"""Offline, label-free fixed-budget intervention diagnostics for one clip."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vadbench.contracts import ClipBatch
from vadbench.data.batches import clean_encoder_batch
from vadbench.research.interventions import (
    fixed_budget_indices,
    paired_spatial_indices,
    relative_branch_update,
    same_position_temporal_change,
)
from vadbench.token_reduction.bridges import (
    IndexedTokenIntervention,
    create_observation_bridge,
    identity_indices,
)

Candidate = Literal["relative_attention_update", "midlayer_temporal_change"]
Strategy = Literal["uniform", "seeded_random", "score_high", "score_low"]


class InterventionRunnerUnsupportedError(RuntimeError):
    """The actual adapter cannot expose a truthful shortened-token output."""


@dataclass(frozen=True)
class PooledComparison:
    pooled_shape: tuple[int, ...]
    feature_shape: tuple[int, ...]
    relative_l2: float
    cosine: float
    max_absolute_error: float


@dataclass(frozen=True)
class InterventionResult:
    name: str
    indices_sha256: str | None
    effective_budget: int
    gathered_shape: tuple[int, ...] | None
    suffix_shapes: Mapping[int, tuple[int, ...]]
    pooled: PooledComparison


@dataclass(frozen=True)
class InterventionReceipt:
    encoder_id: str
    candidate: Candidate
    intervention_depth: int
    requested_budget: int
    offline_prefix_score_diagnostic: bool
    results: Mapping[str, InterventionResult]


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim < 2 or not torch.is_floating_point(value):
        raise InterventionRunnerUnsupportedError(f"adapter {name} must be a floating tensor with batch axis")
    return value.detach().float().cpu()


def _pooled_comparison(native: Any, output: Any) -> PooledComparison:
    reference = _tensor(native.pooled, name="native pooled")
    actual = _tensor(output.pooled, name="pooled")
    if reference.shape != actual.shape:
        raise InterventionRunnerUnsupportedError("adapter pooled shape changed after indexed intervention")
    difference = actual - reference
    denominator = torch.linalg.vector_norm(reference)
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all() or denominator <= 0:
        raise InterventionRunnerUnsupportedError("pooled comparison requires finite outputs and nonzero reference norm")
    relative_l2 = float(torch.linalg.vector_norm(difference) / denominator)
    cosine = float(torch.nn.functional.cosine_similarity(reference.reshape(1, -1), actual.reshape(1, -1)).item())
    return PooledComparison(
        pooled_shape=tuple(actual.shape),
        feature_shape=tuple(_tensor(output.features, name="features").shape),
        relative_l2=relative_l2,
        cosine=cosine,
        max_absolute_error=float(difference.abs().max()),
    )


def _indices_digest(indices: torch.Tensor) -> str:
    values = indices.detach().cpu().to(torch.int64).contiguous().numpy()
    return "sha256:" + hashlib.sha256(values.tobytes()).hexdigest()


def _first_tensor(value: Any, *, site: str) -> torch.Tensor:
    result = value[0] if isinstance(value, tuple) and value else value
    if not isinstance(result, torch.Tensor) or result.ndim != 3:
        raise InterventionRunnerUnsupportedError(f"{site} did not emit a [B,N,D] tensor")
    return result


def _restore_timesformer(value: torch.Tensor, layout: Any) -> torch.Tensor:
    grid = layout.provenance.get("grid")
    if not isinstance(grid, list) or len(grid) != 3 or not all(type(item) is int and item > 0 for item in grid):
        raise InterventionRunnerUnsupportedError("TimeSformer requires verified [T,H,W] geometry")
    frames, height, width = grid
    patches = height * width
    batch = layout.batch_size
    if value.shape[:2] != (batch * frames, 1 + patches):
        raise InterventionRunnerUnsupportedError(
            "TimeSformer spatial attention did not expose the verified B*T by P+CLS layout"
        )
    restored = value.reshape(batch, frames, 1 + patches, value.shape[-1])
    cls = restored[:, :, :1].mean(dim=1)
    patch_tokens = restored[:, :, 1:].permute(0, 2, 1, 3).reshape(batch, patches * frames, -1)
    return torch.cat((cls, patch_tokens), dim=1)


def _capture_scores(
    adapter: Any, bridge: Any, clean: ClipBatch, depth: int, candidate: Candidate, layout: Any
) -> tuple[torch.Tensor, dict[int, tuple[int, ...]]]:
    sites = bridge.observation_sites([depth])
    prefix = f"block.{depth}"
    captured: dict[str, torch.Tensor] = {}
    suffix_shapes: dict[int, tuple[int, ...]] = {}

    def capture_input(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                raise InterventionRunnerUnsupportedError(f"{name} did not receive tensor input")
            captured[name] = _first_tensor(inputs[0], site=name)

        return hook

    def capture_output(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            captured[name] = _first_tensor(output, site=name)

        return hook

    handles = [sites[f"{prefix}.input"].register_forward_pre_hook(capture_input("block_input"))]
    for suffix_depth in range(depth + 1, bridge.receipt().block_count):
        suffix = bridge._blocks[suffix_depth]

        def capture_suffix(_module: Any, inputs: tuple[Any, ...], *, observed_depth: int = suffix_depth) -> None:
            if not inputs:
                raise InterventionRunnerUnsupportedError("native suffix did not receive tensor hidden states")
            suffix_shapes[observed_depth] = tuple(_first_tensor(inputs[0], site="native suffix").shape)

        handles.append(suffix.register_forward_pre_hook(capture_suffix))
    if candidate == "relative_attention_update":
        if bridge.receipt().encoder_id == "timesformer":
            input_site = f"{prefix}.spatial.attn.pre_norm.input"
            output_site = f"{prefix}.spatial.attn.output"
        else:
            input_site = f"{prefix}.norm1"
            output_site = f"{prefix}.attn.output"
        handles.extend(
            (
                sites[input_site].register_forward_pre_hook(capture_input("attention_input")),
                sites[output_site].register_forward_hook(capture_output("attention_output")),
            )
        )
    try:
        with torch.no_grad():
            adapter.encode(clean, train=False)
    finally:
        for handle in handles:
            handle.remove()
    if "block_input" not in captured:
        raise InterventionRunnerUnsupportedError("P07 block input hook did not execute")
    if candidate == "midlayer_temporal_change":
        return same_position_temporal_change(captured["block_input"], layout).scores, suffix_shapes
    if "attention_input" not in captured or "attention_output" not in captured:
        raise InterventionRunnerUnsupportedError("P16 attention hooks did not execute")
    attention_input = captured["attention_input"]
    attention_output = captured["attention_output"]
    if bridge.receipt().encoder_id == "timesformer":
        attention_input = _restore_timesformer(attention_input, layout)
        attention_output = _restore_timesformer(attention_output, layout)
    return relative_branch_update(attention_output, attention_input), suffix_shapes


def _run_indexed(adapter: Any, bridge: Any, clean: ClipBatch, depth: int, layout: Any, indices: torch.Tensor) -> tuple[Any, dict[str, Any]]:
    with IndexedTokenIntervention(bridge, depth, indices, layout) as intervention:
        with torch.no_grad():
            output = adapter.encode(clean, train=False)
        execution = intervention.validate_execution()
    features = _tensor(output.features, name="features")
    expected = int(execution["gathered_tokens"])
    if features.ndim != 3 or features.shape[1] != expected:
        raise InterventionRunnerUnsupportedError(
            "adapter output token timeline/reshape does not match the verified shortened suffix"
        )
    return output, execution


def run_intervention_diagnostic(
    *,
    adapter: Any,
    encoder_id: str,
    batch: ClipBatch,
    candidate: Candidate,
    relative_depth: float = 0.5,
    budget_ratio: float = 0.5,
    seed: int = 0,
    include_paired: bool = False,
) -> InterventionReceipt:
    """Measure explicit offline index controls on one clean B=1 fixed clip."""

    if candidate not in {"relative_attention_update", "midlayer_temporal_change"}:
        raise ValueError(f"unknown candidate={candidate!r}")
    if not isinstance(batch, ClipBatch) or batch.batch_size != 1:
        raise ValueError("intervention diagnostic accepts exactly one ClipBatch item")
    if type(relative_depth) is not float or not 0 < relative_depth <= 1:
        raise ValueError("relative_depth must be in (0, 1]")
    if type(budget_ratio) is not float or not 0 < budget_ratio <= 1:
        raise ValueError("budget_ratio must be in (0, 1]")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if include_paired and budget_ratio != 0.5:
        raise ValueError("paired coverage diagnostic requires budget_ratio=0.5")
    clean = clean_encoder_batch(batch)
    if clean.frame_indices is None or clean.valid_mask is None:
        raise ValueError("intervention diagnostic requires explicit frame_indices and valid_mask")
    bridge = create_observation_bridge(encoder_id, adapter)
    model = bridge.model
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        geometry = bridge.geometry(clean.frame_indices, clean.valid_mask)
        with torch.no_grad(), geometry:
            dense = adapter.encode(clean, train=False)
        geometry_verified = geometry.receipt.get("flatten_verified") or (
            geometry.receipt.get("patch_flatten_verified")
            and geometry.receipt.get("divided_layout_verified")
        )
        if not geometry_verified:
            raise InterventionRunnerUnsupportedError("dense bridge geometry was not verified")
        layout = geometry.layout
        if tuple(dense.features.shape[:2]) != tuple(layout.valid_mask.shape):
            raise InterventionRunnerUnsupportedError("dense adapter token timeline/reshape differs from verified geometry")
        depth = max(0, math.ceil(relative_depth * bridge.receipt().block_count) - 1)
        if depth >= bridge.receipt().block_count - 1:
            raise InterventionRunnerUnsupportedError("relative_depth leaves no native suffix block")
        scores, dense_suffix_shapes = _capture_scores(adapter, bridge, clean, depth, candidate, layout)
        budget = math.ceil(int(layout.valid_token_counts[0]) * budget_ratio)
        controls: dict[str, InterventionResult] = {}
        identity = identity_indices(layout)
        identity_output, identity_execution = _run_indexed(adapter, bridge, clean, depth, layout, identity)
        identity_metrics = _pooled_comparison(dense, identity_output)
        if not torch.allclose(
            _tensor(dense.pooled, name="native pooled"),
            _tensor(identity_output.pooled, name="identity pooled"),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise InterventionRunnerUnsupportedError("identity indexed suffix changed the adapter pooled output")
        controls["identity"] = InterventionResult(
            name="identity",
            indices_sha256=_indices_digest(identity),
            effective_budget=int(identity_execution["gathered_tokens"]),
            gathered_shape=tuple(identity_execution["gathered_shape"]),
            suffix_shapes={int(key): tuple(value) for key, value in identity_execution["suffix_shapes"].items()},
            pooled=identity_metrics,
        )
        selections = {
            strategy: fixed_budget_indices(
                scores, layout, budget, strategy, seed=seed if strategy == "seeded_random" else None
            )
            for strategy in ("uniform", "seeded_random", "score_high", "score_low")
        }
        if include_paired:
            for choice in ("first", "random", "high", "low"):
                selection = paired_spatial_indices(
                    scores, layout, choice, seed=seed if choice == "random" else None
                )
                if selection.effective_budget != budget:
                    raise InterventionRunnerUnsupportedError("paired and global control budgets differ")
                selections[f"paired_{choice}"] = selection
        for strategy, selection in selections.items():
            output, execution = _run_indexed(adapter, bridge, clean, depth, layout, selection.indices)
            controls[strategy] = InterventionResult(
                name=strategy,
                indices_sha256=_indices_digest(selection.indices),
                effective_budget=selection.effective_budget,
                gathered_shape=tuple(execution["gathered_shape"]),
                suffix_shapes={int(key): tuple(value) for key, value in execution["suffix_shapes"].items()},
                pooled=_pooled_comparison(dense, output),
            )
        controls["dense"] = InterventionResult(
            name="dense",
            indices_sha256=None,
            effective_budget=int(layout.valid_token_counts[0]),
            gathered_shape=tuple(_tensor(dense.features, name="native features").shape),
            suffix_shapes=dense_suffix_shapes,
            pooled=PooledComparison(
                pooled_shape=tuple(_tensor(dense.pooled, name="native pooled").shape),
                feature_shape=tuple(_tensor(dense.features, name="native features").shape),
                relative_l2=0.0,
                cosine=1.0,
                max_absolute_error=0.0,
            ),
        )
        return InterventionReceipt(
            encoder_id=encoder_id,
            candidate=candidate,
            intervention_depth=depth,
            requested_budget=budget,
            offline_prefix_score_diagnostic=True,
            results=controls,
        )
    finally:
        for module, mode in modes:
            module.training = mode


__all__ = [
    "InterventionReceipt",
    "InterventionResult",
    "InterventionRunnerUnsupportedError",
    "PooledComparison",
    "run_intervention_diagnostic",
]
