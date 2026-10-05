"""Frozen, single-depth pair reduction around an unchanged native adapter.

Setup consumes a bridge-verified dense CPU topology. Each allowed batch size
gets one device plan and reusable hook context; inference only reads the
current block output. Absolute sampled frame indices belong to extraction
records, never to these clip-local plans.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from vadbench.contracts import ClipBatch

from .bridges import EncoderBridge
from .bridges.indexed import IndexedTokenIntervention, indexed_gather
from .contracts import TokenLayout
from .pair_merge import (
    PairLinearGate,
    PairMergeError,
    PairWeightedMerge,
    horizontal_pair_merge_spec,
)
from .token_selection import (
    GROUP_TOKENS,
    PAIR_MEMBER_RULES,
    SIGNAL_NAME,
    SUPPORTED_KEEP_RATIOS,
    GroupBudgetSelector,
    GroupSelectSpec,
    group_select_spec,
)


def _implementation_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expand_layout(layout: TokenLayout, batch_size: int) -> TokenLayout:
    def expand(value: torch.Tensor) -> torch.Tensor:
        return value.expand(batch_size, *value.shape[1:]).clone()

    # Copy only local geometry. No source-frame indices or sample metadata can
    # be carried from the setup clip into later extraction batches.
    provenance = {
        key: list(value) if isinstance(value, list) else value
        for key, value in layout.provenance.items()
        if key in {"coordinate_source", "grid", "kernel", "stride", "patch_stride"}
    }
    return TokenLayout(
        valid_mask=expand(layout.valid_mask),
        original_token_ids=expand(layout.original_token_ids),
        special_token_mask=expand(layout.special_token_mask),
        mass=expand(layout.mass),
        source_coordinates=expand(layout.source_coordinates),
        position_contract=layout.position_contract,
        provenance=provenance,
    )


class PairMergeDeployment:
    """Prepare fixed-half horizontal pair plans before native extraction.

    ``dense_layout`` must come from one completed ``bridge.geometry`` B=1
    observation of this model. Only the first merge of an unpadded, unit-mass
    layout is supported. A learned gate requires its frozen calibration
    manifest SHA-256, and is privately cloned so later training cannot change
    the declared reducer. ``paired_random`` chooses one member per pair at
    setup, sharing a fixed seed/mask across all clips and batch sizes. It is
    distinct from pilot controls that resample with a different clip seed.
    ``global_uniform`` is a same-budget static control, spacing selections
    uniformly in native patch order (complete trajectories for TimeSformer).
    These operators make no anomaly-specific or quality claim. Execution is
    sequential, like the native encoder they temporarily hook.
    """

    def __init__(
        self,
        bridge: EncoderBridge,
        dense_layout: TokenLayout,
        *,
        depth: int,
        dim: int,
        batch_sizes: Iterable[int] = range(1, 9),
        device: torch.device | str | None = None,
        gate: PairLinearGate | None = None,
        calibration_manifest_digest: str | None = None,
        strategy: str = "mean",
        seed: int = 0,
    ) -> None:
        if strategy not in ("mean", "paired_random", "global_uniform"):
            raise PairMergeError("strategy 必须是 mean、paired_random 或 global_uniform")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise PairMergeError("seed 必须为 [0, 2**63) 内的整数")
        if strategy != "mean" and gate is not None:
            raise PairMergeError(f"{strategy} 不接受 gate")
        if type(dim) is not int or dim <= 0:
            raise PairMergeError("dim 必须为正整数")
        if dense_layout.batch_size != 1:
            raise PairMergeError("deployment setup 需要 bridge 验证的 B=1 dense layout")
        for name in ("valid_mask", "original_token_ids", "special_token_mask", "mass", "source_coordinates"):
            value = getattr(dense_layout, name)
            if value is None or value.device.type != "cpu":
                raise PairMergeError("deployment setup 需要完整 CPU dense layout")
        if not bool(dense_layout.valid_mask.all()) or not bool(dense_layout.mass.eq(1).all()):
            raise PairMergeError("deployment 仅支持无 padding、全 unit mass 的第一处 merge")
        if "pair_merge" in dense_layout.provenance:
            raise PairMergeError("deployment 不支持已有 merge history")
        if not torch.equal(dense_layout.original_token_ids, torch.arange(dense_layout.token_capacity)[None]):
            raise PairMergeError("deployment 需要完整 native dense token IDs")
        encoder_id = bridge.receipt().encoder_id
        coordinate_source = dense_layout.provenance.get("coordinate_source")
        expected_source = (
            "verified-timesformer-spatial-major-time-minor"
            if encoder_id == "timesformer" else "verified-conv3d-flatten-t-h-w"
        )
        if coordinate_source != expected_source:
            raise PairMergeError("deployment 需要当前 native bridge 验证的 coordinate_source")
        sizes = tuple(batch_sizes)
        if not sizes or any(type(size) is not int or size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
            raise PairMergeError("batch_sizes 必须为非空、不重复的正整数")
        if gate is not None and not isinstance(gate, PairLinearGate):
            raise PairMergeError("deployment gate 必须是 PairLinearGate")
        if gate is not None and calibration_manifest_digest is None:
            raise PairMergeError("pair_linear 需要冻结的 calibration manifest digest")
        if calibration_manifest_digest is not None and (
            len(calibration_manifest_digest) != 64
            or any(char not in "0123456789abcdef" for char in calibration_manifest_digest)
        ):
            raise PairMergeError("calibration_manifest_digest 必须为小写 SHA-256")

        parameter = next(bridge.model.parameters())
        execution_device = parameter.device if device is None else torch.device(device)
        frozen_gate = None
        gate_digest = None
        gate_dtype = None
        if gate is not None:
            if gate.weight.numel() != dim or not bool(torch.isfinite(gate.weight).all()):
                raise PairMergeError("gate 必须具有有限的 [dim] 权重")
            frozen_gate = PairLinearGate(dim).to(device=execution_device, dtype=gate.weight.dtype)
            with torch.no_grad():
                frozen_gate.weight.copy_(gate.weight.detach())
            frozen_gate.eval().requires_grad_(False)
            weight = frozen_gate.weight.detach().cpu().contiguous()
            gate_dtype = str(weight.dtype)
            gate_digest = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
        self._merger = PairWeightedMerge(dim, frozen_gate).eval().requires_grad_(False)

        base = _expand_layout(dense_layout, 1)
        base_plan = horizontal_pair_merge_spec(encoder_id, base)
        selected_indices = None
        if strategy == "paired_random":
            # Draw exactly one bit per complete TimeSformer trajectory (or
            # per tubelet pair). Neither setup nor forward changes global RNG.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            groups = base_plan.pair_indices.shape[1] // base_plan.trajectory_length
            choices = torch.randint(2, (groups,), generator=generator)
            choices = choices.repeat_interleave(base_plan.trajectory_length)
            selected = base_plan.pair_indices[0].gather(1, choices[:, None]).flatten()
            specials = torch.nonzero(base.special_token_mask[0], as_tuple=False).flatten()
            selected_indices = torch.sort(torch.cat((specials, selected))).values[None]
        elif strategy == "global_uniform":
            # Match the neutral uniform control exactly: choose half of the
            # native units with linspace -> round -> int64, then retain order.
            patches = torch.nonzero(~base.special_token_mask[0], as_tuple=False).flatten()
            units = patches.reshape(-1, base_plan.trajectory_length)
            slots = torch.linspace(0, len(units) - 1, len(units) // 2).round().long()
            specials = torch.nonzero(base.special_token_mask[0], as_tuple=False).flatten()
            selected_indices = torch.sort(torch.cat((specials, units[slots].flatten()))).values[None]
        grid = base.provenance["grid"]
        if encoder_id == "timesformer":
            self._clip_frames = grid[0]
        else:
            kernel, stride = base.provenance.get("kernel"), base.provenance.get("stride")
            if not kernel or not stride or kernel[0] != stride[0]:
                raise PairMergeError("deployment 需要已验证的非重叠 temporal tubelet kernel/stride")
            self._clip_frames = grid[0] * stride[0]
        module = Path(__file__)
        identity = {
            "name": "pair_mean" if gate is None else "pair_linear",
            "strategy": strategy,
            "encoder_id": encoder_id,
            "depth": depth,
            "hidden_dim": dim,
            "patch_keep_ratio": 0.5,
            "operator": "horizontal_pair_weighted_mean",
            "pairing": "adjacent_width_even_left_odd_right",
            "position_anchor": "first_member_native_index",
            "position_semantics": "approximate_anchor_position",
            "position_strategy": "native_position_algorithm_unchanged",
            "mass_convention": "pair_sum_special_unit_mass",
            "special_tokens": "preserved_unmodified",
            "gate_rule": "uniform_mean" if gate is None else "bias_free_linear_two_member_softmax",
            "gate_trajectory_length": base_plan.trajectory_length,
            "gate_trajectory_rule": "mean_logits_over_complete_trajectory" if encoder_id == "timesformer" else "independent_pairs",
            "clip_frames": self._clip_frames,
            "input_topology_sha256": base_plan.input_topology_sha256,
            "deployment_sha256": _implementation_digest(module),
            "pair_merge_sha256": _implementation_digest(module.with_name("pair_merge.py")),
            "indexed_sha256": _implementation_digest(module.parent / "bridges" / "indexed.py"),
            "gate_tensor_sha256": gate_digest,
            "gate_dtype": gate_dtype,
            "calibration_manifest_digest": calibration_manifest_digest,
        }
        if strategy != "mean":
            identity.update({
                "name": strategy,
                "operator": f"{strategy}_subsample",
                "position_anchor": "selected_member_native_index",
                "position_semantics": "retained_native_position",
                "mass_convention": "retained_unit_mass",
                "discarded_mass": "discarded_not_redistributed",
                "gate_rule": None,
                "gate_trajectory_rule": None,
                "seed": seed if strategy == "paired_random" else None,
                "random_generator": "torch_cpu_local_generator" if strategy == "paired_random" else None,
                "mask_scope": "static_setup_mask_shared_across_all_clips_and_batch_sizes",
                "random_trajectory_rule": "one_bit_per_complete_trajectory" if encoder_id == "timesformer" else "one_bit_per_pair",
                "selected_indices_sha256": hashlib.sha256(selected_indices.numpy().tobytes()).hexdigest(),
            })
            if strategy == "global_uniform":
                identity.update({
                    "pairing": None,
                    "position_anchor": "selected_native_index",
                    "random_trajectory_rule": None,
                    "selection_rule": "linspace_zero_to_last_unit_round_int64",
                    "selection_unit": "complete_spatial_trajectory" if encoder_id == "timesformer" else "patch_token",
                })
        self._reducer_identity = MappingProxyType(identity)
        self._contexts: dict[int, IndexedTokenIntervention] = {}
        for size in sizes:
            layout = base if size == 1 else _expand_layout(base, size)
            transform = None
            if strategy != "mean":
                indices = selected_indices.expand(size, -1).clone().to(execution_device)
            else:
                plan = (base_plan if size == 1 else horizontal_pair_merge_spec(encoder_id, layout)).to(execution_device)
                indices = plan.output_indices
                def transform(hidden, _indices, plan=plan):
                    return self._merger(hidden, plan)
            self._contexts[size] = indexed_gather(
                bridge, depth, indices, layout.to(execution_device),
                transform=transform,
                record_position_masks=False,
            )

    @property
    def reducer_identity(self) -> Mapping[str, Any]:
        """Immutable, path-free identity of the actual frozen operator."""
        return self._reducer_identity

    def __call__(self, clean_batch: ClipBatch) -> IndexedTokenIntervention:
        # ClipBatch already validates shapes/types. Do not read IDs, labels,
        # timestamps, sampled source frames, or pixels to make merge decisions.
        size = clean_batch.batch_size
        if size not in self._contexts:
            raise PairMergeError(f"batch size {size} 未在 setup 准备；禁止 dense 回退")
        if clean_batch.num_frames != self._clip_frames:
            raise PairMergeError("当前 clip 帧数与 verified geometry 不一致")
        if clean_batch.valid_mask is not None and not bool(clean_batch.valid_mask.all()):
            raise PairMergeError("deployment 不支持 padded clip")
        return self._contexts[size]


__all__ = ["GroupSelectDeployment", "PairMergeDeployment"]


class GroupSelectDeployment:
    """Frozen three-tier group-budget selection around an unchanged adapter.

    ``rule`` is one of ``pair_select`` (dynamic signal ranking; the current
    candidate, not a settled method), ``group_uniform`` or ``group_random``
    (static same-budget controls). Uniform/random are exact static gathers:
    retained tokens keep their native positions. ``pair_select`` fills the
    same quota slots dynamically and discloses
    ``position_semantics=approximate_anchor_position``; the real per-forward
    chosen indices are recorded by the selector and exposed through
    ``selection_digest``/the intervention receipt. Setup mirrors
    :class:`PairMergeDeployment`: a bridge-verified B=1 dense CPU topology,
    one context per allowed batch size, no dense fallback at inference.
    """

    def __init__(
        self,
        bridge: EncoderBridge,
        dense_layout: TokenLayout,
        *,
        depth: int,
        dim: int,
        keep_ratio: float,
        rule: str = "pair_select",
        batch_sizes: Iterable[int] = range(1, 9),
        device: torch.device | str | None = None,
        signal_fn: Any = None,
        seed: int = 0,
        record_diagnostics: bool = True,
    ) -> None:
        if rule not in set(PAIR_MEMBER_RULES) | {"group_uniform", "group_random"}:
            raise PairMergeError(
                f"rule 必须是 {PAIR_MEMBER_RULES} 之一或 group_uniform/group_random"
            )
        if not isinstance(record_diagnostics, bool):
            raise PairMergeError("record_diagnostics 必须为布尔")
        if keep_ratio not in SUPPORTED_KEEP_RATIOS:
            raise PairMergeError(
                f"keep_ratio 必须是 {SUPPORTED_KEEP_RATIOS} 之一；dense 基线请用 identity"
            )
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise PairMergeError("seed 必须为 [0, 2**63) 内的整数")
        if rule not in PAIR_MEMBER_RULES and signal_fn is not None:
            raise PairMergeError("signal_fn 仅 pair_member 规则接受")
        if type(dim) is not int or dim <= 0:
            raise PairMergeError("dim 必须为正整数")
        if dense_layout.batch_size != 1:
            raise PairMergeError("deployment setup 需要 bridge 验证的 B=1 dense layout")
        for name in ("valid_mask", "original_token_ids", "special_token_mask", "mass", "source_coordinates"):
            value = getattr(dense_layout, name)
            if value is None or value.device.type != "cpu":
                raise PairMergeError("deployment setup 需要完整 CPU dense layout")
        if not bool(dense_layout.valid_mask.all()) or not bool(dense_layout.mass.eq(1).all()):
            raise PairMergeError("deployment 仅支持无 padding、全 unit mass 的压缩点")
        if "pair_merge" in dense_layout.provenance:
            raise PairMergeError("deployment 不支持已有 merge history")
        if not torch.equal(dense_layout.original_token_ids, torch.arange(dense_layout.token_capacity)[None]):
            raise PairMergeError("deployment 需要完整 native dense token IDs")
        encoder_id = bridge.receipt().encoder_id
        if encoder_id not in {"videomaev2", "timesformer", "videomae"}:
            raise PairMergeError(f"不支持的 active encoder={encoder_id!r}")
        coordinate_source = dense_layout.provenance.get("coordinate_source")
        expected_source = (
            "verified-timesformer-spatial-major-time-minor"
            if encoder_id == "timesformer" else "verified-conv3d-flatten-t-h-w"
        )
        if coordinate_source != expected_source:
            raise PairMergeError("deployment 需要当前 native bridge 验证的 coordinate_source")
        sizes = tuple(batch_sizes)
        if not sizes or any(type(size) is not int or size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
            raise PairMergeError("batch_sizes 必须为非空、不重复的正整数")

        parameter = next(bridge.model.parameters())
        execution_device = parameter.device if device is None else torch.device(device)

        base = _expand_layout(dense_layout, 1)
        spec = group_select_spec(encoder_id, base, keep_ratio)
        specials = torch.nonzero(base.special_token_mask[0], as_tuple=False).flatten()
        if rule in PAIR_MEMBER_RULES:
            chosen = spec.skeleton_indices  # slot skeleton; real choice is dynamic
        elif rule == "group_uniform":
            chosen = spec.skeleton_indices
        else:
            probe = GroupBudgetSelector(dim, rule="group_random", seed=seed)
            chosen = probe._random_slots(spec, torch.device("cpu"))
        indices_base = torch.sort(torch.cat((specials, chosen))).values[None]
        grid = base.provenance["grid"]
        if encoder_id == "timesformer":
            self._clip_frames = grid[0]
        else:
            kernel, stride = base.provenance.get("kernel"), base.provenance.get("stride")
            if not kernel or not stride or kernel[0] != stride[0]:
                raise PairMergeError("deployment 需要已验证的非重叠 temporal tubelet kernel/stride")
            self._clip_frames = grid[0] * stride[0]

        module = Path(__file__)
        identity: dict[str, Any] = {
            "name": rule,
            "operator": "group_budget_select_v1" if rule in PAIR_MEMBER_RULES else f"{rule}_subsample",
            "budget_rule": "group_budget_v1",
            "keep_ratio": keep_ratio,
            "encoder_id": encoder_id,
            "depth": depth,
            "hidden_dim": dim,
            "group_size": GROUP_TOKENS,
            "selection_unit": spec.budget_receipt()["selection_unit"],
            "tokens_per_slot": spec.tokens_per_slot,
            "quotas": list(spec.quotas),
            "kept_units": spec.kept_units,
            "reducible_tokens": spec.reducible_tokens,
            "kept_tokens": indices_base.numel() - specials.numel(),
            "actual_keep_ratio": spec.actual_keep_ratio,
            "rounding": "round_half_up_floor_x_plus_0.5",
            "same_budget_for_all_controls": True,
            "special_tokens": "preserved_unmodified",
            "mass_convention": "retained_unit_mass",
            "discarded_mass": "discarded_not_redistributed",
            "clip_frames": self._clip_frames,
            "input_topology_sha256": spec.input_topology_sha256,
            "selection_signal": SIGNAL_NAME if rule in PAIR_MEMBER_RULES else None,
            "signal_scope": "current_block_output_only" if rule in PAIR_MEMBER_RULES else None,
            "leakage_contract": "no_labels_no_filenames_no_test_statistics_no_future_layers",
            "deployment_sha256": _implementation_digest(module),
            "token_selection_sha256": _implementation_digest(module.with_name("token_selection.py")),
            "indexed_sha256": _implementation_digest(module.parent / "bridges" / "indexed.py"),
            "seed": seed if rule in {"group_random", "pair_random_member"} else None,
            "random_generator": "torch_cpu_local_generator" if rule in {"group_random", "pair_random_member"} else None,
            "mask_scope": (
                "current_forward_mask_per_batch_item" if rule in PAIR_MEMBER_RULES
                else "static_setup_mask_shared_across_all_clips_and_batch_sizes"
            ),
            "record_diagnostics": record_diagnostics,
            "tie_breaking": "stable_argsort_signal_desc_then_native_order",
            "within_unit_preference": {
                "pair_select": "max_norm_member_first",
                "pair_fixed": "native_first_member_always",
                "pair_random_member": "seeded_member_per_pair",
                "pair_reverse": "min_norm_member_first",
            }.get(rule),
            "ablation_note": (
                "pair_fixed/pair_random_member/pair_reverse share the identical pair "
                "table, quota table, pair ordering and slot layout with pair_select; "
                "only the within-pair member preference differs (batch-1b ordering "
                "ablation); pair ordering signal is unchanged"
            ),
            "geometry_snap": spec.geometry_snap,
            "selection_unit_note": (
                "videomaev2/videomae: native pair (budget slot = token); "
                "timesformer: complete spatial trajectory from verified coordinates "
                "(whole trajectories only, never per-frame spatial subsampling)"
            ),
        }
        if rule in PAIR_MEMBER_RULES:
            identity.update({
                "dynamic_selection": True,
                "position_anchor": "skeleton_slot_native_index",
                "position_semantics": "approximate_anchor_position",
                "position_note": (
                    "kept unit tokens fill quota slots of their group; slot coordinates "
                    "come from the uniform skeleton, real chosen indices are recorded "
                    "per forward via selection_digest"
                ),
                "slot_layout": "identical_across_rules_at_same_keep_ratio",
            })
        else:
            identity.update({
                "dynamic_selection": False,
                "position_anchor": "selected_native_index",
                "position_semantics": "retained_native_position",
                "selected_indices_sha256": hashlib.sha256(indices_base.numpy().tobytes()).hexdigest(),
            })
        self._reducer_identity = MappingProxyType(identity)
        self._selectors: dict[int, GroupBudgetSelector] = {}
        self._contexts: dict[int, IndexedTokenIntervention] = {}
        for size in sizes:
            layout = base if size == 1 else _expand_layout(base, size)
            indices = indices_base.expand(size, -1).clone().to(execution_device)
            transform = None
            if rule in PAIR_MEMBER_RULES:
                selector = GroupBudgetSelector(
                    dim, rule=rule, signal_fn=signal_fn, seed=seed,
                    record_diagnostics=record_diagnostics,
                ).to(execution_device).eval().requires_grad_(False)
                self._selectors[size] = selector
                def transform(hidden, _indices, selector=selector, spec=spec, specials=specials):
                    return selector(hidden, spec, specials.to(hidden.device))
            self._contexts[size] = indexed_gather(
                bridge, depth, indices, layout.to(execution_device),
                transform=transform,
                record_position_masks=False,
            )
        self._spec = spec

    @property
    def reducer_identity(self) -> Mapping[str, Any]:
        return self._reducer_identity

    @property
    def selection_spec(self) -> GroupSelectSpec:
        return self._spec

    def selection_digest(self, batch_size: int) -> str | None:
        selector = self._selectors.get(batch_size)
        return None if selector is None else selector.selection_digest()

    def __call__(self, clean_batch: ClipBatch) -> IndexedTokenIntervention:
        size = clean_batch.batch_size
        if size not in self._contexts:
            raise PairMergeError(f"batch size {size} 未在 setup 准备；禁止 dense 回退")
        if clean_batch.num_frames != self._clip_frames:
            raise PairMergeError("当前 clip 帧数与 verified geometry 不一致")
        if clean_batch.valid_mask is not None and not bool(clean_batch.valid_mask.all()):
            raise PairMergeError("deployment 不支持 padded clip")
        return self._contexts[size]
