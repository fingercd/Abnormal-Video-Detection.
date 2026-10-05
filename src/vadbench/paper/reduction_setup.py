"""Verify native geometry and frozen fit calibration before paper extraction."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from vadbench.checkpoints import sha256_file
from vadbench.contracts import ClipBatch
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.deployment import GroupSelectDeployment, PairMergeDeployment
from vadbench.token_reduction.deployment_contracts import ReductionDeployment
from vadbench.token_reduction.pair_merge import PairLinearGate
from vadbench.token_reduction.token_selection import SUPPORTED_KEEP_RATIOS

from vadbench.workflows.extraction import adapter_runtime_summary
from vadbench.data.batches import clean_encoder_batch

_ROLE_LOCK_SHA256 = "53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59"
_GEOMETRY_KEYS = ("processed_shape", "processed_layout", "grid", "token_order", "patch_size", "tubelet_size")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _checkpoint(identity: Mapping[str, Any]) -> Any:
    asset = identity.get("checkpoint")
    if not isinstance(asset, Mapping):
        raise ValueError("verified encoder identity requires checkpoint SHA-256 evidence")
    checkpoint = asset.get("sha256")
    if isinstance(checkpoint, Mapping):
        if not checkpoint:
            raise ValueError("verified checkpoint SHA-256 mapping must not be empty")
        for digest in checkpoint.values():
            _sha256(digest, "checkpoint.sha256")
    else:
        _sha256(checkpoint, "checkpoint.sha256")
    return checkpoint


def _readout(identity: Mapping[str, Any]) -> dict[str, Any]:
    constructor = identity.get("constructor")
    if not isinstance(constructor, Mapping):
        raise ValueError("verified encoder identity requires a constructor mapping")
    return {key: constructor.get(key) for key in ("pooling", "feature_stage")}


@contextmanager
def _frozen_native_model(model: torch.nn.Module):
    """Freeze the actual bridge model, also for adapters with only .model."""
    modes = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            yield
    finally:
        for parameter, required in gradients:
            parameter.requires_grad_(required)
        for module, training in modes:
            module.training = training


def _calibrated_gate(
    run: Path, *, encoder_id: str, output_dim: int, depth: int,
    verified_encoder_identity: Mapping[str, Any], geometry: Mapping[str, Any],
) -> tuple[PairLinearGate, str, dict[str, Any]]:
    run = run.resolve()
    receipt_bytes = (run / "receipt.json").read_bytes()
    source = json.loads(receipt_bytes.decode("utf-8"))
    if not isinstance(source, dict):
        raise ValueError("calibration receipt must be an object")
    required = {
        "status": "completed", "encoder": encoder_id, "epochs": 3,
        "sample_count": 256, "steps": 768, "seed": 0, "batch_size": 1,
        "weights_changed": True, "backbone_all_grad_none": True,
        "backbone_eval": True, "backbone_all_requires_grad_false": True,
        "optimizer_only_gate": True, "teacher_detached": True, "depth": depth,
    }
    for key, expected in required.items():
        if type(source.get(key)) is not type(expected) or source[key] != expected:
            raise ValueError(f"calibration receipt differs at {key}")
    identity = source.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("plan"), dict):
        raise ValueError("calibration requires the frozen fit128 identity.plan")
    plan = identity["plan"]
    plan_expected = {
        "schema_version": 1, "protocol": "W", "role": "fit", "video_count": 128,
        "clip_count": 256, "epochs": 3, "seed": 0, "role_lock_sha256": _ROLE_LOCK_SHA256,
        "fine_annotations_or_captions_used": False, "test_annotations_read": False,
        "sampling": {"windows_per_video": 8, "selected_window_indices": [1, 6],
                     "stride": 2, "short_policy": "stride1_if_needed"},
    }
    for key, expected in plan_expected.items():
        if plan.get(key) != expected:
            raise ValueError(f"calibration plan differs at {key}")
    manifest_digest = _sha256(plan.get("manifest_sha256"), "calibration manifest_sha256")
    for key in ("grad_l1", "losses"):
        values = source.get(key)
        if not isinstance(values, list) or len(values) != 768 or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values
        ):
            raise ValueError(f"calibration {key} must contain 768 finite nonnegative observations")
    nonzero = sum(value > 0 for value in source["grad_l1"])
    if nonzero == 0 or source.get("nonzero_gradient_steps") != nonzero:
        raise ValueError("calibration must observe and count nonzero gate gradients")
    reload_error = source.get("reload_max_absolute_error")
    if type(reload_error) not in (int, float) or not math.isfinite(reload_error) or not 0 <= reload_error <= 1e-6:
        raise ValueError("calibration reload_max_absolute_error exceeds finite 1e-6 tolerance")
    calibrated_identity = identity.get("encoder_identity")
    if not isinstance(calibrated_identity, dict) or calibrated_identity.get("adapter") != encoder_id:
        raise ValueError("calibration encoder identity does not match encoder_id")
    if _checkpoint(calibrated_identity) != _checkpoint(verified_encoder_identity):
        raise ValueError("calibration checkpoint SHA-256 differs from the current verified checkpoint")
    if _readout(calibrated_identity) != _readout(verified_encoder_identity):
        raise ValueError("calibration readout constructor differs from the current native readout")
    calibrated_geometry = source.get("geometry")
    if not isinstance(calibrated_geometry, dict) or any(
        calibrated_geometry.get(key) != geometry.get(key) for key in _GEOMETRY_KEYS
    ):
        raise ValueError("calibration geometry differs from the current native geometry")
    for key in ("setup_execution", "trained_execution", "reload_execution"):
        execution = source.get(key)
        if not isinstance(execution, dict) or execution.get("intervention_depth") != depth:
            raise ValueError(f"calibration {key} must verify the selected depth")
        shape = execution.get("gathered_shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1 or shape[-1] != output_dim:
            raise ValueError(f"calibration {key} does not match output_dim")

    name = source.get("gate_state_dict")
    if not isinstance(name, str) or name in ("", ".", "..") or any(char in name for char in "/\\:"):
        raise ValueError("gate_state_dict must be a same-directory basename")
    gate_path = (run / name).resolve()
    if gate_path.parent != run:
        raise ValueError("gate_state_dict escapes the calibration directory")
    expected_digest = _sha256(source.get("gate_state_sha256"), "gate_state_sha256")
    if sha256_file(gate_path) != expected_digest:
        raise ValueError("gate checkpoint SHA-256 differs from calibration receipt")
    state = torch.load(gate_path, weights_only=True, map_location="cpu")
    if not isinstance(state, Mapping) or set(state) != {"weight"}:
        raise ValueError("gate checkpoint must contain only weight")
    weight = state["weight"]
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (output_dim,) or not torch.is_floating_point(weight):
        raise ValueError("gate weight must be floating [output_dim]")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("gate weight must be finite")
    gate = PairLinearGate(output_dim).to(dtype=weight.dtype)
    gate.load_state_dict(state)
    gate.eval().requires_grad_(False)
    return gate, manifest_digest, {
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "source_receipt": source,
        "source_runtime": {key: source.get(key) for key in (
            "python_executable", "python_version", "torch", "torch_cuda",
            "backbone_parameter_device", "backbone_parameter_dtype", "gate_device", "gate_dtype",
        )},
        "source_reload_absolute_tolerance": 1e-6,
    }


def prepare_reduction(
    adapter: Any, encoder_id: str, clean_setup_batch: ClipBatch, *, reducer: str,
    output_dim: int, verified_encoder_identity: Mapping[str, Any],
    calibration_run: Path | None = None, seed: int = 0, batch_sizes: Iterable[int] = range(1, 9),
    keep_ratio: float | None = None, signal_fn: Any = None, record_diagnostics: bool = False,
) -> tuple[ReductionDeployment, dict[str, Any]]:
    """Observe one dense B=1 clip and prepare a frozen mid-depth intervention.

    Calibration and inference runtime evidence remain separate. Matching
    checkpoint, geometry and readout establishes structural compatibility;
    this setup does not certify cross-runtime numerical equivalence or VAD
    quality. The formal evaluation gate belongs to the paper controller.
    Selection reducers (pair_select/group_uniform/group_random) use the
    frozen three-tier group budget: ``keep_ratio`` ∈ {0.80, 0.60, 0.40} with
    identical actual budgets across all controls at the same tier.
    """
    selection_reducers = {
        "pair_select", "pair_fixed", "pair_random_member", "pair_reverse",
        "group_uniform", "group_random",
    }
    if reducer not in {"global_uniform", "paired_random", "pair_linear"} | selection_reducers:
        raise ValueError(
            "reducer must be global_uniform, paired_random, pair_linear, "
            "pair_select, pair_fixed, pair_random_member, pair_reverse, "
            "group_uniform or group_random"
        )
    if (reducer == "pair_linear") != (calibration_run is not None):
        raise ValueError("calibration_run is required only for pair_linear")
    if (reducer in selection_reducers) != (keep_ratio is not None):
        raise ValueError("keep_ratio is required exactly for the selection reducers")
    if keep_ratio is not None and keep_ratio not in SUPPORTED_KEEP_RATIOS:
        raise ValueError(f"keep_ratio must be one of {SUPPORTED_KEEP_RATIOS}")

    if type(output_dim) is not int or output_dim <= 0:
        raise ValueError("output_dim must be a positive integer")
    if verified_encoder_identity.get("adapter") != encoder_id:
        raise ValueError("verified encoder identity does not match encoder_id")
    _checkpoint(verified_encoder_identity)
    readout = _readout(verified_encoder_identity)
    clean = clean_encoder_batch(clean_setup_batch)
    if clean.batch_size != 1 or clean.frame_indices is None or clean.valid_mask is None or not bool(clean.valid_mask.all()):
        raise ValueError("setup requires one complete B=1 clip with actual source indices and validity")
    bridge = create_observation_bridge(encoder_id, adapter)
    depth = math.ceil(bridge.receipt().block_count * 0.5) - 1
    geometry = bridge.geometry(clean.frame_indices, clean.valid_mask)
    with _frozen_native_model(bridge.model), geometry:
        output = adapter.encode(clean, train=False)
    evidence = geometry.receipt
    if geometry.layout is None or not (
        evidence.get("flatten_verified") is True or (
            evidence.get("patch_flatten_verified") is True and evidence.get("divided_layout_verified") is True
        )
    ):
        raise RuntimeError("actual native dense token geometry was not verified")
    tokens, pooled = output.features, output.pooled
    if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (1, geometry.layout.token_capacity, output_dim):
        raise ValueError("native dense features do not match [1,N,output_dim]")
    if not isinstance(pooled, torch.Tensor) or tuple(pooled.shape) != (1, output_dim):
        raise ValueError("actual adapter pooled output does not match [1,output_dim]")
    if not bool(torch.isfinite(tokens).all()) or not bool(torch.isfinite(pooled).all()):
        raise ValueError("native dense output must be finite")
    runtime = adapter_runtime_summary(adapter)
    actual_readout = {
        **runtime["runtime_configurations"].get("encoder_cfg", {}), **runtime["properties"],
    }
    for key, value in readout.items():
        if value is not None and actual_readout.get(key) != value:
            raise ValueError(f"actual adapter readout differs from verified constructor: {key}")
    gate, manifest_digest, calibration = None, None, None
    if calibration_run is not None:
        gate, manifest_digest, calibration = _calibrated_gate(
            Path(calibration_run), encoder_id=encoder_id, output_dim=output_dim, depth=depth,
            verified_encoder_identity=verified_encoder_identity, geometry=evidence,
        )
        if gate.weight.dtype != tokens.dtype:
            raise ValueError("calibrated gate dtype differs from current native features; implicit conversion is forbidden")
    if reducer in selection_reducers:
        deployment = GroupSelectDeployment(
            bridge, geometry.layout, depth=depth, dim=output_dim, batch_sizes=batch_sizes,
            device=tokens.device, keep_ratio=keep_ratio, rule=reducer, signal_fn=signal_fn,
            seed=seed, record_diagnostics=record_diagnostics,
        )
    else:
        deployment = PairMergeDeployment(
            bridge, geometry.layout, depth=depth, dim=output_dim, batch_sizes=batch_sizes,
            device=tokens.device, gate=gate, calibration_manifest_digest=manifest_digest,
            strategy="mean" if reducer == "pair_linear" else reducer, seed=seed,
        )
    receipt = {
        "schema_version": 1, "status": "prepared", "encoder_id": encoder_id,
        "reducer": reducer, "depth": depth, "output_dim": output_dim,
        "keep_ratio": keep_ratio,
        "record_diagnostics": record_diagnostics,
        "dense_geometry": evidence,
        "dense_output": {"features_shape": list(tokens.shape), "pooled_shape": list(pooled.shape),
                         "features_dtype": str(tokens.dtype), "pooled_dtype": str(pooled.dtype)},
        "readout_structure": readout, "verified_encoder_identity": dict(verified_encoder_identity),
        "current_runtime": {"adapter": runtime, "python_executable": sys.executable,
                            "python_version": sys.version, "torch": str(torch.__version__),
                            "torch_cuda": torch.version.cuda, "device": str(tokens.device)},
        "calibration": calibration,
        "cross_runtime_numerical_validation": "not_performed",
        "reducer_identity": dict(deployment.reducer_identity),
    }
    return deployment, receipt


__all__ = ["prepare_reduction"]
