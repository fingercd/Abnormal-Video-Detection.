"""Fit-only calibration of one bias-free pair gate against native pooled output.

This is a provisional representation-preservation prototype, not evidence of
anomaly detection quality. Labels are audited at the asset boundary and never
enter the model, merger, optimizer, or per-clip loss.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

import vadbench
from vadbench.checkpoints import sha256_file
from vadbench.contracts import ClipBatch
from vadbench.data.manifest import SupervisionScope, VideoManifestRecord, load_manifest_jsonl
from vadbench.features import atomic_write_json
from vadbench.integrations.common import pool_feature_sequence, select_feature_tensor
from vadbench.paper.stages import clean_encoder_batch

from .bridges import create_observation_bridge, indexed_gather
from .pair_merge import PairLinearGate, PairWeightedMerge, horizontal_pair_merge_spec

ROLE_LOCK_SHA256 = "53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59"
EPOCHS = 3
ENCODERS = ("videomaev2", "timesformer", "vjepa2", "videomae")
SAMPLING = {"windows_per_video": 8, "selected_window_indices": [1, 6], "stride": 2,
            "short_policy": "stride1_if_needed"}


@dataclass(frozen=True)
class CalibrationSample:
    """Audit identity owned by the trainer; never passed to the merger."""

    video_id: str
    window_index: int

    @property
    def key(self) -> str:
        return f"{self.video_id}:window{self.window_index}"


def _reject_fine_labels(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = {"frame_labels", "segment_labels", "captions", "caption", "temporal_annotations",
                     "ground_truth", "frame_annotations", "segment_annotations"}
        if forbidden.intersection(value):
            raise ValueError("calibration assets must not contain frame/segment/caption ground truth")
        for child in value.values():
            _reject_fine_labels(child)
    elif isinstance(value, list):
        for child in value:
            _reject_fine_labels(child)


def validate_calibration_assets(
    *, plan_path: Path, cases_path: Path, manifest_path: Path, role_lock_path: Path,
    dataset_root: Path, verify_video_bytes: bool = False,
) -> tuple[dict[str, Any], tuple[CalibrationSample, ...], dict[str, VideoManifestRecord]]:
    """Require the frozen 128-video/256-window fit plan and exact file identities."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    cases_doc = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(cases_doc, dict):
        raise ValueError("plan and cases must be JSON objects")
    for field, expected in {"schema_version": 1, "protocol": "W", "role": "fit", "video_count": 128,
                            "clip_count": 256, "epochs": EPOCHS, "seed": 0, "sampling": SAMPLING}.items():
        if plan.get(field) != expected:
            raise ValueError(f"frozen calibration plan differs at {field}")
    if cases_doc.get("schema_version") != 1 or set(cases_doc) != {"schema_version", "cases"}:
        raise ValueError("unsupported calibration cases schema")
    for field, path in (("manifest_sha256", manifest_path), ("cases_sha256", cases_path),
                        ("role_lock_sha256", role_lock_path)):
        if sha256_file(path) != plan.get(field):
            raise ValueError(f"calibration identity mismatch: {field}")
    if plan["role_lock_sha256"] != ROLE_LOCK_SHA256:
        raise ValueError("role lock differs from the authorized frozen fit lock")
    lock = json.loads(role_lock_path.read_text(encoding="utf-8"))
    if lock.get("basis") != "complete_official_train_source_groups":
        raise ValueError("role lock must originate from complete official training source groups")
    cases = cases_doc["cases"]
    if not isinstance(cases, list) or len(cases) != 256:
        raise ValueError("calibration requires exactly 256 planned windows")
    samples = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"video_id", "window_index"}:
            raise ValueError("cases may contain only video_id and window_index")
        if not isinstance(case["video_id"], str) or type(case["window_index"]) is not int or case["window_index"] not in (1, 6):
            raise ValueError("invalid calibration window identity")
        samples.append(CalibrationSample(**case))
    if len({sample.key for sample in samples}) != 256:
        raise ValueError("calibration window identities must be unique")
    records = load_manifest_jsonl(manifest_path)
    by_id = {record.video_id: record for record in records}
    if len(records) != 128 or len(by_id) != 128 or set(by_id) != {sample.video_id for sample in samples}:
        raise ValueError("manifest and cases must contain the same 128 unique videos")
    if sum(record.is_anomaly for record in records) != 64:
        raise ValueError("calibration requires 64 normal and 64 positive training videos")
    root = dataset_root.resolve()
    _reject_fine_labels(plan)
    for record in records:
        if lock.get("partitions", {}).get(record.video_id) != "fit" or record.split.value != "train":
            raise ValueError("calibration refuses test/confirm/select and non-training records")
        if any(annotation.scope != SupervisionScope.VIDEO for annotation in record.annotations):
            raise ValueError("calibration permits video-level annotations only")
        _reject_fine_labels(dict(record.metadata))
        digest = record.metadata.get("training_asset_sha256")
        size = record.metadata.get("size_bytes")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("manifest requires a training_asset_sha256 for every video")
        if type(size) is not int or size <= 0 or record.num_frames is None or record.fps is None:
            raise ValueError("manifest requires positive bytes, actual num_frames and fps")
        path = (root / record.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("video path escapes dataset root")
        if verify_video_bytes and (not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest):
            raise ValueError(f"video content identity mismatch: {record.video_id}")
    return plan, tuple(samples), by_id


def relative_pooled_mse(prediction: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """The sole objective: one clip's squared error / detached teacher energy."""
    if prediction.ndim != 2 or prediction.shape[0] != 1 or prediction.shape != teacher.shape:
        raise ValueError("pair calibration requires aligned [1,D] pooled tensors")
    if teacher.requires_grad or teacher.grad_fn is not None:
        raise ValueError("dense teacher must be detached")
    denominator = teacher.float().square().sum()
    if not bool(torch.isfinite(denominator)) or denominator.item() <= 0:
        raise RuntimeError("dense teacher pooled energy must be finite and positive")
    return (prediction.float() - teacher.float()).square().sum() / denominator


def gate_training_step(
    gate: PairLinearGate, backbone: torch.nn.Module, optimizer: torch.optim.Optimizer,
    forward: Callable[[], torch.Tensor], teacher: torch.Tensor,
) -> dict[str, float]:
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(parameters) != 1 or parameters[0] is not gate.weight:
        raise RuntimeError("optimizer must contain only the shared linear gate weight")
    if any(module.training for module in backbone.modules()) or any(p.requires_grad for p in backbone.parameters()):
        raise RuntimeError("backbone must be entirely frozen and eval, including dropout")
    optimizer.zero_grad(set_to_none=True)
    loss = relative_pooled_mse(forward(), teacher)
    if not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise RuntimeError("gate loss is nonfinite or disconnected from autograd")
    loss.backward()
    gradient = gate.weight.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("gate gradient is missing or nonfinite")
    if any(parameter.grad is not None for parameter in backbone.parameters()):
        raise RuntimeError("frozen backbone accumulated gradients")
    if any(module.training for module in backbone.modules()):
        raise RuntimeError("native forward enabled model training/dropout")
    result = {"loss": float(loss.detach()), "grad_l1": float(gradient.detach().float().abs().sum())}
    optimizer.step()
    if not bool(torch.isfinite(gate.weight).all()):
        raise RuntimeError("optimizer produced nonfinite gate weights")
    return result


def native_pooled_forward(adapter: Any, encoder: str, batch: ClipBatch, *, gradients: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain native adapter readout without enabling model train/dropout mode."""
    if encoder == "vjepa2":
        raw = adapter.encode_with_grad(batch) if gradients else adapter.bridge.encoder.encode(batch)
        tokens, _source = select_feature_tensor(raw, batch_size=batch.batch_size)
        if tokens.ndim != 3:
            raise RuntimeError("V-JEPA calibration requires native [B,N,D] tokens")
        pooled = pool_feature_sequence(tokens, np.ones(tuple(tokens.shape[:2]), dtype=bool))
    elif encoder in {"timesformer", "videomae"}:
        inputs, lengths = adapter._prepare_inputs(batch)
        raw = adapter._forward(inputs, train=gradients)
        output = adapter._output_from_raw(raw, batch, lengths=lengths)
        tokens, pooled = output.features, output.pooled
    elif encoder == "videomaev2":
        output = adapter.encode(batch, train=gradients)
        tokens, pooled = output.features, output.pooled
    else:
        raise ValueError(f"unsupported active encoder: {encoder}")
    return tokens, pooled


def _input_receipt(batch: ClipBatch) -> dict[str, Any]:
    frames = np.asarray(batch.frames)
    return {"frames_shape": list(frames.shape), "frames_dtype": str(frames.dtype),
            "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "frame_indices": np.asarray(batch.frame_indices).tolist(),
            "valid_mask": np.asarray(batch.valid_mask).tolist(),
            "timestamps_s": np.asarray(batch.timestamps_s).tolist()}


def _epoch_orders(sample_count: int) -> tuple[tuple[int, ...], ...]:
    """Use a private CPU generator; permutations do not consume global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    return tuple(tuple(torch.randperm(sample_count, generator=generator, device="cpu").tolist()) for _ in range(EPOCHS))


def train_pair_gate(
    adapter: Any, encoder: str, samples: Sequence[CalibrationSample],
    load_batch: Callable[[CalibrationSample], ClipBatch], *, output_dir: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Run exactly three seeded shuffled epochs; complete only after reload QA.

    The CLI enforces the 256-window frozen plan. This core also accepts small
    fit fixtures so tests exercise the native hook/optimizer/disk path.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    progress: dict[str, Any] = {"status": "running", "epoch": 0, "completed_steps": 0,
                              "expected_steps": EPOCHS * len(samples)}
    base = {"encoder": encoder, "identity": dict(identity), "epochs": EPOCHS, "seed": 0,
            "sample_order": "seeded_epoch_permutation", "sample_order_generator": "cpu_torch_generator_seed_0",
            "batch_size": 1, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.0,
            "initialization": "zeros", "selection": "last_epoch_no_early_stopping",
            "loss": "per_clip_relative_actual_adapter_pooled_mse", "provisional_prototype": True,
            "vad_quality_established": False, "python_executable": sys.executable,
            "vadbench_file": vadbench.__file__, "python_version": sys.version,
            "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "source_sha256": {"training": sha256_file(Path(__file__)),
                              "adapter": sha256_file(Path(inspect.getfile(type(adapter)))),
                              "pool_feature_sequence": sha256_file(Path(inspect.getfile(pool_feature_sequence))),
                              "clean_encoder_batch": sha256_file(Path(inspect.getfile(clean_encoder_batch)))}}
    atomic_write_json(output_dir / "resolved.json", base)
    atomic_write_json(output_dir / "progress.json", progress)
    try:
        if not samples or len({sample.key for sample in samples}) != len(samples):
            raise ValueError("training samples must be nonempty and unique")
        torch.manual_seed(0)
        bridge = create_observation_bridge(encoder, adapter)
        model = bridge.model
        model.eval().requires_grad_(False)
        for parameter in model.parameters():
            parameter.grad = None
        hooks_before = [(len(block._forward_hooks), len(block._forward_pre_hooks)) for block in bridge._blocks]
        first = clean_encoder_batch(load_batch(samples[0]))
        if first.batch_size != 1 or first.frame_indices is None or first.valid_mask is None or not bool(first.valid_mask.all()):
            raise ValueError("training requires complete native B=1 clips with actual frame indices")
        geometry = bridge.geometry(first.frame_indices, first.valid_mask)
        with torch.no_grad(), geometry:
            dense = adapter.encode(first, train=False)
        evidence = geometry.receipt
        if geometry.layout is None or not (evidence.get("flatten_verified") or
                (evidence.get("patch_flatten_verified") and evidence.get("divided_layout_verified"))):
            raise RuntimeError("actual native token geometry was not verified")
        # Absolute source-frame provenance is recorded per clip below. The
        # cached merge plan carries only static clip-local native geometry.
        layout = replace(geometry.layout, provenance={k: v for k, v in geometry.layout.provenance.items()
                         if k in {"coordinate_source", "grid", "kernel", "stride", "patch_stride"}})
        depth = math.ceil(bridge.receipt().block_count * 0.5) - 1
        if depth >= bridge.receipt().block_count - 1:
            raise RuntimeError("pair calibration needs a real suffix block")
        tokens = dense.features
        plan = horizontal_pair_merge_spec(encoder, layout).to(tokens.device)
        gate = PairLinearGate(tokens.shape[-1]).to(device=tokens.device, dtype=tokens.dtype)
        merger = PairWeightedMerge(tokens.shape[-1], gate)
        optimizer = torch.optim.AdamW([gate.weight], lr=1e-3, weight_decay=0.0)
        initial_weight = gate.weight.detach().clone()
        del tokens

        def merged(batch: ClipBatch, current: PairWeightedMerge, gradients: bool) -> tuple[torch.Tensor, dict[str, Any]]:
            with torch.set_grad_enabled(gradients), indexed_gather(
                bridge, depth, plan.output_indices, layout,
                transform=lambda hidden, _indices: current(hidden, plan), record_position_masks=False,
            ) as run:
                sequence, pooled = native_pooled_forward(adapter, encoder, batch, gradients=gradients)
                receipt = run.validate_execution()
            if sequence.shape[1] != plan.output_indices.shape[1]:
                raise RuntimeError("native suffix failed to execute the shortened token sequence")
            return pooled, receipt

        with torch.no_grad():
            _sequence, native_dense_pooled = native_pooled_forward(adapter, encoder, first, gradients=True)
        torch.testing.assert_close(native_dense_pooled.detach(), dense.pooled.detach(), rtol=1e-5, atol=1e-6)
        dense_parity = float((native_dense_pooled.detach() - dense.pooled.detach()).abs().max())
        with torch.no_grad(), indexed_gather(
            bridge, depth, plan.output_indices, layout,
            transform=lambda hidden, _indices: merger(hidden, plan), record_position_masks=False,
        ) as run:
            reduced_reference = adapter.encode(first, train=False).pooled.detach()
            run.validate_execution()
        reduced_native, setup_execution = merged(first, merger, True)
        torch.testing.assert_close(reduced_native.detach(), reduced_reference, rtol=1e-5, atol=1e-6)
        reduced_parity = float((reduced_native.detach() - reduced_reference).abs().max())
        del native_dense_pooled, reduced_native, reduced_reference, _sequence
        teachers = {samples[0].key: dense.pooled.detach().cpu().clone()}
        del dense
        inputs = {samples[0].key: _input_receipt(first)}
        losses: list[float] = []
        grad_l1: list[float] = []
        epoch_results = []
        for epoch, order in enumerate(_epoch_orders(len(samples))):
            epoch_losses = []
            ordered_sample_keys = [samples[index].key for index in order]
            for index in order:
                sample = samples[index]
                batch = first if epoch == 0 and sample == samples[0] else clean_encoder_batch(load_batch(sample))
                if batch.batch_size != 1 or batch.num_frames != first.num_frames or batch.frame_indices is None or batch.valid_mask is None or not bool(batch.valid_mask.all()):
                    raise ValueError("calibration sample differs from the fixed native complete input geometry")
                current_input = _input_receipt(batch)
                if sample.key in inputs and inputs[sample.key] != current_input:
                    raise ValueError("calibration input identity changed between epochs")
                inputs[sample.key] = current_input
                if sample.key not in teachers:
                    with torch.no_grad():
                        teachers[sample.key] = adapter.encode(batch, train=False).pooled.detach().cpu().clone()
                target = teachers[sample.key].to(device=gate.weight.device)
                result = gate_training_step(gate, model, optimizer, lambda batch=batch: merged(batch, merger, True)[0], target)
                losses.append(result["loss"])
                epoch_losses.append(result["loss"])
                grad_l1.append(result["grad_l1"])
                progress.update(epoch=epoch + 1, completed_steps=len(losses), current_sample=sample.key,
                                latest_loss=result["loss"], latest_grad_l1=result["grad_l1"])
                atomic_write_json(output_dir / "progress.json", progress)
            epoch_results.append({"epoch": epoch + 1, "steps": len(epoch_losses),
                                  "ordered_sample_keys": ordered_sample_keys,
                                  "mean_loss": sum(epoch_losses) / len(epoch_losses)})
            atomic_write_json(output_dir / f"epoch-{epoch + 1}.json", epoch_results[-1])
            if epoch == 0:
                torch.save(teachers, output_dir / "teacher_pooled_cache.pt")
                atomic_write_json(output_dir / "input_receipts.json", inputs)
        if len(losses) != EPOCHS * len(samples) or not any(value > 0 for value in grad_l1):
            raise RuntimeError("training did not finish every job with at least one nonzero gate gradient")
        if torch.equal(initial_weight, gate.weight.detach()):
            raise RuntimeError("gate weights did not update")
        gate_path = output_dir / "pair_linear_gate.pt"
        torch.save(gate.state_dict(), gate_path)
        reloaded = PairLinearGate(gate.weight.numel()).to(device=gate.weight.device, dtype=gate.weight.dtype)
        reloaded.load_state_dict(torch.load(gate_path, weights_only=True, map_location=gate.weight.device))
        trained_output, trained_execution = merged(first, merger, False)
        reloaded_output, reload_execution = merged(first, PairWeightedMerge(reloaded.weight.numel(), reloaded), False)
        torch.testing.assert_close(reloaded_output, trained_output, rtol=1e-5, atol=1e-6)
        if hooks_before != [(len(block._forward_hooks), len(block._forward_pre_hooks)) for block in bridge._blocks]:
            raise RuntimeError("training leaked native model hooks")
        parameter = next(model.parameters())
        receipt = {**base, "status": "completed", "steps": len(losses), "sample_count": len(samples),
                   "epoch_results": epoch_results, "losses": losses, "grad_l1": grad_l1,
                   "nonzero_gradient_steps": sum(value > 0 for value in grad_l1), "weights_changed": True,
                   "backbone_all_grad_none": all(p.grad is None for p in model.parameters()),
                   "backbone_eval": not any(module.training for module in model.modules()),
                   "backbone_all_requires_grad_false": all(not p.requires_grad for p in model.parameters()),
                   "backbone_parameter_dtype": str(parameter.dtype), "backbone_parameter_device": str(parameter.device),
                   "optimizer_only_gate": True, "teacher_detached": True,
                   "teacher_cache_sha256": sha256_file(output_dir / "teacher_pooled_cache.pt"),
                   "gate_state_sha256": sha256_file(gate_path), "gate_state_dict": gate_path.name,
                   "gate_dtype": str(gate.weight.dtype), "gate_device": str(gate.weight.device),
                   "reload_max_absolute_error": float((trained_output - reloaded_output).abs().max()),
                   "adapter_pooled_parity": {"dense_max_absolute_error": dense_parity,
                                            "reduced_max_absolute_error": reduced_parity, "rtol": 1e-5, "atol": 1e-6},
                   "depth": depth, "architecture": bridge.architecture(), "geometry": evidence,
                   "pair_plan": plan.receipt(), "setup_execution": setup_execution,
                   "trained_execution": trained_execution, "reload_execution": reload_execution}
        atomic_write_json(output_dir / "receipt.json", receipt)
        atomic_write_json(output_dir / "progress.json", {**progress, "status": "completed"})
        return receipt
    except BaseException as error:
        failure = {**base, **progress, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
        atomic_write_json(output_dir / "receipt.json", failure)
        atomic_write_json(output_dir / "progress.json", failure)
        raise
