"""One-fit-video engineering QA for the active pair-merge path.

This is deliberately not a detector experiment.  It verifies native geometry,
real shortened suffix execution, frozen-backbone gradients into a local gate,
and checkpoint reloadability on one centre clip from a frozen ``fit`` video.
It reports neither VAD quality nor speed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_neutral_pilot import ROLE_LOCK_SHA256

import vadbench
from vadbench.artifacts import new_run_id
from vadbench.checkpoints import sha256_file
from vadbench.data.sampling import sample_fixed_clip
from vadbench.data.video import build_clip_batch, probe_video
from vadbench.features import atomic_write_json
from vadbench.orchestration import encoder_identity
from vadbench.paper.profile import load_project
from vadbench.paper.stages import clean_encoder_batch
from vadbench.registry import ENCODER_REGISTRY
from vadbench.token_reduction.bridges import (
    create_observation_bridge,
    identity_indices,
    indexed_gather,
)
from vadbench.token_reduction.pair_merge import (
    PairLinearGate,
    PairWeightedMerge,
    horizontal_pair_merge_spec,
)

ROOT = Path(__file__).resolve().parents[2]
ROLE_LOCK = ROOT / "outputs/icassp2027/assets/frozen192-verified-20260917T200000Z/frozen-partition-lock.json"
DEFAULT_VIDEO = ROOT / "data/ucf-debug-mirror/Abuse/Abuse005_x264.mp4"
ENCODERS = ("videomaev2", "timesformer", "videomae", "vjepa2")
STEPS = 2


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(value))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _sequence(value: Any) -> torch.Tensor:
    for name in ("last_hidden_state", "hidden_states", "features", "tokens"):
        candidate = getattr(value, name, None)
        if candidate is not None:
            value = candidate
            break
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise RuntimeError(
            "native pair-merge QA requires a [B,N,D] token sequence, "
            f"got {getattr(value, 'shape', None)!r}"
        )
    return value


def _geometry_verified(geometry: Any) -> bool:
    receipt = geometry.receipt
    return bool(receipt.get("flatten_verified")) or bool(
        receipt.get("patch_flatten_verified") and receipt.get("divided_layout_verified")
    )


def _source_hashes() -> dict[str, str]:
    from vadbench.token_reduction import pair_merge
    from vadbench.token_reduction.bridges import indexed

    paths = {
        "verify_pair_merge_active": Path(__file__),
        "indexed_bridge": Path(indexed.__file__),
        "pair_merge": Path(pair_merge.__file__),
    }
    return {name: sha256_file(path.resolve()) for name, path in paths.items()}


def _fit_video_id(video: Path, lock_path: Path | None = None) -> str:
    lock_path = ROLE_LOCK if lock_path is None else lock_path
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing frozen role lock: {lock_path}")
    if sha256_file(lock_path) != ROLE_LOCK_SHA256:
        raise ValueError("frozen role lock SHA-256 differs from the authorized lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    partitions = lock.get("partitions") if isinstance(lock, dict) else None
    video_id = video.stem
    if not isinstance(partitions, dict) or partitions.get(video_id) != "fit":
        raise ValueError(f"pair-merge engineering QA only permits frozen fit videos: {video_id}")
    return video_id


def _native_forward(
    adapter: Any, encoder: str, batch: Any, *, gradients: bool
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Run the native path and the narrowest differentiable adapter readout."""

    if encoder == "vjepa2":
        if gradients:
            tokens = _sequence(adapter.encode_with_grad(batch))
        else:
            worker = adapter.bridge.encoder
            tokens = _sequence(worker.encode(batch))
        # V-JEPA's public grad route intentionally exposes raw tokens, before
        # its Foundation adapter normalizer/pooling.  Its engineering loss is
        # therefore explicitly native mean; adapter.encode parity is recorded
        # separately below and is never presented as detector distillation.
        return tokens, _pooled(tokens), "native_token_mean_engineering_only"
    if encoder in {"timesformer", "videomae"}:
        inputs, lengths = adapter._prepare_inputs(batch)
        raw = adapter._forward(inputs, train=gradients)
        tokens = _sequence(raw)
        output = adapter._output_from_raw(raw, batch, lengths=lengths)
        return tokens, output.pooled, "actual_adapter_pooled"
    if encoder == "videomaev2":
        output = adapter.encode(batch, train=gradients)
        return _sequence(output.features), output.pooled, "actual_adapter_pooled"
    raise ValueError(f"unknown active encoder={encoder!r}")


def _freeze(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _pooled(tokens: torch.Tensor) -> torch.Tensor:
    """A fixed native-token mean readout for this isolated engineering loss."""

    return tokens.mean(dim=1)


def _relative_pooled_mse(prediction: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    numerator = (prediction.float() - teacher.float()).square().sum()
    denominator = teacher.float().square().sum()
    if not bool(torch.isfinite(denominator)) or denominator.item() <= 0:
        raise RuntimeError("dense teacher pooled norm must be finite and positive")
    return numerator / denominator


def _block_depth(bridge: Any) -> int:
    depth = max(0, math.ceil(bridge.receipt().block_count * 0.5) - 1)
    if depth >= bridge.receipt().block_count - 1:
        raise RuntimeError("mid-depth pair merge needs at least one native suffix block")
    return depth


def _run_pair_merge_qa(
    adapter: Any, encoder: str, batch: Any, *, gate_path: Path
) -> dict[str, Any]:
    """Run dense, identity, pair mean, two frozen-backbone gate steps, and reload."""

    clean = clean_encoder_batch(batch)
    if clean.frame_indices is None or clean.valid_mask is None:
        raise ValueError("pair-merge QA requires explicit frame_indices and valid_mask")
    bridge = create_observation_bridge(encoder, adapter)
    model = bridge.model
    _freeze(model)
    initial_hooks = [
        (len(block._forward_hooks), len(block._forward_pre_hooks)) for block in bridge._blocks
    ]
    geometry = bridge.geometry(clean.frame_indices, clean.valid_mask)
    with torch.no_grad(), geometry:
        dense, teacher, readout = _native_forward(adapter, encoder, clean, gradients=False)
    if geometry.layout is None or not _geometry_verified(geometry):
        raise RuntimeError("native geometry did not verify its actual flatten/layout")
    layout = geometry.layout
    depth = _block_depth(bridge)
    spec = horizontal_pair_merge_spec(encoder, layout).to(dense.device)
    if spec.output_indices.shape[1] >= layout.token_capacity:
        raise RuntimeError("pair plan did not shorten the native suffix")
    teacher = teacher.detach()
    with torch.no_grad():
        adapter_reference = adapter.encode(clean, train=False).pooled.detach()
    native_mean = _pooled(dense)
    identity = identity_indices(layout)
    with torch.no_grad(), indexed_gather(
        bridge, depth, identity, layout, record_position_masks=False
    ) as run:
        identity_output, identity_pooled, _identity_readout = _native_forward(
            adapter, encoder, clean, gradients=False
        )
        identity_receipt = run.validate_execution()
    torch.testing.assert_close(identity_output, dense, rtol=1e-5, atol=1e-6)

    mean = PairWeightedMerge(dense.shape[-1]).to(device=dense.device, dtype=dense.dtype)

    def merged_forward(
        merger: PairWeightedMerge, *, gradients: bool
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        context = torch.enable_grad() if gradients else torch.no_grad()
        with context, indexed_gather(
            bridge,
            depth,
            spec.output_indices,
            layout,
            transform=lambda hidden, _indices: merger(hidden, spec),
            record_position_masks=False,
        ) as run:
            output, pooled, _output_readout = _native_forward(
                adapter, encoder, clean, gradients=gradients
            )
            receipt = run.validate_execution()
        if output.shape[1] != spec.output_indices.shape[1]:
            raise RuntimeError("native suffix did not retain the pair-merge token length")
        return output, pooled, receipt

    pair_mean, pair_mean_pooled, pair_mean_receipt = merged_forward(mean, gradients=False)
    gate = PairLinearGate(dense.shape[-1]).to(device=dense.device, dtype=dense.dtype)
    weighted = PairWeightedMerge(dense.shape[-1], gate).to(
        device=dense.device, dtype=dense.dtype
    )
    zero_gate, zero_gate_pooled, zero_gate_receipt = merged_forward(weighted, gradients=False)
    torch.testing.assert_close(zero_gate, pair_mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(zero_gate_pooled, pair_mean_pooled, rtol=1e-5, atol=1e-6)

    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=0.0)
    before = gate.weight.detach().clone()
    losses: list[float] = []
    gradients: list[float] = []
    for _step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        _output, pooled, _receipt = merged_forward(weighted, gradients=True)
        loss = _relative_pooled_mse(pooled, teacher)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("pair gate loss is not finite")
        loss.backward()
        gradient = gate.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()) or gradient.abs().sum() <= 0:
            raise RuntimeError("pair linear gate did not receive a finite non-zero gradient")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen backbone unexpectedly accumulated gradients")
        losses.append(float(loss.detach().cpu()))
        gradients.append(float(gradient.abs().sum().detach().cpu()))
        optimizer.step()
    if torch.equal(before, gate.weight.detach()):
        raise RuntimeError("pair linear gate weights did not change after AdamW")

    with torch.no_grad():
        trained_output, trained_pooled, trained_receipt = merged_forward(weighted, gradients=False)
    torch.save(gate.state_dict(), gate_path)
    reloaded_gate = PairLinearGate(dense.shape[-1]).to(device=dense.device, dtype=dense.dtype)
    reloaded_gate.load_state_dict(
        torch.load(gate_path, weights_only=True, map_location=dense.device)
    )
    reloaded = PairWeightedMerge(dense.shape[-1], reloaded_gate).to(
        device=dense.device, dtype=dense.dtype
    )
    with torch.no_grad():
        reloaded_output, reloaded_pooled, reload_receipt = merged_forward(
            reloaded, gradients=False
        )
    torch.testing.assert_close(reloaded_output, trained_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reloaded_pooled, trained_pooled, rtol=1e-5, atol=1e-6)
    final_hooks = [
        (len(block._forward_hooks), len(block._forward_pre_hooks)) for block in bridge._blocks
    ]
    if initial_hooks != final_hooks:
        raise RuntimeError("pair-merge contexts leaked native hooks")
    parameter = next(model.parameters(), None)
    return {
        "engineering_only_one_fit_center_clip": True,
        "not_a_trained_plugin_or_detector_claim": True,
        "loss_readout": f"{readout}; dense teacher detached; relative pooled MSE",
        "bridge": bridge.architecture(),
        "geometry": geometry.receipt,
        "depth": depth,
        "dense": {"shape": list(dense.shape), "dtype": str(dense.dtype), "device": str(dense.device)},
        "adapter_readout_parity": {
            "adapter_pooled_shape": list(adapter_reference.shape),
            "native_mean_shape": list(native_mean.shape),
            "adapter_vs_native_mean_max_absolute_error": float(
                (adapter_reference - native_mean).abs().max().detach().cpu()
            ),
            "teacher_vs_adapter_pooled_max_absolute_error": float(
                (teacher - adapter_reference).abs().max().detach().cpu()
            ),
        },
        "identity": {
            "max_absolute_error": float((identity_output - dense).abs().max().detach().cpu()),
            "pooled_max_absolute_error": float(
                (identity_pooled - teacher).abs().max().detach().cpu()
            ),
            "receipt": identity_receipt,
        },
        "pair_mean": {"shape": list(pair_mean.shape), "receipt": pair_mean_receipt},
        "zero_gate": {
            "max_absolute_error_to_mean": float((zero_gate - pair_mean).abs().max().detach().cpu()),
            "receipt": zero_gate_receipt,
        },
        "pair_plan": spec.receipt(),
        "gate": {
            "steps": STEPS,
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "losses": losses,
            "losses_finite": all(math.isfinite(value) for value in losses),
            "grad_l1": gradients,
            "gradients_finite_nonzero": all(math.isfinite(value) and value > 0 for value in gradients),
            "weights_changed": True,
            "initial_weight": before.detach().float().cpu().tolist(),
            "final_weight": gate.weight.detach().float().cpu().tolist(),
            "weight_dtype": str(gate.weight.dtype),
            "weight_device": str(gate.weight.device),
            "state_dict": "pair_linear_gate.pt",
            "state_dict_sha256": sha256_file(gate_path),
            "reload_max_absolute_error": float(
                (reloaded_output - trained_output).abs().max().detach().cpu()
            ),
            "reload_receipt": reload_receipt,
            "trained_receipt": trained_receipt,
        },
        "backbone": {
            "eval": not model.training,
            "all_requires_grad_false": all(not parameter.requires_grad for parameter in model.parameters()),
            "all_grad_none": all(parameter.grad is None for parameter in model.parameters()),
            "parameter": None
            if parameter is None
            else {"dtype": str(parameter.dtype), "device": str(parameter.device)},
        },
        "initial_hook_counts": initial_hooks,
        "final_hook_counts": final_hooks,
        "run_batch": {
            "video_ids": list(batch.video_ids),
            "frame_indices": None
            if batch.frame_indices is None
            else np.asarray(batch.frame_indices).tolist(),
            "valid_mask": None if batch.valid_mask is None else np.asarray(batch.valid_mask).tolist(),
            "frames_shape": list(batch.frames.shape),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "projects/icassp2027/profile.yaml")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    device = torch.device(args.device)
    if args.cuda_memory_fraction is not None:
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError("--cuda-memory-fraction must be in (0, 1]")
        if device.type != "cuda":
            raise ValueError("--cuda-memory-fraction requires a CUDA device")
    project = load_project(args.profile)
    definition = dict(project.encoder(args.encoder)["definition"])
    constructor = dict(definition["constructor"])
    definition["constructor"] = constructor
    video = args.video.resolve()
    video_id = _fit_video_id(video)
    frames = constructor.get("num_frames", constructor.get("clip_frames"))
    if type(frames) is not int or frames <= 0:
        raise ValueError("active encoder constructor must declare positive native clip frames")
    checkpoint = Path(definition["checkpoint"]["local_path"])
    plan = {
        "status": "planned" if not args.execute else "running",
        "engineering_only_one_fit_center_clip": True,
        "not_a_trained_plugin_or_detector_claim": True,
        "not_quality_or_speed_measurement": True,
        "encoder": args.encoder,
        "profile": str(project.path),
        "video": str(video),
        "video_id": video_id,
        "partition": "fit",
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "role_lock": str(ROLE_LOCK),
        "role_lock_sha256": ROLE_LOCK_SHA256,
        "clip_frames": frames,
        "frame_stride": 2,
        "position": "center",
        "steps": STEPS,
        "requested_device": str(device),
        "threads": args.threads,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "execute": bool(args.execute),
    }
    if not args.execute:
        print(_json(plan))
        return 0
    output = (
        args.output.resolve()
        if args.output is not None
        else (project.root / "outputs/icassp2027/pair-merge-active" / new_run_id(args.encoder)).resolve()
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    receipt_path = output / "receipt.json"
    try:
        if not checkpoint.exists() or not video.is_file():
            raise FileNotFoundError(_json(plan))
        if args.cuda_memory_fraction is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
        torch.set_num_threads(args.threads)
        info = probe_video(video)
        sample = sample_fixed_clip(info.num_frames, clip_frames=frames, frame_stride=2, position="center")
        batch = build_clip_batch(video, video_id, [sample])
        constructor["device"] = str(device)
        identity = encoder_identity(definition, project_root=project.root)
        adapter = ENCODER_REGISTRY.create(args.encoder, **constructor)
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        result = _run_pair_merge_qa(
            adapter, args.encoder, batch, gate_path=output / "pair_linear_gate.pt"
        )
        receipt = {
            **plan,
            "status": "completed",
            "output": str(output),
            "python_executable": sys.executable,
            "vadbench_file": vadbench.__file__,
            "torch": torch.__version__,
            "source_sha256": _source_hashes(),
            "encoder_identity": identity,
            "gate_state_sha256": sha256_file(output / "pair_linear_gate.pt"),
            "video_sha256": sha256_file(video),
            "video_info": {"frames": info.num_frames, "fps": info.fps, "height": info.height, "width": info.width},
            "sampled_frame_indices": list(sample.frame_indices),
            "sampled_valid_mask": list(sample.valid_mask),
            "input_shape": list(batch.frames.shape),
            "result": result,
        }
    except Exception as error:
        _write(
            receipt_path,
            {
                **plan,
                "status": "failed",
                "output": str(output),
                "failure": "cuda_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else type(error).__name__,
                "detail": str(error),
            },
        )
        raise
    _write(receipt_path, receipt)
    print(_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
