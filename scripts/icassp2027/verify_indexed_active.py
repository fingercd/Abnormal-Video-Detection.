"""Explicit-index gather verification for one active real-weight clip.

Run without ``--execute`` to inspect only the profile, assets, and deterministic
clip request.  Execution uses no selector: it checks identity, a fixed external
half-budget index set, and a zero-initialized external additive bias gradient.
It never reports quality or speed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

import vadbench
from vadbench.data.sampling import sample_fixed_clip
from vadbench.data.video import build_clip_batch, probe_video
from vadbench.orchestration import encoder_identity
from vadbench.paper.profile import load_project
from vadbench.registry import ENCODER_REGISTRY
from vadbench.token_reduction.bridges import (
    create_observation_bridge,
    identity_indices,
    indexed_gather,
)

ROOT = Path(__file__).resolve().parents[2]


def _sequence(output: Any) -> torch.Tensor:
    value = getattr(output, "last_hidden_state", output)
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise RuntimeError(f"native direct output must expose [B,N,D], got {getattr(value, 'shape', None)}")
    return value


def _fixed_indices(encoder: str, layout: Any) -> torch.Tensor:
    if encoder != "timesformer":
        return torch.arange(0, layout.token_capacity, 2, dtype=torch.long).unsqueeze(0)
    grid = layout.provenance.get("grid")
    if not isinstance(grid, list) or len(grid) != 3:
        raise RuntimeError("TimeSformer verified geometry lacks [T,H,W]")
    frames, _height, width = grid
    spatial_total = (layout.token_capacity - 1) // frames
    retained_spatial = spatial_total // 2
    retained_spatial -= retained_spatial % width
    if retained_spatial <= 0:
        raise RuntimeError("TimeSformer half budget cannot retain a complete native spatial row")
    indices = [0]
    for spatial in range(retained_spatial):
        indices.extend(1 + spatial * frames + time for time in range(frames))
    return torch.tensor([indices], dtype=torch.long)


def _components(
    adapter: Any, encoder: str, batch: Any
) -> tuple[torch.nn.Module, Callable[[], Any]]:
    if encoder in {"timesformer", "videomae"}:
        inputs, _ = adapter._prepare_inputs(batch)
        return adapter.model, lambda: adapter.model(**dict(inputs))
    # The published V-JEPA adapter's worker intentionally uses no_grad.  The
    # verification route calls the same loaded native model directly so an
    # external leaf-bias gradient can be checked without claiming adapter
    # training support.
    worker = adapter.bridge.encoder
    model = worker.model
    inputs = worker._prepare_inputs(batch)
    if not isinstance(inputs, Mapping):
        raise RuntimeError("V-JEPA worker preprocessing must return a mapping for get_vision_features")
    prepared = dict(inputs)
    return model, lambda: model.get_vision_features(**prepared)


def _geometry_verified(geometry: Any) -> bool:
    receipt = geometry.receipt
    return bool(receipt.get("flatten_verified")) or bool(
        receipt.get("patch_flatten_verified") and receipt.get("divided_layout_verified")
    )


def _verify_native(
    *, model: torch.nn.Module, forward: Callable[[], Any], encoder: str, frame_indices: Any,
    valid_mask: Any, depth: int,
) -> dict[str, Any]:
    """Run exactly four native forwards and return identity/shape/gradient evidence."""

    model.eval()
    bridge = create_observation_bridge(encoder, model)
    initial_hooks = [
        (len(block._forward_hooks), len(block._forward_pre_hooks)) for block in bridge._blocks
    ]
    geometry = bridge.geometry(frame_indices, valid_mask)
    with torch.no_grad(), geometry:
        dense = _sequence(forward())
    if geometry.layout is None or not _geometry_verified(geometry):
        raise RuntimeError("native geometry did not complete its verified flatten/layout condition")
    layout = geometry.layout
    dense_pooled = dense.mean(dim=1)
    with torch.no_grad(), indexed_gather(bridge, depth, identity_indices(layout), layout) as run:
        identity = _sequence(forward())
        identity_receipt = run.validate_execution()
    identity_pooled = identity.mean(dim=1)
    torch.testing.assert_close(identity, dense, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(identity_pooled, dense_pooled, rtol=1e-5, atol=1e-6)
    kept = _fixed_indices(encoder, layout)
    with torch.no_grad(), indexed_gather(bridge, depth, kept, layout) as run:
        reduced = _sequence(forward())
        reduced_receipt = run.validate_execution()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bias = torch.zeros(dense.shape[-1], dtype=dense.dtype, requires_grad=True)

    def add_external_bias(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = bridge.block_output_tensor(depth, output)
        return bridge.replace_block_output_tensor(
            depth, output, hidden + bias, allow_sequence_shrink=True
        )

    with indexed_gather(bridge, depth, kept, layout) as run:
        handle = bridge._blocks[depth].register_forward_hook(add_external_bias)
        try:
            biased = _sequence(forward())
        finally:
            handle.remove()
        loss = biased.mean(dim=1)[:, 0].sum()
        loss.backward()
        gradient_receipt = run.validate_execution()
    if bias.grad is None or not torch.isfinite(bias.grad).all() or bias.grad.abs().sum() <= 0:
        raise RuntimeError("zero-initialized external bias did not receive a non-zero finite gradient")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("frozen native backbone unexpectedly accumulated parameter gradients")
    final_hooks = [
        (len(block._forward_hooks), len(block._forward_pre_hooks)) for block in bridge._blocks
    ]
    if final_hooks != initial_hooks:
        raise RuntimeError("indexed contexts did not restore the initial hook counts")
    return {
        "bridge": bridge.architecture(),
        "geometry": geometry.receipt,
        "dense_shape": list(dense.shape),
        "dense_mean_pooled_shape": list(dense_pooled.shape),
        "identity_max_abs": float((identity - dense).abs().max().cpu()),
        "identity_mean_pooled_max_abs": float((identity_pooled - dense_pooled).abs().max().cpu()),
        "identity_receipt": identity_receipt,
        "fixed_indices": kept[0, : min(16, kept.shape[1])].tolist(),
        "reduced_shape": list(reduced.shape),
        "reduced_receipt": reduced_receipt,
        "leaf_bias_shape": list(bias.shape),
        "leaf_bias_grad_l1": float(bias.grad.abs().sum().cpu()),
        "pooled_first_dimension_loss": float(loss.detach().cpu()),
        "gradient_receipt": gradient_receipt,
        "initial_hook_counts": initial_hooks,
        "final_hook_counts": final_hooks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("timesformer", "videomae", "vjepa2"), required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "projects/icassp2027/profile.yaml")
    parser.add_argument(
        "--video", type=Path, default=ROOT / "data/ucf-debug-mirror/Abuse/Abuse005_x264.mp4"
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = load_project(args.profile)
    definition = project.encoder(args.encoder)["definition"]
    constructor = dict(definition["constructor"])
    video = args.video.resolve()
    weight_path = Path(definition["checkpoint"]["local_path"])
    frames = int(constructor.get("clip_frames", constructor.get("num_frames")))
    plan = {
        "encoder": args.encoder,
        "profile": str(project.path),
        "checkpoint_path": str(weight_path),
        "checkpoint_exists": weight_path.exists(),
        "video": str(video),
        "video_exists": video.is_file(),
        "clip_frames": frames,
        "frame_stride": 2,
        "position": "center",
        "depth": args.depth,
        "execution": bool(args.execute),
        "scope": "identity/external-index/leaf-bias engineering verification only",
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if not video.is_file() or not weight_path.exists():
        raise FileNotFoundError(json.dumps(plan, ensure_ascii=False))

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    info = probe_video(video)
    sample = sample_fixed_clip(info.num_frames, clip_frames=frames, frame_stride=2, position="center")
    batch = build_clip_batch(video, "indexed-active", [sample])
    constructor["device"] = "cpu"
    adapter = ENCODER_REGISTRY.create(args.encoder, **constructor)
    model, forward = _components(adapter, args.encoder, batch)
    verification = _verify_native(
        model=model,
        forward=forward,
        encoder=args.encoder,
        frame_indices=batch.frame_indices,
        valid_mask=batch.valid_mask,
        depth=args.depth,
    )

    output = (
        args.output
        if args.output is not None
        else ROOT / "outputs/icassp2027/verify-indexed-active" / f"{args.encoder}.json"
    )
    receipt = {
        **plan,
        "python_executable": sys.executable,
        "vadbench_file": vadbench.__file__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "verified_encoder_identity": encoder_identity(definition, project_root=project.root),
        "video_info": {"frames": info.num_frames, "fps": info.fps, "height": info.height, "width": info.width},
        "sampled_frame_indices": list(sample.frame_indices),
        "input_shape": list(batch.frames.shape),
        **verification,
        "adapter_training_claim": False if args.encoder == "vjepa2" else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
