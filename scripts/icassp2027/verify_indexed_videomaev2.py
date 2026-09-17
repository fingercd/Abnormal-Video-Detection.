"""One-clip CPU verification of native VideoMAEv2 indexed gather.

This script is an engineering receipt only.  Its even-index gather has no
anomaly score, label, selector, efficiency claim, or research conclusion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import vadbench
from vadbench.data.sampling import sample_fixed_clip
from vadbench.data.video import build_clip_batch, probe_video
from vadbench.hashing import sha256_file
from vadbench.integrations.videomaev2_encoder import VideoMAEv2Encoder, VideoMAEv2EncoderConfig
from vadbench.token_reduction.bridges import (
    create_observation_bridge,
    identity_indices,
    indexed_gather,
)

ROOT = Path(__file__).resolve().parents[2]


def _array(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _max_abs(left: Any, right: Any) -> float:
    left_array, right_array = _array(left), _array(right)
    if left_array.shape != right_array.shape:
        raise RuntimeError(f"parity shape differs: {left_array.shape} != {right_array.shape}")
    return float(np.max(np.abs(left_array - right_array)))


def _clean_hooks(bridge: Any) -> bool:
    return all(not block._forward_hooks and not block._forward_pre_hooks for block in bridge._blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "data/ucf-debug-mirror/Abuse/Abuse005_x264.mp4",
    )
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "weights/videomaev2-base-hf"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/icassp2027/verify-indexed-videomaev2.json",
    )
    parser.add_argument("--depth", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    video = args.video.resolve()
    weights = args.weights.resolve()
    if not video.is_file() or not weights.is_dir():
        raise FileNotFoundError(f"video={video}; weights={weights}")

    encoder = VideoMAEv2Encoder(
        VideoMAEv2EncoderConfig(
            model_name=str(weights), image_size=224, num_frames=16, use_half=False, pooling="auto"
        ),
        device="cpu",
    )
    encoder.freeze_backbone()
    encoder.eval()
    if any(parameter.requires_grad for parameter in encoder.backbone.parameters()):
        raise RuntimeError("freeze_backbone failed")

    info = probe_video(video)
    sample = sample_fixed_clip(info.num_frames, clip_frames=16, frame_stride=2, position="center")
    batch = build_clip_batch(video, "Abuse005_x264", [sample])
    clips = [
        [np.asarray(frame, dtype=np.uint8) for frame in batch.frames[row, : int(length)]]
        for row, length in enumerate(batch.valid_lengths)
    ]
    pixel_values = encoder._tensor_from_rgb_lists(clips).float().to("cpu")
    bridge = create_observation_bridge("videomaev2", encoder)
    geometry = bridge.geometry(batch.frame_indices, batch.valid_mask)

    # Forward 1: native dense execution plus runtime geometry proof.
    with torch.no_grad(), geometry:
        dense = encoder.backbone(pixel_values=pixel_values)
    if geometry.layout is None or not geometry.receipt.get("flatten_verified"):
        raise RuntimeError("native VideoMAEv2 geometry was not verified")
    layout = geometry.layout
    native_pooled = encoder._pool(dense)

    # Forward 2: no-op indexed intervention must preserve native output.
    with torch.no_grad(), indexed_gather(bridge, args.depth, identity_indices(layout), layout) as identity:
        identity_output = encoder.backbone(pixel_values=pixel_values)
        identity_receipt = identity.validate_execution()
    identity_pooled = encoder._pool(identity_output)

    # Forward 3: fixed, external even original indices preserve neither a score
    # nor a learned selector.  It only proves native suffix shape shortening.
    kept = torch.arange(0, layout.token_capacity, 2, dtype=torch.long).unsqueeze(0)
    with torch.no_grad(), indexed_gather(bridge, args.depth, kept, layout) as reduced:
        reduced_output = encoder.backbone(pixel_values=pixel_values)
        reduced_receipt = reduced.validate_execution()
    reduced_pooled = encoder._pool(reduced_output)

    # Forward 4: frozen backbone still allows a leaf multiplier to receive a
    # gradient through the direct native route and gathered suffix.
    multiplier = torch.ones((), dtype=pixel_values.dtype, requires_grad=True)
    with indexed_gather(bridge, args.depth, kept, layout) as gradient_run:
        gradient_output = encoder.backbone(pixel_values=pixel_values * multiplier)
        gradient_loss = gradient_output.square().mean()
        gradient_loss.backward()
        gradient_receipt = gradient_run.validate_execution()
    if multiplier.grad is None or not torch.isfinite(multiplier.grad):
        raise RuntimeError("leaf multiplier did not receive a finite gradient")

    # Context cleanup is checked without an extra model forward.
    try:
        with indexed_gather(bridge, args.depth, kept, layout):
            raise RuntimeError("intentional cleanup check")
    except RuntimeError as error:
        if str(error) != "intentional cleanup check":
            raise
    if not _clean_hooks(bridge):
        raise RuntimeError("indexed intervention leaked block hooks")

    model_file = weights / "model.safetensors"
    receipt = {
        "kind": "single_clip_real_weight_indexed_gather_verification",
        "scope": "engineering shape/position/gradient verification; no anomaly or efficiency conclusion",
        "python_executable": sys.executable,
        "vadbench_file": vadbench.__file__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "checkpoint": {
            "path": str(weights),
            "model_safetensors_sha256": sha256_file(model_file),
            "config_sha256": sha256_file(weights / "config.json"),
            "modeling_source_sha256": sha256_file(weights / "modeling_videomaev2.py"),
        },
        "input": {
            "video": str(video),
            "video_bytes": video.stat().st_size,
            "video_info": {"frames": info.num_frames, "fps": info.fps, "height": info.height, "width": info.width},
            "sample": {"frame_indices": list(sample.frame_indices), "valid_mask": list(sample.valid_mask)},
            "clip_batch_shape": list(batch.frames.shape),
            "pixel_values_shape": list(pixel_values.shape),
        },
        "bridge": bridge.architecture(),
        "geometry": geometry.receipt,
        "frozen_backbone": True,
        "dense_output_shape": list(dense.shape),
        "native_pooled_shape": list(native_pooled.shape),
        "identity": {
            "receipt": identity_receipt,
            "output_max_abs": _max_abs(identity_output, dense),
            "pooled_max_abs": _max_abs(identity_pooled, native_pooled),
        },
        "fixed_even_index_gather": {
            "depth": args.depth,
            "kept_token_count": int(kept.shape[1]),
            "first_indices": kept[0, :12].tolist(),
            "receipt": reduced_receipt,
            "native_pooled_shape": list(reduced_pooled.shape),
        },
        "gradient": {
            "direct_native_route": True,
            "leaf_multiplier_grad": float(multiplier.grad.detach().cpu()),
            "loss": float(gradient_loss.detach().cpu()),
            "receipt": gradient_receipt,
            "backbone_parameters_require_grad": False,
        },
        "exception_cleanup_hooks_clear": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
