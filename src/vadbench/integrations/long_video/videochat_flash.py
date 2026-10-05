"""Native VideoChat-Flash projected-visual adapter.

The adapter uses the official Hugging Face custom-code snapshot and loads only
``model.vision_tower`` plus ``model.mm_projector`` from the published 2B
checkpoint.  The language-model weights are deliberately not materialised:
this route exports the exact projected visual representation used immediately
before the decoder, with upstream visual compression disabled at the LLM
level.  It does not claim streaming or decoder-KV support.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from vadbench.contracts import ClipBatch, EncoderCapabilities, EncoderOutput

from .base import ExternalFixedVideoAdapter, LongVideoAssetError


def _load_source_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 VideoChat-Flash 官方模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prefixed_state(module: Any, checkpoint: Path, prefix: str) -> int:
    from safetensors import safe_open

    state: dict[str, Any] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        checkpoint_keys = handle.keys()
        for key in checkpoint_keys:
            if key.startswith(prefix):
                state[key[len(prefix) :]] = handle.get_tensor(key)
    if not state:
        raise RuntimeError(f"VideoChat-Flash checkpoint 不含前缀 {prefix!r}")
    result = module.load_state_dict(state, strict=False)
    missing = tuple(getattr(result, "missing_keys", ()) or ())
    unexpected = tuple(getattr(result, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise RuntimeError(
            f"VideoChat-Flash {prefix} state mismatch: "
            f"missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}"
        )
    return len(state)


class _NativeVideoChatFlashWorker:
    native_upstream = True

    def __init__(
        self,
        vision_tower: Any,
        projector: Any,
        *,
        device: str,
        num_frames: int,
        local_num_frames: int,
        checkpoint_path: Path,
        vision_state_keys: int,
        projector_state_keys: int,
    ) -> None:
        self.vision_tower = vision_tower
        self.projector = projector
        self.device = device
        self.num_frames = int(num_frames)
        self.local_num_frames = int(local_num_frames)
        self.checkpoint_path = checkpoint_path
        self.vision_state_keys = int(vision_state_keys)
        self.projector_state_keys = int(projector_state_keys)

    @staticmethod
    def _pad_or_sample(frames: np.ndarray, target: int) -> np.ndarray:
        if frames.shape[0] == target:
            return frames
        if frames.shape[0] > target:
            indices = np.linspace(0, frames.shape[0] - 1, num=target).round().astype(np.int64)
            return frames[indices]
        return np.concatenate((frames, np.repeat(frames[-1:], target - frames.shape[0], axis=0)))

    def encode(self, batch: ClipBatch, **_: Any) -> dict[str, Any]:
        import torch

        processor = self.vision_tower.image_processor
        rows = []
        for row, length in enumerate(batch.valid_lengths):
            source = np.asarray(batch.frames[row, : int(length)], dtype=np.uint8)
            source = self._pad_or_sample(source, self.num_frames)
            pixels = processor.preprocess(list(source), return_tensors="pt")["pixel_values"]
            rows.append(pixels)
        pixel_values = torch.stack(rows, dim=0).to(self.device)
        batch_size, frames, channels, height, width = pixel_values.shape
        groups = frames // self.local_num_frames
        grouped = pixel_values.reshape(
            batch_size * groups,
            self.local_num_frames,
            channels,
            height,
            width,
        )
        with torch.no_grad():
            visual = self.vision_tower(grouped)
            outputs = []
            for row in range(batch_size):
                per_video = visual[row * groups : (row + 1) * groups]
                per_frame = per_video.reshape(
                    -1,
                    per_video.shape[-2] // self.local_num_frames,
                    per_video.shape[-1],
                )
                projected = self.projector(
                    per_frame,
                    compress=True,
                    local_num_frames=self.local_num_frames,
                )
                outputs.append(projected.flatten(0, 1))
            features = torch.stack(outputs, dim=0)
        return {
            "features": features,
            "pooled": features.mean(dim=1),
            "aux": {
                "native_route_available": True,
                "implementation_source": "native_upstream",
                "checkpoint_path": str(self.checkpoint_path),
                "vision_state_keys": self.vision_state_keys,
                "projector_state_keys": self.projector_state_keys,
                "input_frames": self.num_frames,
                "local_num_frames": self.local_num_frames,
                "visual_token_count": int(features.shape[1]),
                "projector_dimension": int(features.shape[2]),
                "mm_llm_compress": False,
            },
        }


def load_videochat_flash(
    model_path: str | Path,
    device: str = "cpu",
    num_frames: int = 32,
    **_: Any,
) -> _NativeVideoChatFlashWorker:
    root = Path(model_path).expanduser().resolve()
    checkpoint = root / "model.safetensors"
    required = (
        checkpoint,
        root / "config.json",
        root / "vision_tower_builder.py",
        root / "mm_projector_builder.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise LongVideoAssetError(
            integration_id="videochat_flash",
            code="missing_asset",
            message="VideoChat-Flash 官方 HF snapshot 不完整",
            details={"missing": missing},
        )
    config_data = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config_data["mm_llm_compress"] = False
    config = SimpleNamespace(**config_data)
    local_num_frames = int(config.mm_local_num_frames)
    if num_frames <= 0 or num_frames % local_num_frames:
        raise ValueError(
            f"VideoChat-Flash num_frames={num_frames} 必须是 mm_local_num_frames={local_num_frames} 的正整数倍"
        )
    vision_source = _load_source_module(
        root / "vision_tower_builder.py", "vadbench_videochat_flash_vision"
    )
    projector_source = _load_source_module(
        root / "mm_projector_builder.py", "vadbench_videochat_flash_projector"
    )
    vision_tower = vision_source.build_vision_tower(config)
    projector = projector_source.build_vision_projector(config, vision_cfg=vision_tower.config)
    if str(device).startswith("cpu"):
        # The official builder selects FlashAttention when the optional package
        # is importable, although that kernel has no CPU backend.  Switch only
        # the attention implementation to the upstream eager path; parameters
        # and checkpoint tensors remain unchanged.
        import torch.nn as nn

        for attention in vision_tower.modules():
            if getattr(attention, "attn_type", None) == "flash_v2":
                attention.attn_type = "origin"
                probability = attention.attn_drop if isinstance(attention.attn_drop, float) else 0.0
                attention.attn_drop = nn.Dropout(probability)
    vision_keys = _load_prefixed_state(vision_tower, checkpoint, "model.vision_tower.")
    projector_keys = _load_prefixed_state(projector, checkpoint, "model.mm_projector.")
    vision_tower.to(device).eval()
    projector.to(device).eval()
    return _NativeVideoChatFlashWorker(
        vision_tower,
        projector,
        device=str(device),
        num_frames=num_frames,
        local_num_frames=local_num_frames,
        checkpoint_path=checkpoint,
        vision_state_keys=vision_keys,
        projector_state_keys=projector_keys,
    )


class VideoChatFlashAdapter(ExternalFixedVideoAdapter):
    """Expose official VideoChat-Flash projected visual tokens."""

    integration_id = "videochat_flash"
    default_feature_stage = "projected_visual"
    feature_stage = default_feature_stage
    default_checkout_path = "weights/videochat-flash"
    default_model_path = "weights/videochat-flash"
    backend = "videochat-flash"
    cache_owner = None
    run_mode = "long"
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        supports_streaming=False,
        supports_kv_cache=False,
        supports_token_cache=False,
        supports_visual_memory_cache=False,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )

    def __init__(self, **kwargs: Any) -> None:
        if (
            "load_model_fn" not in kwargs
            and "model_loader" not in kwargs
            and "worker_factory" not in kwargs
        ):
            requested_frames = int(kwargs.pop("num_frames", kwargs.pop("clip_frames", 32)))

            def _loader(**load_kwargs: Any) -> Any:
                return load_videochat_flash(num_frames=requested_frames, **load_kwargs)

            kwargs["load_model_fn"] = _loader
        super().__init__(**kwargs)
        if bool(getattr(self.worker, "native_upstream", False)):
            self.implementation_source = "native_upstream"

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        output = super().encode(batch, train=train)
        if not bool(getattr(self.worker, "native_upstream", False)):
            return output
        aux = dict(output.aux)
        aux.update(
            {
                "native_route_available": True,
                "implementation_source": "native_upstream",
            }
        )
        return EncoderOutput(
            features=output.features,
            pooled=output.pooled,
            timeline=output.timeline,
            aux=aux,
        )


DEFAULT_CAPABILITIES = VideoChatFlashAdapter.capabilities
FEATURE_STAGE = VideoChatFlashAdapter.default_feature_stage

__all__ = [
    "DEFAULT_CAPABILITIES",
    "FEATURE_STAGE",
    "VideoChatFlashAdapter",
    "load_videochat_flash",
]
