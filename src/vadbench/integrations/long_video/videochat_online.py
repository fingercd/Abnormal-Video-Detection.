"""Native VideoChat-Online visual-memory streaming adapter.

This adapter follows the official ``VideoChatOnline_Stream.extract_feature_bank``
implementation.  The persisted state is the model's hierarchical visual
memory (not language-model decoder KV); each new chunk updates the upstream
memory bank and returns the current time-ordered memory tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import EncoderCapabilities, EncoderOutput, StreamState, StreamStep

from .base import ExternalStreamingVideoAdapter, LongVideoAssetError


class _NativeVideoChatOnlineWorker:
    native_upstream = True

    def __init__(self, model: Any, transform: Any, *, device: str, checkpoint_path: Path) -> None:
        self.model = model
        self.transform = transform
        self.device = str(device)
        self.checkpoint_path = checkpoint_path
        self.seen = 0

    def init_state(self, video_id: str, **_: Any) -> dict[str, Any]:
        self.model.memorybank = None
        self.seen = 0
        return {"video_id": str(video_id), "seen": 0}

    def _preprocess(self, frames: np.ndarray) -> Any:
        import torch
        from PIL import Image

        values = [self.transform(Image.fromarray(frame, mode="RGB")) for frame in frames]
        return torch.stack(values, dim=0).to(self.device)

    def encode_step(self, chunk: Any, state: Any, **_: Any) -> dict[str, Any]:
        import torch

        opaque = state.opaque if isinstance(state, StreamState) else state
        if not isinstance(opaque, dict):
            opaque = {"seen": self.seen}
        start_id = int(opaque.get("seen", self.seen))
        frames = np.asarray(chunk.frames[0, : int(chunk.valid_lengths[0])], dtype=np.uint8)
        pixel_values = self._preprocess(frames)
        with torch.no_grad():
            features, scale, indices = self.model.extract_feature_bank(
                pixel_values,
                torch.tensor([pixel_values.shape[0]], dtype=torch.long, device=pixel_values.device),
                torch.tensor([1], dtype=torch.long, device=pixel_values.device),
                start_id=start_id,
            )
        self.seen = start_id + int(frames.shape[0])
        if features.ndim == 2:
            features = features.unsqueeze(0)
        return {
            "features": features,
            "pooled": features.mean(dim=1),
            "state": {"seen": self.seen},
            "telemetry": {
                "memory_kind": "visual_memory",
                "memory_token_count": int(features.shape[1]),
                "memory_scale": [int(item) for item in (scale or [])],
                "memory_indices": [int(item) for item in (indices or [])],
                "native_route_available": True,
                "implementation_source": "native_upstream",
                "checkpoint_path": str(self.checkpoint_path),
            },
        }

    def finalize(self, state: Any, **_: Any) -> None:
        return None


def _load_state_from_index(model: Any, root: Path) -> tuple[int, int]:
    from safetensors import safe_open

    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    prefixes = ("vision_model.", "mlp1.")
    tensors: dict[str, Any] = {}
    for key, filename in weight_map.items():
        if not key.startswith(prefixes):
            continue
        with safe_open(str(root / filename), framework="pt", device="cpu") as handle:
            tensors[key] = handle.get_tensor(key)
    vision_state = {
        key[len("vision_model.") :]: value
        for key, value in tensors.items()
        if key.startswith("vision_model.")
    }
    mlp_state = {
        key[len("mlp1.") :]: value for key, value in tensors.items() if key.startswith("mlp1.")
    }
    if not vision_state or not mlp_state:
        raise RuntimeError("VideoChat-Online checkpoint 缺少 vision_model/mlp1 权重")
    vision_result = model.vision_model.load_state_dict(vision_state, strict=False)
    mlp_result = model.mlp1.load_state_dict(mlp_state, strict=False)
    if vision_result.missing_keys or vision_result.unexpected_keys:
        raise RuntimeError(
            f"VideoChat-Online vision state mismatch: missing={vision_result.missing_keys[:8]}, "
            f"unexpected={vision_result.unexpected_keys[:8]}"
        )
    if mlp_result.missing_keys or mlp_result.unexpected_keys:
        raise RuntimeError(
            f"VideoChat-Online mlp1 state mismatch: missing={mlp_result.missing_keys[:8]}, "
            f"unexpected={mlp_result.unexpected_keys[:8]}"
        )
    return len(vision_state), len(mlp_state)


def load_videochat_online(
    checkout_path: str | Path,
    model_path: str | Path,
    device: str = "cpu",
    **_: Any,
) -> _NativeVideoChatOnlineWorker:
    import torch
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    checkout = Path(checkout_path).expanduser().resolve()
    root = Path(model_path).expanduser().resolve()
    required = (checkout / "internvl", root / "config.json", root / "model.safetensors.index.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise LongVideoAssetError(
            integration_id="videochat_online",
            code="missing_asset",
            message="VideoChat-Online 官方 checkout/HF checkpoint 不完整",
            details={"missing": missing},
        )
    sys.path.insert(0, str(checkout))
    try:
        from internvl.model.videochat_online.configuration_internvl_chat import InternVLChatConfig
        from internvl.model.videochat_online.modeling_intern_vit import InternVisionModel
        from internvl.model.videochat_online.modeling_videochat_online import VideoChatOnline_Stream
    except Exception as exc:
        raise RuntimeError(f"无法导入 VideoChat-Online 官方模块: {exc}") from exc
    # ``from_pretrained`` in this pinned upstream calls ``repr(config)``;
    # its default constructor is incompatible with the current Transformers
    # logger.  Construct from the exact local JSON instead, preserving every
    # nested vision/LLM field without any network resolution.
    config_data = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config = InternVLChatConfig(**config_data)
    vision_model = InternVisionModel(config.vision_config)
    model = VideoChatOnline_Stream(
        config,
        vision_model=vision_model,
        language_model=torch.nn.Identity(),
    )
    vision_keys, mlp_keys = _load_state_from_index(model, root)
    model.long_bank = 64
    model.mid_bank = 64
    model.short_bank = 64
    model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    worker = _NativeVideoChatOnlineWorker(model, transform, device=device, checkpoint_path=root)
    worker.vision_state_keys = vision_keys
    worker.mlp_state_keys = mlp_keys
    return worker


class VideoChatOnlineAdapter(ExternalStreamingVideoAdapter):
    """Expose VideoChat-Online's official hierarchical visual memory."""

    integration_id = "videochat_online"
    default_feature_stage = "visual_memory"
    feature_stage = default_feature_stage
    default_checkout_path = "external/videochat-online"
    default_model_path = "weights/videochat-online"
    backend = "videochat-online"
    cache_owner = "visual_memory"
    capabilities = EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_kv_cache=False,
        supports_token_cache=False,
        supports_visual_memory_cache=True,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("load_model_fn", load_videochat_online)
        super().__init__(**kwargs)
        if bool(getattr(self.worker, "native_upstream", False)):
            self.implementation_source = "native_upstream"

    def encode_step(
        self,
        chunk: Any,
        state: StreamState,
        train: bool = False,
        compression: Any = None,
    ) -> StreamStep:
        result = super().encode_step(chunk, state, train=train, compression=compression)
        if not bool(getattr(self.worker, "native_upstream", False)) or result.output is None:
            return result
        aux = dict(result.output.aux)
        aux.update(
            {
                "native_route_available": True,
                "implementation_source": "native_upstream",
                "cache_owner": "visual_memory",
            }
        )
        output = EncoderOutput(
            features=result.output.features,
            pooled=result.output.pooled,
            timeline=result.output.timeline,
            aux=aux,
        )
        return StreamStep(
            output=output,
            state=result.state,
            cache_updates=result.cache_updates,
            telemetry=result.telemetry,
            final=result.final,
        )


DEFAULT_CAPABILITIES = VideoChatOnlineAdapter.capabilities
FEATURE_STAGE = VideoChatOnlineAdapter.default_feature_stage

__all__ = [
    "DEFAULT_CAPABILITIES",
    "FEATURE_STAGE",
    "VideoChatOnlineAdapter",
    "load_videochat_online",
]
