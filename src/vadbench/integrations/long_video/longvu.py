"""Native LongVU projected-visual adapter.

LongVU publishes a Cambrian/Qwen checkpoint whose vision towers are declared as
SigLIP SO400M and DINOv2-Giant.  This adapter follows the official SVA connector
path and loads the two published base vision checkpoints plus LongVU's own
projector/sampler weights.  The language model is intentionally not materialised
because VADBench consumes the projected visual representation only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from vadbench.contracts import ClipBatch, EncoderCapabilities, EncoderOutput

from .base import ExternalFixedVideoAdapter, LongVideoAssetError


def _load_source_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 LongVU 官方模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_prefix(module: Any, checkpoint: Path, prefix: str) -> int:
    from safetensors import safe_open

    state: dict[str, Any] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        for key in keys:
            if key.startswith(prefix):
                state[key[len(prefix) :]] = handle.get_tensor(key)
    if not state:
        raise RuntimeError(f"LongVU checkpoint 不含前缀 {prefix!r}")
    result = module.load_state_dict(state, strict=False)
    missing = tuple(getattr(result, "missing_keys", ()) or ())
    unexpected = tuple(getattr(result, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise RuntimeError(
            f"LongVU {prefix} state mismatch: missing={list(missing)[:8]}, "
            f"unexpected={list(unexpected)[:8]}"
        )
    return len(state)


def _resize_tokens(tokens: Any, side: int = 24) -> Any:
    import torch.nn.functional as F

    if tokens.shape[1] == side * side:
        return tokens
    height = width = int(tokens.shape[1] ** 0.5)
    if height * width != tokens.shape[1]:
        raise ValueError(f"LongVU vision token 数不是平方数：{tokens.shape}")
    value = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], height, width)
    value = F.interpolate(value.float(), size=(side, side), mode="bilinear", align_corners=False)
    return value.to(dtype=tokens.dtype).flatten(2).transpose(1, 2).contiguous()


class _NativeLongVUWorker:
    native_upstream = True

    def __init__(
        self,
        *,
        siglip: Any,
        dino: Any,
        connector: Any,
        processors: tuple[Any, Any],
        device: str,
        checkpoint: Path,
        metadata: dict[str, Any],
    ) -> None:
        self.siglip = siglip
        self.dino = dino
        self.connector = connector
        self.processors = processors
        self.device = str(device)
        self.checkpoint = checkpoint
        self.metadata = metadata

    def encode(self, batch: ClipBatch, **_: Any) -> dict[str, Any]:
        import torch

        sig_processor, dino_processor = self.processors
        sig_values = []
        dino_values = []
        for row, length in enumerate(batch.valid_lengths):
            frames = [batch.frames[row, idx] for idx in range(int(length))]
            sig_values.extend(sig_processor(images=frames, return_tensors="pt")["pixel_values"])
            dino_values.extend(dino_processor(images=frames, return_tensors="pt")["pixel_values"])
        sig_pixels = torch.stack(sig_values, dim=0).to(self.device)
        dino_pixels = torch.stack(dino_values, dim=0).to(self.device)
        with torch.no_grad():
            sig_tokens = self.siglip(pixel_values=sig_pixels).last_hidden_state
            dino_tokens = self.dino(pixel_values=dino_pixels).last_hidden_state[:, 1:]
            sig_tokens = _resize_tokens(sig_tokens, 24)
            dino_tokens = _resize_tokens(dino_tokens, 24)
            sig_tokens = self.connector.mm_projector_aux_0(sig_tokens)
            dino_tokens = self.connector.mm_projector_aux_1(dino_tokens)
            bs = batch.batch_size
            frames = int(batch.valid_lengths[0])
            sig_tokens = sig_tokens.reshape(bs, frames, 576, -1).mean(dim=1)
            dino_tokens = dino_tokens.reshape(bs, frames, 576, -1).mean(dim=1)

            def rearrange(tokens: Any) -> Any:
                value = tokens.view(bs, 12, 2, 12, 2, tokens.shape[-1])
                return value.permute(0, 1, 3, 2, 4, 5).contiguous().view(bs * 144, 4, -1)

            context = (
                sig_tokens.mean(dim=1, keepdim=True).expand(-1, 144, -1).reshape(bs * 144, 1, -1)
            )
            query = self.connector.vision_query.view(1, 1, -1).expand(bs * 144, 1, -1)
            query = self.connector.vision_sampler_0(
                query,
                context,
                rearrange(sig_tokens),
                rearrange(dino_tokens),
                torch.ones(bs * 144, 4, dtype=torch.bool, device=query.device),
                torch.ones(bs * 144, 4, dtype=torch.bool, device=query.device),
            )
            query = query.view(bs, 144, -1)
            features = self.connector.mm_projector(query)
        return {
            "features": features,
            "pooled": features.mean(dim=1),
            "aux": {
                "native_route_available": True,
                "implementation_source": "native_upstream",
                "checkpoint_path": str(self.checkpoint),
                "visual_token_count": int(features.shape[1]),
                "projector_dimension": int(features.shape[2]),
                **self.metadata,
            },
        }


def load_longvu(
    checkout_path: str | Path,
    model_path: str | Path,
    device: str = "cpu",
    **_: Any,
) -> _NativeLongVUWorker:
    import torch
    from transformers import (
        AutoImageProcessor,
        Dinov2Model,
        SiglipImageProcessor,
        SiglipVisionModel,
    )

    checkout = Path(checkout_path).expanduser().resolve()
    root = Path(model_path).expanduser().resolve()
    checkpoint = root / "model.safetensors"
    sig_root = root / "siglip"
    dino_root = root / "dinov2"
    required = (
        checkout / "longvu" / "vision_sampler.py",
        checkpoint,
        root / "config.json",
        sig_root / "model.safetensors",
        dino_root / "model.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise LongVideoAssetError(
            integration_id="longvu",
            code="missing_asset",
            message="LongVU checkpoint 或 auxiliary vision tower 不完整",
            details={"missing": missing},
        )
    config_data = json.loads((root / "config.json").read_text(encoding="utf-8"))
    sampler_module = _load_source_module(
        checkout / "longvu" / "vision_sampler.py", "vadbench_longvu_sampler"
    )
    connector = SimpleNamespace()
    connector.mm_projector_aux_0 = torch.nn.Sequential(
        torch.nn.Linear(1152, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.LayerNorm(1024),
    )
    connector.mm_projector_aux_1 = torch.nn.Sequential(
        torch.nn.Linear(1536, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.LayerNorm(1024),
    )
    connector.mm_projector = torch.nn.Sequential(
        torch.nn.Linear(1024, int(config_data["hidden_size"])),
        torch.nn.GELU(),
        torch.nn.Linear(int(config_data["hidden_size"]), int(config_data["hidden_size"])),
    )
    connector.vision_sampler_0 = sampler_module.VisionTokenSampler(
        1024, 1024, [1024, 1024], [2, 2], 1024, int(config_data.get("connector_depth", 1))
    )
    connector.vision_query = torch.nn.Parameter(torch.empty(1, 1024))
    keys = {
        "mm_projector_aux_0": _load_prefix(
            connector.mm_projector_aux_0, checkpoint, "model.mm_projector_aux_0."
        ),
        "mm_projector_aux_1": _load_prefix(
            connector.mm_projector_aux_1, checkpoint, "model.mm_projector_aux_1."
        ),
        "mm_projector": _load_prefix(connector.mm_projector, checkpoint, "model.mm_projector."),
        "vision_sampler_0": _load_prefix(
            connector.vision_sampler_0, checkpoint, "model.vision_sampler_0."
        ),
    }
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        connector.vision_query.data.copy_(handle.get_tensor("model.vision_query"))
    siglip = SiglipVisionModel.from_pretrained(str(sig_root), local_files_only=True)
    dino = Dinov2Model.from_pretrained(str(dino_root), local_files_only=True)
    sig_processor = SiglipImageProcessor.from_pretrained(str(sig_root), local_files_only=True)
    dino_processor = AutoImageProcessor.from_pretrained(str(dino_root), local_files_only=True)
    siglip.to(device).eval()
    dino.to(device).eval()
    connector.mm_projector_aux_0.to(device).eval()
    connector.mm_projector_aux_1.to(device).eval()
    connector.mm_projector.to(device).eval()
    connector.vision_sampler_0.to(device).eval()
    connector.vision_query.data = connector.vision_query.data.to(device)
    return _NativeLongVUWorker(
        siglip=siglip,
        dino=dino,
        connector=connector,
        processors=(sig_processor, dino_processor),
        device=str(device),
        checkpoint=checkpoint,
        metadata={
            "vision_state_sources": ["google/siglip-so400m-patch14-384", "facebook/dinov2-giant"],
            "connector_state_keys": keys,
        },
    )


class LongVUAdapter(ExternalFixedVideoAdapter):
    """Expose LongVU's official SVA projected visual tokens."""

    integration_id = "longvu"
    default_feature_stage = "projected_visual"
    feature_stage = default_feature_stage
    default_checkout_path = "external/longvu"
    default_model_path = "weights/longvu"
    backend = "longvu"
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
        kwargs.setdefault("load_model_fn", load_longvu)
        super().__init__(**kwargs)
        if bool(getattr(self.worker, "native_upstream", False)):
            self.implementation_source = "native_upstream"

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        output = super().encode(batch, train=train)
        if not bool(getattr(self.worker, "native_upstream", False)):
            return output
        aux = dict(output.aux)
        aux.update({"native_route_available": True, "implementation_source": "native_upstream"})
        return EncoderOutput(
            features=output.features, pooled=output.pooled, timeline=output.timeline, aux=aux
        )


DEFAULT_CAPABILITIES = LongVUAdapter.capabilities
FEATURE_STAGE = LongVUAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "LongVUAdapter", "load_longvu"]
