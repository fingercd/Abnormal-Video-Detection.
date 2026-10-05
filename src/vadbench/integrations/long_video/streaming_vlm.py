"""Native StreamingVLM decoder-KV adapter.

The upstream model is Qwen2.5-VL with the repository's streaming patches.  A
chunk is encoded by the official vision tower, then fed to the causal decoder
with ``use_cache=True`` and the previous DynamicCache.  The adapter exposes the
cache layers as the framework's explicit ``decoder_kv`` CacheView; no
compression is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import EncoderCapabilities, EncoderOutput, StreamState, StreamStep

from .base import ExternalStreamingVideoAdapter, LongVideoAssetError


class _NativeStreamingVLMWorker:
    native_upstream = True

    def __init__(self, model: Any, processor: Any, *, device: str, checkpoint: Path) -> None:
        self.model = model
        self.processor = processor
        self.device = str(device)
        self.checkpoint = checkpoint
        self.past = None
        self.offset = 0

    def init_state(self, video_id: str, **_: Any) -> dict[str, Any]:
        self.past = None
        self.offset = 0
        return {"video_id": str(video_id), "offset": 0}

    def _cache_mapping(self) -> dict[str, Any]:
        if self.past is None:
            return {}
        tensors: dict[str, Any] = {}
        layers = getattr(self.past, "layers", None)
        if layers is None and hasattr(self.past, "to_legacy_cache"):
            layers = self.past.to_legacy_cache()
        for index, layer in enumerate(layers or ()):
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                key, value = layer.keys, layer.values
            else:
                key, value = layer[0], layer[1]
            tensors[f"layer_{index}.key"] = key
            tensors[f"layer_{index}.value"] = value
        return {"tensors": tensors, "sequence_axis": -2}

    def encode_step(self, chunk: Any, state: Any, **_: Any) -> dict[str, Any]:
        import torch
        from PIL import Image

        frames = np.asarray(chunk.frames[0, : int(chunk.valid_lengths[0])], dtype=np.uint8)
        images = [Image.fromarray(frame, mode="RGB") for frame in frames]
        text = "<|vision_start|><|video_pad|><|vision_end|>"
        inputs = self.processor(text=[text], videos=[images], return_tensors="pt", use_fast=False)
        pixel_values = inputs.pixel_values_videos.to(self.device, dtype=torch.bfloat16)
        grid = inputs.video_grid_thw.to(self.device)
        with torch.no_grad():
            visual = self.model.get_video_features(pixel_values, grid)[0].unsqueeze(0)
            token_count = int(visual.shape[1])
            input_ids = torch.zeros((1, token_count), dtype=torch.long, device=self.device)
            position_ids = (
                torch.arange(self.offset, self.offset + token_count, device=self.device)
                .view(1, 1, token_count)
                .expand(3, 1, token_count)
            )
            result = self.model.model(
                input_ids=input_ids,
                inputs_embeds=visual,
                position_ids=position_ids,
                past_key_values=self.past,
                use_cache=True,
                return_dict=True,
            )
        self.past = result.past_key_values
        self.offset += token_count
        contextual = result.last_hidden_state
        return {
            "features": contextual,
            "pooled": contextual.mean(dim=1),
            "state": {"offset": self.offset},
            "cache": self._cache_mapping(),
            "aux": {
                "cache_hit": self.offset > token_count,
                "cache_sequence_length": self.offset,
                "cache_layer_count": len(getattr(self.past, "layers", ()) or ()),
            },
            "telemetry": {
                "cache_owner": "language_model_decoder",
                "cache_kind": "decoder_kv",
                "cache_hit": self.offset > token_count,
                "cache_sequence_length": self.offset,
                "visual_token_count": token_count,
                "native_route_available": True,
                "implementation_source": "native_upstream",
                "checkpoint_path": str(self.checkpoint),
            },
        }

    def finalize(self, state: Any, **_: Any) -> None:
        self.past = None


def load_streaming_vlm(
    model_path: str | Path,
    device: str = "cpu",
    **_: Any,
) -> _NativeStreamingVLMWorker:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    root = Path(model_path).expanduser().resolve()
    required = (root / "config.json", root / "model.safetensors.index.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise LongVideoAssetError(
            integration_id="streaming_vlm",
            code="missing_asset",
            message="StreamingVLM HF checkpoint 不完整",
            details={"missing": missing},
        )
    dtype = torch.bfloat16
    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(root), local_files_only=True, dtype=dtype, low_cpu_mem_usage=True
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, use_fast=False)
    return _NativeStreamingVLMWorker(model, processor, device=str(device), checkpoint=root)


class StreamingVLMAdapter(ExternalStreamingVideoAdapter):
    """Expose the upstream Qwen2.5-VL decoder context and DynamicCache."""

    integration_id = "streaming_vlm"
    default_feature_stage = "decoder_contextual"
    feature_stage = default_feature_stage
    default_checkout_path = "external/streaming-vlm"
    default_model_path = "weights/streaming-vlm"
    backend = "streaming-vlm"
    cache_owner = "decoder_kv"
    capabilities = EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_kv_cache=True,
        supports_token_cache=False,
        supports_visual_memory_cache=False,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("load_model_fn", load_streaming_vlm)
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
                "cache_owner": "language_model_decoder",
                "cache_kind": "decoder_kv",
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


DEFAULT_CAPABILITIES = StreamingVLMAdapter.capabilities
FEATURE_STAGE = StreamingVLMAdapter.default_feature_stage

__all__ = [
    "DEFAULT_CAPABILITIES",
    "FEATURE_STAGE",
    "StreamingVLMAdapter",
    "load_streaming_vlm",
]
