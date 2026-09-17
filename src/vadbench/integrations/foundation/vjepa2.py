"""Native Meta V-JEPA 2 fixed-clip integration.

The Hugging Face V-JEPA 2 model exposes ``get_vision_features`` rather than a
standard ``forward(pixel_values=...)`` method.  The loader below follows the
official model-card API and keeps all preprocessing inside this adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import ClipBatch, validate_clip_for_capabilities
from vadbench.integrations.foundation.base import (
    FOUNDATION_CAPABILITIES,
    FoundationVideoAdapter,
)


def _move(value: Any, device: str | None) -> Any:
    if isinstance(value, Mapping):
        return {key: _move(item, device) for key, item in value.items()}
    if device is not None and hasattr(value, "to"):
        try:
            return value.to(device)
        except (TypeError, RuntimeError):
            return value
    return value


class _VJEPA2Encoder:
    def __init__(self, model: Any, processor: Any, *, device: str | None) -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def _prepare_inputs(self, batch: ClipBatch) -> Any:
        """Keep the native V-JEPA preprocessing shared by inference and hooks."""

        videos = [
            [
                np.asarray(frame, dtype=np.uint8)
                for frame in np.asarray(batch.frames)[row, : int(length)]
            ]
            for row, length in enumerate(batch.valid_lengths)
        ]
        processed = self.processor(videos, return_tensors="pt")
        if isinstance(processed, Mapping):
            return _move(dict(processed), self.device)
        return _move(processed, self.device)

    def _get_vision_features(self, inputs: Any) -> Any:
        get_features = getattr(self.model, "get_vision_features", None)
        if not callable(get_features):
            raise RuntimeError("V-JEPA 2 model 缺少官方 get_vision_features 接口")
        if isinstance(inputs, Mapping):
            return get_features(**inputs)
        return get_features(inputs)

    def _freeze_backbone_for_internal_plugin(self) -> None:
        """Freeze the native model while preserving a differentiable suffix.

        An external, temporary forward hook may replace a block output with a
        trainable plugin output.  Frozen upstream parameters do not require an
        autograd graph until that plugin creates one; downstream frozen blocks
        then remain differentiable with respect to the plugin without exposing
        any backbone parameter to the optimizer.
        """

        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, batch: ClipBatch) -> Any:
        inputs = self._prepare_inputs(batch)
        import torch

        with torch.no_grad():
            return self._get_vision_features(inputs)

    def encode_with_grad(self, batch: ClipBatch) -> Any:
        """Run the native vision encoder for a temporary internal plugin hook.

        The model remains entirely frozen and in ``eval`` mode.  This method
        does not install a selector or modify any model method: callers must
        own a short-lived hook on a verified native block and remove it after
        the forward.  With no hook installed, this route is numerically the
        same as :meth:`encode` but deliberately enables autograd for a later
        hook-created trainable tensor to connect to the frozen suffix.
        """

        self._freeze_backbone_for_internal_plugin()
        inputs = self._prepare_inputs(batch)
        import torch

        with torch.enable_grad():
            return self._get_vision_features(inputs)


def load_vjepa2(model_path: str | Path, device: str | None = None, **_: Any) -> Any:
    """Load the pinned local V-JEPA 2 checkpoint through Transformers."""

    try:
        from transformers import AutoModel, AutoVideoProcessor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("V-JEPA 2 需要 transformers>=4.56 与 AutoVideoProcessor") from exc
    path = str(model_path)
    model = AutoModel.from_pretrained(path, local_files_only=True)
    processor = AutoVideoProcessor.from_pretrained(path, local_files_only=True)
    if device is not None and callable(getattr(model, "to", None)):
        model.to(device)
    if callable(getattr(model, "eval", None)):
        model.eval()
    return _VJEPA2Encoder(model, processor, device=device)


class VJEPA2Adapter(FoundationVideoAdapter):
    """Adapt the official V-JEPA 2 video encoder to VADBench."""

    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "vjepa2"
    FEATURE_STAGE = "backbone_tokens"
    PREPROCESS_PROFILE = "vjepa2-bthwc-v1"
    DEFAULT_MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
    DEFAULT_RUNTIME = "in_process"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("runtime", "in_process")
        kwargs.setdefault("loader", load_vjepa2)
        kwargs.pop("upstream_entrypoint", None)
        kwargs.pop("entrypoint", None)
        super().__init__(**kwargs)

    def encode_with_grad(self, batch: ClipBatch) -> Any:
        """Expose V-JEPA's frozen-backbone internal-plugin route explicitly.

        This intentionally returns the raw native vision tokens so an external
        context-managed hook can train against a downstream loss without the
        generic feature normalizer detaching or pooling the intervening graph.
        Standard ``encode`` remains the inference API and stays no-grad.
        """

        validate_clip_for_capabilities(batch, self.capabilities, train=True)
        worker = self.bridge.encoder
        route = getattr(worker, "encode_with_grad", None)
        if not callable(route):
            raise RuntimeError(
                "V-JEPA 2 loaded worker 不支持 encode_with_grad；"
                "需要官方 _VJEPA2Encoder 冻结 backbone 路径"
            )
        self.encoder = worker
        return route(batch)


VJepa2Adapter = VJEPA2Adapter
VJepaAdapter = VJEPA2Adapter

__all__ = ["VJEPA2Adapter", "VJepa2Adapter", "VJepaAdapter", "load_vjepa2"]
