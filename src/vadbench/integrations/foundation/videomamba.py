"""Native VideoMamba-Tiny fixed-clip adapter.

The official VideoMamba model is loaded from the pinned checkout and its K400
checkpoint.  For CPU smoke the adapter uses the upstream reference selective
scan and PyTorch depthwise convolution; on CUDA it leaves the official fused
kernels enabled.  This is a runtime choice inside the same model, not a model
or weight substitution.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import ClipBatch
from vadbench.integrations.foundation.base import FOUNDATION_CAPABILITIES, FoundationVideoAdapter


class _TorchRMSNorm:
    """Small torch-only RMSNorm used when Triton is unavailable on CPU."""

    def __new__(cls, hidden_size: int, eps: float = 1e-5, **_: Any) -> Any:
        import torch
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.bias = None

            def forward(
                self, x: Any, residual: Any = None, prenorm: bool = False, **__: Any
            ) -> Any:
                if residual is not None:
                    x = x + residual
                out = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
                out = out * self.weight.to(dtype=out.dtype)
                return (out, x) if prenorm else out

        return _Impl()


def _load_official_model(checkout_path: Path, checkpoint_path: Path, device: str) -> Any:
    for candidate in (
        checkout_path / "videomamba" / "video_sm",
        checkout_path / "video_sm",
        checkout_path / "videomamba" / "video_sm" / "models",
    ):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    mamba_root = checkout_path / "mamba"
    if mamba_root.is_dir() and str(mamba_root) not in sys.path:
        sys.path.insert(0, str(mamba_root))
    if str(device).startswith("cpu"):
        # The official Python reference path can run on CPU, but its package
        # imports the optional CUDA extension at module import time.  A module
        # placeholder is safe here because we immediately replace the Mamba
        # call with the upstream ``selective_scan_ref`` implementation below;
        # no CUDA kernel or alternate weight is introduced.
        import types

        sys.modules.setdefault("selective_scan_cuda", types.ModuleType("selective_scan_cuda"))
    module = importlib.import_module("models.videomamba")
    vision_mamba = getattr(module, "VisionMamba", None)
    if vision_mamba is None:
        raise ImportError("官方 VideoMamba checkout 缺少 models.videomamba.VisionMamba")
    import torch

    model = vision_mamba(
        img_size=224,
        patch_size=16,
        depth=24,
        embed_dim=192,
        channels=3,
        num_classes=400,
        num_frames=16,
        rms_norm=True,
        fused_add_norm=True,
        residual_in_fp32=True,
        bimamba=True,
    )
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("VideoMamba checkpoint 必须是 state_dict mapping")
    state = dict(payload)
    # CPU has no usable Triton fused norm; replace only the implementation while
    # retaining the exact checkpoint weights and model dimensions.
    if device.startswith("cpu"):
        model.fused_add_norm = False
        for block in model.layers:
            block.fused_add_norm = False
            if block.norm.__class__.__name__ == "RMSNorm":
                replacement = _TorchRMSNorm(model.embed_dim)
                replacement.weight.data.copy_(block.norm.weight.data)
                block.norm = replacement
        if model.norm_f.__class__.__name__ == "RMSNorm":
            replacement = _TorchRMSNorm(model.embed_dim)
            replacement.weight.data.copy_(model.norm_f.weight.data)
            model.norm_f = replacement
        mamba_simple = importlib.import_module("mamba_ssm.modules.mamba_simple")
        scan = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        mamba_simple.causal_conv1d_fn = None
        mamba_simple.selective_scan_fn = scan.selective_scan_ref
        for block in model.layers:
            block.mixer.use_fast_path = False
    missing_head = ["head.weight", "head.bias"]
    # Keep the classification head in the module for architecture parity, but
    # it is not part of the representation used by the adapter.
    result = model.load_state_dict(state, strict=False)
    missing = set(getattr(result, "missing_keys", ()) or ())
    unexpected = set(getattr(result, "unexpected_keys", ()) or ())
    allowed_missing = set(missing_head)
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            "VideoMamba checkpoint state mismatch: "
            f"missing={sorted(missing - allowed_missing)[:8]}, unexpected={sorted(unexpected)[:8]}"
        )
    model.to(device)
    model.eval()
    return model


def _prepare(batch: ClipBatch, *, image_size: int = 224, num_frames: int = 16) -> Any:
    import torch
    import torch.nn.functional as F

    frames = np.asarray(batch.frames, dtype=np.uint8)
    values = []
    for row, length in enumerate(batch.valid_lengths):
        source = frames[row, : int(length)]
        if source.shape[0] < num_frames:
            source = np.concatenate(
                (source, np.repeat(source[-1:], num_frames - source.shape[0], axis=0))
            )
        source = source[:num_frames]
        tensor = torch.from_numpy(np.transpose(source, (3, 0, 1, 2))).float().div(255.0)
        values.append(tensor)
    value = torch.stack(values, dim=0)
    b, c, t, h, w = value.shape
    scale = image_size / min(h, w)
    nh, nw = max(image_size, round(h * scale)), max(image_size, round(w * scale))
    if (nh, nw) != (h, w):
        flat = value.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(nh, nw), mode="bilinear", align_corners=False)
        value = flat.reshape(b, t, c, nh, nw).permute(0, 2, 1, 3, 4)
        h, w = nh, nw
    top, left = max(0, (h - image_size) // 2), max(0, (w - image_size) // 2)
    value = value[..., top : top + image_size, left : left + image_size]
    mean = value.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    std = value.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    return (value - mean) / std


def load_videomamba(
    model_path: str | Path,
    checkout_path: str | Path | None = None,
    device: str = "cpu",
    **_: Any,
) -> Any:
    if checkout_path is None:
        raise ValueError("VideoMamba loader 需要显式 checkout_path")
    model = _load_official_model(
        Path(checkout_path).expanduser().resolve(),
        Path(model_path).expanduser().resolve(),
        str(device),
    )

    class _Encoder:
        def encode(self, batch: ClipBatch) -> Any:
            import torch

            inputs = _prepare(batch)
            inputs = inputs.to(device)
            with torch.no_grad():
                return model.forward_features(inputs)

    return _Encoder()


class VideoMambaAdapter(FoundationVideoAdapter):
    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "videomamba"
    FEATURE_STAGE = "backbone_tokens"
    PREPROCESS_PROFILE = "videomamba-bthwc-v1"
    DEFAULT_MODEL_NAME = "OpenGVLab/VideoMamba-Tiny-K400"
    DEFAULT_RUNTIME = "in_process"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("loader", load_videomamba)
        checkout_path = kwargs.get("checkout_path")
        model_kwargs = dict(kwargs.get("model_kwargs") or {})
        if checkout_path is not None:
            model_kwargs.setdefault("checkout_path", checkout_path)
        kwargs["model_kwargs"] = model_kwargs
        kwargs.pop("upstream_entrypoint", None)
        kwargs.pop("entrypoint", None)
        super().__init__(**kwargs)


__all__ = ["VideoMambaAdapter", "load_videomamba"]
