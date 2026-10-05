"""Lazy TorchVision video-model adapter.

The adapter keeps TorchVision out of catalog/list imports and exposes the
three video models used by the first integration batch through one small
constructor.  It accepts only local checkpoints; a model object may be
injected by contract tests.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import (
    ClipBatch,
    EncoderCapabilities,
    EncoderOutput,
    TokenTimeline,
    VideoEncoderAdapter,
    validate_clip_for_capabilities,
    validate_encoder_output,
)
from vadbench.integrations.common import normalize_encoder_output, normalize_feature_tensor

DEFAULT_CAPABILITIES = EncoderCapabilities(
    supports_fixed_clip=True,
    supports_streaming=False,
    supports_kv_cache=False,
    supports_token_cache=False,
    supports_visual_memory_cache=False,
    supports_external_cache_policy=False,
    supports_training=True,
    fixed_num_frames=None,
    min_frames=1,
    max_frames=None,
)

_VARIANTS = {
    "r2plus1d": "r2plus1d_18",
    "r2plus1d_18": "r2plus1d_18",
    "r2plus1d-18": "r2plus1d_18",
    "mvit": "mvit_v2_s",
    "mvitv2": "mvit_v2_s",
    "mvit_v2_s": "mvit_v2_s",
    "mvit-v2-s": "mvit_v2_s",
    "swin3d": "swin3d_t",
    "swin3d_t": "swin3d_t",
    "video_swin": "swin3d_t",
    "video-swin": "swin3d_t",
}

_CONSTRUCTORS = {
    "r2plus1d_18": "r2plus1d_18",
    "mvit_v2_s": "mvit_v2_s",
    "swin3d_t": "swin3d_t",
}


def _variant(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("variant 必须是非空字符串")
    normalized = value.strip().lower().replace(" ", "_")
    try:
        return _VARIANTS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"不支持 TorchVision video variant={value!r}；允许：r2plus1d_18、mvit_v2_s、swin3d_t"
        ) from exc


def _shape(value: Any) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        try:
            raw = np.asarray(value).shape
        except Exception:
            return None
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError):
        return None


def _is_torch(value: Any) -> bool:
    return type(value).__module__.split(".", 1)[0] == "torch"


def _numpy(value: Any, name: str) -> np.ndarray:
    try:
        if _is_torch(value) and hasattr(value, "detach"):
            tensor = value.detach().cpu()
            if str(getattr(tensor, "dtype", "")).lower() in {"torch.bfloat16", "bfloat16"}:
                tensor = tensor.float()
            return tensor.numpy()
        return np.asarray(value)
    except Exception as exc:
        raise ValueError(f"{name} 无法转换为 NumPy") from exc


def _checkpoint_file(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"TorchVision checkpoint 不存在：{path}")
    if path.is_file():
        return path.resolve()
    candidates = sorted(
        item.resolve()
        for suffix in ("*.pth", "*.pt", "*.bin", "*.safetensors")
        for item in path.glob(suffix)
    )
    if len(candidates) != 1:
        raise ValueError(
            f"checkpoint 目录必须恰好包含一个权重文件，实际 {len(candidates)}："
            f"{[item.name for item in candidates]}"
        )
    return candidates[0]


def _load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("TorchVision adapter 需要 torch/torchvision") from exc
    return torch


def _make_model(variant: str) -> Any:
    try:
        from torchvision.models import video
    except ImportError as exc:
        raise ImportError("TorchVision video models 不可用，请安装 torchvision") from exc
    constructor = getattr(video, _CONSTRUCTORS[variant], None)
    if not callable(constructor):
        raise ImportError(f"torchvision.models.video 缺少 {_CONSTRUCTORS[variant]}")
    try:
        signature = inspect.signature(constructor)
        kwargs = {"weights": None, "progress": False}
        if not any(
            item.kind is inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()
        ):
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        return constructor(**kwargs)
    except (TypeError, ValueError):
        return constructor(weights=None)


def _load_state(model: Any, path: Path, *, torch: Any, strict: bool) -> dict[str, Any]:
    if not callable(getattr(model, "load_state_dict", None)):
        raise TypeError("TorchVision 模型没有 load_state_dict")
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    state = payload
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state", "model", "model_state_dict", "net"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping) and candidate:
                state = candidate
                break
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"checkpoint 不包含 state_dict：{path}")
    variants = [dict(state)]
    for prefix in ("module.", "model.", "backbone."):
        variants.append(
            {
                (
                    key[len(prefix) :] if isinstance(key, str) and key.startswith(prefix) else key
                ): value
                for key, value in state.items()
            }
        )
    try:
        model_keys = set(model.state_dict())
        selected = max(variants, key=lambda item: sum(key in model_keys for key in item))
    except Exception:
        selected = variants[0]
    try:
        result = model.load_state_dict(selected, strict=strict)
    except TypeError:
        result = model.load_state_dict(selected)
    return {
        "path": str(path),
        "strict": bool(strict),
        "state_keys": len(selected),
        "missing_keys": list(getattr(result, "missing_keys", ()) or ()),
        "unexpected_keys": list(getattr(result, "unexpected_keys", ()) or ()),
    }


def _timeline(batch: ClipBatch, token_count: int) -> TokenTimeline:
    if token_count <= 0:
        raise ValueError("token_count 必须大于 0")
    timestamps = _numpy(batch.timestamps_s, "timestamps_s").astype(np.float64, copy=False)
    indices = (
        np.broadcast_to(
            np.arange(batch.num_frames, dtype=np.int64),
            (batch.batch_size, batch.num_frames),
        )
        if batch.frame_indices is None
        else _numpy(batch.frame_indices, "frame_indices").astype(np.int64, copy=False)
    )
    lengths = np.asarray(batch.valid_lengths, dtype=np.int64)
    starts = np.empty((batch.batch_size, token_count), dtype=np.float64)
    ends = np.empty_like(starts)
    source_start = np.empty((batch.batch_size, token_count), dtype=np.int64)
    source_end = np.empty_like(source_start)
    for row, raw_length in enumerate(lengths):
        length = int(raw_length)
        times = timestamps[row, :length]
        frame_ids = indices[row, :length]
        slots = np.minimum(
            (np.arange(token_count, dtype=np.int64) * length) // token_count,
            length - 1,
        )
        starts[row] = times[slots]
        if length > 1:
            differences = np.diff(times)
            positive = differences[differences > 0]
            delta = float(np.median(positive)) if positive.size else 0.0
        else:
            delta = 0.0
        frame_ends = np.empty(length, dtype=np.float64)
        if length > 1:
            frame_ends[:-1] = times[1:]
        frame_ends[-1] = times[-1] + delta
        ends[row] = frame_ends[slots]
        source_start[row] = frame_ids[slots]
        source_end[row] = source_start[row] + 1
    return TokenTimeline(
        start_s=starts,
        end_s=ends,
        source_frame_start=source_start,
        source_frame_end=source_end,
    )


def _candidate_activation(value: Any, batch_size: int) -> Any | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("last_hidden_state", "features", "x", "pooled", "pooler_output"):
            if key in value:
                candidate = _candidate_activation(value[key], batch_size)
                if candidate is not None:
                    return candidate
        for candidate in reversed(tuple(value.values())):
            selected = _candidate_activation(candidate, batch_size)
            if selected is not None:
                return selected
        return None
    if isinstance(value, (tuple, list)) and not isinstance(value, (str, bytes, bytearray)):
        for candidate in reversed(value):
            selected = _candidate_activation(candidate, batch_size)
            if selected is not None:
                return selected
        return None
    shape = _shape(value)
    if (
        shape is None
        or len(shape) < 2
        or shape[0] != batch_size
        or any(item <= 0 for item in shape)
    ):
        return None
    if len(shape) == 2:
        return value
    if len(shape) == 3:
        return value
    # TorchVision Swin's norm output is [B,T,H,W,C].  Other video blocks may
    # expose [B,C,T,H,W]; choose the larger trailing channel-like dimension.
    if shape[-1] >= shape[1]:
        if hasattr(value, "reshape"):
            return value.reshape(batch_size, -1, shape[-1])
        return np.asarray(value).reshape(batch_size, -1, shape[-1])
    if _is_torch(value):
        return value.flatten(2).transpose(1, 2)
    return np.asarray(value).transpose(0, 2, 3, 4, 1).reshape(batch_size, -1, shape[1])


def _prepare_input(batch: ClipBatch, *, torch: Any, image_size: int, device: str | None) -> Any:
    frames = _numpy(batch.frames, "frames")
    lengths = batch.valid_lengths
    padded = np.array(frames, dtype=np.uint8, copy=True)
    for row, raw_length in enumerate(lengths):
        length = int(raw_length)
        if length < padded.shape[1]:
            padded[row, length:] = padded[row, length - 1 : length]
    value = torch.from_numpy(np.transpose(padded, (0, 4, 1, 2, 3))).float().div(255.0)
    value = (value - value.new_tensor((0.45, 0.45, 0.45)).view(1, 3, 1, 1, 1)) / value.new_tensor(
        (0.225, 0.225, 0.225)
    ).view(1, 3, 1, 1, 1)
    import torch.nn.functional as functional

    batch_size, channels, time, height, width = value.shape
    scale = image_size / min(height, width)
    new_h = max(image_size, int(round(height * scale)))
    new_w = max(image_size, int(round(width * scale)))
    if (new_h, new_w) != (height, width):
        images = value.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        images = functional.interpolate(
            images, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        value = images.reshape(batch_size, time, channels, new_h, new_w).permute(0, 2, 1, 3, 4)
        height, width = new_h, new_w
    top = max(0, (height - image_size) // 2)
    left = max(0, (width - image_size) // 2)
    value = value[..., top : top + image_size, left : left + image_size]
    return value.to(device) if device else value


class TorchvisionVideoAdapter(VideoEncoderAdapter):
    """Shared adapter for TorchVision R(2+1)D, MViTv2 and Video Swin."""

    capabilities = DEFAULT_CAPABILITIES

    def __init__(
        self,
        *,
        variant: str = "r2plus1d_18",
        model: Any | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        image_size: int | None = None,
        clip_frames: int | None = None,
        num_frames: int | None = None,
        frame_stride: int = 1,
        feature_stage: str = "pooled",
        strict_checkpoint: bool = False,
        dtype: str | None = None,
        torch_module: Any | None = None,
        **_: Any,
    ) -> None:
        self.variant = _variant(variant)
        if (
            checkpoint_path is not None
            and checkpoint is not None
            and Path(checkpoint_path) != Path(checkpoint)
        ):
            raise ValueError("checkpoint_path 与 checkpoint 指向不同文件")
        selected_checkpoint = checkpoint_path if checkpoint_path is not None else checkpoint
        self.checkpoint_path = (
            None if selected_checkpoint is None else _checkpoint_file(selected_checkpoint)
        )
        self.device = device
        self.image_size = int(image_size or (112 if self.variant == "r2plus1d_18" else 224))
        self.clip_frames = num_frames if num_frames is not None else clip_frames
        if self.clip_frames is not None and (
            type(self.clip_frames) is not int or self.clip_frames <= 0
        ):
            raise ValueError("clip_frames/num_frames 必须是正整数")
        if type(frame_stride) is not int or frame_stride <= 0:
            raise ValueError("frame_stride 必须是正整数")
        self.frame_stride = frame_stride
        self.feature_stage = str(feature_stage).strip().lower().replace("-", "_")
        if self.feature_stage not in {"pooled", "backbone_tokens", "observed_backbone"}:
            raise ValueError("feature_stage 必须是 pooled、backbone_tokens 或 observed_backbone")
        self.dtype = dtype
        self.strict_checkpoint = bool(strict_checkpoint)
        self.model = model
        self._checkpoint_report: dict[str, Any] | None = None
        if self.model is None:
            if self.checkpoint_path is None:
                raise ValueError("未提供本地 checkpoint；TorchVision adapter 不会隐式联网")
            torch = torch_module or _load_torch()
            self.model = _make_model(self.variant)
            self._checkpoint_report = _load_state(
                self.model,
                self.checkpoint_path,
                torch=torch,
                strict=self.strict_checkpoint,
            )
        elif self.checkpoint_path is not None:
            torch = torch_module or _load_torch()
            self._checkpoint_report = _load_state(
                self.model,
                self.checkpoint_path,
                torch=torch,
                strict=self.strict_checkpoint,
            )
        self._prepare_model()

    def _prepare_model(self) -> None:
        if self.device and callable(getattr(self.model, "to", None)):
            self.model.to(self.device)
        if callable(getattr(self.model, "eval", None)):
            self.model.eval()

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        validate_clip_for_capabilities(batch, self.capabilities, train=train)
        lengths = np.asarray(batch.valid_lengths, dtype=np.int64)
        if self.clip_frames is not None and np.any(lengths != self.clip_frames):
            raise ValueError(
                f"{self.variant} 要求 clip_frames={self.clip_frames}，实际={lengths.tolist()}"
            )
        torch = _load_torch()
        model_input = _prepare_input(
            batch, torch=torch, image_size=self.image_size, device=self.device
        )
        observed: list[tuple[str, Any]] = []
        handles: list[Any] = []
        target = getattr(self.model, "module", self.model)
        hook_paths = ["norm"] if self.variant == "swin3d_t" else []
        hook_paths.extend(["head", "fc", "classifier"])
        for path in hook_paths:
            module = getattr(target, path, None)
            if module is None:
                continue
            register = getattr(module, "register_forward_pre_hook", None)
            if not callable(register):
                register = getattr(module, "register_forward_hook", None)
            if not callable(register):
                continue

            def capture(
                module_obj: Any, inputs: Any, output: Any = None, *, name: str = path
            ) -> None:
                value = (
                    inputs[0] if output is None and isinstance(inputs, tuple) and inputs else output
                )
                observed.append((name, value))

            handles.append(register(capture))
        try:
            if train:
                raw = self.model(model_input)
            else:
                with torch.no_grad():
                    raw = self.model(model_input)
        finally:
            for handle in handles:
                remove = getattr(handle, "remove", None)
                if callable(remove):
                    remove()
        batch_size = batch.batch_size
        sequence: Any | None = None
        source = "model_output"
        if self.feature_stage in {"backbone_tokens", "observed_backbone"}:
            for path, value in reversed(observed):
                if path == "norm":
                    candidate = _candidate_activation(value, batch_size)
                    if candidate is not None:
                        sequence, source = candidate, "hook:norm"
                        break
        if sequence is None:
            for path, value in reversed(observed):
                candidate = _candidate_activation(value, batch_size)
                if candidate is not None:
                    sequence, source = candidate, f"hook:{path}"
                    break
        if sequence is None:
            sequence = _candidate_activation(raw, batch_size)
            source = "model_output"
        if sequence is None:
            raise RuntimeError("无法从 TorchVision 输出提取表征")
        sequence = normalize_feature_tensor(sequence, batch_size=batch_size)
        token_count = int(sequence.shape[1])
        pooled = sequence.mean(dim=1) if _is_torch(sequence) else np.asarray(sequence).mean(axis=1)
        timeline = _timeline(batch, token_count)
        output = normalize_encoder_output(
            sequence,
            timeline=timeline,
            feature_stage=self.feature_stage,
            pooled=pooled,
            sequence_source=source,
            preprocess_profile=f"torchvision-{self.variant}-v1",
            aux={
                "adapter": "torchvision_video",
                "variant": self.variant,
                "feature_source": source,
                "frame_stride": self.frame_stride,
                "image_size": self.image_size,
                "checkpoint_loaded": self._checkpoint_report is not None,
                "checkpoint": self._checkpoint_report,
            },
        )
        validate_encoder_output(output, batch)
        return output


R2Plus1DAdapter = TorchvisionVideoAdapter
MViTv2Adapter = TorchvisionVideoAdapter
VideoSwinAdapter = TorchvisionVideoAdapter

__all__ = [
    "DEFAULT_CAPABILITIES",
    "MViTv2Adapter",
    "R2Plus1DAdapter",
    "TorchvisionVideoAdapter",
    "VideoSwinAdapter",
]
