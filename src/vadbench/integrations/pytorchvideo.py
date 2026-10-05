"""PyTorchVideo family adapter.

The upstream project is intentionally optional.  Importing this module only
loads NumPy and the VADBench contracts; ``torch`` and ``pytorchvideo`` are
resolved when a real model is constructed.  This keeps catalog/list commands
usable on machines that do not have the (rather old) PyTorchVideo dependency
stack installed.

The adapter owns the small amount of model-family-specific plumbing required
by PyTorchVideo:

* canonical ``ClipBatch.frames`` (``BTHWC uint8``) become normalized ``BCTHW``;
* SlowFast receives ``[slow_pathway, fast_pathway]`` with a configurable
  temporal alpha;
* a pre-classification block is observed when available, so the classifier
  logits are not silently presented as a representation;
* all outputs go through the shared ``normalize_encoder_output`` helper and a
  one-token temporal provenance record.

Only local checkpoints are accepted.  In particular, this adapter never calls
``torch.hub`` and never asks an upstream constructor for ``pretrained=True``.
Tests and downstream integrations may inject a model/factory, which is useful
for dependency-isolated contract tests and does not change the real loading
policy.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
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
from vadbench.integrations.common import normalize_encoder_output

DEFAULT_MEAN = (0.45, 0.45, 0.45)
DEFAULT_STD = (0.225, 0.225, 0.225)

# Keep this object exactly equal to the static catalog capability declaration.
# The catalog deliberately leaves fixed_num_frames unset: temporal sample
# lengths are selected by the definition/sampler, while the CNNs themselves
# can consume more than one nominal clip length.
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


_VARIANT_ALIASES = {
    "i3d": "i3d_r50",
    "i3d_r50": "i3d_r50",
    "i3d-r50": "i3d_r50",
    "i3d_8x8_r50": "i3d_r50",
    "i3d-8x8-r50": "i3d_r50",
    "x3d": "x3d_s",
    "x3d_s": "x3d_s",
    "x3d-s": "x3d_s",
    "x3d_small": "x3d_s",
    "x3d-small": "x3d_s",
    "slowfast": "slowfast_r50",
    "slowfast_r50": "slowfast_r50",
    "slowfast-r50": "slowfast_r50",
    "slowfast_8x8_r50": "slowfast_r50",
    "slowfast-8x8-r50": "slowfast_r50",
}

_CONSTRUCTOR_NAMES = {
    "i3d_r50": ("i3d_r50", "create_i3d_r50", "create_i3d"),
    "x3d_s": ("x3d_s", "create_x3d_s", "create_x3d"),
    "slowfast_r50": ("slowfast_r50", "create_slowfast_r50", "create_slowfast"),
}

_SEQUENCE_KEYS = (
    "features",
    "feature",
    "x",
    "last_hidden_state",
    "hidden_states",
    "backbone_features",
    "backbone",
    "pooler_output",
    "pooled_output",
    "pooled",
    "logits",
)


def _normalize_variant(value: str | None, model_name: str | None = None) -> str:
    candidate = value if value is not None else model_name
    if candidate is None:
        candidate = "x3d_s"
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError("variant/model_name 必须是非空字符串")
    normalized = candidate.strip().lower().replace(" ", "_")
    try:
        return _VARIANT_ALIASES[normalized]
    except KeyError as exc:
        expected = ", ".join(sorted({"i3d_r50", "x3d_s", "slowfast_r50"}))
        raise ValueError(
            f"不支持的 PyTorchVideo variant={candidate!r}；允许值：{expected}"
        ) from exc


def _positive_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数" + (" 或 None" if allow_none else ""))
    return int(value)


def _float_tuple(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray)) or len(value) != 3:
        raise ValueError(f"{name} 必须包含 3 个数值")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} 必须全部是有限数值")
    return result  # type: ignore[return-value]


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


def _module_root(value: Any) -> str:
    return type(value).__module__.split(".", 1)[0]


def _to_numpy(value: Any, name: str = "value") -> np.ndarray:
    try:
        if _module_root(value) == "torch" and hasattr(value, "detach"):
            tensor = value.detach().cpu()
            if str(getattr(tensor, "dtype", "")).lower() in {"torch.bfloat16", "bfloat16"}:
                tensor = tensor.float()
            return tensor.numpy()
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - exotic third-party tensor
        raise ValueError(f"{name} 无法转换为 NumPy 数组") from exc


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return (
        cls.__qualname__ if cls.__module__ == "builtins" else f"{cls.__module__}.{cls.__qualname__}"
    )


def _get_path(value: Any, path: str) -> Any | None:
    current = value
    for part in path.split("."):
        if not part:
            return None
        if part.isdigit():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError):
                return None
            continue
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _load_torch(module: Any | None = None) -> Any:
    if module is not None:
        return module
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on minimal envs
        raise ImportError(
            "PyTorchVideo adapter 需要可选依赖 torch；请安装 pytorchvideo 对应环境"
        ) from exc
    return torch


def _load_pytorchvideo_constructor(variant: str) -> Callable[..., Any]:
    """Resolve a constructor without invoking ``torch.hub``.

    PyTorchVideo releases have exposed the model-zoo constructors from slightly
    different modules.  We probe those import paths lazily and let the caller
    pass ``pretrained=False`` explicitly.  Import failures are aggregated into
    one actionable message.
    """

    import_errors: list[str] = []
    names = _CONSTRUCTOR_NAMES[variant]
    module_candidates = (
        "pytorchvideo.models.hub",
        "pytorchvideo.models.hub.resnet",
        "pytorchvideo.models.hub.x3d",
        "pytorchvideo.models.hub.slowfast",
        "pytorchvideo.models",
        "pytorchvideo.models.resnet",
        "pytorchvideo.models.x3d",
        "pytorchvideo.models.slowfast",
    )
    for module_name in module_candidates:
        try:
            module = __import__(module_name, fromlist=["*"])
        except ImportError as exc:
            import_errors.append(f"{module_name}: {exc}")
            continue
        for name in names:
            constructor = getattr(module, name, None)
            if callable(constructor):
                return constructor
    detail = "; ".join(import_errors[-3:])
    raise ImportError(
        f"当前环境未找到 PyTorchVideo {variant} 构造器；请安装 pytorchvideo（{detail}）"
    )


def _call_constructor(constructor: Callable[..., Any]) -> Any:
    """Call an upstream constructor while forbidding online pretrained loads."""

    # Most releases accept (pretrained=False, progress=True).  Signature
    # filtering avoids passing unsupported kwargs to small fake constructors.
    try:
        parameters = inspect.signature(constructor).parameters
    except (TypeError, ValueError):  # pragma: no cover - C extension callable
        parameters = {}
    accepts_any = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if not parameters or accepts_any or "pretrained" in parameters:
        kwargs["pretrained"] = False
    if not parameters or accepts_any or "progress" in parameters:
        kwargs["progress"] = False
    # A constructor that does not expose either argument is still safe: it is
    # expected to create an uninitialized model, and checkpoint loading follows.
    try:
        model = constructor(**kwargs)
    except TypeError:
        # Some old functions use a positional ``pretrained`` argument despite
        # advertising ``*args``.  Retry only with the explicit false value;
        # never retry with True or call torch.hub.
        if kwargs.get("pretrained") is not False:
            raise
        try:
            model = constructor(False)
        except TypeError:
            model = constructor()
    if model is None:
        raise RuntimeError("PyTorchVideo 构造器返回 None")
    return model


def _checkpoint_file(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"本地 checkpoint 不存在：{checkpoint}")
    if checkpoint.is_file():
        return checkpoint.resolve()
    candidates: list[Path] = []
    for suffix in ("*.pyth", "*.pth", "*.pt", "*.bin", "*.safetensors"):
        candidates.extend(checkpoint.glob(suffix))
    candidates = sorted({item.resolve() for item in candidates})
    if not candidates:
        raise FileNotFoundError(f"checkpoint 目录中没有可加载权重：{checkpoint}")
    if len(candidates) > 1:
        names = ", ".join(item.name for item in candidates)
        raise ValueError(f"checkpoint 目录包含多个候选文件，请显式指定一个：{names}")
    return candidates[0]


def _torch_load(torch: Any, path: Path) -> Any:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("加载 .safetensors 需要安装 safetensors") from exc
        return load_file(str(path), device="cpu")
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only keyword
        return torch.load(str(path), map_location="cpu")


def _looks_like_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(isinstance(key, str) for key in value) and any(
        hasattr(item, "shape") for item in value.values()
    )


def _extract_state_dict(payload: Any) -> Mapping[str, Any]:
    if _looks_like_state_dict(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in (
            "model_state",
            "state_dict",
            "model_state_dict",
            "state",
            "model",
            "ema_state_dict",
        ):
            candidate = payload.get(key)
            if _looks_like_state_dict(candidate):
                return candidate
    raise ValueError("checkpoint 不包含可识别的 model state_dict/model_state")


def _strip_prefix(state: Mapping[str, Any], prefixes: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in state.items():
        normalized = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
                    break
        result[normalized] = value
    return result


def _load_local_checkpoint(model: Any, path: Path, *, strict: bool, torch: Any) -> dict[str, Any]:
    if not callable(getattr(model, "load_state_dict", None)):
        raise TypeError("PyTorchVideo model 缺少 load_state_dict，无法加载本地 checkpoint")
    payload = _torch_load(torch, path)
    state = _extract_state_dict(payload)
    variants = [
        dict(state),
        _strip_prefix(state, ("module.",)),
        _strip_prefix(state, ("module.", "model.")),
        _strip_prefix(state, ("model.", "backbone.")),
    ]
    selected = variants[0]
    model_state = None
    with suppress(Exception):
        model_state = model.state_dict()
    if isinstance(model_state, Mapping) and model_state:
        selected = max(
            variants,
            key=lambda candidate: sum(key in model_state for key in candidate),
        )
    try:
        result = model.load_state_dict(selected, strict=strict)
    except TypeError:
        result = model.load_state_dict(selected)
    missing = list(getattr(result, "missing_keys", ()) or ())
    unexpected = list(getattr(result, "unexpected_keys", ()) or ())
    return {
        "path": str(path),
        "strict": bool(strict),
        "state_keys": len(selected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _valid_frame_arrays(batch: ClipBatch) -> np.ndarray:
    """Copy frames to a dense uint8 array and replicate padded suffixes."""

    frames = _to_numpy(batch.frames, "frames")
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8, copy=False)
    # ClipBatch already checked the dimensions.  A copy lets us replace padded
    # suffixes without mutating caller-owned data.
    frames = np.array(frames, dtype=np.uint8, copy=True)
    lengths = batch.valid_lengths
    for row, length_value in enumerate(lengths):
        length = int(length_value)
        if length < frames.shape[1]:
            frames[row, length:] = frames[row, length - 1 : length]
    return frames


def _resize_crop_numpy(
    value: np.ndarray, resize_size: int | None, crop_size: int | None
) -> np.ndarray:
    """Small nearest-neighbor fallback used only when torch is unavailable."""

    if resize_size is None and crop_size is None:
        return value
    batch, channels, time, height, width = value.shape
    target_resize = resize_size
    if target_resize is not None:
        scale = target_resize / min(height, width)
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        y = np.minimum((np.arange(new_h) * height / new_h).astype(int), height - 1)
        x = np.minimum((np.arange(new_w) * width / new_w).astype(int), width - 1)
        value = value[:, :, :, y][:, :, :, :, x]
        height, width = new_h, new_w
    if crop_size is None:
        return value
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if pad_h or pad_w:
        value = np.pad(
            value,
            (
                (0, 0),
                (0, 0),
                (0, 0),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            ),
            mode="edge",
        )
        height, width = value.shape[-2:]
    top = max(0, (height - crop_size) // 2)
    left = max(0, (width - crop_size) // 2)
    return value[:, :, :, top : top + crop_size, left : left + crop_size]


def _prepare_numpy(
    frames: np.ndarray,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    resize_size: int | None,
    crop_size: int | None,
) -> np.ndarray:
    value = np.transpose(frames, (0, 4, 1, 2, 3)).astype(np.float32) / 255.0
    value = (value - np.asarray(mean, dtype=np.float32)[None, :, None, None, None]) / np.asarray(
        std, dtype=np.float32
    )[None, :, None, None, None]
    return _resize_crop_numpy(value, resize_size, crop_size)


def _prepare_torch(
    torch: Any,
    frames: np.ndarray,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    resize_size: int | None,
    crop_size: int | None,
    device: str | None,
    dtype: str | None,
) -> Any:
    value = torch.from_numpy(np.transpose(frames, (0, 4, 1, 2, 3))).to(dtype=torch.float32)
    value = value.div(255.0)
    mean_tensor = torch.tensor(mean, dtype=value.dtype, device=value.device).view(1, 3, 1, 1, 1)
    std_tensor = torch.tensor(std, dtype=value.dtype, device=value.device).view(1, 3, 1, 1, 1)
    value = (value - mean_tensor) / std_tensor

    if resize_size is not None or crop_size is not None:
        import torch.nn.functional as functional

        batch, channels, time, height, width = value.shape
        if resize_size is not None:
            scale = resize_size / min(height, width)
            new_h = max(1, int(round(height * scale)))
            new_w = max(1, int(round(width * scale)))
            if (new_h, new_w) != (height, width):
                image = value.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width)
                image = functional.interpolate(
                    image, size=(new_h, new_w), mode="bilinear", align_corners=False
                )
                value = image.reshape(batch, time, channels, new_h, new_w).permute(0, 2, 1, 3, 4)
                height, width = new_h, new_w
        if crop_size is not None:
            pad_h = max(0, crop_size - height)
            pad_w = max(0, crop_size - width)
            if pad_h or pad_w:
                value = functional.pad(
                    value,
                    (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                    mode="replicate",
                )
                height, width = value.shape[-2:]
            top = max(0, (height - crop_size) // 2)
            left = max(0, (width - crop_size) // 2)
            value = value[..., top : top + crop_size, left : left + crop_size]
    if device:
        value = value.to(device)
    if dtype:
        torch_dtype = getattr(torch, dtype, None)
        if torch_dtype is None:
            raise ValueError(f"不支持的 torch dtype={dtype!r}")
        value = value.to(dtype=torch_dtype)
    return value


def _shape_is_batch_tensor(value: Any, batch_size: int) -> bool:
    shape = _shape(value)
    return bool(
        shape and len(shape) >= 2 and shape[0] == batch_size and all(item > 0 for item in shape)
    )


def _reduce_activation(value: Any, batch_size: int) -> Any | None:
    """Turn one activation into a pooled ``[B,D]`` tensor/array."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        # SlowFast and a few wrappers name their two paths explicitly.
        for keys in (("slow_pathway", "fast_pathway"), ("slow", "fast")):
            if all(key in value for key in keys):
                left = _reduce_activation(value[keys[0]], batch_size)
                right = _reduce_activation(value[keys[1]], batch_size)
                if left is not None and right is not None:
                    return _concat_features(left, right)
        for key in _SEQUENCE_KEYS:
            if key in value:
                found = _reduce_activation(value[key], batch_size)
                if found is not None:
                    return found
        for candidate in reversed(tuple(value.values())):
            found = _reduce_activation(candidate, batch_size)
            if found is not None:
                return found
        return None
    # Objects such as ModelOutput expose fields as attributes rather than a
    # Mapping.  Prefer semantic fields before treating the object as a tensor.
    for key in _SEQUENCE_KEYS:
        candidate = getattr(value, key, None)
        if candidate is not None and candidate is not value:
            found = _reduce_activation(candidate, batch_size)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)) and not isinstance(value, (str, bytes, bytearray)):
        tensor_candidates = [
            _reduce_activation(item, batch_size)
            for item in value
            if _shape_is_batch_tensor(item, batch_size)
        ]
        tensor_candidates = [item for item in tensor_candidates if item is not None]
        # A two-path SlowFast output has 4-D/5-D members and should retain both
        # pathways.  Hidden-state tuples, by contrast, are 3-D and should use
        # the final layer only.
        if (
            len(value) >= 2
            and len(tensor_candidates) >= 2
            and all(
                (_shape(item) or ())[0:1] == (batch_size,) and len(_shape(item) or ()) >= 4
                for item in value
            )
        ):
            result = tensor_candidates[0]
            for item in tensor_candidates[1:]:
                result = _concat_features(result, item)
            return result
        for item in reversed(value):
            found = _reduce_activation(item, batch_size)
            if found is not None:
                return found
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
    module_root = _module_root(value)
    # A three-dimensional activation is conventionally [B,S,D], while a
    # convolutional activation has channel at axis 1 and spatial/temporal
    # dimensions from axis 2 onward. Preserve the representation dimension in
    # both cases; averaging all axes would collapse it to a scalar.
    axes = (1,) if len(shape) == 3 else tuple(range(2, len(shape)))
    # [B,S,D] and [B,C,T,H,W] both become [B,D/C]. For torch, keeping the
    # operation in the original device/dtype avoids a needless host copy.
    if module_root == "torch" and hasattr(value, "mean"):
        return value.mean(dim=axes)
    return np.asarray(value).mean(axis=axes)


def _concat_features(left: Any, right: Any) -> Any:
    if _module_root(left) == "torch" and _module_root(right) == "torch":
        import torch

        return torch.cat((left, right), dim=-1)
    return np.concatenate((np.asarray(left), np.asarray(right)), axis=-1)


def _candidate_modules(model: Any, feature_layer: str | None) -> list[tuple[str, Any]]:
    target = model.module if getattr(model, "module", None) is not None else model
    candidates: list[tuple[str, Any]] = []
    if feature_layer:
        module = _get_path(target, feature_layer)
        if module is None:
            raise ValueError(f"feature_layer 不存在：{feature_layer}")
        candidates.append((feature_layer, module))
    blocks = getattr(target, "blocks", None)
    try:
        if blocks is not None and len(blocks) >= 2:
            candidates.append(("blocks[-2]", blocks[-2]))
    except (TypeError, IndexError):
        pass
    for path in ("layer4", "features", "backbone", "trunk"):
        module = getattr(target, path, None)
        if module is not None:
            candidates.append((path, module))
    # Remove duplicate module identities while preserving preference order.
    unique: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for path, module in candidates:
        if id(module) not in seen:
            seen.add(id(module))
            unique.append((path, module))
    return unique


def _infer_device(model: Any) -> str | None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        parameter = next(parameters())
    except (StopIteration, TypeError):
        return None
    device = getattr(parameter, "device", None)
    return None if device is None else str(device)


def _singleton_timeline(batch: ClipBatch) -> TokenTimeline:
    times = _to_numpy(batch.timestamps_s, "timestamps_s").astype(np.float64, copy=False)
    indices = (
        np.broadcast_to(
            np.arange(batch.num_frames, dtype=np.int64), (batch.batch_size, batch.num_frames)
        )
        if batch.frame_indices is None
        else _to_numpy(batch.frame_indices, "frame_indices").astype(np.int64, copy=False)
    )
    valid = (
        np.ones((batch.batch_size, batch.num_frames), dtype=bool)
        if batch.valid_mask is None
        else _to_numpy(batch.valid_mask, "valid_mask").astype(bool, copy=False)
    )
    starts: list[float] = []
    ends: list[float] = []
    frame_starts: list[int] = []
    frame_ends: list[int] = []
    for row in range(batch.batch_size):
        row_times = times[row, valid[row]]
        row_indices = indices[row, valid[row]]
        starts.append(float(row_times[0]))
        positive_diffs = np.diff(row_times)
        positive_diffs = positive_diffs[positive_diffs > 0]
        if positive_diffs.size:
            fallback = float(np.median(positive_diffs))
        else:
            fps_value = batch.metadata.get("fps")
            try:
                fps = float(fps_value)
            except (TypeError, ValueError):
                fps = 0.0
            fallback = 1.0 / fps if math.isfinite(fps) and fps > 0 else 0.0
        ends.append(float(row_times[-1] + fallback))
        frame_starts.append(int(row_indices[0]))
        frame_ends.append(int(row_indices[-1]) + 1)
    return TokenTimeline(
        start_s=np.asarray(starts, dtype=np.float64)[:, None],
        end_s=np.asarray(ends, dtype=np.float64)[:, None],
        source_frame_start=np.asarray(frame_starts, dtype=np.int64)[:, None],
        source_frame_end=np.asarray(frame_ends, dtype=np.int64)[:, None],
    )


class PytorchVideoAdapter(VideoEncoderAdapter):
    """Adapt I3D-R50, X3D-S and SlowFast-R50 to VADBench."""

    capabilities = DEFAULT_CAPABILITIES

    def __init__(
        self,
        *,
        variant: str | None = None,
        model_name: str | None = None,
        checkpoint: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        model: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
        image_size: int | None = 224,
        resize_size: int | None = None,
        crop_size: int | None = None,
        clip_frames: int | None = None,
        num_frames: int | None = None,
        frame_stride: int = 1,
        stride: int | None = None,
        slow_fast_alpha: int = 4,
        pathway_alpha: int | None = None,
        mean: Sequence[float] = DEFAULT_MEAN,
        std: Sequence[float] = DEFAULT_STD,
        device: str | None = None,
        dtype: str | None = None,
        feature_stage: str = "pooled",
        torch_module: Any | None = None,
        input_backend: str = "auto",
        feature_layer: str | None = None,
        strict_checkpoint: bool = False,
        pretrained: bool = False,
        integration_id: str | None = None,
        preprocess_profile: str | None = None,
        **_: Any,
    ) -> None:
        self.variant = _normalize_variant(variant, model_name)
        self.backend = "pytorchvideo"
        self.integration_id = integration_id
        if (
            checkpoint is not None
            and checkpoint_path is not None
            and Path(checkpoint) != Path(checkpoint_path)
        ):
            raise ValueError("checkpoint 与 checkpoint_path 指向不同文件")
        selected_checkpoint = checkpoint if checkpoint is not None else checkpoint_path
        if pretrained:
            raise ValueError(
                "PyTorchVideo adapter 只允许本地 checkpoint，禁止 pretrained/torch.hub 下载"
            )
        self.checkpoint_path = (
            None if selected_checkpoint is None else _checkpoint_file(selected_checkpoint)
        )
        self.image_size = _positive_int(image_size, "image_size", allow_none=True)
        self.resize_size = _positive_int(resize_size, "resize_size", allow_none=True)
        self.crop_size = _positive_int(crop_size, "crop_size", allow_none=True)
        if self.resize_size is None:
            self.resize_size = self.image_size
        if self.crop_size is None:
            self.crop_size = self.image_size
        self.clip_frames = _positive_int(
            num_frames if num_frames is not None else clip_frames,
            "clip_frames",
            allow_none=True,
        )
        selected_stride = stride if stride is not None else frame_stride
        self.frame_stride = _positive_int(selected_stride, "frame_stride")
        selected_alpha = pathway_alpha if pathway_alpha is not None else slow_fast_alpha
        self.slow_fast_alpha = _positive_int(selected_alpha, "slow_fast_alpha")
        self.mean = _float_tuple(mean, "mean")
        self.std = _float_tuple(std, "std")
        if any(item == 0 for item in self.std):
            raise ValueError("std 不能包含 0")
        if input_backend not in {"auto", "torch", "numpy"}:
            raise ValueError("input_backend 必须是 auto、torch 或 numpy")
        self.input_backend = input_backend
        normalized_stage = str(feature_stage).strip().lower().replace("-", "_")
        if normalized_stage not in {"pooled", "backbone_tokens", "observed_backbone"}:
            raise ValueError("feature_stage 必须是 pooled、backbone_tokens 或 observed_backbone")
        self.feature_stage = normalized_stage
        self._torch_module = torch_module
        self.preprocess_profile = preprocess_profile
        self.device = device
        self.dtype = dtype
        self.feature_layer = feature_layer
        self.strict_checkpoint = bool(strict_checkpoint)
        self._checkpoint_report: dict[str, Any] | None = None
        self._last_feature_source = "model_output"
        self.model = model

        if self.model is None:
            if model_factory is not None:
                if self.checkpoint_path is None:
                    raise ValueError("model_factory 路径也必须提供本地 checkpoint")
                self.model = self._call_model_factory(model_factory)
            else:
                if self.checkpoint_path is None:
                    raise ValueError(
                        "未提供本地 checkpoint；必须提供本地 checkpoint 才能构造真实 PyTorchVideo model，不允许自动下载权重"
                    )
                torch = _load_torch(self._torch_module)
                constructor = _load_pytorchvideo_constructor(self.variant)
                self.model = _call_constructor(constructor)
            torch = _load_torch(self._torch_module)
            self._checkpoint_report = _load_local_checkpoint(
                self.model,
                self.checkpoint_path,
                strict=self.strict_checkpoint,
                torch=torch,
            )
        elif self.checkpoint_path is not None:
            # Injection is primarily a testing/advanced-use escape hatch, but
            # an explicitly supplied local checkpoint must still be honoured.
            torch = _load_torch(self._torch_module)
            self._checkpoint_report = _load_local_checkpoint(
                self.model,
                self.checkpoint_path,
                strict=self.strict_checkpoint,
                torch=torch,
            )
        self._prepare_model()

    def _call_model_factory(self, factory: Callable[..., Any]) -> Any:
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_any = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {}
        if not parameters or accepts_any or "variant" in parameters:
            kwargs["variant"] = self.variant
        if not parameters or accepts_any or "pretrained" in parameters:
            kwargs["pretrained"] = False
        try:
            model = factory(**kwargs)
        except TypeError:
            try:
                model = factory(self.variant)
            except TypeError:
                model = factory()
        if model is None:
            raise RuntimeError("model_factory 返回 None")
        return model

    def _prepare_model(self) -> None:
        if self.device and callable(getattr(self.model, "to", None)):
            self.model.to(self.device)
        if callable(getattr(self.model, "eval", None)):
            self.model.eval()

    def _model_input(self, batch: ClipBatch) -> tuple[Any, dict[str, Any]]:
        frames = _valid_frame_arrays(batch)
        use_numpy = self.input_backend == "numpy"
        torch = None
        if not use_numpy:
            try:
                torch = _load_torch(self._torch_module)
            except ImportError:
                if self.input_backend == "torch":
                    raise
                use_numpy = True
        if use_numpy:
            tensor = _prepare_numpy(
                frames,
                mean=self.mean,
                std=self.std,
                resize_size=self.resize_size,
                crop_size=self.crop_size,
            )
        else:
            tensor = _prepare_torch(
                torch,
                frames,
                mean=self.mean,
                std=self.std,
                resize_size=self.resize_size,
                crop_size=self.crop_size,
                device=self.device,
                dtype=self.dtype,
            )
        tensor_dtype = getattr(tensor, "dtype", None)
        if tensor_dtype is None:
            tensor_dtype = np.asarray(tensor).dtype
        metadata: dict[str, Any] = {
            "input_layout": "BCTHW",
            "input_dtype": str(tensor_dtype),
            "input_shape": list(_shape(tensor) or ()),
            "resize_size": self.resize_size,
            "crop_size": self.crop_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "clip_frames": self.clip_frames,
            "sampling_frame_stride": self.frame_stride,
            "frame_stride": self.frame_stride,
            "sampling_applied_externally": True,
        }
        if self.variant == "slowfast_r50":
            if _module_root(tensor) == "torch":
                slow = tensor[:, :, :: self.slow_fast_alpha, :, :]
            else:
                slow = np.asarray(tensor)[:, :, :: self.slow_fast_alpha, :, :]
            # A one-frame clip still needs a non-empty slow pathway.
            if (_shape(slow) or (0, 0, 0))[2] == 0:
                slow = tensor[:, :, :1, :, :]
            model_input: Any = [slow, tensor]
            metadata.update(
                {
                    "pathways": ["slow", "fast"],
                    "slow_fast_alpha": self.slow_fast_alpha,
                    "slow_pathway_shape": list(_shape(slow) or ()),
                    "fast_pathway_shape": list(_shape(tensor) or ()),
                }
            )
        else:
            model_input = tensor
            metadata["pathways"] = ["rgb"]
        return model_input, metadata

    def _forward(self, model_input: Any, *, train: bool) -> tuple[Any, str, Any]:
        if callable(getattr(self.model, "train", None)):
            self.model.train(bool(train))
        observed: list[Any] = []
        handles: list[Any] = []
        source = "model_output"
        for path, module in _candidate_modules(self.model, self.feature_layer):
            register = getattr(module, "register_forward_hook", None)
            if callable(register):

                def observe(_module: Any, _inputs: Any, output: Any, *, _path: str = path) -> None:
                    observed.append((_path, output))

                handles.append(register(observe))
        try:
            if train:
                raw_output = self.model(model_input)
            else:
                torch = None
                with suppress(ImportError):
                    torch = _load_torch(self._torch_module)
                if torch is not None:
                    with torch.no_grad():
                        raw_output = self.model(model_input)
                else:
                    raw_output = self.model(model_input)
        finally:
            for handle in handles:
                remove = getattr(handle, "remove", None)
                if callable(remove):
                    remove()

        # Prefer the first (most specific) pre-head module that yielded a
        # usable activation. Candidate ordering is blocks[-2], layer4, ...;
        # do not let a broad ``backbone`` wrapper's later hook hide the more
        # useful final stage. Inspect in reverse within each path to select the
        # latest output for repeated calls.
        batch_size = _batch_size_from_input(model_input)
        preferred_paths = [
            path for path, _module in _candidate_modules(self.model, self.feature_layer)
        ]
        for preferred_path in preferred_paths:
            for path, candidate in reversed(observed):
                if path != preferred_path:
                    continue
                pooled = _reduce_activation(candidate, batch_size)
                if pooled is not None:
                    shape = _shape(pooled)
                    if shape is not None and len(shape) == 2 and shape[0] == batch_size:
                        self._last_feature_source = f"hook:{path}"
                        return pooled, self._last_feature_source, raw_output
        pooled = _reduce_activation(raw_output, batch_size)
        if pooled is None:
            raise RuntimeError(
                f"无法从 PyTorchVideo 输出提取 [B,D] 表征；output_type={_qualified_type(raw_output)}"
            )
        self._last_feature_source = source
        return pooled, source, raw_output

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        validate_clip_for_capabilities(batch, self.capabilities, train=train)
        lengths = np.asarray(batch.valid_lengths, dtype=np.int64)
        if self.clip_frames is not None and np.any(lengths != self.clip_frames):
            raise ValueError(
                f"{self.variant} 要求 clip_frames={self.clip_frames}，实际={lengths.tolist()}"
            )
        model_input, input_metadata = self._model_input(batch)
        pooled, source, raw_output = self._forward(model_input, train=train)
        timeline = _singleton_timeline(batch)
        # Pass the pooled tensor as raw output so the common normalizer creates
        # the canonical [B,1,D] feature sequence without guessing which field
        # in a classification ModelOutput is meaningful.
        output = normalize_encoder_output(
            pooled,
            timeline=timeline,
            feature_stage=self.feature_stage,
            pooled=pooled,
            sequence_source="pooled_singleton",
            preprocess_profile=self.preprocess_profile or f"pytorchvideo-{self.variant}-v1",
            aux={
                "adapter": "pytorchvideo",
                "variant": self.variant,
                "backend": "pytorchvideo",
                "integration_id": self.integration_id,
                "feature_source": source,
                "raw_model_output_type": _qualified_type(raw_output),
                "checkpoint_loaded": self._checkpoint_report is not None,
                "checkpoint": None
                if self._checkpoint_report is None
                else dict(self._checkpoint_report),
                **input_metadata,
            },
        )
        validate_encoder_output(output, batch)
        return output


def _batch_size_from_input(value: Any) -> int:
    shape = _shape(value[-1]) if isinstance(value, (list, tuple)) and value else _shape(value)
    if not shape or shape[0] <= 0:
        raise RuntimeError("模型输入缺少有效 batch 维度")
    return int(shape[0])


# Common capitalization variants keep downstream configs/imports readable while
# the catalog's canonical lazy target remains ``PytorchVideoAdapter``.
PyTorchVideoAdapter = PytorchVideoAdapter
I3DAdapter = PytorchVideoAdapter
X3DAdapter = PytorchVideoAdapter
SlowFastAdapter = PytorchVideoAdapter

__all__ = [
    "DEFAULT_CAPABILITIES",
    "I3DAdapter",
    "PytorchVideoAdapter",
    "PyTorchVideoAdapter",
    "SlowFastAdapter",
    "X3DAdapter",
]
