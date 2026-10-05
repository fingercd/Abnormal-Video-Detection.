"""Lazy bridge for the historical C3D (Caffe) video model.

The original C3D release is a Python 2/Caffe project and cannot be imported in
the normal VADBench process.  This adapter therefore keeps the public boundary
small and dependency-free: callers provide a canonical :class:`ClipBatch`,
while the adapter owns the legacy ``BTHWC -> BCTHW`` conversion and translates a
legacy ``fc6``/``fc7`` activation into :class:`EncoderOutput`.

No Caffe, Torch, or upstream checkout is imported at module import time.  A
model/loader can be injected for tests or for an isolated worker.  Without an
injected object, only explicitly supplied local checkout/checkpoint assets are
accepted; this module never downloads, clones, or follows a URL implicitly.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from vadbench.contracts import (
    ClipBatch,
    ContractError,
    EncoderCapabilities,
    EncoderOutput,
    TokenTimeline,
    VideoEncoderAdapter,
    validate_clip_for_capabilities,
    validate_encoder_output,
)
from vadbench.integrations.common import (
    normalize_encoder_output,
    normalize_feature_tensor,
    select_feature_tensor,
    validate_output_health,
)

# Keep this object byte-for-byte equivalent (semantically) to the catalog's
# generic fixed-clip capability declaration.  The 16x112 values belong to the
# smoke/profile configuration, not to the static capability contract.
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
C3D_CAPABILITIES = DEFAULT_CAPABILITIES

_KNOWN_PROTOTXT_NAMES = (
    "c3d_feature_extraction.prototxt",
    "deploy.prototxt",
    "c3d_deploy.prototxt",
    "train_val.prototxt",
)


def _jsonable(value: Any) -> Any:
    """Convert small error/metadata values without importing a serializer."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class LegacyIntegrationError(RuntimeError):
    """Base class for machine-readable C3D bridge failures."""

    integration_id = "c3d"

    def __init__(
        self,
        message: str,
        *,
        code: str = "legacy_error",
        details: Mapping[str, Any] | None = None,
        integration_id: str = "c3d",
    ) -> None:
        self.integration_id = str(integration_id)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "integration_id": self.integration_id,
            "message": self.message,
            "details": _jsonable(self.details),
        }


class LegacyAssetError(FileNotFoundError, LegacyIntegrationError):
    """Fail-closed error for absent local assets or unsafe entrypoints."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "missing_asset",
        details: Mapping[str, Any] | None = None,
        integration_id: str = "c3d",
    ) -> None:
        LegacyIntegrationError.__init__(
            self,
            message,
            code=code,
            details=details,
            integration_id=integration_id,
        )


class LegacyDependencyError(LegacyAssetError):
    """Fail-closed error when the isolated legacy dependency is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "missing_dependency",
        details: Mapping[str, Any] | None = None,
        integration_id: str = "c3d",
    ) -> None:
        super().__init__(
            message,
            code=code,
            details=details,
            integration_id=integration_id,
        )


class LegacyWorkerError(LegacyIntegrationError):
    """Error returned by an injected or dynamically loaded legacy model."""


# Friendly aliases used by matrix/worker callers that share terminology with
# the long-video integrations.
MissingLegacyAssetError = LegacyAssetError
MissingAssetError = LegacyAssetError
ExternalAssetError = LegacyAssetError
StructuredLegacyError = LegacyIntegrationError


def _module_root(value: Any) -> str:
    return type(value).__module__.split(".", 1)[0]


def _to_numpy(value: Any, *, name: str = "value") -> np.ndarray:
    try:
        if _module_root(value) == "torch" and hasattr(value, "detach"):
            tensor = value.detach().cpu()
            # NumPy cannot represent torch bfloat16 directly.
            if str(getattr(tensor, "dtype", "")).lower() in {"torch.bfloat16", "bfloat16"}:
                tensor = tensor.float()
            return tensor.numpy()
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - unusual third-party array
        raise ValueError(f"{name} 无法转换为 NumPy") from exc


def _resolve_project_root(value: str | Path | None) -> Path:
    return Path.cwd().resolve() if value is None else Path(value).expanduser().resolve()


def _resolve_local_path(
    value: str | Path | None,
    *,
    root: Path,
    kind: str,
    explicit: bool = False,
) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "://" in text:
        raise LegacyAssetError(
            f"{kind} 不接受 URL；legacy adapter 禁止隐式联网",
            code="network_disabled",
            details={"kind": kind, "value": text},
        )
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    # ``explicit`` is kept for readable call sites and future policy checks;
    # resolving a missing path is intentionally allowed so the structured
    # asset report can name it precisely.
    _ = explicit
    return resolved


def _invoke_loader(
    loader: Callable[..., Any],
    *,
    checkout_path: Path | None,
    checkpoint_path: Path | None,
    prototxt_path: Path | None,
    device: str,
    feature_layer: str,
) -> Any:
    values = {
        "checkout_path": None if checkout_path is None else str(checkout_path),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "prototxt_path": None if prototxt_path is None else str(prototxt_path),
        "device": device,
        "feature_layer": feature_layer,
    }
    parameters = inspect.signature(loader).parameters
    kwargs = (
        values
        if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
        else {key: value for key, value in values.items() if key in parameters}
    )
    return loader(**kwargs)


def _load_entrypoint(
    checkout_path: Path,
    entrypoint: str,
    *,
    checkpoint_path: Path | None,
    prototxt_path: Path | None,
    device: str,
    feature_layer: str,
) -> Any:
    """Load ``relative_file.py:function`` without mutating ``sys.path``."""

    text = str(entrypoint).strip()
    file_name, separator, function_name = text.partition(":")
    if not separator or not file_name or not function_name or ":" in function_name:
        raise LegacyAssetError(
            "legacy entrypoint 必须采用 relative_file.py:function 格式",
            code="invalid_entrypoint",
            details={"entrypoint": text},
        )
    components = function_name.split(".")
    if any(not component.isidentifier() for component in components):
        raise LegacyAssetError(
            f"legacy entrypoint 函数名非法：{function_name!r}",
            code="invalid_entrypoint",
            details={"entrypoint": text},
        )
    relative = Path(file_name)
    if (
        relative.is_absolute()
        or "\\" in file_name
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise LegacyAssetError(
            "legacy entrypoint 必须是 checkout 内的相对路径，不能包含 ..",
            code="invalid_entrypoint",
            details={"entrypoint": text, "checkout_path": str(checkout_path)},
        )
    if relative.suffix.lower() != ".py":
        raise LegacyAssetError(
            "legacy entrypoint 文件必须是 .py",
            code="invalid_entrypoint",
            details={"entrypoint": text},
        )
    source = (checkout_path / relative).resolve()
    try:
        source.relative_to(checkout_path.resolve())
    except ValueError as exc:  # pragma: no cover - defensive after part check
        raise LegacyAssetError(
            "legacy entrypoint 不能越出 checkout_path",
            code="invalid_entrypoint",
            details={"entrypoint": text},
        ) from exc
    if not source.is_file():
        raise LegacyAssetError(
            f"legacy entrypoint 文件不存在：{source}",
            code="missing_entrypoint",
            details={"entrypoint": text, "path": str(source)},
        )
    module_name = f"vadbench_external_c3d_{abs(hash(source))}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise LegacyAssetError(
            f"无法建立 legacy entrypoint module spec：{source}",
            code="invalid_entrypoint",
            details={"path": str(source)},
        )
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise LegacyWorkerError(
            f"导入 legacy entrypoint 失败：{exc}",
            code="entrypoint_import_failed",
            details={"entrypoint": text, "path": str(source)},
        ) from exc
    target: Any = module
    for component in components:
        try:
            target = getattr(target, component)
        except AttributeError as exc:
            raise LegacyAssetError(
                f"legacy entrypoint 缺少 callable：{function_name}",
                code="missing_entrypoint",
                details={"entrypoint": text, "path": str(source)},
            ) from exc
    if not callable(target):
        raise LegacyAssetError(
            f"legacy entrypoint 不是 callable：{function_name}",
            code="invalid_entrypoint",
            details={"entrypoint": text},
        )
    return _invoke_loader(
        target,
        checkout_path=checkout_path,
        checkpoint_path=checkpoint_path,
        prototxt_path=prototxt_path,
        device=device,
        feature_layer=feature_layer,
    )


def _select_loaded_encoder(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("encoder", "model", "net", "network"):
            if key in value:
                return value[key]
    if isinstance(value, (tuple, list)) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if callable(getattr(item, "encode", None)) or callable(getattr(item, "forward", None)):
                return item
            if hasattr(item, "blobs"):
                return item
            if callable(item) and not isinstance(item, (str, bytes, bytearray)):
                return item
    return value


def _valid_frames(batch: ClipBatch) -> np.ndarray:
    frames = _to_numpy(batch.frames, name="frames")
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError(f"frames 必须是 BTHWC，实际 shape={frames.shape}")
    if frames.dtype != np.uint8:
        # ClipBatch normally catches this; retain an explicit guard for injected
        # duck-typed batches.
        raise ValueError(f"frames 必须是 uint8，实际 dtype={frames.dtype}")
    return frames


def _resize_nearest(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if (height, width) == (size, size):
        return frame
    y = np.rint(np.linspace(0, max(height - 1, 0), size)).astype(np.int64)
    x = np.rint(np.linspace(0, max(width - 1, 0), size)).astype(np.int64)
    return frame[y[:, None], x[None, :], :]


def _preprocess_c3d(
    batch: ClipBatch,
    *,
    clip_frames: int,
    image_size: int,
    channel_order: str,
    mean: Sequence[float],
    scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames = _valid_frames(batch)
    if clip_frames <= 0:
        raise ValueError("clip_frames 必须是正整数")
    if image_size <= 0:
        raise ValueError("image_size 必须是正整数")
    if channel_order not in {"RGB", "BGR"}:
        raise ValueError("channel_order 必须是 RGB 或 BGR")
    mean_array = np.asarray(tuple(mean), dtype=np.float32)
    if mean_array.shape != (3,) or not np.all(np.isfinite(mean_array)):
        raise ValueError("mean 必须是三个有限数值")
    if not math.isfinite(float(scale)):
        raise ValueError("scale 必须是有限数值")

    lengths = batch.valid_lengths.astype(np.int64, copy=False)
    processed: list[np.ndarray] = []
    adjustments: list[str] = []
    for row, raw_length in enumerate(lengths):
        length = int(raw_length)
        source = frames[row, :length]
        if length >= clip_frames:
            source = source[:clip_frames]
            adjustments.append("truncate_first_contiguous" if length > clip_frames else "none")
        else:
            pad_count = clip_frames - length
            source = np.concatenate((source, np.repeat(source[-1:], pad_count, axis=0)), axis=0)
            adjustments.append("repeat_last_frame")
        spatial = np.stack([_resize_nearest(frame, image_size) for frame in source], axis=0)
        if channel_order == "BGR":
            spatial = spatial[..., ::-1]
        spatial = spatial.astype(np.float32, copy=False)
        spatial = (spatial - mean_array.reshape(1, 1, 1, 3)) * float(scale)
        # T,H,W,C -> C,T,H,W; batch is added below.
        processed.append(np.transpose(spatial, (3, 0, 1, 2)))
    tensor = np.stack(processed, axis=0).astype(np.float32, copy=False)
    metadata = {
        "input_layout": "BCTHW",
        "input_dtype": str(tensor.dtype),
        "input_shape": list(tensor.shape),
        "clip_frames": int(clip_frames),
        "image_size": int(image_size),
        "channel_order": channel_order,
        "mean": mean_array.tolist(),
        "scale": float(scale),
        "temporal_policy": "first_contiguous_pad_or_truncate",
        "temporal_adjustment": adjustments,
    }
    return tensor, metadata


def _timeline_for_batch(batch: ClipBatch, token_count: int) -> TokenTimeline:
    if token_count <= 0:
        raise ValueError("token_count 必须大于 0")
    times = _to_numpy(batch.timestamps_s, name="timestamps_s").astype(np.float64, copy=False)
    if batch.frame_indices is None:
        indices = np.broadcast_to(
            np.arange(batch.num_frames, dtype=np.int64),
            (batch.batch_size, batch.num_frames),
        )
    else:
        indices = _to_numpy(batch.frame_indices, name="frame_indices").astype(np.int64, copy=False)
    starts = np.empty((batch.batch_size, token_count), dtype=np.float64)
    ends = np.empty_like(starts)
    frame_starts = np.empty((batch.batch_size, token_count), dtype=np.int64)
    frame_ends = np.empty_like(frame_starts)
    for row, raw_length in enumerate(batch.valid_lengths):
        length = int(raw_length)
        row_times = times[row, :length]
        row_indices = indices[row, :length]
        if length <= 0:  # pragma: no cover - ClipBatch rejects this
            raise ValueError("batch 至少需要一帧有效视频")
        positive = np.diff(row_times)
        positive = positive[positive > 0]
        delta = float(np.median(positive)) if positive.size else (1.0 / 30.0)
        if token_count == 1:
            starts[row, 0] = row_times[0]
            ends[row, 0] = row_times[-1] + delta
            frame_starts[row, 0] = row_indices[0]
            frame_ends[row, 0] = row_indices[-1] + 1
            continue
        slots = np.minimum((np.arange(token_count) * length) // token_count, length - 1)
        starts[row] = row_times[slots]
        per_frame_end = np.empty(length, dtype=np.float64)
        if length > 1:
            per_frame_end[:-1] = row_times[1:]
        per_frame_end[-1] = row_times[-1] + delta
        ends[row] = per_frame_end[slots]
        frame_starts[row] = row_indices[slots]
        frame_ends[row] = frame_starts[row] + 1
    return TokenTimeline(
        start_s=starts,
        end_s=ends,
        source_frame_start=frame_starts,
        source_frame_end=frame_ends,
    )


def _normalize_c3d_feature(
    raw_output: Any,
    *,
    batch_size: int,
    feature_layer: str,
    sequence_source: str | None = None,
) -> tuple[Any, str]:
    """Select one declared C3D activation and normalize it through ``common``."""

    candidate = raw_output
    source = sequence_source
    if source is None and isinstance(raw_output, Mapping):
        if feature_layer not in raw_output:
            raise LegacyWorkerError(
                f"legacy encoder 输出缺少 feature_layer={feature_layer!r}",
                code="invalid_output",
            )
        candidate = raw_output[feature_layer]
        source = feature_layer
    elif source is None:
        named = getattr(raw_output, feature_layer, None)
        if named is not None:
            candidate = named
            source = feature_layer
        elif not hasattr(raw_output, "shape"):
            raise LegacyWorkerError(
                f"legacy encoder 输出缺少 feature_layer={feature_layer!r}",
                code="invalid_output",
            )

    if not hasattr(candidate, "shape") and hasattr(candidate, "data"):
        candidate = candidate.data
    if not hasattr(candidate, "shape"):
        raise LegacyWorkerError(
            f"feature_layer={feature_layer!r} 必须直接提供数组或张量",
            code="invalid_output",
        )
    shape = tuple(int(item) for item in getattr(candidate, "shape", ()))
    if len(shape) == 5 and shape[2:] == (1, 1, 1):
        if shape[0] != batch_size:
            raise LegacyWorkerError(
                f"C3D feature batch={shape[0]} 与输入 batch={batch_size} 不一致",
                code="invalid_output",
            )
        reshape = getattr(candidate, "reshape", None)
        candidate = (
            reshape(shape[0], shape[1])
            if callable(reshape)
            else np.asarray(candidate).reshape(shape[0], shape[1])
        )

    try:
        selected, observed_source = select_feature_tensor(candidate, batch_size=batch_size)
        features = normalize_feature_tensor(selected, batch_size=batch_size)
    except ContractError as exc:
        raise LegacyWorkerError(
            f"C3D feature_layer={feature_layer!r} 无法规范化：{exc}",
            code="invalid_output",
        ) from exc
    return features, source or observed_source


def _caffe_available() -> bool:
    try:
        return importlib.util.find_spec("caffe") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


class LegacyVideoAdapter(VideoEncoderAdapter):
    """Adapt an explicit C3D/Caffe model or local legacy loader.

    Parameters intentionally accept several historical spellings (``model``,
    ``checkpoint``, ``model_path``) so an isolated worker can map upstream
    configuration without changing the VADBench contract.
    """

    capabilities = DEFAULT_CAPABILITIES
    integration_id = "c3d"
    backend = "legacy"

    def __init__(
        self,
        *,
        encoder: Any | None = None,
        model: Any | None = None,
        loader: Callable[..., Any] | None = None,
        load_model_fn: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
        checkout_path: str | Path | None = None,
        checkout: str | Path | None = None,
        checkpoint: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        model_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        prototxt_path: str | Path | None = None,
        deploy_prototxt: str | Path | None = None,
        project_root: str | Path | None = None,
        root: str | Path | None = None,
        entrypoint: str | None = None,
        clip_frames: int = 16,
        num_frames: int | None = None,
        image_size: int = 112,
        resize_size: int | None = None,
        frame_stride: int = 2,
        feature_layer: str = "fc6",
        data_blob: str = "data",
        channel_order: str = "BGR",
        mean: Sequence[float] | None = (90.0, 98.0, 102.0),
        scale: float = 1.0,
        device: str = "cpu",
        strict_assets: bool = True,
        caffe_module: Any | None = None,
        **_: Any,
    ) -> None:
        self.integration_id = type(self).integration_id
        self.backend = type(self).backend
        if encoder is not None and model is not None and encoder is not model:
            raise ValueError("encoder 与 model 不能同时指向不同对象")
        if loader is not None and load_model_fn is not None and loader is not load_model_fn:
            raise ValueError("loader 与 load_model_fn 不能同时指定不同 callable")
        selected_encoder = encoder if encoder is not None else model
        selected_loader = loader or load_model_fn or model_loader
        self.project_root = _resolve_project_root(
            project_root if project_root is not None else root
        )
        selected_checkout = checkout_path if checkout_path is not None else checkout
        selected_checkpoint = checkpoint
        for candidate in (checkpoint_path, model_path, weights_path):
            if candidate is not None:
                if selected_checkpoint is not None:
                    first = _resolve_local_path(
                        selected_checkpoint,
                        root=self.project_root,
                        kind="checkpoint_path",
                    )
                    second = _resolve_local_path(
                        candidate,
                        root=self.project_root,
                        kind="checkpoint_path",
                    )
                    if first != second:
                        raise ValueError("checkpoint、checkpoint_path、model_path 指向不同文件")
                selected_checkpoint = candidate
        selected_prototxt = prototxt_path if prototxt_path is not None else deploy_prototxt
        self.checkout_path = _resolve_local_path(
            selected_checkout, root=self.project_root, kind="checkout_path"
        )
        self.checkpoint_path = _resolve_local_path(
            selected_checkpoint, root=self.project_root, kind="checkpoint_path"
        )
        self.prototxt_path = _resolve_local_path(
            selected_prototxt, root=self.project_root, kind="prototxt_path"
        )
        self.clip_frames = int(num_frames if num_frames is not None else clip_frames)
        self.image_size = int(resize_size if resize_size is not None else image_size)
        self.frame_stride = int(frame_stride)
        self.feature_layer = str(feature_layer).strip()
        self.data_blob = str(data_blob).strip()
        self.channel_order = str(channel_order).upper().strip()
        self.mean = (0.0, 0.0, 0.0) if mean is None else tuple(float(item) for item in mean)
        self.scale = float(scale)
        self.device = str(device)
        self.strict_assets = bool(strict_assets)
        self.caffe_module = caffe_module
        self.entrypoint = entrypoint
        self.loader = selected_loader
        self.implementation_source = "injected_encoder"
        self.encoder: Any | None = selected_encoder
        # ``model`` is a conventional spelling in the other VADBench adapters.
        # Keep it as an alias for callers that inspect the loaded legacy object.
        self.model: Any | None = selected_encoder
        self._checkpoint_report: dict[str, Any] | None = None

        if not self.feature_layer:
            raise ValueError("feature_layer 不能为空")
        if not self.data_blob:
            raise ValueError("data_blob 不能为空")
        if self.clip_frames <= 0 or self.image_size <= 0 or self.frame_stride <= 0:
            raise ValueError("clip_frames、image_size、frame_stride 必须是正整数")
        if self.channel_order not in {"RGB", "BGR"}:
            raise ValueError("channel_order 必须是 RGB 或 BGR")
        if self.encoder is not None and self.entrypoint is not None:
            raise ValueError("已注入 encoder 时不能再指定 entrypoint")

        if isinstance(selected_loader, str):
            if self.entrypoint is not None:
                raise ValueError("loader 字符串与 entrypoint 不能同时指定")
            self.entrypoint = selected_loader
            selected_loader = None
            self.loader = None

        # An injected encoder is a complete alternative to local legacy
        # assets.  This is important for dependency-isolated contract tests and
        # does not weaken the fail-closed policy for loader/entrypoint paths.
        if self.encoder is None:
            self._check_explicit_paths()
        if self.encoder is None and selected_loader is not None:
            try:
                loaded = _invoke_loader(
                    selected_loader,
                    checkout_path=self.checkout_path,
                    checkpoint_path=self._checkpoint_file_if_present(required=False),
                    prototxt_path=self.prototxt_path,
                    device=self.device,
                    feature_layer=self.feature_layer,
                )
            except LegacyIntegrationError:
                raise
            except Exception as exc:
                raise LegacyWorkerError(
                    f"无法加载 C3D legacy model：{exc}",
                    code="model_load_failed",
                    details={"loader": getattr(selected_loader, "__name__", repr(selected_loader))},
                ) from exc
            self.encoder = _select_loaded_encoder(loaded)
            self.model = self.encoder
            self.implementation_source = "explicit_loader"
        elif self.encoder is None and self.entrypoint is not None:
            if self.checkout_path is None:
                raise LegacyAssetError(
                    "指定 entrypoint 时必须提供 checkout_path",
                    code="missing_asset",
                    details={"entrypoint": self.entrypoint, "checkout_path": None},
                )
            self._require_assets(require_checkout=True, require_checkpoint=self.strict_assets)
            try:
                loaded = _load_entrypoint(
                    self.checkout_path,
                    self.entrypoint,
                    checkpoint_path=self._checkpoint_file_if_present(required=False),
                    prototxt_path=self.prototxt_path,
                    device=self.device,
                    feature_layer=self.feature_layer,
                )
            except LegacyIntegrationError:
                raise
            except Exception as exc:
                raise LegacyWorkerError(
                    f"无法加载 C3D legacy entrypoint：{exc}",
                    code="model_load_failed",
                    details={"entrypoint": self.entrypoint},
                ) from exc
            self.encoder = _select_loaded_encoder(loaded)
            self.model = self.encoder
            self.implementation_source = "pinned_upstream_entrypoint"
        elif self.encoder is None:
            self._require_assets(require_checkout=True, require_checkpoint=True)
            self.encoder = self._load_caffe_encoder()
            self.model = self.encoder
            self.implementation_source = "caffe_local_assets"

        if self.encoder is None or not (
            callable(getattr(self.encoder, "encode", None))
            or callable(self.encoder)
            or (hasattr(self.encoder, "blobs") and callable(getattr(self.encoder, "forward", None)))
        ):
            raise LegacyWorkerError(
                "C3D legacy encoder 必须提供 encode/callable 或 Caffe blobs+forward 接口",
                code="worker_protocol_error",
            )
        self._prepare_encoder()

    def _check_explicit_paths(self) -> None:
        missing: list[dict[str, str]] = []
        # If a caller explicitly names an asset, report a missing path even for
        # an injected loader; this catches typos without imposing assets on fake
        # encoders that use no path at all.
        for kind, path in (
            ("checkout", self.checkout_path),
            ("checkpoint", self.checkpoint_path),
            ("prototxt", self.prototxt_path),
        ):
            if path is not None and not path.exists():
                missing.append({"kind": kind, "path": str(path)})
        if missing:
            raise LegacyAssetError(
                "C3D legacy 指定的本地资产不存在；已 fail-closed",
                code="missing_asset",
                details={"missing": missing},
            )

    def _require_assets(self, *, require_checkout: bool, require_checkpoint: bool) -> None:
        missing: list[dict[str, str]] = []
        if require_checkout:
            if self.checkout_path is None:
                missing.append({"kind": "checkout", "path": "<unset>"})
            elif not self.checkout_path.is_dir():
                missing.append({"kind": "checkout", "path": str(self.checkout_path)})
        if require_checkpoint:
            if self.checkpoint_path is None:
                missing.append({"kind": "checkpoint", "path": "<unset>"})
            elif not self.checkpoint_path.exists():
                missing.append({"kind": "checkpoint", "path": str(self.checkpoint_path)})
        if self.prototxt_path is not None and not self.prototxt_path.is_file():
            missing.append({"kind": "prototxt", "path": str(self.prototxt_path)})
        if missing:
            raise LegacyAssetError(
                "C3D legacy 缺少本地 upstream 资产；不会自动联网下载",
                code="missing_asset",
                details={"missing": missing},
            )

    def _checkpoint_file_if_present(self, *, required: bool) -> Path | None:
        path = self.checkpoint_path
        if path is None:
            if required:
                raise LegacyAssetError(
                    "未提供 C3D checkpoint",
                    code="missing_asset",
                    details={"kind": "checkpoint", "path": "<unset>"},
                )
            return None
        if path.is_file():
            return path
        if path.is_dir():
            candidates = sorted(
                item
                for item in path.iterdir()
                if item.is_file()
                and item.suffix.lower() in {".caffemodel", ".model", ".bin", ".pth"}
            )
            if len(candidates) == 1:
                return candidates[0]
            if required or candidates:
                raise LegacyAssetError(
                    "C3D checkpoint 目录必须恰好包含一个权重文件",
                    code="invalid_asset",
                    details={"path": str(path), "candidates": [str(item) for item in candidates]},
                )
            return None
        if required:
            raise LegacyAssetError(
                f"C3D checkpoint 不存在：{path}",
                code="missing_asset",
                details={"kind": "checkpoint", "path": str(path)},
            )
        return None

    def _find_prototxt(self) -> Path:
        if self.prototxt_path is not None:
            if not self.prototxt_path.is_file():
                raise LegacyAssetError(
                    f"C3D prototxt 不存在：{self.prototxt_path}",
                    code="missing_asset",
                    details={"kind": "prototxt", "path": str(self.prototxt_path)},
                )
            return self.prototxt_path
        assert self.checkout_path is not None
        for name in _KNOWN_PROTOTXT_NAMES:
            matches = sorted(self.checkout_path.rglob(name))
            if matches:
                return matches[0].resolve()
        matches = sorted(self.checkout_path.rglob("*.prototxt"))
        if matches:
            return matches[0].resolve()
        raise LegacyAssetError(
            "C3D checkout 中未找到 prototxt；请显式提供 prototxt_path",
            code="missing_asset",
            details={"checkout_path": str(self.checkout_path)},
        )

    def _load_caffe_encoder(self) -> Any:
        caffe = self.caffe_module
        if caffe is None:
            try:
                caffe = importlib.import_module("caffe")
            except (ImportError, ModuleNotFoundError) as exc:
                raise LegacyDependencyError(
                    "当前环境缺少 Caffe；请在隔离 legacy 环境安装 Caffe，或注入 loader/encoder",
                    details={
                        "dependency": "caffe",
                        "checkpoint": None
                        if self.checkpoint_path is None
                        else str(self.checkpoint_path),
                    },
                ) from exc
        checkpoint = self._checkpoint_file_if_present(required=True)
        prototxt = self._find_prototxt()
        if self.device.startswith("cuda"):
            set_mode_gpu = getattr(caffe, "set_mode_gpu", None)
            if callable(set_mode_gpu):
                set_mode_gpu()
            set_device = getattr(caffe, "set_device", None)
            if callable(set_device):
                try:
                    index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
                except ValueError:
                    index = 0
                set_device(index)
        else:
            set_mode_cpu = getattr(caffe, "set_mode_cpu", None)
            if callable(set_mode_cpu):
                set_mode_cpu()
        net_constructor = getattr(caffe, "Net", None)
        if not callable(net_constructor):
            raise LegacyDependencyError(
                "Caffe 模块缺少 Net callable",
                code="invalid_dependency",
                details={"dependency": "caffe", "module": repr(caffe)},
            )
        test_mode = getattr(caffe, "TEST", 0)
        try:
            return net_constructor(str(prototxt), str(checkpoint), test_mode)
        except Exception as exc:
            raise LegacyWorkerError(
                f"Caffe Net 初始化失败：{exc}",
                code="model_load_failed",
                details={"prototxt": str(prototxt), "checkpoint": str(checkpoint)},
            ) from exc

    def _prepare_encoder(self) -> None:
        if self.encoder is None:
            return
        to = getattr(self.encoder, "to", None)
        if callable(to) and self.device:
            with suppress(TypeError):
                to(self.device)
        eval_fn = getattr(self.encoder, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def validate_assets(self) -> dict[str, Any]:
        """Return a JSON-ready local asset report without importing Caffe."""

        return {
            "integration_id": self.integration_id,
            "checkout_path": None if self.checkout_path is None else str(self.checkout_path),
            "checkout_exists": bool(self.checkout_path and self.checkout_path.is_dir()),
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "checkpoint_exists": bool(self.checkpoint_path and self.checkpoint_path.exists()),
            "prototxt_path": None if self.prototxt_path is None else str(self.prototxt_path),
            "prototxt_exists": bool(self.prototxt_path and self.prototxt_path.is_file()),
            "caffe_available": _caffe_available(),
            "implementation_source": self.implementation_source,
        }

    def _call_encoder(self, model_input: np.ndarray, *, train: bool) -> Any:
        encode = getattr(self.encoder, "encode", None)
        if callable(encode):
            return encode(model_input, train=train)
        return self.encoder(model_input)

    def _call_caffe_net(self, model_input: np.ndarray) -> tuple[Any, str, Any]:
        assert self.encoder is not None
        blobs = getattr(self.encoder, "blobs", None)
        if not isinstance(blobs, Mapping) or self.data_blob not in blobs:
            raise LegacyWorkerError(
                f"Caffe Net 缺少 data blob={self.data_blob!r}",
                code="worker_protocol_error",
                details={"available_blobs": [] if blobs is None else list(blobs)},
            )
        data_blob = blobs[self.data_blob]
        reshape = getattr(data_blob, "reshape", None)
        if callable(reshape):
            reshape(*model_input.shape)
        target = getattr(data_blob, "data", data_blob)
        try:
            target[...] = model_input
        except Exception as exc:
            try:
                data_blob.data = model_input
            except Exception as second:
                raise LegacyWorkerError(
                    f"无法写入 Caffe data blob：{exc}",
                    code="worker_protocol_error",
                ) from second
        forward = getattr(self.encoder, "forward", None)
        if not callable(forward):
            raise LegacyWorkerError("Caffe Net 缺少 forward()", code="worker_protocol_error")
        raw = forward()
        if self.feature_layer not in blobs:
            raise LegacyWorkerError(
                f"Caffe forward 未找到 feature_layer={self.feature_layer!r} activation",
                code="invalid_output",
            )
        return (
            getattr(blobs[self.feature_layer], "data", blobs[self.feature_layer]),
            f"caffe:{self.feature_layer}",
            raw,
        )

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        validate_clip_for_capabilities(batch, self.capabilities, train=train)
        model_input, input_metadata = _preprocess_c3d(
            batch,
            clip_frames=self.clip_frames,
            image_size=self.image_size,
            channel_order=self.channel_order,
            mean=self.mean,
            scale=self.scale,
        )
        raw_container: Any
        source: str | None = None
        try:
            if callable(getattr(self.encoder, "encode", None)):
                raw_container = self._call_encoder(model_input, train=train)
                raw = raw_container
            elif hasattr(self.encoder, "blobs") and callable(
                getattr(self.encoder, "forward", None)
            ):
                raw, source, raw_container = self._call_caffe_net(model_input)
            else:
                raw_container = self._call_encoder(model_input, train=train)
                raw = raw_container
            features, source = _normalize_c3d_feature(
                raw,
                batch_size=batch.batch_size,
                feature_layer=self.feature_layer,
                sequence_source=source,
            )
        except LegacyIntegrationError:
            raise
        except Exception as exc:
            raise LegacyWorkerError(
                f"C3D legacy forward 失败：{exc}",
                code="forward_failed",
            ) from exc

        timeline = _timeline_for_batch(batch, int(features.shape[1]))
        pooled = None
        for name in ("pooler_output", "pooled_output", "pooled"):
            pooled = (
                raw_container.get(name)
                if isinstance(raw_container, Mapping)
                else getattr(raw_container, name, None)
            )
            if pooled is not None:
                break
        output = normalize_encoder_output(
            features,
            timeline=timeline,
            feature_stage="fc_features",
            pooled=pooled,
            sequence_source=source,
            preprocess_profile="c3d-16x112-v1",
            aux={
                "adapter": "legacy",
                "integration_id": self.integration_id,
                "backend": self.backend,
                "implementation_source": self.implementation_source,
                "feature_layer": self.feature_layer,
                "frame_stride": self.frame_stride,
                "sampling_frame_stride": self.frame_stride,
                "checkpoint_loaded": self.checkpoint_path is not None,
                "feature_source": source,
                "checkout_path": None if self.checkout_path is None else str(self.checkout_path),
                **input_metadata,
            },
        )
        raw_type = type(raw_container)
        model_output_type = raw_type.__qualname__
        if raw_type.__module__ != "builtins":
            model_output_type = f"{raw_type.__module__}.{model_output_type}"
        output = EncoderOutput(
            features=output.features,
            pooled=output.pooled,
            timeline=output.timeline,
            aux={**output.aux, "model_output_type": model_output_type},
        )
        validate_encoder_output(output, batch)
        try:
            validate_output_health(output)
        except Exception as exc:
            raise LegacyWorkerError(
                f"C3D legacy 输出健康检查失败：{exc}",
                code="invalid_output_health",
            ) from exc
        return output


class C3DAdapter(LegacyVideoAdapter):
    """Compatibility alias for callers that name the model rather than runtime."""


DEFAULT_C3D_CAPABILITIES = DEFAULT_CAPABILITIES


__all__ = [
    "C3DAdapter",
    "DEFAULT_C3D_CAPABILITIES",
    "C3D_CAPABILITIES",
    "DEFAULT_CAPABILITIES",
    "ExternalAssetError",
    "LegacyAssetError",
    "LegacyDependencyError",
    "LegacyIntegrationError",
    "LegacyWorkerError",
    "MissingAssetError",
    "MissingLegacyAssetError",
    "StructuredLegacyError",
    "LegacyVideoAdapter",
]
