"""Shared bridge and output normalization for video foundation adapters.

The four foundation-model integrations intentionally keep their upstream
dependencies out of the import path.  A catalog/list operation therefore only
loads this small module; a model library is imported only when an adapter is
constructed with an explicit local asset and loader.  ``encoder=`` remains the
supported dependency-injection seam for tests and for future upstream
implementations.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.contracts import (
    ClipBatch,
    EncoderCapabilities,
    EncoderOutput,
    VideoEncoderAdapter,
    validate_clip_for_capabilities,
    validate_encoder_output,
)
from vadbench.integrations.common import (
    normalize_encoder_output,
    select_feature_tensor,
)

FOUNDATION_CAPABILITIES = EncoderCapabilities(
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


class FoundationIntegrationError(RuntimeError):
    """Base error for explicit, fail-closed foundation integration failures."""


class FoundationAssetError(FoundationIntegrationError, FileNotFoundError):
    """Raised when a foundation model's explicitly requested local asset is absent."""


class FoundationUpstreamError(FoundationIntegrationError, ImportError):
    """Raised when an explicit upstream loader/entrypoint cannot be resolved."""


def _to_numpy(value: Any) -> np.ndarray:
    """Copy only small timeline metadata to CPU when the upstream uses torch."""

    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if str(getattr(tensor, "dtype", "")).lower() in {"bfloat16", "torch.bfloat16"}:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _is_path_like_reference(value: str) -> bool:
    """Return whether ``value`` should be interpreted as a local path.

    Model identifiers such as ``OpenGVLab/UMT-B`` are deliberately not fetched
    or handed to a hub client.  They are retained as provenance only unless a
    separate, existing local ``model_path`` is supplied.
    """

    return value.startswith((".", "/", "~")) or "\\" in value or "/" in value


def _resolve_local_asset(
    *, backend: str, model_path: str | Path | None, model_name: str | None
) -> Path:
    selected: str | Path | None = model_path
    if selected is None and isinstance(model_name, str) and _is_path_like_reference(model_name):
        selected = model_name
    if selected is None:
        hint = f" model_name={model_name!r}" if model_name else ""
        raise FoundationAssetError(
            f"{backend} 缺少本地 model_path；{hint}。"
            "foundation adapter 禁止隐式联网，请先准备权重并显式传入 model_path"
        )
    if not isinstance(selected, (str, Path)):
        raise FoundationAssetError(
            f"{backend} model_path 必须是本地路径字符串或 Path，实际为 {type(selected).__name__}"
        )
    text = str(selected)
    if text.startswith(("http://", "https://", "hf://", "s3://")):
        raise FoundationAssetError(
            f"{backend} 的 model_path 不能是网络 URI：{text!r}；请使用已下载的本地资产"
        )
    path = Path(selected).expanduser().resolve()
    if not path.exists():
        raise FoundationAssetError(f"{backend} 本地模型/权重不存在：{path}；不会自动联网下载")
    if path.is_dir():
        try:
            has_asset = any(path.iterdir())
        except OSError as exc:
            raise FoundationAssetError(f"{backend} 无法读取本地模型目录：{path}") from exc
        if not has_asset:
            raise FoundationAssetError(f"{backend} 本地模型目录为空，未发现权重：{path}")
    elif path.stat().st_size <= 0:
        raise FoundationAssetError(f"{backend} 本地权重文件为空：{path}")
    return path


def _lookup_attribute(value: Any, attribute_path: str) -> Any:
    for component in attribute_path.split("."):
        if not component.isidentifier():
            raise FoundationUpstreamError(f"entrypoint 属性路径非法：{attribute_path!r}")
        try:
            value = getattr(value, component)
        except AttributeError as exc:
            raise FoundationUpstreamError(
                f"entrypoint 缺少属性 {component!r}：{attribute_path!r}"
            ) from exc
    return value


def _load_entrypoint(spec: str, *, checkout_path: Path | None, backend: str) -> Any:
    """Resolve an explicit ``module:attribute`` or ``file.py:attribute`` target."""

    module_name, separator, attribute_path = spec.partition(":")
    if not separator or not module_name or not attribute_path:
        raise FoundationUpstreamError(
            f"{backend} upstream_entrypoint 必须是 module:attribute：{spec!r}"
        )

    module_path = Path(module_name)
    if module_name.endswith(".py") or "/" in module_name or "\\" in module_name:
        if not module_path.is_absolute():
            if checkout_path is None:
                raise FoundationUpstreamError(
                    f"{backend} 的文件 entrypoint 需要显式 checkout_path：{spec!r}"
                )
            module_path = checkout_path / module_path
        module_path = module_path.expanduser().resolve()
        if not module_path.is_file():
            raise FoundationUpstreamError(
                f"{backend} upstream entrypoint 文件不存在：{module_path}"
            )
        safe_backend = "".join(char if char.isalnum() else "_" for char in backend)
        unique_name = f"vadbench_foundation_{safe_backend}_upstream"
        module_spec = importlib.util.spec_from_file_location(unique_name, module_path)
        if module_spec is None or module_spec.loader is None:
            raise FoundationUpstreamError(f"无法创建 {backend} entrypoint loader：{module_path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules.setdefault(unique_name, module)
        try:
            module_spec.loader.exec_module(module)
        except Exception as exc:
            raise FoundationUpstreamError(
                f"加载 {backend} upstream entrypoint 失败：{module_path}"
            ) from exc
        target = module
    else:
        if checkout_path is not None:
            checkout_text = str(checkout_path.expanduser().resolve())
            if checkout_text not in sys.path:
                sys.path.insert(0, checkout_text)
        try:
            target = importlib.import_module(module_name)
        except Exception as exc:
            raise FoundationUpstreamError(
                f"导入 {backend} upstream module 失败：{module_name!r}"
            ) from exc
    target = _lookup_attribute(target, attribute_path)
    if not callable(target):
        raise FoundationUpstreamError(f"{backend} entrypoint 不是 callable：{spec!r}")
    return target


def _call_loader(
    loader: Callable[..., Any],
    *,
    model_path: Path,
    model_name: str | None,
    device: str | None,
    model_kwargs: Mapping[str, Any],
    backend: str,
) -> Any:
    """Call a loader with only parameters it explicitly accepts.

    The helper accepts both modern keyword loaders and legacy one-path factory
    functions.  It never catches a loader's runtime exception and retries with
    a different signature, which keeps upstream failures observable.
    """

    try:
        parameters = inspect.signature(loader).parameters
    except (TypeError, ValueError) as exc:
        raise FoundationUpstreamError(
            f"{backend} upstream loader 必须提供可检查的显式签名"
        ) from exc
    values: dict[str, Any] = {
        "model_path": str(model_path),
        "checkpoint_path": str(model_path),
        "model_name": model_name,
        "device": device,
        **dict(model_kwargs),
    }
    accepts_var_kw = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    keyword_values = {
        key: value
        for key, value in values.items()
        if value is not None and (accepts_var_kw or key in parameters)
    }
    positional: list[Any] = []
    for parameter in parameters.values():
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            if parameter.name in {"path", "checkpoint", "checkpoint_path", "model_path"}:
                positional.append(str(model_path))
            elif parameter.name in {"name", "model_name"}:
                positional.append(model_name)
            elif parameter.default is inspect.Parameter.empty and parameter.name in values:
                positional.append(values[parameter.name])
            elif parameter.default is inspect.Parameter.empty and not positional:
                positional.append(str(model_path))
            else:
                break
        else:
            break
    # Positional-or-keyword values already supplied positionally must not be
    # repeated as keywords.
    for parameter in list(parameters.values())[: len(positional)]:
        keyword_values.pop(parameter.name, None)
    try:
        return loader(*positional, **keyword_values)
    except Exception as exc:
        if isinstance(exc, FoundationUpstreamError):
            raise
        raise FoundationUpstreamError(f"{backend} upstream loader 执行失败：{exc}") from exc


def _invoke_encoder(encoder: Any, batch: ClipBatch) -> Any:
    """Invoke the internal worker's explicit ``ClipBatch`` contract."""

    method = getattr(encoder, "encode", None)
    if callable(method):
        return method(batch)
    if callable(encoder):
        return encoder(batch)
    raise TypeError(
        f"upstream {type(encoder).__name__} 必须实现 encode(ClipBatch) 或 callable(ClipBatch)"
    )


def _uniform_timeline(batch: ClipBatch, token_count: int):
    """Map output tokens to monotonically increasing source-frame intervals."""

    timestamps = _to_numpy(batch.timestamps_s)
    frame_indices = None if batch.frame_indices is None else _to_numpy(batch.frame_indices)
    starts = np.empty((batch.batch_size, token_count), dtype=np.float64)
    ends = np.empty_like(starts)
    source_start = np.empty((batch.batch_size, token_count), dtype=np.int64)
    source_end = np.empty_like(source_start)

    for row in range(batch.batch_size):
        valid_frames = int(batch.valid_lengths[row])
        times = timestamps[row, :valid_frames].astype(np.float64, copy=False)
        if valid_frames == 1:
            delta = 1.0 / 30.0
            frame_ends = np.asarray([times[0] + delta], dtype=np.float64)
        else:
            differences = np.diff(times)
            positive = differences[np.isfinite(differences) & (differences > 0)]
            delta = float(np.median(positive)) if positive.size else 1.0 / 30.0
            frame_ends = np.empty(valid_frames, dtype=np.float64)
            frame_ends[:-1] = times[1:]
            frame_ends[-1] = times[-1] + delta
        slots = np.minimum(
            (np.arange(token_count, dtype=np.int64) * valid_frames) // token_count,
            valid_frames - 1,
        )
        starts[row] = times[slots]
        ends[row] = frame_ends[slots]
        if frame_indices is None:
            source = np.arange(valid_frames, dtype=np.int64)
        else:
            source = frame_indices[row, :valid_frames].astype(np.int64, copy=False)
        source_start[row] = source[slots]
        source_end[row] = source[slots] + 1

    from vadbench.contracts import TokenTimeline

    return TokenTimeline(
        start_s=starts,
        end_s=ends,
        source_frame_start=source_start,
        source_frame_end=source_end,
    )


class LazyFoundationBridge:
    """Lazy local-asset bridge shared by in-process and external integrations.

    ``runtime`` is metadata for the eventual worker orchestration.  In-process
    tests and future adapters use the same explicit loader seam; no subprocess,
    hub client, or network operation is started implicitly here.
    """

    def __init__(
        self,
        *,
        backend: str,
        model_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        local_path: str | Path | None = None,
        model_name: str | None = None,
        device: str | None = None,
        runtime: str = "in_process",
        checkout_path: str | Path | None = None,
        upstream_entrypoint: str | None = None,
        entrypoint: str | None = None,
        loader: Callable[..., Any] | None = None,
        factory: Callable[..., Any] | None = None,
        encoder: Any | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if runtime not in {"in_process", "external_python"}:
            raise ValueError(f"runtime 必须是 in_process 或 external_python，实际为 {runtime!r}")
        if loader is not None and factory is not None:
            raise ValueError("loader 与 factory 只能显式提供一个")
        if upstream_entrypoint is not None and entrypoint is not None:
            raise ValueError("upstream_entrypoint 与 entrypoint 只能显式提供一个")
        self.backend = str(backend)
        self.model_name = model_name
        self.device = device
        self.runtime = runtime
        self.checkout_path = (
            None if checkout_path is None else Path(checkout_path).expanduser().resolve()
        )
        self.upstream_entrypoint = upstream_entrypoint or entrypoint
        self.loader = loader or factory
        self.model_kwargs = dict(model_kwargs or {})
        self.model_path: Path | None = None
        self._encoder = encoder
        selected_model_path = model_path
        aliases = [
            value for value in (checkpoint_path, weights_path, local_path) if value is not None
        ]
        if selected_model_path is not None and aliases:
            if any(
                Path(value).expanduser().resolve()
                != Path(selected_model_path).expanduser().resolve()
                for value in aliases
            ):
                raise ValueError(
                    "model_path、checkpoint_path、weights_path 和 local_path 只能指向同一资产"
                )
        elif selected_model_path is None and aliases:
            selected_model_path = aliases[0]

        if encoder is None:
            self.model_path = _resolve_local_asset(
                backend=self.backend,
                model_path=selected_model_path,
                model_name=model_name,
            )
        elif selected_model_path is not None:
            # Keep provenance but do not reject a test double merely because its
            # eventual checkpoint has not been materialized yet.
            self.model_path = Path(selected_model_path).expanduser().resolve()

    @property
    def loaded(self) -> bool:
        return self._encoder is not None

    @property
    def encoder(self) -> Any:
        return self._encoder if self._encoder is not None else self.load()

    def load(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        target = self.loader
        if target is None and self.upstream_entrypoint is not None:
            target = _load_entrypoint(
                self.upstream_entrypoint,
                checkout_path=self.checkout_path,
                backend=self.backend,
            )
        if target is None:
            raise FoundationUpstreamError(
                f"{self.backend} 已找到本地资产 {self.model_path}，但未配置显式 upstream loader；"
                "请传入 loader/factory 或 upstream_entrypoint。不会猜测接口或联网"
            )
        self._encoder = _call_loader(
            target,
            model_path=self.model_path,
            model_name=self.model_name,
            device=self.device,
            model_kwargs=self.model_kwargs,
            backend=self.backend,
        )
        if self._encoder is None:
            raise FoundationUpstreamError(f"{self.backend} upstream loader 返回了 None")
        return self._encoder

    def encode(self, batch: ClipBatch) -> Any:
        return _invoke_encoder(self.encoder, batch)


class InProcessFoundationBridge(LazyFoundationBridge):
    """Named bridge for adapters whose upstream runs in the current process."""


class ExternalPythonFoundationBridge(LazyFoundationBridge):
    """Named bridge for adapters reserved for an isolated Python worker."""


class FoundationVideoAdapter(VideoEncoderAdapter):
    """Base fixed-clip adapter for UMT/InternVideo2/VideoMamba/V-JEPA2."""

    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "foundation"
    FEATURE_STAGE = "backbone_tokens"
    PREPROCESS_PROFILE = "foundation-bthwc-v1"
    DEFAULT_MODEL_NAME: str | None = None
    DEFAULT_RUNTIME = "external_python"

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        local_path: str | Path | None = None,
        model_name: str | None = None,
        device: str | None = None,
        encoder: Any | None = None,
        model: Any | None = None,
        runtime: str | None = None,
        checkout_path: str | Path | None = None,
        upstream_entrypoint: str | None = None,
        entrypoint: str | None = None,
        loader: Callable[..., Any] | None = None,
        factory: Callable[..., Any] | None = None,
        feature_stage: str | None = None,
        preprocess_profile: str | None = None,
        num_frames: int | None = None,
        image_size: int | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if encoder is not None and model is not None and encoder is not model:
            raise ValueError("encoder 与 model 不能同时指向不同对象")
        injected = encoder if encoder is not None else model
        if num_frames is not None and (isinstance(num_frames, bool) or num_frames <= 0):
            raise ValueError("num_frames 必须是正整数或 None")
        if image_size is not None and (isinstance(image_size, bool) or image_size <= 0):
            raise ValueError("image_size 必须是正整数或 None")
        from vadbench.integrations.common import normalize_feature_stage

        self.backend = self.BACKEND
        # An injected test/model object is not evidence that the native upstream
        # loader ran. Keep that distinction in smoke provenance.
        self._native_upstream = injected is None and (
            loader is not None
            or factory is not None
            or upstream_entrypoint is not None
            or entrypoint is not None
        )
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = device
        self.runtime = runtime or self.DEFAULT_RUNTIME
        self.feature_stage = normalize_feature_stage(feature_stage or self.FEATURE_STAGE)
        self.preprocess_profile = preprocess_profile or self.PREPROCESS_PROFILE
        self.num_frames = num_frames
        self.image_size = image_size
        merged_kwargs = dict(model_kwargs or {})
        merged_kwargs.update(kwargs)
        bridge_class = (
            ExternalPythonFoundationBridge
            if self.runtime == "external_python"
            else InProcessFoundationBridge
        )
        self.bridge = bridge_class(
            backend=self.backend,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            weights_path=weights_path,
            local_path=local_path,
            model_name=self.model_name,
            device=device,
            runtime=self.runtime,
            checkout_path=checkout_path,
            upstream_entrypoint=upstream_entrypoint,
            entrypoint=entrypoint,
            loader=loader,
            factory=factory,
            encoder=injected,
            model_kwargs=merged_kwargs,
        )
        # Keep a stable public reference for injected fakes.  For lazy models it
        # remains None until the first encode/load call.
        self.encoder = injected
        self.model_path = self.bridge.model_path

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        validate_clip_for_capabilities(batch, self.capabilities, train=train)
        raw_output = self.bridge.encode(batch)
        if self.encoder is None:
            self.encoder = self.bridge.encoder
        selected, observed_source = select_feature_tensor(raw_output, batch_size=batch.batch_size)
        token_count = 1 if selected.ndim == 2 else int(selected.shape[1])
        sequence_source = "pooled_singleton" if selected.ndim == 2 else observed_source
        timeline = _uniform_timeline(batch, token_count)
        output = normalize_encoder_output(
            raw_output,
            timeline=timeline,
            feature_stage=self.feature_stage,
            sequence_source=sequence_source,
            preprocess_profile=self.preprocess_profile,
            aux={
                "backend": self.backend,
                "implementation_source": "native_upstream"
                if self._native_upstream
                else "lazy_upstream_bridge",
                "native_route_available": bool(self._native_upstream),
                "runtime": self.runtime,
                "model_name": self.model_name,
                "model_path": None if self.model_path is None else str(self.model_path),
                "input_layout": "BTHWC",
            },
        )
        validate_encoder_output(output, batch)
        return output


__all__ = [
    "ExternalPythonFoundationBridge",
    "FOUNDATION_CAPABILITIES",
    "FoundationAssetError",
    "FoundationIntegrationError",
    "FoundationUpstreamError",
    "FoundationVideoAdapter",
    "InProcessFoundationBridge",
    "LazyFoundationBridge",
]
