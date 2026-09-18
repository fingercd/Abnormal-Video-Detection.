"""Lazy Hugging Face Transformers adapters for fixed-frame video models.

The adapter deliberately keeps the Transformers dependency behind the
constructor.  Catalog/list commands can therefore import this module without
importing ``torch`` or ``transformers``.  Both TimeSformer and VideoMAE use the
same boundary: a :class:`~vadbench.contracts.ClipBatch` in ``BTHWC`` layout and
an :class:`~vadbench.contracts.EncoderOutput` with a provenance timeline.

Only local checkpoints are accepted.  ``from_pretrained`` is always called
with ``local_files_only=True`` so a typo or a missing asset fails loudly rather
than causing an implicit network download.
"""

from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Mapping
from contextlib import nullcontext
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
from vadbench.integrations.common import (
    normalize_encoder_output,
    normalize_feature_tensor,
    pool_feature_sequence,
    select_feature_tensor,
)

# Keep this identical to the generic fixed-clip capabilities in the catalog.
# Model-specific frame-count constraints are checked by the adapter itself;
# using a generic declaration lets one class serve checkpoints with different
# temporal configurations while preserving lazy registry validation.
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
    "timesformer": "timesformer",
    "time_sformer": "timesformer",
    "timesformer_base": "timesformer",
    "timesformer-base": "timesformer",
    "videomae": "videomae",
    "video_mae": "videomae",
    "videomae_base": "videomae",
    "videomae-base": "videomae",
}
_MODEL_CLASS_NAMES = {
    "timesformer": ("TimesformerModel", "TimeSformerModel"),
    "videomae": ("VideoMAEModel",),
}


def _normalize_variant(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("variant 必须是 timesformer 或 videomae")
    normalized = value.strip().lower().replace(" ", "_")
    normalized = _VARIANT_ALIASES.get(normalized, normalized)
    if normalized not in _MODEL_CLASS_NAMES:
        raise ValueError(
            f"不支持的 Transformers 视频模型 variant={value!r}；允许值为 timesformer、videomae"
        )
    return normalized


def _shape(value: Any, name: str = "tensor") -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raw = np.asarray(value).shape
    try:
        return tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}.shape 无法解析：{raw!r}") from exc


def _to_numpy(value: Any, name: str = "tensor") -> np.ndarray:
    try:
        module_root = type(value).__module__.split(".", 1)[0]
        if module_root == "torch" and hasattr(value, "detach"):
            tensor = value.detach().cpu()
            # NumPy cannot represent bfloat16 on some torch versions.
            if str(getattr(tensor, "dtype", "")).lower() in {
                "torch.bfloat16",
                "bfloat16",
            }:
                tensor = tensor.float()
            return tensor.numpy()
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - exotic tensor backends
        raise ValueError(f"{name} 无法转换为 numpy") from exc


def _field(value: Any, name: str) -> Any | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _coerce_sequence(value: Any, *, batch_size: int, name: str) -> Any:
    """Normalize a model activation to ``[B,S,D]`` without copying torch data."""

    shape = _shape(value, name)
    if len(shape) == 2:
        if shape[0] != batch_size or min(shape) <= 0:
            raise ValueError(f"{name} 必须是 [B,D]，实际 shape={shape}")
        if hasattr(value, "unsqueeze"):
            return value.unsqueeze(1)
        return np.expand_dims(np.asarray(value), axis=1)
    if len(shape) == 3:
        if shape[0] != batch_size or min(shape) <= 0:
            raise ValueError(f"{name} 必须是非空 [B,S,D]，实际 shape={shape}")
        return value
    # A few wrappers expose [B,T,H,W,D] or [B,T,N,D].  Flatten all token
    # axes while retaining batch and embedding dimensions.
    if len(shape) > 3 and shape[0] == batch_size and min(shape) > 0:
        if hasattr(value, "reshape"):
            return value.reshape(batch_size, -1, shape[-1])
        return np.asarray(value).reshape(batch_size, -1, shape[-1])
    raise ValueError(f"{name} 必须以 batch 维开头并包含 token/embedding 维，实际 shape={shape}")


def _extract_last_hidden(raw_output: Any, *, batch_size: int) -> Any | None:
    """Read the semantic last-hidden field from model output objects."""

    for name in ("last_hidden_state", "hidden_state", "features", "tokens"):
        candidate = _field(raw_output, name)
        if candidate is None:
            continue
        try:
            return _coerce_sequence(candidate, batch_size=batch_size, name=name)
        except ValueError:
            continue
    # A tuple output from ``return_dict=False`` is handled by the common
    # selector, but retain the same shape normalization here.
    try:
        selected, _ = select_feature_tensor(raw_output, batch_size=batch_size)
    except Exception:
        return None
    try:
        return _coerce_sequence(selected, batch_size=batch_size, name="model output")
    except ValueError:
        return None


def _has_named_pooler(raw_output: Any) -> bool:
    return any(
        _field(raw_output, name) is not None
        for name in ("pooler_output", "pooled_output", "pooled")
    )


def _has_named_sequence(raw_output: Any) -> bool:
    return any(
        _field(raw_output, name) is not None
        for name in ("last_hidden_state", "hidden_state", "features", "tokens")
    )


def _extract_pooler(raw_output: Any, *, batch_size: int, feature_dim: int) -> Any | None:
    for name in ("pooler_output", "pooled_output", "pooled"):
        candidate = _field(raw_output, name)
        if candidate is None:
            continue
        try:
            if _shape(candidate, name) == (batch_size, feature_dim):
                return candidate
        except ValueError:
            continue
    return None


def _normalizer_sequence_source(raw_output: Any, sequence: Any) -> Any:
    """Keep the original model-output type when its hidden field is usable."""

    candidate = _field(raw_output, "last_hidden_state")
    if candidate is not None:
        try:
            if _shape(candidate, "last_hidden_state") == _shape(sequence, "features"):
                return raw_output
        except ValueError:
            pass
    if isinstance(raw_output, (tuple, list)):
        return raw_output
    return {"last_hidden_state": sequence}


def _call_with_supported_kwargs(callable_obj: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a loader while preserving ``local_files_only``.

    Test doubles and older Transformers releases sometimes omit optional
    ``revision``/``trust_remote_code`` parameters.  We filter only those
    optional keys based on the signature; ``local_files_only`` is never
    removed, and an incompatible loader fails explicitly.
    """

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return callable_obj(*args, **kwargs)
    parameters = signature.parameters.values()
    accepts_var_kwargs = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters)
    if not accepts_var_kwargs:
        allowed = set(signature.parameters)
        if "local_files_only" not in allowed:
            raise TypeError(
                "本地 Transformers loader 必须接受 local_files_only 参数，拒绝不安全的隐式联网调用"
            )
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key == "local_files_only" or key in allowed
        }
    return callable_obj(*args, **kwargs)


def _local_checkpoint_path(
    model_path: str | os.PathLike[str] | None,
    model_name: str | os.PathLike[str] | None,
    checkpoint_path: str | os.PathLike[str] | None,
) -> Path | None:
    # ``model_path`` is the authoritative local asset.  ``model_name`` is
    # retained as a backwards-compatible alias, but a remote-looking name is
    # ignored when an explicit local path is present (and is never fetched).
    if model_path is not None:
        selected = model_path
        if (
            checkpoint_path is not None
            and Path(checkpoint_path).expanduser() != Path(model_path).expanduser()
        ):
            raise ValueError("model_path 与 checkpoint_path 必须指向同一个本地 checkpoint")
    elif checkpoint_path is not None:
        selected = checkpoint_path
    else:
        selected = model_name
    if selected is None:
        return None
    path = Path(selected).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Transformers 视频模型本地权重不存在：{path}；"
            "adapter 不会隐式联网下载，请先准备 checkpoint"
        )
    if not path.is_dir() and not path.is_file():
        raise ValueError(f"Transformers 视频模型路径必须是本地目录或权重文件：{path}")
    return path


def _transformers_module(injected: Any | None) -> Any:
    if injected is not None:
        return injected
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "Transformers 视频适配器需要可选依赖 transformers/torch；请在隔离环境安装对应版本"
        ) from exc


def _load_model(
    module: Any,
    variant: str,
    path: Path,
    *,
    revision: str | None,
    trust_remote_code: bool,
    loader: Any | None,
) -> Any:
    model_cls = None
    if loader is None:
        for class_name in _MODEL_CLASS_NAMES[variant]:
            model_cls = getattr(module, class_name, None)
            if model_cls is not None:
                break
        if model_cls is None:
            names = ", ".join(_MODEL_CLASS_NAMES[variant])
            raise ImportError(f"当前 transformers 缺少 {names}")
        loader = getattr(model_cls, "from_pretrained", None)
    else:
        # A supplied loader may be a classmethod, a callable class, or a
        # test double.  It is invoked directly below.
        model_cls = getattr(loader, "__self__", None)
    if not callable(loader):
        raise TypeError(f"{variant} 模型没有可调用的 from_pretrained")
    kwargs: dict[str, Any] = {"local_files_only": True}
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return _call_with_supported_kwargs(loader, str(path), **kwargs)


def _load_processor(
    module: Any,
    path: Path,
    *,
    revision: str | None,
    trust_remote_code: bool,
    loader: Any | None,
) -> Any:
    if loader is None:
        loader = getattr(module, "AutoImageProcessor", None)
        if loader is None:
            loader = getattr(module, "AutoProcessor", None)
        if loader is None:
            # Older Transformers releases did not expose AutoImageProcessor
            # for the video configs but did export the concrete processor.
            loader = getattr(module, "VideoMAEImageProcessor", None)
        if loader is None:
            raise ImportError("当前 transformers 缺少 AutoImageProcessor/AutoProcessor")
        loader = getattr(loader, "from_pretrained", None)
    if not callable(loader):
        raise TypeError("processor 没有可调用的 from_pretrained")
    kwargs: dict[str, Any] = {"local_files_only": True}
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return _call_with_supported_kwargs(loader, str(path), **kwargs)


def _frame_lists(batch: ClipBatch) -> tuple[list[list[np.ndarray]], np.ndarray]:
    frames = _to_numpy(batch.frames, "frames")
    lengths = np.asarray(batch.valid_lengths, dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("ClipBatch 至少需要一个有效帧")
    videos = [
        [np.asarray(frame, dtype=np.uint8) for frame in frames[index, : int(length)]]
        for index, length in enumerate(lengths)
    ]
    # Image processors generally require a rectangular temporal batch.  Pad
    # only for preprocessing; timeline construction below still uses each
    # sample's original valid length.
    max_length = int(lengths.max())
    for video in videos:
        if len(video) < max_length:
            video.extend([video[-1].copy() for _ in range(max_length - len(video))])
    return videos, lengths


def _invoke_processor(
    processor: Any, videos: list[list[np.ndarray]], kwargs: Mapping[str, Any]
) -> Any:
    call = processor
    # Positional invocation is accepted by both VideoMAEImageProcessor and
    # AutoImageProcessor wrappers.  Keyword fallbacks support strict fakes and
    # a few third-party processors that expose ``videos=`` only.
    call_kwargs = dict(kwargs)
    try:
        return call(videos, **call_kwargs)
    except TypeError as first_error:
        # Filter optional preprocessing knobs for strict processors whose
        # signature only accepts ``videos``/``images`` and ``return_tensors``.
        try:
            signature = inspect.signature(call)
            params = signature.parameters.values()
            if not any(item.kind is inspect.Parameter.VAR_KEYWORD for item in params):
                allowed = set(signature.parameters)
                filtered = {key: value for key, value in call_kwargs.items() if key in allowed}
                if filtered != call_kwargs:
                    return call(videos, **filtered)
        except (TypeError, ValueError):
            pass
        for key in ("videos", "images"):
            try:
                return call(**{key: videos, **call_kwargs})
            except TypeError:
                try:
                    signature = inspect.signature(call)
                    params = signature.parameters.values()
                    if not any(item.kind is inspect.Parameter.VAR_KEYWORD for item in params):
                        allowed = set(signature.parameters)
                        filtered = {
                            name: value
                            for name, value in {key: videos, **call_kwargs}.items()
                            if name in allowed
                        }
                        return call(**filtered)
                except (TypeError, ValueError):
                    pass
                continue
        raise first_error


def _move_to_device(value: Any, device: str | None) -> Any:
    if device is None:
        return value
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    # Do not call ``.to`` on numpy arrays or arbitrary user objects.  A real
    # torch tensor and BatchFeature both expose a compatible method.
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "to"):
        return value.to(device)
    return value


def _model_device(model: Any) -> str | None:
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            parameter = next(parameters())
            device = getattr(parameter, "device", None)
            if device is not None:
                return str(device)
        except (StopIteration, TypeError):
            pass
    return None


def _timeline_for_tokens(batch: ClipBatch, token_count: int) -> TokenTimeline:
    """Map token positions to source frames with an explicit approximation."""

    if token_count <= 0:
        raise ValueError("token_count 必须大于 0")
    timestamps = _to_numpy(batch.timestamps_s, "timestamps_s").astype(np.float64, copy=False)
    indices = (
        np.broadcast_to(
            np.arange(batch.num_frames, dtype=np.int64),
            (batch.batch_size, batch.num_frames),
        )
        if batch.frame_indices is None
        else _to_numpy(batch.frame_indices, "frame_indices").astype(np.int64, copy=False)
    )
    lengths = np.asarray(batch.valid_lengths, dtype=np.int64)
    starts = np.empty((batch.batch_size, token_count), dtype=np.float64)
    ends = np.empty_like(starts)
    source_start = np.empty((batch.batch_size, token_count), dtype=np.int64)
    source_end = np.empty_like(source_start)
    for row, length in enumerate(lengths):
        count = int(length)
        times = timestamps[row, :count]
        frame_ids = indices[row, :count]
        slots = np.minimum(
            (np.arange(token_count, dtype=np.int64) * count) // token_count,
            count - 1,
        )
        starts[row] = times[slots]
        if count > 1:
            deltas = np.diff(times)
            positive = deltas[deltas > 0]
            fallback = float(np.median(positive)) if positive.size else 0.0
            frame_ends = np.empty(count, dtype=np.float64)
            frame_ends[:-1] = times[1:]
            frame_ends[-1] = times[-1] + fallback
        else:
            frame_ends = np.asarray([times[0]], dtype=np.float64)
        ends[row] = frame_ends[slots]
        source_start[row] = frame_ids[slots]
        # Non-contiguous source indices are legal (sampling stride); a
        # one-frame half-open span remains the least surprising provenance.
        source_end[row] = source_start[row] + 1
    return TokenTimeline(
        start_s=starts,
        end_s=ends,
        source_frame_start=source_start,
        source_frame_end=source_end,
    )


def _torch_context(train: bool) -> Any:
    if train:
        return nullcontext()
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return nullcontext()
    no_grad = getattr(torch, "no_grad", None)
    return no_grad() if callable(no_grad) else nullcontext()


class TransformersVideoAdapter(VideoEncoderAdapter):
    """Shared fixed-clip adapter for TimeSformer and VideoMAEModel."""

    capabilities = DEFAULT_CAPABILITIES

    def __init__(
        self,
        *,
        variant: str = "timesformer",
        model_name: str | os.PathLike[str] | None = None,
        model_path: str | os.PathLike[str] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        device: str | None = None,
        processor: Any | None = None,
        model: Any | None = None,
        image_size: int | None = None,
        clip_frames: int | None = None,
        num_frames: int | None = None,
        frame_stride: int = 1,
        feature_stage: str = "last_hidden_state",
        pooling: str = "auto",
        preprocess_profile: str = "transformers-video-v1",
        revision: str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        processor_kwargs: Mapping[str, Any] | None = None,
        processor_tensor_type: str = "pt",
        transformers_module: Any | None = None,
        model_loader: Any | None = None,
        processor_loader: Any | None = None,
    ) -> None:
        self.variant = _normalize_variant(variant)
        if type(local_files_only) is not bool or not local_files_only:
            raise ValueError("Transformers 视频 adapter 只允许 local_files_only=True，禁止隐式联网")
        if clip_frames is not None and num_frames is not None and clip_frames != num_frames:
            raise ValueError("clip_frames 与 num_frames 必须一致")
        selected_frames = clip_frames if clip_frames is not None else num_frames
        if selected_frames is not None and (
            type(selected_frames) is not int or selected_frames <= 0
        ):
            raise ValueError("clip_frames/num_frames 必须是正整数")
        if type(frame_stride) is not int or frame_stride <= 0:
            raise ValueError("frame_stride 必须是正整数")
        normalized_stage = str(feature_stage).strip().lower().replace("-", "_")
        if normalized_stage not in {"last_hidden_state", "pooled"}:
            raise ValueError("feature_stage 必须是 last_hidden_state 或 pooled")
        normalized_pooling = str(pooling).strip().lower()
        if normalized_pooling not in {"mean", "auto", "pooler"}:
            raise ValueError("pooling 必须是 mean、auto 或 pooler")
        if not isinstance(preprocess_profile, str) or not preprocess_profile.strip():
            raise ValueError("preprocess_profile 必须是非空字符串")
        if processor_tensor_type not in ("pt", "np"):
            raise ValueError("processor_tensor_type 必须是 pt 或 np")

        # Fully injected test/worker objects do not need an asset path; in
        # that case ignore a merely informational (possibly remote-looking)
        # model_name instead of attempting any filesystem or network access.
        path = (
            None
            if model is not None and processor is not None
            else _local_checkpoint_path(model_path, model_name, checkpoint_path)
        )
        self.model_path = path
        self.revision = revision
        self.image_size = image_size
        self.frame_stride = frame_stride
        self.feature_stage = normalized_stage
        self.pooling = normalized_pooling
        self.preprocess_profile = preprocess_profile.strip()
        self.processor_kwargs = dict(processor_kwargs or {})
        self.processor_kwargs.pop("return_tensors", None)
        self.processor_tensor_type = processor_tensor_type
        self.transformers_module = transformers_module

        # If a model/processor is injected, no optional package import or file
        # lookup is needed.  This is useful for contract tests and external
        # workers.  For a real path, load both pieces with local-only flags.
        module = None
        needs_model_module = model is None and model_loader is None
        needs_processor_module = processor is None and processor_loader is None
        if needs_model_module or needs_processor_module:
            if path is None:
                raise ValueError(
                    "必须提供本地 model_path/checkpoint_path（或同时注入 model 与 processor）；"
                    "不会从远程 model_name 自动下载"
                )
            module = _transformers_module(transformers_module)
        if model is None:
            assert path is not None
            model = _load_model(
                module,
                self.variant,
                path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                loader=model_loader,
            )
        if processor is None:
            assert path is not None
            processor = _load_processor(
                module,
                path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                loader=processor_loader,
            )

        self.model = model
        self.processor = processor
        # Infer the checkpoint's temporal contract when config supplied no
        # explicit clip_frames.  Fake models without config stay flexible.
        if selected_frames is None:
            config = getattr(model, "config", None)
            inferred = getattr(config, "num_frames", None)
            if type(inferred) is int and inferred > 0:
                selected_frames = inferred
        self.clip_frames = selected_frames
        self.device = device
        if self.device is None:
            self.device = _model_device(model)
        self._move_model_to_device()

    def _move_model_to_device(self) -> None:
        evaluator = getattr(self.model, "eval", None)
        if callable(evaluator):
            evaluator()
        if self.device is None:
            return
        mover = getattr(self.model, "to", None)
        if callable(mover):
            try:
                mover(self.device)
            except TypeError:
                # Some test doubles expose ``to(device=...)`` only.
                mover(device=self.device)

    def _prepare_inputs(self, batch: ClipBatch) -> tuple[Mapping[str, Any], np.ndarray]:
        videos, lengths = _frame_lists(batch)
        kwargs = dict(self.processor_kwargs)
        kwargs["return_tensors"] = self.processor_tensor_type
        if self.image_size is not None and "size" not in kwargs:
            # Processors differ in whether ``size`` accepts an integer or a
            # dict.  Let an explicit processor config win; only use a square
            # dict when caller supplied image_size.
            kwargs["size"] = {"height": self.image_size, "width": self.image_size}
        processed = _invoke_processor(self.processor, videos, kwargs)
        if isinstance(processed, Mapping):
            inputs = dict(processed)
        elif hasattr(processed, "pixel_values"):
            inputs = {"pixel_values": processed.pixel_values}
        else:
            inputs = {"pixel_values": processed}
        if self.processor_tensor_type == "np":
            # Classic video processors otherwise construct torch tensors from
            # lists of frame arrays. NumPy batches the same processed pixels
            # first; from_numpy preserves their dtype/layout without a copy.
            torch = importlib.import_module("torch")
            inputs = {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in inputs.items()
            }
        return _move_to_device(inputs, self.device), lengths

    def _forward(self, inputs: Mapping[str, Any], *, train: bool) -> Any:
        kwargs = dict(inputs)
        # Both supported base models accept return_dict; inspect fakes and
        # older wrappers before adding it so a strict signature still works.
        target = getattr(self.model, "forward", self.model)
        try:
            signature = inspect.signature(target)
            params = signature.parameters.values()
            if "return_dict" in signature.parameters or any(
                item.kind is inspect.Parameter.VAR_KEYWORD for item in params
            ):
                kwargs["return_dict"] = True
        except (TypeError, ValueError):
            kwargs["return_dict"] = True
        with _torch_context(train):
            try:
                return self.model(**kwargs)
            except TypeError as first_error:
                # A small compatibility fallback for strict test doubles or
                # old model wrappers that reject return_dict.
                if "return_dict" in kwargs:
                    kwargs.pop("return_dict", None)
                    try:
                        return self.model(**kwargs)
                    except TypeError:
                        pass
                raise first_error

    def _set_model_mode(self, train: bool) -> None:
        if train:
            trainer = getattr(self.model, "train", None)
            if callable(trainer):
                try:
                    trainer(True)
                except TypeError:
                    trainer()
        else:
            evaluator = getattr(self.model, "eval", None)
            if callable(evaluator):
                evaluator()

    def _output_from_raw(
        self,
        raw_output: Any,
        batch: ClipBatch,
        *,
        lengths: np.ndarray,
    ) -> EncoderOutput:
        # Do not mistake a named pooled-only output for a token sequence.  A
        # tuple/raw tensor remains eligible for the generic common selector.
        sequence = (
            _extract_last_hidden(raw_output, batch_size=batch.batch_size)
            if _has_named_sequence(raw_output) or not _has_named_pooler(raw_output)
            else None
        )
        pooler = None
        actual_stage = self.feature_stage
        sequence_source = "last_hidden_state"
        if sequence is not None:
            sequence = normalize_feature_tensor(sequence, batch_size=batch.batch_size)
            feature_dim = _shape(sequence, "last_hidden_state")[-1]
            pooler = _extract_pooler(
                raw_output,
                batch_size=batch.batch_size,
                feature_dim=feature_dim,
            )
            if self.pooling == "pooler" and pooler is None:
                raise ValueError("pooling=pooler 但模型输出没有 pooler_output")
            if pooler is None or self.pooling == "mean":
                pooler = pool_feature_sequence(sequence)
            if self.feature_stage == "pooled":
                raw_for_normalizer: Any = pooler
                sequence_source = (
                    "pooler_output"
                    if _field(raw_output, "pooler_output") is not None
                    else "mean_pool"
                )
            else:
                raw_for_normalizer = _normalizer_sequence_source(raw_output, sequence)
        else:
            # A few classifier wrappers expose only a pooled representation.
            # Keep the output usable, but record the actual stage in aux rather
            # than pretending a missing token sequence exists.
            selected, selected_source = select_feature_tensor(
                raw_output, batch_size=batch.batch_size
            )
            pooled = _coerce_sequence(selected, batch_size=batch.batch_size, name="pooled output")
            if _shape(pooled, "pooled output")[1] != 1:
                pooled = pool_feature_sequence(pooled)
            else:
                pooled = pooled[:, 0]
            raw_for_normalizer = pooled
            actual_stage = "pooled"
            sequence_source = selected_source or "pooled_singleton"

        if sequence is not None and self.feature_stage != "pooled":
            token_count = _shape(sequence, "features")[1]
        else:
            token_count = 1
        timeline = _timeline_for_tokens(batch, token_count)
        output = normalize_encoder_output(
            raw_for_normalizer,
            timeline=timeline,
            feature_stage=actual_stage,
            pooled=pooler if sequence is not None else pooled,
            sequence_source=sequence_source,
            preprocess_profile=self.preprocess_profile,
            aux={
                "adapter": "transformers_video",
                "variant": self.variant,
                "input_layout": "BTHWC",
                "requested_feature_stage": self.feature_stage,
                "frame_count": [int(item) for item in lengths],
                "frame_stride": self.frame_stride,
                "timeline_policy": "uniform_token_to_frame_approximation",
                "native_route_available": True,
                "implementation_source": "native_upstream",
                "native_model_path": None if self.model_path is None else str(self.model_path),
                "native_revision": self.revision,
            },
        )
        validate_encoder_output(output, batch)
        return output

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        validate_clip_for_capabilities(batch, self.capabilities, train=train)
        lengths = np.asarray(batch.valid_lengths, dtype=np.int64)
        if self.clip_frames is not None and np.any(lengths != self.clip_frames):
            raise ValueError(
                f"{self.variant} checkpoint 要求 clip_frames={self.clip_frames}，"
                f"实际有效帧数={lengths.tolist()}"
            )
        self._set_model_mode(train)
        inputs, lengths = self._prepare_inputs(batch)
        raw_output = self._forward(inputs, train=train)
        return self._output_from_raw(raw_output, batch, lengths=lengths)


# Explicit aliases make configs and downstream type annotations readable while
# retaining one implementation and one lazy import target.
TimesformerAdapter = TransformersVideoAdapter
TimeSformerAdapter = TransformersVideoAdapter
VideoMAEAdapter = TransformersVideoAdapter
VideoMAEModelAdapter = TransformersVideoAdapter

TIMESFORMER_CAPABILITIES = DEFAULT_CAPABILITIES
VIDEOMAE_CAPABILITIES = DEFAULT_CAPABILITIES


__all__ = [
    "DEFAULT_CAPABILITIES",
    "TIMESFORMER_CAPABILITIES",
    "VIDEOMAE_CAPABILITIES",
    "TimeSformerAdapter",
    "TimesformerAdapter",
    "TransformersVideoAdapter",
    "VideoMAEAdapter",
    "VideoMAEModelAdapter",
]
