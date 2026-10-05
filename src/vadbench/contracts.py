"""Core contracts shared by VAD encoder integrations.

The framework deliberately has one decoded-video boundary: ``uint8`` frames in
``BTHWC`` layout.  Model adapters own all layout conversion and normalization.
This keeps datasets, feature extraction and encoder benchmarks independent of a
particular model family.

The cache types in this module describe *observable, reusable* cache state.  An
adapter must not advertise a cache capability merely because its implementation
contains attention layers; it must expose that cache through :class:`StreamState`
and accept cache policies in ``encode_step``.
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

ArrayLike = Any


class ContractError(ValueError):
    """Raised when data does not satisfy a framework contract."""


class CapabilityError(ContractError):
    """Raised when an adapter's behavior contradicts declared capabilities."""


def _shape(value: Any, name: str) -> tuple[int, ...]:
    """Return an integer shape without moving a tensor between devices."""

    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        try:
            raw_shape = np.asarray(value).shape
        except Exception as exc:  # pragma: no cover - defensive for exotic arrays
            raise ContractError(f"{name} 必须是具有 shape 的数组或张量") from exc
    try:
        shape = tuple(int(dim) for dim in raw_shape)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name}.shape 必须由整数维度组成，实际为 {raw_shape!r}") from exc
    if any(dim < 0 for dim in shape):
        raise ContractError(f"{name}.shape 不能包含负维度，实际为 {shape}")
    return shape


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return str(dtype).lower()


def _to_numpy_for_validation(value: Any, name: str) -> np.ndarray:
    """Copy small contract metadata to CPU when it is a torch tensor.

    Frames and features are never copied by contract validation.  This helper is
    used only for timestamps, masks and frame indices, which are intentionally
    small compared with the video tensor.
    """

    module_root = type(value).__module__.split(".", 1)[0]
    try:
        if module_root == "torch" and hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    except Exception as exc:
        raise ContractError(f"{name} 无法转换为可校验数组") from exc


def _freeze_mapping(value: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} 必须是 mapping，实际为 {type(value).__name__}")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ContractError(f"{name} 的键必须是非空字符串，实际为 {key!r}")
        copied[key] = item
    return MappingProxyType(copied)


def _validate_prefix_mask(mask: np.ndarray, name: str) -> None:
    if mask.ndim != 2:
        raise ContractError(f"{name} 必须是二维 [B,T] 或 [B,S] mask")
    if mask.dtype.kind != "b":
        raise ContractError(f"{name} 必须是 bool，实际 dtype={mask.dtype}")
    if np.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ContractError(f"{name} 的有效位置必须是连续前缀，不能在 padding 后重新出现")
    if np.any(mask.sum(axis=1) == 0):
        raise ContractError(f"{name} 的每个样本至少要有一个有效位置")


def _slice_array(value: Any, axis: int, start: int, stop: int) -> Any:
    slices = [slice(None)] * len(_shape(value, "cache tensor"))
    slices[axis] = slice(start, stop)
    return value[tuple(slices)]


@dataclass(frozen=True, slots=True)
class EncoderCapabilities:
    """Truthful, machine-checkable capabilities of one encoder adapter.

    ``supports_kv_cache`` and ``supports_token_cache`` mean the corresponding
    reusable state is actually exposed as :class:`CacheView`.  Internal model
    activations that callers cannot reuse do not count.
    """

    supports_fixed_clip: bool
    supports_streaming: bool = False
    supports_kv_cache: bool = False
    supports_token_cache: bool = False
    supports_visual_memory_cache: bool = False
    supports_external_cache_policy: bool = False
    supports_training: bool = False
    fixed_num_frames: int | None = None
    min_frames: int = 1
    max_frames: int | None = None

    def __post_init__(self) -> None:
        bool_fields = (
            "supports_fixed_clip",
            "supports_streaming",
            "supports_kv_cache",
            "supports_token_cache",
            "supports_visual_memory_cache",
            "supports_external_cache_policy",
            "supports_training",
        )
        for name in bool_fields:
            if type(getattr(self, name)) is not bool:
                raise CapabilityError(f"{name} 必须是 bool")
        if not self.supports_fixed_clip and not self.supports_streaming:
            raise CapabilityError("编码器必须至少支持 fixed-clip 或 streaming 中的一种")
        if (
            self.supports_kv_cache or self.supports_token_cache or self.supports_visual_memory_cache
        ) and not self.supports_streaming:
            raise CapabilityError(
                "只有真实 streaming 编码器才能声明 decoder KV、vision token 或 visual memory cache"
            )
        if self.supports_external_cache_policy and not (
            self.supports_kv_cache or self.supports_token_cache or self.supports_visual_memory_cache
        ):
            raise CapabilityError("外部 cache policy 需要至少一种可观察 cache")
        if type(self.min_frames) is not int or self.min_frames <= 0:
            raise CapabilityError("min_frames 必须是正整数")
        if self.max_frames is not None and (
            type(self.max_frames) is not int or self.max_frames < self.min_frames
        ):
            raise CapabilityError("max_frames 必须是不小于 min_frames 的整数")
        if self.fixed_num_frames is not None:
            if not self.supports_fixed_clip:
                raise CapabilityError("fixed_num_frames 只能用于 fixed-clip 编码器")
            if type(self.fixed_num_frames) is not int or self.fixed_num_frames <= 0:
                raise CapabilityError("fixed_num_frames 必须是正整数")
            if self.fixed_num_frames < self.min_frames:
                raise CapabilityError("fixed_num_frames 不能小于 min_frames")
            if self.max_frames is not None and self.fixed_num_frames > self.max_frames:
                raise CapabilityError("fixed_num_frames 不能大于 max_frames")

    @property
    def cache_kinds(self) -> frozenset[CacheKind]:
        kinds: set[CacheKind] = set()
        if self.supports_kv_cache:
            kinds.add(CacheKind.KV)
        if self.supports_token_cache:
            kinds.add(CacheKind.TOKEN)
        if self.supports_visual_memory_cache:
            kinds.add(CacheKind.VISUAL_MEMORY)
        return frozenset(kinds)

    @property
    def supports_grad(self) -> bool:
        """Compatibility spelling used by experiment capability negotiation."""

        return self.supports_training

    @property
    def cache_access(self) -> str:
        """Return ``none``, ``read`` or ``replace`` for config checks."""

        if self.supports_external_cache_policy:
            return "replace"
        if self.supports_kv_cache or self.supports_token_cache or self.supports_visual_memory_cache:
            return "read"
        return "none"

    def require(self, capability: str) -> None:
        """Raise a clear error when a requested declared capability is absent."""

        if capability not in self.__dataclass_fields__:
            raise CapabilityError(f"未知编码器能力：{capability}")
        value = getattr(self, capability)
        if type(value) is not bool:
            raise CapabilityError(f"{capability} 不是布尔能力字段")
        if not value:
            raise CapabilityError(f"编码器未声明能力：{capability}")


@dataclass(frozen=True, slots=True)
class ClipBatch:
    """Decoded video clips at the framework/adapter boundary.

    Attributes:
        frames: ``uint8`` frames with exact shape ``[B,T,H,W,3]`` (BTHWC).
        timestamps_s: Per-frame timestamps in seconds, shape ``[B,T]``.
        video_ids: One stable identifier per batch item.
        valid_mask: Optional prefix mask for a padded batch, shape ``[B,T]``.
        frame_indices: Optional zero-based source-frame indices, shape ``[B,T]``.
    """

    frames: ArrayLike
    timestamps_s: ArrayLike
    video_ids: Sequence[str]
    valid_mask: ArrayLike | None = None
    frame_indices: ArrayLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    layout: ClassVar[str] = "BTHWC"
    dtype: ClassVar[str] = "uint8"

    def __post_init__(self) -> None:
        frame_shape = _shape(self.frames, "frames")
        if len(frame_shape) != 5:
            raise ContractError(
                f"frames 必须是 BTHWC 五维数组 [B,T,H,W,3]，实际 shape={frame_shape}"
            )
        batch, steps, height, width, channels = frame_shape
        if min(batch, steps, height, width) <= 0 or channels != 3:
            raise ContractError(f"frames 必须满足 B/T/H/W>0 且通道数 C=3，实际 shape={frame_shape}")
        dtype_name = _dtype_name(self.frames)
        if dtype_name not in {"uint8", "torch.uint8"}:
            raise ContractError(f"frames 必须是 uint8，实际 dtype={dtype_name}")

        timestamp_shape = _shape(self.timestamps_s, "timestamps_s")
        if timestamp_shape != (batch, steps):
            raise ContractError(
                f"timestamps_s 必须是 [B,T]={(batch, steps)}，实际为 {timestamp_shape}"
            )
        timestamps = _to_numpy_for_validation(self.timestamps_s, "timestamps_s")
        if timestamps.dtype.kind not in "fiu":
            raise ContractError("timestamps_s 必须是数值类型")

        video_ids = tuple(self.video_ids)
        if len(video_ids) != batch:
            raise ContractError(f"video_ids 长度必须等于 batch={batch}")
        if any(not isinstance(video_id, str) or not video_id for video_id in video_ids):
            raise ContractError("video_ids 中每一项都必须是非空字符串")
        object.__setattr__(self, "video_ids", video_ids)

        if self.valid_mask is None:
            valid = np.ones((batch, steps), dtype=bool)
        else:
            mask_shape = _shape(self.valid_mask, "valid_mask")
            if mask_shape != (batch, steps):
                raise ContractError(
                    f"valid_mask 必须是 [B,T]={(batch, steps)}，实际为 {mask_shape}"
                )
            valid = _to_numpy_for_validation(self.valid_mask, "valid_mask")
            _validate_prefix_mask(valid, "valid_mask")

        for row in range(batch):
            row_times = timestamps[row, valid[row]].astype(np.float64, copy=False)
            if not np.all(np.isfinite(row_times)):
                raise ContractError(f"timestamps_s 第 {row} 个样本含 NaN/Inf")
            if np.any(row_times < 0):
                raise ContractError(f"timestamps_s 第 {row} 个样本含负时间")
            if np.any(np.diff(row_times) < 0):
                raise ContractError(f"timestamps_s 第 {row} 个样本不是单调不减")

        if self.frame_indices is not None:
            index_shape = _shape(self.frame_indices, "frame_indices")
            if index_shape != (batch, steps):
                raise ContractError(
                    f"frame_indices 必须是 [B,T]={(batch, steps)}，实际为 {index_shape}"
                )
            indices = _to_numpy_for_validation(self.frame_indices, "frame_indices")
            if indices.dtype.kind not in "iu":
                raise ContractError("frame_indices 必须是整数类型")
            for row in range(batch):
                row_indices = indices[row, valid[row]]
                if np.any(row_indices < 0) or np.any(np.diff(row_indices) <= 0):
                    raise ContractError(f"frame_indices 第 {row} 个样本必须非负且严格递增")

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @property
    def batch_size(self) -> int:
        return _shape(self.frames, "frames")[0]

    @property
    def num_frames(self) -> int:
        return _shape(self.frames, "frames")[1]

    @property
    def spatial_size(self) -> tuple[int, int]:
        shape = _shape(self.frames, "frames")
        return shape[2], shape[3]

    @property
    def valid_lengths(self) -> np.ndarray:
        if self.valid_mask is None:
            return np.full(self.batch_size, self.num_frames, dtype=np.int64)
        return _to_numpy_for_validation(self.valid_mask, "valid_mask").sum(axis=1)


@dataclass(frozen=True, slots=True)
class TokenTimeline:
    """Temporal provenance for encoder tokens, using half-open frame ranges."""

    start_s: ArrayLike
    end_s: ArrayLike
    valid_mask: ArrayLike | None = None
    source_frame_start: ArrayLike | None = None
    source_frame_end: ArrayLike | None = None

    def __post_init__(self) -> None:
        start_shape = _shape(self.start_s, "start_s")
        end_shape = _shape(self.end_s, "end_s")
        if len(start_shape) != 2 or start_shape != end_shape:
            raise ContractError(
                f"TokenTimeline start_s/end_s 必须同为 [B,S]，实际为 {start_shape}/{end_shape}"
            )
        batch, steps = start_shape
        if batch <= 0 or steps <= 0:
            raise ContractError("TokenTimeline 的 B 和 S 必须大于 0")
        starts = _to_numpy_for_validation(self.start_s, "start_s")
        ends = _to_numpy_for_validation(self.end_s, "end_s")
        if starts.dtype.kind not in "fiu" or ends.dtype.kind not in "fiu":
            raise ContractError("TokenTimeline 时间必须是数值类型")

        if self.valid_mask is None:
            valid = np.ones((batch, steps), dtype=bool)
        else:
            if _shape(self.valid_mask, "valid_mask") != start_shape:
                raise ContractError("TokenTimeline.valid_mask 必须与 start_s 同形")
            valid = _to_numpy_for_validation(self.valid_mask, "valid_mask")
            _validate_prefix_mask(valid, "TokenTimeline.valid_mask")

        for row in range(batch):
            row_starts = starts[row, valid[row]].astype(np.float64, copy=False)
            row_ends = ends[row, valid[row]].astype(np.float64, copy=False)
            if not np.all(np.isfinite(row_starts)) or not np.all(np.isfinite(row_ends)):
                raise ContractError(f"TokenTimeline 第 {row} 个样本含 NaN/Inf")
            if np.any(row_starts < 0) or np.any(row_ends < row_starts):
                raise ContractError(f"TokenTimeline 第 {row} 个样本含负时间或 end<start")
            if np.any(np.diff(row_starts) < 0) or np.any(np.diff(row_ends) < 0):
                raise ContractError(f"TokenTimeline 第 {row} 个样本必须按时间单调不减")

        frame_fields = (self.source_frame_start, self.source_frame_end)
        if (frame_fields[0] is None) != (frame_fields[1] is None):
            raise ContractError("source_frame_start/source_frame_end 必须同时提供或同时省略")
        if frame_fields[0] is not None:
            if (
                _shape(frame_fields[0], "source_frame_start") != start_shape
                or _shape(frame_fields[1], "source_frame_end") != start_shape
            ):
                raise ContractError("source frame 范围必须与 start_s 同形")
            frame_starts = _to_numpy_for_validation(frame_fields[0], "source_frame_start")
            frame_ends = _to_numpy_for_validation(frame_fields[1], "source_frame_end")
            if frame_starts.dtype.kind not in "iu" or frame_ends.dtype.kind not in "iu":
                raise ContractError("source frame 范围必须是整数类型")
            for row in range(batch):
                starts_valid = frame_starts[row, valid[row]]
                ends_valid = frame_ends[row, valid[row]]
                if np.any(starts_valid < 0) or np.any(ends_valid <= starts_valid):
                    raise ContractError("source frame 使用 [start,end) 且必须满足 0<=start<end")
                if np.any(np.diff(starts_valid) < 0) or np.any(np.diff(ends_valid) < 0):
                    raise ContractError("source frame 范围必须单调不减")

    @property
    def batch_size(self) -> int:
        return _shape(self.start_s, "start_s")[0]

    @property
    def num_tokens(self) -> int:
        return _shape(self.start_s, "start_s")[1]

    @property
    def start_seconds(self) -> ArrayLike:
        """Verbose alias for serialization code."""

        return self.start_s

    @property
    def end_seconds(self) -> ArrayLike:
        """Verbose alias for serialization code."""

        return self.end_s

    @property
    def valid_lengths(self) -> np.ndarray:
        if self.valid_mask is None:
            return np.full(self.batch_size, self.num_tokens, dtype=np.int64)
        return _to_numpy_for_validation(self.valid_mask, "valid_mask").sum(axis=1)

    @property
    def valid(self) -> ArrayLike | None:
        """Short compatibility alias for heads and artifact writers."""

        return self.valid_mask

    def slice_tokens(self, start: int, stop: int) -> TokenTimeline:
        if not (0 <= start < stop <= self.num_tokens):
            raise ContractError(
                f"timeline slice 必须满足 0<=start<stop<={self.num_tokens}，实际 {start}:{stop}"
            )

        def sliced(value: Any | None) -> Any | None:
            return None if value is None else _slice_array(value, 1, start, stop)

        return TokenTimeline(
            start_s=sliced(self.start_s),
            end_s=sliced(self.end_s),
            valid_mask=sliced(self.valid_mask),
            source_frame_start=sliced(self.source_frame_start),
            source_frame_end=sliced(self.source_frame_end),
        )


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    """Canonical token/features emitted by every encoder adapter."""

    features: ArrayLike
    timeline: TokenTimeline
    pooled: ArrayLike | None = None
    aux: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        feature_shape = _shape(self.features, "features")
        if len(feature_shape) != 3 or any(dim <= 0 for dim in feature_shape):
            raise ContractError(f"features 必须是非空 [B,S,D] 三维数组，实际 shape={feature_shape}")
        batch, tokens, dim = feature_shape
        if (self.timeline.batch_size, self.timeline.num_tokens) != (batch, tokens):
            raise ContractError(
                "features 的 [B,S] 必须与 timeline 一致，"
                f"实际 features={feature_shape}, timeline="
                f"{(self.timeline.batch_size, self.timeline.num_tokens)}"
            )
        if self.pooled is not None:
            pooled_shape = _shape(self.pooled, "pooled")
            if pooled_shape != (batch, dim):
                raise ContractError(
                    f"pooled 必须是 [B,D]={(batch, dim)}，实际 shape={pooled_shape}"
                )
        object.__setattr__(self, "aux", _freeze_mapping(self.aux, "aux"))

    @property
    def tokens(self) -> ArrayLike:
        """Compatibility alias: canonical storage remains ``features``."""

        return self.features

    @property
    def batch_size(self) -> int:
        return _shape(self.features, "features")[0]

    @property
    def num_tokens(self) -> int:
        return _shape(self.features, "features")[1]

    @property
    def feature_dim(self) -> int:
        return _shape(self.features, "features")[2]


class CacheKind(str, Enum):
    # Values match experiment YAML vocabulary.  The verbose members are aliases.
    KV = "decoder_kv"
    DECODER_KV = "decoder_kv"
    TOKEN = "vision_tokens"
    VISION_TOKENS = "vision_tokens"
    VISUAL_MEMORY = "visual_memory"


class CacheUpdateMode(str, Enum):
    APPEND = "append"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class CacheView:
    """Normalized view of one cache with a shared sequence axis.

    ``tensors`` flattens model-specific nested cache structures into stable
    names (for example ``layer.0.key`` and ``layer.0.value``).  Every tensor has
    batch on axis 0 and the same sequence length on ``sequence_axis``.
    """

    kind: CacheKind | str
    tensors: Mapping[str, ArrayLike]
    sequence_axis: int
    timeline: TokenTimeline
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, CacheKind) else CacheKind(self.kind)
        except ValueError as exc:
            raise ContractError(f"未知 cache kind：{self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        if type(self.sequence_axis) is not int:
            raise ContractError("sequence_axis 必须是 int")
        tensors = _freeze_mapping(self.tensors, "tensors")
        if not tensors:
            raise ContractError(
                "CacheView.tensors 不能为空；空初始状态请使用 StreamState.caches={}"
            )

        sequence_length: int | None = None
        batch_size: int | None = None
        normalized_axis: int | None = None
        for name, tensor in tensors.items():
            shape = _shape(tensor, f"tensors[{name!r}]")
            if len(shape) < 2:
                raise ContractError(f"cache tensor {name!r} 至少需要 batch 和 sequence 两个维度")
            axis = (
                self.sequence_axis if self.sequence_axis >= 0 else len(shape) + self.sequence_axis
            )
            if axis <= 0 or axis >= len(shape):
                raise ContractError(
                    f"sequence_axis={self.sequence_axis} 对 {name!r} shape={shape} 无效，且不能是 batch 轴"
                )
            if normalized_axis is None:
                normalized_axis = axis
            elif axis != normalized_axis:
                raise ContractError("所有 cache tensor 必须对 sequence_axis 得到同一规范化轴")
            if batch_size is None:
                batch_size = shape[0]
                sequence_length = shape[axis]
            elif shape[0] != batch_size or shape[axis] != sequence_length:
                raise ContractError("所有 cache tensor 的 batch/sequence 长度必须一致")
            if shape[0] <= 0 or shape[axis] <= 0:
                raise ContractError("CacheView 不接受空 batch 或空 sequence")

        assert sequence_length is not None and batch_size is not None
        if (self.timeline.batch_size, self.timeline.num_tokens) != (batch_size, sequence_length):
            raise ContractError(
                "cache timeline 必须精确对应 cache 的 batch/sequence，"
                f"实际 cache={(batch_size, sequence_length)}, timeline="
                f"{(self.timeline.batch_size, self.timeline.num_tokens)}"
            )
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "sequence_axis", normalized_axis)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @property
    def batch_size(self) -> int:
        first = next(iter(self.tensors.values()))
        return _shape(first, "cache tensor")[0]

    @property
    def sequence_length(self) -> int:
        first = next(iter(self.tensors.values()))
        return _shape(first, "cache tensor")[self.sequence_axis]

    @property
    def nbytes(self) -> int:
        total = 0
        for value in self.tensors.values():
            size = getattr(value, "nbytes", None)
            if size is not None:
                total += int(size)
            elif hasattr(value, "element_size") and hasattr(value, "numel"):
                total += int(value.element_size()) * int(value.numel())
        return total

    def slice_sequence(self, start: int, stop: int) -> CacheView:
        if not (0 <= start < stop <= self.sequence_length):
            raise ContractError(
                f"cache slice 必须满足 0<=start<stop<={self.sequence_length}，实际 {start}:{stop}"
            )
        sliced = {
            name: _slice_array(value, self.sequence_axis, start, stop)
            for name, value in self.tensors.items()
        }
        return CacheView(
            kind=self.kind,
            tensors=sliced,
            sequence_axis=self.sequence_axis,
            timeline=self.timeline.slice_tokens(start, stop),
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class CacheUpdate:
    """One model-produced replacement or append operation for a named cache."""

    view: CacheView
    mode: CacheUpdateMode | str = CacheUpdateMode.APPEND
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            mode = (
                self.mode if isinstance(self.mode, CacheUpdateMode) else CacheUpdateMode(self.mode)
            )
        except ValueError as exc:
            raise ContractError(f"未知 cache update mode：{self.mode!r}") from exc
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    @property
    def cache(self) -> CacheView:
        return self.view

    @classmethod
    def append(cls, view: CacheView, **metadata: Any) -> CacheUpdate:
        return cls(view=view, mode=CacheUpdateMode.APPEND, metadata=metadata)

    @classmethod
    def replace(cls, view: CacheView, **metadata: Any) -> CacheUpdate:
        return cls(view=view, mode=CacheUpdateMode.REPLACE, metadata=metadata)


@runtime_checkable
class CachePolicy(Protocol):
    """Policy that merges a cache update and optionally compresses the result."""

    name: str

    def apply(self, current: CacheView | None, update: CacheUpdate) -> CacheView: ...

    def compress(self, view: CacheView) -> CacheView: ...


@dataclass(frozen=True, slots=True)
class StreamState:
    """Per-video streaming state passed explicitly between encoder steps."""

    video_id: str
    step_index: int = 0
    caches: Mapping[str, CacheView] = field(default_factory=dict)
    opaque: Any = None
    next_timestamp_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id:
            raise ContractError("StreamState.video_id 必须是非空字符串")
        if type(self.step_index) is not int or self.step_index < 0:
            raise ContractError("StreamState.step_index 必须是非负整数")
        caches = _freeze_mapping(self.caches, "caches")
        if any(not isinstance(cache, CacheView) for cache in caches.values()):
            raise ContractError("StreamState.caches 的值必须是 CacheView")
        object.__setattr__(self, "caches", caches)
        if self.next_timestamp_s is not None:
            if not isinstance(self.next_timestamp_s, (int, float)) or not math.isfinite(
                float(self.next_timestamp_s)
            ):
                raise ContractError("next_timestamp_s 必须是有限数值或 None")
            if self.next_timestamp_s < 0:
                raise ContractError("next_timestamp_s 不能为负")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "metadata"))

    def replace(self, **changes: Any) -> StreamState:
        """Typed convenience wrapper around :func:`dataclasses.replace`."""

        return dataclass_replace(self, **changes)


@dataclass(frozen=True, slots=True)
class StreamStep:
    """Result of consuming one streaming chunk."""

    output: EncoderOutput | None
    state: StreamState
    cache_updates: Mapping[str, CacheUpdate] = field(default_factory=dict)
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    final: bool = False

    def __post_init__(self) -> None:
        if self.output is not None and not isinstance(self.output, EncoderOutput):
            raise ContractError("StreamStep.output 必须是 EncoderOutput 或 None")
        updates = _freeze_mapping(self.cache_updates, "cache_updates")
        if any(not isinstance(update, CacheUpdate) for update in updates.values()):
            raise ContractError("StreamStep.cache_updates 的值必须是 CacheUpdate")
        object.__setattr__(self, "cache_updates", updates)
        object.__setattr__(self, "telemetry", _freeze_mapping(self.telemetry, "telemetry"))
        if type(self.final) is not bool:
            raise ContractError("StreamStep.final 必须是 bool")


class VideoEncoderAdapter(ABC):
    """Fixed-clip encoder adapter contract."""

    capabilities: EncoderCapabilities

    @abstractmethod
    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        """Encode a canonical BTHWC clip batch."""


class StreamingVideoEncoderAdapter(VideoEncoderAdapter):
    """Encoder contract for persistent, explicitly managed video state."""

    @abstractmethod
    def init_state(self, video_id: str) -> StreamState:
        """Create an empty state for exactly one video."""

    @abstractmethod
    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression: CachePolicy | None = None,
    ) -> StreamStep:
        """Consume one B=1 chunk and return the next explicit state."""

    @abstractmethod
    def finalize(self, state: StreamState) -> EncoderOutput | None:
        """Flush buffered output, if any, without inventing additional frames."""


def validate_clip_for_capabilities(
    batch: ClipBatch,
    capabilities: EncoderCapabilities,
    *,
    streaming: bool = False,
    train: bool = False,
) -> None:
    """Validate an invocation against advertised adapter capabilities."""

    if streaming:
        capabilities.require("supports_streaming")
        if batch.batch_size != 1:
            raise CapabilityError("streaming encode_step 只接受 B=1 的单视频 chunk")
    else:
        capabilities.require("supports_fixed_clip")
    if train and not capabilities.supports_training:
        raise CapabilityError("该编码器未声明训练/梯度能力")
    lengths = batch.valid_lengths
    if np.any(lengths < capabilities.min_frames):
        raise CapabilityError(
            f"有效帧数不能小于 min_frames={capabilities.min_frames}，实际 {lengths.tolist()}"
        )
    if capabilities.max_frames is not None and np.any(lengths > capabilities.max_frames):
        raise CapabilityError(
            f"有效帧数不能大于 max_frames={capabilities.max_frames}，实际 {lengths.tolist()}"
        )
    if capabilities.fixed_num_frames is not None and np.any(
        lengths != capabilities.fixed_num_frames
    ):
        raise CapabilityError(
            f"编码器要求 fixed_num_frames={capabilities.fixed_num_frames}，实际 {lengths.tolist()}"
        )


def validate_encoder_output(output: EncoderOutput, batch: ClipBatch) -> None:
    if not isinstance(output, EncoderOutput):
        raise ContractError(f"adapter 必须返回 EncoderOutput，实际为 {type(output).__name__}")
    if output.batch_size != batch.batch_size:
        raise ContractError(
            f"EncoderOutput batch={output.batch_size} 与输入 batch={batch.batch_size} 不一致"
        )


def validate_stream_step(
    step: StreamStep,
    *,
    previous_state: StreamState,
    chunk: ClipBatch,
    capabilities: EncoderCapabilities,
) -> None:
    """Validate state progression and prevent undeclared cache behavior."""

    if not isinstance(step, StreamStep):
        raise ContractError(f"encode_step 必须返回 StreamStep，实际为 {type(step).__name__}")
    validate_clip_for_capabilities(chunk, capabilities, streaming=True)
    if chunk.video_ids != (previous_state.video_id,):
        raise ContractError("streaming chunk.video_ids 必须与 previous_state.video_id 一致")
    if step.state.video_id != previous_state.video_id:
        raise ContractError("encode_step 不能切换 StreamState.video_id")
    if step.state.step_index <= previous_state.step_index:
        raise ContractError("encode_step 必须递增 StreamState.step_index")
    if step.output is not None:
        validate_encoder_output(step.output, chunk)
    for name, update in step.cache_updates.items():
        if update.view.kind not in capabilities.cache_kinds:
            raise CapabilityError(
                f"adapter 返回了未声明的 {update.view.kind.value} cache update：{name}"
            )
    for name, view in step.state.caches.items():
        if view.kind not in capabilities.cache_kinds:
            raise CapabilityError(f"adapter state 暴露了未声明的 {view.kind.value} cache：{name}")


def validate_encoder_adapter(
    adapter: Any, expected: EncoderCapabilities | None = None
) -> EncoderCapabilities:
    """Validate a concrete (possibly duck-typed) adapter instance."""

    capabilities = getattr(adapter, "capabilities", None)
    if not isinstance(capabilities, EncoderCapabilities):
        raise CapabilityError("adapter.capabilities 必须是 EncoderCapabilities")
    if expected is not None and capabilities != expected:
        raise CapabilityError(
            "adapter 实际 capabilities 与 registry 声明不一致："
            f"expected={expected!r}, actual={capabilities!r}"
        )
    if capabilities.supports_fixed_clip and not callable(getattr(adapter, "encode", None)):
        raise CapabilityError("声明 supports_fixed_clip 的 adapter 必须实现 encode")
    if capabilities.supports_streaming:
        for method_name in ("init_state", "encode_step", "finalize"):
            if not callable(getattr(adapter, method_name, None)):
                raise CapabilityError(f"声明 supports_streaming 的 adapter 必须实现 {method_name}")
    if capabilities.supports_external_cache_policy:
        encode_step = adapter.encode_step
        try:
            parameters = inspect.signature(encode_step).parameters.values()
        except (TypeError, ValueError) as exc:
            raise CapabilityError("无法检查 encode_step 的 compression 参数") from exc
        accepts_compression = any(
            parameter.name == "compression" or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if not accepts_compression:
            raise CapabilityError(
                "声明 supports_external_cache_policy 的 encode_step 必须接收 compression 参数"
            )
    return capabilities


__all__ = [
    "ArrayLike",
    "CacheKind",
    "CachePolicy",
    "CacheUpdate",
    "CacheUpdateMode",
    "CacheView",
    "CapabilityError",
    "ClipBatch",
    "ContractError",
    "EncoderCapabilities",
    "EncoderOutput",
    "StreamState",
    "StreamStep",
    "StreamingVideoEncoderAdapter",
    "TokenTimeline",
    "VideoEncoderAdapter",
    "validate_clip_for_capabilities",
    "validate_encoder_adapter",
    "validate_encoder_output",
    "validate_stream_step",
]
