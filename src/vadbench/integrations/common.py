"""Shared, dependency-light helpers for video encoder integrations.

Adapters receive model-specific outputs but VADBench persists one public shape:
``features[B,S,D]`` plus ``pooled[B,D]`` and a matching
:class:`~vadbench.contracts.TokenTimeline`.  This module deliberately avoids
importing torch or any upstream model package so catalog and preflight commands
remain lightweight.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vadbench.contracts import ContractError, EncoderOutput, TokenTimeline

FEATURE_STAGES = frozenset(
    {
        "pooled",
        "fc_features",
        "backbone_tokens",
        "observed_backbone",
        "last_hidden_state",
        "vision_tokens",
        "projected_visual",
        "visual_memory",
        "decoder_contextual",
    }
)

_FEATURE_STAGE_ALIASES = {
    "fc": "fc_features",
    "backbone": "backbone_tokens",
    "tokens": "backbone_tokens",
    "hidden_state": "last_hidden_state",
    "projected": "projected_visual",
    "visual": "projected_visual",
    "memory": "visual_memory",
    "decoder": "decoder_contextual",
    "contextual": "decoder_contextual",
}

_SEQUENCE_FIELDS = (
    "last_hidden_state",
    "hidden_states",
    "features",
    "tokens",
    "video_features",
    "visual_features",
    "backbone_features",
    "projected_visual",
    "visual_memory",
    "decoder_contextual",
    "pooler_output",
    "pooled_output",
    "pooled",
)

_POOLED_FIELDS = ("pooler_output", "pooled_output", "pooled")


class OutputHealthError(ContractError):
    """Raised when an output cannot be recorded as a healthy smoke result."""

    def __init__(self, message: str, *, health: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.health = dict(health or {})


def _shape(value: Any, name: str) -> tuple[int, ...]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        try:
            raw_shape = np.asarray(value).shape
        except Exception as exc:  # pragma: no cover - exotic upstream output
            raise ContractError(f"{name} 必须是具有 shape 的数组或张量") from exc
    try:
        shape = tuple(int(dimension) for dimension in raw_shape)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name}.shape 必须由整数维度组成，实际为 {raw_shape!r}") from exc
    if any(dimension < 0 for dimension in shape):
        raise ContractError(f"{name}.shape 不能包含负维度，实际为 {shape}")
    return shape


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return str(dtype).lower()


def _to_numpy(value: Any, name: str) -> np.ndarray:
    try:
        if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
            tensor = value.detach().cpu()
            # NumPy has no native bfloat16 dtype, and torch consequently rejects
            # ``tensor.numpy()`` for bfloat16 CPU tensors.  The conversion is
            # validation-only: callers keep the original tensor and dtype while
            # finite checks inspect a temporary float32 view.
            if str(getattr(tensor, "dtype", "")).lower() in {
                "bfloat16",
                "torch.bfloat16",
            }:
                as_float = getattr(tensor, "float", None)
                if not callable(as_float):  # pragma: no cover - real torch exposes it
                    raise TypeError("bfloat16 torch-like tensor 缺少 float()")
                tensor = as_float()
            return tensor.numpy()
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - exotic upstream output
        raise ContractError(f"{name} 无法转换为可校验数组") from exc


def _get_field(value: Any, name: str) -> Any | None:
    if isinstance(value, Mapping) and name in value:
        return value[name]
    return getattr(value, name, None)


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    module = value_type.__module__
    qualified = value_type.__qualname__
    return qualified if module == "builtins" else f"{module}.{qualified}"


def _non_empty_label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} 必须是非空字符串")
    return value.strip()


def normalize_feature_stage(value: str) -> str:
    """Normalize one feature-stage label and reject taxonomy drift."""

    normalized = _non_empty_label(value, "feature_stage").lower().replace("-", "_")
    normalized = normalized.replace(" ", "_")
    normalized = _FEATURE_STAGE_ALIASES.get(normalized, normalized)
    if normalized not in FEATURE_STAGES:
        expected = ", ".join(sorted(FEATURE_STAGES))
        raise ContractError(f"未知 feature_stage={value!r}；允许值为 {expected}")
    return normalized


def normalize_sequence_source(value: str) -> str:
    """Validate the human-readable source used to obtain a token sequence."""

    return _non_empty_label(value, "sequence_source")


def normalize_feature_tensor(value: Any, *, batch_size: int | None = None) -> Any:
    """Return a non-empty ``[B,S,D]`` tensor without importing its backend.

    A pooled ``[B,D]`` tensor becomes a singleton token sequence.  Existing
    ``[B,S,D]`` tensors are returned by identity so adapters do not copy model
    activations merely to satisfy the framework boundary.
    """

    shape = _shape(value, "features")
    if len(shape) == 2:
        if hasattr(value, "unsqueeze"):
            normalized = value.unsqueeze(1)
        else:
            normalized = np.expand_dims(np.asarray(value), axis=1)
    elif len(shape) == 3:
        normalized = value
    else:
        raise ContractError(f"features 必须是 [B,D] 或 [B,S,D]，实际 shape={shape}")

    normalized_shape = _shape(normalized, "features")
    if any(dimension <= 0 for dimension in normalized_shape):
        raise ContractError(f"features 必须是非空 [B,S,D]，实际 shape={normalized_shape}")
    if batch_size is not None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ContractError("batch_size 必须是正整数")
        if normalized_shape[0] != batch_size:
            raise ContractError(
                f"features batch={normalized_shape[0]} 与期望 batch={batch_size} 不一致"
            )
    return normalized


def normalize_pooled_tensor(value: Any, *, batch_size: int, feature_dim: int) -> Any:
    """Validate a model-provided pooled tensor as exact ``[B,D]``."""

    shape = _shape(value, "pooled")
    expected = (batch_size, feature_dim)
    if shape != expected:
        raise ContractError(f"pooled 必须是 [B,D]={expected}，实际 shape={shape}")
    return value


def pool_feature_sequence(features: Any, valid_mask: Any | None = None) -> Any:
    """Mean-pool ``[B,S,D]``, respecting an optional prefix-valid mask."""

    normalized = normalize_feature_tensor(features)
    batch_size, token_count, _ = _shape(normalized, "features")
    module_root = type(normalized).__module__.split(".", 1)[0]
    if valid_mask is None:
        if module_root == "torch":
            return normalized.mean(dim=1)
        return np.asarray(normalized).mean(axis=1)

    mask = _to_numpy(valid_mask, "valid_mask")
    if mask.shape != (batch_size, token_count):
        raise ContractError(
            "valid_mask 必须与 features 的 [B,S] 一致，"
            f"实际 mask={mask.shape}, features={(batch_size, token_count)}"
        )
    if mask.dtype.kind != "b":
        raise ContractError(f"valid_mask 必须是 bool，实际 dtype={mask.dtype}")
    if np.any(mask.sum(axis=1) == 0):
        raise ContractError("valid_mask 每个样本至少需要一个有效 token")

    if module_root == "torch":
        new_tensor = getattr(normalized, "new_tensor", None)
        if not callable(new_tensor):  # pragma: no cover - real torch tensors expose it
            raise ContractError("torch-like features 缺少 new_tensor，无法应用 valid_mask")
        weights = new_tensor(mask.astype(np.float32, copy=False)).unsqueeze(-1)
        numerator = (normalized * weights).sum(dim=1)
        denominator = weights.sum(dim=1)
        return numerator / denominator

    array = np.asarray(normalized)
    weights = mask.astype(np.float32, copy=False)[..., None]
    return (array * weights).sum(axis=1) / weights.sum(axis=1)


def _compatible_feature_tensor(value: Any, batch_size: int | None) -> bool:
    try:
        shape = _shape(value, "model output candidate")
    except ContractError:
        return False
    if len(shape) not in {2, 3} or any(dimension <= 0 for dimension in shape):
        return False
    return batch_size is None or shape[0] == batch_size


def _select_from_value(
    value: Any,
    *,
    batch_size: int | None,
    source: str,
    seen: set[int],
) -> tuple[Any, str] | None:
    if _compatible_feature_tensor(value, batch_size):
        return value, source

    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index in range(len(value) - 1, -1, -1):
            selected = _select_from_value(
                value[index],
                batch_size=batch_size,
                source=f"{source}[{index}]",
                seen=seen,
            )
            if selected is not None:
                return selected
    return None


def select_feature_tensor(
    raw_output: Any,
    *,
    batch_size: int | None = None,
) -> tuple[Any, str]:
    """Select a feature tensor from common model-output objects.

    Explicit semantic fields are preferred over arbitrary tuple members.  This
    prevents a classification ``logits`` tensor from silently becoming the
    encoder representation when a hidden-state field is available.
    """

    if _compatible_feature_tensor(raw_output, batch_size):
        return raw_output, "pooled_tensor" if len(
            _shape(raw_output, "raw_output")
        ) == 2 else "tensor"

    seen = {id(raw_output)}
    for field_name in _SEQUENCE_FIELDS:
        candidate = _get_field(raw_output, field_name)
        if candidate is None:
            continue
        selected = _select_from_value(
            candidate,
            batch_size=batch_size,
            source=field_name,
            seen=seen,
        )
        if selected is not None:
            return selected

    if isinstance(raw_output, Sequence) and not isinstance(raw_output, (str, bytes, bytearray)):
        selected = _select_from_value(
            raw_output,
            batch_size=batch_size,
            source="output",
            seen=set(),
        )
        if selected is not None:
            return selected
    raise ContractError(
        f"无法从模型输出选择 [B,D] 或 [B,S,D] 表征；output_type={_qualified_type_name(raw_output)}"
    )


def _select_pooled_tensor(raw_output: Any, *, batch_size: int, feature_dim: int) -> Any | None:
    if _compatible_feature_tensor(raw_output, batch_size) and _shape(raw_output, "raw_output") == (
        batch_size,
        feature_dim,
    ):
        return raw_output
    for field_name in _POOLED_FIELDS:
        candidate = _get_field(raw_output, field_name)
        if candidate is None:
            continue
        if _shape(candidate, field_name) == (batch_size, feature_dim):
            return candidate
    return None


def build_output_aux(
    *,
    feature_stage: str,
    sequence_source: str,
    model_output_type: str,
    preprocess_profile: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical, serializable output identity fields for ``aux``."""

    aux = dict(extra or {})
    aux.update(
        {
            "feature_stage": normalize_feature_stage(feature_stage),
            "sequence_source": normalize_sequence_source(sequence_source),
            "model_output_type": _non_empty_label(model_output_type, "model_output_type"),
        }
    )
    if preprocess_profile is not None:
        aux["preprocess_profile"] = _non_empty_label(preprocess_profile, "preprocess_profile")
    return aux


def normalize_encoder_output(
    raw_output: Any,
    *,
    timeline: TokenTimeline,
    feature_stage: str,
    pooled: Any | None = None,
    sequence_source: str | None = None,
    preprocess_profile: str | None = None,
    aux: Mapping[str, Any] | None = None,
) -> EncoderOutput:
    """Convert a tensor or model-output object into canonical ``EncoderOutput``."""

    if not isinstance(timeline, TokenTimeline):
        raise ContractError("timeline 必须是 TokenTimeline")
    selected, observed_source = select_feature_tensor(raw_output, batch_size=timeline.batch_size)
    features = normalize_feature_tensor(selected, batch_size=timeline.batch_size)
    _, token_count, feature_dim = _shape(features, "features")
    if timeline.num_tokens != token_count:
        raise ContractError(
            f"timeline token={timeline.num_tokens} 与 features token={token_count} 不一致"
        )

    pooled_value = pooled
    if pooled_value is None:
        pooled_value = _select_pooled_tensor(
            raw_output,
            batch_size=timeline.batch_size,
            feature_dim=feature_dim,
        )
    if pooled_value is None:
        pooled_value = pool_feature_sequence(features, timeline.valid_mask)
    pooled_value = normalize_pooled_tensor(
        pooled_value,
        batch_size=timeline.batch_size,
        feature_dim=feature_dim,
    )

    source = sequence_source or (
        "pooled_singleton" if len(_shape(selected, "selected features")) == 2 else observed_source
    )
    return EncoderOutput(
        features=features,
        pooled=pooled_value,
        timeline=timeline,
        aux=build_output_aux(
            feature_stage=feature_stage,
            sequence_source=source,
            model_output_type=_qualified_type_name(raw_output),
            preprocess_profile=preprocess_profile,
            extra=aux,
        ),
    )


@dataclass(frozen=True, slots=True)
class TensorHealth:
    """JSON-ready health summary for one emitted tensor."""

    shape: tuple[int, ...]
    dtype: str
    finite: bool
    non_finite_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "finite": self.finite,
            "non_finite_count": self.non_finite_count,
        }


@dataclass(frozen=True, slots=True)
class TimelineHealth:
    """Checks linking token provenance to emitted features and source video."""

    token_count: int
    valid_tokens_per_batch: tuple[int, ...]
    token_count_matches: bool
    monotonic: bool
    in_video_range: bool
    video_bounds_checked: bool
    has_source_frames: bool
    min_start_seconds: float
    max_end_seconds: float
    min_source_frame: int | None
    max_source_frame_end: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "valid_tokens_per_batch": list(self.valid_tokens_per_batch),
            "token_count_matches": self.token_count_matches,
            "monotonic": self.monotonic,
            "in_video_range": self.in_video_range,
            "video_bounds_checked": self.video_bounds_checked,
            "has_source_frames": self.has_source_frames,
            "min_start_seconds": self.min_start_seconds,
            "max_end_seconds": self.max_end_seconds,
            "min_source_frame": self.min_source_frame,
            "max_source_frame_end": self.max_source_frame_end,
        }


@dataclass(frozen=True, slots=True)
class OutputHealth:
    """Complete machine-readable health report for one encoder output."""

    feature_stage: str
    sequence_source: str
    identity_inferred: bool
    features: TensorHealth
    pooled: TensorHealth | None
    timeline: TimelineHealth
    pooled_required: bool = True
    bounds_required: bool = False

    @property
    def passed(self) -> bool:
        pooled_ok = self.pooled is not None and self.pooled.finite
        if not self.pooled_required and self.pooled is None:
            pooled_ok = True
        bounds_ok = self.timeline.in_video_range
        if self.bounds_required:
            bounds_ok = bounds_ok and self.timeline.video_bounds_checked
        return bool(
            self.feature_stage
            and self.sequence_source
            and self.features.finite
            and pooled_ok
            and self.timeline.token_count_matches
            and self.timeline.monotonic
            and bounds_ok
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "feature_stage": self.feature_stage,
            "sequence_source": self.sequence_source,
            "identity_inferred": self.identity_inferred,
            "features": self.features.to_dict(),
            "pooled": None if self.pooled is None else self.pooled.to_dict(),
            "timeline": self.timeline.to_dict(),
        }


def inspect_tensor_health(value: Any, *, name: str) -> TensorHealth:
    """Inspect finite values without requiring the tensor backend at import time."""

    shape = _shape(value, name)
    dtype = _dtype_name(value)
    array = _to_numpy(value, name)
    try:
        finite_mask = np.isfinite(array)
    except TypeError:
        return TensorHealth(
            shape=shape,
            dtype=dtype,
            finite=False,
            non_finite_count=None,
        )
    non_finite_count = int(finite_mask.size - np.count_nonzero(finite_mask))
    return TensorHealth(
        shape=shape,
        dtype=dtype,
        finite=non_finite_count == 0,
        non_finite_count=non_finite_count,
    )


def inspect_timeline_health(
    timeline: TokenTimeline,
    *,
    expected_tokens: int,
    video_duration_seconds: float | None = None,
    video_num_frames: int | None = None,
    tolerance_seconds: float = 1e-6,
) -> TimelineHealth:
    """Inspect token count, monotonicity, and optional source-video bounds."""

    if not isinstance(timeline, TokenTimeline):
        raise ContractError("timeline 必须是 TokenTimeline")
    if type(expected_tokens) is not int or expected_tokens <= 0:
        raise ContractError("expected_tokens 必须是正整数")
    if video_duration_seconds is not None:
        if not isinstance(video_duration_seconds, (int, float)) or not math.isfinite(
            float(video_duration_seconds)
        ):
            raise ContractError("video_duration_seconds 必须是有限数值或 None")
        if video_duration_seconds < 0:
            raise ContractError("video_duration_seconds 不能为负")
    if video_num_frames is not None and (
        type(video_num_frames) is not int or video_num_frames <= 0
    ):
        raise ContractError("video_num_frames 必须是正整数或 None")
    if tolerance_seconds < 0 or not math.isfinite(float(tolerance_seconds)):
        raise ContractError("tolerance_seconds 必须是有限非负数")

    starts = _to_numpy(timeline.start_s, "timeline.start_s").astype(np.float64, copy=False)
    ends = _to_numpy(timeline.end_s, "timeline.end_s").astype(np.float64, copy=False)
    if timeline.valid_mask is None:
        valid = np.ones(starts.shape, dtype=bool)
    else:
        valid = _to_numpy(timeline.valid_mask, "timeline.valid_mask").astype(bool, copy=False)

    monotonic = True
    in_video_range = True
    valid_starts: list[np.ndarray] = []
    valid_ends: list[np.ndarray] = []
    for row in range(timeline.batch_size):
        row_starts = starts[row, valid[row]]
        row_ends = ends[row, valid[row]]
        valid_starts.append(row_starts)
        valid_ends.append(row_ends)
        monotonic = monotonic and bool(
            np.all(np.isfinite(row_starts))
            and np.all(np.isfinite(row_ends))
            and np.all(row_ends >= row_starts)
            and np.all(np.diff(row_starts) >= 0)
            and np.all(np.diff(row_ends) >= 0)
        )
        in_video_range = in_video_range and bool(np.all(row_starts >= -tolerance_seconds))
        if video_duration_seconds is not None:
            in_video_range = in_video_range and bool(
                np.all(row_ends <= float(video_duration_seconds) + tolerance_seconds)
            )

    merged_starts = np.concatenate(valid_starts)
    merged_ends = np.concatenate(valid_ends)
    has_source_frames = timeline.source_frame_start is not None
    min_source_frame: int | None = None
    max_source_frame_end: int | None = None
    frame_bounds_checked = False
    if has_source_frames:
        frame_starts = _to_numpy(timeline.source_frame_start, "timeline.source_frame_start")
        frame_ends = _to_numpy(timeline.source_frame_end, "timeline.source_frame_end")
        valid_frame_starts = np.concatenate(
            [frame_starts[row, valid[row]] for row in range(timeline.batch_size)]
        )
        valid_frame_ends = np.concatenate(
            [frame_ends[row, valid[row]] for row in range(timeline.batch_size)]
        )
        min_source_frame = int(valid_frame_starts.min())
        max_source_frame_end = int(valid_frame_ends.max())
        in_video_range = in_video_range and bool(
            np.all(valid_frame_starts >= 0) and np.all(valid_frame_ends > valid_frame_starts)
        )
        if video_num_frames is not None:
            frame_bounds_checked = True
            in_video_range = in_video_range and bool(np.all(valid_frame_ends <= video_num_frames))

    return TimelineHealth(
        token_count=timeline.num_tokens,
        valid_tokens_per_batch=tuple(int(item) for item in timeline.valid_lengths),
        token_count_matches=timeline.num_tokens == expected_tokens,
        monotonic=monotonic,
        in_video_range=in_video_range,
        video_bounds_checked=(video_duration_seconds is not None) or frame_bounds_checked,
        has_source_frames=has_source_frames,
        min_start_seconds=float(merged_starts.min()),
        max_end_seconds=float(merged_ends.max()),
        min_source_frame=min_source_frame,
        max_source_frame_end=max_source_frame_end,
    )


def _resolve_output_identity(
    output: EncoderOutput,
    *,
    feature_stage: str | None,
    sequence_source: str | None,
) -> tuple[str, str, bool]:
    stage_value = feature_stage or output.aux.get("feature_stage")
    source_value = sequence_source or output.aux.get("sequence_source")
    inferred = stage_value is None or source_value is None
    if stage_value is None:
        stage_value = "pooled" if output.num_tokens == 1 else "backbone_tokens"
    if source_value is None:
        source_value = "pooled_singleton" if output.num_tokens == 1 else "features"
    return (
        normalize_feature_stage(str(stage_value)),
        normalize_sequence_source(str(source_value)),
        inferred,
    )


def inspect_output_health(
    output: EncoderOutput,
    *,
    video_duration_seconds: float | None = None,
    video_num_frames: int | None = None,
    feature_stage: str | None = None,
    sequence_source: str | None = None,
    require_pooled: bool = True,
    require_video_bounds: bool = False,
) -> OutputHealth:
    """Create a JSON-ready report without raising for unhealthy values."""

    if not isinstance(output, EncoderOutput):
        raise ContractError("output 必须是 EncoderOutput")
    resolved_stage, resolved_source, inferred = _resolve_output_identity(
        output,
        feature_stage=feature_stage,
        sequence_source=sequence_source,
    )
    features = inspect_tensor_health(output.features, name="features")
    pooled = None if output.pooled is None else inspect_tensor_health(output.pooled, name="pooled")
    timeline = inspect_timeline_health(
        output.timeline,
        expected_tokens=output.num_tokens,
        video_duration_seconds=video_duration_seconds,
        video_num_frames=video_num_frames,
    )
    return OutputHealth(
        feature_stage=resolved_stage,
        sequence_source=resolved_source,
        identity_inferred=inferred,
        features=features,
        pooled=pooled,
        timeline=timeline,
        pooled_required=require_pooled,
        bounds_required=require_video_bounds,
    )


def validate_output_health(
    output: EncoderOutput,
    *,
    video_duration_seconds: float | None = None,
    video_num_frames: int | None = None,
    feature_stage: str | None = None,
    sequence_source: str | None = None,
    require_pooled: bool = True,
    require_video_bounds: bool = False,
) -> OutputHealth:
    """Return complete output health or raise with identical serialized evidence."""

    health = inspect_output_health(
        output,
        video_duration_seconds=video_duration_seconds,
        video_num_frames=video_num_frames,
        feature_stage=feature_stage,
        sequence_source=sequence_source,
        require_pooled=require_pooled,
        require_video_bounds=require_video_bounds,
    )
    if not health.passed:
        raise OutputHealthError("encoder 输出健康检查失败", health=health.to_dict())
    return health


__all__ = [
    "FEATURE_STAGES",
    "OutputHealth",
    "OutputHealthError",
    "TensorHealth",
    "TimelineHealth",
    "build_output_aux",
    "inspect_output_health",
    "inspect_tensor_health",
    "inspect_timeline_health",
    "normalize_encoder_output",
    "normalize_feature_stage",
    "normalize_feature_tensor",
    "normalize_pooled_tensor",
    "normalize_sequence_source",
    "pool_feature_sequence",
    "select_feature_tensor",
    "validate_output_health",
]
