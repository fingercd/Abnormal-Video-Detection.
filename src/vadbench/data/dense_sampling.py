"""Fixed-encoder dense sliding-window sampling and score aggregation.

The sampler owns two deliberately separate concepts: ``frame_indices`` are
the frames actually handed to an encoder, whereas ``score_frame_start/end``
are the half-open source interval to which that clip score is projected.
Keeping both prevents a padded tail clip or an inferred uniform timeline from
silently becoming a claimed observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .sampling import FixedClipSample, SamplingError, sample_fixed_clip, uniform_segments

DenseReduction = Literal["mean", "max"]


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SamplingError(f"{name} 必须是正整数")
    return value


@dataclass(frozen=True)
class DenseClipSample:
    """One dense encoder window with its explicit output scoring support."""

    clip_index: int
    clip: FixedClipSample
    score_frame_start: int
    score_frame_end: int
    end_anchored: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.clip_index, bool) or not isinstance(self.clip_index, int):
            raise SamplingError("clip_index 必须是整数")
        if self.clip_index < 0:
            raise SamplingError("clip_index 必须非负")
        if not isinstance(self.clip, FixedClipSample):
            raise TypeError("clip 必须是 FixedClipSample")
        if (
            isinstance(self.score_frame_start, bool)
            or isinstance(self.score_frame_end, bool)
            or not isinstance(self.score_frame_start, int)
            or not isinstance(self.score_frame_end, int)
            or self.score_frame_start < 0
            or self.score_frame_end <= self.score_frame_start
        ):
            raise SamplingError("score interval 必须满足 0 <= start < end")
        valid_indices = self.clip.valid_frame_indices
        if self.score_frame_start != valid_indices[0]:
            raise SamplingError("score interval 起点必须等于第一个真实输入帧")
        if self.score_frame_end <= valid_indices[-1]:
            raise SamplingError("score interval 必须覆盖最后一个真实输入帧")
        if not isinstance(self.end_anchored, bool):
            raise SamplingError("end_anchored 必须是 boolean")

    @property
    def frame_indices(self) -> tuple[int, ...]:
        """Actual decoder indices; valid dense windows never contain padding."""

        return self.clip.frame_indices

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        """Mask for actual input positions, always all true for a valid window."""

        return self.clip.valid_mask


@dataclass(frozen=True)
class DenseSamplingPlan:
    """A fixed-clip dense protocol expressed entirely in source-frame units."""

    clip_frames: int
    frame_stride: int
    window_stride: int = 1

    def __post_init__(self) -> None:
        _positive_int(self.clip_frames, "clip_frames")
        _positive_int(self.frame_stride, "frame_stride")
        _positive_int(self.window_stride, "window_stride")
        if self.window_stride > self.temporal_span:
            raise SamplingError(
                "window_stride 不能大于固定 clip 的时间跨度，否则 dense 计分区间会有缺口"
            )

    @property
    def temporal_span(self) -> int:
        return (self.clip_frames - 1) * self.frame_stride + 1

    def sample(self, num_frames: int) -> tuple[DenseClipSample, ...]:
        """Return complete fixed-size windows plus one end-anchored tail window.

        For a source video at least as long as the encoder's native clip span,
        every returned window has ``valid_frames == clip_frames``.  Regular
        starts follow ``window_stride``; if that sequence does not land on the
        final complete start, one explicitly marked ``end_anchored`` window is
        added.  A shorter video cannot satisfy a fixed-clip adapter contract,
        so it fails rather than manufacturing a partial valid mask.
        """

        _positive_int(num_frames, "num_frames")
        if num_frames < self.temporal_span:
            raise SamplingError(
                "固定 dense sampler 要求 num_frames >= temporal_span；"
                "短视频需要单独、已声明的原生 clip 策略"
            )
        final_start = num_frames - self.temporal_span
        starts = list(range(0, final_start + 1, self.window_stride))
        end_anchored_start: int | None = None
        if starts[-1] != final_start:
            starts.append(final_start)
            end_anchored_start = final_start
        samples: list[DenseClipSample] = []
        for clip_index, start in enumerate(starts):
            end = start + self.temporal_span
            clip = sample_fixed_clip(
                num_frames,
                clip_frames=self.clip_frames,
                frame_stride=self.frame_stride,
                start_frame=start,
                end_frame=end,
                position="start",
            )
            samples.append(
                DenseClipSample(
                    clip_index=clip_index,
                    clip=clip,
                    score_frame_start=start,
                    score_frame_end=end,
                    end_anchored=start == end_anchored_start,
                )
            )
        return tuple(samples)


@dataclass(frozen=True)
class UniformFullClipSample:
    """One full native input clip scored over a logical uniform segment.

    The input window is deliberately allowed to extend outside its scoring
    segment.  This makes 32-segment MIL sampling usable with fixed encoders
    when an individual logical segment is shorter than a native clip.
    """

    clip_index: int
    clip: FixedClipSample
    score_frame_start: int
    score_frame_end: int
    requested_input_start: int
    input_start_frame: int
    input_end_frame: int
    input_window_reused: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.clip_index, bool) or not isinstance(self.clip_index, int):
            raise SamplingError("clip_index 必须是整数")
        if self.clip_index < 0 or not isinstance(self.clip, FixedClipSample):
            raise SamplingError("uniform full clip 必须包含非负索引和 FixedClipSample")
        values = (
            self.score_frame_start,
            self.score_frame_end,
            self.requested_input_start,
            self.input_start_frame,
            self.input_end_frame,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise SamplingError("uniform full clip 坐标必须是整数")
        if self.score_frame_start < 0 or self.score_frame_end <= self.score_frame_start:
            raise SamplingError("score interval 必须满足 0 <= start < end")
        if self.input_start_frame < 0 or self.input_end_frame <= self.input_start_frame:
            raise SamplingError("input window 必须满足 0 <= start < end")
        if self.clip.frame_indices[0] != self.input_start_frame:
            raise SamplingError("input_start_frame 必须等于真实输入首帧")
        if self.clip.frame_indices[-1] >= self.input_end_frame:
            raise SamplingError("input_end_frame 必须在真实输入末帧之后")
        if not all(self.clip.valid_mask):
            raise SamplingError("uniform full clip 不允许 padding valid_mask")
        if not isinstance(self.input_window_reused, bool):
            raise SamplingError("input_window_reused 必须是 boolean")

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return self.clip.frame_indices

    @property
    def valid_mask(self) -> tuple[bool, ...]:
        return self.clip.valid_mask


def sample_uniform_full_clips(
    num_frames: int,
    *,
    num_segments: int = 32,
    clip_frames: int = 16,
    frame_stride: int = 2,
) -> tuple[UniformFullClipSample, ...]:
    """Sample fixed-size clips around uniform segment centers across a video.

    Each output's score interval is exactly the corresponding
    :func:`uniform_segments` interval.  Its input starts at the centered
    complete window clamped to ``[0, num_frames - temporal_span]``; it may
    therefore cross either segment boundary.  The returned reuse flags make
    edge clamping and any repeated short-video logical score interval explicit
    instead of silently treating them as independent encoder observations.
    """

    num_frames = _positive_int(num_frames, "num_frames")
    num_segments = _positive_int(num_segments, "num_segments")
    clip_frames = _positive_int(clip_frames, "clip_frames")
    frame_stride = _positive_int(frame_stride, "frame_stride")
    temporal_span = (clip_frames - 1) * frame_stride + 1
    if num_frames < temporal_span:
        raise SamplingError(
            "固定 uniform full sampler 要求 num_frames >= temporal_span；"
            "短视频需要单独、已声明的原生 clip 策略"
        )
    if num_frames < num_segments:
        raise SamplingError(
            "uniform full sampler 无法把 num_segments 个计分区间无重叠地放入该视频；"
            "短视频需要单独、已声明的计分策略"
        )

    max_start = num_frames - temporal_span
    samples: list[UniformFullClipSample] = []
    seen_inputs: set[tuple[int, int]] = set()
    for segment in uniform_segments(num_frames, num_segments):
        requested_start = segment.start_frame + (segment.length - temporal_span) // 2
        input_start = min(max(requested_start, 0), max_start)
        input_end = input_start + temporal_span
        input_key = (input_start, input_end)
        clip = sample_fixed_clip(
            num_frames,
            clip_frames=clip_frames,
            frame_stride=frame_stride,
            start_frame=input_start,
            end_frame=input_end,
            position="start",
        )
        samples.append(
            UniformFullClipSample(
                clip_index=segment.index,
                clip=clip,
                score_frame_start=segment.start_frame,
                score_frame_end=segment.end_frame,
                requested_input_start=requested_start,
                input_start_frame=input_start,
                input_end_frame=input_end,
                input_window_reused=input_key in seen_inputs,
            )
        )
        seen_inputs.add(input_key)
    return tuple(samples)


@dataclass(frozen=True)
class DenseScoreAggregation:
    """Frame-aligned scores and the number of contributing dense windows."""

    scores: np.ndarray
    contributors: np.ndarray
    reduction: DenseReduction

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float64)
        contributors = np.asarray(self.contributors, dtype=np.int64)
        if scores.ndim != 1 or contributors.shape != scores.shape or scores.size == 0:
            raise ValueError("dense aggregation must contain aligned non-empty frame vectors")
        if not np.all(np.isfinite(scores)):
            raise ValueError("dense aggregation scores must be finite")
        if np.any(contributors <= 0):
            raise ValueError("dense aggregation must cover every source frame")
        if self.reduction not in {"mean", "max"}:
            raise ValueError("reduction must be 'mean' or 'max'")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "contributors", contributors)


def aggregate_interval_scores(
    frame_starts: Sequence[int] | np.ndarray,
    frame_ends: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    num_frames: int,
    reduction: DenseReduction = "mean",
) -> DenseScoreAggregation:
    """Aggregate overlapping half-open window scores into one score per frame.

    A gap, a malformed interval, or a non-finite score is an error.  This is
    the point where dense overlapping windows become legal, non-overlapping
    frame prediction intervals for the existing official evaluator.
    """

    _positive_int(num_frames, "num_frames")
    if reduction not in {"mean", "max"}:
        raise ValueError("reduction must be 'mean' or 'max'")
    starts = np.asarray(frame_starts)
    ends = np.asarray(frame_ends)
    values = np.asarray(scores, dtype=np.float64)
    if any(value.ndim != 1 for value in (starts, ends, values)):
        raise ValueError("frame_starts, frame_ends, and scores must be one-dimensional")
    if not starts.size or starts.shape != ends.shape or starts.shape != values.shape:
        raise ValueError("dense score inputs must be aligned and non-empty")
    if starts.dtype.kind not in "iu" or ends.dtype.kind not in "iu":
        raise ValueError("dense score frame intervals must contain integers")
    if not np.all(np.isfinite(values)):
        raise ValueError("dense score inputs contain NaN or Infinity")
    if np.any(starts < 0) or np.any(ends <= starts) or np.any(ends > num_frames):
        raise ValueError("dense score interval is invalid or exceeds num_frames")

    contributors = np.zeros(num_frames, dtype=np.int64)
    if reduction == "mean":
        total = np.zeros(num_frames, dtype=np.float64)
        for start, end, score in zip(starts, ends, values, strict=True):
            total[int(start) : int(end)] += score
            contributors[int(start) : int(end)] += 1
        result = np.divide(total, contributors, out=np.zeros_like(total), where=contributors > 0)
    else:
        result = np.full(num_frames, -np.inf, dtype=np.float64)
        for start, end, score in zip(starts, ends, values, strict=True):
            result[int(start) : int(end)] = np.maximum(result[int(start) : int(end)], score)
            contributors[int(start) : int(end)] += 1
    if np.any(contributors == 0):
        missing = int(np.count_nonzero(contributors == 0))
        raise ValueError(f"dense score intervals leave uncovered frames={missing}")
    return DenseScoreAggregation(scores=result, contributors=contributors, reduction=reduction)


def aggregate_dense_scores(
    samples: Sequence[DenseClipSample],
    scores: Sequence[float] | np.ndarray,
    *,
    num_frames: int,
    reduction: DenseReduction = "mean",
) -> DenseScoreAggregation:
    """Aggregate scores for samples created by :class:`DenseSamplingPlan`."""

    if not samples:
        raise ValueError("dense samples must not be empty")
    return aggregate_interval_scores(
        [item.score_frame_start for item in samples],
        [item.score_frame_end for item in samples],
        scores,
        num_frames=num_frames,
        reduction=reduction,
    )


__all__ = [
    "DenseClipSample",
    "DenseReduction",
    "DenseSamplingPlan",
    "DenseScoreAggregation",
    "UniformFullClipSample",
    "aggregate_dense_scores",
    "aggregate_interval_scores",
    "sample_uniform_full_clips",
]
