"""Dependency-light metrics and temporal score projection utilities.

The UCF-Crime protocol reports one ROC-AUC value after concatenating the
frame-level predictions and labels of every test video.  This module keeps
that protocol independent from scikit-learn so feature extraction and metric
smoke tests can run in a minimal environment.

Intervals are interpreted as half-open ranges, ``[start, end)``.  Frame zero
has timestamp/index zero; with ``fps`` set, frame ``i`` is sampled at
``i / fps`` seconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

Reduction = Literal["max", "mean", "last", "first"]


def _as_finite_1d(values: Any, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _binary_targets(values: Any) -> np.ndarray:
    targets = _as_finite_1d(values, name="y_true")
    unique = np.unique(targets)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(f"y_true must contain only 0/1 labels, got {unique.tolist()}")
    return targets.astype(np.uint8, copy=False)


def roc_auc_score(
    y_true: Any,
    y_score: Any,
    *,
    undefined: Literal["nan", "raise"] = "nan",
) -> float:
    """Compute binary ROC-AUC with exact, tie-aware NumPy ranks.

    The result is identical to the Mann-Whitney interpretation of ROC-AUC.
    Equal scores receive their average rank, so an all-tied predictor has an
    AUC of ``0.5``.  A dataset with only one class is undefined and returns
    ``nan`` by default (or raises when ``undefined='raise'``).
    """

    targets = _binary_targets(y_true)
    scores = _as_finite_1d(y_score, name="y_score")
    if targets.shape != scores.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape, got {targets.shape} and {scores.shape}"
        )

    num_positive = int(targets.sum())
    num_negative = int(targets.size - num_positive)
    if num_positive == 0 or num_negative == 0:
        if undefined == "raise":
            raise ValueError("ROC-AUC is undefined when y_true contains one class")
        if undefined != "nan":
            raise ValueError("undefined must be either 'nan' or 'raise'")
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, scores.size + 1, dtype=np.float64)

    # Replace ordinal ranks by the mean rank of each exact-score tie group.
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], scores.size]
    for start, end in zip(starts, ends, strict=True):
        ranks[start:end] = (start + 1 + end) / 2.0

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    positive_rank_sum = float(ranks[inverse_order][targets.astype(bool)].sum())
    u_statistic = positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    return float(u_statistic / (num_positive * num_negative))


def average_precision_score(
    y_true: Any,
    y_score: Any,
    *,
    undefined: Literal["nan", "raise"] = "nan",
) -> float:
    """Compute binary average precision using grouped score thresholds.

    Grouping equal thresholds makes the result invariant to the input order
    within ties and matches the non-interpolated precision-recall definition
    commonly used by ``sklearn.metrics.average_precision_score``.
    """

    targets = _binary_targets(y_true)
    scores = _as_finite_1d(y_score, name="y_score")
    if targets.shape != scores.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape, got {targets.shape} and {scores.shape}"
        )

    num_positive = int(targets.sum())
    if num_positive == 0:
        if undefined == "raise":
            raise ValueError("average precision is undefined without positives")
        if undefined != "nan":
            raise ValueError("undefined must be either 'nan' or 'raise'")
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    true_positives = np.cumsum(sorted_targets, dtype=np.float64)
    false_positives = np.cumsum(1 - sorted_targets, dtype=np.float64)

    # Only evaluate at the end of each tied-score group.
    threshold_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), scores.size - 1]
    tp = true_positives[threshold_ends]
    fp = false_positives[threshold_ends]
    precision = tp / (tp + fp)
    recall = tp / num_positive
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_delta * precision))


def _validate_intervals(intervals: Any, *, expected_count: int | None = None) -> np.ndarray:
    array = np.asarray(intervals, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"intervals must have shape [N, 2], got {array.shape}")
    if expected_count is not None and array.shape[0] != expected_count:
        raise ValueError(
            f"interval count ({array.shape[0]}) does not match score count ({expected_count})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("intervals contain NaN or infinite values")
    if np.any(array[:, 1] <= array[:, 0]):
        raise ValueError("every interval must satisfy end > start")
    return array


def project_intervals_to_grid(
    intervals: Any,
    scores: Any,
    grid: Any,
    *,
    reduction: Reduction = "max",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Project interval scores onto arbitrary frame/time sample locations.

    Args:
        intervals: Source ``[start, end)`` ranges with shape ``[N, 2]``.
        scores: Values with shape ``[N]`` or ``[N, ...]``.
        grid: One-dimensional target sample locations (frame indices or time).
        reduction: How overlapping intervals are combined.
        fill_value: Value assigned where no interval covers a grid location.

    Returns:
        An array of shape ``[len(grid), ...]``.  A scalar score input produces
        a one-dimensional result.
    """

    score_array = np.asarray(scores)
    if score_array.ndim == 0:
        raise ValueError("scores must have a leading interval dimension")
    interval_array = _validate_intervals(intervals, expected_count=score_array.shape[0])
    grid_array = np.asarray(grid, dtype=np.float64)
    if grid_array.ndim != 1:
        raise ValueError(f"grid must be one-dimensional, got shape {grid_array.shape}")
    if not np.all(np.isfinite(grid_array)):
        raise ValueError("grid contains NaN or infinite values")
    if reduction not in {"max", "mean", "last", "first"}:
        raise ValueError(f"unsupported reduction: {reduction}")

    output_shape = (grid_array.size,) + score_array.shape[1:]
    output_dtype = np.result_type(score_array.dtype, np.asarray(fill_value).dtype, np.float32)
    output = np.full(output_shape, fill_value, dtype=output_dtype)
    covered = np.zeros(grid_array.size, dtype=bool)

    if reduction == "mean":
        totals = np.zeros(output_shape, dtype=output_dtype)
        counts = np.zeros(grid_array.size, dtype=np.int64)

    for interval, score in zip(interval_array, score_array, strict=True):
        selected = (grid_array >= interval[0]) & (grid_array < interval[1])
        if not np.any(selected):
            continue
        if reduction == "last":
            output[selected] = score
        elif reduction == "first":
            new_locations = selected & ~covered
            output[new_locations] = score
        elif reduction == "max":
            new_locations = selected & ~covered
            existing_locations = selected & covered
            output[new_locations] = score
            output[existing_locations] = np.maximum(output[existing_locations], score)
        else:  # mean
            totals[selected] += score
            counts[selected] += 1
        covered[selected] = True

    if reduction == "mean":
        if score_array.ndim == 1:
            output[covered] = totals[covered] / counts[covered]
        else:
            divisor_shape = (int(covered.sum()),) + (1,) * (score_array.ndim - 1)
            output[covered] = totals[covered] / counts[covered].reshape(divisor_shape)
    return output


def project_intervals_to_frames(
    intervals: Any,
    scores: Any,
    num_frames: int,
    *,
    fps: float | None = None,
    frame_times: Any | None = None,
    reduction: Reduction = "max",
    fill_value: float = 0.0,
) -> np.ndarray:
    """Project frame- or second-based intervals to a frame score sequence.

    When ``frame_times`` is supplied it is authoritative.  Otherwise a grid
    of frame indices is used when ``fps`` is ``None``, and timestamps
    ``arange(num_frames) / fps`` are used when ``fps`` is set.
    """

    if not isinstance(num_frames, (int, np.integer)) or num_frames < 0:
        raise ValueError("num_frames must be a non-negative integer")
    if frame_times is not None:
        grid = np.asarray(frame_times, dtype=np.float64)
        if grid.shape != (num_frames,):
            raise ValueError(f"frame_times must have shape ({num_frames},), got {grid.shape}")
    elif fps is None:
        grid = np.arange(num_frames, dtype=np.float64)
    else:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and greater than zero")
        grid = np.arange(num_frames, dtype=np.float64) / float(fps)
    return project_intervals_to_grid(
        intervals,
        scores,
        grid,
        reduction=reduction,
        fill_value=fill_value,
    )


def resample_scores_to_frames(scores: Any, num_frames: int) -> np.ndarray:
    """Piecewise-constantly expand uniformly spaced snippet scores to frames.

    This is the fallback for legacy UCF-Crime feature files that contain only
    a score sequence and no explicit snippet timeline.  For auditable runs,
    prefer :func:`project_intervals_to_frames` with recorded intervals.
    """

    score_array = np.asarray(scores)
    if score_array.ndim == 0 or score_array.shape[0] == 0:
        raise ValueError("scores must have a non-empty leading time dimension")
    if not isinstance(num_frames, (int, np.integer)) or num_frames < 0:
        raise ValueError("num_frames must be a non-negative integer")
    if num_frames == 0:
        return np.empty((0,) + score_array.shape[1:], dtype=score_array.dtype)
    source_indices = np.floor(
        np.arange(num_frames, dtype=np.float64) * score_array.shape[0] / num_frames
    ).astype(np.int64)
    source_indices = np.minimum(source_indices, score_array.shape[0] - 1)
    return score_array[source_indices]


def _flatten_video_values(values: Any, *, name: str) -> np.ndarray:
    if isinstance(values, Mapping):
        arrays = [np.asarray(value).reshape(-1) for value in values.values()]
        if not arrays:
            raise ValueError(f"{name} mapping must not be empty")
        return np.concatenate(arrays)

    if isinstance(values, np.ndarray):
        return values.reshape(-1)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if not values:
            raise ValueError(f"{name} must not be empty")
        if all(np.isscalar(value) for value in values):
            return np.asarray(values).reshape(-1)
        arrays = [np.asarray(value).reshape(-1) for value in values]
        return np.concatenate(arrays)
    return np.asarray(values).reshape(-1)


@dataclass(frozen=True)
class UCFFrameMetrics:
    """Standard global frame-level UCF-Crime metrics."""

    frame_auc: float
    frame_ap: float
    num_frames: int
    num_positive_frames: int
    num_negative_frames: int
    num_videos: int

    @property
    def roc_auc(self) -> float:
        return self.frame_auc

    @property
    def average_precision(self) -> float:
        return self.frame_ap

    def to_dict(self) -> dict[str, float | int]:
        result = asdict(self)
        # Include conventional aliases to avoid metric-name ambiguity in logs.
        result["roc_auc"] = self.frame_auc
        result["average_precision"] = self.frame_ap
        return result


def compute_ucf_frame_metrics(
    frame_labels: Any,
    frame_scores: Any,
    *,
    undefined: Literal["nan", "raise"] = "nan",
) -> UCFFrameMetrics:
    """Concatenate all videos, then compute the standard frame AUC and AP.

    ``frame_labels`` and ``frame_scores`` may be flat arrays, sequences of
    per-video arrays, or mappings keyed by video id.  For mappings the key
    sets must match; labels define the deterministic concatenation order.
    """

    num_videos = 1
    if isinstance(frame_labels, Mapping) or isinstance(frame_scores, Mapping):
        if not isinstance(frame_labels, Mapping) or not isinstance(frame_scores, Mapping):
            raise ValueError("frame_labels and frame_scores must both be mappings")
        label_keys = list(frame_labels)
        if set(label_keys) != set(frame_scores):
            missing_scores = sorted(set(label_keys) - set(frame_scores))
            missing_labels = sorted(set(frame_scores) - set(label_keys))
            raise ValueError(
                "video id mismatch: "
                f"missing scores={missing_scores}, missing labels={missing_labels}"
            )
        label_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        for video_id in label_keys:
            labels = np.asarray(frame_labels[video_id]).reshape(-1)
            scores = np.asarray(frame_scores[video_id]).reshape(-1)
            if labels.shape != scores.shape:
                raise ValueError(
                    f"video {video_id!r} has {labels.size} labels but {scores.size} scores"
                )
            label_parts.append(labels)
            score_parts.append(scores)
        labels_flat = np.concatenate(label_parts)
        scores_flat = np.concatenate(score_parts)
        num_videos = len(label_keys)
    else:
        labels_flat = _flatten_video_values(frame_labels, name="frame_labels")
        scores_flat = _flatten_video_values(frame_scores, name="frame_scores")
        if labels_flat.shape != scores_flat.shape:
            raise ValueError(
                f"frame label/score lengths differ: {labels_flat.size} vs {scores_flat.size}"
            )
        if isinstance(frame_labels, Sequence) and frame_labels:
            first = frame_labels[0]
            if not np.isscalar(first):
                num_videos = len(frame_labels)

    labels_binary = _binary_targets(labels_flat)
    scores_finite = _as_finite_1d(scores_flat, name="frame_scores")
    positive = int(labels_binary.sum())
    return UCFFrameMetrics(
        frame_auc=roc_auc_score(labels_binary, scores_finite, undefined=undefined),
        frame_ap=average_precision_score(labels_binary, scores_finite, undefined=undefined),
        num_frames=int(labels_binary.size),
        num_positive_frames=positive,
        num_negative_frames=int(labels_binary.size - positive),
        num_videos=num_videos,
    )


def ucf_frame_metrics(
    frame_labels: Any,
    frame_scores: Any,
    *,
    undefined: Literal["nan", "raise"] = "nan",
) -> dict[str, float | int]:
    """Dictionary-returning convenience wrapper for experiment logging."""

    return compute_ucf_frame_metrics(frame_labels, frame_scores, undefined=undefined).to_dict()


__all__ = [
    "UCFFrameMetrics",
    "average_precision_score",
    "compute_ucf_frame_metrics",
    "project_intervals_to_frames",
    "project_intervals_to_grid",
    "resample_scores_to_frames",
    "roc_auc_score",
    "ucf_frame_metrics",
]
