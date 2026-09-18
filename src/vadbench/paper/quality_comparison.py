"""Single-head-seed, video-paired frame quality comparisons.

This pure module opens no score/annotation files. The caller must authorize
and SHA-bind those files, then align labels and BOTH final frame scores into
one complete, non-overlapping half-open partition per video. These are final
frame-score intervals, not overlapping clip predictions awaiting projection.

Metric references in this repository:
* ``vadbench.metrics.roc_auc_score`` (exact ties count half a win).
* ``research.xd_feature_grid.precision_recall_trapezoid_auc`` (PR trapezoids,
  including precision=1, recall=0; deliberately not step average precision).

The comparison protocol requires both frame classes even where standalone
PR conventions define a value for a one-class sample. Resampling preserves
the number of videos in each weak-label stratum, not each video's duration.
Head seeds are not additional videos and are intentionally outside this API.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

QualityMetric = Literal["frame_roc_auc", "frame_pr_auc"]
BOOTSTRAP_SEED = 20260918
BOOTSTRAP_DRAWS = 10000
MIN_VALID_DRAWS = 9990
NONINFERIORITY_MARGIN = 0.005
# A disclosed decision precision on the metrics' [0,1] scale, not an extra
# scientific degradation margin. Eight binary64 ULPs cover a small arithmetic
# budget for metric evaluation, subtraction and percentile interpolation.
DECISION_ULPS = 8
DECISION_TOLERANCE = float(DECISION_ULPS * np.spacing(np.float64(1.0)))


@dataclass(frozen=True)
class PairedVideoIntervals:
    """One video with aligned constant-label/constant-score frame intervals.

    ``intervals`` is integer [K,2], covers [0,num_frames) exactly, and has no
    gaps/overlap. ``labels``, ``dense_scores`` and ``method_scores`` are [K].
    Different source interval boundaries must be split/aligned by the caller.
    A weak-positive video may contain zero positive frames; a weak-normal
    video containing positive frame truth is rejected as an identity mismatch.
    """

    video_id: str
    weak_label: int
    num_frames: int
    intervals: Any
    labels: Any
    dense_scores: Any
    method_scores: Any


@dataclass(frozen=True)
class _Histogram:
    scores: np.ndarray
    positive: np.ndarray
    negative: np.ndarray


def _validated_video(video: PairedVideoIntervals):
    if not isinstance(video, PairedVideoIntervals):
        raise TypeError("videos must contain PairedVideoIntervals")
    if not isinstance(video.video_id, str) or not video.video_id.strip():
        raise ValueError("video_id must be a nonempty string")
    if type(video.weak_label) is not int or video.weak_label not in (0, 1):
        raise ValueError("weak_label must be integer 0 or 1")
    if isinstance(video.num_frames, bool) or not isinstance(video.num_frames, (int, np.integer)) or video.num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    intervals = np.asarray(video.intervals)
    if intervals.ndim != 2 or intervals.shape[1] != 2 or not len(intervals) or intervals.dtype.kind not in "iu":
        raise ValueError("intervals must be nonempty integer [K,2]")
    if (intervals[:, 0] < 0).any() or (intervals[:, 1] <= intervals[:, 0]).any():
        raise ValueError("intervals must be positive-length half-open frame ranges")
    if intervals[0, 0] != 0 or intervals[-1, 1] != video.num_frames or not np.array_equal(intervals[1:, 0], intervals[:-1, 1]):
        raise ValueError("both methods must share exact complete [0,num_frames) coverage without gaps/overlap")
    labels = np.asarray(video.labels)
    if labels.shape != (len(intervals),) or not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary and aligned with intervals")
    if video.weak_label == 0 and labels.any():
        raise ValueError("weak-normal video contains positive frame labels")
    scores = []
    for name in ("dense_scores", "method_scores"):
        values = np.asarray(getattr(video, name), dtype=np.float64)
        if values.shape != labels.shape or not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite and aligned with the same frame partition")
        scores.append(values)
    lengths = intervals[:, 1] - intervals[:, 0]
    return lengths, labels.astype(bool), scores


def _histogram(scores, lengths, labels):
    # Only sorting/compression in preparation. Every bootstrap draw uses counts.
    unique, inverse = np.unique(scores, return_inverse=True)
    positive = np.bincount(inverse, weights=lengths * labels, minlength=len(unique))
    negative = np.bincount(inverse, weights=lengths * ~labels, minlength=len(unique))
    return _Histogram(unique, positive, negative)


def _auc_pair_matrix(histograms: Sequence[_Histogram]) -> np.ndarray:
    """A[i,j] counts wins/ties of positive frames i versus negative frames j.

    For video multiplicities w, the exact pooled AUC numerator is w.T A w.
    The diagonal therefore correctly scales as w_i**2 for repeated videos.
    """
    size = len(histograms)
    result = np.zeros((size, size), dtype=np.float64)
    for j, negative in enumerate(histograms):
        mask = negative.negative > 0
        score = negative.scores[mask]
        if not len(score):
            continue
        cumulative = np.r_[0., np.cumsum(negative.negative[mask])]
        for i, positive in enumerate(histograms):
            mask = positive.positive > 0
            target, counts = positive.scores[mask], positive.positive[mask]
            if not len(target):
                continue
            below = np.searchsorted(score, target, side="left")
            through = np.searchsorted(score, target, side="right")
            result[i, j] = np.dot(counts, cumulative[below] + .5 * (cumulative[through] - cumulative[below]))
    return result


@dataclass(frozen=True)
class _PRCurve:
    video_index: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    starts: np.ndarray

    @classmethod
    def prepare(cls, histograms):
        scores = np.concatenate([h.scores for h in histograms])
        video_index = np.concatenate([np.full(len(h.scores), i, dtype=np.int64) for i, h in enumerate(histograms)])
        positive = np.concatenate([h.positive for h in histograms])
        negative = np.concatenate([h.negative for h in histograms])
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        return cls(video_index[order], positive[order], negative[order], starts)

    def evaluate(self, weights):
        """Weighted official PR trapezoids; no per-draw sorting or frame expansion."""
        output = np.full(len(weights), np.nan)
        # Bound temporaries by compressed bins, not potentially millions of frames.
        batch_size = max(1, min(64, (128 * 1024 * 1024) // max(1, 64 * len(self.video_index))))
        for begin in range(0, len(weights), batch_size):
            selected = weights[begin:begin + batch_size, self.video_index]
            positives = np.add.reduceat(selected * self.positive, self.starts, axis=1)
            negatives = np.add.reduceat(selected * self.negative, self.starts, axis=1)
            del selected
            tp = np.cumsum(positives, axis=1)
            total = np.cumsum(positives + negatives, axis=1)
            precision = np.divide(tp, total, out=np.ones_like(tp), where=total > 0)
            # Empty leading thresholds retain the PR origin (precision=1).
            # Empty interior thresholds retain the preceding point, so ties and
            # unsampled-video thresholds need not be removed separately.
            previous = np.concatenate([np.ones((len(tp), 1)), precision[:, :-1]], axis=1)
            numerator = np.sum(positives * (previous + precision) * .5, axis=1)
            valid = (tp[:, -1] > 0) & (total[:, -1] > tp[:, -1])
            batch = output[begin:begin + len(tp)]
            np.divide(numerator, tp[:, -1], out=batch, where=valid)
        return output


def _stratified_weights(labels: np.ndarray) -> np.ndarray:
    """Exactly 10000 shared video-count draws, preserving weak-label group sizes."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = np.zeros((BOOTSTRAP_DRAWS, len(labels)), dtype=np.int32)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices):
            result[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1. / len(indices)), size=BOOTSTRAP_DRAWS)
    return result


def _auc_delta(weights, difference, positives, negatives):
    result = np.full(len(weights), np.nan)
    for start in range(0, len(weights), 256):
        w = weights[start:start + 256].astype(np.float64)
        denominator = (w @ positives) * (w @ negatives)
        numerator = np.sum((w @ difference) * w, axis=1)
        np.divide(numerator, denominator, out=result[start:start + len(w)], where=denominator > 0)
    return result


def _decision_value(value):
    boundary = -NONINFERIORITY_MARGIN
    return boundary if abs(value - boundary) <= DECISION_TOLERANCE else value


def _interval_and_decision(delta_draws, point_delta):
    finite = np.isfinite(delta_draws)
    valid_count = int(finite.sum())
    if point_delta is None:
        return {"status": "NA", "reason": "both_frame_classes_required_by_comparison_protocol", "ci_low": None, "ci_high": None, "noninferiority": "NA", "point_noninferiority_pass": None, "decision_point_delta": None, "decision_ci_low": None, "decision_ci_high": None}
    decision_point = _decision_value(point_delta)
    point_pass = bool(decision_point >= -NONINFERIORITY_MARGIN)
    if valid_count < MIN_VALID_DRAWS:
        return {"status": "NA", "reason": "fewer_than_99.9_percent_valid_draws", "ci_low": None, "ci_high": None, "noninferiority": "NA", "point_noninferiority_pass": point_pass, "decision_point_delta": decision_point, "decision_ci_low": None, "decision_ci_high": None}
    # Never condition the claimed interval on a silently filtered subset.
    # Both metrics lie in [0,1]. For each invalid draw use -1 for the lower
    # endpoint calculation and +1 for the upper: an explicit conservative
    # bound over all possible missing delta values, retaining all 10000 draws.
    low = float(np.quantile(np.where(finite, delta_draws, -1.), .025))
    high = float(np.quantile(np.where(finite, delta_draws, 1.), .975))
    decision_low, decision_high = _decision_value(low), _decision_value(high)
    decision = "supported" if decision_low >= -NONINFERIORITY_MARGIN else "inferior" if decision_high < -NONINFERIORITY_MARGIN else "inconclusive"
    return {"status": "available", "reason": None, "ci_low": low, "ci_high": high, "noninferiority": decision, "point_noninferiority_pass": point_pass, "decision_point_delta": decision_point, "decision_ci_low": decision_low, "decision_ci_high": decision_high}


def compare_paired_quality(
    videos: Sequence[PairedVideoIntervals],
    *,
    metric: QualityMetric,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Return one head seed's paired quality delta and video bootstrap CI.

    Positive delta is better. Noninferiority requires CI lower >= -0.005 at
    the disclosed eight-ULP binary64 decision precision on the [0,1] metric
    scale. Only values within 8*spacing(1.0) of the boundary are snapped to it;
    all raw values are preserved alongside their explicit decision values.
    A passing point estimate alone is inconclusive when its CI crosses that
    boundary. At least 9990/10000 draws must contain both frame classes and
    finite metrics. No redraws. Failed draws are retained through conservative
    percentile endpoint bounds, not dropped from a supporting interval.
    ROC point and bootstrap deltas both divide the paired win-count difference
    once, avoiding disagreement from subtracting two separately rounded AUCs.

    ``source_sha256`` is supplied/verified by the future export-loading CLI;
    this function validates the digest format but never reads those files.
    It does not authorize access to annotations, scores, or official tests.
    """
    if metric not in ("frame_roc_auc", "frame_pr_auc"):
        raise ValueError("metric must be frame_roc_auc or frame_pr_auc (XD trapezoids, not step AP)")
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise ValueError("caller-supplied source SHA-256 bindings are required")
    for name, digest in source_sha256.items():
        if not isinstance(name, str) or not name or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("source_sha256 must map nonempty names to lowercase 64-hex digests")
    items = list(videos)
    if not items:
        raise ValueError("at least one complete video is required")
    validated = [_validated_video(v) for v in items]
    if len({v.video_id for v in items}) != len(items):
        raise ValueError("duplicate input video_id; bootstrap duplicates must be multiplicities, not extra videos")
    order = sorted(range(len(items)), key=lambda i: items[i].video_id)
    items, validated = [items[i] for i in order], [validated[i] for i in order]
    # Exact integer frame-count representation in float64, including any draw.
    frame_counts = [int(v.num_frames) for v in items]
    if max(frame_counts) * len(items) > 2 ** 53:
        raise ValueError("frame counts exceed exact float64 integer range for video bootstrap")
    histograms = [[], []]
    for lengths, labels, scores in validated:
        for method in (0, 1):
            histograms[method].append(_histogram(scores[method], lengths, labels))
    positive = np.array([h.positive.sum() for h in histograms[0]])
    negative = np.array([h.negative.sum() for h in histograms[0]])
    labels = np.array([v.weak_label for v in items])
    weights = _stratified_weights(labels)
    positive_draws, negative_draws = weights @ positive, weights @ negative
    class_valid = (positive_draws > 0) & (negative_draws > 0)
    point_valid = bool(positive.sum() > 0 and negative.sum() > 0)
    original = np.ones((1, len(items)), dtype=np.int32)
    if metric == "frame_roc_auc":
        dense_matrix, method_matrix = (_auc_pair_matrix(h) for h in histograms)
        denominator = positive.sum() * negative.sum()
        dense_point = float(dense_matrix.sum() / denominator) if point_valid else None
        method_point = float(method_matrix.sum() / denominator) if point_valid else None
        difference = method_matrix - dense_matrix
        delta = _auc_delta(weights, difference, positive, negative)
        point_delta = float(_auc_delta(original, difference, positive, negative)[0]) if point_valid else None
        engine = "exact_video_pair_matrix_quadratic_form"
    else:
        dense_curve, method_curve = (_PRCurve.prepare(h) for h in histograms)
        dense_point = float(dense_curve.evaluate(original)[0]) if point_valid else None
        method_point = float(method_curve.evaluate(original)[0]) if point_valid else None
        delta = method_curve.evaluate(weights) - dense_curve.evaluate(weights)
        point_delta = method_point - dense_point if point_valid else None
        engine = "compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids"
    valid = class_valid & np.isfinite(delta)
    delta[~valid] = np.nan
    decision = _interval_and_decision(delta, point_delta)
    return {
        **decision,
        "metric": metric,
        "metric_definition": "global_concatenated_frame_tie_aware_ROC_AUC" if metric == "frame_roc_auc" else "auc(recall,precision)_trapezoids_with_PR_origin_not_step_AP",
        "dense_value": dense_point,
        "method_value": method_point,
        "delta_method_minus_dense": point_delta,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "decision_precision": {
            "dtype": "float64",
            "unit_metric_scale": 1.0,
            "ulps": DECISION_ULPS,
            "absolute_tolerance": DECISION_TOLERANCE,
            "rule": "snap only values within this arithmetic resolution of -margin to the boundary; preserve raw delta/CI",
        },
        "n_videos": len(items),
        "n_frames": sum(frame_counts),
        "positive_frames": int(positive.sum()),
        "negative_frames": int(negative.sum()),
        "weak_label_strata": {"normal": int((labels == 0).sum()), "positive": int((labels == 1).sum())},
        "video_order": [v.video_id for v in items],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "valid_draws": int(valid.sum()),
        "invalid_draws": int((~valid).sum()),
        "invalid_reasons": {"no_positive_frames": int((positive_draws == 0).sum()), "no_negative_frames": int((negative_draws == 0).sum()), "nonfinite_metric_with_both_classes": int((class_valid & ~np.isfinite(delta)).sum())},
        "minimum_valid_draws": MIN_VALID_DRAWS,
        "bootstrap_frame_count_range": [int((positive_draws + negative_draws).min()), int((positive_draws + negative_draws).max())],
        "resampling": "paired_video_multiplicities_stratified_by_weak_label_with_fixed_stratum_sizes",
        "ci_rule": "95pct_percentile; invalid draws retained as [-1,+1] endpoint bounds; no redraw/filtering",
        "bootstrap_weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "engine": engine,
        "compressed_video_score_rows": {"dense": sum(len(h.scores) for h in histograms[0]), "method": sum(len(h.scores) for h in histograms[1])},
        "source_sha256": dict(source_sha256),
        "source_binding": "caller_verified_file_digests; this module performs no file access",
        "numpy_version": np.__version__,
    }


__all__ = ["PairedVideoIntervals", "QualityMetric", "compare_paired_quality"]
