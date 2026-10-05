"""Compare two UR-DMU prediction-only runs on one sealed frame truth.

The prediction writer deliberately does not open official frame truth.  This
module is the narrow consumer-side bridge: a caller supplies a separately
audited truth JSONL, two complete prediction JSONL files, and provenance
paths.  It aligns the dense and method intervals before calling the shared
video-paired bootstrap engine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.metrics import average_precision_score, roc_auc_score
from vadbench.workflows.quality_comparison import PairedVideoIntervals, compare_paired_quality


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"prediction/truth JSONL is empty: {path}")
    return rows


def _truth_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("truth row lacks a nonempty video_id")
        if video_id in result:
            raise ValueError(f"duplicate truth video_id: {video_id}")
        weak_label = row.get("weak_label")
        num_frames = row.get("num_frames")
        intervals = row.get("intervals")
        if type(weak_label) is not int or weak_label not in (0, 1):
            raise ValueError(f"{video_id}: weak_label must be 0 or 1")
        if type(num_frames) is not int or num_frames <= 0:
            raise ValueError(f"{video_id}: num_frames must be positive int")
        if not isinstance(intervals, list) or not intervals:
            raise ValueError(f"{video_id}: truth intervals are missing")
        parsed = []
        for item in intervals:
            if not isinstance(item, Mapping):
                raise ValueError(f"{video_id}: truth interval must be an object")
            start, end, label = item.get("start"), item.get("end"), item.get("label")
            if type(start) is not int or type(end) is not int or end <= start:
                raise ValueError(f"{video_id}: invalid truth interval")
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f"{video_id}: invalid truth interval label")
            parsed.append((start, end, label))
        _validate_partition(video_id, num_frames, [(a, b) for a, b, _ in parsed])
        if weak_label == 0 and any(label for _, _, label in parsed):
            raise ValueError(f"{video_id}: weak-normal truth contains positive frames")
        result[video_id] = {
            "weak_label": weak_label,
            "num_frames": num_frames,
            "intervals": parsed,
            "truncate_predictions": bool(row.get("truncate_predictions", False)),
        }
    return result


def _prediction_rows(path: str | Path) -> dict[str, list[tuple[int, int, float]]]:
    result: dict[str, list[tuple[int, int, float]]] = {}
    for row in _jsonl(path):
        video_id = row.get("video_id")
        start, end, score = row.get("frame_start"), row.get("frame_end"), row.get("anomaly_score")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("prediction row lacks a nonempty video_id")
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError(f"{video_id}: invalid prediction interval")
        if not isinstance(score, (int, float)) or not np.isfinite(score):
            raise ValueError(f"{video_id}: prediction score is not finite")
        result.setdefault(video_id, []).append((start, end, float(score)))
    return result


def _validate_partition(video_id: str, num_frames: int, intervals: Sequence[tuple[int, int]]) -> None:
    ordered = sorted(intervals)
    if not ordered or ordered[0][0] != 0 or ordered[-1][1] != num_frames:
        raise ValueError(f"{video_id}: intervals do not cover [0,num_frames)")
    if any(start < 0 or end > num_frames or end <= start for start, end in ordered):
        raise ValueError(f"{video_id}: interval lies outside the frame range")
    if any(left[1] != right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"{video_id}: intervals have a gap or overlap")


def _score_at(rows: Sequence[tuple[int, int, float]], frame: int, video_id: str) -> float:
    matches = [score for start, end, score in rows if start <= frame < end]
    if len(matches) != 1:
        raise ValueError(f"{video_id}: prediction intervals do not form a unique cover at frame {frame}")
    return matches[0]


def _aligned_video(
    video_id: str,
    truth: Mapping[str, Any],
    dense: Sequence[tuple[int, int, float]],
    method: Sequence[tuple[int, int, float]],
) -> PairedVideoIntervals:
    num_frames = int(truth["num_frames"])
    if truth.get("truncate_predictions"):
        def clip(rows: Sequence[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
            return [(start, min(end, num_frames), score) for start, end, score in rows if start < num_frames and min(end, num_frames) > start]
        dense = clip(dense)
        method = clip(method)
    _validate_partition(video_id, num_frames, [(a, b) for a, b, _ in dense])
    _validate_partition(video_id, num_frames, [(a, b) for a, b, _ in method])
    boundaries = {0, num_frames}
    boundaries.update(end for start, end, _ in truth["intervals"] for end in (start, end))
    boundaries.update(end for start, end, _ in dense for end in (start, end))
    boundaries.update(end for start, end, _ in method for end in (start, end))
    points = sorted(boundaries)
    intervals, labels, dense_scores, method_scores = [], [], [], []
    truth_intervals = truth["intervals"]
    for start, end in zip(points, points[1:]):
        label = next((label for a, b, label in truth_intervals if a <= start < b), None)
        if label is None:
            raise ValueError(f"{video_id}: truth does not cover frame {start}")
        intervals.append((start, end))
        labels.append(label)
        dense_scores.append(_score_at(dense, start, video_id))
        method_scores.append(_score_at(method, start, video_id))
    return PairedVideoIntervals(
        video_id=video_id,
        weak_label=int(truth["weak_label"]),
        num_frames=num_frames,
        intervals=np.asarray(intervals, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.uint8),
        dense_scores=np.asarray(dense_scores, dtype=np.float64),
        method_scores=np.asarray(method_scores, dtype=np.float64),
    )


def _weighted_average_precision(videos: Sequence[PairedVideoIntervals], *, method: bool) -> float:
    labels, scores, weights = [], [], []
    for video in videos:
        labels.append(np.asarray(video.labels, dtype=np.uint8))
        scores.append(np.asarray(video.method_scores if method else video.dense_scores, dtype=np.float64))
        intervals = np.asarray(video.intervals, dtype=np.int64)
        weights.append(intervals[:, 1] - intervals[:, 0])
    y = np.concatenate(labels)
    s = np.concatenate(scores)
    w = np.concatenate(weights).astype(np.float64)
    # The interval weights are exact frame multiplicities.  This is the same
    # grouped threshold definition as average_precision_score, without
    # expanding potentially millions of frames in memory.
    order = np.argsort(-s, kind="mergesort")
    ss, yy, ww = s[order], y[order], w[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    tp = np.add.reduceat(ww * yy, starts)
    fp = np.add.reduceat(ww * (1 - yy), starts)
    if tp.sum() <= 0:
        return float("nan")
    precision = np.divide(tp.cumsum(), (tp + fp).cumsum(), out=np.zeros_like(tp), where=(tp + fp).cumsum() > 0)
    recall_delta = np.diff(np.r_[0.0, tp.cumsum() / tp.sum()])
    return float(np.sum(recall_delta * precision))


def compare_prediction_runs(
    *,
    dense_predictions: str | Path,
    method_predictions: str | Path,
    truth_jsonl: str | Path,
    dataset: str,
    source_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Return one paired quality export for two complete prediction runs."""

    if dataset not in {"ucf_crime", "xd_violence"}:
        raise ValueError("dataset must be ucf_crime or xd_violence")
    truth = _truth_rows(truth_jsonl)
    dense = _prediction_rows(dense_predictions)
    method = _prediction_rows(method_predictions)
    if set(dense) != set(method) or set(dense) != set(truth):
        raise ValueError("dense, method and truth video identities differ")
    videos = [_aligned_video(video_id, truth[video_id], dense[video_id], method[video_id]) for video_id in sorted(truth)]
    metric = "frame_roc_auc" if dataset == "ucf_crime" else "frame_pr_auc"
    paths = [Path(dense_predictions), Path(method_predictions), Path(truth_jsonl), *(Path(p) for p in source_paths)]
    source_sha = {str(p.resolve()): _sha256(p) for p in paths}
    comparison = compare_paired_quality(videos, metric=metric, source_sha256=source_sha)
    step_ap = {
        "dense": _weighted_average_precision(videos, method=False),
        "method": _weighted_average_precision(videos, method=True),
    }
    result = {
        "schema": "icassp2027.urdmu-quality-export/v1",
        "dataset": dataset,
        "metric": comparison,
        "video_level": {
            "dense_video_roc_auc": roc_auc_score(
                [video.weak_label for video in videos],
                [float(np.max(video.dense_scores)) for video in videos],
            ),
            "method_video_roc_auc": roc_auc_score(
                [video.weak_label for video in videos],
                [float(np.max(video.method_scores)) for video in videos],
            ),
        },
        "frame_ap_step": {
            **step_ap,
            "definition": "grouped score-threshold step average precision with exact interval frame weights",
        },
        "average_precision": {
            **step_ap,
            "definition": "sklearn average_precision_score equivalent after exact interval frame weighting; reported separately from XD PR trapezoids",
        },
        "actual_frame_counts": {video.video_id: int(video.num_frames) for video in videos},
        "source_sha256": source_sha,
    }
    return result


__all__ = ["compare_prediction_runs"]
