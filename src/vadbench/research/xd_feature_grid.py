"""Strict XD official feature-grid PR area; no raw-video frame alignment.

Callers supply frozen, independently verified metadata and already pooled
per-video feature-row scores. This module never opens test annotations, GT
files, feature arrays, or videos. A GT header digest is not a GT file digest:
the caller must bind the canonical GT file when loading it after method freeze.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PROTOCOL_ID = "xd-violence/official-feature-grid-pr-auc-v1"
SCOPE = "canonical feature coordinates; not raw frame alignment"
OFFICIAL_VIDEO_COUNT = 800
OFFICIAL_GT_LENGTH = 2_330_384
OFFICIAL_FEATURE_ROWS = 145_649
REPEAT_FACTOR = 16
OFFICIAL_TEST_LIST_SHA256 = "c8793a06117b0b3a90a8478185a048e98be3c96803fd28fb9136cf345af6d816"
OFFICIAL_GT_HEADER_SHA256 = "7e0ec983ee64c64497f72ad1f876a05f7c9275bdedd7daf8073bf3daa6e750ee"


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_grid_table_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Digest every ordered metadata row, including direct-header evidence."""
    return _json_sha256(records)


def feature_grid_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    """Compute the digest for an external, immutable metadata freeze receipt."""
    return _json_sha256(metadata)


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _numeric_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numeric values")
    return array


def precision_recall_trapezoid_auc(y_true: Any, y_score: Any) -> float:
    """Match ``auc(recall, precision)`` after sklearn's binary PR curve.

    Equal scores form one threshold. The terminal PR point (recall=0,
    precision=1) is included. The no-positive convention also matches sklearn:
    every threshold has recall=1, so its trapezoid area is 0.5. This is not
    non-interpolated average precision.
    """
    targets = _numeric_vector(y_true, "y_true")
    scores = _numeric_vector(y_score, "y_score")
    if targets.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("y_true must contain only binary 0/1 labels")
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    threshold_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), scores.size - 1]
    true_positives = np.cumsum(targets[order], dtype=np.float64)[threshold_ends]
    precision = np.r_[1.0, true_positives / (threshold_ends + 1)]
    positives = float(true_positives[-1])
    recall = np.r_[
        0.0,
        true_positives / positives if positives else np.ones_like(true_positives),
    ]
    return float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) * 0.5))


def _verified_rows(
    metadata: Mapping[str, Any], frozen_metadata_sha256: str
) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    expected_digest = _sha256(frozen_metadata_sha256, "frozen_metadata_sha256")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    if feature_grid_metadata_sha256(metadata) != expected_digest:
        raise ValueError("metadata does not match its external frozen digest")
    if metadata.get("status") != "ok":
        raise ValueError("feature-grid metadata is provisional or incomplete")
    if _integer(metadata.get("video_count"), "video_count") != OFFICIAL_VIDEO_COUNT:
        raise ValueError("the official feature grid requires exactly 800 videos")
    if _integer(metadata.get("gt_reference_length"), "gt_reference_length") != OFFICIAL_GT_LENGTH:
        raise ValueError("the official GT reference length must be 2330384")
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != OFFICIAL_VIDEO_COUNT:
        raise ValueError("metadata.records must contain all 800 ordered videos")
    source = metadata.get("provenance")
    if not isinstance(source, Mapping):
        raise ValueError("frozen metadata requires source provenance")
    provenance = {
        name: _sha256(source.get(name), f"provenance.{name}")
        for name in ("official_test_list_sha256", "gt_npy_header_sha256", "ti_table_sha256")
    }
    if provenance["official_test_list_sha256"] != OFFICIAL_TEST_LIST_SHA256:
        raise ValueError(
            "official ordered test-list source digest does not match the verified source"
        )
    if provenance["gt_npy_header_sha256"] != OFFICIAL_GT_HEADER_SHA256:
        raise ValueError("official GT NPY header source digest does not match the verified source")
    if provenance["ti_table_sha256"] != feature_grid_table_sha256(records):
        raise ValueError("Ti table does not match its provenance digest")
    offset = 0
    seen = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ValueError(f"metadata record {index} must be a mapping")
        if _integer(row.get("video_index"), "video_index") != index:
            raise ValueError("metadata video order/index is not canonical")
        name = row.get("feature_name")
        if not isinstance(name, str) or not name or name != name.strip() or name in seen:
            raise ValueError("metadata feature_name must be nonempty and unique")
        # The frozen official RGB list contributes one crop-0 identity per video.
        if not name.endswith("__0.npy") or "/" in name or "\\" in name:
            raise ValueError("metadata feature_name must be the canonical crop-0 basename")
        seen.add(name)
        if (
            row.get("header_verified") is not True
            or row.get("status") != "header_verified"
            or row.get("verification_method") != "direct_npy_header"
        ):
            raise ValueError(f"video {index} lacks directly verified NPY header evidence")
        version = row.get("npy_version")
        if (
            version != [1, 0]
            or any(type(part) is not int for part in version)
            or row.get("npy_dtype") != "<f4"
        ):
            raise ValueError("direct NPY evidence must verify version 1.0 and dtype <f4")
        if _integer(row.get("feature_dim"), "feature_dim") != 1024:
            raise ValueError("direct NPY evidence must verify feature dimension 1024")
        _integer(row.get("npy_header_bytes"), "npy_header_bytes", minimum=1)
        rows = _integer(row.get("t"), "t", minimum=1)
        start = _integer(row.get("gt_start"), "gt_start")
        end = _integer(row.get("gt_end_exclusive"), "gt_end_exclusive", minimum=1)
        if start != offset or end != start + rows * REPEAT_FACTOR:
            raise ValueError("GT offsets must be contiguous half-open Ti*16 slices")
        offset = end
    if offset != OFFICIAL_GT_LENGTH:
        raise ValueError("sum(Ti*16) must equal the complete official GT reference length")
    return records, provenance


def evaluate_xd_feature_grid(
    video_scores: Sequence[Mapping[str, Any]],
    ground_truth: Any,
    metadata: Mapping[str, Any],
    *,
    method_frozen: bool,
    frozen_metadata_sha256: str,
) -> dict[str, Any]:
    """Evaluate complete, ordered scores in the canonical official feature grid.

    ``scores`` is one scalar per feature row after the caller's required crop
    aggregation, not a raw frame score or a fixed-length resampled sequence.
    Every input mapping supplies ``video_index``, ``feature_name`` and ``scores``.
    A separately frozen digest binds metadata and its source evidence. Missing
    header verification is never waived by ``method_frozen=True``.
    """
    if method_frozen is not True:
        raise ValueError("official test evaluation requires method_frozen=True")
    records, provenance = _verified_rows(metadata, frozen_metadata_sha256)
    if not isinstance(video_scores, Sequence) or len(video_scores) != OFFICIAL_VIDEO_COUNT:
        raise ValueError("scores must contain all 800 videos in the frozen official order")
    arrays = []
    for index, (item, row) in enumerate(zip(video_scores, records, strict=True)):
        if not isinstance(item, Mapping):
            raise ValueError(f"score record {index} must be a mapping")
        if set(item) != {"video_index", "feature_name", "scores"}:
            raise ValueError("score records require exactly video_index, feature_name and scores")
        if _integer(item.get("video_index"), "score video_index") != index:
            raise ValueError("score video order/index does not match frozen metadata")
        if item.get("feature_name") != row["feature_name"]:
            raise ValueError("score video identity/order does not match frozen metadata")
        scores = _numeric_vector(item.get("scores"), f"video {index} scores")
        if scores.size != row["t"]:
            raise ValueError(f"video {index} score length must equal its directly verified Ti")
        arrays.append(scores)
    truth = _numeric_vector(ground_truth, "ground_truth")
    if truth.size != OFFICIAL_GT_LENGTH:
        raise ValueError("ground_truth must have the complete official length 2330384")
    if not np.all((truth == 0) | (truth == 1)):
        raise ValueError("ground_truth must contain only binary 0/1 labels")
    expanded = np.repeat(np.concatenate(arrays), REPEAT_FACTOR)
    return {
        "protocol_id": PROTOCOL_ID,
        "scope": SCOPE,
        "frame_pr_auc": precision_recall_trapezoid_auc(truth, expanded),
        "video_count": OFFICIAL_VIDEO_COUNT,
        "feature_rows": OFFICIAL_FEATURE_ROWS,
        "gt_length": OFFICIAL_GT_LENGTH,
        "repeat_factor": REPEAT_FACTOR,
        "metadata_sha256": frozen_metadata_sha256,
        "provenance": {**provenance, "metadata_sha256": frozen_metadata_sha256},
        "gt_file_sha256": None,
        "gt_value_provenance": "caller-supplied; header digest does not authenticate array values",
        "raw_video_alignment": "not_implemented",
    }
