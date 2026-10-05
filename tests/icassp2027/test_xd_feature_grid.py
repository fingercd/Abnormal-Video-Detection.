from __future__ import annotations

import copy
import hashlib
import json
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from vadbench.research.xd_feature_grid import (
    OFFICIAL_GT_HEADER_SHA256,
    OFFICIAL_TEST_LIST_SHA256,
    evaluate_xd_feature_grid,
    feature_grid_metadata_sha256,
    feature_grid_table_sha256,
    precision_recall_trapezoid_auc,
)

VIDEO_COUNT = 800
FEATURE_ROWS = 145_649
GT_LENGTH = 2_330_384
OFFICIAL_LIST_SHA256 = "c8793a06117b0b3a90a8478185a048e98be3c96803fd28fb9136cf345af6d816"
OFFICIAL_GT_HEADER_SHA = "7e0ec983ee64c64497f72ad1f876a05f7c9275bdedd7daf8073bf3daa6e750ee"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sklearn_pr_area(y_true: Any, y_score: Any) -> float:
    metrics = pytest.importorskip("sklearn.metrics")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    return float(metrics.auc(recall, precision))


@pytest.fixture(scope="module")
def full_feature_grid() -> dict[str, Any]:
    rng = np.random.default_rng(202709)
    lengths = [182] * VIDEO_COUNT
    lengths[0] += FEATURE_ROWS - sum(lengths)
    assert sum(lengths) == FEATURE_ROWS

    records: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    offset = 0
    for video_index, length in enumerate(lengths):
        feature_name = f"video_{video_index:04d}__0.npy"
        records.append(
            {
                "video_index": video_index,
                "feature_name": feature_name,
                "t": length,
                "feature_dim": 1024,
                "npy_version": [1, 0],
                "npy_dtype": "<f4",
                "npy_header_bytes": 128,
                "status": "header_verified",
                "header_verified": True,
                "verification_method": "direct_npy_header",
                "gt_start": offset,
                "gt_end_exclusive": offset + length * 16,
            }
        )
        scores.append(
            {
                "video_index": video_index,
                "feature_name": feature_name,
                "scores": rng.normal(size=length),
            }
        )
        offset += length * 16
    assert offset == GT_LENGTH
    flat_scores = np.concatenate([item["scores"] for item in scores])
    repeated_scores = np.repeat(flat_scores, 16)
    ground_truth = (repeated_scores > np.quantile(repeated_scores, 0.8)).astype(np.uint8)
    metadata = {
        "status": "ok",
        "video_count": VIDEO_COUNT,
        "gt_reference_length": GT_LENGTH,
        "records": records,
        "provenance": {
            "official_test_list_sha256": OFFICIAL_LIST_SHA256,
            "gt_npy_header_sha256": OFFICIAL_GT_HEADER_SHA,
            "ti_table_sha256": _canonical_sha256(records),
        },
        "source": {"repository": "official-feature-grid-fixture", "revision": "synthetic"},
    }
    return {
        "scores": scores,
        "ground_truth": ground_truth,
        "metadata": metadata,
        "repeated_scores": repeated_scores,
    }


def _metadata_with(
    metadata: dict[str, Any],
    *,
    records: list[dict[str, Any]] | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(metadata)
    if records is not None:
        result["records"] = records
    if mutate is not None:
        mutate(result)
    result["provenance"]["ti_table_sha256"] = _canonical_sha256(result["records"])
    return result


@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]),
        ([0, 1, 0, 1, 1, 0], [0.2, 0.2, 0.8, 0.3, 0.3, 0.8]),
        ([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5]),
        ([0, 0], [0.1, 0.2]),
    ],
)
def test_precision_recall_trapezoid_auc_matches_sklearn_for_ties_and_no_positive(
    y_true: list[int], y_score: list[float]
) -> None:
    expected = _sklearn_pr_area(y_true, y_score)
    assert precision_recall_trapezoid_auc(y_true, y_score) == pytest.approx(expected)


def test_pr_trapezoid_area_is_not_average_precision() -> None:
    metrics = pytest.importorskip("sklearn.metrics")
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]

    assert _sklearn_pr_area(y_true, y_score) == pytest.approx(19 / 24)
    assert metrics.average_precision_score(y_true, y_score) == pytest.approx(5 / 6)
    assert precision_recall_trapezoid_auc(y_true, y_score) != pytest.approx(
        metrics.average_precision_score(y_true, y_score)
    )


def test_feature_grid_table_and_metadata_hash_include_all_declared_source_metadata(
    full_feature_grid: dict[str, Any],
) -> None:
    metadata = full_feature_grid["metadata"]
    assert OFFICIAL_TEST_LIST_SHA256 == OFFICIAL_LIST_SHA256
    assert OFFICIAL_GT_HEADER_SHA256 == OFFICIAL_GT_HEADER_SHA
    assert feature_grid_table_sha256(metadata["records"]) == _canonical_sha256(metadata["records"])
    original = feature_grid_metadata_sha256(metadata)
    changed = _metadata_with(
        metadata,
        mutate=lambda item: item["source"].update({"revision": "changed"}),
    )
    assert feature_grid_metadata_sha256(changed) != original


def test_evaluate_feature_grid_matches_repeat_16_and_sklearn_pr_area(
    full_feature_grid: dict[str, Any],
) -> None:
    metadata = full_feature_grid["metadata"]
    digest = feature_grid_metadata_sha256(metadata)

    result = evaluate_xd_feature_grid(
        full_feature_grid["scores"],
        full_feature_grid["ground_truth"],
        metadata,
        method_frozen=True,
        frozen_metadata_sha256=digest,
    )

    assert result["protocol_id"] == "xd-violence/official-feature-grid-pr-auc-v1"
    assert result["scope"] == "canonical feature coordinates; not raw frame alignment"
    assert result["frame_pr_auc"] == pytest.approx(
        _sklearn_pr_area(full_feature_grid["ground_truth"], full_feature_grid["repeated_scores"])
    )
    assert result["video_count"] == VIDEO_COUNT
    assert result["feature_rows"] == FEATURE_ROWS
    assert result["gt_length"] == GT_LENGTH
    assert result["metadata_sha256"] == digest
    assert result["gt_file_sha256"] is None
    assert result["provenance"] == {**metadata["provenance"], "metadata_sha256": digest}
    assert result["repeat_factor"] == 16
    assert result["raw_video_alignment"] == "not_implemented"
    assert (
        result["gt_value_provenance"]
        == "caller-supplied; header digest does not authenticate array values"
    )


@pytest.mark.parametrize(
    "change_scores",
    [
        lambda scores: scores.__setitem__(0, {**scores[0], "scores": np.ones((2, 1))}),
        lambda scores: scores.__setitem__(slice(0, 2), [scores[1], scores[0]]),
        lambda scores: scores.pop(),
        lambda scores: scores.__setitem__(
            0, {key: value for key, value in scores[0].items() if key != "scores"}
        ),
        lambda scores: scores.__setitem__(1, {**scores[1], "video_index": 0}),
        lambda scores: scores.__setitem__(0, {**scores[0], "feature_name": "wrong.npy"}),
        lambda scores: scores.__setitem__(0, {**scores[0], "unexpected": True}),
    ],
)
def test_evaluate_rejects_score_shape_order_count_index_name_and_extra_fields(
    full_feature_grid: dict[str, Any], change_scores: Callable[[list[dict[str, Any]]], None]
) -> None:
    scores = list(full_feature_grid["scores"])
    change_scores(scores)
    metadata = full_feature_grid["metadata"]

    with pytest.raises(ValueError):
        evaluate_xd_feature_grid(
            scores,
            full_feature_grid["ground_truth"],
            metadata,
            method_frozen=True,
            frozen_metadata_sha256=feature_grid_metadata_sha256(metadata),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda metadata: metadata.update({"status": "provisional"}),
        lambda metadata: metadata["records"][0].update({"header_verified": False}),
        lambda metadata: metadata["records"][0].update({"header_verified": 1}),
        lambda metadata: metadata["records"][0].update({"status": "layout_derived"}),
        lambda metadata: metadata["records"][0].update({"verification_method": "layout_inference"}),
        lambda metadata: metadata["records"][0].update({"npy_version": [2, 0]}),
        lambda metadata: metadata["records"][0].update({"npy_version": [True, 0]}),
        lambda metadata: metadata["records"][0].update({"npy_dtype": "<f8"}),
        lambda metadata: metadata["records"][0].update({"npy_header_bytes": 0}),
        lambda metadata: metadata["records"][0].pop("feature_dim"),
        lambda metadata: metadata["records"][0].update({"t": 183}),
        lambda metadata: metadata["records"][1].update({"gt_start": 16}),
        lambda metadata: metadata.update({"video_count": 799}),
    ],
)
def test_evaluate_rejects_provisional_or_inconsistent_metadata_before_gt_access(
    full_feature_grid: dict[str, Any], mutate: Callable[[dict[str, Any]], None]
) -> None:
    metadata = _metadata_with(full_feature_grid["metadata"], mutate=mutate)

    class GroundTruthMustNotBeRead:
        def __array__(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
            pytest.fail(
                "invalid metadata must be rejected before ground-truth validation or scoring"
            )

    with pytest.raises(ValueError):
        evaluate_xd_feature_grid(
            full_feature_grid["scores"],
            GroundTruthMustNotBeRead(),
            metadata,
            method_frozen=True,
            frozen_metadata_sha256=feature_grid_metadata_sha256(metadata),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda metadata: metadata["provenance"].pop("official_test_list_sha256"),
        lambda metadata: metadata["provenance"].update({"official_test_list_sha256": "b" * 64}),
        lambda metadata: metadata["provenance"].pop("gt_npy_header_sha256"),
        lambda metadata: metadata["provenance"].update({"gt_npy_header_sha256": "not-a-sha"}),
        lambda metadata: metadata["provenance"].update({"gt_npy_header_sha256": "a" * 64}),
        lambda metadata: metadata["provenance"].pop("ti_table_sha256"),
        lambda metadata: metadata["provenance"].update({"ti_table_sha256": "0" * 64}),
    ],
)
def test_evaluate_rejects_missing_invalid_or_mismatched_provenance_hashes(
    full_feature_grid: dict[str, Any], mutate: Callable[[dict[str, Any]], None]
) -> None:
    metadata = _metadata_with(full_feature_grid["metadata"])
    mutate(metadata)

    with pytest.raises(ValueError):
        evaluate_xd_feature_grid(
            full_feature_grid["scores"],
            full_feature_grid["ground_truth"],
            metadata,
            method_frozen=True,
            frozen_metadata_sha256=feature_grid_metadata_sha256(metadata),
        )


@pytest.mark.parametrize(
    ("method_frozen", "frozen_hash"),
    [(False, None), (True, "0" * 64), (1, None)],
)
def test_evaluate_requires_frozen_method_and_exact_metadata_identity(
    full_feature_grid: dict[str, Any], method_frozen: bool | int, frozen_hash: str | None
) -> None:
    metadata = full_feature_grid["metadata"]
    if method_frozen == 1 and type(method_frozen) is int:
        frozen_hash = feature_grid_metadata_sha256(metadata)
    with pytest.raises(ValueError):
        evaluate_xd_feature_grid(
            full_feature_grid["scores"],
            full_feature_grid["ground_truth"],
            metadata,
            method_frozen=method_frozen,
            frozen_metadata_sha256=frozen_hash,
        )


@pytest.mark.parametrize(
    "ground_truth",
    [
        np.zeros((1, GT_LENGTH), dtype=np.uint8),
        np.zeros(GT_LENGTH - 1, dtype=np.uint8),
        np.full(GT_LENGTH, 2, dtype=np.uint8),
        np.full(GT_LENGTH, np.nan),
    ],
)
def test_evaluate_rejects_invalid_canonical_ground_truth(
    full_feature_grid: dict[str, Any], ground_truth: np.ndarray
) -> None:
    metadata = full_feature_grid["metadata"]
    with pytest.raises(ValueError):
        evaluate_xd_feature_grid(
            full_feature_grid["scores"],
            ground_truth,
            metadata,
            method_frozen=True,
            frozen_metadata_sha256=feature_grid_metadata_sha256(metadata),
        )
