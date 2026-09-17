from __future__ import annotations

import math
import unittest

import numpy as np
import pytest

from vadbench.data.manifest import (
    SupervisionAnnotation,
    TemporalSpan,
    VideoManifestRecord,
    write_manifest_jsonl,
)
from vadbench.engine.evaluate import (
    evaluate_manifest_predictions,
    evaluate_ucf_prediction_records,
    evaluate_ucf_predictions,
    prediction_records_to_temporal,
    project_video_prediction,
)
from vadbench.metrics import (
    average_precision_score,
    compute_ucf_frame_metrics,
    project_intervals_to_frames,
    project_intervals_to_grid,
    resample_scores_to_frames,
    roc_auc_score,
)


class BinaryMetricTests(unittest.TestCase):
    def test_roc_auc_and_average_precision_known_example(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(roc_auc_score(labels, scores), 0.75)
        self.assertAlmostEqual(average_precision_score(labels, scores), 5.0 / 6.0)

    def test_metrics_are_tie_aware_and_order_invariant(self) -> None:
        labels = np.array([0, 1, 1, 0])
        tied = np.ones(4)
        self.assertAlmostEqual(roc_auc_score(labels, tied), 0.5)
        self.assertAlmostEqual(average_precision_score(labels, tied), 0.5)
        permutation = np.array([2, 0, 3, 1])
        self.assertEqual(
            average_precision_score(labels, tied),
            average_precision_score(labels[permutation], tied[permutation]),
        )

    def test_undefined_single_class_behavior(self) -> None:
        self.assertTrue(math.isnan(roc_auc_score([1, 1], [0.1, 0.2])))
        self.assertTrue(math.isnan(average_precision_score([0, 0], [0.1, 0.2])))
        with self.assertRaises(ValueError):
            roc_auc_score([1, 1], [0.1, 0.2], undefined="raise")

    def test_rejects_non_binary_or_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            roc_auc_score([0, 2], [0.0, 1.0])
        with self.assertRaises(ValueError):
            average_precision_score([0, 1], [0.0, np.nan])


class ProjectionTests(unittest.TestCase):
    def test_grid_projection_overlap_reductions_and_gap(self) -> None:
        intervals = [[0.0, 2.0], [1.0, 3.0]]
        scores = [0.2, 0.8]
        grid = [0.0, 1.0, 2.0, 3.0]
        np.testing.assert_allclose(
            project_intervals_to_grid(intervals, scores, grid, reduction="max"),
            [0.2, 0.8, 0.8, 0.0],
        )
        np.testing.assert_allclose(
            project_intervals_to_grid(intervals, scores, grid, reduction="mean"),
            [0.2, 0.5, 0.8, 0.0],
        )
        np.testing.assert_allclose(
            project_intervals_to_grid(intervals, scores, grid, reduction="first"),
            [0.2, 0.2, 0.8, 0.0],
        )
        np.testing.assert_allclose(
            project_intervals_to_grid(intervals, scores, grid, reduction="last"),
            [0.2, 0.8, 0.8, 0.0],
        )

    def test_multichannel_mean_projection(self) -> None:
        result = project_intervals_to_grid(
            [[0, 2], [1, 3]],
            [[1.0, 3.0], [3.0, 5.0]],
            [0, 1, 2],
            reduction="mean",
        )
        np.testing.assert_allclose(result, [[1, 3], [2, 4], [3, 5]])

    def test_seconds_to_frame_projection_uses_half_open_intervals(self) -> None:
        result = project_intervals_to_frames(
            [[0.0, 1.0], [1.0, 2.0]], [0.1, 0.9], num_frames=4, fps=2.0
        )
        np.testing.assert_allclose(result, [0.1, 0.1, 0.9, 0.9])

    def test_uniform_resampling_is_piecewise_constant(self) -> None:
        np.testing.assert_array_equal(
            resample_scores_to_frames([1.0, 2.0], 5), [1.0, 1.0, 1.0, 2.0, 2.0]
        )
        np.testing.assert_array_equal(
            project_video_prediction([0.2, 0.8], 4), [0.2] * 2 + [0.8] * 2
        )

    def test_prediction_valid_mask_filters_padded_scores_and_intervals(self) -> None:
        result = project_video_prediction(
            {
                "scores": [0.1, 0.9, 100.0],
                "intervals": [[0, 2], [2, 4], [4, 6]],
                "valid_mask": [True, True, False],
            },
            4,
        )
        np.testing.assert_allclose(result, [0.1, 0.1, 0.9, 0.9])

    def test_task_prediction_timeline_supplies_frame_intervals(self) -> None:
        from types import SimpleNamespace

        prediction = SimpleNamespace(
            predictions=np.array([[0.1, 0.9, 100.0]]),
            auxiliary={
                "timeline": SimpleNamespace(
                    source_frame_start=np.array([[0, 2, 4]]),
                    source_frame_end=np.array([[2, 4, 6]]),
                    valid_mask=np.array([[True, True, False]]),
                )
            },
        )
        np.testing.assert_allclose(project_video_prediction(prediction, 4), [0.1, 0.1, 0.9, 0.9])

    def test_invalid_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            project_intervals_to_grid([[1, 1]], [0.5], [1])


class UCFProtocolTests(unittest.TestCase):
    def test_global_frame_metrics_concatenate_videos(self) -> None:
        labels = {"normal": np.array([0, 0]), "abnormal": np.array([1, 1])}
        scores = {"normal": np.array([0.1, 0.2]), "abnormal": np.array([0.8, 0.9])}
        result = compute_ucf_frame_metrics(labels, scores)
        self.assertEqual(result.num_videos, 2)
        self.assertEqual(result.num_frames, 4)
        self.assertEqual(result.num_positive_frames, 2)
        self.assertAlmostEqual(result.frame_auc, 1.0)
        self.assertAlmostEqual(result.frame_ap, 1.0)

    def test_variable_length_video_sequences_are_supported(self) -> None:
        labels = [np.array([0]), np.array([0, 1, 1])]
        scores = [np.array([0.1]), np.array([0.2, 0.8, 0.9])]
        result = compute_ucf_frame_metrics(labels, scores)
        self.assertEqual(result.num_videos, 2)
        self.assertEqual(result.num_frames, 4)
        self.assertAlmostEqual(result.frame_auc, 1.0)

    def test_ucf_evaluation_projects_recorded_intervals(self) -> None:
        labels = {"v": np.array([0, 0, 1, 1])}
        predictions = {
            "v": {
                "scores": np.array([0.1, 0.9]),
                "intervals": np.array([[0, 2], [2, 4]]),
            }
        }
        result = evaluate_ucf_predictions(predictions, labels)
        np.testing.assert_allclose(result.frame_scores["v"], [0.1, 0.1, 0.9, 0.9])
        self.assertAlmostEqual(result.metrics.frame_auc, 1.0)
        self.assertAlmostEqual(result.metrics.frame_ap, 1.0)

    def test_artifact_prediction_records_use_frame_ranges(self) -> None:
        from types import SimpleNamespace

        records = [
            SimpleNamespace(
                video_id="v",
                clip_index=1,
                frame_start=2,
                frame_end=4,
                start_s=1.0,
                end_s=2.0,
                anomaly_score=0.9,
            ),
            SimpleNamespace(
                video_id="v",
                clip_index=0,
                frame_start=0,
                frame_end=2,
                start_s=0.0,
                end_s=1.0,
                anomaly_score=0.1,
            ),
        ]
        grouped, interval_unit = prediction_records_to_temporal(records)
        self.assertEqual(interval_unit, "frames")
        np.testing.assert_allclose(grouped["v"].scores, [0.1, 0.9])
        result = evaluate_ucf_prediction_records(records, {"v": [0, 0, 1, 1]})
        self.assertAlmostEqual(result.metrics.frame_auc, 1.0)

    def test_second_records_require_timing_information(self) -> None:
        from types import SimpleNamespace

        records = [
            SimpleNamespace(
                video_id="v",
                clip_index=0,
                frame_start=None,
                frame_end=None,
                start_s=0.0,
                end_s=1.0,
                anomaly_score=0.2,
            )
        ]
        with self.assertRaises(ValueError):
            evaluate_ucf_prediction_records(records, {"v": [0, 0]})

    def test_mapping_key_and_length_mismatch_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_ucf_frame_metrics({"a": [0]}, {"b": [0.1]})
        with self.assertRaises(ValueError):
            compute_ucf_frame_metrics({"a": [0, 1]}, {"a": [0.1]})


def _protocol_manifest(video_id: str, *, anomaly: bool) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{'Abuse' if anomaly else 'Normal'}/{video_id}.mp4",
        split="test",
        category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly,
        num_frames=4,
        fps=2.0,
        annotations=(
            SupervisionAnnotation(
                scope="frame",
                label="Abuse",
                is_anomaly=True,
                span=TemporalSpan(2, 4, "frame"),
            ),
        )
        if anomaly
        else (),
    )


def _protocol_predictions(video_ids: list[str], intervals=((0, 2), (2, 4))):
    from types import SimpleNamespace

    return [
        SimpleNamespace(
            video_id=video_id,
            clip_id=f"{video_id}:{index}",
            clip_index=index,
            frame_start=start,
            frame_end=end,
            start_s=start / 2,
            end_s=end / 2,
            anomaly_score=0.1 if index == 0 else 0.9,
            encoder_fingerprint=None,
        )
        for video_id in video_ids
        for index, (start, end) in enumerate(intervals)
    ]


def test_subset_protocol_marks_scope_and_requires_complete_coverage() -> None:
    manifests = (
        _protocol_manifest("normal", anomaly=False),
        _protocol_manifest("abuse", anomaly=True),
    )
    result = evaluate_manifest_predictions(
        _protocol_predictions(["normal", "abuse"]),
        manifests,
        protocol="subset",
    )
    payload = result.to_dict()
    assert payload["protocol"] == "ucf-crime/subset-frameauc-v1"
    assert payload["coverage"] == {
        "status": "validated",
        "num_videos": 2,
        "total_frames": 8,
        "covered_frames": 8,
        "gap_frames": 0,
        "overlap_frames": 0,
        "coverage_ratio": 1.0,
        "complete": True,
        "incomplete_video_ids": [],
    }
    assert payload["input_identity"]["videos"] == 2
    assert payload["input_identity"]["manifest_sha256"].startswith("sha256:")

    with pytest.raises(ValueError, match="gap"):
        evaluate_manifest_predictions(
            _protocol_predictions(["normal", "abuse"], intervals=((0, 1), (2, 4))),
            manifests,
            protocol="subset",
        )


@pytest.mark.parametrize("field", ["run_id", "encoder_fingerprint", "checkpoint_sha256"])
def test_subset_rejects_mixed_experiment_identity(field: str) -> None:
    manifests = (
        _protocol_manifest("normal", anomaly=False),
        _protocol_manifest("abuse", anomaly=True),
    )
    predictions = _protocol_predictions(["normal", "abuse"])
    for index, item in enumerate(predictions):
        value = "run-a" if index < 2 else "run-b"
        if field == "checkpoint_sha256":
            item.metadata = {field: value}
        else:
            setattr(item, field, value)
    with pytest.raises(ValueError, match="mix incompatible"):
        evaluate_manifest_predictions(predictions, manifests, protocol="subset")


def test_official_protocol_rejects_a_two_video_subset() -> None:
    manifests = (
        _protocol_manifest("normal", anomaly=False),
        _protocol_manifest("abuse", anomaly=True),
    )
    with pytest.raises(ValueError, match="exactly 290"):
        evaluate_manifest_predictions(
            _protocol_predictions(["normal", "abuse"]),
            manifests,
            protocol="official",
        )


def test_official_protocol_requires_matching_ready_audit(tmp_path) -> None:
    from test_data_audit import _fake_probe, _official_records_and_files, _official_source_registry

    from vadbench.data.audit import audit_ucf_crime_dataset

    train, manifests = _official_records_and_files(tmp_path)
    source_registry = _official_source_registry(tmp_path, train, manifests)
    train_path = write_manifest_jsonl(train, tmp_path / "train.jsonl")
    manifest_path = write_manifest_jsonl(manifests, tmp_path / "test.jsonl")
    predictions = _protocol_predictions(
        [item.video_id for item in manifests], intervals=((0, 50), (50, 100))
    )
    audit = audit_ucf_crime_dataset(
        tmp_path,
        train_path,
        manifest_path,
        probe_fn=_fake_probe,
        official_source_registry=source_registry,
    )
    assert audit["passed"]
    result = evaluate_manifest_predictions(
        predictions,
        manifest_path,
        protocol="official",
        audit_report=audit,
        official_source_registry=source_registry,
    )
    payload = result.to_dict()
    assert payload["protocol"] == "ucf-crime/official-frameauc-v1"
    assert payload["coverage"]["num_videos"] == 290
    assert payload["input_identity"]["dataset_audit"]["evaluation_ready"] is True

    mismatched = {**audit, "manifests": {**audit["manifests"], "test": "other.jsonl"}}
    with pytest.raises(ValueError, match="does not match"):
        evaluate_manifest_predictions(
            predictions,
            manifest_path,
            protocol="official",
            audit_report=mismatched,
            official_source_registry=source_registry,
        )

    stale = {
        **audit,
        "manifest_sha256": {**audit["manifest_sha256"], "test": "0" * 64},
    }
    with pytest.raises(ValueError, match="SHA256"):
        evaluate_manifest_predictions(
            predictions,
            manifest_path,
            protocol="official",
            audit_report=stale,
            official_source_registry=source_registry,
        )

    partial = {
        key: audit[key] for key in ("schema_version", "dataset", "passed", "evaluation_readiness")
    }
    with pytest.raises(ValueError, match="v2 schema"):
        evaluate_manifest_predictions(
            predictions,
            manifest_path,
            protocol="official",
            audit_report=partial,
            official_source_registry=source_registry,
        )
    altered = {
        **audit,
        "official_source_identity": {
            **audit["official_source_identity"],
            "source_commit": "b" * 40,
        },
    }
    with pytest.raises(ValueError, match="source identity changed"):
        evaluate_manifest_predictions(
            predictions,
            manifest_path,
            protocol="official",
            audit_report=altered,
            official_source_registry=source_registry,
        )


if __name__ == "__main__":
    unittest.main()
