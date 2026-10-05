from __future__ import annotations

import json
from pathlib import Path

import pytest

from vadbench.paper.urdmu_quality_export import compare_prediction_runs


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_compare_prediction_runs_aligns_intervals_and_reports_both_ap(tmp_path: Path):
    truth = _write_jsonl(
        tmp_path / "truth.jsonl",
        [{
            "video_id": "v0",
            "weak_label": 1,
            "num_frames": 10,
            "intervals": [{"start": 0, "end": 5, "label": 0}, {"start": 5, "end": 10, "label": 1}],
        }, {
            "video_id": "v1",
            "weak_label": 0,
            "num_frames": 10,
            "intervals": [{"start": 0, "end": 10, "label": 0}],
        }],
    )
    dense = _write_jsonl(
        tmp_path / "dense.jsonl",
        [{"video_id": "v0", "frame_start": 0, "frame_end": 10, "anomaly_score": 0.2},
         {"video_id": "v1", "frame_start": 0, "frame_end": 10, "anomaly_score": 0.1}],
    )
    method = _write_jsonl(
        tmp_path / "method.jsonl",
        [{"video_id": "v0", "frame_start": 0, "frame_end": 5, "anomaly_score": 0.1},
         {"video_id": "v0", "frame_start": 5, "frame_end": 10, "anomaly_score": 0.9},
         {"video_id": "v1", "frame_start": 0, "frame_end": 10, "anomaly_score": 0.1}],
    )
    result = compare_prediction_runs(
        dense_predictions=dense,
        method_predictions=method,
        truth_jsonl=truth,
        dataset="xd_violence",
    )
    assert result["metric"]["metric"] == "frame_pr_auc"
    assert result["metric"]["n_videos"] == 2
    assert result["metric"]["valid_draws"] >= 9990
    assert result["frame_ap_step"]["method"] >= result["frame_ap_step"]["dense"]
    assert result["video_level"]["method_video_roc_auc"] >= result["video_level"]["dense_video_roc_auc"]


def test_compare_prediction_runs_rejects_incomplete_prediction_cover(tmp_path: Path):
    truth = _write_jsonl(tmp_path / "truth.jsonl", [{"video_id": "v", "weak_label": 0, "num_frames": 4, "intervals": [{"start": 0, "end": 4, "label": 0}]}])
    dense = _write_jsonl(tmp_path / "dense.jsonl", [{"video_id": "v", "frame_start": 0, "frame_end": 4, "anomaly_score": 0.1}])
    method = _write_jsonl(tmp_path / "method.jsonl", [{"video_id": "v", "frame_start": 1, "frame_end": 4, "anomaly_score": 0.1}])
    with pytest.raises(ValueError, match="cover"):
        compare_prediction_runs(dense_predictions=dense, method_predictions=method, truth_jsonl=truth, dataset="ucf_crime")
