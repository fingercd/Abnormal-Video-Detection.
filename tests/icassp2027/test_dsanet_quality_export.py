"""Check DSANet's 16-frame score mapping before frame-level evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


from vadbench.workflows.dsanet import _quality_runtime as module


def test_convert_preserves_tail_frame_and_sealed_xd_identity(tmp_path: Path) -> None:
    name = "Example.Movie__#00-01-00_label_B1-0-0"
    truth_id = module.truth_id("xd", name)
    truth = {
        truth_id: {"num_frames": 16},
    }
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps(
            {
                "video_id": name,
                "raw_frames": 17,
                "feature_rows": 2,
                "coarse_scores": [0.2, 0.8],
            }
        ) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "intervals.jsonl"
    receipt = module.convert("xd", "coarse", scores, truth, destination)
    intervals = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert receipt["videos"] == 1 and receipt["score_intervals"] == 2
    assert [row["video_id"] for row in intervals] == [truth_id, truth_id]
    assert [(row["frame_start"], row["frame_end"]) for row in intervals] == [(0, 16), (16, 17)]
    assert [row["anomaly_score"] for row in intervals] == [0.2, 0.8]


def test_convert_rejects_missing_canonical_frames(tmp_path: Path) -> None:
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        json.dumps({"video_id": "v", "raw_frames": 15, "feature_rows": 1,
                    "coarse_scores": [0.3]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shorter"):
        module.convert("ucf", "coarse", scores, {"v": {"num_frames": 16}}, tmp_path / "out.jsonl")
