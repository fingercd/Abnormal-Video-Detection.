from __future__ import annotations

import json
from pathlib import Path

from vadbench.cli import main
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl


def test_cli_audit_writes_failed_report_for_incomplete_dataset(tmp_path: Path, capsys) -> None:
    train = write_manifest_jsonl(
        (
            VideoManifestRecord(
                video_id="train-normal",
                path="Training_Normal_Videos_Anomaly/train-normal.mp4",
                split="train",
                category="Normal",
                is_anomaly=False,
            ),
        ),
        tmp_path / "train.jsonl",
    )
    test = write_manifest_jsonl(
        (
            VideoManifestRecord(
                video_id="test-normal",
                path="Testing_Normal_Videos_Anomaly/test-normal.mp4",
                split="test",
                category="Normal",
                is_anomaly=False,
            ),
        ),
        tmp_path / "test.jsonl",
    )
    output = tmp_path / "audit.json"
    code = main(
        [
            "manifest",
            "audit-ucf",
            "--dataset-root",
            str(tmp_path / "videos"),
            "--train-manifest",
            str(train),
            "--test-manifest",
            str(test),
            "--output",
            str(output),
        ]
    )
    assert code == 3
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert summary["passed"] is False
    assert report["files"]["missing"] == 2
    assert report["status"] == "failed"
