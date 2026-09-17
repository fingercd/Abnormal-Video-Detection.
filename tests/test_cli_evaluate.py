from __future__ import annotations

import json
from pathlib import Path

import yaml

from vadbench.artifacts import PredictionRecord
from vadbench.cli import main
from vadbench.data.manifest import (
    SupervisionAnnotation,
    TemporalSpan,
    VideoManifestRecord,
    write_manifest_jsonl,
)


def test_cli_evaluate_writes_frame_metrics(tmp_path: Path, capsys) -> None:
    normal = VideoManifestRecord(
        video_id="normal",
        path="Normal/normal.mp4",
        split="test",
        category="Normal",
        is_anomaly=False,
        num_frames=4,
        fps=1.0,
    )
    anomaly = VideoManifestRecord(
        video_id="anomaly",
        path="Abuse/anomaly.mp4",
        split="test",
        category="Abuse",
        is_anomaly=True,
        num_frames=4,
        fps=1.0,
        annotations=(
            SupervisionAnnotation(
                scope="frame",
                label="Abuse",
                is_anomaly=True,
                span=TemporalSpan(start=2, end=4, unit="frame"),
            ),
        ),
    )
    manifest_path = write_manifest_jsonl((normal, anomaly), tmp_path / "manifest.jsonl")
    predictions = [
        PredictionRecord("run", "normal", "normal@0", 0, 0, 2, 0.1, 0, 2),
        PredictionRecord("run", "normal", "normal@1", 1, 2, 4, 0.1, 2, 4),
        PredictionRecord("run", "anomaly", "anomaly@0", 0, 0, 2, 0.1, 0, 2),
        PredictionRecord("run", "anomaly", "anomaly@1", 1, 2, 4, 0.9, 2, 4),
    ]
    prediction_path = tmp_path / "predictions.jsonl"
    prediction_path.write_text(
        "".join(json.dumps(item.to_dict()) + "\n" for item in predictions),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "dataset": {"root": str(tmp_path), "test_manifest": str(manifest_path)},
        "encoder": {"adapter": "fake"},
        "task": {"supervision": "frame"},
        "output": {"root": str(tmp_path / "outputs"), "run_name": "eval"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "metrics.json"

    assert (
        main(
            [
                "evaluate",
                "-c",
                str(config_path),
                "--protocol",
                "subset",
                "--predictions",
                str(prediction_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["frame_auc"] == 1.0
    assert output.is_file()
    assert output.with_name("frame_scores.npz").is_file()
    assert payload["metrics"]["protocol"] == "ucf-crime/subset-frameauc-v1"
    assert (
        main(
            [
                "evaluate",
                "-c",
                str(config_path),
                "--predictions",
                str(prediction_path),
                "--output",
                str(tmp_path / "official.json"),
            ]
        )
        == 2
    )
    attempts = [json.loads(p.read_text()) for p in (tmp_path / "provenance/stages").glob("*.json")]
    assert attempts[-1]["status"] == "failed"
