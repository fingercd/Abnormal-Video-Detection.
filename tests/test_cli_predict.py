from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

import vadbench.cli as cli


def test_cli_predict_wires_strict_coverage(tmp_path: Path, monkeypatch, capsys) -> None:
    config = {
        "schema_version": 1,
        "dataset": {
            "root": str(tmp_path),
            "train_manifest": str(tmp_path / "train.jsonl"),
            "test_manifest": str(tmp_path / "test.jsonl"),
        },
        "encoder": {"adapter": "cached_features", "trainable": False},
        "task": {"kind": "weak_mil", "supervision": "video"},
        "output": {"root": str(tmp_path / "outputs"), "run_name": "predict"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    captured = {}

    def fake_predict(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [SimpleNamespace(video_id="video-a"), SimpleNamespace(video_id="video-a")]

    monkeypatch.setattr(cli, "predict_feature_head", fake_predict)
    output = tmp_path / "predictions.jsonl"
    code = cli.main(
        [
            "predict",
            "-c",
            str(config_path),
            "--features",
            str(tmp_path / "features"),
            "--checkpoint",
            str(tmp_path / "final.pt"),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["prediction_records"] == 2
    assert summary["videos"] == 1
    assert captured["kwargs"] == {"device": "cpu", "strict_coverage": True}
