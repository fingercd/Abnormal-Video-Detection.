from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

import vadbench.cli as cli
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl


def test_cli_train_passes_max_steps_to_runner(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest = write_manifest_jsonl(
        (
            VideoManifestRecord(
                video_id="normal",
                path="Normal/normal.mp4",
                split="train",
                category="Normal",
                is_anomaly=False,
            ),
        ),
        tmp_path / "train.jsonl",
    )
    config = {
        "schema_version": 1,
        "dataset": {
            "root": str(tmp_path),
            "train_manifest": str(manifest),
            "test_manifest": str(manifest),
        },
        "encoder": {"adapter": "fake", "trainable": False},
        "task": {"kind": "weak_mil", "supervision": "video"},
        "output": {"root": str(tmp_path / "outputs"), "run_name": "train"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    captured = {}

    def fake_train(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"global_step": 1})

    monkeypatch.setattr(cli, "train_feature_head", fake_train)
    assert (
        cli.main(
            [
                "train",
                "-c",
                str(config_path),
                "--features",
                str(tmp_path / "features"),
                "--max-steps",
                "1",
            ]
        )
        == 0
    )
    assert captured["config"]["training"]["max_steps"] == 1
    assert json.loads(capsys.readouterr().out)["global_step"] == 1
