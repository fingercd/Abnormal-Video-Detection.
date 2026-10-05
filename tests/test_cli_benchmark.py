from __future__ import annotations

import json
from pathlib import Path

import yaml

import vadbench.cli as cli


def test_cli_benchmark_forwards_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    plan = {
        "benchmark": {"output": str(tmp_path / "default.json")},
        "input": {},
        "cases": [],
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "comparison": {"comparable": False, "reasons": ["different adapters"]},
            "cases": [
                {
                    "name": "fake",
                    "mode": "fixed",
                    "aggregate": {"throughput": {}},
                    "accuracy_eligibility": {"eligible": True},
                }
            ],
        }

    monkeypatch.setattr(cli, "run_benchmark_plan", fake_run)
    output = tmp_path / "result.json"
    code = cli.main(
        [
            "benchmark",
            "-c",
            str(plan_path),
            "--video",
            str(tmp_path / "video.mp4"),
            "--device",
            "cuda:2",
            "--warmup",
            "2",
            "--repeat",
            "3",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert captured["kwargs"]["device"] == "cuda:2"
    assert captured["kwargs"]["warmup"] == 2
    assert captured["kwargs"]["repeat_count"] == 3
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"] == str(output.resolve())
    assert summary["comparison"]["comparable"] is False
