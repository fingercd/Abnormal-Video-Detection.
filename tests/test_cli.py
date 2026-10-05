from __future__ import annotations

import json

from vadbench.cli import main


def test_cli_lists_pinned_weights(capsys) -> None:
    assert main(["weights", "list"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {item["id"] for item in payload} >= {
        "videomaev2-base-hf",
        "hermes-llava-ov-0.5b",
    }


def test_cli_validates_reference_config(capsys) -> None:
    assert main(["config", "validate", "configs/experiments/ucf_videomaev2_weak.yaml"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == 1


def test_cli_lists_encoder_cache_semantics(capsys) -> None:
    assert main(["encoders", "list"]) == 0
    payload = {item["id"]: item for item in json.loads(capsys.readouterr().out)}
    assert payload["videomaev2"]["capabilities"]["cache_kinds"] == []
    assert payload["hermes_llava_ov"]["capabilities"]["cache_kinds"] == ["decoder_kv"]


def test_cli_validates_manifest(tmp_path, capsys) -> None:
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(
        '{"schema_version":1,"video_id":"Normal001","path":"Normal/Normal001.mp4",'
        '"split":"test","category":"Normal","is_anomaly":false,"annotations":[]}\n',
        encoding="utf-8",
    )
    assert main(["manifest", "validate", str(manifest)]) == 0
    assert json.loads(capsys.readouterr().out)["videos"] == 1
