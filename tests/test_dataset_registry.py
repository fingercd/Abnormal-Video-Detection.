from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml


def test_ucf_crime_registry_pins_official_protocol_files() -> None:
    registry = yaml.safe_load(Path("registry/datasets.yaml").read_text(encoding="utf-8"))
    spec = registry["datasets"]["ucf-crime"]
    assert re.fullmatch(r"[0-9a-f]{40}", spec["source"]["commit"])
    assert spec["expected_split"]["train"] == {
        "total": 1610,
        "normal": 800,
        "anomaly": 810,
    }
    assert spec["expected_split"]["test"] == {
        "total": 290,
        "normal": 150,
        "anomaly": 140,
    }
    for file_spec in spec["files"].values():
        assert re.fullmatch(r"[0-9a-f]{64}", file_spec["sha256"])
        assert Path(file_spec["local_path"]).is_relative_to("data/splits/ucf_crime")
        source = Path(file_spec["local_path"])
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == file_spec["sha256"]
    assert spec["coordinates"]["conversion"] == "[raw_start - 1, raw_end)"
