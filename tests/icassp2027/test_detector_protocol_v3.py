"""official-detector-protocol v3: the XD videomae scope revision (2026-09-20).

Pins the real new protocol version: the only substantive difference versus v2
is ``datasets.xd_violence.encoders`` gaining ``videomae`` (user directive,
all three XD encoders). v2 stays byte-identical in place; consumers switch by
passing the v3 path + SHA.
"""

from __future__ import annotations

import json
from pathlib import Path

from vadbench.checkpoints import sha256_file

DECISIONS = Path(__file__).resolve().parents[2] / "projects" / "icassp2027" / "decisions"
V2 = DECISIONS / "official-detector-protocol-v2.json"
V3 = DECISIONS / "official-detector-protocol-v3.json"

V3_SHA256 = "b85bed6ca2c416c80f0582ca84d1f7aa61c86a836126fd3916e13121988677a9"


def test_v3_is_the_only_xd_videomae_scope_revision() -> None:
    assert V2.is_file() and V3.is_file()
    assert sha256_file(V3) == V3_SHA256
    old = json.loads(V2.read_text(encoding="utf-8"))
    new = json.loads(V3.read_text(encoding="utf-8"))

    # The schema field intentionally stays v2 so every existing validator
    # (official_extraction / urdmu_training protocol gates) applies unchanged.
    assert new["schema"] == "icassp2027.official-detector-protocol/v2"
    assert old["schema"] == new["schema"]

    top_diffs = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    assert top_diffs == {"datasets", "decision_date", "revision_note"}
    assert "v3" in new["revision_note"] and "videomae" in new["revision_note"]

    # UCF is untouched; XD differs only in the encoders list.
    assert new["datasets"]["ucf_crime"] == old["datasets"]["ucf_crime"]
    xd_diffs = {
        key
        for key in set(old["datasets"]["xd_violence"]) | set(new["datasets"]["xd_violence"])
        if old["datasets"]["xd_violence"].get(key) != new["datasets"]["xd_violence"].get(key)
    }
    assert xd_diffs == {"encoders"}
    assert new["datasets"]["xd_violence"]["encoders"] == ["videomaev2", "timesformer", "videomae"]
    assert old["datasets"]["xd_violence"]["encoders"] == ["videomaev2", "timesformer"]
    assert new["author_backend"] == old["author_backend"]

    # Consumer gate simulation: both existing predicates accept v3 for XD
    # videomae and keep every other (dataset, encoder) pair identical to v2.
    def gate(document: dict, dataset: str, encoder: str) -> bool:
        return (
            document.get("schema") == "icassp2027.official-detector-protocol/v2"
            and encoder in document["datasets"][dataset]["encoders"]
        )

    assert gate(new, "xd_violence", "videomae") is True
    assert gate(old, "xd_violence", "videomae") is False
    for dataset in ("ucf_crime", "xd_violence"):
        for encoder in ("videomaev2", "timesformer", "vjepa2", "videomae"):
            assert gate(new, dataset, encoder) == gate(old, dataset, encoder) or (
                dataset == "xd_violence" and encoder == "videomae"
            )
