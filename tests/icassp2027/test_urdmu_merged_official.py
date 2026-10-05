"""Formal/development UR-DMU training over merged feature views.

A merged/subset view may back formal/development training only through the
narrow adapter: bound view contract, identity identical to the native dense
extraction contract, all-identical duplicate audit, exact target list, and
per-file SHA bindings. These tests pin acceptance and every rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_feature_merge import _merge
from test_urdmu_v0 import _run768

from vadbench.checkpoints import sha256_file
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.paper import urdmu_training as training
from vadbench.paper.urdmu_training import URDMUTrainingRequest


def _records(tmp_path: Path, video_ids: list[str]) -> list[VideoManifestRecord]:
    records = []
    for index, video_id in enumerate(video_ids):
        path = tmp_path / f"{video_id}.mp4"
        path.touch(exist_ok=True)
        records.append(
            VideoManifestRecord(
                video_id=video_id,
                path=path.name,
                split="train",
                category="Anomaly" if index % 2 else "Normal",
                is_anomaly=index % 2 == 1,
                num_frames=64,
                fps=8.0,
                duration_seconds=8.0,
                metadata={
                    "original_role": "fit" if index % 2 == 0 else "select",
                    "content_sha256": sha256_file(path),
                    "content_size_bytes": path.stat().st_size,
                },
            )
        )
    return records


@pytest.fixture
def merged_case(tmp_path, monkeypatch):
    """Native 768-dim run, a merged view over it, and bound native contract."""
    native = _run768(tmp_path, "native-run", ["sample-a", "sample-b"])
    assert native.completed
    merged = _merge(tmp_path, (native,), ["sample-a", "sample-b"], run_id="merged-view")
    merged_root = Path(merged["run_dir"])
    records = _records(tmp_path, ["sample-a", "sample-b"])
    manifest = write_manifest_jsonl(records, tmp_path / "train.jsonl")
    source_contract = tmp_path / "view-contract.json"
    source_contract.write_text("{}", encoding="utf-8")
    protocol_path = Path("projects/icassp2027/decisions/official-detector-protocol-v2.json").resolve()
    native_root = Path(native.run_dir)
    extraction_contract = tmp_path / "extraction-contract.json"
    extraction_contract.write_text(
        json.dumps(
            {
                "schema": "icassp2027.official-dense-extraction/v1",
                "status": "ready",
                "data_role": "official-fulltrain-final",
                "dataset": "ucf_crime",
                "training_view_contract": {
                    "path": str(source_contract.resolve()),
                    "sha256": sha256_file(source_contract),
                },
                "training_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
                "feature_store": {
                    "root": str(native_root),
                    **{
                        name: {"path": str(native_root / filename), "sha256": sha256_file(native_root / filename)}
                        for name, filename in (
                            ("resolved", "resolved.json"),
                            ("status", "status.json"),
                            ("index", "index.jsonl"),
                        )
                    },
                },
                "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            }
        ),
        encoding="utf-8",
    )

    def _request(**overrides):
        values = {
            "dataset": "ucf_crime",
            "feature_store": str(merged_root),
            "train_manifest": str(manifest),
            "source_contract_path": str(source_contract),
            "source_contract_sha256": sha256_file(source_contract),
            "upstream_dir": "not-accessed",
            "output_root": str(tmp_path / "runs"),
            "device": "cpu",
            "run_mode": "formal",
            "steps": 3000,
            "bags_per_class": 64,
            "extraction_contract_path": str(extraction_contract),
            "extraction_contract_sha256": sha256_file(extraction_contract),
            "feature_view_contract_path": str(merged_root / "merge-contract.json"),
            "feature_view_contract_sha256": sha256_file(merged_root / "merge-contract.json"),
        }
        values.update(overrides)
        return URDMUTrainingRequest(**values)

    return _request, records, merged_root, extraction_contract


def test_merged_view_backs_formal_dense_source(merged_case):
    _request, records, merged_root, _contract = merged_case
    document, representation, sampling, receipt, sequences = training._dense_source(
        _request(), records, {}, None
    )
    assert document["merged_view"] is True and document["merged_view_digest"]
    assert representation.output_dim == 768
    assert sampling.regime == "test_dense"
    assert receipt["merged_view"] is True
    assert len(sequences) == 2
    assert {sequence["video_id"] for sequence in sequences} == {"sample-a", "sample-b"}
    assert receipt["feature_files"]["index"]["path"].endswith("index.jsonl")


def test_merged_view_requires_bound_view_contract(merged_case):
    _request, records, _root, _contract = merged_case
    with pytest.raises(ValueError, match="externally bound feature view contract"):
        training._dense_source(
            _request(feature_view_contract_path=None, feature_view_contract_sha256=None),
            records,
            {},
            None,
        )


def test_merged_view_rejects_partial_target_list(merged_case):
    _request, records, _root, _contract = merged_case
    with pytest.raises(ValueError, match="target list differs"):
        training._dense_source(_request(), records[:1], {}, None)


def test_merged_view_rejects_nonidentical_duplicate_audit(merged_case, tmp_path):
    _request, records, merged_root, _contract = merged_case
    contract_path = merged_root / "merge-contract.json"
    document = json.loads(contract_path.read_text())
    document["duplicate_audit"] = {
        "sample-a": {"verdict": "duplicate_divergent", "kept_run": "x", "dropped_run": "y"}
    }
    contract_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="non-identical duplicate verdict"):
        training._dense_source(
            _request(feature_view_contract_sha256=sha256_file(contract_path)),
            records,
            {},
            None,
        )


def test_merged_view_rejects_identity_drift(merged_case):
    _request, records, merged_root, _contract = merged_case
    contract_path = merged_root / "merge-contract.json"
    document = json.loads(contract_path.read_text())
    document["identity"]["spec"]["representation"]["reducer"] = {"name": "global_uniform"}
    contract_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="identity .* differs from the bound dense extraction"):
        training._dense_source(
            _request(feature_view_contract_sha256=sha256_file(contract_path)),
            records,
            {},
            None,
        )


def test_merged_view_rejects_unbound_native_store(merged_case, tmp_path):
    _request, records, _root, extraction_contract = merged_case
    document = json.loads(extraction_contract.read_text())
    document["feature_store"]["index"]["sha256"] = "0" * 64
    extraction_contract.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        training._dense_source(
            _request(extraction_contract_sha256=sha256_file(extraction_contract)),
            records,
            {},
            None,
        )
