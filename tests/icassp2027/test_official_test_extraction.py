"""Sealed-test dense extraction: evaluation-only identity, leak guards, resume."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_official_extraction import _CV, _Adapter

from vadbench.checkpoints import sha256_file
from vadbench.data import official_training
from vadbench.data.manifest import (
    VideoManifestRecord,
    load_manifest_jsonl,
    write_manifest_jsonl,
)
from vadbench.features import FeatureStore
from vadbench.paper.official_extraction import (
    OfficialDenseExtractionRequest,
    run_official_dense_extraction,
)


def _case(tmp_path, monkeypatch, *, with_content_evidence: bool):
    train_records = []
    for number in range(2):
        video_id = f"source-{number}"
        (tmp_path / f"{video_id}.mp4").write_bytes(f"fixture-{number}".encode())
        train_records.append(
            VideoManifestRecord(
                video_id=video_id,
                path=f"{video_id}.mp4",
                split="train",
                category="Abuse" if number else "Normal",
                is_anomaly=bool(number),
                num_frames=64,
                fps=8.0,
                metadata={
                    "original_role": "fit" if number == 0 else "select",
                    "training_role": "official-fulltrain-final",
                    "content_sha256": sha256_file(tmp_path / f"{video_id}.mp4"),
                    "content_size_bytes": (tmp_path / f"{video_id}.mp4").stat().st_size,
                },
            )
        )
    # Test videos live in their own tree; the training root must never be
    # assumed to contain them.
    test_root = tmp_path / "test-root"
    test_root.mkdir()
    test_records = []
    for number in range(2):
        video_id = f"sample-test-{number}"
        path = test_root / f"{video_id}.mp4"
        path.write_bytes(f"sealed-test-{number}".encode())
        metadata = {}
        if with_content_evidence:
            metadata = {
                "content_sha256": sha256_file(path),
                "content_size_bytes": path.stat().st_size,
            }
        test_records.append(
            VideoManifestRecord(
                video_id=video_id,
                path=f"{video_id}.mp4",
                split="test",
                category="Abuse" if number else "Normal",
                is_anomaly=bool(number),
                num_frames=64,
                fps=8.0,
                metadata=metadata,
            )
        )
    test_manifest = write_manifest_jsonl(test_records, tmp_path / "sealed-test.jsonl")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": "icassp2027.official-detector-protocol/v2",
                "datasets": {"ucf_crime": {"encoders": ["videomaev2"]}},
            }
        )
    )
    contract = tmp_path / "view-contract.json"
    contract.write_text("{}")
    calls = []

    def load(path, expected_sha256, **_kwargs):
        assert Path(path) == contract and expected_sha256 == sha256_file(contract)
        calls.append(str(path))
        return tuple(train_records), {
            "dataset": "ucf_crime",
            "dataset_root": str(tmp_path),
            "inputs": {},
        }

    monkeypatch.setattr(official_training, "load_official_training_view", load)
    identity = {
        "adapter": "videomaev2",
        "constructor": {"clip_frames": 4},
        "checkpoint": {"sha256": {"fixture.bin": "fixture"}},
    }
    def factory(_request):
        return _Adapter(), {"identity": identity}
    return train_records, test_records, test_manifest, protocol, contract, factory, calls


def _request(tmp_path, contract, protocol, test_manifest, **overrides):
    values = {
        "encoder": "videomaev2",
        "device": "cpu",
        "training_contract_path": str(contract),
        "training_contract_sha256": sha256_file(contract),
        "protocol_path": str(protocol),
        "protocol_sha256": sha256_file(protocol),
        "output_root": str(tmp_path / "runs"),
        "run_id": "sealed-test",
        "test_manifest_path": str(test_manifest),
        "test_manifest_sha256": sha256_file(test_manifest),
        "test_dataset_root": str(tmp_path / "test-root"),
    }
    values.update(overrides)
    return OfficialDenseExtractionRequest(**values)


def test_sealed_test_extraction_publishes_evaluation_only_contract(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, with_content_evidence=True)
    _train, test_records, test_manifest, protocol, contract, _factory, _calls = case
    adapter = _Adapter()
    def factory(_request):
        return adapter, {"identity": {
            "adapter": "videomaev2", "constructor": {"clip_frames": 4},
            "checkpoint": {"sha256": {"fixture.bin": "fixture"}},
        }}
    result = run_official_dense_extraction(
        _request(tmp_path, contract, protocol, test_manifest),
        adapter_factory=factory,
        video_backend=_CV(),
    )
    assert result["status"] == "completed"
    assert result["data_role"] == "official-sealed-test"
    assert result["test_only"] is True
    document = json.loads(Path(result["contract"]["path"]).read_text())
    assert document["data_role"] == "official-sealed-test"
    assert document["test_only"] is True
    assert document["detector_training_executed"] is False
    assert document["test_scoring_executed"] is False
    assert document["content_cross_check"] == "performed"
    assert document["sealed_test_manifest"]["sha256"] == sha256_file(test_manifest)
    assert "never" in document["leakage_note"]
    assert "evaluation only" in document["training_sampling_note"]
    frozen = load_manifest_jsonl(document["training_manifest"]["path"])
    assert [row.video_id for row in frozen] == [row.video_id for row in test_records]
    assert all(row.split.value == "test" for row in frozen)
    assert document["training_manifest"]["path"].endswith("test.jsonl")
    rows = tuple(FeatureStore(document["feature_store"]["root"]).iter_records())
    assert {row.video_id for row in rows} == {"sample-test-0", "sample-test-1"}
    assert adapter.calls > 0


def test_sealed_test_extraction_without_content_hashes_is_explicit(tmp_path, monkeypatch):
    _train, _test, test_manifest, protocol, contract, factory, _calls = _case(
        tmp_path, monkeypatch, with_content_evidence=False
    )
    result = run_official_dense_extraction(
        _request(tmp_path, contract, protocol, test_manifest),
        adapter_factory=factory,
        video_backend=_CV(),
    )
    assert result["status"] == "completed"
    document = json.loads(Path(result["contract"]["path"]).read_text())
    assert document["content_cross_check"] == "unavailable-in-sealed-test-manifest"


def test_sealed_test_overlapping_training_view_is_a_hard_leak_failure(tmp_path, monkeypatch):
    train_records, _test, _manifest, protocol, contract, factory, _calls = _case(
        tmp_path, monkeypatch, with_content_evidence=True
    )
    # A training-view video relabelled as test is the exact mixed-identity
    # leak the sealed test mode must refuse.
    stolen = replace(train_records[0], split="test", metadata={})
    overlapped = [stolen, *_test_records(tmp_path)]
    bad_manifest = write_manifest_jsonl(overlapped, tmp_path / "overlapped.jsonl")
    # The train/test leak guard fires before any model or output exists.
    with pytest.raises(ValueError, match="切分泄漏"):
        run_official_dense_extraction(
            _request(tmp_path, contract, protocol, bad_manifest),
            adapter_factory=factory,
        )


def _test_records(tmp_path):
    return [
        VideoManifestRecord(
            video_id="sample-test-0",
            path="sample-test-0.mp4",
            split="test",
            category="Normal",
            is_anomaly=False,
            num_frames=64,
            fps=8.0,
        )
    ]


def test_sealed_test_manifest_with_train_record_rejected(tmp_path, monkeypatch):
    train_records, _test, _manifest, protocol, contract, factory, _calls = _case(
        tmp_path, monkeypatch, with_content_evidence=True
    )
    mixed = [*train_records[:1], *_test_records(tmp_path)]
    bad_manifest = write_manifest_jsonl(mixed, tmp_path / "mixed-split.jsonl")
    with pytest.raises(ValueError, match="非 test 记录"):
        run_official_dense_extraction(
            _request(tmp_path, contract, protocol, bad_manifest),
            adapter_factory=factory,
        )


def test_test_mode_rejects_role_subset_and_partial_bindings(tmp_path, monkeypatch):
    _train, _test, test_manifest, protocol, contract, _factory, _calls = _case(
        tmp_path, monkeypatch, with_content_evidence=True
    )
    base = _request(tmp_path, contract, protocol, test_manifest)
    with pytest.raises(ValueError, match="provided together"):
        replace(base, test_manifest_sha256=None)
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(base, development_role="fit")
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(base, engineering_video_ids=("source-0",))
    with pytest.raises(ValueError, match="only meaningful with a sealed test manifest"):
        OfficialDenseExtractionRequest(
            encoder="videomaev2",
            device="cpu",
            training_contract_path=str(contract),
            training_contract_sha256=sha256_file(contract),
            protocol_path=str(protocol),
            protocol_sha256=sha256_file(protocol),
            output_root=str(tmp_path / "runs"),
            test_dataset_root=str(tmp_path),
        )


def test_sealed_test_extraction_resumes_completed_test_run(tmp_path, monkeypatch):
    _train, _test, test_manifest, protocol, contract, factory, _calls = _case(
        tmp_path, monkeypatch, with_content_evidence=True
    )
    first = run_official_dense_extraction(
        _request(tmp_path, contract, protocol, test_manifest, run_id="test-a"),
        adapter_factory=factory,
        video_backend=_CV(),
    )
    assert first["status"] == "completed"
    resumed_adapter = _Adapter()
    def resumed_factory(_request):
        return resumed_adapter, {"identity": {
            "adapter": "videomaev2", "constructor": {"clip_frames": 4},
            "checkpoint": {"sha256": {"fixture.bin": "fixture"}},
        }}
    first_root = json.loads(Path(first["contract"]["path"]).read_text())["feature_store"]["root"]
    second = run_official_dense_extraction(
        _request(
            tmp_path,
            contract,
            protocol,
            test_manifest,
            run_id="test-b",
            resume_source=first_root,
        ),
        adapter_factory=resumed_factory,
        video_backend=_CV(),
    )
    assert second["status"] == "completed"
    assert second["data_role"] == "official-sealed-test"
    assert resumed_adapter.calls == 0
