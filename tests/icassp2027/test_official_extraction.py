"""Controller wiring tests; native model/data acceptance is a separate run."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vadbench.checkpoints import sha256_file
from vadbench.contracts import EncoderCapabilities, EncoderOutput, TokenTimeline
from vadbench.data import official_training
from vadbench.data.manifest import VideoManifestRecord, load_manifest_jsonl
from vadbench.features import FeatureStore
from vadbench.paper.official_extraction import (
    OfficialDenseExtractionRequest,
    run_official_dense_extraction,
)


class _Capture:
    def __init__(self):
        self.position = 0

    def isOpened(self):  # noqa: N802
        return True

    def get(self, key):
        return {1: 64, 2: 8, 3: 2, 4: 2}[key]

    def set(self, _key, value):
        self.position = int(value)
        return True

    def read(self):
        value = np.full((2, 2, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, value

    def release(self):
        pass


class _CV:
    CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_POS_FRAMES = range(1, 6)

    def VideoCapture(self, _path):  # noqa: N802
        return _Capture()


class _Adapter:
    capabilities = EncoderCapabilities(supports_fixed_clip=True, fixed_num_frames=4)
    preprocess_profile = "fixture-rgb"
    pooling = "mean"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def encode(self, batch, train=False):
        assert not train
        assert all(value.startswith("sample-") for value in batch.video_ids)
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture extraction failure")
        value = np.asarray(batch.frame_indices, dtype=np.float32).mean(axis=1)
        pooled = value[:, None] + np.arange(768, dtype=np.float32)[None, :]
        return EncoderOutput(
            features=pooled[:, None, :],
            pooled=pooled,
            timeline=TokenTimeline(
                start_s=np.zeros((len(value), 1), dtype=np.float32),
                end_s=np.ones((len(value), 1), dtype=np.float32),
                valid_mask=np.ones((len(value), 1), dtype=bool),
            ),
        )


def _case(tmp_path, monkeypatch):
    records = []
    for number in range(2):
        video_id = f"source-{number}"
        (tmp_path / f"{video_id}.mp4").write_bytes(f"fixture-{number}".encode())
        records.append(VideoManifestRecord(
            video_id=video_id, path=f"{video_id}.mp4", split="train",
            category="Abuse" if number else "Normal", is_anomaly=bool(number),
            num_frames=64, fps=8.0,
            metadata={
                "original_role": "fit" if number == 0 else "select",
                "training_role": "official-fulltrain-final",
                "content_sha256": sha256_file(tmp_path / f"{video_id}.mp4"),
                "content_size_bytes": (tmp_path / f"{video_id}.mp4").stat().st_size,
            },
        ))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema": "icassp2027.official-detector-protocol/v2",
        "datasets": {"ucf_crime": {"encoders": ["videomaev2"]}},
    }))
    contract = tmp_path / "view-contract.json"
    contract.write_text("{}")
    role_lock = tmp_path / "role-lock.json"
    role_lock.write_text(json.dumps({"partitions": {
        "source-0": "fit", "source-1": "select"
    }}))
    calls = []

    def load(path, expected_sha256, **_kwargs):
        assert Path(path) == contract and expected_sha256 == sha256_file(contract)
        calls.append(str(path))
        return tuple(records), {
            "dataset": "ucf_crime",
            "dataset_root": str(tmp_path),
            "inputs": {
                "role_lock": {"path": role_lock.name, "sha256": sha256_file(role_lock)}
            },
        }

    monkeypatch.setattr(official_training, "load_official_training_view", load)
    request = OfficialDenseExtractionRequest(
        encoder="videomaev2", device="cpu", training_contract_path=str(contract),
        training_contract_sha256=sha256_file(contract), protocol_path=str(protocol),
        protocol_sha256=sha256_file(protocol), output_root=str(tmp_path / "runs"),
        run_id="engineering", engineering_video_ids=("source-0",),
    )
    adapter = _Adapter()
    identity = {
        "adapter": "videomaev2", "constructor": {"clip_frames": 4},
        "checkpoint": {"sha256": {"fixture.bin": "fixture"}},
    }
    return request, adapter, lambda _r: (adapter, {"identity": identity}), calls


def test_subset_real_feature_store_complete_and_bound(tmp_path, monkeypatch):
    request, adapter, factory, authority_calls = _case(tmp_path, monkeypatch)
    result = run_official_dense_extraction(request, adapter_factory=factory, video_backend=_CV())
    assert result["status"] == "completed" and len(authority_calls) == 2
    assert result["data_role"] == "engineering-subset-of-official-fulltrain"
    assert result["development_role"] is None
    contract_path = Path(result["contract"]["path"])
    assert result["contract"]["sha256"] == sha256_file(contract_path)
    contract = json.loads(contract_path.read_text())
    assert contract["data_role"] == "engineering-subset-of-official-fulltrain"
    assert contract["videos"] == 1 and contract["clips"] == 16 and adapter.calls == 2
    assert contract["coverage"]["input_union_complete"]
    assert contract["detector_training_executed"] is False
    assert contract["test_scoring_executed"] is False
    rows = tuple(FeatureStore(contract["feature_store"]["root"]).iter_records())
    assert len(rows) == 16 and {r.video_id for r in rows} == {"source-0"}
    assert all(r.arrays["pooled"].shape == (768,) for r in rows)
    manifest = load_manifest_jsonl(contract["training_manifest"]["path"])
    assert manifest[0].metadata["original_role"] == "fit"
    for key in ("resolved", "status", "index"):
        entry = contract["feature_store"][key]
        assert sha256_file(entry["path"]) == entry["sha256"]


def test_development_fit_extracts_exact_original_role_and_binds_role_lock(tmp_path, monkeypatch):
    request, adapter, factory, authority_calls = _case(tmp_path, monkeypatch)
    result = run_official_dense_extraction(
        replace(request, run_id="development-fit", engineering_video_ids=(), development_role="fit"),
        adapter_factory=factory,
        video_backend=_CV(),
    )
    assert result["status"] == "completed" and len(authority_calls) == 2
    assert result["data_role"] == "development-fit" and result["development_role"] == "fit"
    contract = json.loads(Path(result["contract"]["path"]).read_text())
    assert contract["data_role"] == "development-fit"
    assert contract["development_role"] == "fit"
    assert contract["engineering_video_ids"] == []
    assert contract["original_role_lock"]["sha256"] == sha256_file(
        contract["original_role_lock"]["path"]
    )
    manifest = load_manifest_jsonl(contract["training_manifest"]["path"])
    assert {row.metadata["original_role"] for row in manifest} == {"fit"}
    assert contract["videos"] == len(manifest) == 1


def test_development_select_extracts_exact_original_role(tmp_path, monkeypatch):
    request, _adapter, factory, _ = _case(tmp_path, monkeypatch)
    result = run_official_dense_extraction(
        replace(request, run_id="development-select", engineering_video_ids=(), development_role="select"),
        adapter_factory=factory,
        video_backend=_CV(),
    )
    contract = json.loads(Path(result["contract"]["path"]).read_text())
    manifest = load_manifest_jsonl(contract["training_manifest"]["path"])
    assert contract["data_role"] == "development-select"
    assert contract["development_role"] == "select"
    assert contract["videos"] == len(manifest) == 1
    assert manifest[0].video_id == "source-1"
    assert manifest[0].metadata["original_role"] == "select"


def test_development_role_rejects_engineering_ids_before_output(tmp_path, monkeypatch):
    request, _adapter, _factory, _ = _case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(request, development_role="fit")
    assert not Path(request.output_root).exists()


def test_development_role_rejects_confirm_before_authority_access(tmp_path, monkeypatch):
    request, _adapter, _factory, _ = _case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="must be fit, select, or null"):
        replace(request, engineering_video_ids=(), development_role="confirm")
    assert not Path(request.output_root).exists()


def test_protocol_tamper_rejects_before_model_or_output(tmp_path, monkeypatch):
    request, adapter, factory, _ = _case(tmp_path, monkeypatch)
    Path(request.protocol_path).write_text("{}")
    with pytest.raises(ValueError, match="SHA-256"):
        run_official_dense_extraction(request, adapter_factory=factory, video_backend=_CV())
    assert adapter.calls == 0 and not Path(request.output_root).exists()


def test_subset_outside_training_rejects_before_model(tmp_path, monkeypatch):
    request, adapter, factory, _ = _case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="outside"):
        run_official_dense_extraction(
            replace(request, engineering_video_ids=("test-video",)), adapter_factory=factory
        )
    assert adapter.calls == 0 and not Path(request.output_root).exists()


def test_failed_feature_stage_never_publishes_ready_contract(tmp_path, monkeypatch):
    request, adapter, factory, _ = _case(tmp_path, monkeypatch)
    adapter.fail = True
    result = run_official_dense_extraction(request, adapter_factory=factory, video_backend=_CV())
    assert result["status"] == "failed"
    assert not (Path(request.output_root) / request.run_id / "extraction-contract.json").exists()
    assert result["extraction"]["failures"][0]["type"] == "RuntimeError"


def test_training_authority_failure_cannot_construct_adapter(tmp_path, monkeypatch):
    request, adapter, factory, _ = _case(tmp_path, monkeypatch)

    def reject(*_args):
        raise ValueError("official identity audit failed")

    monkeypatch.setattr(official_training, "load_official_training_view", reject)
    with pytest.raises(ValueError, match="identity audit"):
        run_official_dense_extraction(request, adapter_factory=factory)
    assert adapter.calls == 0 and not Path(request.output_root).exists()


def test_different_raw_bytes_cannot_publish_training_contract(tmp_path, monkeypatch):
    request, _adapter, factory, _ = _case(tmp_path, monkeypatch)
    (tmp_path / "source-0.mp4").write_bytes(b"different-source-with-same-probed-shape")
    with pytest.raises(ValueError, match="independently audited"):
        run_official_dense_extraction(request, adapter_factory=factory, video_backend=_CV())
    assert not (Path(request.output_root) / request.run_id / "extraction-contract.json").exists()
