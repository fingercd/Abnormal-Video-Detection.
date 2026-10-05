"""CPU-only XD repeat-head and cached-test provenance tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vadbench.data.manifest import VideoManifestRecord, load_manifest_jsonl, write_manifest_jsonl
from vadbench.paper import evaluation, xd_evaluation
from vadbench.paper.compatibility import BackboneIdentity, RepresentationIdentity, SamplingIdentity
from vadbench.paper.extraction import make_data_content_evidence
from vadbench.paper.xd_head_repeats import _source_config
from vadbench.paper.xd_repeat_evaluation import _cache


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _identity() -> tuple[RepresentationIdentity, SamplingIdentity]:
    representation = RepresentationIdentity(
        BackboneIdentity("videomaev2", "weights", "code", {"frames": 16}, {"kind": "pooled"}),
        {"name": "paired_random", "seed": 0},
        4,
        "float32",
        {"kind": "native"},
    )
    sampling = SamplingIdentity(
        "source",
        "test_dense",
        {"clips": 3},
        {"frames": 16},
        {"frames": 1},
        {"kind": "strict"},
        {"kind": "frame"},
    )
    return representation, sampling


def test_xd_repeat_config_preserves_source_budget_and_changes_only_seed() -> None:
    representation, sampling = _identity()
    training = evaluation.TrainingIdentity(
        {"kind": "topk_mil", "k": 3}, "fit", 0, {"epochs": 20, "lr": 0.001}
    )
    source = SimpleNamespace(
        checkpoint_metadata={
            "config": {
                "head": "topk",
                "head_kwargs": {"k": 3},
                "expected_clips": 32,
                "epochs": 20,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            }
        },
        training_identity=training,
        representation=representation,
        sampling=sampling,
        encoder_fingerprint="sha256:" + "a" * 64,
    )
    config = _source_config(source, seed=2)
    assert config.seed == 2
    assert config.epochs == 20 and config.batch_size == 16
    assert config.declaration.training_identity.seed == 2


def test_xd_cached_test_store_is_seed0_bound_and_tamper_checked(
    tmp_path: Path, monkeypatch
) -> None:
    representation, sampling = _identity()
    test = write_manifest_jsonl(
        (
            VideoManifestRecord(
                "test", "test.mp4", "test", "Normal", False, num_frames=32, fps=24.0
            ),
        ),
        tmp_path / "test.jsonl",
    )
    raw = tmp_path / "test.mp4"
    raw.write_bytes(b"synthetic-raw-byte-identity")
    records = load_manifest_jsonl(test)
    evidence = make_data_content_evidence(records, dataset_root=tmp_path)
    coordinates = SimpleNamespace(
        manifest=records,
        raw_rows=[{"video_id": "test", "raw_sha256": _sha(raw), "raw_bytes": raw.stat().st_size}],
    )
    paths = {
        name: _write(tmp_path / f"{name}.json", {"name": name})
        for name in ("audit", "freeze", "scope", "original", "contract", "grid")
    }
    gt = tmp_path / "gt.npy"
    gt.write_bytes(b"fixture")
    root = tmp_path / "seed0"
    feature = root / "features" / "test"
    feature.mkdir(parents=True)
    index = feature / "index.jsonl"
    index.write_text("", encoding="utf-8")
    feature_resolved = _write(
        feature / "resolved.json",
        {
            "spec": {
                "representation": representation.to_dict(),
                "sampling": sampling.to_dict(),
                "sampling_kind": "dense",
            },
            "encoder_fingerprint": "sha256:" + "b" * 64,
            "data_content_evidence": evidence,
        },
    )
    status = _write(
        feature / "status.json",
        {
            "completed": True,
            "feature_root": str(feature),
            "encoder_fingerprint": "sha256:" + "b" * 64,
        },
    )
    source_receipt = {"run_dir": "source", "head_seed": 0}
    artifacts = {
        "test_feature_store": {
            "root": str(feature),
            **{
                name: {"path": str(path), "sha256": _sha(path)}
                for name, path in (
                    ("resolved", feature_resolved),
                    ("status", status),
                    ("index", index),
                )
            },
        }
    }
    resolved = _write(
        root / "resolved.json",
        {
            "method_source": source_receipt,
            "sealed_test_manifest_sha256": _sha(test),
            "sealed_audit_sha256": _sha(paths["audit"]),
            "evaluation_representation": representation.to_dict(),
            "evaluation_sampling": sampling.to_dict(),
            "evaluation_encoder_fingerprint": "sha256:" + "b" * 64,
            "artifacts": artifacts,
            "coverage": {"complete": True},
        },
    )
    _write(
        root / "result.json",
        {
            "status": "completed",
            "protocol": xd_evaluation.PROTOCOL_ID,
            "resolved": {"path": str(resolved), "sha256": _sha(resolved)},
            "artifacts": artifacts,
        },
    )
    config = {
        "encoder": "videomaev2",
        "device": "cpu",
        "dataset_root": str(tmp_path),
        "test_manifest": str(test),
        "audit_report": str(paths["audit"]),
        "method_source_run": str(tmp_path / "controller"),
        "output_root": str(root),
        "canonical_metadata": str(paths["grid"]),
        "ground_truth": str(gt),
        "raw_coordinate_receipt_sha256": "a" * 64,
        "head_data_contract_sha256": _sha(paths["contract"]),
        "original_role_lock_path": str(paths["original"]),
        "scope_path": str(paths["scope"]),
        "freeze_path": str(paths["freeze"]),
        "role_lock_path": str(paths["scope"]),
        "head_data_contract_path": str(paths["contract"]),
        "source_manifest_root": str(tmp_path),
    }
    inputs = {
        name: {"location": str(path), "sha256": _sha(path)}
        for name, path in {
            "test_manifest": test,
            "audit_report": paths["audit"],
            "method_freeze": paths["freeze"],
            "xd_scope": paths["scope"],
            "xd_canonical_metadata": paths["grid"],
            "xd_gt": gt,
            "xd_original_role_lock": paths["original"],
            "training_role_lock": paths["scope"],
            "head_data_contract": paths["contract"],
        }.items()
    }
    _write(
        root / "provenance" / "stages" / "stage.json",
        {
            "stage": "frozen_xd_evaluation",
            "status": "completed",
            "config": config,
            "inputs": inputs,
        },
    )
    source = SimpleNamespace()
    monkeypatch.setattr(xd_evaluation, "verify_raw_coordinates", lambda request: coordinates)
    monkeypatch.setattr(
        xd_evaluation, "load_frozen_xd_detector_source", lambda *args, **kwargs: source
    )
    monkeypatch.setattr(evaluation, "_source_receipt", lambda value: source_receipt)
    monkeypatch.setattr(evaluation, "_load_freeze", lambda value: ({}, "freeze"))
    cached = _cache(root, expected_encoder="videomaev2")
    assert cached[2]["root"] == str(feature)
    index.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="FeatureStore binding"):
        _cache(root, expected_encoder="videomaev2")


def test_xd_repeat_clis_expose_only_cpu_frozen_inputs() -> None:
    root = Path(__file__).resolve().parents[2]
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(root / "src")}
    for name in ("repeat_frozen_xd_head.py", "evaluate_frozen_xd_head_repeat.py"):
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "icassp2027" / name), "--help"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--device" in result.stdout
