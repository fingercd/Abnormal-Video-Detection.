"""UR-DMU scoring contracts use synthetic dense vectors, never research scores."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from vadbench.checkpoints import sha256_file
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.features import FeatureStore, compute_encoder_fingerprint
from vadbench.paper import urdmu_evaluation as evaluation
from vadbench.paper.compatibility import (
    BackboneIdentity,
    RepresentationIdentity,
    feature_cache_key,
)
from vadbench.paper.extraction import (
    _semantic_runtime_identity,
    _verified_code_digest,
    _verified_weights_digest,
    make_data_content_evidence,
    make_sampling_identity,
)
from vadbench.paper.urdmu_backend import UPSTREAM_COMMIT
from vadbench.paper.urdmu_evaluation import (
    URDMUEvaluationRequest,
    _attention_workspace_lower_bound_bytes,
    _query_chunk_receipt,
    run_urdmu_evaluation,
)
from vadbench.paper.urdmu_training import COMPONENTS


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    return path


def _source(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, tuple[VideoManifestRecord, ...]]:
    raw = tmp_path / "raw"
    raw.mkdir()
    records = []
    for index, anomaly in enumerate((False, True)):
        video_id = f"select-{index}"
        path = raw / f"{video_id}.mp4"
        path.write_bytes(f"synthetic {video_id}".encode())
        records.append(
            VideoManifestRecord(
                video_id=video_id,
                path=path.name,
                split="train",
                category="Anomaly" if anomaly else "Normal",
                is_anomaly=anomaly,
                num_frames=32,
                fps=8.0,
                duration_seconds=4.0,
                metadata={
                    "original_role": "select",
                    "content_sha256": sha256_file(path),
                    "content_size_bytes": path.stat().st_size,
                },
            )
        )
    manifest = write_manifest_jsonl(records, tmp_path / "select.jsonl")
    fit_path = raw / "fit-0.mp4"
    fit_path.write_bytes(b"synthetic fit")
    fit = VideoManifestRecord(
        video_id="fit-0",
        path=fit_path.name,
        split="train",
        category="Normal",
        is_anomaly=False,
        num_frames=32,
        fps=8.0,
        duration_seconds=4.0,
        metadata={
            "original_role": "fit",
            "content_sha256": sha256_file(fit_path),
            "content_size_bytes": fit_path.stat().st_size,
        },
    )
    authority_records = (fit, *records)
    fit_manifest = write_manifest_jsonl((fit,), tmp_path / "fit.jsonl")
    verified = {
        "adapter": "toy",
        "constructor": {"clip_frames": 4},
        "checkpoint": {"sha256": {"fixture": "a" * 64}},
    }
    runtime = {
        "adapter_type": "fixture.Adapter",
        "adapter_library_version": None,
        "encoder_type": "fixture.Encoder",
        "encoder_library_version": None,
        "capabilities": {"fixed_num_frames": 4, "supports_fixed_clip": True},
        "properties": {"pooling": "mean"},
        "runtime_configurations": {"model_config": {"hidden_size": 768}},
        "implementation_files": {"model": {"role": "model", "module": "fixture", "path": str(tmp_path / "native.py"), "sha256": "b" * 64}},
        "loaded_library_versions": {"torch": str(torch.__version__)},
    }
    representation = RepresentationIdentity(
        BackboneIdentity(
            "toy",
            _verified_weights_digest(verified),
            _verified_code_digest(verified, runtime),
            {"profile": "fixture"},
            {"kind": "mean"},
        ),
        {"name": "identity"},
        768,
        "float32",
        {"kind": "native"},
    )
    sampling = make_sampling_identity(
        records,
        dataset_root=raw,
        sampling_kind="dense",
        clip_frames=4,
        frame_stride=2,
        window_stride=4,
    )
    runtime.update(
        runtime_id="toy",
        representation_fingerprint=representation.fingerprint,
        sampling_fingerprint=sampling.fingerprint,
        source_data_digest=sampling.source_digest,
        declared_readout=dict(representation.backbone.readout),
        verified_encoder_identity=verified,
    )
    fingerprint = compute_encoder_fingerprint({"paper_pooled_cache": _semantic_runtime_identity(runtime)})
    paper_identity = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    features = tmp_path / "features"
    store = FeatureStore(features)
    for record in records:
        for sample in DenseSamplingPlan(4, 2, 4).sample(record.num_frames):
            vector = np.full(768, float(record.is_anomaly) + sample.clip_index / 100, dtype=np.float32)
            store.write(
                video_id=record.video_id,
                clip_id=f"{record.video_id}:{sample.clip_index}",
                clip_index=sample.clip_index,
                encoder_fingerprint=fingerprint,
                features=vector[None, :],
                pooled=vector,
                start_s=sample.score_frame_start / record.fps,
                end_s=sample.score_frame_end / record.fps,
                frame_start=sample.score_frame_start,
                frame_end=sample.score_frame_end,
                metadata={"paper_identity": paper_identity, "sampling": {"kind": "dense", "clip_index": sample.clip_index}},
            )
    _write(features / "resolved.json", {"spec": {"representation": representation.to_dict(), "sampling": sampling.to_dict(), "sampling_kind": "dense"}, "runtime": runtime, "data_content_evidence": make_data_content_evidence(records, dataset_root=raw), "encoder_fingerprint": fingerprint, "paper_identity": paper_identity})
    count = len(list(store.iter_records()))
    _write(features / "status.json", {"status": "completed", "completed": True, "failures": [], "feature_root": str(features.resolve()), "encoder_fingerprint": fingerprint, "records_written_to_shards": count})
    role_lock = _write(
        tmp_path / "roles.json",
        {"partitions": {record.video_id: record.metadata["original_role"] for record in authority_records}},
    )
    original_role_lock = {"path": str(role_lock.resolve()), "sha256": sha256_file(role_lock)}
    authority_contract = _write(tmp_path / "official-training-contract.json", {"fixture": True})
    contract = tmp_path / "development-select-contract.json"
    _write(contract, {"schema": "icassp2027.official-dense-extraction/v1", "status": "ready", "data_role": "development-select", "development_role": "select", "dataset": "ucf_crime", "dataset_root": str(raw.resolve()), "training_view_contract": {"path": str(authority_contract.resolve()), "sha256": sha256_file(authority_contract)}, "original_role_lock": original_role_lock, "training_manifest": {"path": str(manifest.resolve()), "sha256": sha256_file(manifest)}, "feature_store": {"root": str(features.resolve()), **{name: {"path": str((features / filename).resolve()), "sha256": sha256_file(features / filename)} for name, filename in (("resolved", "resolved.json"), ("status", "status.json"), ("index", "index.jsonl"))}}})

    run = tmp_path / "head"
    checkpoint = run / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True)
    metadata = {
        "schema": "urdmu.training/v1",
        "status": "completed_training",
        "run_mode": "development",
        "data_role": "development-fit",
        "development_role": "fit",
        "dataset": "ucf_crime",
        "checkpoint_role": "dense_reference",
        "representation": representation.to_dict(),
        "representation_fingerprint": representation.fingerprint,
        "sampling": sampling.to_dict(),
        "sampling_fingerprint": sampling.fingerprint,
        "encoder_fingerprint": fingerprint,
        "source_training_view": {
            "development_role_lock": original_role_lock,
            "manifest_sha256": "sha256:fixture-fit-manifest",
        },
        "source_sha256": {
            str(authority_contract.resolve()): sha256_file(authority_contract),
            str(fit_manifest.resolve()): sha256_file(fit_manifest),
        },
        "backend": {"upstream_commit": UPSTREAM_COMMIT},
        "checkpoint_selection": "fixed_final_step_no_test_selection",
        "steps": 3000,
        "bags_per_class": 64,
        "seed": 0,
        "nonzero_gradient_steps": 3000,
    }
    torch.save({"model_state_dict": {"weight": torch.ones(1)}, "metadata": metadata}, checkpoint)
    digest = sha256_file(checkpoint)
    _write(checkpoint.with_suffix(".pt.json"), {"sha256": digest, "size_bytes": checkpoint.stat().st_size})
    _write(run / "training_qa.json", {"status": "passed", "checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest, "step": 3000}, "nonzero_gradient_steps": 3000, "changed_parameters_by_component": {component: 1 for component in COMPONENTS}, "reload_parity": {"exact_equal": True, "flag": "Test", "training": False}})
    _write(run / "result.json", {"status": "completed", "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": digest})
    _write(
        run / "provenance" / "stages" / "training.json",
        {
            "stage": "urdmu_training",
            "status": "completed",
            "config": {"run_mode": "development", "train_manifest": str(fit_manifest.resolve())},
            "inputs": {
                "train_manifest": {
                    "location": str(fit_manifest.resolve()),
                    "sha256": sha256_file(fit_manifest),
                }
            },
        },
    )
    return run, features, manifest, contract, authority_contract, authority_records


def _patch_authority(monkeypatch, authority_contract: Path, records, role_lock: Path) -> None:
    from vadbench.data import official_training

    expected = sha256_file(authority_contract)

    def load(path, digest, *, dataset_root=None):
        assert Path(path).resolve() == authority_contract.resolve() and digest == expected
        return tuple(records), {
            "dataset": "ucf_crime",
            "contract_path": str(authority_contract.resolve()),
            "contract_sha256": expected,
            "inputs": {"role_lock": {"path": role_lock.name, "sha256": sha256_file(role_lock)}},
        }

    monkeypatch.setattr(official_training, "load_official_training_view", load)


def _request(tmp_path: Path, run: Path, features: Path, manifest: Path, contract: Path) -> URDMUEvaluationRequest:
    return URDMUEvaluationRequest(
        dataset="ucf_crime",
        phase="development_video",
        mode="direct_insert",
        trained_run=str(run),
        feature_store=str(features),
        evaluation_manifest=str(manifest),
        upstream_dir="must-not-load-in-unit-test",
        output_root=str(tmp_path / "outputs"),
        device="cpu",
        feature_contract_path=str(contract),
        feature_contract_sha256=sha256_file(contract),
        run_id="scores",
    )


def test_development_scoring_is_select_only_and_writes_standard_frame_records(tmp_path, monkeypatch):
    run, features, manifest, contract, authority_contract, authority_records = _source(tmp_path)
    _patch_authority(monkeypatch, authority_contract, authority_records, tmp_path / "roles.json")

    def fake_scores(source, target, request):
        return (
            {
                record.video_id: np.full(
                    len(target.rows_by_video[record.video_id]), float(record.is_anomaly)
                )
                for record in target.records
            },
            {},
        )

    monkeypatch.setattr("vadbench.paper.urdmu_evaluation._run_scores", fake_scores)
    result = run_urdmu_evaluation(_request(tmp_path, run, features, manifest, contract))
    assert result.completed and result.video_metrics["video_roc_auc"] == 1.0
    rows = [json.loads(line) for line in Path(result.predictions).read_text(encoding="utf8").splitlines()]
    assert rows and {row["metadata"]["score_level"] for row in rows} == {"frame"}
    assert {row["metadata"]["ground_truth_scope"] for row in rows} == {"video"}
    resolved = json.loads((Path(result.run_dir) / "resolved.json").read_text(encoding="utf8"))
    assert resolved["official_frame_scores_read"] is False
    assert resolved["test_sequence"]["kind"] == "complete_native_dense_windows_no_200_bin_aggregation"


def test_development_scoring_rejects_engineering_or_nonselect_sources(tmp_path, monkeypatch):
    run, features, manifest, contract, authority_contract, authority_records = _source(tmp_path)
    _patch_authority(monkeypatch, authority_contract, authority_records, tmp_path / "roles.json")
    result = json.loads((run / "result.json").read_text(encoding="utf8"))
    checkpoint = Path(result["checkpoint_path"])
    payload = torch.load(checkpoint, weights_only=False)
    payload["metadata"]["run_mode"] = "engineering"
    torch.save(payload, checkpoint)
    digest = sha256_file(checkpoint)
    _write(checkpoint.with_suffix(".pt.json"), {"sha256": digest, "size_bytes": checkpoint.stat().st_size})
    result["checkpoint_sha256"] = digest
    _write(run / "result.json", result)
    qa = json.loads((run / "training_qa.json").read_text(encoding="utf8"))
    qa["checkpoint"]["sha256"] = digest
    _write(run / "training_qa.json", qa)
    with pytest.raises(ValueError, match="development-fit"):
        run_urdmu_evaluation(_request(tmp_path, run, features, manifest, contract))


def test_direct_insert_rejects_another_d768_encoder_even_when_output_dimension_matches(tmp_path):
    run, features, manifest, _contract, _authority_contract, _authority_records = _source(tmp_path)
    request = _request(tmp_path, run, features, manifest, _contract)
    source = evaluation._load_source(request)
    target = evaluation._load_dense_feature_source(features, manifest)
    different_backbone = replace(target.representation.backbone, runtime_id="timesformer")
    different_representation = replace(target.representation, backbone=different_backbone)
    with pytest.raises(ValueError, match="direct_insert may only change the reducer"):
        evaluation._verify_pairing(
            request, source, replace(target, representation=different_representation)
        )


def test_direct_insert_rejects_different_dense_window_strategy(tmp_path):
    run, features, manifest, contract, _authority_contract, _authority_records = _source(tmp_path)
    request = _request(tmp_path, run, features, manifest, contract)
    source = evaluation._load_source(request)
    target = evaluation._load_dense_feature_source(features, manifest)
    different_sampling = replace(target.sampling, stride={"frame_stride": 1})
    with pytest.raises(ValueError, match="sampling definitions differ"):
        evaluation._verify_pairing(request, source, replace(target, sampling=different_sampling))


def test_development_gate_rejects_empty_component_update_receipt(tmp_path, monkeypatch):
    run, features, manifest, contract, authority_contract, authority_records = _source(tmp_path)
    _patch_authority(monkeypatch, authority_contract, authority_records, tmp_path / "roles.json")
    qa = json.loads((run / "training_qa.json").read_text(encoding="utf8"))
    qa["changed_parameters_by_component"] = {}
    _write(run / "training_qa.json", qa)
    with pytest.raises(ValueError, match="all seven"):
        run_urdmu_evaluation(_request(tmp_path, run, features, manifest, contract))


def test_development_gate_rejects_missing_member_from_authoritative_select(tmp_path, monkeypatch):
    run, features, manifest, contract, authority_contract, authority_records = _source(tmp_path)
    _patch_authority(
        monkeypatch,
        authority_contract,
        tuple(record for record in authority_records if record.video_id != "select-1"),
        tmp_path / "roles.json",
    )
    with pytest.raises(ValueError, match="role lock"):
        run_urdmu_evaluation(_request(tmp_path, run, features, manifest, contract))


def test_development_gate_rejects_fit_manifest_that_differs_from_authoritative_role(tmp_path, monkeypatch):
    run, features, manifest, contract, authority_contract, authority_records = _source(tmp_path)
    altered_fit = replace(authority_records[0], category="DifferentNormal")
    _patch_authority(
        monkeypatch,
        authority_contract,
        (altered_fit, *authority_records[1:]),
        tmp_path / "roles.json",
    )
    with pytest.raises(ValueError, match="fit manifest differs"):
        run_urdmu_evaluation(_request(tmp_path, run, features, manifest, contract))


def test_development_gate_rejects_tampered_authoritative_role_lock(tmp_path, monkeypatch):
    run, features, manifest, contract, authority_contract, authority_records = _source(tmp_path)
    lock = tmp_path / "roles.json"
    _write(lock, {"partitions": {record.video_id: "fit" for record in authority_records}})
    _patch_authority(monkeypatch, authority_contract, authority_records, lock)
    with pytest.raises(ValueError, match="role lock"):
        run_urdmu_evaluation(_request(tmp_path, run, features, manifest, contract))


def test_dense_test_path_never_collapses_to_200_bins_and_can_fail_before_model(tmp_path, monkeypatch):
    run, features, manifest, contract, authority_contract, authority_records = _source(tmp_path)
    _patch_authority(monkeypatch, authority_contract, authority_records, tmp_path / "roles.json")
    request = _request(tmp_path, run, features, manifest, contract)
    request = URDMUEvaluationRequest(**{**request.__dict__, "max_attention_workspace_bytes": 1})
    monkeypatch.setattr("vadbench.paper.urdmu_evaluation.build_urdmu", lambda *args, **kwargs: pytest.fail("model must not load after workspace gate"))
    with pytest.raises(MemoryError, match="No 200-bin aggregation"):
        run_urdmu_evaluation(request)
    assert _attention_workspace_lower_bound_bytes(10) == 13 * 10 * 10 * 4


def test_exact_query_chunk_receipt_must_bind_full_keys_and_unchanged_model_state():
    receipt = {
        "schema": "urdmu.query-chunk-exact-attention/v1",
        "status": "completed",
        "execution": "eval_only_exact_all_key_query_chunk_attention",
        "approximation": False,
        "token_reduction": False,
        "complete_key_value_sequence": True,
        "source_identity_complete": True,
        "sequence_length": 17,
        "query_chunk_size": 4,
        "state_dict": {"unchanged": True},
    }
    _query_chunk_receipt(receipt, expected_chunk=4, length=17)
    receipt["state_dict"] = {"unchanged": False}
    with pytest.raises(ValueError, match="changed the UR-DMU state"):
        _query_chunk_receipt(receipt, expected_chunk=4, length=17)


def test_official_phase_requires_freeze_and_audit_at_request_construction(tmp_path):
    run, features, manifest, contract, _authority_contract, _authority_records = _source(tmp_path)
    with pytest.raises(ValueError, match="method freeze"):
        URDMUEvaluationRequest(
            **{**_request(tmp_path, run, features, manifest, contract).__dict__, "phase": "official_frame"}
        )


def test_official_phase_is_blocked_before_any_test_artifact_is_opened(tmp_path):
    run, features, manifest, contract, _authority_contract, _authority_records = _source(tmp_path)
    request = URDMUEvaluationRequest(
        **{
            **_request(tmp_path, run, features, manifest, contract).__dict__,
            "phase": "official_frame",
            "freeze_path": str(tmp_path / "old-freeze-not-opened.json"),
            "audit_report": str(tmp_path / "audit-not-opened.json"),
        }
    )
    with pytest.raises(NotImplementedError, match="no official test source or score was read"):
        run_urdmu_evaluation(request)
