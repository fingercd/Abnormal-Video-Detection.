"""Synthetic-only UCF export comparisons; no official scores/GT are opened."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from vadbench.artifacts import PredictionRecord
from vadbench.data.audit import audit_ucf_crime_dataset, compute_manifest_sha256
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.labels import frame_labels_for_record
from vadbench.data.manifest import (
    SupervisionAnnotation,
    TemporalSpan,
    VideoManifestRecord,
    write_manifest_jsonl,
)
from vadbench.engine.coverage import validate_frame_coverage
from vadbench.engine.evaluate import evaluate_manifest_predictions, project_video_prediction
from vadbench.features import ArrayReference, FeatureRecord, compute_encoder_fingerprint
from vadbench.paper import evaluation, quality_export, repeat_evaluation
from vadbench.paper.compatibility import (
    BackboneIdentity,
    RepresentationIdentity,
    TrainingIdentity,
    feature_cache_key,
    validate_compatibility,
)
from vadbench.paper.extraction import (
    _semantic_runtime_identity,
    _verified_code_digest,
    _verified_weights_digest,
    make_data_content_evidence,
    make_sampling_identity,
)
from vadbench.paper.quality_comparison import compare_paired_quality
from vadbench.paper.quality_export import paired_frame_intervals


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path):
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _paper_identity(representation, sampling):
    return {"representation_fingerprint": representation.fingerprint, "sampling_fingerprint": sampling.fingerprint, "feature_cache_fingerprint": feature_cache_key(representation, sampling)}


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory):
    # Reuse the repository's synthetic official-size dataset/registry builder;
    # the real audit and official-source parser execute on these local fixtures.
    root = tmp_path_factory.mktemp("synthetic-quality-export")
    helper_path = Path(__file__).parents[1] / "test_data_audit.py"
    spec = importlib.util.spec_from_file_location("quality_audit_fixture", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    train, test = helper._official_records_and_files(root)
    registry = helper._official_source_registry(root, train, test)
    train_path = write_manifest_jsonl(train, root / "all-train.jsonl")
    manifest = write_manifest_jsonl(test, root / "sealed-test.jsonl")
    report = audit_ucf_crime_dataset(root, train_path, manifest, probe_fn=helper._fake_probe, official_source_registry=registry)
    assert report["passed"] and len(test) == 290
    audit = root / "sealed-audit.json"
    _write(audit, report)
    fit = [replace(train[0], num_frames=100, fps=25.), replace(train[800], num_frames=100, fps=25.)]
    select = [replace(train[1], num_frames=100, fps=25.)]
    sources = root / "head-sources"
    fit_path = write_manifest_jsonl(fit, sources / "fit.jsonl")
    select_path = write_manifest_jsonl(select, sources / "select.jsonl")
    roles = {r.video_id: "confirm" for r in train}
    roles.update({r.video_id: "fit" for r in fit})
    roles.update({r.video_id: "select" for r in select})
    lock = root / "role-lock.json"
    _write(lock, {"schema_version": 1, "basis": "complete_official_train_source_groups", "seed": 20260918, "partitions": roles})
    freeze_doc = {"status": "algorithm_and_budget_frozen_before_official_model_scores", "annotation_policy": "W", "active_encoders": ["toy"], "methods": {"dense": {"reducer": "identity"}, "same_budget_control": {"reducer": "global_uniform"}, "training_free": {"reducer": "paired_random", "seed": 0}, "trainable": {"reducer": "pair_linear"}}, "detection_head": {"primary_seed": 0, "key_configuration_seeds": [0, 1, 2], "epochs": 20, "batch_size": 16, "learning_rate": .001, "weight_decay": 0.}, "ucf_evaluation": {"sealed_test_manifest_sha256": _sha(manifest), "sealed_audit_sha256": _sha(audit)}, "sampling": {"native_clip_frames": {"toy": 16}, "frame_stride": 2, "short_policy": "strict", "evaluation_window_stride": {"toy": 16}}}
    freeze = root / "freeze.json"
    _write(freeze, freeze_doc)
    contract = root / "head-contract.json"
    _write(contract, {"schema_version": 1, "status": "fixed_before_official_model_scores", "dataset": "ucf-crime", "official_training_videos": 1610, "method_freeze_canonical_sha256": evaluation._canonical_sha256(freeze_doc), "role_lock_sha256": _sha(lock), "role_lock_seed": 20260918, "roles": {role: {"videos": len(records), "source_manifest_sha256": _sha(path), "source_manifest": f"manifests/{role}.jsonl", "source_split": "train", "controller_split": split} for role, records, path, split in (("fit", fit, fit_path, "train"), ("select", select, select_path, "val"))}})
    verified = {"adapter": "toy", "constructor": {"clip_frames": 16}, "checkpoint": {"sha256": {"fixture-native": "a" * 64}}}
    runtime = {"adapter_type": "fixture.Adapter", "adapter_library_version": None, "encoder_type": "fixture.Encoder", "encoder_library_version": None, "capabilities": {"fixed_num_frames": 16, "supports_fixed_clip": True}, "properties": {"pooling": "mean"}, "runtime_configurations": {"model_config": {"hidden_size": 4}}, "implementation_files": {"model": {"role": "model", "module": "fixture", "path": str(root / "native.py"), "sha256": "b" * 64}}, "loaded_library_versions": {"torch": str(torch.__version__)}}
    backbone = BackboneIdentity("toy", _verified_weights_digest(verified), _verified_code_digest(verified, runtime), {"profile": "fixture"}, {"kind": "mean"})
    return SimpleNamespace(root=root, test=test, manifest=manifest, audit=audit, registry=registry, fit=fit, select=select, lock=lock, contract=contract, source_root=sources, freeze=freeze, freeze_doc=freeze_doc, verified=verified, runtime=runtime, backbone=backbone)


def _apply_seals(monkeypatch, data):
    # Only the externally trusted byte digests change to the synthetic domain;
    # no coverage count, verifier, reader, projection or statistic is mocked.
    monkeypatch.setattr(evaluation, "_UCF_TEST_MANIFEST_SHA256", _sha(data.manifest))
    monkeypatch.setattr(evaluation, "_UCF_AUDIT_SHA256", _sha(data.audit))
    monkeypatch.setattr(evaluation, "_FROZEN_ROLE_LOCK_SHA256", _sha(data.lock))
    monkeypatch.setattr(evaluation, "_FROZEN_HEAD_DATA_CONTRACT_SHA256", _sha(data.contract))


def _index_row(record, index, fingerprint, paper, *, frame_start=0, frame_end=100, metadata=None):
    arrays = {"features": ArrayReference(path="fixture.npz", key="features", shape=(1, 4), dtype="<f4", sha256="c" * 64, nbytes=16), "pooled": ArrayReference(path="fixture.npz", key="pooled", shape=(4,), dtype="<f4", sha256="c" * 64, nbytes=16)}
    return FeatureRecord(record.video_id, f"{record.video_id}:clip-{index:06d}", index, fingerprint, "npz", arrays, frame_start / 25., frame_end / 25., frame_start, frame_end, {"paper_identity": paper, **(metadata or {})}).to_dict()


def _write_index(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf8")


def _source_head(root, data, representation, *, seed=0, weight=1.):
    fit_path = write_manifest_jsonl(data.fit, root / "frozen/train.jsonl")
    select = [replace(r, split="val") for r in data.select]
    write_manifest_jsonl(select, root / "frozen/val.jsonl")
    sampling = make_sampling_identity(data.fit, dataset_root=data.root, sampling_kind="uniform_full", clip_frames=16, frame_stride=2)
    train_fp = compute_encoder_fingerprint({"source_representation": representation.fingerprint, "sampling": sampling.fingerprint})
    for role, records in (("train", data.fit), ("validation", select)):
        sample = make_sampling_identity(records, dataset_root=data.root, sampling_kind="uniform_full", clip_frames=16, frame_stride=2)
        fingerprint = train_fp if role == "train" else compute_encoder_fingerprint({"representation": representation.fingerprint, "sampling": sample.fingerprint})
        paper = _paper_identity(representation, sample)
        folder = root / "features" / role
        _write_index(folder / "index.jsonl", [_index_row(r, i, fingerprint, paper) for r in records for i in range(32)])
        _write(folder / "resolved.json", {"spec": {"representation": representation.to_dict(), "sampling": sample.to_dict(), "sampling_kind": "uniform_full"}, "encoder_fingerprint": fingerprint, "paper_identity": paper})
        _write(folder / "status.json", {"completed": True, "feature_root": str(folder.resolve())})
    training = TrainingIdentity({"kind": "topk_mil", "k": 3}, "sha256:" + compute_manifest_sha256(data.fit), seed, {"epochs": 20, "lr": .001})
    config = {"task": "weak_mil", "feature_level": "clip", "head": "topk", "head_kwargs": {"k": 3, "dropout": 0.}, "task_kwargs": {"ranking_weight": 0.}, "expected_clips": 32, "epochs": 20, "batch_size": 16, "learning_rate": .001, "weight_decay": 0., "seed": seed}
    metadata = {"status": "completed", "encoder_fingerprint": train_fp, "feature_dim": 4, "config": config, "training_qa": {"status": "passed", "nonzero_gradient_steps": 20, "changed_parameter_count": 1, "parameter_delta_l2": 1., "reload_parity_exact": True}, "paper_detector": {"training_representation_fingerprint": representation.fingerprint, "training_sampling_fingerprint": sampling.fingerprint, "training_feature_cache_fingerprint": feature_cache_key(representation, sampling), "training_identity_fingerprint": training.fingerprint}}
    checkpoint = root / "head/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {"weight": torch.full((1, 4), weight)}, "epoch": 20, "step": 20, "metadata": metadata}, checkpoint)
    _write(checkpoint.with_suffix(".pt.json"), {"sha256": _sha(checkpoint), "metadata": metadata})
    _write(root / "head/training_qa.json", {"status": "passed", "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint), "epoch": 20, "step": 20}, "nonzero_gradient_steps": 20, "changed_parameter_count": 1, "parameter_delta_l2": 1., "reload_parity": {"outputs": {"snippet_scores": {"exact_equal": True, "max_abs_difference": 0.}}}, "final_checkpoint_load": {"missing_keys": [], "unexpected_keys": []}})
    _write(root / "result.json", {"status": "completed", "checkpoint_path": str(checkpoint.resolve())})
    _write(root / "provenance/stages/source.json", {"stage": "paper_detection", "status": "completed", "config": {"encoder": "toy", "epochs": 20, "batch_size": 16, "learning_rate": .001, "seed": seed}})
    assert fit_path.is_file()
    return evaluation.load_frozen_detector_source(root, freeze=data.freeze_doc, expected_encoder="toy", role_lock_path=data.lock, head_data_contract_path=data.contract, source_manifest_root=data.source_root)


def _evaluation_run(root, data, source, *, dense_source=None, split_scores=False):
    representation = source.representation
    sampling = make_sampling_identity(data.test, dataset_root=data.root, sampling_kind="dense", clip_frames=16, frame_stride=2, window_stride=16)
    runtime = {**copy.deepcopy(data.runtime), "runtime_id": "toy", "representation_fingerprint": representation.fingerprint, "sampling_fingerprint": sampling.fingerprint, "source_data_digest": sampling.source_digest, "declared_readout": dict(representation.backbone.readout), "verified_encoder_identity": data.verified}
    fingerprint = compute_encoder_fingerprint({"paper_pooled_cache": _semantic_runtime_identity(runtime)})
    paper = _paper_identity(representation, sampling)
    features = root / "features/test"
    input_rows = []
    for record in sorted(data.test, key=lambda r: r.video_id):
        for sample in DenseSamplingPlan(16, 2, 16).sample(record.num_frames):
            metadata = {"sampling": {"clip_index": sample.clip_index, "actual_frame_stride": 2, "end_anchored": sample.end_anchored, "kind": "dense"}, "source_video": {"manifest_path": record.path, "actual_num_frames": record.num_frames, "actual_fps": record.fps, "width": 320, "height": 240, "content_sha256": _sha(record.resolve_path(data.root)), "content_size_bytes": record.resolve_path(data.root).stat().st_size}}
            input_rows.append(_index_row(record, sample.clip_index, fingerprint, paper, frame_start=sample.score_frame_start, frame_end=sample.score_frame_end, metadata=metadata))
    _write_index(features / "index.jsonl", input_rows)
    _write(features / "resolved.json", {"spec": {"representation": representation.to_dict(), "sampling": sampling.to_dict(), "sampling_kind": "dense"}, "runtime": runtime, "data_content_evidence": make_data_content_evidence(data.test, dataset_root=data.root), "encoder_fingerprint": fingerprint, "paper_identity": paper})
    _write(features / "status.json", {"status": "completed", "completed": True, "failures": [], "feature_root": str(features.resolve()), "encoder_fingerprint": fingerprint, "records_written_to_shards": len(input_rows)})
    artifacts = {"test_feature_store": {"root": str(features.resolve()), **{name: _binding(features / filename) for name, filename in (("resolved", "resolved.json"), ("status", "status.json"), ("index", "index.jsonl"))}}}
    for branch, mode, head in (("primary", "refit_head", source), ("secondary", "direct_insert", dense_source)):
        if head is None:
            artifacts[branch + "_predictions"] = artifacts[branch + "_metrics"] = None
            continue
        permit = validate_compatibility(head.detection_config(mode=mode, evaluation_representation=representation, evaluation_sampling=sampling, evaluation_encoder_fingerprint=fingerprint).declaration)
        predictions = []
        for record in data.test:
            spans = [(0, 15, .3), (15, 100, .7)] if record.is_anomaly else [(0, 100, .1)]
            if split_scores:
                spans = [(a, (a + b) // 2, score) for a, b, score in spans] + [((a + b) // 2, b, score) for a, b, score in spans]
            for start, end, score in spans:
                pred = _prediction(record, start, end, score)
                head_receipt = head.receipt() if isinstance(head, repeat_evaluation.FrozenRepeatHeadSource) else evaluation._source_receipt(head)
                predictions.append(replace(pred, run_id=str(root.name) + branch, encoder_fingerprint=fingerprint, metadata={"checkpoint_sha256": head_receipt["checkpoint_sha256"], "paper_compatibility": permit, "score_level": "frame", "ground_truth_scope": "video", "dense_aggregation": {"reduction": "mean", "contributing_windows": 1, "interval_source": "exact_equal_frame_run"}}))
        suffix = "primary-refit-head" if branch == "primary" else "secondary-dense-head-direct-insert"
        pred_path, metric_path = root / f"predictions-{suffix}.jsonl", root / f"metrics-{suffix}.json"
        _write_index(pred_path, [r.to_dict() for r in predictions])
        metrics = evaluate_manifest_predictions(predictions, data.manifest, audit_report=data.audit, official_source_registry=data.registry)
        _write(metric_path, metrics.to_dict())
        artifacts[branch + "_predictions"], artifacts[branch + "_metrics"] = _binding(pred_path), _binding(metric_path)
    config = {"encoder": "toy", "dataset_root": str(data.root), "test_manifest": str(data.manifest), "audit_report": str(data.audit), "freeze_path": str(data.freeze), "role_lock_path": str(data.lock), "head_data_contract_path": str(data.contract), "source_manifest_root": str(data.source_root)}
    inputs = {name: {"location": str(path.resolve()), "sha256": _sha(path)} for name, path in (("test_manifest", data.manifest), ("audit_report", data.audit), ("method_freeze", data.freeze), ("training_role_lock", data.lock), ("head_data_contract", data.contract), ("method_checkpoint", source.checkpoint))}
    if dense_source is not None:
        inputs["dense_checkpoint"] = {"location": str(dense_source.checkpoint), "sha256": _sha(dense_source.checkpoint)}
    is_repeat = isinstance(source, repeat_evaluation.FrozenRepeatHeadSource)
    if is_repeat:
        inputs["repeat_checkpoint"] = inputs.pop("method_checkpoint")
        if dense_source is not None:
            inputs["dense_repeat_checkpoint"] = inputs.pop("dense_checkpoint")
    _write(root / "provenance/stages/evaluation.json", {"stage": "frozen_repeat_ucf_evaluation" if is_repeat else "frozen_ucf_evaluation", "status": "completed", "config": config, "config_sha256": hashlib.sha256(quality_export._canonical(config)).hexdigest(), "inputs": inputs})
    resolved = {"schema_version": 1, "freeze_sha256": _sha(data.freeze), "sealed_test_manifest_sha256": _sha(data.manifest), "sealed_audit_sha256": _sha(data.audit), "method_source": source.receipt() if is_repeat else evaluation._source_receipt(source), "dense_source": None if dense_source is None else dense_source.receipt() if isinstance(dense_source, repeat_evaluation.FrozenRepeatHeadSource) else evaluation._source_receipt(dense_source), "evaluation_representation": representation.to_dict(), "evaluation_representation_fingerprint": representation.fingerprint, "evaluation_sampling": sampling.to_dict(), "evaluation_sampling_fingerprint": sampling.fingerprint, "evaluation_encoder_fingerprint": fingerprint, "coverage": {"input_union_complete": True, "videos": 290, "overlap_is_expected_before_prediction_aggregation": True}, "artifacts": artifacts}
    if is_repeat:
        resolved["secondary_status"] = "completed" if dense_source is not None else "not_requested"
    _write(root / "resolved.json", resolved)
    _write(root / "result.json", {"schema_version": 1, "status": "completed", "protocol": "ucf-official-frozen-v1", "resolved": _binding(root / "resolved.json"), "artifacts": artifacts})
    return root


def _repeat_head(root, data, base, seed):
    metadata = copy.deepcopy(base.checkpoint_metadata)
    metadata["config"]["seed"] = seed
    training = replace(base.training_identity, seed=seed)
    metadata["paper_detector"]["training_identity_fingerprint"] = training.fingerprint
    checkpoint = root / "head/checkpoints/final.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {"weight": torch.full((1, 4), float(seed))}, "epoch": 20, "step": 20, "metadata": metadata}, checkpoint)
    _write(checkpoint.with_suffix(".pt.json"), {"sha256": _sha(checkpoint), "metadata": metadata})
    qa = json.loads((base.root / "head/training_qa.json").read_text(encoding="utf8"))
    qa["checkpoint"].update(path=str(checkpoint), sha256=_sha(checkpoint))
    _write(root / "head/training_qa.json", qa)
    _write(root / "result.json", {"schema_version": 1, "status": "completed", "seed": seed, "freeze_sha256": _sha(data.freeze), "source_controller_run": str(base.root), "source_artifact_sha256": dict(base.source_hashes), "training_role_lock": {"path": str(base.role_lock_path), "sha256": base.role_lock_sha256}, "head_data_contract": {"path": str(base.head_data_contract_path), "sha256": base.head_data_contract_sha256}, "source_manifest_root": str(base.source_manifest_root), "source_manifests_sha256": dict(base.source_manifest_sha256), "checkpoint_path": str(checkpoint), "checkpoint_sha256": _sha(checkpoint), "checkpoint_sidecar_sha256": _sha(checkpoint.with_suffix(".pt.json")), "training_qa_sha256": _sha(root / "head/training_qa.json"), "training_identity_fingerprint": training.fingerprint, "training_representation_fingerprint": base.representation.fingerprint, "training_sampling_fingerprint": base.sampling.fingerprint})
    _write(root / "provenance/stages/repeat.json", {"stage": "frozen_head_repeat", "status": "completed", "config": {"encoder": "toy", "seed": seed, "freeze_path": str(data.freeze)}, "inputs": {"source_checkpoint": {"location": str(base.checkpoint), "sha256": _sha(base.checkpoint)}, "source_train_manifest": {"location": str(base.train_manifest), "sha256": _sha(base.train_manifest)}}})
    return repeat_evaluation.load_frozen_repeat_head_source(root, freeze=data.freeze_doc, expected_encoder="toy", role_lock_path=data.lock, head_data_contract_path=data.contract, source_manifest_root=data.source_root)


def _reuse_store(repeat, source):
    resolved_path = repeat / "resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf8"))
    original = json.loads((source / "resolved.json").read_text(encoding="utf8"))
    resolved["artifacts"]["test_feature_store"] = original["artifacts"]["test_feature_store"]
    resolved["reused_test_feature_store_from"] = {"run_dir": str(source), "result": _binding(source / "result.json"), "resolved": _binding(source / "resolved.json")}
    _write(resolved_path, resolved)
    _refresh(repeat)


@pytest.fixture(scope="module")
def _base_frozen_pair(tmp_path_factory, synthetic_dataset):
    tmp_path = tmp_path_factory.mktemp("base-quality-pair")
    data = synthetic_dataset
    with pytest.MonkeyPatch.context() as monkeypatch:
        _apply_seals(monkeypatch, data)
        dense_rep = RepresentationIdentity(data.backbone, {"name": "identity", "calibration": "none"}, 4, "float32", {"kind": "native"})
        method_rep = replace(dense_rep, reducer={"name": "paired_random", "seed": 0})
        dense_head = _source_head(tmp_path / "dense-head", data, dense_rep)
        method_head = _source_head(tmp_path / "method-head", data, method_rep)
        dense = _evaluation_run(tmp_path / "dense-eval", data, dense_head)
        method = _evaluation_run(tmp_path / "method-eval", data, method_head, dense_source=dense_head, split_scores=True)
    return SimpleNamespace(data=data, dense=dense, method=method, dense_head=dense_head, method_head=method_head, dense_rep=dense_rep, method_rep=method_rep)


def _clone_evaluation(source, destination):
    shutil.copytree(source, destination)

    def rewrite(value):
        if isinstance(value, str) and value.startswith(str(source)):
            return str(destination) + value[len(str(source)):]
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        return value

    for path in destination.rglob("*.json"):
        document = rewrite(json.loads(path.read_text(encoding="utf8")))
        if document.get("stage") == "frozen_ucf_evaluation":
            document["config_sha256"] = hashlib.sha256(quality_export._canonical(document["config"])).hexdigest()
        _write(path, document)
    resolved_path = destination / "resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf8"))
    for name, entry in resolved["artifacts"].items():
        if name == "test_feature_store":
            for field in ("resolved", "status", "index"):
                entry[field]["sha256"] = _sha(Path(entry[field]["path"]))
        elif entry is not None:
            entry["sha256"] = _sha(Path(entry["path"]))
    _write(resolved_path, resolved)
    _refresh(destination)
    return destination


@pytest.fixture
def frozen_pair(tmp_path, _base_frozen_pair, monkeypatch):
    base = _base_frozen_pair
    _apply_seals(monkeypatch, base.data)
    return SimpleNamespace(**{**base.__dict__, "dense": _clone_evaluation(base.dense, tmp_path / "dense-eval"), "method": _clone_evaluation(base.method, tmp_path / "method-eval")})


def test_full_290_video_run_comparison_preserves_same_score_under_different_partitions(frozen_pair):
    pair = frozen_pair
    for mode in ("refit_head", "direct_insert"):
        result = quality_export.compare_frozen_ucf_quality(pair.dense, pair.method, mode=mode)
        assert result["n_videos"] == 290 and result["n_frames"] == 29000
        assert result["positive_frames"] == 1400  # never the 14000 weak-positive-video frames
        assert result["valid_draws"] == 10000
        assert result["delta_method_minus_dense"] == result["ci_low"] == result["ci_high"] == 0
        assert result["noninferiority"] == "supported" and result["head_seed"] == 0
        assert result["mode"] == mode
        assert result["decision_precision"]["ulps"] == 8
        assert result["source_sha256"][str(pair.method / "result.json")] == _sha(pair.method / "result.json")
        assert result["source_sha256"][str(pair.data.lock)] == _sha(pair.data.lock)


def _refresh(run, *artifact_keys):
    resolved = json.loads((run / "resolved.json").read_text(encoding="utf8"))
    for key in artifact_keys:
        entry = resolved["artifacts"][key]
        entry["sha256"] = _sha(Path(entry["path"]))
    _write(run / "resolved.json", resolved)
    result = json.loads((run / "result.json").read_text(encoding="utf8"))
    result.update(resolved=_binding(run / "resolved.json"), artifacts=resolved["artifacts"])
    _write(run / "result.json", result)


def test_pending_failed_and_prediction_sha_tampering_fail_before_comparison(frozen_pair, monkeypatch):
    pair = frozen_pair
    result_path = pair.method / "result.json"
    original = json.loads(result_path.read_text(encoding="utf8"))
    for status in ("running", "failed"):
        _write(result_path, {**original, "status": status})
        with pytest.raises(ValueError, match="not a completed"):
            quality_export._load_completed_run(pair.method, "refit_head")
    _write(result_path, original)
    pred = pair.method / "predictions-primary-refit-head.jsonl"
    pred.write_text(pred.read_text(encoding="utf8") + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        quality_export._load_completed_run(pair.method, "refit_head")


def test_coverage_and_checkpoint_identity_fail_even_if_file_ledger_is_rebound(frozen_pair):
    pair = frozen_pair
    path = pair.method / "predictions-primary-refit-head.jsonl"
    original = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    altered = copy.deepcopy(original)
    altered[0]["metadata"]["checkpoint_sha256"] = "0" * 64
    _write_index(path, altered)
    _refresh(pair.method, "primary_predictions")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        quality_export._load_completed_run(pair.method, "refit_head")
    _write_index(path, original[1:])
    _refresh(pair.method, "primary_predictions")
    with pytest.raises(ValueError, match="gap"):
        quality_export._load_completed_run(pair.method, "refit_head")


def test_feature_runtime_or_input_index_sha_mutation_fails(frozen_pair):
    pair = frozen_pair
    index = pair.method / "features/test/index.jsonl"
    index.write_text(index.read_text(encoding="utf8") + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="index SHA"):
        quality_export._load_completed_run(pair.method, "refit_head")


def test_missing_native_window_is_rejected_even_when_both_runs_still_have_union_coverage(frozen_pair):
    for run in (frozen_pair.dense, frozen_pair.method):
        index = run / "features/test/index.jsonl"
        rows = [json.loads(line) for line in index.read_text(encoding="utf8").splitlines()]
        video_id = rows[0]["video_id"]
        subset = [r for r in rows if r["video_id"] == video_id]
        removed = subset[-2]["clip_index"]
        rows = [r for r in rows if (r["video_id"], r["clip_index"]) != (video_id, removed)]
        remaining = [r for r in rows if r["video_id"] == video_id]
        coverage = validate_frame_coverage(video_id=video_id, clip_indices=np.array([r["clip_index"] for r in remaining]), frame_starts=np.array([r["frame_start"] for r in remaining]), frame_ends=np.array([r["frame_end"] for r in remaining]), num_frames=100, fps=25., require_complete=False)
        assert coverage["gap_frames"] == 0  # union alone would miss this corruption
        _write_index(index, rows)
        status_path = run / "features/test/status.json"
        status = json.loads(status_path.read_text(encoding="utf8"))
        status["records_written_to_shards"] -= 1
        _write(status_path, status)
        resolved_path = run / "resolved.json"
        resolved = json.loads(resolved_path.read_text(encoding="utf8"))
        resolved["artifacts"]["test_feature_store"]["index"] = _binding(index)
        resolved["artifacts"]["test_feature_store"]["status"] = _binding(status_path)
        _write(resolved_path, resolved)
        _refresh(run)
    with pytest.raises(ValueError, match="every frozen sampler window"):
        quality_export.compare_frozen_ucf_quality(frozen_pair.dense, frozen_pair.method)


def test_seed_mismatch_and_different_direct_insert_checkpoint_fail(frozen_pair, tmp_path):
    pair = frozen_pair
    seed1 = _source_head(tmp_path / "seed1-head", pair.data, pair.method_rep, seed=1)
    seed1_run = _evaluation_run(tmp_path / "seed1-eval", pair.data, seed1)
    with pytest.raises(ValueError, match="heads differ in seed"):
        quality_export.compare_frozen_ucf_quality(pair.dense, seed1_run)
    different_dense = _source_head(tmp_path / "other-dense", pair.data, pair.dense_rep, weight=2.)
    wrong_direct = _evaluation_run(tmp_path / "wrong-direct", pair.data, pair.method_head, dense_source=different_dense)
    with pytest.raises(ValueError, match="exact dense baseline head"):
        quality_export.compare_frozen_ucf_quality(pair.dense, wrong_direct, mode="direct_insert")


def test_semantically_failed_audit_is_rejected_despite_self_consistent_new_file_hashes(frozen_pair, tmp_path, monkeypatch):
    pair = frozen_pair
    failed_audit = json.loads(pair.data.audit.read_text(encoding="utf8"))
    failed_audit.update(status="failed", passed=False)
    audit_path = tmp_path / "failed-audit.json"
    _write(audit_path, failed_audit)
    changed_freeze = copy.deepcopy(pair.data.freeze_doc)
    changed_freeze["ucf_evaluation"]["sealed_audit_sha256"] = _sha(audit_path)
    freeze_path = tmp_path / "failed-audit-freeze.json"
    _write(freeze_path, changed_freeze)
    monkeypatch.setattr(evaluation, "_UCF_AUDIT_SHA256", _sha(audit_path))
    stage_path = pair.method / "provenance/stages/evaluation.json"
    stage = json.loads(stage_path.read_text(encoding="utf8"))
    stage["inputs"]["audit_report"] = {"location": str(audit_path), "sha256": _sha(audit_path)}
    stage["inputs"]["method_freeze"] = {"location": str(freeze_path), "sha256": _sha(freeze_path)}
    _write(stage_path, stage)
    resolved_path = pair.method / "resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf8"))
    resolved.update(freeze_sha256=_sha(freeze_path), sealed_audit_sha256=_sha(audit_path))
    _write(resolved_path, resolved)
    _refresh(pair.method)
    with pytest.raises(ValueError, match="audit"):
        quality_export._load_completed_run(pair.method, "refit_head")


def test_saved_metric_numbers_are_recomputed_not_trusted(frozen_pair):
    pair = frozen_pair
    path = pair.method / "metrics-primary-refit-head.json"
    metrics = json.loads(path.read_text(encoding="utf8"))
    metrics["frame_auc"] += .01
    _write(path, metrics)
    _refresh(pair.method, "primary_metrics")
    with pytest.raises(ValueError, match="saved official metric disagrees"):
        quality_export.compare_frozen_ucf_quality(pair.dense, pair.method)


def test_external_feature_store_requires_completed_source_evaluation_and_exact_binding(frozen_pair, tmp_path):
    pair = frozen_pair
    repeat = _evaluation_run(tmp_path / "repeat", pair.data, pair.method_head, dense_source=pair.dense_head)
    resolved_path = repeat / "resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf8"))
    source = json.loads((pair.method / "resolved.json").read_text(encoding="utf8"))
    resolved["artifacts"]["test_feature_store"] = source["artifacts"]["test_feature_store"]
    _write(resolved_path, resolved)
    _refresh(repeat)
    with pytest.raises(ValueError, match="external FeatureStore lacks"):
        quality_export._load_completed_run(repeat, "refit_head")
    resolved["reused_test_feature_store_from"] = {"run_dir": str(pair.method), "result": _binding(pair.method / "result.json"), "resolved": _binding(pair.method / "resolved.json")}
    _write(resolved_path, resolved)
    _refresh(repeat)
    result = quality_export.compare_frozen_ucf_quality(pair.dense, repeat)
    assert result["n_videos"] == 290 and result["delta_method_minus_dense"] == 0.
    source_stage = pair.method / "provenance/stages/evaluation.json"
    stage = json.loads(source_stage.read_text(encoding="utf8"))
    stage["status"] = "failed"
    _write(source_stage, stage)
    with pytest.raises(ValueError, match="source stage is not completed"):
        quality_export._load_completed_run(repeat, "refit_head")


def test_repeat_head_routes_use_actual_seed_and_reuse_only_bound_seed0_features(frozen_pair, tmp_path, monkeypatch):
    pair = frozen_pair
    dense_repeat = _repeat_head(tmp_path / "dense-repeat1", pair.data, pair.dense_head, 1)
    method_repeat = _repeat_head(tmp_path / "method-repeat1", pair.data, pair.method_head, 1)
    def predict(config, *, feature_store, evaluation_manifest, training, output_path, device):
        source_run = pair.dense if config.declaration.evaluation_representation.reducer["name"] == "identity" else pair.method
        values = [PredictionRecord.from_dict(json.loads(line)) for line in (source_run / "predictions-primary-refit-head.jsonl").read_text(encoding="utf8").splitlines()]
        permit = validate_compatibility(config.declaration)
        values = [replace(r, run_id=Path(output_path).parent.name, metadata={**r.metadata, "checkpoint_sha256": _sha(Path(training)), "paper_compatibility": permit}) for r in values]
        _write_index(Path(output_path), [r.to_dict() for r in values])
        return values

    def evaluate(records, manifest, *, protocol, audit_report):
        return evaluate_manifest_predictions(records, manifest, protocol=protocol, audit_report=audit_report, official_source_registry=pair.data.registry)

    monkeypatch.setattr(repeat_evaluation, "predict_detector", predict)  # synthetic model scores only
    monkeypatch.setattr(repeat_evaluation, "evaluate_detector", evaluate)  # real evaluator, synthetic registry

    def execute(name, head, source_run, dense=None):
        request = repeat_evaluation.FrozenRepeatEvaluationRequest(encoder="toy", repeat_head_run=str(head.root), source_evaluation_run=str(source_run), output_root=str(tmp_path), dense_head_repeat_run=None if dense is None else str(dense.root), freeze_path=str(pair.data.freeze), role_lock_path=str(pair.data.lock), head_data_contract_path=str(pair.data.contract), source_manifest_root=str(pair.data.source_root), run_id=name)
        return Path(repeat_evaluation.run_frozen_repeat_ucf_evaluation(request).run_dir)

    dense_eval = execute("dense-eval1", dense_repeat, pair.dense)
    method_eval = execute("method-eval1", method_repeat, pair.method, dense_repeat)
    for mode in ("refit_head", "direct_insert"):
        result = quality_export.compare_frozen_ucf_quality(dense_eval, method_eval, mode=mode)
        assert result["head_seed"] == result["dense_head"]["head_seed"] == result["method_head"]["head_seed"] == 1
        assert result["method_head"]["kind"] == "frozen_head_repeat_v1"
        assert result["method_head"]["source_controller"]["head_seed"] == 0
        assert result["n_videos"] == 290 and result["delta_method_minus_dense"] == 0.
    with pytest.raises(ValueError, match="heads differ in seed"):
        quality_export.compare_frozen_ucf_quality(pair.dense, method_eval, mode="direct_insert")
    missing = execute("no-secondary", method_repeat, pair.method)
    with pytest.raises(ValueError, match="no frozen head source"):
        quality_export.compare_frozen_ucf_quality(dense_eval, missing, mode="direct_insert")


def test_repeat_secondary_may_not_use_a_different_head_seed(frozen_pair, tmp_path):
    pair = frozen_pair
    method_repeat = _repeat_head(tmp_path / "method-repeat1", pair.data, pair.method_head, 1)
    dense_repeat2 = _repeat_head(tmp_path / "dense-repeat2", pair.data, pair.dense_head, 2)
    run = _evaluation_run(tmp_path / "mismatched-secondary", pair.data, method_repeat, dense_source=dense_repeat2)
    _reuse_store(run, pair.method)
    with pytest.raises(ValueError, match="primary and secondary head seeds differ"):
        quality_export._load_completed_run(run, "direct_insert")


def test_seed_summary_never_increases_video_n_or_selects_best(frozen_pair):
    result = quality_export.compare_frozen_ucf_quality(frozen_pair.dense, frozen_pair.method)
    results = [{**result, "head_seed": seed, "delta_method_minus_dense": delta, "dense_head": {**result["dense_head"], "head_seed": seed, "checkpoint_sha256": str(seed) * 64}, "method_head": {**result["method_head"], "head_seed": seed, "checkpoint_sha256": str(seed + 3) * 64}} for seed, delta in ((0, -.003), (1, .001), (2, .002))]
    summary = quality_export.summarize_quality_seeds(results)
    assert summary["head_seed_count"] == 3 and summary["n_videos_per_seed"] == 290
    assert summary["mean_delta_across_head_seeds"] == pytest.approx(0.)
    assert summary["std_delta_across_head_seeds_ddof1"] == pytest.approx(np.std([-.003, .001, .002], ddof=1))
    assert summary["delta_range_across_head_seeds"] == [-.003, .002]
    assert summary["best_seed_selection"] is False
    with pytest.raises(ValueError, match="complete 0/1/2"):
        quality_export.summarize_quality_seeds(results[:2])
    forged = [{**result, "head_seed": seed} for seed in (0, 1, 2)]
    with pytest.raises(ValueError, match="actual paired source heads"):
        quality_export.summarize_quality_seeds(forged)


def test_cli_writes_self_contained_completed_comparison(frozen_pair, tmp_path, capsys):
    path = Path(__file__).parents[2] / "scripts/icassp2027/compare_frozen_quality.py"
    spec = importlib.util.spec_from_file_location("compare_quality_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "comparison"
    assert cli.main(["--dense-run", str(frozen_pair.dense), "--method-run", str(frozen_pair.method), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["n_videos_per_seed"] == 290
    summary = json.loads((output / "summary.json").read_text(encoding="utf8"))
    assert summary["head_seeds"] == [0]
    stages = list((output / "provenance/stages").glob("*.json"))
    assert len(stages) == 1 and json.loads(stages[0].read_text(encoding="utf8"))["status"] == "completed"


def _record(video_id, anomalous, *, spans=()):
    return VideoManifestRecord(video_id, f"{video_id}.mp4", "test", "Abuse" if anomalous else "Normal", anomalous,
                               annotations=spans, num_frames=12, fps=2.)


def _prediction(record, start, end, score):
    return PredictionRecord("fixture", record.video_id, f"{record.video_id}:{start}-{end}", start,
                            start / record.fps, end / record.fps, score, frame_start=start, frame_end=end,
                            ground_truth=record.is_anomaly, encoder_fingerprint="sha256:" + "0" * 64)


def test_gt_and_differently_split_score_boundaries_align_exactly():
    positive = _record("p", True, spans=(SupervisionAnnotation("frame", is_anomaly=True, span=TemporalSpan(3, 7, "frame")),))
    normal = _record("n", False)
    dense = [_prediction(positive, 0, 5, .2), _prediction(positive, 5, 12, .8), _prediction(normal, 0, 12, .1)]
    method = [_prediction(positive, 0, 3, .2), _prediction(positive, 3, 5, .2), _prediction(positive, 5, 7, .8), _prediction(positive, 7, 12, .8), _prediction(normal, 0, 4, .1), _prediction(normal, 4, 12, .1)]
    videos = paired_frame_intervals((positive, normal), dense, list(reversed(method)))
    assert videos[0].intervals.tolist() == [[0, 3], [3, 5], [5, 7], [7, 12]]
    assert videos[0].labels.tolist() == [0, 1, 1, 0]
    # PredictionRecord.ground_truth is video-level True for all four intervals;
    # it must never become the frame truth used in this comparison.
    np.testing.assert_array_equal(np.repeat(videos[0].labels, np.diff(videos[0].intervals).ravel()), frame_labels_for_record(positive))
    np.testing.assert_array_equal(videos[0].dense_scores, videos[0].method_scores)
    projected = project_video_prediction({"scores": [.2, .8], "intervals": [[0, 5], [5, 12]]}, 12, allow_uniform_resample=False)
    np.testing.assert_array_equal(np.repeat(videos[0].dense_scores, np.diff(videos[0].intervals).ravel()), projected)
    result = compare_paired_quality(videos, metric="frame_roc_auc", source_sha256={"synthetic": "0" * 64})
    assert result["positive_frames"] == 4 and result["n_frames"] == 24
    assert result["delta_method_minus_dense"] == result["ci_low"] == result["ci_high"] == 0


def test_second_annotation_projection_reuses_repository_floor_ceil_rules():
    positive = _record("p", True, spans=(SupervisionAnnotation("segment", is_anomaly=True, span=TemporalSpan(1.25, 2.25, "second")),))
    records = [_prediction(positive, 0, 12, .3)]
    result = paired_frame_intervals((positive,), records, records)[0]
    assert result.intervals.tolist() == [[0, 2], [2, 5], [5, 12]]
    assert result.labels.tolist() == [0, 1, 0]


@pytest.mark.parametrize("ranges", [[[0, 4], [5, 12]], [[0, 6], [5, 12]], [[0, 11]], [[0, 13]]])
def test_final_frame_gap_overlap_tail_and_out_of_bounds_fail(ranges):
    normal = _record("n", False)
    dense = [_prediction(normal, 0, 12, .1)]
    method = [_prediction(normal, a, b, .1) for a, b in ranges]
    with pytest.raises(ValueError):
        paired_frame_intervals((normal,), dense, method)


def test_video_set_and_missing_temporal_truth_fail():
    normal, positive = _record("n", False), _record("p", True)
    n = [_prediction(normal, 0, 12, .1)]
    p = [_prediction(positive, 0, 12, .8)]
    with pytest.raises(ValueError, match="exact manifest"):
        paired_frame_intervals((normal,), n, p)
    with pytest.raises(ValueError, match="弱标签"):
        paired_frame_intervals((positive,), p, p)


def test_normal_record_cannot_disguise_positive_frame_annotations():
    normal = _record("n", False)
    inconsistent = replace(normal, annotations=(SupervisionAnnotation("frame", is_anomaly=True, span=TemporalSpan(2, 3, "frame")),))
    records = [_prediction(inconsistent, 0, 12, .1)]
    videos = paired_frame_intervals((inconsistent,), records, records)
    with pytest.raises(ValueError, match="weak-normal"):
        compare_paired_quality(videos, metric="frame_roc_auc", source_sha256={"synthetic": "0" * 64})
