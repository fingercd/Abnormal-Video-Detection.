from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.engine.runner import TORCH_AVAILABLE
from vadbench.features import FeatureStore, compute_encoder_fingerprint
from vadbench.paper.compatibility import (
    BackboneIdentity,
    CompatibilityDeclaration,
    RepresentationIdentity,
    SamplingIdentity,
    TrainingIdentity,
    feature_cache_key,
)
from vadbench.paper.detection import DetectionConfig, train_detector
from vadbench.paper.evaluation import (
    FrozenEvaluationRequest,
    _coverage,
    load_frozen_detector_source,
    run_frozen_ucf_evaluation,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _representation(reducer: dict[str, object]) -> RepresentationIdentity:
    return RepresentationIdentity(
        backbone=BackboneIdentity(
            "toy", "sha256:weights", "sha256:code", {"profile": "toy"}, {"kind": "mean"}
        ),
        reducer=reducer,
        output_dim=4,
        precision="float32",
        position_strategy={"kind": "native"},
    )


def _sampling(source: str, regime: str) -> SamplingIdentity:
    from vadbench.paper.extraction import _sampling_implementation_identity

    selection = {
        "kind": "uniform_full" if regime == "train_32" else "dense_sliding",
        "short_video_policy": "strict",
        "implementation": _sampling_implementation_identity(),
    }
    if regime == "train_32":
        selection["num_segments"] = 32
    else:
        selection["window_stride"] = 2
    return SamplingIdentity(
        source,
        regime,
        selection,
        {"clip_frames": 4},
        {"frame_stride": 2},
        {"kind": "forbid"},
        {"kind": "frame_intervals_v1"},
    )


def _record(video_id: str, split: str = "train") -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{video_id}.mp4",
        split=split,
        category="Normal",
        is_anomaly=False,
        num_frames=8,
        fps=2.0,
        duration_seconds=4.0,
    )


def _freeze_document() -> dict[str, object]:
    return {
        "status": "algorithm_and_budget_frozen_before_official_model_scores",
        "annotation_policy": "W",
        "methods": {
            "dense": {"reducer": "identity"},
            "same_budget_control": {"reducer": "global_uniform"},
            "training_free": {"reducer": "paired_random", "seed": 0},
            "trainable": {"reducer": "pair_linear"},
        },
        "detection_head": {
            "primary_seed": 0,
            "key_configuration_seeds": [0, 1, 2],
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "ucf_evaluation": {
            "sealed_test_manifest_sha256": "a2a66ae55db22f1a6ece0da6b87e77b8b92a0031e1aafc4c10d32aa956c7e17c",
            "sealed_audit_sha256": "d7568104f4607059fa02b1303ea5461123c15eb4b4a78e0aafd00351ee2d9e4a",
        },
        "sampling": {
            "native_clip_frames": {"toy": 4},
            "frame_stride": 2,
            "short_policy": "strict",
            "evaluation_window_stride": {"toy": 2},
        },
    }


@pytest.fixture(autouse=True)
def _synthetic_full_head_data_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a tiny but independently sealed fit/select contract for unit tests."""
    from vadbench.paper import evaluation

    fit = [
        _record("fit-normal"),
        VideoManifestRecord(
            **{**_record("fit-positive").to_dict(), "is_anomaly": True, "category": "Abuse"}
        ),
    ]
    select_source = [_record("select-normal")]
    source_root = tmp_path / "head-source"
    fit_path = write_manifest_jsonl(fit, source_root / "fit.jsonl")
    select_path = write_manifest_jsonl(select_source, source_root / "select.jsonl")
    lock = tmp_path / "role-lock.json"
    _write_json(
        lock,
        {
            "schema_version": 1,
            "basis": "complete_official_train_source_groups",
            "seed": 20260918,
            "partitions": {"fit-normal": "fit", "fit-positive": "fit", "select-normal": "select"},
        },
    )
    contract = tmp_path / "head-data-contract.json"
    _write_json(
        contract,
        {
            "schema_version": 1,
            "status": "fixed_before_official_model_scores",
            "dataset": "ucf-crime",
            "official_training_videos": 1610,
            "method_freeze_canonical_sha256": evaluation._canonical_sha256(_freeze_document()),
            "role_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "role_lock_seed": 20260918,
            "roles": {
                "fit": {
                    "videos": 2,
                    "source_manifest_sha256": hashlib.sha256(fit_path.read_bytes()).hexdigest(),
                    "source_manifest": "manifests/fit.jsonl",
                    "source_split": "train",
                    "controller_split": "train",
                },
                "select": {
                    "videos": 1,
                    "source_manifest_sha256": hashlib.sha256(select_path.read_bytes()).hexdigest(),
                    "source_manifest": "manifests/select.jsonl",
                    "source_split": "train",
                    "controller_split": "val",
                },
            },
        },
    )
    monkeypatch.setattr(
        evaluation, "_FROZEN_ROLE_LOCK_SHA256", hashlib.sha256(lock.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        evaluation,
        "_FROZEN_HEAD_DATA_CONTRACT_SHA256",
        hashlib.sha256(contract.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(evaluation, "_DEFAULT_ROLE_LOCK_PATH", lock)
    monkeypatch.setattr(evaluation, "_DEFAULT_HEAD_DATA_CONTRACT_PATH", contract)
    monkeypatch.setattr(evaluation, "_DEFAULT_HEAD_SOURCE_MANIFEST_ROOT", source_root)


def _source_run(
    tmp_path: Path, name: str, reducer: dict[str, object], *, bad_budget: bool = False
) -> tuple[Path, RepresentationIdentity]:
    root = tmp_path / name
    representation, sampling = _representation(reducer), _sampling("sha256:fit", "train_32")
    train = [_record("fit-normal"), _record("fit-positive")]
    train[1] = VideoManifestRecord(
        **{**train[1].to_dict(), "is_anomaly": True, "category": "Abuse"}
    )
    write_manifest_jsonl(train, root / "frozen" / "train.jsonl")
    training = TrainingIdentity(
        {"kind": "topk_mil", "k": 3},
        "sha256:" + compute_manifest_sha256(train),
        0,
        {"epochs": 20, "lr": 0.001},
    )
    fingerprint = compute_encoder_fingerprint({"source": name})
    checkpoint = root / "head" / "checkpoints" / "final.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "task": "weak_mil",
        "feature_level": "clip",
        "head": "topk",
        "head_kwargs": {"k": 3, "dropout": 0.0},
        "task_kwargs": {"ranking_weight": 0.0},
        "expected_clips": 32,
        "batch_size": 16,
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "seed": 0,
    }
    qa = {
        "status": "passed",
        "checkpoint": {"sha256": "", "path": str(checkpoint.resolve()), "epoch": 20, "step": 1},
        "nonzero_gradient_steps": 1,
        "changed_parameter_count": 1,
        "parameter_delta_l2": 1.0,
        "reload_parity": {
            "outputs": {"snippet_scores": {"exact_equal": True, "max_abs_difference": 0.0}}
        },
        "final_checkpoint_load": {"missing_keys": [], "unexpected_keys": []},
    }
    metadata = {
        "status": "completed",
        "encoder_fingerprint": fingerprint,
        "feature_dim": 4,
        "config": config,
        "training_qa": {
            "status": "passed",
            "nonzero_gradient_steps": 1,
            "changed_parameter_count": 1,
            "parameter_delta_l2": 1.0,
            "reload_parity_exact": True,
        },
        "paper_detector": {
            "training_representation_fingerprint": representation.fingerprint,
            "training_sampling_fingerprint": sampling.fingerprint,
            "training_feature_cache_fingerprint": feature_cache_key(representation, sampling),
            "training_identity_fingerprint": training.fingerprint,
        },
    }
    import torch

    torch.save(
        {
            "model_state_dict": {"weight": torch.ones(1)},
            "epoch": 20,
            "step": 1,
            "metadata": metadata,
        },
        checkpoint,
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    qa["checkpoint"]["sha256"] = checkpoint_sha
    _write_json(
        checkpoint.with_suffix(".pt.json"), {"sha256": checkpoint_sha, "metadata": metadata}
    )
    feature = root / "features" / "train"
    feature.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(feature)
    paper_identity = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    for record in train:
        for index in range(32):
            store.write(
                video_id=record.video_id,
                clip_id=f"{record.video_id}:{index}",
                clip_index=index,
                encoder_fingerprint=fingerprint,
                features=np.full((1, 4), float(index), dtype=np.float32),
                pooled=np.full(4, float(index), dtype=np.float32),
                start_s=float(index),
                end_s=float(index + 1),
                metadata={"paper_identity": paper_identity},
            )
    _write_json(
        feature / "status.json", {"completed": True, "feature_root": str(feature.resolve())}
    )
    _write_json(
        feature / "resolved.json",
        {
            "spec": {
                "representation": representation.to_dict(),
                "sampling": sampling.to_dict(),
                "sampling_kind": "uniform_full",
            },
            "encoder_fingerprint": fingerprint,
            "paper_identity": paper_identity,
        },
    )
    _write_json(root / "head" / "training_qa.json", qa)
    _write_json(
        root / "result.json", {"status": "completed", "checkpoint_path": str(checkpoint.resolve())}
    )
    _write_json(
        root / "provenance" / "stages" / "stage.json",
        {
            "stage": "paper_detection",
            "status": "completed",
            "config": {
                "encoder": "toy",
                "epochs": 99 if bad_budget else 20,
                "batch_size": 16,
                "learning_rate": 0.001,
                "seed": 0,
            },
        },
    )
    select = [_record("select-normal", "val")]
    write_manifest_jsonl(select, root / "frozen" / "val.jsonl")
    validation = root / "features" / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    val_store = FeatureStore(validation)
    for index in range(32):
        val_store.write(
            video_id="select-normal",
            clip_id=f"select-normal:{index}",
            clip_index=index,
            encoder_fingerprint=fingerprint,
            features=np.full((1, 4), float(index), dtype=np.float32),
            pooled=np.full(4, float(index), dtype=np.float32),
            start_s=float(index),
            end_s=float(index + 1),
            metadata={"paper_identity": paper_identity},
        )
    _write_json(
        validation / "status.json", {"completed": True, "feature_root": str(validation.resolve())}
    )
    _write_json(
        validation / "resolved.json",
        {
            "spec": {
                "representation": representation.to_dict(),
                "sampling": sampling.to_dict(),
                "sampling_kind": "uniform_full",
            },
            "encoder_fingerprint": fingerprint,
            "paper_identity": paper_identity,
        },
    )
    return root, representation


def _freeze(path: Path) -> None:
    _write_json(path, _freeze_document())


class _Adapter:
    capabilities = SimpleNamespace(fixed_num_frames=4)
    preprocess_profile = "toy"
    pooling = "mean"


def _request(
    tmp_path: Path, method: Path, dense: Path | None, manifest: Path, audit: Path, freeze: Path
) -> FrozenEvaluationRequest:
    from vadbench.paper import evaluation

    return FrozenEvaluationRequest(
        "toy",
        "cpu",
        str(tmp_path),
        str(manifest),
        str(audit),
        str(method),
        str(tmp_path / "out"),
        None if dense is None else str(dense),
        freeze_path=str(freeze),
        role_lock_path=str(evaluation._DEFAULT_ROLE_LOCK_PATH),
        head_data_contract_path=str(evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH),
        source_manifest_root=str(evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT),
        run_id="official",
    )


def _patch_complete_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    representation: RepresentationIdentity,
    *,
    tamper: bool = False,
    missing_coverage: bool = False,
) -> None:
    from vadbench.paper import evaluation

    original_digest = evaluation._digest

    def digest(path, *, name):
        value = original_digest(path, name=name)
        if name == "sealed UCF test manifest":
            return evaluation._UCF_TEST_MANIFEST_SHA256
        if name == "sealed UCF audit":
            return evaluation._UCF_AUDIT_SHA256
        return value

    monkeypatch.setattr(evaluation, "_digest", digest)
    monkeypatch.setattr(
        evaluation, "representation_from_verified_encoder", lambda **_kwargs: representation
    )
    monkeypatch.setattr(
        evaluation,
        "make_sampling_identity",
        lambda *_args, **_kwargs: _sampling("sha256:test", "test_dense"),
    )
    feature_root = tmp_path / "out" / "official" / "features" / "test"

    def extract(*_args, **_kwargs):
        store = FeatureStore(feature_root)
        if not missing_coverage:
            from vadbench.data.dense_sampling import DenseSamplingPlan

            samples = DenseSamplingPlan(clip_frames=4, frame_stride=2, window_stride=2).sample(8)
            assert len(samples) > 1
            for sample in samples:
                store.write(
                    video_id="test",
                    clip_id=f"test:{sample.clip_index}",
                    clip_index=sample.clip_index,
                    encoder_fingerprint="sha256:" + "a" * 64,
                    features=np.ones((1, 4), dtype=np.float32),
                    pooled=np.ones(4, dtype=np.float32),
                    start_s=sample.score_frame_start / 2.0,
                    end_s=sample.score_frame_end / 2.0,
                    frame_start=sample.score_frame_start,
                    frame_end=sample.score_frame_end,
                    metadata={},
                )
        if tamper:
            (tmp_path / "method" / "head" / "checkpoints" / "final.pt").write_bytes(b"tampered")
        sampling = _sampling("sha256:test", "test_dense")
        _write_json(
            feature_root / "resolved.json",
            {
                "spec": {
                    "representation": representation.to_dict(),
                    "sampling": sampling.to_dict(),
                    "sampling_kind": "dense",
                },
                "encoder_fingerprint": "sha256:" + "a" * 64,
                "runtime": {"fixture": True},
            },
        )
        _write_json(
            feature_root / "status.json",
            {
                "completed": True,
                "status": "completed",
                "failures": [],
                "feature_root": str(feature_root.resolve()),
                "encoder_fingerprint": "sha256:" + "a" * 64,
            },
        )
        return SimpleNamespace(
            completed=True, feature_root=str(feature_root), encoder_fingerprint="sha256:" + "a" * 64
        )

    monkeypatch.setattr(evaluation, "extract_pooled_features", extract)

    def predict(*_args, **kwargs):
        Path(kwargs["output_path"]).write_text('{"fixture":true}\n', encoding="utf-8")
        return [{"prediction": str(kwargs["output_path"])}]

    monkeypatch.setattr(evaluation, "predict_detector", predict)

    class Metrics:
        def to_dict(self):
            return {"metric": "fixture"}

    monkeypatch.setattr(evaluation, "evaluate_detector", lambda *_args, **_kwargs: Metrics())


def test_frozen_evaluation_reuses_two_qa_passed_heads_and_one_test_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, representation = _source_run(
        tmp_path, "method", {"name": "identity", "calibration": "none"}
    )
    dense, _ = _source_run(tmp_path, "dense", {"name": "identity", "calibration": "none"})
    test = write_manifest_jsonl([_record("test", "test")], tmp_path / "sealed.jsonl")
    audit, freeze = tmp_path / "audit.json", tmp_path / "freeze.json"
    _write_json(audit, {})
    _freeze(freeze)
    _patch_complete_run(monkeypatch, tmp_path, representation)
    result = run_frozen_ucf_evaluation(
        _request(tmp_path, method, dense, test, audit, freeze),
        adapter_factory=lambda _r: (
            _Adapter(),
            {"identity": {"adapter": "toy", "constructor": {"clip_frames": 4}}},
        ),
    )
    assert result.completed and result.secondary_predictions is not None
    resolved = json.loads((tmp_path / "out" / "official" / "resolved.json").read_text())
    assert resolved["coverage"]["input_union_complete"] is True
    assert (tmp_path / "out" / "official" / "result.json").is_file()


def test_source_head_rejects_failed_qa_and_wrong_budget(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    payload = json.loads(freeze.read_text())
    qa_failed, _ = _source_run(tmp_path, "qa", {"name": "identity", "calibration": "none"})
    _write_json(qa_failed / "head" / "training_qa.json", {"status": "failed"})
    with pytest.raises(ValueError, match="training QA"):
        load_frozen_detector_source(qa_failed, freeze=payload, expected_encoder="toy")
    wrong_budget, _ = _source_run(
        tmp_path, "budget", {"name": "identity", "calibration": "none"}, bad_budget=True
    )
    with pytest.raises(ValueError, match="training budget"):
        load_frozen_detector_source(wrong_budget, freeze=payload, expected_encoder="toy")


def test_source_head_rejects_qa_from_another_checkpoint_or_missing_update_evidence(
    tmp_path: Path,
) -> None:
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    payload = json.loads(freeze.read_text())
    source, _ = _source_run(tmp_path, "qa-binding", {"name": "identity", "calibration": "none"})
    qa_path = source / "head" / "training_qa.json"
    qa = json.loads(qa_path.read_text())
    qa["checkpoint"]["sha256"] = "0" * 64
    _write_json(qa_path, qa)
    with pytest.raises(ValueError, match="not bound"):
        load_frozen_detector_source(source, freeze=payload, expected_encoder="toy")
    qa["checkpoint"]["sha256"] = hashlib.sha256(
        (source / "head" / "checkpoints" / "final.pt").read_bytes()
    ).hexdigest()
    qa.pop("changed_parameter_count")
    _write_json(qa_path, qa)
    with pytest.raises(ValueError, match="changed_parameter_count"):
        load_frozen_detector_source(source, freeze=payload, expected_encoder="toy")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_source_reader_accepts_real_runner_final_qa_receipt(tmp_path: Path) -> None:
    import shutil

    source, representation = _source_run(
        tmp_path, "real-qa", {"name": "identity", "calibration": "none"}
    )
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    train = [
        _record("fit-normal"),
        VideoManifestRecord(
            video_id="fit-positive",
            path="fit-positive.mp4",
            split="train",
            category="Abuse",
            is_anomaly=True,
            num_frames=8,
            fps=2.0,
            duration_seconds=4.0,
        ),
    ]
    write_manifest_jsonl(train, source / "frozen" / "train.jsonl")
    sampling = _sampling("sha256:fit", "train_32")
    training = TrainingIdentity(
        {"kind": "topk_mil", "k": 3},
        "sha256:" + compute_manifest_sha256(train),
        0,
        {"epochs": 20, "lr": 0.001},
    )
    fingerprint = compute_encoder_fingerprint({"source": "real-qa"})
    declaration = CompatibilityDeclaration(
        "refit_head",
        representation,
        representation,
        sampling,
        _sampling("sha256:test", "test_dense"),
        _sampling("sha256:test", "test_dense"),
        training,
        "train32_to_testdense",
    )
    config = DetectionConfig(
        declaration,
        fingerprint,
        fingerprint,
        head="topk",
        head_kwargs={"k": 3},
        epochs=20,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=0.0,
        seed=0,
        expected_training_clips=32,
        expected_evaluation_clips=None,
    )
    store = FeatureStore(source / "features" / "train")
    paper_identity = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    for record in train:
        for index in range(32):
            value = float(record.is_anomaly) + index / 100.0
            store.write(
                video_id=record.video_id,
                clip_id=f"{record.video_id}:{index}",
                clip_index=index,
                encoder_fingerprint=fingerprint,
                features=np.full((1, 4), value, dtype=np.float32),
                pooled=np.full(4, value, dtype=np.float32),
                start_s=float(index),
                end_s=float(index + 1),
                metadata={"paper_identity": paper_identity},
            )
    shutil.rmtree(source / "head")
    trained = train_detector(
        config, feature_store=store, train_manifest=train, output_dir=source / "head", device="cpu"
    )
    _write_json(
        source / "features" / "train" / "resolved.json",
        {
            "spec": {
                "representation": representation.to_dict(),
                "sampling": sampling.to_dict(),
                "sampling_kind": "uniform_full",
            },
            "encoder_fingerprint": fingerprint,
            "paper_identity": paper_identity,
        },
    )
    _write_json(
        source / "result.json", {"status": "completed", "checkpoint_path": trained.checkpoint_path}
    )
    loaded = load_frozen_detector_source(
        source, freeze=json.loads(freeze.read_text()), expected_encoder="toy"
    )
    assert loaded.checkpoint == Path(trained.checkpoint_path)
    assert (
        json.loads((source / "head" / "training_qa.json").read_text())["checkpoint"]["sha256"]
        == loaded.source_hashes["head/checkpoints/final.pt"]
    )


def test_source_head_rejects_unfrozen_reducer_seed(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    payload = json.loads(freeze.read_text())
    source, _ = _source_run(tmp_path, "bad", {"name": "paired_random", "seed": 1})
    with pytest.raises(ValueError, match="seed"):
        load_frozen_detector_source(source, freeze=payload, expected_encoder="toy")


def test_source_head_rejects_small_or_identity_drifted_fit_select_manifests(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    payload = json.loads(freeze.read_text())
    source, _ = _source_run(tmp_path, "role-contract", {"name": "identity", "calibration": "none"})
    train_path = source / "frozen" / "train.jsonl"
    train_path.write_text(
        train_path.read_text(encoding="utf-8").replace("fit-normal.mp4", "renamed.mp4"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source fit manifest differs"):
        load_frozen_detector_source(source, freeze=payload, expected_encoder="toy")
    source, _ = _source_run(
        tmp_path, "role-contract-small", {"name": "identity", "calibration": "none"}
    )
    write_manifest_jsonl([_record("fit-normal")], source / "frozen" / "train.jsonl")
    with pytest.raises(ValueError, match="fit manifest differs"):
        load_frozen_detector_source(source, freeze=payload, expected_encoder="toy")


def test_source_head_rejects_unbound_role_lock(tmp_path: Path) -> None:
    from vadbench.paper import evaluation

    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    payload = json.loads(freeze.read_text())
    source, _ = _source_run(
        tmp_path, "binding-contract", {"name": "identity", "calibration": "none"}
    )
    lock = evaluation._DEFAULT_ROLE_LOCK_PATH
    lock.write_text(lock.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="role lock SHA-256"):
        load_frozen_detector_source(source, freeze=payload, expected_encoder="toy")


def test_source_head_rejects_unbound_head_data_contract(tmp_path: Path) -> None:
    from vadbench.paper import evaluation

    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    payload = json.loads(freeze.read_text())
    source, _ = _source_run(
        tmp_path, "contract-binding", {"name": "identity", "calibration": "none"}
    )
    contract = evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH
    contract.write_text(contract.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="head data contract SHA-256"):
        load_frozen_detector_source(source, freeze=payload, expected_encoder="toy")


def test_frozen_evaluation_rejects_actual_representation_identity_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, representation = _source_run(
        tmp_path, "method", {"name": "identity", "calibration": "none"}
    )
    test = write_manifest_jsonl([_record("test", "test")], tmp_path / "sealed.jsonl")
    audit, freeze = tmp_path / "audit.json", tmp_path / "freeze.json"
    _write_json(audit, {})
    _freeze(freeze)
    changed = RepresentationIdentity(
        representation.backbone, {"name": "global_uniform"}, 4, "float32", {"kind": "native"}
    )
    _patch_complete_run(monkeypatch, tmp_path, changed)
    with pytest.raises(ValueError, match="representation differs"):
        run_frozen_ucf_evaluation(
            _request(tmp_path, method, None, test, audit, freeze),
            adapter_factory=lambda _r: (
                _Adapter(),
                {"identity": {"adapter": "toy", "constructor": {"clip_frames": 4}}},
            ),
        )


def test_input_coverage_accepts_expected_dense_window_overlap(tmp_path: Path) -> None:
    from vadbench.data.dense_sampling import DenseSamplingPlan

    fingerprint = "sha256:" + "b" * 64
    manifest = (_record("overlap", "test"),)
    store = FeatureStore(tmp_path / "features")
    samples = DenseSamplingPlan(clip_frames=4, frame_stride=1, window_stride=2).sample(8)
    assert len(samples) > 1 and any(
        sample.score_frame_start < samples[index - 1].score_frame_end
        for index, sample in enumerate(samples)
        if index
    )
    for sample in samples:
        store.write(
            video_id="overlap",
            clip_id=f"overlap:{sample.clip_index}",
            clip_index=sample.clip_index,
            encoder_fingerprint=fingerprint,
            features=np.ones((1, 4), dtype=np.float32),
            pooled=np.ones(4, dtype=np.float32),
            start_s=sample.score_frame_start / 2.0,
            end_s=sample.score_frame_end / 2.0,
            frame_start=sample.score_frame_start,
            frame_end=sample.score_frame_end,
            metadata={},
        )
    receipt = _coverage(manifest, store.root, fingerprint)
    assert receipt["input_union_complete"] is True
    assert receipt["overlap_is_expected_before_prediction_aggregation"] is True


def test_global_uniform_descriptor_with_null_seed_reaches_reduction_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    method, representation = _source_run(
        tmp_path, "method", {"name": "global_uniform", "seed": None}
    )
    test = write_manifest_jsonl([_record("test", "test")], tmp_path / "sealed.jsonl")
    audit, freeze = tmp_path / "audit.json", tmp_path / "freeze.json"
    _write_json(audit, {})
    _freeze(freeze)
    _patch_complete_run(monkeypatch, tmp_path, representation)
    from vadbench.paper import evaluation, reduction_setup

    received: list[int] = []

    class Context:
        reducer_identity = {"name": "global_uniform", "seed": None}

    def prepare(*_args, **kwargs):
        received.append(kwargs["seed"])
        return Context(), {"status": "fixture"}

    monkeypatch.setattr(evaluation, "build_clip_batch", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(reduction_setup, "prepare_reduction", prepare)
    result = run_frozen_ucf_evaluation(
        _request(tmp_path, method, None, test, audit, freeze),
        adapter_factory=lambda _r: (
            _Adapter(),
            {"identity": {"adapter": "toy", "constructor": {"clip_frames": 4}}},
        ),
    )
    assert result.completed and received == [0]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_frozen_head_repeat_trains_only_a_new_seed_over_verified_source_features(
    tmp_path: Path,
) -> None:
    """The repeat path must reuse the sealed source inputs while emitting a new head only."""
    from vadbench.paper import evaluation
    from vadbench.paper.head_repeats import FrozenHeadRepeatRequest, run_frozen_head_repeat

    source, _ = _source_run(tmp_path, "repeat-source", {"name": "identity", "calibration": "none"})
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    before = hashlib.sha256(
        (source / "features" / "train" / "index.jsonl").read_bytes()
    ).hexdigest()

    result = run_frozen_head_repeat(
        FrozenHeadRepeatRequest(
            "toy",
            str(source),
            str(tmp_path / "repeats"),
            1,
            freeze_path=str(freeze),
            role_lock_path=str(evaluation._DEFAULT_ROLE_LOCK_PATH),
            head_data_contract_path=str(evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH),
            source_manifest_root=str(evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT),
            device="cpu",
            run_id="seed-1",
        )
    )

    receipt = json.loads((Path(result.run_dir) / "result.json").read_text(encoding="utf-8"))
    assert Path(result.checkpoint_path).is_file()
    assert receipt["seed"] == 1
    assert receipt["source_artifact_sha256"]["features/train/index.jsonl"] == before
    assert (
        hashlib.sha256((source / "features" / "train" / "index.jsonl").read_bytes()).hexdigest()
        == before
    )
    assert (
        json.loads((Path(result.run_dir) / "head" / "training_qa.json").read_text())["status"]
        == "passed"
    )


def test_frozen_head_repeat_rejects_out_of_protocol_seed_before_reading_source(
    tmp_path: Path,
) -> None:
    from vadbench.paper.head_repeats import FrozenHeadRepeatRequest

    with pytest.raises(ValueError, match="only seed 1 or 2"):
        FrozenHeadRepeatRequest("toy", "unused", str(tmp_path), 0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is optional")
def test_repeat_official_evaluation_reuses_seed0_test_store_without_adapter_or_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed repeats score the sealed cache; they never recreate the encoder path."""
    from vadbench.paper import evaluation, repeat_evaluation
    from vadbench.paper.head_repeats import FrozenHeadRepeatRequest, run_frozen_head_repeat
    from vadbench.paper.repeat_evaluation import (
        FrozenRepeatEvaluationRequest,
        run_frozen_repeat_ucf_evaluation,
    )

    method, representation = _source_run(
        tmp_path, "repeat-method", {"name": "identity", "calibration": "none"}
    )
    dense, _ = _source_run(tmp_path, "repeat-dense", {"name": "identity", "calibration": "none"})
    test = write_manifest_jsonl([_record("test", "test")], tmp_path / "sealed.jsonl")
    audit, freeze = tmp_path / "audit.json", tmp_path / "freeze.json"
    _write_json(audit, {})
    _freeze(freeze)
    _patch_complete_run(monkeypatch, tmp_path, representation)
    seed0 = run_frozen_ucf_evaluation(
        _request(tmp_path, method, dense, test, audit, freeze),
        adapter_factory=lambda _r: (
            _Adapter(),
            {"identity": {"adapter": "toy", "constructor": {"clip_frames": 4}}},
        ),
    )
    source_stage_path = next((Path(seed0.run_dir) / "provenance" / "stages").glob("*.json"))
    source_stage = json.loads(source_stage_path.read_text(encoding="utf-8"))
    source_stage["inputs"]["test_manifest"]["sha256"] = evaluation._UCF_TEST_MANIFEST_SHA256
    source_stage["inputs"]["audit_report"]["sha256"] = evaluation._UCF_AUDIT_SHA256
    _write_json(source_stage_path, source_stage)
    real_sha256_file = repeat_evaluation.sha256_file
    monkeypatch.setattr(
        repeat_evaluation,
        "sha256_file",
        lambda path: (
            evaluation._UCF_TEST_MANIFEST_SHA256
            if Path(path).resolve() == test.resolve()
            else evaluation._UCF_AUDIT_SHA256
            if Path(path).resolve() == audit.resolve()
            else real_sha256_file(path)
        ),
    )
    repeated = run_frozen_head_repeat(
        FrozenHeadRepeatRequest(
            "toy",
            str(method),
            str(tmp_path / "head-repeats"),
            1,
            freeze_path=str(freeze),
            role_lock_path=str(evaluation._DEFAULT_ROLE_LOCK_PATH),
            head_data_contract_path=str(evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH),
            source_manifest_root=str(evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT),
            device="cpu",
            run_id="seed-1",
        )
    )
    dense_repeated = run_frozen_head_repeat(
        FrozenHeadRepeatRequest(
            "toy",
            str(dense),
            str(tmp_path / "head-repeats"),
            1,
            freeze_path=str(freeze),
            role_lock_path=str(evaluation._DEFAULT_ROLE_LOCK_PATH),
            head_data_contract_path=str(evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH),
            source_manifest_root=str(evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT),
            device="cpu",
            run_id="dense-seed-1",
        )
    )
    calls: list[tuple[Path, Path]] = []

    def predict(config, *, feature_store, evaluation_manifest, training, output_path, device):
        calls.append((Path(feature_store), Path(training)))
        Path(output_path).write_text('{"fixture":true}\n', encoding="utf-8")
        return [{"fixture": True}]

    class Metrics:
        def to_dict(self):
            return {"fixture": True}

    monkeypatch.setattr(repeat_evaluation, "predict_detector", predict)
    monkeypatch.setattr(repeat_evaluation, "evaluate_detector", lambda *_args, **_kwargs: Metrics())
    monkeypatch.setattr(
        evaluation,
        "_make_adapter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("repeat evaluation must not create an adapter")
        ),
    )
    monkeypatch.setattr(
        evaluation,
        "extract_pooled_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("repeat evaluation must not extract features")
        ),
    )
    result = run_frozen_repeat_ucf_evaluation(
        FrozenRepeatEvaluationRequest(
            encoder="toy",
            repeat_head_run=repeated.run_dir,
            source_evaluation_run=seed0.run_dir,
            output_root=str(tmp_path / "repeat-evals"),
            dense_head_repeat_run=dense_repeated.run_dir,
            freeze_path=str(freeze),
            role_lock_path=str(evaluation._DEFAULT_ROLE_LOCK_PATH),
            head_data_contract_path=str(evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH),
            source_manifest_root=str(evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT),
            run_id="seed-1",
        )
    )
    cache_root = tmp_path / "out" / "official" / "features" / "test"
    assert result.completed and [checkpoint for _feature, checkpoint in calls] == [
        Path(repeated.checkpoint_path),
        Path(dense_repeated.checkpoint_path),
    ]
    assert all(feature == cache_root for feature, _checkpoint in calls)
    resolved = json.loads((Path(result.run_dir) / "resolved.json").read_text(encoding="utf-8"))
    assert resolved["reused_test_feature_store_from"]["run_dir"] == seed0.run_dir
    assert (
        resolved["dense_source"]["checkpoint_sha256"]
        == hashlib.sha256(Path(dense_repeated.checkpoint_path).read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="frozen_head_repeat"):
        run_frozen_repeat_ucf_evaluation(
            FrozenRepeatEvaluationRequest(
                encoder="toy",
                repeat_head_run=repeated.run_dir,
                source_evaluation_run=seed0.run_dir,
                output_root=str(tmp_path / "invalid-repeat-evals"),
                dense_head_repeat_run=str(dense),
                freeze_path=str(freeze),
                role_lock_path=str(evaluation._DEFAULT_ROLE_LOCK_PATH),
                head_data_contract_path=str(evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH),
                source_manifest_root=str(evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT),
                run_id="dense0",
            )
        )
    with pytest.raises(ValueError, match="does not originate"):
        run_frozen_repeat_ucf_evaluation(
            FrozenRepeatEvaluationRequest(
                encoder="toy",
                repeat_head_run=repeated.run_dir,
                source_evaluation_run=seed0.run_dir,
                output_root=str(tmp_path / "invalid-repeat-evals"),
                dense_head_repeat_run=repeated.run_dir,
                freeze_path=str(freeze),
                role_lock_path=str(evaluation._DEFAULT_ROLE_LOCK_PATH),
                head_data_contract_path=str(evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH),
                source_manifest_root=str(evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT),
                run_id="method-as-dense",
            )
        )
    wrong_seed = Path(dense_repeated.run_dir)
    wrong_result = wrong_seed / "result.json"
    payload = json.loads(wrong_result.read_text(encoding="utf-8"))
    payload["seed"] = 2
    _write_json(wrong_result, payload)
    stage_path = next((wrong_seed / "provenance" / "stages").glob("*.json"))
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["config"]["seed"] = 2
    _write_json(stage_path, stage)
    with pytest.raises(ValueError, match="training budget"):
        run_frozen_repeat_ucf_evaluation(
            FrozenRepeatEvaluationRequest(
                encoder="toy",
                repeat_head_run=repeated.run_dir,
                source_evaluation_run=seed0.run_dir,
                output_root=str(tmp_path / "invalid-repeat-evals"),
                dense_head_repeat_run=str(wrong_seed),
                freeze_path=str(freeze),
                role_lock_path=str(evaluation._DEFAULT_ROLE_LOCK_PATH),
                head_data_contract_path=str(evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH),
                source_manifest_root=str(evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT),
                run_id="dense-wrong-seed",
            )
        )
    repeat_receipt = Path(repeated.run_dir) / "result.json"
    tampered = json.loads(repeat_receipt.read_text(encoding="utf-8"))
    tampered["source_artifact_sha256"]["frozen/train.jsonl"] = "0" * 64
    _write_json(repeat_receipt, tampered)
    with pytest.raises(ValueError, match="source artifacts differ"):
        repeat_evaluation.load_frozen_repeat_head_source(
            repeated.run_dir,
            freeze=json.loads(freeze.read_text()),
            expected_encoder="toy",
            role_lock_path=evaluation._DEFAULT_ROLE_LOCK_PATH,
            head_data_contract_path=evaluation._DEFAULT_HEAD_DATA_CONTRACT_PATH,
            source_manifest_root=evaluation._DEFAULT_HEAD_SOURCE_MANIFEST_ROOT,
        )


@pytest.mark.parametrize(
    ("tamper", "missing_coverage", "message"),
    [(True, False, "changed after verification"), (False, True, "no FeatureStore rows")],
)
def test_frozen_evaluation_rejects_source_mutation_and_incomplete_dense_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: bool,
    missing_coverage: bool,
    message: str,
) -> None:
    method, representation = _source_run(
        tmp_path, "method", {"name": "identity", "calibration": "none"}
    )
    test = write_manifest_jsonl([_record("test", "test")], tmp_path / "sealed.jsonl")
    audit, freeze = tmp_path / "audit.json", tmp_path / "freeze.json"
    _write_json(audit, {})
    _freeze(freeze)
    _patch_complete_run(
        monkeypatch, tmp_path, representation, tamper=tamper, missing_coverage=missing_coverage
    )
    with pytest.raises(ValueError, match=message):
        run_frozen_ucf_evaluation(
            _request(tmp_path, method, None, test, audit, freeze),
            adapter_factory=lambda _r: (
                _Adapter(),
                {"identity": {"adapter": "toy", "constructor": {"clip_frames": 4}}},
            ),
        )
