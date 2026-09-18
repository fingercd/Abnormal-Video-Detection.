from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vadbench.artifacts import PredictionRecord
from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.manifest import VideoManifestRecord
from vadbench.engine.predict import (
    TORCH_AVAILABLE,
    _aggregate_overlapping_records,
    predict_feature_head,
)
from vadbench.engine.train import save_checkpoint
from vadbench.features import FeatureStore, compute_encoder_fingerprint
from vadbench.paper.compatibility import (
    BackboneIdentity,
    CompatibilityDeclaration,
    RepresentationIdentity,
    SamplingIdentity,
    TrainingIdentity,
    feature_cache_key,
)
from vadbench.paper.detection import (
    DetectionConfig,
    issue_prediction_permit,
    predict_detector,
    train_detector,
)
from vadbench.tasks import build_task

if TORCH_AVAILABLE:
    import torch


def _manifest(video_id: str, *, split: str, anomaly: bool, frames: int = 8) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{video_id}.mp4",
        split=split,
        category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly,
        num_frames=frames,
        fps=2.0,
        duration_seconds=frames / 2.0,
    )


def _representation(*, reducer: dict[str, object]) -> RepresentationIdentity:
    return RepresentationIdentity(
        backbone=BackboneIdentity(
            runtime_id="videomaev2",
            weights_digest="sha256:encoder-weights",
            code_digest="sha256:encoder-code",
            preprocessing={"frames": 4, "resolution": 224},
            readout={"kind": "pooled", "source": "last_hidden_state"},
        ),
        reducer=reducer,
        output_dim=4,
        precision="float32",
        position_strategy={"kind": "native"},
    )


def _sampling(*, source: str, regime: str, clips: int) -> SamplingIdentity:
    return SamplingIdentity(
        source_digest=source,
        regime=regime,
        frame_selection={"clips_per_video": clips, "method": "fixed"},
        window={"frames": 4},
        stride={"frames": 1},
        padding={"kind": "repeat_last"},
        projection={"kind": "frame_intervals_v1"},
    )


def _declaration() -> CompatibilityDeclaration:
    train = _representation(reducer={"name": "identity", "calibration": "none"})
    evaluation = _representation(reducer={"name": "token_merge", "calibration": "fit-normal-v1"})
    train_sampling = _sampling(source="sha256:fit", regime="train_32", clips=32)
    test_sampling = _sampling(source="sha256:test", regime="test_dense", clips=8)
    return CompatibilityDeclaration(
        mode="direct_insert",
        training_representation=train,
        evaluation_representation=evaluation,
        training_sampling=train_sampling,
        evaluation_sampling=test_sampling,
        baseline_evaluation_sampling=test_sampling,
        training_identity=TrainingIdentity(
            head={"kind": "topk_mil", "k": 1},
            fit_split_digest="sha256:fit",
            seed=13,
            optimization={"epochs": 2, "lr": 0.05},
        ),
        sampling_change="train32_to_testdense",
    )


def _config(source: str, target: str) -> DetectionConfig:
    return DetectionConfig(
        declaration=_declaration(),
        training_encoder_fingerprint=source,
        evaluation_encoder_fingerprint=target,
        head="topk",
        head_kwargs={"k": 1},
        epochs=2,
        batch_size=2,
        learning_rate=0.05,
        max_steps=2,
        seed=13,
        expected_training_clips=32,
        expected_evaluation_clips=None,
    )


def _with_fit_digest(config: DetectionConfig, records: list[VideoManifestRecord]) -> DetectionConfig:
    training_identity = replace(
        config.declaration.training_identity,
        fit_split_digest="sha256:" + compute_manifest_sha256(records),
    )
    return replace(config, declaration=replace(config.declaration, training_identity=training_identity))


def test_paper_detector_enables_validated_training_sequence_cache() -> None:
    fingerprint = compute_encoder_fingerprint({"adapter": "paper-head-cache"})
    settings = _config(fingerprint, fingerprint).training_config()
    assert settings.cache_sequences is True
    assert settings.feature_level == "clip"
    assert settings.num_workers == 0


def _identity(declaration: CompatibilityDeclaration, *, training: bool) -> dict[str, str]:
    representation = (
        declaration.training_representation if training else declaration.evaluation_representation
    )
    sampling = declaration.training_sampling if training else declaration.evaluation_sampling
    identity = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
    }
    identity["feature_cache_fingerprint"] = feature_cache_key(representation, sampling)
    return identity


def _write_features(
    store: FeatureStore,
    videos: list[VideoManifestRecord],
    *,
    fingerprint: str,
    declaration: CompatibilityDeclaration,
    training: bool,
) -> None:
    identity = _identity(declaration, training=training)
    for video in videos:
        count = 32 if training else 5
        for index in range(count):
            if training:
                start, end = index % 8, index % 8 + 1
            else:
                start, end = [(0, 3), (1, 5), (3, 7), (5, 8), (7, 8)][index]
            value = float(video.is_anomaly) * 2.0 + index / max(count, 1)
            store.write(
                video_id=video.video_id,
                clip_id=f"{video.video_id}:clip-{index}",
                clip_index=index,
                encoder_fingerprint=fingerprint,
                features=np.full((1, 4), value, dtype=np.float32),
                pooled=np.full(4, value, dtype=np.float32),
                start_s=start / float(video.fps),
                end_s=end / float(video.fps),
                frame_start=start,
                frame_end=end,
                metadata={
                    "source": {"split": video.split.value, "is_anomaly": video.is_anomaly},
                    "paper_identity": identity,
                },
            )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_direct_insert_train32_to_dense_executes_with_bound_permit(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    source = compute_encoder_fingerprint({"adapter": "source-train32"})
    target = compute_encoder_fingerprint({"adapter": "target-dense"})
    config = _config(source, target)
    train = [
        _manifest("fit-normal", split="train", anomaly=False),
        _manifest("fit-abnormal", split="train", anomaly=True),
    ]
    config = _with_fit_digest(config, train)
    test = [_manifest("test-video", split="test", anomaly=True)]
    _write_features(store, train, fingerprint=source, declaration=config.declaration, training=True)
    _write_features(store, test, fingerprint=target, declaration=config.declaration, training=False)

    torch.manual_seed(config.seed)
    initial = torch.cat(
        [
            parameter.detach().flatten()
            for parameter in build_task(
                "weak_mil",
                None,
                feature_dim=4,
                head="topk",
                head_kwargs={"k": 1},
                task_kwargs={"ranking_weight": 0.0},
            ).parameters()
        ]
    )
    training = train_detector(
        config,
        feature_store=store,
        train_manifest=train,
        output_dir=tmp_path / "run",
        device="cpu",
    )
    assert np.isfinite(training.history["epochs"][-1]["train"]["loss"])
    trained = torch.cat([parameter.detach().flatten() for parameter in training.model.parameters()])
    assert not torch.equal(initial, trained)

    with pytest.raises(ValueError, match="no features match encoder_fingerprint"):
        predict_feature_head(
            config.training_config(),
            store,
            test,
            training.checkpoint_path,
            tmp_path / "legacy.jsonl",
            device="cpu",
        )

    records = predict_detector(
        config,
        feature_store=store,
        evaluation_manifest=test,
        training=training,
        output_path=tmp_path / "dense.jsonl",
        device="cpu",
    )
    expanded = [
        (frame, item.anomaly_score)
        for item in records
        for frame in range(item.frame_start, item.frame_end)
    ]
    assert [frame for frame, _score in expanded] == list(range(8))
    assert all(item.predicted_label == (item.anomaly_score >= 0.5) for item in records)
    assert all(item.ground_truth is True for item in records)
    assert all(item.metadata["ground_truth_scope"] == "video" for item in records)
    assert all(item.metadata["dense_aggregation"]["reduction"] == "mean" for item in records)
    assert all(item.metadata["paper_compatibility"]["mode"] == "direct_insert" for item in records)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_one_trained_head_issues_independent_permits_for_multiple_direct_targets(
    tmp_path: Path,
) -> None:
    store = FeatureStore(tmp_path / "features")
    source = compute_encoder_fingerprint({"adapter": "source"})
    target_one = compute_encoder_fingerprint({"adapter": "target-one"})
    target_two = compute_encoder_fingerprint({"adapter": "target-two"})
    train = [
        _manifest("fit-normal", split="train", anomaly=False),
        _manifest("fit-abnormal", split="train", anomaly=True),
    ]
    first = _with_fit_digest(_config(source, target_one), train)
    second_sampling = _sampling(source="sha256:second-evaluation", regime="test_dense", clips=8)
    second_declaration = replace(
        first.declaration,
        evaluation_representation=_representation(
            reducer={"name": "uniform_keep", "calibration": "none"}
        ),
        evaluation_sampling=second_sampling,
        baseline_evaluation_sampling=second_sampling,
    )
    second = replace(
        first,
        declaration=second_declaration,
        evaluation_encoder_fingerprint=target_two,
    )
    evaluation_one = [_manifest("evaluation-one", split="test", anomaly=True)]
    evaluation_two = [_manifest("evaluation-two", split="test", anomaly=False)]
    _write_features(store, train, fingerprint=source, declaration=first.declaration, training=True)
    _write_features(
        store, evaluation_one, fingerprint=target_one, declaration=first.declaration, training=False
    )
    _write_features(
        store, evaluation_two, fingerprint=target_two, declaration=second.declaration, training=False
    )

    training = train_detector(
        first, feature_store=store, train_manifest=train, output_dir=tmp_path / "run", device="cpu"
    )
    checkpoint_before = Path(training.checkpoint_path).read_bytes()
    sidecar = Path(training.checkpoint_path).with_suffix(".pt.json")
    sidecar_before = sidecar.read_bytes()
    records_one = predict_detector(
        first,
        feature_store=store,
        evaluation_manifest=evaluation_one,
        training=training,
        output_path=tmp_path / "one.jsonl",
        device="cpu",
    )
    records_two = predict_detector(
        second,
        feature_store=store,
        evaluation_manifest=evaluation_two,
        training=training,
        output_path=tmp_path / "two.jsonl",
        device="cpu",
    )

    assert records_one and records_two
    assert Path(training.checkpoint_path).read_bytes() == checkpoint_before
    assert sidecar.read_bytes() == sidecar_before
    assert records_one[0].metadata["paper_compatibility"]["evaluation_sampling_fingerprint"] != (
        records_two[0].metadata["paper_compatibility"]["evaluation_sampling_fingerprint"]
    )
    assert records_one[0].metadata["paper_compatibility"][
        "evaluation_representation_fingerprint"
    ] != records_two[0].metadata["paper_compatibility"]["evaluation_representation_fingerprint"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_permit_rejects_tampered_checkpoint_and_target_identity(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    source = compute_encoder_fingerprint({"adapter": "source"})
    target = compute_encoder_fingerprint({"adapter": "target"})
    config = _config(source, target)
    train = [
        _manifest("fit-n", split="train", anomaly=False),
        _manifest("fit-a", split="train", anomaly=True),
    ]
    config = _with_fit_digest(config, train)
    test = [_manifest("test", split="test", anomaly=True)]
    _write_features(store, train, fingerprint=source, declaration=config.declaration, training=True)
    _write_features(store, test, fingerprint=target, declaration=config.declaration, training=False)
    training = train_detector(
        config, feature_store=store, train_manifest=train, output_dir=tmp_path / "run", device="cpu"
    )
    permit = issue_prediction_permit(config, training)

    with Path(training.checkpoint_path).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        predict_feature_head(
            config.training_config(),
            store,
            test,
            training.checkpoint_path,
            tmp_path / "bad.jsonl",
            device="cpu",
            compatibility_permit=permit,
            overlap_reduction="mean",
        )

    # A fresh valid checkpoint still rejects FeatureStore rows whose paper
    # identity names a different reducer calibration.
    training = train_detector(
        config,
        feature_store=store,
        train_manifest=train,
        output_dir=tmp_path / "run-2",
        device="cpu",
    )
    permit = issue_prediction_permit(config, training)
    record = next(item for item in store.records() if item.video_id == "test")
    # Rewriting one target row preserves the ordinary cache fingerprint but changes its declared calibration.
    bundle = store.load_bundle(record)
    bad_identity = _identity(config.declaration, training=False) | {"calibration": "other"}
    store.write(
        video_id=record.video_id,
        clip_id=record.clip_id,
        clip_index=record.clip_index,
        encoder_fingerprint=record.encoder_fingerprint,
        features=bundle["features"],
        pooled=bundle["pooled"],
        start_s=record.start_s,
        end_s=record.end_s,
        frame_start=record.frame_start,
        frame_end=record.frame_end,
        metadata={"source": {"split": "test", "is_anomaly": True}, "paper_identity": bad_identity},
    )
    with pytest.raises(ValueError, match="target FeatureStore identity"):
        predict_feature_head(
            config.training_config(),
            store,
            test,
            training.checkpoint_path,
            tmp_path / "bad-identity.jsonl",
            device="cpu",
            compatibility_permit=permit,
            overlap_reduction="mean",
        )


def test_config_rejects_unapproved_identity_or_test_training_leakage(tmp_path: Path) -> None:
    source = compute_encoder_fingerprint({"adapter": "source"})
    target = compute_encoder_fingerprint({"adapter": "target"})
    declaration = _declaration()
    with pytest.raises(ValueError, match="output_dim"):
        DetectionConfig(
            declaration=replace(
                declaration,
                evaluation_representation=replace(
                    declaration.evaluation_representation, output_dim=8
                ),
            ),
            training_encoder_fingerprint=source,
            evaluation_encoder_fingerprint=target,
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is an optional dependency")
def test_permit_rejects_legacy_checkpoint_without_paper_source_identity(tmp_path: Path) -> None:
    source = compute_encoder_fingerprint({"adapter": "source"})
    target = compute_encoder_fingerprint({"adapter": "target"})
    config = _config(source, target)
    checkpoint = tmp_path / "legacy.pt"
    save_checkpoint(
        checkpoint,
        build_task("weak_mil", None, feature_dim=4, head="topk", head_kwargs={"k": 1}),
        metadata={
            "task": "wsvad",
            "encoder_fingerprint": source,
            "feature_dim": 4,
            "feature_level": "clip",
        },
    )

    with pytest.raises(
        ValueError, match="paper detector training/representation/sampling identity"
    ):
        issue_prediction_permit(config, checkpoint)

    store = FeatureStore(tmp_path / "features")
    config = _config(source, target)
    with pytest.raises(ValueError, match="only train"):
        train_detector(
            config,
            feature_store=store,
            train_manifest=[_manifest("test", split="test", anomaly=True)],
            output_dir=tmp_path / "run",
            device="cpu",
        )


def test_train_detector_rejects_manifest_not_matching_training_identity_digest(
    tmp_path: Path,
) -> None:
    source = compute_encoder_fingerprint({"adapter": "source"})
    target = compute_encoder_fingerprint({"adapter": "target"})
    declared_fit = [
        _manifest("fit-normal", split="train", anomaly=False),
        _manifest("fit-abnormal", split="train", anomaly=True),
    ]
    actual_fit = [
        _manifest("other-normal", split="train", anomaly=False),
        _manifest("other-abnormal", split="train", anomaly=True),
    ]
    config = _with_fit_digest(_config(source, target), declared_fit)

    with pytest.raises(ValueError, match="fit_split_digest"):
        train_detector(
            config,
            feature_store=FeatureStore(tmp_path / "features"),
            train_manifest=actual_fit,
            output_dir=tmp_path / "run",
            device="cpu",
        )


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("weights", "backbone"),
        ("readout", "backbone"),
        ("sampling", "clip/projection"),
    ],
)
def test_config_rejects_unapproved_weight_readout_and_sampling_changes(
    changed: str, message: str
) -> None:
    source = compute_encoder_fingerprint({"adapter": "source"})
    target = compute_encoder_fingerprint({"adapter": "target"})
    declaration = _declaration()
    if changed == "weights":
        evaluation = replace(
            declaration.evaluation_representation,
            backbone=replace(
                declaration.evaluation_representation.backbone,
                weights_digest="sha256:other-weights",
            ),
        )
        declaration = replace(declaration, evaluation_representation=evaluation)
    elif changed == "readout":
        evaluation = replace(
            declaration.evaluation_representation,
            backbone=replace(
                declaration.evaluation_representation.backbone,
                readout={"kind": "other"},
            ),
        )
        declaration = replace(declaration, evaluation_representation=evaluation)
    else:
        evaluation_sampling = replace(declaration.evaluation_sampling, window={"frames": 8})
        declaration = replace(
            declaration,
            evaluation_sampling=evaluation_sampling,
            baseline_evaluation_sampling=evaluation_sampling,
        )
    with pytest.raises(ValueError, match=message):
        DetectionConfig(
            declaration=declaration,
            training_encoder_fingerprint=source,
            evaluation_encoder_fingerprint=target,
        )


def test_dense_aggregation_rle_is_exact_and_retains_video_level_truth_scope() -> None:
    manifest = _manifest("video", split="test", anomaly=True, frames=6)
    raw = [
        PredictionRecord(
            run_id="run",
            video_id="video",
            clip_id=f"clip-{index}",
            clip_index=index,
            start_s=start / 2,
            end_s=end / 2,
            frame_start=start,
            frame_end=end,
            anomaly_score=score,
            ground_truth=True,
            encoder_fingerprint=compute_encoder_fingerprint({"adapter": "fixture"}),
            metadata={"ground_truth_scope": "video"},
        )
        for index, (start, end, score) in enumerate(((0, 2, 0.2), (2, 4, 0.2), (4, 6, 0.8)))
    ]

    compressed = _aggregate_overlapping_records(
        raw, {"video": manifest}, reduction="mean", strict_coverage=True
    )

    assert [(item.frame_start, item.frame_end) for item in compressed] == [(0, 4), (4, 6)]
    expanded = [
        item.anomaly_score
        for item in compressed
        for _frame in range(item.frame_start, item.frame_end)
    ]
    assert expanded == [0.2, 0.2, 0.2, 0.2, 0.8, 0.8]
    assert all(
        item.metadata["dense_aggregation"]["interval_source"] == "exact_equal_frame_run"
        for item in compressed
    )
    assert all(item.metadata["ground_truth_scope"] == "video" for item in compressed)
