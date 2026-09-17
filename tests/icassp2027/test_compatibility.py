from __future__ import annotations

from dataclasses import replace

import pytest

from vadbench.paper.compatibility import (
    BackboneIdentity,
    CompatibilityDeclaration,
    RepresentationIdentity,
    SamplingIdentity,
    TrainingIdentity,
    feature_cache_key,
    validate_compatibility,
)


def _backbone(*, weights_digest: str = "sha256:weights-v1") -> BackboneIdentity:
    return BackboneIdentity(
        runtime_id="videomaev2",
        weights_digest=weights_digest,
        code_digest="sha256:code-v1",
        preprocessing={"frames": 16, "resolution": 224, "normalization": "imagenet"},
        readout={"kind": "mean_pool", "source": "last_hidden_state"},
    )


def _representation(
    *,
    reducer: str = "identity",
    output_dim: int = 768,
    precision: str = "float32",
    position: str = "native",
    weights_digest: str = "sha256:weights-v1",
) -> RepresentationIdentity:
    return RepresentationIdentity(
        backbone=_backbone(weights_digest=weights_digest),
        reducer={"name": reducer, "keep_ratio": 1.0 if reducer == "identity" else 0.5},
        output_dim=output_dim,
        precision=precision,
        position_strategy={"kind": position},
    )


def _sampling(*, source_digest: str, regime: str, clips: int) -> SamplingIdentity:
    return SamplingIdentity(
        source_digest=source_digest,
        regime=regime,
        frame_selection={"clips_per_video": clips, "method": "uniform"},
        window={"frames": 16},
        stride={"frames": 4},
        padding={"kind": "repeat_last"},
        projection={"kind": "frame_intervals_v1"},
    )


def _training() -> TrainingIdentity:
    return TrainingIdentity(
        head={"kind": "attention_mil", "hidden_dim": 256},
        fit_split_digest="sha256:fit-v1",
        seed=7,
        optimization={"epochs": 10, "lr": 0.001},
    )


def _declaration(
    *,
    mode: str = "direct_insert",
    train_representation: RepresentationIdentity | None = None,
    eval_representation: RepresentationIdentity | None = None,
    train_sampling: SamplingIdentity | None = None,
    eval_sampling: SamplingIdentity | None = None,
    baseline_sampling: SamplingIdentity | None = None,
    sampling_change: str = "train32_to_testdense",
) -> CompatibilityDeclaration:
    train_representation = train_representation or _representation()
    eval_representation = eval_representation or _representation(reducer="token_merge")
    train_sampling = train_sampling or _sampling(
        source_digest="sha256:fit-videos", regime="train_32", clips=32
    )
    eval_sampling = eval_sampling or _sampling(
        source_digest="sha256:test-videos", regime="test_dense", clips=128
    )
    return CompatibilityDeclaration(
        mode=mode,  # type: ignore[arg-type]
        training_representation=train_representation,
        evaluation_representation=eval_representation,
        training_sampling=train_sampling,
        evaluation_sampling=eval_sampling,
        baseline_evaluation_sampling=baseline_sampling or eval_sampling,
        training_identity=_training(),
        sampling_change=sampling_change,  # type: ignore[arg-type]
    )


def test_direct_insert_accepts_explicit_train32_to_testdense_and_only_reducer_change() -> None:
    result = validate_compatibility(_declaration())

    assert result["mode"] == "direct_insert"
    assert (
        result["training_representation_fingerprint"]
        != result["evaluation_representation_fingerprint"]
    )
    assert result["evaluation_cache_fingerprint"].startswith("sha256:")


def test_refit_head_accepts_matching_compressed_representation() -> None:
    compressed = _representation(reducer="token_merge")
    declaration = _declaration(
        mode="refit_head",
        train_representation=compressed,
        eval_representation=compressed,
    )

    assert validate_compatibility(declaration)["mode"] == "refit_head"


@pytest.mark.parametrize(
    "changed_representation, message",
    [
        (_representation(weights_digest="sha256:weights-v2"), "backbone"),
        (_representation(output_dim=512), "output_dim"),
        (_representation(precision="bfloat16"), "precision"),
        (_representation(position="interpolated"), "position_strategy"),
    ],
)
def test_direct_insert_rejects_non_reducer_representation_changes(
    changed_representation: RepresentationIdentity, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_compatibility(_declaration(eval_representation=changed_representation))


def test_rejects_mismatched_dense_and_compressed_evaluation_sampling() -> None:
    changed_baseline = _sampling(source_digest="sha256:test-videos", regime="test_dense", clips=64)
    with pytest.raises(ValueError, match="evaluation sampling"):
        validate_compatibility(_declaration(baseline_sampling=changed_baseline))


def test_sampling_change_requires_declared_train32_to_testdense_pair() -> None:
    wrong_train = _sampling(source_digest="sha256:fit-videos", regime="train_16", clips=16)
    with pytest.raises(ValueError, match="train32_to_testdense"):
        validate_compatibility(_declaration(train_sampling=wrong_train))


def test_refit_head_rejects_changed_representation() -> None:
    with pytest.raises(ValueError, match="identical training and evaluation representations"):
        validate_compatibility(_declaration(mode="refit_head"))


def test_unknown_declaration_field_and_missing_declaration_fail() -> None:
    declaration = _declaration()
    payload = declaration.to_dict() | {"unrecognized": True}
    with pytest.raises(ValueError, match="unknown fields"):
        CompatibilityDeclaration.from_mapping(payload)
    with pytest.raises(TypeError, match="CompatibilityDeclaration"):
        validate_compatibility(None)  # type: ignore[arg-type]


def test_unknown_identity_field_fails_instead_of_becoming_hash_noise() -> None:
    payload = _backbone().to_dict() | {"unrecognized": True}

    with pytest.raises(ValueError, match="unknown fields"):
        BackboneIdentity.from_mapping(payload)


def test_cache_identity_isolated_by_representation_and_sampling() -> None:
    dense = _representation()
    compressed = _representation(reducer="token_merge")
    test_dense = _sampling(source_digest="sha256:test-videos", regime="test_dense", clips=128)
    another_test = replace(test_dense, source_digest="sha256:other-test-videos")

    assert feature_cache_key(dense, test_dense) != feature_cache_key(compressed, test_dense)
    assert feature_cache_key(dense, test_dense) != feature_cache_key(dense, another_test)


@pytest.mark.parametrize(
    "field, change",
    [("window", {"frames": 8}), ("stride", {"frames": 2}), ("projection", {"kind": "changed"})],
)
def test_declared_sampling_change_does_not_permit_different_clip_semantics(field, change):
    declaration = _declaration()
    altered = replace(declaration.evaluation_sampling, **{field: change})
    declaration = replace(
        declaration, evaluation_sampling=altered, baseline_evaluation_sampling=altered
    )
    with pytest.raises(ValueError, match="clip/projection"):
        validate_compatibility(declaration)


def test_same_regime_cannot_hide_undeclared_frame_selection_change():
    train = _sampling(source_digest="fit", regime="uniform", clips=32)
    evaluation = _sampling(source_digest="test", regime="uniform", clips=16)
    with pytest.raises(ValueError, match="without an explicit declaration"):
        validate_compatibility(
            _declaration(train_sampling=train, eval_sampling=evaluation, sampling_change="none")
        )
