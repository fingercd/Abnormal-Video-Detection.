"""Run-scoped runtime references retain strict FeatureStore identity checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vadbench.checkpoints import sha256_file
from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.features import FeatureStore, compute_encoder_fingerprint
from vadbench.paper import detection
from vadbench.paper.compatibility import (
    BackboneIdentity,
    CompatibilityDeclaration,
    RepresentationIdentity,
    SamplingIdentity,
    TrainingIdentity,
    feature_cache_key,
)


def _case(tmp_path, *, document_change=None, reference_change=None, row_runtime_id="timesformer", inline=False):
    representation = RepresentationIdentity(
        backbone=BackboneIdentity(
            runtime_id="timesformer", weights_digest="fixture-weights", code_digest="fixture-code",
            preprocessing={"clip_frames": 4}, readout={"pooling": "mean"},
        ),
        reducer={"name": "identity"}, output_dim=4, precision="float32", position_strategy={"name": "native"},
    )
    sampling = SamplingIdentity(
        source_digest="fixture-fit", regime="train_32", frame_selection={"clips": 3},
        window={"frames": 4}, stride={"frames": 1}, padding={"kind": "strict"},
        projection={"kind": "frame_intervals_v1"},
    )
    fingerprint = compute_encoder_fingerprint({"fixture": "native-cache"})
    paper_identity = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    document = {
        "spec": {"runtime_id": "timesformer", "representation": representation.to_dict(),
                 "sampling": sampling.to_dict()},
        "runtime": {"runtime_id": "timesformer", "representation_fingerprint": representation.fingerprint,
                    "sampling_fingerprint": sampling.fingerprint,
                    "runtime_configurations": {"model_config": {"id2label": {"0": "fixture"}}}},
        "paper_identity": paper_identity,
        "encoder_fingerprint": fingerprint,
    }
    # Keep row identities independent of deliberately corrupted resolved metadata.
    document = json.loads(json.dumps(document))
    if document_change is not None:
        document_change(document)
    store = FeatureStore(tmp_path / "features")
    store.root.mkdir(parents=True, exist_ok=True)
    path = store.root / "resolved.json"
    if not inline:
        path.write_text(json.dumps(document), encoding="utf-8")
    records = tuple(VideoManifestRecord(
        video_id=f"fit-{label}", path=f"fit-{label}.mp4", split="train", category="fixture",
        is_anomaly=bool(label), num_frames=8, fps=2.0,
    ) for label in (0, 1))
    for record in records:
        for clip in range(3):
            metadata = {
                "paper_identity": paper_identity,
                "runtime": {"runtime_id": row_runtime_id, "adapter_type": "fixture", "loaded_library_versions": {}},
                "source": {"split": "train", "is_anomaly": record.is_anomaly},
            }
            if not inline:
                reference = {"base": "extraction_run", "path": "resolved.json", "sha256": sha256_file(path)}
                if reference_change is not None:
                    reference = reference_change(reference, clip)
                metadata["runtime_reference"] = reference
            store.write(
                video_id=record.video_id, clip_id=f"{record.video_id}:clip-{clip}", clip_index=clip,
                encoder_fingerprint=fingerprint, features=np.ones((1, 4), dtype=np.float32),
                pooled=np.ones(4, dtype=np.float32), start_s=clip, end_s=clip + 1,
                frame_start=clip, frame_end=clip + 1, metadata=metadata,
            )
    return store, records, representation, sampling, fingerprint


def _verify(case):
    store, records, representation, sampling, fingerprint = case
    detection._verify_feature_identity(
        store, records, encoder_fingerprint=fingerprint, representation=representation, sampling=sampling,
    )


def test_valid_reference_reads_and_hashes_one_snapshot_for_all_clips(tmp_path, monkeypatch):
    case = _case(tmp_path)
    expected_path = case[0].root / "resolved.json"
    read_bytes = Path.read_bytes
    reads = []

    def observed(path):
        if path == expected_path:
            reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observed)
    _verify(case)
    assert reads == [expected_path]


def test_legacy_inline_records_need_no_resolved_file_or_new_reference(tmp_path, monkeypatch):
    case = _case(tmp_path, inline=True)
    assert not (case[0].root / "resolved.json").exists()
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("legacy inline record read a runtime reference"))
    _verify(case)
    assert all("runtime_reference" not in row.metadata for row in case[0].iter_records())


@pytest.mark.parametrize("failure", ["missing", "tampered"])
def test_missing_or_changed_resolved_bytes_are_rejected(tmp_path, failure):
    case = _case(tmp_path)
    path = case[0].root / "resolved.json"
    if failure == "missing":
        path.unlink()
        expected = FileNotFoundError
    else:
        path.write_bytes(path.read_bytes() + b"\n")
        expected = ValueError
    with pytest.raises(expected):
        _verify(case)


@pytest.mark.parametrize("reference_change", [
    lambda ref, _clip: ref | {"base": "other_run"},
    lambda ref, _clip: ref | {"path": "../resolved.json"},
    lambda ref, _clip: ref | {"path": "C:/other/resolved.json"},
    lambda ref, _clip: ref | {"extra": "unexpected"},
    lambda _ref, _clip: None,
])
def test_reference_cannot_select_arbitrary_paths_or_unknown_contracts(tmp_path, reference_change, monkeypatch):
    case = _case(tmp_path, reference_change=reference_change)
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("invalid reference selected a file"))
    with pytest.raises(ValueError, match="exactly"):
        _verify(case)


@pytest.mark.parametrize("document_change,expected", [
    (lambda doc: doc.update(encoder_fingerprint="sha256:" + "f" * 64), "encoder fingerprint"),
    (lambda doc: doc["paper_identity"].update(sampling_fingerprint="other"), "paper identity"),
    (lambda doc: doc["spec"].update(runtime_id="videomaev2"), "runtime_id"),
    (lambda doc: doc["runtime"].update(runtime_id="videomaev2"), "runtime_id"),
    (lambda doc: doc["runtime"].update(representation_fingerprint="other"), "runtime fingerprints"),
    (lambda doc: doc["spec"]["representation"].update(reducer={"name": "pair_linear", "calibration": "other"}), "representation/sampling content"),
    (lambda doc: doc["spec"]["sampling"].update(window={"frames": 16}), "representation/sampling content"),
])
def test_wrong_run_or_changed_reducer_and_sampling_content_cannot_reuse_labels(tmp_path, document_change, expected):
    # The row reference hashes the altered file correctly: these are semantic
    # identity failures, not only a check against outdated byte checksums.
    case = _case(tmp_path, document_change=document_change)
    with pytest.raises(ValueError, match=expected):
        _verify(case)


def test_referenced_row_runtime_id_must_match_resolved_and_configuration(tmp_path):
    case = _case(tmp_path, row_runtime_id="videomaev2")
    with pytest.raises(ValueError, match="row runtime_id"):
        _verify(case)


def test_later_wrong_checksum_does_not_hide_behind_cached_first_reference(tmp_path, monkeypatch):
    case = _case(tmp_path, reference_change=lambda ref, clip: ref | ({"sha256": "f" * 64} if clip == 2 else {}))
    read_bytes = Path.read_bytes
    reads = []

    def observed(path):
        reads.append(path)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", observed)
    with pytest.raises(ValueError, match="SHA-256"):
        _verify(case)
    assert reads == [case[0].root / "resolved.json"]


def _config(case):
    _store, records, representation, sampling, fingerprint = case
    declaration = CompatibilityDeclaration(
        mode="refit_head", training_representation=representation, evaluation_representation=representation,
        training_sampling=sampling, evaluation_sampling=sampling, baseline_evaluation_sampling=sampling,
        training_identity=TrainingIdentity(
            head={"kind": "topk_mil", "k": 1}, seed=0,
            fit_split_digest="sha256:" + compute_manifest_sha256(records),
            optimization={"epochs": 1, "lr": 1e-3},
        ), sampling_change="none",
    )
    return detection.DetectionConfig(
        declaration=declaration, training_encoder_fingerprint=fingerprint,
        evaluation_encoder_fingerprint=fingerprint, head_kwargs={"k": 1},
        expected_training_clips=3, expected_evaluation_clips=3,
    )


def test_bad_reference_stops_training_and_prediction_before_model_or_checkpoint_loading(tmp_path, monkeypatch):
    case = _case(tmp_path)
    store, records, _representation, _sampling, _fingerprint = case
    config = _config(case)
    (store.root / "resolved.json").write_text("{}", encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid runtime reference reached training/prediction/checkpoint loading")

    monkeypatch.setattr(detection, "train_feature_head", forbidden)
    monkeypatch.setattr(detection, "predict_feature_head", forbidden)
    monkeypatch.setattr(detection, "issue_prediction_permit", forbidden)
    with pytest.raises(ValueError, match="SHA-256"):
        detection.train_detector(config, feature_store=store, train_manifest=records, output_dir=tmp_path / "train")
    with pytest.raises(ValueError, match="SHA-256"):
        detection.predict_detector(
            config, feature_store=store, evaluation_manifest=records, training=tmp_path / "unused.pt",
            output_path=tmp_path / "unused.jsonl",
        )


@pytest.mark.parametrize("manifest_kind", ["path", "generator"])
def test_prediction_preserves_manifest_path_and_generator_support(tmp_path, monkeypatch, manifest_kind):
    case = _case(tmp_path)
    store, records, _representation, _sampling, _fingerprint = case
    manifest = (record.to_dict() for record in records)
    if manifest_kind == "path":
        manifest = tmp_path / "fit.jsonl"
        write_manifest_jsonl(records, manifest)
    monkeypatch.setattr(detection, "issue_prediction_permit", lambda *_args: object())
    observed = []

    def predict(_settings, _store, manifest, *_args, **_kwargs):
        observed.extend(manifest)
        return []

    monkeypatch.setattr(detection, "predict_feature_head", predict)
    assert detection.predict_detector(
        _config(case), feature_store=store, evaluation_manifest=manifest,
        training=tmp_path / "unused.pt", output_path=tmp_path / "unused.jsonl",
    ) == []
    assert tuple(observed) == records
