from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vadbench.data.features_dataset import FeatureDataset, collate_feature_batch
from vadbench.data.manifest import (
    SupervisionAnnotation,
    TemporalSpan,
    VideoManifestRecord,
)
from vadbench.features import FeatureStore, compute_encoder_fingerprint


def _fingerprint(name: str = "encoder") -> str:
    return compute_encoder_fingerprint({"adapter": name, "revision": "test"})


def _manifest(
    video_id: str,
    *,
    anomaly: bool,
    annotations=(),
    split: str = "train",
) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=f"{video_id}.mp4",
        split=split,
        category="Abuse" if anomaly else "Normal",
        is_anomaly=anomaly,
        annotations=tuple(annotations),
        num_frames=40,
        fps=10.0,
        duration_seconds=4.0,
    )


def _write_clip(
    store: FeatureStore,
    *,
    video_id: str,
    clip_index: int,
    value: float,
    anomaly: bool,
    fingerprint: str | None = None,
    tokens: int = 2,
    pooled: bool = True,
) -> None:
    start_frame, end_frame = clip_index * 10, (clip_index + 1) * 10
    features = np.stack(
        [
            np.full(3, value, dtype=np.float32),
            np.full(3, value + 2.0, dtype=np.float32),
        ][:tokens]
    )
    store.write(
        video_id=video_id,
        clip_id=f"{video_id}:clip-{clip_index}",
        clip_index=clip_index,
        encoder_fingerprint=fingerprint or _fingerprint(),
        features=features,
        pooled=np.full(3, value, dtype=np.float32) if pooled else None,
        start_s=clip_index + 0.2,
        end_s=clip_index + 0.8,
        frame_start=start_frame + 2,
        frame_end=end_frame - 2,
        timeline_start_s=np.linspace(clip_index, clip_index + 0.4, tokens),
        timeline_end_s=np.linspace(clip_index + 0.4, clip_index + 0.8, tokens),
        timeline_valid=np.ones(tokens, dtype=bool),
        source_frame_start=np.arange(tokens) * 2 + start_frame,
        source_frame_end=np.arange(tokens) * 2 + start_frame + 2,
        metadata={
            "source": {
                "sampling_kind": "uniform_segments",
                "segment_start_frames": start_frame,
                "segment_end_frames": end_frame,
                "split": "train",
                "is_anomaly": anomaly,
            }
        },
    )


def test_dataset_groups_by_video_and_sorts_clips_using_pooled_features(
    tmp_path: Path,
) -> None:
    store = FeatureStore(tmp_path / "features")
    _write_clip(store, video_id="normal", clip_index=1, value=11, anomaly=False)
    _write_clip(store, video_id="normal", clip_index=0, value=3, anomaly=False)

    dataset = FeatureDataset(store, [_manifest("normal", anomaly=False)])
    sample = dataset[0]
    np.testing.assert_array_equal(
        sample["features"],
        np.asarray([[3, 3, 3], [11, 11, 11]], dtype=np.float32),
    )
    np.testing.assert_array_equal(sample["clip_indices"], [0, 1])
    # Full MIL segment boundaries take precedence over sampled center clips.
    np.testing.assert_array_equal(sample["source_frame_start"], [0, 10])
    np.testing.assert_array_equal(sample["source_frame_end"], [10, 20])
    np.testing.assert_allclose(sample["timeline_start_s"], [0.0, 1.0])
    np.testing.assert_allclose(sample["timeline_end_s"], [1.0, 2.0])


def test_sequence_cache_is_opt_in_reuses_validated_video_and_returns_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FeatureStore(tmp_path / "features")
    manifest = [_manifest("normal", anomaly=False), _manifest("abuse", anomaly=True)]
    for record in manifest:
        for clip_index in range(2):
            _write_clip(
                store,
                video_id=record.video_id,
                clip_index=clip_index,
                value=float(clip_index),
                anomaly=record.is_anomaly,
            )

    original_load_bundle = store.load_bundle
    reads = 0

    def counted_load_bundle(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_load_bundle(*args, **kwargs)

    monkeypatch.setattr(store, "load_bundle", counted_load_bundle)
    legacy = FeatureDataset(store, manifest)
    for _epoch in range(3):
        for index in range(len(legacy)):
            legacy[index]
    assert reads == 12

    reads = 0
    cached = FeatureDataset(store, manifest, cache_sequences=True)
    first = cached[0]
    first["features"][:] = -1.0
    first["feature_valid_mask"][:] = False
    again = cached[0]
    np.testing.assert_array_equal(again["features"], [[0, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(again["feature_valid_mask"], [True, True])
    for _epoch in range(2):
        for index in range(len(cached)):
            cached[index]
    assert reads == 4


def test_sequence_cache_still_checks_first_feature_bundle_load(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    _write_clip(store, video_id="normal", clip_index=0, value=1, anomaly=False)
    record = store.records()[0]
    bundle_path = store.root / record.feature_path
    bundle_path.write_bytes(b"tampered")

    cached = FeatureDataset(
        store,
        [_manifest("normal", anomaly=False)],
        cache_sequences=True,
    )
    with pytest.raises(OSError, match="checksum mismatch"):
        cached[0]


def test_sequence_cache_rejects_token_arrays(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    _write_clip(store, video_id="normal", clip_index=0, value=1, anomaly=False)
    with pytest.raises(ValueError, match="clip-level"):
        FeatureDataset(
            store,
            [_manifest("normal", anomaly=False)],
            feature_level="token",
            cache_sequences=True,
        )


def test_clip_fallback_pools_only_valid_tokens(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    fingerprint = _fingerprint()
    store.write(
        video_id="normal",
        clip_id="normal:clip-0",
        clip_index=0,
        encoder_fingerprint=fingerprint,
        features=np.asarray([[1, 1], [9, 9]], dtype=np.float32),
        timeline_valid=np.asarray([True, False]),
        start_s=0,
        end_s=1,
        metadata={"source": {"split": "train", "is_anomaly": False}},
    )
    sample = FeatureDataset(store, [_manifest("normal", anomaly=False)])[0]
    np.testing.assert_array_equal(sample["features"], [[1, 1]])


def test_collate_pads_weak_video_bags_and_labels(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    for clip_index in range(2):
        _write_clip(
            store,
            video_id="normal",
            clip_index=clip_index,
            value=clip_index,
            anomaly=False,
        )
    _write_clip(store, video_id="abuse", clip_index=0, value=7, anomaly=True)
    dataset = FeatureDataset(
        store,
        [_manifest("normal", anomaly=False), _manifest("abuse", anomaly=True)],
    )
    batch = collate_feature_batch([dataset[0], dataset[1]], as_torch=False)
    assert batch["features"].shape == (2, 2, 3)
    np.testing.assert_array_equal(batch["valid_mask"], [[True, True], [True, False]])
    np.testing.assert_array_equal(batch["video_labels"], [0.0, 1.0])
    assert batch["video_ids"] == ("normal", "abuse")


def test_multiple_fingerprints_require_explicit_selection(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    _write_clip(store, video_id="normal", clip_index=0, value=1, anomaly=False)
    _write_clip(
        store,
        video_id="normal",
        clip_index=0,
        value=2,
        anomaly=False,
        fingerprint=_fingerprint("other"),
    )
    with pytest.raises(ValueError, match="multiple encoder fingerprints"):
        FeatureDataset(store, [_manifest("normal", anomaly=False)])
    selected = FeatureDataset(
        store,
        [_manifest("normal", anomaly=False)],
        encoder_fingerprint=_fingerprint("other"),
    )
    np.testing.assert_array_equal(selected[0]["features"], [[2, 2, 2]])


def test_strong_targets_use_explicit_frame_intervals_and_ignore_captions(
    tmp_path: Path,
) -> None:
    store = FeatureStore(tmp_path / "features")
    for clip_index in range(3):
        _write_clip(
            store,
            video_id="abuse",
            clip_index=clip_index,
            value=clip_index,
            anomaly=True,
        )
    annotations = (
        SupervisionAnnotation(
            scope="frame",
            is_anomaly=True,
            span=TemporalSpan(10, 20, "frame"),
        ),
        SupervisionAnnotation(scope="caption", text="ordinary description"),
        SupervisionAnnotation(scope="video", is_anomaly=True),
    )
    dataset = FeatureDataset(
        store,
        [_manifest("abuse", anomaly=True, annotations=annotations)],
        supervision="strong",
    )
    sample = dataset[0]
    np.testing.assert_array_equal(sample["temporal_labels"], [0, 1, 0])
    np.testing.assert_array_equal(sample["temporal_valid_mask"], [True, True, True])


def test_caption_and_video_only_strong_sample_never_becomes_negative(
    tmp_path: Path,
) -> None:
    store = FeatureStore(tmp_path / "features")
    _write_clip(store, video_id="abuse", clip_index=0, value=1, anomaly=True)
    annotations = (
        SupervisionAnnotation(scope="caption", text="timestamped description"),
        SupervisionAnnotation(scope="video", is_anomaly=True),
    )
    manifest = [_manifest("abuse", anomaly=True, annotations=annotations)]
    with pytest.raises(ValueError, match="no manifest videos"):
        FeatureDataset(store, manifest, supervision="strong")

    audit_dataset = FeatureDataset(
        store,
        manifest,
        supervision="strong",
        strong_unlabeled="mask",
    )
    sample = audit_dataset[0]
    np.testing.assert_array_equal(sample["temporal_labels"], [0])
    np.testing.assert_array_equal(sample["temporal_valid_mask"], [False])
    batch = collate_feature_batch([sample], as_torch=False)
    np.testing.assert_array_equal(batch["valid_mask"], [[False]])
    assert "video_labels" not in batch


def test_token_level_strong_targets_use_stored_source_frame_arrays(
    tmp_path: Path,
) -> None:
    store = FeatureStore(tmp_path / "features")
    _write_clip(
        store,
        video_id="abuse",
        clip_index=0,
        value=1,
        anomaly=True,
        tokens=2,
    )
    annotation = SupervisionAnnotation(
        scope="frame",
        is_anomaly=True,
        span=TemporalSpan(2, 4, "frame"),
    )
    dataset = FeatureDataset(
        store,
        [_manifest("abuse", anomaly=True, annotations=(annotation,))],
        supervision="strong",
        feature_level="token",
    )
    np.testing.assert_array_equal(dataset[0]["temporal_labels"], [0, 1])


def test_expected_clip_indices_are_a_hard_protocol_gate(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    _write_clip(store, video_id="normal", clip_index=0, value=1, anomaly=False)
    with pytest.raises(ValueError, match="must be exactly"):
        FeatureDataset(
            store,
            [_manifest("normal", anomaly=False)],
            expected_clips=2,
        )
