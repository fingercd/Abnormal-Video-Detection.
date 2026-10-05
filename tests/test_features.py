from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from vadbench.contracts import (
    ClipBatch,
    EncoderCapabilities,
    EncoderOutput,
    StreamingVideoEncoderAdapter,
    StreamState,
    StreamStep,
    TokenTimeline,
    VideoEncoderAdapter,
)
from vadbench.engine.extract import FeatureExtractionEngine
from vadbench.features import FeatureStore, compute_encoder_fingerprint


def _fingerprint() -> str:
    return compute_encoder_fingerprint({"adapter": "synthetic", "hidden_size": 4})


def test_encoder_fingerprint_is_canonical_and_location_independent(tmp_path: Path) -> None:
    first = compute_encoder_fingerprint(
        {"name": "encoder", "depth": 12, "checkpoint_path": "C:/machine-a/model.pt"}
    )
    second = compute_encoder_fingerprint(
        {"checkpoint_path": "/machine-b/model.pt", "depth": 12, "name": "encoder"}
    )
    changed = compute_encoder_fingerprint({"name": "encoder", "depth": 24})
    assert first == second
    assert first != changed
    assert first.startswith("sha256:") and len(first) == 71

    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"weights-v1")
    with_weights = compute_encoder_fingerprint({"name": "encoder"}, checkpoint=checkpoint)
    checkpoint.write_bytes(b"weights-v2")
    assert with_weights != compute_encoder_fingerprint({"name": "encoder"}, checkpoint=checkpoint)


@pytest.mark.parametrize("storage_format", ["npz", "npy"])
def test_feature_store_round_trip_keeps_dense_arrays_out_of_jsonl(
    tmp_path: Path, storage_format: str
) -> None:
    store = FeatureStore(tmp_path / storage_format, storage_format=storage_format)
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    timeline = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    record = store.write(
        video_id="Abuse/001",
        clip_id="Abuse001:clip-0",
        clip_index=0,
        encoder_fingerprint=_fingerprint(),
        features=features,
        start_s=0.0,
        end_s=1.5,
        frame_start=0,
        frame_end=16,
        timeline_start_s=timeline,
        timeline_end_s=timeline + 0.5,
        pooled=features.mean(axis=0),
        metadata={"split": "train"},
    )

    rows = store.records()
    assert rows == [record]
    np.testing.assert_array_equal(store.load_array(record), features)
    bundle = store.load_bundle(record)
    np.testing.assert_array_equal(bundle["timeline_start_s"], timeline)
    np.testing.assert_array_equal(bundle["pooled"], features.mean(axis=0))

    raw_index = store.index_path.read_text(encoding="utf-8")
    row = json.loads(raw_index)
    assert row["feature_path"].endswith(f".{storage_format}")
    assert row["shape"] == [3, 4]
    assert "0.0, 1.0, 2.0" not in raw_index
    assert "features" not in row or isinstance(row["features"], dict)
    assert set(row["arrays"]) >= {
        "features",
        "timeline_start_s",
        "timeline_end_s",
        "pooled",
    }


def test_feature_store_upserts_index_without_partial_or_duplicate_rows(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    common = {
        "video_id": "video-1",
        "clip_id": "clip-1",
        "clip_index": 0,
        "encoder_fingerprint": _fingerprint(),
        "start_s": 0.0,
        "end_s": 1.0,
    }
    store.write(features=np.zeros((2, 3), dtype=np.float32), **common)
    replacement = store.write(features=np.ones((2, 3), dtype=np.float32), **common)
    assert len(store.records()) == 1
    np.testing.assert_array_equal(store.load_array(replacement), np.ones((2, 3), dtype=np.float32))
    assert store.index_path.read_bytes().endswith(b"\n")

    with pytest.raises(FileExistsError):
        store.write(features=np.zeros((2, 3), dtype=np.float32), overwrite=False, **common)
    # A failed no-overwrite attempt leaves the prior complete record readable.
    assert len(store.records()) == 1
    np.testing.assert_array_equal(store.load_array(replacement), np.ones((2, 3), dtype=np.float32))


def test_feature_store_rejects_tensor_like_metadata(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    with pytest.raises(TypeError, match="NPZ/NPY"):
        store.write(
            video_id="video",
            clip_id="clip",
            clip_index=0,
            encoder_fingerprint=_fingerprint(),
            features=np.zeros((1, 2), dtype=np.float32),
            start_s=0.0,
            end_s=1.0,
            metadata={"forbidden": np.zeros((128, 128), dtype=np.float32)},
        )
    assert not store.index_path.exists()


def test_feature_index_keeps_all_concurrent_writer_rows(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "features")
    fingerprint = _fingerprint()

    def write(index: int) -> None:
        store.write(
            video_id=f"video-{index % 2}",
            clip_id=f"clip-{index}",
            clip_index=index,
            encoder_fingerprint=fingerprint,
            features=np.full((2, 3), index, dtype=np.float32),
            start_s=float(index),
            end_s=float(index + 1),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(8)))
    records = store.records()
    assert len(records) == 8
    assert {record.clip_id for record in records} == {f"clip-{index}" for index in range(8)}


class _SyntheticFixedAdapter(VideoEncoderAdapter):
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        fixed_num_frames=4,
        min_frames=4,
        max_frames=4,
    )

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        del train
        batch_size = batch.batch_size
        features = np.arange(batch_size * 2 * 4, dtype=np.float32).reshape(batch_size, 2, 4)
        starts = np.asarray(batch.timestamps_s)[:, [0, 2]]
        ends = np.asarray(batch.timestamps_s)[:, [1, 3]] + 0.25
        return EncoderOutput(
            features=features,
            timeline=TokenTimeline(
                start_s=starts,
                end_s=ends,
                source_frame_start=np.tile(np.asarray([0, 2]), (batch_size, 1)),
                source_frame_end=np.tile(np.asarray([2, 4]), (batch_size, 1)),
            ),
            pooled=features.mean(axis=1),
            aux={"quality": np.ones((batch_size, 2), dtype=np.float32)},
        )


def test_extraction_engine_consumes_only_contracts_and_synthetic_frames(tmp_path: Path) -> None:
    batch = ClipBatch(
        frames=np.zeros((2, 4, 8, 8, 3), dtype=np.uint8),
        timestamps_s=np.asarray([[0.0, 0.25, 0.5, 0.75], [4.0, 4.25, 4.5, 4.75]]),
        video_ids=("normal-001", "abuse-001"),
        frame_indices=np.asarray([[0, 1, 2, 3], [100, 101, 102, 103]]),
        metadata={
            "clip_ids": ["n-0", "a-0"],
            "clip_indices": [0, 7],
            "labels": [0, 1],
            "split": "test",
        },
    )
    store = FeatureStore(tmp_path / "features")
    engine = FeatureExtractionEngine(
        adapter=_SyntheticFixedAdapter(),
        manifest={"adapter": "synthetic-fixed", "revision": "test"},
        feature_store=store,
    )
    records = engine.extract_batch(batch)
    assert [(record.video_id, record.clip_id, record.clip_index) for record in records] == [
        ("normal-001", "n-0", 0),
        ("abuse-001", "a-0", 7),
    ]
    assert records[0].shape == (2, 4)
    assert records[0].frame_start == 0 and records[0].frame_end == 4
    assert records[1].start_s == 4.0 and records[1].end_s == 5.0
    assert records[0].metadata["source"] == {"labels": 0, "split": "test"}
    assert records[1].metadata["source"] == {"labels": 1, "split": "test"}
    assert set(store.load_bundle(records[0])) >= {
        "features",
        "pooled",
        "timeline_start_s",
        "timeline_end_s",
        "source_frame_start",
        "source_frame_end",
        "aux_quality",
    }


class _SyntheticStreamingAdapter(StreamingVideoEncoderAdapter):
    capabilities = EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_token_cache=True,
        min_frames=2,
    )

    def encode(self, batch: ClipBatch, train: bool = False) -> EncoderOutput:
        raise NotImplementedError

    def init_state(self, video_id: str) -> StreamState:
        return StreamState(video_id=video_id)

    def encode_step(
        self,
        chunk: ClipBatch,
        state: StreamState,
        train: bool = False,
        compression: object | None = None,
    ) -> StreamStep:
        del train, compression
        start = float(np.asarray(chunk.timestamps_s)[0, 0])
        end = float(np.asarray(chunk.timestamps_s)[0, -1]) + 0.5
        output = EncoderOutput(
            features=np.full((1, 1, 3), state.step_index + 1, dtype=np.float32),
            timeline=TokenTimeline(
                start_s=np.asarray([[start]]),
                end_s=np.asarray([[end]]),
            ),
        )
        return StreamStep(
            output=output,
            state=state.replace(step_index=state.step_index + 1),
            telemetry={"reused_tokens": 0 if state.step_index == 0 else 1},
        )

    def finalize(self, state: StreamState) -> EncoderOutput | None:
        return None


def test_extraction_engine_streaming_assigns_stable_clip_indices(tmp_path: Path) -> None:
    chunks = [
        ClipBatch(
            frames=np.zeros((1, 2, 4, 4, 3), dtype=np.uint8),
            timestamps_s=np.asarray([[offset, offset + 0.5]]),
            video_ids=("video",),
        )
        for offset in (0.0, 1.0)
    ]
    store = FeatureStore(tmp_path / "features")
    engine = FeatureExtractionEngine(
        adapter=_SyntheticStreamingAdapter(),
        manifest={"adapter": "synthetic-streaming"},
        feature_store=store,
    )
    records = engine.extract_stream(chunks, video_id="video")
    assert [record.clip_index for record in records] == [0, 1]
    assert [record.metadata["extraction_mode"] for record in records] == [
        "streaming",
        "streaming",
    ]
    np.testing.assert_array_equal(store.load_array(records[1]), np.full((1, 3), 2.0))


def test_feature_index_rows_validate_against_published_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    store = FeatureStore(tmp_path / "features")
    store.write(
        video_id="video",
        clip_id="clip",
        clip_index=0,
        encoder_fingerprint=_fingerprint(),
        features=np.zeros((2, 4), dtype=np.float32),
        start_s=0.0,
        end_s=1.0,
    )
    schema_path = Path(__file__).parents[1] / "schemas" / "feature-index-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    row = json.loads(store.index_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(row)
