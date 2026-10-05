from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import vadbench.smoke as smoke
from vadbench.contracts import (
    EncoderCapabilities,
    EncoderOutput,
    StreamState,
    StreamStep,
    TokenTimeline,
)


class _Info:
    num_frames = 16
    fps = 8.0
    duration_seconds = 2.0
    width = 4
    height = 3


def _batch(start: int = 0):
    from vadbench.contracts import ClipBatch

    frames = np.zeros((1, 2, 3, 4, 3), dtype=np.uint8)
    return ClipBatch(
        frames=frames,
        timestamps_s=np.asarray([[start / 8, (start + 1) / 8]], dtype=np.float64),
        video_ids=("video",),
        frame_indices=np.asarray([[start, start + 1]], dtype=np.int64),
    )


def _output(batch, value: float = 1.0):
    features = np.full((1, 1, 2), value, dtype=np.float32)
    start = np.asarray(batch.timestamps_s)[:, :1]
    end = np.asarray(batch.timestamps_s)[:, 1:2]
    frame_start = np.asarray(batch.frame_indices)[:, :1]
    return EncoderOutput(
        features=features,
        pooled=features[:, 0],
        timeline=TokenTimeline(
            start_s=start,
            end_s=end,
            source_frame_start=frame_start,
            source_frame_end=frame_start + 1,
        ),
        aux={"feature_stage": "pooled", "sequence_source": "unit"},
    )


class _Fixed:
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True, fixed_num_frames=2, min_frames=2, max_frames=2
    )

    def encode(self, batch, train=False):
        return _output(batch)


class _Stream:
    capabilities = EncoderCapabilities(supports_fixed_clip=False, supports_streaming=True)

    def init_state(self, video_id):
        return StreamState(video_id=video_id)

    def encode_step(self, chunk, state, train=False, compression=None):
        next_state = state.replace(step_index=state.step_index + 1)
        return StreamStep(
            _output(chunk, next_state.step_index),
            next_state,
            telemetry={"step": next_state.step_index},
        )

    def finalize(self, state):
        return None


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"unit-video")
    return path


def test_fixed_v2_has_batches_health_and_schema(monkeypatch, fake_video: Path) -> None:
    monkeypatch.setattr(smoke, "probe_video", lambda path: _Info())
    monkeypatch.setattr(smoke, "build_clip_batch", lambda *args, **kwargs: _batch(0))
    result = smoke.run_encoder_smoke_v2(
        {"encoder": {"adapter": "unit"}, "sampler": {"clip_frames": 2}},
        fake_video,
        project_root=fake_video.parent,
        adapter_instance=_Fixed(),
        run_id="unit-fixed",
    )
    assert result["status"] == "smoke_pass"
    assert result["input"]["batches"][0]["layout"] == "BTHWC"
    assert result["outputs"][0]["features"]["finite"] is True
    validator = pytest.importorskip("jsonschema")
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / "encoder-smoke-v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator.Draft202012Validator(schema).validate(result)


def test_stream_v2_requires_two_steps_and_records_state(monkeypatch, fake_video: Path) -> None:
    monkeypatch.setattr(smoke, "probe_video", lambda path: _Info())
    monkeypatch.setattr(
        smoke, "iter_streaming_chunk_batches", lambda *args, **kwargs: iter((_batch(0), _batch(2)))
    )
    result = smoke.run_encoder_smoke_v2(
        {
            "encoder": {"adapter": "unit"},
            "streaming": {"enabled": True, "chunk_frames": 2, "frame_stride": 1},
        },
        fake_video,
        project_root=fake_video.parent,
        adapter_instance=_Stream(),
        run_id="unit-stream",
    )
    assert result["status"] == "smoke_pass"
    assert result["streaming"]["state_steps"] == [1, 2]
    assert len(result["outputs"]) == 2
    assert result["outputs"][1]["aux"]["stream_telemetry"] == {"step": 2}


def test_v2_writer_validates_and_requires_explicit_overwrite(monkeypatch, fake_video: Path) -> None:
    monkeypatch.setattr(smoke, "probe_video", lambda path: _Info())
    monkeypatch.setattr(smoke, "build_clip_batch", lambda *args, **kwargs: _batch(0))
    first = smoke.run_encoder_smoke_v2(
        {"encoder": {"adapter": "unit"}, "sampler": {"clip_frames": 2}},
        fake_video,
        project_root=fake_video.parent,
        adapter_instance=_Fixed(),
        run_id="unit-first",
    )
    result_path = fake_video.parent / "out" / "result.json"
    smoke.write_smoke_result_v2(first, result_path, output_root=fake_video.parent / "out")
    with pytest.raises(FileExistsError, match="显式允许覆盖"):
        smoke.write_smoke_result_v2(first, result_path, output_root=fake_video.parent / "out")

    second = deepcopy(first)
    second["run_id"] = "unit-second"
    smoke.write_smoke_result_v2(
        second,
        result_path,
        output_root=fake_video.parent / "out",
        overwrite_success=True,
    )
    assert json.loads(result_path.read_text())["run_id"] == "unit-second"
    assert smoke.read_smoke_result_v2(result_path)["run_id"] == "unit-second"

    with pytest.raises(ValueError, match="不符合 v2 schema"):
        smoke.write_smoke_result_v2(
            {"status": "failed"},
            fake_video.parent / "out" / "invalid.json",
            output_root=fake_video.parent / "out",
        )
    with pytest.raises(ValueError):
        smoke.write_smoke_result_v2(
            first,
            fake_video.parent / "outside.json",
            output_root=fake_video.parent / "out",
        )
