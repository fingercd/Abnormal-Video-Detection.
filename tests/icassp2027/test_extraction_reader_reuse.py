"""Reader reuse preserves the existing dense extraction data contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from test_paper_extraction import _Adapter, _record, _representation, _sampling, _spec, _touch

from vadbench.data.sampling import FixedClipSample
from vadbench.data.video import OpenCVVideoReader, _batch_from_reader, build_clip_batch
from vadbench.paper.extraction import extract_pooled_features


class _Capture:
    def __init__(self, backend: _TrackingCV2) -> None:
        self.backend = backend
        self.position = 0
        self.released = False
        self.read_calls: list[int] = []

    def isOpened(self) -> bool:  # noqa: N802
        return True

    def get(self, property_id: int) -> float:
        values = {
            self.backend.CAP_PROP_FRAME_COUNT: len(self.backend.frames),
            self.backend.CAP_PROP_FPS: self.backend.fps,
            self.backend.CAP_PROP_FRAME_WIDTH: 3,
            self.backend.CAP_PROP_FRAME_HEIGHT: 2,
        }
        return float(values[property_id])

    def set(self, _property_id: int, value: float) -> bool:
        self.position = int(value)
        return True

    def read(self):
        self.read_calls.append(self.position)
        if self.position >= len(self.backend.frames):
            return False, None
        frame = self.backend.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class _TrackingCV2:
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_FPS = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 5

    def __init__(self, count: int = 64, fps: float = 8.0) -> None:
        self.fps = fps
        self.frames = []
        for index in range(count):
            frame = np.zeros((2, 3, 3), dtype=np.uint8)
            frame[..., 0] = index
            frame[..., 1] = (index * 3) % 256
            frame[..., 2] = (255 - index) % 256
            self.frames.append(frame)
        self.captures: list[_Capture] = []

    def VideoCapture(self, _path: str) -> _Capture:  # noqa: N802
        capture = _Capture(self)
        self.captures.append(capture)
        return capture


def _assert_same_batch(actual, expected) -> None:
    assert actual.video_ids == expected.video_ids
    assert actual.metadata == expected.metadata
    assert np.array_equal(actual.frames, expected.frames)
    assert np.array_equal(actual.timestamps_s, expected.timestamps_s)
    assert np.array_equal(actual.frame_indices, expected.frame_indices)
    assert np.array_equal(actual.valid_mask, expected.valid_mask)


def test_reused_reader_matches_legacy_batch_decoding_for_overlapping_noncontiguous_groups(tmp_path):
    path = tmp_path / "video.mp4"
    path.touch()
    groups = (
        (
            FixedClipSample((0, 2, 4, 4), (True, True, True, False)),
            FixedClipSample((1, 3, 5, 5), (True, True, True, False)),
        ),
        (
            FixedClipSample((13, 15, 17, 17), (True, True, True, False)),
            FixedClipSample((14, 16, 18, 18), (True, True, True, False)),
        ),
    )
    metadata = {"source_num_frames": 64, "source_fps": 8.0}
    legacy_backend = _TrackingCV2()
    legacy = [
        build_clip_batch(path, "sample-video", group, backend=legacy_backend, metadata=metadata)
        for group in groups
    ]
    reused_backend = _TrackingCV2()
    with OpenCVVideoReader(path, backend=reused_backend) as reader:
        reused = [
            _batch_from_reader(reader, "sample-video", group, metadata=metadata)
            for group in groups
        ]

    assert len(legacy_backend.captures) == len(groups)
    assert len(reused_backend.captures) == 1 and reused_backend.captures[0].released
    for actual, expected in zip(reused, legacy, strict=True):
        _assert_same_batch(actual, expected)


def _run_extraction(tmp_path: Path, *, run_id: str, backend, adapter=None, resume_source=None):
    records = [_record("video", split="train", anomaly=False)]
    _touch(tmp_path, records)
    selected_adapter = _Adapter() if adapter is None else adapter
    representation = _representation(selected_adapter, reducer="identity")
    sampling = _sampling(records, root=tmp_path, regime="train_32", clips=32)
    return extract_pooled_features(
        _spec(representation, sampling, kind="uniform_full"),
        adapter=selected_adapter,
        manifest=records,
        dataset_root=tmp_path,
        output_root=tmp_path / "runs",
        run_id=run_id,
        backend=backend,
        resume_source=resume_source,
    ), selected_adapter


def test_extraction_opens_one_reader_per_decoded_video_and_closes_it(tmp_path):
    backend = _TrackingCV2()
    result, adapter = _run_extraction(tmp_path, run_id="decoded", backend=backend)

    assert result.completed and len(adapter.seen_batches) > 1
    assert len(backend.captures) == 1
    assert backend.captures[0].released


def test_failed_encode_closes_reused_reader(tmp_path):
    class _FailingAdapter(_Adapter):
        def encode(self, batch, train=False):
            raise RuntimeError("fixture encode failure")

    backend = _TrackingCV2()
    result, _adapter = _run_extraction(
        tmp_path, run_id="failed", backend=backend, adapter=_FailingAdapter()
    )

    assert not result.completed and result.failures[0]["type"] == "RuntimeError"
    assert len(backend.captures) == 1 and backend.captures[0].released


def test_resumed_complete_video_keeps_zero_decode_and_closes_probe_reader(tmp_path):
    source, _ = _run_extraction(tmp_path, run_id="source", backend=_TrackingCV2())
    assert source.completed
    backend = _TrackingCV2()
    resumed, adapter = _run_extraction(
        tmp_path, run_id="resumed", backend=backend, resume_source=source.run_dir
    )

    assert resumed.completed and adapter.seen_batches == []
    assert len(backend.captures) == 1
    assert backend.captures[0].read_calls == [] and backend.captures[0].released
