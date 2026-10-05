from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vadbench.data import video
from vadbench.data.manifest import VideoManifestRecord
from vadbench.data.sampling import FixedClipSample


class _FakeCapture:
    def __init__(self, backend: _FakeCV2) -> None:
        self.backend = backend
        self.position = 0
        self.opened = backend.opened
        self.released = False
        self.set_calls: list[int] = []
        self.read_calls: list[int] = []

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV
        return self.opened

    def get(self, prop: int) -> float:
        if prop == self.backend.CAP_PROP_FRAME_COUNT:
            return float(len(self.backend.frames))
        if prop == self.backend.CAP_PROP_FPS:
            return float(self.backend.fps)
        if prop == self.backend.CAP_PROP_FRAME_WIDTH:
            return float(self.backend.frames[0].shape[1])
        if prop == self.backend.CAP_PROP_FRAME_HEIGHT:
            return float(self.backend.frames[0].shape[0])
        raise AssertionError(f"unexpected property {prop}")

    def set(self, prop: int, value: float) -> bool:
        assert prop == self.backend.CAP_PROP_POS_FRAMES
        self.position = int(value)
        self.set_calls.append(self.position)
        return self.backend.seek_ok

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls.append(self.position)
        if self.position in self.backend.fail_indices:
            return False, None
        if self.position in self.backend.fail_once_indices:
            self.backend.fail_once_indices.remove(self.position)
            return False, None
        frame = self.backend.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class _FakeCV2:
    CAP_PROP_FRAME_COUNT = 1
    CAP_PROP_FPS = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 5

    def __init__(
        self,
        num_frames: int,
        *,
        fps: float = 10.0,
        height: int = 2,
        width: int = 3,
    ) -> None:
        self.fps = fps
        self.opened = True
        self.seek_ok = True
        self.fail_indices: set[int] = set()
        self.fail_once_indices: set[int] = set()
        self.captures: list[_FakeCapture] = []
        self.frames: list[np.ndarray] = []
        for index in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = index % 256  # B
            frame[..., 1] = 50  # G
            frame[..., 2] = (100 + index) % 256  # R
            self.frames.append(frame)

    def VideoCapture(self, path: str) -> _FakeCapture:  # noqa: N802 - mirrors OpenCV
        assert path
        capture = _FakeCapture(self)
        self.captures.append(capture)
        return capture


def _touch_video(tmp_path: Path, relative: str = "Abuse/Abuse001.mp4") -> Path:
    path = tmp_path / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _record(
    *,
    path: str = "Abuse/Abuse001.mp4",
    num_frames: int | None = None,
    fps: float | None = None,
) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id="Abuse001",
        path=path,
        split="test",
        category="Abuse",
        is_anomaly=True,
        num_frames=num_frames,
        fps=fps,
    )


def test_module_imports_without_cv2_and_raises_only_when_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _touch_video(tmp_path)
    monkeypatch.setattr(video, "_cv2", None)
    assert video.cv2_available() is False
    with pytest.raises(video.OpenCVUnavailableError, match="video extra"):
        video.probe_video(path)


def test_probe_video_reads_metadata_and_releases_capture(tmp_path: Path) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(12, fps=6.0, height=4, width=5)

    info = video.probe_video(path, backend=backend)

    assert info.path == path.resolve()
    assert info.num_frames == 12
    assert info.fps == 6.0
    assert (info.width, info.height) == (5, 4)
    assert info.duration_seconds == 2.0
    assert backend.captures[0].released is True
    assert backend.captures[0].read_calls == []


def test_sparse_decode_deduplicates_indices_and_converts_bgr_to_rgb(tmp_path: Path) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(10)

    frames = video.decode_rgb_frames(
        path,
        np.asarray([7, 2, 7, 2], dtype=np.int64),
        backend=backend,
    )

    capture = backend.captures[0]
    assert capture.set_calls == [2, 7]
    assert capture.read_calls == [2, 7]
    assert frames.shape == (4, 2, 3, 3)
    assert frames.dtype == np.uint8
    assert frames[0, 0, 0].tolist() == [107, 50, 7]
    assert frames[1, 0, 0].tolist() == [102, 50, 2]
    assert np.array_equal(frames[0], frames[2])


def test_sparse_decode_uses_bounded_nearby_reads_and_restores_unsorted_padding(
    tmp_path: Path,
) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(10)

    frames = video.decode_rgb_frames(path, [5, 1, 3, 3, 5], backend=backend)

    capture = backend.captures[0]
    assert capture.set_calls == [1]
    assert capture.read_calls == [1, 2, 3, 4, 5]
    assert np.asarray(frames[:, 0, 0, 2]).tolist() == [5, 1, 3, 3, 5]


def test_sparse_decode_keeps_long_jumps_as_seeks(tmp_path: Path) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(10)

    video.decode_rgb_frames(path, [0, 2, 8], backend=backend)

    capture = backend.captures[0]
    assert capture.set_calls == [0, 8]
    assert capture.read_calls == [0, 1, 2, 8]


def test_sparse_decode_falls_back_to_target_seek_after_nearby_read_failure(tmp_path: Path) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(10)
    backend.fail_once_indices.add(2)

    frames = video.decode_rgb_frames(path, [1, 3], backend=backend)

    capture = backend.captures[0]
    assert capture.set_calls == [1, 3]
    assert capture.read_calls == [1, 2, 3]
    assert np.asarray(frames[:, 0, 0, 2]).tolist() == [1, 3]


def test_sparse_decode_falls_back_to_target_seek_after_target_read_failure(tmp_path: Path) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(10)
    backend.fail_once_indices.add(3)

    frames = video.decode_rgb_frames(path, [1, 3], backend=backend)

    capture = backend.captures[0]
    assert capture.set_calls == [1, 3]
    assert capture.read_calls == [1, 2, 3, 3]
    assert np.asarray(frames[:, 0, 0, 2]).tolist() == [1, 3]


def test_sparse_decode_releases_capture_when_seek_and_read_fail(tmp_path: Path) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(10)
    backend.fail_indices.add(1)

    with pytest.raises(video.VideoIOError, match="frame=1"):
        video.decode_rgb_frames(path, [1], backend=backend)

    assert backend.captures[0].released is True


def test_build_clip_batch_preserves_padding_mask_and_deduplicates_batch(
    tmp_path: Path,
) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(8, fps=4.0)
    samples = (
        FixedClipSample((0, 2, 4, 4), (True, True, True, False)),
        FixedClipSample((1, 3, 5, 5), (True, True, True, False)),
    )

    batch = video.build_clip_batch(
        path,
        "Abuse001",
        samples,
        backend=backend,
        metadata={"clip_ids": ["a", "b"], "clip_indices": [0, 1]},
    )

    assert batch.frames.shape == (2, 4, 2, 3, 3)
    assert batch.frames.dtype == np.uint8
    assert np.asarray(batch.frame_indices).tolist() == [[0, 2, 4, 4], [1, 3, 5, 5]]
    assert np.asarray(batch.valid_mask).tolist() == [
        [True, True, True, False],
        [True, True, True, False],
    ]
    assert np.asarray(batch.timestamps_s).tolist() == [
        [0.0, 0.5, 1.0, 1.0],
        [0.25, 0.75, 1.25, 1.25],
    ]
    capture = backend.captures[0]
    assert capture.read_calls == [0, 1, 2, 3, 4, 5]


def test_fixed_segment_manifest_batch_uses_sparse_seeks_not_full_scan(tmp_path: Path) -> None:
    _touch_video(tmp_path)
    backend = _FakeCV2(320, fps=10.0)
    record = _record(num_frames=320, fps=10.0)

    batches = list(
        video.iter_fixed_segment_batches(
            [record],
            tmp_path,
            num_segments=32,
            clip_frames=2,
            frame_stride=2,
            backend=backend,
        )
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch.frames.shape == (32, 2, 2, 3, 3)
    assert batch.video_ids == ("Abuse001",) * 32
    assert batch.metadata["sampling_kind"] == "uniform_segments"
    assert batch.metadata["clip_indices"] == list(range(32))
    assert len(backend.captures) == 1
    capture = backend.captures[0]
    unique_indices = sorted(set(np.asarray(batch.frame_indices).reshape(-1)))
    assert capture.set_calls == unique_indices[::2]
    assert len(capture.read_calls) == 96
    assert len(capture.read_calls) < backend.frames.__len__()


def test_streaming_chunks_are_chronological_and_last_padding_is_invalid(
    tmp_path: Path,
) -> None:
    _touch_video(tmp_path)
    backend = _FakeCV2(5, fps=2.0)
    record = _record(num_frames=5, fps=2.0)

    batches = list(
        video.iter_streaming_chunk_batches(
            [record],
            tmp_path,
            chunk_frames=3,
            frame_stride=1,
            backend=backend,
        )
    )

    assert len(batches) == 2
    assert np.asarray(batches[0].frame_indices).tolist() == [[0, 1, 2]]
    assert np.asarray(batches[0].valid_mask).tolist() == [[True, True, True]]
    assert np.asarray(batches[1].frame_indices).tolist() == [[3, 4, 4]]
    assert np.asarray(batches[1].valid_mask).tolist() == [[True, True, False]]
    assert batches[0].metadata["is_last_chunk"] is False
    assert batches[1].metadata["is_last_chunk"] is True
    assert len(backend.captures) == 1
    assert backend.captures[0].read_calls == [0, 1, 2, 3, 4]


def test_streaming_sample_fps_controls_sparse_stride(tmp_path: Path) -> None:
    _touch_video(tmp_path)
    backend = _FakeCV2(12, fps=10.0)
    record = _record(num_frames=12, fps=10.0)

    batches = list(
        video.iter_streaming_chunk_batches(
            [record],
            tmp_path,
            chunk_frames=2,
            sample_fps=2.0,
            backend=backend,
        )
    )

    assert [np.asarray(batch.frame_indices).tolist() for batch in batches] == [
        [[0, 5]],
        [[10, 10]],
    ]
    assert np.asarray(batches[-1].valid_mask).tolist() == [[True, False]]
    assert all(batch.metadata["sampling_stride"] == 5 for batch in batches)
    assert backend.captures[0].read_calls == [0, 5, 10]


def test_manifest_metadata_mismatch_is_a_hard_error(tmp_path: Path) -> None:
    _touch_video(tmp_path)
    backend = _FakeCV2(10, fps=5.0)
    record = _record(num_frames=11, fps=5.0)
    with pytest.raises(video.VideoIOError, match="manifest num_frames"):
        list(video.iter_fixed_segment_batches([record], tmp_path, backend=backend))


def test_decode_rejects_out_of_range_frame_without_reading(tmp_path: Path) -> None:
    path = _touch_video(tmp_path)
    backend = _FakeCV2(4)
    with pytest.raises(video.VideoIOError, match="越界"):
        video.decode_rgb_frames(path, [4], backend=backend)
    assert backend.captures[0].read_calls == []
