from __future__ import annotations

from pathlib import Path

import pytest

from vadbench.data.enrich import enrich_video_info
from vadbench.data.manifest import VideoManifestRecord
from vadbench.data.video import VideoInfo


def _record() -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id="clip",
        path="Normal/clip.mp4",
        split="test",
        category="Normal",
        is_anomaly=False,
    )


def test_enrich_uses_real_video_metadata(tmp_path: Path) -> None:
    video = tmp_path / "Normal" / "clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"placeholder")

    def fake_probe(path, *, backend=None):
        assert Path(path) == video
        assert backend == "fake"
        return VideoInfo(path=path, num_frames=100, fps=25.0, width=320, height=240)

    record = enrich_video_info([_record()], tmp_path, backend="fake", probe_fn=fake_probe)[0]
    assert record.num_frames == 100
    assert record.duration_seconds == 4.0
    assert record.metadata["video_probe"]["width"] == 320


def test_enrich_rejects_manifest_container_conflict(tmp_path: Path) -> None:
    video = tmp_path / "Normal" / "clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"placeholder")
    record = VideoManifestRecord(**{**_record().__dict__, "num_frames": 99})

    def fake_probe(path, *, backend=None):
        return VideoInfo(path=path, num_frames=100, fps=25.0, width=320, height=240)

    with pytest.raises(ValueError, match="视频元数据冲突"):
        enrich_video_info([record], tmp_path, probe_fn=fake_probe)
