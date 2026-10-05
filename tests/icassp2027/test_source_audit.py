from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vadbench.data.manifest import VideoManifestRecord
from vadbench.research.source_audit import (
    SourceAuditConfig,
    audit_source_candidates,
    write_contact_sheets,
    write_overview_contact_sheet,
    write_source_audit,
)


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _record(video_id: str, relative: str) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=relative,
        split="train",
        category="fixture",
        is_anomaly=False,
    )


def _pattern_frames(offset: int = 0) -> list[np.ndarray]:
    frames = []
    for index in range(36):
        y, x = np.indices((48, 64))
        frame = np.stack(
            ((x * 3 + index * 4 + offset) % 256, (y * 5 + index * 6) % 256, (x + y + index * 8) % 256),
            axis=-1,
        ).astype(np.uint8)
        frames.append(frame)
    return frames


def test_detects_true_duplicate_without_auto_source_group_and_writes_contact_sheet(tmp_path: Path):
    root = tmp_path / "videos"
    frames = _pattern_frames()
    _write_video(root / "a.mp4", frames)
    _write_video(root / "b.mp4", frames)
    _write_video(root / "different.mp4", _pattern_frames(71))
    records = [_record("a", "a.mp4"), _record("b", "b.mp4"), _record("different", "different.mp4")]

    result = audit_source_candidates(records, root, config=SourceAuditConfig())
    pairs = {(item["video_id_a"], item["video_id_b"]) for item in result.candidates}
    assert ("a", "b") in pairs
    evidence = next(item for item in result.candidates if (item["video_id_a"], item["video_id_b"]) == ("a", "b"))
    assert evidence["matching_frames"] >= 4
    assert evidence["automatic_source_group"] is None
    assert result.as_dict()["execution"] == "single_process_sequential"

    audit = write_source_audit(result, tmp_path / "audit")
    sheets = write_contact_sheets(result, records, root, tmp_path / "sheets")
    assert audit.is_file()
    assert sheets and all(path.is_file() and path.stat().st_size > 0 for path in sheets)
    overview = write_overview_contact_sheet(result, records, root, tmp_path / "overview.png")
    assert overview.is_file() and overview.stat().st_size > 0


def test_empty_black_frames_do_not_connect_every_video(tmp_path: Path):
    root = tmp_path / "videos"
    black = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(24)]
    _write_video(root / "black-a.mp4", black)
    _write_video(root / "black-b.mp4", black)
    _write_video(root / "black-c.mp4", black)
    records = [
        _record("black-a", "black-a.mp4"),
        _record("black-b", "black-b.mp4"),
        _record("black-c", "black-c.mp4"),
    ]

    result = audit_source_candidates(records, root)
    assert not result.candidates
    assert all(not frame.informative for video in result.videos for frame in video.frames)
