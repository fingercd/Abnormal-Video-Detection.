"""用实际视频容器元数据补全可迁移 manifest。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from vadbench.data.manifest import VideoManifestRecord, validate_manifest
from vadbench.data.video import VideoInfo, probe_video


def enrich_video_info(
    records: Iterable[VideoManifestRecord],
    dataset_root: str | Path,
    *,
    backend: Any | None = None,
    probe_fn: Callable[..., VideoInfo] = probe_video,
) -> tuple[VideoManifestRecord, ...]:
    """以容器实际值补齐 num_frames/fps/duration，并拒绝已有元数据冲突。"""

    items = validate_manifest(records, dataset_root=dataset_root, require_files=True)
    enriched: list[VideoManifestRecord] = []
    for record in items:
        path = record.resolve_path(dataset_root)
        info = probe_fn(path, backend=backend)
        conflicts: list[str] = []
        if record.num_frames is not None and record.num_frames != info.num_frames:
            conflicts.append(f"num_frames manifest={record.num_frames} video={info.num_frames}")
        if record.fps is not None and abs(record.fps - info.fps) > 1e-3:
            conflicts.append(f"fps manifest={record.fps} video={info.fps}")
        if record.duration_seconds is not None and abs(
            record.duration_seconds - info.duration_seconds
        ) > max(0.1, 1.0 / info.fps):
            conflicts.append(
                f"duration manifest={record.duration_seconds} video={info.duration_seconds}"
            )
        if conflicts:
            raise ValueError(f"{record.video_id}: 视频元数据冲突：{'; '.join(conflicts)}")
        metadata = dict(record.metadata)
        metadata["video_probe"] = {
            "width": info.width,
            "height": info.height,
            "backend": "opencv" if backend is None else type(backend).__name__,
        }
        enriched.append(
            replace(
                record,
                num_frames=info.num_frames,
                fps=info.fps,
                duration_seconds=info.duration_seconds,
                metadata=metadata,
            )
        )
    return validate_manifest(enriched, dataset_root=dataset_root, require_files=True)
