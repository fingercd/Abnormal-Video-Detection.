"""把显式时序标注投影为评测/强监督使用的帧标签。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from vadbench.data.manifest import SpanUnit, SupervisionScope, VideoManifestRecord


class LabelProjectionError(ValueError):
    pass


def frame_labels_for_record(record: VideoManifestRecord) -> np.ndarray:
    """生成一段视频的二值帧标签；拒绝把弱标签或 caption 当强真值。"""

    if record.num_frames is None:
        raise LabelProjectionError(f"{record.video_id}: 缺少 num_frames，无法生成帧标签")
    labels = np.zeros(record.num_frames, dtype=np.uint8)
    temporal_truth = 0
    for annotation in record.annotations:
        if annotation.scope not in {SupervisionScope.SEGMENT, SupervisionScope.FRAME}:
            continue
        if annotation.is_anomaly is None or annotation.span is None:
            continue
        temporal_truth += 1
        span = annotation.span
        if span.unit == SpanUnit.FRAME:
            start = int(span.start)
            end = int(span.end)
        elif span.unit == SpanUnit.SECOND:
            if record.fps is None:
                raise LabelProjectionError(
                    f"{record.video_id}: second annotation 需要 fps 才能投影到帧"
                )
            start = int(np.floor(float(span.start) * record.fps))
            end = int(np.ceil(float(span.end) * record.fps))
        else:  # pragma: no cover - enum exhaustiveness
            raise LabelProjectionError(f"{record.video_id}: 不支持 span unit={span.unit}")
        start = max(0, min(start, record.num_frames))
        end = max(start, min(end, record.num_frames))
        labels[start:end] = 1 if annotation.is_anomaly else 0

    if record.is_anomaly and temporal_truth == 0:
        raise LabelProjectionError(
            f"{record.video_id}: 只有 video/caption 弱标签，不能生成异常帧真值"
        )
    return labels


def frame_labels_from_manifest(
    records: Iterable[VideoManifestRecord],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for record in records:
        if record.video_id in result:
            raise LabelProjectionError(f"重复 video_id：{record.video_id}")
        result[record.video_id] = frame_labels_for_record(record)
    if not result:
        raise LabelProjectionError("manifest 为空")
    return result
