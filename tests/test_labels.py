from __future__ import annotations

import numpy as np
import pytest

from vadbench.data.labels import LabelProjectionError, frame_labels_for_record
from vadbench.data.manifest import (
    SupervisionAnnotation,
    SupervisionScope,
    TemporalSpan,
    VideoManifestRecord,
)


def _record(*, is_anomaly: bool, annotations=()):
    return VideoManifestRecord(
        video_id="video",
        path="category/video.mp4",
        split="test",
        category="Abuse" if is_anomaly else "Normal",
        is_anomaly=is_anomaly,
        annotations=annotations,
        num_frames=10,
        fps=2.0,
    )


def test_frame_annotation_projects_half_open_interval() -> None:
    annotation = SupervisionAnnotation(
        scope=SupervisionScope.FRAME,
        label="anomaly",
        is_anomaly=True,
        span=TemporalSpan(start=2, end=5, unit="frame"),
    )
    assert np.array_equal(
        frame_labels_for_record(_record(is_anomaly=True, annotations=(annotation,))),
        np.asarray([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8),
    )


def test_caption_is_never_treated_as_temporal_ground_truth() -> None:
    caption = SupervisionAnnotation(
        scope=SupervisionScope.CAPTION,
        text="person walks",
        is_anomaly=None,
        span=TemporalSpan(start=0.0, end=1.0, unit="second"),
    )
    with pytest.raises(LabelProjectionError, match="不能生成异常帧真值"):
        frame_labels_for_record(_record(is_anomaly=True, annotations=(caption,)))


def test_normal_video_without_intervals_is_all_zero() -> None:
    assert not frame_labels_for_record(_record(is_anomaly=False)).any()
