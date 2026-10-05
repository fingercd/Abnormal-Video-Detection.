from __future__ import annotations

import json
from pathlib import Path

import pytest

from vadbench.data.manifest import (
    DatasetSplit,
    ManifestError,
    SpanUnit,
    SupervisionAnnotation,
    SupervisionScope,
    TemporalSpan,
    VideoManifestRecord,
    load_manifest_jsonl,
    validate_manifest,
    validate_manifest_pair,
    write_manifest_jsonl,
)


def _record(*, video_id: str = "Abuse001_x264", split: str = "train") -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=f"Abuse/{video_id}.mp4",
        split=split,
        category="Abuse",
        is_anomaly=True,
        annotations=(
            SupervisionAnnotation(
                scope="video",
                label="Abuse",
                is_anomaly=True,
                source="official-split",
            ),
        ),
    )


def test_jsonl_round_trip_preserves_all_supervision_scopes(tmp_path: Path) -> None:
    record = VideoManifestRecord(
        video_id="Arson011_x264",
        path="Arson/Arson011_x264.mp4",
        split=DatasetSplit.TEST,
        category="Arson",
        is_anomaly=True,
        num_frames=1500,
        fps=30.0,
        duration_seconds=50.0,
        annotations=(
            SupervisionAnnotation(scope="video", label="Arson", is_anomaly=True),
            SupervisionAnnotation(
                scope="segment",
                label="event",
                is_anomaly=True,
                span=TemporalSpan(2, 4, "segment"),
            ),
            SupervisionAnnotation(
                scope="frame",
                label="Arson",
                is_anomaly=True,
                span=TemporalSpan(150, 420, SpanUnit.FRAME),
            ),
            SupervisionAnnotation(
                scope=SupervisionScope.CAPTION,
                span=TemporalSpan(0.0, 5.3, "second"),
                text="一名行人走过画面。",
                is_anomaly=None,
                source="uca",
            ),
        ),
        metadata={"note": "中文可追溯"},
    )
    output = write_manifest_jsonl([record], tmp_path / "manifest.jsonl")

    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["annotations"][-1]["is_anomaly"] is None
    assert raw["annotations"][-1]["text"] == "一名行人走过画面。"

    loaded = load_manifest_jsonl(output)
    assert loaded == (record,)
    assert loaded[0].supervision_scopes == {
        SupervisionScope.VIDEO,
        SupervisionScope.SEGMENT,
        SupervisionScope.FRAME,
        SupervisionScope.CAPTION,
    }


@pytest.mark.parametrize(
    "path",
    [
        "C:/datasets/UCF/Abuse001.mp4",
        "/datasets/UCF/Abuse001.mp4",
        "../outside.mp4",
        "Abuse/../../outside.mp4",
        "Abuse\\..\\outside.mp4",
    ],
)
def test_manifest_rejects_absolute_or_traversal_paths(path: str) -> None:
    with pytest.raises(ManifestError, match="path"):
        VideoManifestRecord(
            video_id="Abuse001",
            path=path,
            split="train",
            category="Abuse",
            is_anomaly=True,
        )


def test_manifest_strictly_rejects_train_test_leakage() -> None:
    train = _record(split="train")
    test = VideoManifestRecord(
        video_id="abuse001_X264",
        path="different/Abuse001_x264.mp4",
        split="test",
        category="Abuse",
        is_anomaly=True,
    )
    with pytest.raises(ManifestError, match="泄漏"):
        validate_manifest_pair([train], [test])


def test_feature_chunk_identity_cannot_bypass_leakage_check() -> None:
    train = VideoManifestRecord(
        video_id="Abuse001_x264__0",
        path="features/Abuse001_x264__0.npy",
        split="train",
        category="Abuse",
        is_anomaly=True,
    )
    test = VideoManifestRecord(
        video_id="Abuse001_x264",
        path="raw/Abuse001_x264.mp4",
        split="test",
        category="Abuse",
        is_anomaly=True,
    )
    with pytest.raises(ManifestError, match="泄漏"):
        validate_manifest([train, test])


def test_validate_manifest_can_require_files_under_dataset_root(tmp_path: Path) -> None:
    video = tmp_path / "Abuse" / "Abuse001_x264.mp4"
    video.parent.mkdir()
    video.touch()
    record = _record()
    assert validate_manifest([record], dataset_root=tmp_path, require_files=True) == (record,)

    missing = _record(video_id="Abuse002_x264")
    with pytest.raises(ManifestError, match="不存在"):
        validate_manifest([missing], dataset_root=tmp_path, require_files=True)


def test_video_annotation_must_match_record_label() -> None:
    with pytest.raises(ManifestError, match="不一致"):
        VideoManifestRecord(
            video_id="Normal001",
            path="Normal/Normal001.mp4",
            split="test",
            category="Normal",
            is_anomaly=False,
            annotations=(SupervisionAnnotation(scope="video", is_anomaly=True),),
        )


def test_frame_annotation_cannot_exceed_known_num_frames() -> None:
    with pytest.raises(ManifestError, match="越过 num_frames"):
        VideoManifestRecord(
            video_id="Abuse001",
            path="Abuse/Abuse001.mp4",
            split="test",
            category="Abuse",
            is_anomaly=True,
            num_frames=100,
            annotations=(
                SupervisionAnnotation(
                    scope="frame",
                    span=TemporalSpan(90, 101, "frame"),
                    is_anomaly=True,
                ),
            ),
        )


def test_caption_span_cannot_exceed_known_duration() -> None:
    with pytest.raises(ManifestError, match="越过 duration_seconds"):
        VideoManifestRecord(
            video_id="Normal001",
            path="Normal/Normal001.mp4",
            split="test",
            category="Normal",
            is_anomaly=False,
            duration_seconds=10.0,
            annotations=(
                SupervisionAnnotation(
                    scope="caption",
                    span=TemporalSpan(9.5, 10.1, "second"),
                    text="A person leaves the scene.",
                    is_anomaly=None,
                ),
            ),
        )


def test_bad_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{}\nnot-json\n", encoding="utf-8")
    with pytest.raises(ManifestError, match=r"broken\.jsonl:1"):
        load_manifest_jsonl(path)


def test_schema_file_declares_protocol_v1_and_caption_scope() -> None:
    schema_path = Path(__file__).parents[1] / "schemas" / "video-manifest-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == 1
    assert "caption" in schema["$defs"]["annotation"]["properties"]["scope"]["enum"]
