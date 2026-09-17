from __future__ import annotations

import json
from pathlib import Path

import pytest

from vadbench.data.manifest import SpanUnit, SupervisionScope
from vadbench.data.ucf_crime import (
    UCFCrimeError,
    attach_uca_captions,
    import_ucf_crime,
    parse_uca_captions,
    parse_ucf_split_file,
    parse_ucf_temporal_annotations,
    reconcile_ucf_test_records_to_decoded_frames,
)
from vadbench.data.video import VideoInfo


def test_parse_official_and_derived_temporal_formats(tmp_path: Path) -> None:
    annotations = tmp_path / "Temporal_Anomaly_Annotation.txt"
    annotations.write_text(
        "Abuse028_x264.mp4 Abuse 165 240 -1 -1\n"
        "Arson/Arson011_x264.mp4 1500 Arson 150 420 680 1267\n"
        "Testing_Normal_Videos_Anomaly/Normal_Videos_001_x264.mp4 "
        "900 Normal -1 -1 -1 -1\n",
        encoding="utf-8",
    )
    parsed = parse_ucf_temporal_annotations(annotations)

    abuse = parsed["abuse028_x264"]
    assert [(span.start, span.end, span.unit) for span in abuse.spans] == [
        (164, 240, SpanUnit.FRAME)
    ]
    assert abuse.raw_spans_1based_inclusive == ((165, 240),)
    assert abuse.spans[0].end - abuse.spans[0].start == 76
    arson = parsed["arson011_x264"]
    assert arson.num_frames == 1500
    assert [(span.start, span.end) for span in arson.spans] == [(149, 420), (679, 1267)]
    assert arson.raw_spans_1based_inclusive == ((150, 420), (680, 1267))
    assert parsed["normal_videos_001_x264"].spans == ()


@pytest.mark.parametrize(
    "line",
    [
        "Abuse028_x264.mp4 Abuse -1 240 -1 -1",
        "Abuse028_x264.mp4 Abuse 240 165 -1 -1",
        "Normal001_x264.mp4 Normal 1 2 -1 -1",
        "Abuse028_x264.mp4 Abuse 1 10 8 12",
    ],
)
def test_temporal_parser_rejects_invalid_intervals(tmp_path: Path, line: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(f"{line}\n", encoding="utf-8")
    with pytest.raises(UCFCrimeError):
        parse_ucf_temporal_annotations(path)


def test_feature_split_is_deduplicated_to_original_video(tmp_path: Path) -> None:
    split = tmp_path / "UCF_Train.list"
    split.write_text(
        "dataset/Abuse/Abuse001_x264__0.npy\n"
        "dataset/Abuse/Abuse001_x264__1.npy\n"
        "dataset/Abuse/Abuse002_x264__0.npy\n",
        encoding="utf-8",
    )
    entries = parse_ucf_split_file(split, "train")
    assert [entry.video_id for entry in entries] == ["Abuse001_x264", "Abuse002_x264"]
    assert entries[0].path == "dataset/Abuse/Abuse001_x264.mp4"


def _write_minimal_ucf_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "UCF-Crime"
    relative_paths = (
        "Abuse/Abuse001_x264.mp4",
        "Training_Normal_Videos_Anomaly/Normal_Videos_001_x264.mp4",
        "Abuse/Abuse028_x264.mp4",
        "Testing_Normal_Videos_Anomaly/Normal_Videos_900_x264.mp4",
    )
    for relative in relative_paths:
        path = root / Path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    train = tmp_path / "train.txt"
    train.write_text("\n".join(relative_paths[:2]) + "\n", encoding="utf-8")
    test = tmp_path / "test.txt"
    test.write_text("\n".join(relative_paths[2:]) + "\n", encoding="utf-8")
    temporal = tmp_path / "temporal.txt"
    temporal.write_text(
        "Abuse028_x264.mp4 Abuse 165 240 -1 -1\n",
        encoding="utf-8",
    )
    return root, train, test, temporal


def test_import_builds_relative_manifests_and_frame_truth(tmp_path: Path) -> None:
    root, train, test, temporal = _write_minimal_ucf_sources(tmp_path)
    result = import_ucf_crime(
        dataset_root=root,
        train_split=train,
        test_split=test,
        temporal_annotations=temporal,
        require_files=True,
    )

    assert len(result.train) == 2
    assert len(result.test) == 2
    assert all(not Path(record.path).is_absolute() for record in result.all_records)
    normal_train = result.train[1]
    assert normal_train.category == "Normal"
    assert normal_train.is_anomaly is False

    abnormal_test = result.test[0]
    assert abnormal_test.is_anomaly is True
    frame_annotations = [
        item for item in abnormal_test.annotations if item.scope == SupervisionScope.FRAME
    ]
    assert len(frame_annotations) == 1
    assert frame_annotations[0].start_frame == 164
    assert frame_annotations[0].end_frame == 240
    assert frame_annotations[0].metadata == {
        "raw_start_frame": 165,
        "raw_end_frame": 240,
        "raw_coordinate_system": "matlab_1based_inclusive",
        "internal_coordinate_system": "zero_based_half_open",
    }


def test_official_temporal_file_can_derive_abnormal_and_normal_test_records(
    tmp_path: Path,
) -> None:
    root = tmp_path / "UCF-Crime"
    train_video = root / "Abuse" / "Abuse001_x264.mp4"
    abnormal_test = root / "Abuse" / "Abuse028_x264.mp4"
    normal_test = root / "Testing_Normal_Videos_Anomaly" / "Normal_Videos_003_x264.mp4"
    for video in (train_video, abnormal_test, normal_test):
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()
    train = tmp_path / "Anomaly_Train.txt"
    train.write_text("Abuse/Abuse001_x264.mp4\n", encoding="utf-8")
    temporal = tmp_path / "Temporal_Anomaly_Annotation.txt"
    temporal.write_text(
        "Abuse028_x264.mp4 Abuse 165 240 -1 -1\nNormal_Videos_003_x264.mp4 Normal -1 -1 -1 -1\n",
        encoding="utf-8",
    )

    result = import_ucf_crime(
        dataset_root=root,
        train_split=train,
        temporal_annotations=temporal,
        require_files=True,
    )

    assert [record.path for record in result.test] == [
        "Abuse/Abuse028_x264.mp4",
        "Testing_Normal_Videos_Anomaly/Normal_Videos_003_x264.mp4",
    ]
    assert [record.is_anomaly for record in result.test] == [True, False]
    normal_annotations = result.test[1].annotations
    assert [item.scope for item in normal_annotations] == [SupervisionScope.VIDEO]


def test_import_strictly_rejects_train_test_leakage(tmp_path: Path) -> None:
    root, train, test, temporal = _write_minimal_ucf_sources(tmp_path)
    test.write_text(
        test.read_text(encoding="utf-8") + "Abuse/Abuse001_x264.mp4\n",
        encoding="utf-8",
    )
    with pytest.raises(UCFCrimeError, match="泄漏"):
        import_ucf_crime(
            dataset_root=root,
            train_split=train,
            test_split=test,
            temporal_annotations=temporal,
        )


def test_import_requires_temporal_truth_for_abnormal_test_video(tmp_path: Path) -> None:
    root, train, test, temporal = _write_minimal_ucf_sources(tmp_path)
    temporal.write_text("", encoding="utf-8")
    with pytest.raises(UCFCrimeError, match="缺少时序标注"):
        import_ucf_crime(
            dataset_root=root,
            train_split=train,
            test_split=test,
            temporal_annotations=temporal,
        )


def test_decoded_source_end_reconciliation_keeps_raw_endpoints_and_clamps_only_decoded_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "UCF-Crime"
    test_video = root / "Abuse" / "Abuse900_x264.mp4"
    test_video.parent.mkdir(parents=True)
    test_video.touch()
    train = tmp_path / "Anomaly_Train.txt"
    train.write_text("Abuse/Abuse001_x264.mp4\n", encoding="utf-8")
    (root / "Abuse" / "Abuse001_x264.mp4").touch()
    temporal = tmp_path / "Temporal_Anomaly_Annotation.txt"
    temporal.write_text("Abuse900_x264.mp4 Abuse 9 12 -1 -1\n", encoding="utf-8")
    result = import_ucf_crime(
        dataset_root=root,
        train_split=train,
        temporal_annotations=temporal,
        require_files=True,
    )
    monkeypatch.setattr(
        "vadbench.data.ucf_crime.probe_video",
        lambda path: VideoInfo(path=path, num_frames=10, fps=25.0, width=2, height=2),
    )
    reconciled, receipt = reconcile_ucf_test_records_to_decoded_frames(result.test, root)
    annotation = next(item for item in reconciled[0].annotations if item.scope == SupervisionScope.FRAME)
    assert (annotation.start_frame, annotation.end_frame) == (8, 10)
    detail = annotation.metadata["source_end_reconciliation"]
    assert detail["raw_start_1based_inclusive"] == 9
    assert detail["raw_end_1based_inclusive"] == 12
    assert detail["author_frame_count_round_duration_x30"] == 12
    assert detail["decoded_frame_count"] == 10
    assert detail["decoded_clamped"] is True
    assert receipt["decoded_clamped_spans"] == 1
    assert receipt["author_decode_frame_count_different_records"] == 1


def test_decoded_reconciliation_never_uses_author_30fps_end_as_effective_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "UCF-Crime"
    for relative in ("Abuse/Abuse001_x264.mp4", "Abuse/Abuse900_x264.mp4"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    train = tmp_path / "Anomaly_Train.txt"
    train.write_text("Abuse/Abuse001_x264.mp4\n", encoding="utf-8")
    temporal = tmp_path / "Temporal_Anomaly_Annotation.txt"
    temporal.write_text("Abuse900_x264.mp4 Abuse 9 10 -1 -1\n", encoding="utf-8")
    result = import_ucf_crime(
        dataset_root=root,
        train_split=train,
        temporal_annotations=temporal,
        require_files=True,
    )
    # N_author=round(10 / 40 * 30)=8 while N_decode=10.  The historical
    # author clamp must remain diagnostic; decoded end stays raw_end=10.
    monkeypatch.setattr(
        "vadbench.data.ucf_crime.probe_video",
        lambda path: VideoInfo(path=path, num_frames=10, fps=40.0, width=2, height=2),
    )
    reconciled, _ = reconcile_ucf_test_records_to_decoded_frames(result.test, root)
    annotation = next(item for item in reconciled[0].annotations if item.scope == SupervisionScope.FRAME)
    assert (annotation.start_frame, annotation.end_frame) == (8, 10)
    detail = annotation.metadata["source_end_reconciliation"]
    assert detail["end_author_1based_inclusive"] == 8
    assert detail["end_effective_1based_inclusive"] == 10
    assert detail["author_clamped"] is True
    assert detail["decoded_clamped"] is False


def test_uca_json_caption_is_preserved_without_anomaly_inference(tmp_path: Path) -> None:
    root, train, test, temporal = _write_minimal_ucf_sources(tmp_path)
    uca = tmp_path / "UCFCrime_Test.json"
    uca.write_text(
        json.dumps(
            {
                "Normal_Videos_900_x264": {
                    "duration": 10.0,
                    "timestamps": [[0.0, 2.5]],
                    "sentences": ["Two people fight near a vehicle."],
                }
            }
        ),
        encoding="utf-8",
    )
    result = import_ucf_crime(
        dataset_root=root,
        train_split=train,
        test_split=test,
        temporal_annotations=temporal,
        uca_captions=uca,
    )
    normal = result.test[1]
    captions = [item for item in normal.annotations if item.scope == SupervisionScope.CAPTION]

    assert normal.is_anomaly is False
    assert len(captions) == 1
    assert captions[0].is_anomaly is None
    assert captions[0].span is not None
    assert captions[0].span.unit == SpanUnit.SECOND


def test_uca_txt_parser_and_attach_do_not_change_record_label(tmp_path: Path) -> None:
    root, train, test, temporal = _write_minimal_ucf_sources(tmp_path)
    result = import_ucf_crime(
        dataset_root=root,
        train_split=train,
        test_split=test,
        temporal_annotations=temporal,
    )
    caption_file = tmp_path / "UCFCrime_Test.txt"
    caption_file.write_text(
        "Normal_Videos_900_x264 00:00.0 00:02.5 ##A robbery is described.\n",
        encoding="utf-8",
    )
    captions = parse_uca_captions(caption_file)
    updated = attach_uca_captions(result.test, captions)
    normal = updated[1]
    caption = normal.annotations[-1]

    assert caption.text == "A robbery is described."
    assert caption.is_anomaly is None
    assert normal.is_anomaly is False
