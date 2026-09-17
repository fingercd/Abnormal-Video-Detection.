from __future__ import annotations

import json
from pathlib import Path

import pytest

from vadbench.research.data_prepare import (
    CohortLimits,
    ResearchDataPreparationError,
    prepare_ucf_research_data,
    write_prepared_research_data,
)


def _sources(tmp_path: Path) -> tuple[Path, Path, Path, list[str]]:
    root = tmp_path / "UCF-Crime"
    train_paths = [
        *(f"Abuse/Abuse{index:03d}_x264.mp4" for index in range(1, 11)),
        *(
            f"Training_Normal_Videos_Anomaly/Normal_Videos{index:03d}_x264.mp4"
            for index in range(1, 11)
        ),
    ]
    for relative in train_paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    test = root / "Abuse" / "Abuse900_x264.mp4"
    test.parent.mkdir(parents=True, exist_ok=True)
    test.touch()
    train_split = tmp_path / "Anomaly_Train.txt"
    train_split.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    temporal = tmp_path / "Temporal_Anomaly_Annotation.txt"
    temporal.write_text("Abuse900_x264.mp4 Abuse 1 2 -1 -1\n", encoding="utf-8")
    return root, train_split, temporal, train_paths


def _audit(*_args, **_kwargs):
    return {"passed": False, "status": "partial", "errors": [{"code": "missing_files"}]}


def _prepare(root: Path, train_split: Path, temporal: Path, **kwargs):
    return prepare_ucf_research_data(
        dataset_root=root,
        train_split=train_split,
        temporal_annotations=temporal,
        limits=CohortLimits(1, 4, 2, 2),
        audit_fn=_audit,
        **kwargs,
    )


def test_roles_are_frozen_from_full_train_before_availability_changes(tmp_path: Path):
    root, train_split, temporal, _ = _sources(tmp_path)
    complete = _prepare(root, train_split, temporal)
    delayed_id = complete.candidate_videos["explore"][0]
    delayed = next(record for record in complete.imported.train if record.video_id == delayed_id)
    delayed.resolve_path(root).unlink()

    before = _prepare(root, train_split, temporal)
    delayed.resolve_path(root).touch()
    after = _prepare(root, train_split, temporal)

    assert before.partition_lock == after.partition_lock == complete.partition_lock
    assert before.candidate_videos == after.candidate_videos == complete.candidate_videos
    assert delayed_id not in {record.video_id for record in before.cohorts["explore"].records}
    assert delayed_id in {record.video_id for record in after.cohorts["explore"].records}
    assert before.receipt()["cohorts"]["explore"]["status"] == "partial"


def test_debug_is_an_available_subset_of_explore_and_windows_match_run_probe(tmp_path: Path):
    root, train_split, temporal, _ = _sources(tmp_path)
    result = _prepare(root, train_split, temporal)
    assert set(result.candidate_videos["debug"]).issubset(result.candidate_videos["explore"])
    debug_videos = {record.video_id for record in result.cohorts["debug"].records}
    explore_videos = {record.video_id for record in result.cohorts["explore"].records}
    assert debug_videos.issubset(explore_videos)

    for cohort in result.cohorts.values():
        by_video: dict[str, set[str]] = {}
        for record in cohort.records:
            by_video.setdefault(record.video_id, set()).add(record.clip_id)
            assert record.official_split == "train"
            assert record.temporal_label is None
            assert record.label_source == "video_weak"
            assert record.category_for_analysis_only is not None
        for video_id, clip_ids in by_video.items():
            assert clip_ids == {f"{video_id}:segment-{index:02d}" for index in range(8)}


def test_source_groups_stay_in_one_partition_and_unknown_groups_fail(tmp_path: Path):
    root, train_split, temporal, _ = _sources(tmp_path)
    source_groups = {"Abuse001_x264": "same-source", "Abuse002_x264": "same-source"}
    result = _prepare(root, train_split, temporal, source_groups=source_groups)
    assert result.partition_lock["Abuse001_x264"] == result.partition_lock["Abuse002_x264"]
    assert result.source_grouping_status == "provided_source_groups"

    with pytest.raises(ResearchDataPreparationError, match="未知训练视频"):
        _prepare(root, train_split, temporal, source_groups={"missing": "source"})


def test_writer_persists_full_partition_lock_and_refuses_replacement(tmp_path: Path):
    root, train_split, temporal, _ = _sources(tmp_path)
    result = _prepare(root, train_split, temporal)
    output = tmp_path / "run"
    written = write_prepared_research_data(result, output)
    receipt = json.loads(written["receipt"].read_text(encoding="utf-8"))
    lock = json.loads(written["partition_lock"].read_text(encoding="utf-8"))
    candidates = json.loads(written["cohort_candidates"].read_text(encoding="utf-8"))
    assert receipt["policy"] == "W"
    assert lock["seed"] == 202709
    assert len(lock["partitions"]) == 20
    assert candidates["windows_per_video"] == 8
    assert written["audit"].parent.name == "audit"
    assert not (output / "manifests" / "train.jsonl").exists()
    assert not (output / "manifests" / "test.jsonl").exists()
    with pytest.raises(FileExistsError):
        write_prepared_research_data(result, output)
