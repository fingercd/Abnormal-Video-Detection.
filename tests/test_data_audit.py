from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from vadbench.data import audit as audit_module
from vadbench.data.audit import DATASET_AUDIT_SCHEMA_VERSION, audit_ucf_crime_dataset
from vadbench.data.manifest import SupervisionAnnotation, TemporalSpan, VideoManifestRecord
from vadbench.data.ucf_crime import UCF_CRIME_CATEGORIES
from vadbench.data.video import VideoInfo


def _record(
    *,
    video_id: str,
    path: str,
    split: str,
    category: str,
    num_frames: int | None = None,
    fps: float | None = None,
    duration_seconds: float | None = None,
    annotations: tuple[SupervisionAnnotation, ...] = (),
) -> VideoManifestRecord:
    return VideoManifestRecord(
        video_id=video_id,
        path=path,
        split=split,
        category=category,
        is_anomaly=category != "Normal",
        num_frames=num_frames,
        fps=fps,
        duration_seconds=duration_seconds,
        annotations=annotations,
    )


def _fake_probe(path: str | Path, *, backend=None) -> VideoInfo:
    assert backend in {None, "fake"}
    return VideoInfo(path=path, num_frames=100, fps=25.0, width=320, height=240)


def _touch(root: Path, relative_path: str, content: bytes = b"video") -> Path:
    path = root / Path(relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _error_codes(report: dict) -> set[str]:
    return {item["code"] for item in report["errors"]}


def test_default_audit_probes_container_without_reading_full_file_for_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _touch(tmp_path, "train/Normal_train.mp4")
    _touch(tmp_path, "test/Abuse_test.mp4")
    train = (
        _record(
            video_id="Normal_train",
            path="train/Normal_train.mp4",
            split="train",
            category="Normal",
        ),
    )
    test = (
        _record(
            video_id="Abuse_test",
            path="test/Abuse_test.mp4",
            split="test",
            category="Abuse",
        ),
    )

    def forbidden_hash(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
        del path, chunk_bytes
        raise AssertionError("默认审计不得读取完整视频做 SHA256")

    monkeypatch.setattr(audit_module, "_sha256_file", forbidden_hash)
    report = audit_ucf_crime_dataset(
        tmp_path,
        train,
        test,
        probe_fn=_fake_probe,
        backend="fake",
    )

    assert report["schema_version"] == DATASET_AUDIT_SCHEMA_VERSION
    assert set(report["manifest_sha256"]) == {"train", "test"}
    assert all(len(value) == 64 for value in report["manifest_sha256"].values())
    assert report["deep_hash"] is False
    assert report["hashing"] == {
        "algorithm": "sha256",
        "requested": False,
        "status": "not_run",
        "hashed_files": 0,
        "hash_errors": 0,
        "duplicate_groups": [],
    }
    assert report["files"]["present"] == 2
    assert report["files"]["probed"] == 2
    assert report["videos"][0]["probe"]["num_frames"] == 100
    assert report["near_duplicate_detection"]["status"] == "not_run"
    assert "visual_near_duplicate_not_run" in {item["code"] for item in report["warnings"]}
    assert report["passed"] is False
    assert "official_count_mismatch" in _error_codes(report)
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_missing_file_and_probe_failure_are_separate_hard_errors(tmp_path: Path) -> None:
    _touch(tmp_path, "test/Abuse_test.mp4")
    train = (
        _record(
            video_id="Normal_missing",
            path="train/Normal_missing.mp4",
            split="train",
            category="Normal",
        ),
    )
    test = (
        _record(
            video_id="Abuse_test",
            path="test/Abuse_test.mp4",
            split="test",
            category="Abuse",
        ),
    )

    def broken_probe(path: str | Path, *, backend=None) -> VideoInfo:
        del path, backend
        raise RuntimeError("broken container")

    report = audit_ucf_crime_dataset(tmp_path, train, test, probe_fn=broken_probe)

    assert report["passed"] is False
    assert report["status"] == "failed"
    assert report["files"]["missing"] == 1
    assert report["files"]["probe_errors"] == 1
    assert report["missing_files"] == [
        {
            "split": "train",
            "video_id": "Normal_missing",
            "path": "train/Normal_missing.mp4",
        }
    ]
    assert {"video_file_missing", "video_probe_error"} <= _error_codes(report)


def test_duplicate_canonical_id_and_normalized_path_are_reported(tmp_path: Path) -> None:
    _touch(tmp_path, "shared/Abuse001.mp4")
    train = (
        _record(
            video_id="Abuse001_x264",
            path="shared/Abuse001.mp4",
            split="train",
            category="Abuse",
        ),
    )
    test = (
        _record(
            video_id="abuse001_X264",
            path="shared/Abuse001.mp4",
            split="test",
            category="Abuse",
        ),
    )

    report = audit_ucf_crime_dataset(tmp_path, train, test, probe_fn=_fake_probe)

    assert len(report["duplicates"]["canonical_video_ids"]) == 1
    assert len(report["duplicates"]["normalized_paths"]) == 1
    assert {
        "duplicate_canonical_video_id",
        "duplicate_normalized_path",
    } <= _error_codes(report)
    assert report["passed"] is False


def test_deep_hash_is_explicit_and_reports_exact_content_duplicates(tmp_path: Path) -> None:
    _touch(tmp_path, "train/Abuse_a.mp4", b"same complete video bytes")
    _touch(tmp_path, "train/Abuse_b.mp4", b"same complete video bytes")
    train = (
        _record(
            video_id="Abuse_a",
            path="train/Abuse_a.mp4",
            split="train",
            category="Abuse",
        ),
        _record(
            video_id="Abuse_b",
            path="train/Abuse_b.mp4",
            split="train",
            category="Abuse",
        ),
    )

    report = audit_ucf_crime_dataset(
        tmp_path,
        train,
        (),
        deep_hash=True,
        probe_fn=_fake_probe,
    )

    assert report["hashing"]["requested"] is True
    assert report["hashing"]["status"] == "complete"
    assert report["hashing"]["hashed_files"] == 2
    assert len(report["hashing"]["duplicate_groups"]) == 1
    assert len(report["videos"][0]["sha256"]) == 64
    assert "duplicate_sha256" in {item["code"] for item in report["warnings"]}


def test_manifest_and_container_metadata_conflict_is_a_hard_error(tmp_path: Path) -> None:
    _touch(tmp_path, "test/Normal_test.mp4")
    test = (
        _record(
            video_id="Normal_test",
            path="test/Normal_test.mp4",
            split="test",
            category="Normal",
            num_frames=99,
            fps=30.0,
            duration_seconds=3.0,
        ),
    )

    report = audit_ucf_crime_dataset(tmp_path, (), test, probe_fn=_fake_probe)

    conflicts = [item for item in report["errors"] if item["code"] == "manifest_container_mismatch"]
    assert len(conflicts) == 1
    assert set(conflicts[0]["details"]["conflicts"]) == {
        "num_frames",
        "fps",
        "duration_seconds",
    }


def test_anomaly_test_record_without_positive_frame_span_is_not_evaluation_ready(
    tmp_path: Path,
) -> None:
    _touch(tmp_path, "test/Abuse_test.mp4")
    test = (
        _record(
            video_id="Abuse_test",
            path="test/Abuse_test.mp4",
            split="test",
            category="Abuse",
            num_frames=100,
            fps=25.0,
        ),
    )

    report = audit_ucf_crime_dataset(tmp_path, (), test, probe_fn=_fake_probe)

    readiness = report["evaluation_readiness"]
    assert readiness["ready"] is False
    assert readiness["anomaly_records"] == 1
    assert readiness["anomaly_records_with_1_or_2_positive_frame_spans"] == 0
    assert readiness["not_ready_records"][0]["reasons"] == [
        "anomaly_positive_frame_span_count=0;expected=1_or_2"
    ]
    assert "evaluation_not_ready" in _error_codes(report)


def test_normal_test_record_with_anomaly_frame_span_is_not_evaluation_ready(
    tmp_path: Path,
) -> None:
    _touch(tmp_path, "test/Normal_test.mp4")
    test = (
        _record(
            video_id="Normal_test",
            path="test/Normal_test.mp4",
            split="test",
            category="Normal",
            num_frames=100,
            fps=25.0,
            annotations=(
                SupervisionAnnotation(
                    scope="frame",
                    is_anomaly=True,
                    span=TemporalSpan(10, 20, "frame"),
                    source="invalid-normal-gt",
                ),
            ),
        ),
    )

    report = audit_ucf_crime_dataset(tmp_path, (), test, probe_fn=_fake_probe)

    readiness = report["evaluation_readiness"]
    assert readiness["ready"] is False
    assert readiness["normal_records"] == 1
    assert readiness["normal_records_without_positive_frame_spans"] == 0
    assert readiness["not_ready_records"][0]["reasons"] == [
        "normal_positive_frame_span_count=1;expected=0"
    ]
    assert "evaluation_not_ready" in _error_codes(report)


def test_test_records_missing_num_frames_or_fps_are_not_evaluation_ready(
    tmp_path: Path,
) -> None:
    _touch(tmp_path, "test/Normal_missing_frames.mp4")
    _touch(tmp_path, "test/Normal_missing_fps.mp4")
    test = (
        _record(
            video_id="Normal_missing_frames",
            path="test/Normal_missing_frames.mp4",
            split="test",
            category="Normal",
            fps=25.0,
        ),
        _record(
            video_id="Normal_missing_fps",
            path="test/Normal_missing_fps.mp4",
            split="test",
            category="Normal",
            num_frames=100,
        ),
    )

    report = audit_ucf_crime_dataset(tmp_path, (), test, probe_fn=_fake_probe)

    readiness = report["evaluation_readiness"]
    assert readiness["ready"] is False
    assert readiness["records_with_num_frames"] == 1
    assert readiness["records_with_fps"] == 1
    assert readiness["records_with_frame_metadata"] == 0
    reasons_by_id = {item["video_id"]: item["reasons"] for item in readiness["not_ready_records"]}
    assert reasons_by_id == {
        "Normal_missing_frames": ["missing_num_frames"],
        "Normal_missing_fps": ["missing_fps"],
    }
    assert sum(item["code"] == "evaluation_not_ready" for item in report["errors"]) >= 2


def _official_records_and_files(root: Path) -> tuple[tuple[VideoManifestRecord, ...], ...]:
    categories = tuple(category for category in UCF_CRIME_CATEGORIES if category != "Normal")

    def build(split: str, *, normal_count: int, anomaly_count: int):
        records: list[VideoManifestRecord] = []
        for index in range(normal_count):
            video_id = f"Normal_{split}_{index:04d}"
            relative = f"{split}/{video_id}.mp4"
            _touch(root, relative, b"n")
            records.append(
                _record(
                    video_id=video_id,
                    path=relative,
                    split=split,
                    category="Normal",
                    num_frames=100 if split == "test" else None,
                    fps=25.0 if split == "test" else None,
                )
            )
        for index in range(anomaly_count):
            category = categories[index % len(categories)]
            video_id = f"{category}_{split}_{index:04d}"
            relative = f"{split}/{video_id}.mp4"
            _touch(root, relative, b"a")
            records.append(
                _record(
                    video_id=video_id,
                    path=relative,
                    split=split,
                    category=category,
                    num_frames=100 if split == "test" else None,
                    fps=25.0 if split == "test" else None,
                    annotations=(
                        SupervisionAnnotation(
                            scope="frame",
                            is_anomaly=True,
                            span=TemporalSpan(10, 20, "frame"),
                            source="official-test-gt",
                        ),
                    )
                    if split == "test"
                    else (),
                )
            )
        return tuple(records)

    return (
        build("train", normal_count=800, anomaly_count=810),
        build("test", normal_count=150, anomaly_count=140),
    )


def _official_source_registry(
    root: Path,
    train: tuple[VideoManifestRecord, ...],
    test: tuple[VideoManifestRecord, ...],
) -> Path:
    train_source = root / "official" / "Anomaly_Train.txt"
    temporal_source = root / "official" / "Temporal_Anomaly_Annotation.txt"
    train_source.parent.mkdir(parents=True, exist_ok=True)
    train_source.write_text("\n".join(item.path for item in train) + "\n", encoding="utf-8")
    temporal_lines = []
    for item in test:
        if item.is_anomaly:
            annotation = item.annotations[0]
            assert annotation.span is not None
            temporal_lines.append(
                f"{Path(item.path).name} {item.category} "
                f"{int(annotation.span.start) + 1} {int(annotation.span.end)} -1 -1"
            )
        else:
            temporal_lines.append(f"{Path(item.path).name} Normal -1 -1 -1 -1")
    temporal_source.write_text("\n".join(temporal_lines) + "\n", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    registry = root / "official" / "datasets.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "ucf-crime": {
                        "source": {"commit": "a" * 40},
                        "files": {
                            "train_split": {
                                "local_path": str(train_source),
                                "sha256": digest(train_source),
                            },
                            "temporal_test_annotations": {
                                "local_path": str(temporal_source),
                                "sha256": digest(temporal_source),
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_complete_official_split_passes_and_report_validates_against_schema(
    tmp_path: Path,
) -> None:
    train, test = _official_records_and_files(tmp_path)
    source_registry = _official_source_registry(tmp_path, train, test)

    report = audit_ucf_crime_dataset(
        tmp_path,
        train,
        test,
        probe_fn=_fake_probe,
        official_source_registry=source_registry,
    )

    assert report["passed"] is True
    assert report["status"] == "passed_with_warnings"
    assert report["errors"] == []
    assert report["observed"] == {
        "train": {"total": 1610, "normal": 800, "anomaly": 810},
        "test": {"total": 290, "normal": 150, "anomaly": 140},
        "all": {"total": 1900, "normal": 950, "anomaly": 950},
    }
    assert all(check["passed"] for check in report["count_checks"])
    assert report["evaluation_readiness"] == {
        "ready": True,
        "expected_test_records": 290,
        "test_records": 290,
        "records_with_num_frames": 290,
        "records_with_fps": 290,
        "records_with_frame_metadata": 290,
        "expected_anomaly_records": 140,
        "anomaly_records": 140,
        "anomaly_records_with_1_or_2_positive_frame_spans": 140,
        "expected_normal_records": 150,
        "normal_records": 150,
        "normal_records_without_positive_frame_spans": 150,
        "positive_frame_spans": 140,
        "not_ready_records": [],
    }
    assert report["official_source_identity"]["status"] == "verified"
    assert set(report["category_distribution"]["train"]) == set(UCF_CRIME_CATEGORIES)
    assert report["files"]["present"] == 1900
    assert report["files"]["probed"] == 1900

    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).parents[1] / "schemas" / "dataset-audit-v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(report)


def test_malformed_manifest_line_is_retained_as_error_not_exception(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.jsonl"
    test_manifest = tmp_path / "test.jsonl"
    train_manifest.write_text("not-json\n", encoding="utf-8")
    test_manifest.write_text("", encoding="utf-8")

    report = audit_ucf_crime_dataset(
        tmp_path,
        train_manifest,
        test_manifest,
        probe_fn=_fake_probe,
    )

    assert report["passed"] is False
    assert "manifest_record_invalid" in _error_codes(report)
    assert report["files"]["manifest_records"] == 0
