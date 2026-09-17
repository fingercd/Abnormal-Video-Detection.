"""UCF-Crime 实际落盘数据与官方切分的只读审计。

默认审计只读取 manifest、文件系统元数据，并通过 :func:`probe_video` 查询容器
信息；不会顺序解码视频，也不会读取完整文件计算内容哈希。只有调用方显式设置
``deep_hash=True`` 时才会逐文件计算 SHA256。视觉近重复检测不属于本模块的隐式
行为，报告会始终明确记录其未执行状态。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from vadbench.hashing import sha256_file
from vadbench.hashing import sha256_file as _sha256_file

from .manifest import (
    DatasetSplit,
    ManifestError,
    SupervisionScope,
    VideoManifestRecord,
    canonical_video_id,
)
from .ucf_crime import (
    UCF_CRIME_CATEGORIES,
    parse_ucf_split_file,
    parse_ucf_temporal_annotations,
)
from .video import VideoInfo, probe_video

DATASET_AUDIT_SCHEMA_VERSION = "vadbench.dataset-audit.v2"

OFFICIAL_UCF_CRIME_COUNTS: Mapping[str, Mapping[str, int]] = {
    "train": {"total": 1610, "normal": 800, "anomaly": 810},
    "test": {"total": 290, "normal": 150, "anomaly": 140},
    "all": {"total": 1900, "normal": 950, "anomaly": 950},
}

ManifestSource: TypeAlias = str | Path | Iterable[VideoManifestRecord]
ProbeFunction: TypeAlias = Callable[..., VideoInfo]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _issue(
    code: str,
    message: str,
    *,
    split: str | None = None,
    video_id: str | None = None,
    path: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if split is not None:
        result["split"] = split
    if video_id is not None:
        result["video_id"] = video_id
    if path is not None:
        result["path"] = path
    if details:
        result["details"] = dict(details)
    return result


def _source_label(source: ManifestSource) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return "<in-memory>"


def compute_manifest_sha256(records: Iterable[VideoManifestRecord]) -> str:
    """Hash canonical manifest semantics, independent of JSONL whitespace."""

    payload = json.dumps(
        [record.to_dict() for record in records],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_official_source_identity(
    records_by_split: Mapping[str, tuple[VideoManifestRecord, ...]],
    *,
    registry_path: str | Path,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify manifests against the repository-pinned official protocol files."""

    registry = Path(registry_path).expanduser().resolve()
    result: dict[str, Any] = {
        "status": "unavailable",
        "registry": str(registry),
        "source_commit": None,
        "train_split": None,
        "temporal_test_annotations": None,
        "train_identity_sha256": None,
        "test_identity_sha256": None,
    }
    try:
        payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
        specification = payload["datasets"]["ucf-crime"]
        files = specification["files"]
        source_commit = str(specification["source"]["commit"])
        result["source_commit"] = source_commit
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        errors.append(
            _issue(
                "official_source_unavailable",
                f"无法读取 UCF-Crime 官方来源登记：{exc}",
                path=str(registry),
            )
        )
        return result

    project_root = registry.parent.parent
    resolved_files: dict[str, Path] = {}
    for name in ("train_split", "temporal_test_annotations"):
        file_spec = files.get(name)
        if not isinstance(file_spec, Mapping):
            errors.append(
                _issue("official_source_unavailable", f"dataset registry 缺少 {name} 登记")
            )
            return result
        raw_path = file_spec.get("local_path")
        expected_sha256 = str(file_spec.get("sha256", ""))
        if not isinstance(raw_path, str) or not raw_path or len(expected_sha256) != 64:
            errors.append(
                _issue(
                    "official_source_unavailable", f"dataset registry 的 {name} 缺少路径或 SHA256"
                )
            )
            return result
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            errors.append(
                _issue(
                    "official_source_unavailable",
                    f"冻结的官方协议文件不存在：{path}",
                    path=str(path),
                )
            )
            return result
        actual_sha256 = sha256_file(path)
        result[name] = {
            "path": str(path),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
        }
        if actual_sha256 != expected_sha256:
            errors.append(
                _issue(
                    "official_source_checksum_mismatch",
                    f"冻结的官方协议文件 SHA256 不匹配：{path}",
                    path=str(path),
                    details={"expected": expected_sha256, "actual": actual_sha256},
                )
            )
            return result
        resolved_files[name] = path

    try:
        official_train = parse_ucf_split_file(resolved_files["train_split"], DatasetSplit.TRAIN)
        official_test = parse_ucf_temporal_annotations(resolved_files["temporal_test_annotations"])
    except Exception as exc:
        errors.append(
            _issue(
                "official_source_invalid",
                f"冻结的官方协议文件无法解析：{exc}",
                details={"exception_type": type(exc).__name__},
            )
        )
        return result

    train_identity = [
        {
            "video_id": canonical_video_id(item.video_id),
            "category": item.category,
        }
        for item in official_train
    ]
    test_identity = [
        {
            "video_id": key,
            "category": item.category,
            "spans": [[span.start, span.end] for span in item.spans],
        }
        for key, item in official_test.items()
    ]
    result["train_identity_sha256"] = _identity_sha256(train_identity)
    result["test_identity_sha256"] = _identity_sha256(test_identity)

    mismatches: list[str] = []
    for split, source_rows in (("train", train_identity), ("test", test_identity)):
        if split not in records_by_split:
            continue
        manifests = {canonical_video_id(item.video_id): item for item in records_by_split[split]}
        if set(manifests) != {item["video_id"] for item in source_rows}:
            mismatches.append(f"{split}_video_set")
            continue
        for source in source_rows:
            record = manifests[source["video_id"]]
            if record.category != source["category"] or record.is_anomaly != (
                source["category"] != "Normal"
            ):
                mismatches.append(f"{split}_label:{record.video_id}")
                break
            if split == "test":
                spans = [
                    [a.span.start, a.span.end]
                    for a in record.annotations
                    if a.scope == SupervisionScope.FRAME
                    and a.is_anomaly is True
                    and a.span is not None
                ]
                if spans != source["spans"]:
                    mismatches.append(f"test_truth:{record.video_id}")
                    break
    if mismatches:
        errors.append(
            _issue(
                "official_source_manifest_mismatch",
                "manifest 的视频集合、类别或测试区间与冻结官方来源不一致",
                details={"mismatches": mismatches[:10]},
            )
        )
        result["status"] = "mismatch"
        return result
    result["status"] = "verified"
    return result


def _load_manifest_source(
    source: ManifestSource,
    *,
    source_split: str,
    errors: list[dict[str, Any]],
) -> tuple[VideoManifestRecord, ...]:
    """尽可能保留合法行，让格式错误与计数缺口同时出现在报告中。"""

    if not isinstance(source, (str, Path)):
        records: list[VideoManifestRecord] = []
        try:
            iterator = iter(source)
            for index, record in enumerate(iterator, start=1):
                if not isinstance(record, VideoManifestRecord):
                    errors.append(
                        _issue(
                            "manifest_record_invalid",
                            f"{source_split} manifest 第 {index} 项不是 VideoManifestRecord",
                            split=source_split,
                            details={"index": index, "type": type(record).__name__},
                        )
                    )
                    continue
                records.append(record)
        except Exception as exc:  # pragma: no cover - defensive for user iterators
            errors.append(
                _issue(
                    "manifest_read_error",
                    f"读取 {source_split} manifest 迭代器失败：{exc}",
                    split=source_split,
                    details={"exception_type": type(exc).__name__},
                )
            )
        return tuple(records)

    manifest_path = Path(source).expanduser()
    if not manifest_path.is_file():
        errors.append(
            _issue(
                "manifest_missing",
                f"{source_split} manifest 文件不存在：{manifest_path}",
                split=source_split,
                path=str(manifest_path),
            )
        )
        return ()

    records = []
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, Mapping):
                        raise ManifestError("顶层必须是 JSON object")
                    records.append(VideoManifestRecord.from_dict(payload))
                except (json.JSONDecodeError, ManifestError, TypeError, ValueError) as exc:
                    errors.append(
                        _issue(
                            "manifest_record_invalid",
                            f"{manifest_path}:{line_number}: {exc}",
                            split=source_split,
                            path=str(manifest_path),
                            details={"line_number": line_number},
                        )
                    )
    except (OSError, UnicodeError) as exc:
        errors.append(
            _issue(
                "manifest_read_error",
                f"读取 {source_split} manifest 失败：{exc}",
                split=source_split,
                path=str(manifest_path),
                details={"exception_type": type(exc).__name__},
            )
        )
    return tuple(records)


def _occurrence(source_split: str, index: int, record: VideoManifestRecord) -> dict[str, Any]:
    return {
        "split": source_split,
        "index": index,
        "video_id": record.video_id,
        "path": record.path,
    }


def _duplicate_groups(
    records_by_split: Mapping[str, tuple[VideoManifestRecord, ...]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identities: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    paths: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_split, records in records_by_split.items():
        for index, record in enumerate(records):
            occurrence = _occurrence(source_split, index, record)
            identities[canonical_video_id(record.video_id)].append(occurrence)
            paths[record.path.casefold()].append(occurrence)

    duplicate_ids = [
        {"key": key, "occurrences": occurrences}
        for key, occurrences in sorted(identities.items())
        if len(occurrences) > 1
    ]
    duplicate_paths = [
        {"key": key, "occurrences": occurrences}
        for key, occurrences in sorted(paths.items())
        if len(occurrences) > 1
    ]
    return duplicate_ids, duplicate_paths


def _observed_counts(records: Iterable[VideoManifestRecord]) -> dict[str, int]:
    items = tuple(records)
    anomaly = sum(record.is_anomaly for record in items)
    return {"total": len(items), "normal": len(items) - anomaly, "anomaly": anomaly}


def _check_count(
    *,
    name: str,
    expected: int,
    actual: int,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    passed = actual == expected
    if not passed:
        split, field = name.split(".", maxsplit=1)
        errors.append(
            _issue(
                "official_count_mismatch",
                f"{name} 应为 {expected}，实际为 {actual}",
                split=None if split == "all" else split,
                details={"field": field, "expected": expected, "actual": actual},
            )
        )
    return {"name": name, "expected": expected, "actual": actual, "passed": passed}


def _probe_payload(info: VideoInfo) -> dict[str, Any]:
    return {
        "status": "ok",
        "num_frames": info.num_frames,
        "fps": info.fps,
        "duration_seconds": info.duration_seconds,
        "width": info.width,
        "height": info.height,
    }


def _evaluation_readiness(
    test_records: tuple[VideoManifestRecord, ...],
    *,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """检查官方 290 视频 frame-AUC 所需的 manifest 元数据与真值。"""

    records_with_num_frames = 0
    records_with_fps = 0
    records_with_frame_metadata = 0
    anomaly_records = 0
    anomaly_records_with_valid_spans = 0
    normal_records = 0
    normal_records_without_anomaly_spans = 0
    positive_frame_spans = 0
    not_ready_records: list[dict[str, Any]] = []

    for record in test_records:
        reasons: list[str] = []
        has_num_frames = record.num_frames is not None
        has_fps = record.fps is not None
        records_with_num_frames += int(has_num_frames)
        records_with_fps += int(has_fps)
        records_with_frame_metadata += int(has_num_frames and has_fps)
        if not has_num_frames:
            reasons.append("missing_num_frames")
        if not has_fps:
            reasons.append("missing_fps")

        anomaly_spans = tuple(
            annotation
            for annotation in record.annotations
            if annotation.scope == SupervisionScope.FRAME and annotation.is_anomaly is True
        )
        positive_frame_spans += len(anomaly_spans)
        if record.is_anomaly:
            anomaly_records += 1
            if 1 <= len(anomaly_spans) <= 2:
                anomaly_records_with_valid_spans += 1
            else:
                reasons.append(
                    f"anomaly_positive_frame_span_count={len(anomaly_spans)};expected=1_or_2"
                )
        else:
            normal_records += 1
            if not anomaly_spans:
                normal_records_without_anomaly_spans += 1
            else:
                reasons.append(f"normal_positive_frame_span_count={len(anomaly_spans)};expected=0")

        if reasons:
            not_ready = {
                "split": "test",
                "video_id": record.video_id,
                "path": record.path,
                "reasons": reasons,
            }
            not_ready_records.append(not_ready)
            errors.append(
                _issue(
                    "evaluation_not_ready",
                    f"{record.video_id} 不满足正式 frame-AUC 输入门禁：{'; '.join(reasons)}",
                    split="test",
                    video_id=record.video_id,
                    path=record.path,
                    details={"reasons": reasons},
                )
            )

    test_count_ready = len(test_records) == OFFICIAL_UCF_CRIME_COUNTS["test"]["total"]
    if not test_count_ready:
        errors.append(
            _issue(
                "evaluation_not_ready",
                "正式 frame-AUC 必须使用完整 290 条官方 test manifest，"
                f"实际为 {len(test_records)} 条",
                split="test",
                details={
                    "expected_test_records": OFFICIAL_UCF_CRIME_COUNTS["test"]["total"],
                    "actual_test_records": len(test_records),
                },
            )
        )

    ready = (
        test_count_ready
        and records_with_frame_metadata == len(test_records)
        and anomaly_records == OFFICIAL_UCF_CRIME_COUNTS["test"]["anomaly"]
        and anomaly_records_with_valid_spans == anomaly_records
        and normal_records == OFFICIAL_UCF_CRIME_COUNTS["test"]["normal"]
        and normal_records_without_anomaly_spans == normal_records
    )
    return {
        "ready": ready,
        "expected_test_records": OFFICIAL_UCF_CRIME_COUNTS["test"]["total"],
        "test_records": len(test_records),
        "records_with_num_frames": records_with_num_frames,
        "records_with_fps": records_with_fps,
        "records_with_frame_metadata": records_with_frame_metadata,
        "expected_anomaly_records": OFFICIAL_UCF_CRIME_COUNTS["test"]["anomaly"],
        "anomaly_records": anomaly_records,
        "anomaly_records_with_1_or_2_positive_frame_spans": (anomaly_records_with_valid_spans),
        "expected_normal_records": OFFICIAL_UCF_CRIME_COUNTS["test"]["normal"],
        "normal_records": normal_records,
        "normal_records_without_positive_frame_spans": (normal_records_without_anomaly_spans),
        "positive_frame_spans": positive_frame_spans,
        "not_ready_records": not_ready_records,
    }


def _check_manifest_probe_consistency(
    record: VideoManifestRecord,
    info: VideoInfo,
    *,
    source_split: str,
    errors: list[dict[str, Any]],
) -> None:
    conflicts: dict[str, dict[str, int | float]] = {}
    if record.num_frames is not None and record.num_frames != info.num_frames:
        conflicts["num_frames"] = {"manifest": record.num_frames, "container": info.num_frames}
    if record.fps is not None and not math.isclose(record.fps, info.fps, rel_tol=1e-3):
        conflicts["fps"] = {"manifest": record.fps, "container": info.fps}
    tolerance = max(0.1, 1.0 / info.fps)
    if record.duration_seconds is not None and not math.isclose(
        record.duration_seconds,
        info.duration_seconds,
        abs_tol=tolerance,
        rel_tol=0.0,
    ):
        conflicts["duration_seconds"] = {
            "manifest": record.duration_seconds,
            "container": info.duration_seconds,
        }
    if conflicts:
        errors.append(
            _issue(
                "manifest_container_mismatch",
                f"{record.video_id} 的 manifest 元数据与视频容器不一致",
                split=source_split,
                video_id=record.video_id,
                path=record.path,
                details={"conflicts": conflicts},
            )
        )


def audit_ucf_crime_dataset(
    dataset_root: str | Path,
    train_manifest: ManifestSource,
    test_manifest: ManifestSource,
    *,
    deep_hash: bool = False,
    probe_fn: ProbeFunction = probe_video,
    backend: Any | None = None,
    official_source_registry: str | Path = "registry/datasets.yaml",
) -> dict[str, Any]:
    """审计真实 UCF-Crime 数据、官方划分、容器元数据和可选内容哈希。

    返回值只包含严格 JSON 可序列化类型。任何缺文件、manifest 解析失败、官方
    计数不符、重复身份/路径、非法类别、容器探测失败或元数据冲突都会令
    ``passed=false``。警告会被单独保留，不会被误算为硬错误。

    Args:
        dataset_root: manifest 中相对视频路径的解析根目录。
        train_manifest: train JSONL 路径或内存记录 iterable。
        test_manifest: test JSONL 路径或内存记录 iterable。
        deep_hash: 仅为 ``True`` 时读取每个完整视频并计算 SHA256。
        probe_fn: 默认使用只探测容器的 :func:`probe_video`；参数用于测试或后端适配。
        backend: 原样传递给 ``probe_fn`` 的可选视频后端。
        official_source_registry: 冻结官方 split/temporal 文件路径与 SHA256 的登记表。
    """

    if not isinstance(deep_hash, bool):
        raise TypeError("deep_hash 必须是 boolean")
    if not callable(probe_fn):
        raise TypeError("probe_fn 必须可调用")

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    root_input = Path(dataset_root).expanduser()
    root = root_input.resolve()
    if not root.exists():
        errors.append(
            _issue(
                "dataset_root_missing",
                f"dataset_root 不存在：{root_input}",
                path=str(root_input),
            )
        )
    elif not root.is_dir():
        errors.append(
            _issue(
                "dataset_root_not_directory",
                f"dataset_root 不是目录：{root_input}",
                path=str(root_input),
            )
        )

    records_by_split = {
        "train": _load_manifest_source(
            train_manifest,
            source_split="train",
            errors=errors,
        ),
        "test": _load_manifest_source(
            test_manifest,
            source_split="test",
            errors=errors,
        ),
    }
    official_source_identity = verify_official_source_identity(
        records_by_split,
        registry_path=official_source_registry,
        errors=errors,
    )

    for source_split, records in records_by_split.items():
        expected_enum = DatasetSplit(source_split)
        for record in records:
            if record.split != expected_enum:
                errors.append(
                    _issue(
                        "record_split_mismatch",
                        f"{source_split} manifest 中的 {record.video_id} 声明 split={record.split.value}",
                        split=source_split,
                        video_id=record.video_id,
                        path=record.path,
                        details={"declared_split": record.split.value},
                    )
                )
            if record.category not in UCF_CRIME_CATEGORIES:
                errors.append(
                    _issue(
                        "unknown_category",
                        f"{record.video_id} 使用未知 UCF-Crime 类别 {record.category!r}",
                        split=source_split,
                        video_id=record.video_id,
                        path=record.path,
                    )
                )
            expected_anomaly = record.category != "Normal"
            if record.is_anomaly != expected_anomaly:
                errors.append(
                    _issue(
                        "category_label_mismatch",
                        f"{record.video_id} 的 category={record.category!r} 与 is_anomaly 不一致",
                        split=source_split,
                        video_id=record.video_id,
                        path=record.path,
                    )
                )

    duplicate_ids, duplicate_paths = _duplicate_groups(records_by_split)
    for group in duplicate_ids:
        errors.append(
            _issue(
                "duplicate_canonical_video_id",
                f"规范 video_id={group['key']!r} 出现 {len(group['occurrences'])} 次",
                details={"key": group["key"], "occurrences": group["occurrences"]},
            )
        )
    for group in duplicate_paths:
        errors.append(
            _issue(
                "duplicate_normalized_path",
                f"规范 path={group['key']!r} 出现 {len(group['occurrences'])} 次",
                details={"key": group["key"], "occurrences": group["occurrences"]},
            )
        )

    category_distribution: dict[str, dict[str, int]] = {}
    observed: dict[str, dict[str, int]] = {}
    for source_split, records in records_by_split.items():
        category_distribution[source_split] = dict(
            sorted(Counter(record.category for record in records).items())
        )
        observed[source_split] = _observed_counts(records)
        missing_categories = sorted(
            set(UCF_CRIME_CATEGORIES) - set(category_distribution[source_split])
        )
        if missing_categories:
            errors.append(
                _issue(
                    "official_categories_missing",
                    f"{source_split} manifest 缺少官方类别：{missing_categories}",
                    split=source_split,
                    details={"missing_categories": missing_categories},
                )
            )

    all_records = (*records_by_split["train"], *records_by_split["test"])
    category_distribution["all"] = dict(
        sorted(Counter(record.category for record in all_records).items())
    )
    observed["all"] = _observed_counts(all_records)
    count_checks: list[dict[str, Any]] = []
    for split_name in ("train", "test", "all"):
        for field_name in ("total", "normal", "anomaly"):
            count_checks.append(
                _check_count(
                    name=f"{split_name}.{field_name}",
                    expected=OFFICIAL_UCF_CRIME_COUNTS[split_name][field_name],
                    actual=observed[split_name][field_name],
                    errors=errors,
                )
            )

    evaluation_readiness = _evaluation_readiness(
        records_by_split["test"],
        errors=errors,
    )

    video_entries: list[dict[str, Any]] = []
    missing_files: list[dict[str, str]] = []
    files_present = 0
    stat_errors = 0
    total_size_bytes = 0
    probed = 0
    probe_errors = 0
    total_num_frames = 0
    total_duration_seconds = 0.0
    hashed_files = 0
    hash_errors = 0
    hashes: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for source_split, records in records_by_split.items():
        for index, record in enumerate(records):
            entry: dict[str, Any] = {
                "split": source_split,
                "index": index,
                "video_id": record.video_id,
                "path": record.path,
                "exists": False,
                "file_size_bytes": None,
                "probe": {"status": "missing"},
            }
            try:
                resolved_path = record.resolve_path(root)
            except (ManifestError, OSError) as exc:
                errors.append(
                    _issue(
                        "video_path_resolution_error",
                        f"{record.video_id} 的路径无法安全解析：{exc}",
                        split=source_split,
                        video_id=record.video_id,
                        path=record.path,
                    )
                )
                entry["probe"] = {"status": "error", "message": str(exc)}
                video_entries.append(entry)
                continue

            if not resolved_path.is_file():
                missing = {
                    "split": source_split,
                    "video_id": record.video_id,
                    "path": record.path,
                }
                missing_files.append(missing)
                errors.append(
                    _issue(
                        "video_file_missing",
                        f"视频文件不存在：{record.path}",
                        split=source_split,
                        video_id=record.video_id,
                        path=record.path,
                    )
                )
                video_entries.append(entry)
                continue

            entry["exists"] = True
            files_present += 1
            try:
                file_size = resolved_path.stat().st_size
                entry["file_size_bytes"] = file_size
                total_size_bytes += file_size
                if file_size <= 0:
                    errors.append(
                        _issue(
                            "video_file_empty",
                            f"视频文件为空：{record.path}",
                            split=source_split,
                            video_id=record.video_id,
                            path=record.path,
                        )
                    )
            except OSError as exc:
                stat_errors += 1
                errors.append(
                    _issue(
                        "video_stat_error",
                        f"读取视频文件元数据失败：{exc}",
                        split=source_split,
                        video_id=record.video_id,
                        path=record.path,
                        details={"exception_type": type(exc).__name__},
                    )
                )

            try:
                info = probe_fn(resolved_path, backend=backend)
                if not isinstance(info, VideoInfo):
                    raise TypeError(f"probe_fn 必须返回 VideoInfo，实际为 {type(info).__name__}")
                entry["probe"] = _probe_payload(info)
                probed += 1
                total_num_frames += info.num_frames
                total_duration_seconds += info.duration_seconds
                _check_manifest_probe_consistency(
                    record,
                    info,
                    source_split=source_split,
                    errors=errors,
                )
            except Exception as exc:
                probe_errors += 1
                entry["probe"] = {
                    "status": "error",
                    "message": str(exc),
                    "exception_type": type(exc).__name__,
                }
                errors.append(
                    _issue(
                        "video_probe_error",
                        f"探测视频容器失败：{exc}",
                        split=source_split,
                        video_id=record.video_id,
                        path=record.path,
                        details={"exception_type": type(exc).__name__},
                    )
                )

            if deep_hash:
                try:
                    digest = _sha256_file(resolved_path)
                    entry["sha256"] = digest
                    hashed_files += 1
                    hashes[digest].append(_occurrence(source_split, index, record))
                except OSError as exc:
                    hash_errors += 1
                    entry["sha256"] = None
                    errors.append(
                        _issue(
                            "video_hash_error",
                            f"计算 SHA256 失败：{exc}",
                            split=source_split,
                            video_id=record.video_id,
                            path=record.path,
                            details={"exception_type": type(exc).__name__},
                        )
                    )
            video_entries.append(entry)

    duplicate_hashes = [
        {"sha256": digest, "occurrences": occurrences}
        for digest, occurrences in sorted(hashes.items())
        if len(occurrences) > 1
    ]
    for group in duplicate_hashes:
        split_names = {item["split"] for item in group["occurrences"]}
        issue_target = errors if len(split_names) > 1 else warnings
        issue_target.append(
            _issue(
                "duplicate_sha256_cross_split" if len(split_names) > 1 else "duplicate_sha256",
                f"SHA256={group['sha256']} 对应 {len(group['occurrences'])} 个文件",
                details={"sha256": group["sha256"], "occurrences": group["occurrences"]},
            )
        )

    warnings.append(
        _issue(
            "visual_near_duplicate_not_run",
            "未执行基于感知哈希或视觉嵌入的近重复检测；该结果不能排除内容近重复。",
        )
    )

    requested_hashes = files_present
    if not deep_hash:
        hash_status = "not_run"
    elif hash_errors or hashed_files != requested_hashes:
        hash_status = "partial"
    else:
        hash_status = "complete"

    passed = not errors and not missing_files and evaluation_readiness["ready"]
    status = "failed" if not passed else ("passed_with_warnings" if warnings else "passed")
    report: dict[str, Any] = {
        "schema_version": DATASET_AUDIT_SCHEMA_VERSION,
        "generated_at": _utc_timestamp(),
        "dataset": "ucf_crime",
        "dataset_root": str(root_input),
        "manifests": {
            "train": _source_label(train_manifest),
            "test": _source_label(test_manifest),
        },
        "manifest_sha256": {
            "train": compute_manifest_sha256(records_by_split["train"]),
            "test": compute_manifest_sha256(records_by_split["test"]),
        },
        "official_source_identity": official_source_identity,
        "deep_hash": deep_hash,
        "status": status,
        "passed": passed,
        "expected": {split: dict(counts) for split, counts in OFFICIAL_UCF_CRIME_COUNTS.items()},
        "observed": observed,
        "count_checks": count_checks,
        "category_distribution": category_distribution,
        "evaluation_readiness": evaluation_readiness,
        "files": {
            "manifest_records": len(all_records),
            "present": files_present,
            "missing": len(missing_files),
            "stat_errors": stat_errors,
            "total_size_bytes": total_size_bytes,
            "probed": probed,
            "probe_errors": probe_errors,
            "total_num_frames": total_num_frames,
            "total_duration_seconds": total_duration_seconds,
        },
        "duplicates": {
            "canonical_video_ids": duplicate_ids,
            "normalized_paths": duplicate_paths,
        },
        "hashing": {
            "algorithm": "sha256",
            "requested": deep_hash,
            "status": hash_status,
            "hashed_files": hashed_files,
            "hash_errors": hash_errors,
            "duplicate_groups": duplicate_hashes,
        },
        "near_duplicate_detection": {
            "status": "not_run",
            "method": None,
            "reason": "本模块未执行感知哈希或视觉嵌入近重复检测。",
        },
        "videos": video_entries,
        "missing_files": missing_files,
        "errors": errors,
        "warnings": warnings,
    }
    # 在返回前执行一次严格 JSON 编码，防止 probe/backend 意外泄漏不可序列化值。
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report


__all__ = [
    "DATASET_AUDIT_SCHEMA_VERSION",
    "OFFICIAL_UCF_CRIME_COUNTS",
    "audit_ucf_crime_dataset",
    "compute_manifest_sha256",
]
