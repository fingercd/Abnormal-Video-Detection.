"""JSONL 视频清单协议、数据类与跨切分一致性校验。

清单中的路径始终相对于数据集根目录，并使用 POSIX 分隔符。时间区间统一采用
左闭右开语义 ``[start, end)``，避免训练采样与逐帧评测之间产生边界歧义。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from vadbench.features import atomic_write_jsonl

MANIFEST_SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """清单内容、路径或切分关系不合法。"""


class SupervisionScope(str, Enum):
    """一条标注所表达的监督粒度。"""

    VIDEO = "video"
    SEGMENT = "segment"
    FRAME = "frame"
    CAPTION = "caption"


class SpanUnit(str, Enum):
    """时间区间坐标单位。"""

    FRAME = "frame"
    SECOND = "second"
    SEGMENT = "segment"


class DatasetSplit(str, Enum):
    """框架支持的数据切分。"""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def _as_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ManifestError(f"{field_name} 必须是 {choices} 之一，实际为 {value!r}") from exc


def _json_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    result = dict(value or {})
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field_name} 必须可序列化为严格 JSON") from exc
    return result


def normalize_relative_path(path: str | Path) -> str:
    """规范化并校验数据集相对路径，拒绝绝对路径和目录穿越。"""

    raw = str(path).strip().replace("\\", "/")
    if not raw:
        raise ManifestError("视频 path 不能为空")
    windows_path = PureWindowsPath(raw)
    if windows_path.drive or windows_path.is_absolute() or PurePosixPath(raw).is_absolute():
        raise ManifestError(f"视频 path 必须相对于 dataset root：{path!r}")

    raw_parts = raw.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ManifestError(f"视频 path 含非法目录片段：{path!r}")
    parts = PurePosixPath(raw).parts
    normalized = PurePosixPath(*parts).as_posix()
    if normalized.startswith("//"):
        raise ManifestError(f"视频 path 不能是网络绝对路径：{path!r}")
    return normalized


_FEATURE_SUFFIX = re.compile(r"(?:__\d+|_C)$", re.IGNORECASE)


def canonical_video_id(value: str | Path) -> str:
    """生成用于去重和防泄漏的稳定视频身份。

    除普通视频扩展名外，也兼容 ``video__17.npy`` 和 ``video_C.txt`` 这类
    已切片特征列表，确保同一原视频不会借由特征文件名绕过泄漏检测。
    """

    name = PurePosixPath(str(value).replace("\\", "/")).name
    lowered = name.casefold()
    known_suffixes = (
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".webm",
        ".npy",
        ".npz",
        ".txt",
        ".fc6-1",
    )
    for suffix in known_suffixes:
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = _FEATURE_SUFFIX.sub("", name)
    identity = name.strip().casefold()
    if not identity:
        raise ManifestError(f"无法从 {value!r} 提取 video_id")
    return identity


@dataclass(frozen=True)
class TemporalSpan:
    """左闭右开的时间区间。"""

    start: float
    end: float
    unit: SpanUnit | str

    def __post_init__(self) -> None:
        unit = _as_enum(self.unit, SpanUnit, "span.unit")
        object.__setattr__(self, "unit", unit)
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise ManifestError("span.start/end 必须是数值")
        try:
            start = float(self.start)
            end = float(self.end)
        except (TypeError, ValueError) as exc:
            raise ManifestError("span.start/end 必须是数值") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ManifestError("span.start/end 必须是有限数值")
        if start < 0 or end <= start:
            raise ManifestError(f"span 必须满足 0 <= start < end，实际为 [{start}, {end})")
        if unit in {SpanUnit.FRAME, SpanUnit.SEGMENT} and (
            not start.is_integer() or not end.is_integer()
        ):
            raise ManifestError(f"{unit.value} 坐标必须是整数")
        object.__setattr__(self, "start", int(start) if start.is_integer() else start)
        object.__setattr__(self, "end", int(end) if end.is_integer() else end)

    @property
    def start_frame(self) -> int | None:
        return int(self.start) if self.unit == SpanUnit.FRAME else None

    @property
    def end_frame(self) -> int | None:
        return int(self.end) if self.unit == SpanUnit.FRAME else None

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "unit": self.unit.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TemporalSpan:
        allowed = {"start", "end", "unit"}
        unknown = set(data) - allowed
        if unknown:
            raise ManifestError(f"span 含未知字段：{sorted(unknown)}")
        missing = allowed - set(data)
        if missing:
            raise ManifestError(f"span 缺少字段：{sorted(missing)}")
        return cls(start=data["start"], end=data["end"], unit=data["unit"])


@dataclass(frozen=True)
class SupervisionAnnotation:
    """视频、区间、逐帧或自然语言标注。

    ``caption`` 标注的 ``is_anomaly`` 默认并保持为 ``None``。描述文本本身不是
    异常真值，调用方必须另行显式映射后才能把它用于异常监督。
    """

    scope: SupervisionScope | str
    label: str | None = None
    is_anomaly: bool | None = None
    span: TemporalSpan | None = None
    text: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scope = _as_enum(self.scope, SupervisionScope, "annotation.scope")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "annotation.metadata"))

        if self.label is not None and not str(self.label).strip():
            raise ManifestError("annotation.label 不能是空字符串")
        if self.is_anomaly is not None and not isinstance(self.is_anomaly, bool):
            raise ManifestError("annotation.is_anomaly 必须是 boolean 或 null")
        if self.source is not None and not str(self.source).strip():
            raise ManifestError("annotation.source 不能是空字符串")

        if scope == SupervisionScope.VIDEO:
            if self.span is not None:
                raise ManifestError("video scope 不应携带 span")
            if self.label is None and self.is_anomaly is None:
                raise ManifestError("video scope 至少需要 label 或 is_anomaly")
        elif scope == SupervisionScope.FRAME:
            if self.span is None or self.span.unit != SpanUnit.FRAME:
                raise ManifestError("frame scope 必须携带 unit=frame 的 span")
        elif scope == SupervisionScope.SEGMENT:
            if self.span is None:
                raise ManifestError("segment scope 必须携带 span")
        elif scope == SupervisionScope.CAPTION:
            if self.text is None or not self.text.strip():
                raise ManifestError("caption scope 必须携带非空 text")

        if scope != SupervisionScope.CAPTION and self.text is not None:
            raise ManifestError("只有 caption scope 可以携带 text")

    @property
    def start_frame(self) -> int | None:
        return self.span.start_frame if self.span is not None else None

    @property
    def end_frame(self) -> int | None:
        return self.span.end_frame if self.span is not None else None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "scope": self.scope.value,
            # null 明确表示“未提供异常真值”，尤其用于 UCA caption。
            "is_anomaly": self.is_anomaly,
        }
        if self.label is not None:
            result["label"] = self.label
        if self.span is not None:
            result["span"] = self.span.to_dict()
        if self.text is not None:
            result["text"] = self.text
        if self.source is not None:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SupervisionAnnotation:
        allowed = {"scope", "label", "is_anomaly", "span", "text", "source", "metadata"}
        unknown = set(data) - allowed
        if unknown:
            raise ManifestError(f"annotation 含未知字段：{sorted(unknown)}")
        missing = {"scope", "is_anomaly"} - set(data)
        if missing:
            raise ManifestError(f"annotation 缺少字段：{sorted(missing)}")
        raw_span = data.get("span")
        if raw_span is not None and not isinstance(raw_span, Mapping):
            raise ManifestError("annotation.span 必须是对象")
        return cls(
            scope=data["scope"],
            label=data.get("label"),
            is_anomaly=data.get("is_anomaly"),
            span=TemporalSpan.from_dict(raw_span) if raw_span is not None else None,
            text=data.get("text"),
            source=data.get("source"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class VideoManifestRecord:
    """JSONL 中的一条视频记录。"""

    video_id: str
    path: str
    split: DatasetSplit | str
    category: str
    is_anomaly: bool
    annotations: Sequence[SupervisionAnnotation] = field(default_factory=tuple)
    num_frames: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"仅支持 schema_version={MANIFEST_SCHEMA_VERSION}，实际为 {self.schema_version!r}"
            )
        video_id = str(self.video_id).strip()
        if not video_id or any(char in video_id for char in "/\\"):
            raise ManifestError(f"video_id 必须是非空 basename：{self.video_id!r}")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "path", normalize_relative_path(self.path))
        split = _as_enum(self.split, DatasetSplit, "split")
        object.__setattr__(self, "split", split)

        category = str(self.category).strip()
        if not category:
            raise ManifestError("category 不能为空")
        object.__setattr__(self, "category", category)
        if not isinstance(self.is_anomaly, bool):
            raise ManifestError("is_anomaly 必须是 boolean")

        annotations = tuple(self.annotations)
        if any(not isinstance(item, SupervisionAnnotation) for item in annotations):
            raise ManifestError("annotations 必须由 SupervisionAnnotation 组成")
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))

        if self.num_frames is not None and (
            isinstance(self.num_frames, bool)
            or not isinstance(self.num_frames, int)
            or self.num_frames <= 0
        ):
            raise ManifestError("num_frames 必须是正整数或 null")
        if self.fps is not None and (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (int, float))
            or not math.isfinite(float(self.fps))
            or float(self.fps) <= 0
        ):
            raise ManifestError("fps 必须是有限正数或 null")
        if self.duration_seconds is not None and (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or float(self.duration_seconds) <= 0
        ):
            raise ManifestError("duration_seconds 必须是有限正数或 null")

        for annotation in annotations:
            if (
                annotation.scope == SupervisionScope.VIDEO
                and annotation.is_anomaly is not None
                and annotation.is_anomaly != self.is_anomaly
            ):
                raise ManifestError(f"{video_id} 的 video annotation 与 record.is_anomaly 不一致")
            if annotation.span is None:
                continue
            if (
                annotation.span.unit == SpanUnit.FRAME
                and self.num_frames is not None
                and annotation.span.end > self.num_frames
            ):
                raise ManifestError(
                    f"{video_id} 的 frame annotation 结束于 {annotation.span.end}，"
                    f"越过 num_frames={self.num_frames}"
                )
            if (
                annotation.span.unit == SpanUnit.SECOND
                and self.duration_seconds is not None
                and float(annotation.span.end) > float(self.duration_seconds) + 1e-6
            ):
                raise ManifestError(
                    f"{video_id} 的 second annotation 结束于 {annotation.span.end}，"
                    f"越过 duration_seconds={self.duration_seconds}"
                )

    @property
    def supervision_scopes(self) -> frozenset[SupervisionScope]:
        return frozenset(item.scope for item in self.annotations)

    def resolve_path(self, dataset_root: str | Path) -> Path:
        root = Path(dataset_root).expanduser().resolve()
        resolved = root.joinpath(*PurePosixPath(self.path).parts).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestError(f"视频路径逃逸 dataset root：{self.path!r}") from exc
        return resolved

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "path": self.path,
            "split": self.split.value,
            "category": self.category,
            "is_anomaly": self.is_anomaly,
            "annotations": [item.to_dict() for item in self.annotations],
        }
        if self.num_frames is not None:
            result["num_frames"] = self.num_frames
        if self.fps is not None:
            result["fps"] = self.fps
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VideoManifestRecord:
        allowed = {
            "schema_version",
            "video_id",
            "path",
            "split",
            "category",
            "is_anomaly",
            "annotations",
            "num_frames",
            "fps",
            "duration_seconds",
            "metadata",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ManifestError(f"manifest record 含未知字段：{sorted(unknown)}")
        required = {
            "schema_version",
            "video_id",
            "path",
            "split",
            "category",
            "is_anomaly",
            "annotations",
        }
        missing = required - set(data)
        if missing:
            raise ManifestError(f"manifest record 缺少字段：{sorted(missing)}")
        raw_annotations = data.get("annotations", [])
        if not isinstance(raw_annotations, list):
            raise ManifestError("annotations 必须是数组")
        if any(not isinstance(item, Mapping) for item in raw_annotations):
            raise ManifestError("annotations 的每一项必须是对象")
        return cls(
            schema_version=data["schema_version"],
            video_id=data["video_id"],
            path=data["path"],
            split=data["split"],
            category=data["category"],
            is_anomaly=data["is_anomaly"],
            annotations=tuple(SupervisionAnnotation.from_dict(item) for item in raw_annotations),
            num_frames=data.get("num_frames"),
            fps=data.get("fps"),
            duration_seconds=data.get("duration_seconds"),
            metadata=data.get("metadata", {}),
        )


# 协议文档承诺保留的兼容名称。
ManifestRecord = VideoManifestRecord


def assert_no_split_leakage(records: Iterable[VideoManifestRecord]) -> None:
    """确保同一原视频不会同时出现在不同切分。"""

    seen_ids: dict[str, DatasetSplit] = {}
    seen_paths: dict[str, DatasetSplit] = {}
    for record in records:
        keys = (
            (seen_ids, canonical_video_id(record.video_id), "video_id"),
            (seen_paths, canonical_video_id(record.path), "path"),
        )
        for seen, key, key_kind in keys:
            previous = seen.get(key)
            if previous is not None and previous != record.split:
                raise ManifestError(
                    f"检测到切分泄漏：{key_kind}={key!r} 同时出现在 "
                    f"{previous.value} 与 {record.split.value}"
                )
            seen[key] = record.split


def validate_manifest(
    records: Iterable[VideoManifestRecord],
    *,
    dataset_root: str | Path | None = None,
    require_files: bool = False,
) -> tuple[VideoManifestRecord, ...]:
    """完整校验记录、同切分唯一性和跨切分泄漏，并返回不可变副本。"""

    items = tuple(records)
    if require_files and dataset_root is None:
        raise ManifestError("require_files=True 时必须提供 dataset_root")

    seen_ids_per_split: set[tuple[DatasetSplit, str]] = set()
    seen_paths_per_split: set[tuple[DatasetSplit, str]] = set()
    for record in items:
        if not isinstance(record, VideoManifestRecord):
            raise ManifestError("manifest 只能包含 VideoManifestRecord")
        id_key = (record.split, canonical_video_id(record.video_id))
        if id_key in seen_ids_per_split:
            raise ManifestError(f"{record.split.value} 内出现重复视频：{record.video_id!r}")
        seen_ids_per_split.add(id_key)
        path_key = (record.split, record.path.casefold())
        if path_key in seen_paths_per_split:
            raise ManifestError(f"{record.split.value} 内出现重复路径：{record.path!r}")
        seen_paths_per_split.add(path_key)
        if require_files and not record.resolve_path(dataset_root).is_file():
            raise ManifestError(f"视频文件不存在：{record.resolve_path(dataset_root)}")

    assert_no_split_leakage(items)
    return items


def load_manifest_jsonl(
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    require_files: bool = False,
) -> tuple[VideoManifestRecord, ...]:
    """以 UTF-8 读取并校验 JSONL 清单。"""

    manifest_path = Path(path)
    records: list[VideoManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"{manifest_path}:{line_number} 不是合法 JSON：{exc.msg}"
                ) from exc
            if not isinstance(data, Mapping):
                raise ManifestError(f"{manifest_path}:{line_number} 顶层必须是对象")
            try:
                records.append(VideoManifestRecord.from_dict(data))
            except ManifestError as exc:
                raise ManifestError(f"{manifest_path}:{line_number}: {exc}") from exc
    return validate_manifest(records, dataset_root=dataset_root, require_files=require_files)


def write_manifest_jsonl(
    records: Iterable[VideoManifestRecord],
    path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    require_files: bool = False,
) -> Path:
    """校验后原子写入 UTF-8 JSONL。"""

    items = validate_manifest(records, dataset_root=dataset_root, require_files=require_files)
    output_path = Path(path)
    atomic_write_jsonl(output_path, (record.to_dict() for record in items))
    return output_path


def validate_manifest_pair(
    train_records: Iterable[VideoManifestRecord],
    test_records: Iterable[VideoManifestRecord],
    *,
    dataset_root: str | Path | None = None,
    require_files: bool = False,
) -> tuple[tuple[VideoManifestRecord, ...], tuple[VideoManifestRecord, ...]]:
    """校验 train/test 各自内容以及两者之间的严格无泄漏约束。"""

    train = tuple(train_records)
    test = tuple(test_records)
    for record in train:
        if record.split != DatasetSplit.TRAIN:
            raise ManifestError(f"train manifest 含非 train 记录：{record.video_id}")
    for record in test:
        if record.split != DatasetSplit.TEST:
            raise ManifestError(f"test manifest 含非 test 记录：{record.video_id}")
    validate_manifest((*train, *test), dataset_root=dataset_root, require_files=require_files)
    return train, test
