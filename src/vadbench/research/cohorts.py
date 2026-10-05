"""Cohort index validation and label-access boundaries for observations."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class CohortError(ValueError):
    """A cohort sidecar is malformed or would violate the research protocol."""


class LabelPolicy(str, Enum):
    """Permitted development-label paths from the observation protocol."""

    WEAK = "W"
    DIAGNOSTIC = "D"


_PARTITIONS = frozenset({"fit", "confirm", "select", "test"})
_WEAK_LABELS = frozenset({0, 1})


@dataclass(frozen=True, slots=True)
class CohortRecord:
    """One clip selection in a label sidecar, never passed to an encoder."""

    video_id: str
    clip_id: str
    official_split: str
    partition: str
    role: str
    weak_label: int | None
    label_source: str
    source_id: str | None = None
    temporal_label: Any | None = None
    source_frames: tuple[int, ...] | None = None
    clip_start_s: float | None = None
    clip_end_s: float | None = None
    scene_group: str | None = None
    motion_bin: str | None = None
    category_for_analysis_only: str | None = None
    pseudo_positive: bool = False

    def __post_init__(self) -> None:
        for name in ("video_id", "clip_id", "official_split", "partition", "role", "label_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CohortError(f"{name} 必须是非空字符串")
        if self.partition not in _PARTITIONS:
            raise CohortError(f"partition 必须是 {sorted(_PARTITIONS)} 之一：{self.partition!r}")
        if self.weak_label is not None and (
            type(self.weak_label) is not int or self.weak_label not in _WEAK_LABELS
        ):
            raise CohortError("weak_label 只能是 0、1 或 null")
        if self.source_id is not None and (
            not isinstance(self.source_id, str) or not self.source_id
        ):
            raise CohortError("source_id 必须为非空字符串或 null")
        if self.source_frames is not None:
            if not self.source_frames or any(
                not isinstance(value, int) or value < 0 for value in self.source_frames
            ):
                raise CohortError("source_frames 必须是非空的非负整数序列")
            if any(
                right <= left
                for left, right in zip(self.source_frames, self.source_frames[1:], strict=False)
            ):
                raise CohortError("source_frames 必须严格递增")
        bounds = (self.clip_start_s, self.clip_end_s)
        if (bounds[0] is None) != (bounds[1] is None):
            raise CohortError("clip_start_s 和 clip_end_s 必须同时提供或同时省略")
        if bounds[0] is not None:
            if not isinstance(bounds[0], (float, int)) or not isinstance(bounds[1], (float, int)):
                raise CohortError("clip 时间边界必须是数值")
            if (
                not all(math.isfinite(float(x)) for x in bounds)
                or float(bounds[0]) < 0
                or float(bounds[1]) <= float(bounds[0])
            ):
                raise CohortError("clip 时间边界必须满足 0 <= start < end")

    @property
    def grouping_id(self) -> str:
        """Source/near-duplicate grouping key used to prevent partition leakage."""

        return self.source_id or self.video_id

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CohortRecord:
        if not isinstance(value, Mapping):
            raise CohortError("cohort 行必须是对象")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - allowed
        if unknown:
            raise CohortError(f"cohort 行有未知字段：{sorted(unknown)}")
        payload = dict(value)
        frames = payload.get("source_frames")
        if frames is not None:
            if not isinstance(frames, (list, tuple)):
                raise CohortError("source_frames 必须是数组")
            payload["source_frames"] = tuple(frames)
        try:
            return cls(**payload)
        except TypeError as exc:
            raise CohortError(f"cohort 行缺少字段或字段不合法：{exc}") from exc

    def as_dict(self) -> dict[str, Any]:
        result = {
            field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()
        }
        if self.source_frames is not None:
            result["source_frames"] = list(self.source_frames)
        return result


@dataclass(frozen=True, slots=True)
class CohortIndex:
    """Validated collection of observation-sidecar records."""

    records: tuple[CohortRecord, ...]
    policy: LabelPolicy

    def __post_init__(self) -> None:
        if not self.records:
            raise CohortError("cohort 不能为空")
        clip_ids: set[str] = set()
        grouping_partitions: dict[str, str] = {}
        video_partitions: dict[str, str] = {}
        video_labels: dict[str, int | None] = {}
        for record in self.records:
            if record.clip_id in clip_ids:
                raise CohortError(f"clip_id 重复：{record.clip_id}")
            clip_ids.add(record.clip_id)
            previous = grouping_partitions.setdefault(record.grouping_id, record.partition)
            if previous != record.partition:
                raise CohortError(
                    f"source/video 分组跨 partition：{record.grouping_id} ({previous}, {record.partition})"
                )
            if video_partitions.setdefault(record.video_id, record.partition) != record.partition:
                raise CohortError(f"video 跨 partition：{record.video_id}")
            if video_labels.setdefault(record.video_id, record.weak_label) != record.weak_label:
                raise CohortError(f"同一 video 的 weak_label 不一致：{record.video_id}")
            if record.official_split != "train" or record.partition == "test":
                raise CohortError(
                    "development cohort only accepts official train videos; test access is separate"
                )
            roles = {"debug": "fit", "explore": "fit", "confirm": "confirm", "select": "select"}
            if roles.get(record.role) != record.partition:
                raise CohortError("role 与 partition 不匹配")
            self._validate_record_for_policy(record)

    def _validate_record_for_policy(self, record: CohortRecord) -> None:
        if self.policy is LabelPolicy.WEAK:
            if record.temporal_label is not None:
                raise CohortError("W 路径不允许 cohort sidecar 含 temporal_label（包括 test）")
            if record.weak_label is None:
                raise CohortError("W 路径的 cohort 行必须有训练视频级 weak_label")
            if record.label_source != "video_weak":
                raise CohortError("W 路径 label_source 必须是 video_weak")
        elif self.policy is LabelPolicy.DIAGNOSTIC:
            if record.weak_label is None and record.temporal_label is None:
                raise CohortError("D 路径每行至少需要 weak_label 或 temporal_label")
        else:  # defensive against deserialised invalid enums
            raise CohortError(f"未知标签策略：{self.policy!r}")

    @classmethod
    def from_records(
        cls, records: Iterable[CohortRecord | Mapping[str, Any]], *, policy: LabelPolicy | str
    ) -> CohortIndex:
        try:
            selected_policy = LabelPolicy(policy)
        except ValueError as exc:
            raise CohortError("policy 必须是 W 或 D") from exc
        normalized = tuple(
            item if isinstance(item, CohortRecord) else CohortRecord.from_mapping(item)
            for item in records
        )
        return cls(records=normalized, policy=selected_policy)

    @classmethod
    def load_jsonl(cls, path: str | Path, *, policy: LabelPolicy | str) -> CohortIndex:
        selected = Path(path)
        records: list[CohortRecord] = []
        try:
            with selected.open("r", encoding="utf-8") as stream:
                for number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise CohortError(f"{selected}:{number} 不是合法 JSON") from exc
                    try:
                        records.append(CohortRecord.from_mapping(value))
                    except CohortError as exc:
                        raise CohortError(f"{selected}:{number}: {exc}") from exc
        except OSError as exc:
            raise CohortError(f"无法读取 cohort 文件 {selected}") from exc
        return cls.from_records(records, policy=policy)

    def record_for_video(self, video_id: str) -> CohortRecord:
        matches = [record for record in self.records if record.video_id == video_id]
        if not matches:
            raise CohortError(f"cohort 中不存在 video_id：{video_id}")
        if len(matches) != 1:
            raise CohortError(f"一个 video_id 有多个 clip；请按 clip_id join：{video_id}")
        return matches[0]

    def record_for_clip(self, clip_id: str) -> CohortRecord:
        for record in self.records:
            if record.clip_id == clip_id:
                return record
        raise CohortError(f"cohort 中不存在 clip_id：{clip_id}")

    def label_sidecar(self) -> dict[str, dict[str, Any]]:
        """Return analysis-only labels keyed by clip ID, never for model calls."""

        return {
            record.clip_id: {
                "video_id": record.video_id,
                "partition": record.partition,
                "weak_label": record.weak_label,
                "temporal_label": record.temporal_label,
                "label_source": record.label_source,
                "scene_group": record.scene_group,
                "motion_bin": record.motion_bin,
                "category_for_analysis_only": record.category_for_analysis_only,
                "pseudo_positive": record.pseudo_positive,
            }
            for record in self.records
        }
