"""UCF-Crime 官方切分、帧级真值和 UCA caption 导入。

UCF-Crime 官方帧级文件每行格式为::

    filename category start1 end1 start2 end2

``-1 -1`` 表示该事件区间不存在。官方端点与作者 MATLAB evaluator 一致，采用
1-based inclusive；导入时转换为框架统一的 zero-based half-open 区间，即
``[raw_start - 1, raw_end)``。社区常用评测清单还会在 filename 后插入总帧数；
本模块显式兼容两种格式。UCA caption 是事件描述而非异常真值，导入时始终保留
``is_anomaly=None``，不得由文本内容自动推断异常标签。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .manifest import (
    DatasetSplit,
    ManifestError,
    SpanUnit,
    SupervisionAnnotation,
    SupervisionScope,
    TemporalSpan,
    VideoManifestRecord,
    canonical_video_id,
    normalize_relative_path,
    validate_manifest_pair,
    write_manifest_jsonl,
)
from .video import probe_video

UCF_CRIME_CATEGORIES = (
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
    "Normal",
)
_CATEGORY_LOOKUP = {name.casefold(): name for name in UCF_CRIME_CATEGORIES}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
_FEATURE_SUFFIXES = {".npy", ".npz", ".txt"}
_CHUNK_SUFFIX = re.compile(r"(?:__\d+|_C)$", re.IGNORECASE)


class UCFCrimeError(ManifestError):
    """UCF-Crime 源文件格式或切分关系错误。"""


UCF_DECODED_SOURCE_END_POLICY = "source_end_reconciled_decoded_v1"


@dataclass(frozen=True)
class UCFSourceEndReconciliation:
    """One immutable source frame span projected onto a decoded-frame timeline."""

    raw_start_1based_inclusive: int
    raw_end_1based_inclusive: int
    author_frame_count: int
    decoded_frame_count: int
    start_author: int
    end_author: int
    start_effective: int
    end_effective: int
    author_clamped: bool
    decoded_clamped: bool
    empty_after_reconciliation: bool

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "raw_start_1based_inclusive": self.raw_start_1based_inclusive,
            "raw_end_1based_inclusive": self.raw_end_1based_inclusive,
            "author_frame_count_round_duration_x30": self.author_frame_count,
            "decoded_frame_count": self.decoded_frame_count,
            "start_author_1based_inclusive": self.start_author,
            "end_author_1based_inclusive": self.end_author,
            "start_effective_1based_inclusive": self.start_effective,
            "end_effective_1based_inclusive": self.end_effective,
            "author_clamped": self.author_clamped,
            "decoded_clamped": self.decoded_clamped,
            "empty_after_source_end_reconciliation": self.empty_after_reconciliation,
        }


def _display_video_id(value: str | Path) -> str:
    name = PurePosixPath(str(value).replace("\\", "/")).name
    lowered = name.casefold()
    for suffix in (*_VIDEO_SUFFIXES, *_FEATURE_SUFFIXES, ".fc6-1"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = _CHUNK_SUFFIX.sub("", name).strip()
    if not name:
        raise UCFCrimeError(f"无法从 {value!r} 提取视频名")
    return name


def normalize_ucf_category(value: str) -> str:
    """将官方类别名和常见 Normal 目录名归一化。"""

    raw = str(value).strip().replace(" ", "").replace("_", "").casefold()
    if "normal" in raw:
        return "Normal"
    for key, category in _CATEGORY_LOOKUP.items():
        if raw == key.replace("_", ""):
            return category
    raise UCFCrimeError(f"未知 UCF-Crime 类别：{value!r}")


def infer_ucf_category(path_or_id: str | Path) -> str:
    """从类别目录或官方视频名前缀推断类别。"""

    text = str(path_or_id).replace("\\", "/")
    parts = PurePosixPath(text).parts
    for part in parts:
        try:
            return normalize_ucf_category(part)
        except UCFCrimeError:
            pass

    video_id = _display_video_id(text)
    if "normal" in video_id.casefold():
        return "Normal"
    for category in UCF_CRIME_CATEGORIES:
        if category != "Normal" and video_id.casefold().startswith(category.casefold()):
            return category
    raise UCFCrimeError(f"无法从路径推断 UCF-Crime 类别：{path_or_id!r}")


@dataclass(frozen=True)
class UCFTemporalAnnotation:
    """官方测试视频的一行帧级真值。"""

    video_id: str
    filename: str
    category: str
    spans: tuple[TemporalSpan, ...]
    raw_spans_1based_inclusive: tuple[tuple[int, int], ...]
    num_frames: int | None = None


@dataclass(frozen=True)
class UCFSplitEntry:
    """官方 split 文件中的一条去重后视频记录。"""

    video_id: str
    path: str
    split: DatasetSplit
    category: str
    source: str


@dataclass(frozen=True)
class UCACaption:
    """UCA 的秒级描述；不携带异常真值。"""

    video_id: str
    start_seconds: float
    end_seconds: float
    text: str
    duration_seconds: float | None = None
    source: str = "uca"

    def __post_init__(self) -> None:
        if not self.video_id.strip():
            raise UCFCrimeError("UCA caption 的 video_id 不能为空")
        if (
            not math.isfinite(float(self.start_seconds))
            or not math.isfinite(float(self.end_seconds))
            or self.start_seconds < 0
            or self.end_seconds <= self.start_seconds
        ):
            raise UCFCrimeError(f"UCA caption 时间非法：[{self.start_seconds}, {self.end_seconds})")
        if not self.text.strip():
            raise UCFCrimeError("UCA caption 文本不能为空")
        if self.duration_seconds is not None and (
            not math.isfinite(float(self.duration_seconds)) or self.duration_seconds <= 0
        ):
            raise UCFCrimeError("UCA duration 必须是有限正数")

    def to_annotation(self) -> SupervisionAnnotation:
        """转换为 caption scope，并明确不生成异常标签。"""

        return SupervisionAnnotation(
            scope=SupervisionScope.CAPTION,
            span=TemporalSpan(
                start=self.start_seconds,
                end=self.end_seconds,
                unit=SpanUnit.SECOND,
            ),
            text=self.text.strip(),
            is_anomaly=None,
            source=self.source,
        )


@dataclass(frozen=True)
class UCFCrimeImportResult:
    """可直接写成 train/test JSONL 的导入结果。"""

    train: tuple[VideoManifestRecord, ...]
    test: tuple[VideoManifestRecord, ...]

    def __post_init__(self) -> None:
        train, test = validate_manifest_pair(self.train, self.test)
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)

    @property
    def all_records(self) -> tuple[VideoManifestRecord, ...]:
        return (*self.train, *self.test)

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        return write_ucf_crime_manifests(self, output_dir)


def _parse_event_pair(
    start_token: str,
    end_token: str,
    *,
    context: str,
) -> tuple[TemporalSpan, tuple[int, int]] | None:
    try:
        start = int(start_token)
        end = int(end_token)
    except ValueError as exc:
        raise UCFCrimeError(f"{context}: 帧区间必须是整数") from exc
    if start == -1 and end == -1:
        return None
    if start == -1 or end == -1:
        raise UCFCrimeError(f"{context}: -1 必须成对出现")
    if start < 1:
        raise UCFCrimeError(f"{context}: 官方异常起始帧必须是 1-based 正整数")
    try:
        # 官方 MATLAB: GT(st_fr:end_fr)=1（1-based inclusive）。框架内部统一使用
        # zero-based half-open，所以 [165,240] 必须转成 [164,240)。
        span = TemporalSpan(start=start - 1, end=end, unit=SpanUnit.FRAME)
    except ManifestError as exc:
        raise UCFCrimeError(f"{context}: {exc}") from exc
    return span, (start, end)


def parse_ucf_temporal_annotations(
    path: str | Path,
) -> dict[str, UCFTemporalAnnotation]:
    """读取官方 6 列或带总帧数的 7 列测试标注。"""

    annotation_path = Path(path)
    result: dict[str, UCFTemporalAnnotation] = {}
    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            context = f"{annotation_path}:{line_number}"
            if len(tokens) == 6:
                filename, category_token, *event_tokens = tokens
                num_frames = None
            elif len(tokens) == 7:
                filename, frame_count_token, category_token, *event_tokens = tokens
                try:
                    num_frames = int(frame_count_token)
                except ValueError as exc:
                    raise UCFCrimeError(f"{context}: 总帧数必须是整数") from exc
                if num_frames <= 0:
                    raise UCFCrimeError(f"{context}: 总帧数必须大于 0")
            else:
                raise UCFCrimeError(
                    f"{context}: 期望 6 列（官方）或 7 列（含总帧数），实际 {len(tokens)} 列"
                )

            category = normalize_ucf_category(category_token)
            parsed_pairs = tuple(
                pair
                for pair in (
                    _parse_event_pair(*event_tokens[:2], context=context),
                    _parse_event_pair(*event_tokens[2:], context=context),
                )
                if pair is not None
            )
            spans = tuple(pair[0] for pair in parsed_pairs)
            raw_spans = tuple(pair[1] for pair in parsed_pairs)
            if category == "Normal" and spans:
                raise UCFCrimeError(f"{context}: Normal 视频不能携带异常帧区间")
            if len(spans) == 2 and spans[1].start < spans[0].end:
                raise UCFCrimeError(f"{context}: 两个异常区间重叠或顺序颠倒")

            video_id = _display_video_id(filename)
            key = canonical_video_id(video_id)
            if key in result:
                raise UCFCrimeError(f"{context}: 重复标注视频 {video_id!r}")
            result[key] = UCFTemporalAnnotation(
                video_id=video_id,
                filename=filename.replace("\\", "/"),
                category=category,
                spans=spans,
                raw_spans_1based_inclusive=raw_spans,
                num_frames=num_frames,
            )
    return result


def _coerce_split_video_path(raw_path: str) -> str:
    normalized = normalize_relative_path(raw_path)
    pure = PurePosixPath(normalized)
    suffix = pure.suffix.casefold()
    name = pure.name

    if name.casefold().endswith(".fc6-1"):
        stem = name[: -len(".fc6-1")]
        name = f"{_CHUNK_SUFFIX.sub('', stem)}.mp4"
    elif suffix in _FEATURE_SUFFIXES:
        stem = name[: -len(pure.suffix)]
        name = f"{_CHUNK_SUFFIX.sub('', stem)}.mp4"
    elif suffix not in _VIDEO_SUFFIXES:
        raise UCFCrimeError(f"split 路径不是支持的视频或特征文件：{raw_path!r}")
    return PurePosixPath(*pure.parts[:-1], name).as_posix()


def parse_ucf_split_file(
    path: str | Path,
    split: DatasetSplit | str,
) -> tuple[UCFSplitEntry, ...]:
    """读取一行一个路径的官方 split；特征分块列表会按原视频去重。"""

    try:
        split_value = DatasetSplit(split)
    except ValueError as exc:
        raise UCFCrimeError(f"未知 split：{split!r}") from exc
    split_path = Path(path)
    entries: list[UCFSplitEntry] = []
    by_id: dict[str, UCFSplitEntry] = {}

    with split_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            raw_video_path = line.split()[0]
            try:
                video_path = _coerce_split_video_path(raw_video_path)
                category = infer_ucf_category(video_path)
            except ManifestError as exc:
                raise UCFCrimeError(f"{split_path}:{line_number}: {exc}") from exc
            video_id = _display_video_id(video_path)
            key = canonical_video_id(video_id)
            entry = UCFSplitEntry(
                video_id=video_id,
                path=video_path,
                split=split_value,
                category=category,
                source=str(split_path),
            )
            previous = by_id.get(key)
            if previous is not None:
                if previous.path != entry.path:
                    raise UCFCrimeError(
                        f"{split_path}:{line_number}: 同一视频映射到不同路径："
                        f"{previous.path!r} 与 {entry.path!r}"
                    )
                continue
            by_id[key] = entry
            entries.append(entry)
    return tuple(entries)


def derive_ucf_test_split(
    temporal_annotations: Mapping[str, UCFTemporalAnnotation],
    *,
    source: str = "ucf-crime-temporal-annotation",
) -> tuple[UCFSplitEntry, ...]:
    """从官方 290 行时序文件确定性地构造完整测试切分。

    官方时序文件同时列出 140 个异常测试视频和 150 个 ``-1`` Normal 视频，
    因此它本身就是测试清单的权威来源，不需要扫描或猜测数据目录。
    """

    entries: list[UCFSplitEntry] = []
    for annotation in temporal_annotations.values():
        filename = annotation.filename.replace("\\", "/")
        if "/" in filename:
            video_path = _coerce_split_video_path(filename)
        elif annotation.category == "Normal":
            video_path = normalize_relative_path(f"Testing_Normal_Videos_Anomaly/{filename}")
        else:
            video_path = normalize_relative_path(f"{annotation.category}/{filename}")
        entries.append(
            UCFSplitEntry(
                video_id=annotation.video_id,
                path=video_path,
                split=DatasetSplit.TEST,
                category=annotation.category,
                source=source,
            )
        )
    return tuple(entries)


def _parse_uca_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    try:
        numbers = [float(item) for item in parts]
    except ValueError as exc:
        raise UCFCrimeError(f"UCA 时间戳非法：{value!r}") from exc
    if len(numbers) == 2:
        minutes, seconds = numbers
        total = minutes * 60 + seconds
    elif len(numbers) == 3:
        hours, minutes, seconds = numbers
        total = hours * 3600 + minutes * 60 + seconds
    else:
        raise UCFCrimeError(f"UCA 时间戳非法：{value!r}")
    if any(number < 0 for number in numbers) or not math.isfinite(total):
        raise UCFCrimeError(f"UCA 时间戳非法：{value!r}")
    return total


def _parse_uca_json(path: Path) -> tuple[UCACaption, ...]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise UCFCrimeError(f"{path}: UCA JSON 顶层必须是对象")

    captions: list[UCACaption] = []
    for raw_video_id, raw_payload in data.items():
        if not isinstance(raw_payload, Mapping):
            raise UCFCrimeError(f"{path}: {raw_video_id} 的值必须是对象")
        timestamps = raw_payload.get("timestamps")
        sentences = raw_payload.get("sentences")
        if not isinstance(timestamps, list) or not isinstance(sentences, list):
            raise UCFCrimeError(f"{path}: {raw_video_id} 缺少 timestamps/sentences 数组")
        if len(timestamps) != len(sentences):
            raise UCFCrimeError(f"{path}: {raw_video_id} 的时间戳与句子数量不一致")
        duration_raw = raw_payload.get("duration")
        try:
            duration = float(duration_raw) if duration_raw is not None else None
        except (TypeError, ValueError) as exc:
            raise UCFCrimeError(f"{path}: {raw_video_id} 的 duration 非法") from exc
        video_id = _display_video_id(str(raw_video_id))
        for index, (timestamp, sentence) in enumerate(zip(timestamps, sentences, strict=True)):
            if not isinstance(timestamp, list) or len(timestamp) != 2:
                raise UCFCrimeError(f"{path}: {video_id} 第 {index} 个 timestamp 必须含两项")
            try:
                caption = UCACaption(
                    video_id=video_id,
                    start_seconds=float(timestamp[0]),
                    end_seconds=float(timestamp[1]),
                    text=str(sentence),
                    duration_seconds=duration,
                    source=f"uca:{path.name}",
                )
            except (TypeError, ValueError) as exc:
                raise UCFCrimeError(f"{path}: {video_id} 第 {index} 条 caption 非法") from exc
            captions.append(caption)
    return tuple(captions)


def _parse_uca_txt(path: Path) -> tuple[UCACaption, ...]:
    captions: list[UCACaption] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            prefix, separator, text = line.partition("##")
            tokens = prefix.split()
            if not separator or len(tokens) != 3:
                raise UCFCrimeError(f"{path}:{line_number}: 期望 'video start end ##caption'")
            captions.append(
                UCACaption(
                    video_id=_display_video_id(tokens[0]),
                    start_seconds=_parse_uca_timestamp(tokens[1]),
                    end_seconds=_parse_uca_timestamp(tokens[2]),
                    text=text,
                    source=f"uca:{path.name}",
                )
            )
    return tuple(captions)


def parse_uca_captions(path: str | Path) -> tuple[UCACaption, ...]:
    """读取 UCA 官方 JSON 或 ``video start end ##caption`` 文本格式。"""

    caption_path = Path(path)
    if caption_path.suffix.casefold() == ".json":
        return _parse_uca_json(caption_path)
    return _parse_uca_txt(caption_path)


def _path_sequence(value: str | Path | Sequence[str | Path] | None) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (Path(value),)
    return tuple(Path(item) for item in value)


def _read_split_files(
    paths: str | Path | Sequence[str | Path], split: DatasetSplit
) -> tuple[UCFSplitEntry, ...]:
    entries: list[UCFSplitEntry] = []
    by_id: dict[str, UCFSplitEntry] = {}
    for path in _path_sequence(paths):
        for entry in parse_ucf_split_file(path, split):
            key = canonical_video_id(entry.video_id)
            previous = by_id.get(key)
            if previous is not None:
                if previous.path != entry.path:
                    raise UCFCrimeError(
                        f"多个 {split.value} split 文件为 {entry.video_id!r} 指定了不同路径"
                    )
                continue
            by_id[key] = entry
            entries.append(entry)
    return tuple(entries)


def _caption_index(
    paths: str | Path | Sequence[str | Path] | None,
) -> dict[str, tuple[UCACaption, ...]]:
    grouped: dict[str, list[UCACaption]] = {}
    for path in _path_sequence(paths):
        for caption in parse_uca_captions(path):
            grouped.setdefault(canonical_video_id(caption.video_id), []).append(caption)
    return {key: tuple(items) for key, items in grouped.items()}


def _record_from_entry(
    entry: UCFSplitEntry,
    *,
    temporal: UCFTemporalAnnotation | None,
    captions: Sequence[UCACaption],
) -> VideoManifestRecord:
    is_anomaly = entry.category != "Normal"
    annotations: list[SupervisionAnnotation] = [
        SupervisionAnnotation(
            scope=SupervisionScope.VIDEO,
            label=entry.category,
            is_anomaly=is_anomaly,
            source="ucf-crime-official-split",
        )
    ]
    if temporal is not None:
        if temporal.category != entry.category:
            raise UCFCrimeError(
                f"{entry.video_id}: split 类别 {entry.category} 与时序标注类别 "
                f"{temporal.category} 不一致"
            )
        annotations.extend(
            SupervisionAnnotation(
                scope=SupervisionScope.FRAME,
                label=entry.category,
                is_anomaly=True,
                span=span,
                source="ucf-crime-temporal-annotation",
                metadata={
                    "raw_start_frame": raw_span[0],
                    "raw_end_frame": raw_span[1],
                    "raw_coordinate_system": "matlab_1based_inclusive",
                    "internal_coordinate_system": "zero_based_half_open",
                },
            )
            for span, raw_span in zip(
                temporal.spans,
                temporal.raw_spans_1based_inclusive,
                strict=True,
            )
        )
    annotations.extend(caption.to_annotation() for caption in captions)

    durations = {
        caption.duration_seconds for caption in captions if caption.duration_seconds is not None
    }
    if len(durations) > 1:
        raise UCFCrimeError(f"{entry.video_id}: UCA 中出现不一致的 duration")
    duration = next(iter(durations), None)
    return VideoManifestRecord(
        video_id=entry.video_id,
        path=entry.path,
        split=entry.split,
        category=entry.category,
        is_anomaly=is_anomaly,
        annotations=tuple(annotations),
        num_frames=temporal.num_frames if temporal is not None else None,
        duration_seconds=duration,
        metadata={
            "dataset": "ucf_crime",
            "split_source": entry.source,
            "uca_caption_count": len(captions),
        },
    )


def import_ucf_crime(
    *,
    dataset_root: str | Path,
    train_split: str | Path | Sequence[str | Path],
    temporal_annotations: str | Path,
    test_split: str | Path | Sequence[str | Path] | None = None,
    uca_captions: str | Path | Sequence[str | Path] | None = None,
    require_files: bool = False,
    strict_temporal: bool = True,
) -> UCFCrimeImportResult:
    """导入 UCF-Crime，并严格阻止 train/test 视频泄漏。

    官方 ``Anomaly_Train.txt`` 已包含 1610 个训练视频。未传 ``test_split`` 时，
    直接从 290 行官方时序文件构造 140 个异常 + 150 个 Normal 测试视频；若显式
    传入一个或多个测试列表，则逐项与时序文件交叉核验。``strict_temporal`` 默认
    要求每个异常测试视频都有官方帧区间，且时序标注绝不能指向训练视频。
    """

    train_entries = _read_split_files(train_split, DatasetSplit.TRAIN)
    temporal_by_id = parse_ucf_temporal_annotations(temporal_annotations)
    test_entries = (
        derive_ucf_test_split(temporal_by_id, source=str(Path(temporal_annotations)))
        if test_split is None
        else _read_split_files(test_split, DatasetSplit.TEST)
    )
    captions_by_id = _caption_index(uca_captions)

    train_ids = {canonical_video_id(entry.video_id) for entry in train_entries}
    test_ids = {canonical_video_id(entry.video_id) for entry in test_entries}
    overlap = train_ids & test_ids
    if overlap:
        raise UCFCrimeError(f"官方切分发生 train/test 泄漏：{sorted(overlap)}")

    temporal_train_overlap = train_ids & temporal_by_id.keys()
    if temporal_train_overlap:
        raise UCFCrimeError(f"测试时序标注指向了训练视频：{sorted(temporal_train_overlap)}")
    unknown_temporal = temporal_by_id.keys() - test_ids
    if strict_temporal and unknown_temporal:
        raise UCFCrimeError(f"时序标注包含 test split 中不存在的视频：{sorted(unknown_temporal)}")

    if strict_temporal:
        missing = {
            canonical_video_id(entry.video_id)
            for entry in test_entries
            if entry.category != "Normal"
            and canonical_video_id(entry.video_id) not in temporal_by_id
        }
        if missing:
            raise UCFCrimeError(f"异常测试视频缺少时序标注：{sorted(missing)}")

    train = tuple(
        _record_from_entry(
            entry,
            temporal=None,
            captions=captions_by_id.get(canonical_video_id(entry.video_id), ()),
        )
        for entry in train_entries
    )
    test = tuple(
        _record_from_entry(
            entry,
            temporal=temporal_by_id.get(canonical_video_id(entry.video_id)),
            captions=captions_by_id.get(canonical_video_id(entry.video_id), ()),
        )
        for entry in test_entries
    )
    train, test = validate_manifest_pair(
        train,
        test,
        dataset_root=dataset_root,
        require_files=require_files,
    )
    return UCFCrimeImportResult(train=train, test=test)


def reconcile_ucf_test_records_to_decoded_frames(
    records: Iterable[VideoManifestRecord],
    dataset_root: str | Path,
) -> tuple[tuple[VideoManifestRecord, ...], dict[str, object]]:
    """Explicitly project immutable source ends onto verified decoded frames.

    This is an opt-in decoded-frame evaluation policy.  It leaves the official
    annotation source untouched and records both the author-style duration
    clamp and the decoded-frame clamp in annotation metadata.  It must never
    be used for train/development records.
    """

    source_records = tuple(records)
    if any(record.split != DatasetSplit.TEST for record in source_records):
        raise UCFCrimeError("decoded source-end reconciliation 只接受 official test records")
    reconciled: list[VideoManifestRecord] = []
    aggregate = {
        "policy": UCF_DECODED_SOURCE_END_POLICY,
        "test_records": len(source_records),
        "frame_annotations": 0,
        "author_clamped_spans": 0,
        "decoded_clamped_spans": 0,
        "empty_after_reconciliation_spans": 0,
        "author_decode_frame_count_different_records": 0,
    }
    for record in source_records:
        info = probe_video(record.resolve_path(dataset_root))
        author_count = round(info.duration_seconds * 30)
        if author_count <= 0:
            raise UCFCrimeError(f"{record.video_id}: N_author 非法")
        if author_count != info.num_frames:
            aggregate["author_decode_frame_count_different_records"] += 1
        annotations: list[SupervisionAnnotation] = []
        per_record: list[dict[str, int | bool]] = []
        for annotation in record.annotations:
            if annotation.scope != SupervisionScope.FRAME:
                annotations.append(annotation)
                continue
            raw_start = annotation.metadata.get("raw_start_frame")
            raw_end = annotation.metadata.get("raw_end_frame")
            if type(raw_start) is not int or type(raw_end) is not int:
                raise UCFCrimeError(f"{record.video_id}: 缺少 immutable raw source endpoint")
            start_author = max(raw_start, 1)
            end_author = min(raw_end, author_count)
            start_effective = start_author
            # The decoded metric clamps directly against immutable raw_end.
            # N_author remains a separately recorded historical-evaluator
            # diagnostic and must not move a modern decoded-frame endpoint.
            end_effective = min(raw_end, info.num_frames)
            detail = UCFSourceEndReconciliation(
                raw_start_1based_inclusive=raw_start,
                raw_end_1based_inclusive=raw_end,
                author_frame_count=author_count,
                decoded_frame_count=info.num_frames,
                start_author=start_author,
                end_author=end_author,
                start_effective=start_effective,
                end_effective=end_effective,
                author_clamped=(start_author != raw_start or end_author != raw_end),
                decoded_clamped=(end_effective != raw_end),
                empty_after_reconciliation=(end_effective < start_effective),
            )
            detail_dict = detail.as_dict()
            per_record.append(detail_dict)
            aggregate["frame_annotations"] += 1
            aggregate["author_clamped_spans"] += int(detail.author_clamped)
            aggregate["decoded_clamped_spans"] += int(detail.decoded_clamped)
            aggregate["empty_after_reconciliation_spans"] += int(
                detail.empty_after_reconciliation
            )
            if detail.empty_after_reconciliation:
                # A zero-support source span cannot be silently expanded.  It
                # remains in the sealed receipt but is deliberately absent from
                # the decoded-frame annotation list.
                continue
            metadata = dict(annotation.metadata)
            metadata["source_end_reconciliation"] = detail_dict
            metadata["evaluation_policy"] = UCF_DECODED_SOURCE_END_POLICY
            annotations.append(
                replace(
                    annotation,
                    span=TemporalSpan(
                        start=start_effective - 1,
                        end=end_effective,
                        unit=SpanUnit.FRAME,
                    ),
                    metadata=metadata,
                )
            )
        record_metadata = dict(record.metadata)
        record_metadata["source_end_reconciliation_policy"] = UCF_DECODED_SOURCE_END_POLICY
        record_metadata["source_end_reconciliation"] = per_record
        reconciled.append(
            replace(
                record,
                annotations=tuple(annotations),
                num_frames=info.num_frames,
                fps=info.fps,
                duration_seconds=info.duration_seconds,
                metadata=record_metadata,
            )
        )
    return tuple(reconciled), aggregate


def attach_uca_captions(
    records: Iterable[VideoManifestRecord],
    captions: Iterable[UCACaption],
) -> tuple[VideoManifestRecord, ...]:
    """给现有记录附加 UCA 描述，不改变任何视频异常标签。"""

    grouped: dict[str, list[UCACaption]] = {}
    for caption in captions:
        grouped.setdefault(canonical_video_id(caption.video_id), []).append(caption)
    result: list[VideoManifestRecord] = []
    for record in records:
        additions = grouped.get(canonical_video_id(record.video_id), [])
        duration_values = {
            item.duration_seconds for item in additions if item.duration_seconds is not None
        }
        if len(duration_values) > 1:
            raise UCFCrimeError(f"{record.video_id}: UCA 中出现不一致的 duration")
        duration = record.duration_seconds or next(iter(duration_values), None)
        metadata = dict(record.metadata)
        metadata["uca_caption_count"] = int(metadata.get("uca_caption_count", 0)) + len(additions)
        result.append(
            replace(
                record,
                annotations=(*record.annotations, *(item.to_annotation() for item in additions)),
                duration_seconds=duration,
                metadata=metadata,
            )
        )
    return tuple(result)


def write_ucf_crime_manifests(
    result: UCFCrimeImportResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """写入 ``train.jsonl`` 和 ``test.jsonl``。"""

    train, test = validate_manifest_pair(result.train, result.test)
    output = Path(output_dir)
    train_path = write_manifest_jsonl(train, output / "train.jsonl")
    test_path = write_manifest_jsonl(test, output / "test.jsonl")
    return train_path, test_path


# 语义明确的兼容别名。
parse_temporal_annotations = parse_ucf_temporal_annotations
parse_official_split = parse_ucf_split_file
build_ucf_crime_manifests = import_ucf_crime
