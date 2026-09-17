"""Freeze availability-aware, W-safe UCF-Crime research cohorts.

Partition and candidate selection use the complete official training list. A
later download only changes a candidate from pending to available; it never
changes its partition or replaces it with another video.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vadbench.data.audit import audit_ucf_crime_dataset
from vadbench.data.manifest import VideoManifestRecord, write_manifest_jsonl
from vadbench.data.ucf_crime import UCFCrimeImportResult, import_ucf_crime
from vadbench.features import atomic_write_json, atomic_write_jsonl

from .cohorts import CohortIndex, CohortRecord, LabelPolicy


class ResearchDataPreparationError(ValueError):
    """A requested cohort would violate the data or label-access protocol."""


@dataclass(frozen=True, slots=True)
class CohortLimits:
    """Maximum unique videos per weak label in each frozen role cohort."""

    debug_per_label: int = 4
    explore_per_label: int = 64
    confirm_per_label: int = 32
    select_per_label: int = 32

    def __post_init__(self) -> None:
        for name in (
            "debug_per_label",
            "explore_per_label",
            "confirm_per_label",
            "select_per_label",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ResearchDataPreparationError(f"{name} 必须是非负整数")
        if self.debug_per_label > self.explore_per_label:
            raise ResearchDataPreparationError("debug_per_label 不能大于 explore_per_label")


@dataclass(frozen=True, slots=True)
class PreparedResearchData:
    """Frozen full-list roles plus the currently available materialisation."""

    imported: UCFCrimeImportResult
    available_train: tuple[VideoManifestRecord, ...]
    unavailable_train: tuple[VideoManifestRecord, ...]
    partitions: Mapping[str, tuple[VideoManifestRecord, ...]]
    available_partitions: Mapping[str, tuple[VideoManifestRecord, ...]]
    partition_lock: Mapping[str, str]
    candidate_videos: Mapping[str, tuple[str, ...]]
    cohorts: Mapping[str, CohortIndex]
    cohort_status: Mapping[str, str]
    windows_per_video: int
    seed: int
    audit: Mapping[str, Any]
    source_grouping_status: str
    source_groups: Mapping[str, str]

    def receipt(self) -> dict[str, Any]:
        audit_passed = self.audit.get("passed")
        candidates = {name: set(video_ids) for name, video_ids in self.candidate_videos.items()}
        materialized = {
            name: {record.video_id for record in index.records}
            for name, index in self.cohorts.items()
        }
        return {
            "schema_version": 2,
            "policy": "W",
            "selection_scope": "official_train_video_weak_labels_only",
            "test_temporal_truth_used_for_selection": False,
            "partition_basis": "complete_official_train_source_groups",
            "partition_ratio": {"fit": 0.8, "confirm": 0.1, "select": 0.1},
            "windows_per_video": self.windows_per_video,
            "available_train_videos": len(self.available_train),
            "unavailable_train_videos": len(self.unavailable_train),
            "available_train_by_weak_label": _label_counts(self.available_train),
            "partition_videos": {
                name: len(records) for name, records in sorted(self.partitions.items())
            },
            "available_partition_videos": {
                name: len(records) for name, records in sorted(self.available_partitions.items())
            },
            "cohorts": {
                name: {
                    "status": self.cohort_status[name],
                    "candidate_videos": len(candidates[name]),
                    "available_videos": len(materialized.get(name, set())),
                    "pending_videos": len(candidates[name] - materialized.get(name, set())),
                    "available_by_weak_label": (
                        _cohort_label_counts(self.cohorts[name]) if name in self.cohorts else {}
                    ),
                }
                for name in sorted(candidates)
            },
            "source_grouping": self.source_grouping_status,
            "audit_passed": audit_passed if isinstance(audit_passed, bool) else None,
            "dataset_status": "full" if audit_passed is True else "partial_or_unverified",
        }


def _label_counts(records: Sequence[VideoManifestRecord]) -> dict[str, int]:
    return {
        "normal": sum(not record.is_anomaly for record in records),
        "anomaly": sum(record.is_anomaly for record in records),
    }


def _cohort_label_counts(cohort: CohortIndex) -> dict[str, int]:
    by_video = {record.video_id: record.weak_label for record in cohort.records}
    return {
        "normal": sum(label == 0 for label in by_video.values()),
        "anomaly": sum(label == 1 for label in by_video.values()),
    }


def _stable_order(items: Sequence[str], *, seed: int) -> list[str]:
    def key(value: str) -> tuple[bytes, str]:
        return hashlib.sha256(f"{seed}\0{value}".encode()).digest(), value

    return sorted(items, key=key)


def _validate_source_groups(
    train: Sequence[VideoManifestRecord], source_groups: Mapping[str, str] | None
) -> dict[str, str]:
    if source_groups is None:
        return {}
    expected = {record.video_id for record in train}
    unknown = set(source_groups) - expected
    if unknown:
        raise ResearchDataPreparationError(f"source_groups 含未知训练视频：{sorted(unknown)}")
    normalized: dict[str, str] = {}
    for video_id, source_id in source_groups.items():
        if not isinstance(source_id, str) or not source_id.strip():
            raise ResearchDataPreparationError(f"{video_id} 的 source_id 必须为非空字符串")
        normalized[video_id] = source_id.strip()
    return normalized


def _groups(
    records: Sequence[VideoManifestRecord], source_groups: Mapping[str, str]
) -> dict[str, list[VideoManifestRecord]]:
    result: dict[str, list[VideoManifestRecord]] = defaultdict(list)
    for record in records:
        result[source_groups.get(record.video_id, f"video:{record.video_id}")].append(record)
    for group_id, members in result.items():
        if len({record.is_anomaly for record in members}) != 1:
            raise ResearchDataPreparationError(
                f"source group {group_id!r} 同时含正常与异常训练视频，无法做 W 分层"
            )
    return result


def _partition_complete_train(
    records: Sequence[VideoManifestRecord],
    *,
    source_groups: Mapping[str, str],
    seed: int,
) -> tuple[dict[str, tuple[VideoManifestRecord, ...]], dict[str, str]]:
    groups = _groups(records, source_groups)
    by_label: dict[int, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        by_label[int(members[0].is_anomaly)].append(group_id)

    assignments: dict[str, str] = {}
    for label, group_ids in by_label.items():
        ordered = _stable_order(group_ids, seed=seed + label)
        count = len(ordered)
        fit_count = (count * 8) // 10
        confirm_count = count // 10
        for group_id in ordered[:fit_count]:
            assignments[group_id] = "fit"
        for group_id in ordered[fit_count : fit_count + confirm_count]:
            assignments[group_id] = "confirm"
        for group_id in ordered[fit_count + confirm_count :]:
            assignments[group_id] = "select"

    partitions: dict[str, list[VideoManifestRecord]] = {"fit": [], "confirm": [], "select": []}
    lock: dict[str, str] = {}
    for group_id, members in groups.items():
        partition = assignments[group_id]
        partitions[partition].extend(members)
        lock.update({record.video_id: partition for record in members})
    return (
        {
            name: tuple(sorted(items, key=lambda record: record.video_id.casefold()))
            for name, items in partitions.items()
        },
        dict(sorted(lock.items(), key=lambda item: item[0].casefold())),
    )


def _select_candidates(
    records: Sequence[VideoManifestRecord],
    *,
    per_label: int,
    source_groups: Mapping[str, str],
    seed: int,
) -> tuple[str, ...]:
    groups = _groups(records, source_groups)
    selected: list[VideoManifestRecord] = []
    for label in (0, 1):
        remaining = per_label
        group_ids = [
            group_id for group_id, members in groups.items() if int(members[0].is_anomaly) == label
        ]
        for group_id in _stable_order(group_ids, seed=seed + label):
            members = groups[group_id]
            if len(members) > remaining:
                continue
            selected.extend(members)
            remaining -= len(members)
            if remaining == 0:
                break
    return tuple(record.video_id for record in sorted(selected, key=lambda item: item.video_id.casefold()))


def _take_labelled(
    video_ids: Sequence[str], records: Sequence[VideoManifestRecord], label: int, limit: int
) -> tuple[str, ...]:
    by_id = {record.video_id: record for record in records}
    return tuple(video_id for video_id in video_ids if int(by_id[video_id].is_anomaly) == label)[:limit]


def _available_candidates(
    candidate_ids: Sequence[str], available_by_id: Mapping[str, VideoManifestRecord]
) -> tuple[VideoManifestRecord, ...]:
    return tuple(available_by_id[video_id] for video_id in candidate_ids if video_id in available_by_id)


def _materialize_cohort(
    records: Sequence[VideoManifestRecord],
    *,
    partition: str,
    role: str,
    source_groups: Mapping[str, str],
    windows_per_video: int,
) -> CohortIndex | None:
    if not records:
        return None
    rows = [
        CohortRecord(
            video_id=record.video_id,
            clip_id=f"{record.video_id}:segment-{segment:02d}",
            official_split="train",
            partition=partition,
            role=role,
            weak_label=int(record.is_anomaly),
            label_source="video_weak",
            source_id=source_groups.get(record.video_id),
            category_for_analysis_only=record.category,
        )
        for record in sorted(records, key=lambda item: item.video_id.casefold())
        for segment in range(windows_per_video)
    ]
    return CohortIndex.from_records(rows, policy=LabelPolicy.WEAK)


def prepare_ucf_research_data(
    *,
    dataset_root: str | Path,
    train_split: str | Path,
    temporal_annotations: str | Path,
    test_split: str | Path | None = None,
    source_groups: Mapping[str, str] | None = None,
    limits: CohortLimits | None = None,
    windows_per_video: int = 8,
    seed: int = 202709,
    audit_fn: Callable[..., dict[str, Any]] = audit_ucf_crime_dataset,
) -> PreparedResearchData:
    """Freeze roles from the full official list, then materialize available videos only."""

    if type(seed) is not int:
        raise ResearchDataPreparationError("seed 必须是整数")
    if type(windows_per_video) is not int or windows_per_video <= 0:
        raise ResearchDataPreparationError("windows_per_video 必须是正整数")
    limits = limits or CohortLimits()
    imported = import_ucf_crime(
        dataset_root=dataset_root,
        train_split=train_split,
        temporal_annotations=temporal_annotations,
        test_split=test_split,
        require_files=False,
    )
    normalized_groups = _validate_source_groups(imported.train, source_groups)
    partitions, partition_lock = _partition_complete_train(
        imported.train, source_groups=normalized_groups, seed=seed
    )
    root = Path(dataset_root)
    available = tuple(record for record in imported.train if record.resolve_path(root).is_file())
    unavailable = tuple(record for record in imported.train if not record.resolve_path(root).is_file())
    if not available:
        raise ResearchDataPreparationError("没有实际可用的官方训练视频，不能创建 cohort")
    available_ids = {record.video_id for record in available}
    available_by_id = {record.video_id: record for record in available}
    available_partitions = {
        name: tuple(record for record in records if record.video_id in available_ids)
        for name, records in partitions.items()
    }

    explore = _select_candidates(
        partitions["fit"],
        per_label=limits.explore_per_label,
        source_groups=normalized_groups,
        seed=seed + 20,
    )
    candidate_videos = {
        "debug": tuple(
            video_id
            for label in (0, 1)
            for video_id in _take_labelled(explore, imported.train, label, limits.debug_per_label)
        ),
        "explore": explore,
        "confirm": _select_candidates(
            partitions["confirm"],
            per_label=limits.confirm_per_label,
            source_groups=normalized_groups,
            seed=seed + 30,
        ),
        "select": _select_candidates(
            partitions["select"],
            per_label=limits.select_per_label,
            source_groups=normalized_groups,
            seed=seed + 40,
        ),
    }
    role_spec = {
        "debug": ("fit", "debug"),
        "explore": ("fit", "explore"),
        "confirm": ("confirm", "confirm"),
        "select": ("select", "select"),
    }
    cohorts: dict[str, CohortIndex] = {}
    cohort_status: dict[str, str] = {}
    for name, candidate_ids in candidate_videos.items():
        partition, role = role_spec[name]
        cohort = _materialize_cohort(
            _available_candidates(candidate_ids, available_by_id),
            partition=partition,
            role=role,
            source_groups=normalized_groups,
            windows_per_video=windows_per_video,
        )
        if cohort is None:
            cohort_status[name] = "pending_no_available_candidate"
        else:
            cohorts[name] = cohort
            cohort_status[name] = (
                "ready" if len({row.video_id for row in cohort.records}) == len(candidate_ids) else "partial"
            )
    audit = audit_fn(dataset_root, imported.train, imported.test)
    return PreparedResearchData(
        imported=imported,
        available_train=available,
        unavailable_train=unavailable,
        partitions=partitions,
        available_partitions=available_partitions,
        partition_lock=partition_lock,
        candidate_videos=candidate_videos,
        cohorts=cohorts,
        cohort_status=cohort_status,
        windows_per_video=windows_per_video,
        seed=seed,
        audit=audit,
        source_grouping_status=(
            "provided_source_groups" if normalized_groups else "video_id_only_near_duplicate_unavailable"
        ),
        source_groups=normalized_groups,
    )


def write_prepared_research_data(result: PreparedResearchData, output_dir: str | Path) -> dict[str, Path]:
    """Write a new isolated run; full-list locks and pending candidates remain explicit."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    manifests = destination / "manifests"
    partitions = destination / "available-partitions"
    cohorts = destination / "cohorts"
    audit_dir = destination / "audit"
    for path in (manifests, partitions, cohorts, audit_dir):
        path.mkdir()

    written: dict[str, Path] = {
        "available_train_manifest": write_manifest_jsonl(
            result.available_train, manifests / "available-train.jsonl"
        )
    }
    for name, records in result.available_partitions.items():
        written[f"available_partition_{name}"] = write_manifest_jsonl(
            records, partitions / f"{name}.jsonl"
        )
    for name, cohort in result.cohorts.items():
        path = cohorts / f"{name}.jsonl"
        atomic_write_jsonl(path, (record.as_dict() for record in cohort.records))
        written[f"cohort_{name}"] = path

    lock_path = destination / "partition-lock.json"
    candidates_path = destination / "cohort-candidates.json"
    audit_path = audit_dir / "ucf-audit.json"
    receipt_path = destination / "preparation-receipt.json"
    atomic_write_json(
        lock_path,
        {
            "schema_version": 1,
            "seed": result.seed,
            "basis": "complete_official_train_source_groups",
            "partitions": result.partition_lock,
            "source_groups": result.source_groups,
        },
    )
    available_by_role = {
        name: [record.video_id for record in index.records[:: result.windows_per_video]]
        for name, index in result.cohorts.items()
    }
    atomic_write_json(
        candidates_path,
        {
            "schema_version": 1,
            "windows_per_video": result.windows_per_video,
            "candidates": result.candidate_videos,
            "available": available_by_role,
            "pending": {
                name: [video_id for video_id in ids if video_id not in set(available_by_role.get(name, []))]
                for name, ids in result.candidate_videos.items()
            },
        },
    )
    atomic_write_json(audit_path, dict(result.audit))
    atomic_write_json(receipt_path, result.receipt())
    written.update(
        {
            "partition_lock": lock_path,
            "cohort_candidates": candidates_path,
            "audit": audit_path,
            "receipt": receipt_path,
        }
    )
    return written


__all__ = [
    "CohortLimits",
    "PreparedResearchData",
    "ResearchDataPreparationError",
    "prepare_ucf_research_data",
    "write_prepared_research_data",
]
