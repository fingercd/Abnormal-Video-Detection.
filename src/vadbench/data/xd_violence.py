"""XD-Violence raw-video identities and train-only weak supervision.

Only ZIP central-directory metadata is read. Official test identities are
kept outside the training Manifest format; no event annotations, ground-truth
arrays, frame-coordinate conversion, or evaluation protocol are implemented.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from vadbench.data.manifest import (
    ManifestError,
    SupervisionAnnotation,
    VideoManifestRecord,
    normalize_relative_path,
    validate_manifest,
    write_manifest_jsonl,
)
from vadbench.features import atomic_write_json, atomic_write_jsonl

OFFICIAL_VOLUMES = {
    "train_0001_1004": 1004,
    "train_1005_2004": 1000,
    "train_2005_2804": 800,
    "train_2805_3319": 515,
    "train_3320_3954": 635,
    "test_videos": 800,
}
ANOMALY_TOKENS = frozenset({"B1", "B2", "B4", "B5", "B6", "G"})
SOURCE_GROUP_RULE = "xd-name-prefix-before-double-underscore-hash-v1"


class XDImportError(ManifestError):
    """An archive identity or weak-label contract is not established."""


def _relative_path(value: str) -> str:
    try:
        return normalize_relative_path(value)
    except ManifestError as exc:
        raise XDImportError(str(exc)) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_xd_labels(member_path: str) -> tuple[str, ...]:
    """Parse the observed official label grammar, preserving trailing 0 slots.

    A is the sole normal marker. Anomaly names have one to three known class
    tokens, optionally followed by zero placeholders up to three total slots.
    Unknown tokens are errors, never an implicit abnormal label.
    """
    path = PurePosixPath(_relative_path(member_path))
    if path.suffix != ".mp4" or path.stem.count("_label_") != 1:
        raise XDImportError(f"XD video needs one _label_ field and .mp4 suffix: {member_path}")
    prefix, field = path.stem.rsplit("_label_", 1)
    if not prefix:
        raise XDImportError(f"XD video identity is empty: {member_path}")
    tokens = tuple(field.split("-"))
    if tokens == ("A",):
        return tokens
    if not 1 <= len(tokens) <= 3 or tokens[0] not in ANOMALY_TOKENS:
        raise XDImportError(f"invalid XD label tokens: {tokens}")
    classes: list[str] = []
    padding = False
    for token in tokens:
        if token == "0":
            padding = True
        elif token in ANOMALY_TOKENS and not padding and token not in classes:
            classes.append(token)
        else:
            raise XDImportError(f"invalid XD label tokens: {tokens}")
    return tokens


@dataclass(frozen=True)
class XDArchiveMember:
    volume: str
    path: str
    crc32: int
    compressed_size: int
    uncompressed_size: int


@dataclass(frozen=True)
class XDArchiveInventory:
    members: tuple[XDArchiveMember, ...]
    central_sha256: Mapping[str, str]


def _zip64_sizes(extra: bytes, size: int, compressed: int) -> tuple[int, int]:
    offset = 0
    while offset + 4 <= len(extra):
        tag, length = struct.unpack_from("<HH", extra, offset)
        offset += 4
        if offset + length > len(extra):
            raise XDImportError("truncated ZIP extra field")
        if tag == 1:
            values = extra[offset : offset + length]
            position = 0
            for field in ("size", "compressed"):
                if (size if field == "size" else compressed) != 0xFFFFFFFF:
                    continue
                if position + 8 > len(values):
                    raise XDImportError("truncated ZIP64 size field")
                value = struct.unpack_from("<Q", values, position)[0]
                position += 8
                if field == "size":
                    size = value
                else:
                    compressed = value
            return size, compressed
        offset += length
    raise XDImportError("ZIP64 size field is missing")


def parse_xd_central_directory(data: bytes, volume: str) -> tuple[XDArchiveMember, ...]:
    """Read metadata from one complete central directory, never member bodies."""
    if volume not in OFFICIAL_VOLUMES:
        raise XDImportError(f"unknown XD archive volume: {volume}")
    offset = 0
    members = []
    while offset < len(data):
        if offset + 46 > len(data) or data[offset : offset + 4] != b"PK\x01\x02":
            raise XDImportError(f"invalid central directory at {volume}:{offset}")
        fields = struct.unpack_from("<4s6H3L5H2L", data, offset)
        flags, crc, compressed, size = fields[3], fields[7], fields[8], fields[9]
        name_length, extra_length, comment_length = fields[10:13]
        start = offset + 46
        extra_start = start + name_length
        end = extra_start + extra_length + comment_length
        if end > len(data):
            raise XDImportError(f"truncated central directory in {volume}")
        try:
            raw_name = data[start:extra_start].decode("utf-8" if flags & 0x800 else "cp437")
        except UnicodeDecodeError as exc:
            raise XDImportError(f"invalid ZIP filename encoding in {volume}") from exc
        path = _relative_path(raw_name.rstrip("/"))
        if not raw_name.endswith("/"):
            if PurePosixPath(path).suffix != ".mp4" or flags & 1:
                raise XDImportError(f"unexpected or encrypted XD video member: {raw_name}")
            if size == 0xFFFFFFFF or compressed == 0xFFFFFFFF:
                size, compressed = _zip64_sizes(
                    data[extra_start : extra_start + extra_length], size, compressed
                )
            members.append(XDArchiveMember(volume, path, crc, compressed, size))
        offset = end
    return tuple(members)


def _validate_inventory(inventory: XDArchiveInventory) -> None:
    observed = Counter(member.volume for member in inventory.members)
    if dict(observed) != OFFICIAL_VOLUMES:
        raise XDImportError(f"complete official XD volumes are required: {dict(observed)}")
    if set(inventory.central_sha256) != set(OFFICIAL_VOLUMES):
        raise XDImportError("all six central-directory source digests are required")
    for digest in inventory.central_sha256.values():
        if not isinstance(digest, str) or len(digest) != 64:
            raise XDImportError("central-directory SHA256 must be 64 hexadecimal characters")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise XDImportError("invalid central-directory SHA256") from exc


def load_xd_archive_members(metadata_dir: str | Path) -> XDArchiveInventory:
    """Load the six previously captured official central-directory files.

    ``central_plan.json`` binds each volume name to its exact directory byte
    count; content digests are recorded in the new frozen identity receipt.
    """
    root = Path(metadata_dir)
    plan = json.loads((root / "central_plan.json").read_text(encoding="utf-8"))
    entries = plan["volumes"]
    if len(entries) != 6 or {entry["volume"] for entry in entries} != set(OFFICIAL_VOLUMES):
        raise XDImportError("central plan must contain the six official XD volumes exactly once")
    members: list[XDArchiveMember] = []
    digests: dict[str, str] = {}
    for entry in entries:
        volume = entry["volume"]
        if entry["status"] != "need_central":
            raise XDImportError(f"central-directory source is unavailable: {volume}")
        source = root / f"{volume}.cd"
        expected_size = entry["central_directory_bytes"]
        if (
            type(expected_size) is not int
            or not 0 < expected_size <= 16 * 1024 * 1024
            or source.stat().st_size != expected_size
        ):
            raise XDImportError(f"central-directory byte length mismatch: {volume}")
        data = source.read_bytes()
        if len(data) != expected_size:
            raise XDImportError(f"central-directory byte length mismatch: {volume}")
        digests[volume] = hashlib.sha256(data).hexdigest()
        members.extend(parse_xd_central_directory(data, volume))
    inventory = XDArchiveInventory(tuple(members), digests)
    _validate_inventory(inventory)
    return inventory


def _member_identity(member: XDArchiveMember) -> tuple[str, str]:
    path = PurePosixPath(_relative_path(member.path))
    if path.parts[0].casefold() == "videos":
        path = PurePosixPath(*path.parts[1:])
    if path.suffix != ".mp4":
        raise XDImportError(f"unexpected XD video extension: {member.path}")
    # Include all nested path components; duplicate basenames remain distinct.
    identity = path.with_suffix("").as_posix().casefold()
    return "xd-" + hashlib.sha256(identity.encode()).hexdigest(), identity


def _source_group(member: XDArchiveMember) -> tuple[str, str, str | None]:
    name = PurePosixPath(member.path).stem.rsplit("_label_", 1)[0]
    if name.count("__#") != 1 or not all(name.split("__#")):
        return "xd-source:unknown", "unknown_conservatively_fit", None
    prefix = name.split("__#", 1)[0]
    group = "xd-source:" + hashlib.sha256(prefix.casefold().encode()).hexdigest()
    return group, "name_derived_conservative_not_content_verified", prefix


def _member_metadata(member: XDArchiveMember, inventory: XDArchiveInventory) -> dict[str, Any]:
    group, status, prefix = _source_group(member)
    return {
        "dataset": "xd_violence",
        "archive_volume": member.volume,
        "archive_member": member.path,
        "central_directory_sha256": inventory.central_sha256[member.volume],
        "provider_crc32": f"{member.crc32:08x}",
        "compressed_size": member.compressed_size,
        "uncompressed_size": member.uncompressed_size,
        "source_group": group,
        "source_group_status": status,
        "source_verified": "unknown",
        "source_name_prefix": prefix,
        "source_group_rule": SOURCE_GROUP_RULE,
    }


@dataclass(frozen=True)
class XDPreparedData:
    train: tuple[VideoManifestRecord, ...]
    test_index: tuple[dict[str, Any], ...]
    partition_lock: Mapping[str, str]
    available_train: tuple[VideoManifestRecord, ...]
    pending_train: tuple[VideoManifestRecord, ...]
    partitions: Mapping[str, tuple[VideoManifestRecord, ...]]
    availability: tuple[dict[str, Any], ...]
    central_sha256: Mapping[str, str]
    shared_official_source_groups: tuple[str, ...]
    seed: int
    dataset_root: str

    def role_lock(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": "xd_violence",
            "partition_basis": "complete_3954_official_train_archive_members",
            "source_group_rule": SOURCE_GROUP_RULE,
            "source_verified": "unknown",
            "assignment_rule": "sha256_seeded_source_group_order_80_10_10_v1",
            "central_directory_sha256": dict(self.central_sha256),
            "seed": self.seed,
            "partitions": dict(self.partition_lock),
            "source_groups": {row.video_id: row.metadata["source_group"] for row in self.train},
        }

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset": "xd_violence",
            "dataset_root": self.dataset_root,
            "seed": self.seed,
            "policy": "W",
            "train_candidates": len(self.train),
            "test_identity_candidates": len(self.test_index),
            "train_labels": dict(
                Counter("anomaly" if row.is_anomaly else "normal" for row in self.train)
            ),
            "available_train_files": len(self.available_train),
            "pending_train_files": len(self.pending_train),
            "availability_status": "file_size_only_not_decode_or_crc_verified",
            "download_complete_verified": False,
            "availability_counts": dict(Counter(row["status"] for row in self.availability)),
            "partition_videos": {role: len(rows) for role, rows in self.partitions.items()},
            "role_lock_sha256": _digest(self.role_lock()),
            "source_group_status": "name_derived_conservative_not_content_verified",
            "source_verified": "unknown",
            "derived_group_missing_train_videos": sum(
                row.metadata["source_group_status"].startswith("unknown") for row in self.train
            ),
            "official_train_test_shared_name_source_group_count": len(
                self.shared_official_source_groups
            ),
            "official_train_test_shared_name_source_groups": list(
                self.shared_official_source_groups
            ),
            "official_split_modified": False,
            "source_disjoint_official_train_test_verified": False,
            "test_event_annotations_read": False,
            "test_truth_available": False,
            "raw_video_frame_ap_status": "explicit_blocked_coordinate_and_timebase_contract_unknown",
        }


def prepare_xd_data(
    inventory: XDArchiveInventory, dataset_root: str | Path, *, seed: int = 202709
) -> XDPreparedData:
    """Freeze full-list train roles before checking local video availability."""
    _validate_inventory(inventory)
    identities: dict[str, str] = {}
    train: list[VideoManifestRecord] = []
    test_index: list[dict[str, Any]] = []
    for member in inventory.members:
        video_id, canonical_member = _member_identity(member)
        if video_id in identities:
            raise XDImportError(
                f"duplicate video identity or train/test leakage: {canonical_member}"
            )
        identities[video_id] = member.volume
        metadata = _member_metadata(member, inventory)
        relative_path = _relative_path(f"{member.volume}/{member.path}")
        if member.volume == "test_videos":
            test_index.append(
                {"video_id": video_id, "path": relative_path, "official_split": "test", **metadata}
            )
            continue
        tokens = parse_xd_labels(member.path)
        abnormal = tokens != ("A",)
        train.append(
            VideoManifestRecord(
                video_id=video_id,
                path=relative_path,
                split="train",
                category="Anomaly" if abnormal else "Normal",
                is_anomaly=abnormal,
                annotations=(
                    SupervisionAnnotation(
                        scope="video",
                        is_anomaly=abnormal,
                        source="xd-official-archive-member-filename-weak-label",
                    ),
                ),
                metadata={
                    **metadata,
                    "raw_label_tokens": list(tokens),
                    "anomaly_class_tokens": [token for token in tokens if token in ANOMALY_TOKENS],
                    "label_metadata_usage": "offline_audit_only_not_reducer_input",
                },
            )
        )
    records = validate_manifest(sorted(train, key=lambda row: row.video_id))
    groups: dict[str, list[VideoManifestRecord]] = defaultdict(list)
    for row in records:
        groups[row.metadata["source_group"]].append(row)
    known_groups = sorted(
        (group for group in groups if group != "xd-source:unknown"),
        key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode()).digest(),
    )
    fit_end = len(known_groups) * 8 // 10
    confirm_end = fit_end + len(known_groups) // 10
    group_roles = {
        group: "fit" if i < fit_end else "confirm" if i < confirm_end else "select"
        for i, group in enumerate(known_groups)
    }
    group_roles["xd-source:unknown"] = "fit"
    lock = {row.video_id: group_roles[row.metadata["source_group"]] for row in records}
    partitions = {
        role: tuple(row for row in records if lock[row.video_id] == role)
        for role in ("fit", "confirm", "select")
    }
    available, pending, availability = [], [], []
    for row in records:
        path = row.resolve_path(dataset_root)
        expected = row.metadata["uncompressed_size"]
        actual = path.stat().st_size if path.is_file() else None
        status = (
            "missing"
            if actual is None
            else "available_size_matched"
            if actual == expected
            else "size_mismatch"
        )
        (available if status == "available_size_matched" else pending).append(row)
        availability.append(
            {
                "video_id": row.video_id,
                "path": row.path,
                "status": status,
                "expected_bytes": expected,
                "observed_bytes": actual,
            }
        )
    test_groups = {row["source_group"] for row in test_index} - {"xd-source:unknown"}
    return XDPreparedData(
        records,
        tuple(sorted(test_index, key=lambda row: row["video_id"])),
        lock,
        tuple(available),
        tuple(pending),
        partitions,
        tuple(availability),
        dict(inventory.central_sha256),
        tuple(sorted(set(groups) & test_groups)),
        seed,
        str(Path(dataset_root).expanduser().resolve()),
    )


def write_prepared_xd_data(prepared: XDPreparedData, output_dir: str | Path) -> dict[str, Path]:
    """Write a new preparation run; existing destinations are never replaced."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    written = {}
    for name, rows in (
        ("train.full", prepared.train),
        ("train.available", prepared.available_train),
        ("train.pending", prepared.pending_train),
    ):
        written[name] = write_manifest_jsonl(rows, root / f"{name}.jsonl")
    available_ids = {row.video_id for row in prepared.available_train}
    for role, rows in prepared.partitions.items():
        for suffix, subset in (
            ("full", rows),
            ("available", tuple(row for row in rows if row.video_id in available_ids)),
        ):
            name = f"{role}.{suffix}"
            written[name] = write_manifest_jsonl(subset, root / f"{name}.jsonl")
    for name, rows in (
        ("test.identity", prepared.test_index),
        ("availability", prepared.availability),
    ):
        path = root / f"{name}.jsonl"
        atomic_write_jsonl(path, rows)
        written[name] = path
    for name, value in (("role-lock", prepared.role_lock()), ("receipt", prepared.receipt())):
        path = root / f"{name}.json"
        atomic_write_json(path, value)
        written[name] = path
    return written
