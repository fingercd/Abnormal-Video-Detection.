"""Read-only near-duplicate candidate audit for frozen video manifests.

The output is evidence for an operator to inspect.  A perceptual match is
never converted into a source group, partition change, or verified provenance.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.data.manifest import VideoManifestRecord
from vadbench.data.video import decode_rgb_frames, probe_video
from vadbench.features import atomic_write_json


class SourceAuditError(ValueError):
    """The frozen input cannot be audited without weakening its evidence."""


@dataclass(frozen=True, slots=True)
class SourceAuditConfig:
    """Fixed, label-independent candidate thresholds."""

    samples_per_video: int = 9
    thumbnail_size: int = 16
    max_hamming_distance: int = 6
    min_matching_frames: int = 4
    min_consecutive_matches: int = 3
    minimum_luminance_std: float = 4.0
    max_contact_sheets: int = 20

    def __post_init__(self) -> None:
        for name in (
            "samples_per_video",
            "thumbnail_size",
            "max_hamming_distance",
            "min_matching_frames",
            "min_consecutive_matches",
            "max_contact_sheets",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SourceAuditError(f"{name} 必须是正整数")
        if self.max_hamming_distance > self.thumbnail_size * self.thumbnail_size:
            raise SourceAuditError("max_hamming_distance 超过摘要位数")
        if self.min_matching_frames > self.samples_per_video:
            raise SourceAuditError("min_matching_frames 不能超过 samples_per_video")
        if self.min_consecutive_matches > self.samples_per_video:
            raise SourceAuditError("min_consecutive_matches 不能超过 samples_per_video")
        if self.minimum_luminance_std < 0:
            raise SourceAuditError("minimum_luminance_std 必须非负")


@dataclass(frozen=True, slots=True)
class FrameDigest:
    index: int
    perceptual_hash: str | None
    content_digest: str
    luminance_std: float
    informative: bool


@dataclass(frozen=True, slots=True)
class VideoDigest:
    video_id: str
    relative_path: str
    file_size: int
    known_sha256: str | None
    num_frames: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    sampled_indices: tuple[int, ...]
    frames: tuple[FrameDigest, ...]
    verified_source_id: str | None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["sampled_indices"] = list(self.sampled_indices)
        result["frames"] = [asdict(frame) for frame in self.frames]
        return result


@dataclass(frozen=True, slots=True)
class SourceAuditResult:
    videos: tuple[VideoDigest, ...]
    candidates: tuple[dict[str, Any], ...]
    config: SourceAuditConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scope": "read_only_near_duplicate_candidates_not_verified_source_groups",
            "execution": "single_process_sequential",
            "config": asdict(self.config),
            "video_count": len(self.videos),
            "candidate_pair_count": len(self.candidates),
            "videos": [item.as_dict() for item in self.videos],
            "candidate_pairs": list(self.candidates),
            "automatic_source_group_updates": False,
        }


def _uniform_indices(num_frames: int, count: int) -> tuple[int, ...]:
    return tuple(sorted({int(round(value)) for value in np.linspace(0, num_frames - 1, count)}))


def _thumbnail(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    rows = np.linspace(0, height - 1, size).round().astype(np.int64)
    cols = np.linspace(0, width - 1, size).round().astype(np.int64)
    rgb = frame[rows][:, cols].astype(np.float32)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _digest_frame(index: int, frame: np.ndarray, config: SourceAuditConfig) -> FrameDigest:
    thumbnail = _thumbnail(frame, config.thumbnail_size)
    luminance_std = float(thumbnail.std())
    informative = luminance_std >= config.minimum_luminance_std
    bits = thumbnail > float(thumbnail.mean())
    packed = np.packbits(bits.reshape(-1)).tobytes()
    return FrameDigest(
        index=index,
        perceptual_hash=packed.hex() if informative else None,
        content_digest=hashlib.blake2b(thumbnail.astype(np.uint8).tobytes(), digest_size=12).hexdigest(),
        luminance_std=round(luminance_std, 6),
        informative=informative,
    )


def _hamming(first: str, second: str) -> int:
    return sum(
        (left ^ right).bit_count()
        for left, right in zip(bytes.fromhex(first), bytes.fromhex(second), strict=True)
    )


def _longest_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _candidate_pairs(videos: Sequence[VideoDigest], config: SourceAuditConfig) -> tuple[dict[str, Any], ...]:
    # LSH-style blocks avoid comparing every pair for every frame on full train.
    buckets: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    by_id = {video.video_id: video for video in videos}
    for video in videos:
        for position, frame in enumerate(video.frames):
            if frame.perceptual_hash is None:
                continue
            encoded = bytes.fromhex(frame.perceptual_hash)
            for block in range(4):
                key = encoded[block * 8 : (block + 1) * 8].hex()
                buckets[position, block, key].add(video.video_id)
    pairs: set[tuple[str, str]] = set()
    for video_ids in buckets.values():
        if len(video_ids) > 1:
            pairs.update(tuple(sorted(pair)) for pair in itertools.combinations(sorted(video_ids), 2))

    result: list[dict[str, Any]] = []
    for first_id, second_id in sorted(pairs):
        first, second = by_id[first_id], by_id[second_id]
        matches: list[bool] = []
        distances: list[int | None] = []
        for left, right in zip(first.frames, second.frames, strict=True):
            if left.perceptual_hash is None or right.perceptual_hash is None:
                matches.append(False)
                distances.append(None)
                continue
            distance = _hamming(left.perceptual_hash, right.perceptual_hash)
            distances.append(distance)
            matches.append(distance <= config.max_hamming_distance)
        matching = sum(matches)
        longest = _longest_run(matches)
        if matching < config.min_matching_frames or longest < config.min_consecutive_matches:
            continue
        finite = [item for item in distances if item is not None]
        result.append(
            {
                "video_id_a": first.video_id,
                "video_id_b": second.video_id,
                "evidence": "perceptual_candidate_requires_human_review",
                "matching_sample_positions": [index for index, value in enumerate(matches) if value],
                "matching_frames": matching,
                "longest_consecutive_match": longest,
                "mean_hamming_distance_informative": round(float(np.mean(finite)), 6),
                "verified_source_a": first.verified_source_id,
                "verified_source_b": second.verified_source_id,
                "automatic_source_group": None,
            }
        )
    return tuple(result)


def audit_source_candidates(
    records: Sequence[VideoManifestRecord],
    dataset_root: str | Path,
    *,
    known_sha256: Mapping[str, str] | None = None,
    verified_sources: Mapping[str, str] | None = None,
    config: SourceAuditConfig | None = None,
) -> SourceAuditResult:
    """Create per-video streaming digests and unverified near-duplicate candidates."""

    if not records:
        raise SourceAuditError("manifest 不能为空")
    config = config or SourceAuditConfig()
    ids = [record.video_id for record in records]
    if len(ids) != len(set(ids)):
        raise SourceAuditError("manifest video_id 必须唯一")
    known_sha256 = dict(known_sha256 or {})
    verified_sources = dict(verified_sources or {})
    unknown = (set(known_sha256) | set(verified_sources)) - set(ids)
    if unknown:
        raise SourceAuditError(f"身份映射含 manifest 外 video_id：{sorted(unknown)}")

    videos: list[VideoDigest] = []
    for record in sorted(records, key=lambda item: item.video_id.casefold()):
        path = record.resolve_path(dataset_root)
        info = probe_video(path)
        indices = _uniform_indices(info.num_frames, config.samples_per_video)
        frames = decode_rgb_frames(path, indices)
        digests = tuple(
            _digest_frame(index, frame, config) for index, frame in zip(indices, frames, strict=True)
        )
        videos.append(
            VideoDigest(
                video_id=record.video_id,
                relative_path=record.path,
                file_size=path.stat().st_size,
                known_sha256=known_sha256.get(record.video_id),
                num_frames=info.num_frames,
                fps=info.fps,
                duration_seconds=info.duration_seconds,
                width=info.width,
                height=info.height,
                sampled_indices=indices,
                frames=digests,
                verified_source_id=verified_sources.get(record.video_id),
            )
        )
    frozen = tuple(videos)
    return SourceAuditResult(videos=frozen, candidates=_candidate_pairs(frozen, config), config=config)


def write_contact_sheets(
    result: SourceAuditResult,
    records: Sequence[VideoManifestRecord],
    dataset_root: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Write small visual evidence for candidates; never encode a source decision."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional environment dependency
        raise SourceAuditError("contact sheet 需要 OpenCV") from exc
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    by_id = {video.video_id: video for video in result.videos}
    record_by_id = {record.video_id: record for record in records}
    sheets: list[Path] = []
    for ordinal, pair in enumerate(result.candidates[: result.config.max_contact_sheets], start=1):
        first, second = by_id[pair["video_id_a"]], by_id[pair["video_id_b"]]
        rows: list[np.ndarray] = []
        for video in (first, second):
            frames = decode_rgb_frames(
                record_by_id[video.video_id].resolve_path(dataset_root), video.sampled_indices
            )
            tiles = [cv2.resize(frame[..., ::-1], (128, 96), interpolation=cv2.INTER_AREA) for frame in frames]
            row = np.concatenate(tiles, axis=1)
            cv2.putText(
                row,
                video.video_id,
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            rows.append(row)
        sheet = np.concatenate(rows, axis=0)
        output = destination / f"candidate-{ordinal:03d}-{first.video_id}--{second.video_id}.png"
        if not cv2.imwrite(str(output), sheet):
            raise SourceAuditError(f"无法写入 contact sheet：{output}")
        sheets.append(output)
    return sheets


def write_overview_contact_sheet(
    result: SourceAuditResult,
    records: Sequence[VideoManifestRecord],
    dataset_root: str | Path,
    output_path: str | Path,
) -> Path:
    """Write an all-video review sheet even when no pair crosses the threshold."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional environment dependency
        raise SourceAuditError("contact sheet 需要 OpenCV") from exc
    record_by_id = {record.video_id: record for record in records}
    rows: list[np.ndarray] = []
    for video in result.videos:
        frames = decode_rgb_frames(
            record_by_id[video.video_id].resolve_path(dataset_root), video.sampled_indices
        )
        tiles = [cv2.resize(frame[..., ::-1], (96, 72), interpolation=cv2.INTER_AREA) for frame in frames]
        row = np.concatenate(tiles, axis=1)
        cv2.putText(
            row,
            video.video_id,
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(row)
    output = Path(output_path)
    if not cv2.imwrite(str(output), np.concatenate(rows, axis=0)):
        raise SourceAuditError(f"无法写入 overview contact sheet：{output}")
    return output


def write_source_audit(result: SourceAuditResult, output_dir: str | Path) -> Path:
    """Write JSON evidence; the caller controls any visual rendering separately."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "source-audit.json"
    atomic_write_json(output, result.as_dict())
    return output


__all__ = [
    "SourceAuditConfig",
    "SourceAuditError",
    "SourceAuditResult",
    "audit_source_candidates",
    "write_source_audit",
    "write_overview_contact_sheet",
]
