"""Video-level datasets backed by a clip-level FeatureStore.

Feature records are grouped by video and immutable encoder fingerprint, sorted
by clip index, then exposed as one variable-length sequence per video.  The
module keeps PyTorch optional so feature inspection still works in the minimal
installation.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from vadbench.data.manifest import (
    DatasetSplit,
    SupervisionAnnotation,
    VideoManifestRecord,
    load_manifest_jsonl,
    validate_manifest,
)
from vadbench.features import FeatureRecord, FeatureStore
from vadbench.tasks import build_temporal_targets

try:
    import torch
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by the minimal environment.
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False


SupervisionKind = Literal["weak", "strong"]
FeatureLevel = Literal["clip", "token"]
StrongUnlabeledPolicy = Literal["exclude", "mask", "error"]


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _supervision(value: str) -> SupervisionKind:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"weak", "weak_mil", "weakly_supervised", "mil", "wsvad"}:
        return "weak"
    if normalized in {
        "strong",
        "supervised",
        "temporal",
        "temporal_supervised",
        "frame_supervised",
    }:
        return "strong"
    raise ValueError(f"unknown supervision kind: {value!r}")


def _feature_level(value: str) -> FeatureLevel:
    normalized = value.strip().lower()
    if normalized in {"clip", "clips", "pooled", "pool"}:
        return "clip"
    if normalized in {"token", "tokens", "feature", "features"}:
        return "token"
    raise ValueError("feature_level must be 'clip' or 'token'")


def _manifest_records(
    value: str | Path | Iterable[VideoManifestRecord | Mapping[str, Any]],
) -> tuple[VideoManifestRecord, ...]:
    if isinstance(value, (str, Path)):
        return load_manifest_jsonl(value)
    records = [
        item if isinstance(item, VideoManifestRecord) else VideoManifestRecord.from_dict(item)
        for item in value
    ]
    return validate_manifest(records)


def _has_temporal_truth(record: VideoManifestRecord) -> bool:
    return any(
        _enum_value(item.scope) in {"frame", "segment"}
        and isinstance(item.is_anomaly, bool)
        and item.span is not None
        for item in record.annotations
    )


def _feature_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError(f"{name} must have shape [S,D], [1,S,D], or [D], got {array.shape}")
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be numeric, got dtype={array.dtype}")
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Infinity")
    return result


def _array_1d(
    bundle: Mapping[str, np.ndarray],
    name: str,
    length: int,
    *,
    required: bool = False,
    dtype: Any | None = None,
) -> np.ndarray | None:
    if name not in bundle:
        if required:
            raise ValueError(f"feature bundle is missing required array {name!r}")
        return None
    result = np.asarray(bundle[name], dtype=dtype).reshape(-1)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {result.shape}")
    return result


def _source_metadata(record: FeatureRecord) -> Mapping[str, Any]:
    value = record.metadata.get("source", {})
    return value if isinstance(value, Mapping) else {}


def _clip_frame_interval(record: FeatureRecord) -> tuple[int, int] | None:
    """Prefer the complete MIL segment, not merely its sampled center clip."""

    metadata = _source_metadata(record)
    start = metadata.get("segment_start_frames")
    end = metadata.get("segment_end_frames")
    if (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and 0 <= start < end
    ):
        return start, end
    if record.frame_start is not None and record.frame_end is not None:
        return record.frame_start, record.frame_end
    return None


@dataclass(frozen=True)
class _Entry:
    manifest: VideoManifestRecord
    records: tuple[FeatureRecord, ...]


@dataclass(frozen=True)
class FeatureSequence:
    """One aggregated, unpadded video sequence."""

    video_id: str
    encoder_fingerprint: str
    features: np.ndarray
    feature_valid_mask: np.ndarray
    timeline_start_s: np.ndarray
    timeline_end_s: np.ndarray
    source_frame_start: np.ndarray | None
    source_frame_end: np.ndarray | None
    clip_indices: np.ndarray
    clip_ids: tuple[str, ...]
    video_label: float
    annotations: tuple[SupervisionAnnotation, ...]
    temporal_labels: np.ndarray | None = None
    temporal_valid_mask: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "video_id": self.video_id,
            "encoder_fingerprint": self.encoder_fingerprint,
            "features": self.features,
            "feature_valid_mask": self.feature_valid_mask,
            "valid_mask": (
                self.feature_valid_mask
                if self.temporal_valid_mask is None
                else self.temporal_valid_mask
            ),
            "timeline_start_s": self.timeline_start_s,
            "timeline_end_s": self.timeline_end_s,
            "source_frame_start": self.source_frame_start,
            "source_frame_end": self.source_frame_end,
            "clip_indices": self.clip_indices,
            "clip_ids": self.clip_ids,
            "video_label": self.video_label,
            "annotations": self.annotations,
        }
        if self.temporal_labels is not None:
            result["temporal_labels"] = self.temporal_labels
            result["temporal_valid_mask"] = self.temporal_valid_mask
        return result


class FeatureDataset(Sequence[dict[str, Any]]):
    """Aggregate FeatureStore records into weak or strong video samples.

    Clip-level pooled features are the default because the UCF-Crime MIL
    protocol expects one instance per temporal segment.  Set feature_level to
    token to concatenate the full encoder token sequences instead.

    In strong mode only explicit binary frame/segment annotations are targets.
    Videos carrying only a video label or caption are excluded by default.
    The mask policy retains them with an all-false loss mask for auditing.
    """

    def __init__(
        self,
        feature_store: FeatureStore | str | Path,
        manifest: str | Path | Iterable[VideoManifestRecord | Mapping[str, Any]],
        *,
        encoder_fingerprint: str | None = None,
        supervision: str = "weak",
        feature_level: str = "clip",
        split: DatasetSplit | str | None = None,
        require_all_features: bool = True,
        expected_clips: int | None = None,
        strong_unlabeled: StrongUnlabeledPolicy = "exclude",
        min_overlap_fraction: float = 0.0,
        overlap_reference: Literal["token", "iou"] = "token",
        assume_unannotated_is_normal: bool = True,
    ) -> None:
        self.feature_store = (
            feature_store
            if isinstance(feature_store, FeatureStore)
            else FeatureStore(feature_store)
        )
        self.supervision = _supervision(supervision)
        self.feature_level = _feature_level(feature_level)
        if strong_unlabeled not in {"exclude", "mask", "error"}:
            raise ValueError("strong_unlabeled must be 'exclude', 'mask', or 'error'")
        if not 0.0 <= min_overlap_fraction <= 1.0:
            raise ValueError("min_overlap_fraction must be in [0, 1]")
        if overlap_reference not in {"token", "iou"}:
            raise ValueError("overlap_reference must be 'token' or 'iou'")
        if expected_clips is not None and (isinstance(expected_clips, bool) or expected_clips <= 0):
            raise ValueError("expected_clips must be a positive integer or None")
        self.strong_unlabeled = strong_unlabeled
        self.min_overlap_fraction = float(min_overlap_fraction)
        self.overlap_reference = overlap_reference
        self.assume_unannotated_is_normal = bool(assume_unannotated_is_normal)

        manifests = list(_manifest_records(manifest))
        if split is not None:
            selected_split = DatasetSplit(_enum_value(split))
            manifests = [item for item in manifests if item.split == selected_split]
        if not manifests:
            raise ValueError("manifest selection is empty")

        manifest_ids = {item.video_id for item in manifests}
        indexed = [
            item
            for item in self.feature_store.iter_records()
            if item.video_id in manifest_ids
            and (encoder_fingerprint is None or item.encoder_fingerprint == encoder_fingerprint)
        ]
        fingerprints = {item.encoder_fingerprint for item in indexed}
        if encoder_fingerprint is None:
            if not fingerprints:
                raise ValueError("no feature records match the selected manifest")
            if len(fingerprints) != 1:
                raise ValueError(
                    "multiple encoder fingerprints match; pass encoder_fingerprint explicitly"
                )
            encoder_fingerprint = next(iter(fingerprints))
        elif not fingerprints:
            raise ValueError(f"no features match encoder_fingerprint={encoder_fingerprint!r}")
        self.encoder_fingerprint = encoder_fingerprint

        grouped: dict[tuple[str, str], list[FeatureRecord]] = {}
        for item in indexed:
            grouped.setdefault((item.video_id, item.encoder_fingerprint), []).append(item)

        entries: list[_Entry] = []
        missing: list[str] = []
        for manifest_item in manifests:
            if self.supervision == "strong" and not _has_temporal_truth(manifest_item):
                if strong_unlabeled == "exclude":
                    continue
                if strong_unlabeled == "error":
                    raise ValueError(
                        f"{manifest_item.video_id} has no explicit binary frame/segment annotation"
                    )
            feature_records = grouped.get((manifest_item.video_id, self.encoder_fingerprint), [])
            if not feature_records:
                missing.append(manifest_item.video_id)
                continue
            ordered = sorted(feature_records, key=lambda item: (item.clip_index, item.clip_id))
            indices = [item.clip_index for item in ordered]
            if len(indices) != len(set(indices)):
                raise ValueError(f"{manifest_item.video_id} has duplicate clip_index values")
            if expected_clips is not None and indices != list(range(expected_clips)):
                raise ValueError(
                    f"{manifest_item.video_id} clip_index values must be exactly "
                    f"0..{expected_clips - 1}, got {indices}"
                )
            for feature_record in ordered:
                source = _source_metadata(feature_record)
                source_split = source.get("split")
                if (
                    source_split is not None
                    and _enum_value(source_split) != manifest_item.split.value
                ):
                    raise ValueError(
                        f"{manifest_item.video_id}/{feature_record.clip_id}: "
                        "feature source split conflicts with manifest"
                    )
                source_label = source.get("is_anomaly")
                if source_label is not None and (
                    not isinstance(source_label, bool) or source_label != manifest_item.is_anomaly
                ):
                    raise ValueError(
                        f"{manifest_item.video_id}/{feature_record.clip_id}: "
                        "feature source label conflicts with manifest"
                    )
            entries.append(_Entry(manifest_item, tuple(ordered)))

        if missing and require_all_features:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(
                f"{len(missing)} manifest videos have no matching features: {preview}{suffix}"
            )
        if not entries:
            raise ValueError("no manifest videos with usable matching features remain")
        self._entries = tuple(entries)

        dimensions = {self._record_feature_dim(item) for entry in entries for item in entry.records}
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
        self.feature_dim = next(iter(dimensions))

    def _record_feature_dim(self, record: FeatureRecord) -> int:
        reference = record.arrays.get("pooled") if self.feature_level == "clip" else None
        if reference is None:
            return record.feature_dim
        if len(reference.shape) == 1 and reference.shape[0] > 0:
            return int(reference.shape[0])
        if len(reference.shape) == 2 and reference.shape[0] == 1:
            return int(reference.shape[1])
        raise ValueError(f"{record.video_id}/{record.clip_id}: invalid pooled shape")

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(item.manifest.video_id for item in self._entries)

    @property
    def manifest_records(self) -> tuple[VideoManifestRecord, ...]:
        return tuple(item.manifest for item in self._entries)

    def _load_clip(
        self, record: FeatureRecord
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        bundle = self.feature_store.load_bundle(record)
        location = f"{record.video_id}/{record.clip_id}"
        if "pooled" in bundle:
            features = _feature_matrix(bundle["pooled"], name=f"{location}.pooled")
            if features.shape[0] != 1:
                raise ValueError(f"{location}.pooled must contain one clip vector")
        else:
            tokens = _feature_matrix(bundle["features"], name=f"{location}.features")
            token_valid = _array_1d(bundle, "timeline_valid", tokens.shape[0], dtype=bool)
            if token_valid is None:
                token_valid = np.ones(tokens.shape[0], dtype=bool)
            if not token_valid.any():
                raise ValueError(f"{location} has no valid token to pool")
            features = tokens[token_valid].mean(axis=0, dtype=np.float32)[None, :]
        valid = np.ones(1, dtype=bool)
        starts = np.asarray([record.start_s], dtype=np.float64)
        ends = np.asarray([record.end_s], dtype=np.float64)
        frame_interval = _clip_frame_interval(record)
        source_starts = (
            None if frame_interval is None else np.asarray([frame_interval[0]], dtype=np.int64)
        )
        source_ends = (
            None if frame_interval is None else np.asarray([frame_interval[1]], dtype=np.int64)
        )
        return features, valid, starts, ends, source_starts, source_ends

    def _load_tokens(
        self, record: FeatureRecord
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        bundle = self.feature_store.load_bundle(record)
        location = f"{record.video_id}/{record.clip_id}"
        features = _feature_matrix(bundle["features"], name=f"{location}.features")
        count = features.shape[0]
        starts = _array_1d(
            bundle,
            "timeline_start_s",
            count,
            required=self.supervision == "strong",
            dtype=np.float64,
        )
        ends = _array_1d(
            bundle,
            "timeline_end_s",
            count,
            required=self.supervision == "strong",
            dtype=np.float64,
        )
        if starts is None or ends is None:
            edges = np.linspace(record.start_s, record.end_s, count + 1, dtype=np.float64)
            starts, ends = edges[:-1], edges[1:]
        if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
            raise ValueError(f"{location} timeline contains NaN or Infinity")
        if np.any(starts < 0) or np.any(ends < starts):
            raise ValueError(f"{location} has invalid timeline intervals")
        valid = _array_1d(bundle, "timeline_valid", count, dtype=bool)
        if valid is None:
            valid = np.ones(count, dtype=bool)
        source_starts = _array_1d(bundle, "source_frame_start", count, dtype=np.int64)
        source_ends = _array_1d(bundle, "source_frame_end", count, dtype=np.int64)
        if (source_starts is None) != (source_ends is None):
            raise ValueError(f"{location} source frame arrays must be stored together")
        return features, valid, starts, ends, source_starts, source_ends

    def _strong_targets(
        self,
        annotations: Sequence[SupervisionAnnotation],
        *,
        starts: np.ndarray,
        ends: np.ndarray,
        source_starts: np.ndarray | None,
        source_ends: np.ndarray | None,
        clip_indices: np.ndarray,
        feature_valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        explicit = [
            item
            for item in annotations
            if _enum_value(item.scope) in {"frame", "segment"}
            and isinstance(item.is_anomaly, bool)
            and item.span is not None
        ]
        aligned = [
            item
            for item in explicit
            if _enum_value(item.span.unit) in {"frame", "second", "seconds"}
        ]
        segment_indexed = [item for item in explicit if _enum_value(item.span.unit) == "segment"]
        timeline = SimpleNamespace(
            start_s=starts[None, :],
            end_s=ends[None, :],
            valid_mask=feature_valid[None, :],
            source_frame_start=(None if source_starts is None else source_starts[None, :]),
            source_frame_end=None if source_ends is None else source_ends[None, :],
        )
        target = build_temporal_targets(
            timeline,
            [aligned],
            min_overlap_fraction=self.min_overlap_fraction,
            overlap_reference=self.overlap_reference,
            assume_unannotated_is_normal=False,
        )
        labels = np.asarray(target.labels[0], dtype=np.float32).copy()
        valid = np.asarray(target.valid_mask[0], dtype=bool).copy()

        token_starts = clip_indices.astype(np.float64, copy=False)
        token_ends = token_starts + 1.0
        for item in segment_indexed:
            span = item.span
            assert span is not None
            annotation_start, annotation_end = float(span.start), float(span.end)
            overlap = np.maximum(
                0.0,
                np.minimum(token_ends, annotation_end) - np.maximum(token_starts, annotation_start),
            )
            if self.overlap_reference == "token":
                denominator = token_ends - token_starts
            else:
                denominator = (
                    token_ends - token_starts + annotation_end - annotation_start - overlap
                )
            fraction = np.divide(
                overlap,
                denominator,
                out=np.zeros_like(overlap),
                where=denominator > 0,
            )
            selected = (overlap > 0) & feature_valid
            if self.min_overlap_fraction > 0:
                selected &= fraction >= self.min_overlap_fraction
            if item.is_anomaly:
                labels[selected] = 1.0
            valid[selected] = True

        if explicit and self.assume_unannotated_is_normal:
            valid = feature_valid.copy()
        labels[~valid] = 0.0
        return labels, valid

    def _sequence(self, entry: _Entry) -> FeatureSequence:
        feature_parts: list[np.ndarray] = []
        valid_parts: list[np.ndarray] = []
        start_parts: list[np.ndarray] = []
        end_parts: list[np.ndarray] = []
        source_start_parts: list[np.ndarray | None] = []
        source_end_parts: list[np.ndarray | None] = []
        clip_index_parts: list[np.ndarray] = []

        for record in entry.records:
            loaded = (
                self._load_clip(record)
                if self.feature_level == "clip"
                else self._load_tokens(record)
            )
            features, valid, starts, ends, source_starts, source_ends = loaded
            if (
                self.feature_level == "clip"
                and entry.manifest.fps is not None
                and source_starts is not None
                and source_ends is not None
                and _source_metadata(record).get("sampling_kind") == "uniform_segments"
            ):
                starts = source_starts.astype(np.float64) / float(entry.manifest.fps)
                ends = source_ends.astype(np.float64) / float(entry.manifest.fps)
            feature_parts.append(features)
            valid_parts.append(valid)
            start_parts.append(starts)
            end_parts.append(ends)
            source_start_parts.append(source_starts)
            source_end_parts.append(source_ends)
            clip_index_parts.append(np.full(features.shape[0], record.clip_index, dtype=np.int64))

        features = np.concatenate(feature_parts, axis=0)
        feature_valid = np.concatenate(valid_parts).astype(bool, copy=False)
        starts = np.concatenate(start_parts)
        ends = np.concatenate(end_parts)
        clip_indices = np.concatenate(clip_index_parts)
        complete_frames = all(item is not None for item in source_start_parts)
        source_starts = (
            np.concatenate([item for item in source_start_parts if item is not None])
            if complete_frames
            else None
        )
        source_ends = (
            np.concatenate([item for item in source_end_parts if item is not None])
            if complete_frames
            else None
        )
        temporal_labels: np.ndarray | None = None
        temporal_valid: np.ndarray | None = None
        if self.supervision == "strong":
            temporal_labels, temporal_valid = self._strong_targets(
                entry.manifest.annotations,
                starts=starts,
                ends=ends,
                source_starts=source_starts,
                source_ends=source_ends,
                clip_indices=clip_indices,
                feature_valid=feature_valid,
            )
        return FeatureSequence(
            video_id=entry.manifest.video_id,
            encoder_fingerprint=self.encoder_fingerprint,
            features=features,
            feature_valid_mask=feature_valid,
            timeline_start_s=starts,
            timeline_end_s=ends,
            source_frame_start=source_starts,
            source_frame_end=source_ends,
            clip_indices=clip_indices,
            clip_ids=tuple(item.clip_id for item in entry.records),
            video_label=float(entry.manifest.is_anomaly),
            annotations=tuple(entry.manifest.annotations),
            temporal_labels=temporal_labels,
            temporal_valid_mask=temporal_valid,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._sequence(self._entries[index]).to_dict()


def _copy_row(destination: np.ndarray, row: int, value: Any, length: int) -> None:
    source = np.asarray(value).reshape(-1)
    if source.shape != (length,):
        raise ValueError(f"row value must have shape {(length,)}, got {source.shape}")
    destination[row, :length] = source


def _tensor(value: Any) -> Any:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for tensor collation; install the train extra")
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Mapping):
        return {key: _tensor(item) for key, item in value.items()}
    return value


def collate_feature_batch(
    samples: Sequence[Mapping[str, Any]],
    *,
    supervision: str | None = None,
    as_torch: bool | None = None,
) -> dict[str, Any]:
    """Pad video sequences to features [B,S,D] plus an explicit valid mask."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    strong_flags = ["temporal_labels" in item for item in samples]
    if any(strong_flags) and not all(strong_flags):
        raise ValueError("cannot mix weak and strong samples")
    kind = _supervision(supervision) if supervision else ("strong" if all(strong_flags) else "weak")
    if (kind == "strong") != all(strong_flags):
        raise ValueError("collate supervision does not match sample targets")

    arrays = [np.asarray(item["features"], dtype=np.float32) for item in samples]
    if any(item.ndim != 2 or min(item.shape) <= 0 for item in arrays):
        raise ValueError("every feature sample must be a non-empty [S,D] array")
    dimensions = {int(item.shape[1]) for item in arrays}
    if len(dimensions) != 1:
        raise ValueError(f"feature dimensions differ in batch: {sorted(dimensions)}")
    batch_size = len(samples)
    max_length = max(item.shape[0] for item in arrays)
    feature_dim = next(iter(dimensions))

    features = np.zeros((batch_size, max_length, feature_dim), dtype=np.float32)
    feature_valid = np.zeros((batch_size, max_length), dtype=bool)
    starts = np.zeros((batch_size, max_length), dtype=np.float64)
    ends = np.zeros((batch_size, max_length), dtype=np.float64)
    clip_indices = np.full((batch_size, max_length), -1, dtype=np.int64)
    source_available = all(item.get("source_frame_start") is not None for item in samples)
    source_starts = (
        np.full((batch_size, max_length), -1, dtype=np.int64) if source_available else None
    )
    source_ends = (
        np.full((batch_size, max_length), -1, dtype=np.int64) if source_available else None
    )

    for row, (sample, array) in enumerate(zip(samples, arrays, strict=True)):
        length = array.shape[0]
        features[row, :length] = array
        _copy_row(feature_valid, row, sample["feature_valid_mask"], length)
        _copy_row(starts, row, sample["timeline_start_s"], length)
        _copy_row(ends, row, sample["timeline_end_s"], length)
        _copy_row(clip_indices, row, sample["clip_indices"], length)
        if source_available:
            _copy_row(source_starts, row, sample["source_frame_start"], length)
            _copy_row(source_ends, row, sample["source_frame_end"], length)

    timeline: dict[str, Any] = {
        "start_s": starts,
        "end_s": ends,
        "valid_mask": feature_valid,
        "source_frame_start": source_starts,
        "source_frame_end": source_ends,
    }
    batch: dict[str, Any] = {
        "features": features,
        "feature_valid_mask": feature_valid,
        "timeline": timeline,
        "video_ids": tuple(str(item["video_id"]) for item in samples),
        "encoder_fingerprints": tuple(str(item["encoder_fingerprint"]) for item in samples),
        "clip_indices": clip_indices,
        "clip_ids": tuple(tuple(item["clip_ids"]) for item in samples),
    }
    if kind == "weak":
        batch["valid_mask"] = feature_valid.copy()
        batch["video_labels"] = np.asarray(
            [float(item["video_label"]) for item in samples], dtype=np.float32
        )
    else:
        labels = np.zeros((batch_size, max_length), dtype=np.float32)
        temporal_valid = np.zeros((batch_size, max_length), dtype=bool)
        for row, sample in enumerate(samples):
            length = arrays[row].shape[0]
            _copy_row(labels, row, sample["temporal_labels"], length)
            _copy_row(temporal_valid, row, sample["temporal_valid_mask"], length)
        temporal_valid &= feature_valid
        batch["valid_mask"] = temporal_valid
        batch["temporal_labels"] = labels
        batch["temporal_valid_mask"] = temporal_valid

    use_torch = TORCH_AVAILABLE if as_torch is None else as_torch
    if use_torch:
        tensor_fields = {
            "features",
            "feature_valid_mask",
            "timeline",
            "clip_indices",
            "valid_mask",
            "video_labels",
            "temporal_labels",
            "temporal_valid_mask",
        }
        for name in tensor_fields & batch.keys():
            batch[name] = _tensor(batch[name])
    return batch


def build_feature_dataloader(
    dataset: FeatureDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    generator: Any | None = None,
) -> Any:
    """Construct a DataLoader using the supervision-safe padded collator."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DataLoader; install the train extra")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    collate = functools.partial(
        collate_feature_batch,
        supervision=dataset.supervision,
        as_torch=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate,
        generator=generator,
    )


__all__ = [
    "FeatureDataset",
    "FeatureLevel",
    "FeatureSequence",
    "StrongUnlabeledPolicy",
    "SupervisionKind",
    "TORCH_AVAILABLE",
    "build_feature_dataloader",
    "collate_feature_batch",
]
