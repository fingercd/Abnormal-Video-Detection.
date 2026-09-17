"""Durable, model-agnostic feature artifacts.

The feature index deliberately contains metadata only.  Dense encoder outputs are
stored in ``.npz`` or ``.npy`` files and referenced from a JSONL record.  This
keeps the index streamable and prevents a common (and very costly) failure mode:
serialising token tensors as JSON lists.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.hashing import sha256_file as _sha256_file

FEATURE_INDEX_SCHEMA_VERSION = "vadbench.feature-index.v1"
FINGERPRINT_ALGORITHM = "sha256"
MAX_INLINE_METADATA_ITEMS = 2048


def utc_now_iso() -> str:
    """Return a stable UTC timestamp suitable for provenance records."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_path(path: Path) -> str:
    """Hash a checkpoint file or a sharded checkpoint directory reproducibly."""

    path = path.resolve()
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


_LOCATION_ONLY_KEYS = {
    "cache_dir",
    "checkpoint_path",
    "local_files_only",
    "local_path",
    "weights_path",
    "work_dir",
    "device",
}


def _manifest_value(value: Any) -> Any:
    """Convert a manifest to canonical, location-independent JSON data."""

    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    elif hasattr(value, "model_dump") and callable(value.model_dump):
        value = value.model_dump(mode="json")
    elif not isinstance(
        value, (Mapping, str, bytes, int, float, bool, type(None), Sequence)
    ) and hasattr(value, "__dict__"):
        value = vars(value)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _LOCATION_ONLY_KEYS:
                continue
            result[key] = _manifest_value(item)
        return result
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        raise TypeError("encoder manifest must not contain tensor/ndarray values")
    if isinstance(value, (list, tuple)):
        return [_manifest_value(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("encoder manifest contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def compute_encoder_fingerprint(
    manifest: Any,
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    checkpoint_id: str | None = None,
) -> str:
    """Return a portable fingerprint for an encoder configuration and weights.

    Local-only fields such as ``device`` and ``checkpoint_path`` are omitted from
    the manifest projection.  When a checkpoint path is supplied its *contents*,
    not its machine-specific path, participate in the fingerprint.  For remotely
    hosted weights, pass an immutable revision/checkpoint id.
    """

    if checkpoint is not None and checkpoint_id is not None:
        raise ValueError("checkpoint and checkpoint_id are mutually exclusive")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": FINGERPRINT_ALGORITHM,
        "manifest": _manifest_value(manifest),
    }
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        payload["checkpoint"] = {
            "sha256": _sha256_path(checkpoint_path),
            "kind": "directory" if checkpoint_path.is_dir() else "file",
        }
    elif checkpoint_id is not None:
        payload["checkpoint"] = {"id": checkpoint_id}
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return f"{FINGERPRINT_ALGORITHM}:{digest}"


# Friendly alias used by config/CLI callers.
encoder_fingerprint = compute_encoder_fingerprint


def _count_inline_items(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(1 + _count_inline_items(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(1 + _count_inline_items(item) for item in value)
    return 1


def ensure_json_metadata(value: Any, *, max_items: int = MAX_INLINE_METADATA_ITEMS) -> Any:
    """Return JSON-safe metadata while rejecting tensors and oversized lists."""

    def convert(item: Any, location: str) -> Any:
        if isinstance(item, np.ndarray):
            raise TypeError(f"{location}: ndarray must be stored in NPZ/NPY, not JSON")
        item_type = type(item)
        if item_type.__module__.startswith("torch") and item_type.__name__ == "Tensor":
            raise TypeError(f"{location}: tensor must be stored in NPZ/NPY, not JSON")
        if isinstance(item, np.generic):
            return item.item()
        if dataclasses.is_dataclass(item):
            item = dataclasses.asdict(item)
        if isinstance(item, Path):
            return item.as_posix()
        if isinstance(item, Mapping):
            return {str(key): convert(child, f"{location}.{key}") for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(child, f"{location}[{index}]") for index, child in enumerate(item)]
        if isinstance(item, float) and not np.isfinite(item):
            raise ValueError(f"{location}: JSON metadata cannot contain NaN/Infinity")
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"{location}: unsupported JSON metadata type {type(item).__name__}")

    converted = convert(value, "metadata")
    item_count = _count_inline_items(converted)
    if item_count > max_items:
        raise ValueError(
            f"metadata contains {item_count} inline items (limit {max_items}); "
            "store dense values as NPZ/NPY arrays"
        )
    # Validate the same strict JSON settings used by the writers.
    _canonical_json(converted)
    return converted


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically write strict UTF-8 JSON."""

    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_replace_bytes(Path(path), payload)


def atomic_write_jsonl(path: str | os.PathLike[str], records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically replace a JSONL file with complete, newline-terminated rows."""

    rows = []
    for record in records:
        rows.append(
            json.dumps(
                record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
            )
        )
    payload = (("\n".join(rows) + "\n") if rows else "").encode("utf-8")
    _atomic_replace_bytes(Path(path), payload)


class _InterProcessLock:
    """Tiny lock-file guard for short index transactions.

    Feature extraction commonly uses more than one worker.  ``os.replace`` makes
    readers safe, while this lock prevents two writers from losing one another's
    index updates.
    """

    _thread_locks: MutableMapping[str, threading.RLock] = {}
    _thread_locks_guard = threading.Lock()

    def __init__(self, target: Path, timeout_s: float = 30.0, stale_after_s: float = 300.0) -> None:
        self.path = target.with_name(f".{target.name}.lock")
        self.timeout_s = timeout_s
        self.stale_after_s = stale_after_s
        with self._thread_locks_guard:
            self._thread_lock = self._thread_locks.setdefault(
                str(self.path.resolve()), threading.RLock()
            )
        self._fd: int | None = None

    def __enter__(self) -> _InterProcessLock:
        self._thread_lock.acquire()
        deadline = time.monotonic() + self.timeout_s
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            while True:
                try:
                    self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(self._fd, f"pid={os.getpid()} time={time.time()}\n".encode("ascii"))
                    return self
                except FileExistsError:
                    with contextlib.suppress(FileNotFoundError):
                        if time.time() - self.path.stat().st_mtime > self.stale_after_s:
                            self.path.unlink()
                            continue
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out acquiring feature index lock: {self.path}"
                        ) from None
                    time.sleep(0.02)
        except BaseException:
            self._thread_lock.release()
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self.path.unlink(missing_ok=True)
        finally:
            self._thread_lock.release()


@dataclass(frozen=True)
class ArrayReference:
    """Location and integrity metadata for one dense array."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    nbytes: int
    key: str | None = None

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute():
            raise ValueError("array path must be a non-empty path relative to the feature store")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("array shape dimensions must be non-negative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("array sha256 must be 64 lowercase hexadecimal characters")
        if self.nbytes < 0:
            raise ValueError("array nbytes must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
            "nbytes": self.nbytes,
        }
        if self.key is not None:
            result["key"] = self.key
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArrayReference:
        return cls(
            path=str(value["path"]),
            key=str(value["key"]) if value.get("key") is not None else None,
            shape=tuple(int(item) for item in value["shape"]),
            dtype=str(value["dtype"]),
            sha256=str(value["sha256"]),
            nbytes=int(value["nbytes"]),
        )


@dataclass(frozen=True)
class FeatureRecord:
    """One clip-level row in ``index.jsonl``."""

    video_id: str
    clip_id: str
    clip_index: int
    encoder_fingerprint: str
    storage_format: str
    arrays: Mapping[str, ArrayReference]
    start_s: float
    end_s: float
    frame_start: int | None = None
    frame_end: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = FEATURE_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_INDEX_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature index schema: {self.schema_version}")
        if not self.video_id or not self.clip_id:
            raise ValueError("video_id and clip_id must be non-empty")
        if self.clip_index < 0:
            raise ValueError("clip_index must be non-negative")
        if self.storage_format not in {"npz", "npy"}:
            raise ValueError("storage_format must be 'npz' or 'npy'")
        if "features" not in self.arrays:
            raise ValueError("arrays must contain the primary 'features' array")
        if (
            not np.isfinite(self.start_s)
            or not np.isfinite(self.end_s)
            or self.end_s < self.start_s
        ):
            raise ValueError("clip time interval must be finite and end_s >= start_s")
        if (self.frame_start is None) != (self.frame_end is None):
            raise ValueError("frame_start and frame_end must either both be set or both be null")
        if (
            self.frame_start is not None
            and self.frame_end is not None
            and (self.frame_start < 0 or self.frame_end < self.frame_start)
        ):
            raise ValueError("frame interval must satisfy 0 <= frame_start <= frame_end")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.encoder_fingerprint):
            raise ValueError(
                "encoder_fingerprint must use the form sha256:<64 lowercase hex chars>"
            )
        object.__setattr__(self, "arrays", dict(self.arrays))
        object.__setattr__(self, "metadata", ensure_json_metadata(self.metadata))

    @property
    def feature_path(self) -> str:
        """Compatibility shortcut for consumers interested only in features."""

        return self.arrays["features"].path

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arrays["features"].shape

    @property
    def dtype(self) -> str:
        return self.arrays["features"].dtype

    @property
    def token_count(self) -> int:
        return self.shape[-2] if len(self.shape) >= 2 else self.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.shape[-1]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "clip_id": self.clip_id,
            "clip_index": self.clip_index,
            "encoder_fingerprint": self.encoder_fingerprint,
            "storage_format": self.storage_format,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "feature_path": self.feature_path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "arrays": {
                name: reference.to_dict() for name, reference in sorted(self.arrays.items())
            },
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FeatureRecord:
        arrays = {
            str(name): ArrayReference.from_dict(reference)
            for name, reference in dict(value["arrays"]).items()
        }
        return cls(
            schema_version=str(value.get("schema_version", FEATURE_INDEX_SCHEMA_VERSION)),
            video_id=str(value["video_id"]),
            clip_id=str(value["clip_id"]),
            clip_index=int(value["clip_index"]),
            encoder_fingerprint=str(value["encoder_fingerprint"]),
            storage_format=str(value["storage_format"]),
            arrays=arrays,
            start_s=float(value["start_s"]),
            end_s=float(value["end_s"]),
            frame_start=int(value["frame_start"]) if value.get("frame_start") is not None else None,
            frame_end=int(value["frame_end"]) if value.get("frame_end") is not None else None,
            metadata=dict(value.get("metadata", {})),
            created_at=str(value.get("created_at", utc_now_iso())),
        )


def _slug(value: str, *, fallback: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")[:64] or fallback
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{compact}-{suffix}"


def _normalise_array(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"{name} has object dtype, which is not a portable feature artifact")
    if array.dtype.kind not in "biufc":
        raise TypeError(f"{name} must be a numeric or boolean array, got dtype={array.dtype}")
    return np.ascontiguousarray(array)


def _save_temp_array(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _save_temp_npz(path: Path, arrays: Mapping[str, np.ndarray], *, compressed: bool) -> None:
    with path.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _promote_content_addressed(
    temporary: Path, directory: Path, stem: str, extension: str
) -> tuple[Path, str]:
    digest = _sha256_file(temporary)
    destination = directory / f"{stem}-{digest[:20]}.{extension}"
    if destination.exists():
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    return destination, digest


class FeatureStore:
    """Content-addressed NPZ/NPY store with an atomic clip-level JSONL index."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        storage_format: str = "npz",
        compressed: bool = True,
        index_name: str = "index.jsonl",
    ) -> None:
        if storage_format not in {"npz", "npy"}:
            raise ValueError("storage_format must be 'npz' or 'npy'")
        self.root = Path(root).resolve()
        self.storage_format = storage_format
        self.compressed = compressed
        self.index_path = self.root / index_name
        self.blob_root = self.root / "blobs"
        self.blob_root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"feature path escapes store root: {relative_path}") from error
        return candidate

    def iter_records(self) -> Iterator[FeatureRecord]:
        if not self.index_path.exists():
            return
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield FeatureRecord.from_dict(json.loads(line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"invalid feature index row {line_number}: {self.index_path}"
                    ) from error

    def records(self) -> list[FeatureRecord]:
        return list(self.iter_records())

    def find(
        self,
        *,
        video_id: str | None = None,
        clip_id: str | None = None,
        encoder_fingerprint: str | None = None,
    ) -> list[FeatureRecord]:
        return [
            record
            for record in self.iter_records()
            if (video_id is None or record.video_id == video_id)
            and (clip_id is None or record.clip_id == clip_id)
            and (encoder_fingerprint is None or record.encoder_fingerprint == encoder_fingerprint)
        ]

    def _write_npz(
        self, directory: Path, stem: str, arrays: Mapping[str, np.ndarray]
    ) -> Mapping[str, ArrayReference]:
        temporary_path: Path | None = None
        try:
            fd, raw_path = tempfile.mkstemp(prefix=f".{stem}.", suffix=".tmp", dir=directory)
            os.close(fd)
            temporary_path = Path(raw_path)
            _save_temp_npz(temporary_path, arrays, compressed=self.compressed)
            destination, digest = _promote_content_addressed(temporary_path, directory, stem, "npz")
            temporary_path = None
            relative = destination.relative_to(self.root).as_posix()
            return {
                name: ArrayReference(
                    path=relative,
                    key=name,
                    shape=tuple(int(item) for item in array.shape),
                    dtype=array.dtype.str,
                    sha256=digest,
                    nbytes=int(array.nbytes),
                )
                for name, array in arrays.items()
            }
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _write_npy(
        self, directory: Path, stem: str, arrays: Mapping[str, np.ndarray]
    ) -> Mapping[str, ArrayReference]:
        references: dict[str, ArrayReference] = {}
        for name, array in arrays.items():
            temporary_path: Path | None = None
            try:
                fd, raw_path = tempfile.mkstemp(
                    prefix=f".{stem}.{name}.", suffix=".tmp", dir=directory
                )
                os.close(fd)
                temporary_path = Path(raw_path)
                _save_temp_array(temporary_path, array)
                destination, digest = _promote_content_addressed(
                    temporary_path, directory, f"{stem}.{name}", "npy"
                )
                temporary_path = None
                references[name] = ArrayReference(
                    path=destination.relative_to(self.root).as_posix(),
                    shape=tuple(int(item) for item in array.shape),
                    dtype=array.dtype.str,
                    sha256=digest,
                    nbytes=int(array.nbytes),
                )
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        return references

    def write(
        self,
        *,
        video_id: str,
        clip_id: str,
        clip_index: int,
        encoder_fingerprint: str,
        features: Any,
        start_s: float,
        end_s: float,
        frame_start: int | None = None,
        frame_end: int | None = None,
        timeline_start_s: Any | None = None,
        timeline_end_s: Any | None = None,
        timeline_valid: Any | None = None,
        source_frame_start: Any | None = None,
        source_frame_end: Any | None = None,
        pooled: Any | None = None,
        aux_arrays: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = True,
    ) -> FeatureRecord:
        """Persist one clip and atomically upsert its index row.

        ``features`` is normally ``[S,D]`` for one clip.  Token timelines and
        pooled embeddings remain binary arrays in the same bundle.  ``metadata``
        is checked explicitly and rejects NumPy/PyTorch tensors.
        """

        arrays: dict[str, np.ndarray] = {"features": _normalise_array(features, name="features")}
        optional_arrays = {
            "timeline_start_s": timeline_start_s,
            "timeline_end_s": timeline_end_s,
            "timeline_valid": timeline_valid,
            "source_frame_start": source_frame_start,
            "source_frame_end": source_frame_end,
            "pooled": pooled,
        }
        for name, value in optional_arrays.items():
            if value is not None:
                arrays[name] = _normalise_array(value, name=name)
        for raw_name, value in (aux_arrays or {}).items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", raw_name):
                raise ValueError(f"invalid auxiliary array name: {raw_name!r}")
            name = f"aux_{raw_name}"
            if name in arrays:
                raise ValueError(f"duplicate array name: {name}")
            arrays[name] = _normalise_array(value, name=name)
        if arrays["features"].ndim < 1:
            raise ValueError("features must have at least one dimension")

        clean_metadata = ensure_json_metadata(metadata or {})
        fingerprint_part = encoder_fingerprint.removeprefix("sha256:")[:16]
        video_part = _slug(video_id, fallback="video")
        clip_part = _slug(clip_id, fallback="clip")
        directory = self.blob_root / fingerprint_part / video_part
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{clip_index:08d}-{clip_part}"
        key = (encoder_fingerprint, video_id, clip_id)

        with _InterProcessLock(self.index_path):
            current = list(self.iter_records())
            existing = [
                record
                for record in current
                if (record.encoder_fingerprint, record.video_id, record.clip_id) == key
            ]
            if existing and not overwrite:
                raise FileExistsError(f"feature record already exists: {video_id}/{clip_id}")
            references = (
                self._write_npz(directory, stem, arrays)
                if self.storage_format == "npz"
                else self._write_npy(directory, stem, arrays)
            )
            record = FeatureRecord(
                video_id=video_id,
                clip_id=clip_id,
                clip_index=clip_index,
                encoder_fingerprint=encoder_fingerprint,
                storage_format=self.storage_format,
                arrays=references,
                start_s=float(start_s),
                end_s=float(end_s),
                frame_start=frame_start,
                frame_end=frame_end,
                metadata=clean_metadata,
            )
            updated = [
                item
                for item in current
                if (item.encoder_fingerprint, item.video_id, item.clip_id) != key
            ]
            updated.append(record)
            updated.sort(
                key=lambda item: (
                    item.encoder_fingerprint,
                    item.video_id,
                    item.clip_index,
                    item.clip_id,
                )
            )
            atomic_write_jsonl(self.index_path, (item.to_dict() for item in updated))
            return record

    # Compatibility name used by extraction pipelines.
    put = write

    def _load_npy(self, reference: ArrayReference, mmap_mode: str | None) -> np.ndarray:
        path = self._resolve(reference.path)
        if _sha256_file(path) != reference.sha256:
            raise OSError(f"feature array checksum mismatch: {path}")
        return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)

    def load_bundle(
        self, record: FeatureRecord | Mapping[str, Any], *, mmap_mode: str | None = None
    ) -> dict[str, np.ndarray]:
        if not isinstance(record, FeatureRecord):
            record = FeatureRecord.from_dict(record)
        if record.storage_format == "npz":
            paths = {reference.path for reference in record.arrays.values()}
            if len(paths) != 1:
                raise ValueError("NPZ record must reference exactly one bundle path")
            path = self._resolve(next(iter(paths)))
            expected_digest = next(iter(record.arrays.values())).sha256
            if _sha256_file(path) != expected_digest:
                raise OSError(f"feature bundle checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as bundle:
                return {
                    name: np.asarray(bundle[reference.key or name])
                    for name, reference in record.arrays.items()
                }

        loaded: dict[str, np.ndarray] = {}
        for name, reference in record.arrays.items():
            loaded[name] = self._load_npy(reference, mmap_mode)
        return loaded

    def load_array(
        self,
        record: FeatureRecord | Mapping[str, Any],
        name: str = "features",
        *,
        mmap_mode: str | None = None,
    ) -> np.ndarray:
        if not isinstance(record, FeatureRecord):
            record = FeatureRecord.from_dict(record)
        if name not in record.arrays:
            raise KeyError(name)
        if record.storage_format == "npy":
            return self._load_npy(record.arrays[name], mmap_mode)
        return self.load_bundle(record)[name]

    def load_record(
        self,
        *,
        video_id: str,
        clip_id: str,
        encoder_fingerprint: str | None = None,
    ) -> tuple[FeatureRecord, np.ndarray]:
        matches = self.find(
            video_id=video_id,
            clip_id=clip_id,
            encoder_fingerprint=encoder_fingerprint,
        )
        if not matches:
            raise KeyError(f"feature record not found: {video_id}/{clip_id}")
        if len(matches) > 1:
            raise ValueError("multiple encoders match; pass encoder_fingerprint explicitly")
        record = matches[0]
        return record, self.load_array(record)


__all__ = [
    "ArrayReference",
    "FEATURE_INDEX_SCHEMA_VERSION",
    "FeatureRecord",
    "FeatureStore",
    "atomic_write_json",
    "atomic_write_jsonl",
    "compute_encoder_fingerprint",
    "encoder_fingerprint",
    "ensure_json_metadata",
    "utc_now_iso",
]
