"""Versioned JSON and NumPy sidecar protocol for isolated encoder workers.

The protocol is a dependency-isolation boundary, not a sandbox for untrusted
upstream code.  It deliberately rejects pickle, arbitrary import targets and
non-portable Python objects.  Fixed encoders consume one :class:`ClipBatch`;
streaming encoders consume all chunks in one worker process so ``opaque`` model
state never crosses the process boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np

from vadbench.contracts import (
    CacheUpdate,
    CacheView,
    ClipBatch,
    EncoderOutput,
    StreamState,
    StreamStep,
    TokenTimeline,
)
from vadbench.hashing import sha256_file as _sha256_file
from vadbench.integrations.common import validate_output_health

PROTOCOL_NAME = "vadbench.external_python"
PROTOCOL_VERSION = 1
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ENCODER_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
_NPZ_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class WorkerProtocolError(ValueError):
    """Raised when the external-worker wire contract is invalid."""


class SidecarIntegrityError(WorkerProtocolError):
    """Raised when an array sidecar differs from its declared identity."""


class RemoteWorkerError(RuntimeError):
    """Raised by a controller when a worker returned a structured error."""

    def __init__(self, error: WorkerErrorInfo) -> None:
        self.error = error
        super().__init__(f"{error.code} ({error.stage}/{error.exception_type}): {error.message}")


@dataclass(frozen=True, slots=True)
class ProtocolLimits:
    """Resource limits applied before loading sidecars into memory."""

    max_json_bytes: int = 16 * 1024 * 1024
    max_array_count: int = 8192
    max_array_bytes: int = 32 * 1024**3
    max_total_array_bytes: int = 64 * 1024**3
    max_ndim: int = 16
    max_header_bytes: int = 1024 * 1024
    max_json_depth: int = 32
    max_string_chars: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise WorkerProtocolError(f"{name} must be a positive integer")


DEFAULT_LIMITS = ProtocolLimits()


def _require_exact_fields(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise WorkerProtocolError(f"{name} fields mismatch: missing={missing}, extra={extra}")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkerProtocolError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise WorkerProtocolError(f"{name} keys must be strings")
    return value


def _require_non_empty_string(value: Any, *, name: str, limit: int = 2048) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or "\x00" in value:
        raise WorkerProtocolError(f"{name} must be a non-empty bounded string")
    return value


def _dataclass_payload(value: Any, /, **overrides: Any) -> dict[str, Any]:
    return {
        item.name: overrides.get(item.name, getattr(value, item.name)) for item in fields(value)
    }


def _dataclass_kwargs(cls: type[Any], value: Any, *, name: str) -> dict[str, Any]:
    value = _require_mapping(value, name=name)
    _require_exact_fields(value, {item.name for item in fields(cls)}, name=name)
    return dict(value)


def _validate_relative_posix_path(
    value: Any,
    *,
    name: str,
    required_prefix: str | None = None,
) -> str:
    text = _require_non_empty_string(value, name=name, limit=1024)
    if "\\" in text:
        raise WorkerProtocolError(f"{name} must use POSIX separators")
    raw_parts = text.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise WorkerProtocolError(f"{name} contains an empty/dot/traversal segment")
    path = PurePosixPath(text)
    if path.is_absolute() or ":" in raw_parts[0]:
        raise WorkerProtocolError(f"{name} must be relative and cannot contain a drive")
    if required_prefix is not None:
        prefix = PurePosixPath(
            _validate_relative_posix_path(required_prefix, name="required_prefix")
        )
        try:
            path.relative_to(prefix)
        except ValueError as exc:
            raise WorkerProtocolError(
                f"{name} must stay below {prefix.as_posix()!r}: {text!r}"
            ) from exc
        if path == prefix:
            raise WorkerProtocolError(f"{name} must name a file below {prefix.as_posix()!r}")
    return path.as_posix()


def ensure_json_value(
    value: Any,
    *,
    name: str = "value",
    limits: ProtocolLimits = DEFAULT_LIMITS,
    _depth: int = 0,
) -> Any:
    """Return a strict JSON-safe copy without stringifying arbitrary objects."""

    if _depth > limits.max_json_depth:
        raise WorkerProtocolError(f"{name} exceeds maximum JSON depth")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkerProtocolError(f"{name} contains NaN/Inf")
        return value
    if isinstance(value, str):
        if len(value) > limits.max_string_chars or "\x00" in value:
            raise WorkerProtocolError(f"{name} contains an invalid/oversized string")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 512 or "\x00" in key:
                raise WorkerProtocolError(f"{name} contains an invalid mapping key")
            result[key] = ensure_json_value(
                item,
                name=f"{name}.{key}",
                limits=limits,
                _depth=_depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            ensure_json_value(
                item,
                name=f"{name}[{index}]",
                limits=limits,
                _depth=_depth + 1,
            )
            for index, item in enumerate(value)
        ]
    raise WorkerProtocolError(
        f"{name} contains non-portable type {type(value).__module__}.{type(value).__name__}"
    )


def _freeze_json_mapping(
    value: Mapping[str, Any], *, name: str, limits: ProtocolLimits = DEFAULT_LIMITS
) -> Mapping[str, Any]:
    copied = ensure_json_value(value, name=name, limits=limits)
    return MappingProxyType(copied)


def _dtype(value: Any) -> np.dtype[Any]:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError(f"invalid NumPy dtype: {value!r}") from exc
    if dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in "biuf":
        raise WorkerProtocolError(f"unsupported portable dtype: {dtype}")
    return dtype


def _shape_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise WorkerProtocolError(f"{name} must be an array of dimensions")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise WorkerProtocolError(f"{name} contains an invalid dimension")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ArraySidecarRef:
    """Relative path and integrity identity for one NPY/NPZ array."""

    path: str
    format: str
    key: str | None
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    file_size: int
    sha256: str
    source_dtype: str | None = None

    def __post_init__(self) -> None:
        path = _validate_relative_posix_path(self.path, name="array.path")
        if not isinstance(self.format, str) or self.format not in {"npy", "npz"}:
            raise WorkerProtocolError("array.format must be 'npy' or 'npz'")
        if self.format == "npy":
            if self.key is not None or not path.endswith(".npy"):
                raise WorkerProtocolError("NPY references require .npy and key=null")
        elif (
            not path.endswith(".npz")
            or not isinstance(self.key, str)
            or _NPZ_KEY_RE.fullmatch(self.key) is None
        ):
            raise WorkerProtocolError("NPZ references require .npz and a safe key")
        shape = _shape_tuple(self.shape, name="array.shape")
        dtype = _dtype(self.dtype)
        if isinstance(self.nbytes, bool) or not isinstance(self.nbytes, int) or self.nbytes < 0:
            raise WorkerProtocolError("array.nbytes must be a non-negative integer")
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if self.nbytes != expected_nbytes:
            raise WorkerProtocolError(
                f"array.nbytes mismatch: declared={self.nbytes}, expected={expected_nbytes}"
            )
        if (
            isinstance(self.file_size, bool)
            or not isinstance(self.file_size, int)
            or self.file_size <= 0
        ):
            raise WorkerProtocolError("array.file_size must be a positive integer")
        if not isinstance(self.sha256, str) or _SHA256_RE.fullmatch(self.sha256) is None:
            raise WorkerProtocolError("array.sha256 must be 64 lowercase hexadecimal characters")
        if self.source_dtype is not None and (
            not isinstance(self.source_dtype, str) or not self.source_dtype
        ):
            raise WorkerProtocolError("array.source_dtype must be null or a non-empty string")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.str)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_payload(self, shape=list(self.shape))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArraySidecarRef:
        return cls(**_dataclass_kwargs(cls, value, name="array reference"))


def _sha256_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _validate_npy_header(
    handle: Any,
    reference: ArraySidecarRef,
    *,
    member_size: int,
    limits: ProtocolLimits,
) -> None:
    """Inspect a NPY header before NumPy can allocate its declared shape."""

    try:
        version = np.lib.format.read_magic(handle)
        reader = {
            (1, 0): np.lib.format.read_array_header_1_0,
            (2, 0): np.lib.format.read_array_header_2_0,
            (3, 0): np.lib.format.read_array_header_2_0,
        }[version]
        shape, _fortran_order, dtype_value = reader(handle, max_header_size=limits.max_header_bytes)
        actual_shape = tuple(int(item) for item in shape)
        actual_dtype = _dtype(dtype_value)
    except (EOFError, KeyError, TypeError, ValueError) as exc:
        raise SidecarIntegrityError(f"invalid NPY header: {reference.path}") from exc
    if len(actual_shape) > limits.max_ndim or any(item < 0 for item in actual_shape):
        raise SidecarIntegrityError("NPY header contains an invalid/oversized rank")
    actual_nbytes = math.prod(actual_shape) * actual_dtype.itemsize
    if actual_nbytes > limits.max_array_bytes:
        raise SidecarIntegrityError("NPY header array bytes exceed protocol limit")
    if actual_shape != reference.shape:
        raise SidecarIntegrityError(
            f"sidecar shape mismatch for {reference.path}: "
            f"expected={reference.shape}, actual={actual_shape}"
        )
    if actual_dtype.str != reference.dtype:
        raise SidecarIntegrityError(
            f"sidecar dtype mismatch for {reference.path}: "
            f"expected={reference.dtype}, actual={actual_dtype.str}"
        )
    if actual_nbytes != reference.nbytes:
        raise SidecarIntegrityError(f"sidecar nbytes mismatch: {reference.path}")
    expected_member_size = int(handle.tell()) + actual_nbytes
    if member_size != expected_member_size:
        raise SidecarIntegrityError(
            f"NPY payload size mismatch for {reference.path}: "
            f"expected={expected_member_size}, actual={member_size}"
        )


def _portable_array(value: Any, *, name: str) -> tuple[np.ndarray, str | None]:
    source_dtype: str | None = None
    if type(value).__module__.split(".", 1)[0] == "torch" and hasattr(value, "detach"):
        source_dtype = str(getattr(value, "dtype", "unknown"))
        value = value.detach().cpu()
        if source_dtype == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise WorkerProtocolError(f"{name} cannot be converted to NumPy") from exc
    _dtype(array.dtype)
    return np.ascontiguousarray(array), source_dtype


class SidecarStore:
    """Fail-closed array and JSON storage rooted at one exchange directory."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        create: bool = False,
        limits: ProtocolLimits = DEFAULT_LIMITS,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.limits = limits
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise WorkerProtocolError(f"exchange root is not a directory: {self.root}")
        self._array_count = 0
        self._total_array_bytes = 0

    def _read_path(self, relative_path: str, *, required_prefix: str | None = None) -> Path:
        relative = _validate_relative_posix_path(
            relative_path,
            name="sidecar path",
            required_prefix=required_prefix,
        )
        cursor = self.root
        for part in PurePosixPath(relative).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise WorkerProtocolError(f"sidecar path cannot contain symlinks: {relative}")
        try:
            resolved = cursor.resolve(strict=True)
            resolved.relative_to(self.root)
        except (FileNotFoundError, ValueError) as exc:
            raise WorkerProtocolError(
                f"sidecar path is missing or escapes root: {relative}"
            ) from exc
        mode = resolved.stat().st_mode
        if not stat.S_ISREG(mode):
            raise WorkerProtocolError(f"sidecar path is not a regular file: {relative}")
        return resolved

    def _write_path(self, relative_path: str, *, required_prefix: str | None = None) -> Path:
        relative = _validate_relative_posix_path(
            relative_path,
            name="sidecar path",
            required_prefix=required_prefix,
        )
        path = self.root.joinpath(*PurePosixPath(relative).parts)
        try:
            path.parent.resolve().relative_to(self.root)
        except ValueError as exc:
            raise WorkerProtocolError(f"sidecar parent escapes root: {relative}") from exc
        cursor = self.root
        for part in PurePosixPath(relative).parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink():
                raise WorkerProtocolError(f"sidecar parent cannot be a symlink: {relative}")
            cursor.mkdir(exist_ok=True)
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite sidecar: {relative}")
        return path

    @staticmethod
    def _promote_temp(temporary: Path, destination: Path) -> None:
        if os.path.lexists(destination):
            raise FileExistsError(f"refusing to overwrite sidecar: {destination}")
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)

    def write_array(self, relative_path: str, value: Any) -> ArraySidecarRef:
        path = self._write_path(relative_path)
        if path.suffix != ".npy":
            raise WorkerProtocolError("write_array requires a .npy destination")
        array, source_dtype = _portable_array(value, name=relative_path)
        self._account_array(array)
        temporary: Path | None = None
        try:
            descriptor_id, raw_path = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            os.close(descriptor_id)
            temporary = Path(raw_path)
            with temporary.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            file_size = temporary.stat().st_size
            digest = _sha256_file(temporary)
            self._promote_temp(temporary, path)
            temporary = None
            return ArraySidecarRef(
                path=path.relative_to(self.root).as_posix(),
                format="npy",
                key=None,
                shape=tuple(array.shape),
                dtype=array.dtype.str,
                nbytes=int(array.nbytes),
                file_size=file_size,
                sha256=digest,
                source_dtype=source_dtype,
            )
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def write_npz(
        self, relative_path: str, arrays: Mapping[str, Any]
    ) -> Mapping[str, ArraySidecarRef]:
        path = self._write_path(relative_path)
        if path.suffix != ".npz" or not arrays:
            raise WorkerProtocolError("write_npz requires a .npz destination and non-empty arrays")
        portable: dict[str, np.ndarray] = {}
        source_dtypes: dict[str, str | None] = {}
        for key, value in arrays.items():
            if not isinstance(key, str) or _NPZ_KEY_RE.fullmatch(key) is None:
                raise WorkerProtocolError(f"unsafe NPZ key: {key!r}")
            array, source_dtype = _portable_array(value, name=f"{relative_path}:{key}")
            self._account_array(array)
            portable[key] = array
            source_dtypes[key] = source_dtype
        temporary: Path | None = None
        try:
            descriptor_id, raw_path = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            os.close(descriptor_id)
            temporary = Path(raw_path)
            with temporary.open("wb") as handle:
                np.savez(handle, **portable)
                handle.flush()
                os.fsync(handle.fileno())
            with zipfile.ZipFile(temporary) as archive:
                infos = archive.infolist()
                if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
                    raise WorkerProtocolError("NPZ sidecars must use ZIP_STORED")
            file_size = temporary.stat().st_size
            digest = _sha256_file(temporary)
            self._promote_temp(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        relative = path.relative_to(self.root).as_posix()
        return MappingProxyType(
            {
                key: ArraySidecarRef(
                    path=relative,
                    format="npz",
                    key=key,
                    shape=tuple(array.shape),
                    dtype=array.dtype.str,
                    nbytes=int(array.nbytes),
                    file_size=file_size,
                    sha256=digest,
                    source_dtype=source_dtypes[key],
                )
                for key, array in portable.items()
            }
        )

    def _account_array(self, array: np.ndarray | ArraySidecarRef) -> None:
        ndim = array.ndim if isinstance(array, np.ndarray) else len(array.shape)
        nbytes = int(array.nbytes)
        if ndim > self.limits.max_ndim:
            raise WorkerProtocolError(f"array rank {ndim} exceeds limit")
        if nbytes > self.limits.max_array_bytes:
            raise WorkerProtocolError(f"array bytes {nbytes} exceed per-array limit")
        if self._array_count + 1 > self.limits.max_array_count:
            raise WorkerProtocolError("array count exceeds protocol limit")
        if self._total_array_bytes + nbytes > self.limits.max_total_array_bytes:
            raise WorkerProtocolError("total array bytes exceed protocol limit")
        self._array_count += 1
        self._total_array_bytes += nbytes

    def load_array(self, reference: ArraySidecarRef, *, required_prefix: str) -> np.ndarray:
        self._account_array(reference)
        max_file_bytes = (
            self.limits.max_total_array_bytes
            + self.limits.max_header_bytes * self.limits.max_array_count
        )
        if reference.file_size > max_file_bytes:
            raise WorkerProtocolError("sidecar file size exceeds protocol limit")
        path = self._read_path(reference.path, required_prefix=required_prefix)
        actual_size = path.stat().st_size
        if actual_size != reference.file_size:
            raise SidecarIntegrityError(
                f"sidecar file_size mismatch for {reference.path}: "
                f"expected={reference.file_size}, actual={actual_size}"
            )
        try:
            with path.open("rb") as handle:
                digest = _sha256_handle(handle)
                if digest != reference.sha256:
                    raise SidecarIntegrityError(f"sidecar SHA256 mismatch: {reference.path}")
                handle.seek(0)
                if reference.format == "npy":
                    _validate_npy_header(
                        handle,
                        reference,
                        member_size=actual_size,
                        limits=self.limits,
                    )
                    handle.seek(0)
                    loaded = np.load(
                        handle,
                        allow_pickle=False,
                        max_header_size=self.limits.max_header_bytes,
                    )
                    if not isinstance(loaded, np.ndarray):
                        raise SidecarIntegrityError("NPY sidecar did not contain an ndarray")
                    array = np.asarray(loaded)
                else:
                    with zipfile.ZipFile(handle) as archive:
                        infos = archive.infolist()
                        names = [info.filename for info in infos]
                        if not infos or len(infos) > self.limits.max_array_count:
                            raise SidecarIntegrityError("NPZ member count exceeds protocol limit")
                        if len(names) != len(set(names)):
                            raise SidecarIntegrityError("NPZ contains duplicate members")
                        if any(
                            info.is_dir()
                            or info.compress_type != zipfile.ZIP_STORED
                            or "/" in info.filename
                            or "\\" in info.filename
                            or not info.filename.endswith(".npy")
                            or info.file_size
                            > self.limits.max_array_bytes + self.limits.max_header_bytes
                            for info in infos
                        ):
                            raise SidecarIntegrityError("NPZ contains unsafe members")
                        expected_member = f"{reference.key}.npy"
                        if expected_member not in names:
                            raise SidecarIntegrityError(f"NPZ member is missing: {reference.key!r}")
                        info = archive.getinfo(expected_member)
                        with archive.open(info) as member:
                            _validate_npy_header(
                                member,
                                reference,
                                member_size=info.file_size,
                                limits=self.limits,
                            )
                    handle.seek(0)
                    with np.load(
                        handle,
                        allow_pickle=False,
                        max_header_size=self.limits.max_header_bytes,
                    ) as bundle:
                        array = np.asarray(bundle[reference.key])
        except (EOFError, OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            if isinstance(exc, SidecarIntegrityError):
                raise
            raise SidecarIntegrityError(f"cannot safely load sidecar: {reference.path}") from exc

        if tuple(array.shape) != reference.shape:
            raise SidecarIntegrityError(
                f"sidecar shape mismatch for {reference.path}: "
                f"expected={reference.shape}, actual={tuple(array.shape)}"
            )
        if array.dtype.str != reference.dtype:
            raise SidecarIntegrityError(
                f"sidecar dtype mismatch for {reference.path}: "
                f"expected={reference.dtype}, actual={array.dtype.str}"
            )
        if int(array.nbytes) != reference.nbytes:
            raise SidecarIntegrityError(f"sidecar nbytes mismatch: {reference.path}")
        _dtype(array.dtype)
        return np.ascontiguousarray(array)

    def write_json(self, relative_path: str, payload: Mapping[str, Any]) -> Path:
        path = self._write_path(relative_path)
        value = ensure_json_value(payload, name="JSON payload", limits=self.limits)
        encoded = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > self.limits.max_json_bytes:
            raise WorkerProtocolError("JSON payload exceeds protocol size limit")
        descriptor_id, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor_id)
        temporary = Path(raw_path)
        try:
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._promote_temp(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def read_json(self, relative_path: str) -> Mapping[str, Any]:
        path = self._read_path(relative_path)
        if path.stat().st_size > self.limits.max_json_bytes:
            raise WorkerProtocolError("JSON payload exceeds protocol size limit")

        def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise WorkerProtocolError(f"duplicate JSON key: {key!r}")
                result[key] = item
            return result

        def reject_constant(token: str) -> None:
            raise WorkerProtocolError(f"non-finite JSON number: {token}")

        try:
            payload = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=object_pairs,
                parse_constant=reject_constant,
            )
        except WorkerProtocolError:
            raise
        except (RecursionError, UnicodeDecodeError, ValueError) as exc:
            raise WorkerProtocolError(f"invalid UTF-8 JSON: {relative_path}") from exc
        result = ensure_json_value(payload, name="JSON payload", limits=self.limits)
        return _require_mapping(result, name="JSON payload")


def _ref_payload(reference: ArraySidecarRef | None) -> dict[str, Any] | None:
    return None if reference is None else reference.to_dict()


def _load_ref(
    value: Any,
    store: SidecarStore,
    *,
    required_prefix: str,
    name: str,
) -> np.ndarray:
    mapping = _require_mapping(value, name=name)
    return store.load_array(ArraySidecarRef.from_dict(mapping), required_prefix=required_prefix)


def serialize_clip_batch(batch: ClipBatch, store: SidecarStore, *, prefix: str) -> dict[str, Any]:
    prefix = _validate_relative_posix_path(prefix, name="clip prefix", required_prefix="input")
    return {
        "frames": store.write_array(f"{prefix}/frames.npy", batch.frames).to_dict(),
        "timestamps_s": store.write_array(
            f"{prefix}/timestamps_s.npy", batch.timestamps_s
        ).to_dict(),
        "video_ids": list(batch.video_ids),
        "valid_mask": _ref_payload(
            None
            if batch.valid_mask is None
            else store.write_array(f"{prefix}/valid_mask.npy", batch.valid_mask)
        ),
        "frame_indices": _ref_payload(
            None
            if batch.frame_indices is None
            else store.write_array(f"{prefix}/frame_indices.npy", batch.frame_indices)
        ),
        "metadata": ensure_json_value(batch.metadata, name="ClipBatch.metadata"),
    }


def deserialize_clip_batch(
    payload: Mapping[str, Any], store: SidecarStore, *, required_prefix: str
) -> ClipBatch:
    payload = _require_mapping(payload, name="ClipBatch payload")
    _require_exact_fields(
        payload,
        {"frames", "timestamps_s", "video_ids", "valid_mask", "frame_indices", "metadata"},
        name="ClipBatch payload",
    )
    video_ids = payload["video_ids"]
    if not isinstance(video_ids, list):
        raise WorkerProtocolError("ClipBatch.video_ids must be an array")
    valid = payload["valid_mask"]
    frame_indices = payload["frame_indices"]
    metadata = _require_mapping(payload["metadata"], name="ClipBatch.metadata")
    return ClipBatch(
        frames=_load_ref(payload["frames"], store, required_prefix=required_prefix, name="frames"),
        timestamps_s=_load_ref(
            payload["timestamps_s"],
            store,
            required_prefix=required_prefix,
            name="timestamps_s",
        ),
        video_ids=tuple(video_ids),
        valid_mask=(
            None
            if valid is None
            else _load_ref(valid, store, required_prefix=required_prefix, name="valid_mask")
        ),
        frame_indices=(
            None
            if frame_indices is None
            else _load_ref(
                frame_indices,
                store,
                required_prefix=required_prefix,
                name="frame_indices",
            )
        ),
        metadata=ensure_json_value(metadata, name="ClipBatch.metadata"),
    )


def _serialize_timeline(
    timeline: TokenTimeline, store: SidecarStore, *, prefix: str
) -> dict[str, Any]:
    return {
        "start_s": store.write_array(f"{prefix}/timeline_start_s.npy", timeline.start_s).to_dict(),
        "end_s": store.write_array(f"{prefix}/timeline_end_s.npy", timeline.end_s).to_dict(),
        "valid_mask": _ref_payload(
            None
            if timeline.valid_mask is None
            else store.write_array(f"{prefix}/timeline_valid_mask.npy", timeline.valid_mask)
        ),
        "source_frame_start": _ref_payload(
            None
            if timeline.source_frame_start is None
            else store.write_array(
                f"{prefix}/timeline_source_frame_start.npy", timeline.source_frame_start
            )
        ),
        "source_frame_end": _ref_payload(
            None
            if timeline.source_frame_end is None
            else store.write_array(
                f"{prefix}/timeline_source_frame_end.npy", timeline.source_frame_end
            )
        ),
    }


def _deserialize_timeline(
    payload: Mapping[str, Any], store: SidecarStore, *, required_prefix: str
) -> TokenTimeline:
    payload = _require_mapping(payload, name="TokenTimeline payload")
    _require_exact_fields(
        payload,
        {"start_s", "end_s", "valid_mask", "source_frame_start", "source_frame_end"},
        name="TokenTimeline payload",
    )

    def optional(name: str) -> np.ndarray | None:
        value = payload[name]
        return (
            None
            if value is None
            else _load_ref(value, store, required_prefix=required_prefix, name=name)
        )

    return TokenTimeline(
        start_s=_load_ref(
            payload["start_s"], store, required_prefix=required_prefix, name="start_s"
        ),
        end_s=_load_ref(payload["end_s"], store, required_prefix=required_prefix, name="end_s"),
        valid_mask=optional("valid_mask"),
        source_frame_start=optional("source_frame_start"),
        source_frame_end=optional("source_frame_end"),
    )


def serialize_encoder_output(
    output: EncoderOutput, store: SidecarStore, *, prefix: str
) -> dict[str, Any]:
    prefix = _validate_relative_posix_path(prefix, name="output prefix", required_prefix="output")
    return {
        "features": store.write_array(f"{prefix}/features.npy", output.features).to_dict(),
        "pooled": _ref_payload(
            None
            if output.pooled is None
            else store.write_array(f"{prefix}/pooled.npy", output.pooled)
        ),
        "timeline": _serialize_timeline(output.timeline, store, prefix=prefix),
        "aux": ensure_json_value(output.aux, name="EncoderOutput.aux"),
    }


def deserialize_encoder_output(
    payload: Mapping[str, Any], store: SidecarStore, *, required_prefix: str
) -> EncoderOutput:
    payload = _require_mapping(payload, name="EncoderOutput payload")
    _require_exact_fields(
        payload, {"features", "pooled", "timeline", "aux"}, name="EncoderOutput payload"
    )
    pooled = payload["pooled"]
    return EncoderOutput(
        features=_load_ref(
            payload["features"], store, required_prefix=required_prefix, name="features"
        ),
        pooled=(
            None
            if pooled is None
            else _load_ref(pooled, store, required_prefix=required_prefix, name="pooled")
        ),
        timeline=_deserialize_timeline(
            _require_mapping(payload["timeline"], name="timeline"),
            store,
            required_prefix=required_prefix,
        ),
        aux=ensure_json_value(
            _require_mapping(payload["aux"], name="EncoderOutput.aux"),
            name="EncoderOutput.aux",
        ),
    )


def _array_shape(value: Any) -> list[int]:
    return [int(item) for item in value.shape]


def _array_dtype(value: Any) -> str:
    return str(getattr(value, "dtype", "unknown"))


def _timeline_summary(timeline: TokenTimeline) -> dict[str, Any]:
    starts, _ = _portable_array(timeline.start_s, name="timeline.start_s")
    ends, _ = _portable_array(timeline.end_s, name="timeline.end_s")
    valid = (
        np.ones(starts.shape, dtype=bool)
        if timeline.valid_mask is None
        else _portable_array(timeline.valid_mask, name="timeline.valid_mask")[0]
    )
    return {
        "batch_size": timeline.batch_size,
        "num_tokens": timeline.num_tokens,
        "start_s_min": float(starts[valid].min()),
        "end_s_max": float(ends[valid].max()),
        "has_source_frames": timeline.source_frame_start is not None,
    }


def summarize_cache_view(view: CacheView) -> dict[str, Any]:
    return {
        "kind": view.kind.value,
        "sequence_axis": view.sequence_axis,
        "sequence_length": view.sequence_length,
        "tensor_count": len(view.tensors),
        "tensor_shapes": {name: _array_shape(value) for name, value in view.tensors.items()},
        "tensor_dtypes": {name: _array_dtype(value) for name, value in view.tensors.items()},
        "nbytes": view.nbytes,
        "timeline": _timeline_summary(view.timeline),
        "metadata": ensure_json_value(view.metadata, name="CacheView.metadata"),
    }


def summarize_stream_state(state: StreamState) -> dict[str, Any]:
    return {
        "video_id": state.video_id,
        "step_index": state.step_index,
        "next_timestamp_s": state.next_timestamp_s,
        "caches": {name: summarize_cache_view(view) for name, view in sorted(state.caches.items())},
        "opaque_present": state.opaque is not None,
        "metadata": ensure_json_value(state.metadata, name="StreamState.metadata"),
    }


def summarize_cache_update(update: CacheUpdate) -> dict[str, Any]:
    return {
        "mode": update.mode.value,
        "view": summarize_cache_view(update.view),
        "metadata": ensure_json_value(update.metadata, name="CacheUpdate.metadata"),
    }


@dataclass(frozen=True, slots=True)
class StreamStepRecord:
    """Portable observation of one step; model-private state is intentionally absent."""

    output: EncoderOutput | None
    state: Mapping[str, Any]
    cache_updates: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    final: bool

    def __post_init__(self) -> None:
        if self.output is not None and not isinstance(self.output, EncoderOutput):
            raise WorkerProtocolError("StreamStepRecord.output must be EncoderOutput or null")
        object.__setattr__(self, "state", _freeze_json_mapping(self.state, name="state"))
        object.__setattr__(
            self,
            "cache_updates",
            _freeze_json_mapping(self.cache_updates, name="cache_updates"),
        )
        object.__setattr__(
            self, "telemetry", _freeze_json_mapping(self.telemetry, name="telemetry")
        )
        if type(self.final) is not bool:
            raise WorkerProtocolError("StreamStepRecord.final must be bool")


@dataclass(frozen=True, slots=True)
class StreamWorkerResult:
    steps: tuple[StreamStepRecord, ...]
    final_output: EncoderOutput | None = None

    def __post_init__(self) -> None:
        if not self.steps or any(not isinstance(step, StreamStepRecord) for step in self.steps):
            raise WorkerProtocolError("stream result must contain StreamStepRecord entries")
        if self.final_output is not None and not isinstance(self.final_output, EncoderOutput):
            raise WorkerProtocolError("final_output must be EncoderOutput or null")


def serialize_stream_result(
    steps: Sequence[StreamStep],
    final_output: EncoderOutput | None,
    store: SidecarStore,
    *,
    prefix: str,
) -> dict[str, Any]:
    if not steps:
        raise WorkerProtocolError("stream result requires at least one step")
    serialized: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        step_prefix = f"{prefix}/step-{index:04d}"
        serialized.append(
            {
                "output": (
                    None
                    if step.output is None
                    else serialize_encoder_output(step.output, store, prefix=step_prefix)
                ),
                "state": summarize_stream_state(step.state),
                "cache_updates": {
                    name: summarize_cache_update(update)
                    for name, update in sorted(step.cache_updates.items())
                },
                "telemetry": ensure_json_value(step.telemetry, name="StreamStep.telemetry"),
                "final": step.final,
            }
        )
    return {
        "steps": serialized,
        "final_output": (
            None
            if final_output is None
            else serialize_encoder_output(final_output, store, prefix=f"{prefix}/final-output")
        ),
    }


def deserialize_stream_result(
    payload: Mapping[str, Any], store: SidecarStore, *, required_prefix: str
) -> StreamWorkerResult:
    payload = _require_mapping(payload, name="stream result")
    _require_exact_fields(payload, {"steps", "final_output"}, name="stream result")
    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise WorkerProtocolError("stream result.steps must be non-empty")
    steps: list[StreamStepRecord] = []
    for index, raw_step in enumerate(raw_steps):
        raw_step = _require_mapping(raw_step, name=f"stream result.steps[{index}]")
        _require_exact_fields(
            raw_step,
            {"output", "state", "cache_updates", "telemetry", "final"},
            name=f"stream result.steps[{index}]",
        )
        output_payload = raw_step["output"]
        steps.append(
            StreamStepRecord(
                output=(
                    None
                    if output_payload is None
                    else deserialize_encoder_output(
                        _require_mapping(output_payload, name="step.output"),
                        store,
                        required_prefix=required_prefix,
                    )
                ),
                state=_require_mapping(raw_step["state"], name="step.state"),
                cache_updates=_require_mapping(
                    raw_step["cache_updates"], name="step.cache_updates"
                ),
                telemetry=_require_mapping(raw_step["telemetry"], name="step.telemetry"),
                final=raw_step["final"],
            )
        )
    final_payload = payload["final_output"]
    return StreamWorkerResult(
        steps=tuple(steps),
        final_output=(
            None
            if final_payload is None
            else deserialize_encoder_output(
                _require_mapping(final_payload, name="final_output"),
                store,
                required_prefix=required_prefix,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    request_id: str
    encoder_id: str
    operation: str
    clips: tuple[Mapping[str, Any], ...]
    adapter_kwargs: Mapping[str, Any] = field(default_factory=dict)
    train: bool = False
    output_dir: str = "output"
    protocol: str = PROTOCOL_NAME
    version: int = PROTOCOL_VERSION
    kind: str = "request"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or _REQUEST_ID_RE.fullmatch(self.request_id) is None
        ):
            raise WorkerProtocolError("request_id is invalid")
        if (
            not isinstance(self.encoder_id, str)
            or _ENCODER_ID_RE.fullmatch(self.encoder_id) is None
        ):
            raise WorkerProtocolError("encoder_id is invalid")
        if not isinstance(self.operation, str) or self.operation not in {
            "encode",
            "encode_stream",
        }:
            raise WorkerProtocolError("operation must be encode or encode_stream")
        if (
            self.protocol != PROTOCOL_NAME
            or type(self.version) is not int
            or self.version != PROTOCOL_VERSION
            or self.kind != "request"
        ):
            raise WorkerProtocolError("unsupported worker request protocol/version/kind")
        if (
            not isinstance(self.clips, (list, tuple))
            or not self.clips
            or any(not isinstance(item, Mapping) for item in self.clips)
        ):
            raise WorkerProtocolError("clips must be a non-empty array of objects")
        if self.operation == "encode" and len(self.clips) != 1:
            raise WorkerProtocolError("encode requires exactly one ClipBatch")
        if type(self.train) is not bool:
            raise WorkerProtocolError("train must be bool")
        output_dir = _validate_relative_posix_path(
            self.output_dir, name="output_dir", required_prefix="output"
        )
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(
            self,
            "adapter_kwargs",
            _freeze_json_mapping(
                _require_mapping(self.adapter_kwargs, name="adapter_kwargs"),
                name="adapter_kwargs",
            ),
        )
        object.__setattr__(
            self, "clips", tuple(MappingProxyType(dict(item)) for item in self.clips)
        )

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_payload(
            self,
            clips=[dict(item) for item in self.clips],
            adapter_kwargs=dict(self.adapter_kwargs),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerRequest:
        kwargs = _dataclass_kwargs(cls, value, name="worker request")
        if not isinstance(kwargs["clips"], list):
            raise WorkerProtocolError("clips must be an array")
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class WorkerErrorInfo:
    code: str
    stage: str
    exception_type: str
    message: str

    def __post_init__(self) -> None:
        for name in ("code", "stage", "exception_type", "message"):
            _require_non_empty_string(getattr(self, name), name=f"error.{name}", limit=2048)

    def to_dict(self) -> dict[str, str]:
        return _dataclass_payload(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerErrorInfo:
        return cls(**_dataclass_kwargs(cls, value, name="worker error"))


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    request_id: str
    status: str
    output_dir: str
    result_kind: str | None = None
    result: Mapping[str, Any] | None = None
    error: WorkerErrorInfo | None = None
    protocol: str = PROTOCOL_NAME
    version: int = PROTOCOL_VERSION
    kind: str = "response"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or _REQUEST_ID_RE.fullmatch(self.request_id) is None
        ):
            raise WorkerProtocolError("response request_id is invalid")
        if (
            self.protocol != PROTOCOL_NAME
            or type(self.version) is not int
            or self.version != PROTOCOL_VERSION
            or self.kind != "response"
        ):
            raise WorkerProtocolError("unsupported worker response protocol/version/kind")
        output_dir = _validate_relative_posix_path(
            self.output_dir, name="response.output_dir", required_prefix="output"
        )
        object.__setattr__(self, "output_dir", output_dir)
        if self.status == "ok":
            if not isinstance(self.result_kind, str) or self.result_kind not in {
                "encoder_output",
                "stream_result",
            }:
                raise WorkerProtocolError("ok response has invalid result_kind")
            if self.result is None or self.error is not None:
                raise WorkerProtocolError("ok response requires result and forbids error")
            object.__setattr__(
                self,
                "result",
                _freeze_json_mapping(
                    _require_mapping(self.result, name="response.result"),
                    name="response.result",
                ),
            )
        elif self.status == "error":
            if self.result_kind is not None or self.result is not None or self.error is None:
                raise WorkerProtocolError("error response requires only error")
            if not isinstance(self.error, WorkerErrorInfo):
                raise WorkerProtocolError("error response.error must be WorkerErrorInfo")
        else:
            raise WorkerProtocolError("response.status must be ok or error")

    @classmethod
    def success(
        cls,
        *,
        request_id: str,
        output_dir: str,
        result_kind: str,
        result: Mapping[str, Any],
    ) -> WorkerResponse:
        return cls(
            request_id=request_id,
            status="ok",
            output_dir=output_dir,
            result_kind=result_kind,
            result=result,
        )

    @classmethod
    def failure(cls, *, request_id: str, output_dir: str, error: WorkerErrorInfo) -> WorkerResponse:
        return cls(
            request_id=request_id,
            status="error",
            output_dir=output_dir,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_payload(
            self,
            result=None if self.result is None else dict(self.result),
            error=None if self.error is None else self.error.to_dict(),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerResponse:
        kwargs = _dataclass_kwargs(cls, value, name="worker response")
        if kwargs["error"] is not None:
            kwargs["error"] = WorkerErrorInfo.from_dict(kwargs["error"])
        return cls(**kwargs)

    def raise_for_error(self) -> None:
        if self.error is not None:
            raise RemoteWorkerError(self.error)


def write_worker_request(
    exchange_root: str | os.PathLike[str],
    request_path: str,
    *,
    request_id: str,
    encoder_id: str,
    operation: str,
    clips: Sequence[ClipBatch],
    adapter_kwargs: Mapping[str, Any] | None = None,
    train: bool = False,
    output_dir: str | None = None,
    limits: ProtocolLimits = DEFAULT_LIMITS,
) -> WorkerRequest:
    store = SidecarStore(exchange_root, create=True, limits=limits)
    input_prefix = f"input/{request_id}"
    payloads = tuple(
        serialize_clip_batch(batch, store, prefix=f"{input_prefix}/clip-{index:04d}")
        for index, batch in enumerate(clips)
    )
    request = WorkerRequest(
        request_id=request_id,
        encoder_id=encoder_id,
        operation=operation,
        clips=payloads,
        adapter_kwargs={} if adapter_kwargs is None else adapter_kwargs,
        train=train,
        output_dir=output_dir or f"output/{request_id}",
    )
    store.write_json(request_path, request.to_dict())
    return request


def read_worker_request(
    exchange_root: str | os.PathLike[str],
    request_path: str,
    *,
    limits: ProtocolLimits = DEFAULT_LIMITS,
) -> tuple[WorkerRequest, tuple[ClipBatch, ...]]:
    store = SidecarStore(exchange_root, limits=limits)
    request = WorkerRequest.from_dict(store.read_json(request_path))
    required_prefix = f"input/{request.request_id}"
    clips = tuple(
        deserialize_clip_batch(payload, store, required_prefix=required_prefix)
        for payload in request.clips
    )
    return request, clips


def read_worker_response(
    exchange_root: str | os.PathLike[str],
    response_path: str,
    *,
    expected_request_id: str | None = None,
    limits: ProtocolLimits = DEFAULT_LIMITS,
) -> tuple[WorkerResponse, EncoderOutput | StreamWorkerResult | None]:
    store = SidecarStore(exchange_root, limits=limits)
    response = WorkerResponse.from_dict(store.read_json(response_path))
    if expected_request_id is not None and response.request_id != expected_request_id:
        raise WorkerProtocolError(
            f"response request_id mismatch: {response.request_id!r} != {expected_request_id!r}"
        )
    if response.status == "error":
        return response, None
    if response.result_kind == "encoder_output":
        result: EncoderOutput | StreamWorkerResult = deserialize_encoder_output(
            response.result,
            store,
            required_prefix=response.output_dir,
        )
        validate_output_health(result)
    else:
        result = deserialize_stream_result(
            response.result,
            store,
            required_prefix=response.output_dir,
        )
        for step in result.steps:
            if step.output is not None:
                validate_output_health(step.output)
        if result.final_output is not None:
            validate_output_health(result.final_output)
    return response, result


__all__ = [
    "ArraySidecarRef",
    "DEFAULT_LIMITS",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "ProtocolLimits",
    "RemoteWorkerError",
    "SidecarIntegrityError",
    "SidecarStore",
    "StreamStepRecord",
    "StreamWorkerResult",
    "WorkerErrorInfo",
    "WorkerProtocolError",
    "WorkerRequest",
    "WorkerResponse",
    "deserialize_clip_batch",
    "deserialize_encoder_output",
    "deserialize_stream_result",
    "ensure_json_value",
    "read_worker_request",
    "read_worker_response",
    "serialize_clip_batch",
    "serialize_encoder_output",
    "serialize_stream_result",
    "summarize_cache_update",
    "summarize_cache_view",
    "summarize_stream_state",
    "write_worker_request",
]
