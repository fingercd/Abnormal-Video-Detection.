"""Run-level provenance, metric, prediction, and cache telemetry artifacts."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .features import (
    _InterProcessLock,
    atomic_write_json,
    atomic_write_jsonl,
    ensure_json_metadata,
    utc_now_iso,
)

RUN_SCHEMA_VERSION = "vadbench.run.v1"
METRICS_SCHEMA_VERSION = "vadbench.metrics.v1"
PREDICTION_SCHEMA_VERSION = "vadbench.prediction.v1"
CACHE_TELEMETRY_SCHEMA_VERSION = "vadbench.cache-telemetry.v1"

_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:password|passwd|secret|token|api[_-]?key|cookie|credentials?|authorization)$",
    re.I,
)


def _redact_secrets(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if _SECRET_KEY.search(str(key)) else _redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    return value


def _run_git(command: Sequence[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *command],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def collect_git_provenance(repository: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Collect commit/branch/dirty state without reading repository secrets."""

    cwd = Path(repository or Path.cwd()).resolve()
    commit = _run_git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"available": False}
    branch = _run_git(["branch", "--show-current"], cwd) or None
    status = _run_git(["status", "--porcelain", "--untracked-files=no"], cwd)
    code_digest = hashlib.sha256()
    for folder in ("src", "scripts"):
        for path in sorted((cwd / folder).rglob("*.py")):
            code_digest.update(path.relative_to(cwd).as_posix().encode("utf-8"))
            code_digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "source_sha256": code_digest.hexdigest(),
    }


def collect_runtime_provenance(
    packages: Iterable[str] = ("numpy", "torch", "transformers"),
) -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": versions,
    }


def new_run_id(prefix: str = "run") -> str:
    timestamp = utc_now_iso().replace("-", "").replace(":", "").replace(".", "")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def file_identity(path: str | Path, *, project_root: str | Path = ".") -> dict[str, Any]:
    """Keep content identity separate from the machine-specific location."""

    from .checkpoints import sha256_file

    source = Path(path).resolve()
    root = Path(project_root).resolve()
    logical_path = (
        source.relative_to(root).as_posix() if source.is_relative_to(root) else source.name
    )
    return {
        "path": logical_path,
        "location": str(source),
        "exists": source.is_file(),
        "sha256": sha256_file(source) if source.is_file() else None,
    }


@contextmanager
def record_stage(
    output_dir: str | Path,
    stage: str,
    *,
    config: Mapping[str, Any],
    inputs: Mapping[str, str | Path | None],
    project_root: str | Path = ".",
) -> Iterator[dict[str, Any]]:
    """Persist each attempt independently, including failures before model loading."""

    run = RunProvenance(
        run_id=new_run_id(stage),
        config=config,
        git=collect_git_provenance(project_root),
    ).to_dict()
    run.update(schema_version="vadbench.stage.v1", stage=stage, status="running")
    run["config_sha256"] = hashlib.sha256(
        json.dumps(run["config"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    destination = Path(output_dir) / "provenance" / "stages" / f"{run['run_id']}.json"
    atomic_write_json(destination, run)
    try:
        run["inputs"] = {
            key: file_identity(path, project_root=project_root)
            for key, path in inputs.items()
            if path is not None
        }
        yield run
    except BaseException as exc:
        run.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    else:
        run["status"] = "completed"
    finally:
        run["finished_at"] = utc_now_iso()
        atomic_write_json(destination, run)


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    command: tuple[str, ...] = field(default_factory=lambda: tuple(sys.argv))
    config: Mapping[str, Any] = field(default_factory=dict)
    dataset: Mapping[str, Any] = field(default_factory=dict)
    encoder_fingerprint: str | None = None
    git: Mapping[str, Any] = field(default_factory=collect_git_provenance)
    runtime: Mapping[str, Any] = field(default_factory=collect_runtime_provenance)
    inputs: Mapping[str, Any] = field(default_factory=dict)
    notes: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.schema_version != RUN_SCHEMA_VERSION:
            raise ValueError(f"unsupported run provenance schema: {self.schema_version}")
        if self.encoder_fingerprint is not None and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.encoder_fingerprint
        ):
            raise ValueError("encoder_fingerprint must use sha256:<64 lowercase hex chars>")
        object.__setattr__(self, "command", tuple(str(item) for item in self.command))
        object.__setattr__(self, "config", ensure_json_metadata(_redact_secrets(self.config)))
        object.__setattr__(self, "dataset", ensure_json_metadata(_redact_secrets(self.dataset)))
        object.__setattr__(self, "git", ensure_json_metadata(self.git))
        object.__setattr__(self, "runtime", ensure_json_metadata(self.runtime))
        object.__setattr__(self, "inputs", ensure_json_metadata(_redact_secrets(self.inputs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "command": list(self.command),
            "config": dict(self.config),
            "dataset": dict(self.dataset),
            "encoder_fingerprint": self.encoder_fingerprint,
            "git": dict(self.git),
            "runtime": dict(self.runtime),
            "inputs": dict(self.inputs),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunProvenance:
        return cls(
            schema_version=str(value.get("schema_version", RUN_SCHEMA_VERSION)),
            run_id=str(value["run_id"]),
            command=tuple(str(item) for item in value.get("command", ())),
            config=dict(value.get("config", {})),
            dataset=dict(value.get("dataset", {})),
            encoder_fingerprint=(
                str(value["encoder_fingerprint"])
                if value.get("encoder_fingerprint") is not None
                else None
            ),
            git=dict(value.get("git", {})),
            runtime=dict(value.get("runtime", {})),
            inputs=dict(value.get("inputs", {})),
            notes=str(value["notes"]) if value.get("notes") is not None else None,
            created_at=str(value.get("created_at", utc_now_iso())),
        )


LabelValue = int | str | bool


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    video_id: str
    clip_id: str
    clip_index: int
    start_s: float
    end_s: float
    anomaly_score: float
    frame_start: int | None = None
    frame_end: int | None = None
    predicted_label: LabelValue | None = None
    ground_truth: LabelValue | None = None
    encoder_fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = PREDICTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_SCHEMA_VERSION:
            raise ValueError(f"unsupported prediction schema: {self.schema_version}")
        if not self.run_id or not self.video_id or not self.clip_id:
            raise ValueError("run_id, video_id, and clip_id must be non-empty")
        if self.clip_index < 0:
            raise ValueError("clip_index must be non-negative")
        if (
            not math.isfinite(self.start_s)
            or not math.isfinite(self.end_s)
            or self.end_s < self.start_s
        ):
            raise ValueError("prediction interval must be finite and end_s >= start_s")
        if not math.isfinite(self.anomaly_score):
            raise ValueError("anomaly_score must be finite")
        if (self.frame_start is None) != (self.frame_end is None):
            raise ValueError("frame_start and frame_end must either both be set or both be null")
        if (
            self.frame_start is not None
            and self.frame_end is not None
            and (self.frame_start < 0 or self.frame_end < self.frame_start)
        ):
            raise ValueError("frame interval must satisfy 0 <= frame_start <= frame_end")
        if self.encoder_fingerprint is not None and not re.fullmatch(
            r"sha256:[0-9a-f]{64}", self.encoder_fingerprint
        ):
            raise ValueError("encoder_fingerprint must use sha256:<64 lowercase hex chars>")
        object.__setattr__(self, "metadata", ensure_json_metadata(self.metadata))

    @property
    def score(self) -> float:
        return self.anomaly_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "video_id": self.video_id,
            "clip_id": self.clip_id,
            "clip_index": self.clip_index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "anomaly_score": self.anomaly_score,
            "predicted_label": self.predicted_label,
            "ground_truth": self.ground_truth,
            "encoder_fingerprint": self.encoder_fingerprint,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PredictionRecord:
        score = value.get("anomaly_score", value.get("score"))
        if score is None:
            raise KeyError("anomaly_score")
        return cls(
            schema_version=str(value.get("schema_version", PREDICTION_SCHEMA_VERSION)),
            run_id=str(value["run_id"]),
            video_id=str(value["video_id"]),
            clip_id=str(value["clip_id"]),
            clip_index=int(value["clip_index"]),
            start_s=float(value["start_s"]),
            end_s=float(value["end_s"]),
            anomaly_score=float(score),
            frame_start=int(value["frame_start"]) if value.get("frame_start") is not None else None,
            frame_end=int(value["frame_end"]) if value.get("frame_end") is not None else None,
            predicted_label=value.get("predicted_label"),
            ground_truth=value.get("ground_truth"),
            encoder_fingerprint=(
                str(value["encoder_fingerprint"])
                if value.get("encoder_fingerprint") is not None
                else None
            ),
            metadata=dict(value.get("metadata", {})),
            created_at=str(value.get("created_at", utc_now_iso())),
        )


@dataclass(frozen=True)
class CacheTelemetryRecord:
    run_id: str
    encoder_fingerprint: str
    video_id: str
    clip_id: str
    mode: str
    cache_type: str
    cache_hit: bool
    input_tokens: int
    reused_tokens: int
    output_tokens: int
    cache_bytes: int
    encode_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    schema_version: str = CACHE_TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_TELEMETRY_SCHEMA_VERSION:
            raise ValueError(f"unsupported cache telemetry schema: {self.schema_version}")
        if not self.run_id or not self.video_id or not self.clip_id:
            raise ValueError("run_id, video_id, and clip_id must be non-empty")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.encoder_fingerprint):
            raise ValueError("encoder_fingerprint must use sha256:<64 lowercase hex chars>")
        if self.mode not in {"fixed", "streaming"}:
            raise ValueError("mode must be 'fixed' or 'streaming'")
        if self.cache_type not in {
            "none",
            "kv",
            "token",
            "visual_memory",
            "state",
            "hybrid",
        }:
            raise ValueError("unsupported cache_type")
        for name in ("input_tokens", "reused_tokens", "output_tokens", "cache_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.reused_tokens > self.input_tokens:
            raise ValueError("reused_tokens cannot exceed input_tokens")
        if not math.isfinite(self.encode_ms) or self.encode_ms < 0:
            raise ValueError("encode_ms must be finite and non-negative")
        object.__setattr__(self, "metadata", ensure_json_metadata(self.metadata))

    @property
    def reuse_ratio(self) -> float:
        return self.reused_tokens / self.input_tokens if self.input_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "encoder_fingerprint": self.encoder_fingerprint,
            "video_id": self.video_id,
            "clip_id": self.clip_id,
            "mode": self.mode,
            "cache_type": self.cache_type,
            "cache_hit": self.cache_hit,
            "input_tokens": self.input_tokens,
            "reused_tokens": self.reused_tokens,
            "output_tokens": self.output_tokens,
            "cache_bytes": self.cache_bytes,
            "encode_ms": self.encode_ms,
            "reuse_ratio": self.reuse_ratio,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CacheTelemetryRecord:
        return cls(
            schema_version=str(value.get("schema_version", CACHE_TELEMETRY_SCHEMA_VERSION)),
            run_id=str(value["run_id"]),
            encoder_fingerprint=str(value["encoder_fingerprint"]),
            video_id=str(value["video_id"]),
            clip_id=str(value["clip_id"]),
            mode=str(value["mode"]),
            cache_type=str(value["cache_type"]),
            cache_hit=bool(value["cache_hit"]),
            input_tokens=int(value["input_tokens"]),
            reused_tokens=int(value["reused_tokens"]),
            output_tokens=int(value["output_tokens"]),
            cache_bytes=int(value["cache_bytes"]),
            encode_ms=float(value["encode_ms"]),
            metadata=dict(value.get("metadata", {})),
            created_at=str(value.get("created_at", utc_now_iso())),
        )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {line_number}: {path}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            yield value


def _append_jsonl_row(path: Path, value: Mapping[str, Any]) -> None:
    """Durably append one complete JSONL row under the caller's writer lock.

    Full-file replacement is used when producing a snapshot.  Rewriting a
    growing predictions file for every clip, however, is quadratic, so event
    append uses one ``O_APPEND`` transaction and truncates back to the original
    size if a normal write error occurs.
    """

    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_APPEND | os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o666)
    original_size = os.fstat(descriptor).st_size
    try:
        if original_size:
            os.lseek(descriptor, original_size - 1, os.SEEK_SET)
            if os.read(descriptor, 1) != b"\n":
                raise ValueError(f"refusing to append to incomplete JSONL file: {path}")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"failed to append JSONL row: {path}")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.ftruncate(descriptor, original_size)
        os.fsync(descriptor)
        raise
    finally:
        os.close(descriptor)


class ArtifactStore:
    """Canonical directory layout for one reproducible benchmark run.

    Layout::

        <run>/provenance/run.json
        <run>/metrics/metrics.json
        <run>/metrics/history.jsonl
        <run>/predictions/predictions.jsonl
        <run>/cache_telemetry/events.jsonl
    """

    def __init__(self, run_dir: str | os.PathLike[str], *, run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.run_id = run_id or self.run_dir.name
        if (
            not self.run_id
            or self.run_id in {".", ".."}
            or any(separator in self.run_id for separator in ("/", "\\"))
        ):
            raise ValueError("run_id must be a single safe path component")
        self.provenance_dir = self.run_dir / "provenance"
        self.metrics_dir = self.run_dir / "metrics"
        self.predictions_dir = self.run_dir / "predictions"
        self.cache_telemetry_dir = self.run_dir / "cache_telemetry"
        self.provenance_path = self.provenance_dir / "run.json"
        self.metrics_path = self.metrics_dir / "metrics.json"
        self.metrics_history_path = self.metrics_dir / "history.jsonl"
        self.predictions_path = self.predictions_dir / "predictions.jsonl"
        self.cache_telemetry_path = self.cache_telemetry_dir / "events.jsonl"
        for directory in (
            self.provenance_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.cache_telemetry_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        base_dir: str | os.PathLike[str],
        *,
        run_id: str | None = None,
        provenance: RunProvenance | Mapping[str, Any] | None = None,
    ) -> ArtifactStore:
        resolved_run_id = (
            run_id
            or (provenance.run_id if isinstance(provenance, RunProvenance) else None)
            or new_run_id()
        )
        store = cls(Path(base_dir) / "runs" / resolved_run_id, run_id=resolved_run_id)
        if provenance is not None:
            store.write_provenance(provenance)
        return store

    def write_provenance(self, provenance: RunProvenance | Mapping[str, Any]) -> RunProvenance:
        if not isinstance(provenance, RunProvenance):
            provenance = RunProvenance.from_dict(provenance)
        if provenance.run_id != self.run_id:
            raise ValueError(
                f"provenance run_id {provenance.run_id!r} != store run_id {self.run_id!r}"
            )
        atomic_write_json(self.provenance_path, provenance.to_dict())
        return provenance

    def read_provenance(self) -> RunProvenance:
        with self.provenance_path.open("r", encoding="utf-8") as handle:
            return RunProvenance.from_dict(json.load(handle))

    def write_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        split: str | None = None,
        step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_metrics = ensure_json_metadata(metrics)
        clean_metadata = ensure_json_metadata(metadata or {})
        if step is not None and step < 0:
            raise ValueError("step must be non-negative")
        value = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "updated_at": utc_now_iso(),
            "split": split,
            "step": step,
            "metrics": clean_metrics,
            "metadata": clean_metadata,
        }
        atomic_write_json(self.metrics_path, value)
        return value

    def read_metrics(self) -> dict[str, Any]:
        with self.metrics_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if (
            value.get("schema_version") != METRICS_SCHEMA_VERSION
            or value.get("run_id") != self.run_id
        ):
            raise ValueError(
                f"metrics file does not belong to run {self.run_id}: {self.metrics_path}"
            )
        return value

    def append_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        split: str | None = None,
        step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an epoch/step metric event without discarding prior history."""

        if step is not None and step < 0:
            raise ValueError("step must be non-negative")
        value = {
            "schema_version": METRICS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": utc_now_iso(),
            "split": split,
            "step": step,
            "metrics": ensure_json_metadata(metrics),
            "metadata": ensure_json_metadata(metadata or {}),
        }
        with _InterProcessLock(self.metrics_history_path):
            _append_jsonl_row(self.metrics_history_path, value)
        return value

    def iter_metric_history(self) -> Iterator[dict[str, Any]]:
        for value in _read_jsonl(self.metrics_history_path):
            if (
                value.get("schema_version") != METRICS_SCHEMA_VERSION
                or value.get("run_id") != self.run_id
            ):
                raise ValueError(
                    f"metrics history row does not belong to run {self.run_id}: "
                    f"{self.metrics_history_path}"
                )
            yield value

    def write_predictions(
        self, records: Iterable[PredictionRecord | Mapping[str, Any]]
    ) -> list[PredictionRecord]:
        normalised: list[PredictionRecord] = []
        for item in records:
            record = (
                item if isinstance(item, PredictionRecord) else PredictionRecord.from_dict(item)
            )
            if record.run_id != self.run_id:
                raise ValueError(
                    f"prediction run_id {record.run_id!r} != store run_id {self.run_id!r}"
                )
            normalised.append(record)
        atomic_write_jsonl(self.predictions_path, (record.to_dict() for record in normalised))
        return normalised

    def append_prediction(self, record: PredictionRecord | Mapping[str, Any]) -> PredictionRecord:
        if not isinstance(record, PredictionRecord):
            record = PredictionRecord.from_dict(record)
        if record.run_id != self.run_id:
            raise ValueError(f"prediction run_id {record.run_id!r} != store run_id {self.run_id!r}")
        with _InterProcessLock(self.predictions_path):
            _append_jsonl_row(self.predictions_path, record.to_dict())
        return record

    def iter_predictions(self) -> Iterator[PredictionRecord]:
        for value in _read_jsonl(self.predictions_path):
            yield PredictionRecord.from_dict(value)

    def write_cache_telemetry(
        self, records: Iterable[CacheTelemetryRecord | Mapping[str, Any]]
    ) -> list[CacheTelemetryRecord]:
        normalised: list[CacheTelemetryRecord] = []
        for item in records:
            record = (
                item
                if isinstance(item, CacheTelemetryRecord)
                else CacheTelemetryRecord.from_dict(item)
            )
            if record.run_id != self.run_id:
                raise ValueError(
                    f"telemetry run_id {record.run_id!r} != store run_id {self.run_id!r}"
                )
            normalised.append(record)
        atomic_write_jsonl(self.cache_telemetry_path, (record.to_dict() for record in normalised))
        return normalised

    def append_cache_telemetry(
        self, record: CacheTelemetryRecord | Mapping[str, Any]
    ) -> CacheTelemetryRecord:
        if not isinstance(record, CacheTelemetryRecord):
            record = CacheTelemetryRecord.from_dict(record)
        if record.run_id != self.run_id:
            raise ValueError(f"telemetry run_id {record.run_id!r} != store run_id {self.run_id!r}")
        with _InterProcessLock(self.cache_telemetry_path):
            _append_jsonl_row(self.cache_telemetry_path, record.to_dict())
        return record

    def iter_cache_telemetry(self) -> Iterator[CacheTelemetryRecord]:
        for value in _read_jsonl(self.cache_telemetry_path):
            yield CacheTelemetryRecord.from_dict(value)


__all__ = [
    "ArtifactStore",
    "CACHE_TELEMETRY_SCHEMA_VERSION",
    "CacheTelemetryRecord",
    "METRICS_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "PredictionRecord",
    "RUN_SCHEMA_VERSION",
    "RunProvenance",
    "collect_git_provenance",
    "collect_runtime_provenance",
    "new_run_id",
]
