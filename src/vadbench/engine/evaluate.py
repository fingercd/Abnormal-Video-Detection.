"""Evaluation helpers for fixed-window and streaming VAD predictions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..data.audit import (
    OFFICIAL_UCF_CRIME_COUNTS,
    compute_manifest_sha256,
    verify_official_source_identity,
)
from ..data.labels import frame_labels_from_manifest
from ..data.manifest import (
    DatasetSplit,
    VideoManifestRecord,
    load_manifest_jsonl,
    validate_manifest,
)
from ..metrics import (
    UCFFrameMetrics,
    compute_ucf_frame_metrics,
    project_intervals_to_frames,
    resample_scores_to_frames,
)
from .coverage import aggregate_coverage, validate_frame_coverage

try:  # Model-free artifact evaluation works without PyTorch.
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in minimal environments.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class TemporalPrediction:
    """One video's temporal prediction with an optional explicit timeline."""

    video_id: str
    scores: Any
    intervals: Any | None = None
    fps: float | None = None
    frame_times: Any | None = None
    valid_mask: Any | None = None


@dataclass(frozen=True)
class UCFEvaluationResult:
    """Global UCF-Crime metrics plus the exact projected frame sequences."""

    metrics: UCFFrameMetrics
    frame_scores: dict[str, np.ndarray]
    protocol: str | None = None
    coverage: Mapping[str, Any] | None = None
    input_identity: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = self.metrics.to_dict()
        if self.protocol is not None:
            result["protocol"] = self.protocol
        if self.coverage is not None:
            result["coverage"] = dict(self.coverage)
        if self.input_identity is not None:
            result["input_identity"] = dict(self.input_identity)
        return result


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _to_numpy(value: Any) -> np.ndarray:
    if TORCH_AVAILABLE and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _prediction_parts(
    value: Any,
) -> tuple[np.ndarray, Any | None, float | None, Any | None, Any | None]:
    scores = _field(value, "scores")
    if scores is None:
        scores = _field(value, "frame_scores")
    if scores is None:
        scores = _field(value, "snippet_scores")
    if scores is None:
        predictions = _field(value, "predictions")
        if predictions is not None and predictions is not value:
            scores = _field(predictions, "snippet_scores", predictions)
    if scores is None:
        scores = value
    if TORCH_AVAILABLE and torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    score_array = np.asarray(scores)
    if score_array.ndim == 2 and score_array.shape[0] == 1:
        score_array = score_array[0]
    if score_array.ndim != 1 or score_array.size == 0:
        raise ValueError(
            f"each prediction must contain a non-empty 1D score sequence, got {score_array.shape}"
        )
    if not np.all(np.isfinite(score_array)):
        raise ValueError("prediction contains NaN or infinite scores")
    auxiliary = _field(value, "auxiliary")
    valid_mask = _field(value, "valid_mask")
    if valid_mask is None and auxiliary is not None:
        valid_mask = _field(auxiliary, "valid_mask")
    intervals = _field(value, "intervals")
    if intervals is None and auxiliary is not None:
        timeline = _field(auxiliary, "timeline")
        if timeline is not None:
            if valid_mask is None:
                valid_mask = _field(timeline, "valid_mask", _field(timeline, "valid"))
            frame_start = _field(timeline, "source_frame_start")
            frame_end = _field(timeline, "source_frame_end")
            if frame_start is not None and frame_end is not None:
                starts = _to_numpy(frame_start)
                ends = _to_numpy(frame_end)
            else:
                starts = _to_numpy(_field(timeline, "start_s", _field(timeline, "start_seconds")))
                ends = _to_numpy(_field(timeline, "end_s", _field(timeline, "end_seconds")))
            if starts.ndim == 2 and starts.shape[0] == 1:
                starts = starts[0]
                ends = ends[0]
            if starts.shape != score_array.shape or ends.shape != score_array.shape:
                raise ValueError("prediction timeline must match its score sequence")
            intervals = np.column_stack((starts, ends))
    return (
        score_array,
        intervals,
        _field(value, "fps"),
        _field(value, "frame_times"),
        valid_mask,
    )


def project_video_prediction(
    prediction: Any,
    num_frames: int,
    *,
    intervals: Any | None = None,
    fps: float | None = None,
    frame_times: Any | None = None,
    reduction: str = "max",
    fill_value: float = 0.0,
    allow_uniform_resample: bool = True,
) -> np.ndarray:
    """Convert one frame/snippet/interval prediction to exactly ``num_frames``."""

    scores, embedded_intervals, embedded_fps, embedded_times, valid_mask = _prediction_parts(
        prediction
    )
    intervals = embedded_intervals if intervals is None else intervals
    fps = embedded_fps if fps is None else fps
    frame_times = embedded_times if frame_times is None else frame_times
    if valid_mask is not None:
        if TORCH_AVAILABLE and torch.is_tensor(valid_mask):
            valid_mask = valid_mask.detach().cpu().numpy()
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.ndim == 2 and valid.shape[0] == 1:
            valid = valid[0]
        if valid.shape != scores.shape:
            raise ValueError(
                f"prediction valid_mask must have shape {scores.shape}, got {valid.shape}"
            )
        scores = scores[valid]
        if scores.size == 0:
            raise ValueError("prediction valid_mask selects no scores")
        if intervals is not None:
            interval_array = np.asarray(intervals)
            if interval_array.shape != (valid.size, 2):
                raise ValueError("prediction intervals must match the unmasked score sequence")
            intervals = interval_array[valid]
    if intervals is not None:
        return project_intervals_to_frames(
            intervals,
            scores,
            num_frames,
            fps=fps,
            frame_times=frame_times,
            reduction=reduction,  # type: ignore[arg-type]
            fill_value=fill_value,
        )
    if scores.shape[0] == num_frames:
        return scores.astype(np.float64, copy=False)
    if not allow_uniform_resample:
        raise ValueError(f"got {scores.shape[0]} scores for {num_frames} frames and no intervals")
    return resample_scores_to_frames(scores, num_frames).astype(np.float64, copy=False)


def evaluate_ucf_predictions(
    predictions: Mapping[str, Any],
    frame_labels: Mapping[str, Any],
    *,
    intervals: Mapping[str, Any] | None = None,
    fps: Mapping[str, float] | float | None = None,
    frame_times: Mapping[str, Any] | None = None,
    reduction: str = "max",
    fill_value: float = 0.0,
    allow_uniform_resample: bool = True,
    undefined: Literal["nan", "raise"] = "nan",
) -> UCFEvaluationResult:
    """Run the canonical global UCF-Crime frame ROC-AUC/AP evaluation.

    Predictions may already be frame aligned or may provide interval/snippet
    scores.  Explicit recorded intervals are preferred; uniform expansion is
    retained for legacy 32-segment UCF feature files.
    """

    if not predictions or not frame_labels:
        raise ValueError("predictions and frame_labels must not be empty")
    if set(predictions) != set(frame_labels):
        raise ValueError(
            "prediction/label video ids differ: "
            f"prediction_only={sorted(set(predictions) - set(frame_labels))}, "
            f"label_only={sorted(set(frame_labels) - set(predictions))}"
        )

    projected: dict[str, np.ndarray] = {}
    normalized_labels: dict[str, np.ndarray] = {}
    for video_id, labels_value in frame_labels.items():
        labels = np.asarray(labels_value).reshape(-1)
        if labels.size == 0:
            raise ValueError(f"video {video_id!r} has no frame labels")
        normalized_labels[video_id] = labels
        video_intervals = None if intervals is None else intervals.get(video_id)
        video_times = None if frame_times is None else frame_times.get(video_id)
        video_fps = fps.get(video_id) if isinstance(fps, Mapping) else fps
        projected[video_id] = project_video_prediction(
            predictions[video_id],
            labels.size,
            intervals=video_intervals,
            fps=video_fps,
            frame_times=video_times,
            reduction=reduction,
            fill_value=fill_value,
            allow_uniform_resample=allow_uniform_resample,
        )

    metrics = compute_ucf_frame_metrics(normalized_labels, projected, undefined=undefined)
    return UCFEvaluationResult(metrics=metrics, frame_scores=projected)


def prediction_records_to_temporal(
    records: Iterable[Any],
) -> tuple[dict[str, TemporalPrediction], str]:
    """Group artifact ``PredictionRecord`` objects into temporal sequences.

    The function is structural to avoid coupling the evaluator to one storage
    backend.  It prefers recorded half-open frame ranges when every record has
    them, otherwise it uses ``start_s/end_s`` and reports ``"seconds"`` so the
    caller can provide FPS or exact frame timestamps.
    """

    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        video_id = _field(record, "video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("every prediction record must have a non-empty video_id")
        score = _field(record, "anomaly_score", _field(record, "score"))
        if score is None or not np.isfinite(float(score)):
            raise ValueError(f"prediction record for {video_id!r} has no finite score")
        grouped[video_id].append(record)
    if not grouped:
        raise ValueError("prediction records must not be empty")

    use_frames = all(
        _field(record, "frame_start") is not None and _field(record, "frame_end") is not None
        for video_records in grouped.values()
        for record in video_records
    )
    predictions: dict[str, TemporalPrediction] = {}
    for video_id, video_records in grouped.items():
        ordered = sorted(
            video_records,
            key=lambda record: (
                int(_field(record, "clip_index", 0)),
                float(_field(record, "start_s", 0.0)),
            ),
        )
        scores = np.asarray(
            [float(_field(record, "anomaly_score", _field(record, "score"))) for record in ordered],
            dtype=np.float64,
        )
        if use_frames:
            intervals = np.asarray(
                [
                    [int(_field(record, "frame_start")), int(_field(record, "frame_end"))]
                    for record in ordered
                ],
                dtype=np.int64,
            )
        else:
            intervals = np.asarray(
                [
                    [float(_field(record, "start_s")), float(_field(record, "end_s"))]
                    for record in ordered
                ],
                dtype=np.float64,
            )
        predictions[video_id] = TemporalPrediction(
            video_id=video_id, scores=scores, intervals=intervals
        )
    return predictions, "frames" if use_frames else "seconds"


def evaluate_ucf_prediction_records(
    records: Iterable[Any],
    frame_labels: Mapping[str, Any],
    *,
    fps: Mapping[str, float] | float | None = None,
    frame_times: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> UCFEvaluationResult:
    """Evaluate JSONL-style prediction records emitted by ``ArtifactStore``."""

    predictions, interval_unit = prediction_records_to_temporal(records)
    if interval_unit == "seconds" and fps is None and frame_times is None:
        raise ValueError(
            "second-based prediction records require fps or exact frame_times for projection"
        )
    return evaluate_ucf_predictions(
        predictions,
        frame_labels,
        fps=fps if interval_unit == "seconds" else None,
        frame_times=frame_times if interval_unit == "seconds" else None,
        **kwargs,
    )


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _manifest_records(
    manifest: str | Path | Iterable[VideoManifestRecord],
) -> tuple[tuple[VideoManifestRecord, ...], str, str | None]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).expanduser().resolve()
        records = load_manifest_jsonl(path)
        digest = "sha256:" + compute_manifest_sha256(records)
        return records, digest, str(path)
    records = validate_manifest(manifest)
    return records, "sha256:" + compute_manifest_sha256(records), None


def _prediction_digest(records: Iterable[Any]) -> str:
    rows = []
    for record in records:
        rows.append(
            {
                "video_id": _field(record, "video_id"),
                "clip_id": _field(record, "clip_id"),
                "clip_index": _field(record, "clip_index"),
                "frame_start": _field(record, "frame_start"),
                "frame_end": _field(record, "frame_end"),
                "start_s": _field(record, "start_s"),
                "end_s": _field(record, "end_s"),
                "anomaly_score": _field(record, "anomaly_score", _field(record, "score")),
                "encoder_fingerprint": _field(record, "encoder_fingerprint"),
            }
        )
    return _canonical_digest(rows)


def _load_audit_report(value: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        raise ValueError("official protocol requires a passed dataset audit report")
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    else:
        report = dict(value)
    from jsonschema import Draft202012Validator, FormatChecker

    from vadbench.resources import package_resource_path

    schema_path = Path(__file__).resolve().parents[3] / "schemas/dataset-audit-v2.schema.json"
    if not schema_path.is_file():
        schema_path = package_resource_path("schemas/dataset-audit-v2.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    error = next(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(report), None
    )
    if error is not None:
        raise ValueError(f"dataset audit does not conform to v2 schema: {error.message}")
    if report["errors"] or report["status"] not in {"passed", "passed_with_warnings"}:
        raise ValueError("official protocol rejects a dataset audit with errors")
    if report["passed"] is not True:
        raise ValueError("official protocol requires dataset audit passed=true")
    if report["evaluation_readiness"]["ready"] is not True:
        raise ValueError("official protocol requires evaluation_readiness.ready=true")
    source_identity = report["official_source_identity"]
    if source_identity["status"] != "verified":
        raise ValueError("official protocol requires verified official source identity")
    if not source_identity.get("source_commit") or not source_identity.get("test_identity_sha256"):
        raise ValueError("official source identity is missing commit or test identity SHA256")
    return report


def _validate_official_audit(
    report: Mapping[str, Any],
    manifests: tuple[VideoManifestRecord, ...],
    manifest_path: str | None,
    manifest_digest: str,
) -> dict[str, Any]:
    if manifest_path is None:
        raise ValueError("official protocol requires a manifest file for audit identity")
    audit_manifests = report.get("manifests")
    if not isinstance(audit_manifests, Mapping) or not audit_manifests.get("test"):
        raise ValueError("dataset audit is missing its test manifest identity")
    audited_path = Path(str(audit_manifests["test"])).expanduser().resolve()
    if audited_path != Path(manifest_path):
        raise ValueError("dataset audit test manifest does not match evaluation manifest")
    manifest_hashes = report.get("manifest_sha256")
    if not isinstance(manifest_hashes, Mapping) or manifest_hashes.get("test") != (
        manifest_digest.removeprefix("sha256:")
    ):
        raise ValueError("dataset audit test manifest SHA256 does not match evaluation manifest")

    expected = OFFICIAL_UCF_CRIME_COUNTS["test"]
    observed = report.get("observed", {}).get("test", {})
    if any(observed.get(name) != count for name, count in expected.items()):
        raise ValueError("dataset audit does not contain the official test counts")
    audited_ids = {
        str(item.get("video_id"))
        for item in report.get("videos", [])
        if isinstance(item, Mapping) and item.get("split") == "test"
    }
    manifest_ids = {item.video_id for item in manifests}
    if audited_ids != manifest_ids:
        raise ValueError("dataset audit test video set does not match evaluation manifest")
    return {
        "schema_version": report["schema_version"],
        "generated_at": report.get("generated_at"),
        "status": report.get("status"),
        "test_manifest": str(audited_path),
        "test_manifest_sha256": manifest_digest,
        "evaluation_ready": True,
        "official_source_commit": report["official_source_identity"]["source_commit"],
        "official_test_identity_sha256": report["official_source_identity"]["test_identity_sha256"],
    }


def evaluate_manifest_predictions(
    records: Iterable[Any],
    manifest: str | Path | Iterable[VideoManifestRecord],
    *,
    protocol: Literal["official", "subset", "generic"] = "official",
    audit_report: str | Path | Mapping[str, Any] | None = None,
    official_source_registry: str | Path = "registry/datasets.yaml",
    reduction: str = "max",
    undefined: Literal["nan", "raise"] = "nan",
) -> UCFEvaluationResult:
    """Evaluate prediction records under an explicit UCF-Crime protocol mode.

    ``official`` requires the complete audited 290-video test split. ``subset``
    permits a named test subset but keeps strict frame coverage. ``generic``
    retains the lower-level projection behavior for diagnostics and tests.
    """

    if protocol not in {"official", "subset", "generic"}:
        raise ValueError("protocol must be official, subset, or generic")
    prediction_records = tuple(records)
    if not prediction_records:
        raise ValueError("prediction records must not be empty")
    run_identity = {}
    if protocol != "generic":
        for key in ("run_id", "encoder_fingerprint", "checkpoint_sha256"):
            values = [
                _field(_field(item, "metadata", {}), key)
                if key == "checkpoint_sha256"
                else _field(item, key)
                for item in prediction_records
            ]
            present = {value for value in values if value is not None}
            if len(present) > 1 or (present and any(value is None for value in values)):
                raise ValueError(f"prediction records mix incompatible {key} identities")
            if present:
                run_identity[key] = next(iter(present))
    manifests, manifest_digest, manifest_path = _manifest_records(manifest)
    if any(item.split != DatasetSplit.TEST for item in manifests):
        raise ValueError(f"{protocol} evaluation requires only test split records")
    if protocol == "official" and len(manifests) != OFFICIAL_UCF_CRIME_COUNTS["test"]["total"]:
        raise ValueError(
            "official protocol requires exactly "
            f"{OFFICIAL_UCF_CRIME_COUNTS['test']['total']} test videos"
        )

    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in prediction_records:
        grouped[str(_field(record, "video_id"))].append(record)
    manifest_by_id = {item.video_id: item for item in manifests}
    if set(grouped) != set(manifest_by_id):
        raise ValueError(
            "prediction/manifest video ids differ: "
            f"prediction_only={sorted(set(grouped) - set(manifest_by_id))}, "
            f"manifest_only={sorted(set(manifest_by_id) - set(grouped))}"
        )

    audit_identity = None
    if protocol == "official":
        report = _load_audit_report(audit_report)
        audit_identity = _validate_official_audit(report, manifests, manifest_path, manifest_digest)
        source_errors: list[dict[str, Any]] = []
        source_identity = verify_official_source_identity(
            {"test": manifests},
            registry_path=official_source_registry,
            errors=source_errors,
        )
        recorded_source = report["official_source_identity"]
        if source_errors or source_identity["status"] != "verified":
            raise ValueError(f"official source verification failed: {source_errors}")
        for key in ("source_commit", "train_identity_sha256", "test_identity_sha256"):
            if source_identity[key] != recorded_source[key]:
                raise ValueError(f"dataset audit official source identity changed: {key}")

    coverage_summary: dict[str, Any]
    if protocol in {"official", "subset"}:
        per_video = []
        for video_id, video_records in grouped.items():
            ordered = sorted(
                video_records,
                key=lambda item: (
                    int(_field(item, "clip_index", 0)),
                    float(_field(item, "start_s", 0.0)),
                ),
            )
            if any(
                _field(item, "frame_start") is None or _field(item, "frame_end") is None
                for item in ordered
            ):
                raise ValueError(f"{video_id}: {protocol} protocol requires frame intervals")
            manifest_item = manifest_by_id[video_id]
            per_video.append(
                validate_frame_coverage(
                    video_id=video_id,
                    clip_indices=np.asarray(
                        [int(_field(item, "clip_index", 0)) for item in ordered],
                        dtype=np.int64,
                    ),
                    frame_starts=np.asarray(
                        [int(_field(item, "frame_start")) for item in ordered],
                        dtype=np.int64,
                    ),
                    frame_ends=np.asarray(
                        [int(_field(item, "frame_end")) for item in ordered],
                        dtype=np.int64,
                    ),
                    num_frames=manifest_item.num_frames,
                    fps=manifest_item.fps,
                    require_fps=True,
                )
            )
        coverage_summary = {"status": "validated", **aggregate_coverage(per_video)}
    else:
        coverage_summary = {
            "status": "not_required",
            "complete": None,
            "reason": "generic protocol permits lower-level projection semantics",
        }

    labels = frame_labels_from_manifest(manifests)
    result = evaluate_ucf_prediction_records(
        prediction_records,
        labels,
        fps={item.video_id: item.fps for item in manifests if item.fps is not None},
        reduction=reduction,
        undefined=undefined,
    )
    protocol_id = {
        "official": "ucf-crime/official-frameauc-v1",
        "subset": "ucf-crime/subset-frameauc-v1",
        "generic": "ucf-crime/generic-frameauc-v1",
    }[protocol]
    input_identity = {
        **run_identity,
        "manifest_sha256": manifest_digest,
        "predictions_sha256": _prediction_digest(prediction_records),
        "videos": len(manifests),
        "prediction_records": len(prediction_records),
    }
    if manifest_path is not None:
        input_identity["manifest_path"] = manifest_path
    if audit_identity is not None:
        input_identity["dataset_audit"] = audit_identity
    return UCFEvaluationResult(
        metrics=result.metrics,
        frame_scores=result.frame_scores,
        protocol=protocol_id,
        coverage=coverage_summary,
        input_identity=input_identity,
    )


def evaluate_ucf_frame_auc(
    predictions: Mapping[str, Any], frame_labels: Mapping[str, Any], **kwargs: Any
) -> float:
    """Return only the benchmark's primary frame ROC-AUC scalar."""

    return evaluate_ucf_predictions(predictions, frame_labels, **kwargs).metrics.frame_auc


def evaluate_batches(
    model: Any,
    batches: Iterable[Any],
    *,
    prediction_fn: Any | None = None,
    device: Any | None = None,
) -> list[Any]:
    """Run inference without gradients; projection/metrics remain separate.

    ``prediction_fn(output, batch)`` can turn task-specific outputs into
    :class:`TemporalPrediction` records. Tasks with ``prediction_step`` retain
    their timeline/mask automatically; otherwise raw outputs are returned.
    """

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to evaluate a model")
    if device is not None:
        model.to(device)
    model.eval()
    collected: list[Any] = []
    with torch.inference_mode():
        for batch in batches:
            if device is not None:
                from .train import move_to_device

                batch = move_to_device(batch, device)
            prediction_step = getattr(model, "prediction_step", None)
            output = prediction_step(batch) if callable(prediction_step) else model(batch)
            collected.append(prediction_fn(output, batch) if prediction_fn is not None else output)
    return collected


__all__ = [
    "TORCH_AVAILABLE",
    "TemporalPrediction",
    "UCFEvaluationResult",
    "evaluate_batches",
    "evaluate_manifest_predictions",
    "evaluate_ucf_frame_auc",
    "evaluate_ucf_prediction_records",
    "evaluate_ucf_predictions",
    "prediction_records_to_temporal",
    "project_video_prediction",
]
