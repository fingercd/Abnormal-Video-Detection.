"""Checkpointed head inference over persisted video features."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.artifacts import PredictionRecord
from vadbench.checkpoints import sha256_file
from vadbench.data.features_dataset import FeatureDataset, build_feature_dataloader
from vadbench.data.manifest import (
    VideoManifestRecord,
    load_manifest_jsonl,
    validate_manifest,
)
from vadbench.engine.coverage import validate_frame_coverage
from vadbench.engine.runner import HeadOnlyTrainingConfig, normalize_task_name
from vadbench.engine.train import load_checkpoint, move_to_device
from vadbench.features import FeatureStore, atomic_write_jsonl
from vadbench.tasks import build_task

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by the minimal environment.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _records(
    value: str | Path | Iterable[VideoManifestRecord | Mapping[str, Any]],
) -> tuple[VideoManifestRecord, ...]:
    if isinstance(value, (str, Path)):
        return load_manifest_jsonl(value)
    normalized = [
        item if isinstance(item, VideoManifestRecord) else VideoManifestRecord.from_dict(item)
        for item in value
    ]
    return validate_manifest(normalized)


def _checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint checksum manifest is missing: {sidecar}")
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid checkpoint checksum manifest: {sidecar}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint checksum manifest must be an object: {sidecar}")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be an object")
    return dict(metadata)


def _configured_fingerprint(config: Mapping[str, Any] | None) -> str | None:
    if config is None:
        return None
    direct = config.get("encoder_fingerprint")
    if direct is not None:
        return str(direct)
    features = config.get("features", {})
    if isinstance(features, Mapping) and features.get("encoder_fingerprint") is not None:
        return str(features["encoder_fingerprint"])
    return None


def _settings(
    config: HeadOnlyTrainingConfig | Mapping[str, Any],
) -> HeadOnlyTrainingConfig:
    if isinstance(config, HeadOnlyTrainingConfig):
        return config
    task = config.get("task")
    if isinstance(task, str):
        normalized = dict(config)
        task_section = {"kind": task}
        for name in ("head", "head_kwargs", "task_kwargs", "pooling", "top_k"):
            if name in config:
                task_section[name] = config[name]
        normalized["task"] = task_section
        return HeadOnlyTrainingConfig.from_mapping(normalized)
    return HeadOnlyTrainingConfig.from_mapping(config)


def _run_id(config: Mapping[str, Any] | None, checkpoint_path: str | Path) -> str:
    if config is not None:
        direct = config.get("run_id")
        if direct is not None and str(direct).strip():
            return str(direct)
        output = config.get("output", {})
        if isinstance(output, Mapping):
            run_name = output.get("run_name")
            if run_name is not None and str(run_name).strip():
                return str(run_name)
    path = Path(checkpoint_path).expanduser().resolve()
    candidate = path.parent.parent.name if path.parent.name == "checkpoints" else path.stem
    return candidate or "prediction"


def _numpy(value: Any) -> np.ndarray:
    if TORCH_AVAILABLE and torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scores(step_output: Any, task_name: str) -> np.ndarray:
    predictions = getattr(step_output, "predictions", None)
    if predictions is None and isinstance(step_output, Mapping):
        predictions = step_output.get("predictions")
    if task_name == "wsvad":
        scores = getattr(predictions, "snippet_scores", None)
        if scores is None and isinstance(predictions, Mapping):
            scores = predictions.get("snippet_scores")
        if scores is None:
            raise TypeError("weak MIL prediction must expose snippet_scores")
    else:
        scores = predictions
    result = _numpy(scores)
    if result.ndim != 2 or min(result.shape) <= 0:
        raise ValueError(f"prediction scores must have shape [B,S], got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("prediction scores contain NaN or Infinity")
    return result.astype(np.float64, copy=False)


def predict_feature_head(
    config: HeadOnlyTrainingConfig | Mapping[str, Any],
    feature_store: FeatureStore | str | Path,
    manifest: str | Path | Iterable[VideoManifestRecord | Mapping[str, Any]],
    checkpoint_path: str | Path,
    output_path: str | Path,
    device: Any | None = None,
    strict_coverage: bool = True,
) -> list[PredictionRecord]:
    """Load a verified head checkpoint and emit standard clip prediction JSONL."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for prediction; install the train extra")
    if not isinstance(strict_coverage, bool):
        raise TypeError("strict_coverage must be boolean")
    raw_config = config if isinstance(config, Mapping) else None
    settings = _settings(config)
    task_name = normalize_task_name(settings.task)
    checkpoint_metadata = _checkpoint_metadata(checkpoint_path)
    configured_fingerprint = _configured_fingerprint(raw_config)
    checkpoint_fingerprint = checkpoint_metadata.get("encoder_fingerprint")
    if (
        configured_fingerprint is not None
        and checkpoint_fingerprint is not None
        and configured_fingerprint != checkpoint_fingerprint
    ):
        raise ValueError("config encoder fingerprint conflicts with checkpoint metadata")
    fingerprint = (
        str(checkpoint_fingerprint)
        if checkpoint_fingerprint is not None
        else configured_fingerprint
    )

    manifests = _records(manifest)
    manifest_by_id = {item.video_id: item for item in manifests}
    # Prediction never consumes target annotations.  Weak dataset mode preserves
    # every feature-valid position even for a temporal-supervised checkpoint.
    dataset = FeatureDataset(
        feature_store,
        manifests,
        encoder_fingerprint=fingerprint,
        supervision="weak",
        feature_level=settings.feature_level,
        expected_clips=settings.expected_clips,
    )
    if checkpoint_metadata.get("feature_dim") is not None and (
        int(checkpoint_metadata["feature_dim"]) != dataset.feature_dim
    ):
        raise ValueError("checkpoint feature_dim does not match prediction features")
    if checkpoint_metadata.get("feature_level") is not None and (
        str(checkpoint_metadata["feature_level"]) != dataset.feature_level
    ):
        raise ValueError("checkpoint feature_level does not match prediction config")
    if checkpoint_metadata.get("task") is not None and (
        normalize_task_name(str(checkpoint_metadata["task"])) != task_name
    ):
        raise ValueError("checkpoint task does not match prediction config")

    model = build_task(
        task_name,
        None,
        feature_dim=dataset.feature_dim,
        head=settings.head,
        head_kwargs=settings.head_kwargs,
        task_kwargs=settings.task_kwargs,
    )
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    load_checkpoint(
        checkpoint_path,
        model,
        map_location="cpu",
        strict=True,
        verify=True,
    )
    checkpoint_digest = sha256_file(checkpoint_path)
    model.to(resolved_device)
    model.eval()
    loader = build_feature_dataloader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
    )

    prediction_records: list[PredictionRecord] = []
    run_id = _run_id(raw_config, checkpoint_path)
    with torch.no_grad():
        for batch in loader:
            moved = move_to_device(batch, resolved_device)
            step_output = model.prediction_step(moved)
            score_array = _scores(step_output, task_name)
            valid = _numpy(batch["valid_mask"]).astype(bool, copy=False)
            clip_indices = _numpy(batch["clip_indices"]).astype(np.int64, copy=False)
            timeline = batch["timeline"]
            frame_starts_raw = timeline.get("source_frame_start")
            frame_ends_raw = timeline.get("source_frame_end")
            if frame_starts_raw is None or frame_ends_raw is None:
                raise ValueError("feature records/timeline do not contain real source frame ranges")
            frame_starts = _numpy(frame_starts_raw).astype(np.int64, copy=False)
            frame_ends = _numpy(frame_ends_raw).astype(np.int64, copy=False)
            start_seconds = _numpy(timeline["start_s"]).astype(np.float64, copy=False)
            end_seconds = _numpy(timeline["end_s"]).astype(np.float64, copy=False)
            if score_array.shape != valid.shape:
                raise ValueError("prediction scores and valid_mask shapes differ")
            if any(
                value.shape != score_array.shape
                for value in (
                    clip_indices,
                    frame_starts,
                    frame_ends,
                    start_seconds,
                    end_seconds,
                )
            ):
                raise ValueError("prediction metadata must align with [B,S] scores")

            for row, video_id in enumerate(batch["video_ids"]):
                positions = np.flatnonzero(valid[row])
                if positions.size == 0:
                    raise ValueError(f"{video_id}: prediction selects no valid snippets")
                row_indices = clip_indices[row, positions]
                row_frame_starts = frame_starts[row, positions]
                row_frame_ends = frame_ends[row, positions]
                manifest_record = manifest_by_id[video_id]
                if strict_coverage:
                    validate_frame_coverage(
                        video_id=video_id,
                        clip_indices=row_indices,
                        frame_starts=row_frame_starts,
                        frame_ends=row_frame_ends,
                        num_frames=manifest_record.num_frames,
                        fps=manifest_record.fps,
                        require_fps=True,
                    )
                elif manifest_record.num_frames is not None and (
                    np.any(row_frame_starts < 0)
                    or np.any(row_frame_ends <= row_frame_starts)
                    or np.any(row_frame_ends > manifest_record.num_frames)
                ):
                    raise ValueError(f"{video_id}: invalid or out-of-bounds frame range")

                source_clip_ids = batch["clip_ids"][row]
                one_score_per_clip = len(source_clip_ids) == positions.size
                for output_index, position in enumerate(positions):
                    source_clip_index = int(row_indices[output_index])
                    if one_score_per_clip:
                        clip_id = str(source_clip_ids[output_index])
                        prediction_index = source_clip_index
                    else:
                        clip_id = f"{video_id}:snippet-{output_index:06d}"
                        prediction_index = output_index
                    start_s = float(start_seconds[row, position])
                    end_s = float(end_seconds[row, position])
                    if manifest_record.fps is not None:
                        start_s = int(row_frame_starts[output_index]) / float(manifest_record.fps)
                        end_s = int(row_frame_ends[output_index]) / float(manifest_record.fps)
                    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
                        raise ValueError(f"{video_id}: invalid stored prediction timeline")
                    score = float(score_array[row, position])
                    prediction_records.append(
                        PredictionRecord(
                            run_id=run_id,
                            video_id=video_id,
                            clip_id=clip_id,
                            clip_index=prediction_index,
                            start_s=start_s,
                            end_s=end_s,
                            frame_start=int(row_frame_starts[output_index]),
                            frame_end=int(row_frame_ends[output_index]),
                            anomaly_score=score,
                            predicted_label=bool(score >= 0.5),
                            ground_truth=manifest_record.is_anomaly,
                            encoder_fingerprint=dataset.encoder_fingerprint,
                            metadata={
                                "task": task_name,
                                "score_level": ("clip" if one_score_per_clip else "snippet"),
                                "ground_truth_scope": "video",
                                "source_clip_index": source_clip_index,
                                "checkpoint": Path(checkpoint_path).name,
                                "checkpoint_sha256": checkpoint_digest,
                            },
                        )
                    )
    atomic_write_jsonl(output_path, (item.to_dict() for item in prediction_records))
    return prediction_records


__all__ = ["TORCH_AVAILABLE", "predict_feature_head"]
