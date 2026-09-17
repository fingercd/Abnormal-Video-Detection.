"""Multi-epoch head-only training over persisted video features."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.features_dataset import FeatureDataset, build_feature_dataloader
from vadbench.data.manifest import DatasetSplit, validate_manifest
from vadbench.engine.train import move_to_device, save_checkpoint, train_one_step
from vadbench.features import FeatureStore, atomic_write_json
from vadbench.tasks import build_task

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - covered by the minimal environment.
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def normalize_task_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"weak", "weak_mil", "weakly_supervised", "mil", "wsvad"}:
        return "wsvad"
    if normalized in {
        "strong",
        "supervised",
        "temporal",
        "temporal_supervised",
        "frame_supervised",
    }:
        return "temporal"
    raise ValueError(f"unknown task kind: {value!r}")


def _float_metric(value: Any) -> float:
    value = (
        value.detach().cpu() if TORCH_AVAILABLE and torch.is_tensor(value) else np.asarray(value)
    )
    size = value.numel() if TORCH_AVAILABLE and torch.is_tensor(value) else value.size
    if size != 1:
        raise ValueError("runner metrics must be scalar")
    return float(value.item())


@dataclass(frozen=True)
class HeadOnlyTrainingConfig:
    """Validated runner settings independent of large encoder dependencies."""

    task: str = "weak_mil"
    feature_level: str = "clip"
    head: str | Any | None = None
    head_kwargs: Mapping[str, Any] = field(default_factory=dict)
    task_kwargs: Mapping[str, Any] = field(default_factory=dict)
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float | None = None
    max_steps: int | None = None
    seed: int = 0
    num_workers: int = 0
    pin_memory: bool = False
    expected_clips: int | None = None
    strong_unlabeled: str = "exclude"
    min_overlap_fraction: float = 0.0
    overlap_reference: str = "token"
    assume_unannotated_is_normal: bool = True

    def __post_init__(self) -> None:
        normalize_task_name(self.task)
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.expected_clips is not None and self.expected_clips <= 0:
            raise ValueError("expected_clips must be positive or None")
        object.__setattr__(self, "head_kwargs", dict(self.head_kwargs))
        object.__setattr__(self, "task_kwargs", dict(self.task_kwargs))

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> HeadOnlyTrainingConfig:
        """Read either a flat runner mapping or the repository experiment YAML shape."""

        sections = (config.get(name) for name in ("task", "training", "sampler", "encoder"))
        task_section, training, sampler, encoder = (
            {} if section is None else section for section in sections
        )
        if not all(
            isinstance(item, Mapping) for item in (task_section, training, sampler, encoder)
        ):
            raise TypeError("task, training, sampler, and encoder config sections must be mappings")
        if bool(encoder.get("trainable", False)):
            raise ValueError(
                "head-only feature training cannot use encoder.trainable=true; "
                "run an end-to-end encoder pipeline instead"
            )

        def take(name: str, default: Any) -> Any:
            if name in training:
                return training[name]
            if name in config and not isinstance(config[name], Mapping):
                return config[name]
            return default

        kind = str(task_section.get("kind", config.get("task_kind", "weak_mil")))
        head = task_section.get("head")
        pooling = task_section.get("pooling")
        if head is None and pooling is not None:
            head = "topk" if str(pooling).lower() in {"topk", "top_k"} else pooling
        head_kwargs = dict(task_section.get("head_kwargs", {}))
        if head in {"topk", "top_k"} and "k" not in head_kwargs:
            head_kwargs["k"] = int(task_section.get("top_k", 3))
        task_kwargs = dict(task_section.get("task_kwargs", {}))
        for name in (
            "classification_weight",
            "ranking_weight",
            "ranking_margin",
            "smoothness_weight",
            "sparsity_weight",
            "positive_weight",
            "focal_gamma",
        ):
            if name in task_section:
                task_kwargs[name] = task_section[name]
        expected = sampler.get("segments_per_video")
        if expected is None:
            expected = take("expected_clips", None)
        return cls(
            task=kind,
            feature_level=str(take("feature_level", "clip")),
            head=head,
            head_kwargs=head_kwargs,
            task_kwargs=task_kwargs,
            batch_size=int(take("batch_size", 2)),
            epochs=int(take("epochs", 1)),
            learning_rate=float(take("learning_rate", take("lr", 1e-3))),
            weight_decay=float(take("weight_decay", 0.0)),
            max_grad_norm=take("max_grad_norm", None),
            max_steps=take("max_steps", None),
            seed=int(take("seed", 0)),
            num_workers=int(take("num_workers", 0)),
            pin_memory=bool(take("pin_memory", False)),
            expected_clips=None if expected is None else int(expected),
            strong_unlabeled=str(take("strong_unlabeled", "exclude")),
            min_overlap_fraction=float(take("min_overlap_fraction", 0.0)),
            overlap_reference=str(take("overlap_reference", "token")),
            assume_unannotated_is_normal=bool(take("assume_unannotated_is_normal", True)),
        )


@dataclass(frozen=True)
class TrainingRunResult:
    """Final model plus traceable artifacts from one training invocation."""

    model: Any = field(repr=False)
    optimizer: Any = field(repr=False)
    checkpoint_path: str
    checkpoint_manifest_path: str
    history_path: str
    history: Mapping[str, Any]
    global_step: int
    epochs_completed: int
    feature_dim: int
    encoder_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_manifest_path": self.checkpoint_manifest_path,
            "history_path": self.history_path,
            "history": dict(self.history),
            "global_step": self.global_step,
            "epochs_completed": self.epochs_completed,
            "feature_dim": self.feature_dim,
            "encoder_fingerprint": self.encoder_fingerprint,
        }


def _accumulate_epoch(
    result: Any,
    losses: list[float],
    metrics: dict[str, tuple[float, int]],
    phase: str,
) -> None:
    loss = _float_metric(result.loss)
    if not np.isfinite(loss):
        raise FloatingPointError(f"non-finite {phase} loss: {loss}")
    losses.append(loss)
    for name, raw_value in result.metrics.items():
        value = _float_metric(raw_value)
        if np.isfinite(value):
            total, count = metrics.get(name, (0.0, 0))
            metrics[name] = total + value, count + 1


def _epoch_summary(
    losses: list[float], metrics: Mapping[str, tuple[float, int]], count_name: str, phase: str
) -> dict[str, Any]:
    if not losses:
        raise ValueError(f"{phase} DataLoader produced no batches")
    return {
        "loss": float(np.mean(losses)),
        count_name: len(losses),
        "metrics": {name: total / count for name, (total, count) in sorted(metrics.items())},
    }


def _evaluate_epoch(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    metrics: dict[str, tuple[float, int]] = {}
    with torch.no_grad():
        for batch in loader:
            result = model.training_step(move_to_device(batch, device))
            _accumulate_epoch(result, losses, metrics, "validation")
    return _epoch_summary(losses, metrics, "batches", "validation")


def _logged_losses(epoch: Mapping[str, Any]) -> dict[str, float]:
    losses = {"train_loss": epoch["train"]["loss"]}
    if "validation" in epoch:
        losses["validation_loss"] = epoch["validation"]["loss"]
    return losses


def _resolved_output_dir(
    raw_config: Mapping[str, Any] | None,
    output_dir: str | Path | None,
    artifact_store: Any | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    if artifact_store is not None and hasattr(artifact_store, "run_dir"):
        return Path(artifact_store.run_dir).resolve()
    output = {} if raw_config is None else raw_config.get("output", {})
    if isinstance(output, Mapping):
        root = output.get("root")
        run_name = output.get("run_name")
        if root is not None and run_name is not None:
            return (Path(str(root)) / str(run_name)).expanduser().resolve()
    raise ValueError("output_dir is required when config has no output.root/run_name")


def _config_metadata(config: HeadOnlyTrainingConfig) -> dict[str, Any]:
    value = asdict(config)
    head = value.get("head")
    if head is not None and not isinstance(head, (str, int, float, bool)):
        value["head"] = f"{type(config.head).__module__}.{type(config.head).__qualname__}"
    return value


def train_feature_head(
    config: HeadOnlyTrainingConfig | Mapping[str, Any],
    *,
    feature_store: FeatureStore | str | Path,
    train_manifest: Any,
    validation_manifest: Any | None = None,
    output_dir: str | Path | None = None,
    encoder_fingerprint: str | None = None,
    device: Any | None = None,
    artifact_store: Any | None = None,
) -> TrainingRunResult:
    """Build the canonical task and train only its head over cached features."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training; install the train extra")
    raw_config = config if isinstance(config, Mapping) else None
    settings = HeadOnlyTrainingConfig.from_mapping(config) if raw_config is not None else config
    task_name = normalize_task_name(settings.task)
    supervision = "weak" if task_name == "wsvad" else "strong"

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    generator = torch.Generator()
    generator.manual_seed(settings.seed)

    store = FeatureStore(feature_store) if isinstance(feature_store, (str, Path)) else feature_store
    dataset_options = {
        "encoder_fingerprint": encoder_fingerprint,
        "supervision": supervision,
        "feature_level": settings.feature_level,
        "expected_clips": settings.expected_clips,
        "strong_unlabeled": settings.strong_unlabeled,
        "min_overlap_fraction": settings.min_overlap_fraction,
        "overlap_reference": settings.overlap_reference,
        "assume_unannotated_is_normal": settings.assume_unannotated_is_normal,
    }
    train_dataset = FeatureDataset(store, train_manifest, **dataset_options)
    if any(item.split != DatasetSplit.TRAIN for item in train_dataset.manifest_records):
        raise ValueError("training manifest must contain only train videos")
    validation_dataset = None
    if validation_manifest is not None:
        validation_dataset = FeatureDataset(
            store,
            validation_manifest,
            **{**dataset_options, "encoder_fingerprint": train_dataset.encoder_fingerprint},
        )
        if any(item.split == DatasetSplit.TEST for item in validation_dataset.manifest_records):
            raise ValueError("official test videos cannot be used for validation/model selection")
        validate_manifest((*train_dataset.manifest_records, *validation_dataset.manifest_records))
    if (
        validation_dataset is not None
        and validation_dataset.feature_dim != train_dataset.feature_dim
    ):
        raise ValueError("training and validation feature dimensions differ")

    train_loader = build_feature_dataloader(
        train_dataset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        generator=generator,
    )
    validation_loader = None
    if validation_dataset is not None:
        validation_loader = build_feature_dataloader(
            validation_dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            pin_memory=settings.pin_memory,
        )
    model = build_task(
        task_name,
        None,
        feature_dim=train_dataset.feature_dim,
        head=settings.head,
        head_kwargs=settings.head_kwargs,
        task_kwargs=settings.task_kwargs,
    )
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model.to(resolved_device)
    parameters = [item for item in model.parameters() if item.requires_grad]
    if not parameters:
        raise ValueError("head-only task has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    run_dir = _resolved_output_dir(raw_config, output_dir, artifact_store)
    run_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    epoch_history: list[dict[str, Any]] = []
    stopped_for_max_steps = False
    for epoch in range(1, settings.epochs + 1):
        losses: list[float] = []
        metrics: dict[str, tuple[float, int]] = {}
        for batch in train_loader:
            batch = move_to_device(batch, resolved_device)
            result = train_one_step(
                model,
                batch,
                optimizer,
                step=global_step,
                max_grad_norm=settings.max_grad_norm,
            )
            global_step = result.step
            _accumulate_epoch(result, losses, metrics, "training")
            if settings.max_steps is not None and global_step >= settings.max_steps:
                stopped_for_max_steps = True
                break
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train": _epoch_summary(losses, metrics, "steps", "training"),
        }
        if validation_loader is not None:
            epoch_record["validation"] = _evaluate_epoch(model, validation_loader, resolved_device)
        epoch_history.append(epoch_record)
        if artifact_store is not None and hasattr(artifact_store, "append_metrics"):
            artifact_store.append_metrics(
                _logged_losses(epoch_record),
                split="train",
                step=global_step,
                metadata={"epoch": epoch},
            )
        if stopped_for_max_steps:
            break

    history: dict[str, Any] = {
        "schema_version": 1,
        "status": "max_steps_reached" if stopped_for_max_steps else "completed",
        "task": task_name,
        "encoder_fingerprint": train_dataset.encoder_fingerprint,
        "feature_level": train_dataset.feature_level,
        "feature_dim": train_dataset.feature_dim,
        "train_samples": len(train_dataset),
        "validation_samples": (0 if validation_dataset is None else len(validation_dataset)),
        "epochs_requested": settings.epochs,
        "epochs_completed": len(epoch_history),
        "global_step": global_step,
        "max_steps": settings.max_steps,
        "train_manifest": {
            "records": len(train_dataset),
            "sha256": compute_manifest_sha256(train_dataset.manifest_records),
        },
        "validation_manifest": None
        if validation_dataset is None
        else {
            "records": len(validation_dataset),
            "sha256": compute_manifest_sha256(validation_dataset.manifest_records),
        },
        "config": _config_metadata(settings),
        "epochs": epoch_history,
    }
    checkpoint_path = run_dir / "checkpoints" / "final.pt"
    artifact = save_checkpoint(
        checkpoint_path,
        model,
        optimizer=optimizer,
        step=global_step,
        epoch=len(epoch_history),
        metadata={
            "task": task_name,
            "encoder_fingerprint": train_dataset.encoder_fingerprint,
            "feature_dim": train_dataset.feature_dim,
            "feature_level": train_dataset.feature_level,
            "status": history["status"],
            "train_manifest": history["train_manifest"],
            "validation_manifest": history["validation_manifest"],
            "config": history["config"],
        },
    )
    history["checkpoint"] = artifact.to_dict()
    history_path = run_dir / "history.json"
    atomic_write_json(history_path, history)
    if artifact_store is not None and hasattr(artifact_store, "write_metrics"):
        artifact_store.write_metrics(
            _logged_losses(epoch_history[-1]),
            split="train",
            step=global_step,
            metadata={
                "epochs_completed": len(epoch_history),
                "history_path": str(history_path),
                "checkpoint_path": artifact.path,
            },
        )
    return TrainingRunResult(
        model=model,
        optimizer=optimizer,
        checkpoint_path=artifact.path,
        checkpoint_manifest_path=artifact.manifest_path,
        history_path=str(history_path),
        history=history,
        global_step=global_step,
        epochs_completed=len(epoch_history),
        feature_dim=train_dataset.feature_dim,
        encoder_fingerprint=train_dataset.encoder_fingerprint,
    )


__all__ = [
    "HeadOnlyTrainingConfig",
    "TORCH_AVAILABLE",
    "TrainingRunResult",
    "normalize_task_name",
    "train_feature_head",
]
