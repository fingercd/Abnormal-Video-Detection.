"""Task composition between encoder adapters and trainable VAD heads.

The encoder contract is intentionally consumed structurally: an output needs
``features`` with shape ``[B, S, D]`` and may expose
``timeline.valid`` with shape ``[B, S]``.  This keeps the task layer usable by
fixed-window and streaming/cache-aware adapters without copying their state or
feature artifact types.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .models.heads import (
    TORCH_AVAILABLE,
    AttentionMILHead,
    MILHeadOutput,
    TemporalSupervisedHead,
    TopKMILHead,
    temporal_supervised_loss,
    weakly_supervised_ranking_loss,
)

try:  # Keep all non-neural modules importable in a NumPy-only environment.
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - covered by minimal dependency tests.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for vadbench task modules. Install the "
            "project's 'torch' extra or a compatible PyTorch build."
        )


def _value(container: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(container, Mapping) and name in container:
            return container[name]
        if hasattr(container, name):
            return getattr(container, name)
    return default


@dataclass(frozen=True)
class EncoderFeatures:
    """Task-facing view of an ``EncoderOutput`` without a duplicate protocol."""

    features: Any
    valid_mask: Any | None
    raw_output: Any


@dataclass(frozen=True)
class TemporalTargetBatch:
    """Strong temporal targets aligned to an encoder token timeline."""

    labels: np.ndarray
    valid_mask: np.ndarray


def _numpy_metadata(value: Any, *, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def build_temporal_targets(
    timeline: Any,
    annotations_by_video: Any,
    *,
    min_overlap_fraction: float = 0.0,
    overlap_reference: Literal["token", "iou"] = "token",
    assume_unannotated_is_normal: bool = True,
) -> TemporalTargetBatch:
    """Map explicit anomaly spans onto ``EncoderOutput.timeline`` tokens.

    Positive ``segment``/``frame`` annotations are matched using half-open
    overlap. Frame-unit annotations use ``source_frame_start/end``; second-unit
    annotations use ``start_s/end_s``. Caption and video-only annotations are
    deliberately ignored because they are not strong temporal truth.

    ``min_overlap_fraction=0`` means any positive overlap.  Higher thresholds
    can require either a fraction of the token duration (``token``) or IoU.
    A row with only caption/video-level annotations receives an all-false target
    mask and therefore cannot silently become a negative training example. If
    at least one explicit temporal truth exists, the default treats uncovered
    valid tokens as normal (the UCF temporal annotation convention). Set
    ``assume_unannotated_is_normal=False`` for partially annotated datasets.
    """

    if not 0.0 <= min_overlap_fraction <= 1.0:
        raise ValueError("min_overlap_fraction must be in [0, 1]")
    if overlap_reference not in {"token", "iou"}:
        raise ValueError("overlap_reference must be 'token' or 'iou'")
    if not isinstance(assume_unannotated_is_normal, bool):
        raise TypeError("assume_unannotated_is_normal must be boolean")

    starts_s = _numpy_metadata(_value(timeline, "start_s", "start_seconds"), name="start_s")
    ends_s = _numpy_metadata(_value(timeline, "end_s", "end_seconds"), name="end_s")
    if starts_s.ndim != 2 or starts_s.shape != ends_s.shape:
        raise ValueError("timeline start/end seconds must have equal shape [B, S]")
    frame_starts_raw = _value(timeline, "source_frame_start")
    frame_ends_raw = _value(timeline, "source_frame_end")
    if (frame_starts_raw is None) != (frame_ends_raw is None):
        raise ValueError("timeline source frame start/end must be supplied together")
    frame_starts = (
        None
        if frame_starts_raw is None
        else _numpy_metadata(frame_starts_raw, name="source_frame_start")
    )
    frame_ends = (
        None if frame_ends_raw is None else _numpy_metadata(frame_ends_raw, name="source_frame_end")
    )
    if frame_starts is not None and (
        frame_starts.shape != starts_s.shape or frame_ends.shape != starts_s.shape
    ):
        raise ValueError("timeline source frame ranges must match [B, S]")

    valid_raw = _value(timeline, "valid_mask", "valid")
    valid = (
        np.ones(starts_s.shape, dtype=bool)
        if valid_raw is None
        else _numpy_metadata(valid_raw, name="valid_mask").astype(bool, copy=False)
    )
    if valid.shape != starts_s.shape:
        raise ValueError("timeline valid mask must match [B, S]")

    batch_size = starts_s.shape[0]
    if (
        batch_size == 1
        and isinstance(annotations_by_video, (list, tuple))
        and (not annotations_by_video or _value(annotations_by_video[0], "scope") is not None)
    ):
        annotations_by_video = [annotations_by_video]
    try:
        grouped_annotations = list(annotations_by_video)
    except TypeError as exc:
        raise TypeError("annotations_by_video must contain one iterable per batch item") from exc
    if len(grouped_annotations) != batch_size:
        raise ValueError(
            f"annotations_by_video length must equal batch size {batch_size}, "
            f"got {len(grouped_annotations)}"
        )

    labels = np.zeros(starts_s.shape, dtype=np.float32)
    target_valid = np.zeros(starts_s.shape, dtype=bool)
    for row, video_annotations in enumerate(grouped_annotations):
        has_temporal_truth = False
        for annotation in video_annotations:
            scope = _enum_value(_value(annotation, "scope"))
            is_anomaly = _value(annotation, "is_anomaly")
            if scope not in {"frame", "segment"} or not isinstance(is_anomaly, bool):
                continue
            has_temporal_truth = True
            span = _value(annotation, "span")
            if span is None:
                raise ValueError("temporal anomaly annotation must provide a span")
            unit = _enum_value(_value(span, "unit"))
            annotation_start = float(_value(span, "start"))
            annotation_end = float(_value(span, "end"))
            if not np.isfinite(annotation_start) or not np.isfinite(annotation_end):
                raise ValueError("annotation span must be finite")
            if annotation_end <= annotation_start:
                raise ValueError("annotation span must satisfy end > start")
            if unit == "frame":
                if frame_starts is None or frame_ends is None:
                    raise ValueError("frame annotations require timeline.source_frame_start/end")
                token_starts = frame_starts[row].astype(np.float64, copy=False)
                token_ends = frame_ends[row].astype(np.float64, copy=False)
            elif unit in {"second", "seconds"}:
                token_starts = starts_s[row].astype(np.float64, copy=False)
                token_ends = ends_s[row].astype(np.float64, copy=False)
            else:
                raise ValueError(f"annotation unit {unit!r} cannot be aligned to a token timeline")

            overlap = np.maximum(
                0.0,
                np.minimum(token_ends, annotation_end) - np.maximum(token_starts, annotation_start),
            )
            if overlap_reference == "token":
                denominator = token_ends - token_starts
            else:
                denominator = (
                    token_ends - token_starts + annotation_end - annotation_start - overlap
                )
            fraction = np.divide(
                overlap,
                denominator,
                out=np.zeros_like(overlap, dtype=np.float64),
                where=denominator > 0,
            )
            selected = overlap > 0
            if min_overlap_fraction > 0:
                selected &= fraction >= min_overlap_fraction
            selected &= valid[row]
            if is_anomaly:
                labels[row, selected] = 1.0
            if not assume_unannotated_is_normal:
                target_valid[row, selected] = True

        if has_temporal_truth and assume_unannotated_is_normal:
            target_valid[row] = valid[row]

    labels[~target_valid] = 0.0
    return TemporalTargetBatch(labels=labels, valid_mask=target_valid)


def extract_encoder_features(output: Any) -> EncoderFeatures:
    """Extract ``features`` and timeline validity from common output forms.

    Canonical adapters return ``EncoderOutput.features`` and
    ``EncoderOutput.timeline.valid``.  A direct tensor/array or a mapping is
    accepted for cached feature training and lightweight unit tests.
    """

    if output is None:
        raise ValueError("encoder output must not be None")

    features = _value(output, "features")
    # A direct tensor/array is itself the feature payload.
    if features is None and (
        (TORCH_AVAILABLE and torch.is_tensor(output)) or hasattr(output, "shape")
    ):
        features = output
    # A legacy ``tokens`` fallback eases migration, while the canonical field
    # and all emitted artifacts remain named ``features``.
    if features is None:
        features = _value(output, "tokens")
    if features is None:
        raise TypeError("encoder output must expose 'features' (or be a direct feature tensor)")

    valid_mask = _value(output, "valid_mask", "mask")
    if valid_mask is None:
        timeline = _value(output, "timeline")
        if timeline is not None:
            valid_mask = _value(timeline, "valid", "valid_mask", "mask")
    return EncoderFeatures(features=features, valid_mask=valid_mask, raw_output=output)


@dataclass
class TaskStepOutput(Mapping[str, Any]):
    """Uniform result consumed by :func:`vadbench.engine.train.train_one_step`."""

    loss: Any
    predictions: Any
    metrics: dict[str, Any] = field(default_factory=dict)
    auxiliary: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in {"loss", "predictions", "metrics", "auxiliary"}:
            return getattr(self, key)
        if key in self.metrics:
            return self.metrics[key]
        if key in self.auxiliary:
            return self.auxiliary[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("loss", "predictions", "metrics", "auxiliary"))

    def __len__(self) -> int:
        return 4


if TORCH_AVAILABLE:

    def _adapter_trainable_module(encoder: Any) -> nn.Module | None:
        if isinstance(encoder, nn.Module):
            return None  # ``self.encoder = encoder`` already registers it.
        capabilities = getattr(encoder, "capabilities", None)
        if capabilities is not None and not bool(getattr(capabilities, "supports_training", False)):
            return None
        for name in ("encoder", "model", "backbone"):
            candidate = getattr(encoder, name, None)
            if isinstance(candidate, nn.Module):
                return candidate
        return None

    class _EncoderTask(nn.Module):
        def __init__(self, encoder: Any, head: nn.Module) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = head
            # Contract adapters are not required to inherit nn.Module.  A
            # trainable adapter commonly wraps one complete module; registering
            # it here makes parameters/to/train/state_dict/checkpoints truthful.
            self.registered_encoder_module = _adapter_trainable_module(encoder)

        def _prepare_for_head(self, encoded: EncoderFeatures) -> EncoderFeatures:
            try:
                reference = next(self.head.parameters())
            except StopIteration:
                reference = next(self.head.buffers(), None)
            features = encoded.features
            if torch.is_tensor(features):
                if reference is not None:
                    target_dtype = reference.dtype if reference.is_floating_point() else None
                    features = features.to(device=reference.device, dtype=target_dtype)
            else:
                features = torch.as_tensor(features)
                if not features.is_floating_point():
                    features = features.float()
                if reference is not None:
                    target_dtype = reference.dtype if reference.is_floating_point() else None
                    features = features.to(device=reference.device, dtype=target_dtype)
            valid_mask = encoded.valid_mask
            if valid_mask is not None:
                valid_mask = torch.as_tensor(valid_mask, device=features.device, dtype=torch.bool)
            return EncoderFeatures(
                features=features,
                valid_mask=valid_mask,
                raw_output=encoded.raw_output,
            )

        def _encode(self, batch: Any, *, train: bool) -> EncoderFeatures:
            supplied_output = _value(batch, "encoder_output")
            if supplied_output is not None:
                return self._prepare_for_head(extract_encoder_features(supplied_output))

            supplied_features = _value(batch, "features")
            if supplied_features is not None:
                supplied_mask = _value(batch, "valid_mask", "mask")
                return self._prepare_for_head(
                    EncoderFeatures(supplied_features, supplied_mask, batch)
                )

            encode = getattr(self.encoder, "encode", None)
            if callable(encode):
                output = encode(batch, train=train)
            elif callable(self.encoder):
                output = self.encoder(batch)
            else:
                raise TypeError("encoder must provide encode(batch, train=...) or be callable")
            return self._prepare_for_head(extract_encoder_features(output))

        @staticmethod
        def _tensor(value: Any, *, device: Any, dtype: Any | None = None) -> Tensor:
            if value is None:
                raise KeyError("required batch target is missing")
            if torch.is_tensor(value):
                return value.to(device=device, dtype=dtype)
            return torch.as_tensor(value, device=device, dtype=dtype)

    class WeaklySupervisedMILTask(_EncoderTask):
        """Video-level binary MIL task for UCF-Crime weak supervision."""

        def __init__(
            self,
            encoder: Any,
            head: nn.Module,
            *,
            classification_weight: float = 1.0,
            ranking_weight: float = 0.0,
            ranking_margin: float = 1.0,
            smoothness_weight: float = 0.0,
            sparsity_weight: float = 0.0,
        ) -> None:
            super().__init__(encoder, head)
            if (
                min(
                    classification_weight,
                    ranking_weight,
                    ranking_margin,
                    smoothness_weight,
                    sparsity_weight,
                )
                < 0
            ):
                raise ValueError("task weights and ranking margin must be non-negative")
            self.classification_weight = float(classification_weight)
            self.ranking_weight = float(ranking_weight)
            self.ranking_margin = float(ranking_margin)
            self.smoothness_weight = float(smoothness_weight)
            self.sparsity_weight = float(sparsity_weight)

        def _forward_with_features(
            self, batch: Any, *, train: bool
        ) -> tuple[EncoderFeatures, MILHeadOutput]:
            encoded = self._encode(batch, train=train)
            output = self.head(encoded.features, mask=encoded.valid_mask)
            if not isinstance(output, MILHeadOutput):
                raise TypeError("MIL head must return MILHeadOutput")
            return encoded, output

        def forward(self, batch: Any) -> MILHeadOutput:
            _, output = self._forward_with_features(batch, train=self.training)
            return output

        def prediction_step(self, batch: Any) -> TaskStepOutput:
            encoded, output = self._forward_with_features(batch, train=False)
            return TaskStepOutput(
                loss=None,
                predictions=output,
                auxiliary={
                    "valid_mask": encoded.valid_mask,
                    "timeline": _value(encoded.raw_output, "timeline"),
                },
            )

        def training_step(self, batch: Any) -> TaskStepOutput:
            encoded, output = self._forward_with_features(batch, train=self.training)
            labels_raw = _value(
                batch,
                "video_labels",
                "video_label",
                "labels",
                "label",
                "is_anomaly",
            )
            labels = self._tensor(
                labels_raw, device=output.video_logits.device, dtype=output.video_logits.dtype
            ).reshape(-1)
            if labels.shape != output.video_logits.shape:
                raise ValueError(
                    f"video labels must have shape {tuple(output.video_logits.shape)}, "
                    f"got {tuple(labels.shape)}"
                )
            classification = F.binary_cross_entropy_with_logits(output.video_logits, labels)
            ranking = classification.detach() * 0.0
            ranking_pairs = 0
            if self.ranking_weight > 0 and (labels > 0.5).any() and (labels <= 0.5).any():
                ranking = weakly_supervised_ranking_loss(
                    output.snippet_scores,
                    labels,
                    mask=encoded.valid_mask,
                    margin=self.ranking_margin,
                    smoothness_weight=self.smoothness_weight,
                    sparsity_weight=self.sparsity_weight,
                )
                ranking_pairs = int((labels > 0.5).sum() * (labels <= 0.5).sum())
            loss = self.classification_weight * classification + self.ranking_weight * ranking
            return TaskStepOutput(
                loss=loss,
                predictions=output,
                metrics={
                    "classification_loss": classification.detach(),
                    "ranking_loss": ranking.detach(),
                    "ranking_pairs": ranking_pairs,
                },
                auxiliary={
                    "valid_mask": encoded.valid_mask,
                    "timeline": _value(encoded.raw_output, "timeline"),
                },
            )

    class TemporalSupervisedTask(_EncoderTask):
        """Frame/snippet-level binary task for strongly supervised VAD."""

        def __init__(
            self,
            encoder: Any,
            head: nn.Module,
            *,
            positive_weight: float | None = None,
            focal_gamma: float | None = None,
        ) -> None:
            super().__init__(encoder, head)
            self.positive_weight = positive_weight
            self.focal_gamma = focal_gamma

        def forward(self, batch: Any) -> Tensor:
            encoded = self._encode(batch, train=self.training)
            return self.head(encoded.features, mask=encoded.valid_mask)

        def prediction_step(self, batch: Any) -> TaskStepOutput:
            encoded = self._encode(batch, train=False)
            logits = self.head(encoded.features, mask=encoded.valid_mask)
            return TaskStepOutput(
                loss=None,
                predictions=torch.sigmoid(logits),
                auxiliary={
                    "logits": logits,
                    "valid_mask": encoded.valid_mask,
                    "timeline": _value(encoded.raw_output, "timeline"),
                },
            )

        def training_step(self, batch: Any) -> TaskStepOutput:
            encoded = self._encode(batch, train=self.training)
            logits = self.head(encoded.features, mask=encoded.valid_mask)
            target_bundle = _value(batch, "temporal_targets")
            if target_bundle is not None:
                targets_raw = _value(target_bundle, "labels")
                target_valid_raw = _value(target_bundle, "valid_mask")
            else:
                targets_raw = _value(
                    batch,
                    "temporal_labels",
                    "frame_labels",
                    "targets",
                    "labels",
                )
                target_valid_raw = _value(batch, "temporal_valid_mask", "target_valid_mask")
            targets = self._tensor(targets_raw, device=logits.device, dtype=logits.dtype)
            loss_mask = (
                torch.ones_like(logits, dtype=torch.bool)
                if encoded.valid_mask is None
                else torch.as_tensor(encoded.valid_mask, device=logits.device, dtype=torch.bool)
            )
            if target_valid_raw is not None:
                target_valid = torch.as_tensor(
                    target_valid_raw, device=logits.device, dtype=torch.bool
                )
                if target_valid.shape != logits.shape:
                    raise ValueError("temporal target valid mask must match logits")
                loss_mask &= target_valid
            loss = temporal_supervised_loss(
                logits,
                targets,
                mask=loss_mask,
                positive_weight=self.positive_weight,
                focal_gamma=self.focal_gamma,
            )
            with torch.no_grad():
                prediction = torch.sigmoid(logits)
                accuracy = ((prediction >= 0.5) == (targets >= 0.5))[loss_mask].float().mean()
            return TaskStepOutput(
                loss=loss,
                predictions=prediction,
                metrics={"temporal_loss": loss.detach(), "temporal_accuracy": accuracy},
                auxiliary={
                    "logits": logits,
                    "valid_mask": loss_mask,
                    "timeline": _value(encoded.raw_output, "timeline"),
                },
            )


else:

    class _TorchRequiredTask:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class WeaklySupervisedMILTask(_TorchRequiredTask):
        pass

    class TemporalSupervisedTask(_TorchRequiredTask):
        pass


def build_task(
    task: str,
    encoder: Any,
    *,
    feature_dim: int,
    head: str | Any | None = None,
    head_kwargs: Mapping[str, Any] | None = None,
    task_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Build one of the two canonical benchmark tasks from a small config."""

    _require_torch()
    normalized = task.strip().lower().replace("-", "_")
    head_options = dict(head_kwargs or {})
    task_options = dict(task_kwargs or {})

    if normalized in {"weak", "weak_mil", "weakly_supervised", "mil", "wsvad"}:
        if head is None or head in {"attention", "attention_mil"}:
            head_module = AttentionMILHead(feature_dim, **head_options)
        elif head in {"topk", "top_k"}:
            head_module = TopKMILHead(feature_dim, **head_options)
        elif isinstance(head, str):
            raise ValueError(f"unknown MIL head: {head}")
        else:
            head_module = head
        return WeaklySupervisedMILTask(encoder, head_module, **task_options)

    if normalized in {
        "strong",
        "supervised",
        "temporal",
        "temporal_supervised",
        "frame_supervised",
    }:
        if head is None or head in {"temporal", "supervised"}:
            head_module = TemporalSupervisedHead(feature_dim, **head_options)
        elif isinstance(head, str):
            raise ValueError(f"unknown temporal head: {head}")
        else:
            head_module = head
        return TemporalSupervisedTask(encoder, head_module, **task_options)

    raise ValueError(f"unknown task: {task}")


__all__ = [
    "EncoderFeatures",
    "TaskStepOutput",
    "TemporalTargetBatch",
    "TemporalSupervisedTask",
    "WeaklySupervisedMILTask",
    "build_task",
    "build_temporal_targets",
    "extract_encoder_features",
]
