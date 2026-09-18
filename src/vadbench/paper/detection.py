"""Paper-scoped weak-MIL detector entry points.

This module is intentionally a thin layer over the existing FeatureStore,
head-only runner, prediction writer, and evaluator.  Its only extra authority
is an explicit, checkpoint-bound permit for a reviewed train/evaluation
sampling or direct-insert representation pairing.  The legacy prediction API
continues to reject every fingerprint mismatch unless this typed permit is
passed by these paper entry points.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vadbench.checkpoints import sha256_file
from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.manifest import DatasetSplit, VideoManifestRecord, validate_manifest
from vadbench.engine.evaluate import UCFEvaluationResult, evaluate_manifest_predictions
from vadbench.engine.predict import _records as _prediction_records
from vadbench.engine.predict import predict_feature_head
from vadbench.engine.runner import HeadOnlyTrainingConfig, TrainingRunResult, train_feature_head
from vadbench.features import FeatureStore

from .compatibility import (
    CompatibilityDeclaration,
    RepresentationIdentity,
    SamplingIdentity,
    feature_cache_key,
    validate_compatibility,
)

DetectorHead = Literal["topk", "attention"]
OverlapReduction = Literal["mean", "max"]


def _fingerprint(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{name} must be a sha256 feature fingerprint")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a sha256 feature fingerprint") from exc
    return value


def _metadata(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint checksum manifest is missing: {sidecar}")
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid checkpoint checksum manifest: {sidecar}") from exc
    if not isinstance(document, Mapping) or not isinstance(document.get("metadata"), Mapping):
        raise ValueError("checkpoint checksum manifest must contain metadata")
    if document.get("sha256") != sha256_file(checkpoint_path):
        raise ValueError("checkpoint checksum manifest does not match checkpoint bytes")
    return dict(document["metadata"])


def _source_checkpoint_identity(declaration: CompatibilityDeclaration) -> dict[str, Any]:
    """The immutable paper provenance that must be embedded in a new head checkpoint."""

    return {
        "schema_version": 1,
        "training_representation_fingerprint": declaration.training_representation.fingerprint,
        "training_sampling_fingerprint": declaration.training_sampling.fingerprint,
        "training_feature_cache_fingerprint": feature_cache_key(
            declaration.training_representation, declaration.training_sampling
        ),
        "training_identity_fingerprint": declaration.training_identity.fingerprint,
    }


@dataclass(frozen=True)
class PredictionCompatibilityPermit:
    """A reviewed exception bound to one source checkpoint and target cache.

    It does not change a checkpoint, a feature index, or any computed
    fingerprint.  ``authorize`` is called by the narrow prediction API before
    target features are opened.
    """

    declaration: CompatibilityDeclaration
    compatibility_receipt: Mapping[str, str]
    source_checkpoint_sha256: str
    source_encoder_fingerprint: str
    target_encoder_fingerprint: str
    evaluation_expected_clips: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, CompatibilityDeclaration):
            raise TypeError("declaration must be a CompatibilityDeclaration")
        receipt = validate_compatibility(self.declaration)
        if dict(self.compatibility_receipt) != receipt:
            raise ValueError("compatibility permit receipt does not match declaration")
        if not isinstance(self.source_checkpoint_sha256, str):
            raise ValueError("source_checkpoint_sha256 must be a SHA-256 digest")
        if len(self.source_checkpoint_sha256) != 64:
            raise ValueError("source_checkpoint_sha256 must be a SHA-256 digest")
        try:
            int(self.source_checkpoint_sha256, 16)
        except ValueError as exc:
            raise ValueError("source_checkpoint_sha256 must be a SHA-256 digest") from exc
        _fingerprint(self.source_encoder_fingerprint, "source_encoder_fingerprint")
        _fingerprint(self.target_encoder_fingerprint, "target_encoder_fingerprint")
        if self.evaluation_expected_clips is not None and (
            isinstance(self.evaluation_expected_clips, bool)
            or not isinstance(self.evaluation_expected_clips, int)
            or self.evaluation_expected_clips <= 0
        ):
            raise ValueError("evaluation_expected_clips must be a positive integer or null")
        object.__setattr__(self, "compatibility_receipt", dict(self.compatibility_receipt))

    @classmethod
    def issue(
        cls,
        declaration: CompatibilityDeclaration,
        *,
        checkpoint_path: str | Path,
        source_encoder_fingerprint: str,
        target_encoder_fingerprint: str,
        evaluation_expected_clips: int | None,
    ) -> PredictionCompatibilityPermit:
        """Verify and bind a permit after the source head checkpoint exists."""

        source_encoder_fingerprint = _fingerprint(
            source_encoder_fingerprint, "source_encoder_fingerprint"
        )
        target_encoder_fingerprint = _fingerprint(
            target_encoder_fingerprint, "target_encoder_fingerprint"
        )
        receipt = validate_compatibility(declaration)
        metadata = _metadata(checkpoint_path)
        if metadata.get("encoder_fingerprint") != source_encoder_fingerprint:
            raise ValueError("checkpoint source encoder fingerprint does not match permit source")
        if metadata.get("feature_dim") != declaration.training_representation.output_dim:
            raise ValueError(
                "checkpoint feature_dim does not match declared training representation"
            )
        if metadata.get("paper_detector") != _source_checkpoint_identity(declaration):
            raise ValueError(
                "checkpoint lacks the verified paper detector training/representation/sampling identity"
            )
        return cls(
            declaration=declaration,
            compatibility_receipt=receipt,
            source_checkpoint_sha256=sha256_file(checkpoint_path),
            source_encoder_fingerprint=source_encoder_fingerprint,
            target_encoder_fingerprint=target_encoder_fingerprint,
            evaluation_expected_clips=evaluation_expected_clips,
        )

    def authorize(
        self,
        *,
        checkpoint_path: str | Path,
        checkpoint_metadata: Mapping[str, Any],
        checkpoint_sha256: str,
    ) -> dict[str, Any]:
        """Return the only target fingerprint this permit authorizes."""

        if sha256_file(checkpoint_path) != self.source_checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 does not match compatibility permit")
        if checkpoint_sha256 != self.source_checkpoint_sha256:
            raise ValueError("prediction checkpoint SHA-256 does not match compatibility permit")
        if checkpoint_metadata.get("encoder_fingerprint") != self.source_encoder_fingerprint:
            raise ValueError("checkpoint encoder fingerprint does not match compatibility permit")
        if (
            checkpoint_metadata.get("feature_dim")
            != self.declaration.training_representation.output_dim
        ):
            raise ValueError("checkpoint feature_dim does not match compatibility declaration")
        # Revalidation makes a stale permit fail if a caller somehow mutates a
        # nested mapping after issuance.
        if validate_compatibility(self.declaration) != dict(self.compatibility_receipt):
            raise ValueError("compatibility declaration changed after permit issuance")
        if checkpoint_metadata.get("paper_detector") != _source_checkpoint_identity(self.declaration):
            raise ValueError(
                "checkpoint paper detector identity does not match compatibility permit"
            )
        return {
            "target_encoder_fingerprint": self.target_encoder_fingerprint,
            "evaluation_expected_clips": self.evaluation_expected_clips,
            "receipt": dict(self.compatibility_receipt),
            "target_paper_identity": {
                "representation_fingerprint": self.declaration.evaluation_representation.fingerprint,
                "sampling_fingerprint": self.declaration.evaluation_sampling.fingerprint,
                "feature_cache_fingerprint": feature_cache_key(
                    self.declaration.evaluation_representation,
                    self.declaration.evaluation_sampling,
                ),
            },
        }

    def audit_dict(self) -> dict[str, Any]:
        return {
            "mode": self.declaration.mode,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_encoder_fingerprint": self.source_encoder_fingerprint,
            "target_encoder_fingerprint": self.target_encoder_fingerprint,
            "evaluation_expected_clips": self.evaluation_expected_clips,
            "compatibility": dict(self.compatibility_receipt),
            "target_paper_identity": {
                "representation_fingerprint": self.declaration.evaluation_representation.fingerprint,
                "sampling_fingerprint": self.declaration.evaluation_sampling.fingerprint,
                "feature_cache_fingerprint": feature_cache_key(
                    self.declaration.evaluation_representation,
                    self.declaration.evaluation_sampling,
                ),
            },
        }


@dataclass(frozen=True)
class DetectionConfig:
    """The fixed, simple head protocol shared by dense and reducer runs."""

    declaration: CompatibilityDeclaration
    training_encoder_fingerprint: str
    evaluation_encoder_fingerprint: str
    head: DetectorHead = "topk"
    head_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"k": 3})
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_steps: int | None = None
    seed: int = 0
    expected_training_clips: int = 32
    expected_evaluation_clips: int | None = None
    overlap_reduction: OverlapReduction = "mean"

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, CompatibilityDeclaration):
            raise TypeError("declaration must be a CompatibilityDeclaration")
        validate_compatibility(self.declaration)
        _fingerprint(self.training_encoder_fingerprint, "training_encoder_fingerprint")
        _fingerprint(self.evaluation_encoder_fingerprint, "evaluation_encoder_fingerprint")
        if self.head not in {"topk", "attention"}:
            raise ValueError("head must be 'topk' or 'attention'")
        if self.overlap_reduction not in {"mean", "max"}:
            raise ValueError("overlap_reduction must be 'mean' or 'max'")
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or null")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.expected_training_clips <= 0:
            raise ValueError("expected_training_clips must be positive")
        if self.expected_evaluation_clips is not None and self.expected_evaluation_clips <= 0:
            raise ValueError("expected_evaluation_clips must be positive or null")
        if self.declaration.sampling_change == "train32_to_testdense" and (
            self.expected_training_clips != 32 or self.expected_evaluation_clips is not None
        ):
            raise ValueError(
                "train32_to_testdense requires expected_training_clips=32 and "
                "expected_evaluation_clips=null"
            )
        declared_head = self.declaration.training_identity.head
        declared_kind = str(declared_head.get("kind", "")).strip().lower().replace("-", "_")
        allowed_kinds = {
            "topk": {"topk", "top_k", "topk_mil", "top_k_mil"},
            "attention": {"attention", "attention_mil"},
        }
        if declared_kind not in allowed_kinds[self.head]:
            raise ValueError("detector head differs from compatibility training identity")
        if "k" in declared_head and self.head_kwargs.get("k") != declared_head["k"]:
            raise ValueError("detector TopK budget differs from compatibility training identity")
        declared_optimization = self.declaration.training_identity.optimization
        expected_values = {
            "epochs": self.epochs,
            "lr": self.learning_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        for name, actual in expected_values.items():
            if name in declared_optimization and declared_optimization[name] != actual:
                raise ValueError(f"detector {name} differs from compatibility training identity")
        if self.declaration.training_identity.seed != self.seed:
            raise ValueError("detector seed differs from compatibility training identity")
        object.__setattr__(self, "head_kwargs", dict(self.head_kwargs))

    def training_config(self) -> HeadOnlyTrainingConfig:
        return HeadOnlyTrainingConfig(
            task="weak_mil",
            feature_level="clip",
            head=self.head,
            head_kwargs=self.head_kwargs,
            task_kwargs={"ranking_weight": 0.0},
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_steps=self.max_steps,
            seed=self.seed,
            expected_clips=self.expected_training_clips,
            verify_training=True,
            # train_detector verifies this paper FeatureStore snapshot before
            # entering the runner; cache its fully validated video sequences.
            cache_sequences=True,
        )


def _train_records(value: Any) -> tuple[VideoManifestRecord, ...]:
    records = validate_manifest(value)
    if any(item.split != DatasetSplit.TRAIN for item in records):
        raise ValueError("detector training may use only train videos")
    labels = {item.is_anomaly for item in records}
    if labels != {False, True}:
        raise ValueError("detector training requires both positive and negative video bags")
    return records


def _validation_records(value: Any | None) -> tuple[VideoManifestRecord, ...] | None:
    if value is None:
        return None
    records = validate_manifest(value)
    if any(item.split == DatasetSplit.TEST for item in records):
        raise ValueError("detector validation must not use test videos or test truth")
    return records


def _verify_feature_identity(
    feature_store: FeatureStore | str | Path,
    records: tuple[VideoManifestRecord, ...],
    *,
    encoder_fingerprint: str,
    representation: Any,
    sampling: Any,
) -> None:
    """Check row identities and any run-scoped runtime evidence before use."""

    store = (
        feature_store if isinstance(feature_store, FeatureStore) else FeatureStore(feature_store)
    )
    required_ids = {item.video_id for item in records}
    matching = [
        item
        for item in store.iter_records()
        if item.video_id in required_ids and item.encoder_fingerprint == encoder_fingerprint
    ]
    if not matching:
        raise ValueError("no FeatureStore rows match the declared detector identity")
    expected = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    runtime_id = representation.backbone.runtime_id
    resolved_digest = None
    for item in matching:
        identity = item.metadata.get("paper_identity")
        if not isinstance(identity, Mapping) or dict(identity) != expected:
            raise ValueError(
                f"{item.video_id}/{item.clip_id}: FeatureStore paper identity does not match "
                "the declared representation and sampling"
            )
        if "runtime_reference" not in item.metadata:
            continue  # Older inline records retain their existing identity contract.
        reference = item.metadata["runtime_reference"]
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"base", "path", "sha256"}
            or reference["base"] != "extraction_run"
            or reference["path"] != "resolved.json"
        ):
            raise ValueError("runtime_reference must name extraction_run/resolved.json exactly")
        declared_digest = reference["sha256"]
        if (
            not isinstance(declared_digest, str)
            or len(declared_digest) != 64
            or any(char not in "0123456789abcdef" for char in declared_digest)
        ):
            raise ValueError("runtime_reference.sha256 must be a lowercase SHA-256")
        if resolved_digest is None:
            # Only this fixed, run-local file is eligible. Read/hash one byte
            # snapshot once for all clips, including mixed reference digests.
            resolved_path = (store.root / "resolved.json").resolve()
            if resolved_path.parent != store.root:
                raise ValueError("runtime_reference resolved.json escapes the extraction run")
            content = resolved_path.read_bytes()
            resolved_digest = hashlib.sha256(content).hexdigest()
            if resolved_digest != declared_digest:
                raise ValueError("runtime_reference SHA-256 differs from resolved.json bytes")
            resolved = json.loads(content.decode("utf-8"))
            if not isinstance(resolved, Mapping):
                raise ValueError("runtime_reference resolved.json must contain an object")
            spec = resolved.get("spec")
            runtime = resolved.get("runtime")
            if not isinstance(spec, Mapping) or not isinstance(runtime, Mapping):
                raise ValueError("runtime_reference resolved.json lacks extraction spec/runtime")
            if resolved.get("encoder_fingerprint") != encoder_fingerprint:
                raise ValueError("runtime_reference resolved encoder fingerprint does not match the rows")
            if resolved.get("paper_identity") != expected:
                raise ValueError("runtime_reference resolved paper identity does not match the detector")
            if spec.get("runtime_id") != runtime_id or runtime.get("runtime_id") != runtime_id:
                raise ValueError("runtime_reference resolved runtime_id differs from the representation")
            if (
                runtime.get("representation_fingerprint") != representation.fingerprint
                or runtime.get("sampling_fingerprint") != sampling.fingerprint
            ):
                raise ValueError("runtime_reference resolved runtime fingerprints do not match the detector")
            if not isinstance(spec.get("representation"), Mapping) or not isinstance(spec.get("sampling"), Mapping):
                raise ValueError("runtime_reference resolved spec lacks representation/sampling identities")
            resolved_representation = RepresentationIdentity.from_mapping(spec["representation"])
            resolved_sampling = SamplingIdentity.from_mapping(spec["sampling"])
            if (
                resolved_representation.fingerprint != representation.fingerprint
                or resolved_sampling.fingerprint != sampling.fingerprint
            ):
                raise ValueError("runtime_reference resolved representation/sampling content does not match the detector")
        if declared_digest != resolved_digest:
            raise ValueError("runtime_reference SHA-256 differs from resolved.json bytes")
        row_runtime = item.metadata.get("runtime")
        if not isinstance(row_runtime, Mapping) or row_runtime.get("runtime_id") != runtime_id:
            raise ValueError("runtime_reference row runtime_id differs from the representation")


def train_detector(
    config: DetectionConfig,
    *,
    feature_store: FeatureStore | str | Path,
    train_manifest: Any,
    validation_manifest: Any | None = None,
    validation_feature_store: FeatureStore | str | Path | None = None,
    validation_encoder_fingerprint: str | None = None,
    validation_sampling: Any | None = None,
    output_dir: str | Path,
    device: Any | None = None,
) -> TrainingRunResult:
    """Train the shared BCE-only weak-MIL head on the declared fit features."""

    if not isinstance(config, DetectionConfig):
        raise TypeError("config must be a DetectionConfig")
    train_records = _train_records(train_manifest)
    validation_records = _validation_records(validation_manifest)
    actual_fit_digest = "sha256:" + compute_manifest_sha256(train_records)
    if config.declaration.training_identity.fit_split_digest != actual_fit_digest:
        raise ValueError("TrainingIdentity.fit_split_digest does not match the actual train manifest")
    _verify_feature_identity(
        feature_store,
        train_records,
        encoder_fingerprint=config.training_encoder_fingerprint,
        representation=config.declaration.training_representation,
        sampling=config.declaration.training_sampling,
    )
    if validation_records is not None:
        if validation_encoder_fingerprint is None or validation_sampling is None:
            raise ValueError("validation manifest requires its explicit feature store fingerprint and sampling")
        training_policy = config.declaration.training_sampling.to_dict()
        validation_policy = validation_sampling.to_dict()
        training_policy.pop("source_digest")
        validation_policy.pop("source_digest")
        if validation_policy != training_policy:
            raise ValueError("validation sampling policy must match the training sampling policy")
        _verify_feature_identity(
            feature_store if validation_feature_store is None else validation_feature_store,
            validation_records,
            encoder_fingerprint=validation_encoder_fingerprint,
            representation=config.declaration.training_representation,
            sampling=validation_sampling,
        )
    result = train_feature_head(
        config.training_config(),
        feature_store=feature_store,
        train_manifest=train_records,
        validation_manifest=validation_records,
        output_dir=output_dir,
        encoder_fingerprint=config.training_encoder_fingerprint,
        device=device,
        checkpoint_metadata={
            "paper_detector": _source_checkpoint_identity(config.declaration),
        },
        validation_feature_store=validation_feature_store,
        validation_encoder_fingerprint=validation_encoder_fingerprint,
    )
    if result.encoder_fingerprint != config.training_encoder_fingerprint:
        raise RuntimeError("training returned an unexpected FeatureStore fingerprint")
    return result


def issue_prediction_permit(
    config: DetectionConfig,
    training: TrainingRunResult | str | Path,
) -> PredictionCompatibilityPermit:
    """Bind an approved target feature identity to one trained source head."""

    if not isinstance(config, DetectionConfig):
        raise TypeError("config must be a DetectionConfig")
    if isinstance(training, TrainingRunResult):
        if training.encoder_fingerprint != config.training_encoder_fingerprint:
            raise ValueError("training result fingerprint differs from detector configuration")
        checkpoint_path = training.checkpoint_path
    else:
        checkpoint_path = training
    return PredictionCompatibilityPermit.issue(
        config.declaration,
        checkpoint_path=checkpoint_path,
        source_encoder_fingerprint=config.training_encoder_fingerprint,
        target_encoder_fingerprint=config.evaluation_encoder_fingerprint,
        evaluation_expected_clips=config.expected_evaluation_clips,
    )


def predict_detector(
    config: DetectionConfig,
    *,
    feature_store: FeatureStore | str | Path,
    evaluation_manifest: Any,
    training: TrainingRunResult | str | Path,
    output_path: str | Path,
    device: Any | None = None,
) -> list[Any]:
    """Predict dense features through a bound permit and frame aggregation."""

    if not isinstance(config, DetectionConfig):
        raise TypeError("config must be a DetectionConfig")
    evaluation_records = _prediction_records(evaluation_manifest)
    _verify_feature_identity(
        feature_store,
        evaluation_records,
        encoder_fingerprint=config.evaluation_encoder_fingerprint,
        representation=config.declaration.evaluation_representation,
        sampling=config.declaration.evaluation_sampling,
    )
    permit = issue_prediction_permit(config, training)
    checkpoint_path = (
        training.checkpoint_path if isinstance(training, TrainingRunResult) else training
    )
    return predict_feature_head(
        config.training_config(),
        feature_store,
        evaluation_records,
        checkpoint_path,
        output_path,
        device=device,
        compatibility_permit=permit,
        overlap_reduction=config.overlap_reduction,
    )


def evaluate_detector(
    records: Any,
    manifest: Any,
    *,
    protocol: Literal["official", "subset", "generic"],
    audit_report: Any | None = None,
) -> UCFEvaluationResult:
    """Use the existing audited evaluator; this module adds no parallel metric."""

    return evaluate_manifest_predictions(
        records,
        manifest,
        protocol=protocol,
        audit_report=audit_report,
    )


__all__ = [
    "DetectionConfig",
    "DetectorHead",
    "OverlapReduction",
    "PredictionCompatibilityPermit",
    "evaluate_detector",
    "issue_prediction_permit",
    "predict_detector",
    "train_detector",
]
