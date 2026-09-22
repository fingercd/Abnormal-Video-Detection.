"""Explicit paper-planning identities and narrow compatibility declarations.

This module validates *planned* feature/head pairings before a paper run.  It
does not alter checkpoint metadata and is deliberately not wired into the
existing prediction path: that path continues to require an exact encoder
fingerprint.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from vadbench.features import compute_encoder_fingerprint


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        # Round-tripping both checks JSON compatibility and breaks aliases to
        # caller-owned mutable mappings.
        normalized = json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain JSON-compatible values") from exc
    if not isinstance(normalized, dict):  # Defensive only for unusual Mapping subclasses.
        raise TypeError(f"{name} must be an object")
    return normalized


def _fields(
    value: Mapping[str, Any],
    *,
    name: str,
    required: set[str],
) -> dict[str, Any]:
    copied = dict(value)
    missing = sorted(required - copied.keys())
    unknown = sorted(copied.keys() - required)
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")
    return copied


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class BackboneIdentity:
    """The immutable encoder portion shared by dense and reduced features."""

    runtime_id: str
    weights_digest: str
    code_digest: str
    preprocessing: Mapping[str, Any]
    readout: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_id", _text(self.runtime_id, "runtime_id"))
        object.__setattr__(self, "weights_digest", _text(self.weights_digest, "weights_digest"))
        object.__setattr__(self, "code_digest", _text(self.code_digest, "code_digest"))
        object.__setattr__(
            self, "preprocessing", _json_mapping(self.preprocessing, "preprocessing")
        )
        object.__setattr__(self, "readout", _json_mapping(self.readout, "readout"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BackboneIdentity:
        fields = _fields(
            value,
            name="backbone_identity",
            required={"runtime_id", "weights_digest", "code_digest", "preprocessing", "readout"},
        )
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "weights_digest": self.weights_digest,
            "code_digest": self.code_digest,
            "preprocessing": dict(self.preprocessing),
            "readout": dict(self.readout),
        }

    @property
    def fingerprint(self) -> str:
        return compute_encoder_fingerprint({"paper_backbone_identity": self.to_dict()})


@dataclass(frozen=True)
class RepresentationIdentity:
    """Backbone output plus the exact reducer, precision and position policy."""

    backbone: BackboneIdentity
    reducer: Mapping[str, Any]
    output_dim: int
    precision: str
    position_strategy: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, BackboneIdentity):
            raise TypeError("backbone must be a BackboneIdentity")
        object.__setattr__(self, "reducer", _json_mapping(self.reducer, "reducer"))
        object.__setattr__(self, "output_dim", _integer(self.output_dim, "output_dim"))
        object.__setattr__(self, "precision", _text(self.precision, "precision"))
        object.__setattr__(
            self, "position_strategy", _json_mapping(self.position_strategy, "position_strategy")
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RepresentationIdentity:
        fields = _fields(
            value,
            name="representation_identity",
            required={"backbone", "reducer", "output_dim", "precision", "position_strategy"},
        )
        backbone = fields["backbone"]
        if isinstance(backbone, Mapping):
            fields["backbone"] = BackboneIdentity.from_mapping(backbone)
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone.to_dict(),
            "reducer": dict(self.reducer),
            "output_dim": self.output_dim,
            "precision": self.precision,
            "position_strategy": dict(self.position_strategy),
        }

    @property
    def fingerprint(self) -> str:
        return compute_encoder_fingerprint({"paper_representation_identity": self.to_dict()})


@dataclass(frozen=True)
class SamplingIdentity:
    """Provenance for source video selection and the full sampling/projection path."""

    source_digest: str
    regime: str
    frame_selection: Mapping[str, Any]
    window: Mapping[str, Any]
    stride: Mapping[str, Any]
    padding: Mapping[str, Any]
    projection: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_digest", _text(self.source_digest, "source_digest"))
        object.__setattr__(self, "regime", _text(self.regime, "regime"))
        for field_name in ("frame_selection", "window", "stride", "padding", "projection"):
            object.__setattr__(
                self, field_name, _json_mapping(getattr(self, field_name), field_name)
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SamplingIdentity:
        fields = _fields(
            value,
            name="sampling_identity",
            required={
                "source_digest",
                "regime",
                "frame_selection",
                "window",
                "stride",
                "padding",
                "projection",
            },
        )
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_digest": self.source_digest,
            "regime": self.regime,
            "frame_selection": dict(self.frame_selection),
            "window": dict(self.window),
            "stride": dict(self.stride),
            "padding": dict(self.padding),
            "projection": dict(self.projection),
        }

    @property
    def fingerprint(self) -> str:
        return compute_encoder_fingerprint({"paper_sampling_identity": self.to_dict()})


@dataclass(frozen=True)
class TrainingIdentity:
    """Head fit provenance.  It is recorded, never inferred from a checkpoint."""

    head: Mapping[str, Any]
    fit_split_digest: str
    seed: int
    optimization: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "head", _json_mapping(self.head, "head"))
        object.__setattr__(
            self, "fit_split_digest", _text(self.fit_split_digest, "fit_split_digest")
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        object.__setattr__(self, "optimization", _json_mapping(self.optimization, "optimization"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrainingIdentity:
        fields = _fields(
            value,
            name="training_identity",
            required={"head", "fit_split_digest", "seed", "optimization"},
        )
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": dict(self.head),
            "fit_split_digest": self.fit_split_digest,
            "seed": self.seed,
            "optimization": dict(self.optimization),
        }

    @property
    def fingerprint(self) -> str:
        return compute_encoder_fingerprint({"paper_training_identity": self.to_dict()})


@dataclass(frozen=True)
class FeatureCacheIdentity:
    """The cache namespace; it intentionally includes both representation and sampling."""

    representation: RepresentationIdentity
    sampling: SamplingIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.representation, RepresentationIdentity):
            raise TypeError("representation must be a RepresentationIdentity")
        if not isinstance(self.sampling, SamplingIdentity):
            raise TypeError("sampling must be a SamplingIdentity")

    def to_dict(self) -> dict[str, str]:
        return {
            "representation_fingerprint": self.representation.fingerprint,
            "sampling_fingerprint": self.sampling.fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return compute_encoder_fingerprint({"paper_feature_cache_identity": self.to_dict()})


def feature_cache_key(representation: RepresentationIdentity, sampling: SamplingIdentity) -> str:
    """Return the cache key that prevents cross-representation/sampling reuse."""

    return FeatureCacheIdentity(representation, sampling).fingerprint


SamplingChange = Literal["none", "train32_to_testdense"]
CompatibilityMode = Literal["refit_head", "direct_insert"]


@dataclass(frozen=True)
class CompatibilityDeclaration:
    """A narrow, explicit exception to exact feature/checkpoint identity.

    ``refit_head`` requires the training and evaluation representations to be
    identical.  ``direct_insert`` permits only the reducer to differ.  Neither
    mode changes the strict legacy prediction contract.
    """

    mode: CompatibilityMode
    training_representation: RepresentationIdentity
    evaluation_representation: RepresentationIdentity
    training_sampling: SamplingIdentity
    evaluation_sampling: SamplingIdentity
    baseline_evaluation_sampling: SamplingIdentity
    training_identity: TrainingIdentity
    sampling_change: SamplingChange

    def __post_init__(self) -> None:
        if self.mode not in {"refit_head", "direct_insert"}:
            raise ValueError("mode must be 'refit_head' or 'direct_insert'")
        if self.sampling_change not in {"none", "train32_to_testdense"}:
            raise ValueError("sampling_change must be 'none' or 'train32_to_testdense'")
        for field_name, expected in (
            ("training_representation", RepresentationIdentity),
            ("evaluation_representation", RepresentationIdentity),
            ("training_sampling", SamplingIdentity),
            ("evaluation_sampling", SamplingIdentity),
            ("baseline_evaluation_sampling", SamplingIdentity),
            ("training_identity", TrainingIdentity),
        ):
            if not isinstance(getattr(self, field_name), expected):
                raise TypeError(f"{field_name} must be a {expected.__name__}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CompatibilityDeclaration:
        fields = _fields(
            value,
            name="compatibility_declaration",
            required={
                "mode",
                "training_representation",
                "evaluation_representation",
                "training_sampling",
                "evaluation_sampling",
                "baseline_evaluation_sampling",
                "training_identity",
                "sampling_change",
            },
        )
        nested = {
            "training_representation": RepresentationIdentity,
            "evaluation_representation": RepresentationIdentity,
            "training_sampling": SamplingIdentity,
            "evaluation_sampling": SamplingIdentity,
            "baseline_evaluation_sampling": SamplingIdentity,
            "training_identity": TrainingIdentity,
        }
        for field_name, identity_type in nested.items():
            if isinstance(fields[field_name], Mapping):
                fields[field_name] = identity_type.from_mapping(fields[field_name])
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "training_representation": self.training_representation.to_dict(),
            "evaluation_representation": self.evaluation_representation.to_dict(),
            "training_sampling": self.training_sampling.to_dict(),
            "evaluation_sampling": self.evaluation_sampling.to_dict(),
            "baseline_evaluation_sampling": self.baseline_evaluation_sampling.to_dict(),
            "training_identity": self.training_identity.to_dict(),
            "sampling_change": self.sampling_change,
        }


# 2026-09-22: code digest pairs proven to be the SAME adapter file content
# stored with different line endings (LF vs CRLF).  Each pair was verified
# byte-identical modulo CRLF/LF on both snapshots (diff empty, dual hashing of
# the same file).  Every other BackboneIdentity field and the
# output_dim/precision/position_strategy still have to match exactly.
_LINE_ENDING_EQUIVALENT_CODE_DIGESTS = frozenset(
    {
        (
            "sha256:81e3144071f0a2c0b3f7696a0c88c8853d7f225be7ef2a92ec458f7503e4f579",
            "sha256:30fe33c1cedd8fe065b2985514549b196241dfc51ebc37fa00025b5e5aada264",
        ),
        (
            "sha256:7f9a3ac6b5528cf0d02281a35a7d2093bf927de70421c1f57fc78f8ed3c9c4de",
            "sha256:e5fd0bdf4679f16622b5b3d3ea6dcf74ed0a6a1139607f3424e0cc515a879eed",
        ),
        (
            "sha256:9a23c5da46de06dd166b314207b85d72822048c6f3d25c492fd763d8e65bc999",
            "sha256:30fe33c1cedd8fe065b2985514549b196241dfc51ebc37fa00025b5e5aada264",
        ),
    }
)


def _same_direct_insert_contract(
    training: RepresentationIdentity, evaluation: RepresentationIdentity
) -> bool:
    if (
        training.output_dim != evaluation.output_dim
        or training.precision != evaluation.precision
        or training.position_strategy != evaluation.position_strategy
    ):
        return False
    training_backbone = training.backbone
    evaluation_backbone = evaluation.backbone
    if training_backbone == evaluation_backbone:
        return True
    if (
        training_backbone.runtime_id != evaluation_backbone.runtime_id
        or training_backbone.weights_digest != evaluation_backbone.weights_digest
        or training_backbone.preprocessing != evaluation_backbone.preprocessing
        or training_backbone.readout != evaluation_backbone.readout
    ):
        return False
    pair = (training_backbone.code_digest, evaluation_backbone.code_digest)
    return (
        pair in _LINE_ENDING_EQUIVALENT_CODE_DIGESTS
        or (pair[1], pair[0]) in _LINE_ENDING_EQUIVALENT_CODE_DIGESTS
    )


def validate_compatibility(declaration: CompatibilityDeclaration) -> dict[str, str]:
    """Validate a declared pairing and return its auditable identity summary.

    Calling this function is mandatory for an exception.  Without a declaration
    there is no implicit compatibility path.
    """

    if not isinstance(declaration, CompatibilityDeclaration):
        raise TypeError("a CompatibilityDeclaration is required")
    if declaration.evaluation_sampling != declaration.baseline_evaluation_sampling:
        raise ValueError("dense and compressed evaluation sampling identities must match exactly")
    if declaration.sampling_change == "none":
        train_sampling = declaration.training_sampling.to_dict()
        eval_sampling = declaration.evaluation_sampling.to_dict()
        train_sampling.pop("source_digest")
        eval_sampling.pop("source_digest")
        if train_sampling != eval_sampling:
            raise ValueError("sampling definitions differ without an explicit declaration")
    elif not (
        declaration.training_sampling.regime == "train_32"
        and declaration.evaluation_sampling.regime == "test_dense"
    ):
        raise ValueError(
            "train32_to_testdense requires training regime 'train_32' and evaluation regime "
            "'test_dense'"
        )
    if declaration.sampling_change == "train32_to_testdense":
        for name in ("short_video_policy", "implementation"):
            if declaration.training_sampling.frame_selection.get(name) != declaration.evaluation_sampling.frame_selection.get(name):
                raise ValueError(
                    f"train32_to_testdense must preserve the sampler contract: {name}"
                )
        for name in ("window", "stride", "padding", "projection"):
            if getattr(declaration.training_sampling, name) != getattr(
                declaration.evaluation_sampling, name
            ):
                raise ValueError(
                    f"train32_to_testdense must preserve the clip/projection contract: {name}"
                )

    if declaration.mode == "refit_head":
        if declaration.training_representation != declaration.evaluation_representation:
            raise ValueError(
                "refit_head requires identical training and evaluation representations"
            )
    elif not _same_direct_insert_contract(
        declaration.training_representation, declaration.evaluation_representation
    ):
        raise ValueError(
            "direct_insert may only change the reducer; backbone, output_dim, precision, and "
            "position_strategy must match"
        )

    return {
        "mode": declaration.mode,
        "training_representation_fingerprint": declaration.training_representation.fingerprint,
        "evaluation_representation_fingerprint": declaration.evaluation_representation.fingerprint,
        "training_sampling_fingerprint": declaration.training_sampling.fingerprint,
        "evaluation_sampling_fingerprint": declaration.evaluation_sampling.fingerprint,
        "baseline_evaluation_sampling_fingerprint": declaration.baseline_evaluation_sampling.fingerprint,
        "training_identity_fingerprint": declaration.training_identity.fingerprint,
        "evaluation_cache_fingerprint": feature_cache_key(
            declaration.evaluation_representation, declaration.evaluation_sampling
        ),
    }


__all__ = [
    "BackboneIdentity",
    "CompatibilityDeclaration",
    "CompatibilityMode",
    "FeatureCacheIdentity",
    "RepresentationIdentity",
    "SamplingChange",
    "SamplingIdentity",
    "TrainingIdentity",
    "feature_cache_key",
    "validate_compatibility",
]
