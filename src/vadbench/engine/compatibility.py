"""Checkpoint-bound permits for explicitly reviewed feature pairings.

The on-disk ``paper_detector`` and ``target_paper_identity`` field names are
retained as versioned artifact fields. No project configuration is imported.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vadbench.checkpoints import sha256_file
from vadbench.data.feature_contracts import (
    CompatibilityDeclaration,
    feature_cache_key,
    validate_compatibility,
)

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

