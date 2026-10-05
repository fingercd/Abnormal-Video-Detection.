"""Repeat frozen XD detector heads on their already verified fit/select stores.

This module never opens an XD test FeatureStore.  It creates only a fresh
seed-1/2 head checkpoint from a completed, contract-verified XD controller
source.  Cached-test scoring lives in :mod:`xd_repeat_evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vadbench.artifacts import new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.manifest import load_manifest_jsonl
from vadbench.features import atomic_write_json
from vadbench.paper import evaluation
from vadbench.paper import xd_evaluation
from vadbench.data.feature_contracts import (
    CompatibilityDeclaration,
    SamplingIdentity,
    TrainingIdentity,
)
from vadbench.workflows.detection import DetectionConfig, train_detector
from vadbench.paper.repeat_evaluation import FrozenRepeatHeadSource, _require_qa

_REPEAT_REDUCERS = {"identity", "paired_random", "pair_linear"}


@dataclass(frozen=True)
class FrozenXDHeadRepeatRequest:
    encoder: str
    source_controller_run: str
    output_root: str
    seed: int
    role_lock_path: str
    head_data_contract_path: str
    head_data_contract_sha256: str
    source_manifest_root: str
    original_role_lock_path: str
    scope_path: str = "projects/icassp2027/decisions/xd-validation-scope-v1.json"
    freeze_path: str = "projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json"
    device: str = "cpu"
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.seed not in {1, 2}:
            raise ValueError("frozen XD head repeats permit only seed 1 or 2")


@dataclass(frozen=True)
class FrozenXDHeadRepeatResult:
    run_dir: str
    checkpoint_path: str
    checkpoint_sha256: str
    seed: int


class FrozenXDRepeatHeadSource(FrozenRepeatHeadSource):
    """A standard repeat checkpoint with an XD-specific provenance receipt."""

    def receipt(self) -> dict[str, Any]:
        receipt = super().receipt()
        receipt["kind"] = "frozen_xd_head_repeat_v1"
        receipt["dataset"] = "xd_violence"
        return receipt


def _source_kwargs(request: FrozenXDHeadRepeatRequest) -> dict[str, str]:
    return {
        "role_lock_path": request.role_lock_path,
        "head_data_contract_path": request.head_data_contract_path,
        "source_manifest_root": request.source_manifest_root,
        "scope_path": request.scope_path,
        "original_role_lock_path": request.original_role_lock_path,
        "method_freeze_path": request.freeze_path,
        "head_data_contract_sha256": request.head_data_contract_sha256,
    }


def _stage(root: Path) -> tuple[dict[str, Any], Path]:
    matches = []
    for path in sorted((root / "provenance" / "stages").glob("*.json")):
        document = evaluation._json(path, name="XD repeat-head stage")
        if (
            document.get("stage") == "frozen_xd_head_repeat"
            and document.get("status") == "completed"
        ):
            matches.append((document, path))
    if len(matches) != 1:
        raise ValueError("XD repeat head run must contain exactly one completed repeat stage")
    return matches[0]


def _source_config(source: Any, *, seed: int) -> DetectionConfig:
    config = evaluation._mapping(
        source.checkpoint_metadata.get("config"), "source checkpoint config"
    )
    if config.get("head") != "topk" or config.get("expected_clips") != 32:
        raise ValueError("XD source checkpoint does not use the frozen TopKMIL/32-bag protocol")
    head_kwargs = evaluation._mapping(config.get("head_kwargs"), "source checkpoint head_kwargs")
    training = TrainingIdentity(
        source.training_identity.head,
        source.training_identity.fit_split_digest,
        seed,
        source.training_identity.optimization,
    )
    return DetectionConfig(
        declaration=CompatibilityDeclaration(
            "refit_head",
            source.representation,
            source.representation,
            source.sampling,
            source.sampling,
            source.sampling,
            training,
            "none",
        ),
        training_encoder_fingerprint=source.encoder_fingerprint,
        evaluation_encoder_fingerprint=source.encoder_fingerprint,
        head="topk",
        head_kwargs=head_kwargs,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        seed=seed,
        expected_training_clips=32,
        expected_evaluation_clips=None,
    )


def run_frozen_xd_head_repeat(request: FrozenXDHeadRepeatRequest) -> FrozenXDHeadRepeatResult:
    freeze, freeze_sha = evaluation._load_freeze(request.freeze_path)
    if evaluation._canonical_sha256(freeze) != xd_evaluation.METHOD_FREEZE_CANONICAL_SHA256:
        raise ValueError("XD head repeat requires the unchanged frozen method definition")
    source = xd_evaluation.load_frozen_xd_detector_source(
        request.source_controller_run,
        freeze=freeze,
        expected_encoder=request.encoder,
        **_source_kwargs(request),
    )
    if source.reducer.get("name") not in _REPEAT_REDUCERS:
        raise ValueError(
            "XD multi-seed head repeats permit only identity, paired_random, and pair_linear"
        )
    root = Path(request.output_root).expanduser().resolve()
    run_dir = root / (request.run_id or new_run_id("frozen-xd-head-repeat"))
    if run_dir.exists():
        raise FileExistsError(f"XD repeat head run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    source.verify_unchanged()
    detector = _source_config(source, seed=request.seed)
    inputs = {
        "source_checkpoint": source.checkpoint,
        "source_train_manifest": source.train_manifest,
        "source_validation_manifest": source.validation_manifest,
        "method_freeze": request.freeze_path,
        "xd_scope": request.scope_path,
        "training_role_lock": source.role_lock_path,
        "xd_original_role_lock": request.original_role_lock_path,
        "head_data_contract": source.head_data_contract_path,
    }
    with record_stage(
        run_dir,
        "frozen_xd_head_repeat",
        config=request.__dict__,
        inputs=inputs,
        project_root=Path.cwd(),
    ):
        if source.validation_manifest is None or source.validation_feature_store is None:
            raise ValueError("XD repeat head source lacks its required select FeatureStore")
        validation_resolved = evaluation._json(
            source.validation_feature_store / "resolved.json",
            name="XD select FeatureStore resolution",
        )
        validation_sampling = SamplingIdentity.from_mapping(
            evaluation._mapping(validation_resolved.get("spec"), "XD select FeatureStore spec")[
                "sampling"
            ]
        )
        trained = train_detector(
            detector,
            feature_store=source.train_feature_store,
            train_manifest=load_manifest_jsonl(source.train_manifest),
            validation_manifest=load_manifest_jsonl(source.validation_manifest),
            validation_feature_store=source.validation_feature_store,
            validation_encoder_fingerprint=str(validation_resolved["encoder_fingerprint"]),
            validation_sampling=validation_sampling,
            output_dir=run_dir / "head",
            device=request.device,
        )
        source.verify_unchanged()
        checkpoint = Path(trained.checkpoint_path)
        atomic_write_json(
            run_dir / "result.json",
            {
                "schema_version": 1,
                "status": "completed",
                "dataset": "xd_violence",
                "kind": "frozen_xd_head_repeat_v1",
                "seed": request.seed,
                "freeze_sha256": freeze_sha,
                "source_controller_run": str(source.root),
                "source_artifact_sha256": dict(source.source_hashes),
                "training_role_lock": {
                    "path": str(source.role_lock_path),
                    "sha256": source.role_lock_sha256,
                },
                "head_data_contract": {
                    "path": str(source.head_data_contract_path),
                    "sha256": source.head_data_contract_sha256,
                },
                "xd_scope": {
                    "path": str(Path(request.scope_path).resolve()),
                    "sha256": evaluation._digest(Path(request.scope_path), name="XD scope"),
                },
                "xd_original_role_lock": {
                    "path": str(Path(request.original_role_lock_path).resolve()),
                    "sha256": evaluation._digest(
                        Path(request.original_role_lock_path), name="XD original role lock"
                    ),
                },
                "source_manifest_root": str(source.source_manifest_root),
                "source_manifests_sha256": dict(source.source_manifest_sha256),
                "training_representation_fingerprint": source.representation.fingerprint,
                "training_sampling_fingerprint": source.sampling.fingerprint,
                "training_identity_fingerprint": detector.declaration.training_identity.fingerprint,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_sidecar_sha256": sha256_file(checkpoint.with_suffix(".pt.json")),
                "training_qa_path": str(run_dir / "head" / "training_qa.json"),
                "training_qa_sha256": sha256_file(run_dir / "head" / "training_qa.json"),
            },
        )
    return FrozenXDHeadRepeatResult(
        str(run_dir), str(checkpoint), sha256_file(checkpoint), request.seed
    )


def load_frozen_xd_repeat_head_source(
    path: str | Path,
    *,
    freeze: dict[str, Any],
    expected_encoder: str,
    role_lock_path: str,
    head_data_contract_path: str,
    source_manifest_root: str,
    scope_path: str,
    original_role_lock_path: str,
    method_freeze_path: str,
    head_data_contract_sha256: str,
) -> FrozenXDRepeatHeadSource:
    root = Path(path).expanduser().resolve()
    result_path, qa_path = root / "result.json", root / "head" / "training_qa.json"
    checkpoint, sidecar = (
        root / "head" / "checkpoints" / "final.pt",
        root / "head" / "checkpoints" / "final.pt.json",
    )
    stage, stage_path = _stage(root)
    result = evaluation._json(result_path, name="XD repeat head result")
    qa = evaluation._json(qa_path, name="XD repeat head QA")
    sidecar_doc = evaluation._json(sidecar, name="XD repeat checkpoint sidecar")
    if (
        result.get("status") != "completed"
        or result.get("kind") != "frozen_xd_head_repeat_v1"
        or result.get("seed") not in {1, 2}
    ):
        raise ValueError("XD repeat head result is not a completed permitted seed-1/2 run")
    if (
        stage.get("config", {}).get("encoder") != expected_encoder
        or stage.get("config", {}).get("seed") != result["seed"]
    ):
        raise ValueError("XD repeat head stage differs from the requested encoder/seed")
    base = xd_evaluation.load_frozen_xd_detector_source(
        result.get("source_controller_run", ""),
        freeze=freeze,
        expected_encoder=expected_encoder,
        role_lock_path=role_lock_path,
        head_data_contract_path=head_data_contract_path,
        source_manifest_root=source_manifest_root,
        scope_path=scope_path,
        original_role_lock_path=original_role_lock_path,
        method_freeze_path=method_freeze_path,
        head_data_contract_sha256=head_data_contract_sha256,
    )
    if result.get("source_artifact_sha256") != dict(base.source_hashes):
        raise ValueError(
            "XD repeat head result source artifacts differ from the verified controller source"
        )
    if (
        result.get("training_role_lock")
        != {"path": str(base.role_lock_path), "sha256": base.role_lock_sha256}
        or result.get("head_data_contract")
        != {
            "path": str(base.head_data_contract_path),
            "sha256": base.head_data_contract_sha256,
        }
        or result.get("source_manifest_root") != str(base.source_manifest_root)
        or result.get("source_manifests_sha256") != dict(base.source_manifest_sha256)
    ):
        raise ValueError("XD repeat head result data-contract sources differ from its controller")
    for name, value in (
        ("xd_scope", Path(scope_path)),
        ("xd_original_role_lock", Path(original_role_lock_path)),
    ):
        expected = {
            "path": str(value.resolve()),
            "sha256": evaluation._digest(value, name=f"XD repeat {name}"),
        }
        if result.get(name) != expected:
            raise ValueError(f"XD repeat head result {name} differs from its verified source")
    metadata = evaluation._mapping(sidecar_doc.get("metadata"), "XD repeat checkpoint metadata")
    if (
        sidecar_doc.get("sha256") != sha256_file(checkpoint)
        or evaluation._checkpoint_metadata_bytes(checkpoint) != metadata
    ):
        raise ValueError("XD repeat checkpoint metadata differs from its checksum-bound bytes")
    training = TrainingIdentity(
        base.training_identity.head,
        base.training_identity.fit_split_digest,
        int(result["seed"]),
        base.training_identity.optimization,
    )
    detector = evaluation._mapping(
        metadata.get("paper_detector"), "XD repeat checkpoint paper_detector"
    )
    if (
        metadata.get("status") != "completed"
        or metadata.get("encoder_fingerprint") != base.encoder_fingerprint
        or detector.get("training_identity_fingerprint") != training.fingerprint
        or result.get("training_identity_fingerprint") != training.fingerprint
    ):
        raise ValueError("XD repeat head training identity differs from its controller source")
    _require_qa(qa, metadata, checkpoint)
    hashes = {
        "result.json": evaluation._digest(result_path, name="XD repeat head result"),
        "head/training_qa.json": evaluation._digest(qa_path, name="XD repeat head QA"),
        "head/checkpoints/final.pt": evaluation._digest(checkpoint, name="XD repeat checkpoint"),
        "head/checkpoints/final.pt.json": evaluation._digest(
            sidecar, name="XD repeat checkpoint sidecar"
        ),
        str(stage_path.relative_to(root)).replace("\\", "/"): evaluation._digest(
            stage_path, name="XD repeat head stage"
        ),
    }
    if result.get("checkpoint_sha256") != hashes["head/checkpoints/final.pt"] or result.get(
        "checkpoint_path"
    ) != str(checkpoint):
        raise ValueError("XD repeat head result does not bind its final checkpoint")
    return FrozenXDRepeatHeadSource(root, checkpoint, metadata, base, training, hashes)


__all__ = [
    "FrozenXDHeadRepeatRequest",
    "FrozenXDHeadRepeatResult",
    "FrozenXDRepeatHeadSource",
    "load_frozen_xd_repeat_head_source",
    "run_frozen_xd_head_repeat",
]
