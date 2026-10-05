"""Frozen seed-1/2 detector-head repeats over an existing controller FeatureStore."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vadbench.artifacts import new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.manifest import load_manifest_jsonl
from vadbench.features import atomic_write_json
from vadbench.data.feature_contracts import (
    CompatibilityDeclaration,
    SamplingIdentity,
    TrainingIdentity,
)
from vadbench.workflows.detection import DetectionConfig, train_detector
from vadbench.paper.evaluation import _load_freeze, load_frozen_detector_source


@dataclass(frozen=True)
class FrozenHeadRepeatRequest:
    encoder: str
    source_controller_run: str
    output_root: str
    seed: int
    freeze_path: str = "projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json"
    role_lock_path: str = (
        "outputs/icassp2027/assets/gate-calibration-20260918T083000Z/frozen-partition-lock.json"
    )
    head_data_contract_path: str = "projects/icassp2027/decisions/head-data-contract-v1.json"
    source_manifest_root: str = "outputs/icassp2027/assets/full-ucf-head-data-contract-20260918"
    device: str = "cpu"
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.seed not in {1, 2}:
            raise ValueError("frozen head repeats permit only seed 1 or 2")


@dataclass(frozen=True)
class FrozenHeadRepeatResult:
    run_dir: str
    checkpoint_path: str
    checkpoint_sha256: str
    seed: int


def run_frozen_head_repeat(request: FrozenHeadRepeatRequest) -> FrozenHeadRepeatResult:
    freeze, freeze_sha = _load_freeze(request.freeze_path)
    allowed = set(freeze["detection_head"]["key_configuration_seeds"])
    if request.seed not in allowed:
        raise ValueError("requested seed is not in the frozen key-configuration seeds")
    source = load_frozen_detector_source(
        request.source_controller_run,
        freeze=freeze,
        expected_encoder=request.encoder,
        role_lock_path=request.role_lock_path,
        head_data_contract_path=request.head_data_contract_path,
        source_manifest_root=request.source_manifest_root,
    )
    if source.reducer.get("name") not in {"identity", "paired_random", "pair_linear"}:
        raise ValueError(
            "only dense, paired_random, and pair_linear have frozen multi-seed head repeats"
        )
    root = Path(request.output_root).expanduser().resolve()
    run_dir = root / (request.run_id or new_run_id("frozen-head-repeat"))
    if run_dir.exists():
        raise FileExistsError(f"head repeat run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    source.verify_unchanged()
    training = TrainingIdentity(
        source.training_identity.head,
        source.training_identity.fit_split_digest,
        request.seed,
        source.training_identity.optimization,
    )
    declaration = CompatibilityDeclaration(
        "refit_head",
        source.representation,
        source.representation,
        source.sampling,
        source.sampling,
        source.sampling,
        training,
        "none",
    )
    config = DetectionConfig(
        declaration,
        source.encoder_fingerprint,
        source.encoder_fingerprint,
        head="topk",
        head_kwargs={"k": 3},
        epochs=20,
        batch_size=16,
        learning_rate=0.001,
        weight_decay=0.0,
        seed=request.seed,
        expected_training_clips=32,
        expected_evaluation_clips=None,
    )
    inputs = {
        "source_train_manifest": source.train_manifest,
        "source_checkpoint": source.checkpoint,
        "freeze": request.freeze_path,
        "training_role_lock": request.role_lock_path,
        "head_data_contract": request.head_data_contract_path,
        "source_validation_manifest": source.validation_manifest,
    }
    with record_stage(
        run_dir,
        "frozen_head_repeat",
        config=request.__dict__,
        inputs=inputs,
        project_root=Path.cwd(),
    ):
        validation_sampling = validation_fingerprint = validation_manifest = validation_store = None
        if source.validation_manifest is not None and source.validation_feature_store is not None:
            validation_manifest = load_manifest_jsonl(source.validation_manifest)
            validation_store = source.validation_feature_store
            resolved = json.loads((validation_store / "resolved.json").read_text(encoding="utf-8"))
            validation_sampling = SamplingIdentity.from_mapping(resolved["spec"]["sampling"])
            validation_fingerprint = resolved["encoder_fingerprint"]
        trained = train_detector(
            config,
            feature_store=source.train_feature_store,
            train_manifest=load_manifest_jsonl(source.train_manifest),
            validation_manifest=validation_manifest,
            validation_feature_store=validation_store,
            validation_encoder_fingerprint=validation_fingerprint,
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
                "source_manifest_root": str(source.source_manifest_root),
                "source_manifests_sha256": dict(source.source_manifest_sha256),
                "training_representation_fingerprint": source.representation.fingerprint,
                "training_sampling_fingerprint": source.sampling.fingerprint,
                "training_identity_fingerprint": training.fingerprint,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_sidecar_sha256": sha256_file(checkpoint.with_suffix(".pt.json")),
                "training_qa_path": str(run_dir / "head" / "training_qa.json"),
                "training_qa_sha256": sha256_file(run_dir / "head" / "training_qa.json"),
            },
        )
    return FrozenHeadRepeatResult(
        str(run_dir), str(checkpoint), sha256_file(checkpoint), request.seed
    )


__all__ = ["FrozenHeadRepeatRequest", "FrozenHeadRepeatResult", "run_frozen_head_repeat"]
