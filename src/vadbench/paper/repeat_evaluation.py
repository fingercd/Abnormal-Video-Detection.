"""Evaluate a frozen seed-1/2 detector head using a sealed seed-0 test cache."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vadbench.artifacts import new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.manifest import load_manifest_jsonl
from vadbench.features import atomic_write_json
from vadbench.paper import evaluation
from vadbench.data.feature_contracts import (
    CompatibilityDeclaration,
    TrainingIdentity,
    feature_cache_key,
)
from vadbench.workflows.detection import DetectionConfig, evaluate_detector, predict_detector


@dataclass(frozen=True)
class FrozenRepeatEvaluationRequest:
    encoder: str
    repeat_head_run: str
    source_evaluation_run: str
    output_root: str
    dense_head_repeat_run: str | None = None
    device: str = "cpu"
    freeze_path: str = "projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json"
    role_lock_path: str = (
        "outputs/icassp2027/assets/gate-calibration-20260918T083000Z/frozen-partition-lock.json"
    )
    head_data_contract_path: str = "projects/icassp2027/decisions/head-data-contract-v1.json"
    source_manifest_root: str = "outputs/icassp2027/assets/full-ucf-head-data-contract-20260918"
    run_id: str | None = None


@dataclass(frozen=True)
class FrozenRepeatEvaluationResult:
    run_dir: str
    completed: bool
    primary_predictions: str
    primary_metrics: str
    secondary_predictions: str | None
    secondary_metrics: str | None


@dataclass(frozen=True)
class FrozenRepeatHeadSource:
    root: Path
    checkpoint: Path
    checkpoint_metadata: Mapping[str, Any]
    base: evaluation.FrozenDetectorSource
    training_identity: TrainingIdentity
    artifact_sha256: Mapping[str, str]

    @property
    def representation(self):
        return self.base.representation

    @property
    def sampling(self):
        return self.base.sampling

    @property
    def encoder_fingerprint(self) -> str:
        return self.base.encoder_fingerprint

    @property
    def reducer(self):
        return self.base.reducer

    def verify_unchanged(self) -> None:
        self.base.verify_unchanged()
        for relative, expected in self.artifact_sha256.items():
            if (
                evaluation._digest(self.root / relative, name=f"repeat-head artifact {relative}")
                != expected
            ):
                raise ValueError(
                    f"repeat-head artifact changed after verification: {self.root / relative}"
                )

    def detection_config(
        self,
        *,
        mode: str,
        evaluation_representation,
        evaluation_sampling,
        evaluation_encoder_fingerprint: str,
    ) -> DetectionConfig:
        config = evaluation._mapping(
            self.checkpoint_metadata.get("config"), "repeat checkpoint metadata.config"
        )
        return DetectionConfig(
            declaration=CompatibilityDeclaration(
                mode=mode,  # type: ignore[arg-type]
                training_representation=self.representation,
                evaluation_representation=evaluation_representation,
                training_sampling=self.sampling,
                evaluation_sampling=evaluation_sampling,
                baseline_evaluation_sampling=evaluation_sampling,
                training_identity=self.training_identity,
                sampling_change="train32_to_testdense",
            ),
            training_encoder_fingerprint=self.encoder_fingerprint,
            evaluation_encoder_fingerprint=evaluation_encoder_fingerprint,
            head="topk",
            head_kwargs={"k": 3},
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            seed=self.training_identity.seed,
            expected_training_clips=32,
            expected_evaluation_clips=None,
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "kind": "frozen_head_repeat_v1",
            "run_dir": str(self.root),
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": self.artifact_sha256["head/checkpoints/final.pt"],
            "repeat_artifact_sha256": dict(self.artifact_sha256),
            "source_controller": evaluation._source_receipt(self.base),
            "training_representation_fingerprint": self.representation.fingerprint,
            "training_sampling_fingerprint": self.sampling.fingerprint,
            "training_encoder_fingerprint": self.encoder_fingerprint,
            "training_identity_fingerprint": self.training_identity.fingerprint,
            "head_seed": self.training_identity.seed,
            "head_budget": {
                key: self.checkpoint_metadata["config"][key]
                for key in (
                    "head",
                    "head_kwargs",
                    "epochs",
                    "batch_size",
                    "learning_rate",
                    "weight_decay",
                )
            },
        }


def _stage(root: Path) -> tuple[dict[str, Any], Path]:
    matches = []
    for path in sorted((root / "provenance" / "stages").glob("*.json")):
        document = evaluation._json(path, name="repeat-head stage")
        if document.get("stage") == "frozen_head_repeat" and document.get("status") == "completed":
            matches.append((document, path))
    if len(matches) != 1:
        raise ValueError(
            "repeat head run must contain exactly one completed frozen_head_repeat stage"
        )
    return matches[0]


def _require_qa(qa: Mapping[str, Any], metadata: Mapping[str, Any], checkpoint: Path) -> None:
    embedded = evaluation._mapping(
        metadata.get("training_qa"), "repeat checkpoint metadata.training_qa"
    )
    bound = evaluation._mapping(qa.get("checkpoint"), "repeat training QA checkpoint")
    if qa.get("status") != "passed" or embedded.get("status") != "passed":
        raise ValueError("repeat head training QA did not pass")
    if (
        bound.get("path") != str(checkpoint)
        or bound.get("sha256") != sha256_file(checkpoint)
        or bound.get("epoch") != 20
    ):
        raise ValueError("repeat head training QA is not bound to its final checkpoint")
    if (
        isinstance(bound.get("step"), bool)
        or not isinstance(bound.get("step"), int)
        or bound["step"] <= 0
    ):
        raise ValueError("repeat head training QA lacks a completed checkpoint step")
    for key in ("nonzero_gradient_steps", "changed_parameter_count"):
        if (
            isinstance(qa.get(key), bool)
            or not isinstance(qa.get(key), int)
            or qa[key] <= 0
            or embedded.get(key) != qa[key]
        ):
            raise ValueError(f"repeat head training QA lacks a bound positive {key}")
    delta = qa.get("parameter_delta_l2")
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(float(delta))
        or delta <= 0
        or embedded.get("parameter_delta_l2") != delta
    ):
        raise ValueError("repeat head training QA lacks a bound positive parameter_delta_l2")
    parity = evaluation._mapping(qa.get("reload_parity"), "repeat training QA reload_parity")
    outputs = evaluation._mapping(parity.get("outputs"), "repeat training QA reload_parity.outputs")
    if not outputs or any(
        evaluation._mapping(value, "repeat reload output").get("exact_equal") is not True
        or evaluation._mapping(value, "repeat reload output").get("max_abs_difference") != 0.0
        for value in outputs.values()
    ):
        raise ValueError("repeat head training QA reload parity is incomplete")
    if embedded.get("reload_parity_exact") is not True or qa.get("final_checkpoint_load") != {
        "missing_keys": [],
        "unexpected_keys": [],
    }:
        raise ValueError("repeat head training QA final reload receipt is incomplete")


def load_frozen_repeat_head_source(
    path: str | Path,
    *,
    freeze: Mapping[str, Any],
    expected_encoder: str,
    role_lock_path: str | Path,
    head_data_contract_path: str | Path,
    source_manifest_root: str | Path,
) -> FrozenRepeatHeadSource:
    root = Path(path).expanduser().resolve()
    result_path, qa_path = root / "result.json", root / "head" / "training_qa.json"
    checkpoint, sidecar = (
        root / "head" / "checkpoints" / "final.pt",
        root / "head" / "checkpoints" / "final.pt.json",
    )
    stage, stage_path = _stage(root)
    result, qa, sidecar_doc = (
        evaluation._json(result_path, name="repeat head result"),
        evaluation._json(qa_path, name="repeat head QA"),
        evaluation._json(sidecar, name="repeat checkpoint sidecar"),
    )
    if (
        result.get("schema_version") != 1
        or result.get("status") != "completed"
        or result.get("seed") not in {1, 2}
    ):
        raise ValueError("repeat head result is not a completed permitted seed-1/2 run")
    stage_freeze, stage_freeze_sha = evaluation._load_freeze(stage["config"].get("freeze_path", ""))
    if stage_freeze != dict(freeze) or result.get("freeze_sha256") != stage_freeze_sha:
        raise ValueError("repeat head result freeze identity differs from its stage")
    if (
        stage["config"].get("encoder") != expected_encoder
        or stage["config"].get("seed") != result["seed"]
    ):
        raise ValueError("repeat head stage differs from the requested encoder/seed")
    base = evaluation.load_frozen_detector_source(
        result.get("source_controller_run", ""),
        freeze=freeze,
        expected_encoder=expected_encoder,
        role_lock_path=role_lock_path,
        head_data_contract_path=head_data_contract_path,
        source_manifest_root=source_manifest_root,
    )
    if result.get("source_artifact_sha256") != dict(base.source_hashes):
        raise ValueError(
            "repeat head result source artifacts differ from the verified controller source"
        )
    if (
        result.get("training_role_lock")
        != {"path": str(base.role_lock_path), "sha256": base.role_lock_sha256}
        or result.get("head_data_contract")
        != {"path": str(base.head_data_contract_path), "sha256": base.head_data_contract_sha256}
        or result.get("source_manifest_root") != str(base.source_manifest_root)
        or result.get("source_manifests_sha256") != dict(base.source_manifest_sha256)
    ):
        raise ValueError(
            "repeat head result data-contract sources differ from the verified controller source"
        )
    inputs = evaluation._mapping(stage.get("inputs"), "repeat head stage.inputs")
    for name, file_path, expected in (
        ("source_checkpoint", base.checkpoint, base.source_hashes["head/checkpoints/final.pt"]),
        ("source_train_manifest", base.train_manifest, base.source_hashes["frozen/train.jsonl"]),
    ):
        entry = evaluation._mapping(inputs.get(name), f"repeat head stage input {name}")
        if (
            Path(entry.get("location", "")).resolve() != file_path
            or entry.get("sha256") != expected
        ):
            raise ValueError(f"repeat head stage input {name} differs from its verified source")
    hashes = {
        "result.json": evaluation._digest(result_path, name="repeat head result"),
        "head/training_qa.json": evaluation._digest(qa_path, name="repeat head QA"),
        "head/checkpoints/final.pt": evaluation._digest(checkpoint, name="repeat checkpoint"),
        "head/checkpoints/final.pt.json": evaluation._digest(
            sidecar, name="repeat checkpoint sidecar"
        ),
        str(stage_path.relative_to(root)).replace("\\", "/"): evaluation._digest(
            stage_path, name="repeat head stage"
        ),
    }
    if (
        result.get("checkpoint_path") != str(checkpoint)
        or result.get("checkpoint_sha256") != hashes["head/checkpoints/final.pt"]
        or result.get("checkpoint_sidecar_sha256") != hashes["head/checkpoints/final.pt.json"]
        or result.get("training_qa_sha256") != hashes["head/training_qa.json"]
    ):
        raise ValueError("repeat head result does not bind its final checkpoint and QA artifacts")
    if sidecar_doc.get("sha256") != hashes["head/checkpoints/final.pt"]:
        raise ValueError("repeat checkpoint sidecar SHA-256 does not match checkpoint bytes")
    metadata = evaluation._mapping(sidecar_doc.get("metadata"), "repeat checkpoint metadata")
    if evaluation._checkpoint_metadata_bytes(checkpoint) != metadata:
        raise ValueError("repeat checkpoint embedded metadata differs from its checksum sidecar")
    training = evaluation._head_training_identity(
        source_train_manifest=base.train_manifest,
        stage={"epochs": 20, "batch_size": 16, "learning_rate": 0.001, "seed": result["seed"]},
        checkpoint_metadata=metadata,
        freeze=freeze,
    )
    if (
        training.seed != result["seed"]
        or result.get("training_identity_fingerprint") != training.fingerprint
    ):
        raise ValueError("repeat head training identity differs from the final checkpoint")
    detector = evaluation._mapping(
        metadata.get("paper_detector"), "repeat checkpoint paper_detector"
    )
    if (
        metadata.get("status") != "completed"
        or metadata.get("encoder_fingerprint") != base.encoder_fingerprint
        or detector.get("training_representation_fingerprint") != base.representation.fingerprint
        or detector.get("training_sampling_fingerprint") != base.sampling.fingerprint
        or detector.get("training_feature_cache_fingerprint")
        != feature_cache_key(base.representation, base.sampling)
        or detector.get("training_identity_fingerprint") != training.fingerprint
        or result.get("training_representation_fingerprint") != base.representation.fingerprint
        or result.get("training_sampling_fingerprint") != base.sampling.fingerprint
    ):
        raise ValueError(
            "repeat head representation/sampling/fit identity differs from its controller source"
        )
    _require_qa(qa, metadata, checkpoint)
    return FrozenRepeatHeadSource(root, checkpoint, metadata, base, training, hashes)


def _cache(
    path: str | Path, *, repeat: FrozenRepeatHeadSource, freeze_sha256: str
) -> tuple[Path, dict[str, Any], dict[str, Any], Path, Path, Mapping[str, Any] | None]:
    root = Path(path).expanduser().resolve()
    result = evaluation._json(root / "result.json", name="source official evaluation result")
    if result.get("status") != "completed" or result.get("protocol") != "ucf-official-frozen-v1":
        raise ValueError(
            "repeat evaluation cache source is not a completed frozen official evaluation"
        )
    resolved_entry = evaluation._mapping(
        result.get("resolved"), "source official evaluation resolved binding"
    )
    resolved_path = root / "resolved.json"
    if resolved_entry.get("path") != str(resolved_path) or resolved_entry.get(
        "sha256"
    ) != evaluation._digest(resolved_path, name="source official evaluation resolved"):
        raise ValueError("source official evaluation result does not bind resolved receipt")
    resolved = evaluation._json(resolved_path, name="source official evaluation resolved")
    if result.get("artifacts") != resolved.get("artifacts"):
        raise ValueError("source official evaluation result artifacts differ from resolved receipt")
    if resolved.get("freeze_sha256") != freeze_sha256:
        raise ValueError("source official evaluation freeze differs from repeat evaluation freeze")
    source = resolved.get("method_source")
    if source != evaluation._source_receipt(repeat.base) or source.get("head_seed") != 0:
        raise ValueError(
            "source official evaluation method source is not the verified seed-0 repeat base"
        )
    if (
        evaluation.RepresentationIdentity.from_mapping(
            evaluation._mapping(
                resolved.get("evaluation_representation"), "source evaluation representation"
            )
        )
        != repeat.representation
    ):
        raise ValueError("source official evaluation cache representation differs from repeat head")
    artifacts = evaluation._mapping(
        resolved.get("artifacts"), "source official evaluation artifacts"
    )
    feature = evaluation._mapping(
        artifacts.get("test_feature_store"), "source official evaluation test FeatureStore"
    )
    feature_root = Path(feature.get("root", "")).resolve()
    if feature_root != (root / "features" / "test").resolve():
        raise ValueError(
            "source official evaluation test FeatureStore must be its writer-owned features/test cache"
        )
    for name in ("resolved", "status", "index"):
        entry = evaluation._mapping(feature.get(name), f"source test FeatureStore {name}")
        expected = (
            feature_root / f"{name}.json" if name != "index" else feature_root / "index.jsonl"
        )
        if Path(entry.get("path", "")).resolve() != expected or entry.get(
            "sha256"
        ) != evaluation._digest(expected, name=f"source test FeatureStore {name}"):
            raise ValueError("source official evaluation test FeatureStore binding is invalid")
    status = evaluation._json(feature_root / "status.json", name="source test FeatureStore status")
    if (
        status.get("completed") is not True
        or Path(status.get("feature_root", "")).resolve() != feature_root
    ):
        raise ValueError("source official evaluation test FeatureStore is incomplete")
    if resolved.get("evaluation_encoder_fingerprint") != status.get("encoder_fingerprint"):
        raise ValueError(
            "source official evaluation cache encoder fingerprint differs from FeatureStore"
        )
    document = evaluation._json(
        feature_root / "resolved.json", name="source test FeatureStore resolution"
    )
    spec = evaluation._mapping(document.get("spec"), "source test FeatureStore spec")
    if (
        evaluation.RepresentationIdentity.from_mapping(
            evaluation._mapping(
                spec.get("representation"), "source test FeatureStore representation"
            )
        )
        != evaluation.RepresentationIdentity.from_mapping(resolved["evaluation_representation"])
        or evaluation.SamplingIdentity.from_mapping(
            evaluation._mapping(spec.get("sampling"), "source test FeatureStore sampling")
        )
        != evaluation.SamplingIdentity.from_mapping(resolved["evaluation_sampling"])
        or spec.get("sampling_kind") != "dense"
        or document.get("encoder_fingerprint") != resolved.get("evaluation_encoder_fingerprint")
        or not isinstance(document.get("runtime"), Mapping)
    ):
        raise ValueError(
            "source official evaluation test FeatureStore resolution differs from cache contract"
        )
    stage_files = sorted((root / "provenance" / "stages").glob("*.json"))
    stages = [
        evaluation._json(item, name="source official evaluation stage") for item in stage_files
    ]
    stages = [
        item
        for item in stages
        if item.get("stage") == "frozen_ucf_evaluation" and item.get("status") == "completed"
    ]
    if len(stages) != 1:
        raise ValueError(
            "source official evaluation must have exactly one completed evaluation stage"
        )
    inputs = evaluation._mapping(stages[0].get("inputs"), "source official evaluation stage.inputs")
    test_manifest = Path(
        evaluation._mapping(inputs.get("test_manifest"), "source test manifest input").get(
            "location", ""
        )
    ).resolve()
    audit_report = Path(
        evaluation._mapping(inputs.get("audit_report"), "source audit input").get("location", "")
    ).resolve()
    sealed = {
        "test_manifest": evaluation._UCF_TEST_MANIFEST_SHA256,
        "audit_report": evaluation._UCF_AUDIT_SHA256,
    }
    resolved_sealed = {
        "test_manifest": resolved.get("sealed_test_manifest_sha256"),
        "audit_report": resolved.get("sealed_audit_sha256"),
    }
    for name, file_path in (("test_manifest", test_manifest), ("audit_report", audit_report)):
        entry = evaluation._mapping(inputs.get(name), f"source {name} input")
        actual = sha256_file(file_path) if file_path.is_file() else None
        if (
            actual is None
            or actual != entry.get("sha256")
            or actual != resolved_sealed[name]
            or actual != sealed[name]
        ):
            raise ValueError(f"source official evaluation stage {name} binding is invalid")
    dense_source = None
    if resolved.get("dense_source") is not None:
        dense = evaluation._mapping(
            resolved["dense_source"], "source official evaluation dense source"
        )
        dense_checkpoint = Path(dense.get("checkpoint_path", "")).resolve()
        if dense.get("head_seed") != 0 or dense.get("checkpoint_sha256") != evaluation._digest(
            dense_checkpoint, name="source official dense checkpoint"
        ):
            raise ValueError(
                "source official evaluation dense source is not its frozen seed-0 checkpoint"
            )
        dense_source = dense
    return root, resolved, feature, test_manifest, audit_report, dense_source


def run_frozen_repeat_ucf_evaluation(
    request: FrozenRepeatEvaluationRequest,
) -> FrozenRepeatEvaluationResult:
    freeze, freeze_sha = evaluation._load_freeze(request.freeze_path)
    source_kwargs = {
        "role_lock_path": request.role_lock_path,
        "head_data_contract_path": request.head_data_contract_path,
        "source_manifest_root": request.source_manifest_root,
    }
    repeat = load_frozen_repeat_head_source(
        request.repeat_head_run, freeze=freeze, expected_encoder=request.encoder, **source_kwargs
    )
    source_root, cached, feature_artifacts, test_manifest, audit_report, cached_dense = _cache(
        request.source_evaluation_run, repeat=repeat, freeze_sha256=freeze_sha
    )
    manifest = load_manifest_jsonl(test_manifest)
    representation = evaluation.RepresentationIdentity.from_mapping(
        cached["evaluation_representation"]
    )
    sampling = evaluation.SamplingIdentity.from_mapping(cached["evaluation_sampling"])
    fingerprint = cached["evaluation_encoder_fingerprint"]
    feature_root = Path(feature_artifacts["root"])
    run_dir = Path(request.output_root).expanduser().resolve() / (
        request.run_id or new_run_id("frozen-repeat-ucf-evaluation")
    )
    if run_dir.exists():
        raise FileExistsError(f"repeat official evaluation run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    dense_repeat = None
    if request.dense_head_repeat_run is not None:
        if cached_dense is None:
            raise ValueError(
                "dense head repeat requires a source evaluation with a frozen dense seed-0 source"
            )
        dense_repeat = load_frozen_repeat_head_source(
            request.dense_head_repeat_run,
            freeze=freeze,
            expected_encoder=request.encoder,
            **source_kwargs,
        )
        dense_base = evaluation.load_frozen_detector_source(
            cached_dense["run_dir"],
            freeze=freeze,
            expected_encoder=request.encoder,
            **source_kwargs,
        )
        if (
            dense_base.reducer.get("name") != "identity"
            or evaluation._source_receipt(dense_base) != cached_dense
            or evaluation._source_receipt(dense_repeat.base) != cached_dense
        ):
            raise ValueError(
                "dense head repeat does not originate from the source evaluation dense head"
            )
        if dense_repeat.training_identity.seed != repeat.training_identity.seed:
            raise ValueError("dense head repeat seed differs from the method repeat seed")
    inputs = {
        "source_evaluation_result": source_root / "result.json",
        "source_evaluation_resolved": source_root / "resolved.json",
        "method_freeze": request.freeze_path,
        "training_role_lock": repeat.base.role_lock_path,
        "head_data_contract": repeat.base.head_data_contract_path,
        "test_feature_resolved": feature_root / "resolved.json",
        "test_feature_status": feature_root / "status.json",
        "test_feature_index": feature_root / "index.jsonl",
        "repeat_checkpoint": repeat.checkpoint,
        "test_manifest": test_manifest,
        "audit_report": audit_report,
    }
    if dense_repeat is not None:
        inputs["dense_repeat_checkpoint"] = dense_repeat.checkpoint
    with record_stage(
        run_dir,
        "frozen_repeat_ucf_evaluation",
        config=request.__dict__,
        inputs=inputs,
        project_root=Path.cwd(),
    ):
        primary = repeat.detection_config(
            mode="refit_head",
            evaluation_representation=representation,
            evaluation_sampling=sampling,
            evaluation_encoder_fingerprint=fingerprint,
        )
        primary_path = run_dir / "predictions-primary-refit-head.jsonl"
        primary_records = predict_detector(
            primary,
            feature_store=feature_root,
            evaluation_manifest=manifest,
            training=repeat.checkpoint,
            output_path=primary_path,
            device=request.device,
        )
        primary_metrics_path = run_dir / "metrics-primary-refit-head.json"
        atomic_write_json(
            primary_metrics_path,
            evaluate_detector(
                primary_records, test_manifest, protocol="official", audit_report=audit_report
            ).to_dict(),
        )
        secondary_path = secondary_metrics_path = None
        if dense_repeat is not None:
            dense_repeat.verify_unchanged()
            secondary = dense_repeat.detection_config(
                mode="direct_insert",
                evaluation_representation=representation,
                evaluation_sampling=sampling,
                evaluation_encoder_fingerprint=fingerprint,
            )
            secondary_path = run_dir / "predictions-secondary-dense-head-direct-insert.jsonl"
            records = predict_detector(
                secondary,
                feature_store=feature_root,
                evaluation_manifest=manifest,
                training=dense_repeat.checkpoint,
                output_path=secondary_path,
                device=request.device,
            )
            secondary_metrics_path = run_dir / "metrics-secondary-dense-head-direct-insert.json"
            atomic_write_json(
                secondary_metrics_path,
                evaluate_detector(
                    records, test_manifest, protocol="official", audit_report=audit_report
                ).to_dict(),
            )
        repeat.verify_unchanged()
        artifacts = {
            "test_feature_store": feature_artifacts,
            "primary_predictions": {
                "path": str(primary_path),
                "sha256": evaluation._digest(primary_path, name="repeat primary predictions"),
            },
            "primary_metrics": {
                "path": str(primary_metrics_path),
                "sha256": evaluation._digest(primary_metrics_path, name="repeat primary metrics"),
            },
            "secondary_predictions": None
            if secondary_path is None
            else {
                "path": str(secondary_path),
                "sha256": evaluation._digest(secondary_path, name="repeat secondary predictions"),
            },
            "secondary_metrics": None
            if secondary_metrics_path is None
            else {
                "path": str(secondary_metrics_path),
                "sha256": evaluation._digest(
                    secondary_metrics_path, name="repeat secondary metrics"
                ),
            },
        }
        resolved_path = run_dir / "resolved.json"
        source_result = source_root / "result.json"
        source_resolved = source_root / "resolved.json"
        atomic_write_json(
            resolved_path,
            {
                "schema_version": 1,
                "freeze_sha256": freeze_sha,
                "sealed_test_manifest_sha256": evaluation._digest(
                    test_manifest, name="sealed UCF test manifest"
                ),
                "sealed_audit_sha256": evaluation._digest(audit_report, name="sealed UCF audit"),
                "method_source": repeat.receipt(),
                "dense_source": None if dense_repeat is None else dense_repeat.receipt(),
                "secondary_status": "not_requested" if dense_repeat is None else "completed",
                "evaluation_representation": representation.to_dict(),
                "evaluation_representation_fingerprint": representation.fingerprint,
                "evaluation_sampling": sampling.to_dict(),
                "evaluation_sampling_fingerprint": sampling.fingerprint,
                "evaluation_encoder_fingerprint": fingerprint,
                "coverage": cached["coverage"],
                "reused_test_feature_store_from": {
                    "run_dir": str(source_root),
                    "result": {
                        "path": str(source_result),
                        "sha256": evaluation._digest(
                            source_result, name="source official evaluation result"
                        ),
                    },
                    "resolved": {
                        "path": str(source_resolved),
                        "sha256": evaluation._digest(
                            source_resolved, name="source official evaluation resolved"
                        ),
                    },
                },
                "artifacts": artifacts,
            },
        )
        atomic_write_json(
            run_dir / "result.json",
            {
                "schema_version": 1,
                "status": "completed",
                "protocol": "ucf-official-frozen-v1",
                "resolved": {
                    "path": str(resolved_path),
                    "sha256": evaluation._digest(
                        resolved_path, name="repeat official evaluation resolved"
                    ),
                },
                "artifacts": artifacts,
            },
        )
    return FrozenRepeatEvaluationResult(
        str(run_dir),
        True,
        str(primary_path),
        str(primary_metrics_path),
        None if secondary_path is None else str(secondary_path),
        None if secondary_metrics_path is None else str(secondary_metrics_path),
    )


__all__ = [
    "FrozenRepeatEvaluationRequest",
    "FrozenRepeatEvaluationResult",
    "FrozenRepeatHeadSource",
    "load_frozen_repeat_head_source",
    "run_frozen_repeat_ucf_evaluation",
]
