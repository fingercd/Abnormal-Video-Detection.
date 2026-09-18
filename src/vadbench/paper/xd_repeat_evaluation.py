"""Score XD seed-1/2 heads only on the exact completed seed-0 XD test cache."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vadbench.artifacts import new_run_id, record_stage
from vadbench.data.manifest import load_manifest_jsonl
from vadbench.features import atomic_write_json
from vadbench.paper import evaluation, xd_evaluation
from vadbench.paper.compatibility import RepresentationIdentity, SamplingIdentity
from vadbench.paper.detection import predict_detector
from vadbench.paper.xd_head_repeats import (
    load_frozen_xd_repeat_head_source,
)


@dataclass(frozen=True)
class FrozenXDRepeatEvaluationRequest:
    encoder: str
    repeat_head_run: str
    source_evaluation_run: str
    output_root: str
    dense_head_repeat_run: str | None = None
    device: str = "cpu"
    run_id: str | None = None


@dataclass(frozen=True)
class FrozenXDRepeatEvaluationResult:
    run_dir: str
    completed: bool
    primary_predictions: str
    primary_metrics: str
    secondary_predictions: str | None
    secondary_metrics: str | None


def _entry(stage: Mapping[str, Any], name: str) -> tuple[Path, str]:
    item = evaluation._mapping(
        evaluation._mapping(stage.get("inputs"), "XD source evaluation stage.inputs").get(name),
        f"XD source evaluation stage input {name}",
    )
    path = Path(item.get("location", "")).resolve()
    actual = evaluation._digest(path, name=f"XD source evaluation input {name}")
    if actual != item.get("sha256"):
        raise ValueError(f"XD source evaluation stage input {name} binding is invalid")
    return path, actual


def _stage(root: Path) -> tuple[dict[str, Any], Path]:
    matches = []
    for path in sorted((root / "provenance" / "stages").glob("*.json")):
        document = evaluation._json(path, name="XD source evaluation stage")
        if (
            document.get("stage") == "frozen_xd_evaluation"
            and document.get("status") == "completed"
        ):
            matches.append((document, path))
    if len(matches) != 1:
        raise ValueError(
            "XD source evaluation must contain exactly one completed frozen_xd_evaluation stage"
        )
    return matches[0]


def _source_request(stage: Mapping[str, Any]) -> xd_evaluation.FrozenXDEvaluationRequest:
    config = evaluation._mapping(stage.get("config"), "XD source evaluation stage.config")
    try:
        return xd_evaluation.FrozenXDEvaluationRequest(**config)
    except TypeError as exc:
        raise ValueError(
            "XD source evaluation stage lacks the frozen XD gate configuration"
        ) from exc


def _source_kwargs(request: xd_evaluation.FrozenXDEvaluationRequest) -> dict[str, str]:
    return {
        "role_lock_path": request.role_lock_path,
        "head_data_contract_path": request.head_data_contract_path,
        "source_manifest_root": request.source_manifest_root,
        "scope_path": request.scope_path,
        "original_role_lock_path": request.original_role_lock_path,
        "method_freeze_path": request.freeze_path,
        "head_data_contract_sha256": request.head_data_contract_sha256,
    }


def _cache(
    path: str | Path, *, expected_encoder: str
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    xd_evaluation.FrozenXDEvaluationRequest,
    Any,
    Mapping[str, Any] | None,
    dict[str, tuple[Path, str]],
]:
    root = Path(path).expanduser().resolve()
    result = evaluation._json(root / "result.json", name="XD source evaluation result")
    if result.get("status") != "completed" or result.get("protocol") != xd_evaluation.PROTOCOL_ID:
        raise ValueError("repeat evaluation cache source is not a completed frozen XD evaluation")
    stage, _stage_path = _stage(root)
    source_request = _source_request(stage)
    if source_request.encoder != expected_encoder:
        raise ValueError("XD source evaluation encoder differs from the requested repeat encoder")
    coordinates = xd_evaluation.verify_raw_coordinates(source_request)
    inputs = {
        name: _entry(stage, name)
        for name in (
            "test_manifest",
            "audit_report",
            "method_freeze",
            "xd_scope",
            "xd_canonical_metadata",
            "xd_gt",
            "xd_original_role_lock",
            "training_role_lock",
            "head_data_contract",
        )
    }
    expected_paths = {
        "test_manifest": Path(source_request.test_manifest).resolve(),
        "audit_report": Path(source_request.audit_report).resolve(),
        "method_freeze": Path(source_request.freeze_path).resolve(),
        "xd_scope": Path(source_request.scope_path).resolve(),
        "xd_canonical_metadata": Path(source_request.canonical_metadata).resolve(),
        "xd_gt": Path(source_request.ground_truth).resolve(),
        "xd_original_role_lock": Path(source_request.original_role_lock_path).resolve(),
        "training_role_lock": Path(source_request.role_lock_path).resolve(),
        "head_data_contract": Path(source_request.head_data_contract_path).resolve(),
    }
    if any(inputs[name][0] != expected for name, expected in expected_paths.items()):
        raise ValueError(
            "XD source evaluation stage inputs differ from its frozen gate configuration"
        )
    resolved_path = root / "resolved.json"
    resolved_entry = evaluation._mapping(
        result.get("resolved"), "XD source evaluation resolved binding"
    )
    if resolved_entry.get("path") != str(resolved_path) or resolved_entry.get(
        "sha256"
    ) != evaluation._digest(resolved_path, name="XD source resolved"):
        raise ValueError("XD source evaluation result does not bind resolved receipt")
    resolved = evaluation._json(resolved_path, name="XD source evaluation resolved")
    if result.get("artifacts") != resolved.get("artifacts"):
        raise ValueError("XD source evaluation artifacts differ from resolved receipt")
    if (
        resolved.get("sealed_test_manifest_sha256") != inputs["test_manifest"][1]
        or resolved.get("sealed_audit_sha256") != inputs["audit_report"][1]
    ):
        raise ValueError("XD source evaluation sealed test/audit identities differ from its gate")
    method = xd_evaluation.load_frozen_xd_detector_source(
        source_request.method_source_run,
        freeze=evaluation._load_freeze(source_request.freeze_path)[0],
        expected_encoder=expected_encoder,
        **_source_kwargs(source_request),
    )
    method_receipt = evaluation._source_receipt(method)
    if resolved.get("method_source") != method_receipt or method_receipt.get("head_seed") != 0:
        raise ValueError(
            "XD source evaluation cache is not bound to its verified seed-0 method head"
        )
    representation = RepresentationIdentity.from_mapping(
        evaluation._mapping(
            resolved.get("evaluation_representation"), "XD evaluation representation"
        )
    )
    sampling = SamplingIdentity.from_mapping(
        evaluation._mapping(resolved.get("evaluation_sampling"), "XD evaluation sampling")
    )
    artifacts = evaluation._mapping(resolved.get("artifacts"), "XD source evaluation artifacts")
    feature = evaluation._mapping(
        artifacts.get("test_feature_store"), "XD source test FeatureStore"
    )
    feature_root = Path(feature.get("root", "")).resolve()
    if feature_root != (root / "features" / "test").resolve():
        raise ValueError("XD source test FeatureStore must be its writer-owned features/test cache")
    for name, filename in (
        ("resolved", "resolved.json"),
        ("status", "status.json"),
        ("index", "index.jsonl"),
    ):
        item = evaluation._mapping(feature.get(name), f"XD source test FeatureStore {name}")
        expected = feature_root / filename
        if Path(item.get("path", "")).resolve() != expected or item.get(
            "sha256"
        ) != evaluation._digest(expected, name=f"XD test FeatureStore {name}"):
            raise ValueError("XD source test FeatureStore binding is invalid")
    status = evaluation._json(
        feature_root / "status.json", name="XD source test FeatureStore status"
    )
    document = evaluation._json(
        feature_root / "resolved.json", name="XD source test FeatureStore resolution"
    )
    spec = evaluation._mapping(document.get("spec"), "XD source test FeatureStore spec")
    if (
        status.get("completed") is not True
        or Path(status.get("feature_root", "")).resolve() != feature_root
        or document.get("encoder_fingerprint") != resolved.get("evaluation_encoder_fingerprint")
        or RepresentationIdentity.from_mapping(
            evaluation._mapping(spec.get("representation"), "XD cache representation")
        )
        != representation
        or SamplingIdentity.from_mapping(
            evaluation._mapping(spec.get("sampling"), "XD cache sampling")
        )
        != sampling
        or spec.get("sampling_kind") != "dense"
    ):
        raise ValueError("XD source test FeatureStore resolution differs from cache contract")
    dense_source = resolved.get("dense_source")
    if dense_source is not None:
        dense_source = evaluation._mapping(dense_source, "XD source dense head")
        if dense_source.get("head_seed") != 0:
            raise ValueError("XD source dense head is not the frozen seed-0 checkpoint")
    xd_evaluation.validate_xd_data_content_evidence(
        evaluation._mapping(document.get("data_content_evidence"), "XD cache data content evidence"),
        coordinates,
    )
    return root, resolved, feature, source_request, coordinates, dense_source, inputs


def run_frozen_xd_repeat_evaluation(
    request: FrozenXDRepeatEvaluationRequest,
) -> FrozenXDRepeatEvaluationResult:
    (
        source_root,
        cached,
        feature_artifacts,
        source_request,
        coordinates,
        cached_dense,
        gate_inputs,
    ) = _cache(request.source_evaluation_run, expected_encoder=request.encoder)
    freeze, freeze_sha = evaluation._load_freeze(source_request.freeze_path)
    repeat = load_frozen_xd_repeat_head_source(
        request.repeat_head_run,
        freeze=freeze,
        expected_encoder=request.encoder,
        **_source_kwargs(source_request),
    )
    if cached.get("method_source") != evaluation._source_receipt(repeat.base):
        raise ValueError(
            "XD repeat head does not originate from the seed-0 source evaluation method head"
        )
    representation = RepresentationIdentity.from_mapping(cached["evaluation_representation"])
    sampling = SamplingIdentity.from_mapping(cached["evaluation_sampling"])
    fingerprint = str(cached["evaluation_encoder_fingerprint"])
    feature_root = Path(feature_artifacts["root"])
    manifest = load_manifest_jsonl(source_request.test_manifest)
    run_dir = Path(request.output_root).expanduser().resolve() / (
        request.run_id or new_run_id("frozen-xd-repeat-evaluation")
    )
    if run_dir.exists():
        raise FileExistsError(f"XD repeat evaluation run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    dense_repeat = None
    if request.dense_head_repeat_run is not None:
        if cached_dense is None:
            raise ValueError(
                "dense repeat requires a source XD evaluation with a frozen dense seed-0 source"
            )
        dense_repeat = load_frozen_xd_repeat_head_source(
            request.dense_head_repeat_run,
            freeze=freeze,
            expected_encoder=request.encoder,
            **_source_kwargs(source_request),
        )
        if (
            dense_repeat.base.reducer.get("name") != "identity"
            or evaluation._source_receipt(dense_repeat.base) != cached_dense
            or dense_repeat.training_identity.seed != repeat.training_identity.seed
        ):
            raise ValueError(
                "XD dense head repeat does not originate from the matching seed-0 dense head"
            )
    inputs = {
        "source_evaluation_result": source_root / "result.json",
        "source_evaluation_resolved": source_root / "resolved.json",
        "test_feature_resolved": feature_root / "resolved.json",
        "test_feature_status": feature_root / "status.json",
        "test_feature_index": feature_root / "index.jsonl",
        "repeat_checkpoint": repeat.checkpoint,
        **{name: path for name, (path, _sha) in gate_inputs.items()},
    }
    if dense_repeat is not None:
        inputs["dense_repeat_checkpoint"] = dense_repeat.checkpoint
    with record_stage(
        run_dir,
        "frozen_xd_repeat_evaluation",
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
            xd_evaluation.score_xd_predictions(
                primary_records, coordinates, source_request.ground_truth
            ),
        )
        secondary_path = secondary_metrics_path = None
        if dense_repeat is not None:
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
                xd_evaluation.score_xd_predictions(
                    records, coordinates, source_request.ground_truth
                ),
            )
        repeat.verify_unchanged()
        if dense_repeat is not None:
            dense_repeat.verify_unchanged()
        artifacts = {
            "test_feature_store": feature_artifacts,
            "primary_predictions": {
                "path": str(primary_path),
                "sha256": evaluation._digest(primary_path, name="XD repeat primary predictions"),
            },
            "primary_metrics": {
                "path": str(primary_metrics_path),
                "sha256": evaluation._digest(
                    primary_metrics_path, name="XD repeat primary metrics"
                ),
            },
            "secondary_predictions": None
            if secondary_path is None
            else {
                "path": str(secondary_path),
                "sha256": evaluation._digest(
                    secondary_path, name="XD repeat secondary predictions"
                ),
            },
            "secondary_metrics": None
            if secondary_metrics_path is None
            else {
                "path": str(secondary_metrics_path),
                "sha256": evaluation._digest(
                    secondary_metrics_path, name="XD repeat secondary metrics"
                ),
            },
        }
        resolved_path = run_dir / "resolved.json"
        atomic_write_json(
            resolved_path,
            {
                "schema_version": 1,
                "protocol": xd_evaluation.PROTOCOL_ID,
                "freeze_sha256": freeze_sha,
                "sealed_test_manifest_sha256": cached.get("sealed_test_manifest_sha256"),
                "sealed_audit_sha256": cached.get("sealed_audit_sha256"),
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
                        "path": str(source_root / "result.json"),
                        "sha256": evaluation._digest(
                            source_root / "result.json", name="seed-0 XD evaluation result"
                        ),
                    },
                    "resolved": {
                        "path": str(source_root / "resolved.json"),
                        "sha256": evaluation._digest(
                            source_root / "resolved.json", name="seed-0 XD evaluation resolved"
                        ),
                    },
                },
                "xd_gate_inputs": {
                    name: {"path": str(path), "sha256": digest}
                    for name, (path, digest) in gate_inputs.items()
                },
                "artifacts": artifacts,
            },
        )
        atomic_write_json(
            run_dir / "result.json",
            {
                "schema_version": 1,
                "status": "completed",
                "protocol": xd_evaluation.PROTOCOL_ID,
                "resolved": {
                    "path": str(resolved_path),
                    "sha256": evaluation._digest(
                        resolved_path, name="XD repeat evaluation resolved"
                    ),
                },
                "artifacts": artifacts,
            },
        )
    return FrozenXDRepeatEvaluationResult(
        str(run_dir),
        True,
        str(primary_path),
        str(primary_metrics_path),
        None if secondary_path is None else str(secondary_path),
        None if secondary_metrics_path is None else str(secondary_metrics_path),
    )


__all__ = [
    "FrozenXDRepeatEvaluationRequest",
    "FrozenXDRepeatEvaluationResult",
    "run_frozen_xd_repeat_evaluation",
]
