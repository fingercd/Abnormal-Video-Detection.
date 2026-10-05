"""Read completed sealed XD RGB runs; never train, infer, or select methods.

The unchanged method/scope seals and audited raw prefix coordinate contract are
mandatory. Each head seed is a separate paired-video PR-trapezoid comparison.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vadbench.artifacts import PredictionRecord
from vadbench.data.manifest import VideoManifestRecord
from vadbench.engine.coverage import validate_frame_coverage
from vadbench.engine.evaluate import prediction_records_to_temporal
from vadbench.paper import evaluation
from vadbench.paper import repeat_evaluation
from vadbench.paper import quality_export as shared
from vadbench.paper import xd_evaluation as xd
from vadbench.data.feature_contracts import (
    RepresentationIdentity,
    SamplingIdentity,
    validate_compatibility,
)
from vadbench.workflows.quality_comparison import (
    DECISION_TOLERANCE,
    PairedVideoIntervals,
    compare_paired_quality,
)
from vadbench.paper.quality_export import (
    QualityMode,
    _bound_json,
    _bound_path,
    _canonical,
    _json,
    _prediction_records,
    _read,
    _require,
)

_EVALUATION_STAGES = {"frozen_xd_evaluation", "frozen_xd_repeat_evaluation"}


def _feature_store_root(root, resolved, sources, visited=None):
    visited = set() if visited is None else visited
    _require(root not in visited, "cyclic reused FeatureStore provenance")
    visited.add(root)
    entry = resolved["artifacts"]["test_feature_store"]
    _require(Path(entry["root"]).is_absolute(), "test FeatureStore root must be absolute")
    feature_root = Path(entry["root"]).resolve()
    if feature_root == root / "features/test":
        return feature_root
    reuse = resolved.get("reused_test_feature_store_from")
    _require(
        isinstance(reuse, Mapping),
        "external FeatureStore lacks a completed source evaluation binding",
    )
    origin = Path(reuse["run_dir"]).resolve()
    result = _bound_json(reuse["result"], origin / "result.json", sources)
    source = _bound_json(reuse["resolved"], origin / "resolved.json", sources)
    _require(
        result.get("schema_version") == 1
        and result.get("status") == "completed"
        and result.get("protocol") == "xd-violence/frozen-rgb-raw-prefix-pr-auc-v1",
        "reused FeatureStore source evaluation is not completed sealed XD",
    )
    _require(
        result["resolved"] == reuse["resolved"]
        and result["artifacts"] == source["artifacts"]
        and source["artifacts"]["test_feature_store"] == entry,
        "reused FeatureStore artifact bindings differ from source evaluation",
    )
    for key in (
        "freeze_sha256",
        "sealed_test_manifest_sha256",
        "sealed_audit_sha256",
        "evaluation_representation",
        "evaluation_representation_fingerprint",
        "evaluation_sampling",
        "evaluation_sampling_fingerprint",
        "evaluation_encoder_fingerprint",
    ):
        _require(
            source[key] == resolved[key], f"reused FeatureStore source contract differs: {key}"
        )
    stages = [_json(p, sources) for p in (origin / "provenance/stages").glob("*.json")]
    stages = [s for s in stages if s.get("stage") in _EVALUATION_STAGES]
    _require(
        len(stages) == 1 and stages[0].get("status") == "completed",
        "reused FeatureStore source stage is not completed",
    )
    _require(
        stages[0]["config_sha256"] == hashlib.sha256(_canonical(stages[0]["config"])).hexdigest(),
        "reused source stage config SHA mismatch",
    )
    for name, key in (
        ("test_manifest", "sealed_test_manifest_sha256"),
        ("audit_report", "sealed_audit_sha256"),
        ("method_freeze", "freeze_sha256"),
    ):
        _require(
            stages[0]["inputs"][name]["sha256"] == source[key],
            "reused source stage sealed input mismatch",
        )
    _require(
        _feature_store_root(origin, source, sources, visited) == feature_root,
        "reused FeatureStore origin differs",
    )
    return feature_root


@dataclass(frozen=True)
class _CompletedRun:
    root: Path
    mode: QualityMode
    records: tuple[PredictionRecord, ...]
    manifest: tuple[VideoManifestRecord, ...]
    metrics: Mapping[str, Any]
    source: evaluation.FrozenDetectorSource | repeat_evaluation.FrozenRepeatHeadSource
    representation: RepresentationIdentity
    sampling: SamplingIdentity
    native_runtime: Mapping[str, Any]
    input_windows_sha256: str
    data_evidence: Mapping[str, Any]
    freeze_sha256: str
    manifest_sha256: str
    audit_sha256: str
    artifact_sha256: Mapping[str, str]
    source_receipt: Mapping[str, Any]
    coordinates: xd.SealedXDCoordinates
    ground_truth: Path


def _load_completed_run(path: str | Path, mode: QualityMode) -> _CompletedRun:
    _require(mode in ("refit_head", "direct_insert"), "mode must be refit_head or direct_insert")
    root, sources = Path(path).expanduser().resolve(), {}
    result = _json(root / "result.json", sources)
    _require(
        result.get("schema_version") == 1
        and result.get("status") == "completed"
        and result.get("protocol") == "xd-violence/frozen-rgb-raw-prefix-pr-auc-v1",
        "run is not a completed frozen sealed XD evaluation",
    )
    resolved = _bound_json(result["resolved"], root / "resolved.json", sources)
    _require(
        resolved.get("schema_version") == 1 and result["artifacts"] == resolved["artifacts"],
        "completed artifact ledger differs from resolved ledger",
    )
    stages = []
    for stage_path in sorted((root / "provenance/stages").glob("*.json")):
        stage = _json(stage_path, sources)
        if stage.get("stage") in _EVALUATION_STAGES:
            stages.append(stage)
    _require(
        len(stages) == 1 and stages[0].get("status") == "completed",
        "requires exactly one completed official evaluation stage",
    )
    stage = stages[0]
    is_repeat = stage["stage"] == "frozen_xd_repeat_evaluation"
    _require(
        (resolved["method_source"].get("kind") == "frozen_xd_head_repeat_v1") == is_repeat,
        "head source kind differs from evaluation stage",
    )
    if is_repeat:
        reuse = resolved.get("reused_test_feature_store_from")
        _require(
            isinstance(reuse, Mapping),
            "repeat evaluation must bind its original completed test cache",
        )
        origin = Path(reuse["run_dir"]).resolve()
        original = _bound_json(reuse["resolved"], origin / "resolved.json", sources)
        for key in ("method_source", "dense_source"):
            if resolved.get(key) is not None:
                _require(
                    resolved[key]["source_controller"] == original.get(key),
                    "repeat head does not derive from the corresponding cached seed-0 source",
                )
                _require(
                    resolved[key]["source_controller"]["head_seed"] == 0,
                    "repeat cache source must be the frozen seed-0 head",
                )
        _require(
            resolved.get("secondary_status")
            == ("completed" if resolved.get("dense_source") is not None else "not_requested"),
            "repeat secondary route completion status mismatch",
        )
    config = stage["config"]
    _require(
        stage["config_sha256"] == hashlib.sha256(_canonical(config)).hexdigest(),
        "official stage config SHA mismatch",
    )
    # A repeat inherits the sealed input paths from its SHA-bound seed-0 run.
    input_config = config
    if is_repeat:
        origin_stages = [_json(p, sources) for p in (origin / "provenance/stages").glob("*.json")]
        origin_stages = [s for s in origin_stages if s.get("stage") == "frozen_xd_evaluation"]
        _require(
            len(origin_stages) == 1 and origin_stages[0].get("status") == "completed",
            "repeat cache requires a completed seed-0 XD stage",
        )
        original_stage = origin_stages[0]
        _require(
            original_stage["config_sha256"]
            == hashlib.sha256(_canonical(original_stage["config"])).hexdigest(),
            "source stage config SHA mismatch",
        )
        input_config = original_stage["config"]
    request = xd.FrozenXDEvaluationRequest(**input_config)
    _require(
        request.encoder == config["encoder"], "repeat encoder differs from seed-0 input config"
    )
    coordinates = xd.verify_raw_coordinates(request)
    manifest = coordinates.manifest
    manifest_path, audit_path, freeze_path = (
        Path(p).resolve()
        for p in (request.test_manifest, request.audit_report, request.freeze_path)
    )
    freeze, freeze_sha = evaluation._load_freeze(freeze_path)
    _require(
        evaluation._canonical_sha256(freeze) == xd.METHOD_FREEZE_CANONICAL_SHA256,
        "XD method freeze differs from frozen canonical SHA",
    )
    scope = xd._scope(request.scope_path)
    expected = {
        "test_manifest": (manifest_path, coordinates.input_hashes[manifest_path]),
        "audit_report": (audit_path, request.raw_coordinate_receipt_sha256),
        "method_freeze": (freeze_path, freeze_sha),
        "xd_scope": (Path(request.scope_path).resolve(), xd.SCOPE_FILE_SHA256),
        "xd_original_role_lock": (
            Path(request.original_role_lock_path).resolve(),
            scope["original_role_lock_file_sha256"],
        ),
        "xd_canonical_metadata": (
            Path(request.canonical_metadata).resolve(),
            coordinates.input_hashes[Path(request.canonical_metadata).resolve()],
        ),
        "xd_gt": (Path(request.ground_truth).resolve(), xd.GT_FILE_SHA256),
    }
    for name, (file_path, digest) in expected.items():
        _require(
            stage["inputs"][name]["sha256"] == digest
            and Path(stage["inputs"][name]["location"]).resolve() == file_path,
            f"XD stage input identity mismatch: {name}",
        )
        _read(file_path, sources, digest)
        if is_repeat:
            original_input = original_stage["inputs"][name]
            _require(
                original_input["sha256"] == digest
                and Path(original_input["location"]).resolve() == file_path,
                f"repeat input differs from seed-0 stage: {name}",
            )
    for file_path, digest in coordinates.input_hashes.items():
        # The coordinate verifier has already hashed these files, including
        # potentially large raw videos. Register its hashes without loading
        # video bytes into memory; the final source pass hashes as a stream.
        _require(
            str(file_path) not in sources or sources[str(file_path)] == digest,
            "coordinate proof artifact changed across reads",
        )
        sources[str(file_path)] = digest
    _require(
        resolved["freeze_sha256"] == freeze_sha
        and resolved["sealed_test_manifest_sha256"] == sources[str(manifest_path)]
        and resolved["sealed_audit_sha256"] == sources[str(audit_path)],
        "sealed freeze/manifest/audit identity mismatch",
    )
    representation, sampling, runtime, windows_sha, data_evidence, feature_videos = (
        shared._feature_contract(
            root,
            resolved["artifacts"]["test_feature_store"],
            resolved,
            manifest,
            sources,
            expected_video_count=xd.OFFICIAL_VIDEO_COUNT,
            feature_root_loader=_feature_store_root,
        )
    )
    xd.validate_xd_data_content_evidence(data_evidence, coordinates)
    _require(
        feature_videos == {r.video_id for r in manifest},
        "actual test input window IDs differ from sealed manifest",
    )
    _require(
        config["encoder"] == representation.backbone.runtime_id,
        "stage encoder differs from test representation",
    )
    frozen_sampling = freeze["sampling"]
    _require(
        sampling.regime == "test_dense"
        and sampling.frame_selection.get("kind") == "dense_sliding"
        and sampling.window
        == {"clip_frames": frozen_sampling["native_clip_frames"][config["encoder"]]}
        and sampling.stride == {"frame_stride": frozen_sampling["frame_stride"]}
        and sampling.frame_selection.get("window_stride")
        == frozen_sampling["evaluation_window_stride"][config["encoder"]]
        and sampling.frame_selection.get("short_video_policy") == frozen_sampling["short_policy"],
        "evaluation sampling differs from the frozen native-input contract",
    )
    source_objects = {}
    for key in ("method_source", "dense_source"):
        snapshot = resolved.get(key)
        if snapshot is None:
            continue
        _require(
            snapshot.get("kind") in (None, "frozen_xd_head_repeat_v1"),
            "unknown frozen head source kind",
        )
        _require(
            (snapshot.get("kind") == "frozen_xd_head_repeat_v1") == is_repeat,
            "both prediction routes must use the evaluation's head source kind",
        )
        base_snapshot = snapshot["source_controller"] if is_repeat else snapshot
        if is_repeat:
            from vadbench.paper.xd_repeat_evaluation import load_frozen_xd_repeat_head_source

            source_loader = load_frozen_xd_repeat_head_source
        else:
            source_loader = xd.load_frozen_xd_detector_source
        loaded = source_loader(
            snapshot["run_dir"],
            freeze=freeze,
            expected_encoder=config["encoder"],
            role_lock_path=base_snapshot["training_role_lock"]["path"],
            head_data_contract_path=base_snapshot["head_data_contract"]["path"],
            source_manifest_root=base_snapshot["source_manifest_root"],
            scope_path=request.scope_path,
            original_role_lock_path=request.original_role_lock_path,
            method_freeze_path=request.freeze_path,
            head_data_contract_sha256=request.head_data_contract_sha256,
        )
        actual_receipt = loaded.receipt() if is_repeat else evaluation._source_receipt(loaded)
        _require(
            actual_receipt == snapshot,
            "frozen head source changed or differs from resolved receipt",
        )
        config_key = (
            ("repeat_head_run" if key == "method_source" else "dense_head_repeat_run")
            if is_repeat
            else ("method_source_run" if key == "method_source" else "dense_source_run")
        )
        _require(
            Path(config[config_key]).resolve() == loaded.root,
            "stage source run differs from verified source receipt",
        )
        checkpoint_key = (
            ("repeat_checkpoint" if key == "method_source" else "dense_repeat_checkpoint")
            if is_repeat
            else ("method_checkpoint" if key == "method_source" else "dense_checkpoint")
        )
        _require(
            stage["inputs"][checkpoint_key]["sha256"] == snapshot["checkpoint_sha256"]
            and Path(stage["inputs"][checkpoint_key]["location"]).resolve()
            == loaded.checkpoint.resolve(),
            "stage checkpoint differs from verified source",
        )
        base = loaded.base if is_repeat else loaded
        for relative, digest in base.source_hashes.items():
            _read(base.root / relative, sources, digest)
        if is_repeat:
            for relative, digest in loaded.artifact_sha256.items():
                _read(loaded.root / relative, sources, digest)
        for field, stage_field in (
            ("training_role_lock", "training_role_lock"),
            ("head_data_contract", "head_data_contract"),
        ):
            entry = base_snapshot[field]
            _require(
                stage["inputs"][stage_field]["sha256"] == entry["sha256"]
                and Path(stage["inputs"][stage_field]["location"]).resolve()
                == Path(entry["path"]).resolve(),
                "head data-contract/role-lock differs from the evaluated source",
            )
            _read(Path(entry["path"]), sources, entry["sha256"])
        for source_path, digest in base_snapshot["source_manifests_sha256"].items():
            _read(Path(source_path), sources, digest)
        source_objects[key] = loaded
    _require(
        "method_source" in source_objects
        and source_objects["method_source"].representation == representation,
        "evaluated representation differs from its frozen method source",
    )
    if "dense_source" in source_objects:
        _require(
            source_objects["dense_source"].training_identity.seed
            == source_objects["method_source"].training_identity.seed,
            "primary and secondary head seeds differ within the evaluation run",
        )
    _require(
        (resolved["artifacts"]["secondary_predictions"] is not None)
        == ("dense_source" in source_objects),
        "secondary prediction artifacts differ from the declared dense source",
    )
    source_key = "method_source" if mode == "refit_head" else "dense_source"
    _require(source_key in source_objects, "requested prediction branch has no frozen head source")
    source, source_receipt = source_objects[source_key], resolved[source_key]
    source_stage_key = (
        ("repeat_checkpoint" if mode == "refit_head" else "dense_repeat_checkpoint")
        if is_repeat
        else ("method_checkpoint" if mode == "refit_head" else "dense_checkpoint")
    )
    _require(
        stage["inputs"][source_stage_key]["sha256"] == source_receipt["checkpoint_sha256"],
        "stage checkpoint differs from selected prediction branch",
    )
    _require(
        Path(stage["inputs"][source_stage_key]["location"]).resolve()
        == source.checkpoint.resolve(),
        "stage checkpoint path differs from selected source",
    )
    declaration = source.detection_config(
        mode=mode,
        evaluation_representation=representation,
        evaluation_sampling=sampling,
        evaluation_encoder_fingerprint=resolved["evaluation_encoder_fingerprint"],
    ).declaration
    permit = validate_compatibility(declaration)
    for branch_name, branch_suffix in (
        ("primary", "primary-refit-head"),
        ("secondary", "secondary-dense-head-direct-insert"),
    ):
        prediction_entry = resolved["artifacts"][branch_name + "_predictions"]
        metric_entry = resolved["artifacts"][branch_name + "_metrics"]
        _require(
            (prediction_entry is None) == (metric_entry is None),
            "completed prediction/metric artifact pair is incomplete",
        )
        _require(
            branch_name != "primary" or prediction_entry is not None,
            "completed run has no primary prediction artifacts",
        )
        if prediction_entry is not None:
            for entry, filename in (
                (prediction_entry, f"predictions-{branch_suffix}.jsonl"),
                (metric_entry, f"metrics-{branch_suffix}.json"),
            ):
                _read(_bound_path(entry, root / filename), sources, entry["sha256"])
    branch = "primary" if mode == "refit_head" else "secondary"
    suffix = "primary-refit-head" if mode == "refit_head" else "secondary-dense-head-direct-insert"
    records = _prediction_records(
        resolved["artifacts"][branch + "_predictions"],
        root / f"predictions-{suffix}.jsonl",
        sources,
    )
    metrics = _bound_json(
        resolved["artifacts"][branch + "_metrics"], root / f"metrics-{suffix}.json", sources
    )
    run_ids = {row.run_id for row in records}
    _require(len(run_ids) == 1, "predictions mix run IDs")
    for row in records:
        _require(
            row.encoder_fingerprint == resolved["evaluation_encoder_fingerprint"],
            "prediction encoder fingerprint mismatch",
        )
        _require(
            row.metadata.get("checkpoint_sha256") == source_receipt["checkpoint_sha256"],
            "prediction checkpoint SHA mismatch",
        )
        _require(
            row.metadata.get("paper_compatibility") == permit,
            "prediction compatibility receipt mismatch",
        )
        aggregation = row.metadata.get("dense_aggregation", {})
        _require(
            row.metadata.get("score_level") == "frame"
            and aggregation.get("interval_source") == "exact_equal_frame_run"
            and aggregation.get("reduction") == "mean",
            "predictions are not the frozen final mean-projected frame intervals",
        )
    # Re-run the official XD scorer to verify full raw coverage before GT access.
    actual_metrics = xd.score_xd_predictions(records, coordinates, request.ground_truth)
    _require(metrics == actual_metrics, "saved XD metric disagrees with audited predictions/GT")
    for loaded in source_objects.values():
        loaded.verify_unchanged()
    return _CompletedRun(
        root,
        mode,
        records,
        manifest,
        metrics,
        source,
        representation,
        sampling,
        runtime,
        windows_sha,
        data_evidence,
        freeze_sha,
        sources[str(manifest_path)],
        sources[str(audit_path)],
        sources,
        source_receipt,
        coordinates,
        Path(request.ground_truth).resolve(),
    )


def paired_xd_frame_intervals(coordinates, ground_truth, dense_records, method_records):
    """Align exact raw score runs and actual GT changes inside each sealed prefix.

    Full raw-video coverage is checked even though the prespecified suffix is
    excluded from the metric. The weak label is used only for video bootstrap
    stratification; GT values come from the independently sealed canonical file.
    """
    coordinates.verify_unchanged()
    truth_path = xd._bound(ground_truth, xd.GT_FILE_SHA256, "canonical XD GT")
    dense, dense_unit = prediction_records_to_temporal(dense_records)
    method, method_unit = prediction_records_to_temporal(method_records)
    by_id = {r.video_id: r for r in coordinates.manifest}
    _require(
        dense_unit == method_unit == "frames" and set(dense) == set(method) == set(by_id),
        "both predictions must cover the exact sealed XD manifest",
    )
    ordered = {}
    for video_id, record in by_id.items():
        ordered[video_id] = []
        for prediction in (dense[video_id], method[video_id]):
            indices = np.argsort(prediction.intervals[:, 0], kind="stable")
            intervals, scores = prediction.intervals[indices], prediction.scores[indices]
            validate_frame_coverage(
                video_id=video_id,
                clip_indices=np.arange(len(intervals)),
                frame_starts=intervals[:, 0],
                frame_ends=intervals[:, 1],
                num_frames=record.num_frames,
                fps=record.fps,
                require_fps=True,
            )
            _require(np.isfinite(scores).all(), "nonfinite XD frame scores")
            ordered[video_id].append((intervals, scores))
    truth = np.load(truth_path, allow_pickle=False)
    _require(
        truth.shape == (xd.OFFICIAL_GT_LENGTH,) and np.isin(truth, (0, 1)).all(),
        "canonical XD GT must be complete and binary",
    )
    output = []
    for raw, grid in zip(coordinates.raw_rows, coordinates.grid_rows, strict=True):
        video_id = raw["video_id"]
        length = grid["t"] * 16
        labels = truth[grid["gt_start"] : grid["gt_end_exclusive"]]
        changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        boundaries = np.unique(
            np.concatenate(
                ([0, length], changes, *(intervals.ravel() for intervals, _ in ordered[video_id]))
            )
        )
        boundaries = boundaries[(boundaries >= 0) & (boundaries <= length)]
        starts, ends = boundaries[:-1], boundaries[1:]
        scores = [
            values[np.searchsorted(intervals[:, 1], starts, side="right")]
            for intervals, values in ordered[video_id]
        ]
        output.append(
            PairedVideoIntervals(
                video_id,
                int(by_id[video_id].is_anomaly),
                length,
                np.column_stack((starts, ends)),
                labels[starts],
                *scores,
            )
        )
    coordinates.verify_unchanged()
    xd._bound(truth_path, xd.GT_FILE_SHA256, "canonical XD GT")
    return tuple(output)


def compare_frozen_xd_quality(
    dense_run: str | Path, method_run: str | Path, *, mode: QualityMode = "refit_head"
) -> dict[str, Any]:
    """Compare one same-seed pair after verifying every frozen source contract."""
    dense = _load_completed_run(dense_run, "refit_head")
    method = _load_completed_run(method_run, mode)
    _require(
        dense.representation.reducer.get("name") == "identity",
        "dense baseline must use the identity reducer",
    )
    for name in (
        "freeze_sha256",
        "manifest_sha256",
        "audit_sha256",
        "sampling",
        "native_runtime",
        "input_windows_sha256",
        "data_evidence",
    ):
        _require(
            getattr(dense, name) == getattr(method, name),
            f"dense/method evaluation contract mismatch: {name}",
        )
    _require(
        dense.coordinates.grid_rows == method.coordinates.grid_rows
        and dense.coordinates.raw_rows == method.coordinates.raw_rows,
        "dense/method canonical coordinate identity differs",
    )
    for name in ("backbone", "output_dim", "precision", "position_strategy"):
        _require(
            getattr(dense.representation, name) == getattr(method.representation, name),
            f"only reducer differences are allowed: {name}",
        )
    _require(
        dense.source.training_identity == method.source.training_identity,
        "heads differ in seed, fit split, or training identity",
    )
    _require(
        dense.source_receipt["head_budget"] == method.source_receipt["head_budget"],
        "head training budgets differ",
    )
    if mode == "direct_insert":
        _require(
            dense.source_receipt["checkpoint_sha256"] == method.source_receipt["checkpoint_sha256"],
            "direct_insert must use the exact dense baseline head checkpoint",
        )
    sources = dict(dense.artifact_sha256)
    for path, digest in method.artifact_sha256.items():
        _require(
            path not in sources or sources[path] == digest,
            "shared artifact changed across paired reads",
        )
        sources[path] = digest
    videos = paired_xd_frame_intervals(
        dense.coordinates, dense.ground_truth, dense.records, method.records
    )
    result = compare_paired_quality(videos, metric="frame_pr_auc", source_sha256=sources)
    for run, key in ((dense, "dense_value"), (method, "method_value")):
        _require(
            result[key] is not None
            and abs(result[key] - run.metrics["frame_pr_auc"]) <= DECISION_TOLERANCE,
            "saved XD metric disagrees with reconstructed paired frame intervals/GT",
        )
        _require(
            run.metrics["video_count"] == result["n_videos"]
            and run.metrics["gt_length"] == result["n_frames"],
            "saved XD counts disagree with canonical prefixes",
        )
    # Do not deliver a snapshot assembled across an artifact mutation.
    for path, digest in tuple(sources.items()):
        _require(
            evaluation._digest(Path(path), name="paired XD source") == digest,
            f"paired XD source artifact SHA mismatch: {path}",
        )
    return {
        **result,
        "comparison_protocol": "xd-completed-frozen-runs-paired-quality-v1",
        "evaluation_protocol": xd.PROTOCOL_ID,
        "mode": mode,
        "head_seed": method.source.training_identity.seed,
        "encoder": method.representation.backbone.runtime_id,
        "sealed_manifest_sha256": method.manifest_sha256,
        "sealed_audit_sha256": method.audit_sha256,
        "freeze_sha256": method.freeze_sha256,
        "scope_sha256": xd.SCOPE_FILE_SHA256,
        "canonical_metadata_sha256": xd.CANONICAL_METADATA_SHA256,
        "gt_file_sha256": xd.GT_FILE_SHA256,
        "mapping": dict(xd.MAPPING),
        "training_scope": "derived train1598; name-group-disjoint fit1267/confirm167/select164; not full official train3954; name-group separation is a proxy",
        "input_windows_sha256": method.input_windows_sha256,
        "input_window_verification": "all frozen DenseSamplingPlan windows including tail/end-anchor, actual frame stride and raw content; native frames/masks reconstructed and exported arrays checked",
        "dense_run": str(dense.root),
        "method_run": str(method.root),
        "dense_head": dict(dense.source_receipt),
        "method_head": dict(method.source_receipt),
        "dense_representation": dense.representation.to_dict(),
        "method_representation": method.representation.to_dict(),
    }


def summarize_xd_quality_seeds(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep per-seed video CIs separate and disclose the fixed XD scope."""
    _require(bool(results), "no XD seed comparisons supplied")
    for result in results:
        _require(
            result["comparison_protocol"] == "xd-completed-frozen-runs-paired-quality-v1"
            and result["evaluation_protocol"] == xd.PROTOCOL_ID,
            "summary requires completed XD comparisons",
        )
        for key in (
            "metric",
            "scope_sha256",
            "canonical_metadata_sha256",
            "gt_file_sha256",
            "mapping",
            "training_scope",
        ):
            _require(result[key] == results[0][key], f"XD seed contracts differ: {key}")
        if result["method_representation"]["reducer"]["name"] == "global_uniform":
            _require(result["head_seed"] == 0, "global_uniform has only frozen XD seed 0")
    return {
        **shared.summarize_quality_seeds(results),
        "evaluation_protocol": xd.PROTOCOL_ID,
        "metric": "frame_pr_auc",
        "training_scope": results[0]["training_scope"],
    }


__all__ = ["compare_frozen_xd_quality", "paired_xd_frame_intervals", "summarize_xd_quality_seeds"]
