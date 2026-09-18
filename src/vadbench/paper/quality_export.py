"""Read completed, sealed UCF evaluations and compare one paired head seed.

This entry is for authorized post-freeze evaluation only. It does not accept
free-standing score JSON, invent an annotation grid, train/load a model, or
support XD. All source artifacts remain read-only.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vadbench.artifacts import PredictionRecord
from vadbench.data.audit import (
    OFFICIAL_UCF_CRIME_COUNTS,
    compute_manifest_sha256,
    verify_official_source_identity,
)
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.labels import frame_labels_for_record
from vadbench.data.manifest import DatasetSplit, VideoManifestRecord, load_manifest_jsonl
from vadbench.engine.coverage import aggregate_coverage, validate_frame_coverage
from vadbench.engine.evaluate import (
    _load_audit_report,
    _prediction_digest,
    _validate_official_audit,
    prediction_records_to_temporal,
)
from vadbench.features import FeatureRecord, compute_encoder_fingerprint
from vadbench.paper import evaluation, repeat_evaluation
from vadbench.paper.compatibility import (
    RepresentationIdentity,
    SamplingIdentity,
    feature_cache_key,
    validate_compatibility,
)
from vadbench.paper.extraction import _semantic_runtime_identity, _verified_code_digest
from vadbench.paper.quality_comparison import (
    DECISION_TOLERANCE,
    PairedVideoIntervals,
    compare_paired_quality,
)

QualityMode = Literal["refit_head", "direct_insert"]
_EVALUATION_STAGES = {"frozen_ucf_evaluation", "frozen_repeat_ucf_evaluation"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def _read(path: Path, sources: dict[str, str], expected: str | None = None) -> bytes:
    path = path.resolve()
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    _require(expected is None or digest == expected, f"artifact SHA mismatch: {path}")
    _require(str(path) not in sources or sources[str(path)] == digest, f"artifact changed while reading: {path}")
    sources[str(path)] = digest
    return data


def _json(path: Path, sources: dict[str, str], expected: str | None = None) -> dict[str, Any]:
    value = json.loads(_read(path, sources, expected))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _bound_path(entry: Mapping[str, Any], expected: Path) -> Path:
    _require(isinstance(entry, Mapping) and set(entry) == {"path", "sha256"}, "missing path/SHA artifact binding")
    _require(Path(entry["path"]).resolve() == expected.resolve(), f"artifact outside its expected run location: {expected}")
    return expected


def _bound_json(entry, path, sources):
    return _json(_bound_path(entry, path), sources, entry["sha256"])


def _prediction_records(entry, path, sources):
    data = _read(_bound_path(entry, path), sources, entry["sha256"])
    output = []
    for line in data.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("clip_index", "frame_start", "frame_end"):
            _require(type(row.get(key)) is int, f"prediction {key} must be an integer, not an implicit coercion")
        output.append(PredictionRecord.from_dict(row))
    _require(bool(output), "empty prediction export")
    return tuple(output)


def _verified_official_inputs(manifest_path: Path, audit_path: Path):
    records = load_manifest_jsonl(manifest_path)
    _require(len(records) == OFFICIAL_UCF_CRIME_COUNTS["test"]["total"] and all(r.split == DatasetSplit.TEST for r in records), "quality comparison requires the complete official 290-video test cohort")
    report = _load_audit_report(audit_path)
    manifest_digest = "sha256:" + compute_manifest_sha256(records)
    identity = _validate_official_audit(report, records, str(manifest_path.resolve()), manifest_digest)
    errors: list[dict[str, Any]] = []
    actual = verify_official_source_identity({"test": records}, registry_path=report["official_source_identity"]["registry"], errors=errors)
    _require(not errors and actual["status"] == "verified", "official source identity verification failed")
    for key in ("source_commit", "train_identity_sha256", "test_identity_sha256"):
        _require(actual[key] == report["official_source_identity"][key], f"official source identity drift: {key}")
    return records, identity


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
    _require(isinstance(reuse, Mapping), "external FeatureStore lacks a completed source evaluation binding")
    origin = Path(reuse["run_dir"]).resolve()
    result = _bound_json(reuse["result"], origin / "result.json", sources)
    source = _bound_json(reuse["resolved"], origin / "resolved.json", sources)
    _require(result.get("schema_version") == 1 and result.get("status") == "completed" and result.get("protocol") == "ucf-official-frozen-v1", "reused FeatureStore source evaluation is not completed official UCF")
    _require(result["resolved"] == reuse["resolved"] and result["artifacts"] == source["artifacts"] and source["artifacts"]["test_feature_store"] == entry, "reused FeatureStore artifact bindings differ from source evaluation")
    for key in ("freeze_sha256", "sealed_test_manifest_sha256", "sealed_audit_sha256", "evaluation_representation", "evaluation_representation_fingerprint", "evaluation_sampling", "evaluation_sampling_fingerprint", "evaluation_encoder_fingerprint"):
        _require(source[key] == resolved[key], f"reused FeatureStore source contract differs: {key}")
    stages = [_json(p, sources) for p in (origin / "provenance/stages").glob("*.json")]
    stages = [s for s in stages if s.get("stage") in _EVALUATION_STAGES]
    _require(len(stages) == 1 and stages[0].get("status") == "completed", "reused FeatureStore source stage is not completed")
    _require(stages[0]["config_sha256"] == hashlib.sha256(_canonical(stages[0]["config"])).hexdigest(), "reused source stage config SHA mismatch")
    for name, key in (("test_manifest", "sealed_test_manifest_sha256"), ("audit_report", "sealed_audit_sha256"), ("method_freeze", "freeze_sha256")):
        _require(stages[0]["inputs"][name]["sha256"] == source[key], "reused source stage sealed input mismatch")
    _require(_feature_store_root(origin, source, sources, visited) == feature_root, "reused FeatureStore origin differs")
    return feature_root


def _feature_contract(root, entry, resolved, manifest, sources):
    feature_root = _feature_store_root(root, resolved, sources)
    document = _bound_json(entry["resolved"], feature_root / "resolved.json", sources)
    status = _bound_json(entry["status"], feature_root / "status.json", sources)
    fingerprint = resolved["evaluation_encoder_fingerprint"]
    _require(status.get("completed") is True and status.get("status") == "completed" and status.get("failures") == [] and Path(status["feature_root"]).resolve() == feature_root, "test FeatureStore was not completed")
    _require(status.get("encoder_fingerprint") == fingerprint == document.get("encoder_fingerprint"), "test FeatureStore fingerprint mismatch")
    representation = RepresentationIdentity.from_mapping(resolved["evaluation_representation"])
    sampling = SamplingIdentity.from_mapping(resolved["evaluation_sampling"])
    _require(representation.fingerprint == resolved["evaluation_representation_fingerprint"] and sampling.fingerprint == resolved["evaluation_sampling_fingerprint"], "declared evaluation fingerprints do not match their definitions")
    _require(document["spec"]["representation"] == representation.to_dict() and document["spec"]["sampling"] == sampling.to_dict() and document["spec"]["sampling_kind"] == "dense", "test FeatureStore differs from declared representation/sampling")
    paper_identity = {"representation_fingerprint": representation.fingerprint, "sampling_fingerprint": sampling.fingerprint, "feature_cache_fingerprint": feature_cache_key(representation, sampling)}
    _require(document["paper_identity"] == paper_identity, "test FeatureStore paper identity mismatch")
    runtime = _semantic_runtime_identity(document["runtime"])
    _require(compute_encoder_fingerprint({"paper_pooled_cache": runtime}) == fingerprint, "actual runtime semantic fingerprint mismatch")
    _require(runtime["representation_fingerprint"] == representation.fingerprint and runtime["sampling_fingerprint"] == sampling.fingerprint and runtime["declared_readout"] == dict(representation.backbone.readout), "runtime declaration differs from representation/sampling/readout")
    _require(runtime["verified_weights_digest"] == representation.backbone.weights_digest and _verified_code_digest(document["runtime"]["verified_encoder_identity"], document["runtime"]) == representation.backbone.code_digest, "native runtime weights/code evidence differs from representation")
    _require(document["data_content_evidence"]["source_digest"] == sampling.source_digest == runtime["source_data_digest"], "test video content evidence differs from sampler/runtime")
    evidence = document["data_content_evidence"]
    _require(evidence["canonical_manifest_sha256"] == "sha256:" + compute_manifest_sha256(manifest), "test feature input manifest differs from sealed manifest")
    _require(evidence["source_digest"] == compute_encoder_fingerprint({"canonical_manifest": evidence["canonical_manifest_sha256"], "video_contents": evidence["videos"]}), "test data-content digest is not self-consistent")
    by_video = {r.video_id: r for r in manifest}
    contents = {r["video_id"]: r for r in evidence["videos"]}
    _require(set(contents) == set(by_video) and len(evidence["videos"]) == 290, "test video content evidence does not cover the sealed cohort")
    index_path = _bound_path(entry["index"], feature_root / "index.jsonl")
    index_digest, windows_digest = hashlib.sha256(), hashlib.sha256()
    keys = set()
    videos = set()
    previous_key = None
    current_video = None
    expected_samples = ()
    expected_window_count = 0
    sampler = DenseSamplingPlan(clip_frames=sampling.window["clip_frames"], frame_stride=sampling.stride["frame_stride"], window_stride=sampling.frame_selection["window_stride"], short_policy=sampling.frame_selection["short_video_policy"])
    with index_path.open("rb") as stream:
        for line in stream:
            index_digest.update(line)
            if not line.strip():
                continue
            raw = json.loads(line)
            _require(all(type(raw.get(k)) is int for k in ("clip_index", "frame_start", "frame_end")), "feature input indices must be exact integers")
            row = FeatureRecord.from_dict(raw)
            key = (row.video_id, row.clip_index)
            _require(key not in keys and (previous_key is None or key > previous_key), "test input windows duplicate or not canonically ordered")
            keys.add(key)
            videos.add(row.video_id)
            previous_key = key
            _require(row.encoder_fingerprint == fingerprint and row.metadata["paper_identity"] == paper_identity, "test feature row identity mismatch")
            _require(row.video_id in by_video, "test feature video is outside the sealed manifest")
            if row.video_id != current_video:
                current_video = row.video_id
                expected_samples = sampler.sample(by_video[row.video_id].num_frames)
                expected_window_count += len(expected_samples)
            _require(0 <= row.clip_index < len(expected_samples), "actual input clip index differs from frozen sampler")
            sample = expected_samples[row.clip_index]
            native_frames, native_mask = list(sample.frame_indices), list(sample.valid_mask)
            actual_sampling = row.metadata["sampling"]
            _require(row.frame_start == sample.score_frame_start and row.frame_end == sample.score_frame_end and actual_sampling.get("kind") == "dense" and actual_sampling.get("clip_index") == row.clip_index and actual_sampling.get("actual_frame_stride") == native_frames[1] - native_frames[0] and actual_sampling.get("end_anchored") == sample.end_anchored, "actual input window differs from frozen DenseSamplingPlan")
            for field, expected_value in (("frame_indices", native_frames), ("valid_mask", native_mask)):
                if field in actual_sampling:
                    _require(actual_sampling[field] == expected_value, f"exported actual input {field} differs from frozen sampler")
            observed = row.metadata["source_video"]
            _require(observed["actual_num_frames"] == by_video[row.video_id].num_frames and observed["actual_fps"] == by_video[row.video_id].fps and observed["content_sha256"] == contents[row.video_id]["sha256"] and observed["content_size_bytes"] == contents[row.video_id]["size_bytes"], "actual input video metadata differs from sealed content/decoded frame metadata")
            windows_digest.update(_canonical({"video_id": row.video_id, "clip_id": row.clip_id, "clip_index": row.clip_index, "frame_start": row.frame_start, "frame_end": row.frame_end, "start_s": row.start_s, "end_s": row.end_s, "native_frame_indices": native_frames, "native_valid_mask": native_mask, "source_video": row.metadata["source_video"]}))
            windows_digest.update(b"\n")
    _require(index_digest.hexdigest() == entry["index"]["sha256"], "test FeatureStore index SHA mismatch")
    sources[str(index_path)] = index_digest.hexdigest()
    _require(len(keys) == status["records_written_to_shards"] == expected_window_count and len(videos) == 290, "test FeatureStore does not contain every frozen sampler window")
    _require(resolved["coverage"].get("input_union_complete") is True and resolved["coverage"].get("videos") == 290, "original dense input windows did not cover every video")
    runtime.pop("representation_fingerprint")  # reducer differences are compared explicitly
    return representation, sampling, runtime, windows_digest.hexdigest(), document["data_content_evidence"], videos


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


def _load_completed_run(path: str | Path, mode: QualityMode) -> _CompletedRun:
    _require(mode in ("refit_head", "direct_insert"), "mode must be refit_head or direct_insert")
    root, sources = Path(path).expanduser().resolve(), {}
    result = _json(root / "result.json", sources)
    _require(result.get("schema_version") == 1 and result.get("status") == "completed" and result.get("protocol") == "ucf-official-frozen-v1", "run is not a completed frozen official UCF evaluation")
    resolved = _bound_json(result["resolved"], root / "resolved.json", sources)
    _require(resolved.get("schema_version") == 1 and result["artifacts"] == resolved["artifacts"], "completed artifact ledger differs from resolved ledger")
    stages = []
    for stage_path in sorted((root / "provenance/stages").glob("*.json")):
        stage = _json(stage_path, sources)
        if stage.get("stage") in _EVALUATION_STAGES:
            stages.append(stage)
    _require(len(stages) == 1 and stages[0].get("status") == "completed", "requires exactly one completed official evaluation stage")
    stage = stages[0]
    is_repeat = stage["stage"] == "frozen_repeat_ucf_evaluation"
    _require((resolved["method_source"].get("kind") == "frozen_head_repeat_v1") == is_repeat, "head source kind differs from evaluation stage")
    if is_repeat:
        reuse = resolved.get("reused_test_feature_store_from")
        _require(isinstance(reuse, Mapping), "repeat evaluation must bind its original completed test cache")
        origin = Path(reuse["run_dir"]).resolve()
        original = _bound_json(reuse["resolved"], origin / "resolved.json", sources)
        for key in ("method_source", "dense_source"):
            if resolved.get(key) is not None:
                _require(resolved[key]["source_controller"] == original.get(key), "repeat head does not derive from the corresponding cached seed-0 source")
                _require(resolved[key]["source_controller"]["head_seed"] == 0, "repeat cache source must be the frozen seed-0 head")
        _require(resolved.get("secondary_status") == ("completed" if resolved.get("dense_source") is not None else "not_requested"), "repeat secondary route completion status mismatch")
    config = stage["config"]
    _require(stage["config_sha256"] == hashlib.sha256(_canonical(config)).hexdigest(), "official stage config SHA mismatch")
    manifest_path, audit_path, freeze_path = (Path(stage["inputs"][k]["location"]).expanduser().resolve() for k in ("test_manifest", "audit_report", "method_freeze"))
    freeze, freeze_sha = evaluation._load_freeze(freeze_path)
    _read(freeze_path, sources, freeze_sha)
    _require(resolved["freeze_sha256"] == freeze_sha, "run method freeze differs from current frozen file")
    expected = {"test_manifest": (manifest_path, evaluation._UCF_TEST_MANIFEST_SHA256), "audit_report": (audit_path, evaluation._UCF_AUDIT_SHA256), "method_freeze": (freeze_path, freeze_sha)}
    for name, (file_path, digest) in expected.items():
        _require(stage["inputs"][name]["sha256"] == digest, f"official stage input identity mismatch: {name}")
        _read(file_path, sources, digest)
    _require(resolved["sealed_test_manifest_sha256"] == sources[str(manifest_path)] and resolved["sealed_audit_sha256"] == sources[str(audit_path)], "sealed manifest/audit identity mismatch")
    manifest, audit_identity = _verified_official_inputs(manifest_path, audit_path)
    representation, sampling, runtime, windows_sha, data_evidence, feature_videos = _feature_contract(root, resolved["artifacts"]["test_feature_store"], resolved, manifest, sources)
    _require(feature_videos == {r.video_id for r in manifest}, "actual test input window IDs differ from sealed manifest")
    _require(config["encoder"] == representation.backbone.runtime_id, "stage encoder differs from test representation")
    frozen_sampling = freeze["sampling"]
    _require(sampling.regime == "test_dense" and sampling.frame_selection.get("kind") == "dense_sliding" and sampling.window == {"clip_frames": frozen_sampling["native_clip_frames"][config["encoder"]]} and sampling.stride == {"frame_stride": frozen_sampling["frame_stride"]} and sampling.frame_selection.get("window_stride") == frozen_sampling["evaluation_window_stride"][config["encoder"]] and sampling.frame_selection.get("short_video_policy") == frozen_sampling["short_policy"], "evaluation sampling differs from the frozen native-input contract")
    source_objects = {}
    for key in ("method_source", "dense_source"):
        snapshot = resolved.get(key)
        if snapshot is None:
            continue
        _require(snapshot.get("kind") in (None, "frozen_head_repeat_v1"), "unknown frozen head source kind")
        _require((snapshot.get("kind") == "frozen_head_repeat_v1") == is_repeat, "both prediction routes must use the evaluation's head source kind")
        base_snapshot = snapshot["source_controller"] if is_repeat else snapshot
        source_roots = {Path(p).resolve().parent for p in base_snapshot["source_manifests_sha256"]}
        _require(source_roots == {Path(base_snapshot["source_manifest_root"]).resolve()} and len(base_snapshot["source_manifests_sha256"]) == 2, "frozen head source manifests must share their audited root")
        loader = repeat_evaluation.load_frozen_repeat_head_source if is_repeat else evaluation.load_frozen_detector_source
        loaded = loader(snapshot["run_dir"], freeze=freeze, expected_encoder=config["encoder"], role_lock_path=base_snapshot["training_role_lock"]["path"], head_data_contract_path=base_snapshot["head_data_contract"]["path"], source_manifest_root=base_snapshot["source_manifest_root"])
        actual_receipt = loaded.receipt() if is_repeat else evaluation._source_receipt(loaded)
        _require(actual_receipt == snapshot, "frozen head source changed or differs from resolved receipt")
        base = loaded.base if is_repeat else loaded
        for relative, digest in base.source_hashes.items():
            _read(base.root / relative, sources, digest)
        if is_repeat:
            for relative, digest in loaded.artifact_sha256.items():
                _read(loaded.root / relative, sources, digest)
        for field, stage_field in (("training_role_lock", "training_role_lock"), ("head_data_contract", "head_data_contract")):
            entry = base_snapshot[field]
            _require(stage["inputs"][stage_field]["sha256"] == entry["sha256"] and Path(stage["inputs"][stage_field]["location"]).resolve() == Path(entry["path"]).resolve(), "head data-contract/role-lock differs from the evaluated source")
            _read(Path(entry["path"]), sources, entry["sha256"])
        for source_path, digest in base_snapshot["source_manifests_sha256"].items():
            _read(Path(source_path), sources, digest)
        source_objects[key] = loaded
    _require("method_source" in source_objects and source_objects["method_source"].representation == representation, "evaluated representation differs from its frozen method source")
    if "dense_source" in source_objects:
        _require(source_objects["dense_source"].training_identity.seed == source_objects["method_source"].training_identity.seed, "primary and secondary head seeds differ within the evaluation run")
    source_key = "method_source" if mode == "refit_head" else "dense_source"
    _require(source_key in source_objects, "requested prediction branch has no frozen head source")
    source, source_receipt = source_objects[source_key], resolved[source_key]
    source_stage_key = ("repeat_checkpoint" if mode == "refit_head" else "dense_repeat_checkpoint") if is_repeat else ("method_checkpoint" if mode == "refit_head" else "dense_checkpoint")
    _require(stage["inputs"][source_stage_key]["sha256"] == source_receipt["checkpoint_sha256"], "stage checkpoint differs from selected prediction branch")
    declaration = source.detection_config(mode=mode, evaluation_representation=representation, evaluation_sampling=sampling, evaluation_encoder_fingerprint=resolved["evaluation_encoder_fingerprint"]).declaration
    permit = validate_compatibility(declaration)
    for branch_name, branch_suffix in (("primary", "primary-refit-head"), ("secondary", "secondary-dense-head-direct-insert")):
        prediction_entry = resolved["artifacts"][branch_name + "_predictions"]
        metric_entry = resolved["artifacts"][branch_name + "_metrics"]
        _require((prediction_entry is None) == (metric_entry is None), "completed prediction/metric artifact pair is incomplete")
        _require(branch_name != "primary" or prediction_entry is not None, "completed run has no primary prediction artifacts")
        if prediction_entry is not None:
            for entry, filename in ((prediction_entry, f"predictions-{branch_suffix}.jsonl"), (metric_entry, f"metrics-{branch_suffix}.json")):
                _read(_bound_path(entry, root / filename), sources, entry["sha256"])
    branch = "primary" if mode == "refit_head" else "secondary"
    suffix = "primary-refit-head" if mode == "refit_head" else "secondary-dense-head-direct-insert"
    records = _prediction_records(resolved["artifacts"][branch + "_predictions"], root / f"predictions-{suffix}.jsonl", sources)
    metrics = _bound_json(resolved["artifacts"][branch + "_metrics"], root / f"metrics-{suffix}.json", sources)
    run_ids = {row.run_id for row in records}
    _require(len(run_ids) == 1, "predictions mix run IDs")
    for row in records:
        _require(row.encoder_fingerprint == resolved["evaluation_encoder_fingerprint"], "prediction encoder fingerprint mismatch")
        _require(row.metadata.get("checkpoint_sha256") == source_receipt["checkpoint_sha256"], "prediction checkpoint SHA mismatch")
        _require(row.metadata.get("paper_compatibility") == permit, "prediction compatibility receipt mismatch")
        aggregation = row.metadata.get("dense_aggregation", {})
        _require(row.metadata.get("score_level") == "frame" and aggregation.get("interval_source") == "exact_equal_frame_run" and aggregation.get("reduction") == "mean", "predictions are not the frozen final mean-projected frame intervals")
    temporal, unit = prediction_records_to_temporal(records)
    _require(unit == "frames" and set(temporal) == {r.video_id for r in manifest}, "prediction video/frame identities differ from sealed manifest")
    grouped = {r.video_id: [] for r in manifest}
    for row in records:
        grouped[row.video_id].append(row)
    coverage = []
    for record in manifest:
        rows = grouped[record.video_id]
        coverage.append(validate_frame_coverage(video_id=record.video_id, clip_indices=np.array([r.clip_index for r in rows]), frame_starts=np.array([r.frame_start for r in rows]), frame_ends=np.array([r.frame_end for r in rows]), num_frames=record.num_frames, fps=record.fps, require_fps=True))
    _require(metrics.get("protocol") == "ucf-crime/official-frameauc-v1" and metrics.get("coverage", {}).get("complete") is True, "metrics do not attest complete official frame evaluation")
    _require(metrics["coverage"] == {"status": "validated", **aggregate_coverage(coverage)}, "metric final-frame coverage receipt mismatch")
    identity = metrics["input_identity"]
    _require(identity.get("predictions_sha256") == _prediction_digest(records) and identity.get("manifest_sha256") == "sha256:" + compute_manifest_sha256(manifest), "metric/prediction canonical identity mismatch")
    _require(identity.get("dataset_audit") == audit_identity and identity.get("videos") == 290 and identity.get("prediction_records") == len(records), "metric official audit/count identity mismatch")
    _require(identity.get("encoder_fingerprint") == resolved["evaluation_encoder_fingerprint"] and identity.get("checkpoint_sha256") == source_receipt["checkpoint_sha256"] and identity.get("run_id") in run_ids, "metric run/head identity mismatch")
    for loaded in source_objects.values():
        loaded.verify_unchanged()
    return _CompletedRun(root, mode, records, manifest, metrics, source, representation, sampling, runtime, windows_sha, data_evidence, freeze_sha, sources[str(manifest_path)], sources[str(audit_path)], sources, source_receipt)


def paired_frame_intervals(manifest, dense_records, method_records) -> tuple[PairedVideoIntervals, ...]:
    """Split final score runs at both methods' boundaries and actual GT changes.

    Labels come only from the repository's source-annotation projection, never
    PredictionRecord.ground_truth (which is merely a video-level weak label).
    """
    dense, dense_unit = prediction_records_to_temporal(dense_records)
    method, method_unit = prediction_records_to_temporal(method_records)
    ids = {r.video_id for r in manifest}
    _require(dense_unit == method_unit == "frames" and set(dense) == set(method) == ids, "both predictions must cover the exact manifest in frame coordinates")
    output = []
    for record in manifest:
        predictions = [dense[record.video_id], method[record.video_id]]
        ordered = []
        for prediction in predictions:
            indices = np.argsort(prediction.intervals[:, 0], kind="stable")
            intervals, scores = prediction.intervals[indices], prediction.scores[indices]
            validate_frame_coverage(video_id=record.video_id, clip_indices=np.arange(len(intervals)), frame_starts=intervals[:, 0], frame_ends=intervals[:, 1], num_frames=record.num_frames, fps=record.fps, require_fps=True)
            ordered.append((intervals, scores))
        labels = frame_labels_for_record(record)
        changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        boundaries = np.unique(np.concatenate([ordered[0][0].ravel(), ordered[1][0].ravel(), changes]))
        starts, ends = boundaries[:-1], boundaries[1:]
        aligned_scores = [scores[np.searchsorted(intervals[:, 1], starts, side="right")] for intervals, scores in ordered]
        output.append(PairedVideoIntervals(record.video_id, int(record.is_anomaly), record.num_frames, np.column_stack([starts, ends]), labels[starts], aligned_scores[0], aligned_scores[1]))
    return tuple(output)


def compare_frozen_ucf_quality(dense_run: str | Path, method_run: str | Path, *, mode: QualityMode = "refit_head") -> dict[str, Any]:
    """Read two completed audited runs and compare one matched head seed."""
    dense = _load_completed_run(dense_run, "refit_head")
    method = _load_completed_run(method_run, mode)
    _require(dense.representation.reducer.get("name") == "identity", "dense baseline must use the identity reducer")
    for name in ("freeze_sha256", "manifest_sha256", "audit_sha256", "sampling", "native_runtime", "input_windows_sha256", "data_evidence"):
        _require(getattr(dense, name) == getattr(method, name), f"dense/method evaluation contract mismatch: {name}")
    for name in ("backbone", "output_dim", "precision", "position_strategy"):
        _require(getattr(dense.representation, name) == getattr(method.representation, name), f"only reducer differences are allowed: {name}")
    _require(dense.source.training_identity == method.source.training_identity, "heads differ in seed, fit split, or training identity")
    _require(dense.source_receipt["head_budget"] == method.source_receipt["head_budget"], "head training budgets differ")
    if mode == "direct_insert":
        _require(dense.source_receipt["checkpoint_sha256"] == method.source_receipt["checkpoint_sha256"], "direct_insert must use the exact dense baseline head checkpoint")
    sources = dict(dense.artifact_sha256)
    for path, digest in method.artifact_sha256.items():
        _require(path not in sources or sources[path] == digest, "shared source artifact changed across paired reads")
        sources[path] = digest
    videos = paired_frame_intervals(dense.manifest, dense.records, method.records)
    result = compare_paired_quality(videos, metric="frame_roc_auc", source_sha256=sources)
    for run, metric_key in ((dense, "dense_value"), (method, "method_value")):
        _require(result[metric_key] is not None and abs(result[metric_key] - run.metrics["frame_auc"]) <= DECISION_TOLERANCE, "saved official metric disagrees with reconstructed frame intervals/GT")
        _require(run.metrics["num_videos"] == result["n_videos"] and run.metrics["num_frames"] == result["n_frames"] and run.metrics["num_positive_frames"] == result["positive_frames"] and run.metrics["num_negative_frames"] == result["negative_frames"], "saved official metric counts disagree with sealed frame truth")
    return {**result, "comparison_protocol": "ucf-completed-frozen-runs-paired-quality-v1", "mode": mode, "head_seed": method.source.training_identity.seed, "encoder": method.representation.backbone.runtime_id, "sealed_manifest_sha256": method.manifest_sha256, "sealed_audit_sha256": method.audit_sha256, "freeze_sha256": method.freeze_sha256, "input_windows_sha256": method.input_windows_sha256, "input_window_verification": "all frozen DenseSamplingPlan windows; exported endpoints/actual_stride/end_anchor checked; native frames/masks reconstructed by exact sampler, explicit exported arrays checked when present", "dense_run": str(dense.root), "method_run": str(method.root), "dense_head": dict(dense.source_receipt), "method_head": dict(method.source_receipt), "dense_representation": dense.representation.to_dict(), "method_representation": method.representation.to_dict()}


def summarize_quality_seeds(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe one seed or all frozen seeds 0/1/2; never select a best seed."""
    _require(bool(results), "no seed comparisons supplied")
    seeds = [r["head_seed"] for r in results]
    _require(all(type(s) is int and s in (0, 1, 2) for s in seeds) and len(set(seeds)) == len(seeds) and (len(seeds) == 1 or set(seeds) == {0, 1, 2}), "use one seed or the complete 0/1/2 set without duplicates")
    first = results[0]
    for result in results:
        for key in ("mode", "encoder", "sealed_manifest_sha256", "sealed_audit_sha256", "freeze_sha256", "input_windows_sha256", "dense_representation", "method_representation", "n_videos", "n_frames", "video_order"):
            _require(result[key] == first[key], f"seed runs are not the same comparison: {key}")
        _require(result["status"] == "available" and result["delta_method_minus_dense"] is not None, "cannot aggregate an unavailable seed comparison")
        _require(result["head_seed"] == result["dense_head"]["head_seed"] == result["method_head"]["head_seed"], "summary seed differs from the actual paired source heads")
    for role in ("dense_head", "method_head"):
        _require(len({r[role]["checkpoint_sha256"] for r in results}) == len(results), "reusing one checkpoint is not an independent head seed")
    deltas = np.array([r["delta_method_minus_dense"] for r in results])
    return {"status": "completed", "head_seeds": sorted(seeds), "head_seed_count": len(seeds), "n_videos_per_seed": first["n_videos"], "n_frames_per_seed": first["n_frames"], "mean_delta_across_head_seeds": float(deltas.mean()), "std_delta_across_head_seeds_ddof1": float(deltas.std(ddof=1)) if len(seeds) > 1 else None, "delta_range_across_head_seeds": [float(deltas.min()), float(deltas.max())], "video_ci": "reported separately per seed; not averaged and seeds are not extra videos", "best_seed_selection": False, "seed_results": [dict(r) for r in sorted(results, key=lambda r: r["head_seed"])]}


__all__ = ["compare_frozen_ucf_quality", "paired_frame_intervals", "summarize_quality_seeds"]
