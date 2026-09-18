"""Provenance-bound UR-DMU prediction over native dense FeatureStores.

UR-DMU is trained on author-style 200-bin bags, while Test-mode inference in
this module retains the complete native dense window sequence and its
source-frame intervals.  It verifies both identities, runs the frozen network
once per full video sequence, and reuses standard dense interval aggregation
to emit ordinary frame prediction records.

It deliberately does *not* open temporal ground truth. Development runs can
report a video-level ROC-AUC from permitted video labels. Official frame
prediction is fail-closed: the historical reducer freeze covers superseded
engineering prototypes and cannot authorize the current property-first method.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vadbench.artifacts import PredictionRecord, new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.dense_sampling import DenseSamplingPlan, aggregate_interval_scores
from vadbench.data.manifest import DatasetSplit, VideoManifestRecord, load_manifest_jsonl
from vadbench.features import FeatureRecord, FeatureStore, atomic_write_json, atomic_write_jsonl
from vadbench.metrics import roc_auc_score
from vadbench.paper.compatibility import (
    CompatibilityDeclaration,
    RepresentationIdentity,
    SamplingIdentity,
    TrainingIdentity,
    feature_cache_key,
    validate_compatibility,
)
from vadbench.paper.urdmu_backend import UPSTREAM_COMMIT, build_urdmu
from vadbench.paper.urdmu_training import COMPONENTS

try:  # Kept optional at import time, like the rest of the paper entry points.
    import torch
except ImportError:  # pragma: no cover - exercised in installations without train extras.
    torch = None


EvaluationPhase = Literal["development_video", "official_frame"]
EvaluationMode = Literal["direct_insert", "refit_head"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: str | Path, *, name: str) -> dict[str, Any]:
    value = Path(path).expanduser().resolve()
    try:
        document = json.loads(value.read_text(encoding="utf8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} is missing: {value}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is invalid JSON: {value}") from exc
    if not isinstance(document, Mapping):
        raise ValueError(f"{name} must contain an object")
    return dict(document)


def _sha(value: str | Path, *, name: str) -> tuple[Path, str]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path, sha256_file(path)


def _same_path(value: str | Path, expected: Path, *, name: str) -> None:
    _require(Path(value).expanduser().resolve() == expected, f"{name} path differs")


@dataclass(frozen=True)
class URDMUEvaluationRequest:
    """One sealed UR-DMU prediction attempt.

    ``development_video`` accepts only a development-trained fit head and a
    select FeatureStore.  It never accepts engineering artifacts.  The
    ``official_frame`` remains explicit so a caller cannot relabel development
    output as an official result. It currently fails before opening any test
    source until a v3 property-evidence and same-mechanism TF/LoRA freeze is
    available. It never writes an official metric.
    """

    dataset: Literal["ucf_crime", "xd_violence"]
    phase: EvaluationPhase
    mode: EvaluationMode
    trained_run: str
    feature_store: str
    evaluation_manifest: str
    upstream_dir: str
    output_root: str
    device: str
    feature_contract_path: str | None = None
    feature_contract_sha256: str | None = None
    freeze_path: str | None = None
    audit_report: str | None = None
    max_attention_workspace_bytes: int | None = None
    query_chunk_size: int | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        _require(self.phase in {"development_video", "official_frame"}, "unknown UR-DMU phase")
        _require(self.mode in {"direct_insert", "refit_head"}, "unknown UR-DMU evaluation mode")
        _require(bool(self.dataset and self.trained_run and self.feature_store), "UR-DMU paths are required")
        _require(bool(self.evaluation_manifest and self.upstream_dir and self.output_root), "UR-DMU paths are required")
        _require(bool(self.device), "device is required")
        _require(
            self.run_id is None
            or bool(self.run_id)
            and self.run_id not in {".", ".."}
            and not any(char in self.run_id for char in "/\\"),
            "run_id must be a basename",
        )
        if self.phase == "official_frame":
            _require(bool(self.freeze_path), "official UR-DMU prediction requires a method freeze")
            _require(bool(self.audit_report), "official UR-DMU prediction requires a dataset audit receipt")
        else:
            _require(self.freeze_path is None and self.audit_report is None, "development UR-DMU prediction cannot take official-test inputs")
            _require(
                bool(self.feature_contract_path and self.feature_contract_sha256),
                "development UR-DMU prediction requires the SHA-bound development-select extraction contract",
            )
        _require(
            self.max_attention_workspace_bytes is None
            or type(self.max_attention_workspace_bytes) is int
            and self.max_attention_workspace_bytes > 0,
            "max_attention_workspace_bytes must be a positive integer or null",
        )
        _require(
            self.query_chunk_size is None
            or type(self.query_chunk_size) is int
            and self.query_chunk_size > 0,
            "query_chunk_size must be a positive integer or null",
        )


@dataclass(frozen=True)
class URDMUEvaluationResult:
    run_dir: str
    predictions: str
    completed: bool
    phase: EvaluationPhase
    video_metrics: Mapping[str, Any] | None


@dataclass(frozen=True)
class _FrozenURDMUSource:
    root: Path
    checkpoint: Path
    checkpoint_sha256: str
    metadata: Mapping[str, Any]
    training_qa: Mapping[str, Any]
    representation: RepresentationIdentity
    sampling: SamplingIdentity
    encoder_fingerprint: str
    fit_manifest_path: Path
    fit_manifest_digest: str
    fit_records: tuple[VideoManifestRecord, ...]
    source_hashes: Mapping[str, str]

    def verify_unchanged(self) -> None:
        for path, digest in self.source_hashes.items():
            if sha256_file(path) != digest:
                raise ValueError(f"UR-DMU source changed after verification: {path}")


@dataclass(frozen=True)
class _DenseFeatureSource:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    records: tuple[VideoManifestRecord, ...]
    rows_by_video: Mapping[str, tuple[FeatureRecord, ...]]
    representation: RepresentationIdentity
    sampling: SamplingIdentity
    encoder_fingerprint: str
    source_hashes: Mapping[str, str]

    def verify_unchanged(self) -> None:
        for path, digest in self.source_hashes.items():
            if sha256_file(path) != digest:
                raise ValueError(f"dense FeatureStore changed after verification: {path}")


def _checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    if torch is None:
        raise ImportError("PyTorch is required for UR-DMU prediction")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"cannot read UR-DMU checkpoint: {path}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("UR-DMU checkpoint lacks a model_state_dict")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("UR-DMU checkpoint lacks training metadata")
    return dict(metadata)


def _fit_manifest_from_training_stage(
    root: Path, metadata: Mapping[str, Any]
) -> tuple[Path, str, tuple[VideoManifestRecord, ...]]:
    """Recover the actual fit source from the recorded training stage.

    ``source_training_view.manifest_sha256`` denotes the full authority view.
    It cannot serve as a fit digest for a development head.  The checkpoint's
    immutable metadata records SHA-bound source files, while the completed run
    stage names which one was actually passed as ``train_manifest``.
    """

    candidates = []
    for path in sorted((root / "provenance" / "stages").glob("*.json")):
        stage = _json(path, name="UR-DMU training provenance stage")
        if stage.get("stage") == "urdmu_training" and stage.get("status") == "completed":
            candidates.append((path, stage))
    _require(
        len(candidates) == 1,
        "UR-DMU run must contain exactly one completed training provenance stage",
    )
    _stage_path, stage = candidates[0]
    config, inputs = stage.get("config"), stage.get("inputs")
    _require(
        isinstance(config, Mapping)
        and config.get("run_mode") == "development"
        and isinstance(config.get("train_manifest"), str)
        and isinstance(inputs, Mapping)
        and isinstance(inputs.get("train_manifest"), Mapping),
        "UR-DMU training stage lacks its development fit manifest binding",
    )
    binding = inputs["train_manifest"]
    manifest_path = Path(config["train_manifest"]).expanduser().resolve()
    _require(
        Path(binding.get("location", "")).expanduser().resolve() == manifest_path
        and isinstance(binding.get("sha256"), str),
        "UR-DMU training stage manifest config/input binding differs",
    )
    sources = metadata.get("source_sha256")
    _require(
        isinstance(sources, Mapping)
        and sources.get(str(manifest_path)) == binding["sha256"]
        and sha256_file(manifest_path) == binding["sha256"],
        "UR-DMU checkpoint does not SHA-bind its actual development fit manifest",
    )
    records = load_manifest_jsonl(manifest_path)
    _require(records, "UR-DMU development fit manifest is empty")
    return manifest_path, "sha256:" + compute_manifest_sha256(records), records


def _load_source(request: URDMUEvaluationRequest) -> _FrozenURDMUSource:
    root = Path(request.trained_run).expanduser().resolve()
    result_path, result_sha = _sha(root / "result.json", name="UR-DMU result receipt")
    qa_path, qa_sha = _sha(root / "training_qa.json", name="UR-DMU training QA")
    result, qa = _json(result_path, name="UR-DMU result receipt"), _json(qa_path, name="UR-DMU training QA")
    _require(result.get("status") == "completed", "UR-DMU training run is not completed")
    _require(qa.get("status") == "passed", "UR-DMU training QA did not pass")
    checkpoint = Path(result.get("checkpoint_path", "")).expanduser().resolve()
    _require(checkpoint.parent == (root / "checkpoints").resolve() and checkpoint.name == "final.pt", "UR-DMU result does not bind final.pt in its own run")
    checkpoint, checkpoint_sha = _sha(checkpoint, name="UR-DMU final checkpoint")
    sidecar, sidecar_sha = _sha(checkpoint.with_suffix(".pt.json"), name="UR-DMU checkpoint manifest")
    _require(result.get("checkpoint_sha256") == checkpoint_sha, "UR-DMU result checkpoint SHA differs")
    qa_checkpoint = qa.get("checkpoint")
    _require(isinstance(qa_checkpoint, Mapping), "UR-DMU QA lacks checkpoint binding")
    _same_path(qa_checkpoint.get("path", ""), checkpoint, name="UR-DMU QA checkpoint")
    _require(qa_checkpoint.get("sha256") == checkpoint_sha, "UR-DMU QA checkpoint SHA differs")
    metadata = _checkpoint_metadata(checkpoint)
    _require(metadata.get("schema") == "urdmu.training/v1", "checkpoint is not a UR-DMU training artifact")
    _require(metadata.get("status") == "completed_training", "UR-DMU checkpoint training is incomplete")
    _require(metadata.get("dataset") == request.dataset, "UR-DMU checkpoint dataset differs")
    _require(metadata.get("encoder_fingerprint") and isinstance(metadata.get("encoder_fingerprint"), str), "UR-DMU checkpoint lacks encoder fingerprint")
    backend = metadata.get("backend")
    _require(isinstance(backend, Mapping) and backend.get("upstream_commit") == UPSTREAM_COMMIT, "UR-DMU checkpoint upstream commit differs")
    _require(metadata.get("checkpoint_selection") == "fixed_final_step_no_test_selection", "UR-DMU checkpoint selection is not the fixed final step")
    _require(
        metadata.get("steps") == 3000
        and metadata.get("bags_per_class") == 64
        and qa_checkpoint.get("step") == 3000,
        "UR-DMU QA does not bind the fixed 3000-step 64+64 training budget",
    )
    _require(
        qa.get("nonzero_gradient_steps") == 3000
        and metadata.get("nonzero_gradient_steps") == 3000,
        "UR-DMU QA lacks nonzero-gradient evidence for every fixed training step",
    )
    changed = qa.get("changed_parameters_by_component")
    _require(
        isinstance(changed, Mapping)
        and set(changed) == set(COMPONENTS)
        and all(type(value) is int and value > 0 for value in changed.values()),
        "UR-DMU QA lacks updates for all seven required components",
    )
    parity = qa.get("reload_parity")
    _require(isinstance(parity, Mapping) and parity.get("exact_equal") is True and parity.get("flag") == "Test" and parity.get("training") is False, "UR-DMU QA lacks exact Test-mode reload parity")
    representation = RepresentationIdentity.from_mapping(metadata.get("representation", {}))
    sampling = SamplingIdentity.from_mapping(metadata.get("sampling", {}))
    _require(metadata.get("representation_fingerprint") == representation.fingerprint, "UR-DMU checkpoint representation fingerprint differs")
    _require(metadata.get("sampling_fingerprint") == sampling.fingerprint, "UR-DMU checkpoint sampling fingerprint differs")
    _require(representation.output_dim in {768, 1024}, "UR-DMU checkpoint feature dimension is unsupported")
    fit_manifest_path, fit_manifest_digest, fit_records = _fit_manifest_from_training_stage(
        root, metadata
    )
    source_hashes = {
        str(result_path): result_sha,
        str(qa_path): qa_sha,
        str(checkpoint): checkpoint_sha,
        str(sidecar): sidecar_sha,
        str(fit_manifest_path): sha256_file(fit_manifest_path),
    }
    return _FrozenURDMUSource(
        root,
        checkpoint,
        checkpoint_sha,
        metadata,
        qa,
        representation,
        sampling,
        str(metadata["encoder_fingerprint"]),
        fit_manifest_path,
        fit_manifest_digest,
        fit_records,
        source_hashes,
    )


def _phase_gate(
    request: URDMUEvaluationRequest, source: _FrozenURDMUSource, target: _DenseFeatureSource
) -> dict[str, str]:
    metadata = source.metadata
    if request.phase == "development_video":
        _require(metadata.get("run_mode") == "development" and metadata.get("data_role") == "development-fit" and metadata.get("development_role") == "fit", "development prediction requires a development-fit UR-DMU head, never an engineering artifact")
        _require(target.records and all(record.split == DatasetSplit.TRAIN for record in target.records), "development prediction may use only train-partition records")
        roles = {record.metadata.get("original_role") for record in target.records}
        _require(roles == {"select"}, "development prediction requires exactly the frozen select role")
        contract_path, contract_sha = _sha(
            request.feature_contract_path, name="development-select extraction contract"
        )
        _require(contract_sha == request.feature_contract_sha256, "development-select extraction contract SHA differs")
        contract = _json(contract_path, name="development-select extraction contract")
        _require(
            contract.get("schema") == "icassp2027.official-dense-extraction/v1"
            and contract.get("status") == "ready"
            and contract.get("data_role") == "development-select"
            and contract.get("development_role") == "select"
            and contract.get("dataset") == request.dataset,
            "development prediction requires the authorized development-select extraction contract",
        )
        head_view = metadata.get("source_training_view")
        head_lock = head_view.get("development_role_lock") if isinstance(head_view, Mapping) else None
        _require(
            isinstance(head_lock, Mapping)
            and contract.get("original_role_lock") == dict(head_lock),
            "development select contract role lock differs from the development-fit head",
        )
        features = contract.get("feature_store")
        _require(isinstance(features, Mapping), "development-select contract lacks FeatureStore binding")
        _require(
            features.get("root") == str(target.root)
            and all(
                isinstance(features.get(name), Mapping)
                and Path(features[name].get("path", "")).resolve() == target.root / filename
                and features[name].get("sha256") == target.source_hashes[str(target.root / filename)]
                for name, filename in (("resolved", "resolved.json"), ("status", "status.json"), ("index", "index.jsonl"))
            ),
            "development-select contract FeatureStore binding differs from scoring input",
        )
        manifest = contract.get("training_manifest")
        _require(
            isinstance(manifest, Mapping)
            and Path(manifest.get("path", "")).resolve() == target.manifest_path
            and manifest.get("sha256") == target.manifest_sha256,
            "development-select contract manifest differs from scoring input",
        )
        view = contract.get("training_view_contract")
        _require(
            isinstance(view, Mapping)
            and isinstance(view.get("path"), str)
            and isinstance(view.get("sha256"), str),
            "development-select contract lacks its authoritative training-view binding",
        )
        view_path, view_sha = _sha(view["path"], name="authoritative training-view contract")
        _require(view_sha == view["sha256"], "development-select authoritative training-view SHA differs")
        recorded_sources = metadata.get("source_sha256")
        _require(
            isinstance(recorded_sources, Mapping)
            and recorded_sources.get(str(view_path)) == view_sha,
            "development-fit head is not bound to this authoritative training-view contract",
        )
        from vadbench.data.official_training import load_official_training_view

        authoritative, authority = load_official_training_view(
            view_path, view_sha, dataset_root=contract.get("dataset_root")
        )
        _require(
            authority.get("dataset") == request.dataset
            and authority.get("contract_path") == str(view_path)
            and authority.get("contract_sha256") == view_sha,
            "authoritative training view differs from the development-select contract",
        )
        inputs = authority.get("inputs")
        _require(
            isinstance(inputs, Mapping) and isinstance(inputs.get("role_lock"), Mapping),
            "authoritative training view lacks an original role lock",
        )
        raw_lock = inputs["role_lock"]
        relative_lock = Path(raw_lock.get("path", ""))
        _require(
            not relative_lock.is_absolute() and ".." not in relative_lock.parts,
            "authoritative role-lock path escapes the training-view contract",
        )
        lock_path, lock_sha = _sha(
            view_path.parent / relative_lock, name="authoritative original role lock"
        )
        _require(lock_sha == raw_lock.get("sha256"), "authoritative original role lock SHA differs")
        expected_lock = {"path": str(lock_path), "sha256": lock_sha}
        _require(
            contract.get("original_role_lock") == expected_lock and dict(head_lock) == expected_lock,
            "development head/select contract role lock differs from authoritative evidence",
        )
        lock = _json(lock_path, name="authoritative original role lock")
        partitions = lock.get("partitions")
        _require(
            isinstance(partitions, Mapping)
            and set(partitions) == {record.video_id for record in authoritative}
            and all(partitions[record.video_id] == record.metadata.get("original_role") for record in authoritative),
            "authoritative original role lock does not match the full training view",
        )
        expected_select = tuple(
            record for record in authoritative if partitions[record.video_id] == "select"
        )
        expected_fit = tuple(
            record for record in authoritative if partitions[record.video_id] == "fit"
        )
        _require(
            expected_fit
            and [record.to_dict() for record in source.fit_records]
            == [record.to_dict() for record in expected_fit],
            "development head fit manifest differs from the complete authoritative fit role",
        )
        _require(
            expected_select
            and [record.to_dict() for record in target.records]
            == [record.to_dict() for record in expected_select],
            "development FeatureStore does not contain the complete authoritative select role",
        )
        expected_content = {
            record.video_id: (
                record.metadata.get("content_sha256"),
                record.metadata.get("content_size_bytes"),
            )
            for record in expected_select
        }
        evidence = _json(target.root / "resolved.json", name="dense FeatureStore resolved receipt")[
            "data_content_evidence"
        ]
        observed_content = {
            item.get("video_id"): (item.get("sha256"), item.get("size_bytes"))
            for item in evidence["videos"]
        }
        _require(
            observed_content == expected_content,
            "development select FeatureStore content evidence differs from authoritative videos",
        )
        return {
            str(contract_path): contract_sha,
            str(view_path): view_sha,
            str(lock_path): lock_sha,
        }
    raise NotImplementedError(
        "official_frame is blocked: reducer-evaluation-freeze-v1 only binds "
        "superseded engineering prototypes. A new v3 property-evidence and "
        "same-mechanism TF/LoRA method-freeze contract is required before this "
        "evaluator may read any official test source."
    )


def _load_dense_feature_source(
    root_value: str | Path, manifest_value: str | Path
) -> _DenseFeatureSource:
    root = Path(root_value).expanduser().resolve()
    manifest_path, manifest_sha = _sha(manifest_value, name="evaluation manifest")
    records = load_manifest_jsonl(manifest_path)
    _require(records, "evaluation manifest is empty")
    by_id = {record.video_id: record for record in records}
    _require(len(by_id) == len(records), "evaluation manifest has duplicate video IDs")
    resolved_path, resolved_sha = _sha(root / "resolved.json", name="dense FeatureStore resolved receipt")
    status_path, status_sha = _sha(root / "status.json", name="dense FeatureStore status receipt")
    index_path, index_sha = _sha(root / "index.jsonl", name="dense FeatureStore index")
    resolved, status = _json(resolved_path, name="dense FeatureStore resolved receipt"), _json(status_path, name="dense FeatureStore status receipt")
    _require(status.get("status") == "completed" and status.get("completed") is True and status.get("failures") == [], "dense FeatureStore is not completed")
    _same_path(status.get("feature_root", ""), root, name="dense FeatureStore status root")
    spec = resolved.get("spec")
    _require(isinstance(spec, Mapping) and spec.get("sampling_kind") == "dense", "FeatureStore is not a native dense extraction")
    representation = RepresentationIdentity.from_mapping(spec.get("representation", {}))
    sampling = SamplingIdentity.from_mapping(spec.get("sampling", {}))
    _require(sampling.regime == "test_dense", "FeatureStore sampling is not the native dense regime")
    fingerprint = resolved.get("encoder_fingerprint")
    _require(isinstance(fingerprint, str) and status.get("encoder_fingerprint") == fingerprint, "dense FeatureStore encoder fingerprint differs")
    paper_identity = {"representation_fingerprint": representation.fingerprint, "sampling_fingerprint": sampling.fingerprint, "feature_cache_fingerprint": feature_cache_key(representation, sampling)}
    _require(resolved.get("paper_identity") == paper_identity, "dense FeatureStore paper identity differs")
    evidence = resolved.get("data_content_evidence")
    _require(isinstance(evidence, Mapping), "dense FeatureStore lacks data content evidence")
    _require(evidence.get("canonical_manifest_sha256") == "sha256:" + compute_manifest_sha256(records), "dense FeatureStore manifest identity differs")
    _require(evidence.get("source_digest") == sampling.source_digest, "dense FeatureStore content/sampling digest differs")
    evidence_videos = evidence.get("videos")
    _require(isinstance(evidence_videos, list) and {row.get("video_id") for row in evidence_videos if isinstance(row, Mapping)} == set(by_id) and len(evidence_videos) == len(records), "dense FeatureStore content evidence does not cover the evaluation manifest")
    store = FeatureStore(root)
    rows = list(store.iter_records())
    _require(rows and len(rows) == status.get("records_written_to_shards"), "dense FeatureStore row count differs from completed receipt")
    rows_by_video: dict[str, list[FeatureRecord]] = {video_id: [] for video_id in by_id}
    for row in rows:
        _require(row.video_id in rows_by_video and row.encoder_fingerprint == fingerprint, "dense FeatureStore contains a foreign video or encoder identity")
        _require(row.metadata.get("paper_identity") == paper_identity, "dense FeatureStore row paper identity differs")
        rows_by_video[row.video_id].append(row)
    sampler = DenseSamplingPlan(
        clip_frames=sampling.window["clip_frames"],
        frame_stride=sampling.stride["frame_stride"],
        window_stride=sampling.frame_selection["window_stride"],
        short_policy=sampling.frame_selection["short_video_policy"],
    )
    complete: dict[str, tuple[FeatureRecord, ...]] = {}
    for video_id, manifest in by_id.items():
        actual = tuple(sorted(rows_by_video[video_id], key=lambda row: row.clip_index))
        expected = sampler.sample(manifest.num_frames)
        _require(len(actual) == len(expected), f"{video_id}: dense FeatureStore lacks sampler windows")
        for row, sample in zip(actual, expected, strict=True):
            _require(row.clip_index == sample.clip_index and row.frame_start == sample.score_frame_start and row.frame_end == sample.score_frame_end, f"{video_id}: dense FeatureStore window differs from the declared sampler")
            _require(math.isclose(row.start_s, sample.score_frame_start / manifest.fps) and math.isclose(row.end_s, sample.score_frame_end / manifest.fps), f"{video_id}: dense FeatureStore timeline differs from manifest FPS")
            observed = row.metadata.get("sampling")
            _require(isinstance(observed, Mapping) and observed.get("kind") == "dense" and observed.get("clip_index") == sample.clip_index, f"{video_id}: dense FeatureStore row lacks native sampling identity")
        complete[video_id] = actual
    source_hashes = {str(manifest_path): manifest_sha, str(resolved_path): resolved_sha, str(status_path): status_sha, str(index_path): index_sha}
    return _DenseFeatureSource(root, manifest_path, manifest_sha, records, complete, representation, sampling, str(fingerprint), source_hashes)


def _verify_pairing(request: URDMUEvaluationRequest, source: _FrozenURDMUSource, target: _DenseFeatureSource) -> None:
    expected_role = "refit_head" if request.mode == "refit_head" else "dense_reference"
    _require(
        source.metadata.get("checkpoint_role") == expected_role,
        f"{request.mode} prediction requires an UR-DMU {expected_role} checkpoint",
    )
    seed = source.metadata.get("seed")
    _require(type(seed) is int, "UR-DMU head lacks an integer training seed")
    declaration = CompatibilityDeclaration(
        mode=request.mode,
        training_representation=source.representation,
        evaluation_representation=target.representation,
        training_sampling=source.sampling,
        evaluation_sampling=target.sampling,
        baseline_evaluation_sampling=target.sampling,
        training_identity=TrainingIdentity(
            {"kind": "urdmu", "checkpoint_role": expected_role},
            source.fit_manifest_digest,
            seed,
            {"steps": source.metadata.get("steps"), "bags_per_class": source.metadata.get("bags_per_class")},
        ),
        sampling_change="none",
    )
    validate_compatibility(declaration)


def _attention_workspace_lower_bound_bytes(length: int) -> int:
    """Lower bound for one original Test-mode attention layer's live tensors.

    The pinned upstream uses four heads and materializes ``dots``, ``attn1``,
    an unheaded temporal-distance matrix, and repeated ``attn2``.  At fp32
    this is ``(3 * 4 + 1) * N² * 4`` bytes.  It intentionally excludes model,
    input, allocator and kernel workspace, so callers treating it as a budget
    receive a fail-closed lower-bound check rather than a capacity promise.
    """

    _require(type(length) is int and length > 0, "UR-DMU dense sequence must be nonempty")
    return 13 * length * length * np.dtype(np.float32).itemsize


def _frame_records(
    *,
    run_id: str,
    source: _FrozenURDMUSource,
    target: _DenseFeatureSource,
    per_video_scores: Mapping[str, np.ndarray],
) -> tuple[list[PredictionRecord], dict[str, float]]:
    records: list[PredictionRecord] = []
    video_scores: dict[str, float] = {}
    for manifest in target.records:
        rows = target.rows_by_video[manifest.video_id]
        window_scores = per_video_scores[manifest.video_id]
        _require(window_scores.shape == (len(rows),) and np.isfinite(window_scores).all(), f"{manifest.video_id}: UR-DMU window scores are invalid")
        frame = aggregate_interval_scores(
            [int(row.frame_start) for row in rows],
            [int(row.frame_end) for row in rows],
            window_scores,
            num_frames=manifest.num_frames,
            reduction="mean",
        )
        video_scores[manifest.video_id] = float(np.max(frame.scores))
        start = 0
        boundaries = [index for index in range(1, manifest.num_frames) if frame.scores[index] != frame.scores[index - 1] or frame.contributors[index] != frame.contributors[index - 1]]
        for end in [*boundaries, manifest.num_frames]:
            score, contributors = float(frame.scores[start]), int(frame.contributors[start])
            records.append(PredictionRecord(
                run_id=run_id,
                video_id=manifest.video_id,
                clip_id=f"{manifest.video_id}:frame-{start:08d}-{end:08d}",
                clip_index=start,
                start_s=start / float(manifest.fps),
                end_s=end / float(manifest.fps),
                frame_start=start,
                frame_end=end,
                anomaly_score=score,
                predicted_label=bool(score >= 0.5),
                ground_truth=manifest.is_anomaly,
                encoder_fingerprint=target.encoder_fingerprint,
                metadata={
                    "task": "urdmu",
                    "score_level": "frame",
                    "ground_truth_scope": "video",
                    "source_clip_index": None,
                    "checkpoint": source.checkpoint.name,
                    "checkpoint_sha256": source.checkpoint_sha256,
                    "urdmu": {
                        "test_sequence": "complete_native_dense_windows_no_200_bin_aggregation",
                        "attention_workspace_lower_bound_bytes": _attention_workspace_lower_bound_bytes(len(rows)),
                    },
                    "dense_aggregation": {"reduction": "mean", "contributing_windows": contributors, "interval_source": "exact_equal_frame_run"},
                },
            ))
            start = end
    return records, video_scores


def _query_chunk_receipt(receipt: Mapping[str, Any], *, expected_chunk: int, length: int) -> None:
    _require(
        receipt.get("schema") == "urdmu.query-chunk-exact-attention/v1"
        and receipt.get("status") == "completed"
        and receipt.get("execution") == "eval_only_exact_all_key_query_chunk_attention"
        and receipt.get("approximation") is False
        and receipt.get("token_reduction") is False
        and receipt.get("complete_key_value_sequence") is True
        and receipt.get("source_identity_complete") is True
        and receipt.get("sequence_length") == length
        and receipt.get("query_chunk_size") == expected_chunk,
        "query-chunk inference receipt does not establish exact full-sequence UR-DMU execution",
    )
    state = receipt.get("state_dict")
    _require(isinstance(state, Mapping) and state.get("unchanged") is True, "query-chunk inference changed the UR-DMU state")


def _run_scores(
    source: _FrozenURDMUSource, target: _DenseFeatureSource, request: URDMUEvaluationRequest
) -> tuple[dict[str, np.ndarray], dict[str, Mapping[str, Any]]]:
    if torch is None:
        raise ImportError("PyTorch is required for UR-DMU prediction")
    workspace_by_video: dict[str, Mapping[str, Any]] = {}
    for manifest in target.records:
        length = len(target.rows_by_video[manifest.video_id])
        if request.query_chunk_size is None:
            workspace: Mapping[str, Any] = {
                "execution": "upstream_full_attention",
                "workspace_lower_bound_bytes": _attention_workspace_lower_bound_bytes(length),
            }
        else:
            from vadbench.paper.urdmu_inference import estimate_query_chunk_attention_workspace

            workspace = estimate_query_chunk_attention_workspace(
                batch_size=1,
                sequence_length=length,
                query_chunk_size=request.query_chunk_size,
            )
        workspace_by_video[manifest.video_id] = workspace
        if (
            request.max_attention_workspace_bytes is not None
            and workspace["workspace_lower_bound_bytes"] > request.max_attention_workspace_bytes
        ):
            raise MemoryError(
                f"{manifest.video_id}: original UR-DMU Test attention needs at least "
                f"{workspace['workspace_lower_bound_bytes']} bytes for N={length} dense windows; "
                "this exceeds max_attention_workspace_bytes. No 200-bin aggregation or "
                "windowing is applied; query_chunk_size is the only explicit exact opt-in."
            )
    device = torch.device(request.device)
    model, receipt = build_urdmu(target.representation.output_dim, request.upstream_dir, mode="Test")
    _require(receipt.get("upstream_commit") == UPSTREAM_COMMIT, "UR-DMU inference backend commit differs")
    from vadbench.engine.train import load_checkpoint

    loaded = load_checkpoint(source.checkpoint, model, map_location="cpu", strict=True, verify=True)
    _require(not loaded["missing_keys"] and not loaded["unexpected_keys"], "UR-DMU final checkpoint is not a strict model load")
    model.to(device)
    model.flag = "Test"
    model.eval()
    store = FeatureStore(target.root)
    output: dict[str, np.ndarray] = {}
    query_receipts: dict[str, Mapping[str, Any]] = {}
    with torch.inference_mode():
        for manifest in target.records:
            rows = target.rows_by_video[manifest.video_id]
            vectors = []
            for row in rows:
                bundle = store.load_bundle(row)
                vector = np.asarray(bundle.get("pooled"), dtype=np.float32)
                _require(vector.shape == (target.representation.output_dim,) and np.isfinite(vector).all(), f"{manifest.video_id}/{row.clip_id}: pooled feature is invalid")
                vectors.append(vector)
            dense = np.stack(vectors)
            inputs = torch.from_numpy(dense).unsqueeze(0).to(device)
            if request.query_chunk_size is None:
                result = model(inputs)
            else:
                from vadbench.paper.urdmu_inference import run_urdmu_query_chunked

                result, receipt = run_urdmu_query_chunked(
                    model,
                    inputs,
                    query_chunk_size=request.query_chunk_size,
                    source_receipt=source.metadata["backend"],
                    return_receipt=True,
                )
                _query_chunk_receipt(
                    receipt, expected_chunk=request.query_chunk_size, length=len(rows)
                )
                _require(
                    receipt.get("memory_estimate") == workspace_by_video[manifest.video_id],
                    "query-chunk runtime receipt memory estimate differs from preflight",
                )
                query_receipts[manifest.video_id] = receipt
            frame = result.get("frame") if isinstance(result, Mapping) else None
            _require(torch.is_tensor(frame) and tuple(frame.shape) == (1, len(rows)) and torch.isfinite(frame).all().item(), f"{manifest.video_id}: UR-DMU Test output is invalid")
            output[manifest.video_id] = frame.detach().cpu().numpy()[0].astype(np.float64, copy=False)
    return output, query_receipts


def run_urdmu_evaluation(request: URDMUEvaluationRequest) -> URDMUEvaluationResult:
    """Verify, predict and publish standard records without opening test truth."""
    if request.phase == "official_frame":
        # Fail before a test FeatureStore, checkpoint, old freeze, or audit is
        # opened. Those historical documents cannot authorize property-first
        # UR-DMU scoring.
        raise NotImplementedError(
            "official_frame is blocked pending a v3 property-evidence and "
            "same-mechanism TF/LoRA method-freeze contract; no official test "
            "source or score was read."
        )
    target = _load_dense_feature_source(request.feature_store, request.evaluation_manifest)
    source = _load_source(request)
    extra_hashes = _phase_gate(request, source, target)
    _verify_pairing(request, source, target)
    run = Path(request.output_root).expanduser().resolve() / (request.run_id or new_run_id("urdmu-evaluation"))
    if run.exists():
        raise FileExistsError(f"UR-DMU evaluation run already exists: {run}")
    run.mkdir(parents=True)
    inputs = {"trained_result": source.root / "result.json", "training_qa": source.root / "training_qa.json", "checkpoint": source.checkpoint, "evaluation_manifest": request.evaluation_manifest, "feature_resolved": target.root / "resolved.json", "feature_status": target.root / "status.json", "feature_index": target.root / "index.jsonl"}
    inputs.update({f"official_gate_{index}": path for index, path in enumerate(extra_hashes)})
    with record_stage(run, "urdmu_evaluation", config=request.__dict__, inputs=inputs, project_root=Path.cwd()):
        source.verify_unchanged()
        target.verify_unchanged()
        scores, query_receipts = _run_scores(source, target, request)
        source.verify_unchanged()
        target.verify_unchanged()
        prediction_records, video_scores = _frame_records(run_id=run.name, source=source, target=target, per_video_scores=scores)
        prediction_path = run / "predictions.jsonl"
        atomic_write_jsonl(prediction_path, (record.to_dict() for record in prediction_records))
        query_receipt_path = None
        if query_receipts:
            query_receipt_path = run / "query-chunk-receipts.json"
            atomic_write_json(query_receipt_path, query_receipts)
        metrics = None
        if request.phase == "development_video":
            labels = np.asarray([record.is_anomaly for record in target.records], dtype=np.uint8)
            values = np.asarray([video_scores[record.video_id] for record in target.records], dtype=np.float64)
            metrics = {"scope": "development_select_video_level_only", "metric": "video_roc_auc", "video_roc_auc": roc_auc_score(labels, values), "videos": len(target.records), "official_frame_scores_read": False}
            atomic_write_json(run / "development-video-metrics.json", metrics)
        max_windows = max(len(rows) for rows in target.rows_by_video.values())
        resolved = {"schema_version": 1, "status": "completed", "phase": request.phase, "mode": request.mode, "official_frame_scores_read": False, "checkpoint_sha256": source.checkpoint_sha256, "evaluation_encoder_fingerprint": target.encoder_fingerprint, "evaluation_representation_fingerprint": target.representation.fingerprint, "evaluation_sampling_fingerprint": target.sampling.fingerprint, "test_sequence": {"kind": "complete_native_dense_windows_no_200_bin_aggregation", "inference": "upstream_full_attention" if request.query_chunk_size is None else "explicit_exact_all_key_query_chunk_attention", "query_chunk_size": request.query_chunk_size, "max_dense_windows_per_video": max_windows, "max_attention_workspace_lower_bound_bytes": _attention_workspace_lower_bound_bytes(max_windows), "configured_max_attention_workspace_bytes": request.max_attention_workspace_bytes}, "artifacts": {"predictions": {"path": str(prediction_path), "sha256": sha256_file(prediction_path)}, "query_chunk_receipts": None if query_receipt_path is None else {"path": str(query_receipt_path), "sha256": sha256_file(query_receipt_path)}, "development_video_metrics": None if metrics is None else {"path": str(run / "development-video-metrics.json"), "sha256": sha256_file(run / "development-video-metrics.json")}}, "official_gate_sha256": extra_hashes}
        atomic_write_json(run / "resolved.json", resolved)
        atomic_write_json(run / "result.json", {"schema_version": 1, "status": "completed", "phase": request.phase, "protocol": "urdmu-prediction-only/v1", "official_frame_scores_read": False, "resolved": {"path": str(run / "resolved.json"), "sha256": sha256_file(run / "resolved.json")}})
    return URDMUEvaluationResult(str(run), str(prediction_path), True, request.phase, metrics)


__all__ = ["URDMUEvaluationRequest", "URDMUEvaluationResult", "run_urdmu_evaluation"]
