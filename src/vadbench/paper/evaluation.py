"""Strict UCF official-test evaluation from already trained frozen detectors.

This is deliberately separate from :mod:`vadbench.paper.controller`: it never
trains a detector or writes into a source run.  It opens two immutable source
heads (the method head and, optionally, the dense head), extracts one isolated
test FeatureStore for the method representation, and uses the existing permit
and official evaluator paths for prediction and scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vadbench.artifacts import new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.manifest import DatasetSplit, VideoManifestRecord, load_manifest_jsonl
from vadbench.data.video import build_clip_batch
from vadbench.engine.coverage import validate_frame_coverage
from vadbench.features import FeatureStore, atomic_write_json
from vadbench.paper.compatibility import (
    CompatibilityDeclaration,
    RepresentationIdentity,
    SamplingIdentity,
    TrainingIdentity,
    feature_cache_key,
)
from vadbench.paper.detection import DetectionConfig, evaluate_detector, predict_detector
from vadbench.paper.extraction import (
    PooledExtractionSpec,
    extract_pooled_features,
    make_sampling_identity,
    representation_from_verified_encoder,
)

_FREEZE_STATUS = "algorithm_and_budget_frozen_before_official_model_scores"
_UCF_TEST_MANIFEST_SHA256 = "a2a66ae55db22f1a6ece0da6b87e77b8b92a0031e1aafc4c10d32aa956c7e17c"
_UCF_AUDIT_SHA256 = "d7568104f4607059fa02b1303ea5461123c15eb4b4a78e0aafd00351ee2d9e4a"
_FROZEN_ROLE_LOCK_SHA256 = "53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59"
_DEFAULT_ROLE_LOCK_PATH = Path(
    "outputs/icassp2027/assets/gate-calibration-20260918T083000Z/frozen-partition-lock.json"
)
_FROZEN_HEAD_DATA_CONTRACT_SHA256 = (
    "5508d900aa97c7bb1ef447dd3b03bfdae2808b56aafe00ac8a250a1aa6d7527b"
)
_DEFAULT_HEAD_DATA_CONTRACT_PATH = Path("projects/icassp2027/decisions/head-data-contract-v1.json")
_DEFAULT_HEAD_SOURCE_MANIFEST_ROOT = Path(
    "outputs/icassp2027/assets/full-ucf-head-data-contract-20260918"
)
_HEAD_BUDGET = {
    "name": "TopKMIL",
    "k": 3,
    "dropout": 0.0,
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
}


def _json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"{name} is missing: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain an object: {path}")
    return dict(value)


def _digest(path: Path, *, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return sha256_file(path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _checkpoint_metadata_bytes(path: Path) -> dict[str, Any]:
    """Read the metadata embedded in the SHA-bound checkpoint, on CPU only."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - a checkpoint cannot be verified without torch.
        raise RuntimeError(
            "PyTorch is required to verify frozen detector checkpoint metadata"
        ) from exc
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot read source checkpoint metadata: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("source checkpoint payload must be an object")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("source checkpoint lacks a non-empty model_state_dict")
    if (
        payload.get("epoch") != 20
        or isinstance(payload.get("step"), bool)
        or not isinstance(payload.get("step"), int)
        or payload["step"] <= 0
    ):
        raise ValueError(
            "source checkpoint does not contain a completed frozen 20-epoch head state"
        )
    return _mapping(payload.get("metadata"), "source checkpoint embedded metadata")


@dataclass(frozen=True)
class FrozenEvaluationRequest:
    """Inputs for one UCF official-test attempt.

    ``method_source_run`` supplies the frozen refit-head checkpoint.  When a
    dense source is supplied, the same extracted method test features are also
    scored by the dense head through the explicit direct-insert permit.
    """

    encoder: str
    device: str
    dataset_root: str
    test_manifest: str
    audit_report: str
    method_source_run: str
    output_root: str
    dense_source_run: str | None = None
    calibration_run: str | None = None
    project: str = "projects/icassp2027/profile.yaml"
    freeze_path: str = "projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json"
    role_lock_path: str = str(_DEFAULT_ROLE_LOCK_PATH)
    head_data_contract_path: str = str(_DEFAULT_HEAD_DATA_CONTRACT_PATH)
    source_manifest_root: str = str(_DEFAULT_HEAD_SOURCE_MANIFEST_ROOT)
    processor_tensor_type: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.encoder:
            raise ValueError("encoder must be non-empty")
        if self.processor_tensor_type not in {None, "pt", "np"}:
            raise ValueError("processor_tensor_type must be pt, np, or null")
        if self.processor_tensor_type is not None and self.encoder not in {
            "timesformer",
            "videomae",
        }:
            raise ValueError(
                "processor_tensor_type is only supported by the active Transformers adapters"
            )


@dataclass(frozen=True)
class FrozenEvaluationResult:
    run_dir: str
    completed: bool
    primary_predictions: str
    primary_metrics: str
    secondary_predictions: str | None
    secondary_metrics: str | None


@dataclass(frozen=True)
class FrozenDetectorSource:
    root: Path
    checkpoint: Path
    checkpoint_metadata: Mapping[str, Any]
    representation: RepresentationIdentity
    sampling: SamplingIdentity
    encoder_fingerprint: str
    training_identity: TrainingIdentity
    reducer: Mapping[str, Any]
    source_hashes: Mapping[str, str]
    train_manifest: Path
    train_feature_store: Path
    validation_manifest: Path | None
    validation_feature_store: Path | None
    stage_config: Mapping[str, Any]
    role_lock_path: Path
    role_lock_sha256: str
    head_data_contract_path: Path
    head_data_contract_sha256: str
    source_manifest_root: Path
    source_manifest_sha256: Mapping[str, str]

    def verify_unchanged(self) -> None:
        for name, expected in self.source_hashes.items():
            path = self.root / name
            if _digest(path, name=f"source artifact {name}") != expected:
                raise ValueError(f"source artifact changed after verification: {path}")
        if _digest(self.role_lock_path, name="frozen training role lock") != self.role_lock_sha256:
            raise ValueError(
                f"frozen training role lock changed after verification: {self.role_lock_path}"
            )
        if (
            _digest(self.head_data_contract_path, name="frozen head data contract")
            != self.head_data_contract_sha256
        ):
            raise ValueError(
                f"frozen head data contract changed after verification: {self.head_data_contract_path}"
            )
        for name, expected in self.source_manifest_sha256.items():
            if _digest(Path(name), name="frozen head source manifest") != expected:
                raise ValueError(f"frozen head source manifest changed after verification: {name}")

    def detection_config(
        self,
        *,
        mode: str,
        evaluation_representation: RepresentationIdentity,
        evaluation_sampling: SamplingIdentity,
        evaluation_encoder_fingerprint: str,
    ) -> DetectionConfig:
        """Build a permit-bound config without changing any source artifact."""

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
            epochs=int(self.stage_config["epochs"]),
            batch_size=int(self.stage_config["batch_size"]),
            learning_rate=float(self.stage_config["learning_rate"]),
            weight_decay=float(self.checkpoint_metadata["config"]["weight_decay"]),
            seed=self.training_identity.seed,
            expected_training_clips=32,
            expected_evaluation_clips=None,
        )


def _load_freeze(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path).expanduser().resolve()
    freeze = _json(path, name="method freeze")
    if freeze.get("status") != _FREEZE_STATUS:
        raise ValueError("method freeze is not approved for official model evaluation")
    if freeze.get("annotation_policy") != "W":
        raise ValueError("official evaluator requires the frozen weak-supervision policy")
    methods = _mapping(freeze.get("methods"), "method freeze.methods")
    if set(methods) != {"dense", "same_budget_control", "training_free", "trainable"}:
        raise ValueError("method freeze does not describe the complete frozen reducer set")
    ucf = _mapping(freeze.get("ucf_evaluation"), "method freeze.ucf_evaluation")
    if (
        ucf.get("sealed_test_manifest_sha256") != _UCF_TEST_MANIFEST_SHA256
        or ucf.get("sealed_audit_sha256") != _UCF_AUDIT_SHA256
    ):
        raise ValueError("method freeze sealed UCF identities differ from this official evaluator")
    return freeze, _digest(path, name="method freeze")


def _stage_config(root: Path) -> tuple[dict[str, Any], Path]:
    stages = root / "provenance" / "stages"
    candidates: list[tuple[dict[str, Any], Path]] = []
    for path in sorted(stages.glob("*.json")):
        document = _json(path, name="source controller stage")
        if document.get("stage") == "paper_detection" and document.get("status") == "completed":
            candidates.append(
                (_mapping(document.get("config"), "source controller stage.config"), path)
            )
    if len(candidates) != 1:
        raise ValueError(
            "source run must contain exactly one completed paper_detection provenance stage"
        )
    return candidates[0]


def _head_training_identity(
    *,
    source_train_manifest: Path,
    stage: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> TrainingIdentity:
    config = _mapping(checkpoint_metadata.get("config"), "checkpoint metadata.config")
    if {"head", "batch_size", "epochs", "learning_rate", "weight_decay", "seed"} - set(config):
        raise ValueError("checkpoint lacks the complete frozen detector budget")
    freeze_head = _mapping(freeze.get("detection_head"), "method freeze.detection_head")
    expected = {
        **_HEAD_BUDGET,
        "epochs": freeze_head.get("epochs"),
        "batch_size": freeze_head.get("batch_size"),
        "learning_rate": freeze_head.get("learning_rate"),
        "weight_decay": freeze_head.get("weight_decay"),
    }
    observed = {
        "name": "TopKMIL" if config.get("head") == "topk" else config.get("head"),
        "k": _mapping(config.get("head_kwargs"), "checkpoint metadata.config.head_kwargs").get("k"),
        "dropout": _mapping(
            config.get("head_kwargs"), "checkpoint metadata.config.head_kwargs"
        ).get("dropout", 0.0),
        "epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
    }
    if observed != expected:
        raise ValueError("source checkpoint does not use the frozen detector head/training budget")
    if (
        config.get("task") != "weak_mil"
        or config.get("feature_level") != "clip"
        or config.get("expected_clips") != 32
    ):
        raise ValueError("source checkpoint does not use the frozen weak-MIL clip/32-bag protocol")
    if _mapping(config.get("task_kwargs"), "checkpoint metadata.config.task_kwargs") != {
        "ranking_weight": 0.0
    }:
        raise ValueError(
            "source checkpoint task loss differs from the frozen BCE-only head protocol"
        )
    allowed_seeds = {
        freeze_head.get("primary_seed"),
        *freeze_head.get("key_configuration_seeds", ()),
    }
    if config.get("seed") not in allowed_seeds or isinstance(config.get("seed"), bool):
        raise ValueError("source checkpoint seed is outside the frozen key-configuration seeds")
    if any(
        stage.get(key) != expected[key] for key in ("epochs", "batch_size", "learning_rate")
    ) or stage.get("seed") != config.get("seed"):
        raise ValueError("source controller stage does not match checkpoint training budget")
    records = load_manifest_jsonl(source_train_manifest)
    from vadbench.data.audit import compute_manifest_sha256

    return TrainingIdentity(
        head={"kind": "topk_mil", "k": 3},
        fit_split_digest="sha256:" + compute_manifest_sha256(records),
        seed=int(config["seed"]),
        # Controller v1 did not include weight decay in this declaration.
        optimization={"epochs": 20, "lr": 0.001},
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _frozen_source_fields(
    records: tuple[VideoManifestRecord, ...],
) -> dict[str, tuple[str, bool, int | None, float | None]]:
    values = {
        record.video_id: (record.path, record.is_anomaly, record.num_frames, record.fps)
        for record in records
    }
    if len(values) != len(records):
        raise ValueError("frozen head source manifest contains duplicate video IDs")
    return values


def _verify_frozen_training_roles(
    *,
    train_manifest: Path,
    validation_manifest: Path | None,
    role_lock_path: str | Path | None,
    freeze: Mapping[str, Any],
    head_data_contract_path: str | Path | None,
    source_manifest_root: str | Path | None,
) -> tuple[Path, str, Path, str, dict[str, str]]:
    """Bind source IDs to the independent full-training partition lock."""

    lock_path = (
        (_DEFAULT_ROLE_LOCK_PATH if role_lock_path is None else Path(role_lock_path))
        .expanduser()
        .resolve()
    )
    lock_sha = _digest(lock_path, name="frozen training role lock")
    if lock_sha != _FROZEN_ROLE_LOCK_SHA256:
        raise ValueError(
            "frozen training role lock SHA-256 differs from the authorized partition lock"
        )
    lock = _json(lock_path, name="frozen training role lock")
    if (
        lock.get("schema_version") != 1
        or lock.get("basis") != "complete_official_train_source_groups"
        or lock.get("seed") != 20260918
    ):
        raise ValueError(
            "frozen training role lock schema differs from the authorized partition contract"
        )
    partitions = _mapping(lock.get("partitions"), "frozen training role lock.partitions")
    if any(
        not isinstance(video_id, str) or role not in {"fit", "confirm", "select"}
        for video_id, role in partitions.items()
    ):
        raise ValueError("frozen training role lock partitions are invalid")
    expected_fit = {video_id for video_id, role in partitions.items() if role == "fit"}
    expected_select = {video_id for video_id, role in partitions.items() if role == "select"}
    contract_path = (
        (
            _DEFAULT_HEAD_DATA_CONTRACT_PATH
            if head_data_contract_path is None
            else Path(head_data_contract_path)
        )
        .expanduser()
        .resolve()
    )
    contract_sha = _digest(contract_path, name="frozen head data contract")
    if contract_sha != _FROZEN_HEAD_DATA_CONTRACT_SHA256:
        raise ValueError("frozen head data contract SHA-256 differs from the authorized contract")
    contract = _json(contract_path, name="frozen head data contract")
    if (
        contract.get("schema_version") != 1
        or contract.get("status") != "fixed_before_official_model_scores"
        or contract.get("dataset") != "ucf-crime"
        or contract.get("official_training_videos") != 1610
        or contract.get("role_lock_sha256") != _FROZEN_ROLE_LOCK_SHA256
        or contract.get("role_lock_seed") != 20260918
        or contract.get("method_freeze_canonical_sha256") != _canonical_sha256(freeze)
    ):
        raise ValueError(
            "frozen head data contract differs from the authorized method/role binding"
        )
    roles = _mapping(contract.get("roles"), "frozen head data contract.roles")
    root = (
        (
            _DEFAULT_HEAD_SOURCE_MANIFEST_ROOT
            if source_manifest_root is None
            else Path(source_manifest_root)
        )
        .expanduser()
        .resolve()
    )
    source_hashes: dict[str, str] = {}
    controller = {
        "fit": (train_manifest, expected_fit),
        "select": (validation_manifest, expected_select),
    }
    for role, (controller_manifest, expected_ids) in controller.items():
        entry = _mapping(roles.get(role), f"frozen head data contract.roles.{role}")
        if (
            entry.get("videos") != len(expected_ids)
            or entry.get("source_split") != "train"
            or entry.get("controller_split") != ("train" if role == "fit" else "val")
        ):
            raise ValueError(f"frozen head data contract {role} role is invalid")
        source_name = entry.get("source_manifest")
        if (
            not isinstance(source_name, str)
            or not source_name
            or Path(source_name).is_absolute()
            or ".." in Path(source_name).parts
        ):
            raise ValueError(f"frozen head data contract {role} source manifest is invalid")
        # The server contract stores the stable archive-relative name
        # ``manifests/<role>.jsonl`` while the local audited mirror stores the
        # same sealed files directly in its supplied root.
        source_path = root / Path(source_name).name
        source_sha = _digest(source_path, name=f"frozen {role} source manifest")
        if source_sha != entry.get("source_manifest_sha256"):
            raise ValueError(f"frozen {role} source manifest SHA-256 differs from the contract")
        source_hashes[str(source_path)] = source_sha
        source_records = load_manifest_jsonl(source_path)
        if set(_frozen_source_fields(source_records)) != expected_ids or len(source_records) != len(
            expected_ids
        ):
            raise ValueError(
                f"frozen {role} source manifest IDs differ from the independent role lock"
            )
        if any(record.split != DatasetSplit.TRAIN for record in source_records):
            raise ValueError(f"frozen {role} source manifest must retain original train split")
        if controller_manifest is None:
            raise ValueError(f"source run lacks the frozen complete {role} manifest")
        controller_records = load_manifest_jsonl(controller_manifest)
        controller_split = DatasetSplit.TRAIN if role == "fit" else DatasetSplit.VAL
        if any(record.split != controller_split for record in controller_records):
            raise ValueError(f"source {role} manifest does not use its frozen controller split")
        if _frozen_source_fields(controller_records) != _frozen_source_fields(source_records):
            raise ValueError(
                f"source {role} manifest differs from the frozen source video identity fields"
            )
    actual_fit = {record.video_id for record in load_manifest_jsonl(train_manifest)}
    if actual_fit != expected_fit:
        raise ValueError("source train manifest IDs differ from the frozen complete fit role set")
    if (
        validation_manifest is None
        or {record.video_id for record in load_manifest_jsonl(validation_manifest)}
        != expected_select
    ):
        raise ValueError(
            "source validation manifest IDs differ from the frozen complete select role set"
        )
    return lock_path, lock_sha, contract_path, contract_sha, source_hashes


def load_frozen_detector_source(
    path: str | Path,
    *,
    freeze: Mapping[str, Any],
    expected_encoder: str,
    role_lock_path: str | Path | None = None,
    head_data_contract_path: str | Path | None = None,
    source_manifest_root: str | Path | None = None,
) -> FrozenDetectorSource:
    root = Path(path).expanduser().resolve()
    stage, stage_path = _stage_config(root)
    if stage.get("encoder") != expected_encoder:
        raise ValueError(
            "source controller encoder differs from requested official evaluator encoder"
        )
    if (
        expected_encoder in {"timesformer", "videomae"}
        and stage.get("processor_tensor_type") != "np"
    ):
        raise ValueError(
            "frozen Transformer source heads require the verified np processor runtime"
        )
    train_feature = root / "features" / "train"
    resolved_path = train_feature / "resolved.json"
    status_path = train_feature / "status.json"
    train_manifest = root / "frozen" / "train.jsonl"
    checkpoint = root / "head" / "checkpoints" / "final.pt"
    sidecar = checkpoint.with_suffix(".pt.json")
    qa_path = root / "head" / "training_qa.json"
    result_path = root / "result.json"
    train_index = train_feature / "index.jsonl"
    validation_manifest = root / "frozen" / "val.jsonl"
    validation_feature = root / "features" / "validation"
    if validation_manifest.exists() != validation_feature.exists():
        raise ValueError(
            "source validation manifest and FeatureStore must either both exist or both be absent"
        )
    (
        frozen_role_lock,
        frozen_role_lock_sha,
        head_contract,
        head_contract_sha,
        source_manifest_hashes,
    ) = _verify_frozen_training_roles(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest if validation_manifest.exists() else None,
        role_lock_path=role_lock_path,
        freeze=freeze,
        head_data_contract_path=head_data_contract_path,
        source_manifest_root=source_manifest_root,
    )
    validation_required: tuple[Path, ...] = ()
    if validation_manifest.exists():
        validation_required = (
            validation_manifest,
            validation_feature / "resolved.json",
            validation_feature / "status.json",
            validation_feature / "index.jsonl",
        )
    required = (
        resolved_path,
        status_path,
        train_manifest,
        train_index,
        checkpoint,
        sidecar,
        qa_path,
        result_path,
        stage_path,
        *validation_required,
    )
    hashes = {
        str(item.relative_to(root)).replace("\\", "/"): _digest(item, name="source artifact")
        for item in required
    }
    status = _json(status_path, name="source training feature status")
    if status.get("completed") is not True or status.get("feature_root") != str(train_feature):
        raise ValueError("source training FeatureStore was not completed and published")
    resolved = _json(resolved_path, name="source training feature resolution")
    spec = _mapping(resolved.get("spec"), "source feature resolved.spec")
    representation = RepresentationIdentity.from_mapping(
        _mapping(spec.get("representation"), "source representation")
    )
    sampling = SamplingIdentity.from_mapping(
        _mapping(spec.get("sampling"), "source training sampling")
    )
    if (
        spec.get("sampling_kind") != "uniform_full"
        or sampling.regime != "train_32"
        or sampling.frame_selection.get("num_segments") != 32
    ):
        raise ValueError("source head must have frozen train_32 pooled features")
    if representation.backbone.runtime_id != expected_encoder:
        raise ValueError("source training representation runtime differs from requested encoder")
    expected_paper_identity = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    if resolved.get("paper_identity") != expected_paper_identity:
        raise ValueError("source training feature resolution paper identity differs from its spec")
    encoder_fingerprint = resolved.get("encoder_fingerprint")
    if not isinstance(encoder_fingerprint, str):
        raise ValueError("source training feature resolution lacks encoder_fingerprint")
    store = FeatureStore(train_feature)
    rows_by_video: dict[str, list[int]] = {
        item.video_id: [] for item in load_manifest_jsonl(train_manifest)
    }
    expected_training_identity = {
        "representation_fingerprint": representation.fingerprint,
        "sampling_fingerprint": sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(representation, sampling),
    }
    for row in store.iter_records():
        if row.video_id not in rows_by_video:
            raise ValueError(
                "source training FeatureStore contains a video outside its frozen train manifest"
            )
        if (
            row.encoder_fingerprint != encoder_fingerprint
            or row.metadata.get("paper_identity") != expected_training_identity
        ):
            raise ValueError(
                "source training FeatureStore row identity differs from its resolved source identity"
            )
        pooled = row.arrays.get("pooled")
        if pooled is None or pooled.shape != (representation.output_dim,):
            raise ValueError(
                "source training FeatureStore row lacks the declared pooled feature dimension"
            )
        rows_by_video[row.video_id].append(row.clip_index)
    for video_id, rows in rows_by_video.items():
        if len(rows) != 32 or set(rows) != set(range(32)):
            raise ValueError(
                f"{video_id}: source training FeatureStore must contain exactly the frozen 32 clips"
            )
    sidecar_doc = _json(sidecar, name="source checkpoint sidecar")
    if sidecar_doc.get("sha256") != _digest(checkpoint, name="source checkpoint"):
        raise ValueError("source checkpoint sidecar SHA-256 does not match checkpoint bytes")
    metadata = _mapping(sidecar_doc.get("metadata"), "source checkpoint metadata")
    if _checkpoint_metadata_bytes(checkpoint) != metadata:
        raise ValueError("source checkpoint embedded metadata differs from its checksum sidecar")
    qa = _json(qa_path, name="source training QA")
    checkpoint_qa = _mapping(qa.get("checkpoint"), "source training QA checkpoint")
    metadata_qa = _mapping(metadata.get("training_qa"), "source checkpoint training_qa")
    actual_checkpoint_sha = _digest(checkpoint, name="source checkpoint")
    if qa.get("status") != "passed" or metadata_qa.get("status") != "passed":
        raise ValueError("source detector training QA did not pass")
    if checkpoint_qa.get("sha256") != actual_checkpoint_sha or checkpoint_qa.get("path") != str(
        checkpoint
    ):
        raise ValueError("source training QA is not bound to the final checkpoint bytes")
    if (
        checkpoint_qa.get("epoch") != 20
        or isinstance(checkpoint_qa.get("step"), bool)
        or not isinstance(checkpoint_qa.get("step"), int)
        or checkpoint_qa["step"] <= 0
    ):
        raise ValueError("source training QA is not bound to a completed 20-epoch checkpoint")
    for name in ("nonzero_gradient_steps", "changed_parameter_count"):
        if isinstance(qa.get(name), bool) or not isinstance(qa.get(name), int) or qa[name] <= 0:
            raise ValueError(f"source training QA lacks a positive {name}")
        if metadata_qa.get(name) != qa[name]:
            raise ValueError(f"source checkpoint QA {name} differs from the training QA receipt")
    if (
        isinstance(qa.get("parameter_delta_l2"), bool)
        or not isinstance(qa.get("parameter_delta_l2"), (int, float))
        or not math.isfinite(float(qa["parameter_delta_l2"]))
        or qa["parameter_delta_l2"] <= 0
    ):
        raise ValueError("source training QA lacks a positive parameter_delta_l2")
    if metadata_qa.get("parameter_delta_l2") != qa["parameter_delta_l2"]:
        raise ValueError(
            "source checkpoint QA parameter_delta_l2 differs from the training QA receipt"
        )
    parity = _mapping(qa.get("reload_parity"), "source training QA reload_parity")
    outputs = _mapping(parity.get("outputs"), "source training QA reload_parity.outputs")
    if not outputs or any(
        _mapping(item, "source training QA reload output").get("exact_equal") is not True
        or _mapping(item, "source training QA reload output").get("max_abs_difference") != 0.0
        for item in outputs.values()
    ):
        raise ValueError("source training QA reload parity is incomplete or non-exact")
    if metadata_qa.get("reload_parity_exact") is not True:
        raise ValueError("source checkpoint metadata does not attest exact QA reload parity")
    final_load = _mapping(
        qa.get("final_checkpoint_load"), "source training QA final_checkpoint_load"
    )
    if (
        set(final_load) != {"missing_keys", "unexpected_keys"}
        or final_load["missing_keys"] != []
        or final_load["unexpected_keys"] != []
    ):
        raise ValueError("source training QA final checkpoint reload receipt is incomplete")
    if (
        metadata.get("status") != "completed"
        or metadata.get("encoder_fingerprint") != encoder_fingerprint
    ):
        raise ValueError("source checkpoint is not the completed source FeatureStore checkpoint")
    result = _json(result_path, name="source controller result")
    if (
        result.get("status") != "completed"
        or Path(str(result.get("checkpoint_path", ""))).resolve() != checkpoint
    ):
        raise ValueError("source controller result does not bind the final detector checkpoint")
    training = _head_training_identity(
        source_train_manifest=train_manifest,
        stage=stage,
        checkpoint_metadata=metadata,
        freeze=freeze,
    )
    paper_detector = _mapping(metadata.get("paper_detector"), "source checkpoint paper_detector")
    if (
        paper_detector.get("training_representation_fingerprint") != representation.fingerprint
        or paper_detector.get("training_sampling_fingerprint") != sampling.fingerprint
        or paper_detector.get("training_feature_cache_fingerprint")
        != feature_cache_key(representation, sampling)
        or paper_detector.get("training_identity_fingerprint") != training.fingerprint
    ):
        raise ValueError(
            "source checkpoint paper detector identity does not match source artifacts"
        )
    reducer = dict(representation.reducer)
    allowed = {"identity", "global_uniform", "paired_random", "pair_linear"}
    if reducer.get("name") not in allowed:
        raise ValueError("source representation reducer is outside the frozen method set")
    method_by_reducer = {
        "identity": "dense",
        "global_uniform": "same_budget_control",
        "paired_random": "training_free",
        "pair_linear": "trainable",
    }
    frozen_method = _mapping(
        _mapping(freeze.get("methods"), "method freeze.methods").get(
            method_by_reducer[reducer["name"]]
        ),
        "method freeze entry",
    )
    if frozen_method.get("reducer") != reducer["name"]:
        raise ValueError("source reducer does not match the frozen method definition")
    if reducer["name"] == "paired_random" and reducer.get("seed") != frozen_method.get("seed"):
        raise ValueError("source paired_random reducer seed differs from frozen method")
    if reducer["name"] == "pair_linear":
        calibration = _mapping(frozen_method.get("calibration"), "frozen pair_linear calibration")
        if reducer.get("calibration_manifest_digest") != calibration.get("manifest_sha256"):
            raise ValueError(
                "source pair_linear calibration manifest differs from the frozen method"
            )
    if validation_feature.exists():
        validation_status = _json(
            validation_feature / "status.json", name="source validation feature status"
        )
        if validation_status.get("completed") is not True or validation_status.get(
            "feature_root"
        ) != str(validation_feature):
            raise ValueError("source validation FeatureStore was not completed and published")
        validation_resolved = _json(
            validation_feature / "resolved.json", name="source validation feature resolution"
        )
        validation_spec = _mapping(
            validation_resolved.get("spec"), "source validation feature resolved.spec"
        )
        validation_representation = RepresentationIdentity.from_mapping(
            _mapping(validation_spec.get("representation"), "source validation representation")
        )
        validation_sampling = SamplingIdentity.from_mapping(
            _mapping(validation_spec.get("sampling"), "source validation sampling")
        )
        if (
            validation_representation != representation
            or validation_spec.get("sampling_kind") != "uniform_full"
        ):
            raise ValueError(
                "source validation FeatureStore representation differs from the frozen training representation"
            )
        train_policy, validation_policy = sampling.to_dict(), validation_sampling.to_dict()
        train_policy.pop("source_digest")
        validation_policy.pop("source_digest")
        if train_policy != validation_policy:
            raise ValueError(
                "source validation FeatureStore sampling differs from the frozen train policy"
            )
        validation_identity = {
            "representation_fingerprint": representation.fingerprint,
            "sampling_fingerprint": validation_sampling.fingerprint,
            "feature_cache_fingerprint": feature_cache_key(representation, validation_sampling),
        }
        validation_rows: dict[str, list[int]] = {
            item.video_id: [] for item in load_manifest_jsonl(validation_manifest)
        }
        for row in FeatureStore(validation_feature).iter_records():
            if (
                row.video_id not in validation_rows
                or row.encoder_fingerprint != validation_resolved.get("encoder_fingerprint")
                or row.metadata.get("paper_identity") != validation_identity
            ):
                raise ValueError(
                    "source validation FeatureStore row identity differs from its resolved source identity"
                )
            validation_rows[row.video_id].append(row.clip_index)
        if any(len(rows) != 32 or set(rows) != set(range(32)) for rows in validation_rows.values()):
            raise ValueError(
                "source validation FeatureStore must contain exactly the frozen 32 clips per video"
            )
    return FrozenDetectorSource(
        root,
        checkpoint,
        metadata,
        representation,
        sampling,
        encoder_fingerprint,
        training,
        reducer,
        hashes,
        train_manifest,
        train_feature,
        validation_manifest if validation_manifest.exists() else None,
        validation_feature if validation_feature.exists() else None,
        stage,
        frozen_role_lock,
        frozen_role_lock_sha,
        head_contract,
        head_contract_sha,
        (
            _DEFAULT_HEAD_SOURCE_MANIFEST_ROOT
            if source_manifest_root is None
            else Path(source_manifest_root)
        )
        .expanduser()
        .resolve(),
        source_manifest_hashes,
    )


def _clip_frames(adapter: Any, identity: Mapping[str, Any]) -> int:
    constructor = _mapping(identity.get("constructor"), "verified encoder constructor")
    value = constructor.get(
        "num_frames",
        constructor.get("clip_frames", getattr(adapter.capabilities, "fixed_num_frames", None)),
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("verified fixed-clip encoder has no positive clip frame count")
    return value


def _make_adapter(request: FrozenEvaluationRequest) -> tuple[Any, Mapping[str, Any]]:
    from vadbench.orchestration import encoder_identity
    from vadbench.paper.profile import load_project
    from vadbench.registry import ENCODER_REGISTRY

    project = load_project(request.project)
    selected = project.encoder(request.encoder)
    definition = dict(selected["definition"])
    constructor = dict(definition["constructor"])
    constructor["device"] = request.device
    if request.processor_tensor_type is not None:
        constructor["processor_tensor_type"] = request.processor_tensor_type
    definition["constructor"] = constructor
    definition["identity"] = encoder_identity(definition, project_root=project.root)
    return ENCODER_REGISTRY.create(request.encoder, **constructor), definition


def _coverage(
    manifest: tuple[VideoManifestRecord, ...], feature_root: Path, fingerprint: str
) -> dict[str, Any]:
    store = FeatureStore(feature_root)
    by_video: dict[str, list[tuple[int, int | None, int | None]]] = {
        item.video_id: [] for item in manifest
    }
    for row in store.iter_records():
        if row.video_id in by_video and row.encoder_fingerprint == fingerprint:
            by_video[row.video_id].append((row.clip_index, row.frame_start, row.frame_end))
    receipt = []
    for item in manifest:
        rows = sorted(by_video[item.video_id])
        if not rows:
            raise ValueError("official dense sampling has no FeatureStore rows for a test video")
        result = validate_frame_coverage(
            video_id=item.video_id,
            clip_indices=__import__("numpy").asarray([row[0] for row in rows]),
            frame_starts=__import__("numpy").asarray([row[1] for row in rows]),
            frame_ends=__import__("numpy").asarray([row[2] for row in rows]),
            num_frames=item.num_frames,
            fps=item.fps,
            require_fps=True,
            require_complete=False,
        )
        if result["gap_frames"]:
            raise ValueError(
                f"{item.video_id}: official dense sampling has uncovered decoded frames"
            )
        receipt.append(result)
    return {
        "videos": len(receipt),
        "input_union_complete": all(item["gap_frames"] == 0 for item in receipt),
        "overlap_is_expected_before_prediction_aggregation": any(
            item["overlap_frames"] > 0 for item in receipt
        ),
        "per_video": receipt,
    }


def _source_receipt(source: FrozenDetectorSource) -> dict[str, Any]:
    """Stable provenance fields for downstream quality aggregation."""

    return {
        "run_dir": str(source.root),
        "checkpoint_path": str(source.checkpoint),
        "checkpoint_sha256": source.source_hashes["head/checkpoints/final.pt"],
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
        "training_encoder_fingerprint": source.encoder_fingerprint,
        "training_identity_fingerprint": source.training_identity.fingerprint,
        "head_seed": source.training_identity.seed,
        "head_budget": {
            "head": source.checkpoint_metadata["config"]["head"],
            "head_kwargs": source.checkpoint_metadata["config"]["head_kwargs"],
            "epochs": source.checkpoint_metadata["config"]["epochs"],
            "batch_size": source.checkpoint_metadata["config"]["batch_size"],
            "learning_rate": source.checkpoint_metadata["config"]["learning_rate"],
            "weight_decay": source.checkpoint_metadata["config"]["weight_decay"],
        },
    }


def run_frozen_ucf_evaluation(
    request: FrozenEvaluationRequest,
    *,
    adapter_factory: Callable[[FrozenEvaluationRequest], tuple[Any, Mapping[str, Any]]]
    | None = None,
    video_backend: Any | None = None,
) -> FrozenEvaluationResult:
    """Extract and score a frozen UCF test run without head training."""

    freeze, freeze_sha = _load_freeze(request.freeze_path)
    test_manifest = Path(request.test_manifest).expanduser().resolve()
    audit_report = Path(request.audit_report).expanduser().resolve()
    if _digest(test_manifest, name="sealed UCF test manifest") != _UCF_TEST_MANIFEST_SHA256:
        raise ValueError(
            "sealed UCF test manifest SHA-256 differs from the frozen evaluation identity"
        )
    if _digest(audit_report, name="sealed UCF audit") != _UCF_AUDIT_SHA256:
        raise ValueError("sealed UCF audit SHA-256 differs from the frozen evaluation identity")
    manifest = load_manifest_jsonl(test_manifest)
    if not manifest or any(item.split != DatasetSplit.TEST for item in manifest):
        raise ValueError("official UCF evaluation manifest must contain only test records")
    source_kwargs = {
        "role_lock_path": request.role_lock_path,
        "head_data_contract_path": request.head_data_contract_path,
        "source_manifest_root": request.source_manifest_root,
    }
    method = load_frozen_detector_source(
        request.method_source_run, freeze=freeze, expected_encoder=request.encoder, **source_kwargs
    )
    dense = (
        None
        if request.dense_source_run is None
        else load_frozen_detector_source(
            request.dense_source_run,
            freeze=freeze,
            expected_encoder=request.encoder,
            **source_kwargs,
        )
    )
    if dense is not None and dense.reducer.get("name") != "identity":
        raise ValueError("secondary direct_insert source must be the frozen dense identity head")
    if method.reducer.get("name") == "pair_linear" and request.calibration_run is None:
        raise ValueError(
            "pair_linear official evaluation requires its completed frozen calibration_run"
        )
    if method.reducer.get("name") != "pair_linear" and request.calibration_run is not None:
        raise ValueError("calibration_run is only permitted for the pair_linear frozen reducer")
    root = Path(request.dataset_root).expanduser().resolve()
    run_dir = Path(request.output_root).expanduser().resolve() / (
        request.run_id or new_run_id("frozen-ucf-evaluation")
    )
    if run_dir.exists():
        raise FileExistsError(f"official evaluation run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    inputs = {
        "test_manifest": test_manifest,
        "audit_report": audit_report,
        "method_freeze": request.freeze_path,
        "training_role_lock": method.role_lock_path,
        "head_data_contract": method.head_data_contract_path,
        "method_checkpoint": method.checkpoint,
        "dense_checkpoint": None if dense is None else dense.checkpoint,
    }
    with record_stage(
        run_dir,
        "frozen_ucf_evaluation",
        config=request.__dict__,
        inputs=inputs,
        project_root=Path.cwd(),
    ):
        adapter, definition = (
            _make_adapter(request) if adapter_factory is None else adapter_factory(request)
        )
        verified = _mapping(
            definition.get("identity"), "official evaluator verified encoder identity"
        )
        if verified.get("adapter") != request.encoder:
            raise ValueError("actual adapter identity differs from requested encoder")
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        clip_frames = _clip_frames(adapter, verified)
        sampling_cfg = _mapping(freeze.get("sampling"), "method freeze.sampling")
        if sampling_cfg.get("native_clip_frames", {}).get(request.encoder) != clip_frames:
            raise ValueError(
                "actual encoder clip frame count differs from the frozen sampling plan"
            )
        frame_stride = sampling_cfg.get("frame_stride")
        short_policy = sampling_cfg.get("short_policy")
        window_stride = _mapping(
            sampling_cfg.get("evaluation_window_stride"), "frozen evaluation stride"
        ).get(request.encoder)
        if not all(
            isinstance(item, int) and item > 0 for item in (frame_stride, window_stride)
        ) or short_policy not in {"strict", "stride1_if_needed"}:
            raise ValueError("frozen UCF sampling plan is invalid")
        from vadbench.paper.reduction_setup import prepare_reduction

        context = None
        reducer = dict(method.reducer)
        if reducer.get("name") != "identity":
            first = manifest[0]
            from vadbench.data.dense_sampling import DenseSamplingPlan

            sample = DenseSamplingPlan(
                clip_frames=clip_frames,
                frame_stride=frame_stride,
                window_stride=window_stride,
                short_policy=short_policy,
            ).sample(first.num_frames)[0]
            batch = build_clip_batch(
                first.resolve_path(root), first.video_id, [sample.clip], backend=video_backend
            )
            reducer_seed = int(reducer["seed"]) if reducer.get("name") == "paired_random" else 0
            context, setup = prepare_reduction(
                adapter,
                request.encoder,
                batch,
                reducer=str(reducer["name"]),
                output_dim=method.representation.output_dim,
                verified_encoder_identity=verified,
                calibration_run=None
                if request.calibration_run is None
                else Path(request.calibration_run),
                seed=reducer_seed,
                batch_sizes=range(1, 9),
            )
            if dict(context.reducer_identity) != reducer:
                raise ValueError(
                    "actual reduction setup identity differs from the frozen source representation"
                )
        else:
            setup = {"status": "identity_no_context"}
        actual_representation = representation_from_verified_encoder(
            runtime_id=request.encoder,
            adapter=adapter,
            verified_encoder_identity=verified,
            preprocessing=method.representation.backbone.preprocessing,
            readout=method.representation.backbone.readout,
            reducer=reducer,
            output_dim=method.representation.output_dim,
            precision=method.representation.precision,
            position_strategy=method.representation.position_strategy,
        )
        if actual_representation != method.representation:
            raise ValueError(
                "actual native encoder/reducer representation differs from the frozen source run"
            )
        evaluation_sampling = make_sampling_identity(
            manifest,
            dataset_root=root,
            sampling_kind="dense",
            clip_frames=clip_frames,
            frame_stride=frame_stride,
            window_stride=window_stride,
            short_policy=short_policy,
        )
        extraction = extract_pooled_features(
            PooledExtractionSpec(
                runtime_id=request.encoder,
                verified_encoder_identity=verified,
                representation=actual_representation,
                sampling=evaluation_sampling,
                sampling_kind="dense",
                clip_frames=clip_frames,
                frame_stride=frame_stride,
                window_stride=window_stride,
                short_policy=short_policy,
            ),
            adapter=adapter,
            manifest=manifest,
            dataset_root=root,
            output_root=run_dir / "features",
            run_id="test",
            backend=video_backend,
            encode_context_factory=context,
        )
        if not extraction.completed or extraction.feature_root is None:
            raise RuntimeError(
                "official test FeatureStore was not completely extracted and published"
            )
        coverage = _coverage(
            manifest, Path(extraction.feature_root), extraction.encoder_fingerprint
        )
        if coverage["input_union_complete"] is not True:
            raise ValueError(
                "official dense sampling does not provide complete decoded frame coverage"
            )
        method.verify_unchanged()
        primary = method.detection_config(
            mode="refit_head",
            evaluation_representation=actual_representation,
            evaluation_sampling=evaluation_sampling,
            evaluation_encoder_fingerprint=extraction.encoder_fingerprint,
        )
        primary_path = run_dir / "predictions-primary-refit-head.jsonl"
        primary_records = predict_detector(
            primary,
            feature_store=extraction.feature_root,
            evaluation_manifest=manifest,
            training=method.checkpoint,
            output_path=primary_path,
            device=request.device,
        )
        primary_metrics = evaluate_detector(
            primary_records, test_manifest, protocol="official", audit_report=audit_report
        )
        primary_metrics_path = run_dir / "metrics-primary-refit-head.json"
        atomic_write_json(primary_metrics_path, primary_metrics.to_dict())
        secondary_path = secondary_metrics_path = None
        if dense is not None:
            dense.verify_unchanged()
            secondary = dense.detection_config(
                mode="direct_insert",
                evaluation_representation=actual_representation,
                evaluation_sampling=evaluation_sampling,
                evaluation_encoder_fingerprint=extraction.encoder_fingerprint,
            )
            secondary_path = run_dir / "predictions-secondary-dense-head-direct-insert.jsonl"
            secondary_records = predict_detector(
                secondary,
                feature_store=extraction.feature_root,
                evaluation_manifest=manifest,
                training=dense.checkpoint,
                output_path=secondary_path,
                device=request.device,
            )
            secondary_metrics = evaluate_detector(
                secondary_records, test_manifest, protocol="official", audit_report=audit_report
            )
            secondary_metrics_path = run_dir / "metrics-secondary-dense-head-direct-insert.json"
            atomic_write_json(secondary_metrics_path, secondary_metrics.to_dict())
        artifacts = {
            "test_feature_store": {
                "root": str(extraction.feature_root),
                "resolved": {
                    "path": str(Path(extraction.feature_root) / "resolved.json"),
                    "sha256": _digest(
                        Path(extraction.feature_root) / "resolved.json",
                        name="test FeatureStore resolved receipt",
                    ),
                },
                "status": {
                    "path": str(Path(extraction.feature_root) / "status.json"),
                    "sha256": _digest(
                        Path(extraction.feature_root) / "status.json",
                        name="test FeatureStore status receipt",
                    ),
                },
                "index": {
                    "path": str(Path(extraction.feature_root) / "index.jsonl"),
                    "sha256": _digest(
                        Path(extraction.feature_root) / "index.jsonl",
                        name="test FeatureStore index",
                    ),
                },
            },
            "primary_predictions": {
                "path": str(primary_path),
                "sha256": _digest(primary_path, name="primary prediction artifact"),
            },
            "primary_metrics": {
                "path": str(primary_metrics_path),
                "sha256": _digest(primary_metrics_path, name="primary metric artifact"),
            },
            "secondary_predictions": None
            if secondary_path is None
            else {
                "path": str(secondary_path),
                "sha256": _digest(secondary_path, name="secondary prediction artifact"),
            },
            "secondary_metrics": None
            if secondary_metrics_path is None
            else {
                "path": str(secondary_metrics_path),
                "sha256": _digest(secondary_metrics_path, name="secondary metric artifact"),
            },
        }
        resolved_path = run_dir / "resolved.json"
        atomic_write_json(
            resolved_path,
            {
                "schema_version": 1,
                "freeze_sha256": freeze_sha,
                "sealed_test_manifest_sha256": _digest(
                    test_manifest, name="sealed UCF test manifest"
                ),
                "sealed_audit_sha256": _digest(audit_report, name="sealed UCF audit"),
                "method_source": _source_receipt(method),
                "dense_source": None if dense is None else _source_receipt(dense),
                "reduction_setup": setup,
                "evaluation_representation": actual_representation.to_dict(),
                "evaluation_representation_fingerprint": actual_representation.fingerprint,
                "evaluation_sampling": evaluation_sampling.to_dict(),
                "evaluation_sampling_fingerprint": evaluation_sampling.fingerprint,
                "evaluation_encoder_fingerprint": extraction.encoder_fingerprint,
                "coverage": coverage,
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
                    "sha256": _digest(resolved_path, name="official evaluation resolved receipt"),
                },
                "artifacts": artifacts,
            },
        )
    return FrozenEvaluationResult(
        str(run_dir),
        True,
        str(primary_path),
        str(primary_metrics_path),
        None if secondary_path is None else str(secondary_path),
        None if secondary_metrics_path is None else str(secondary_metrics_path),
    )


__all__ = [
    "FrozenDetectorSource",
    "FrozenEvaluationRequest",
    "FrozenEvaluationResult",
    "load_frozen_detector_source",
    "run_frozen_ucf_evaluation",
]
