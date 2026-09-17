"""Small, executable controller for frozen paper detector runs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vadbench.artifacts import new_run_id, record_stage
from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.enrich import enrich_video_info
from vadbench.data.manifest import (
    DatasetSplit,
    VideoManifestRecord,
    canonical_video_id,
    load_manifest_jsonl,
    write_manifest_jsonl,
)
from vadbench.features import atomic_write_json
from vadbench.paper.compatibility import CompatibilityDeclaration, TrainingIdentity
from vadbench.paper.detection import (
    DetectionConfig,
    evaluate_detector,
    predict_detector,
    train_detector,
)
from vadbench.paper.extraction import (
    PooledExtractionSpec,
    extract_pooled_features,
    make_sampling_identity,
    representation_from_verified_encoder,
)

RunMode = Literal["engineering", "official"]


@dataclass(frozen=True)
class DetectionExperimentRequest:
    encoder: str
    device: str
    dataset_root: str
    train_manifest: str
    validation_manifest: str | None
    evaluation_manifest: str | None
    output_root: str
    project: str = "projects/icassp2027/profile.yaml"
    reducer: str = "identity"
    frame_stride: int = 2
    dense_window_stride: int | None = None
    output_dim: int = 768
    precision: str = "float32"
    seed: int = 0
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 1e-3
    run_mode: RunMode = "engineering"
    method_frozen: bool = False
    audit_report: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.reducer != "identity":
            raise ValueError("only reducer=identity is currently implemented")
        if self.run_mode not in {"engineering", "official"}:
            raise ValueError("run_mode must be engineering or official")
        if self.run_mode == "official" and (not self.method_frozen or not self.audit_report):
            raise ValueError("official evaluation requires method_frozen=true and an audit_report")
        if self.run_mode == "engineering" and self.evaluation_manifest is None:
            raise ValueError("engineering mode requires a validation evaluation manifest")
        for name in ("frame_stride", "output_dim", "epochs", "batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dense_window_stride is not None and self.dense_window_stride <= 0:
            raise ValueError("dense_window_stride must be positive when specified")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class DetectionExperimentResult:
    run_dir: str
    completed: bool
    training_checkpoint: str | None
    predictions: str | None
    metrics: str | None


def _load(path: str) -> tuple[VideoManifestRecord, ...]:
    return load_manifest_jsonl(path)


def _require_split(records: Iterable[VideoManifestRecord], split: DatasetSplit, role: str) -> None:
    if not records or any(item.split != split for item in records):
        raise ValueError(f"{role} manifest must contain only {split.value} records")


def _disjoint(*groups: Iterable[VideoManifestRecord]) -> dict[str, Any]:
    seen: set[str] = set()
    coverage = []
    for role, records in zip(("fit", "validation", "evaluation"), groups, strict=True):
        records = tuple(records)
        keys = {f"id:{canonical_video_id(item.video_id)}" for item in records} | {
            f"path:{canonical_video_id(item.path)}" for item in records
        }
        known_sources = 0
        for item in records:
            sources = {
                item.metadata.get(field)
                for field in ("source_id", "source_group")
                if item.metadata.get(field) not in (None, "", "unknown")
            }
            if any(not isinstance(source, str) for source in sources):
                raise ValueError("manifest source_id/source_group must be nonempty strings")
            keys.update(f"source:{source}" for source in sources)
            known_sources += bool(sources)
        overlap = seen & keys
        if overlap:
            raise ValueError(f"fit/validation/evaluation video roles overlap: {sorted(overlap)}")
        seen |= keys
        coverage.append(
            {"role": role, "videos": len(records), "known_source_videos": known_sources}
        )
    return {
        "video_identity_disjoint": True,
        "provided_source_groups_disjoint": True,
        "source_coverage": coverage,
        "near_duplicate_exclusion": "not_established_by_video_identity_checks",
    }


def _clip_frames(adapter: Any, definition: Mapping[str, Any]) -> int:
    constructor = definition.get("constructor", {})
    value = constructor.get("num_frames", constructor.get("clip_frames"))
    if not isinstance(value, int) or value <= 0:
        value = getattr(adapter.capabilities, "fixed_num_frames", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("resolved fixed-clip encoder must declare a positive clip frame count")
    return value


def run_detection_experiment(
    request: DetectionExperimentRequest,
    *,
    adapter_factory: Callable[[Mapping[str, Any]], tuple[Any, Mapping[str, Any]]] | None = None,
    video_backend: Any | None = None,
) -> DetectionExperimentResult:
    """Execute frozen fit→extract→head→predict stages using existing APIs."""

    root = Path(request.dataset_root).expanduser().resolve()
    train_raw = _load(request.train_manifest)
    validation_raw = (
        () if request.validation_manifest is None else _load(request.validation_manifest)
    )
    evaluation_raw = (
        () if request.evaluation_manifest is None else _load(request.evaluation_manifest)
    )
    _require_split(train_raw, DatasetSplit.TRAIN, "train")
    if validation_raw:
        _require_split(validation_raw, DatasetSplit.VAL, "validation")
    if request.run_mode == "official":
        _require_split(evaluation_raw, DatasetSplit.TEST, "official evaluation")
    else:
        _require_split(evaluation_raw, DatasetSplit.VAL, "engineering evaluation")
    role_receipt = _disjoint(train_raw, validation_raw, evaluation_raw)

    selected_run = request.run_id or new_run_id("paper-detection")
    run_dir = Path(request.output_root).expanduser().resolve() / selected_run
    if run_dir.exists():
        raise FileExistsError(f"detection run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    summary = {name: getattr(request, name) for name in request.__dataclass_fields__}
    with record_stage(
        run_dir,
        "paper_detection",
        config=summary,
        inputs={
            "train_manifest": request.train_manifest,
            "validation_manifest": request.validation_manifest,
            "evaluation_manifest": request.evaluation_manifest,
        },
        project_root=Path.cwd(),
    ):
        frozen = run_dir / "frozen"
        frozen.mkdir()
        atomic_write_json(frozen / "role_checks.json", role_receipt)

        def freeze(
            records: Iterable[VideoManifestRecord], path: Path
        ) -> tuple[VideoManifestRecord, ...]:
            enriched = enrich_video_info(records, root, backend=video_backend)
            write_manifest_jsonl(enriched, path, dataset_root=root, require_files=True)
            return enriched

        train = freeze(train_raw, frozen / "train.jsonl")
        validation = () if not validation_raw else freeze(validation_raw, frozen / "val.jsonl")
        evaluation = freeze(evaluation_raw, frozen / "evaluation.jsonl")
        if adapter_factory is None:
            from vadbench.orchestration import encoder_identity
            from vadbench.paper.profile import load_project
            from vadbench.registry import ENCODER_REGISTRY

            project = load_project(request.project)
            selected = project.encoder(request.encoder)
            definition = dict(selected["definition"])
            constructor = dict(definition["constructor"])
            constructor["device"] = request.device
            definition["constructor"] = constructor
            definition["identity"] = encoder_identity(definition, project_root=project.root)
            adapter = ENCODER_REGISTRY.create(request.encoder, **constructor)
        else:
            adapter, definition = adapter_factory(summary)
        verified = definition.get("identity")
        if not isinstance(verified, Mapping):
            raise ValueError(
                "adapter factory must return an orchestration verified encoder identity"
            )
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        clip_frames = _clip_frames(adapter, definition)
        dense_window_stride = request.dense_window_stride
        if dense_window_stride is None:
            dense_window_stride = max(1, clip_frames * request.frame_stride // 2)
        atomic_write_json(
            run_dir / "resolved_sampling.json",
            {
                "clip_frames": clip_frames,
                "frame_stride": request.frame_stride,
                "train_segments": 32,
                "dense_window_stride": dense_window_stride,
                "dense_window_stride_source": "explicit"
                if request.dense_window_stride is not None
                else "half_native_window_span",
            },
        )
        representation = representation_from_verified_encoder(
            runtime_id=request.encoder,
            adapter=adapter,
            verified_encoder_identity=verified,
            preprocessing={"profile": getattr(adapter, "preprocess_profile", "unknown")},
            readout={"kind": getattr(adapter, "pooling", "pooled")},
            reducer={"name": "identity", "calibration": "none"},
            output_dim=request.output_dim,
            precision=request.precision,
            position_strategy={"kind": "native"},
        )
        train_sampling = make_sampling_identity(
            train,
            dataset_root=root,
            sampling_kind="uniform_full",
            clip_frames=clip_frames,
            frame_stride=request.frame_stride,
            num_segments=32,
        )
        validation_sampling = (
            None
            if not validation
            else make_sampling_identity(
                validation,
                dataset_root=root,
                sampling_kind="uniform_full",
                clip_frames=clip_frames,
                frame_stride=request.frame_stride,
                num_segments=32,
            )
        )
        evaluation_sampling = make_sampling_identity(
            evaluation,
            dataset_root=root,
            sampling_kind="dense",
            clip_frames=clip_frames,
            frame_stride=request.frame_stride,
            window_stride=dense_window_stride,
        )
        train_features = extract_pooled_features(
            PooledExtractionSpec(
                runtime_id=request.encoder,
                verified_encoder_identity=verified,
                representation=representation,
                sampling=train_sampling,
                sampling_kind="uniform_full",
                clip_frames=clip_frames,
                frame_stride=request.frame_stride,
                num_segments=32,
            ),
            adapter=adapter,
            manifest=train,
            dataset_root=root,
            output_root=run_dir / "features",
            run_id="train",
            backend=video_backend,
        )
        validation_features = (
            None
            if not validation
            else extract_pooled_features(
                PooledExtractionSpec(
                    runtime_id=request.encoder,
                    verified_encoder_identity=verified,
                    representation=representation,
                    sampling=validation_sampling,
                    sampling_kind="uniform_full",
                    clip_frames=clip_frames,
                    frame_stride=request.frame_stride,
                    num_segments=32,
                ),
                adapter=adapter,
                manifest=validation,
                dataset_root=root,
                output_root=run_dir / "features",
                run_id="validation",
                backend=video_backend,
            )
        )
        evaluation_features = extract_pooled_features(
            PooledExtractionSpec(
                runtime_id=request.encoder,
                verified_encoder_identity=verified,
                representation=representation,
                sampling=evaluation_sampling,
                sampling_kind="dense",
                clip_frames=clip_frames,
                frame_stride=request.frame_stride,
                window_stride=dense_window_stride,
            ),
            adapter=adapter,
            manifest=evaluation,
            dataset_root=root,
            output_root=run_dir / "features",
            run_id="evaluation",
            backend=video_backend,
        )
        if (
            not train_features.completed
            or not evaluation_features.completed
            or (validation_features is not None and not validation_features.completed)
        ):
            raise RuntimeError(
                "feature extraction is incomplete; final FeatureStore index was not published"
            )
        fit_digest = "sha256:" + compute_manifest_sha256(train)
        declaration = CompatibilityDeclaration(
            mode="direct_insert",
            training_representation=representation,
            evaluation_representation=representation,
            training_sampling=train_sampling,
            evaluation_sampling=evaluation_sampling,
            baseline_evaluation_sampling=evaluation_sampling,
            training_identity=TrainingIdentity(
                head={"kind": "topk_mil", "k": 3},
                fit_split_digest=fit_digest,
                seed=request.seed,
                optimization={"epochs": request.epochs, "lr": request.learning_rate},
            ),
            sampling_change="train32_to_testdense",
        )
        detector = DetectionConfig(
            declaration=declaration,
            training_encoder_fingerprint=train_features.encoder_fingerprint,
            evaluation_encoder_fingerprint=evaluation_features.encoder_fingerprint,
            head="topk",
            head_kwargs={"k": 3},
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            seed=request.seed,
            expected_training_clips=32,
            expected_evaluation_clips=None,
        )
        trained = train_detector(
            detector,
            feature_store=train_features.feature_root,
            train_manifest=train,
            validation_manifest=validation or None,
            output_dir=run_dir / "head",
            device=request.device,
            validation_feature_store=None
            if validation_features is None
            else validation_features.feature_root,
            validation_encoder_fingerprint=None
            if validation_features is None
            else validation_features.encoder_fingerprint,
            validation_sampling=validation_sampling,
        )
        predictions_path = run_dir / "predictions.jsonl"
        predictions = predict_detector(
            detector,
            feature_store=evaluation_features.feature_root,
            evaluation_manifest=evaluation,
            training=trained,
            output_path=predictions_path,
            device=request.device,
        )
        metrics_path: Path | None = None
        if request.run_mode == "official":
            metrics = evaluate_detector(
                predictions,
                frozen / "evaluation.jsonl",
                protocol="official",
                audit_report=request.audit_report,
            )
            metrics_path = run_dir / "metrics.json"
            atomic_write_json(metrics_path, metrics.to_dict())
        atomic_write_json(
            run_dir / "result.json",
            {
                "status": "completed",
                "checkpoint_path": trained.checkpoint_path,
                "predictions": str(predictions_path),
                "metrics": None if metrics_path is None else str(metrics_path),
            },
        )
    return DetectionExperimentResult(
        run_dir=str(run_dir),
        completed=True,
        training_checkpoint=trained.checkpoint_path,
        predictions=str(predictions_path),
        metrics=None if metrics_path is None else str(metrics_path),
    )


__all__ = ["DetectionExperimentRequest", "DetectionExperimentResult", "run_detection_experiment"]
