"""Extract complete dense training sequences for the fixed author backend.

This entry point produces features only. It cannot train a head, read test
annotations, or publish a detector score. The original paper extractor retains
its native sampling, FeatureStore, runtime identity and strict resume contracts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vadbench.artifacts import new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.manifest import write_manifest_jsonl
from vadbench.data.video import build_clip_batch
from vadbench.features import atomic_write_json
from vadbench.paper.evaluation import _clip_frames, _coverage, _make_adapter
from vadbench.paper.extraction import (
    PooledExtractionSpec,
    extract_pooled_features,
    make_sampling_identity,
    representation_from_verified_encoder,
)


@dataclass(frozen=True)
class OfficialDenseExtractionRequest:
    encoder: str
    device: str
    training_contract_path: str
    training_contract_sha256: str
    protocol_path: str
    protocol_sha256: str
    output_root: str
    project: str = "projects/icassp2027/profile.yaml"
    reducer: str = "identity"
    calibration_run: str | None = None
    processor_tensor_type: str | None = None
    run_id: str | None = None
    resume_source: str | None = None
    engineering_video_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.encoder not in {"videomaev2", "timesformer", "vjepa2", "videomae"}:
            raise ValueError("encoder is outside the active four-encoder scope")
        if self.reducer not in {"identity", "global_uniform", "paired_random", "pair_linear"}:
            raise ValueError("reducer is outside the frozen comparison")
        if (self.reducer == "pair_linear") != (self.calibration_run is not None):
            raise ValueError("only pair_linear requires its frozen calibration run")
        if self.processor_tensor_type not in {None, "pt", "np"}:
            raise ValueError("processor_tensor_type must be pt, np, or null")
        if self.processor_tensor_type is not None and self.encoder not in {"timesformer", "videomae"}:
            raise ValueError("processor override is only supported by TimeSformer and VideoMAE")
        if len(set(self.engineering_video_ids)) != len(self.engineering_video_ids):
            raise ValueError("engineering subset contains duplicate video IDs")


def _bound_json(path: str | Path, expected: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if sha256_file(path) != expected:
        raise ValueError(f"source SHA-256 mismatch: {path.name}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _entry(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def run_official_dense_extraction(
    request: OfficialDenseExtractionRequest,
    *,
    adapter_factory: Callable[[Any], tuple[Any, Mapping[str, Any]]] | None = None,
    video_backend: Any | None = None,
) -> dict[str, Any]:
    """Publish a SHA-bound contract only after every requested video completes."""
    from vadbench.data.official_training import load_official_training_view

    protocol_path, protocol = _bound_json(request.protocol_path, request.protocol_sha256)
    if protocol.get("schema") != "icassp2027.official-detector-protocol/v2":
        raise ValueError("official extraction requires detector protocol v2")
    training_contract_path = Path(request.training_contract_path).expanduser().resolve()
    records, authority = load_official_training_view(
        training_contract_path, request.training_contract_sha256
    )
    dataset = authority["dataset"]
    if request.encoder not in protocol["datasets"][dataset]["encoders"]:
        raise ValueError("encoder is outside this dataset's declared scope")
    root = Path(authority["dataset_root"]).resolve()
    data_role = "official-fulltrain-final"
    if request.engineering_video_ids:
        selected = set(request.engineering_video_ids)
        if not selected <= {item.video_id for item in records}:
            raise ValueError("engineering subset contains a video outside the official training view")
        records = tuple(item for item in records if item.video_id in selected)
        data_role = "engineering-subset-of-official-fulltrain"
    selected_run = request.run_id or new_run_id("official-dense-training-features")
    if not selected_run or any(c in selected_run for c in "/\\"):
        raise ValueError("run_id must be a basename")
    run_dir = Path(request.output_root).expanduser().resolve() / selected_run
    run_dir.mkdir(parents=True, exist_ok=False)
    with record_stage(
        run_dir,
        "official_dense_training_extraction",
        config=asdict(request),
        inputs={"training_contract": str(training_contract_path), "protocol": str(protocol_path)},
        project_root=Path.cwd(),
    ):
        frozen_manifest = run_dir / "train.jsonl"
        write_manifest_jsonl(records, frozen_manifest, dataset_root=root, require_files=True)
        adapter, definition = (adapter_factory or _make_adapter)(request)
        verified = definition["identity"]
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        clip_frames = _clip_frames(adapter, verified)
        output_dim = 1024 if request.encoder == "vjepa2" else 768
        frame_stride = 2
        window_stride = clip_frames
        short_policy = "stride1_if_needed"
        plan = DenseSamplingPlan(
            clip_frames=clip_frames,
            frame_stride=frame_stride,
            window_stride=window_stride,
            short_policy=short_policy,
        )
        # Validate all native inputs before the first extraction; no video may
        # silently disappear because it is shorter than the model's clip.
        expected_clips = sum(len(plan.sample(item.num_frames)) for item in records)
        factory = None
        reducer_identity = {"name": "identity", "calibration": "none"}
        if request.reducer != "identity":
            from vadbench.paper.reduction_setup import prepare_reduction

            sample = plan.sample(records[0].num_frames)[0]
            setup_batch = build_clip_batch(
                records[0].resolve_path(root), records[0].video_id, [sample.clip], backend=video_backend
            )
            factory, reducer_receipt = prepare_reduction(
                adapter,
                request.encoder,
                setup_batch,
                reducer=request.reducer,
                output_dim=output_dim,
                verified_encoder_identity=verified,
                calibration_run=None if request.calibration_run is None else Path(request.calibration_run),
                seed=0,
                batch_sizes=range(1, 9),
            )
            reducer_identity = dict(factory.reducer_identity)
            atomic_write_json(run_dir / "resolved_reducer.json", reducer_receipt)
        representation = representation_from_verified_encoder(
            runtime_id=request.encoder,
            adapter=adapter,
            verified_encoder_identity=verified,
            preprocessing={"profile": getattr(adapter, "preprocess_profile", "unknown")},
            readout={"kind": getattr(adapter, "pooling", "pooled")},
            reducer=reducer_identity,
            output_dim=output_dim,
            precision="float32",
            position_strategy={"kind": "native"},
        )
        sampling = make_sampling_identity(
            records,
            dataset_root=root,
            sampling_kind="dense",
            clip_frames=clip_frames,
            frame_stride=frame_stride,
            window_stride=window_stride,
            short_policy=short_policy,
        )
        result = extract_pooled_features(
            PooledExtractionSpec(
                runtime_id=request.encoder,
                verified_encoder_identity=verified,
                representation=representation,
                sampling=sampling,
                sampling_kind="dense",
                clip_frames=clip_frames,
                frame_stride=frame_stride,
                window_stride=window_stride,
                short_policy=short_policy,
                micro_batch_size=8,
            ),
            adapter=adapter,
            manifest=records,
            dataset_root=root,
            output_root=run_dir / "features",
            run_id="train",
            backend=video_backend,
            encode_context_factory=factory,
            resume_source=request.resume_source,
        )
        if not result.completed:
            failure = {"status": "failed", "extraction": asdict(result), "data_role": data_role}
            atomic_write_json(run_dir / "result.json", failure)
            return failure
        feature_root = Path(result.feature_root).resolve()
        if result.records_written != expected_clips:
            raise ValueError("completed extraction record count differs from the native dense plan")
        coverage = _coverage(records, feature_root, result.encoder_fingerprint)
        actual_contents = json.loads((feature_root / "resolved.json").read_text(encoding="utf-8"))["data_content_evidence"]["videos"]
        observed = {item["video_id"]: (item["sha256"], item["size_bytes"]) for item in actual_contents}
        expected = {
            item.video_id: (item.metadata["content_sha256"], item.metadata["content_size_bytes"])
            for item in records
        }
        if observed != expected:
            raise ValueError("extracted video content differs from the independently audited training view")
        # Recheck the immutable authority files after a potentially long run.
        _bound_json(request.protocol_path, request.protocol_sha256)
        load_official_training_view(training_contract_path, request.training_contract_sha256)
        contract = {
            "schema": "icassp2027.official-dense-extraction/v1",
            "status": "ready",
            "data_role": data_role,
            "dataset": dataset,
            "dataset_root": str(root),
            "encoder": request.encoder,
            "reducer": request.reducer,
            "training_view_contract": _entry(training_contract_path),
            "training_manifest": _entry(frozen_manifest),
            "protocol": _entry(protocol_path),
            "feature_store": {
                "root": str(feature_root),
                **{name: _entry(feature_root / file) for name, file in (
                    ("resolved", "resolved.json"), ("status", "status.json"), ("index", "index.jsonl")
                )},
            },
            "coverage": coverage,
            "videos": len(records),
            "clips": expected_clips,
            "training_sampling_note": "test_dense is the existing sampler's name; all data are official training videos",
            "engineering_video_ids": list(request.engineering_video_ids),
            "detector_training_executed": False,
            "test_scoring_executed": False,
        }
        path = run_dir / "extraction-contract.json"
        atomic_write_json(path, contract)
        result_document = {"status": "completed", "contract": _entry(path), "data_role": data_role}
        atomic_write_json(run_dir / "result.json", result_document)
        return result_document
