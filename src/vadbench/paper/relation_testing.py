"""Frozen relation-ablation orchestration using the existing extraction/scorer.

No new sampler, detector or metric implementation lives here. Test access is
bound to a fresh, user-authorized campaign; historical freeze files are intact.
"""
from __future__ import annotations

import gc
import json
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vadbench.checkpoints import sha256_file
from vadbench.data.batches import clean_encoder_batch
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.feature_contracts import CompatibilityDeclaration, TrainingIdentity, validate_compatibility
from vadbench.data.manifest import load_manifest_jsonl, validate_manifest_pair, write_manifest_jsonl
from vadbench.data.official_training import load_official_training_view
from vadbench.data.video import build_clip_batch
from vadbench.engine.coverage import feature_store_coverage
from vadbench.features import atomic_write_json
from vadbench.paper.evaluation import _make_adapter
from vadbench.paper.urdmu_evaluation import URDMUEvaluationRequest, _load_source, run_urdmu_evaluation
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.relation_deployment import RelationDeployment
from vadbench.workflows.extraction import PooledExtractionSpec, extract_pooled_features, make_sampling_identity, representation_from_verified_encoder
from vadbench.workflows.urdmu_quality_export import compare_prediction_runs


def bound(binding):
    path = Path(binding["path"]).resolve()
    if sha256_file(path) != binding["sha256"]:
        raise ValueError(f"frozen input SHA differs: {path}")
    return path


def load_campaign(path, digest):
    path = bound({"path": str(path), "sha256": digest})
    c = json.loads(path.read_text("utf-8"))
    if c.get("schema") != "vadbench.relation-test-campaign/v1" or c.get("status") != "frozen_before_test":
        raise ValueError("not an active frozen relation test campaign")
    expected = {(d, m, b, k) for d in ("ucf_crime", "xd_violence")
                for m in ("attention_topk", "visionzip_internal", "uniform")
                for b in (5, 8, 11) for k in (0.5, 0.3)}
    expected |= {(d, "identity", None, 1.0) for d in ("ucf_crime", "xd_violence")}
    tasks = c["tasks"]
    actual = {(t["dataset"], t["method"], t["block_index"], t["keep_ratio"]) for t in tasks}
    if len(tasks) != 38 or len({t["id"] for t in tasks}) != 38 or actual != expected:
        raise ValueError("campaign is not the exact frozen 38-condition grid")
    for name, digest in c["implementation_sha256"].items():
        bound({"path": str(Path(c["snapshot_root"]) / name), "sha256": digest})
    for key in ("method_freeze", "freeze_markdown", "engineering_qa"):
        bound(c[key])
    return c


def evaluation_request(c, task, feature_root, manifest, contract=None):
    ds = c["datasets"][task["dataset"]]
    return URDMUEvaluationRequest(
        dataset=task["dataset"], phase="official_frame", mode="direct_insert",
        trained_run=ds["head_run"], feature_store=str(feature_root),
        evaluation_manifest=str(manifest), upstream_dir=c["upstream_dir"],
        output_root=str(Path(c["output_root"]) / task["id"] / "scoring"),
        device="cuda:0", run_id="predictions", query_chunk_size=256,
        max_attention_workspace_bytes=2 * 1024**3,
        freeze_path=c["freeze_markdown"]["path"], audit_report=str(bound(ds["audit"])),
        method_freeze_contract_path=c["method_freeze"]["path"],
        method_freeze_contract_sha256=c["method_freeze"]["sha256"],
        feature_contract_path=None if contract is None else str(contract),
        feature_contract_sha256=None if contract is None else sha256_file(contract),
    )


def run_test(c, task, tracking_url, emit):
    ds = c["datasets"][task["dataset"]]
    out = Path(c["output_root"]) / task["id"]
    out.mkdir(parents=True, exist_ok=False)
    atomic_write_json(out / "task.json", {"status": "running", "task": task, "tracking_url": tracking_url})
    torch.cuda.set_per_process_memory_fraction(c["cuda_memory_fraction"], device=0)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    try:
        training, _ = load_official_training_view(bound(ds["training_contract"]), ds["training_contract"]["sha256"])
        manifest = bound(ds["test_manifest"])
        records = tuple(load_manifest_jsonl(manifest))
        validate_manifest_pair(training, records)
        if len(records) != ds["test_videos"]:
            raise ValueError("sealed test membership count differs")
        bound(ds["head_checkpoint"])
        request = evaluation_request(c, task, out / "features/native", manifest)
        source = _load_source(request)
        adapter, definition = _make_adapter(SimpleNamespace(
            encoder="videomaev2", project=c["project"], device="cuda:0", processor_tensor_type=None))
        bridge = create_observation_bridge("videomaev2", adapter)
        bridge.model.eval().requires_grad_(False)
        sampling_args = dict(sampling_kind="dense", clip_frames=16, frame_stride=2,
                             window_stride=16, short_policy="stride1_if_needed")
        plan = DenseSamplingPlan(clip_frames=16, frame_stride=2, window_stride=16, short_policy="stride1_if_needed")
        factory = None
        reducer = {"name": "identity", "calibration": "none"}
        if task["method"] != "identity":
            first = records[0]
            clip = plan.sample(first.num_frames)[0].clip
            batch = clean_encoder_batch(build_clip_batch(first.resolve_path(ds["dataset_root"]), first.video_id, [clip]))
            with torch.inference_mode(), bridge.geometry(batch.frame_indices, batch.valid_mask) as geometry:
                adapter.encode(batch, train=False)
            factory = RelationDeployment(bridge, geometry.layout, depth=task["block_index"],
                                         method=task["method"], keep_ratio=task["keep_ratio"])
            reducer = factory.reducer_identity
        representation = representation_from_verified_encoder(
            runtime_id="videomaev2", adapter=adapter, verified_encoder_identity=definition["identity"],
            preprocessing={"profile": getattr(adapter, "preprocess_profile", "unknown")},
            readout={"kind": getattr(adapter, "pooling", "pooled")}, reducer=reducer,
            output_dim=768, precision="float32", position_strategy={"kind": "native"})
        # Fail before a full extraction if the original dense head cannot pair.
        declaration = CompatibilityDeclaration(
            mode="direct_insert", training_representation=source.representation,
            evaluation_representation=representation, training_sampling=source.sampling,
            evaluation_sampling=source.sampling, baseline_evaluation_sampling=source.sampling,
            training_identity=TrainingIdentity({"kind": "urdmu", "checkpoint_role": "dense_reference"},
                source.fit_manifest_digest, source.metadata["seed"], {"steps":3000,"bags_per_class":64}),
            sampling_change="none")
        compatibility = validate_compatibility(declaration)
        atomic_write_json(out / "preflight.json", {"compatibility": compatibility,
            "representation": representation.to_dict(), "head_sha256": source.checkpoint_sha256})
        emit("preflight", task_id=task["id"], representation=representation.to_dict(),
             head_sha256=source.checkpoint_sha256, metrics={"test/compatibility_passed":1})
        sampling = make_sampling_identity(records, dataset_root=ds["dataset_root"], **sampling_args)
        frozen_manifest = out / "test.jsonl"
        write_manifest_jsonl(records, frozen_manifest, dataset_root=ds["dataset_root"], require_files=True)
        def progress(row):
            emit("progress", task_id=task["id"], video_id=row["video_id"],
                 metrics={"test/videos_completed":row["videos_completed"],
                          "test/videos_total":row["videos_total"], "test/clips_written":row["clips_written"]})
        result = extract_pooled_features(
            PooledExtractionSpec(runtime_id="videomaev2", verified_encoder_identity=definition["identity"],
                representation=representation, sampling=sampling, micro_batch_size=8, **sampling_args),
            adapter=adapter, manifest=records, dataset_root=ds["dataset_root"],
            output_root=out / "features", run_id="native", encode_context_factory=factory, progress=progress)
        if not result.completed:
            raise RuntimeError(f"extraction failed: {result.failures}")
        feature_root = Path(result.feature_root)
        coverage = feature_store_coverage(records, feature_root, result.encoder_fingerprint)
        bindings = {k:{"path":str(feature_root / name),"sha256":sha256_file(feature_root / name)}
                    for k,name in (("resolved","resolved.json"),("status","status.json"),("index","index.jsonl"))}
        contract = out / "extraction-contract.json"
        atomic_write_json(contract, {"schema":"icassp2027.official-dense-extraction/v1", "status":"ready",
            "dataset":task["dataset"], "data_role":"official-sealed-test", "test_only":True,
            "feature_store":{"root":str(feature_root),**bindings}, "coverage":coverage,
            "training_view_contract":ds["training_contract"], "sealed_test_manifest":ds["test_manifest"],
            "training_manifest":{"path":str(frozen_manifest),"sha256":sha256_file(frozen_manifest)},
            "campaign_task":task, "tracking_url":tracking_url})
        emit("extracted", metrics={"test/extraction_completed":1,"test/clips_total":result.records_written})
        del factory, bridge, adapter
        gc.collect(); torch.cuda.empty_cache()
        scored = run_urdmu_evaluation(evaluation_request(c, task, feature_root, frozen_manifest, contract))
        load_campaign(c["campaign_path"], c["campaign_sha256"])
        final = {"status":"prediction_completed", "task":task,"extraction":asdict(result),
                 "scoring":asdict(scored), "tracking_url":tracking_url,"formal_metrics_read":False}
        atomic_write_json(out / "task.json", final)
        emit("completed", task_id=task["id"], predictions=scored.predictions,
             metrics={"test/completed":1,"test/videos_total":len(records)})
    except Exception as exc:
        atomic_write_json(out / "task.json", {"status":"failed","task":task,"error":str(exc),"tracking_url":tracking_url})
        raise


def run_quality(c, task, tracking_url, emit):
    ds = c["datasets"][task["dataset"]]
    out = Path(c["output_root"]) / task["id"]
    method = out / "scoring/predictions/predictions.jsonl"
    dense = Path(c["output_root"]) / f"{task['dataset']}-dense" / "scoring/predictions/predictions.jsonl"
    for root in {out, dense.parents[2]}:
        if json.loads((root / "task.json").read_text())["status"] != "prediction_completed":
            raise ValueError("quality requires completed native prediction runs")
    target = out / "quality.json"
    if target.exists():
        raise FileExistsError("refusing to overwrite a quality export")
    result = compare_prediction_runs(dense_predictions=dense, method_predictions=method,
        truth_jsonl=bound(ds["truth"]), dataset=task["dataset"],
        source_paths=[c["campaign_path"],out / "extraction-contract.json",out / "scoring/predictions/resolved.json"])
    atomic_write_json(target, result)
    metric = result["metric"]
    emit("completed", task_id=task["id"], quality_path=str(target), quality=result,
         metrics={"test/completed":1, "quality/dense":metric["dense_value"],
                  "quality/method":metric["method_value"], "quality/delta":metric["delta_method_minus_dense"]})
