"""Fit-only native VideoMAEv2 engineering QA; never detector evaluation/timing.

The caller must initialize online SwanLab and stream EVENT lines to that run.
This process preserves artifacts locally on failure; it does not read heads,
test manifests, frame annotations, or AUC/AP. Actual execution loads weights.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import vadbench
from vadbench.checkpoints import sha256_file
from vadbench.contracts import validate_encoder_output
from vadbench.data.dense_sampling import sample_uniform_full_clips
from vadbench.data.manifest import DatasetSplit, load_manifest_jsonl
from vadbench.data.video import build_clip_batch, probe_video
from vadbench.features import atomic_write_json
from vadbench.orchestration import encoder_identity, slice_clip_batch
from vadbench.paper.profile import load_project
from vadbench.paper.stages import clean_encoder_batch
from vadbench.registry import ENCODER_REGISTRY
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.bridges.post_attention import PostAttentionIntervention
from vadbench.token_reduction.relation_deployment import RelationDeployment
from vadbench.token_reduction.relation_selection import METHODS

DEPTHS = (5, 8, 11)
RATIOS = (0.5, 0.3)
WINDOW_INDICES = (1, 6)
POOLED_TOLERANCE = {"rtol": 1e-3, "atol": 1e-3}
KEY_TOLERANCE = {"rtol": 5e-3, "atol": 5e-3}


def emit(kind: str, **values: Any) -> None:
    print("EVENT " + json.dumps({"kind": kind, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def _finite_output(output: Any, batch: Any, tokens: int) -> torch.Tensor:
    validate_encoder_output(output, batch)
    if output.features.shape != (2, tokens, 768) or output.pooled.shape != (2, 768):
        raise ValueError(f"unexpected native features/pooled shapes: {output.features.shape}/{output.pooled.shape}")
    if not bool(torch.isfinite(output.features).all()) or not bool(torch.isfinite(output.pooled).all()):
        raise ValueError("non-finite native features/pooled values")
    return output.pooled.detach().float().cpu()


def _distortion(pooled: torch.Tensor, dense: torch.Tensor) -> dict[str, float]:
    dense_norm = dense.norm(dim=-1)
    if bool(dense_norm.eq(0).any()):
        raise ValueError("dense pooled norm is zero; relative diagnostic is undefined")
    return {
        "qa/pooled_cosine_mean": float(F.cosine_similarity(pooled, dense).mean()),
        "qa/pooled_relative_l2_mean": float(((pooled - dense).norm(dim=-1) / dense_norm).mean()),
        "qa/pooled_norm_ratio_mean": float((pooled.norm(dim=-1) / dense_norm).mean()),
        "qa/finite": 1.0,
    }


def _capture_identity(adapter: Any, bridge: Any, batch: Any, depth: int, dense: torch.Tensor):
    reference: dict[str, Any] = {"calls": 0}
    attention = bridge._blocks[depth].attn

    def full_qkv_reference(module: Any, inputs: tuple[Any, ...]) -> None:
        # This diagnostic adds one full-QKV projection only, without changing
        # native forward/backend/output. It is explicitly NOT efficiency work.
        x = inputs[0]
        bias = None
        if module.q_bias is not None:
            bias = torch.cat((module.q_bias, torch.zeros_like(module.v_bias), module.v_bias))
        qkv = F.linear(x, module.qkv.weight, bias)
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        keys = qkv[1]
        if type(module).__name__ == "CosAttention":
            keys = F.normalize(keys, dim=-1)
        reference["keys"] = keys
        reference["calls"] += 1

    def identity(hidden: torch.Tensor, keys: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if keys.shape != reference["keys"].shape or scores.shape != hidden.shape[:2]:
            raise ValueError("capture-on native signal geometry mismatch")
        if not bool(torch.isfinite(keys).all()) or not bool(torch.isfinite(scores).all()):
            raise ValueError("non-finite native key/attention signal")
        torch.testing.assert_close(keys, reference["keys"], **KEY_TOLERANCE)
        reference["key_max_abs"] = float((keys.float() - reference.pop("keys").float()).abs().max().cpu())
        return hidden

    handle = attention.register_forward_pre_hook(full_qkv_reference)
    try:
        with PostAttentionIntervention(bridge, depth, identity, True, True) as intervention:
            output = adapter.encode(batch, train=False)
            receipt = intervention.validate_execution()
    finally:
        handle.remove()
    pooled = _finite_output(output, batch, 1568)
    torch.testing.assert_close(pooled, dense, **POOLED_TOLERANCE)
    if reference["calls"] != 1:
        raise ValueError("full-QKV reference did not execute exactly once in the same forward")
    return pooled, receipt, {
        **_distortion(pooled, dense),
        "qa/identity_pooled_max_abs": float((pooled - dense).abs().max()),
        "qa/key_slice_full_qkv_max_abs": reference["key_max_abs"],
    }


def _assert_compression(receipt: dict[str, Any], output_tokens: int, depth: int) -> None:
    if receipt["before_tokens"] != 1568 or receipt["gathered_tokens"] != output_tokens:
        raise ValueError("native compression did not execute its frozen budget")
    if receipt["site"] != "post_attention_residual_pre_norm2_mlp" or receipt["block_index"] != depth:
        raise ValueError("compression executed at the wrong native site")
    if receipt["selected_block_mlp_tokens"] != output_tokens:
        raise ValueError("selected block MLP did not receive compressed tokens")
    for layer in range(12):
        if receipt["per_layer_token_counts"][str(layer)] != (1568 if layer <= depth else output_tokens):
            raise ValueError("actual native attention input length mismatch")
        if receipt["per_layer_mlp_token_counts"][str(layer)] != (1568 if layer < depth else output_tokens):
            raise ValueError("actual native MLP input length mismatch")
    for shape in receipt["suffix_shapes"].values():
        if shape != [2, output_tokens, 768]:
            raise ValueError("native suffix did not use the shortened B=2 sequence")


def _save_case(output: Path, result: dict[str, Any], case: dict[str, Any], pooled: torch.Tensor) -> None:
    case_dir = output / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(case_dir / "pooled.npz", pooled=pooled.numpy())
    case["pooled_path"] = str(case_dir / "pooled.npz")
    atomic_write_json(case_dir / "receipt.json", case)
    result["cases"].append(case)
    atomic_write_json(output / "result.json", result)
    emit("case", **case)


def run(args: argparse.Namespace, output: Path, result: dict[str, Any]) -> None:
    if not args.tracking_url.strip():
        raise ValueError("--tracking-url must identify the already initialized online SwanLab run")
    root = args.project_root.resolve()
    manifest = args.manifest.resolve()
    role_path = args.role_lock.resolve()
    if "test" in manifest.name.casefold():
        raise ValueError("test manifest paths are forbidden for engineering QA")
    records = load_manifest_jsonl(manifest)
    roles = json.loads(role_path.read_text(encoding="utf-8"))
    partitions = roles.get("partitions", {})
    if not records or len({record.video_id for record in records}) != len(records):
        raise ValueError("QA manifest must contain nonempty unique video records")
    if any(record.split != DatasetSplit.TRAIN or partitions.get(record.video_id) != "fit" for record in records):
        raise ValueError("all manifest records must be train and role-locked fit; no subset bypass")
    project = load_project(root / "projects/icassp2027/profile.yaml", root=root)
    if set(project.encoders) != {"videomaev2", "videomae", "timesformer"}:
        raise ValueError("project profile must contain the current three dense encoders")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("real-weight QA requires the existing CUDA runtime")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    definition = dict(project.encoder("videomaev2")["definition"])
    constructor = dict(definition["constructor"])
    constructor["device"] = args.device
    definition["constructor"] = constructor
    identity = encoder_identity(definition, project_root=root)
    sources = []
    for record in records:
        path = record.resolve_path(args.dataset_root.resolve())
        sources.append({"video_id": record.video_id, "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    adapter = ENCODER_REGISTRY.create("videomaev2", **constructor)
    bridge = create_observation_bridge("videomaev2", adapter)
    bridge.model.eval()
    for parameter in bridge.model.parameters():
        parameter.requires_grad_(False)
    if bridge.receipt().block_count != 12:
        raise ValueError("QA is frozen to native VideoMAEv2 base with 12 blocks")
    props = torch.cuda.get_device_properties(device)
    provenance = {
        "scope": "engineering_qa_fit_only_not_auc_ap_or_formal_efficiency",
        "encoder_id": "videomaev2", "verified_encoder_identity": identity,
        "constructor": constructor, "use_half_unchanged": constructor.get("use_half"),
        "bridge_architecture": bridge.architecture(),
        "python_executable": sys.executable, "vadbench_file": vadbench.__file__,
        "project_root": str(root), "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "device": args.device, "device_name": props.name, "device_total_memory_bytes": props.total_memory,
        "device_uuid": str(getattr(props, "uuid", "unavailable")),
        "cuda_numeric_settings": {"cudnn_enabled": torch.backends.cudnn.enabled,
                                  "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                  "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                  "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
        "role_lock": {"path": str(role_path), "sha256": sha256_file(role_path)},
        "profile": {"path": str(project.path), "sha256": sha256_file(project.path)},
        "source_videos": sources, "seed": args.seed,
        "sampling": {"kind": "uniform_full_clips", "num_segments": 8, "selected_indices": list(WINDOW_INDICES), "clip_frames": 16, "frame_stride": 2, "short_policy": "stride1_if_needed"},
        "depths": list(DEPTHS), "methods": list(METHODS), "keep_ratios": list(RATIOS),
        "pooled_parity_tolerance": POOLED_TOLERANCE, "key_parity_tolerance": KEY_TOLERANCE,
        "key_reference": "same_forward_input_full_functional_QKV_reference_reprojection_not_native_tensor_capture",
        "tracking_url": args.tracking_url, "command": [sys.executable, *sys.argv],
        "qa_script_sha256": sha256_file(Path(__file__)),
        "implementation_sources": {
            str(Path(inspect.getfile(item)).resolve()): sha256_file(Path(inspect.getfile(item)))
            for item in (RelationDeployment, PostAttentionIntervention, clean_encoder_batch,
                         build_clip_batch, sample_uniform_full_clips, type(adapter))
        } | {
            str(Path(sys.modules["vadbench.token_reduction.relation_selection"].__file__).resolve()):
            sha256_file(Path(sys.modules["vadbench.token_reduction.relation_selection"].__file__))
        },
    }
    atomic_write_json(output / "provenance.json", provenance)
    result["provenance"] = provenance
    atomic_write_json(output / "result.json", result)
    emit("provenance", **provenance)
    deployments = {}
    with torch.no_grad():
        for video_index, (record, source) in enumerate(zip(records, sources, strict=True)):
            path = Path(source["path"])
            info = probe_video(path)
            if record.num_frames is not None and record.num_frames != info.num_frames:
                raise ValueError("manifest num_frames differs from decoded source video")
            if record.fps is not None and not math.isclose(record.fps, info.fps, rel_tol=1e-3):
                raise ValueError("manifest fps differs from decoded source video")
            windows = sample_uniform_full_clips(info.num_frames, num_segments=8, clip_frames=16, frame_stride=2, short_policy="stride1_if_needed")
            if len(windows) != 8:
                raise ValueError("uniform8 sampling did not return eight windows")
            batch = clean_encoder_batch(build_clip_batch(path, record.video_id, [windows[index].clip for index in WINDOW_INDICES]))
            if not bool(np.asarray(batch.valid_mask).all()) or batch.frames.shape[:2] != (2, 16):
                raise ValueError("QA requires two complete 16-frame native clips")
            if video_index == 0:
                first = slice_clip_batch(batch, 0, 1)
                with bridge.geometry(first.frame_indices, first.valid_mask) as geometry:
                    geometry_output = adapter.encode(first, train=False)
                validate_encoder_output(geometry_output, first)
                if geometry.layout is None or geometry.layout.token_capacity != 1568 or geometry_output.features.shape != (1, 1568, 768):
                    raise ValueError("first-video dense geometry is not verified native 1568 tokens")
                atomic_write_json(output / "geometry.json", geometry.receipt)
                deployments = {
                    (method, depth, ratio): RelationDeployment(bridge, geometry.layout, depth=depth, method=method, keep_ratio=ratio)
                    for method in METHODS for depth in DEPTHS for ratio in RATIOS
                }
            case_common = {"video_id": record.video_id, "video_index": video_index,
                           "sampling_frame_indices": np.asarray(batch.frame_indices).tolist(),
                           "actual_video_info": {"num_frames": info.num_frames, "fps": info.fps,
                                                 "width": info.width, "height": info.height},
                           "input_sha256": source["sha256"], "scope": provenance["scope"]}
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            dense_output = adapter.encode(batch, train=False)
            torch.cuda.synchronize(device)
            dense = _finite_output(dense_output, batch, 1568)
            del dense_output
            _save_case(output, result, {**case_common, "case_id": f"v{video_index:04d}-dense",
                "method": "dense", "metrics": {"qa/finite": 1.0, "qa/diagnostic_wall_seconds_not_formal_latency": time.perf_counter() - started}}, dense)
            for depth in DEPTHS:
                pooled, receipt, metrics = _capture_identity(adapter, bridge, batch, depth, dense)
                _save_case(output, result, {**case_common, "case_id": f"v{video_index:04d}-identity-d{depth:02d}", "method": "capture_on_identity", "depth": depth, "receipt": receipt, "metrics": metrics}, pooled)
                result["identity_checks"] += 1
            for (method, depth, ratio), deployment in deployments.items():
                case_id = f"v{video_index:04d}-{method}-d{depth:02d}-k{int(ratio * 100):02d}"
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with deployment(batch) as context:
                    encoded = adapter.encode(batch, train=False)
                    receipt = context.validate_execution()
                torch.cuda.synchronize(device)
                wall = time.perf_counter() - started
                tokens = 784 if ratio == 0.5 else 470
                _assert_compression(receipt, tokens, depth)
                pooled = _finite_output(encoded, batch, tokens)
                del encoded
                if video_index == 0:
                    selection = context.selection
                    selection_dir = output / "membership"
                    selection_dir.mkdir(exist_ok=True)
                    np.savez_compressed(selection_dir / f"{case_id}.npz",
                        source_to_output=selection.source_to_output[0].detach().cpu().numpy(),
                        representatives=selection.representatives[0].detach().cpu().numpy(),
                        mass=selection.mass[0].detach().cpu().numpy())
                _save_case(output, result, {**case_common, "case_id": case_id, "method": method, "depth": depth, "keep_ratio": ratio,
                    "reducer_identity": deployment.reducer_identity, "receipt": receipt,
                    "membership_path": str(output / "membership" / f"{case_id}.npz") if video_index == 0 else None,
                    "metrics": {**_distortion(pooled, dense), "qa/diagnostic_wall_seconds_not_formal_latency": wall}}, pooled)
                result["compressed_cases"] += 1
                del context
            restored = _finite_output(adapter.encode(batch, train=False), batch, 1568)
            torch.testing.assert_close(restored, dense, **POOLED_TOLERANCE)
            _save_case(output, result, {**case_common, "case_id": f"v{video_index:04d}-dense-restored", "method": "dense_after_context_restore",
                "metrics": {**_distortion(restored, dense), "qa/restored_pooled_max_abs": float((restored - dense).abs().max())}}, restored)
    result["status"] = "completed"
    result["metrics"] = {"qa/completed": 1, "qa/compressed_cases": result["compressed_cases"], "qa/identity_checks": result["identity_checks"]}
    atomic_write_json(output / "result.json", result)
    emit("completed", output=str(output), metrics=result["metrics"], scope=provenance["scope"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role-lock", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-url", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--windows-per-video", type=int, default=2, choices=(2,))
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        emit("failed", output=str(output), error="refusing to overwrite an existing output", metrics={"qa/failed": 1})
        return 1
    result: dict[str, Any] = {"status": "running", "scope": "engineering_qa_fit_only", "tracking_url": args.tracking_url,
                              "command": [sys.executable, *sys.argv], "cases": [], "compressed_cases": 0, "identity_checks": 0}
    atomic_write_json(output / "result.json", result)
    try:
        run(args, output, result)
    except Exception as exc:
        stack = traceback.format_exc()
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=stack)
        atomic_write_json(output / "result.json", result)
        (output / "failure.log").write_text(stack, encoding="utf-8")
        emit("failed", output=str(output), error_type=type(exc).__name__, error=str(exc), metrics={"qa/failed": 1})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
