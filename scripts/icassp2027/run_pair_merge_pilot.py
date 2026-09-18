"""Run the frozen 26-fit pair-mean engineering pilot only with --execute."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import run_neutral_pilot as neutral
import torch

import vadbench
from vadbench.artifacts import new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.dense_sampling import sample_uniform_full_clips
from vadbench.data.video import OpenCVVideoReader, _batch_from_reader
from vadbench.features import atomic_write_json
from vadbench.orchestration import encoder_identity
from vadbench.paper.profile import load_project
from vadbench.paper.stages import clean_encoder_batch
from vadbench.registry import ENCODER_REGISTRY
from vadbench.research.interventions import paired_spatial_indices
from vadbench.token_reduction.bridges import (
    create_observation_bridge,
    identity_indices,
    indexed_gather,
)
from vadbench.token_reduction.pair_merge import (
    PairWeightedMerge,
    horizontal_pair_merge_spec,
)

ROOT = Path(__file__).resolve().parents[2]
def _metrics(dense: Any, output: Any) -> dict[str, Any]:
    reference = dense.pooled.detach().float().cpu()
    actual = output.pooled.detach().float().cpu()
    if reference.shape != actual.shape or not torch.isfinite(actual).all():
        raise RuntimeError("adapter pooled output is invalid after pair merge")
    delta = actual - reference
    return {
        "pooled_shape": list(actual.shape),
        "feature_shape": list(output.features.shape),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)),
        "cosine": float(torch.nn.functional.cosine_similarity(reference.reshape(1, -1), actual.reshape(1, -1)).item()),
        "max_absolute_error": float(delta.abs().max()),
    }


def _pair_controls(adapter: Any, encoder: str, batch: Any, *, seed: int) -> dict[str, Any]:
    batch = clean_encoder_batch(batch)
    bridge = create_observation_bridge(encoder, adapter)
    model = bridge.model
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        geometry = bridge.geometry(batch.frame_indices, batch.valid_mask)
        with torch.no_grad(), geometry:
            dense = adapter.encode(batch, train=False)
        verified = geometry.receipt.get("flatten_verified") or (
            geometry.receipt.get("patch_flatten_verified") and geometry.receipt.get("divided_layout_verified")
        )
        if not verified or tuple(dense.features.shape[:2]) != tuple(geometry.layout.valid_mask.shape):
            raise RuntimeError("native bridge geometry/token layout was not verified")
        depth = max(0, math.ceil(0.5 * bridge.receipt().block_count) - 1)
        if depth >= bridge.receipt().block_count - 1:
            raise RuntimeError("relative depth leaves no suffix block")
        layout = geometry.layout
        spec = horizontal_pair_merge_spec(encoder, layout)
        parameter = next(bridge.model.parameters(), None)
        if parameter is None:
            raise RuntimeError("pair merge bridge has no parameter device")
        spec = spec.to(parameter.device)
        dim = int(dense.features.shape[-1])
        mean = PairWeightedMerge(dim)

        def execute(merger: PairWeightedMerge) -> tuple[Any, dict[str, Any]]:
            with torch.no_grad(), indexed_gather(
                bridge,
                depth,
                spec.output_indices,
                layout,
                transform=lambda hidden, _indices: merger(hidden, spec),
                record_position_masks=False,
            ) as run:
                output = adapter.encode(batch, train=False)
                receipt = run.validate_execution()
            if output.features.shape[1] != spec.output_indices.shape[1]:
                raise RuntimeError("adapter feature timeline does not match pair-merge suffix")
            return output, receipt

        identity = identity_indices(layout)
        with torch.no_grad(), indexed_gather(bridge, depth, identity, layout) as run:
            identity_output = adapter.encode(batch, train=False)
            identity_receipt = run.validate_execution()
        torch.testing.assert_close(identity_output.pooled, dense.pooled, rtol=1e-5, atol=1e-6)
        pair_output, pair_receipt = execute(mean)
        scores = torch.zeros(layout.valid_mask.shape, dtype=dense.features.dtype, device=dense.features.device)
        paired_random = paired_spatial_indices(scores, layout, "random", seed=seed)
        with torch.no_grad(), indexed_gather(bridge, depth, paired_random.indices, layout) as run:
            random_output = adapter.encode(batch, train=False)
            random_receipt = run.validate_execution()
        if random_output.features.shape[1] != paired_random.effective_budget:
            raise RuntimeError("adapter feature timeline does not match paired-random suffix")
        return {
            "dense": {"metrics": _metrics(dense, dense), "shape": list(dense.features.shape)},
            "identity": {"metrics": _metrics(dense, identity_output), "receipt": identity_receipt},
            "pairmean": {"metrics": _metrics(dense, pair_output), "receipt": pair_receipt, "plan": spec.receipt(), "indices_sha256": "sha256:" + hashlib.sha256(spec.output_indices.detach().cpu().numpy().tobytes()).hexdigest()},
            "pairedrandom": {"metrics": _metrics(dense, random_output), "receipt": random_receipt, "indices_sha256": "sha256:" + hashlib.sha256(paired_random.indices.detach().cpu().numpy().tobytes()).hexdigest()},
            "pair_spec": {"pair_count": int(spec.pair_indices.shape[1]), "trajectory_length": spec.trajectory_length, "output_tokens": int(spec.output_indices.shape[1]), "encoder_id": encoder},
        }
    finally:
        for module, mode in modes:
            module.training = mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "projects/icassp2027/profile.yaml")
    parser.add_argument("--encoder", choices=("videomaev2", "timesformer", "videomae", "vjepa2"), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--plan", type=Path, default=neutral.ASSET_ROOT / "plan.json")
    parser.add_argument("--cases", type=Path, default=neutral.ASSET_ROOT / "cases.json")
    parser.add_argument("--manifest", type=Path, default=neutral.ASSET_ROOT / "manifest.jsonl")
    parser.add_argument("--role-lock", type=Path, default=neutral.ROLE_LOCK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    project = load_project(args.profile)
    dataset_root = (args.dataset_root or Path(neutral._load_json(args.plan)["dataset_root"])).resolve()
    planning = {"engineering_only_pair_merge": True, "offline_prefix_score_diagnostic": True, "no_actual_vad_quality": True, "no_timing_claims": True, "encoder": args.encoder, "expected_videos": 26, "expected_clips": 52, "execute": bool(args.execute)}
    if not args.execute:
        print(json.dumps(planning, ensure_ascii=False, indent=2))
        return 0
    plan, cases, records = neutral._validate_assets(plan_path=args.plan.resolve(), cases_path=args.cases.resolve(), manifest_path=args.manifest.resolve(), dataset_root=dataset_root, role_lock_path=args.role_lock.resolve(), role_lock_sha256=neutral.ROLE_LOCK_SHA256)
    definition = dict(project.encoder(args.encoder)["definition"])
    constructor = dict(definition["constructor"])
    frames = constructor.get("num_frames", constructor.get("clip_frames"))
    if type(frames) is not int or frames <= 0:
        raise ValueError("encoder constructor lacks native clip frames")
    output = (args.output or project.root / "outputs/icassp2027/pair-merge-pilot" / new_run_id(args.encoder)).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    constructor["device"] = args.device
    definition["constructor"] = constructor
    completed, failures = 0, []
    with record_stage(output, "pair_merge_pilot", config={**planning, "plan": plan}, inputs={"plan": args.plan, "cases": args.cases, "manifest": args.manifest, "role_lock": args.role_lock}, project_root=project.root):
        identity = encoder_identity(definition, project_root=project.root)
        adapter = ENCODER_REGISTRY.create(args.encoder, **constructor)
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        atomic_write_json(output / "resolved.json", {**planning, "identity": identity, "python": sys.executable, "vadbench": vadbench.__file__, "torch": torch.__version__, "transformers": __import__("transformers").__version__, "cudnn": {"enabled": torch.backends.cudnn.enabled, "benchmark": torch.backends.cudnn.benchmark, "deterministic": torch.backends.cudnn.deterministic}, "source_sha256": {"cli": sha256_file(Path(__file__)), "pair_merge": sha256_file(Path(__import__("vadbench.token_reduction.pair_merge", fromlist=["x"]).__file__)), "indexed": sha256_file(Path(__import__("vadbench.token_reduction.bridges.indexed", fromlist=["x"]).__file__))}, "actual_parameter": neutral._adapter_details(adapter, args.encoder)["parameter"]})
        for ordinal, case, window in neutral._clip_jobs(cases):
            record, shard = records[case["video_id"]], output / "shards" / f"{ordinal:03d}.json"
            try:
                with OpenCVVideoReader(dataset_root / record.path) as reader:
                    sample = sample_uniform_full_clips(reader.info.num_frames, num_segments=8, clip_frames=frames, frame_stride=2)[window]
                    batch = _batch_from_reader(reader, record.video_id, [sample.clip])
                pair = _pair_controls(adapter, args.encoder, batch, seed=20260918 + ordinal)
                atomic_write_json(shard, {"status": "completed", "ordinal": ordinal, "window_index": window, "frame_indices": list(sample.frame_indices), "analysis_case": {"video_id": record.video_id, "weak_label": case["weak_label"], "category": case["category_for_analysis_only"]}, "controls": pair})
                completed += 1
            except torch.cuda.OutOfMemoryError as error:
                failures.append({"ordinal": ordinal, "type": type(error).__name__, "message": str(error)})
                atomic_write_json(shard, {"status": "failed", **failures[-1]})
                neutral._write_progress(output, completed=completed, failures=failures)
                raise
            except Exception as error:
                failures.append({"ordinal": ordinal, "type": type(error).__name__, "message": str(error)})
                atomic_write_json(shard, {"status": "failed", **failures[-1]})
            neutral._write_progress(output, completed=completed, failures=failures)
        if failures or completed != neutral.EXPECTED_CLIPS:
            raise RuntimeError("pair merge pilot partial; no completed summary")
        atomic_write_json(output / "summary.json", {"status": "completed", "clips": completed, "offline_prefix_score_diagnostic": True, "no_actual_vad_quality": True, "no_timing_claims": True})
    print(json.dumps({"status": "completed", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
