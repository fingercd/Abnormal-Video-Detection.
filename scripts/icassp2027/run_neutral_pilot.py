"""Run the frozen 26-video neutral intervention pilot only when explicitly requested."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

import vadbench
from vadbench.artifacts import new_run_id, record_stage
from vadbench.checkpoints import sha256_file
from vadbench.data.dense_sampling import sample_uniform_full_clips
from vadbench.data.manifest import SupervisionScope, VideoManifestRecord, load_manifest_jsonl
from vadbench.data.video import OpenCVVideoReader, _batch_from_reader
from vadbench.features import atomic_write_json
from vadbench.orchestration import encoder_identity
from vadbench.paper.profile import load_project
from vadbench.registry import ENCODER_REGISTRY
from vadbench.research.intervention_runner import run_intervention_diagnostic
from vadbench.token_reduction.bridges import create_observation_bridge

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "outputs/icassp2027/assets/neutral-pilot-20260917T220518Z"
ROLE_LOCK = ROOT / "outputs/icassp2027/assets/frozen192-verified-20260917T200000Z/frozen-partition-lock.json"
ROLE_LOCK_SHA256 = "53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59"
EXPECTED_VIDEOS = 26
EXPECTED_CLIPS = 52
WINDOW_INDICES = (1, 6)
CANDIDATES = ("relative_attention_update", "midlayer_temporal_change")
REQUIRED_CONTROLS = {
    "dense",
    "identity",
    "uniform",
    "seeded_random",
    "score_high",
    "score_low",
    "paired_first",
    "paired_random",
    "paired_high",
    "paired_low",
}


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _strict_float_equal(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and math.isclose(left, right, rel_tol=1e-6)


def _validate_role_lock(path: Path, cases: list[dict[str, Any]], *, expected_sha256: str) -> None:
    if sha256_file(path) != expected_sha256:
        raise ValueError("role lock SHA-256 differs from the original frozen data roles")
    lock = _load_json(path)
    if not isinstance(lock, dict) or lock.get("basis") != "complete_official_train_source_groups":
        raise ValueError("role lock must describe the complete official training source groups")
    partitions = lock.get("partitions")
    if not isinstance(partitions, dict) or any(partitions.get(case["video_id"]) != "fit" for case in cases):
        raise ValueError("every neutral pilot video must have fit role in the original lock")


def _validate_assets(
    *, plan_path: Path, cases_path: Path, manifest_path: Path, dataset_root: Path,
    role_lock_path: Path, role_lock_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, VideoManifestRecord]]:
    plan = _load_json(plan_path)
    cases = _load_json(cases_path)
    if not isinstance(plan, dict) or not isinstance(cases, list):
        raise ValueError("neutral pilot plan/cases must be JSON object/list")
    if sha256_file(cases_path) != plan.get("cases_sha256"):
        raise ValueError("cases SHA-256 does not match frozen plan")
    if sha256_file(manifest_path) != plan.get("source_manifest_sha256"):
        raise ValueError("manifest SHA-256 does not match frozen plan")
    if (
        plan.get("videos") != EXPECTED_VIDEOS
        or plan.get("clips_per_encoder") != EXPECTED_CLIPS
        or plan.get("windows_per_video_in_source_sampler") != 8
        or tuple(plan.get("selected_window_indices", ())) != WINDOW_INDICES
        or plan.get("relative_depth") != 0.5
        or plan.get("budget_ratio") != 0.5
    ):
        raise ValueError("frozen plan does not match the neutral 26-video/52-clip contract")
    if len(cases) != EXPECTED_VIDEOS:
        raise ValueError("frozen cases must contain exactly 26 videos")
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("each frozen case must be an object")
    records = load_manifest_jsonl(manifest_path)
    by_id = {record.video_id: record for record in records}
    if len(records) != EXPECTED_VIDEOS or len(by_id) != EXPECTED_VIDEOS or set(by_id) != {case.get("video_id") for case in cases}:
        raise ValueError("cases and manifest must contain the same exactly-26 video IDs")
    _validate_role_lock(role_lock_path, cases, expected_sha256=role_lock_sha256)
    positives = 0
    normals = 0
    categories: set[str] = set()
    for case in cases:
        video_id = case.get("video_id")
        record = by_id.get(video_id)
        if record is None or case.get("partition") != "fit" or record.split.value != "train":
            raise ValueError("neutral pilot permits only frozen fit/train records")
        if case.get("path") != record.path or tuple(case.get("window_indices", ())) != WINDOW_INDICES:
            raise ValueError("case path/window indices differ from frozen manifest contract")
        label = case.get("weak_label")
        if type(label) is not int or label not in (0, 1) or label != int(record.is_anomaly):
            raise ValueError("case weak label differs from frozen manifest")
        if case.get("category_for_analysis_only") != record.category:
            raise ValueError("case category differs from frozen manifest")
        if any(annotation.scope != SupervisionScope.VIDEO for annotation in record.annotations):
            raise ValueError("neutral pilot permits video-level labels only; temporal or caption annotations are forbidden")
        path = (dataset_root / record.path).resolve()
        if not path.is_relative_to(dataset_root.resolve()) or not path.is_file():
            raise FileNotFoundError(f"frozen case video missing: {record.path}")
        if path.stat().st_size != case.get("size_bytes") or sha256_file(path) != case.get("sha256"):
            raise ValueError(f"frozen case content identity mismatch: {record.video_id}")
        with OpenCVVideoReader(path) as reader:
            if reader.info.num_frames != record.num_frames or not _strict_float_equal(reader.info.fps, record.fps):
                raise ValueError(f"manifest video info mismatch: {record.video_id}")
        if record.is_anomaly:
            positives += 1
            categories.add(record.category)
        else:
            normals += 1
    if positives != 13 or normals != 13 or len(categories) != 13:
        raise ValueError("frozen cases must remain 13 category positives plus 13 normals")
    return plan, cases, by_id


def _source_hashes() -> dict[str, str]:
    from vadbench.data import dense_sampling, video
    from vadbench.research import intervention_runner, interventions
    from vadbench.token_reduction.bridges import indexed

    paths = {
        "run_neutral_pilot": Path(__file__),
        "intervention_runner": Path(intervention_runner.__file__),
        "interventions": Path(interventions.__file__),
        "indexed_bridge": Path(indexed.__file__),
        "dense_sampling": Path(dense_sampling.__file__),
        "video_reader": Path(video.__file__),
    }
    return {name: sha256_file(path.resolve()) for name, path in paths.items()}


def _adapter_details(adapter: Any, encoder: str) -> dict[str, Any]:
    bridge = create_observation_bridge(encoder, adapter)
    parameter = next(bridge.model.parameters(), None)
    cfg = getattr(getattr(adapter, "encoder", None), "cfg", None)
    pooling = getattr(adapter, "pooling", getattr(cfg, "pooling", None))
    return {
        "parameter": None if parameter is None else {"dtype": str(parameter.dtype), "device": str(parameter.device)},
        "readout": {
            "feature_stage": getattr(adapter, "feature_stage", getattr(adapter, "FEATURE_STAGE", None)),
            "pooling": pooling,
            "description": "actual adapter.encode pooled output; native mean substitution is not used",
        },
    }


def _write_progress(output: Path, *, completed: int, failures: list[dict[str, Any]]) -> None:
    atomic_write_json(
        output / "progress.json",
        {"expected_clips": EXPECTED_CLIPS, "completed_clips": completed, "failures": failures},
    )


def _validate_diagnostic(diagnostic: Any, *, encoder: str, candidate: str) -> None:
    if diagnostic.encoder_id != encoder or diagnostic.candidate != candidate:
        raise RuntimeError("runner receipt encoder/candidate differs from requested diagnostic")
    if not REQUIRED_CONTROLS.issubset(diagnostic.results):
        raise RuntimeError("runner receipt lacks a required neutral pilot control")
    controls = diagnostic.results
    dense_budget = controls["dense"].effective_budget
    budget = diagnostic.requested_budget
    if budget != math.ceil(dense_budget * 0.5) or controls["identity"].effective_budget != dense_budget:
        raise RuntimeError("runner dense/identity/requested budget differs from the fixed ratio")
    suffix_depths = set(controls["dense"].suffix_shapes)
    if not suffix_depths:
        raise RuntimeError("runner receipt has no native suffix execution")
    for name in REQUIRED_CONTROLS - {"dense"}:
        result = controls[name]
        expected_budget = dense_budget if name == "identity" else budget
        if result.effective_budget != expected_budget:
            raise RuntimeError("runner controls have unequal effective budgets")
        if result.gathered_shape is None or len(result.gathered_shape) != 3 or result.gathered_shape[1] != expected_budget:
            raise RuntimeError("runner gathered shape differs from its effective budget")
        if set(result.suffix_shapes) != suffix_depths or any(tuple(shape) != tuple(result.gathered_shape) for shape in result.suffix_shapes.values()):
            raise RuntimeError("runner native suffix does not preserve its gathered token shape")


def _clip_jobs(cases: list[dict[str, Any]]) -> tuple[tuple[int, dict[str, Any], int], ...]:
    if len(cases) != EXPECTED_VIDEOS:
        raise ValueError("neutral pilot production gate requires exactly 26 frozen cases")
    jobs = tuple(
        (ordinal, case, window_index)
        for ordinal, (case, window_index) in enumerate(
            item
            for frozen_case in cases
            for item in ((frozen_case, WINDOW_INDICES[0]), (frozen_case, WINDOW_INDICES[1]))
        )
    )
    if len(jobs) != EXPECTED_CLIPS:
        raise RuntimeError("neutral pilot job count differs from frozen 52-clip contract")
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "projects/icassp2027/profile.yaml")
    parser.add_argument("--encoder", required=True, choices=("videomaev2", "timesformer", "videomae", "vjepa2"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--plan", type=Path, default=ASSET_ROOT / "plan.json")
    parser.add_argument("--cases", type=Path, default=ASSET_ROOT / "cases.json")
    parser.add_argument("--manifest", type=Path, default=ASSET_ROOT / "manifest.jsonl")
    parser.add_argument("--role-lock", type=Path, default=ROLE_LOCK)
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
    dataset_root = (args.dataset_root or Path(_load_json(args.plan)["dataset_root"])).resolve()
    plan_path, cases_path, manifest_path = args.plan.resolve(), args.cases.resolve(), args.manifest.resolve()
    role_lock_path = args.role_lock.resolve()
    plan = _load_json(plan_path)
    if not isinstance(plan, dict):
        raise ValueError("plan must be a JSON object")
    selected = project.encoder(args.encoder)
    definition = dict(selected["definition"])
    constructor = dict(definition["constructor"])
    frames = constructor.get("num_frames", constructor.get("clip_frames"))
    if type(frames) is not int or frames <= 0:
        raise ValueError("encoder constructor must declare positive native clip frames")
    planning = {
        "engineering_only_neutral_pilot": True,
        "offline_prefix_score_diagnostic": True,
        "no_actual_vad_quality": True,
        "no_timing_claims": True,
        "encoder": args.encoder,
        "dataset_root": str(dataset_root),
        "plan": str(plan_path),
        "cases": str(cases_path),
        "manifest": str(manifest_path),
        "role_lock": str(role_lock_path),
        "expected_videos": EXPECTED_VIDEOS,
        "expected_clips": EXPECTED_CLIPS,
        "windows": list(WINDOW_INDICES),
        "native_clip_frames": frames,
        "frame_stride": 2,
        "device": args.device,
        "threads": args.threads,
        "execute": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(planning, ensure_ascii=False, indent=2))
        return 0
    plan, cases, records = _validate_assets(
        plan_path=plan_path, cases_path=cases_path, manifest_path=manifest_path, dataset_root=dataset_root,
        role_lock_path=role_lock_path, role_lock_sha256=ROLE_LOCK_SHA256,
    )
    data_fingerprints = {
        "plan_sha256": sha256_file(plan_path),
        "cases_sha256": sha256_file(cases_path),
        "manifest_sha256": sha256_file(manifest_path),
        "role_lock_sha256": sha256_file(role_lock_path),
        "videos": [
            {
                "video_id": case["video_id"],
                "path": case["path"],
                "size_bytes": case["size_bytes"],
                "sha256": case["sha256"],
            }
            for case in cases
        ],
    }
    output = (args.output or project.root / "outputs/icassp2027/neutral-pilot" / new_run_id(args.encoder)).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    constructor["device"] = args.device
    definition["constructor"] = constructor
    completed = 0
    failures: list[dict[str, Any]] = []
    config = {**planning, "frozen_plan": plan, "source_sha256": _source_hashes()}
    with record_stage(
        output,
        "neutral_intervention_pilot",
        config=config,
        inputs={"plan": plan_path, "cases": cases_path, "manifest": manifest_path, "role_lock": role_lock_path},
        project_root=project.root,
    ):
        identity = encoder_identity(definition, project_root=project.root)
        adapter = ENCODER_REGISTRY.create(args.encoder, **constructor)
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        atomic_write_json(
            output / "resolved.json",
            {
                **config,
                "python_executable": sys.executable,
                "vadbench_file": vadbench.__file__,
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cudnn": {"enabled": torch.backends.cudnn.enabled, "benchmark": torch.backends.cudnn.benchmark, "deterministic": torch.backends.cudnn.deterministic},
                "encoder_identity": identity,
                "adapter": _adapter_details(adapter, args.encoder),
                "data_fingerprints": data_fingerprints,
            },
        )
        for ordinal, case, window_index in _clip_jobs(cases):
            record = records[case["video_id"]]
            shard = output / "shards" / f"{ordinal:03d}-{record.video_id}-window{window_index}.json"
            try:
                path = (dataset_root / record.path).resolve()
                with OpenCVVideoReader(path) as reader:
                    samples = sample_uniform_full_clips(
                        reader.info.num_frames, num_segments=8, clip_frames=frames, frame_stride=2
                    )
                    sample = samples[window_index]
                    batch = _batch_from_reader(reader, record.video_id, [sample.clip])
                seed = 20260918 + ordinal
                results = {}
                for candidate in CANDIDATES:
                    diagnostic = run_intervention_diagnostic(
                        adapter=adapter,
                        encoder_id=args.encoder,
                        batch=batch,
                        candidate=candidate,
                        relative_depth=0.5,
                        budget_ratio=0.5,
                        seed=seed,
                        include_paired=True,
                    )
                    _validate_diagnostic(diagnostic, encoder=args.encoder, candidate=candidate)
                    results[candidate] = dataclasses.asdict(diagnostic)
                atomic_write_json(
                    shard,
                    {
                        "status": "completed",
                        "ordinal": ordinal,
                        "analysis_case": {"video_id": record.video_id, "weak_label": case["weak_label"], "category": case["category_for_analysis_only"]},
                        "window_index": window_index,
                        "seed": seed,
                        "frame_indices": list(sample.frame_indices),
                        "results": results,
                    },
                )
                completed += 1
            except torch.cuda.OutOfMemoryError as error:
                failure = {"ordinal": ordinal, "video_id": record.video_id, "window_index": window_index, "type": type(error).__name__, "message": str(error)}
                failures.append(failure)
                atomic_write_json(shard, {"status": "failed", **failure})
                _write_progress(output, completed=completed, failures=failures)
                raise
            except Exception as error:
                failure = {"ordinal": ordinal, "video_id": record.video_id, "window_index": window_index, "type": type(error).__name__, "message": str(error)}
                failures.append(failure)
                atomic_write_json(shard, {"status": "failed", **failure})
            _write_progress(output, completed=completed, failures=failures)
        if failures or completed != EXPECTED_CLIPS:
            raise RuntimeError("neutral pilot is partial; no completed summary was written")
        atomic_write_json(output / "summary.json", {"status": "completed", "clips": completed, "offline_prefix_score_diagnostic": True, "no_actual_vad_quality": True, "no_timing_claims": True})
    print(json.dumps({"status": "completed", "output": str(output), "clips": completed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
