"""Time one complete decoded-video frozen detector path; dry-run by default.

The formal execution reads one fixed fit-only video and no temporal labels. It
does not report AUC/AP, write a FeatureStore, or alter any source controller
run.  Each frozen reducer uses its own completed refit TopKMIL head.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # Keep --help/dry-run useful in catalog-only environments.
    torch = None  # type: ignore[assignment]

import vadbench
from vadbench.artifacts import new_run_id
from vadbench.checkpoints import sha256_file
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.manifest import DatasetSplit, load_manifest_jsonl
from vadbench.data.video import build_clip_batch, probe_video
from vadbench.engine.train import load_checkpoint
from vadbench.features import atomic_write_json
from vadbench.paper.evaluation import load_frozen_detector_source
from vadbench.workflows.extraction import representation_from_verified_encoder
from vadbench.paper.profile import load_project
from vadbench.workflows.video_efficiency import (
    VIDEO_EFFICIENCY_SCHEMA_VERSION,
    VideoTimingSettings,
    measure_full_video_detector,
    predictor_path_qa,
    run_full_video_detector,
)
from vadbench.registry import ENCODER_REGISTRY
from vadbench.tasks import build_task

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("dense", "same_budget_control", "training_free", "trainable")
REDUCERS = {"dense": "identity", "same_budget_control": "global_uniform", "training_free": "paired_random", "trainable": "pair_linear"}
CLIP_FRAMES = {"videomaev2": 16, "timesformer": 8, "vjepa2": 64, "videomae": 16}
FROZEN_FIT128_MANIFEST_SHA256 = "7dc3092e63294c65cc8896292cf4f1fea9d95cbaa3f7c5cb15153aa5435b64ac"
FROZEN_ROLE_LOCK_SHA256 = "53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59"
FIXED_VIDEO_ID = "Abuse005_x264"
FIXED_VIDEO_FRAMES = 949
FIXED_VIDEO_FPS = 30.0
DEFAULT_CALIBRATION_ROOT = Path(
    "/users/fotile/icassp2027-runs/code-5e95107/outputs/icassp2027/control/"
    "target-runtime-gate-20260918T120000Z/calibration"
)


def _nvidia_smi() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "command": command, "error": f"{type(error).__name__}: {error}"}
    return {"available": result.returncode == 0, "command": command, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def _json(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != set(METHODS) or any(not isinstance(item, str) or not item for item in value.values()):
        raise ValueError("--method-sources must be a JSON object with exactly dense/same_budget_control/training_free/trainable paths")
    return {str(name): str(location) for name, location in value.items()}


def _freeze(path: Path) -> tuple[dict[str, Any], str]:
    # The frozen-source loader intentionally takes a parsed freeze. Reuse its
    # companion validator rather than duplicating the complete method contract.
    from vadbench.paper.evaluation import _load_freeze

    return _load_freeze(path)


def _fit_video(manifest: Path, role_lock: Path):
    if sha256_file(manifest) != FROZEN_FIT128_MANIFEST_SHA256:
        raise ValueError("fit manifest SHA-256 differs from the frozen fit128 source")
    if sha256_file(role_lock) != FROZEN_ROLE_LOCK_SHA256:
        raise ValueError("fit role-lock SHA-256 differs from the frozen fit128 lock")
    records = load_manifest_jsonl(manifest)
    if len(records) != 128 or any(record.split != DatasetSplit.TRAIN for record in records):
        raise ValueError("full-video timing requires the exact 128-video TRAIN fit manifest")
    first = sorted(records, key=lambda record: record.video_id)[0]
    if first.video_id != FIXED_VIDEO_ID or first.num_frames != FIXED_VIDEO_FRAMES or first.fps != FIXED_VIDEO_FPS:
        raise ValueError("frozen fit128 lexical first video identity differs from Abuse005_x264/949/30fps")
    asset_sha256 = first.metadata.get("training_asset_sha256")
    if not isinstance(asset_sha256, str) or len(asset_sha256) != 64 or any(char not in "0123456789abcdef" for char in asset_sha256):
        raise ValueError("frozen fit video manifest lacks a lowercase training_asset_sha256")
    return first


def _adapter(encoder: str, *, project_path: Path, device: str):
    project = load_project(project_path)
    definition = dict(project.encoder(encoder)["definition"])
    constructor = dict(definition["constructor"])
    constructor["device"] = device
    if encoder in {"timesformer", "videomae"}:
        constructor["processor_tensor_type"] = "np"
    from vadbench.orchestration import encoder_identity

    definition["constructor"] = constructor
    definition["identity"] = encoder_identity(definition, project_root=project.root)
    adapter = ENCODER_REGISTRY.create(encoder, **constructor)
    if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
        adapter.bridge.load()
    return project, adapter, definition


def _actual_parameter(adapter: Any, encoder: str):
    from vadbench.token_reduction.bridges import create_observation_bridge

    model = create_observation_bridge(encoder, adapter).model
    model.eval().requires_grad_(False)
    parameter = next(model.parameters(), None)
    if parameter is None or parameter.dtype != torch.float32 or parameter.device.type != "cuda":
        raise RuntimeError("formal full-video detector requires actual bridge model FP32 on CUDA")
    return model, {"dtype": str(parameter.dtype), "device": str(parameter.device)}


def _head(source: Any, *, device: str):
    # Checkpoint metadata is runner._config_metadata(asdict(...)): its task is
    # a flat string. The predictor's adapter is the established compatibility
    # path for exactly that persisted representation.
    from vadbench.engine.predict import _settings

    settings = _settings(source.checkpoint_metadata["config"])
    task = build_task(settings.task, None, feature_dim=source.representation.output_dim, head=settings.head,
                      head_kwargs=settings.head_kwargs, task_kwargs=settings.task_kwargs)
    receipt = load_checkpoint(source.checkpoint, task, map_location="cpu", strict=True, verify=True)
    if receipt["missing_keys"] or receipt["unexpected_keys"]:
        raise RuntimeError("frozen head checkpoint did not load exactly")
    task.to(device).eval().requires_grad_(False)
    floating = [parameter for parameter in task.parameters() if parameter.is_floating_point()]
    if not floating or any(parameter.dtype != torch.float32 for parameter in floating):
        raise RuntimeError("formal full-video detector requires the loaded TopKMIL head in FP32")
    return task, {"checkpoint": str(source.checkpoint), "checkpoint_sha256": sha256_file(source.checkpoint),
                  "feature_dim": source.representation.output_dim, "config": source.checkpoint_metadata["config"]}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=tuple(CLIP_FRAMES), required=True)
    parser.add_argument("--method-sources", type=Path, help="JSON mapping of the four frozen methods to completed source controller runs")
    parser.add_argument("--fit-manifest", type=Path, help="exact frozen fit128 manifest")
    parser.add_argument("--fit-role-lock", type=Path, help="exact frozen fit128 role lock")
    parser.add_argument("--dataset-root", type=Path, help="root containing the fit video paths")
    parser.add_argument("--project", type=Path, default=ROOT / "projects/icassp2027/profile.yaml")
    parser.add_argument("--freeze-path", type=Path, default=ROOT / "projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json")
    parser.add_argument("--role-lock-path", type=Path, default=ROOT / "outputs/icassp2027/assets/gate-calibration-20260918T083000Z/frozen-partition-lock.json")
    parser.add_argument("--head-data-contract-path", type=Path, default=ROOT / "projects/icassp2027/decisions/head-data-contract-v1.json")
    parser.add_argument("--source-manifest-root", type=Path, default=ROOT / "outputs/icassp2027/assets/full-ucf-head-data-contract-20260918")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = {
        "schema_version": VIDEO_EFFICIENCY_SCHEMA_VERSION,
        "encoder": args.encoder,
        "methods": list(METHODS),
        "video": {"selection": "lexicographic_first_frozen_fit128", "video_id": FIXED_VIDEO_ID,
                  "num_frames": FIXED_VIDEO_FRAMES, "fps": FIXED_VIDEO_FPS},
        "sampling": {"clip_frames": CLIP_FRAMES[args.encoder], "frame_stride": 2,
                     "window_stride": CLIP_FRAMES[args.encoder], "short_policy": "stride1_if_needed"},
        "timing": {"warmup": 1, "repeat": 10, "batch_size": 8, "precision": "float32"},
        "fit_manifest_expected_sha256": FROZEN_FIT128_MANIFEST_SHA256,
        "fit_role_lock_expected_sha256": FROZEN_ROLE_LOCK_SHA256,
        "execute": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if torch is None or not str(args.device).lower().startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal full-video detector benchmark requires an available CUDA runtime")
    required = (args.method_sources, args.fit_manifest, args.fit_role_lock, args.dataset_root)
    if any(value is None for value in required):
        raise ValueError("--execute requires --method-sources, --fit-manifest, --fit-role-lock, and --dataset-root")
    torch.cuda.set_device(torch.device(args.device))
    output = args.output.resolve() if args.output else ROOT / "outputs/icassp2027/frozen-video-efficiency" / new_run_id(args.encoder)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full-video benchmark run: {output}")
    source_paths = _json(args.method_sources)
    freeze, freeze_sha = _freeze(args.freeze_path)
    record = _fit_video(args.fit_manifest, args.fit_role_lock)
    video_path = (args.dataset_root / record.path).resolve()
    expected_asset_sha256 = record.metadata["training_asset_sha256"]
    actual_asset_sha256 = sha256_file(video_path)
    if actual_asset_sha256 != expected_asset_sha256:
        raise ValueError("selected fit video bytes differ from frozen manifest training_asset_sha256")
    info = probe_video(video_path)
    if info.num_frames != FIXED_VIDEO_FRAMES or not np.isclose(info.fps, FIXED_VIDEO_FPS):
        raise ValueError("actual decoded fit video identity differs from frozen Abuse005_x264/949/30fps")
    output.mkdir(parents=True)
    try:
        project, adapter, definition = _adapter(args.encoder, project_path=args.project, device=args.device)
        verified = definition["identity"]
        model, model_receipt = _actual_parameter(adapter, args.encoder)
        sampling = DenseSamplingPlan(clip_frames=CLIP_FRAMES[args.encoder], frame_stride=2,
                                    window_stride=CLIP_FRAMES[args.encoder], short_policy="stride1_if_needed")
        samples = sampling.sample(info.num_frames)
        if not samples or samples[-1].clip_index != len(samples) - 1:
            raise RuntimeError("dense sampler did not emit a stable complete window sequence")
        setup_batch = build_clip_batch(video_path, record.video_id, [samples[0].clip])
        source_kwargs = {"role_lock_path": args.role_lock_path, "head_data_contract_path": args.head_data_contract_path,
                         "source_manifest_root": args.source_manifest_root}
        resolved = {**plan, "status": "running", "freeze_sha256": freeze_sha, "fit_manifest_sha256": sha256_file(args.fit_manifest),
                    "fit_role_lock_sha256": sha256_file(args.fit_role_lock), "method_sources": source_paths,
                    "video_path": str(video_path), "decoded_video": {"num_frames": info.num_frames, "fps": info.fps,
                    "width": info.width, "height": info.height, "training_asset_sha256": actual_asset_sha256}, "verified_encoder_identity": verified,
                    "actual_bridge_model": model_receipt, "python_executable": sys.executable, "vadbench_file": vadbench.__file__,
                    "torch": str(torch.__version__), "torch_cuda": torch.version.cuda, "cuda_current_device": torch.cuda.current_device(), "nvidia_smi_host_device_provenance": _nvidia_smi(),
                    "source_sha256": {"cli": sha256_file(Path(__file__)),
                    "video_efficiency": sha256_file(Path(__import__("vadbench.workflows.video_efficiency", fromlist=["x"]).__file__))}}
        atomic_write_json(output / "resolved.json", resolved)
        outcomes: dict[str, Any] = {}
        for method in METHODS:
            try:
                source = load_frozen_detector_source(source_paths[method], freeze=freeze, expected_encoder=args.encoder, **source_kwargs)
                if source.reducer.get("name") != REDUCERS[method] or source.training_identity.seed != 0:
                    raise ValueError("source controller does not provide the required frozen reducer/headseed0 method")
                source.verify_unchanged()
                context = None
                setup: dict[str, Any] = {"status": "identity_no_context"}
                if method != "dense":
                    from vadbench.paper.reduction_setup import prepare_reduction

                    calibration = args.calibration_root / args.encoder if method == "trainable" else None
                    reducer_seed = int(source.reducer["seed"]) if source.reducer["name"] == "paired_random" else 0
                    context, setup = prepare_reduction(adapter, args.encoder, setup_batch, reducer=source.reducer["name"],
                        output_dim=source.representation.output_dim, verified_encoder_identity=verified, calibration_run=calibration,
                        seed=reducer_seed, batch_sizes=range(1, 9))
                    if dict(context.reducer_identity) != dict(source.reducer):
                        raise ValueError("actual prepared reducer identity differs from frozen head representation")
                actual_representation = representation_from_verified_encoder(runtime_id=args.encoder, adapter=adapter,
                    verified_encoder_identity=verified, preprocessing=source.representation.backbone.preprocessing,
                    readout=source.representation.backbone.readout, reducer=source.reducer,
                    output_dim=source.representation.output_dim, precision=source.representation.precision,
                    position_strategy=source.representation.position_strategy)
                if actual_representation != source.representation:
                    raise ValueError("actual adapter/reducer representation differs from the frozen source head")
                task, head_receipt = _head(source, device=args.device)
                qa = predictor_path_qa(task, feature_dim=source.representation.output_dim, device=args.device)
                preflight = run_full_video_detector(adapter=adapter, detector_task=task, video_path=str(video_path),
                    video_id=record.video_id, samples=samples, num_frames=info.num_frames, fps=info.fps,
                    context_factory=context, batch_size=8)
                measured = measure_full_video_detector(lambda task=task, context=context: run_full_video_detector(adapter=adapter, detector_task=task,
                    video_path=str(video_path), video_id=record.video_id, samples=samples, num_frames=info.num_frames,
                    fps=info.fps, context_factory=context, batch_size=8), torch_module=torch, device=args.device,
                    settings=VideoTimingSettings())
                if any(item["video_receipt"]["token_lengths"] != preflight["token_lengths"] for item in measured["samples"]):
                    raise RuntimeError("formal repeats changed the actual token-length receipt")
                outcomes[method] = {"status": "completed", "source": {"root": str(source.root), "checkpoint": head_receipt,
                                    "reducer": dict(source.reducer), "head_seed": source.training_identity.seed,
                                    "training_role_lock": {"path": str(source.role_lock_path), "sha256": source.role_lock_sha256},
                                    "head_data_contract": {"path": str(source.head_data_contract_path), "sha256": source.head_data_contract_sha256},
                                    "source_manifests_sha256": dict(source.source_manifest_sha256)},
                                    "reduction_setup": setup, "predictor_path_qa": qa, "preflight": preflight, "measurement": measured}
            except BaseException as error:
                outcomes[method] = {"status": "failed", "measurement": None,
                                    "error": {"type": type(error).__name__, "message": str(error)}}
            atomic_write_json(output / "progress.json", {"status": "running", "completed_methods": list(outcomes), "outcomes": outcomes})
        result = {"schema_version": VIDEO_EFFICIENCY_SCHEMA_VERSION,
                  "status": "completed" if all(item["status"] == "completed" for item in outcomes.values()) else "failed",
                  "methods": outcomes, "no_auc_or_label_comparison": True}
        atomic_write_json(output / "result.json", result)
        atomic_write_json(output / "progress.json", {"status": result["status"], "completed_methods": list(outcomes)})
        print(json.dumps({"status": result["status"], "output": str(output)}, ensure_ascii=False))
        return 0 if result["status"] == "completed" else 2
    except BaseException as error:
        atomic_write_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                          "python_executable": sys.executable})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
