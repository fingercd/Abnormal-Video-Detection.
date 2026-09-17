"""Plan or run one active-encoder offline intervention diagnostic.

Without ``--execute`` this only checks the selected profile entry, checkpoint,
video, and deterministic centre-clip request.  Execution is engineering-only:
one clip, two offline prefix-score candidates, no quality, speed, fit, confirm,
or selector claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

import vadbench
from vadbench.artifacts import new_run_id
from vadbench.checkpoints import sha256_file
from vadbench.data.sampling import sample_fixed_clip
from vadbench.data.video import build_clip_batch, probe_video
from vadbench.orchestration import encoder_identity
from vadbench.paper.profile import load_project
from vadbench.registry import ENCODER_REGISTRY
from vadbench.research.intervention_runner import run_intervention_diagnostic
from vadbench.token_reduction.bridges import create_observation_bridge

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = ROOT / "data/ucf-debug-mirror/Abuse/Abuse005_x264.mp4"
ENCODERS = ("videomaev2", "timesformer", "videomae", "vjepa2")
CANDIDATES = ("relative_attention_update", "midlayer_temporal_change")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _source_hashes() -> dict[str, str]:
    from vadbench.research import intervention_runner, interventions
    from vadbench.token_reduction.bridges import indexed

    files = {
        "verify_intervention_active": Path(__file__),
        "intervention_runner": Path(intervention_runner.__file__),
        "interventions": Path(interventions.__file__),
        "indexed_bridge": Path(indexed.__file__),
    }
    return {name: sha256_file(path.resolve()) for name, path in files.items()}


def _adapter_readout(adapter: Any) -> dict[str, Any]:
    encoder = getattr(adapter, "encoder", None)
    cfg = getattr(encoder, "cfg", None)
    pooling = getattr(adapter, "pooling", None)
    if pooling is None:
        pooling = getattr(cfg, "pooling", None)
    return {
        "adapter_type": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "feature_stage": getattr(adapter, "feature_stage", getattr(adapter, "FEATURE_STAGE", None)),
        "pooling": pooling,
        "description": "adapter.encode pooled output; runner rejects a reduced suffix whose adapter feature timeline does not match gathered token count",
    }


def _parameter_details(adapter: Any, encoder: str) -> dict[str, Any] | None:
    bridge = create_observation_bridge(encoder, adapter)
    parameter = next(bridge.model.parameters(), None)
    if parameter is None:
        return None
    return {"dtype": str(parameter.dtype), "device": str(parameter.device)}


def _output_dir(args: argparse.Namespace, project: Any) -> Path:
    if args.output is not None:
        return args.output.resolve()
    return (project.root / "outputs/icassp2027/intervention-active" / new_run_id("offline")).resolve()


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "projects/icassp2027/profile.yaml")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--candidate", choices=CANDIDATES, action="append")
    parser.add_argument("--relative-depth", type=float, default=0.5)
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if not 0 < args.relative_depth <= 1:
        raise ValueError("--relative-depth must be in (0, 1]")
    if not 0 < args.budget_ratio <= 1:
        raise ValueError("--budget-ratio must be in (0, 1]")
    requested_device = torch.device(args.device)
    if args.cuda_memory_fraction is not None:
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError("--cuda-memory-fraction must be in (0, 1]")
        if requested_device.type != "cuda":
            raise ValueError("--cuda-memory-fraction requires a CUDA device")
    project = load_project(args.profile)
    selected = project.encoder(args.encoder)
    definition = dict(selected["definition"])
    constructor = dict(definition["constructor"])
    definition["constructor"] = constructor
    checkpoint = definition["checkpoint"]
    checkpoint_path = Path(checkpoint["local_path"])
    video = args.video.resolve()
    frames = constructor.get("num_frames", constructor.get("clip_frames"))
    if type(frames) is not int or frames <= 0:
        raise ValueError("active encoder constructor must declare positive num_frames or clip_frames")
    candidates = tuple(args.candidate or CANDIDATES)
    plan = {
        "status": "planned" if not args.execute else "running",
        "engineering_only_one_clip": True,
        "offline_prefix_score_diagnostic": True,
        "nonquality": True,
        "nonspeed": True,
        "encoder": args.encoder,
        "profile": str(project.path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "video": str(video),
        "video_exists": video.is_file(),
        "clip_frames": frames,
        "frame_stride": 2,
        "position": "center",
        "candidates": list(candidates),
        "relative_depth": args.relative_depth,
        "budget_ratio": args.budget_ratio,
        "seed": args.seed,
        "requested_device": str(requested_device),
        "requested_threads": args.threads,
        "requested_cuda_memory_fraction": args.cuda_memory_fraction,
        "execution": bool(args.execute),
    }
    if not args.execute:
        print(_json(plan))
        return 0
    if not checkpoint_path.exists() or not video.is_file():
        raise FileNotFoundError(_json(plan))
    output_dir = _output_dir(args, project)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    receipt_path = output_dir / "receipt.json"
    if args.cuda_memory_fraction is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, requested_device)
    torch.set_num_threads(args.threads)
    info = probe_video(video)
    sample = sample_fixed_clip(info.num_frames, clip_frames=frames, frame_stride=2, position="center")
    batch = build_clip_batch(video, "intervention-active", [sample])
    constructor["device"] = str(requested_device)
    try:
        identity = encoder_identity(definition, project_root=project.root)
        adapter = ENCODER_REGISTRY.create(args.encoder, **constructor)
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        results = {
            candidate: dataclasses.asdict(
                run_intervention_diagnostic(
                    adapter=adapter,
                    encoder_id=args.encoder,
                    batch=batch,
                    candidate=candidate,
                    relative_depth=args.relative_depth,
                    budget_ratio=args.budget_ratio,
                    seed=args.seed,
                )
            )
            for candidate in candidates
        }
    except Exception as error:
        _write(
            receipt_path,
            {
                **plan,
                "status": "failed",
                "failure": "cuda_oom" if isinstance(error, torch.cuda.OutOfMemoryError) else type(error).__name__,
                "detail": str(error),
                "output_dir": str(output_dir),
            },
        )
        raise
    receipt = {
        **plan,
        "status": "completed",
        "output_dir": str(output_dir),
        "python_executable": sys.executable,
        "vadbench_file": vadbench.__file__,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "torch_threads": torch.get_num_threads(),
        "verified_encoder_identity": identity,
        "source_sha256": _source_hashes(),
        "video_sha256": sha256_file(video),
        "video_info": {
            "num_frames": info.num_frames,
            "fps": info.fps,
            "height": info.height,
            "width": info.width,
        },
        "sampled_frame_indices": list(sample.frame_indices),
        "sampled_valid_mask": list(sample.valid_mask),
        "input_shape": list(batch.frames.shape),
        "actual_parameter": _parameter_details(adapter, args.encoder),
        "adapter_readout": _adapter_readout(adapter),
        "results": results,
    }
    _write(receipt_path, receipt)
    print(_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
