"""Dry-run by default; calibrate a provisional pair gate on the frozen fit128 plan."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from vadbench.artifacts import new_run_id
from vadbench.checkpoints import sha256_file
from vadbench.data.dense_sampling import sample_uniform_full_clips
from vadbench.data.video import build_clip_batch, probe_video
from vadbench.features import atomic_write_json
from vadbench.orchestration import encoder_identity
from vadbench.paper.profile import load_project
from vadbench.registry import ENCODER_REGISTRY
from vadbench.token_reduction import pair_merge, training
from vadbench.token_reduction.bridges import indexed
from vadbench.token_reduction.training import (
    ENCODERS,
    CalibrationSample,
    train_pair_gate,
    validate_calibration_assets,
)

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "cases", "manifest", "role-lock", "dataset-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "projects/icassp2027/profile.yaml")
    parser.add_argument("--processor-tensor-type", choices=("pt", "np"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    device = torch.device(args.device)
    if args.cuda_memory_fraction is not None:
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError("--cuda-memory-fraction must be in (0, 1]")
        if device.type != "cuda":
            raise ValueError("--cuda-memory-fraction requires a CUDA device")
    output = args.output.resolve() if args.output else ROOT / "outputs/icassp2027/pair-gate-training" / new_run_id(args.encoder)
    if args.execute and output.exists():
        raise FileExistsError(f"training refuses to overwrite a run: {output}")
    try:
        plan, samples, records = validate_calibration_assets(
            plan_path=args.plan, cases_path=args.cases, manifest_path=args.manifest,
            role_lock_path=args.role_lock, dataset_root=args.dataset_root, verify_video_bytes=args.execute,
        )
        project = load_project(args.profile)
        definition = dict(project.encoder(args.encoder)["definition"])
        constructor = dict(definition["constructor"])
        constructor["device"] = args.device
        if args.processor_tensor_type is not None:
            if args.encoder not in {"timesformer", "videomae"}:
                raise ValueError("--processor-tensor-type is only supported by TimeSformer/VideoMAE")
            constructor["processor_tensor_type"] = args.processor_tensor_type
        definition["constructor"] = constructor
        frames = constructor.get("num_frames", constructor.get("clip_frames"))
        if type(frames) is not int or frames <= 0:
            raise ValueError("active encoder must declare its native clip frame count")
        resolved = {"status": "planned", "encoder": args.encoder, "plan": plan,
                    "plan_sha256": sha256_file(args.plan), "profile": str(project.path),
                    "profile_sha256": sha256_file(project.path), "constructor": constructor,
                    "clip_frames": frames, "sample_count": len(samples), "expected_steps": 768,
                    "requested_device": args.device, "threads": args.threads,
                    "cuda_memory_fraction": args.cuda_memory_fraction,
                    "dataset_root": str(args.dataset_root.resolve()), "output": str(output),
                    "execute": args.execute, "video_bytes_verified": bool(args.execute)}
        if not args.execute:
            print(json.dumps(resolved, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        if args.cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
        identity = encoder_identity(definition, project_root=project.root)
        adapter = ENCODER_REGISTRY.create(args.encoder, **constructor)
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        if args.processor_tensor_type is not None and adapter.processor_tensor_type != args.processor_tensor_type:
            raise RuntimeError("loaded adapter does not implement the requested processor tensor constructor")
        verified_videos: set[str] = set()

        def load_batch(sample: CalibrationSample):
            record = records[sample.video_id]
            path = args.dataset_root / record.path
            if sample.video_id not in verified_videos:
                info = probe_video(path)
                if info.num_frames != record.num_frames or not math.isclose(info.fps, record.fps, rel_tol=1e-6):
                    raise ValueError(f"actual video frame/fps identity mismatch: {record.video_id}")
                verified_videos.add(sample.video_id)
            windows = sample_uniform_full_clips(
                record.num_frames, num_segments=8, clip_frames=frames, frame_stride=2,
                short_policy="stride1_if_needed",
            )
            return build_clip_batch(path, record.video_id, [windows[sample.window_index].clip])

        device_info = {"requested": args.device, "cuda_available": torch.cuda.is_available()}
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            device_info.update(name=properties.name, total_memory=properties.total_memory,
                               uuid=str(getattr(properties, "uuid", "unavailable")))
        receipt = train_pair_gate(
            adapter, args.encoder, samples, load_batch, output_dir=output,
            identity={**resolved, "encoder_identity": identity, "device": device_info,
                      "processor_tensor_type_verified": getattr(adapter, "processor_tensor_type", None),
                      "source_sha256": {name: sha256_file(Path(module.__file__)) for name, module in
                                        (("training", training), ("pair_merge", pair_merge), ("indexed", indexed))},
                      "cli_sha256": sha256_file(Path(__file__))},
        )
        print(json.dumps({"status": receipt["status"], "output": str(output), "steps": receipt["steps"],
                          "gate_state_sha256": receipt["gate_state_sha256"]}, ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        if args.execute and not (output / "receipt.json").exists():
            output.mkdir(parents=True, exist_ok=True)
            failure = {"status": "failed", "encoder": args.encoder, "error_type": type(error).__name__,
                       "error": str(error), "python_executable": sys.executable, "completed_steps": 0}
            atomic_write_json(output / "receipt.json", failure)
            atomic_write_json(output / "progress.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
