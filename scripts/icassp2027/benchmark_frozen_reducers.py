"""Formal net timing of the four frozen reducers on one fit-only clip group.

The command is deliberately dry-run by default.  ``--execute`` requires CUDA
and an already-completed fit128 calibration for ``pair_linear``.  It reads no
test manifest, frame annotation, score, detector head, or FeatureStore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Keep --help and the dry-run receipt usable in a catalog-only Python.
    import torch
except ImportError:  # pragma: no cover - exercised by minimal CLI environments
    torch = None  # type: ignore[assignment]

import vadbench
from vadbench.artifacts import new_run_id
from vadbench.checkpoints import sha256_file
from vadbench.data.dense_sampling import sample_uniform_full_clips
from vadbench.data.manifest import DatasetSplit, VideoManifestRecord, load_manifest_jsonl
from vadbench.data.video import build_clip_batch, probe_video
from vadbench.features import atomic_write_json
from vadbench.integrations.common import select_feature_tensor
from vadbench.orchestration import encoder_identity
from vadbench.paper.efficiency import (
    EFFICIENCY_SCHEMA_VERSION,
    CudaTimingRuntime,
    FrozenTimingSettings,
    failed_scope,
    measure_scope,
)
from vadbench.paper.profile import load_project
from vadbench.paper.stages import clean_encoder_batch
from vadbench.registry import ENCODER_REGISTRY
from vadbench.token_reduction.bridges import create_observation_bridge

ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ENCODERS = ("videomaev2", "timesformer", "vjepa2", "videomae")
OUTPUT_DIMENSIONS = {"videomaev2": 768, "timesformer": 768, "vjepa2": 1024, "videomae": 768}
CLIP_FRAMES = {"videomaev2": 16, "timesformer": 8, "vjepa2": 64, "videomae": 16}
FROZEN_FIT128_MANIFEST_SHA256 = "7dc3092e63294c65cc8896292cf4f1fea9d95cbaa3f7c5cb15153aa5435b64ac"
FROZEN_ROLE_LOCK_SHA256 = "53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59"
DEFAULT_CALIBRATION_ROOT = Path(
    "/users/fotile/icassp2027-runs/code-5e95107/outputs/icassp2027/control/"
    "target-runtime-gate-20260918T120000Z/calibration"
)


@dataclass(frozen=True, slots=True)
class NativeRoute:
    """The model's real forward plus its adapter's existing pooled readout."""

    prepare: Callable[[Any], Any]
    run: Callable[[Any, Any], torch.Tensor]
    description: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_receipt(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, Mapping):
        return {str(key): _tensor_receipt(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tensor_receipt(item) for item in value]
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _native_route(adapter: Any, encoder: str, native_model: torch.nn.Module) -> NativeRoute:
    if encoder == "videomaev2":
        worker = adapter.encoder
        backbone = getattr(native_model, "backbone", None)
        tensor_from_lists = getattr(worker, "_tensor_from_rgb_lists", None)
        pool = getattr(worker, "_pool", None)
        if worker is not native_model or backbone is None or not callable(tensor_from_lists) or not callable(pool):
            raise RuntimeError("VideoMAEv2 adapter lacks its verified native tensor/backbone/readout route")

        def prepare(batch: Any) -> torch.Tensor:
            clips = [
                [np.asarray(frame, dtype=np.uint8) for frame in np.asarray(batch.frames)[row, : int(length)]]
                for row, length in enumerate(batch.valid_lengths)
            ]
            parameter = next(backbone.parameters())
            return tensor_from_lists(clips).float().to(parameter.device)

        def run(inputs: torch.Tensor, _batch: Any) -> torch.Tensor:
            # The pinned native backbone already pools and returns [B,D].
            # Internal tokens are verified by the setup adapter/bridge; no
            # observation hook belongs in this pure native timing route.
            return pool(backbone(pixel_values=inputs.float()))

        return NativeRoute(prepare, run, "VideoMAEv2Encoder.backbone(pixel_values)+VideoMAEv2Encoder._pool")

    if encoder in {"timesformer", "videomae"}:
        if not callable(getattr(adapter, "_prepare_inputs", None)) or getattr(adapter, "model", None) is not native_model:
            raise RuntimeError("Transformers adapter lacks its verified native forward/readout route")
        if getattr(adapter, "pooling", None) != "mean" or getattr(adapter, "feature_stage", None) != "last_hidden_state":
            raise RuntimeError("formal pure_model requires the frozen Transformers mean last_hidden_state readout")

        def prepare(batch: Any) -> tuple[Mapping[str, Any], np.ndarray]:
            return adapter._prepare_inputs(batch)

        def run(inputs: tuple[Mapping[str, Any], np.ndarray], batch: Any) -> torch.Tensor:
            raw_inputs, lengths = inputs
            del lengths  # Input conversion has completed; it is not model work.
            raw = native_model(**raw_inputs, return_dict=True)
            tokens = raw.get("last_hidden_state") if isinstance(raw, Mapping) else getattr(raw, "last_hidden_state", None)
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] != batch.batch_size:
                raise RuntimeError("frozen Transformers native forward did not expose [B,N,D] last_hidden_state")
            # Both frozen Transformer configurations use mean pooling.  This
            # deliberately bypasses adapter timeline/metadata construction so
            # pure_model reports only native forward plus the same pooled rule.
            return tokens.mean(dim=1)

        return NativeRoute(prepare, run, "native transformers forward+frozen mean(last_hidden_state) pooled readout")

    if encoder == "vjepa2":
        worker = getattr(getattr(adapter, "bridge", None), "encoder", None)
        if worker is None or getattr(worker, "model", None) is not native_model or not callable(getattr(worker, "_prepare_inputs", None)):
            raise RuntimeError("V-JEPA2 adapter lacks its verified native get_vision_features route")

        def prepare(batch: Any) -> Any:
            return worker._prepare_inputs(batch)

        def run(inputs: Any, batch: Any) -> torch.Tensor:
            if not isinstance(inputs, Mapping):
                raise RuntimeError("V-JEPA2 native processor did not return a keyword mapping")
            raw = native_model.get_vision_features(**inputs)
            tokens, _source = select_feature_tensor(raw, batch_size=batch.batch_size)
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise RuntimeError("V-JEPA2 native vision route did not return [B,N,D] tokens")
            weights = tokens.new_ones((tokens.shape[0], tokens.shape[1], 1))
            return (tokens * weights).sum(dim=1) / weights.sum(dim=1)

        return NativeRoute(prepare, run, "V-JEPA2 _get_vision_features+canonical token pooling")
    raise ValueError(f"unsupported active encoder: {encoder}")


def _freeze_eval(native_model: torch.nn.Module) -> None:
    """Freeze exactly the model selected by the active bridge."""

    native_model.eval().requires_grad_(False)


def _model_details(native_model: torch.nn.Module) -> dict[str, Any]:
    parameter = next(native_model.parameters(), None)
    if parameter is None:
        raise RuntimeError("native bridge model has no parameters for dtype/device provenance")
    if any(parameter.dtype != torch.float32 for parameter in native_model.parameters() if parameter.is_floating_point()):
        raise RuntimeError("formal frozen-reducer timing requires float32 model parameters")
    return {"parameter_dtype": str(parameter.dtype), "parameter_device": str(parameter.device)}


def _batch_receipt(batch: Any) -> dict[str, Any]:
    frames = np.asarray(batch.frames)
    indices = np.asarray(batch.frame_indices)
    return {
        "frames_shape": list(frames.shape),
        "frames_dtype": str(frames.dtype),
        "frames_sha256": _sha256_bytes(frames.tobytes()),
        "frame_indices_shape": list(indices.shape),
        "frame_indices_sha256": _sha256_bytes(indices.tobytes()),
        "frame_indices": indices.tolist(),
        "valid_mask": np.asarray(batch.valid_mask).tolist(),
    }


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "available": False, "error": f"{type(error).__name__}: {error}"}
    return {"command": command, "available": result.returncode == 0, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def _method_specs(calibration_root: Path) -> tuple[tuple[str, str, Path | None], ...]:
    return (
        ("dense", "identity", None),
        ("global_uniform", "global_uniform", None),
        ("paired_random_seed0", "paired_random", None),
        ("pair_linear_gatefit128", "pair_linear", calibration_root),
    )


def _assert_pooled_parity(route: NativeRoute, adapter: Any, batch: Any, deployment: Any | None) -> dict[str, Any]:
    """Setup-only native/adapter parity for exactly one method and batch size."""

    prepared = route.prepare(batch)
    with torch.inference_mode():
        if deployment is None:
            pooled = route.run(prepared, batch)
            adapter_output = adapter.encode(batch, train=False)
            native_execution, adapter_execution = None, None
        else:
            with deployment(batch) as native_context:
                pooled = route.run(prepared, batch)
                native_execution = native_context.validate_execution()
            with deployment(batch) as adapter_context:
                adapter_output = adapter.encode(batch, train=False)
                adapter_execution = adapter_context.validate_execution()
        adapter_pooled = adapter_output.pooled
    torch.testing.assert_close(pooled, adapter_pooled, rtol=1e-5, atol=1e-6)
    if any(value.dtype != torch.float32 for value in (pooled, adapter_output.features, adapter_pooled)):
        raise RuntimeError("formal frozen-reducer timing requires float32 native and adapter outputs")
    receipt = {
        "native_pooled_shape": list(pooled.shape),
        "adapter_pooled_shape": list(adapter_pooled.shape),
        "max_absolute_error": float((pooled.float() - adapter_pooled.float()).abs().max().cpu()),
        "rtol": 1e-5,
        "atol": 1e-6,
        "prepared_inputs": _tensor_receipt(prepared),
        "native_reduction_execution": native_execution,
        "adapter_reduction_execution": adapter_execution,
    }
    # Setup evidence is a CPU-only receipt.  Do not let either parity branch
    # become an accidental resident allocation in a formal peak measurement.
    del pooled, adapter_pooled, adapter_output, prepared
    return receipt


def _token_preflight(adapter: Any, batch: Any, deployment: Any | None) -> dict[str, Any]:
    with torch.inference_mode():
        if deployment is None:
            output = adapter.encode(batch, train=False)
            execution = None
        else:
            with deployment(batch) as context:
                output = adapter.encode(batch, train=False)
                execution = context.validate_execution()
    receipt = {"actual_feature_tokens": int(output.features.shape[1]), "feature_dtype": str(output.features.dtype),
               "pooled_dtype": str(output.pooled.dtype), "reduction_execution": execution}
    del output
    return receipt


def _operation(
    *, scope: str, adapter: Any, route: NativeRoute, batch: Any, deployment: Any | None,
    prepared: Any | None, source_path: Path, video_id: str, samples: Sequence[Any], verified_tokens: int,
) -> Callable[[], Mapping[str, Any]]:
    def run_with(batch_value: Any, *, raw_inputs: Any | None, clean_inside_boundary: bool) -> Mapping[str, Any]:
        clean = clean_encoder_batch(batch_value) if clean_inside_boundary else batch_value
        with torch.inference_mode():
            if deployment is None:
                if scope == "pure_model":
                    assert raw_inputs is not None
                    pooled = route.run(raw_inputs, clean)
                    execution = None
                else:
                    output = adapter.encode(clean, train=False)
                    tokens, pooled, execution = output.features, output.pooled, None
            else:
                with deployment(clean) as context:
                    if scope == "pure_model":
                        assert raw_inputs is not None
                        pooled = route.run(raw_inputs, clean)
                    else:
                        output = adapter.encode(clean, train=False)
                        tokens, pooled = output.features, output.pooled
                    execution = context.validate_execution()
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise RuntimeError("native/adapter timing route did not produce [B,D] pooled readout")
        if scope != "pure_model" and (not isinstance(tokens, torch.Tensor) or tokens.ndim != 3):
            raise RuntimeError("adapter timing route did not expose [B,N,D] features")
        receipt = {"verified_feature_tokens": verified_tokens,
                   "token_verification": "setup adapter/bridge; reduced repeats also validate every actual suffix",
                   "adapter_feature_tokens": None if scope == "pure_model" else int(tokens.shape[1]),
                   "pooled_shape": list(pooled.shape), "pooled_dtype": str(pooled.dtype),
                   "reduction_execution": execution}
        # The receipt is scalar/JSON data.  Dropping raw outputs before return
        # avoids retaining one repeat while the next repeat is timed.
        if scope != "pure_model":
            del output, tokens
        del pooled
        return receipt

    if scope == "pure_model":
        if prepared is None:
            raise ValueError("pure_model scope requires preprocessed GPU inputs")
        return lambda: run_with(batch, raw_inputs=prepared, clean_inside_boundary=False)
    if scope == "adapter":
        return lambda: run_with(batch, raw_inputs=None, clean_inside_boundary=False)
    if scope == "end_to_end":
        def end_to_end() -> Mapping[str, Any]:
            rebuilt = build_clip_batch(source_path, video_id, samples)
            return run_with(rebuilt, raw_inputs=None, clean_inside_boundary=True)
        return end_to_end
    raise ValueError(scope)


def _choose_fit_video(records: Sequence[VideoManifestRecord], requested: str | None) -> VideoManifestRecord:
    if not records or any(record.split != DatasetSplit.TRAIN for record in records):
        raise ValueError("formal timing accepts only the frozen fit128 TRAIN manifest")
    if requested is None:
        raise ValueError("--execute requires an explicit --fit-video-id from the frozen fit128 manifest")
    by_id = {record.video_id: record for record in records}
    if requested not in by_id:
        raise ValueError("--fit-video-id is absent from the frozen fit128 manifest")
    return by_id[requested]


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=ACTIVE_ENCODERS, required=True)
    parser.add_argument("--profile", type=Path, default=ROOT / "projects/icassp2027/profile.yaml")
    parser.add_argument("--dataset-root", type=Path, help="root for the fit-only manifest paths")
    parser.add_argument("--fit-manifest", type=Path, help="exact frozen fit128 TRAIN manifest")
    parser.add_argument("--role-lock", type=Path, help="exact fit128 role-lock paired with --fit-manifest")
    parser.add_argument("--fit-video-id", help="explicit preselected video ID from the frozen fit128 manifest")
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    plan = {
        "schema_version": EFFICIENCY_SCHEMA_VERSION,
        "encoder": args.encoder,
        "methods": [name for name, _reducer, _calibration in _method_specs(args.calibration_root / args.encoder)],
        "batch_sizes": [1, 8],
        "scopes": ["pure_model", "adapter", "end_to_end"],
        "timing": {"warmup": 5, "repeat": 30, "primary_duration": "synchronized_wall_clock",
                   "cuda_event_duration": "diagnostic"},
        "frozen_precision": "float32",
        "processor_tensor_type": "np" if args.encoder in {"timesformer", "videomae"} else None,
        "expected_output_dim": OUTPUT_DIMENSIONS[args.encoder],
        "native_clip_frames": CLIP_FRAMES[args.encoder],
        "fit_manifest": None if args.fit_manifest is None else str(args.fit_manifest),
        "fit_manifest_expected_sha256": FROZEN_FIT128_MANIFEST_SHA256,
        "role_lock": None if args.role_lock is None else str(args.role_lock),
        "role_lock_expected_sha256": FROZEN_ROLE_LOCK_SHA256,
        "dataset_root": None if args.dataset_root is None else str(args.dataset_root),
        "fit_video_id": args.fit_video_id,
        "calibration_root": str(args.calibration_root),
        "execute": bool(args.execute),
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.dataset_root is None or args.fit_manifest is None or args.role_lock is None or args.fit_video_id is None:
        raise ValueError("--execute requires --dataset-root, --fit-manifest, --role-lock, and --fit-video-id")
    if torch is None:
        raise RuntimeError("--execute requires the existing PyTorch encoder runtime")
    if not str(args.device).lower().startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal frozen-reducer timing requires an available --device cuda[:index]")
    requested_device = torch.device(args.device)
    torch.cuda.set_device(torch.cuda.current_device() if requested_device.index is None else requested_device.index)

    output = args.output.resolve() if args.output else ROOT / "outputs/icassp2027/frozen-efficiency" / new_run_id(args.encoder)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing timing run: {output}")
    manifest_sha256 = sha256_file(args.fit_manifest)
    if manifest_sha256 != FROZEN_FIT128_MANIFEST_SHA256:
        raise ValueError("fit manifest SHA-256 differs from the frozen fit128 calibration manifest")
    role_lock_sha256 = sha256_file(args.role_lock)
    if role_lock_sha256 != FROZEN_ROLE_LOCK_SHA256:
        raise ValueError("role-lock SHA-256 differs from the frozen fit128 calibration lock")
    records = load_manifest_jsonl(args.fit_manifest)
    selected = _choose_fit_video(records, args.fit_video_id)
    source_path = (args.dataset_root / selected.path).resolve()
    info = probe_video(source_path)
    frames = CLIP_FRAMES[args.encoder]
    windows = sample_uniform_full_clips(info.num_frames, num_segments=8, clip_frames=frames,
                                        frame_stride=2, short_policy="stride1_if_needed")
    if len(windows) != 8:
        raise RuntimeError("formal timing needs exactly eight deterministic fit clips")
    batches = {
        1: clean_encoder_batch(build_clip_batch(source_path, selected.video_id, [windows[0].clip])),
        8: clean_encoder_batch(build_clip_batch(source_path, selected.video_id, [item.clip for item in windows])),
    }
    output.mkdir(parents=True)
    try:
        torch.set_num_threads(args.threads)
        project = load_project(args.profile)
        definition = dict(project.encoder(args.encoder)["definition"])
        constructor = dict(definition["constructor"])
        constructor["device"] = args.device
        if args.encoder in {"timesformer", "videomae"}:
            constructor["processor_tensor_type"] = "np"
        definition["constructor"] = constructor
        verified_identity = encoder_identity(definition, project_root=project.root)
        adapter = ENCODER_REGISTRY.create(args.encoder, **constructor)
        if hasattr(adapter, "bridge") and not adapter.bridge.loaded:
            adapter.bridge.load()
        native_bridge = create_observation_bridge(args.encoder, adapter)
        native_model = native_bridge.model
        _freeze_eval(native_model)
        runtime = CudaTimingRuntime(torch, device=args.device)
        route = _native_route(adapter, args.encoder, native_model)

        resolved = {
            **plan,
            "status": "running",
            "python_executable": sys.executable,
            "vadbench_file": vadbench.__file__,
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "cuda_device": args.device,
            "nvidia_smi_host_device_provenance": _nvidia_smi(),
            "verified_encoder_identity": verified_identity,
            "constructor": constructor,
            "actual_model": _model_details(native_model),
            "native_bridge": native_bridge.architecture(),
            "native_route": route.description,
            "inputs": {str(size): _batch_receipt(batch) for size, batch in batches.items()},
            "source": {"video_id": selected.video_id, "video_path": selected.path,
                       "num_frames": info.num_frames, "fps": info.fps,
                       "fit_manifest_sha256": manifest_sha256, "role_lock_sha256": role_lock_sha256},
            "source_sha256": {"cli": sha256_file(Path(__file__)),
                              "efficiency": sha256_file(Path(__import__("vadbench.paper.efficiency", fromlist=["x"]).__file__)),
                              "reduction_setup": sha256_file(Path(__import__("vadbench.paper.reduction_setup", fromlist=["x"]).__file__)),
                              "deployment": sha256_file(Path(__import__("vadbench.token_reduction.deployment", fromlist=["x"]).__file__))},
        }
        atomic_write_json(output / "resolved.json", resolved)
        results: dict[str, Any] = {}
        settings = FrozenTimingSettings()
        dense_preflight = {str(size): _token_preflight(adapter, batch, None) for size, batch in batches.items()}
        for method, reducer, calibration in _method_specs(args.calibration_root / args.encoder):
            try:
                deployment, setup = (None, None)
                if reducer != "identity":
                    from vadbench.paper.reduction_setup import prepare_reduction

                    deployment, setup = prepare_reduction(
                        adapter, args.encoder, batches[1], reducer=reducer,
                        output_dim=OUTPUT_DIMENSIONS[args.encoder],
                        verified_encoder_identity=verified_identity, calibration_run=calibration,
                        seed=0, batch_sizes=(1, 8),
                    )
                method_result: dict[str, Any] = {
                    "status": "running", "reducer": reducer, "setup": setup,
                    "reducer_identity": {"name": "identity"} if deployment is None else dict(deployment.reducer_identity),
                    "model_and_plan_resident_gpu_memory_bytes": None,
                    "batches": {},
                }
                method_result["native_pooled_parity_setup_only"] = {
                    str(size): _assert_pooled_parity(route, adapter, batch, deployment)
                    for size, batch in batches.items()
                }
                runtime.synchronize()
                runtime.empty_cache()
                runtime.synchronize()
                method_result["model_and_plan_resident_gpu_memory_bytes"] = {
                    "allocated": runtime.memory("memory_allocated"),
                    "reserved": runtime.memory("memory_reserved"),
                }
                for size in (1, 8):
                    batch = batches[size]
                    batch_result: dict[str, Any] = {
                        "preflight_actual_tokens": {
                            "before_dense": dense_preflight[str(size)],
                            "after_reducer": _token_preflight(adapter, batch, deployment),
                        },
                        "scopes": {},
                    }
                    for scope in ("pure_model", "adapter", "end_to_end"):
                        prepared_inputs: Any | None = None
                        operation: Any | None = None
                        try:
                            if scope == "pure_model":
                                prepared_inputs = route.prepare(batch)
                            operation = _operation(scope=scope, adapter=adapter, route=route, batch=batch,
                                                   deployment=deployment, prepared=prepared_inputs, source_path=source_path,
                                                   video_id=selected.video_id,
                                                   samples=[windows[0].clip] if size == 1 else [item.clip for item in windows],
                                                   verified_tokens=batch_result["preflight_actual_tokens"]["after_reducer"]["actual_feature_tokens"])
                            batch_result["scopes"][scope] = measure_scope(
                                operation, runtime=runtime, settings=settings, scope=scope, batch_size=size,
                                sampled_frames=size * frames,
                            )
                        except BaseException as error:
                            batch_result["scopes"][scope] = failed_scope(scope=scope, error=error)
                        finally:
                            # Pure-model input tensors are intentionally prebuilt
                            # outside its boundary, then released before the next
                            # scope so they cannot inflate adapter/end-to-end peaks.
                            del operation
                            del prepared_inputs
                    method_result["batches"][str(size)] = batch_result
                    atomic_write_json(output / "progress.json", {"status": "running", "current_method": method,
                                      "completed_methods": list(results), "current_batch_size": size})
                method_result["status"] = "completed" if all(
                    scope["status"] == "completed"
                    for batch in method_result["batches"].values() for scope in batch["scopes"].values()
                ) else "failed"
            except BaseException as error:
                method_result = {
                    "status": "failed", "reducer": reducer, "setup": None, "batches": None,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            results[method] = method_result
            atomic_write_json(output / "partial-results.json", results)
        final = {"schema_version": EFFICIENCY_SCHEMA_VERSION,
                 "status": "completed" if all(item["status"] == "completed" for item in results.values()) else "failed",
                 "resolved": "resolved.json", "methods": results}
        atomic_write_json(output / "result.json", final)
        atomic_write_json(output / "progress.json", {"status": final["status"], "completed_methods": list(results)})
        print(json.dumps({"status": final["status"], "output": str(output)}, ensure_ascii=False))
        return 0 if final["status"] == "completed" else 2
    except BaseException as error:
        atomic_write_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                          "error": str(error), "python_executable": sys.executable})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
