"""Pooled-only paper feature extraction over the existing adapter contract.

This is intentionally a small execution layer, not a second extraction
engine.  It supplies paper samplers to ``build_clip_batch``, strips labels
before ``adapter.encode``, and persists only one pooled vector per clip.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vadbench.artifacts import new_run_id
from vadbench.contracts import validate_clip_for_capabilities, validate_encoder_output
from vadbench.data.audit import compute_manifest_sha256
from vadbench.data.dense_sampling import (
    DenseSamplingPlan,
    ShortVideoPolicy,
    sample_uniform_full_clips,
)
from vadbench.data.manifest import VideoManifestRecord, validate_manifest
from vadbench.data.video import OpenCVVideoReader, VideoIOError, build_clip_batch
from vadbench.features import (
    ArrayReference,
    FeatureRecord,
    FeatureStore,
    atomic_write_json,
    atomic_write_jsonl,
    compute_encoder_fingerprint,
)

from .compatibility import (
    BackboneIdentity,
    RepresentationIdentity,
    SamplingIdentity,
    feature_cache_key,
)
from .stages import clean_encoder_batch

SamplingKind = Literal["uniform_full", "dense"]


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _array(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    # NumPy cannot represent torch bfloat16.  Its source dtype is recorded
    # separately, while the cache uses portable float32.
    if str(getattr(value, "dtype", "")) in {"torch.bfloat16", "bfloat16"}:
        value = value.float()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "biufc" or result.dtype == object:
        raise TypeError(f"{name} must be a numeric portable array")
    return np.ascontiguousarray(result)


def _package_version(module_name: str) -> str | None:
    root = module_name.partition(".")[0]
    try:
        return importlib.metadata.version(root)
    except importlib.metadata.PackageNotFoundError:
        return None


def _loaded_library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "timm"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        value = getattr(module, "__version__", None)
        versions[name] = str(value) if value is not None else _package_version(name) or "unknown"
    return versions


def _path_summary(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.exists():
        return {"location": str(path), "exists": False}
    resolved = path.resolve()
    if resolved.is_file():
        from vadbench.checkpoints import sha256_file

        return {
            "location": str(resolved),
            "exists": True,
            "kind": "file",
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    # A checkpoint directory is validated by orchestration before this point.
    # Never recursively inventory it here: model caches and download artifacts
    # would make an otherwise identical loaded checkpoint look different.
    return {
        "location": str(resolved),
        "exists": True,
        "kind": "directory",
    }


def _implementation_file(value: Any, *, role: str) -> dict[str, Any] | None:
    source = inspect.getsourcefile(value)
    if source is None:
        return None
    path = Path(source).resolve()
    if not path.is_file():
        return None
    from vadbench.checkpoints import sha256_file

    return {
        "role": role,
        "module": getattr(value, "__module__", type(value).__module__),
        "path": str(path),
        "sha256": sha256_file(path),
    }


_RUNTIME_DIAGNOSTIC_KEYS = frozenset(
    {
        "cache_dir",
        "checkpoint_path",
        "device",
        "device_str",
        "local_files_only",
        "local_path",
        "model_path",
        "name_or_path",
        "_name_or_path",
        "weights_path",
        "work_dir",
    }
)


def _semantic_json(value: Any, *, name: str) -> dict[str, Any]:
    """Validate a loaded configuration and remove deployment-only values."""

    if is_dataclass(value):
        value = asdict(value)
    elif isinstance(value, Mapping):
        value = dict(value)
    else:
        value = value.to_dict()

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key)
                if key.lower() in _RUNTIME_DIAGNOSTIC_KEYS:
                    continue
                if (
                    key == "model_name"
                    and isinstance(raw_value, (Path, str))
                    and Path(raw_value).is_absolute()
                ):
                    continue
                result[key] = normalize(raw_value)
            return result
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if type(item).__module__ == "torch" and type(item).__qualname__ == "dtype":
            return str(item)
        return item

    try:
        normalized = json.loads(
            json.dumps(normalize(value), ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-compatible values") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{name} must be a JSON object")
    return normalized


def _runtime_configurations(
    *, encoder: Any | None, model: Any | None, backbone: Any | None, processor: Any | None
) -> dict[str, dict[str, Any]]:
    """Capture configuration actually present on loaded inference objects."""

    configurations: dict[str, dict[str, Any]] = {}
    if encoder is not None and (cfg := getattr(encoder, "cfg", None)) is not None:
        configurations["encoder_cfg"] = _semantic_json(cfg, name="encoder.cfg")
    for role, value in (
        ("model", model),
        ("model_nested_model", None if model is None else getattr(model, "model", None)),
        ("backbone", backbone),
        ("backbone_nested_model", None if backbone is None else getattr(backbone, "model", None)),
    ):
        if value is not None and (config := getattr(value, "config", None)) is not None:
            configurations[f"{role}_config"] = _semantic_json(config, name=f"{role}.config")
    if processor is not None:
        configurations["processor"] = _semantic_json(processor.to_dict(), name="processor.to_dict()")
    return configurations


def adapter_runtime_summary(adapter: Any) -> dict[str, Any]:
    """Return observed adapter/model facts used in the actual cache identity."""

    adapter_type = type(adapter)
    encoder = getattr(adapter, "encoder", None)
    properties = {}
    for name in (
        "backend",
        "implementation_source",
        "preprocess_profile",
        "pooling",
        "feature_stage",
        "variant",
        "revision",
        "model_name",
    ):
        value = getattr(adapter, name, None)
        if value is not None and isinstance(value, (str, int, float, bool)):
            properties[name] = value
    model_path = getattr(adapter, "model_path", None)
    if model_path is None and encoder is not None:
        model_path = getattr(getattr(encoder, "cfg", None), "model_name", None)
    bridge = getattr(adapter, "bridge", None)
    if encoder is None and bridge is not None:
        encoder = getattr(bridge, "encoder", None)
    encoder_type = None if encoder is None else f"{type(encoder).__module__}.{type(encoder).__qualname__}"
    model = getattr(adapter, "model", None)
    if model is None and encoder is not None:
        model = getattr(encoder, "model", None)
    backbone = getattr(adapter, "backbone", None)
    if backbone is None and encoder is not None:
        backbone = getattr(encoder, "backbone", None)
    processor = getattr(adapter, "processor", None)
    if processor is None and encoder is not None:
        processor = getattr(encoder, "processor", None)
    candidates = (
        ("adapter_input", adapter_type),
        ("encoder", None if encoder is None else type(encoder)),
        ("model", None if model is None else type(model)),
        ("backbone", None if backbone is None else type(backbone)),
        ("processor", None if processor is None else type(processor)),
    )
    files = {
        role: entry
        for role, value in candidates
        if value is not None
        for entry in [_implementation_file(value, role=role)]
        if entry is not None
    }
    is_active_adapter = type(adapter).__module__.startswith("vadbench.integrations")
    if is_active_adapter and not any(role in files for role in ("model", "backbone", "encoder")):
        raise RuntimeError(
            "active adapter has no loaded model/backbone seam; load its existing local asset before "
            "constructing a paper representation identity"
        )
    return {
        "adapter_type": f"{adapter_type.__module__}.{adapter_type.__qualname__}",
        "adapter_library_version": _package_version(adapter_type.__module__),
        "encoder_type": encoder_type,
        "encoder_library_version": None
        if encoder is None
        else _package_version(type(encoder).__module__),
        "capabilities": {
            name: getattr(adapter.capabilities, name)
            for name in (
                "fixed_num_frames",
                "min_frames",
                "max_frames",
                "supports_fixed_clip",
                "supports_training",
            )
        },
        "properties": properties,
        "runtime_configurations": _runtime_configurations(
            encoder=encoder,
            model=model,
            backbone=backbone,
            processor=processor,
        ),
        "model_asset": _path_summary(model_path),
        "implementation_files": files,
        "loaded_library_versions": _loaded_library_versions(),
    }


def _verified_weights_digest(identity: Mapping[str, Any]) -> str:
    checkpoint = identity.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or not checkpoint.get("sha256"):
        raise ValueError("verified_encoder_identity must contain verified checkpoint SHA-256 values")
    return compute_encoder_fingerprint({"verified_checkpoint": dict(checkpoint)})


def _verified_code_digest(identity: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    files = runtime.get("implementation_files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("adapter runtime has no implementation seam evidence")
    semantic_files = [
        {key: value for key, value in entry.items() if key != "path"}
        for _role, entry in sorted(files.items())
    ]
    constructor = identity.get("constructor")
    if not isinstance(constructor, Mapping):
        raise ValueError("verified_encoder_identity must contain a canonical constructor mapping")
    configurations = runtime.get("runtime_configurations")
    if not isinstance(configurations, Mapping):
        raise ValueError("adapter runtime has no loaded configuration evidence")
    return compute_encoder_fingerprint(
        {
            "verified_constructor": _semantic_json(
                constructor, name="verified_encoder_identity.constructor"
            ),
            "implementation_files": semantic_files,
            "loaded_library_versions": runtime["loaded_library_versions"],
            "runtime_configurations": dict(configurations),
        }
    )


def _semantic_runtime_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Strip diagnostics-only paths and broad Git evidence from the cache key."""

    properties = {
        name: value
        for name, value in dict(runtime.get("properties", {})).items()
        if name not in {"device", "device_str"}
    }
    files = runtime.get("implementation_files", {})
    return {
        "runtime_id": runtime["runtime_id"],
        "representation_fingerprint": runtime["representation_fingerprint"],
        "sampling_fingerprint": runtime["sampling_fingerprint"],
        "source_data_digest": runtime["source_data_digest"],
        "declared_readout": runtime["declared_readout"],
        "adapter_type": runtime["adapter_type"],
        "adapter_library_version": runtime["adapter_library_version"],
        "encoder_type": runtime["encoder_type"],
        "encoder_library_version": runtime["encoder_library_version"],
        "capabilities": runtime["capabilities"],
        "properties": properties,
        "runtime_configurations": runtime["runtime_configurations"],
        "implementation_files": [
            {key: value for key, value in entry.items() if key != "path"}
            for _role, entry in sorted(dict(files).items())
        ],
        "loaded_library_versions": runtime["loaded_library_versions"],
        # Verified weights participate by their digest; full orchestration Git
        # provenance stays in resolved.json and does not perturb cache meaning.
        "verified_weights_digest": _verified_weights_digest(
            runtime["verified_encoder_identity"]
        ),
        "verified_constructor": _semantic_json(
            runtime["verified_encoder_identity"].get("constructor"),
            name="verified_encoder_identity.constructor",
        ),
    }


def representation_from_verified_encoder(
    *,
    runtime_id: str,
    adapter: Any,
    verified_encoder_identity: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    readout: Mapping[str, Any],
    reducer: Mapping[str, Any],
    output_dim: int,
    precision: str,
    position_strategy: Mapping[str, Any],
) -> RepresentationIdentity:
    """Construct a representation identity from actual verified encoder evidence."""

    identity = dict(verified_encoder_identity)
    if identity.get("adapter") != runtime_id:
        raise ValueError("verified_encoder_identity.adapter must match runtime_id")
    runtime = adapter_runtime_summary(adapter)
    return RepresentationIdentity(
        backbone=BackboneIdentity(
            runtime_id=runtime_id,
            weights_digest=_verified_weights_digest(identity),
            code_digest=_verified_code_digest(identity, runtime),
            preprocessing=preprocessing,
            readout=readout,
        ),
        reducer=reducer,
        output_dim=output_dim,
        precision=precision,
        position_strategy=position_strategy,
    )


@dataclass(frozen=True)
class PooledExtractionSpec:
    runtime_id: str
    verified_encoder_identity: Mapping[str, Any]
    representation: RepresentationIdentity
    sampling: SamplingIdentity
    sampling_kind: SamplingKind
    clip_frames: int
    frame_stride: int
    num_segments: int = 32
    window_stride: int = 1
    short_policy: ShortVideoPolicy = "strict"
    micro_batch_size: int = 8
    strict_manifest_info: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise ValueError("runtime_id must be non-empty")
        if not isinstance(self.representation, RepresentationIdentity):
            raise TypeError("representation must be a RepresentationIdentity")
        if not isinstance(self.verified_encoder_identity, Mapping):
            raise TypeError("verified_encoder_identity must be a mapping")
        if not isinstance(self.sampling, SamplingIdentity):
            raise TypeError("sampling must be a SamplingIdentity")
        if self.runtime_id != self.representation.backbone.runtime_id:
            raise ValueError("runtime_id must match representation backbone runtime_id")
        if self.verified_encoder_identity.get("adapter") != self.runtime_id:
            raise ValueError("verified_encoder_identity.adapter must match runtime_id")
        if self.sampling_kind not in {"uniform_full", "dense"}:
            raise ValueError("sampling_kind must be 'uniform_full' or 'dense'")
        if self.short_policy not in {"strict", "stride1_if_needed"}:
            raise ValueError("short_policy must be 'strict' or 'stride1_if_needed'")
        for name in ("clip_frames", "frame_stride", "num_segments", "window_stride", "micro_batch_size"):
            _positive(getattr(self, name), name)
        expected_regime = "train_32" if self.sampling_kind == "uniform_full" else "test_dense"
        if self.sampling.regime != expected_regime:
            raise ValueError(f"{self.sampling_kind} requires sampling.regime={expected_regime!r}")
        selection = self.sampling.frame_selection
        window = self.sampling.window
        stride = self.sampling.stride
        expected_kind = "uniform_full" if self.sampling_kind == "uniform_full" else "dense_sliding"
        if selection.get("kind") != expected_kind:
            raise ValueError("sampling.frame_selection.kind does not match sampling_kind")
        if selection.get("implementation") != _sampling_implementation_identity():
            raise ValueError("sampling implementation evidence does not match the active sampler code")
        if selection.get("short_video_policy") != self.short_policy:
            raise ValueError("sampling.short_video_policy does not match short_policy")
        if window.get("clip_frames") != self.clip_frames:
            raise ValueError("sampling.window.clip_frames does not match clip_frames")
        if stride.get("frame_stride") != self.frame_stride:
            raise ValueError("sampling.stride.frame_stride does not match frame_stride")
        if self.sampling_kind == "uniform_full":
            if selection.get("num_segments") != self.num_segments:
                raise ValueError("sampling.frame_selection.num_segments does not match num_segments")
        elif selection.get("window_stride") != self.window_stride:
            raise ValueError("sampling.frame_selection.window_stride does not match window_stride")
        object.__setattr__(self, "verified_encoder_identity", dict(self.verified_encoder_identity))


@dataclass(frozen=True)
class PooledExtractionResult:
    run_dir: str
    feature_root: str | None
    completed: bool
    encoder_fingerprint: str
    records_written: int
    failures: tuple[dict[str, str], ...]
    status_path: str


def _source_digest(records: Sequence[VideoManifestRecord]) -> str:
    return "sha256:" + compute_manifest_sha256(records)


def make_data_content_evidence(
    records: Iterable[VideoManifestRecord | Mapping[str, Any]], *, dataset_root: str | Path
) -> dict[str, Any]:
    """Freeze canonical manifest and selected source-video bytes identities."""

    from vadbench.checkpoints import sha256_file

    normalized = validate_manifest(records)
    root = Path(dataset_root).expanduser().resolve()
    videos: list[dict[str, Any]] = []
    for record in sorted(normalized, key=lambda item: item.video_id):
        path = record.resolve_path(root)
        if not path.is_file():
            raise FileNotFoundError(f"manifest video is missing: {record.video_id}")
        videos.append(
            {
                "video_id": record.video_id,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest_digest = _source_digest(normalized)
    source_digest = compute_encoder_fingerprint(
        {"canonical_manifest": manifest_digest, "video_contents": videos}
    )
    return {
        "canonical_manifest_sha256": manifest_digest,
        "videos": videos,
        "source_digest": source_digest,
    }


def _sampling_implementation_identity() -> dict[str, str]:
    from vadbench.checkpoints import sha256_file
    from vadbench.data import dense_sampling, sampling, video

    return {
        "dense_sampling_sha256": sha256_file(Path(dense_sampling.__file__).resolve()),
        "base_sampling_sha256": sha256_file(Path(sampling.__file__).resolve()),
        "video_reader_sha256": sha256_file(Path(video.__file__).resolve()),
    }


def make_sampling_identity(
    records: Iterable[VideoManifestRecord | Mapping[str, Any]],
    *,
    dataset_root: str | Path,
    sampling_kind: SamplingKind,
    clip_frames: int,
    frame_stride: int,
    num_segments: int = 32,
    window_stride: int = 1,
    short_policy: ShortVideoPolicy = "strict",
    padding_kind: str = "forbid",
    projection_kind: str = "frame_intervals_v1",
) -> SamplingIdentity:
    """Build a sampling identity coupled to the current sampler implementation."""

    normalized = validate_manifest(records)
    data_evidence = make_data_content_evidence(normalized, dataset_root=dataset_root)
    if sampling_kind not in {"uniform_full", "dense"}:
        raise ValueError("sampling_kind must be 'uniform_full' or 'dense'")
    _positive(clip_frames, "clip_frames")
    _positive(frame_stride, "frame_stride")
    _positive(num_segments, "num_segments")
    _positive(window_stride, "window_stride")
    if short_policy not in {"strict", "stride1_if_needed"}:
        raise ValueError("short_policy must be 'strict' or 'stride1_if_needed'")
    selection: dict[str, Any] = {
        "kind": "uniform_full" if sampling_kind == "uniform_full" else "dense_sliding",
        "short_video_policy": short_policy,
        "implementation": _sampling_implementation_identity(),
    }
    if sampling_kind == "uniform_full":
        selection["num_segments"] = num_segments
        regime = "train_32"
    else:
        selection["window_stride"] = window_stride
        regime = "test_dense"
    return SamplingIdentity(
        source_digest=data_evidence["source_digest"],
        regime=regime,
        frame_selection=selection,
        window={"clip_frames": clip_frames},
        stride={"frame_stride": frame_stride},
        padding={"kind": padding_kind},
        projection={"kind": projection_kind},
    )


def _samples(spec: PooledExtractionSpec, num_frames: int) -> tuple[Any, ...]:
    if spec.sampling_kind == "uniform_full":
        return sample_uniform_full_clips(
            num_frames,
            num_segments=spec.num_segments,
            clip_frames=spec.clip_frames,
            frame_stride=spec.frame_stride,
            short_policy=spec.short_policy,
        )
    return DenseSamplingPlan(
        clip_frames=spec.clip_frames,
        frame_stride=spec.frame_stride,
        window_stride=spec.window_stride,
        short_policy=spec.short_policy,
    ).sample(num_frames)


def _sample_metadata(sample: Any) -> tuple[int, int, int, dict[str, Any]]:
    metadata: dict[str, Any] = {"clip_index": int(sample.clip_index)}
    indices = tuple(int(index) for index in sample.frame_indices)
    if len(indices) > 1:
        strides = {right - left for left, right in zip(indices[:-1], indices[1:], strict=True)}
        if len(strides) == 1:
            metadata["actual_frame_stride"] = strides.pop()
    if hasattr(sample, "end_anchored"):
        metadata["end_anchored"] = bool(sample.end_anchored)
    if hasattr(sample, "requested_input_start"):
        metadata.update(
            {
                "requested_input_start": int(sample.requested_input_start),
                "input_start_frame": int(sample.input_start_frame),
                "input_end_frame": int(sample.input_end_frame),
                "input_window_reused": bool(sample.input_window_reused),
            }
        )
    return int(sample.clip_index), int(sample.score_frame_start), int(sample.score_frame_end), metadata


def _chunked(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _dtype_matches_precision(dtype: str, precision: str) -> bool:
    normalized = precision.strip().lower().replace("torch.", "")
    expected = {
        "float32": {"float32", "torch.float32"},
        "fp32": {"float32", "torch.float32"},
        "float16": {"float16", "torch.float16"},
        "fp16": {"float16", "torch.float16"},
        "bfloat16": {"bfloat16", "torch.bfloat16"},
        "bf16": {"bfloat16", "torch.bfloat16"},
    }.get(normalized)
    return expected is not None and dtype.lower() in expected


def _checked_pooled_output(
    output: Any, representation: RepresentationIdentity
) -> tuple[np.ndarray, int, str, str]:
    """Check shape/dtype without materializing intermediate token tensors."""

    pooled_raw = getattr(output, "pooled", None)
    if pooled_raw is None:
        raise ValueError("pooled-only extraction requires EncoderOutput.pooled")
    pooled = _array(pooled_raw, "output.pooled")
    pooled_dtype = str(getattr(pooled_raw, "dtype", pooled.dtype))
    features = getattr(output, "features", None)
    feature_shape = tuple(int(value) for value in getattr(features, "shape", ()))
    feature_dtype = str(getattr(features, "dtype", ""))
    if (
        pooled.ndim != 2
        or pooled.shape[0] != output.batch_size
        or pooled.shape[1] != representation.output_dim
    ):
        raise ValueError("EncoderOutput.pooled must have shape [B, representation.output_dim]")
    if len(feature_shape) != 3 or feature_shape[0] != output.batch_size:
        raise ValueError("EncoderOutput.features must have shape [B,S,D]")
    if feature_shape[2] != representation.output_dim:
        raise ValueError("EncoderOutput token dimension differs from representation.output_dim")
    if feature_shape[1] <= 0:
        raise ValueError("EncoderOutput emitted no tokens")
    if not _dtype_matches_precision(feature_dtype, representation.precision) or not _dtype_matches_precision(
        pooled_dtype, representation.precision
    ):
        raise ValueError(
            "EncoderOutput dtype does not match declared representation precision: "
            f"features={feature_dtype}, pooled={pooled_dtype}, declared={representation.precision}"
        )
    return pooled, feature_shape[1], feature_dtype, pooled_dtype


@contextmanager
def _frozen_adapter(adapter: Any) -> Iterable[None]:
    """Run inference under no-grad while restoring every reachable module mode."""

    modules: list[Any] = []
    for candidate in (adapter, getattr(adapter, "encoder", None)):
        iterator = getattr(candidate, "modules", None)
        if callable(iterator):
            for module in iterator():
                if module not in modules:
                    modules.append(module)
    modes = [(module, bool(getattr(module, "training", False))) for module in modules]
    for module, _mode in modes:
        evaluate = getattr(module, "eval", None)
        if callable(evaluate):
            evaluate()
    try:
        try:
            import torch

            context = torch.inference_mode()
        except ImportError:  # pragma: no cover - minimal optional-dependency environment.
            context = nullcontext()
        with context:
            yield
    finally:
        for module, mode in modes:
            module.training = mode


def _write_pooled_record(
    store: FeatureStore,
    *,
    row: int,
    record: VideoManifestRecord,
    sample: Any,
    fps: float,
    encoder_fingerprint: str,
    paper_identity: Mapping[str, str],
    runtime: Mapping[str, Any],
    source_video: Mapping[str, Any],
    pooled: np.ndarray,
    token_count: int,
    token_dtype: str,
    pooled_dtype: str,
) -> FeatureRecord:
    clip_index, frame_start, frame_end, sample_metadata = _sample_metadata(sample)
    if not (0 <= frame_start < frame_end):
        raise ValueError("sampler produced an invalid score frame interval")
    return store.write(
        video_id=record.video_id,
        clip_id=f"{record.video_id}:clip-{clip_index:06d}",
        clip_index=clip_index,
        encoder_fingerprint=encoder_fingerprint,
        features=pooled[row : row + 1],
        pooled=pooled[row],
        start_s=frame_start / fps,
        end_s=frame_end / fps,
        frame_start=frame_start,
        frame_end=frame_end,
        metadata={
            "storage": {"feature_level": "pooled_only", "intermediate_tokens_stored": False},
            "readout": {
                "declared": dict(runtime.get("declared_readout", {})),
                "adapter": {
                    name: runtime.get("properties", {}).get(name)
                    for name in ("pooling", "feature_stage")
                    if name in runtime.get("properties", {})
                },
            },
            "encoder_output": {
                "token_count": token_count,
                "token_dtype": token_dtype,
                "pooled_dtype": pooled_dtype,
                "stored_dtype": pooled.dtype.str,
            },
            "paper_identity": dict(paper_identity),
            "source": {"split": record.split.value, "is_anomaly": record.is_anomaly},
            "source_video": dict(source_video),
            "sampling": sample_metadata | {
                "kind": "uniform_full" if hasattr(sample, "requested_input_start") else "dense"
            },
            "runtime": dict(runtime),
        },
    )


def _merge_shards(run_dir: Path, shard_dirs: Sequence[Path]) -> int:
    """Publish a final index only after all per-video shard indexes validate."""

    merged: list[FeatureRecord] = []
    for shard in shard_dirs:
        store = FeatureStore(shard)
        prefix = shard.relative_to(run_dir).as_posix()
        for record in store.iter_records():
            arrays = {
                name: ArrayReference(
                    path=f"{prefix}/{reference.path}",
                    key=reference.key,
                    shape=reference.shape,
                    dtype=reference.dtype,
                    sha256=reference.sha256,
                    nbytes=reference.nbytes,
                )
                for name, reference in record.arrays.items()
            }
            merged.append(
                FeatureRecord(
                    video_id=record.video_id,
                    clip_id=record.clip_id,
                    clip_index=record.clip_index,
                    encoder_fingerprint=record.encoder_fingerprint,
                    storage_format=record.storage_format,
                    arrays=arrays,
                    start_s=record.start_s,
                    end_s=record.end_s,
                    frame_start=record.frame_start,
                    frame_end=record.frame_end,
                    metadata=record.metadata,
                    created_at=record.created_at,
                )
            )
    keys = [(item.encoder_fingerprint, item.video_id, item.clip_id) for item in merged]
    if len(keys) != len(set(keys)):
        raise ValueError("shard merge found duplicate feature identities")
    merged.sort(key=lambda item: (item.encoder_fingerprint, item.video_id, item.clip_index, item.clip_id))
    # Validate every rewritten relative blob path before the final index exists;
    # consumers can never observe an index that still has an unchecked bundle.
    final = FeatureStore(run_dir)
    for record in merged:
        final.load_bundle(record)
    atomic_write_jsonl(run_dir / "index.jsonl", (item.to_dict() for item in merged))
    return len(merged)


def extract_pooled_features(
    spec: PooledExtractionSpec,
    *,
    adapter: Any,
    manifest: Iterable[VideoManifestRecord | Mapping[str, Any]],
    dataset_root: str | Path,
    output_root: str | Path,
    run_id: str | None = None,
    backend: Any | None = None,
) -> PooledExtractionResult:
    """Run one isolated pooled-only paper extraction attempt.

    Per-video shard indexes survive failures for diagnosis.  A root
    ``index.jsonl`` and ``completed=true`` appear only after every requested
    video was extracted and every merged bundle checksum was reopened.
    """

    if not isinstance(spec, PooledExtractionSpec):
        raise TypeError("spec must be a PooledExtractionSpec")
    records = validate_manifest(manifest)
    root = Path(dataset_root).expanduser().resolve()
    data_evidence = make_data_content_evidence(records, dataset_root=root)
    source_digest = data_evidence["source_digest"]
    if spec.sampling.source_digest != source_digest:
        raise ValueError("sampling.source_digest does not match manifest and video content evidence")
    capabilities = adapter.capabilities
    if not bool(getattr(capabilities, "supports_fixed_clip", False)):
        raise ValueError("pooled paper extraction requires a fixed-clip adapter")
    runtime = adapter_runtime_summary(adapter)
    verified_identity = dict(spec.verified_encoder_identity)
    expected_weights = _verified_weights_digest(verified_identity)
    expected_code = _verified_code_digest(verified_identity, runtime)
    if spec.representation.backbone.weights_digest != expected_weights:
        raise ValueError("representation weights_digest does not match verified encoder checkpoint evidence")
    if spec.representation.backbone.code_digest != expected_code:
        raise ValueError("representation code_digest does not match verified adapter implementation evidence")
    runtime["runtime_id"] = spec.runtime_id
    runtime["representation_fingerprint"] = spec.representation.fingerprint
    runtime["sampling_fingerprint"] = spec.sampling.fingerprint
    runtime["source_data_digest"] = source_digest
    runtime["declared_readout"] = dict(spec.representation.backbone.readout)
    runtime["verified_encoder_identity"] = verified_identity
    encoder_fingerprint = compute_encoder_fingerprint(
        {"paper_pooled_cache": _semantic_runtime_identity(runtime)}
    )
    paper_identity = {
        "representation_fingerprint": spec.representation.fingerprint,
        "sampling_fingerprint": spec.sampling.fingerprint,
        "feature_cache_fingerprint": feature_cache_key(spec.representation, spec.sampling),
    }
    selected_run_id = run_id or new_run_id("paper-extract")
    if not selected_run_id or any(char in selected_run_id for char in "/\\"):
        raise ValueError("run_id must be a non-empty basename")
    run_dir = Path(output_root).expanduser().resolve() / selected_run_id
    if run_dir.exists():
        raise FileExistsError(f"paper extraction run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_write_json(
        run_dir / "resolved.json",
        {
            "spec": {
                "runtime_id": spec.runtime_id,
                "representation": spec.representation.to_dict(),
                "sampling": spec.sampling.to_dict(),
                "sampling_kind": spec.sampling_kind,
                "clip_frames": spec.clip_frames,
                "frame_stride": spec.frame_stride,
                "num_segments": spec.num_segments,
                "window_stride": spec.window_stride,
                "short_policy": spec.short_policy,
                "micro_batch_size": spec.micro_batch_size,
            },
            "data_content_evidence": data_evidence,
            "runtime": runtime,
            "encoder_fingerprint": encoder_fingerprint,
            "paper_identity": paper_identity,
        },
    )
    failures: list[dict[str, str]] = []
    shard_dirs: list[Path] = []
    written = 0
    evidence_by_video = {item["video_id"]: item for item in data_evidence["videos"]}
    for record in records:
        shard = run_dir / "shards" / record.video_id
        try:
            path = record.resolve_path(root)
            with OpenCVVideoReader(path, backend=backend) as reader:
                info = reader.info
            if spec.strict_manifest_info and record.num_frames is not None and record.num_frames != info.num_frames:
                raise VideoIOError("manifest num_frames differs from OpenCV probe")
            if spec.strict_manifest_info and record.fps is not None and not math.isclose(
                record.fps, info.fps, rel_tol=1e-3
            ):
                raise VideoIOError("manifest fps differs from OpenCV probe")
            samples = _samples(spec, info.num_frames)
            shard_store = FeatureStore(shard)
            for group in _chunked(samples, spec.micro_batch_size):
                batch = build_clip_batch(
                    path,
                    record.video_id,
                    [item.clip for item in group],
                    backend=backend,
                    metadata={"source_num_frames": info.num_frames, "source_fps": info.fps},
                )
                clean = clean_encoder_batch(batch)
                validate_clip_for_capabilities(clean, capabilities, train=False)
                with _frozen_adapter(adapter):
                    output = adapter.encode(clean, train=False)
                validate_encoder_output(output, clean)
                pooled, token_count, token_dtype, pooled_dtype = _checked_pooled_output(
                    output, spec.representation
                )
                for row, sample in enumerate(group):
                    _write_pooled_record(
                        shard_store,
                        row=row,
                        record=record,
                        sample=sample,
                        fps=info.fps,
                        encoder_fingerprint=encoder_fingerprint,
                        paper_identity=paper_identity,
                        runtime=runtime,
                        source_video={
                            "manifest_path": record.path,
                            "actual_num_frames": info.num_frames,
                            "actual_fps": info.fps,
                            "width": info.width,
                            "height": info.height,
                            "summary_fingerprint": compute_encoder_fingerprint(
                                {
                                    "manifest_path": record.path,
                                    "num_frames": info.num_frames,
                                    "fps": info.fps,
                                    "width": info.width,
                                    "height": info.height,
                                }
                            ),
                            "content_sha256": evidence_by_video[record.video_id]["sha256"],
                            "content_size_bytes": evidence_by_video[record.video_id]["size_bytes"],
                        },
                        pooled=pooled,
                        token_count=token_count,
                        token_dtype=token_dtype,
                        pooled_dtype=pooled_dtype,
                    )
                    written += 1
            if len(shard_store.records()) != len(samples):
                raise RuntimeError("per-video shard index is incomplete")
            shard_dirs.append(shard)
        except Exception as exc:
            failures.append(
                {"video_id": record.video_id, "type": type(exc).__name__, "message": str(exc)}
            )
    completed = not failures
    feature_root: str | None = None
    if completed:
        try:
            merged = _merge_shards(run_dir, shard_dirs)
            if merged != written:
                raise RuntimeError("merged feature record count differs from completed shards")
            feature_root = str(run_dir)
        except Exception as exc:
            completed = False
            failures.append({"video_id": "<merge>", "type": type(exc).__name__, "message": str(exc)})
            (run_dir / "index.jsonl").unlink(missing_ok=True)
    status = {
        "status": "completed" if completed else "partial_failed",
        "completed": completed,
        "records_written_to_shards": written,
        "feature_root": feature_root,
        "encoder_fingerprint": encoder_fingerprint,
        "failures": failures,
    }
    status_path = run_dir / "status.json"
    atomic_write_json(status_path, status)
    return PooledExtractionResult(
        run_dir=str(run_dir),
        feature_root=feature_root,
        completed=completed,
        encoder_fingerprint=encoder_fingerprint,
        records_written=written,
        failures=tuple(failures),
        status_path=str(status_path),
    )


__all__ = [
    "PooledExtractionResult",
    "PooledExtractionSpec",
    "SamplingKind",
    "adapter_runtime_summary",
    "extract_pooled_features",
    "make_data_content_evidence",
    "make_sampling_identity",
    "representation_from_verified_encoder",
]
