"""把实验 YAML、adapter registry、采样 batch 和缓存策略连接起来。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vadbench.checkpoints import load_checkpoint_registry, verify_checkpoint
from vadbench.compression import build_cache_policy
from vadbench.config import load_yaml, validate_capabilities, validate_encoder_config
from vadbench.contracts import ClipBatch
from vadbench.integrations import DEFAULT_INTEGRATION_CATALOG
from vadbench.registry import ENCODER_REGISTRY
from vadbench.resources import package_resource_path

BUILTIN_ENCODER_CONFIGS = dict(DEFAULT_INTEGRATION_CATALOG.definition_paths())


def load_encoder_definition(
    adapter_id: str,
    *,
    project_root: str | Path = ".",
    path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if path is None:
        try:
            selected = Path(BUILTIN_ENCODER_CONFIGS[adapter_id])
        except KeyError as exc:
            raise ValueError(f"catalog 中不存在 encoder definition：{adapter_id!r}") from exc
    else:
        selected = Path(path)
    if not selected.is_absolute():
        selected = root / selected
    selected = selected.resolve()
    if selected != root and root not in selected.parents:
        raise ValueError(f"encoder {adapter_id!r} 的 definition 越出 project_root：{selected}")
    if not selected.is_file():
        if path is not None and Path(path).as_posix() != BUILTIN_ENCODER_CONFIGS.get(adapter_id):
            raise FileNotFoundError(selected)
        selected = package_resource_path(BUILTIN_ENCODER_CONFIGS[adapter_id])
    definition = load_yaml(selected)
    if definition.get("adapter") != adapter_id:
        raise ValueError(
            f"encoder definition adapter={definition.get('adapter')!r}，预期 {adapter_id!r}"
        )
    return definition


def _absolute_constructor_paths(constructor: dict[str, Any], project_root: Path) -> None:
    for key in ("model_name", "model_path", "checkout_path", "checkpoint_path", "prototxt_path"):
        value = constructor.get(key)
        if not isinstance(value, str) or not value:
            continue
        path = Path(value)
        if path.is_absolute():
            continue
        candidate = (project_root / path).resolve()
        if candidate.exists() or key.endswith("_path"):
            constructor[key] = str(candidate)


def resolve_encoder_config(
    config: Mapping[str, Any],
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Resolve one constructor and its actual asset binding without loading a model."""
    encoder_config = config.get("encoder")
    if not isinstance(encoder_config, Mapping):
        raise ValueError("experiment 缺少 encoder 对象")
    validate_encoder_config(encoder_config)
    adapter_id = str(encoder_config.get("adapter", ""))
    if not adapter_id:
        raise ValueError("encoder.adapter 不能为空")
    spec = ENCODER_REGISTRY.get_spec(adapter_id)
    validate_capabilities(config, spec.capabilities)
    definition_path = encoder_config.get("definition")
    definition = load_encoder_definition(
        adapter_id,
        project_root=project_root,
        path=definition_path,
    )
    constructor = {**spec.default_kwargs, **definition.get("constructor", {})}
    params = encoder_config.get("params", {})
    constructor.update(params)
    if encoder_config.get("device") is not None:
        constructor["device"] = str(encoder_config["device"])
    root = Path(project_root).resolve()
    checkpoint = dict(definition.get("checkpoint", {}))
    if not checkpoint and encoder_config.get("checkpoint") is not None:
        raise ValueError(
            "encoder.checkpoint requires a definition checkpoint.constructor_key binding"
        )
    if checkpoint:
        binding = checkpoint.get("constructor_key")
        if not isinstance(binding, str) or binding not in constructor:
            raise ValueError("checkpoint.constructor_key must name the adapter's weight argument")
        registry_path = root / "registry/checkpoints.yaml"
        if not registry_path.is_file():
            registry_path = package_resource_path("registry/checkpoints.yaml")
        checkpoint_id = encoder_config.get("checkpoint")
        if checkpoint_id is None:
            checkpoint_id = checkpoint.get("registry_id")
        if checkpoint_id is None:
            checkpoint_id = DEFAULT_INTEGRATION_CATALOG.get(adapter_id).checkpoint.registry_id
        weights = load_checkpoint_registry(registry_path)
        if checkpoint_id not in weights:
            raise ValueError(f"未知 checkpoint：{checkpoint_id!r}")
        weight = weights[str(checkpoint_id)]
        if weight.adapter != adapter_id:
            raise ValueError(f"checkpoint {weight.id!r} 不属于 adapter {adapter_id!r}")
        if encoder_config.get("checkpoint") is not None:
            if not weight.local_path:
                raise ValueError(f"checkpoint {weight.id!r} 缺少 local_path")
            if binding in params:
                raise ValueError(f"encoder.checkpoint 与 encoder.params.{binding} 不能同时指定")
            constructor[binding] = weight.local_path
        checkpoint.update(
            id=weight.id,
            model_id=weight.repo_id,
            revision=weight.revision,
            license=weight.license,
            source=weight.source,
            sha256=dict(weight.sha256),
        )
    _absolute_constructor_paths(constructor, root)
    if checkpoint:
        # This is the exact argument consumed by the adapter, not a separate metadata path.
        binding = checkpoint["constructor_key"]
        constructor[binding] = str((root / constructor[binding]).resolve())
        checkpoint["local_path"] = constructor[binding]
    return {
        **definition,
        "adapter_target": spec.target_path,
        "constructor": constructor,
        "checkpoint": checkpoint,
    }


def create_encoder_from_experiment(
    config: Mapping[str, Any],
    *,
    project_root: str | Path = ".",
) -> tuple[Any, dict[str, Any]]:
    definition = resolve_encoder_config(config, project_root=project_root)
    definition["identity"] = encoder_identity(definition, project_root=project_root)
    return ENCODER_REGISTRY.create(definition["adapter"], **definition["constructor"]), definition


def encoder_identity(
    definition: Mapping[str, Any],
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Fingerprint effective parameters and verified assets, independent of deployment paths."""

    from vadbench.artifacts import collect_git_provenance

    constructor = dict(definition.get("constructor", {}))
    checkpoint = dict(definition.get("checkpoint", {}))
    if checkpoint:
        registry_path = Path(project_root) / "registry/checkpoints.yaml"
        if not registry_path.is_file():
            registry_path = package_resource_path("registry/checkpoints.yaml")
        spec = load_checkpoint_registry(registry_path)[checkpoint["id"]]
        digests = verify_checkpoint(spec, checkpoint["local_path"])
        constructor[checkpoint["constructor_key"]] = {"checkpoint_sha256": digests}
        checkpoint = {key: checkpoint[key] for key in ("id", "model_id", "revision", "license")} | {
            "sha256": digests
        }
    # Checkout identity is the pinned source lock; local location is diagnostic only.
    lock_path = definition.get("upstream_lock")
    upstream = None
    if lock_path:
        source = Path(project_root) / str(lock_path)
        if not source.is_file():
            source = package_resource_path(str(lock_path))
        upstream = load_yaml(source)
        constructor.pop("checkout_path", None)
    constructor.pop("device", None)
    git = collect_git_provenance(project_root)
    return {
        "adapter": definition["adapter"],
        "target": definition.get("adapter_target"),
        "constructor": constructor,
        "checkpoint": checkpoint,
        "upstream": upstream,
        "code": {key: git[key] for key in ("commit", "source_sha256") if key in git},
    }


def _slice_metadata(value: Any, start: int, stop: int, batch_size: int) -> Any:
    if isinstance(value, (str, bytes, Mapping)):
        return value
    if isinstance(value, Sequence) and len(value) == batch_size:
        sliced = value[start:stop]
        return tuple(sliced) if isinstance(value, tuple) else list(sliced)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == batch_size:
        return value[start:stop]
    return value


def slice_clip_batch(batch: ClipBatch, start: int, stop: int) -> ClipBatch:
    if not 0 <= start < stop <= batch.batch_size:
        raise ValueError(f"非法 batch slice [{start}:{stop}] / {batch.batch_size}")
    metadata = {
        key: _slice_metadata(value, start, stop, batch.batch_size)
        for key, value in batch.metadata.items()
    }
    return ClipBatch(
        frames=batch.frames[start:stop],
        timestamps_s=batch.timestamps_s[start:stop],
        video_ids=batch.video_ids[start:stop],
        valid_mask=None if batch.valid_mask is None else batch.valid_mask[start:stop],
        frame_indices=None if batch.frame_indices is None else batch.frame_indices[start:stop],
        metadata=metadata,
    )


def iter_microbatches(batches: Iterator[ClipBatch], micro_batch_size: int) -> Iterator[ClipBatch]:
    if isinstance(micro_batch_size, bool) or micro_batch_size <= 0:
        raise ValueError("micro_batch_size 必须是正整数")
    for batch in batches:
        for start in range(0, batch.batch_size, micro_batch_size):
            yield slice_clip_batch(batch, start, min(start + micro_batch_size, batch.batch_size))


def compression_from_experiment(config: Mapping[str, Any]) -> Any | None:
    streaming = config.get("streaming", {})
    compression = streaming.get("compression", {}) if isinstance(streaming, Mapping) else {}
    if not isinstance(compression, Mapping):
        raise ValueError("streaming.compression 必须是对象")
    name = str(compression.get("policy", "identity"))
    if name in {"hermes_native", "native"}:
        return None
    max_tokens = compression.get("max_tokens")
    if max_tokens is None and name == "keep_recent":
        max_tokens = compression.get("kv_budget_tokens")
    return build_cache_policy(name, max_tokens=None if max_tokens is None else int(max_tokens))


__all__ = [
    "BUILTIN_ENCODER_CONFIGS",
    "compression_from_experiment",
    "create_encoder_from_experiment",
    "resolve_encoder_config",
    "encoder_identity",
    "iter_microbatches",
    "load_encoder_definition",
    "slice_clip_batch",
]
