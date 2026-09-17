"""Lightweight, fail-closed catalog for all encoder integration targets.

The catalog stores import strings and static capability metadata only. Loading
or listing it must never import Torch, Transformers, or an upstream checkout.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from vadbench.config import load_yaml
from vadbench.contracts import EncoderCapabilities
from vadbench.registry import EncoderRegistry
from vadbench.resources import package_resource_path

CATALOG_SCHEMA_VERSION = 1

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_RELATIVE_YAML_PATH = re.compile(
    r"^(?!/)(?!.*\\)(?!.*//)(?!.*(?:^|/)\.{1,2}(?:/|$))[A-Za-z0-9._/-]+\.ya?ml$"
)
_DOTTED_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_ADAPTER_TARGET = re.compile(rf"^{_DOTTED_IDENTIFIER}:{_DOTTED_IDENTIFIER}$")
_STATUSES = frozenset({"planned", "integrated", "blocked"})
_RUN_MODES = frozenset({"fixed", "long", "streaming"})
_RUNTIMES = frozenset({"in_process", "external_python"})
_CHECKPOINT_STATUSES = frozenset({"planned", "registered", "verified"})
_FEATURE_STAGES = frozenset(
    {
        "pooled",
        "fc_features",
        "backbone_tokens",
        "last_hidden_state",
        "observed_backbone",
        "projected_visual",
        "visual_memory",
        "decoder_contextual",
    }
)
_IMPLEMENTED_STATUSES = frozenset({"integrated"})
_CAPABILITY_KEYS = frozenset(
    {
        "supports_fixed_clip",
        "supports_streaming",
        "supports_kv_cache",
        "supports_token_cache",
        "supports_visual_memory_cache",
        "supports_external_cache_policy",
        "supports_training",
        "fixed_num_frames",
        "min_frames",
        "max_frames",
    }
)
_INTEGRATION_KEYS = frozenset(
    {
        "id",
        "display_name",
        "family",
        "status",
        "definition",
        "adapter_target",
        "backend",
        "run_mode",
        "feature_stage",
        "environment",
        "upstream_lock",
        "checkpoint",
        "smoke_profile",
        "capabilities",
    }
)


class IntegrationCatalogError(ValueError):
    """Raised when catalog structure, references, or registrations are invalid."""


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrationCatalogError(f"{context} 必须是对象")
    if any(not isinstance(key, str) for key in value):
        raise IntegrationCatalogError(f"{context} 的键必须是字符串")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], context: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"缺少字段 {missing}")
        if unknown:
            details.append(f"未知字段 {unknown}")
        raise IntegrationCatalogError(f"{context} 结构非法：{'；'.join(details)}")


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationCatalogError(f"{context} 必须是非空字符串")
    return value.strip()


def _identifier(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    if _IDENTIFIER.fullmatch(text) is None:
        raise IntegrationCatalogError(f"{context} 不是合法标识符：{text!r}")
    return text


def _choice(value: Any, choices: frozenset[str], context: str) -> str:
    text = _nonempty_string(value, context)
    if text not in choices:
        raise IntegrationCatalogError(f"{context}={text!r} 不在 {sorted(choices)} 中")
    return text


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise IntegrationCatalogError(f"{context} 必须是正整数")
    return value


def _relative_yaml_path(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    if _RELATIVE_YAML_PATH.fullmatch(text) is None:
        raise IntegrationCatalogError(
            f"{context} 必须是不可越界的规范 POSIX 相对 YAML 路径：{text!r}"
        )
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise IntegrationCatalogError(f"{context} 必须是不可越界的相对路径：{text!r}")
    return path.as_posix()


def _adapter_target(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    if _ADAPTER_TARGET.fullmatch(text) is None:
        raise IntegrationCatalogError(
            f"{context} 必须采用规范 ASCII 'module:attribute' 格式：{text!r}"
        )
    return text


@dataclass(frozen=True, slots=True)
class IntegrationEnvironment:
    runtime: str
    profile: str


@dataclass(frozen=True, slots=True)
class CheckpointReference:
    registry_id: str
    status: str


@dataclass(frozen=True, slots=True)
class SmokeProfile:
    profile: str
    clip_frames: int
    frame_stride: int
    image_size: int
    chunks: int


@dataclass(frozen=True, slots=True)
class IntegrationRecord:
    id: str
    display_name: str
    family: str
    status: str
    definition: str
    adapter_target: str
    backend: str
    run_mode: str
    feature_stage: str
    environment: IntegrationEnvironment
    upstream_lock: str
    checkpoint: CheckpointReference
    smoke_profile: SmokeProfile
    capabilities: EncoderCapabilities

    def registry_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "display_name": self.display_name,
            "family": self.family,
            "status": self.status,
            "definition": self.definition,
            "backend": self.backend,
            "run_mode": self.run_mode,
            "feature_stage": self.feature_stage,
            "runtime": self.environment.runtime,
            "environment_profile": self.environment.profile,
            "upstream_lock": self.upstream_lock,
            "checkpoint_id": self.checkpoint.registry_id,
            "checkpoint_status": self.checkpoint.status,
            "smoke_profile": self.smoke_profile.profile,
        }
        if self.capabilities.supports_kv_cache:
            metadata["cache_owner"] = "language_model_decoder"
        elif self.capabilities.supports_visual_memory_cache:
            metadata["cache_owner"] = "visual_memory"
        return metadata


@dataclass(frozen=True, slots=True)
class IntegrationCatalog:
    schema_version: int
    integrations: tuple[IntegrationRecord, ...]
    source_path: Path
    _by_id: Mapping[str, IntegrationRecord] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lookup: dict[str, IntegrationRecord] = {}
        for record in self.integrations:
            if record.id in lookup:
                raise IntegrationCatalogError(f"integration id 重复：{record.id!r}")
            lookup[record.id] = record
        object.__setattr__(self, "_by_id", MappingProxyType(lookup))

    def __len__(self) -> int:
        return len(self.integrations)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(record.id for record in self.integrations)

    def get(self, integration_id: str) -> IntegrationRecord:
        try:
            return self._by_id[integration_id]
        except KeyError as exc:
            available = ", ".join(self.ids)
            raise IntegrationCatalogError(
                f"catalog 中不存在 integration {integration_id!r}；可用项：{available}"
            ) from exc

    def definition_paths(self) -> Mapping[str, str]:
        return MappingProxyType({record.id: record.definition for record in self.integrations})

    def validate_references(self, project_root: str | Path) -> None:
        root = Path(project_root).resolve()
        checkpoint_ids: frozenset[str] | None = None
        for record in self.integrations:
            if record.status in _IMPLEMENTED_STATUSES:
                for label, relative in (
                    ("definition", record.definition),
                    ("upstream_lock", record.upstream_lock),
                ):
                    resolved = (root / relative).resolve()
                    if root != resolved and root not in resolved.parents:
                        raise IntegrationCatalogError(
                            f"{record.id}.{label} 解析后越出项目目录：{resolved}"
                        )
                    if not resolved.is_file():
                        raise IntegrationCatalogError(
                            f"{record.id}.{label} 声明为 {record.status} 但文件不存在：{resolved}"
                        )
            if record.checkpoint.status != "planned":
                if checkpoint_ids is None:
                    registry_path = root / "registry" / "checkpoints.yaml"
                    registry_data = load_yaml(registry_path)
                    checkpoints = _mapping(
                        registry_data.get("checkpoints"),
                        f"{registry_path}: checkpoints",
                    )
                    checkpoint_ids = frozenset(str(key) for key in checkpoints)
                if record.checkpoint.registry_id not in checkpoint_ids:
                    raise IntegrationCatalogError(
                        f"{record.id}.checkpoint.registry_id 未登记："
                        f"{record.checkpoint.registry_id!r}"
                    )


def _parse_environment(value: Any, context: str) -> IntegrationEnvironment:
    data = _mapping(value, context)
    _exact_keys(data, frozenset({"runtime", "profile"}), context)
    return IntegrationEnvironment(
        runtime=_choice(data["runtime"], _RUNTIMES, f"{context}.runtime"),
        profile=_identifier(data["profile"], f"{context}.profile"),
    )


def _parse_checkpoint(value: Any, context: str) -> CheckpointReference:
    data = _mapping(value, context)
    _exact_keys(data, frozenset({"registry_id", "status"}), context)
    return CheckpointReference(
        registry_id=_identifier(data["registry_id"], f"{context}.registry_id"),
        status=_choice(data["status"], _CHECKPOINT_STATUSES, f"{context}.status"),
    )


def _parse_smoke_profile(value: Any, context: str) -> SmokeProfile:
    data = _mapping(value, context)
    _exact_keys(
        data,
        frozenset({"profile", "clip_frames", "frame_stride", "image_size", "chunks"}),
        context,
    )
    return SmokeProfile(
        profile=_identifier(data["profile"], f"{context}.profile"),
        clip_frames=_positive_int(data["clip_frames"], f"{context}.clip_frames"),
        frame_stride=_positive_int(data["frame_stride"], f"{context}.frame_stride"),
        image_size=_positive_int(data["image_size"], f"{context}.image_size"),
        chunks=_positive_int(data["chunks"], f"{context}.chunks"),
    )


def _parse_capabilities(value: Any, context: str) -> EncoderCapabilities:
    data = _mapping(value, context)
    _exact_keys(data, _CAPABILITY_KEYS, context)
    try:
        return EncoderCapabilities(**{key: data[key] for key in _CAPABILITY_KEYS})
    except (TypeError, ValueError) as exc:
        raise IntegrationCatalogError(f"{context} 不满足 EncoderCapabilities：{exc}") from exc


def _parse_record(value: Any, index: int) -> IntegrationRecord:
    context = f"integrations[{index}]"
    data = _mapping(value, context)
    _exact_keys(data, _INTEGRATION_KEYS, context)
    record = IntegrationRecord(
        id=_identifier(data["id"], f"{context}.id"),
        display_name=_nonempty_string(data["display_name"], f"{context}.display_name"),
        family=_identifier(data["family"], f"{context}.family"),
        status=_choice(data["status"], _STATUSES, f"{context}.status"),
        definition=_relative_yaml_path(data["definition"], f"{context}.definition"),
        adapter_target=_adapter_target(data["adapter_target"], f"{context}.adapter_target"),
        backend=_identifier(data["backend"], f"{context}.backend"),
        run_mode=_choice(data["run_mode"], _RUN_MODES, f"{context}.run_mode"),
        feature_stage=_choice(data["feature_stage"], _FEATURE_STAGES, f"{context}.feature_stage"),
        environment=_parse_environment(data["environment"], f"{context}.environment"),
        upstream_lock=_relative_yaml_path(data["upstream_lock"], f"{context}.upstream_lock"),
        checkpoint=_parse_checkpoint(data["checkpoint"], f"{context}.checkpoint"),
        smoke_profile=_parse_smoke_profile(data["smoke_profile"], f"{context}.smoke_profile"),
        capabilities=_parse_capabilities(data["capabilities"], f"{context}.capabilities"),
    )
    if record.run_mode == "streaming" and not record.capabilities.supports_streaming:
        raise IntegrationCatalogError(f"{record.id}: streaming 目标必须声明 supports_streaming")
    if record.run_mode != "streaming" and not record.capabilities.supports_fixed_clip:
        raise IntegrationCatalogError(f"{record.id}: fixed/long 目标必须声明 supports_fixed_clip")
    if record.run_mode == "streaming" and record.smoke_profile.chunks < 2:
        raise IntegrationCatalogError(f"{record.id}: streaming smoke 至少需要 2 个 chunks")
    if record.status in _IMPLEMENTED_STATUSES and record.checkpoint.status == "planned":
        raise IntegrationCatalogError(f"{record.id}: {record.status} 不能引用 planned checkpoint")
    return record


def default_project_root() -> Path:
    """Return the source-checkout root without relying on the process CWD."""

    return Path(__file__).resolve().parents[3]


def default_catalog_path(project_root: str | Path | None = None) -> Path:
    root = default_project_root() if project_root is None else Path(project_root).resolve()
    checkout_catalog = root / "registry" / "encoder-integrations.yaml"
    if checkout_catalog.is_file():
        return checkout_catalog.resolve()

    try:
        return package_resource_path("registry/encoder-integrations.yaml")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise IntegrationCatalogError(
            f"未找到 encoder integration catalog：checkout={checkout_catalog}"
        ) from exc


def load_integration_catalog(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    validate_references: bool = True,
) -> IntegrationCatalog:
    """Load and validate one catalog without resolving any adapter target."""

    catalog_path = Path(path).resolve()
    data = load_yaml(catalog_path)
    _exact_keys(data, frozenset({"schema_version", "integrations"}), str(catalog_path))
    if data["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise IntegrationCatalogError(
            f"仅支持 catalog schema_version={CATALOG_SCHEMA_VERSION}，"
            f"实际为 {data['schema_version']!r}"
        )
    raw_integrations = data["integrations"]
    if not isinstance(raw_integrations, list):
        raise IntegrationCatalogError("integrations 必须是列表")
    if not raw_integrations:
        raise IntegrationCatalogError("integrations 不能为空")
    catalog = IntegrationCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        integrations=tuple(
            _parse_record(raw_record, index) for index, raw_record in enumerate(raw_integrations)
        ),
        source_path=catalog_path,
    )
    if validate_references:
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else catalog_path.parent.parent.resolve()
        )
        catalog.validate_references(root)
    return catalog


def load_default_integration_catalog(
    project_root: str | Path | None = None,
) -> IntegrationCatalog:
    root = default_project_root() if project_root is None else Path(project_root).resolve()
    checkout_catalog = (root / "registry" / "encoder-integrations.yaml").resolve()
    selected = default_catalog_path(root)
    validation_root = root if selected == checkout_catalog else selected.parent.parent
    return load_integration_catalog(
        selected,
        project_root=validation_root,
        validate_references=True,
    )


def register_catalog_integrations(
    catalog: IntegrationCatalog,
    registry: EncoderRegistry,
) -> None:
    """Register all import strings idempotently while rejecting capability drift."""

    for record in catalog.integrations:
        if record.id in registry:
            existing = registry.get_spec(record.id)
            if (
                existing.target_path != record.adapter_target
                or existing.capabilities != record.capabilities
            ):
                raise IntegrationCatalogError(
                    f"registry 中 {record.id!r} 与 catalog target/capabilities 不一致"
                )
            continue
        registry.register_lazy(
            record.id,
            record.adapter_target,
            capabilities=record.capabilities,
            metadata=record.registry_metadata(),
        )


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CheckpointReference",
    "IntegrationCatalog",
    "IntegrationCatalogError",
    "IntegrationEnvironment",
    "IntegrationRecord",
    "SmokeProfile",
    "default_catalog_path",
    "default_project_root",
    "load_default_integration_catalog",
    "load_integration_catalog",
    "register_catalog_integrations",
]
