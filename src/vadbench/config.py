"""实验配置加载、覆盖和跨模块约束校验。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """配置结构或能力协商失败。"""


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    """以 UTF-8 读取 YAML，并要求顶层为 mapping。"""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"配置顶层必须是对象：{config_path}")
    return dict(data)


def load_experiment(
    path: str | Path,
    *,
    defaults: str | Path | None = None,
) -> dict[str, Any]:
    """读取实验配置，可选地以另一份 YAML 作为默认值。"""

    data = load_yaml(path)
    if defaults is not None:
        data = _merge(load_yaml(defaults), data)
    validate_experiment_shape(data)
    return data


def validate_experiment_shape(config: Mapping[str, Any]) -> None:
    """做不依赖具体 encoder 的最小结构校验。"""

    version = config.get("schema_version")
    if version != 1:
        raise ConfigError(f"仅支持 schema_version=1，实际为 {version!r}")
    for section in ("dataset", "encoder", "task", "output"):
        value = config.get(section)
        if not isinstance(value, Mapping):
            raise ConfigError(f"缺少对象配置段：{section}")

    validate_encoder_config(config["encoder"])

    supervision = str(config["task"].get("supervision", ""))
    if supervision not in {"video", "segment", "frame"}:
        raise ConfigError("task.supervision 必须是 video、segment 或 frame")

    streaming = config.get("streaming", {})
    if not isinstance(streaming, Mapping):
        raise ConfigError("streaming 必须是对象")
    if streaming.get("enabled") and int(streaming.get("chunk_frames", 0)) <= 0:
        raise ConfigError("启用 streaming 时 chunk_frames 必须大于 0")


def validate_encoder_config(config: Mapping[str, Any]) -> None:
    """Keep experiment overrides explicit; model-specific options belong to params."""

    allowed = {
        "adapter",
        "definition",
        "checkpoint",
        "trainable",
        "device",
        "params",
        "micro_batch_size",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ConfigError(
            f"未知 encoder 字段：{sorted(unknown)}；模型和精度参数请写入 encoder.params，"
            "权重用 encoder.checkpoint 选择 registry 条目"
        )
    if not isinstance(config.get("params", {}), Mapping):
        raise ConfigError("encoder.params 必须是对象")


def validate_capabilities(config: Mapping[str, Any], capabilities: Any) -> None:
    """以 duck typing 校验配置需求，避免 config 层硬依赖某个 adapter 实现。"""

    streaming = config.get("streaming", {})
    compression = streaming.get("compression", {})
    cache_kind = compression.get("cache_kind")
    supports_streaming = bool(getattr(capabilities, "supports_streaming", False))
    supports_grad = bool(getattr(capabilities, "supports_grad", False))
    if streaming.get("enabled") and not supports_streaming:
        raise ConfigError("实验请求 streaming，但所选 encoder 不支持增量状态")
    if config["encoder"].get("trainable") and not supports_grad:
        raise ConfigError("实验请求端到端训练，但所选 encoder 不支持梯度")

    if cache_kind:
        cache_kind = str(cache_kind)
        cache_kinds = {
            str(getattr(item, "value", item)) for item in getattr(capabilities, "cache_kinds", ())
        }
        if cache_kind not in cache_kinds:
            raise ConfigError(
                f"实验请求 cache_kind={cache_kind!r}，adapter 仅声明 {sorted(cache_kinds)}"
            )
    if compression.get("replace"):
        cache_access = getattr(capabilities, "cache_access", "none")
        access = str(getattr(cache_access, "value", cache_access))
        if access != "replace":
            raise ConfigError(f"实验请求替换缓存，但 adapter cache_access={access!r}")
