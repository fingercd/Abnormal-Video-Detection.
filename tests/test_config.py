from __future__ import annotations

from dataclasses import dataclass

import pytest

from vadbench.config import ConfigError, load_experiment, validate_capabilities


@dataclass
class FakeCapabilities:
    supports_streaming: bool = False
    supports_grad: bool = True
    cache_kinds: tuple[str, ...] = ()
    cache_access: str = "none"


def test_load_reference_experiment() -> None:
    config = load_experiment("configs/experiments/ucf_videomaev2_weak.yaml")
    assert config["task"]["supervision"] == "video"
    assert config["sampler"]["segments_per_video"] == 32


def test_streaming_capability_is_not_silently_downgraded() -> None:
    config = load_experiment("configs/experiments/ucf_hermes_stream.yaml")
    with pytest.raises(ConfigError, match="不支持增量状态"):
        validate_capabilities(config, FakeCapabilities())


def test_cache_replacement_requires_replace_access() -> None:
    config = load_experiment("configs/experiments/ucf_hermes_stream.yaml")
    config["streaming"]["compression"]["replace"] = True
    capabilities = FakeCapabilities(
        supports_streaming=True,
        cache_kinds=("decoder_kv",),
        cache_access="read",
    )
    with pytest.raises(ConfigError, match="请求替换缓存"):
        validate_capabilities(config, capabilities)
