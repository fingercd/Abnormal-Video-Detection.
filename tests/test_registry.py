from __future__ import annotations

import sys

import pytest

from vadbench.contracts import EncoderCapabilities
from vadbench.registry import (
    DuplicateEncoderError,
    EncoderLoadError,
    EncoderRegistry,
    RegistryError,
    UnknownEncoderError,
)

FIXED_CAPABILITIES = EncoderCapabilities(supports_fixed_clip=True)


class FixedAdapter:
    capabilities = FIXED_CAPABILITIES

    def __init__(self, *, scale: int = 1) -> None:
        self.scale = scale

    def encode(self, batch, train: bool = False):  # pragma: no cover - contract stub
        raise NotImplementedError


def test_direct_registry_supports_factory_and_decorator_forms() -> None:
    registry = EncoderRegistry()
    registry.register_factory(
        "demo.fixed",
        FixedAdapter,
        default_kwargs={"scale": 2},
        metadata={"family": "test"},
    )

    @registry.register("demo.decorated", capabilities=FIXED_CAPABILITIES)
    class DecoratedAdapter(FixedAdapter):
        pass

    assert registry.names() == ("demo.decorated", "demo.fixed")
    assert registry.get_spec("DEMO.FIXED").metadata["family"] == "test"
    assert registry.get_spec("demo.fixed").is_lazy is False
    assert registry.create("demo.fixed").scale == 2
    assert registry.create("demo.fixed", scale=7).scale == 7
    assert isinstance(registry.create("demo.decorated"), DecoratedAdapter)


def test_registry_rejects_duplicates_unknown_names_and_bad_lazy_paths() -> None:
    registry = EncoderRegistry()
    registry.register_factory("demo", FixedAdapter)
    with pytest.raises(DuplicateEncoderError):
        registry.register_factory("demo", FixedAdapter)
    with pytest.raises(UnknownEncoderError, match="可用项"):
        registry.get_spec("missing")
    with pytest.raises(RegistryError, match="module:attribute"):
        registry.register_lazy(
            "bad",
            "missing_separator",
            capabilities=FIXED_CAPABILITIES,
        )


def test_lazy_registration_does_not_import_until_factory_is_loaded(tmp_path, monkeypatch) -> None:
    module_name = "vadbench_test_lazy_adapter"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "\n".join(
            [
                "from vadbench.contracts import EncoderCapabilities",
                "class LazyAdapter:",
                "    capabilities = EncoderCapabilities(supports_fixed_clip=True)",
                "    def __init__(self, marker='default'):",
                "        self.marker = marker",
                "    def encode(self, batch, train=False):",
                "        raise NotImplementedError",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    registry = EncoderRegistry()
    spec = registry.register_lazy(
        "lazy",
        f"{module_name}:LazyAdapter",
        capabilities=FIXED_CAPABILITIES,
        default_kwargs={"marker": "from-default"},
    )

    assert spec.is_lazy is True
    assert registry.names() == ("lazy",)
    assert registry.get_spec("lazy") is spec
    assert module_name not in sys.modules

    adapter = registry.create("lazy")
    assert module_name in sys.modules
    assert adapter.marker == "from-default"
    assert registry.load_factory("lazy") is registry.load_factory("lazy")


def test_create_rejects_capability_drift_from_lazy_metadata(tmp_path, monkeypatch) -> None:
    module_name = "vadbench_test_capability_drift"
    (tmp_path / f"{module_name}.py").write_text(
        "\n".join(
            [
                "from vadbench.contracts import EncoderCapabilities",
                "class DriftedAdapter:",
                "    capabilities = EncoderCapabilities(",
                "        supports_fixed_clip=True, supports_training=True)",
                "    def encode(self, batch, train=False):",
                "        raise NotImplementedError",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)
    registry = EncoderRegistry()
    registry.register_lazy(
        "drifted",
        f"{module_name}:DriftedAdapter",
        capabilities=FIXED_CAPABILITIES,
    )

    with pytest.raises(EncoderLoadError, match="capabilities"):
        registry.create("drifted")


def test_replace_invalidates_resolved_factory_cache() -> None:
    registry = EncoderRegistry()
    registry.register_factory("demo", FixedAdapter)
    first = registry.load_factory("demo")

    class Replacement(FixedAdapter):
        pass

    registry.register_factory("demo", Replacement, replace=True)
    second = registry.load_factory("demo")
    assert first is FixedAdapter
    assert second is Replacement
