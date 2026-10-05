"""Thread-safe lazy registry for video encoder adapters."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar

from .contracts import EncoderCapabilities, validate_encoder_adapter


class RegistryError(ValueError):
    """Base error for invalid registry operations."""


class DuplicateEncoderError(RegistryError):
    """Raised when an encoder name is registered more than once."""


class UnknownEncoderError(RegistryError, KeyError):
    """Raised when an encoder name is not present."""


class EncoderLoadError(RegistryError, ImportError):
    """Raised when a lazy target cannot be imported or instantiated."""


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise RegistryError("encoder name 必须是非空字符串")
    normalized = name.strip().lower()
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-." for character in normalized):
        raise RegistryError("encoder name 只能包含 ASCII 小写字母、数字、下划线、短横线和点")
    return normalized


def _freeze_string_mapping(value: Mapping[str, Any] | None, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise RegistryError(f"{field_name} 必须是 mapping")
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise RegistryError(f"{field_name} 的键必须是非空字符串")
        copied[key] = item
    return MappingProxyType(copied)


def _validate_target_path(target: str) -> str:
    if not isinstance(target, str):
        raise RegistryError("lazy target 必须是 'module:attribute' 字符串")
    module_name, separator, attribute_path = target.partition(":")
    if not separator or not module_name or not attribute_path:
        raise RegistryError(f"lazy target 必须采用 'module:attribute' 格式，实际为 {target!r}")
    if any(not component.isidentifier() for component in module_name.split(".")):
        raise RegistryError(f"lazy target module 非法：{module_name!r}")
    if any(not component.isidentifier() for component in attribute_path.split(".")):
        raise RegistryError(f"lazy target attribute 非法：{attribute_path!r}")
    return target


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Registration metadata that is safe to inspect without importing a model."""

    name: str
    target: str | Callable[..., Any]
    capabilities: EncoderCapabilities
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        if isinstance(self.target, str):
            object.__setattr__(self, "target", _validate_target_path(self.target))
        elif not callable(self.target):
            raise RegistryError("target 必须是 callable 或 'module:attribute' 字符串")
        if not isinstance(self.capabilities, EncoderCapabilities):
            raise RegistryError("capabilities 必须是 EncoderCapabilities")
        object.__setattr__(
            self,
            "default_kwargs",
            _freeze_string_mapping(self.default_kwargs, "default_kwargs"),
        )
        object.__setattr__(self, "metadata", _freeze_string_mapping(self.metadata, "metadata"))

    @property
    def is_lazy(self) -> bool:
        return isinstance(self.target, str)

    @property
    def target_path(self) -> str | None:
        return self.target if isinstance(self.target, str) else None


FactoryT = TypeVar("FactoryT", bound=Callable[..., Any])


class EncoderRegistry:
    """Register, discover and instantiate adapters without eager model imports."""

    def __init__(self) -> None:
        self._specs: dict[str, EncoderSpec] = {}
        self._resolved: dict[str, Callable[..., Any]] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            normalized = _validate_name(name)
        except RegistryError:
            return False
        with self._lock:
            return normalized in self._specs

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def names(self) -> tuple[str, ...]:
        """Return stable sorted names without resolving lazy targets."""

        with self._lock:
            return tuple(sorted(self._specs))

    def specs(self) -> tuple[EncoderSpec, ...]:
        """Return stable sorted specs without resolving lazy targets."""

        with self._lock:
            return tuple(self._specs[name] for name in sorted(self._specs))

    def get_spec(self, name: str) -> EncoderSpec:
        normalized = _validate_name(name)
        with self._lock:
            try:
                return self._specs[normalized]
            except KeyError as exc:
                available = ", ".join(sorted(self._specs)) or "<empty>"
                raise UnknownEncoderError(
                    f"未注册 encoder {normalized!r}；可用项：{available}"
                ) from exc

    def _install(self, spec: EncoderSpec, *, replace: bool) -> EncoderSpec:
        with self._lock:
            if spec.name in self._specs and not replace:
                raise DuplicateEncoderError(f"encoder {spec.name!r} 已注册")
            self._specs[spec.name] = spec
            # A replacement must never reuse the previous resolved callable.
            self._resolved.pop(spec.name, None)
        return spec

    def register_lazy(
        self,
        name: str,
        target: str,
        *,
        capabilities: EncoderCapabilities,
        default_kwargs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> EncoderSpec:
        """Register an import string without importing its module."""

        spec = EncoderSpec(
            name=name,
            target=target,
            capabilities=capabilities,
            default_kwargs=default_kwargs or {},
            metadata=metadata or {},
        )
        return self._install(spec, replace=replace)

    def register_factory(
        self,
        name: str,
        factory: Callable[..., Any],
        *,
        capabilities: EncoderCapabilities | None = None,
        default_kwargs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> EncoderSpec:
        """Register an already-imported callable.

        Capability metadata may be inferred from a class-level ``capabilities``
        attribute.  It is never inferred by instantiating the factory.
        """

        declared = capabilities or getattr(factory, "capabilities", None)
        if not isinstance(declared, EncoderCapabilities):
            raise RegistryError("direct factory 必须显式传 capabilities，或在类/函数上声明该属性")
        spec = EncoderSpec(
            name=name,
            target=factory,
            capabilities=declared,
            default_kwargs=default_kwargs or {},
            metadata=metadata or {},
        )
        return self._install(spec, replace=replace)

    def register(
        self,
        name: str,
        factory: FactoryT | None = None,
        *,
        capabilities: EncoderCapabilities | None = None,
        default_kwargs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> FactoryT | Callable[[FactoryT], FactoryT]:
        """Register a direct factory, usable as a decorator or normal method."""

        def decorator(resolved: FactoryT) -> FactoryT:
            self.register_factory(
                name,
                resolved,
                capabilities=capabilities,
                default_kwargs=default_kwargs,
                metadata=metadata,
                replace=replace,
            )
            return resolved

        if factory is None:
            return decorator
        return decorator(factory)

    def unregister(self, name: str) -> EncoderSpec:
        normalized = _validate_name(name)
        with self._lock:
            try:
                spec = self._specs.pop(normalized)
            except KeyError as exc:
                raise UnknownEncoderError(f"未注册 encoder {normalized!r}") from exc
            self._resolved.pop(normalized, None)
            return spec

    def clear(self) -> None:
        with self._lock:
            self._specs.clear()
            self._resolved.clear()

    @staticmethod
    def _import_target(path: str) -> Callable[..., Any]:
        module_name, _, attribute_path = path.partition(":")
        try:
            value: Any = importlib.import_module(module_name)
            for component in attribute_path.split("."):
                value = getattr(value, component)
        except (ImportError, AttributeError) as exc:
            raise EncoderLoadError(f"无法加载 encoder target {path!r}: {exc}") from exc
        if not callable(value):
            raise EncoderLoadError(f"encoder target {path!r} 不是 callable")
        return value

    def load_factory(self, name: str) -> Callable[..., Any]:
        """Resolve one target once; list/get operations remain import-free."""

        spec = self.get_spec(name)
        with self._lock:
            cached = self._resolved.get(spec.name)
            if cached is not None:
                return cached
            factory = (
                self._import_target(spec.target) if isinstance(spec.target, str) else spec.target
            )
            self._resolved[spec.name] = factory
            return factory

    # A concise alias reads naturally in dependency-injection code.
    resolve = load_factory

    def create(self, name: str, /, **kwargs: Any) -> Any:
        """Instantiate and capability-check one registered adapter."""

        spec = self.get_spec(name)
        factory = self.load_factory(name)
        parameters = dict(spec.default_kwargs)
        parameters.update(kwargs)
        try:
            adapter = factory(**parameters)
        except Exception as exc:
            raise EncoderLoadError(
                f"实例化 encoder {spec.name!r} ({spec.target!r}) 失败: {exc}"
            ) from exc
        try:
            validate_encoder_adapter(adapter, expected=spec.capabilities)
        except Exception as exc:
            raise EncoderLoadError(f"encoder {spec.name!r} 未满足注册契约: {exc}") from exc
        return adapter


ENCODER_REGISTRY = EncoderRegistry()


def register_encoder(
    name: str,
    *,
    capabilities: EncoderCapabilities | None = None,
    default_kwargs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    replace: bool = False,
) -> Callable[[FactoryT], FactoryT]:
    """Decorator backed by the process-global encoder registry."""

    result = ENCODER_REGISTRY.register(
        name,
        capabilities=capabilities,
        default_kwargs=default_kwargs,
        metadata=metadata,
        replace=replace,
    )
    assert callable(result)
    return result


def create_encoder(name: str, /, **kwargs: Any) -> Any:
    return ENCODER_REGISTRY.create(name, **kwargs)


__all__ = [
    "DuplicateEncoderError",
    "ENCODER_REGISTRY",
    "EncoderLoadError",
    "EncoderRegistry",
    "EncoderSpec",
    "RegistryError",
    "UnknownEncoderError",
    "create_encoder",
    "register_encoder",
]
