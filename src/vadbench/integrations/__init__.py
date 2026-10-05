"""Lazy registrations for optional video-model integrations.

Importing this package only reads the lightweight YAML catalog and installs
``module:attribute`` strings. Model libraries and upstream checkouts are not
imported until a specific registry entry is instantiated.
"""

from __future__ import annotations

from typing import Any

from vadbench.integrations.catalog import (
    IntegrationCatalog,
    load_default_integration_catalog,
    register_catalog_integrations,
)
from vadbench.registry import ENCODER_REGISTRY

DEFAULT_INTEGRATION_CATALOG: IntegrationCatalog = load_default_integration_catalog()

# Compatibility exports used by existing adapter tests and downstream code.
VIDEOMAEV2_CAPABILITIES = DEFAULT_INTEGRATION_CATALOG.get("videomaev2").capabilities
HERMES_LLAVA_OV_CAPABILITIES = DEFAULT_INTEGRATION_CATALOG.get("hermes_llava_ov").capabilities


def register_builtin_integrations(registry: Any = ENCODER_REGISTRY) -> None:
    """Install all catalog targets idempotently without resolving adapters."""

    register_catalog_integrations(DEFAULT_INTEGRATION_CATALOG, registry)


register_builtin_integrations()


__all__ = [
    "DEFAULT_INTEGRATION_CATALOG",
    "HERMES_LLAVA_OV_CAPABILITIES",
    "VIDEOMAEV2_CAPABILITIES",
    "register_builtin_integrations",
]
