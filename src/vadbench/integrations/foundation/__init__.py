"""Lazy-friendly exports for video foundation integrations."""

from __future__ import annotations

from .base import (
    FOUNDATION_CAPABILITIES,
    ExternalPythonFoundationBridge,
    FoundationAssetError,
    FoundationIntegrationError,
    FoundationUpstreamError,
    FoundationVideoAdapter,
    InProcessFoundationBridge,
    LazyFoundationBridge,
)
from .internvideo2 import InternVideo2Adapter
from .videomamba import VideoMambaAdapter
from .vjepa2 import VJEPA2Adapter

__all__ = [
    "ExternalPythonFoundationBridge",
    "FOUNDATION_CAPABILITIES",
    "FoundationAssetError",
    "FoundationIntegrationError",
    "FoundationUpstreamError",
    "FoundationVideoAdapter",
    "InProcessFoundationBridge",
    "InternVideo2Adapter",
    "LazyFoundationBridge",
    "VJEPA2Adapter",
    "VideoMambaAdapter",
]
