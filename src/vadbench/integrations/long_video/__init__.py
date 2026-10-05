"""Lazy-friendly long-video/VLM integration declarations.

Importing this package only imports the dependency-free adapter facade.  The
actual upstream model packages are loaded only when an adapter is constructed
with an explicit local loader, checkout entrypoint, or injected worker.
"""

from .base import (
    DEFAULT_NEUTRAL_PROMPT,
    ExternalAssetError,
    ExternalFixedVideoAdapter,
    ExternalStreamingVideoAdapter,
    ExternalWorkerError,
    LongVideoAssetError,
    LongVideoWorkerError,
    MissingAssetError,
    MissingLongVideoAssetError,
    StructuredLongVideoError,
)
from .longvu import LongVUAdapter
from .ma_lmm import MALMMAdapter
from .moviechat import MovieChatAdapter
from .streaming_vlm import StreamingVLMAdapter
from .videochat import VideoChatAdapter
from .videochat_flash import VideoChatFlashAdapter
from .videochat_online import VideoChatOnlineAdapter

__all__ = [
    "DEFAULT_NEUTRAL_PROMPT",
    "ExternalAssetError",
    "ExternalFixedVideoAdapter",
    "ExternalStreamingVideoAdapter",
    "ExternalWorkerError",
    "LongVUAdapter",
    "LongVideoAssetError",
    "LongVideoWorkerError",
    "MALMMAdapter",
    "MissingAssetError",
    "MissingLongVideoAssetError",
    "StructuredLongVideoError",
    "MovieChatAdapter",
    "StreamingVLMAdapter",
    "VideoChatAdapter",
    "VideoChatFlashAdapter",
    "VideoChatOnlineAdapter",
]
