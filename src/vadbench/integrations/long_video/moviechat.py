"""MovieChat visual-memory integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalStreamingVideoAdapter


class MovieChatAdapter(ExternalStreamingVideoAdapter):
    """Expose MovieChat short/long-term memory through StreamState."""

    integration_id = "moviechat"
    default_feature_stage = "visual_memory"
    feature_stage = default_feature_stage
    default_checkout_path = "external/moviechat"
    default_model_path = "weights/moviechat"
    backend = "moviechat"
    cache_owner = "visual_memory"
    capabilities = EncoderCapabilities(
        supports_fixed_clip=False,
        supports_streaming=True,
        supports_kv_cache=False,
        supports_token_cache=False,
        supports_visual_memory_cache=True,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )


DEFAULT_CAPABILITIES = MovieChatAdapter.capabilities
FEATURE_STAGE = MovieChatAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "MovieChatAdapter"]
