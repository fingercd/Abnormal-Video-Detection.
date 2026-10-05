"""Original VideoChat integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalFixedVideoAdapter


class VideoChatAdapter(ExternalFixedVideoAdapter):
    """Bridge VideoChat's projected visual interface."""

    integration_id = "videochat"
    default_feature_stage = "projected_visual"
    feature_stage = default_feature_stage
    default_checkout_path = "external/videochat"
    default_model_path = "weights/videochat"
    backend = "videochat"
    cache_owner = None
    run_mode = "fixed"
    capabilities = EncoderCapabilities(
        supports_fixed_clip=True,
        supports_streaming=False,
        supports_kv_cache=False,
        supports_token_cache=False,
        supports_visual_memory_cache=False,
        supports_external_cache_policy=False,
        supports_training=False,
        min_frames=1,
    )


DEFAULT_CAPABILITIES = VideoChatAdapter.capabilities
FEATURE_STAGE = VideoChatAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "VideoChatAdapter"]
