"""MA-LMM visual-memory integration declaration."""

from __future__ import annotations

from vadbench.contracts import EncoderCapabilities

from .base import ExternalStreamingVideoAdapter


class MALMMAdapter(ExternalStreamingVideoAdapter):
    """Expose MA-LMM's Q-Former memory bank, not decoder KV, as state."""

    integration_id = "ma_lmm"
    default_feature_stage = "visual_memory"
    feature_stage = default_feature_stage
    default_checkout_path = "external/ma-lmm"
    default_model_path = "weights/ma-lmm"
    backend = "ma-lmm"
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


DEFAULT_CAPABILITIES = MALMMAdapter.capabilities
FEATURE_STAGE = MALMMAdapter.default_feature_stage

__all__ = ["DEFAULT_CAPABILITIES", "FEATURE_STAGE", "MALMMAdapter"]
