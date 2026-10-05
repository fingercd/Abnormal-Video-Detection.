"""InternVideo2 fixed-clip integration."""

from __future__ import annotations

from vadbench.integrations.foundation.base import (
    FOUNDATION_CAPABILITIES,
    FoundationVideoAdapter,
)


class InternVideo2Adapter(FoundationVideoAdapter):
    """Adapt an InternVideo2 visual encoder without importing its optional stack."""

    capabilities = FOUNDATION_CAPABILITIES
    BACKEND = "internvideo2"
    FEATURE_STAGE = "backbone_tokens"
    PREPROCESS_PROFILE = "internvideo2-bthwc-v1"
    DEFAULT_MODEL_NAME = "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
    DEFAULT_RUNTIME = "external_python"


__all__ = ["InternVideo2Adapter"]
