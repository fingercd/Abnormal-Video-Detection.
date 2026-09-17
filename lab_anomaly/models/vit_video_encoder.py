"""Compatibility import for the retained ``lab_anomaly`` prototype.

The single VideoMAE v2 implementation is maintained in
``vadbench.integrations.videomaev2_encoder``. Existing legacy imports remain
valid without retaining a second model loader.
"""

from vadbench.integrations.videomaev2_encoder import (
    VideoMAEv2Encoder,
    VideoMAEv2EncoderConfig,
)

__all__ = ["VideoMAEv2Encoder", "VideoMAEv2EncoderConfig"]
