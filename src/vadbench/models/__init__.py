"""Task heads exposed without making PyTorch a base-package dependency."""

from .heads import (
    TORCH_AVAILABLE,
    AttentionMILHead,
    MILHeadOutput,
    TemporalSupervisedHead,
    TopKMILHead,
    mil_ranking_loss,
    temporal_supervised_loss,
    weakly_supervised_ranking_loss,
)

__all__ = [
    "AttentionMILHead",
    "MILHeadOutput",
    "TORCH_AVAILABLE",
    "TemporalSupervisedHead",
    "TopKMILHead",
    "mil_ranking_loss",
    "temporal_supervised_loss",
    "weakly_supervised_ranking_loss",
]
