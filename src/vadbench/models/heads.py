"""Optional PyTorch temporal heads used by the VAD benchmark tasks.

Importing this module never requires PyTorch.  When PyTorch is unavailable,
constructing a neural head or calling a loss raises a focused ``ImportError``;
the NumPy data/metric pipeline remains usable.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - the two branches are exercised in separate envs.
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in the minimal CI job.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for vadbench.models heads. Install the "
            "project's 'torch' extra or a compatible PyTorch build."
        )


@dataclass
class MILHeadOutput(Mapping[str, Any]):
    """Structured output shared by attention and top-k MIL heads."""

    video_logits: Any
    snippet_logits: Any
    attention: Any | None = None
    selected_mask: Any | None = None
    valid_mask: Any | None = None

    @property
    def video_scores(self) -> Any:
        _require_torch()
        return torch.sigmoid(self.video_logits)

    @property
    def snippet_scores(self) -> Any:
        _require_torch()
        return torch.sigmoid(self.snippet_logits)

    def __getitem__(self, key: str) -> Any:
        if key == "video_scores":
            return self.video_scores
        if key == "snippet_scores":
            return self.snippet_scores
        if key in {
            "video_logits",
            "snippet_logits",
            "attention",
            "selected_mask",
            "valid_mask",
        }:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(
            (
                "video_logits",
                "snippet_logits",
                "video_scores",
                "snippet_scores",
                "attention",
                "selected_mask",
                "valid_mask",
            )
        )

    def __len__(self) -> int:
        return 7

    def to_dict(self) -> dict[str, Any]:
        return {key: self[key] for key in self}


if TORCH_AVAILABLE:

    def _validate_features(features: Tensor, feature_dim: int) -> Tensor:
        if not torch.is_tensor(features):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, S, D], got {tuple(features.shape)}")
        if features.shape[0] <= 0 or features.shape[1] <= 0:
            raise ValueError("features batch and temporal dimensions must be non-empty")
        if features.shape[-1] != feature_dim:
            raise ValueError(f"expected feature dimension {feature_dim}, got {features.shape[-1]}")
        return features

    def _valid_mask(features: Tensor, mask: Tensor | None) -> Tensor:
        expected = features.shape[:2]
        if mask is None:
            return torch.ones(expected, dtype=torch.bool, device=features.device)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=features.device)
        if tuple(mask.shape) != tuple(expected):
            raise ValueError(f"mask must have shape {tuple(expected)}, got {tuple(mask.shape)}")
        return mask.to(device=features.device, dtype=torch.bool)

    def _classifier(feature_dim: int, hidden_dim: int | None, dropout: float) -> nn.Sequential:
        if hidden_dim is None:
            return nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    class AttentionMILHead(nn.Module):
        """Gated-attention MIL classifier for video-level supervision.

        It emits snippet logits for ranking/sparsity losses and learns an
        attention distribution for the video logit.  Invalid/padded timeline
        positions receive exactly zero attention.
        """

        def __init__(
            self,
            feature_dim: int,
            hidden_dim: int = 128,
            *,
            dropout: float = 0.0,
            gated: bool = True,
        ) -> None:
            super().__init__()
            if feature_dim <= 0 or hidden_dim <= 0:
                raise ValueError("feature_dim and hidden_dim must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0, 1)")
            self.feature_dim = int(feature_dim)
            self.hidden_dim = int(hidden_dim)
            self.gated = bool(gated)
            self.dropout = nn.Dropout(dropout)
            self.attention_v = nn.Linear(feature_dim, hidden_dim)
            self.attention_u = nn.Linear(feature_dim, hidden_dim) if gated else None
            self.attention_w = nn.Linear(hidden_dim, 1)
            self.classifier = nn.Linear(feature_dim, 1)

        def forward(self, features: Tensor, mask: Tensor | None = None) -> MILHeadOutput:
            features = _validate_features(features, self.feature_dim)
            valid = _valid_mask(features, mask)
            dropped = self.dropout(features)
            attention_hidden = torch.tanh(self.attention_v(dropped))
            if self.attention_u is not None:
                attention_hidden = attention_hidden * torch.sigmoid(self.attention_u(dropped))
            attention_logits = self.attention_w(attention_hidden).squeeze(-1)
            attention_logits = attention_logits.masked_fill(~valid, -torch.inf)

            # Softmax(all -inf) is NaN.  Re-normalizing the masked weights also
            # gives an all-padding row a deterministic all-zero distribution.
            has_valid = valid.any(dim=1, keepdim=True)
            safe_logits = torch.where(
                has_valid, attention_logits, torch.zeros_like(attention_logits)
            )
            attention = torch.softmax(safe_logits, dim=1) * valid.to(features.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(features.dtype).eps
            )

            snippet_logits = self.classifier(dropped).squeeze(-1)
            video_logits = (attention * snippet_logits).sum(dim=1)
            return MILHeadOutput(
                video_logits=video_logits,
                snippet_logits=snippet_logits,
                attention=attention,
                valid_mask=valid,
            )

    class TopKMILHead(nn.Module):
        """Temporal classifier with masked top-k video aggregation."""

        def __init__(
            self,
            feature_dim: int,
            *,
            k: int | float = 0.125,
            hidden_dim: int | None = None,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if feature_dim <= 0:
                raise ValueError("feature_dim must be positive")
            if isinstance(k, bool) or not isinstance(k, (int, float)) or k <= 0:
                raise ValueError("k must be a positive integer or a ratio in (0, 1]")
            if isinstance(k, float) and k > 1.0:
                raise ValueError("floating-point k must be a ratio in (0, 1]")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0, 1)")
            self.feature_dim = int(feature_dim)
            self.k = k
            self.classifier = _classifier(feature_dim, hidden_dim, dropout)

        def _k_for(self, valid_count: int) -> int:
            if isinstance(self.k, int):
                return min(self.k, valid_count)
            return min(max(1, math.ceil(valid_count * self.k)), valid_count)

        def forward(self, features: Tensor, mask: Tensor | None = None) -> MILHeadOutput:
            features = _validate_features(features, self.feature_dim)
            valid = _valid_mask(features, mask)
            snippet_logits = self.classifier(features).squeeze(-1)
            selected = torch.zeros_like(valid)
            video_logits: list[Tensor] = []
            attention_rows: list[Tensor] = []
            for row_index in range(features.shape[0]):
                valid_indices = torch.nonzero(valid[row_index], as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    # Preserve a differentiable zero for pathological padding-only bags.
                    video_logits.append(snippet_logits[row_index].sum() * 0.0)
                    attention_rows.append(torch.zeros_like(snippet_logits[row_index]))
                    continue
                count = self._k_for(int(valid_indices.numel()))
                valid_logits = snippet_logits[row_index, valid_indices]
                _, local_indices = torch.topk(valid_logits, k=count, largest=True, sorted=False)
                chosen = valid_indices[local_indices]
                selected[row_index, chosen] = True
                video_logits.append(snippet_logits[row_index, chosen].mean())
                weights = torch.zeros_like(snippet_logits[row_index])
                weights = weights.scatter(0, chosen, 1.0 / count)
                attention_rows.append(weights)
            return MILHeadOutput(
                video_logits=torch.stack(video_logits),
                snippet_logits=snippet_logits,
                attention=torch.stack(attention_rows),
                selected_mask=selected,
                valid_mask=valid,
            )

    class TemporalSupervisedHead(nn.Module):
        """Per-timestep binary classifier for strong temporal supervision."""

        def __init__(
            self,
            feature_dim: int,
            *,
            hidden_dim: int | None = None,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if feature_dim <= 0:
                raise ValueError("feature_dim must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0, 1)")
            self.feature_dim = int(feature_dim)
            self.classifier = _classifier(feature_dim, hidden_dim, dropout)

        def forward(self, features: Tensor, mask: Tensor | None = None) -> Tensor:
            features = _validate_features(features, self.feature_dim)
            logits = self.classifier(features).squeeze(-1)
            if mask is not None:
                _valid_mask(features, mask)  # Validate; loss consumes the mask.
            return logits

    def _masked_bag_max(scores: Tensor, mask: Tensor | None, *, name: str) -> Tensor:
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        if scores.ndim != 2:
            raise ValueError(f"{name} must have shape [B, S] or [S]")
        if scores.shape[0] == 0 or scores.shape[1] == 0:
            raise ValueError(f"{name} must contain at least one non-empty bag")
        if mask is None:
            return scores.max(dim=1).values
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, device=scores.device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) != tuple(scores.shape):
            raise ValueError(f"{name}_mask must have shape {tuple(scores.shape)}")
        valid = mask.to(device=scores.device, dtype=torch.bool)
        if not torch.all(valid.any(dim=1)):
            raise ValueError(f"every {name} bag must contain at least one valid score")
        return scores.masked_fill(~valid, -torch.inf).max(dim=1).values

    def _masked_regularizers(scores: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        if mask is None:
            valid = torch.ones_like(scores, dtype=torch.bool)
        else:
            valid = torch.as_tensor(mask, device=scores.device, dtype=torch.bool)
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if tuple(valid.shape) != tuple(scores.shape):
                raise ValueError("anomaly_mask shape must match anomaly_scores")

        valid_float = valid.to(scores.dtype)
        sparsity = (scores.abs() * valid_float).sum() / valid_float.sum().clamp_min(1.0)
        if scores.shape[1] < 2:
            return scores.sum() * 0.0, sparsity
        pair_valid = valid[:, 1:] & valid[:, :-1]
        pair_float = pair_valid.to(scores.dtype)
        squared_diff = (scores[:, 1:] - scores[:, :-1]).square()
        smoothness = (squared_diff * pair_float).sum() / pair_float.sum().clamp_min(1.0)
        return smoothness, sparsity

    def mil_ranking_loss(
        anomaly_scores: Tensor,
        normal_scores: Tensor,
        *,
        anomaly_mask: Tensor | None = None,
        normal_mask: Tensor | None = None,
        margin: float = 1.0,
        smoothness_weight: float = 0.0,
        sparsity_weight: float = 0.0,
        reduction: str = "mean",
    ) -> Tensor:
        """RTFM-style bag ranking loss with optional temporal regularizers.

        Every anomaly bag is compared with every normal bag, which also works
        when a minibatch contains unequal class counts.
        """

        if margin < 0 or smoothness_weight < 0 or sparsity_weight < 0:
            raise ValueError("margin and regularizer weights must be non-negative")
        positive_max = _masked_bag_max(anomaly_scores, anomaly_mask, name="anomaly_scores")
        negative_max = _masked_bag_max(normal_scores, normal_mask, name="normal_scores")
        pair_losses = torch.relu(margin - positive_max[:, None] + negative_max[None, :])
        if reduction == "mean":
            rank_loss = pair_losses.mean()
        elif reduction == "sum":
            rank_loss = pair_losses.sum()
        elif reduction == "none":
            if smoothness_weight or sparsity_weight:
                raise ValueError("regularizers require reduction='mean' or 'sum'")
            return pair_losses
        else:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        smoothness, sparsity = _masked_regularizers(anomaly_scores, anomaly_mask)
        return rank_loss + smoothness_weight * smoothness + sparsity_weight * sparsity

    def weakly_supervised_ranking_loss(
        snippet_scores: Tensor,
        video_labels: Tensor,
        *,
        mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Split a mixed minibatch and delegate to :func:`mil_ranking_loss`."""

        if snippet_scores.ndim != 2:
            raise ValueError("snippet_scores must have shape [B, S]")
        labels = torch.as_tensor(video_labels, device=snippet_scores.device).reshape(-1)
        if labels.shape[0] != snippet_scores.shape[0]:
            raise ValueError("video_labels length must equal batch size")
        positive_rows = labels > 0.5
        negative_rows = ~positive_rows
        if not positive_rows.any() or not negative_rows.any():
            raise ValueError("ranking loss needs at least one anomaly and one normal bag")
        valid = None if mask is None else torch.as_tensor(mask, device=snippet_scores.device)
        return mil_ranking_loss(
            snippet_scores[positive_rows],
            snippet_scores[negative_rows],
            anomaly_mask=None if valid is None else valid[positive_rows],
            normal_mask=None if valid is None else valid[negative_rows],
            **kwargs,
        )

    def temporal_supervised_loss(
        logits: Tensor,
        targets: Tensor,
        *,
        mask: Tensor | None = None,
        positive_weight: float | None = None,
        focal_gamma: float | None = None,
        reduction: str = "mean",
    ) -> Tensor:
        """Masked BCE-with-logits loss for frame/snippet-level labels."""

        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have equal shape, got "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}"
            )
        if positive_weight is not None and positive_weight <= 0:
            raise ValueError("positive_weight must be greater than zero")
        if focal_gamma is not None and focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        pos_weight = None
        if positive_weight is not None:
            pos_weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
        element_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=pos_weight
        )
        if focal_gamma is not None:
            probability = torch.sigmoid(logits)
            p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
            element_loss = element_loss * (1.0 - p_t).pow(focal_gamma)

        if mask is None:
            valid = torch.ones_like(element_loss, dtype=torch.bool)
        else:
            valid = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
            if tuple(valid.shape) != tuple(logits.shape):
                raise ValueError("mask shape must match logits")
        selected = element_loss[valid]
        if selected.numel() == 0:
            raise ValueError("temporal loss mask selects no elements")
        if reduction == "mean":
            return selected.mean()
        if reduction == "sum":
            return selected.sum()
        if reduction == "none":
            masked = torch.zeros_like(element_loss)
            masked[valid] = selected
            return masked
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")


else:

    class _TorchRequiredHead:
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()

    class AttentionMILHead(_TorchRequiredHead):
        pass

    class TopKMILHead(_TorchRequiredHead):
        pass

    class TemporalSupervisedHead(_TorchRequiredHead):
        pass

    def mil_ranking_loss(*_: Any, **__: Any) -> Any:
        _require_torch()

    def weakly_supervised_ranking_loss(*_: Any, **__: Any) -> Any:
        _require_torch()

    def temporal_supervised_loss(*_: Any, **__: Any) -> Any:
        _require_torch()


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
