"""Single-site, ungrouped attention/key reduction for patch-only video ViTs.

VisionZip's selection and ``target + mean(assigned)`` formula follow the local
MLLM-Comp-Platform kernel. This batched implementation keeps all decisions on
the tensor device; it does not import the VLM platform or change native pooling.
See projects/icassp2027/key_attention/README.md for provenance and scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


METHODS = ("attention_topk", "visionzip_internal", "uniform")


@dataclass(frozen=True)
class RelationSelection:
    tokens: torch.Tensor
    representatives: torch.Tensor
    source_to_output: torch.Tensor
    mass: torch.Tensor


def retained_count(token_count: int, keep_ratio: float) -> int:
    """Nearest integer, half upward; ratios refer to ordinary patch tokens."""
    if token_count < 1 or not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError("positive token_count and keep_ratio in (0, 1] are required")
    return max(1, math.floor(token_count * keep_ratio + 0.5))


def _gather(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def select_relational_tokens(
    hidden: torch.Tensor,
    *,
    method: str,
    keep_ratio: float,
    keys: torch.Tensor | None = None,
    attention_scores: torch.Tensor | None = None,
) -> RelationSelection:
    """Reduce unpadded, patch-only ``[B,N,D]`` hidden states exactly once.

    Scores are mean received attention ``[B,N]``. Keys, when required, are the
    actual attention-space keys ``[B,H,N,d]``. No sample metadata or labels are
    accepted. The bridge must exclude unsupported special-token/ragged layouts.
    Pruning uses -1 for discarded sources; VisionZip represents every source.
    Source-group minimum determines VisionZip output order, as in the reference.
    """
    if method not in METHODS:
        raise ValueError(f"unknown relation method: {method}")
    if hidden.ndim != 3 or not hidden.is_floating_point():
        raise ValueError("hidden must be floating [B,N,D]")
    batch, n, _ = hidden.shape
    k = retained_count(n, keep_ratio)
    native = torch.arange(n, device=hidden.device).expand(batch, -1)
    if k == n:
        return RelationSelection(hidden, native, native, torch.ones_like(native))
    if method == "uniform":
        slots = torch.linspace(0, n - 1, k, device=hidden.device).round().long()
        selected = slots.expand(batch, -1)
    else:
        if attention_scores is None or attention_scores.shape != (batch, n):
            raise ValueError("received-attention scores must match [B,N]")
        if not bool(torch.isfinite(attention_scores).all()):
            raise ValueError("received-attention scores must be finite")
        ranked = attention_scores.detach().float().argsort(dim=1, descending=True, stable=True)
        if method == "attention_topk":
            selected = ranked[:, :k].sort(dim=1).values
        else:
            return _visionzip(hidden, keys, ranked, k, native)
    assignment = torch.full_like(native, -1)
    assignment.scatter_(1, selected, torch.arange(k, device=hidden.device).expand(batch, -1))
    return RelationSelection(_gather(hidden, selected), selected, assignment, torch.ones_like(selected))


def _visionzip(hidden, keys, ranked, k, native) -> RelationSelection:
    batch, n, dim = hidden.shape
    if k < 2:
        raise ValueError("VisionZip needs at least one dominant and one contextual token")
    if keys is None or keys.ndim != 4 or keys.shape[0] != batch or keys.shape[2] != n:
        raise ValueError("VisionZip requires actual keys [B,H,N,d]")
    if not bool(torch.isfinite(keys).all()):
        raise ValueError("keys must be finite")
    # Frozen 4:1 output allocation, rounded half upward, not a fitted threshold.
    dominant_n = min(k - 1, max(1, (k * 4 + 2) // 5))
    context_n = k - dominant_n
    dominant = ranked[:, :dominant_n]
    remaining_mask = torch.ones_like(native, dtype=torch.bool)
    remaining_mask.scatter_(1, dominant, False)
    remaining = native[remaining_mask].reshape(batch, n - dominant_n)
    stride = remaining.shape[1] // context_n
    target_slots = torch.arange(context_n, device=hidden.device) * stride
    targets = remaining[:, target_slots]
    source_mask = remaining_mask.clone()
    source_mask.scatter_(1, targets, False)
    sources = native[source_mask].reshape(batch, n - k)
    metric = F.normalize(keys.detach().mean(dim=1).float(), p=2, dim=-1)
    destination = (_gather(metric, sources) @ _gather(metric, targets).transpose(1, 2)).argmax(-1)

    # Reference formula is NOT a mass-weighted mean of the full group.
    sums = torch.zeros(batch, context_n, dim, device=hidden.device, dtype=torch.float32)
    sums.scatter_add_(1, destination.unsqueeze(-1).expand(-1, -1, dim), _gather(hidden, sources).float())
    counts = torch.zeros(batch, context_n, device=hidden.device, dtype=torch.long)
    counts.scatter_add_(1, destination, torch.ones_like(destination))
    context = (_gather(hidden, targets).float() + sums / counts.clamp_min(1).unsqueeze(-1)).to(hidden.dtype)
    values = torch.cat((_gather(hidden, dominant), context), dim=1)
    representatives = torch.cat((dominant, targets), dim=1)
    assignment = torch.empty_like(native)
    assignment.scatter_(1, dominant, torch.arange(dominant_n, device=hidden.device).expand(batch, -1))
    assignment.scatter_(1, targets, (dominant_n + torch.arange(context_n, device=hidden.device)).expand(batch, -1))
    assignment.scatter_(1, sources, dominant_n + destination)
    minima = torch.full_like(representatives, n)
    minima.scatter_reduce_(1, assignment, native, reduce="amin", include_self=True)
    order = minima.argsort(dim=1, stable=True)
    inverse = torch.empty_like(order)
    inverse.scatter_(1, order, torch.arange(k, device=hidden.device).expand(batch, -1))
    mass = torch.cat((torch.ones_like(dominant), counts + 1), dim=1)
    return RelationSelection(
        _gather(values, order), representatives.gather(1, order),
        inverse.gather(1, assignment), mass.gather(1, order),
    )
