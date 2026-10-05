"""ADGS (Attention-Diverse Group Selection) drop-only token selection.

Plan section 3.3 candidate, VideoMAEv2 pair-member layouts: at one native
block depth, each frozen group quota is split into an importance half and a
diversity half.

* importance[i] is the *received* attention of token i: the mean of the
  same-block attention probability matrix over heads and query positions.
  The probabilities are captured in the same forward through the registered
  ``block.{depth}.attn.probs.input`` site (the ``attn_drop`` input, i.e. the
  post-softmax probabilities).  If that capture did not happen in the
  current forward, selection fails loudly with
  :class:`ADGSAttentionUnavailable` — no silent fallback to another signal.
* Grouping and quotas are reused verbatim from
  :func:`group_select_spec`, so the actual kept count K is identical to
  ``group_uniform`` (and every other rule) at the same keep_ratio.
* Per group, ``round_half_up(quota * important_ratio)`` slots keep the
  highest-importance tokens (ties broken by native index ascending).  The
  remaining slots are filled inside the same group by cosine
  farthest-point diversity: repeatedly take the candidate whose minimum
  cosine distance to the already-selected set is largest, ties by native
  index ascending.  The procedure is fully deterministic for a fixed seed
  (the algorithm consumes no RNG; ``seed`` is kept only for interface
  parity with :class:`GroupBudgetSelector`).
* Output is a real gather to the quota length — no padding back to N.

Leakage contract: only the current block output hidden states and the
same-forward same-block attention probabilities are read.  No labels, file
names, test statistics, future-layer activations, or dense-teacher signals
participate, and the selector adds no trainable parameters.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from vadbench.contracts import ClipBatch

from .bridges import EncoderBridge
from .bridges.indexed import IndexedTokenIntervention, indexed_gather
from .contracts import TokenLayout
from .deployment_contracts import ReductionDeployment
from .pair_merge import PairMergeError
from .token_selection import (
    GROUP_TOKENS,
    SUPPORTED_KEEP_RATIOS,
    GroupSelectSpec,
    group_select_spec,
    round_half_up,
)

ADGS_RULE_NAME = "adgs"
ADGS_SIGNAL_NAME = "received_attention_mean_over_heads_and_queries"
DEFAULT_IMPORTANT_RATIO = 0.50


class ADGSAttentionUnavailable(RuntimeError):
    """Same-forward block attention probabilities were not captured.

    ADGS never falls back to another selection signal when attention is
    missing or malformed; deployment setup also fails fast when the bridge
    does not expose the ``block.{depth}.attn.probs.input`` site.
    """


def _implementation_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expand_layout(layout: TokenLayout, batch_size: int) -> TokenLayout:
    def expand(value: torch.Tensor) -> torch.Tensor:
        return value.expand(batch_size, *value.shape[1:]).clone()

    # Copy only local geometry. No source-frame indices or sample metadata can
    # be carried from the setup clip into later extraction batches.
    provenance = {
        key: list(value) if isinstance(value, list) else value
        for key, value in layout.provenance.items()
        if key in {"coordinate_source", "grid", "kernel", "stride", "patch_stride"}
    }
    return TokenLayout(
        valid_mask=expand(layout.valid_mask),
        original_token_ids=expand(layout.original_token_ids),
        special_token_mask=expand(layout.special_token_mask),
        mass=expand(layout.mass),
        source_coordinates=expand(layout.source_coordinates),
        position_contract=layout.position_contract,
        provenance=provenance,
    )


class ADGSSelector(nn.Module):
    """Drop-only group-quota selector driven by received attention + diversity.

    Interface parity with :class:`GroupBudgetSelector`: usable as an
    ``IndexedTokenIntervention`` transform callback (input ``[B, N, D]``
    block-output hidden states, output ``[B, K, D]``), records
    ``last_selection`` and exposes ``selection_digest()``.  Attention
    probabilities for the current forward must be stashed beforehand via
    :meth:`stash_attention` (the deployment registers the capture hook);
    they are consumed exactly once per forward.
    """

    def __init__(
        self,
        dim: int,
        *,
        keep_ratio: float = 0.60,
        important_ratio: float = DEFAULT_IMPORTANT_RATIO,
        seed: int = 0,
        record_diagnostics: bool = True,
    ) -> None:
        super().__init__()
        if type(dim) is not int or dim <= 0:
            raise PairMergeError("dim 必须为正整数")
        if keep_ratio != 1.0 and keep_ratio not in SUPPORTED_KEEP_RATIOS:
            raise PairMergeError(
                f"keep_ratio 必须是 1.0（identity）或 {SUPPORTED_KEEP_RATIOS} 之一"
            )
        if not isinstance(important_ratio, (int, float)) or isinstance(important_ratio, bool):
            raise PairMergeError("important_ratio 必须为数值")
        if not 0.0 < float(important_ratio) <= 1.0:
            raise PairMergeError("important_ratio 必须在 (0, 1] 内")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise PairMergeError("seed 必须为 [0, 2**63) 内的整数")
        if not isinstance(record_diagnostics, bool):
            raise PairMergeError("record_diagnostics 必须为布尔")
        self.dim = dim
        self.keep_ratio = float(keep_ratio)
        self.important_ratio = float(important_ratio)
        self.seed = seed
        self.record_diagnostics = record_diagnostics
        self.last_selection: torch.Tensor | None = None
        self.last_importance: torch.Tensor | None = None
        self._pending_attention: torch.Tensor | None = None

    @property
    def identity_mode(self) -> bool:
        return self.keep_ratio == 1.0

    def stash_attention(self, attention: torch.Tensor) -> None:
        """Record same-forward block attention probabilities for the next selection."""

        if not isinstance(attention, torch.Tensor):
            raise ADGSAttentionUnavailable("attention capture 需要 torch.Tensor")
        self._pending_attention = attention

    def _take_attention(self, batch: int, tokens: int) -> torch.Tensor:
        attention = self._pending_attention
        self._pending_attention = None
        if attention is None:
            raise ADGSAttentionUnavailable(
                "ADGS 需要同一 forward 内经 block.{depth}.attn.probs.input 捕获的注意力概率；"
                "禁止在未捕获时静默回退其他信号"
            )
        if (
            attention.ndim != 4
            or attention.shape[0] != batch
            or attention.shape[2] != tokens
            or attention.shape[3] != tokens
        ):
            raise ADGSAttentionUnavailable(
                f"注意力概率 shape={tuple(attention.shape)} 与 [B,H,N,N] 不一致 "
                f"(B={batch}, N={tokens})"
            )
        if not torch.is_floating_point(attention):
            raise ADGSAttentionUnavailable("注意力概率必须为浮点 tensor")
        if not bool(torch.isfinite(attention).all()):
            raise PairMergeError("注意力概率必须有限")
        return attention

    def _importance(self, attention: torch.Tensor) -> torch.Tensor:
        """Received attention: mean over heads and queries per key token."""

        return attention.mean(dim=(1, 2))  # [B, N]

    def _select_group(
        self,
        importance_b: torch.Tensor,
        hidden_b: torch.Tensor,
        start: int,
        group_slots: int,
        quota: int,
        eps: float,
    ) -> torch.Tensor:
        """Chosen native token indices (ascending) for one quota group."""

        device = hidden_b.device
        token_ids = torch.arange(start, start + group_slots, device=device)
        important_quota = min(round_half_up(quota * self.important_ratio), group_slots)
        fill_quota = quota - important_quota
        scores = importance_b[token_ids]
        # Descending importance; stable argsort keeps native index ascending
        # for ties, the single fixed tie-breaking rule.
        order = torch.argsort(scores, descending=True, stable=True).tolist()
        chosen_set = set(order[:important_quota])
        if fill_quota:
            vectors = hidden_b[token_ids]
            unit = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)
            for _ in range(fill_quota):
                cand = [pos for pos in range(group_slots) if pos not in chosen_set]
                if not cand:
                    break
                cand_t = torch.tensor(cand, dtype=torch.long, device=device)
                sel_t = torch.tensor(sorted(chosen_set), dtype=torch.long, device=device)
                sim = unit[cand_t] @ unit[sel_t].transpose(0, 1)  # [C, S]
                min_dist = (1.0 - sim).min(dim=1).values
                # argmax first occurrence == lowest native index among ties,
                # because ``cand`` is in ascending order.
                pick = int(torch.argmax(min_dist).item())
                chosen_set.add(cand[pick])
        return token_ids[sorted(chosen_set)]

    def _select_slots(self, spec: GroupSelectSpec, importance: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """[B,K] kept reducible-token indices per sample."""

        batch = hidden.shape[0]
        eps = torch.finfo(hidden.dtype).eps
        rows = []
        for b in range(batch):
            kept = []
            start = 0
            for quota in spec.quotas:
                group_slots = min(GROUP_TOKENS, spec.slot_count - start)
                if quota > 0:
                    kept.append(
                        self._select_group(
                            importance[b], hidden[b], start, group_slots, quota, eps
                        )
                    )
                start += group_slots
            rows.append(torch.cat(kept))
        return torch.stack(rows, dim=0)

    def forward(
        self,
        hidden: torch.Tensor,
        spec: GroupSelectSpec | None = None,
        specials: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select quota slots per sample; specials are preserved unmodified.

        ``keep_ratio=1.0`` is the identity path: all tokens are returned
        unmodified (numerically equal to the input) and no attention is
        required.  At every frozen tier the output length equals
        ``spec.output_token_count`` — the same K as ``group_uniform``.
        """

        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[2] != self.dim
        ):
            raise PairMergeError("hidden 必须为 [B,N,D] 且 D 与 dim 一致")
        if not torch.is_floating_point(hidden):
            raise PairMergeError("hidden 必须为浮点 tensor")
        if not bool(torch.isfinite(hidden).all()):
            raise PairMergeError("hidden 必须有限")
        batch, tokens = hidden.shape[0], hidden.shape[1]
        if self.identity_mode:
            selected_index = torch.arange(tokens, device=hidden.device)[None].expand(batch, -1)
            if self.record_diagnostics:
                self.last_selection = selected_index.detach().cpu()
                self.last_importance = None
            return hidden
        if not isinstance(spec, GroupSelectSpec):
            raise PairMergeError("ADGS 非 identity 模式需要 GroupSelectSpec")
        if spec.mode != "pair_member":
            raise PairMergeError("ADGS 当前仅支持 pair_member（videomaev2/videomae）分组")
        if spec.input_token_count != tokens:
            raise PairMergeError("hidden 与 GroupSelectSpec token 数不一致")
        attention = self._take_attention(batch, tokens)
        importance = self._importance(attention)
        selected_slots = self._select_slots(spec, importance, hidden)
        if selected_slots.shape[1] != spec.output_token_count:
            raise PairMergeError("ADGS 选择结果与冻结配额不一致")
        if specials is None:
            specials = torch.empty(0, dtype=torch.int64, device=hidden.device)
        specials = specials.to(hidden.device)
        selected_index = torch.sort(
            torch.cat([specials[None].expand(batch, -1), selected_slots], dim=1), dim=1
        ).values
        output = torch.gather(
            hidden, 1, selected_index.unsqueeze(-1).expand(-1, -1, self.dim)
        )
        if self.record_diagnostics:
            # Dev-only audit exports; formal timing keeps the algorithm cost
            # alone and leaves these CPU transfers disabled.
            self.last_selection = selected_index.detach().cpu()
            self.last_importance = importance.detach().cpu()
        return output

    def selection_digest(self) -> str | None:
        """SHA-256 of the real per-forward chosen token indices (audit)."""

        if self.last_selection is None:
            return None
        return hashlib.sha256(self.last_selection.numpy().tobytes()).hexdigest()

    def importance_digest(self) -> str | None:
        """SHA-256 of the per-forward received-attention importances (audit)."""

        if self.last_importance is None:
            return None
        return hashlib.sha256(self.last_importance.contiguous().numpy().tobytes()).hexdigest()


class _ADGSContext(AbstractContextManager):
    """Attach the same-forward attention capture hook to an indexed intervention."""

    def __init__(
        self,
        inner: IndexedTokenIntervention,
        probs_module: torch.nn.Module,
        selector: ADGSSelector,
    ) -> None:
        self._inner = inner
        self._probs_module = probs_module
        self._selector = selector

    def __enter__(self) -> _ADGSContext:
        self._inner.__enter__()
        # The block-depth probs site executes inside the hooked block's
        # forward, i.e. before the block-output transform consumes the stash.
        handle = self._probs_module.register_forward_pre_hook(self._capture)
        # Reuse the intervention's own handle list so close()/__exit__ removes
        # the capture hook together with the gather and suffix hooks.
        self._inner._handles.append(handle)
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._inner.__exit__(*args)

    def _capture(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ADGSAttentionUnavailable("attn probs hook 未收到注意力概率 tensor")
        self._selector.stash_attention(inputs[0])

    def validate_execution(self) -> dict[str, Any]:
        return self._inner.validate_execution()

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)


class ADGSDeployment(ReductionDeployment):
    """Frozen group-budget ADGS deployment around an unchanged native adapter.

    Mirrors :class:`GroupSelectDeployment` setup discipline: a
    bridge-verified B=1 dense CPU topology, one context per allowed batch
    size, no dense fallback at inference.  In addition to the indexed
    gather hooks it registers a capture pre-hook on the bridge's
    ``block.{depth}.attn.probs.input`` site for the duration of each
    intervention forward, so the selector reads same-forward attention
    probabilities.  Setup fails fast with :class:`ADGSAttentionUnavailable`
    when that site does not exist on the loaded model.  ``keep_ratio=1.0``
    is the identity deployment and needs no attention capture.
    """

    def __init__(
        self,
        bridge: EncoderBridge,
        dense_layout: TokenLayout,
        *,
        depth: int,
        dim: int,
        keep_ratio: float = 0.60,
        important_ratio: float = DEFAULT_IMPORTANT_RATIO,
        batch_sizes: Iterable[int] = range(1, 9),
        device: torch.device | str | None = None,
        seed: int = 0,
        record_diagnostics: bool = True,
    ) -> None:
        if not isinstance(important_ratio, (int, float)) or isinstance(important_ratio, bool):
            raise PairMergeError("important_ratio 必须为数值")
        if not 0.0 < float(important_ratio) <= 1.0:
            raise PairMergeError("important_ratio 必须在 (0, 1] 内")
        if keep_ratio != 1.0 and keep_ratio not in SUPPORTED_KEEP_RATIOS:
            raise PairMergeError(
                f"keep_ratio 必须是 1.0（identity）或 {SUPPORTED_KEEP_RATIOS} 之一"
            )
        if not isinstance(record_diagnostics, bool):
            raise PairMergeError("record_diagnostics 必须为布尔")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise PairMergeError("seed 必须为 [0, 2**63) 内的整数")
        if type(dim) is not int or dim <= 0:
            raise PairMergeError("dim 必须为正整数")
        if dense_layout.batch_size != 1:
            raise PairMergeError("deployment setup 需要 bridge 验证的 B=1 dense layout")
        for name in ("valid_mask", "original_token_ids", "special_token_mask", "mass", "source_coordinates"):
            value = getattr(dense_layout, name)
            if value is None or value.device.type != "cpu":
                raise PairMergeError("deployment setup 需要完整 CPU dense layout")
        if not bool(dense_layout.valid_mask.all()) or not bool(dense_layout.mass.eq(1).all()):
            raise PairMergeError("deployment 仅支持无 padding、全 unit mass 的压缩点")
        if "pair_merge" in dense_layout.provenance:
            raise PairMergeError("deployment 不支持已有 merge history")
        if not torch.equal(dense_layout.original_token_ids, torch.arange(dense_layout.token_capacity)[None]):
            raise PairMergeError("deployment 需要完整 native dense token IDs")
        encoder_id = bridge.receipt().encoder_id
        coordinate_source = dense_layout.provenance.get("coordinate_source")
        expected_source = (
            "verified-timesformer-spatial-major-time-minor"
            if encoder_id == "timesformer" else "verified-conv3d-flatten-t-h-w"
        )
        if coordinate_source != expected_source:
            raise PairMergeError("deployment 需要当前 native bridge 验证的 coordinate_source")
        sizes = tuple(batch_sizes)
        if not sizes or any(type(size) is not int or size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
            raise PairMergeError("batch_sizes 必须为非空、不重复的正整数")

        parameter = next(bridge.model.parameters())
        execution_device = parameter.device if device is None else torch.device(device)

        self._identity_mode = keep_ratio == 1.0
        base = _expand_layout(dense_layout, 1)
        specials = torch.nonzero(base.special_token_mask[0], as_tuple=False).flatten()
        self._probs_site_name = f"block.{depth}.attn.probs.input"
        probs_module = None
        spec = None
        if self._identity_mode:
            indices_base = torch.arange(base.token_capacity, dtype=torch.int64)[None]
        else:
            sites = bridge.observation_sites((depth,))
            probs_module = sites.get(self._probs_site_name)
            if probs_module is None:
                raise ADGSAttentionUnavailable(
                    f"{encoder_id} 未在 block.{depth} 暴露 {self._probs_site_name} "
                    "注意力捕获点；ADGS 禁止无注意力静默回退"
                )
            spec = group_select_spec(encoder_id, base, keep_ratio)
            if spec.mode != "pair_member":
                raise PairMergeError("ADGS 当前仅支持 videomaev2/videomae 的 pair_member 分组")
            indices_base = torch.sort(torch.cat((specials, spec.skeleton_indices))).values[None]
        grid = base.provenance["grid"]
        if encoder_id == "timesformer":
            self._clip_frames = grid[0]
        else:
            kernel, stride = base.provenance.get("kernel"), base.provenance.get("stride")
            if not kernel or not stride or kernel[0] != stride[0]:
                raise PairMergeError("deployment 需要已验证的非重叠 temporal tubelet kernel/stride")
            self._clip_frames = grid[0] * stride[0]

        module = Path(__file__)
        identity: dict[str, Any] = {
            "name": "adgs_identity" if self._identity_mode else ADGS_RULE_NAME,
            "operator": "adgs_attention_diverse_v1",
            "budget_rule": None if self._identity_mode else "group_budget_v1",
            "keep_ratio": keep_ratio,
            "important_ratio": float(important_ratio),
            "encoder_id": encoder_id,
            "depth": depth,
            "hidden_dim": dim,
            "group_size": None if self._identity_mode else GROUP_TOKENS,
            "selection_signal": None if self._identity_mode else ADGS_SIGNAL_NAME,
            "signal_definition": (
                None
                if self._identity_mode
                else "importance[i] = mean over heads and query positions of the "
                "same-block attention probability matrix column key=i (received attention)"
            ),
            "signal_source_site": None if self._identity_mode else self._probs_site_name,
            "signal_scope": (
                None
                if self._identity_mode
                else "same_forward_same_block_attention_probabilities"
            ),
            "attention_capture": "attn_drop_input_pre_hook_stash_consume_once_per_forward",
            "attention_unavailable_policy": "explicit_error_no_fallback",
            "diversity_rule": (
                None
                if self._identity_mode
                else "cosine_farthest_point_min_distance_fill_within_group"
            ),
            "dynamic_selection": not self._identity_mode,
            "position_anchor": None if self._identity_mode else "skeleton_slot_native_index",
            "position_semantics": (
                "retained_native_position" if self._identity_mode else "approximate_anchor_position"
            ),
            "position_note": (
                None
                if self._identity_mode
                else "kept tokens fill quota slots of their group in ascending native order; "
                "the real per-forward chosen indices are recorded via selection_digest"
            ),
            "special_tokens": "preserved_unmodified",
            "mass_convention": "retained_unit_mass",
            "discarded_mass": "discarded_not_redistributed",
            "clip_frames": self._clip_frames,
            "same_budget_for_all_controls": True,
            "rounding": "round_half_up_floor_x_plus_0.5",
            "tie_breaking": "stable_argsort_importance_desc_then_native_index_ascending",
            "seed": seed,
            "deterministic": True,
            "random_generator": None,
            "mask_scope": (
                "static_identity"
                if self._identity_mode
                else "current_forward_mask_per_batch_item"
            ),
            "leakage_contract": "no_labels_no_filenames_no_test_statistics_no_future_layers_no_dense_teacher",
            "trainable_parameters": 0,
            "record_diagnostics": record_diagnostics,
            "deployment_sha256": _implementation_digest(module),
            "indexed_sha256": _implementation_digest(module.parent / "bridges" / "indexed.py"),
            "token_selection_sha256": _implementation_digest(module.with_name("token_selection.py")),
        }
        if not self._identity_mode:
            identity.update({
                "quotas": list(spec.quotas),
                "kept_units": spec.kept_units,
                "reducible_tokens": spec.reducible_tokens,
                "kept_tokens": spec.output_token_count,
                "actual_keep_ratio": spec.actual_keep_ratio,
                "tokens_per_slot": spec.tokens_per_slot,
                "selection_unit": "native_patch_token",
                "slot_layout": "identical_across_rules_at_same_keep_ratio",
                "input_topology_sha256": spec.input_topology_sha256,
            })
        self._reducer_identity = MappingProxyType(identity)
        self._selectors: dict[int, ADGSSelector] = {}
        self._contexts: dict[int, Any] = {}
        for size in sizes:
            layout = base if size == 1 else _expand_layout(base, size)
            indices = indices_base.expand(size, -1).clone().to(execution_device)
            selector = ADGSSelector(
                dim,
                keep_ratio=keep_ratio,
                important_ratio=important_ratio,
                seed=seed,
                record_diagnostics=record_diagnostics,
            ).to(execution_device).eval().requires_grad_(False)
            self._selectors[size] = selector
            if self._identity_mode:
                def transform(hidden, _indices, selector=selector):
                    return selector(hidden)
            else:
                def transform(hidden, _indices, selector=selector, spec=spec, specials=specials):
                    return selector(hidden, spec, specials.to(hidden.device))
            inner = indexed_gather(
                bridge, depth, indices, layout.to(execution_device),
                transform=transform,
                record_position_masks=False,
            )
            if self._identity_mode:
                self._contexts[size] = inner
            else:
                self._contexts[size] = _ADGSContext(inner, probs_module, selector)
        self._spec = spec

    @property
    def reducer_identity(self) -> Mapping[str, Any]:
        """Immutable, path-free identity of the actual frozen operator."""

        return self._reducer_identity

    @property
    def selection_spec(self) -> GroupSelectSpec | None:
        return self._spec

    def selection_digest(self, batch_size: int) -> str | None:
        selector = self._selectors.get(batch_size)
        return None if selector is None else selector.selection_digest()

    def importance_digest(self, batch_size: int) -> str | None:
        selector = self._selectors.get(batch_size)
        return None if selector is None else selector.importance_digest()

    def last_selection(self, batch_size: int) -> torch.Tensor | None:
        selector = self._selectors.get(batch_size)
        return None if selector is None else selector.last_selection

    def last_importance(self, batch_size: int) -> torch.Tensor | None:
        selector = self._selectors.get(batch_size)
        return None if selector is None else selector.last_importance

    def __call__(self, clean_batch: ClipBatch) -> IndexedTokenIntervention | _ADGSContext:
        # ClipBatch already validates shapes/types. Do not read IDs, labels,
        # timestamps, sampled source frames, or pixels to make selections.
        size = clean_batch.batch_size
        if size not in self._contexts:
            raise PairMergeError(f"batch size {size} 未在 setup 准备；禁止 dense 回退")
        if clean_batch.num_frames != self._clip_frames:
            raise PairMergeError("当前 clip 帧数与 verified geometry 不一致")
        if clean_batch.valid_mask is not None and not bool(clean_batch.valid_mask.all()):
            raise PairMergeError("deployment 不支持 padded clip")
        return self._contexts[size]


__all__ = [
    "ADGS_RULE_NAME",
    "ADGS_SIGNAL_NAME",
    "ADGSAttentionUnavailable",
    "ADGSSelector",
    "ADGSDeployment",
    "DEFAULT_IMPORTANT_RATIO",
]
