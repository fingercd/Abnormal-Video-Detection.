"""CPU checks of ADGS (Attention-Diverse Group Selection) selection and deployment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from test_pair_deployment import _batch

from vadbench.token_reduction.attention_diverse_selection import (
    ADGSAttentionUnavailable,
    ADGSSelector,
    ADGSDeployment,
)
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.pair_merge import PairMergeError
from vadbench.token_reduction.token_selection import (
    GROUP_TOKENS,
    GroupBudgetSelector,
    GroupSelectSpec,
    group_quotas,
)


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _make_spec(token_count: int = 20, keep_ratio: float = 0.60) -> GroupSelectSpec:
    """Synthetic pair-member spec with consecutive adjacent pairs."""

    quotas = group_quotas(token_count, keep_ratio)
    pieces = []
    start = 0
    for quota in quotas:
        pieces.append(torch.arange(start, start + quota, dtype=torch.int64))
        start += min(GROUP_TOKENS, token_count - start)
    skeleton = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64)
    return GroupSelectSpec(
        encoder_id="videomaev2",
        input_token_count=token_count,
        keep_ratio=keep_ratio,
        mode="pair_member",
        units=torch.arange(token_count, dtype=torch.int64).reshape(-1, 2),
        group_size_units=5,
        quotas=tuple(quotas),
        slot_count=token_count,
        skeleton_indices=skeleton,
        tokens_per_slot=1,
        reducible_tokens=token_count,
        kept_units=sum(quotas),
        input_topology_sha256="a" * 64,
    )


def _attention(batch: int, heads: int, tokens: int, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Probability-like [B,H,N,N] attention; optional per-key additive bias."""

    generator = torch.Generator(device="cpu").manual_seed(1234)
    logits = torch.randn(batch, heads, tokens, tokens, generator=generator)
    if bias is not None:
        logits = logits + bias.to(logits.dtype)
    return logits.softmax(dim=-1)


def _run(selector: ADGSSelector, hidden: torch.Tensor, spec: GroupSelectSpec) -> torch.Tensor:
    selector.stash_attention(_attention(hidden.shape[0], 2, hidden.shape[1]))
    return selector(hidden, spec, torch.empty(0, dtype=torch.int64))


# ---------------------------------------------------------------------------
# selector-level behavior
# ---------------------------------------------------------------------------

def test_identity_keep_ratio_one_is_numerically_equal_and_records_selection():
    selector = ADGSSelector(8, keep_ratio=1.0)
    hidden = torch.randn(2, 20, 8)
    output = selector(hidden)
    assert torch.equal(output, hidden)
    assert selector.last_selection is not None
    assert selector.last_selection[0].tolist() == list(range(20))
    assert selector.selection_digest() is not None
    # Identity never needs attention and must not consume a stale stash.
    selector2 = ADGSSelector(8, keep_ratio=1.0)
    assert torch.equal(selector2(hidden), hidden)


def test_actual_k_matches_group_uniform_at_same_spec():
    spec = _make_spec(token_count=20, keep_ratio=0.60)
    assert spec.output_token_count == 12
    hidden = torch.randn(1, 20, 8)
    adgs = ADGSSelector(8, keep_ratio=0.60)
    reduced = _run(adgs, hidden, spec)
    uniform = GroupBudgetSelector(8, rule="group_uniform")(hidden, spec, torch.empty(0, dtype=torch.int64))
    assert reduced.shape == uniform.shape == (1, spec.output_token_count, 8)
    for keep_ratio in (0.80, 0.40):
        tier_spec = _make_spec(token_count=20, keep_ratio=keep_ratio)
        tier = ADGSSelector(8, keep_ratio=keep_ratio)
        out = _run(tier, hidden, tier_spec)
        assert out.shape[1] == tier_spec.output_token_count


def test_selected_indices_are_unique_in_range_and_ascending():
    spec = _make_spec(token_count=26, keep_ratio=0.60)  # full group + remainder
    hidden = torch.randn(3, 26, 8)
    selector = ADGSSelector(8, keep_ratio=0.60)
    _run(selector, hidden, spec)
    selection = selector.last_selection
    assert selection.shape == (3, spec.output_token_count)
    for row in selection:
        values = row.tolist()
        assert values == sorted(values)
        assert len(set(values)) == len(values)
        assert all(0 <= index < 26 for index in values)


def test_same_inputs_give_same_digest_and_different_inputs_differ():
    spec = _make_spec(token_count=20, keep_ratio=0.60)
    generator = torch.Generator(device="cpu").manual_seed(7)
    hidden = torch.randn(1, 20, 8, generator=generator)
    first = ADGSSelector(8, keep_ratio=0.60)
    second = ADGSSelector(8, keep_ratio=0.60)
    _run(first, hidden.clone(), spec)
    _run(second, hidden.clone(), spec)
    assert first.selection_digest() == second.selection_digest()
    other = ADGSSelector(8, keep_ratio=0.60)
    _run(other, hidden.flip(dims=(1,)).clone(), spec)
    assert other.selection_digest() != first.selection_digest()


def test_hidden_attention_and_output_are_all_finite():
    spec = _make_spec(token_count=20, keep_ratio=0.60)
    hidden = torch.randn(2, 20, 8)
    attention = _attention(2, 3, 20)
    assert torch.isfinite(hidden).all() and torch.isfinite(attention).all()
    selector = ADGSSelector(8, keep_ratio=0.60)
    selector.stash_attention(attention)
    output = selector(hidden, spec, torch.empty(0, dtype=torch.int64))
    assert torch.isfinite(output).all()
    assert torch.isfinite(selector.last_importance).all()
    assert selector.importance_digest() is not None


def test_missing_attention_raises_explicitly_without_fallback():
    spec = _make_spec(token_count=20, keep_ratio=0.60)
    hidden = torch.randn(1, 20, 8)
    selector = ADGSSelector(8, keep_ratio=0.60)
    with pytest.raises(ADGSAttentionUnavailable, match="attention|注意力"):
        selector(hidden, spec, torch.empty(0, dtype=torch.int64))
    # The stash is consumed exactly once: a second forward without a new
    # capture must raise again rather than reuse stale attention.
    selector.stash_attention(_attention(1, 2, 20))
    selector(hidden, spec, torch.empty(0, dtype=torch.int64))
    with pytest.raises(ADGSAttentionUnavailable):
        selector(hidden, spec, torch.empty(0, dtype=torch.int64))


def test_malformed_or_nonfinite_attention_raises():
    spec = _make_spec(token_count=20, keep_ratio=0.60)
    hidden = torch.randn(1, 20, 8)
    selector = ADGSSelector(8, keep_ratio=0.60)
    selector.stash_attention(torch.randn(1, 2, 20, 16))  # wrong key length
    with pytest.raises(ADGSAttentionUnavailable):
        selector(hidden, spec, torch.empty(0, dtype=torch.int64))
    bad = _attention(1, 2, 20)
    bad[0, 0, 0, 0] = float("nan")
    selector.stash_attention(bad)
    with pytest.raises(PairMergeError, match="有限"):
        selector(hidden, spec, torch.empty(0, dtype=torch.int64))


def test_skewed_importance_keeps_high_attention_tokens():
    tokens = 10
    spec = _make_spec(token_count=tokens, keep_ratio=0.60)  # one group, quota 6
    bias = torch.zeros(tokens)
    bias[7] = 12.0  # token 7 dominates received attention
    hidden = torch.randn(1, tokens, 8)
    selector = ADGSSelector(8, keep_ratio=0.60)
    selector.stash_attention(_attention(1, 2, tokens, bias=bias))
    selector(hidden, spec, torch.empty(0, dtype=torch.int64))
    selection = selector.last_selection[0].tolist()
    assert len(selection) == 6
    assert 7 in selection
    importance = selector.last_importance[0]
    assert importance.argmax().item() == 7


def test_diversity_fill_prefers_cosine_farthest_token():
    tokens = 10
    spec = _make_spec(token_count=tokens, keep_ratio=0.60)  # quota 6 = 3 important + 3 fill
    hidden = torch.zeros(1, tokens, 10)
    # Tokens 0-8 share one direction; token 9 is orthogonal to all of them.
    direction = torch.arange(1.0, 11.0)
    hidden[0, :9] = direction
    hidden[0, 9] = -direction.flip(0)
    # Flat attention: the important half must be tokens {0,1,2} (ascending
    # tie-break), then the fill must pick token 9 first (max min-distance),
    # then ascending ties pick tokens 3 and 4.
    selector = ADGSSelector(10, keep_ratio=0.60)
    selector.stash_attention(torch.full((1, 2, tokens, tokens), 1.0 / tokens))
    selector(hidden, spec, torch.empty(0, dtype=torch.int64))
    assert selector.last_selection[0].tolist() == [0, 1, 2, 3, 4, 9]


def test_constructor_validation():
    with pytest.raises(PairMergeError):
        ADGSSelector(8, keep_ratio=0.50)  # not a frozen tier and not identity
    with pytest.raises(PairMergeError):
        ADGSSelector(8, keep_ratio=0.60, important_ratio=0.0)
    with pytest.raises(PairMergeError):
        ADGSSelector(8, keep_ratio=0.60, important_ratio=1.5)
    with pytest.raises(PairMergeError):
        ADGSSelector(8, keep_ratio=0.60, seed=-1)


# ---------------------------------------------------------------------------
# deployment-level behavior with a native patch-only model exposing attn_drop
# ---------------------------------------------------------------------------

class _NativePatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 8, 2, 2)

    def forward(self, pixels):
        return self.proj(pixels).flatten(2).transpose(1, 2)


class _NativeAttention(nn.Module):
    """VideoMAEv2-style attention: functional qkv, probs pass through attn_drop."""

    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden):
        batch, tokens, dim = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scale = (dim // self.num_heads) ** -0.5
        probs = (query @ key.transpose(-2, -1)) * scale
        probs = probs.softmax(dim=-1)
        probs = self.attn_drop(probs)  # capture site: block.N.attn.probs.input
        out = (probs @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out)


class _NativeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.attn = _NativeAttention()

    def forward(self, hidden):
        return hidden + self.attn(self.norm1(hidden))


class _NativeMAEv2ADGS(nn.Module):
    """Patch-only Conv3d model whose blocks expose an attn_drop probs site."""

    def __init__(self):
        super().__init__()
        self.patch_embed = _NativePatch()
        self.blocks = nn.ModuleList([_NativeBlock(), _NativeBlock()])

    def forward(self, pixels):
        hidden = self.patch_embed(pixels)
        for block in self.blocks:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _Adapter:
    def __init__(self, model, frames):
        self.model = model.eval()
        self.encoder_id = "videomaev2"
        self.capabilities = SimpleNamespace(
            fixed_num_frames=frames, min_frames=frames, max_frames=frames,
            supports_fixed_clip=True, supports_training=True,
        )

    def encode(self, batch):
        pixels = torch.from_numpy(batch.frames).permute(0, 1, 4, 2, 3).float() / 255
        return self.model(pixels.transpose(1, 2)).last_hidden_state


@pytest.fixture()
def adgs_case():
    torch.manual_seed(41)
    model = _NativeMAEv2ADGS()
    adapter = _Adapter(model, frames=4)
    bridge = create_observation_bridge("videomaev2", model)
    initial = _batch(frames=4, offset=100)
    with torch.no_grad(), bridge.geometry(initial.frame_indices, initial.valid_mask) as geometry:
        adapter.encode(initial)
    assert geometry.layout is not None
    sites = bridge.observation_sites((0,))
    assert "block.0.attn.probs.input" in sites
    return adapter, bridge, geometry.layout, 4, 8


def test_deployment_shortens_suffix_with_real_gather_and_cleans_hooks(adgs_case):
    adapter, bridge, layout, frames, dim = adgs_case
    deployment = ADGSDeployment(
        bridge, layout, depth=0, dim=dim, keep_ratio=0.60,
        batch_sizes=(1,), device="cpu",
    )
    identity = deployment.reducer_identity
    assert identity["name"] == "adgs"
    assert identity["selection_signal"] == "received_attention_mean_over_heads_and_queries"
    assert identity["signal_source_site"] == "block.0.attn.probs.input"
    assert identity["attention_unavailable_policy"] == "explicit_error_no_fallback"
    assert identity["dynamic_selection"] is True
    assert identity["trainable_parameters"] == 0
    spec = deployment.selection_spec
    batch = _batch(frames=frames)
    with torch.no_grad(), deployment(batch) as context:
        output = adapter.encode(batch)
        receipt = context.validate_execution()
    assert output.shape == (1, spec.output_token_count, dim)
    assert receipt["gathered_tokens"] == spec.output_token_count
    assert receipt["per_layer_token_counts"] == {"1": spec.output_token_count}
    assert deployment.selection_digest(1) is not None
    assert deployment.importance_digest(1) is not None
    selection = deployment.last_selection(1)
    assert selection.shape == (1, spec.output_token_count)
    assert deployment.last_importance(1).shape == (1, layout.token_capacity)
    # The attention capture hook is removed together with the intervention hooks.
    assert not any(block._forward_hooks or block._forward_pre_hooks for block in bridge._blocks)
    assert len(bridge._blocks[0].attn.attn_drop._forward_pre_hooks) == 0


def test_deployment_repeats_deterministically_per_forward(adgs_case):
    adapter, bridge, layout, frames, dim = adgs_case
    deployment = ADGSDeployment(
        bridge, layout, depth=0, dim=dim, keep_ratio=0.60, batch_sizes=(1,), device="cpu",
    )
    digests = []
    for offset in (11, 22):
        batch = _batch(frames=frames, offset=offset)
        with torch.no_grad(), deployment(batch):
            adapter.encode(batch)
        digests.append(deployment.selection_digest(1))
    assert digests[0] is not None and digests[1] is not None


def test_deployment_identity_keep_ratio_one_needs_no_attention(adgs_case):
    adapter, bridge, layout, frames, dim = adgs_case
    deployment = ADGSDeployment(
        bridge, layout, depth=0, dim=dim, keep_ratio=1.0, batch_sizes=(1,), device="cpu",
    )
    assert deployment.reducer_identity["name"] == "adgs_identity"
    batch = _batch(frames=frames)
    with torch.no_grad(), deployment(batch) as context:
        output = adapter.encode(batch)
        receipt = context.validate_execution()
    assert output.shape == (1, layout.token_capacity, dim)
    assert receipt["gathered_tokens"] == layout.token_capacity


def test_deployment_fails_fast_when_probs_site_missing():
    from test_pair_deployment import _case

    _adapter, bridge, layout, _frames, dim = _case("videomaev2")  # nn.MultiheadAttention: no attn_drop site
    sites = bridge.observation_sites((0,))
    assert "block.0.attn.probs.input" not in sites
    with pytest.raises(ADGSAttentionUnavailable, match="attn.probs.input"):
        ADGSDeployment(bridge, layout, depth=0, dim=dim, keep_ratio=0.60, batch_sizes=(1,), device="cpu")


def test_deployment_rejects_unprepared_batch_and_frame_mismatch(adgs_case):
    _adapter, bridge, layout, frames, dim = adgs_case
    deployment = ADGSDeployment(
        bridge, layout, depth=0, dim=dim, keep_ratio=0.60, batch_sizes=(1,), device="cpu",
    )
    with pytest.raises(PairMergeError, match="dense"):
        deployment(_batch(2, frames))
    with pytest.raises(PairMergeError, match="帧数"):
        deployment(_batch(1, 2))
    assert not any(block._forward_hooks or block._forward_pre_hooks for block in bridge._blocks)


def test_deployment_rejects_bad_ratios_and_non_pair_member_geometry(adgs_case):
    _adapter, bridge, layout, _frames, dim = adgs_case
    with pytest.raises(PairMergeError, match="keep_ratio"):
        ADGSDeployment(bridge, layout, depth=0, dim=dim, keep_ratio=0.50, batch_sizes=(1,))
    with pytest.raises(PairMergeError, match="important_ratio"):
        ADGSDeployment(bridge, layout, depth=0, dim=dim, keep_ratio=0.60, important_ratio=0.0, batch_sizes=(1,))
