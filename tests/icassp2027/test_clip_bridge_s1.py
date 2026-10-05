"""Regression test for the F06-S1 full-length CLIP identity gather."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vadbench.token_reduction.bridges.clip import ClipBridgeError, ClipVisionBridge  # noqa: E402


class _Block(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = torch.nn.MultiheadAttention(width, 1, batch_first=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(value)
        update, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        return value + update


class _Visual(torch.nn.Module):
    input_resolution = 4

    def __init__(self) -> None:
        super().__init__()
        width = 4
        self.conv1 = torch.nn.Conv2d(3, width, 2, 2, bias=False)
        self.class_embedding = torch.nn.Parameter(torch.zeros(width))
        self.positional_embedding = torch.nn.Parameter(torch.zeros(5, width))
        self.ln_pre = torch.nn.LayerNorm(width)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([_Block(width), _Block(width)])
        self.ln_post = torch.nn.LayerNorm(width)
        self.proj = torch.nn.Parameter(torch.eye(width))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.conv1(images).reshape(images.shape[0], 4, -1).permute(2, 0, 1)
        cls = self.class_embedding.reshape(1, 1, 4).expand(1, images.shape[0], 4)
        tokens = self.ln_pre(torch.cat((cls, patches), dim=0) + self.positional_embedding[:, None, :])
        for block in self.transformer.resblocks:
            tokens = block(tokens)
        return self.ln_post(tokens[0]) @ self.proj


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _Visual()

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images)


def test_full_native_identity_gather_is_exact_and_reordered_full_sequence_is_rejected() -> None:
    torch.manual_seed(0)
    bridge = ClipVisionBridge(_Model())
    images = torch.randn(2, 3, 4, 4)
    dense = bridge.forward(images)
    identity = bridge.forward(
        images,
        capture_depth=1,
        keep_indices=torch.arange(bridge.receipt().sequence_length),
    )
    torch.testing.assert_close(identity.pooled, dense.pooled, rtol=1e-5, atol=1e-6)
    with pytest.raises(ClipBridgeError, match="原顺序"):
        bridge.forward(images, capture_depth=1, keep_indices=torch.tensor([0, 2, 1, 3, 4]))


def test_pair_select_uses_each_image_scores_and_preserves_cls() -> None:
    bridge = ClipVisionBridge(_Model())
    tokens = torch.zeros(5, 2, 4)
    tokens[:, 0, 0] = torch.tensor([0.0, 10.0, 9.0, 2.0, 1.0])
    tokens[:, 1, 0] = torch.tensor([0.0, 1.0, 2.0, 9.0, 10.0])
    reduced, indices = bridge._pair_select(tokens, 0.80)
    assert reduced.shape == (4, 2, 4)
    assert indices.tolist() == [[0, 1, 2, 3], [0, 2, 3, 4]]
    torch.testing.assert_close(reduced[:, 0], tokens[indices[0], 0])
    torch.testing.assert_close(reduced[:, 1], tokens[indices[1], 1])


def test_pair_select_budget_lengths_for_clip_vit_b16_grid() -> None:
    bridge = ClipVisionBridge(_Model())
    bridge._receipt = bridge._receipt.__class__(
        **{**bridge._receipt.__dict__, "grid_h": 14, "grid_w": 14, "sequence_length": 197}
    )
    tokens = torch.randn(197, 3, 4)
    for keep_ratio, expected in ((0.80, 158), (0.60, 119), (0.40, 79)):
        reduced, indices = bridge._pair_select(tokens, keep_ratio)
        assert reduced.shape == (expected, 3, 4)
        assert indices.shape == (3, expected)
        assert torch.all(indices[:, 0] == 0)
        assert torch.all(indices[:, 1:] > indices[:, :-1])
        torch.testing.assert_close(reduced[:, 0], tokens[indices[0], 0])
