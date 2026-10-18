"""Small synthetic unit tests; no dataset/checkpoint results are produced."""
import pytest
torch = pytest.importorskip("torch")

from vadbench.token_reduction.bridges.clip import ClipVisionBridge
from vadbench.token_reduction.bridges.clip_post_attention import ClipPostAttentionIntervention


class NativeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(8, 2)
        self.ln_1 = torch.nn.LayerNorm(8)
        self.ln_2 = torch.nn.LayerNorm(8)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
        self.attn_mask = None

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Visual(torch.nn.Module):
    input_resolution = 8

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2, 2, bias=False)
        self.class_embedding = torch.nn.Parameter(torch.randn(8))
        self.positional_embedding = torch.nn.Parameter(torch.randn(17, 8))
        self.ln_pre = torch.nn.LayerNorm(8)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.Sequential(NativeBlock(), NativeBlock())
        self.ln_post = torch.nn.LayerNorm(8)
        self.proj = torch.nn.Parameter(torch.randn(8, 4))

    def forward(self, images):
        x = self.conv1(images).flatten(2).transpose(1, 2)
        x = torch.cat((self.class_embedding.expand(images.shape[0], 1, 8), x), 1)
        x = self.ln_pre(x + self.positional_embedding).transpose(0, 1)
        x = self.transformer.resblocks(x)
        return self.ln_post(x[0]) @ self.proj


@pytest.mark.parametrize("method", ["attention_topk", "visionzip_internal", "uniform"])
@pytest.mark.parametrize("ratio,count", [(1., 17), (.5, 9), (.3, 6)])
@pytest.mark.parametrize("depth", [0, 1])
def test_identity_shortening_cls_and_restoration(method, ratio, count, depth):
    torch.manual_seed(0)
    visual = Visual().eval()
    bridge = ClipVisionBridge(visual)
    images = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        dense = visual(images)
        with ClipPostAttentionIntervention(bridge, depth, method, ratio) as intervention:
            result = visual(images)
            receipt = intervention.receipt()
        restored = visual(images)
    assert receipt["after_tokens"] == count
    assert receipt["mlp_inputs"][depth][0] == count
    torch.testing.assert_close(restored, dense, rtol=0, atol=0)
    if ratio == 1:
        torch.testing.assert_close(result, dense, rtol=0, atol=0)
    elif depth == 1:
        torch.testing.assert_close(result, dense, rtol=1e-5, atol=1e-6)
    assert "forward" not in bridge.blocks[depth].__dict__
    assert not hasattr(visual, "_vadbench_clip_post_attention")


def test_context_restores_after_failure_and_rejects_training():
    visual = Visual()
    bridge = ClipVisionBridge(visual)
    with pytest.raises(ValueError, match="eval"):
        with ClipPostAttentionIntervention(bridge, 0, "uniform", .5):
            pass
    visual.eval()
    with pytest.raises(RuntimeError, match="intentional"):
        with ClipPostAttentionIntervention(bridge, 0, "uniform", .5):
            raise RuntimeError("intentional")
    assert "forward" not in bridge.blocks[0].__dict__
    assert not hasattr(visual, "_vadbench_clip_post_attention")
