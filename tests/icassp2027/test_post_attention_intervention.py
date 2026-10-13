"""CPU unit tests of audited native code, without loading model weights."""

from __future__ import annotations

import ast
import copy
import hashlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.bridges.post_attention import (
    PostAttentionIntervention,
    PostAttentionInterventionError,
    _canonical_ast_dump,
    _forward_identity,
)


@pytest.fixture(scope="module")
def native_classes():
    # Compile only the real tiny native modules; avoid transformers/timm and
    # any pretrained model initialization/download. Retain source identity.
    path = Path(__file__).resolve().parents[2] / "weights/videomaev2-base-hf/modeling_videomaev2.py"
    if not path.is_file():
        pytest.skip("audited native VideoMAEv2 source asset is not installed")
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name in {"Mlp", "Attention", "CosAttention", "Block"}]
    module = ModuleType("vadbench_test_native_post_attention")
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, nn=nn, F=F)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), module.__dict__)
    yield module
    del sys.modules[module.__name__]


class TinyNativeModel(nn.Module):
    def __init__(self, native, *, cosine=False, gamma=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            native.Block(dim=8, num_heads=2, mlp_ratio=2, qkv_bias=True,
                         init_values=gamma, cos_attn=cosine)
            for _ in range(3)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def make_model(native, **kwargs):
    torch.manual_seed(82)
    model = TinyNativeModel(native, **kwargs).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.attn.q_bias.normal_()
            block.attn.v_bias.normal_()
    return model, create_observation_bridge("videomaev2", model), torch.randn(2, 7, 8)


@pytest.mark.parametrize("cosine", [False, True])
@pytest.mark.parametrize("gamma", [0.0, 0.7])
@pytest.mark.parametrize("depth", [0, 2])
def test_capture_on_identity_matches_native_and_same_forward_qkv(native_classes, cosine, gamma, depth):
    model, bridge, x = make_model(native_classes, cosine=cosine, gamma=gamma)
    selected = model.blocks[depth]
    reference = {}

    def save_qkv(attention, inputs):
        hidden = inputs[0]
        bias = torch.cat((attention.q_bias, torch.zeros_like(attention.v_bias), attention.v_bias))
        qkv = F.linear(hidden, attention.qkv.weight, bias)
        qkv = qkv.reshape(hidden.shape[0], hidden.shape[1], 3, attention.num_heads, -1).permute(2, 0, 3, 1, 4)
        keys = F.normalize(qkv[1], dim=-1) if cosine else qkv[1]
        reference["keys"] = keys

    def save_scores(_module, inputs):
        reference["scores"] = inputs[0].mean(dim=(1, 2))

    def reducer(hidden, keys, scores):
        torch.testing.assert_close(keys, reference["keys"], rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(scores, reference["scores"], rtol=0, atol=0)
        torch.testing.assert_close(scores.sum(dim=-1), torch.ones(x.shape[0]))
        return hidden

    with torch.no_grad():
        expected = model(x)
        handles = [selected.attn.register_forward_pre_hook(save_qkv),
                   selected.attn.attn_drop.register_forward_pre_hook(save_scores)]
        try:
            with PostAttentionIntervention(bridge, depth, reducer) as context:
                actual = model(x)
                receipt = context.validate_execution()
        finally:
            for handle in handles:
                handle.remove()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert receipt["gathered_tokens"] == 7
    assert receipt["attention_backend_unchanged"]
    assert "not_capture" in receipt["key_source"]
    assert "forward" not in selected.__dict__


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("signals", [(False, False), (True, False), (False, True), (True, True)])
def test_shrink_occurs_before_this_blocks_mlp(native_classes, depth, signals):
    model, bridge, x = make_model(native_classes, gamma=0.6)
    control = copy.deepcopy(model)
    need_keys, need_attention = signals

    def reducer(hidden, keys, scores):
        assert (keys is not None) == need_keys
        assert (scores is not None) == need_attention
        if need_keys:
            assert keys.shape == (2, 2, 7, 4)
        if need_attention:
            assert scores.shape == (2, 7)
        return hidden[:, [0, 2, 5]]

    with torch.no_grad():
        expected = x
        for index, block in enumerate(control.blocks):
            if index != depth:
                expected = block(expected)
                continue
            expected = expected + block.drop_path(block.gamma_1 * block.attn(block.norm1(expected)))
            expected = expected[:, [0, 2, 5]]
            expected = expected + block.drop_path(block.gamma_2 * block.mlp(block.norm2(expected)))
        with PostAttentionIntervention(bridge, depth, reducer, *signals) as context:
            actual = model(x)
            receipt = context.validate_execution()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 3, 8)
    assert receipt["site"] == "post_attention_residual_pre_norm2_mlp"
    assert receipt["before_tokens"] == 7 and receipt["after_tokens"] == 3
    assert receipt["selected_block_mlp_tokens"] == 3
    assert receipt["per_layer_token_counts"] == {str(d): 7 if d <= depth else 3 for d in range(3)}
    assert receipt["per_layer_mlp_token_counts"] == {str(d): 7 if d < depth else 3 for d in range(3)}
    if not need_keys:
        assert receipt["plugin_overhead_ms"]["key_projection_ms"] == 0
    if not need_attention:
        assert receipt["plugin_overhead_ms"]["attention_score_ms"] == 0
    assert not any(block._forward_hooks or block._forward_pre_hooks for block in model.blocks)


def test_uniform_has_no_signal_computation(native_classes, monkeypatch):
    model, bridge, x = make_model(native_classes)
    context = PostAttentionIntervention(bridge, 1, lambda h, k, a: h[:, :3], False, False)
    def forbidden(*_args):
        raise AssertionError("uniform should not calculate a signal")
    monkeypatch.setattr(context, "_observe_keys", forbidden)
    monkeypatch.setattr(context, "_observe_scores", forbidden)
    with context, torch.no_grad():
        model(x)
        context.validate_execution()


def test_exception_cleanup_restores_original_forward_and_hooks(native_classes):
    model, bridge, x = make_model(native_classes)
    with torch.no_grad():
        expected = model(x)
    original = model.blocks[1].forward
    def fail(*_args):
        raise ValueError("intentional reducer failure")
    context = PostAttentionIntervention(bridge, 1, fail)
    with pytest.raises(ValueError, match="intentional"):
        with context, torch.no_grad():
            model(x)
    assert model.blocks[1].forward == original
    assert not model._forward_hooks and not model._forward_pre_hooks
    with torch.no_grad():
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    with pytest.raises(PostAttentionInterventionError, match="completed"):
        context.validate_execution()


def test_nested_repeated_forward_and_reused_context_rejected(native_classes):
    model, bridge, x = make_model(native_classes)
    first = PostAttentionIntervention(bridge, 1, lambda h, k, a: h)
    second = PostAttentionIntervention(bridge, 2, lambda h, k, a: h)
    with first, torch.no_grad():
        with pytest.raises(PostAttentionInterventionError, match="nested"):
            second.__enter__()
        model(x)
        with pytest.raises(PostAttentionInterventionError, match="exactly one"):
            model(x)
    with pytest.raises(PostAttentionInterventionError, match="reused"):
        first.__enter__()
    with torch.no_grad():
        model(x)


@pytest.mark.parametrize("bad", [lambda h: h[:, :0], lambda h: h[:, :, :2], lambda h: h.double(), lambda h: None])
def test_invalid_reducer_output_fails_closed(native_classes, bad):
    model, bridge, x = make_model(native_classes)
    with pytest.raises(PostAttentionInterventionError, match="reducer must"):
        with PostAttentionIntervention(bridge, 2, lambda h, k, a: bad(h)), torch.no_grad():
            model(x)
    assert "forward" not in model.blocks[2].__dict__


def test_unsupported_configuration_and_eval_gate(native_classes):
    model, bridge, x = make_model(native_classes)
    model.train()
    with pytest.raises(PostAttentionInterventionError, match="eval-only"):
        with PostAttentionIntervention(bridge, 1, lambda h, k, a: h):
            pass
    model.eval()
    model.blocks[1].attn.k_norm = nn.LayerNorm(4)
    with pytest.raises(PostAttentionInterventionError, match="k_norm"):
        PostAttentionIntervention(bridge, 1, lambda h, k, a: h)
    del model.blocks[1].attn.k_norm
    model.with_cp = True
    with pytest.raises(PostAttentionInterventionError, match="checkpoint"):
        with PostAttentionIntervention(bridge, 1, lambda h, k, a: h):
            pass
    model.with_cp = False
    bridge._receipt = type(bridge.receipt())(**{**bridge.receipt().__dict__, "encoder_id": "videomae"})
    with pytest.raises(PostAttentionInterventionError, match="VideoMAEv2 only"):
        PostAttentionIntervention(bridge, 1, lambda h, k, a: h)


def test_receipt_requires_full_model_forward(native_classes):
    model, bridge, x = make_model(native_classes)
    with PostAttentionIntervention(bridge, 2, lambda h, k, a: h) as context:
        with pytest.raises(PostAttentionInterventionError, match="completed"):
            context.validate_execution()
        with pytest.raises(PostAttentionInterventionError, match="bridge model"):
            model.blocks[2](x)


def test_semantically_different_attention_forward_rejected(native_classes):
    model, bridge, _ = make_model(native_classes)
    class UnsupportedAttention(native_classes.Attention):
        def forward(self, x):
            return super().forward(x) + 1
    model.blocks[1].attn = UnsupportedAttention(8, num_heads=2)
    with pytest.raises(PostAttentionInterventionError, match="attention forward"):
        PostAttentionIntervention(bridge, 1, lambda h, k, a: h)


def test_body_exception_and_unused_context_reuse_cleanup(native_classes):
    model, bridge, _ = make_model(native_classes)
    context = PostAttentionIntervention(bridge, 1, lambda h, k, a: h)
    with pytest.raises(ValueError, match="body failure"):
        with context:
            raise ValueError("body failure")
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert "forward" not in model.blocks[1].__dict__
    with pytest.raises(PostAttentionInterventionError, match="reused"):
        context.__enter__()


def test_constructor_does_not_authorize_later_attention_override(native_classes):
    model, bridge, _ = make_model(native_classes)
    context = PostAttentionIntervention(bridge, 1, lambda h, k, a: h)
    model.blocks[1].attn.forward = lambda x: x
    with pytest.raises(PostAttentionInterventionError, match="overridden"):
        context.__enter__()
    del model.blocks[1].attn.forward


def test_native_attention_and_mlp_execute_once(native_classes):
    model, bridge, x = make_model(native_classes)
    counts = {"attention": 0, "mlp": 0}
    def count(kind):
        def hook(_module, _inputs, _output):
            counts[kind] += 1
        return hook
    handles = [model.blocks[1].attn.register_forward_hook(count("attention")),
               model.blocks[1].mlp.register_forward_hook(count("mlp"))]
    try:
        with PostAttentionIntervention(bridge, 1, lambda h, k, a: h[:, :3]) as context, torch.no_grad():
            model(x)
            context.validate_execution()
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"attention": 1, "mlp": 1}


@pytest.mark.parametrize("name", ["Block", "Attention", "CosAttention"])
def test_empty_type_params_schema_difference_is_semantically_canonical(native_classes, name):
    import inspect
    import textwrap

    module_type = getattr(native_classes, name)
    node = ast.parse(textwrap.dedent(inspect.getsource(module_type.forward))).body[0]
    python310 = copy.deepcopy(node)
    python312 = copy.deepcopy(node)
    # Simulate both schemas even when the test interpreter is not Python 3.12.
    for old, new in zip(ast.walk(python310), ast.walk(python312), strict=True):
        if isinstance(old, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            old._fields = tuple(field for field in old._fields if field != "type_params")
            if hasattr(old, "type_params"):
                del old.type_params
            if "type_params" not in new._fields:
                new._fields = (*new._fields, "type_params")
            new.type_params = []
    assert ast.dump(python310, include_attributes=False) != ast.dump(python312, include_attributes=False)
    assert _canonical_ast_dump(python310) == _canonical_ast_dump(python312)
    module = module_type(8, num_heads=2, init_values=0.0) if name == "Block" else module_type(8, num_heads=2)
    assert _forward_identity(module) == hashlib.sha256(_canonical_ast_dump(python310).encode()).hexdigest()


def test_canonical_ast_preserves_nonempty_type_params_and_string_literals():
    node = ast.parse('def forward(self, x):\n    return x + ", type_params=[]"').body[0]
    with_parameters = copy.deepcopy(node)
    if "type_params" not in with_parameters._fields:
        with_parameters._fields = (*with_parameters._fields, "type_params")
    with_parameters.type_params = [ast.Name(id="NonemptyParameter", ctx=ast.Load())]
    assert _canonical_ast_dump(node) != _canonical_ast_dump(with_parameters)
    assert ", type_params=[]" in _canonical_ast_dump(node)


def test_canonical_ast_preserves_changed_operation():
    add = ast.parse("def forward(self, x):\n    return x + 1").body[0]
    multiply = ast.parse("def forward(self, x):\n    return x * 1").body[0]
    assert _canonical_ast_dump(add) != _canonical_ast_dump(multiply)
