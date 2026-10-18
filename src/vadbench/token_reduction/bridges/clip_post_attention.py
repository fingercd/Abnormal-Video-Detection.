"""Audited OpenAI CLIP visual reduction, preserving native attention and CLS.

One context runs one visual forward. Signals are additional same-input Q/K
projections, never a replacement attention backend. Selection sees patches
only; native attention still sees CLS and all patches.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F


_FORWARD = "d8bd873fd7e5bf78b2b0d47c32da54140f282e07bb947e3a0334ad173cccd9ae"
_ATTENTION = "0b267120ff7ebe738ade7553ae8728dcb7f0c900cc9af5552c2561f0dacc62ca"


def _identity(method):
    node = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    # Empty Python 3.12 type_params are the only AST schema normalization.
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and getattr(child, "type_params", None) == []:
            child._fields = tuple(f for f in child._fields if f != "type_params")
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _selector():
    # The legacy Python3.9 launcher cannot import vadbench's top-level registry.
    name = "_vadbench_clip_relation_selection"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "relation_selection.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].select_relational_tokens


class ClipPostAttentionIntervention:
    """Reduce patches after zero-based block ``depth`` attention residual.

    Readout and positional embeddings are native. CLS is kept byte-for-byte
    before the same block's tokenwise norm/MLP. Signals exclude CLS queries
    and keys, including the selection-only softmax denominator.
    """

    def __init__(self, bridge, depth, method, keep_ratio):
        if method not in ("attention_topk", "visionzip_internal", "uniform"):
            raise ValueError("unsupported CLIP relation method")
        bridge._validate_depth(depth)
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0,1]")
        self.bridge, self.depth, self.method, self.keep_ratio = bridge, depth, method, keep_ratio
        self.visual, self.blocks = bridge.visual, bridge.blocks
        for block in self.blocks:
            if "forward" in block.__dict__ or _identity(type(block).forward) != _FORWARD:
                raise ValueError("unsupported native CLIP block forward")
            if "attention" in block.__dict__ or _identity(type(block).attention) != _ATTENTION:
                raise ValueError("unsupported native CLIP attention path")
            attn = block.attn
            if (not isinstance(attn, torch.nn.MultiheadAttention) or attn.batch_first
                    or not attn._qkv_same_embed_dim or attn.bias_k is not None
                    or attn.bias_v is not None or attn.add_zero_attn or attn.dropout != 0
                    or block.attn_mask is not None):
                raise ValueError("unsupported CLIP attention layout/mask")
        self.block = self.blocks[depth]
        self.select = _selector()
        self.handles = []
        self.used = False
        self.active = False
        self.layer_inputs, self.layer_outputs, self.mlp_inputs = {}, {}, {}
        self.before_tokens = self.after_tokens = None
        self.selection = None

    def __enter__(self):
        if self.used or self.active or hasattr(self.visual, "_vadbench_clip_post_attention"):
            raise ValueError("nested/reused CLIP intervention")
        if any(m.training for m in self.visual.modules()):
            raise ValueError("CLIP post-attention requires eval mode")
        for block in self.blocks:
            if "forward" in block.__dict__ or block._forward_hooks or block._forward_pre_hooks:
                raise ValueError("concurrent CLIP block interventions are unsupported")
        setattr(self.visual, "_vadbench_clip_post_attention", self)
        self.active = True
        try:
            self.handles.append(self.visual.register_forward_pre_hook(self._begin))
            for i, block in enumerate(self.blocks):
                self.handles.append(block.register_forward_pre_hook(self._record(self.layer_inputs, i)))
                self.handles.append(block.register_forward_hook(self._record_output(i)))
                self.handles.append(block.mlp.register_forward_pre_hook(self._record(self.mlp_inputs, i)))
            self.block.forward = MethodType(self._forward, self.block)
        except Exception:
            self.close()
            raise
        return self

    def _begin(self, module, inputs):
        if self.used:
            raise ValueError("CLIP context permits exactly one forward")
        self.used = True

    def _record(self, target, index):
        def hook(module, inputs):
            if index in target:
                raise ValueError("reentrant CLIP block execution")
            target[index] = list(inputs[0].shape)
        return hook

    def _record_output(self, index):
        def hook(module, inputs, output):
            self.layer_outputs[index] = list(output.shape)
        return hook

    def _forward(self, block, x):
        normalized = block.ln_1(x)
        hidden = x + block.attention(normalized)
        self.before_tokens = hidden.shape[0]
        if self.keep_ratio == 1:
            reduced = hidden
        else:
            keys = scores = None
            if self.method != "uniform":
                width, heads = block.attn.embed_dim, block.attn.num_heads
                bias = block.attn.in_proj_bias
                # Only Q and K are recomputed; V/native output remain untouched.
                qk = F.linear(normalized, block.attn.in_proj_weight[:2 * width],
                              None if bias is None else bias[:2 * width])
                q, k = qk.chunk(2, dim=-1)
                def head_layout(v):
                    return v.permute(1, 0, 2).reshape(v.shape[1], v.shape[0], heads, width // heads).permute(0, 2, 1, 3)
                q, keys = head_layout(q)[:, :, 1:], head_layout(k)[:, :, 1:]
                probabilities = ((q * (width // heads) ** -0.5) @ keys.transpose(-2, -1)).softmax(-1)
                scores = probabilities.mean(dim=(1, 2))
            self.selection = self.select(hidden.permute(1, 0, 2)[:, 1:], method=self.method,
                                         keep_ratio=self.keep_ratio, keys=keys, attention_scores=scores)
            reduced = torch.cat((hidden[:1], self.selection.tokens.permute(1, 0, 2)), dim=0)
        self.after_tokens = reduced.shape[0]
        return reduced + block.mlp(block.ln_2(reduced))

    def receipt(self):
        expected = set(range(len(self.blocks)))
        if set(self.layer_inputs) != expected or set(self.layer_outputs) != expected or set(self.mlp_inputs) != expected:
            raise ValueError("incomplete CLIP execution")
        for i in expected:
            before = self.before_tokens if i <= self.depth else self.after_tokens
            after = self.before_tokens if i < self.depth else self.after_tokens
            if self.layer_inputs[i][0] != before or self.layer_outputs[i][0] != after or self.mlp_inputs[i][0] != after:
                raise ValueError("CLIP did not execute at contracted shortened lengths")
        return {"site": "post_attention_residual_pre_norm2_mlp", "block_index": self.depth,
                "method": self.method, "keep_ratio": self.keep_ratio,
                "before_tokens": self.before_tokens, "after_tokens": self.after_tokens,
                "cls_preserved": True, "signal": "same_ln1_native_qk_patch_only_softmax",
                "native_attention_unchanged": True, "layer_inputs": self.layer_inputs,
                "layer_outputs": self.layer_outputs, "mlp_inputs": self.mlp_inputs}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.active:
            if "forward" in self.block.__dict__:
                del self.block.forward
            if getattr(self.visual, "_vadbench_clip_post_attention", None) is self:
                delattr(self.visual, "_vadbench_clip_post_attention")
        self.active = False

    def __exit__(self, *args):
        self.close()
