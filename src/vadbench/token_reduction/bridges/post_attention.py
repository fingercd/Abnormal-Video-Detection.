"""Single-forward, instance-local reduction after native attention residual.

Only the audited VideoMAEv2 implementation is supported. Attention execution
is untouched: scores are observed at its actual softmax/dropout boundary.
Functional native QKV has no module hook, so K is recomputed from the *same*
attention input with the native K weight slice, not inferred from hidden states.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import time
from collections.abc import Callable
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostAttentionInterventionError(RuntimeError):
    """The requested native execution cannot be verified safely."""


# AST identities of forward methods in weights/videomaev2-base-hf/
# modeling_videomaev2.py. Ignore formatting only, not semantic changes.
_BLOCK_FORWARD = "10985903796c6b23008db25ddf6eb4b54c9b08d06cde4cb3e5303de94c1a7a63"
_ATTENTION_FORWARD = "dcabbb3fcf71724be3e82235242c8f91e876257387800f8a71093a5b5775359b"
_COS_ATTENTION_FORWARD = "0f32c240660da50135beeebd2c7b9905e8e2c2f905c5ba415fd817911eaf538c"
_OWNER_MARKER = "_vadbench_post_attention_owner"


def _forward_identity(module: nn.Module) -> str:
    if "forward" in module.__dict__:
        raise PostAttentionInterventionError("already overridden instance forward is unsupported")
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(type(module).forward))).body[0]
    except (OSError, TypeError, SyntaxError) as exc:
        raise PostAttentionInterventionError("native forward source identity is unavailable") from exc
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class PostAttentionIntervention:
    """Reduce one native block's residual before its norm2 and MLP.

    ``depth`` is zero-based, including the final block. ``reducer`` receives
    hidden ``[B,N,D]``, keys ``[B,H,N,d]`` or None, received attention
    ``[B,N]`` or None; scores average over heads and source queries. It must
    return a same-device/dtype ``[B,K,D]`` with 0 < K <= N.

    A context supports exactly one model forward in eval mode. It never
    changes global functions, attention output, attention backend, or weights.
    """

    def __init__(
        self,
        bridge: Any,
        depth: int,
        reducer: Callable[[torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor],
        need_keys: bool = True,
        need_attention: bool = True,
    ) -> None:
        self.bridge = bridge
        self.depth = depth
        self.reducer = reducer
        self.need_keys = need_keys
        self.need_attention = need_attention
        receipt = bridge.receipt()
        if receipt.encoder_id != "videomaev2":
            raise PostAttentionInterventionError("post-attention supports audited native VideoMAEv2 only")
        if type(depth) is not int or not 0 <= depth < receipt.block_count:
            raise PostAttentionInterventionError("depth must name an existing zero-based block")
        self.model = bridge.model
        self.blocks = bridge._blocks
        self.block = self.blocks[depth]
        self._attention_kinds: dict[int, str] = {}
        for index, block in enumerate(self.blocks):
            if _forward_identity(block) != _BLOCK_FORWARD:
                raise PostAttentionInterventionError("unsupported native block forward; HF paths fail closed")
            kind = _forward_identity(block.attn)
            if kind not in {_ATTENTION_FORWARD, _COS_ATTENTION_FORWARD}:
                raise PostAttentionInterventionError("unsupported native attention forward/bias/mask handling")
            self._attention_kinds[index] = kind
            for name in ("norm1", "norm2", "mlp", "drop_path"):
                if not isinstance(getattr(block, name, None), nn.Module):
                    raise PostAttentionInterventionError(f"native block lacks {name}")
            attention = block.attn
            if not isinstance(attention.qkv, nn.Linear) or attention.qkv.bias is not None:
                raise PostAttentionInterventionError("native functional QKV requires bias-free qkv module")
            if attention.qkv.out_features % (3 * attention.num_heads):
                raise PostAttentionInterventionError("native QKV/head dimensions are inconsistent")
            if not isinstance(attention.attn_drop, nn.Dropout):
                raise PostAttentionInterventionError("native probability observation requires attn_drop")
            for name in ("relative_position_bias_table", "rel_pos_bias", "attention_mask", "q_norm", "k_norm"):
                value = getattr(attention, name, None)
                if value is not None and not isinstance(value, nn.Identity):
                    raise PostAttentionInterventionError(f"unsupported attention configuration: {name}")
        self._handles: list[Any] = []
        self._active = False
        self._entered = False
        self._used = False
        self._completed = False
        self._original_forward: Any = None
        self._keys: torch.Tensor | None = None
        self._scores: torch.Tensor | None = None
        self.before_shape: tuple[int, ...] | None = None
        self.after_shape: tuple[int, ...] | None = None
        self.layer_shapes: dict[int, tuple[int, ...]] = {}
        self.layer_output_shapes: dict[int, tuple[int, ...]] = {}
        self.mlp_shapes: dict[int, tuple[int, ...]] = {}
        self.key_projection_ms = 0.0
        self.attention_score_ms = 0.0
        self.reducer_ms = 0.0

    def __enter__(self) -> PostAttentionIntervention:
        if self._active or self._entered or hasattr(self.model, _OWNER_MARKER):
            raise PostAttentionInterventionError("nested/reused post-attention context is forbidden")
        if any(module.training for module in self.model.modules()):
            raise PostAttentionInterventionError("post-attention is eval-only")
        if any(getattr(module, "with_cp", False) for module in self.model.modules()):
            raise PostAttentionInterventionError("checkpoint/reentrant native execution is unsupported")
        # Recheck at attachment: another intervention may have been installed
        # since construction, including a block-output gather hook.
        for index, block in enumerate(self.blocks):
            if _forward_identity(block) != _BLOCK_FORWARD or block._forward_hooks or block._forward_pre_hooks:
                raise PostAttentionInterventionError("concurrent native block intervention/hooks are unsupported")
            if _forward_identity(block.attn) != self._attention_kinds[index]:
                raise PostAttentionInterventionError("native attention changed after bridge construction")
        setattr(self.model, _OWNER_MARKER, self)
        self._active = True
        self._entered = True
        try:
            self._handles.append(self.model.register_forward_pre_hook(self._begin_forward))
            self._handles.append(self.model.register_forward_hook(self._end_forward))
            for index, block in enumerate(self.blocks):
                self._handles.append(block.register_forward_pre_hook(self._record_input(index)))
                self._handles.append(block.register_forward_hook(self._record_output(index)))
                self._handles.append(block.mlp.register_forward_pre_hook(self._record_mlp(index)))
            if self.need_keys:
                self._handles.append(self.block.attn.register_forward_pre_hook(self._observe_keys))
            if self.need_attention:
                self._handles.append(self.block.attn.attn_drop.register_forward_pre_hook(self._observe_scores))
            self._original_forward = self.block.forward
            self.block.forward = MethodType(self._forward_block, self.block)
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._original_forward is not None:
            # The audited original is class-defined: remove our instance shadow.
            del self.block.forward
            self._original_forward = None
        if getattr(self.model, _OWNER_MARKER, None) is self:
            delattr(self.model, _OWNER_MARKER)
        self._active = False
        self._keys = self._scores = None

    def _begin_forward(self, _module: Any, _inputs: tuple[Any, ...]) -> None:
        if self._used:
            raise PostAttentionInterventionError("each context permits exactly one forward")
        if any(module.training for module in self.model.modules()):
            raise PostAttentionInterventionError("model changed out of eval mode")
        self._used = True

    def _end_forward(self, _module: Any, _inputs: Any, _output: Any) -> None:
        self._completed = True

    def _record_input(self, depth: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if depth in self.layer_shapes:
                raise PostAttentionInterventionError("native block executed more than once")
            self.layer_shapes[depth] = tuple(inputs[0].shape)
        return hook

    def _record_output(self, depth: int):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.layer_output_shapes[depth] = tuple(output.shape)
        return hook

    def _record_mlp(self, depth: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if depth in self.mlp_shapes:
                raise PostAttentionInterventionError("native MLP executed more than once")
            self.mlp_shapes[depth] = tuple(inputs[0].shape)
        return hook

    def _observe_keys(self, attention: nn.Module, inputs: tuple[Any, ...]) -> None:
        started = time.perf_counter()
        x = inputs[0]
        width = attention.qkv.out_features // 3
        # Native qkv_bias uses an exactly-zero K bias; do not use q_bias/v_bias.
        keys = F.linear(x, attention.qkv.weight[width:2 * width], bias=None)
        keys = keys.reshape(x.shape[0], x.shape[1], attention.num_heads, -1).transpose(1, 2)
        if self._attention_kinds[self.depth] == _COS_ATTENTION_FORWARD:
            keys = F.normalize(keys, dim=-1)
        self._keys = keys
        self.key_projection_ms += (time.perf_counter() - started) * 1000

    def _observe_scores(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        started = time.perf_counter()
        probabilities = inputs[0]
        if probabilities.ndim != 4 or probabilities.shape[-2] != probabilities.shape[-1]:
            raise PostAttentionInterventionError("native attention probabilities are not [B,H,N,N]")
        self._scores = probabilities.mean(dim=(1, 2))
        self.attention_score_ms += (time.perf_counter() - started) * 1000

    def _forward_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if not self._used:
            raise PostAttentionInterventionError("must invoke the bridge model, not a detached block")
        # Same native attention/residual operations and gamma order as Block.forward.
        update = block.attn(block.norm1(x))
        if block.gamma_1 is not None:
            update = block.gamma_1 * update
        hidden = x + block.drop_path(update)
        self.before_shape = tuple(hidden.shape)
        if (self.need_keys and self._keys is None) or (self.need_attention and self._scores is None):
            raise PostAttentionInterventionError("same-forward native signal did not execute")
        started = time.perf_counter()
        reduced = self.reducer(hidden, self._keys, self._scores)
        self.reducer_ms += (time.perf_counter() - started) * 1000
        if (
            not isinstance(reduced, torch.Tensor)
            or reduced.ndim != 3
            or reduced.shape[0] != hidden.shape[0]
            or reduced.shape[2] != hidden.shape[2]
            or not 0 < reduced.shape[1] <= hidden.shape[1]
            or reduced.device != hidden.device
            or reduced.dtype != hidden.dtype
        ):
            raise PostAttentionInterventionError("reducer must return same-device/dtype [B,K,D], 0<K<=N")
        self.after_shape = tuple(reduced.shape)
        self._keys = self._scores = None
        update = block.mlp(block.norm2(reduced))
        if block.gamma_1 is not None:
            update = block.gamma_2 * update
        return reduced + block.drop_path(update)

    def validate_execution(self) -> dict[str, Any]:
        expected = set(range(len(self.blocks)))
        if (
            not self._completed or self.before_shape is None or self.after_shape is None
            or set(self.layer_shapes) != expected or set(self.mlp_shapes) != expected
            or set(self.layer_output_shapes) != expected
        ):
            raise PostAttentionInterventionError("full native model/MLP execution has not completed")
        before, after = self.before_shape[1], self.after_shape[1]
        for depth in expected:
            expected_input = before if depth <= self.depth else after
            expected_output = before if depth < self.depth else after
            if (
                self.layer_shapes[depth][1] != expected_input
                or self.mlp_shapes[depth][1] != expected_output
                or self.layer_output_shapes[depth][1] != expected_output
            ):
                raise PostAttentionInterventionError("observed native token lengths violate post-attention contract")
        suffix = {str(d): list(s) for d, s in self.layer_shapes.items() if d > self.depth}
        return {
            "encoder_id": self.bridge.receipt().encoder_id,
            "intervention_depth": self.depth,
            "block_index": self.depth,
            "site": "post_attention_residual_pre_norm2_mlp",
            "native_input_tokens": before,
            "before_tokens": before,
            "after_tokens": after,
            "gathered_tokens": after,
            "gathered_shape": list(self.after_shape),
            "suffix_shapes": suffix,
            "per_layer_token_counts": {str(d): s[1] for d, s in self.layer_shapes.items()},
            "per_layer_mlp_token_counts": {str(d): s[1] for d, s in self.mlp_shapes.items()},
            "per_layer_output_token_counts": {str(d): s[1] for d, s in self.layer_output_shapes.items()},
            "selected_block_mlp_tokens": self.mlp_shapes[self.depth][1],
            "key_source": "same_forward_attention_input_K_weight_slice_reprojection_not_capture"
            if self.need_keys else "disabled",
            "key_normalization": "native_cosine_l2" if self.need_keys and self._attention_kinds[self.depth] == _COS_ATTENTION_FORWARD else "none",
            "attention_source": "native_softmax_probabilities_before_eval_dropout" if self.need_attention else "disabled",
            "attention_score_definition": "mean_over_heads_and_queries_received_probability",
            "attention_backend_unchanged": True,
            "prefix_recomputed": False,
            "need_keys": self.need_keys,
            "need_attention": self.need_attention,
            "plugin_overhead_ms": {
                "key_projection_ms": self.key_projection_ms,
                "attention_score_ms": self.attention_score_ms,
                "transform_ms": self.reducer_ms,
            },
            "overhead_clock": "host_perf_counter; CUDA enqueue time, not synchronized GPU latency",
        }


__all__ = ["PostAttentionIntervention", "PostAttentionInterventionError"]
