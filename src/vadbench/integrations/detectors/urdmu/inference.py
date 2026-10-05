"""Exact, opt-in long-sequence inference for the pinned UR-DMU Test network.

The author ``Attention`` implementation materializes several ``[B,H,N,N]``
tensors.  Its value for a query token, however, depends only on that query and
the complete, unchanged key/value sequence.  This module computes queries in
chunks while retaining every key and value.  It is an execution strategy for
the fixed author backend, not a new backend, token reduction, windowing, or
approximation.

There is one deliberately unusual compatibility detail.  The upstream code
uses ``attn2 / attn2.sum(-1)`` without ``keepdim=True``.  For a square
``[N,N]`` tensor, PyTorch broadcasts the resulting ``[N]`` denominator along
the *last* dimension.  The denominator consequently belongs to the key index,
not the query row.  The chunked implementation preserves that exact broadcast
direction.
"""

from __future__ import annotations

import copy
import hashlib
import platform
import sys
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch import Tensor, nn

_SCHEMA = "urdmu.query-chunk-exact-attention/v1"
_SOURCE_KEYS = (
    "backend",
    "upstream_dir",
    "upstream_commit",
    "source_sha256",
    "source_hash_basis",
    "raw_source_sha256",
    "selected_definitions",
    "executed_definition_ast_sha256",
    "device_patches",
    "torch_version",
    "einops_version",
    "einops_file",
    "checkpoint",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_chunk_size(query_chunk_size: int) -> int:
    if type(query_chunk_size) is not int or query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be a positive integer")
    return query_chunk_size


def _iter_attention_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield only the two reviewed author Attention instances, or fail closed."""

    found: list[tuple[str, nn.Module]] = []
    for path, module in model.named_modules():
        if type(module).__name__ != "Attention":
            continue
        if type(module).__module__ != "vadbench.integrations.detectors.urdmu.backend":
            raise ValueError(f"{path}: query-chunk inference only accepts the pinned UR-DMU Attention")
        required = ("heads", "scale", "attend", "to_qkv", "to_out")
        if not all(hasattr(module, name) for name in required):
            raise ValueError(f"{path}: an unreviewed Attention-like module cannot be chunked")
        if not isinstance(module.to_qkv, nn.Linear) or module.to_qkv.bias is not None:
            raise ValueError(f"{path}: UR-DMU Attention.to_qkv no longer has the reviewed layout")
        if not isinstance(module.attend, nn.Softmax) or module.attend.dim != -1:
            raise ValueError(f"{path}: UR-DMU Attention softmax differs from the reviewed definition")
        if type(module.heads) is not int or module.heads <= 0:
            raise ValueError(f"{path}: UR-DMU Attention heads are invalid")
        inner_times_four = module.to_qkv.out_features
        if inner_times_four % (4 * module.heads):
            raise ValueError(f"{path}: UR-DMU Attention Q/K/V/T layout is invalid")
        found.append((path, module))
    if len(found) != 2:
        raise ValueError(
            "exact query-chunk inference accepts only the reviewed two-layer UR-DMU self-attention"
        )
    yield from found


def _validate_inference_state(model: nn.Module) -> None:
    if model.training:
        raise ValueError("query-chunk attention is eval-only; call model.eval() first")
    if getattr(model, "flag", None) != "Test":
        raise ValueError("query-chunk attention requires the author's model.flag == 'Test'")
    enabled_dropout = [
        path
        for path, module in model.named_modules()
        if isinstance(module, nn.Dropout) and module.training
    ]
    if enabled_dropout:
        raise ValueError(
            "query-chunk attention rejects active dropout modules: " + ", ".join(enabled_dropout)
        )


def _validate_input(inputs: Tensor) -> None:
    if not torch.is_tensor(inputs):
        raise TypeError("UR-DMU query-chunk input must be a torch Tensor")
    if inputs.ndim not in {3, 4}:
        raise ValueError("UR-DMU query-chunk input must have shape [B,T,D] or [B,crops,T,D]")
    if inputs.shape[-2] <= 0:
        raise ValueError("UR-DMU query-chunk input sequence must be nonempty")
    if inputs.requires_grad:
        raise ValueError("query-chunk attention is inference-only and rejects inputs requiring gradients")
    if not torch.is_floating_point(inputs):
        raise ValueError("UR-DMU query-chunk input must have a floating-point dtype")


def state_dict_sha256(model: nn.Module) -> str:
    """Fingerprint parameter/buffer values without changing model state."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(name.encode("utf8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()


def _source_identity(source_receipt: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, bool]:
    if source_receipt is None:
        return None, False
    copied = {key: copy.deepcopy(source_receipt[key]) for key in _SOURCE_KEYS if key in source_receipt}
    complete = (
        copied.get("backend") == "urdmu"
        and isinstance(copied.get("upstream_commit"), str)
        and isinstance(copied.get("source_sha256"), Mapping)
        and isinstance(copied.get("executed_definition_ast_sha256"), Mapping)
    )
    return copied, complete


def estimate_query_chunk_attention_workspace(
    *,
    batch_size: int,
    sequence_length: int,
    query_chunk_size: int,
    heads: int = 4,
    dim_head: int = 128,
    activation_dtype: torch.dtype = torch.float32,
) -> dict[str, int | str]:
    """Return a conservative tensor-workspace lower bound for one attention layer.

    This deliberately excludes parameters, allocator fragmentation, cuBLAS
    workspaces and upstream convolution/memory modules.  It includes the
    assembled attention output because query chunks must be retained to pass
    the complete sequence to the next author layer.  It is useful for rejecting
    budgets that are already too small, never as a promise that a process will
    fit.  The position branch is float32 because that is what the pinned author
    code constructs.
    """

    for name, value in (
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("heads", heads),
        ("dim_head", dim_head),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _validate_chunk_size(query_chunk_size)
    if not isinstance(activation_dtype, torch.dtype) or not torch.empty(
        (), dtype=activation_dtype
    ).is_floating_point():
        raise ValueError("activation_dtype must be a floating-point torch dtype")

    query_length = min(sequence_length, query_chunk_size)
    activation_bytes = torch.empty((), dtype=activation_dtype).element_size()
    float32_bytes = torch.empty((), dtype=torch.float32).element_size()
    # to_qkv is live while its four views are used.  The two score tensors are
    # ``dots`` and ``attn1``.  The temporal branch retains distance, unnormalised
    # weights and the broadcast-normalised weight for one query chunk.
    qkvt = batch_size * sequence_length * 4 * heads * dim_head * activation_bytes
    appearance_scores = 2 * batch_size * heads * query_length * sequence_length * activation_bytes
    temporal_scores = 3 * query_length * sequence_length * float32_bytes
    chunk_outputs = 2 * batch_size * heads * query_length * dim_head * activation_bytes
    assembled_output = 2 * batch_size * heads * sequence_length * dim_head * activation_bytes
    # ``permute(...).reshape(...)`` can need a contiguous output copy, just as
    # the author's einops rearrange can.  Include it in the lower bound.
    output_relayout = assembled_output
    temporal_vectors = sequence_length * float32_bytes
    lower_bound = (
        qkvt
        + appearance_scores
        + temporal_scores
        + chunk_outputs
        + assembled_output
        + output_relayout
        + temporal_vectors
    )
    return {
        "schema": _SCHEMA,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "query_chunk_size": query_chunk_size,
        "effective_query_chunk_size": query_length,
        "heads": heads,
        "dim_head": dim_head,
        "activation_dtype": str(activation_dtype),
        "qkvt_bytes": qkvt,
        "appearance_score_bytes": appearance_scores,
        "temporal_position_bytes": temporal_scores,
        "temporal_denominator_bytes": temporal_vectors,
        "chunk_output_bytes": chunk_outputs,
        "assembled_attention_output_bytes": assembled_output,
        "output_relayout_bytes": output_relayout,
        "workspace_lower_bound_bytes": lower_bound,
        "workspace_lower_bound_mib": lower_bound / (1024 * 1024),
    }


def _temporal_prior(query_positions: Tensor, key_positions: Tensor) -> Tensor:
    """Match the author's ``exp(-abs(tmp_n - tmp_n.view(-1, 1)) / exp(1))``."""

    # ``key_positions`` has shape [N] and query_positions [Q].  The original
    # multiplication by ones is exact and only turns ``tmp_n`` into this row.
    # Keep torch.tensor(1.) on its default CPU scalar device as in the reviewed
    # source; PyTorch permits it in a CUDA scalar expression.
    distance = torch.abs(key_positions - query_positions.unsqueeze(1))
    return torch.exp(-distance / torch.exp(torch.tensor(1.0)))


def _temporal_denominators(
    key_positions: Tensor,
    *,
    query_chunk_size: int,
) -> Tensor:
    """Compute each original row sum without creating the complete N×N matrix."""

    rows = []
    for start in range(0, key_positions.numel(), query_chunk_size):
        prior = _temporal_prior(key_positions[start : start + query_chunk_size], key_positions)
        rows.append(prior.sum(-1))
    return torch.cat(rows, dim=0)


def _query_chunked_attention_forward(attention: nn.Module, x: Tensor, *, query_chunk_size: int) -> Tensor:
    """The author Attention forward with exact all-key query chunks."""

    b, n, _ = x.size()
    qkvt = attention.to_qkv(x).chunk(4, dim=-1)
    inner_dim = qkvt[0].shape[-1]
    dim_head = inner_dim // attention.heads
    q, k, v, t = (
        value.reshape(b, n, attention.heads, dim_head).permute(0, 2, 1, 3)
        for value in qkvt
    )

    # The pinned source creates both vectors at default float32 then moves them
    # to x.device.  Do the same even if a caller selected a dtype that upstream
    # itself cannot execute with this positional branch.
    key_positions = torch.linspace(1, n, n, device=x.device)
    denominators = _temporal_denominators(key_positions, query_chunk_size=query_chunk_size)
    # This O(B*N*D) output is necessary for the next fixed author layer.  A
    # preallocated result avoids retaining all chunk objects and another large
    # concatenation buffer; it never materializes an N×N attention map.
    out = torch.empty(
        (b, attention.heads, n, 2 * dim_head), device=x.device, dtype=q.dtype
    )
    key_transpose = k.transpose(-1, -2)
    for start in range(0, n, query_chunk_size):
        end = min(start + query_chunk_size, n)
        dots = torch.matmul(q[:, :, start:end], key_transpose) * attention.scale
        attn1 = attention.attend(dots)
        appearance = torch.matmul(attn1, v)

        prior = _temporal_prior(key_positions[start:end], key_positions)
        # Do not change this to denominators[start:end].unsqueeze(-1): the
        # upstream ``[N,N] / [N]`` operation aligns this vector with key columns.
        attn2 = prior / denominators
        temporal = torch.matmul(attn2, t)
        out[:, :, start:end] = torch.cat((appearance, temporal), dim=-1)
    out = out.permute(0, 2, 1, 3).reshape(b, n, 2 * inner_dim)
    return attention.to_out(out)


@dataclass
class QueryChunkedAttention(AbstractContextManager["QueryChunkedAttention"]):
    """Temporarily install exact query-chunk forwards on fixed UR-DMU attention.

    Enter this context only beneath ``torch.no_grad()`` or
    ``torch.inference_mode()``.  The original instance forwards are restored on
    normal exit and on every exception.
    """

    model: nn.Module
    query_chunk_size: int
    source_receipt: Mapping[str, Any] | None = None
    _patched: list[tuple[nn.Module, bool, Any]] | None = None
    _modules: list[tuple[str, nn.Module]] | None = None

    def __enter__(self) -> QueryChunkedAttention:
        _validate_chunk_size(self.query_chunk_size)
        _validate_inference_state(self.model)
        if torch.is_grad_enabled():
            raise ValueError(
                "query-chunk attention is inference-only; enter torch.no_grad() or torch.inference_mode()"
            )
        self._modules = list(_iter_attention_modules(self.model))
        self._patched = []
        try:
            for _, attention in self._modules:
                had_instance_forward = "forward" in attention.__dict__
                previous = attention.__dict__.get("forward")

                def forward(module: nn.Module, x: Tensor, *, _chunk: int = self.query_chunk_size) -> Tensor:
                    return _query_chunked_attention_forward(module, x, query_chunk_size=_chunk)

                self._patched.append((attention, had_instance_forward, previous))
                attention.forward = MethodType(forward, attention)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self) -> None:
        if self._patched is None:
            return
        for attention, had_instance_forward, previous in reversed(self._patched):
            if had_instance_forward:
                attention.forward = previous
            else:
                delattr(attention, "forward")
        self._patched = None

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._restore()

    def receipt(
        self,
        inputs: Tensor,
        *,
        state_dict_before_sha256: str | None = None,
        state_dict_after_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Describe this execution; caller supplies hashes when it requests them."""

        _validate_input(inputs)
        source, source_complete = _source_identity(self.source_receipt)
        modules = self._modules if self._modules is not None else list(_iter_attention_modules(self.model))
        sequence_length = int(inputs.shape[-2])
        attention_modules = []
        for path, module in modules:
            inner_dim = module.to_qkv.out_features // 4
            attention_modules.append(
                {
                    "path": path,
                    "class_module": type(module).__module__,
                    "class_name": type(module).__qualname__,
                    "heads": module.heads,
                    "dim_head": inner_dim // module.heads,
                    "scale": float(module.scale),
                    "qkv_out_features": module.to_qkv.out_features,
                }
            )
        state = None
        if state_dict_before_sha256 is not None or state_dict_after_sha256 is not None:
            state = {
                "before_sha256": state_dict_before_sha256,
                "after_sha256": state_dict_after_sha256,
                "unchanged": state_dict_before_sha256 == state_dict_after_sha256,
            }
        return {
            "schema": _SCHEMA,
            "status": "completed",
            "execution": "eval_only_exact_all_key_query_chunk_attention",
            "approximation": False,
            "token_reduction": False,
            "complete_key_value_sequence": True,
            "sequence_length": sequence_length,
            "query_chunk_size": self.query_chunk_size,
            "query_chunk_count": (sequence_length + self.query_chunk_size - 1) // self.query_chunk_size,
            "input_shape": list(inputs.shape),
            "input_dtype": str(inputs.dtype),
            "input_device": str(inputs.device),
            "attention_modules": attention_modules,
            "temporal_prior": {
                "formula": "exp(-abs(j-i)/exp(1))",
                "normalization": "upstream attn2 / attn2.sum(-1)",
                "broadcast_direction": "last_dimension_key_index",
            },
            "inference_constraints": {
                "model_training": self.model.training,
                "model_flag": getattr(self.model, "flag", None),
                "grad_enabled_during_patch": torch.is_grad_enabled(),
                "active_dropout": any(
                    isinstance(module, nn.Dropout) and module.training
                    for module in self.model.modules()
                ),
            },
            "memory_estimate": estimate_query_chunk_attention_workspace(
                batch_size=int(inputs.reshape(-1, sequence_length, inputs.shape[-1]).shape[0]),
                sequence_length=sequence_length,
                query_chunk_size=self.query_chunk_size,
                heads=attention_modules[0]["heads"],
                dim_head=attention_modules[0]["dim_head"],
                activation_dtype=inputs.dtype,
            ),
            "source_identity": source,
            "source_identity_complete": source_complete,
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "state_dict": state,
        }


def query_chunked_attention(
    model: nn.Module,
    *,
    query_chunk_size: int,
    source_receipt: Mapping[str, Any] | None = None,
) -> QueryChunkedAttention:
    """Return the explicit opt-in context for exact UR-DMU Test attention."""

    return QueryChunkedAttention(model, query_chunk_size, source_receipt)


def run_urdmu_query_chunked(
    model: nn.Module,
    inputs: Tensor,
    *,
    query_chunk_size: int,
    source_receipt: Mapping[str, Any] | None = None,
    return_receipt: bool = False,
    fingerprint_state_dict: bool = True,
) -> Mapping[str, Tensor] | tuple[Mapping[str, Tensor], dict[str, Any]]:
    """Run the unmodified UR-DMU Test network with exact all-key query chunks.

    The runner owns an inference-mode scope.  It rejects an active train mode,
    author Train flag, active dropout, and gradient-requiring input before it
    patches anything.  ``return_receipt=True`` returns the original result
    mapping plus a source/runtime/memory receipt suitable for a scoring stage.
    """

    _validate_input(inputs)
    _validate_inference_state(model)
    before = state_dict_sha256(model) if fingerprint_state_dict else None
    with torch.inference_mode(), query_chunked_attention(
        model,
        query_chunk_size=query_chunk_size,
        source_receipt=source_receipt,
    ) as execution:
        output = model(inputs)
        after = state_dict_sha256(model) if fingerprint_state_dict else None
        receipt = execution.receipt(
            inputs,
            state_dict_before_sha256=before,
            state_dict_after_sha256=after,
        )
    if receipt["state_dict"] is not None and not receipt["state_dict"]["unchanged"]:
        raise RuntimeError("query-chunk inference changed UR-DMU state_dict")
    if not isinstance(output, Mapping):
        raise RuntimeError("reviewed UR-DMU Test forward did not return a result mapping")
    if return_receipt:
        return output, receipt
    return output


__all__ = [
    "QueryChunkedAttention",
    "estimate_query_chunk_attention_workspace",
    "query_chunked_attention",
    "run_urdmu_query_chunked",
    "state_dict_sha256",
]
