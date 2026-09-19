"""Bounded, read-only reconstruction of selected SDPA query rows.

The observer intentionally sits at the attention-dispatch boundary.  It does
not replace SDPA or request native attention probabilities: the original
backend is invoked once and its result is returned unchanged.  For selected
modules in the narrow V-JEPA eval path, it additionally reconstructs a fixed
set of pre-dropout probability rows from the post-RoPE Q and K tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import nn


class UnsupportedSDPAObservationError(RuntimeError):
    """Raised when a selected call is outside the audited reconstruction path."""


@dataclass(frozen=True)
class SDPAQueryCapture:
    """One bounded reconstruction emitted after the native backend returns.

    ``sampled_probabilities`` has shape ``(batch, heads, sampled_queries,
    full_keys)`` and is detached.  ``query_ids`` indexes the original query
    axis exactly; it is shared by every batch item and head.
    """

    site: str
    sampled_probabilities: torch.Tensor
    query_ids: torch.Tensor
    metadata: Mapping[str, Any]


CaptureCallback = Callable[[SDPAQueryCapture], None]


class SDPAQueryRowObserver:
    """Temporarily wrap a registry's ``sdpa`` backend for selected modules.

    ``registry`` may be Hugging Face's ``AttentionInterface`` or a compatible
    mutable mapping.  AttentionInterface has a global mapping plus an
    instance-local override; this class restores the prior *local* state, so a
    temporary wrapper never leaks into later calls.
    """

    _IMPLEMENTATION = "sdpa"

    def __init__(
        self,
        registry: MutableMapping[str, Callable[..., Any]],
        targets: Mapping[str, nn.Module],
        on_capture: CaptureCallback,
        max_queries: int,
        source_identity: Mapping[str, Any],
    ) -> None:
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        if not targets:
            raise ValueError("targets must not be empty")
        if not isinstance(source_identity, Mapping):
            raise TypeError("source_identity must be a mapping of source fingerprints")

        sites_by_id: dict[int, str] = {}
        for site, module in targets.items():
            if not isinstance(site, str) or not site:
                raise ValueError("target sites must be non-empty strings")
            if not isinstance(module, nn.Module):
                raise TypeError(f"target {site!r} is not a torch module")
            previous = sites_by_id.setdefault(id(module), site)
            if previous != site:
                raise ValueError("one attention module cannot be assigned to multiple observer sites")

        self._registry = registry
        self._sites_by_id = sites_by_id
        self._on_capture = on_capture
        self._max_queries = max_queries
        self._source_identity = dict(source_identity)
        self._active = False
        self._had_local_override = False
        self._previous_local_backend: Callable[..., Any] | None = None
        self._previous_mapping_backend: Callable[..., Any] | None = None

    def __enter__(self) -> SDPAQueryRowObserver:
        if self._active:
            raise RuntimeError("SDPAQueryRowObserver cannot be entered twice concurrently")

        self._snapshot_backend_state()
        self._registry[self._IMPLEMENTATION] = self._wrapped_sdpa
        self._active = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self._restore_backend_state()
        finally:
            self._active = False
        return False

    def _snapshot_backend_state(self) -> None:
        local_mapping = getattr(self._registry, "_local_mapping", None)
        if isinstance(local_mapping, MutableMapping):
            self._had_local_override = self._IMPLEMENTATION in local_mapping
            self._previous_local_backend = local_mapping.get(self._IMPLEMENTATION)
            self._previous_mapping_backend = None
            return

        self._had_local_override = self._IMPLEMENTATION in self._registry
        self._previous_mapping_backend = (
            self._registry[self._IMPLEMENTATION] if self._had_local_override else None
        )
        self._previous_local_backend = None

    def _restore_backend_state(self) -> None:
        local_mapping = getattr(self._registry, "_local_mapping", None)
        if isinstance(local_mapping, MutableMapping):
            if self._had_local_override:
                assert self._previous_local_backend is not None
                self._registry[self._IMPLEMENTATION] = self._previous_local_backend
            else:
                # AttentionInterface.__delitem__ removes only a local key,
                # thereby exposing its original global backend again.
                del self._registry[self._IMPLEMENTATION]
            return

        if self._had_local_override:
            assert self._previous_mapping_backend is not None
            self._registry[self._IMPLEMENTATION] = self._previous_mapping_backend
        else:
            del self._registry[self._IMPLEMENTATION]

    def _wrapped_sdpa(self, *args: Any, **kwargs: Any) -> Any:
        backend = self._native_backend()
        module = self._find_module(args, kwargs)
        site = self._sites_by_id.get(id(module)) if module is not None else None
        if site is None:
            return backend(*args, **kwargs)

        query, key, value, attention_mask = self._extract_call_arguments(args, kwargs)
        effective_scale, native_scale = self._validate_selected_call(
            module, query, key, value, attention_mask, kwargs
        )
        result = backend(*args, **kwargs)
        capture = self._reconstruct_capture(
            site, query, key, value, effective_scale, native_scale
        )
        self._on_capture(capture)
        return result

    def _native_backend(self) -> Callable[..., Any]:
        # Plain mutable mappings retain their original backend in this slot.
        # Check it before the AttentionInterface-local branch: both kinds of
        # registry can have a pre-existing ``sdpa`` entry.
        if self._previous_mapping_backend is not None:
            return self._previous_mapping_backend
        if self._had_local_override:
            assert self._previous_local_backend is not None
            return self._previous_local_backend
        # A registry with a global mapping has no local ``sdpa`` item.  Looking
        # it up after installation would recurse into this wrapper, so resolve
        # the global backend directly.
        global_mapping = getattr(self._registry, "_global_mapping", None)
        if isinstance(global_mapping, Mapping) and self._IMPLEMENTATION in global_mapping:
            backend = global_mapping[self._IMPLEMENTATION]
            if callable(backend):
                return backend
        raise RuntimeError("SDPA registry did not expose a native backend")

    @staticmethod
    def _find_module(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> nn.Module | None:
        candidate = kwargs.get("module", args[0] if args else None)
        return candidate if isinstance(candidate, nn.Module) else None

    @staticmethod
    def _extract_call_arguments(
        args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        positional = args[1:4] if args and isinstance(args[0], nn.Module) else args[:3]
        query = kwargs.get("query", positional[0] if len(positional) > 0 else None)
        key = kwargs.get("key", positional[1] if len(positional) > 1 else None)
        value = kwargs.get("value", positional[2] if len(positional) > 2 else None)
        if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
            raise UnsupportedSDPAObservationError("selected SDPA call did not provide tensor Q/K/V")
        # Transformers 4.57 V-JEPA supplies this as the fifth positional
        # argument, while other backends commonly use the keyword.
        mask_position = 4 if args and isinstance(args[0], nn.Module) else 3
        attention_mask = kwargs.get(
            "attention_mask", kwargs.get("attn_mask", args[mask_position] if len(args) > mask_position else None)
        )
        return query, key, value, attention_mask

    @staticmethod
    def _scalar(value: Any, name: str) -> float:
        if isinstance(value, Real):
            return float(value)
        if isinstance(value, torch.Tensor) and value.ndim == 0 and not value.requires_grad:
            return float(value.item())
        raise UnsupportedSDPAObservationError(f"selected SDPA call has unsupported {name}")

    def _validate_selected_call(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
        kwargs: Mapping[str, Any],
    ) -> tuple[float, float | None]:
        if module.training:
            raise UnsupportedSDPAObservationError("selected SDPA observation requires eval mode")
        if attention_mask is not None:
            raise UnsupportedSDPAObservationError("selected SDPA observation does not support attention masks")
        if kwargs.get("head_mask") is not None or kwargs.get("layer_head_mask") is not None:
            raise UnsupportedSDPAObservationError("selected SDPA observation does not support head masks")
        if kwargs.get("is_causal", False) is not False:
            raise UnsupportedSDPAObservationError("selected SDPA observation requires is_causal=False")
        if self._scalar(kwargs.get("dropout", 0.0), "dropout") != 0.0:
            raise UnsupportedSDPAObservationError("selected SDPA observation requires dropout=0")
        if kwargs.get("enable_gqa", False):
            raise UnsupportedSDPAObservationError("selected SDPA observation does not support GQA")
        if getattr(module, "num_key_value_groups", 1) != 1:
            raise UnsupportedSDPAObservationError("selected SDPA observation does not support grouped KV heads")
        if kwargs.get("position_bias") is not None:
            raise UnsupportedSDPAObservationError("selected SDPA observation does not support position bias")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise UnsupportedSDPAObservationError("selected SDPA observation requires rank-4 Q/K/V")
        if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
            raise UnsupportedSDPAObservationError("selected SDPA observation requires equal batch and head axes")
        if (
            key.shape[-2] != value.shape[-2]
            or query.shape[-1] != key.shape[-1]
            or query.shape[-1] != value.shape[-1]
        ):
            raise UnsupportedSDPAObservationError("selected SDPA observation has incompatible Q/K/V axes")

        native_scale_value = kwargs.get("scaling")
        native_scale = (
            None if native_scale_value is None else self._scalar(native_scale_value, "scaling")
        )
        effective_scale = native_scale if native_scale is not None else query.shape[-1] ** -0.5
        return effective_scale, native_scale

    def _reconstruct_capture(
        self,
        site: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        effective_scale: float,
        native_scale: float | None,
    ) -> SDPAQueryCapture:
        query_ids = self._uniform_query_ids(query.shape[-2], query.device)
        # ``Tensor.float()`` alone is insufficient under an enclosing
        # autocast context: matmul could immediately cast its float inputs
        # back down.  Disable autocast only for this read-only side path.
        with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
            sampled_query = query.index_select(-2, query_ids).float()
            float_key = key.float()
            scores = torch.matmul(sampled_query, float_key.transpose(-2, -1))
            scores.mul_(effective_scale)
            probabilities = torch.softmax(scores, dim=-1).detach()

        metadata: dict[str, Any] = {
            "source_kind": "reconstructed_from_native_post_rope_qk",
            "source_identity": dict(self._source_identity),
            "source_fingerprints": dict(self._source_identity),
            "native_query_dtype": self._dtype_name(query),
            "native_key_dtype": self._dtype_name(key),
            "native_value_dtype": self._dtype_name(value),
            "reconstruction_dtype": self._dtype_name(probabilities),
            "native_scaling": native_scale,
            "effective_scale": effective_scale,
            "attention_mask": None,
            "head_mask": None,
            "is_causal": False,
            "dropout": 0.0,
            "query_shape": tuple(query.shape),
            "key_shape": tuple(key.shape),
            "value_shape": tuple(value.shape),
            "sampled_score_shape": tuple(scores.shape),
            "query_count": query.shape[-2],
            "key_count": key.shape[-2],
            "sampled_query_count": query_ids.numel(),
        }
        return SDPAQueryCapture(site, probabilities, query_ids.detach().cpu(), metadata)

    def _uniform_query_ids(self, query_count: int, device: torch.device) -> torch.Tensor:
        count = min(query_count, self._max_queries)
        if count == 1:
            return torch.tensor([query_count // 2], device=device, dtype=torch.long)
        return torch.arange(count, device=device, dtype=torch.long) * (query_count - 1) // (count - 1)

    @staticmethod
    def _dtype_name(tensor: torch.Tensor) -> str:
        return str(tensor.dtype).removeprefix("torch.")
