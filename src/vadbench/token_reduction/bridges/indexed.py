"""Neutral, externally indexed intermediate-token intervention for bridge tests.

This is a bounded measurement aid, not a selector: callers supply every kept
native token index.  It temporarily hooks concrete block instances and never
replaces a model's ``forward`` method or its input preprocessing.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vadbench.token_reduction.contracts import TokenLayout

if TYPE_CHECKING:
    from . import EncoderBridge


class IndexedInterventionError(ValueError):
    """An external gather is structurally incompatible with the loaded bridge."""


def identity_indices(layout: TokenLayout) -> torch.Tensor:
    """Return the verified dense token order for an identity parity forward."""

    if not bool(layout.valid_mask.all()):
        raise IndexedInterventionError("identity gather requires an unpadded fixed-clip layout")
    return torch.arange(layout.token_capacity, device=layout.valid_mask.device).expand(
        layout.batch_size, -1
    )


@dataclass(slots=True)
class IndexedTokenIntervention(AbstractContextManager["IndexedTokenIntervention"]):
    """Gather explicit native token indices after one block and inspect its suffix.

    ``indices`` are source-token positions in the dense block output.  The
    operation has no score input and cannot choose tokens from activations.
    ``validate_execution`` must be called after a successful forward before
    reporting a shortened suffix.
    """

    bridge: EncoderBridge
    depth: int
    indices: torch.Tensor
    layout: TokenLayout
    transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    record_position_masks: bool = True
    _handles: list[Any] = field(default_factory=list, init=False, repr=False)
    gathered_shape: tuple[int, ...] | None = field(default=None, init=False)
    suffix_shapes: dict[int, tuple[int, ...]] = field(default_factory=dict, init=False)
    suffix_position_masks: dict[int, torch.Tensor] = field(default_factory=dict, init=False)
    position_injections: int = field(default=0, init=False)
    transform_ms: float = field(default=0.0, init=False)
    gather_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.bridge._validate_depth(self.depth)
        if self.depth >= self.bridge.receipt().block_count - 1:
            raise IndexedInterventionError(
                "indexed intervention needs at least one native suffix block"
            )
        if not isinstance(self.indices, torch.Tensor) or self.indices.ndim != 2:
            raise IndexedInterventionError("indices 必须为 [B,K] 整数 tensor")
        if self.indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise IndexedInterventionError("indices 必须为整数 tensor")
        if self.indices.shape[0] != self.layout.batch_size or self.indices.shape[1] == 0:
            raise IndexedInterventionError("indices 必须与 layout batch 一致且至少保留一个 token")
        if (self.indices < 0).any() or (self.indices >= self.layout.token_capacity).any():
            raise IndexedInterventionError("indices 超出已验证 dense token 范围")
        for row in self.indices:
            if row.numel() != torch.unique(row).numel():
                raise IndexedInterventionError("每个样本的 indices 不可重复")
        valid = self.layout.valid_mask.to(self.indices.device)
        if not bool(valid.gather(1, self.indices).all()):
            raise IndexedInterventionError("不得 gather padding token")
        if self.bridge.receipt().encoder_id == "timesformer":
            self._validate_timesformer_indices()
        if self.bridge.receipt().encoder_id == "vjepa2" and self.layout.special_token_mask.any():
            raise IndexedInterventionError("V-JEPA2 bridge expects a patch-only RoPE layout")

    def _validate_timesformer_indices(self) -> None:
        grid = self.layout.provenance.get("grid")
        if (
            not isinstance(grid, list)
            or len(grid) != 3
            or not all(type(value) is int and value > 0 for value in grid)
        ):
            raise IndexedInterventionError("TimeSformer requires verified [T,H,W] geometry")
        frames, _height, width = grid
        for row in self.indices.tolist():
            if not row or row[0] != 0:
                raise IndexedInterventionError(
                    "TimeSformer must preserve the actual CLS token at index 0"
                )
            patches = row[1:]
            if len(patches) % frames:
                raise IndexedInterventionError(
                    "TimeSformer must retain every time index for each selected spatial token"
                )
            spatial = []
            for start in range(0, len(patches), frames):
                group = patches[start : start + frames]
                base = group[0] - 1
                if (
                    base < 0
                    or base % frames
                    or group != [base + 1 + time for time in range(frames)]
                ):
                    raise IndexedInterventionError(
                        "TimeSformer indices must be CLS plus spatial-major complete time trajectories"
                    )
                spatial.append(base // frames)
            if spatial != sorted(spatial) or len(spatial) != len(set(spatial)):
                raise IndexedInterventionError(
                    "TimeSformer spatial trajectories must stay in native order"
                )
            if len(spatial) % width:
                raise IndexedInterventionError(
                    "TimeSformer retained spatial-token count must be divisible by native patch_width"
                )

    def __enter__(self) -> IndexedTokenIntervention:
        if self._handles:
            raise IndexedInterventionError("indexed intervention is already active")
        self.gathered_shape = None
        self.suffix_shapes.clear()
        self.suffix_position_masks.clear()
        self.position_injections = 0
        try:
            block = self.bridge._blocks[self.depth]
            self._handles.append(block.register_forward_hook(self._gather_after_block))
            for suffix_depth in range(self.depth + 1, self.bridge.receipt().block_count):
                suffix = self.bridge._blocks[suffix_depth]
                if self.bridge.receipt().encoder_id == "vjepa2":
                    self._handles.append(
                        suffix.register_forward_pre_hook(self._inject_vjepa_positions)
                    )
                self._handles.append(
                    suffix.register_forward_pre_hook(self._record_suffix_shape(suffix_depth))
                )
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        handles, self._handles = self._handles, []
        for handle in handles:
            handle.remove()

    def _gather_after_block(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        import time

        hidden = self.bridge.block_output_tensor(self.depth, output)
        if hidden.shape[:2] != (self.indices.shape[0], self.layout.token_capacity):
            raise IndexedInterventionError(
                "native block output does not match the dense layout supplied to indexed intervention"
            )
        indices = self.indices.to(hidden.device, dtype=torch.long)
        if self.transform is None:
            started = time.perf_counter()
            gathered = hidden.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            )
            self.gather_ms += (time.perf_counter() - started) * 1000.0
        else:
            started = time.perf_counter()
            gathered = self.transform(hidden, indices)
            self.transform_ms += (time.perf_counter() - started) * 1000.0
            if (
                not isinstance(gathered, torch.Tensor)
                or gathered.shape != (
                    hidden.shape[0], self.indices.shape[1], hidden.shape[-1]
                )
                or gathered.device != hidden.device
                or gathered.dtype != hidden.dtype
            ):
                raise IndexedInterventionError(
                    "indexed transform 必须返回同 device/dtype 的 [B,K,D] tensor"
                )
        self.gathered_shape = tuple(gathered.shape)
        return self.bridge.replace_block_output_tensor(
            self.depth, output, gathered, allow_sequence_shrink=True
        )

    def _inject_vjepa_positions(self, _module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs:
            raise IndexedInterventionError("V-JEPA2 suffix did not receive hidden states")
        positions = self.indices.to(inputs[0].device, dtype=torch.long)
        self.position_injections += 1
        if self.record_position_masks:
            self.suffix_position_masks[self.position_injections - 1] = positions.detach().cpu()
        return (inputs[0], positions, *inputs[2:])

    def _record_suffix_shape(self, depth: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise IndexedInterventionError(
                    "native suffix block did not receive tensor hidden states"
                )
            self.suffix_shapes[depth] = tuple(inputs[0].shape)
            if inputs[0].shape[1] != self.indices.shape[1]:
                raise IndexedInterventionError(
                    "native suffix did not receive the gathered token length"
                )
            return None

        return hook

    def validate_execution(self) -> dict[str, Any]:
        """Return a compact receipt only after every suffix used gathered tokens."""

        expected = set(range(self.depth + 1, self.bridge.receipt().block_count))
        if self.gathered_shape is None or set(self.suffix_shapes) != expected:
            raise IndexedInterventionError(
                "indexed gather or one or more native suffix blocks did not execute"
            )
        if self.bridge.receipt().encoder_id == "vjepa2" and self.position_injections != len(expected):
            raise IndexedInterventionError(
                "V-JEPA2 suffix did not receive original RoPE position indices"
            )
        return {
            "encoder_id": self.bridge.receipt().encoder_id,
            "intervention_depth": self.depth,
            "native_input_tokens": self.layout.token_capacity,
            "gathered_tokens": self.indices.shape[1],
            "gathered_shape": list(self.gathered_shape),
            "suffix_shapes": {
                str(depth): list(shape) for depth, shape in self.suffix_shapes.items()
            },
            "per_layer_token_counts": {
                str(depth): int(shape[1]) for depth, shape in self.suffix_shapes.items()
            },
            "plugin_overhead_ms": {
                "gather_ms": self.gather_ms,
                "transform_ms": self.transform_ms,
            },
            "vjepa2_original_rope_positions": self.bridge.receipt().encoder_id == "vjepa2",
            "vjepa2_position_injections": self.position_injections,
            "recorded_position_masks": self.record_position_masks,
        }


def indexed_gather(
    bridge: EncoderBridge,
    depth: int,
    indices: torch.Tensor,
    layout: TokenLayout,
    *,
    transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    record_position_masks: bool = True,
) -> IndexedTokenIntervention:
    """Create an explicit native-index gather, optionally transforming its block output."""

    return IndexedTokenIntervention(
        bridge,
        depth,
        indices,
        layout,
        transform=transform,
        record_position_masks=record_position_masks,
    )


__all__ = [
    "IndexedInterventionError",
    "IndexedTokenIntervention",
    "identity_indices",
    "indexed_gather",
]
