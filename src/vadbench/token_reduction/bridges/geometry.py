"""Runtime-checked tubelet provenance for the native VideoMAEv2 patch embed."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import nn

from vadbench.token_reduction.contracts import TokenLayout


class TubeletGeometry(AbstractContextManager):
    """Observe Conv3d and verify the exact flatten order; never change the output.

    Coordinates index the *processed* clip grid. ``source_frame_indices`` retains
    every original sampled frame in each tubelet, not an approximate timeline.
    Partial/padded tubelets are excluded from statistics, conservatively.
    """

    def __init__(self, patch_embed: nn.Module, frames: Any, valid: Any) -> None:
        self.patch_embed = patch_embed
        self.projection = patch_embed.proj
        if not isinstance(self.projection, nn.Conv3d):
            raise ValueError("tubelet geometry requires the actual Conv3d projection")
        if any(self.projection.padding) or any(d != 1 for d in self.projection.dilation):
            raise ValueError("padded/dilated patch projection is not supported by this bridge")
        self.frames = torch.as_tensor(frames, dtype=torch.int64, device="cpu")
        self.valid = torch.as_tensor(valid, dtype=torch.bool, device="cpu")
        if self.frames.ndim != 2 or self.frames.shape != self.valid.shape:
            raise ValueError("frame indices and validity must have identical [B,T] shape")
        self.layout: TokenLayout | None = None
        self.receipt: dict[str, Any] = {}
        self.handles: list[Any] = []
        self._projection_output: torch.Tensor | None = None

    def __enter__(self) -> TubeletGeometry:
        if self.handles:
            raise RuntimeError("geometry observer is already active")
        self.layout = None
        self.receipt = {}
        self.handles.append(self.projection.register_forward_hook(self._capture_projection))
        self.handles.append(self.patch_embed.register_forward_hook(self._capture_flatten))
        return self

    def __exit__(self, *args: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._projection_output = None

    def _capture_projection(
        self, module: nn.Conv3d, inputs: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        batch, _, frames, height, width = inputs[0].shape
        if (batch, frames) != tuple(self.frames.shape):
            raise ValueError("processed temporal shape differs from supplied source frames")
        grid_t, grid_h, grid_w = output.shape[2:]
        kernel = tuple(module.kernel_size)
        stride = tuple(module.stride)
        if any(size % step for size, step in zip((frames, height, width), stride, strict=True)):
            raise ValueError("input must form an exact tubelet grid")
        self._projection_output = output
        temporal_slots = (
            torch.arange(grid_t)[:, None] * stride[0] + torch.arange(kernel[0])[None, :]
        )
        source = self.frames[:, temporal_slots]
        tubelet_valid = self.valid[:, temporal_slots].all(dim=-1)
        coordinates = torch.cartesian_prod(
            torch.arange(grid_t), torch.arange(grid_h), torch.arange(grid_w)
        )
        coordinates = coordinates.reshape(-1, 3).unsqueeze(0).expand(batch, -1, -1).clone()
        valid = tubelet_valid.repeat_interleave(grid_h * grid_w, dim=1)
        ids = torch.arange(grid_t * grid_h * grid_w).unsqueeze(0).expand(batch, -1).clone()
        ids[~valid] = -1
        self.layout = TokenLayout(
            valid_mask=valid,
            original_token_ids=ids,
            special_token_mask=torch.zeros_like(valid),
            mass=valid.float(),
            source_coordinates=coordinates,
            position_contract="native-dense-position-order-preserved",
            provenance={
                "coordinate_source": "verified-conv3d-flatten-t-h-w",
                "grid": [grid_t, grid_h, grid_w],
                "kernel": list(kernel),
                "stride": list(stride),
            },
        )
        self.receipt = {
            "processed_shape": list(inputs[0].shape),
            "processed_layout": "BCTHW",
            "grid": [grid_t, grid_h, grid_w],
            "patch_size": list(kernel[1:]),
            "tubelet_size": kernel[0],
            "token_order": "t,h,w (w fastest)",
            "source_frame_indices": source.tolist(),
            "coordinate_scope": "processed crop/resize grid; no claim of original-image pixel localisation",
            "valid_token_counts": valid.sum(dim=1).tolist(),
            "flatten_verified": False,
        }

    def _capture_flatten(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        if self._projection_output is None:
            raise RuntimeError("patch projection was not observed")
        expected = self._projection_output.flatten(2).transpose(1, 2)
        if not torch.equal(output, expected):
            raise ValueError("patch output is not Conv3d output flattened in t,h,w order")
        self.receipt["flatten_verified"] = True
        self._projection_output = None
