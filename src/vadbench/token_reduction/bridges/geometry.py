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
        # Native upstream names this Conv3d ``proj`` for VideoMAEv2/V-JEPA2
        # and ``projection`` for HF VideoMAE.  They have identical observable
        # semantics here; accepting only a real Conv3d keeps the geometry
        # fail-closed for other patch modules.
        self.projection = getattr(patch_embed, "proj", getattr(patch_embed, "projection", None))
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


class TimeSformerGeometry(AbstractContextManager):
    """Verify HF TimeSformer's actual divided-space-time token order.

    The public sequence is ``[CLS, spatial-major/time-minor patches]``.  This
    observer proves the claim against the first temporal sublayer input rather
    than deriving it from a nominal configuration alone.  It is intentionally
    observation-only and rejects a non-divided layout.
    """

    def __init__(
        self, embeddings: nn.Module, temporal_layernorm: nn.Module, frames: Any, valid: Any
    ) -> None:
        self.embeddings = embeddings
        self.temporal_layernorm = temporal_layernorm
        self.patch_embed = embeddings.patch_embeddings
        self.projection = self.patch_embed.projection
        if not isinstance(self.projection, nn.Conv2d):
            raise ValueError("TimeSformer geometry requires its actual Conv2d projection")
        self.frames = torch.as_tensor(frames, dtype=torch.int64, device="cpu")
        self.valid = torch.as_tensor(valid, dtype=torch.bool, device="cpu")
        if self.frames.ndim != 2 or self.frames.shape != self.valid.shape:
            raise ValueError("frame indices and validity must have identical [B,T] shape")
        self.layout: TokenLayout | None = None
        self.receipt: dict[str, Any] = {}
        self.handles: list[Any] = []
        self._projection_output: torch.Tensor | None = None
        self._embedding_output: torch.Tensor | None = None

    def __enter__(self) -> TimeSformerGeometry:
        if self.handles:
            raise RuntimeError("geometry observer is already active")
        self.layout = None
        self.receipt = {}
        self.handles.extend(
            (
                self.projection.register_forward_hook(self._capture_projection),
                self.patch_embed.register_forward_hook(self._capture_patch),
                self.embeddings.register_forward_hook(self._capture_embeddings),
                self.temporal_layernorm.register_forward_pre_hook(self._verify_temporal_input),
            )
        )
        return self

    def __exit__(self, *args: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._projection_output = None
        self._embedding_output = None

    def _capture_projection(
        self, _module: nn.Conv2d, inputs: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        batch_times_frames, _, height, width = inputs[0].shape
        batch, frames = self.frames.shape
        if batch_times_frames != batch * frames:
            raise ValueError(
                "TimeSformer patch projection batch does not match supplied source frames"
            )
        grid_h, grid_w = output.shape[-2:]
        if height % self.projection.stride[0] or width % self.projection.stride[1]:
            raise ValueError("TimeSformer input must form an exact patch grid")
        self._projection_output = output
        patch_count = grid_h * grid_w
        coordinates = torch.cartesian_prod(torch.arange(grid_h), torch.arange(grid_w))
        coordinates = coordinates.reshape(-1, 2)
        patch_coordinates = torch.empty((batch, patch_count * frames, 3), dtype=torch.int64)
        for spatial, (row, column) in enumerate(coordinates.tolist()):
            start = spatial * frames
            patch_coordinates[:, start : start + frames, 0] = torch.arange(frames)
            patch_coordinates[:, start : start + frames, 1] = row
            patch_coordinates[:, start : start + frames, 2] = column
        patch_valid = self.valid[:, None, :].expand(batch, patch_count, frames).reshape(batch, -1)
        valid = torch.cat((torch.ones((batch, 1), dtype=torch.bool), patch_valid), dim=1)
        ids = torch.arange(1 + patch_count * frames, dtype=torch.int64).expand(batch, -1).clone()
        ids[~valid] = -1
        full_coordinates = torch.cat(
            (torch.zeros((batch, 1, 3), dtype=torch.int64), patch_coordinates), dim=1
        )
        self.layout = TokenLayout(
            valid_mask=valid,
            original_token_ids=ids,
            special_token_mask=torch.cat(
                (torch.ones((batch, 1), dtype=torch.bool), torch.zeros_like(patch_valid)), dim=1
            ),
            mass=valid.float(),
            source_coordinates=full_coordinates,
            position_contract="hf-timesformer-absolute-spatial-and-temporal-position-before-divided-attention",
            provenance={
                "coordinate_source": "verified-timesformer-spatial-major-time-minor",
                "grid": [frames, grid_h, grid_w],
                "patch_stride": list(self.projection.stride),
            },
        )
        self.receipt = {
            "processed_shape": [batch, frames, int(inputs[0].shape[1]), height, width],
            "processed_layout": "BTCHW",
            "grid": [frames, grid_h, grid_w],
            "token_order": "CLS, spatial-major/time-minor (t fastest within each h,w)",
            "coordinate_scope": "processed crop/resize grid; no claim of original-image pixel localisation",
            "valid_token_counts": valid.sum(dim=1).tolist(),
            "patch_flatten_verified": False,
            "divided_layout_verified": False,
        }

    def _capture_patch(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        if self._projection_output is None or not isinstance(output, tuple) or not output:
            raise ValueError("TimeSformer patch embedding did not expose native tuple output")
        expected = self._projection_output.flatten(2).transpose(1, 2)
        if not torch.equal(output[0], expected):
            raise ValueError("TimeSformer patch output is not Conv2d flatten order")
        self.receipt["patch_flatten_verified"] = True
        self._projection_output = None

    def _capture_embeddings(
        self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise ValueError("TimeSformer embeddings output is not a tensor")
        self._embedding_output = output.detach()

    def _verify_temporal_input(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if self._embedding_output is None or self.layout is None:
            raise ValueError("TimeSformer embeddings were not observed before temporal attention")
        actual = inputs[0]
        batch, tokens, width = self._embedding_output.shape
        frames, grid_h, grid_w = self.receipt["grid"]
        patches = grid_h * grid_w
        if tokens != 1 + patches * frames:
            raise ValueError("TimeSformer embedding sequence does not match observed patch grid")
        expected = self._embedding_output[:, 1:].reshape(batch, grid_h, grid_w, frames, width)
        expected = expected.reshape(batch * patches, frames, width)
        if tuple(actual.shape) != tuple(expected.shape) or not torch.equal(actual, expected):
            raise ValueError(
                "TimeSformer temporal input is not verified spatial-major/time-minor layout"
            )
        self.receipt["divided_layout_verified"] = True
