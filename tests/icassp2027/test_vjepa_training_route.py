from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from vadbench.contracts import ClipBatch
from vadbench.integrations.foundation.vjepa2 import VJEPA2Adapter, _VJEPA2Encoder


class _TinyVideoProcessor:
    """A local processor seam with V-JEPA's B,T,C,H,W tensor contract."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def __call__(self, videos: list[list[np.ndarray]], *, return_tensors: str) -> dict[str, torch.Tensor]:
        self.calls.append((len(videos), return_tensors))
        values = np.asarray(videos, dtype=np.float32) / 255.0
        return {"pixel_values_videos": torch.from_numpy(values).permute(0, 1, 4, 2, 3)}


def _batch() -> ClipBatch:
    frames = np.arange(2 * 16 * 16 * 3, dtype=np.uint8).reshape(1, 2, 16, 16, 3)
    return ClipBatch(
        frames=frames,
        timestamps_s=np.asarray([[0.0, 0.1]], dtype=np.float64),
        video_ids=("fixture",),
        frame_indices=np.asarray([[0, 1]], dtype=np.int64),
    )


def _tiny_vjepa_model() -> nn.Module:
    transformers = pytest.importorskip("transformers")
    config = transformers.VJEPA2Config(
        crop_size=16,
        frames_per_clip=2,
        tubelet_size=2,
        patch_size=8,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=2,
        mlp_ratio=2,
        drop_path_rate=0.2,
        pred_hidden_size=8,
        pred_num_attention_heads=2,
        pred_num_hidden_layers=1,
        num_pooler_layers=1,
    )
    return transformers.VJEPA2Model(config)


def test_vjepa_explicit_grad_route_keeps_default_no_grad_and_trains_only_internal_plugin() -> None:
    torch.manual_seed(7)
    model = _tiny_vjepa_model()
    processor = _TinyVideoProcessor()
    worker = _VJEPA2Encoder(model, processor, device=None)
    batch = _batch()
    grad_modes: list[bool] = []
    mode_handle = model.encoder.register_forward_hook(
        lambda _module, _inputs, _output: grad_modes.append(torch.is_grad_enabled())
    )
    try:
        model.eval()
        dense = worker.encode(batch)
        assert dense.requires_grad is False
        assert grad_modes == [False]

        adapter = VJEPA2Adapter(encoder=worker)
        model.train()
        no_plugin = adapter.encode_with_grad(batch)
        assert grad_modes[-1] is True
        assert not model.training
        assert not model.encoder.layer[0].training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        torch.testing.assert_close(no_plugin, dense)

        plugin = nn.Linear(8, 8, bias=False)
        before_backbone = [parameter.detach().clone() for parameter in model.parameters()]
        before_plugin = plugin.weight.detach().clone()

        def insert_plugin(_module: Any, _inputs: tuple[Any, ...], output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            return (plugin(output[0]), *output[1:])

        internal_handle = model.encoder.layer[0].register_forward_hook(insert_plugin)
        try:
            trained_features = adapter.encode_with_grad(batch)
            loss = trained_features.square().mean()
            loss.backward()
        finally:
            internal_handle.remove()

        assert trained_features.requires_grad
        assert plugin.weight.grad is not None
        assert plugin.weight.grad.abs().sum().item() > 0
        assert all(parameter.grad is None for parameter in model.parameters())

        optimizer = torch.optim.SGD(plugin.parameters(), lr=0.1)
        optimizer.step()
        assert not torch.equal(plugin.weight.detach(), before_plugin)
        assert all(
            torch.equal(parameter.detach(), before)
            for parameter, before in zip(model.parameters(), before_backbone, strict=True)
        )
        assert processor.calls == [(1, "pt"), (1, "pt"), (1, "pt")]
    finally:
        mode_handle.remove()
