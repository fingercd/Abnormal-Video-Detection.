from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vadbench.contracts import ClipBatch
from vadbench.integrations.torchvision_video import (
    DEFAULT_CAPABILITIES,
    TorchvisionVideoAdapter,
)

torch = pytest.importorskip("torch")


def _batch(frames: int = 4, height: int = 8, width: int = 10) -> ClipBatch:
    pixels = np.arange(frames * height * width * 3, dtype=np.uint8).reshape(
        1, frames, height, width, 3
    )
    timestamps = (np.arange(frames, dtype=np.float64) / 2.0)[None, :]
    indices = np.arange(frames, dtype=np.int64)[None, :]
    return ClipBatch(
        frames=pixels,
        timestamps_s=timestamps,
        frame_indices=indices,
        video_ids=("video",),
    )


class _FakeBlock(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _FakeVideoModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = _FakeBlock()
        self.head = _FakeBlock()
        self.last_input = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value
        # [B,C,T,H,W] -> a dense [B,T,C] representation.
        dense = value.mean(dim=(-1, -2)).transpose(1, 2)
        dense = self.norm(dense)
        pooled = dense.mean(dim=1)
        return self.head(pooled)


@pytest.mark.parametrize(
    ("variant", "feature_stage", "image_size", "frames", "expected_tokens"),
    [
        ("r2plus1d_18", "pooled", 112, 4, 1),
        ("mvitv2", "pooled", 224, 4, 1),
        ("video_swin", "backbone_tokens", 224, 4, 4),
    ],
)
def test_variants_normalize_bthwc_and_emit_healthy_output(
    variant: str,
    feature_stage: str,
    image_size: int,
    frames: int,
    expected_tokens: int,
) -> None:
    model = _FakeVideoModel()
    adapter = TorchvisionVideoAdapter(
        variant=variant,
        model=model,
        image_size=image_size,
        clip_frames=frames,
        feature_stage=feature_stage,
    )
    output = adapter.encode(_batch(frames=frames))

    assert tuple(model.last_input.shape[:3]) == (1, 3, frames)
    assert tuple(output.features.shape) == (1, expected_tokens, 3)
    assert tuple(output.pooled.shape) == (1, 3)
    assert output.aux["adapter"] == "torchvision_video"
    assert output.aux["variant"] in {"r2plus1d_18", "mvit_v2_s", "swin3d_t"}
    assert output.timeline.num_tokens == expected_tokens
    assert np.isfinite(output.pooled.detach().cpu().numpy()).all()


def test_capabilities_match_generic_catalog_declaration() -> None:
    adapter = TorchvisionVideoAdapter(variant="r2plus1d_18", model=_FakeVideoModel())
    assert adapter.capabilities == DEFAULT_CAPABILITIES
    assert adapter.capabilities.supports_fixed_clip
    assert not adapter.capabilities.supports_streaming
    assert not adapter.capabilities.supports_kv_cache


def test_missing_checkpoint_and_remote_identifier_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="未提供本地 checkpoint"):
        TorchvisionVideoAdapter(variant="r2plus1d_18")
    with pytest.raises(FileNotFoundError, match="不存在"):
        TorchvisionVideoAdapter(
            variant="r2plus1d_18",
            checkpoint_path=tmp_path / "missing.pth",
        )
