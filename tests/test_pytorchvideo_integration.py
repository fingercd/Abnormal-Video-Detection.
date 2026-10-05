from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vadbench.contracts import ClipBatch
from vadbench.integrations.pytorchvideo import DEFAULT_CAPABILITIES, PytorchVideoAdapter

torch = pytest.importorskip("torch")


def _batch(batch_size: int = 2, frames: int = 4) -> ClipBatch:
    pixels = np.arange(batch_size * frames * 4 * 6 * 3, dtype=np.uint8).reshape(
        batch_size, frames, 4, 6, 3
    )
    timestamps = np.arange(frames, dtype=np.float64)[None, :].repeat(batch_size, axis=0)
    indices = np.arange(frames, dtype=np.int64)[None, :].repeat(batch_size, axis=0)
    return ClipBatch(
        frames=pixels,
        timestamps_s=timestamps,
        frame_indices=indices,
        video_ids=tuple(f"video-{i}" for i in range(batch_size)),
    )


class _Hook:
    def __init__(self, owner: _Block, callback: Any) -> None:
        self.owner = owner
        self.callback = callback

    def remove(self) -> None:
        if self.callback in self.owner.callbacks:
            self.owner.callbacks.remove(self.callback)


class _Block:
    def __init__(self) -> None:
        self.callbacks: list[Any] = []

    def register_forward_hook(self, callback: Any) -> _Hook:
        self.callbacks.append(callback)
        return _Hook(self, callback)

    def emit(self, output: Any) -> None:
        for callback in tuple(self.callbacks):
            callback(self, (), output)


class _FakeModel:
    def __init__(self, *, slowfast: bool = False) -> None:
        self.blocks = [_Block(), _Block(), _Block()]
        self.slowfast = slowfast
        self.last_input: Any = None
        self.training = False

    def eval(self) -> _FakeModel:
        self.training = False
        return self

    def train(self, value: bool = True) -> _FakeModel:
        self.training = bool(value)
        return self

    def __call__(self, value: Any) -> Any:
        self.last_input = value
        output_dtype = value[-1].dtype if isinstance(value, (list, tuple)) else value.dtype
        if self.slowfast:
            slow, fast = value
            batch = int(fast.shape[0])
            slow_feature = torch.ones((batch, 3, slow.shape[2], 2, 2), dtype=fast.dtype)
            fast_feature = torch.full((batch, 5, fast.shape[2], 2, 2), 2.0, dtype=fast.dtype)
            self.blocks[-2].emit([slow_feature, fast_feature])
        else:
            batch = int(value.shape[0])
            feature = torch.arange(batch * 7 * 2 * 2 * 2, dtype=value.dtype).reshape(
                batch, 7, 2, 2, 2
            )
            self.blocks[-2].emit(feature)
        # A classifier output is intentionally different from the observed
        # backbone representation; the adapter must prefer the hook.
        return torch.full((batch, 400), 99.0, dtype=output_dtype)


def test_fixed_adapter_converts_bthwc_to_normalized_bcthw_and_uses_backbone_hook() -> None:
    model = _FakeModel()
    adapter = PytorchVideoAdapter(
        variant="i3d",
        model=model,
        image_size=None,
        resize_size=None,
        crop_size=None,
        frame_stride=8,
    )
    output = adapter.encode(_batch())

    assert model.last_input.shape == (2, 3, 4, 4, 6)
    assert model.last_input.dtype == torch.float32
    assert output.features.shape == (2, 1, 7)
    assert output.pooled.shape == (2, 7)
    assert output.aux["variant"] == "i3d_r50"
    assert output.aux["feature_source"] == "hook:blocks[-2]"
    assert output.aux["input_layout"] == "BCTHW"
    assert output.aux["sampling_frame_stride"] == 8
    assert output.timeline.source_frame_start.shape == (2, 1)
    assert output.timeline.source_frame_end.tolist() == [[4], [4]]
    assert np.isfinite(output.pooled.detach().cpu().numpy()).all()


def test_slowfast_adapter_builds_two_pathways_and_concatenates_pooled_features() -> None:
    model = _FakeModel(slowfast=True)
    adapter = PytorchVideoAdapter(
        variant="slowfast_r50",
        model=model,
        image_size=None,
        slow_fast_alpha=2,
    )
    output = adapter.encode(_batch(batch_size=1, frames=5))

    slow, fast = model.last_input
    assert slow.shape == (1, 3, 3, 4, 6)
    assert fast.shape == (1, 3, 5, 4, 6)
    assert output.pooled.shape == (1, 8)
    assert output.aux["pathways"] == ["slow", "fast"]
    assert output.aux["slow_fast_alpha"] == 2
    assert output.aux["slow_pathway_shape"] == [1, 3, 3, 4, 6]


def test_variant_aliases_and_capabilities_are_fixed_clip_only() -> None:
    model = _FakeModel()
    adapter = PytorchVideoAdapter(variant="x3d-s", model=model, image_size=None)
    assert adapter.variant == "x3d_s"
    assert adapter.capabilities == DEFAULT_CAPABILITIES
    assert adapter.capabilities.supports_fixed_clip
    assert not adapter.capabilities.supports_streaming
    assert not adapter.capabilities.supports_kv_cache
    assert not adapter.capabilities.supports_token_cache


def test_local_checkpoint_is_loaded_without_online_pretrained_mode(tmp_path: Path) -> None:
    class CheckpointModel(_FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def state_dict(self) -> dict[str, Any]:
            return {"weight": self.weight.detach().clone()}

        def load_state_dict(self, state: Any, strict: bool = False) -> Any:
            self.weight.data.copy_(state["weight"])
            return type("Result", (), {"missing_keys": [], "unexpected_keys": []})()

    checkpoint = tmp_path / "model.pth"
    torch.save({"model_state": {"weight": torch.ones(2)}}, checkpoint)
    model = CheckpointModel()
    adapter = PytorchVideoAdapter(
        variant="x3d_s",
        model=model,
        checkpoint=checkpoint,
        image_size=None,
    )
    assert torch.equal(model.weight, torch.ones(2))
    assert adapter._checkpoint_report is not None
    assert adapter._checkpoint_report["state_keys"] == 1


def test_pretrained_true_and_missing_local_checkpoint_fail_closed() -> None:
    with pytest.raises(ValueError, match="本地 checkpoint"):
        PytorchVideoAdapter(variant="i3d_r50", pretrained=True, model=_FakeModel())
    with pytest.raises(ValueError, match="必须提供本地 checkpoint"):
        PytorchVideoAdapter(variant="i3d_r50")
