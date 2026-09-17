from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vadbench.contracts import ClipBatch
from vadbench.integrations.transformers_video import (
    DEFAULT_CAPABILITIES,
    TransformersVideoAdapter,
)


def _batch(frames: int = 8, *, batch_size: int = 1) -> ClipBatch:
    pixels = np.arange(batch_size * frames * 4 * 5 * 3, dtype=np.uint8).reshape(
        batch_size, frames, 4, 5, 3
    )
    timestamps = np.arange(frames, dtype=np.float64)[None, :]
    timestamps = np.broadcast_to(timestamps, (batch_size, frames)).copy()
    indices = np.arange(frames, dtype=np.int64)[None, :]
    indices = np.broadcast_to(indices, (batch_size, frames)).copy()
    return ClipBatch(
        frames=pixels,
        timestamps_s=timestamps,
        video_ids=tuple(f"video-{idx}" for idx in range(batch_size)),
        frame_indices=indices,
    )


class _Processor:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, videos: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((videos, dict(kwargs)))
        # Keep the fake backend numpy-only; the real adapter does not require
        # torch merely to exercise the public BTHWC boundary.
        return {"pixel_values": np.asarray(videos, dtype=np.float32)}


class _Model:
    def __init__(self, *, frames: int = 8, tokens: int = 5, dim: int = 6) -> None:
        self.config = SimpleNamespace(num_frames=frames)
        self.tokens = tokens
        self.dim = dim
        self.calls: list[dict[str, Any]] = []
        self.eval_called = False

    def eval(self) -> _Model:
        self.eval_called = True
        return self

    def __call__(self, *, pixel_values: Any, return_dict: bool = True) -> Any:
        self.calls.append({"pixel_values": pixel_values, "return_dict": return_dict})
        batch = int(np.asarray(pixel_values).shape[0])
        hidden = np.arange(batch * self.tokens * self.dim, dtype=np.float32).reshape(
            batch, self.tokens, self.dim
        )
        return SimpleNamespace(last_hidden_state=hidden)


def test_adapter_normalizes_bthwc_and_last_hidden_state() -> None:
    processor = _Processor()
    model = _Model(frames=8, tokens=5, dim=6)
    adapter = TransformersVideoAdapter(
        variant="timesformer",
        model=model,
        processor=processor,
        clip_frames=8,
        image_size=224,
    )

    output = adapter.encode(_batch(8))

    assert adapter.capabilities == DEFAULT_CAPABILITIES
    assert output.features.shape == (1, 5, 6)
    assert output.pooled.shape == (1, 6)
    np.testing.assert_allclose(output.pooled, output.features.mean(axis=1))
    assert output.timeline.num_tokens == 5
    assert output.timeline.source_frame_start.shape == (1, 5)
    assert output.aux["adapter"] == "transformers_video"
    assert output.aux["variant"] == "timesformer"
    assert output.aux["feature_stage"] == "last_hidden_state"
    assert output.aux["preprocess_profile"] == "transformers-video-v1"
    assert processor.calls[0][0][0][0].dtype == np.uint8
    assert processor.calls[0][1]["return_tensors"] == "pt"
    assert processor.calls[0][1]["size"] == {"height": 224, "width": 224}
    assert model.calls[0]["return_dict"] is True


def test_videomae_variant_accepts_padded_batched_clip_and_pooling_stage() -> None:
    processor = _Processor()
    model = _Model(frames=4, tokens=3, dim=4)
    adapter = TransformersVideoAdapter(
        variant="video_mae",
        model=model,
        processor=processor,
        clip_frames=4,
        feature_stage="pooled",
        pooling="mean",
    )

    output = adapter.encode(_batch(4, batch_size=2))

    assert output.features.shape == (2, 1, 4)
    assert output.pooled.shape == (2, 4)
    assert output.aux["variant"] == "videomae"
    assert output.aux["requested_feature_stage"] == "pooled"
    assert output.aux["sequence_source"] == "mean_pool"


def test_local_loader_forces_local_files_only_and_selects_variant(tmp_path: Path) -> None:
    calls: dict[str, list[dict[str, Any]]] = {"model": [], "processor": []}

    class ModelClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> _Model:
            calls["model"].append({"path": path, **kwargs})
            return _Model(frames=16)

    class ProcessorClass:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> _Processor:
            calls["processor"].append({"path": path, **kwargs})
            return _Processor()

    module = SimpleNamespace(VideoMAEModel=ModelClass, AutoImageProcessor=ProcessorClass)
    adapter = TransformersVideoAdapter(
        variant="videomae",
        model_path=tmp_path,
        revision="a" * 40,
        transformers_module=module,
    )

    assert isinstance(adapter.model, _Model)
    assert calls["model"][0]["local_files_only"] is True
    assert calls["model"][0]["revision"] == "a" * 40
    assert calls["processor"][0]["local_files_only"] is True
    assert calls["processor"][0]["revision"] == "a" * 40


def test_missing_local_model_fails_without_importing_transformers(
    tmp_path: Path, monkeypatch
) -> None:
    imported: list[str] = []

    def fail_import(name: str) -> Any:
        imported.append(name)
        raise AssertionError("transformers must not be imported for a missing path")

    monkeypatch.setattr(
        "vadbench.integrations.transformers_video.importlib.import_module", fail_import
    )
    with pytest.raises(FileNotFoundError, match="本地权重不存在"):
        TransformersVideoAdapter(variant="timesformer", model_path=tmp_path / "missing")
    assert imported == []


def test_remote_download_switch_is_rejected() -> None:
    with pytest.raises(ValueError, match="local_files_only=True"):
        TransformersVideoAdapter(
            variant="timesformer",
            model=object(),
            processor=object(),
            local_files_only=False,
        )


def test_frame_contract_is_checked_before_processor_call() -> None:
    processor = _Processor()
    adapter = TransformersVideoAdapter(
        variant="timesformer",
        model=_Model(frames=8),
        processor=processor,
        clip_frames=8,
    )
    with pytest.raises(ValueError, match="要求 clip_frames=8"):
        adapter.encode(_batch(4))
    assert processor.calls == []


def test_real_videomae_processor_accepts_square_override_and_timesformer_224_8_shape():
    """Exercise the actual processor contract without loading checkpoint weights."""

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    small_processor = transformers.VideoMAEImageProcessor(
        size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}
    )
    small_model = transformers.VideoMAEModel(
        transformers.VideoMAEConfig(
            image_size=16,
            patch_size=8,
            num_frames=4,
            tubelet_size=2,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    small_adapter = TransformersVideoAdapter(
        variant="videomae",
        model=small_model,
        processor=small_processor,
        clip_frames=4,
        image_size=16,
    )
    small_inputs, _ = small_adapter._prepare_inputs(_batch(4))
    assert tuple(small_inputs["pixel_values"].shape) == (1, 4, 3, 16, 16)
    assert tuple(small_adapter.encode(_batch(4)).features.shape) == (1, 8, 8)

    # TimeSformer uses the same processor family.  Its native 224/8 geometry
    # must still be BTCHW and retain its CLS token: 1 + 8 * 14 * 14 = 1569.
    processor = transformers.VideoMAEImageProcessor(
        size={"shortest_edge": 224}, crop_size={"height": 224, "width": 224}
    )
    model = transformers.TimesformerModel(
        transformers.TimesformerConfig(
            image_size=224,
            patch_size=16,
            num_frames=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    adapter = TransformersVideoAdapter(
        variant="timesformer",
        model=model,
        processor=processor,
        clip_frames=8,
        image_size=224,
    )
    inputs, _ = adapter._prepare_inputs(_batch(8))
    assert isinstance(inputs["pixel_values"], torch.Tensor)
    assert tuple(inputs["pixel_values"].shape) == (1, 8, 3, 224, 224)
    output = adapter.encode(_batch(8))
    assert tuple(output.features.shape) == (1, 1569, 8)
    assert tuple(output.pooled.shape) == (1, 8)
