"""Synthetic regression only; no real model/dataset or speed result."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numpy_dedup_candidate import candidate_prepare, NumpyPackedHFVideoCPUPipeline
from test_numpy_packing_candidate import SyntheticProcessor
from vadbench.workflows.hf_video_pipeline import HFVideoCPUPipeline


class CountingProcessor(SyntheticProcessor):
    def __init__(self):
        self.counts = []

    def __call__(self, videos, **kwargs):
        self.counts.append(sum(len(video) for video in videos))
        assert all(len(video) == 16 for video in videos)
        return super().__call__(videos, **kwargs)


def fixture(indices, lengths=None, ids=None):
    indices = np.asarray(indices)
    rng = np.random.default_rng(4)
    source = rng.integers(0, 256, size=(int(indices.max()) + 1, 5, 7, 3), dtype=np.uint8)
    batch = SimpleNamespace(frames=source[indices], frame_indices=indices,
        valid_lengths=lengths or [indices.shape[1]] * len(indices),
        video_ids=ids or ["same-video"] * len(indices))
    adapter = SimpleNamespace(processor=CountingProcessor(), processor_kwargs={"do_resize": True},
                              image_size=None, processor_tensor_type="pt", device="cpu")
    return adapter, batch


@pytest.mark.parametrize("indices,lengths", [
    ([list(range(16))], None),
    ([list(range(16))] * 8, None),
    ([list(range(i * 8, i * 8 + 16)) for i in range(8)], None),
    ([[0] * 16, list(range(16))], [3, 16]),
    ([list(range(16)), list(range(16, 32))], [8, 16]),
])
def test_exact_original_mapping_padding_dtype(indices, lengths):
    adapter, batch = fixture(indices, lengths)
    reference, native_lengths = HFVideoCPUPipeline(adapter).prepare(batch)
    actual, actual_lengths = candidate_prepare(adapter, batch)
    x, y = reference["pixel_values"], actual["pixel_values"]
    assert torch.equal(x, y)
    assert x.numpy().tobytes() == y.numpy().tobytes()
    assert x.shape == y.shape and x.dtype == y.dtype and x.stride() == y.stride()
    assert x.device == y.device == torch.device("cpu")
    assert np.array_equal(native_lengths, actual_lengths)
    assert adapter.processor_tensor_type == "pt"
    assert adapter.processor_kwargs == {"do_resize": True}


def test_eight_repeated_clips_process_sixteen_not_128_frames():
    adapter, batch = fixture([list(range(16))] * 8)
    candidate_prepare(adapter, batch)
    assert adapter.processor.counts == [16]


def test_cleaned_scheduler_ids_supported():
    adapter, batch = fixture([list(range(16))] * 2, ids=["sample-0", "sample-1"])
    candidate_prepare(adapter, batch)


def test_different_video_ids_rejected():
    adapter, batch = fixture([list(range(16))] * 2, ids=["video-a", "video-b"])
    with pytest.raises(ValueError, match="one source video"):
        candidate_prepare(adapter, batch)


def test_conflicting_rgb_same_index_rejected():
    adapter, batch = fixture([list(range(16))] * 2)
    batch.frames[1, 0, 0, 0, 0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="different RGB"):
        candidate_prepare(adapter, batch)


def test_identity_and_no_cross_call_cache():
    adapter, batch = fixture([list(range(16))] * 2)
    pipeline = NumpyPackedHFVideoCPUPipeline(adapter)
    pipeline.prepare(batch)
    pipeline.prepare(batch)
    assert adapter.processor.counts == [16, 16]
    identity = pipeline.execution_identity
    assert identity["deduplicate_frames"] is True
    assert identity["processed_frame_tail_cache"] is False
    assert len(identity["packing_implementation_sha256"]) == 64
