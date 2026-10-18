"""Synthetic regression: native 8/16 frames and no-duplicate fast paths."""
import concurrent.futures
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numpy_dedup_candidate_r03 import candidate_prepare, NumpyPackedHFVideoCPUPipeline
from test_numpy_packing_candidate import SyntheticProcessor
from vadbench.workflows.hf_video_pipeline import HFVideoCPUPipeline


class CountingProcessor(SyntheticProcessor):
    def __init__(self):
        self.groups = []

    def __call__(self, videos, **kwargs):
        self.groups.append([len(video) for video in videos])
        return super().__call__(videos, **kwargs)


def make(indices, lengths=None, tensor_type="np", ids=None):
    indices = np.asarray(indices)
    source = np.random.default_rng(4).integers(0, 256,
        size=(int(indices.max()) + 1, 5, 7, 3), dtype=np.uint8)
    batch = SimpleNamespace(frames=source[indices], frame_indices=indices,
        valid_lengths=lengths or [indices.shape[1]] * len(indices),
        video_ids=ids or ["same-video"] * len(indices))
    a = SimpleNamespace(processor=CountingProcessor(), processor_kwargs={"do_resize": True},
                        image_size=None, processor_tensor_type=tensor_type, device="cpu")
    return a, batch


@pytest.mark.parametrize("steps", [8, 16])
@pytest.mark.parametrize("size", [1, 8])
@pytest.mark.parametrize("tensor_type", ["np", "pt"])
@pytest.mark.parametrize("repeated", [False, True])
def test_exact_native_shapes_no_extra_frame_padding(steps, size, tensor_type, repeated):
    indices = [list(range(steps)) if repeated else list(range(b * steps, (b + 1) * steps))
               for b in range(size)]
    a, batch = make(indices, tensor_type=tensor_type)
    reference, lengths = HFVideoCPUPipeline(a).prepare(batch)
    actual, actual_lengths = candidate_prepare(a, batch)
    x, y = reference["pixel_values"], actual["pixel_values"]
    assert x.numpy().tobytes() == y.numpy().tobytes()
    assert x.shape == y.shape and x.dtype == y.dtype and x.stride() == y.stride()
    assert x.device == y.device == torch.device("cpu")
    assert np.array_equal(lengths, actual_lengths)
    assert all(n == steps for n in a.processor.groups[-1])
    expected_groups = 1 if repeated else size
    assert len(a.processor.groups[-1]) == expected_groups
    assert a.processor_tensor_type == tensor_type
    assert a.processor_kwargs == {"do_resize": True}


@pytest.mark.parametrize("steps,lengths", [(8, [3, 8]), (16, [8, 16]), (8, [3, 5])])
def test_effective_lengths_and_padding(steps, lengths):
    a, batch = make([list(range(steps))] * 2, lengths=lengths)
    reference, native_lengths = HFVideoCPUPipeline(a).prepare(batch)
    actual, actual_lengths = candidate_prepare(a, batch)
    assert torch.equal(reference["pixel_values"], actual["pixel_values"])
    assert np.array_equal(native_lengths, actual_lengths)
    assert all(n == max(lengths) for n in a.processor.groups[-1])


def test_mixed_video_and_conflicting_rgb_still_rejected():
    a, batch = make([list(range(8))] * 2, ids=["video-a", "video-b"])
    with pytest.raises(ValueError, match="one source video"):
        candidate_prepare(a, batch)
    batch.video_ids = ["same-video"] * 2
    batch.frames[1, 0, 0, 0, 0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="different RGB"):
        candidate_prepare(a, batch)


def test_cleaned_scheduler_ids_two_workers_and_identity():
    a, batch = make([list(range(8))] * 2, ids=["sample-0", "sample-1"])
    pipeline = NumpyPackedHFVideoCPUPipeline(a)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(pipeline.prepare, [batch] * 4))
    assert all(torch.equal(results[0][0]["pixel_values"], r[0]["pixel_values"]) for r in results)
    identity = pipeline.execution_identity
    assert identity["native_clip_frames_policy"] == "dynamic_native_input_steps"
    assert identity["processed_frame_tail_cache"] is False
    assert identity["dedup_duplicate_rgb_exact_check"] is True
    assert len(identity["packing_implementation_sha256"]) == 64
