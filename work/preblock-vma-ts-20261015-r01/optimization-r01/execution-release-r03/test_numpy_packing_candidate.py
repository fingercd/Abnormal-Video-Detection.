"""Synthetic unit tests only; no real dataset/model, no speed claims."""
import concurrent.futures
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numpy_packing_candidate import NumpyPackedHFVideoCPUPipeline
from vadbench.workflows.hf_video_pipeline import HFVideoCPUPipeline


class SyntheticProcessor:
    def __call__(self, videos, *, return_tensors, **kwargs):
        # The SAME artificial arithmetic and ordering precedes both packing paths.
        result = [[(frame.astype(np.float32).transpose(2, 0, 1) / np.float32(255)
                    - np.float32(.5)) / np.float32(.5)
                   for frame in video] for video in videos]
        if return_tensors == "pt":
            return {"pixel_values": torch.tensor(result)}
        assert return_tensors == "np"
        return {"pixel_values": np.asarray(result)}


def adapter():
    return SimpleNamespace(processor=SyntheticProcessor(), processor_kwargs={"do_resize": True},
                           image_size=None, processor_tensor_type="pt", device="cpu")


@pytest.mark.parametrize("size,lengths", [(1, [16]), (8, [16] * 8), (2, [8, 16])])
def test_synthetic_equal_shape_dtype_values_lengths(size, lengths):
    a = adapter()
    batch = SimpleNamespace(frames=np.random.default_rng(7).integers(
        0, 256, size=(size, 16, 5, 7, 3), dtype=np.uint8), valid_lengths=lengths)
    native, native_lengths = HFVideoCPUPipeline(a).prepare(batch)
    candidate, candidate_lengths = NumpyPackedHFVideoCPUPipeline(a).prepare(batch)
    x, y = native["pixel_values"], candidate["pixel_values"]
    assert torch.equal(x, y)
    assert x.numpy().tobytes() == y.numpy().tobytes()
    assert x.shape == y.shape and x.dtype == y.dtype == torch.float32
    assert x.device == y.device == torch.device("cpu")
    assert x.stride() == y.stride() and x.is_contiguous() and y.is_contiguous()
    assert np.array_equal(native_lengths, candidate_lengths)
    assert a.processor_tensor_type == "pt"
    assert a.processor_kwargs == {"do_resize": True}
    assert "_prepare_inputs" not in a.__dict__


def test_two_cpu_workers_no_adapter_or_global_mutation():
    a = adapter()
    pipeline = NumpyPackedHFVideoCPUPipeline(a)
    batch = SimpleNamespace(frames=np.zeros((1, 16, 5, 7, 3), dtype=np.uint8), valid_lengths=[16])
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        results = list(executor.map(pipeline.prepare, [batch] * 4))
    assert all(torch.equal(results[0][0]["pixel_values"], r[0]["pixel_values"]) for r in results)
    assert a.processor_tensor_type == "pt"
    assert a.processor_kwargs == {"do_resize": True}


def test_identity_distinguishes_candidate():
    a = adapter()
    old = HFVideoCPUPipeline(a).execution_identity
    new = NumpyPackedHFVideoCPUPipeline(a).execution_identity
    assert new["kind"] != old["kind"]
    assert new["implementation_sha256"] == old["implementation_sha256"]
    assert len(new["packing_implementation_sha256"]) == 64
    assert new["adapter_processor_tensor_type"] == "pt"
