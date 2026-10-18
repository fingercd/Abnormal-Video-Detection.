"""Candidate execution-only packing; NOT installed in a formal snapshot.

Retains adapter constructor/state and native processor operations. Only the native
processor's final requested container is NumPy, then converted to CPU torch with
zero-copy from_numpy. Real native-processor exactness gates are REQUIRED before
any formal task can use this candidate.
"""
from collections.abc import Mapping
import hashlib
from pathlib import Path

import numpy as np
import torch

from vadbench.integrations.transformers_video import _frame_lists, _invoke_processor
from vadbench.workflows.hf_video_pipeline import HFVideoCPUPipeline


class NumpyPackedHFVideoCPUPipeline(HFVideoCPUPipeline):
    """Local pipeline alternative, not a global processor or adapter patch."""

    @property
    def execution_identity(self):
        identity = dict(super().execution_identity)
        identity.update({
            "kind": "hf_video_ordered_cpu_pipeline_numpy_packing_candidate_r01",
            "packing_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_processor_return_tensors": "np",
            "tensor_conversion": "torch.from_numpy",
            "adapter_processor_tensor_type": self.adapter.processor_tensor_type,
        })
        return identity

    def prepare(self, batch):
        return candidate_prepare(self.adapter, batch)


def candidate_prepare(adapter, batch):
    videos, lengths = _frame_lists(batch)
    kwargs = dict(adapter.processor_kwargs)
    kwargs["return_tensors"] = "np"
    if adapter.image_size is not None and "size" not in kwargs:
        kwargs["size"] = {"height": adapter.image_size, "width": adapter.image_size}
    value = _invoke_processor(adapter.processor, videos, kwargs)
    inputs = dict(value) if isinstance(value, Mapping) else {
        "pixel_values": value.pixel_values if hasattr(value, "pixel_values") else value
    }
    inputs = {key: torch.from_numpy(item) if isinstance(item, np.ndarray) else item
              for key, item in inputs.items()}
    if any(torch.is_tensor(item) and item.is_cuda for item in inputs.values()):
        raise ValueError("CPU worker emitted a CUDA tensor")
    return inputs, lengths
