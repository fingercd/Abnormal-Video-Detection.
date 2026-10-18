"""Experimental batch-local frame dedup; native-HF exactness gate required.

Only inherited single-video scheduler batches are supported. No cross-batch or
cross-video cache exists. No running task/snapshot is modified by this module.
"""
from collections.abc import Mapping
import hashlib
from pathlib import Path

import numpy as np
import torch

from vadbench.integrations.transformers_video import _frame_lists, _invoke_processor
from vadbench.workflows.hf_video_pipeline import HFVideoCPUPipeline


class NumpyPackedHFVideoCPUPipeline(HFVideoCPUPipeline):
    @property
    def execution_identity(self):
        identity = dict(super().execution_identity)
        identity.update({
            "kind": "hf_video_batch_local_frame_dedup_candidate_r02",
            "packing_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_processor_return_tensors": "np",
            "tensor_conversion": "torch.from_numpy",
            "adapter_processor_tensor_type": self.adapter.processor_tensor_type,
            "deduplicate_frames": True,
            "dedup_scope": "single_video_single_prepare_call",
            "dedup_duplicate_rgb_exact_check": True,
            "processed_frame_tail_cache": False,
            "native_processor_unique_group_frames": 16,
        })
        return identity

    def prepare(self, batch):
        return candidate_prepare(self.adapter, batch)


def candidate_prepare(adapter, batch):
    videos, lengths = _frame_lists(batch)
    batch_size, steps = len(videos), len(videos[0])
    ids = tuple(batch.video_ids)
    same_video_ids = len(set(ids)) == 1
    # clean_encoder_batch intentionally erases filenames. Its canonical IDs are
    # accepted ONLY under the inherited one-reader/one-video scheduler contract.
    cleaned_scheduler_ids = ids == tuple(f"sample-{i}" for i in range(batch_size))
    if not same_video_ids and not cleaned_scheduler_ids:
        raise ValueError("frame dedup requires one source video per batch")
    if batch.frame_indices is None:
        raise ValueError("frame dedup requires source frame_indices")
    indices = np.asarray(batch.frame_indices)
    if indices.shape != (batch_size, batch.frames.shape[1]):
        raise ValueError("frame_indices do not match decoded batch")
    selected, by_index, inverse = [], {}, []
    for b, video in enumerate(videos):
        length = int(lengths[b])
        for t, frame in enumerate(video):
            # _frame_lists pads short clips with their last valid RGB frame.
            index = int(indices[b, min(t, length - 1)])
            if index < 0:
                raise ValueError("valid frame index must be nonnegative")
            position = by_index.get(index)
            if position is None:
                position = len(selected)
                by_index[index] = position
                selected.append(frame)
            elif not np.array_equal(selected[position], frame):
                raise ValueError("same frame index has different RGB; source identity conflict")
            inverse.append(position)
    count = len(selected)
    selected.extend([selected[-1]] * ((-count) % 16))
    grouped = [selected[i:i + 16] for i in range(0, len(selected), 16)]
    kwargs = dict(adapter.processor_kwargs)
    kwargs["return_tensors"] = "np"
    if adapter.image_size is not None and "size" not in kwargs:
        kwargs["size"] = {"height": adapter.image_size, "width": adapter.image_size}
    value = _invoke_processor(adapter.processor, grouped, kwargs)
    if not isinstance(value, Mapping) or set(value) != {"pixel_values"}:
        raise ValueError("dedup candidate supports only native pixel_values processor output")
    pixels = value["pixel_values"]
    if not isinstance(pixels, np.ndarray) or pixels.ndim != 5 or pixels.shape[:2] != (len(grouped), 16):
        raise ValueError("native processor must return [unique_groups,16,C,H,W] ndarray")
    per_frame = pixels.reshape(-1, *pixels.shape[2:])[:count]
    restored = per_frame[np.asarray(inverse, dtype=np.int64)].reshape(
        batch_size, steps, *pixels.shape[2:])
    return {"pixel_values": torch.from_numpy(restored)}, lengths
