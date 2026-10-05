"""Batch transformations shared by observation and token-reduction workflows."""

from __future__ import annotations

from typing import Any

from vadbench.contracts import ClipBatch


def clean_encoder_batch(batch: Any) -> ClipBatch:
    """Preserve frame tensors while removing labels and identifying filenames."""

    return ClipBatch(
        frames=batch.frames,
        timestamps_s=batch.timestamps_s,
        video_ids=tuple(f"sample-{i}" for i in range(batch.batch_size)),
        valid_mask=batch.valid_mask,
        frame_indices=batch.frame_indices,
        metadata={
            key: batch.metadata[key]
            for key in ("source_num_frames", "source_fps")
            if key in batch.metadata
        },
    )
