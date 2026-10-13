"""Bounded, ordered CPU preparation for the unchanged native V2 forward."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
from pathlib import Path
from queue import Queue, Full
from threading import Event, Thread

import numpy as np

from vadbench.data.batches import clean_encoder_batch
from vadbench.data.video import OpenCVVideoReader, _batch_from_reader, VideoIOError


class ContinuousVideoReader(OpenCVVideoReader):
    """Keep the decoder cursor and 32-frame tail across overlapping groups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor = None
        self.tail = {}

    def read_indices(self, frame_indices):
        requested = tuple(int(i) for i in frame_indices)
        if not requested or min(requested) < 0 or max(requested) >= self.info.num_frames:
            raise VideoIOError("invalid frame indices in continuous reader")
        decoded = {}
        for index in sorted(set(requested)):
            if index in self.tail:
                decoded[index] = self.tail[index]
                continue
            if self.cursor is None or index < self.cursor or index - self.cursor > 2:
                frame = self._seek_and_read(self._capture, index)
            else:
                try:
                    while self.cursor < index:
                        self._read_rgb_frame(self._capture, self.cursor)
                        self.cursor += 1
                    frame = self._read_rgb_frame(self._capture, index)
                except VideoIOError:
                    frame = self._seek_and_read(self._capture, index)
            self.cursor = index + 1
            decoded[index] = frame
        self.tail = {i: decoded[i] for i in sorted(decoded)[-32:]}
        return np.stack([decoded[i] for i in requested])


class VideoMAEv2CPUPipeline:
    """One decoder, bounded preparation workers, native processor, ordered output.

    Only CPU work is done off-thread. Model/hooks/CUDA remain on the consumer.
    Frame-wise processor deduplication must pass a real-video equality gate
    before enabling this execution option for a frozen campaign.
    """

    def __init__(self, encoder, *, workers=2, prefetch=2, deduplicate=True):
        if workers < 1 or prefetch < 1:
            raise ValueError("workers and prefetch must be positive")
        self.encoder = encoder
        self.native_prepare = encoder._tensor_from_rgb_lists
        self.workers = workers
        self.prefetch = prefetch
        self.deduplicate = deduplicate

    @property
    def execution_identity(self):
        return {"kind": "v2_ordered_cpu_pipeline", "workers": self.workers,
                "prefetch": self.prefetch, "deduplicate_frames": self.deduplicate,
                "continuous_reader": True,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def prepare(self, batch):
        frames = batch.frames
        if not self.deduplicate:
            return self.native_prepare([list(clip) for clip in frames])
        # All clips belong to one video; repeated source indices are exact RGB
        # duplicates. Keep first-occurrence order, then restore native BCTHW.
        import torch
        indices = batch.frame_indices.reshape(-1)
        flat = frames.reshape((-1,) + frames.shape[2:])
        unique, first, inverse = np.unique(indices, return_index=True, return_inverse=True)
        count = len(unique)
        selected = [flat[i] for i in first]
        steps = self.encoder.cfg.num_frames
        selected.extend([selected[-1]] * ((-count) % steps))
        pixels = self.native_prepare([selected[i:i + steps] for i in range(0, len(selected), steps)])
        per_frame = pixels.permute(0, 2, 1, 3, 4).reshape(-1, *pixels.shape[1:2], *pixels.shape[3:])[:count]
        restored = per_frame[torch.as_tensor(inverse)].reshape(frames.shape[0], steps, *per_frame.shape[1:])
        return restored.permute(0, 2, 1, 3, 4)

    @contextmanager
    def prepared_context(self, pixels):
        encoder = self.encoder
        had_override = "_tensor_from_rgb_lists" in encoder.__dict__
        previous = encoder.__dict__.get("_tensor_from_rgb_lists")
        encoder._tensor_from_rgb_lists = lambda _clips: pixels
        try:
            yield
        finally:
            if had_override:
                encoder._tensor_from_rgb_lists = previous
            else:
                del encoder._tensor_from_rgb_lists

    @contextmanager
    def batches(self, reader, video_id, samples, micro_batch_size):
        queue = Queue(maxsize=self.prefetch)
        stop = Event()
        sentinel = object()

        def put(item):
            while not stop.is_set():
                try:
                    queue.put(item, timeout=.1)
                    return True
                except Full:
                    pass
            return False

        pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="v2-preprocess")
        # Limit raw-frame futures too: a bounded queue alone does not limit
        # executor submissions while consumer/model is slow.
        outstanding = deque()

        def produce():
            try:
                with ContinuousVideoReader(reader.info.path, backend=reader.backend) as source:
                    for start in range(0, len(samples), micro_batch_size):
                        if stop.is_set():
                            break
                        while len(outstanding) >= self.prefetch + 1:
                            outstanding.popleft().result()
                        group = samples[start:start + micro_batch_size]
                        batch = clean_encoder_batch(_batch_from_reader(source, video_id, [s.clip for s in group],
                            metadata={"source_num_frames": source.info.num_frames, "source_fps": source.info.fps}))
                        future = pool.submit(lambda b: (b, self.prepare(b)), batch)
                        outstanding.append(future)
                        if not put((group, future)):
                            break
            except BaseException as exc:
                put(exc)
            finally:
                put(sentinel)

        producer = Thread(target=produce, name="v2-decode", daemon=True)
        producer.start()

        def consume():
            while True:
                item = queue.get()
                if item is sentinel:
                    return
                if isinstance(item, BaseException):
                    raise item
                group, future = item
                batch, pixels = future.result()
                yield group, batch, self.prepared_context(pixels)

        try:
            yield consume()
        finally:
            stop.set()
            producer.join()
            pool.shutdown(wait=True, cancel_futures=True)
