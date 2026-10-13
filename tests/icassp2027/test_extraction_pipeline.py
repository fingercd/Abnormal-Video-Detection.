from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.video import OpenCVVideoReader, _batch_from_reader
from vadbench.workflows.extraction_pipeline import ContinuousVideoReader, VideoMAEv2CPUPipeline


class Backend:
    CAP_PROP_FRAME_COUNT, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_POS_FRAMES = range(5)

    def __init__(self, n=200):
        self.frames = np.arange(n, dtype=np.uint8)[:, None, None, None] * np.ones((n, 2, 3, 3), dtype=np.uint8)
        self.seeks = []
        self.closed = 0

    def VideoCapture(self, path):
        backend = self
        class Capture:
            position = 0
            def isOpened(self): return True
            def get(self, key): return (len(backend.frames), 30, 3, 2)[key]
            def set(self, key, value):
                self.position = int(value); backend.seeks.append(self.position); return True
            def read(self):
                if self.position >= len(backend.frames): return False, None
                frame = backend.frames[self.position]; self.position += 1; return True, frame
            def release(self): backend.closed += 1
        return Capture()


class Encoder:
    cfg = SimpleNamespace(num_frames=16)
    def _tensor_from_rgb_lists(self, clips):
        return torch.tensor(np.asarray(clips).copy()).permute(0, 4, 1, 2, 3).float() / 255


def test_continuous_reader_matches_native_with_overlap_and_backward(tmp_path):
    path = tmp_path / 'video.mp4'; path.touch()
    baseline, fast = Backend(), Backend()
    groups = [tuple(range(0, 31, 2)), tuple(range(16, 47, 2)), tuple(range(32, 63, 2)), (5, 2, 5)]
    with OpenCVVideoReader(path, backend=baseline) as a, ContinuousVideoReader(path, backend=fast) as b:
        for group in groups:
            np.testing.assert_array_equal(a.read_indices(group), b.read_indices(group))
    assert len(fast.seeks) < len(baseline.seeks)


@pytest.mark.parametrize('workers', [1, 2, 4])
@pytest.mark.parametrize('deduplicate', [False, True])
def test_pipeline_exact_pixels_order_and_restore(tmp_path, workers, deduplicate):
    path = tmp_path / 'video.mp4'; path.touch()
    backend = Backend()
    encoder = Encoder()
    pipeline = VideoMAEv2CPUPipeline(encoder, workers=workers, prefetch=2, deduplicate=deduplicate)
    samples = DenseSamplingPlan(16, 2, 16).sample(200)
    seen = []
    with OpenCVVideoReader(path, backend=backend) as reader:
        with pipeline.batches(reader, 'v', samples, 3) as batches:
            for group, batch, context in batches:
                expected = pipeline.native_prepare([list(c) for c in batch.frames])
                with context:
                    actual = encoder._tensor_from_rgb_lists(None)
                    assert torch.equal(expected, actual)
                assert '_tensor_from_rgb_lists' not in encoder.__dict__
                seen.extend(s.clip_index for s in group)
    assert seen == list(range(len(samples)))
    assert backend.closed == 2
    assert not any(t.name == 'v2-decode' for t in threading.enumerate())


def test_pipeline_propagates_prepare_failure_and_stops(tmp_path):
    path = tmp_path / 'video.mp4'; path.touch()
    backend = Backend()
    pipeline = VideoMAEv2CPUPipeline(Encoder())
    def fail(batch): raise RuntimeError('prepare failed')
    pipeline.prepare = fail
    with OpenCVVideoReader(path, backend=backend) as reader:
        with pytest.raises(RuntimeError, match='prepare failed'):
            with pipeline.batches(reader, 'v', DenseSamplingPlan(16, 2, 16).sample(200), 2) as batches:
                list(batches)
    assert backend.closed == 2
    assert not any(t.name.startswith('v2-') for t in threading.enumerate())


def test_pipeline_closes_on_consumer_failure(tmp_path):
    path = tmp_path / 'video.mp4'; path.touch()
    backend = Backend()
    encoder = Encoder()
    pipeline = VideoMAEv2CPUPipeline(encoder)
    with OpenCVVideoReader(path, backend=backend) as reader:
        with pytest.raises(RuntimeError, match='consumer failed'):
            with pipeline.batches(reader, 'v', DenseSamplingPlan(16, 2, 16).sample(200), 2) as batches:
                for _, _, context in batches:
                    with context:
                        raise RuntimeError('consumer failed')
    assert '_tensor_from_rgb_lists' not in encoder.__dict__
    assert backend.closed == 2
    assert not any(t.name.startswith('v2-') for t in threading.enumerate())
