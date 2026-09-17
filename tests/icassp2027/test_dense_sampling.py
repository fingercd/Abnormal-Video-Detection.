from __future__ import annotations

import numpy as np
import pytest

from vadbench.contracts import ClipBatch, validate_clip_for_capabilities
from vadbench.data.dense_sampling import (
    DenseSamplingPlan,
    aggregate_dense_scores,
    aggregate_interval_scores,
    sample_uniform_full_clips,
)
from vadbench.data.sampling import SamplingError
from vadbench.integrations.videomaev2 import DEFAULT_CAPABILITIES


def test_dense_plan_end_anchors_complete_tail_without_padding() -> None:
    plan = DenseSamplingPlan(clip_frames=3, frame_stride=2, window_stride=3)

    samples = plan.sample(10)

    assert [item.score_frame_start for item in samples] == [0, 3, 5]
    assert [item.score_frame_end for item in samples] == [5, 8, 10]
    assert samples[0].frame_indices == (0, 2, 4)
    assert samples[-1].frame_indices == (5, 7, 9)
    assert all(item.valid_mask == (True, True, True) for item in samples)
    assert [item.end_anchored for item in samples] == [False, False, True]

    aggregation = aggregate_dense_scores(samples, [0.1, 0.4, 0.7], num_frames=10)
    np.testing.assert_allclose(
        aggregation.scores,
        [0.1, 0.1, 0.1, 0.25, 0.25, 0.55, 0.55, 0.55, 0.7, 0.7],
    )
    np.testing.assert_array_equal(aggregation.contributors, [1, 1, 1, 2, 2, 2, 2, 2, 1, 1])


def test_dense_complete_windows_satisfy_existing_videomaev2_fixed_clip_contract() -> None:
    plan = DenseSamplingPlan(clip_frames=16, frame_stride=2, window_stride=4)

    samples = plan.sample(44)  # temporal span=31; 13 is the end-anchored start.

    assert samples[-1].frame_indices == tuple(range(13, 44, 2))
    assert samples[-1].end_anchored
    for item in samples:
        batch = ClipBatch(
            frames=np.zeros((1, 16, 1, 1, 3), dtype=np.uint8),
            timestamps_s=np.asarray([item.frame_indices], dtype=np.float64),
            frame_indices=np.asarray([item.frame_indices], dtype=np.int64),
            valid_mask=np.asarray([item.valid_mask], dtype=bool),
            video_ids=("video",),
        )
        validate_clip_for_capabilities(batch, DEFAULT_CAPABILITIES, train=False)


def test_dense_short_video_fails_instead_of_emitting_partial_fixed_clip() -> None:
    with pytest.raises(SamplingError, match="短视频"):
        DenseSamplingPlan(clip_frames=16, frame_stride=2).sample(30)


def test_encoder_specific_fixed_clips_share_the_same_scoring_time_coverage() -> None:
    # Active encoders may require different native fixed clip lengths.  Their
    # sampled input coordinates differ, but each declared dense protocol still
    # projects only a complete, gap-free [0, num_frames) source timeline.
    plans = (
        DenseSamplingPlan(clip_frames=16, frame_stride=2, window_stride=4),
        DenseSamplingPlan(clip_frames=8, frame_stride=2, window_stride=4),
        DenseSamplingPlan(clip_frames=64, frame_stride=1, window_stride=4),
    )
    for plan in plans:
        samples = plan.sample(200)
        assert samples[0].score_frame_start == 0
        assert samples[-1].score_frame_end == 200
        assert all(item.clip.valid_frames == plan.clip_frames for item in samples)
        aggregation = aggregate_dense_scores(samples, np.ones(len(samples)), num_frames=200)
        np.testing.assert_allclose(aggregation.scores, 1.0)


def test_uniform_full_clips_keep_32_nonoverlapping_score_segments_and_full_inputs() -> None:
    samples = sample_uniform_full_clips(3200, num_segments=32, clip_frames=16, frame_stride=2)

    assert len(samples) == 32
    assert [(item.score_frame_start, item.score_frame_end) for item in samples] == [
        (index * 100, (index + 1) * 100) for index in range(32)
    ]
    assert all(item.valid_mask == (True,) * 16 for item in samples)
    assert not any(item.input_window_reused for item in samples)

    for item in samples:
        batch = ClipBatch(
            frames=np.zeros((1, 16, 1, 1, 3), dtype=np.uint8),
            timestamps_s=np.asarray([item.frame_indices], dtype=np.float64),
            frame_indices=np.asarray([item.frame_indices], dtype=np.int64),
            valid_mask=np.asarray([item.valid_mask], dtype=bool),
            video_ids=("video",),
        )
        validate_clip_for_capabilities(batch, DEFAULT_CAPABILITIES, train=False)


def test_uniform_full_clips_clamp_cross_segment_inputs_and_record_reuse() -> None:
    samples = sample_uniform_full_clips(128, num_segments=32, clip_frames=64, frame_stride=1)

    assert [(item.score_frame_start, item.score_frame_end) for item in samples] == [
        (index * 4, (index + 1) * 4) for index in range(32)
    ]
    assert samples[0].requested_input_start == -30
    assert samples[0].input_start_frame == 0
    assert samples[0].input_end_frame == 64
    assert samples[0].frame_indices[-1] == 63
    assert samples[-1].input_start_frame == 64
    assert samples[-1].input_end_frame == 128
    assert any(item.input_window_reused for item in samples)
    # Inputs span logical segment boundaries by design; score intervals remain
    # the disjoint, complete segment partition used by MIL.
    assert samples[0].score_frame_end < samples[0].input_end_frame


def test_uniform_full_clips_rejects_video_shorter_than_native_temporal_span() -> None:
    with pytest.raises(SamplingError, match="短视频"):
        sample_uniform_full_clips(30, num_segments=32, clip_frames=16, frame_stride=2)
    with pytest.raises(SamplingError, match="无重叠"):
        sample_uniform_full_clips(20, num_segments=32, clip_frames=8, frame_stride=2)


def test_overlap_mean_and_max_are_explicit_and_complete() -> None:
    mean = aggregate_interval_scores([0, 1], [2, 4], [0.2, 0.8], num_frames=4)
    maximum = aggregate_interval_scores([0, 1], [2, 4], [0.2, 0.8], num_frames=4, reduction="max")

    np.testing.assert_allclose(mean.scores, [0.2, 0.5, 0.8, 0.8])
    np.testing.assert_array_equal(mean.contributors, [1, 2, 1, 1])
    np.testing.assert_allclose(maximum.scores, [0.2, 0.8, 0.8, 0.8])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_starts": [0], "frame_ends": [1], "scores": [0.2], "num_frames": 2}, "uncovered"),
        ({"frame_starts": [0], "frame_ends": [3], "scores": [0.2], "num_frames": 2}, "exceeds"),
        (
            {"frame_starts": [0], "frame_ends": [1], "scores": [float("nan")], "num_frames": 1},
            "NaN",
        ),
    ],
)
def test_dense_aggregation_rejects_gaps_invalid_intervals_and_nonfinite_scores(
    kwargs, message
) -> None:
    with pytest.raises(ValueError, match=message):
        aggregate_interval_scores(**kwargs)


def test_dense_plan_rejects_invalid_protocol() -> None:
    with pytest.raises(SamplingError, match="window_stride"):
        DenseSamplingPlan(clip_frames=4, frame_stride=1, window_stride=0)
    with pytest.raises(SamplingError, match="时间跨度"):
        DenseSamplingPlan(clip_frames=4, frame_stride=2, window_stride=8)
