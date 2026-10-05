from __future__ import annotations

import pytest

from vadbench.data.sampling import (
    SamplingError,
    sample_32_segments,
    sample_fixed_clip,
    sample_uniform_segment_clips,
    uniform_segments,
)


def test_uniform_segments_cover_video_without_gaps_for_long_video() -> None:
    segments = uniform_segments(320, 32)
    assert len(segments) == 32
    assert segments[0].start_frame == 0
    assert segments[-1].end_frame == 320
    assert all(segment.length == 10 for segment in segments)
    assert all(
        left.end_frame == right.start_frame
        for left, right in zip(segments, segments[1:], strict=False)
    )


def test_uniform_segments_keep_exact_count_for_short_video() -> None:
    segments = uniform_segments(2, 4)
    assert [(item.start_frame, item.end_frame) for item in segments] == [
        (0, 1),
        (0, 1),
        (1, 2),
        (1, 2),
    ]


def test_fixed_clip_center_sampling_respects_stride() -> None:
    sample = sample_fixed_clip(
        100,
        clip_frames=4,
        frame_stride=3,
        start_frame=10,
        end_frame=30,
        position="center",
    )
    assert sample.frame_indices == (15, 18, 21, 24)
    assert sample.valid_mask == (True, True, True, True)
    assert sample.valid_frame_indices == sample.frame_indices


def test_fixed_clip_repeats_last_frame_when_region_is_short() -> None:
    sample = sample_fixed_clip(
        10,
        clip_frames=6,
        frame_stride=2,
        start_frame=7,
        end_frame=10,
    )
    assert sample.frame_indices == (7, 9, 9, 9, 9, 9)
    assert sample.valid_mask == (True, True, False, False, False, False)
    assert sample.valid_frame_indices == (7, 9)


def test_32_segment_sampling_shape_and_bounds() -> None:
    samples = sample_32_segments(3200, clip_frames=16, frame_stride=2)
    assert len(samples) == 32
    assert all(sample.clip_frames == 16 for sample in samples)
    assert all(len(sample.valid_mask) == 16 for sample in samples)
    assert all(
        sample.segment_start_frame <= index < sample.segment_end_frame
        for sample in samples
        for index in sample.frame_indices
    )
    assert all(
        left < right
        for sample in samples
        for left, right in zip(
            sample.valid_frame_indices, sample.valid_frame_indices[1:], strict=False
        )
    )


def test_random_segment_sampling_is_seed_deterministic() -> None:
    first = sample_uniform_segment_clips(
        1000,
        num_segments=8,
        clip_frames=4,
        frame_stride=2,
        position="random",
        seed=42,
    )
    second = sample_uniform_segment_clips(
        1000,
        num_segments=8,
        clip_frames=4,
        frame_stride=2,
        position="random",
        seed=42,
    )
    other = sample_uniform_segment_clips(
        1000,
        num_segments=8,
        clip_frames=4,
        frame_stride=2,
        position="random",
        seed=43,
    )
    assert first == second
    assert first != other


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_frames": 0, "clip_frames": 4}, "num_frames"),
        ({"num_frames": 10, "clip_frames": 0}, "clip_frames"),
        ({"num_frames": 10, "clip_frames": 4, "frame_stride": 0}, "frame_stride"),
        (
            {"num_frames": 10, "clip_frames": 4, "start_frame": 5, "end_frame": 5},
            "采样区间",
        ),
    ],
)
def test_fixed_clip_rejects_invalid_parameters(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(SamplingError, match=message):
        sample_fixed_clip(**kwargs)
