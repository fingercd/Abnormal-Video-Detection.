from __future__ import annotations

import numpy as np
import pytest

from vadbench.contracts import ClipBatch
from vadbench.research.controls import (
    ControlError,
    aggregate_control_rows,
    apply_control_bins,
    calibrate_control_bins,
    raw_clip_controls,
)


def batch(frames, timestamps, valid_mask=None):
    frames = np.asarray(frames, dtype=np.uint8)
    return ClipBatch(
        frames=frames,
        timestamps_s=np.asarray(timestamps, dtype=float),
        valid_mask=None if valid_mask is None else np.asarray(valid_mask, dtype=bool),
        frame_indices=np.tile(np.arange(frames.shape[1]), (frames.shape[0], 1)),
        video_ids=tuple(f"v{index}" for index in range(frames.shape[0])),
    )


def test_raw_controls_use_only_valid_frames_and_actual_time_gap():
    frames = np.zeros((1, 3, 1, 1, 3), dtype=np.uint8)
    frames[0, 1] = 255
    frames[0, 2] = 10
    controls = raw_clip_controls(
        batch(frames, [[0.0, 0.5, 1.0]], [[True, True, False]]),
        clip_ids=("v0:0",),
        low_resolution=(1, 1),
    )
    record = controls[0]
    assert record["clip_id"] == "v0:0"
    assert record["brightness_mean"] == pytest.approx(0.5)
    assert record["brightness_std"] == pytest.approx(0.5)
    assert record["adjacent_frame_mad_mean"] == pytest.approx(1.0)
    assert record["adjacent_frame_mad_per_s_mean"] == pytest.approx(2.0)
    assert record["motion_status"] == "available"
    assert "not optical flow" in record["motion_detail"]
    assert record["valid_frame_count"] == 2 and record["padded_frame_count"] == 1
    assert record["source_duration_s"] == pytest.approx(0.5)


def test_raw_controls_report_zero_dt_without_inventing_a_rate():
    frames = np.zeros((1, 2, 1, 1, 3), dtype=np.uint8)
    frames[0, 1] = 10
    record = raw_clip_controls(batch(frames, [[1.0, 1.0]]), low_resolution=(1, 1))[0]
    assert record["adjacent_frame_mad_mean"] is not None
    assert record["adjacent_frame_mad_per_s_mean"] is None
    assert record["zero_dt_pair_count"] == 1
    assert record["motion_status"] == "unavailable"


def control(
    video_id,
    brightness,
    motion,
    partition="fit",
    weak_label=0,
    encoder_id="fixture-encoder",
    input_sampling_id="fixture-sampling-v1",
    **extra,
):
    return {
        "encoder_id": encoder_id,
        "video_id": video_id,
        "brightness_mean": brightness,
        "adjacent_frame_mad_per_s_mean": motion,
        "partition": partition,
        "weak_label": weak_label,
        "input_sampling_id": input_sampling_id,
        **extra,
    }


def test_calibration_uses_fit_normal_video_means_then_apply_only_uses_frozen_thresholds():
    rows = [
        control("n0", 0.1, 1.0),
        control("n0", 0.3, 3.0),
        control("n1", 0.5, 5.0),
        control("n2", 0.9, 9.0),
        control("abnormal-not-fit", 100.0, 100.0, weak_label=1),
        control("confirm-not-fit", 100.0, 100.0, partition="confirm"),
    ]
    video_rows = aggregate_control_rows(rows)
    calibration = calibrate_control_bins(
        video_rows,
        encoder_id="fixture-encoder",
        input_sampling_id="fixture-sampling-v1",
        control_input_sha256="a" * 64,
    )
    assert calibration.fit_normal_video_counts == {"brightness_bin": 3, "motion_bin": 3}
    assert calibration.fit_normal_video_ids == ("n0", "n1", "n2")
    applied = apply_control_bins(
        [control("new", 0.5, 5.0, partition="confirm", weak_label=1)], calibration
    )[0]
    assert applied["brightness_bin"] == "mid"
    assert applied["motion_bin"] == "mid"
    assert applied["control_bin_calibration"] == "control-bin-calibration-v1"


def test_apply_rejects_conflicting_existing_bin_and_calibration_needs_three_videos():
    rows = [control("n0", 0.1, 1.0), control("n1", 0.5, 5.0), control("n2", 0.9, 9.0)]
    calibration = calibrate_control_bins(
        aggregate_control_rows(rows),
        encoder_id="fixture-encoder",
        input_sampling_id="fixture-sampling-v1",
    )
    with pytest.raises(ControlError, match="冲突"):
        apply_control_bins([control("x", 0.5, 5.0, motion_bin="high")], calibration)
    with pytest.raises(ControlError, match="三个"):
        calibrate_control_bins(
            aggregate_control_rows(rows[:2]),
            encoder_id="fixture-encoder",
            input_sampling_id="fixture-sampling-v1",
        )


def test_aggregate_keeps_encoders_separate_and_rejects_mixed_sampling_for_one_encoder_video():
    rows = [
        control("same", 0.1, 1.0, encoder_id="encoder-a"),
        control("same", 0.3, 3.0, encoder_id="encoder-a"),
        control("same", 0.9, 9.0, encoder_id="encoder-b", input_sampling_id="other-sampling"),
    ]
    aggregated = aggregate_control_rows(rows)
    assert len(aggregated) == 2
    a = next(item for item in aggregated if item["encoder_id"] == "encoder-a")
    assert a["brightness_mean"] == pytest.approx(0.2)
    assert a["adjacent_frame_mad_per_s_mean"] == pytest.approx(2.0)
    with pytest.raises(ControlError, match="input_sampling_id 不一致"):
        aggregate_control_rows(
            [control("same", 0.1, 1.0), control("same", 0.2, 2.0, input_sampling_id="v2")]
        )
