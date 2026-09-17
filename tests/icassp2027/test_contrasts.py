from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from vadbench.research.contrasts import (
    ContrastConfig,
    ContrastError,
    analyze_contrasts,
    join_video_controls,
)


def row(
    video_id: str,
    clip_id: str,
    value: float | None,
    weak_label: int | None,
    *,
    category: str | None = None,
    scene: str | None = None,
    motion: str | None = None,
    brightness: str | None = None,
    duration: str | None = None,
    site: str = "block.3.output",
    statistic: str = "effective_rank",
) -> dict:
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "encoder_id": "fixture-encoder",
        "layer_index": 3,
        "site": site,
        "sublayer_kind": "output",
        "head_id": None,
        "probe_id": "P02",
        "statistic_name": statistic,
        "statistic_value": value,
        "status": "available" if value is not None else "unavailable",
        "weak_label": weak_label,
        "partition": "fit",
        "category_for_analysis_only": category,
        "scene_group": scene,
        "motion_bin": motion,
        "brightness_bin": brightness,
        "duration_bin": duration,
    }


def primary(analysis):
    return next(item for item in analysis.contrast_rows if item["contrast_id"] == "weak_video")


def test_clip_then_video_aggregation_does_not_weight_videos_by_clip_count():
    rows = [
        row("n0", "n0:0", 0, 0),
        row("n1", "n1:0", 2, 0),
        row("p0", "p0:0", 3, 1),
        row("p0", "p0:1", 5, 1),
        row("p1", "p1:0", 5, 1),
    ]
    analysis = analyze_contrasts(rows, config=ContrastConfig(200, 7, "explore"))
    values = {item["video_id"]: item for item in analysis.video_rows}
    assert values["p0"]["video_statistic_value"] == pytest.approx(4)
    assert values["p0"]["clip_count"] == 2
    # Video means are [0, 2] and [4, 5], so the difference is 3.5, not a clip-weighted value.
    assert primary(analysis)["mean_difference"] == pytest.approx(3.5)
    assert primary(analysis)["effect"] > 0
    assert primary(analysis)["ci_low"] > 0


def test_category_unknown_is_reported_and_matching_excludes_unknown_nuisance():
    rows = [
        row("n0", "n0:0", 0, 0, scene="s1", motion="low"),
        row("p0", "p0:0", 3, 1, category="fight", scene="s1", motion="low"),
        row("n1", "n1:0", 1, 0, scene="s2", motion="high"),
        row("p1", "p1:0", 4, 1, category=None, scene="s2", motion="high"),
        row("n2", "n2:0", 10, 0, scene=None, motion="high"),
        row("p2", "p2:0", -10, 1, category="fight", scene=None, motion="high"),
    ]
    analysis = analyze_contrasts(rows, config=ContrastConfig(200, 3, "explore"))
    categories = {
        item["stratum"]
        for item in analysis.contrast_rows
        if item["contrast_id"] == "category_stratified_weak_video"
    }
    assert categories == {"fight", "unknown"}
    matched = next(
        item for item in analysis.contrast_rows if item["contrast_id"] == "matched_control"
    )
    assert matched["status"] == "available"
    assert matched["num_matched_groups"] == 2
    assert matched["excluded_unknown_nuisance_videos"] == 2
    assert matched["effect"] == pytest.approx(3.0)


def test_missing_labels_are_unavailable_instead_of_inferred():
    analysis = analyze_contrasts(
        [row("unlabelled", "unlabelled:0", 2, None)],
        config=ContrastConfig(20, 1, "explore"),
    )
    result = primary(analysis)
    assert result["status"] == "unavailable"
    assert "弱视频标签" in result["reason"]


def test_confirm_only_evaluates_pre_frozen_exact_signature_and_direction():
    rows = [
        row("n0", "n0:0", 0, 0),
        row("n1", "n1:0", 1, 0),
        row("p0", "p0:0", 3, 1),
        row("p1", "p1:0", 4, 1),
        row("n0", "n0:1", 10, 0, statistic="rank90"),
        row("n1", "n1:1", 11, 0, statistic="rank90"),
        row("p0", "p0:1", 9, 1, statistic="rank90"),
        row("p1", "p1:1", 8, 1, statistic="rank90"),
    ]
    frozen = {
        "frozen_from_run_id": "explore-001",
        "candidates": [
            {
                "candidate_id": "rank",
                "encoder_id": "fixture-encoder",
                "layer_index": 3,
                "site": "block.3.output",
                "sublayer_kind": "output",
                "head_id": None,
                "probe_id": "P02",
                "statistic_name": "rank90",
                "expected_direction": "negative",
            }
        ],
    }
    analysis = analyze_contrasts(
        rows, config=ContrastConfig(100, 4, "confirm"), candidate_definition=frozen
    )
    assert analysis.receipt["analyzed_signatures"] == 1
    result = primary(analysis)
    assert result["statistic_name"] == "rank90"
    assert result["direction_matches_frozen"] is True
    with pytest.raises(ContrastError, match="candidate definition"):
        analyze_contrasts(rows, config=ContrastConfig(10, 1, "confirm"))


def test_rejects_more_than_two_candidates():
    candidate = {
        "encoder_id": "fixture-encoder",
        "layer_index": 3,
        "site": "block.3.output",
        "sublayer_kind": "output",
        "head_id": None,
        "probe_id": "P02",
        "statistic_name": "effective_rank",
        "expected_direction": "positive",
    }
    definition = {"candidates": [{**candidate, "candidate_id": f"c{i}"} for i in range(3)]}
    with pytest.raises(ContrastError, match="最多"):
        analyze_contrasts(
            [], config=ContrastConfig(10, 1, "confirm"), candidate_definition=definition
        )


def test_explicit_motion_brightness_matching_is_not_described_as_scene_control():
    rows = [
        row("n0", "n0:0", 0, 0, scene="normal-camera-a", motion="low", brightness="low"),
        row("p0", "p0:0", 3, 1, scene="positive-camera-x", motion="low", brightness="low"),
        row("n1", "n1:0", 1, 0, scene="normal-camera-b", motion="high", brightness="high"),
        row("p1", "p1:0", 4, 1, scene="positive-camera-y", motion="high", brightness="high"),
    ]
    analysis = analyze_contrasts(
        rows, config=ContrastConfig(100, 5, "explore", ("motion_bin", "brightness_bin"))
    )
    matched = next(
        item for item in analysis.contrast_rows if item["contrast_id"] == "matched_control"
    )
    assert matched["status"] == "available"
    assert matched["matching_fields"] == ["motion_bin", "brightness_bin"]
    assert "scene_group" not in matched["group_definition"]
    assert analysis.receipt["matching_fields"] == ["motion_bin", "brightness_bin"]
    assert analysis.receipt["matching_unknown_coverage"]["motion_bin"] == {
        "unknown_videos": 0,
        "total_videos": 4,
    }


def test_nearly_constant_norm_output_keeps_raw_distribution_and_caution_flag():
    rows = [
        row("n0", "n0:0", 1.0, 0, site="block.3.norm.output", statistic="activation_norm_median"),
        row(
            "n1",
            "n1:0",
            1.0000001,
            0,
            site="block.3.norm.output",
            statistic="activation_norm_median",
        ),
        row(
            "p0",
            "p0:0",
            1.0000002,
            1,
            site="block.3.norm.output",
            statistic="activation_norm_median",
        ),
        row(
            "p1",
            "p1:0",
            1.0000003,
            1,
            site="block.3.norm.output",
            statistic="activation_norm_median",
        ),
    ]
    result = primary(analyze_contrasts(rows, config=ContrastConfig(100, 6, "explore")))
    assert result["normalization_caution"] == "mechanically_near_constant_norm_output"
    assert result["positive_raw_std"] < 1e-5
    assert result["raw_mean_delta_positive_minus_normal"] == pytest.approx(2e-7)


def test_video_control_join_uses_encoder_video_identity_and_rejects_conflicting_bins():
    source = row("v0", "v0:0", 1, 0)
    control = {
        "encoder_id": "fixture-encoder",
        "video_id": "v0",
        "motion_bin": "low",
        "brightness_bin": "high",
        "input_sampling_id": "fixture-sampling",
    }
    joined = join_video_controls([source], [control])[0]
    assert joined["motion_bin"] == "low" and joined["brightness_bin"] == "high"
    assert joined["control_join_status"] == "applied_frozen_video_control"
    with pytest.raises(ContrastError, match="冲突"):
        join_video_controls([{**source, "motion_bin": "high"}], [control])


def test_cli_marks_fixture_export_synthetic_and_writes_machine_tables(tmp_path):
    input_path = tmp_path / "probe_summary.jsonl"
    rows = [
        row("n0", "n0:0", 0, 0, scene="s0", motion="low"),
        row("n1", "n1:0", 1, 0, scene="s1", motion="low"),
        row("p0", "p0:0", 3, 1, category="fixture", scene="s0", motion="low"),
        row("p1", "p1:0", 4, 1, category="fixture", scene="s1", motion="low"),
    ]
    input_path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "icassp2027" / "analyze_probes.py"
    main = runpy.run_path(str(script))["main"]
    output = tmp_path / "analysis"
    assert (
        main(
            [
                str(input_path),
                "--output",
                str(output),
                "--partition",
                "fit",
                "--bootstrap",
                "100",
                "--matching-field",
                "motion_bin",
                "--matching-field",
                "brightness_bin",
                "--data-status",
                "synthetic_test_only",
                "--no-plots",
            ]
        )
        == 0
    )
    receipt = json.loads((output / "analysis_receipt.json").read_text(encoding="utf-8"))
    assert receipt["data_status"] == "synthetic_test_only"
    assert receipt["output_is_research_finding"] is False
    assert receipt["matching_fields"] == ["motion_bin", "brightness_bin"]
    assert (output / "contrast_summary.csv").is_file()
    assert (output / "video_summary.csv").is_file()


def test_cli_fits_encoder_scoped_control_calibration_and_joins_video_bins(tmp_path):
    input_path = tmp_path / "probe_summary.jsonl"
    controls_path = tmp_path / "input_controls.jsonl"
    rows = []
    controls = []
    for index in range(3):
        for prefix, label, value, brightness, motion in (
            ("n", 0, float(index), 0.1 + 0.4 * index, 1.0 + 4 * index),
            ("p", 1, float(3 + index), 0.2 + 0.4 * index, 2.0 + 4 * index),
        ):
            video_id = f"{prefix}{index}"
            rows.append(row(video_id, f"{video_id}:0", value, label))
            controls.append(
                {
                    "encoder_id": "fixture-encoder",
                    "video_id": video_id,
                    "clip_id": f"{video_id}:0",
                    "partition": "fit",
                    "weak_label": label,
                    "input_sampling_id": "fixture-sampling-v1",
                    "brightness_mean": brightness,
                    "adjacent_frame_mad_per_s_mean": motion,
                }
            )
    input_path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")
    controls_path.write_text(
        "\n".join(json.dumps(item) for item in controls) + "\n", encoding="utf-8"
    )
    script = Path(__file__).resolve().parents[2] / "scripts" / "icassp2027" / "analyze_probes.py"
    main = runpy.run_path(str(script))["main"]
    output = tmp_path / "analysis-controls"
    assert (
        main(
            [
                str(input_path),
                "--output",
                str(output),
                "--partition",
                "fit",
                "--bootstrap",
                "100",
                "--controls",
                str(controls_path),
                "--matching-field",
                "motion_bin",
                "--matching-field",
                "brightness_bin",
                "--data-status",
                "synthetic_test_only",
                "--no-plots",
            ]
        )
        == 0
    )
    calibration = json.loads((output / "control_calibration.json").read_text(encoding="utf-8"))
    item = calibration["calibrations"][0]
    assert item["encoder_id"] == "fixture-encoder"
    assert item["input_sampling_id"] == "fixture-sampling-v1"
    assert len(item["control_input_sha256"]) == 64
    receipt = json.loads((output / "analysis_receipt.json").read_text(encoding="utf-8"))
    assert receipt["controls"]["status"] == "available"
