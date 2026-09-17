"""Analyze an already label-joined probe_summary JSONL at video level.

The input is read-only.  Required analysis metadata is ``weak_label`` (0/1),
``partition``, ``scene_group``, ``motion_bin``, ``brightness_bin``,
``duration_bin`` and ``category_for_analysis_only`` in addition to the
collector's probe identity.  Missing control fields remain ``unknown`` and
are reported rather than inferred.
Missing labels result in unavailable rows, never inferred labels.  Use
``--phase confirm`` only with an exploration-after frozen candidate definition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from vadbench.research.contrasts import (
    MATCHING_FIELDS,
    ContrastConfig,
    analyze_contrasts,
    join_video_controls,
    render_candidate_cards,
    validate_candidate_definition,
)
from vadbench.research.controls import (
    ControlBinCalibration,
    aggregate_control_rows,
    apply_control_bins,
    calibrate_control_bins,
)


def _read_jsonl(paths: list[Path], partition: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} 不是合法 JSONL") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
                if partition is None or row.get("partition") == partition:
                    rows.append(row)
    return rows


def _read_one_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl([path], partition=None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sampling_ids(values: list[str] | None) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values or []:
        encoder_id, separator, sampling_id = value.partition("=")
        if not separator or not encoder_id or not sampling_id or encoder_id in output:
            raise ValueError(
                "--control-sampling-id 必须形如 encoder_id=input_sampling_id，且每个 encoder 一次"
            )
        output[encoder_id] = sampling_id
    return output


def _inject_sampling_ids(
    rows: list[dict[str, Any]], supplied: dict[str, str]
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        encoder_id = row.get("encoder_id")
        if not isinstance(encoder_id, str) or not encoder_id:
            raise ValueError("control sidecar 行必须含非空 encoder_id")
        expected = supplied.get(encoder_id)
        existing = row.get("input_sampling_id")
        if expected is not None and existing not in {None, "", expected}:
            raise ValueError("--control-sampling-id 与 control sidecar 的 input_sampling_id 冲突")
        if expected is not None:
            row["input_sampling_id"] = expected
        if not isinstance(row.get("input_sampling_id"), str) or not row["input_sampling_id"]:
            raise ValueError(
                "control sidecar 必须含 input_sampling_id，或为每个 encoder 提供 --control-sampling-id"
            )
        updated.append(row)
    return updated


def _load_calibration_bundle(path: Path) -> dict[str, ControlBinCalibration]:
    value = json.loads(path.read_text(encoding="utf-8"))
    items = value.get("calibrations") if isinstance(value, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("control calibration 文件必须含非空 calibrations 数组")
    calibrations = [ControlBinCalibration.from_mapping(item) for item in items]
    result = {item.encoder_id: item for item in calibrations}
    if len(result) != len(calibrations):
        raise ValueError("control calibration 文件的 encoder_id 重复")
    return result


def _disable_unfrozen_control_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["motion_bin"] = "unknown"
        row["brightness_bin"] = "unknown"
        row["control_join_status"] = "unavailable_without_frozen_control_calibration"
        result.append(row)
    return result


def _prepare_controls(
    probe_rows: list[dict[str, Any]], args
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    matching_fields = (
        tuple(args.matching_field) if args.matching_field else ("scene_group", "motion_bin")
    )
    needs_raw_controls = bool({"motion_bin", "brightness_bin"} & set(matching_fields))
    if not needs_raw_controls:
        return (
            probe_rows,
            {"status": "not_requested", "matching_fields": list(matching_fields)},
            None,
        )
    if args.controls is None:
        return (
            _disable_unfrozen_control_bins(probe_rows),
            {"status": "unavailable", "reason": "raw control sidecar was not provided"},
            None,
        )
    raw = _inject_sampling_ids(
        _read_one_jsonl(args.controls), _sampling_ids(args.control_sampling_id)
    )
    video_controls = aggregate_control_rows(raw)
    control_sha = _sha256(args.controls)
    if args.control_calibration is not None:
        calibrations = _load_calibration_bundle(args.control_calibration)
        source = "frozen_file"
        write_bundle = None
    elif args.phase == "explore" and args.partition == "fit":
        encoders = sorted({row["encoder_id"] for row in probe_rows})
        calibrations = {}
        for encoder_id in encoders:
            matching = [row for row in video_controls if row["encoder_id"] == encoder_id]
            if not matching:
                continue
            sampling_ids = {row["input_sampling_id"] for row in matching}
            if len(sampling_ids) != 1:
                raise ValueError("同一 encoder 的 controls 只能有一个 input_sampling_id")
            calibrations[encoder_id] = calibrate_control_bins(
                matching,
                encoder_id=encoder_id,
                input_sampling_id=sampling_ids.pop(),
                control_input_sha256=control_sha,
            )
        source = "fit_normal_fitted_current_input"
        write_bundle = {
            "schema": "control-calibration-bundle-v1",
            "control_input_sha256": control_sha,
            "calibrations": [item.as_dict() for item in calibrations.values()],
        }
    else:
        return (
            _disable_unfrozen_control_bins(probe_rows),
            {
                "status": "unavailable",
                "reason": "confirm/select 或非-fit 分区不能拟合 controls；需要 --control-calibration",
            },
            None,
        )
    applied: list[dict[str, Any]] = []
    missing_calibration_encoders: set[str] = set()
    for encoder_id in sorted({row["encoder_id"] for row in video_controls}):
        members = [row for row in video_controls if row["encoder_id"] == encoder_id]
        calibration = calibrations.get(encoder_id)
        if calibration is None:
            missing_calibration_encoders.add(encoder_id)
            continue
        applied.extend(apply_control_bins(members, calibration))
    joined = list(join_video_controls(probe_rows, applied))
    for row in joined:
        if row.get("control_join_status") == "missing_video_control":
            row["motion_bin"] = "unknown"
            row["brightness_bin"] = "unknown"
    return (
        joined,
        {
            "status": "available" if not missing_calibration_encoders else "partially_unavailable",
            "source": source,
            "controls_path": str(args.controls),
            "controls_sha256": control_sha,
            "video_control_rows": len(video_controls),
            "applied_video_control_rows": len(applied),
            "missing_calibration_encoders": sorted(missing_calibration_encoders),
        },
        write_bundle,
    )


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plots(output: Path, analysis, candidates, data_status: str) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "unavailable", "reason": "matplotlib is not installed; no plots created"}
    primary = [
        row
        for row in analysis.contrast_rows
        if row["contrast_id"] == "weak_video"
        and row["status"] == "available"
        and row.get("normalization_caution") != "mechanically_near_constant_norm_output"
    ]
    created: list[str] = []
    if primary:
        labels = [
            f"{row['encoder_id']} | L{row['layer_index']} | {row['probe_id']}:{row['statistic_name']}"
            for row in primary
        ]
        y = list(range(len(primary)))
        fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(primary) + 1.5)))
        effects = [row["effect"] for row in primary]
        left = [row["effect"] - row["ci_low"] for row in primary]
        right = [row["ci_high"] - row["effect"] for row in primary]
        ax.errorbar(effects, y, xerr=[left, right], fmt="o", color="#1f4e79", capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Hedges g (weak positive video − normal video)")
        ax.set_title(f"Probe contrasts with video-bootstrap 95% CI ({data_status})")
        fig.tight_layout()
        target = output / "weak_video_effects.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        created.append(target.name)
    candidate_signatures = [
        tuple(
            item[name]
            for name in (
                "encoder_id",
                "layer_index",
                "site",
                "sublayer_kind",
                "head_id",
                "probe_id",
                "statistic_name",
            )
        )
        for item in candidates
    ]
    values = [
        row
        for row in analysis.video_rows
        if tuple(
            row[name]
            for name in (
                "encoder_id",
                "layer_index",
                "site",
                "sublayer_kind",
                "head_id",
                "probe_id",
                "statistic_name",
            )
        )
        in candidate_signatures
        and row["weak_label"] in {0, 1}
    ]
    if values:
        fig, axes = plt.subplots(
            1, len(candidate_signatures), figsize=(5 * len(candidate_signatures), 4), squeeze=False
        )
        for axis, signature in zip(axes[0], candidate_signatures, strict=True):
            subset = [
                row
                for row in values
                if tuple(
                    row[name]
                    for name in (
                        "encoder_id",
                        "layer_index",
                        "site",
                        "sublayer_kind",
                        "head_id",
                        "probe_id",
                        "statistic_name",
                    )
                )
                == signature
            ]
            normal = [row["video_statistic_value"] for row in subset if row["weak_label"] == 0]
            positive = [row["video_statistic_value"] for row in subset if row["weak_label"] == 1]
            axis.boxplot([normal, positive], tick_labels=["V−", "V+"], showfliers=True)
            axis.set_title(f"{signature[0]} / {signature[5]}\n{signature[6]}")
            axis.set_ylabel("video-level probe statistic")
        fig.suptitle(f"Frozen candidate distributions ({data_status})")
        fig.tight_layout()
        target = output / "candidate_video_distributions.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        created.append(target.name)
    return {"status": "created", "files": created}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="label-joined probe_summary JSONL path(s)"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--partition", required=True, help="one frozen cohort partition; files are filtered to it"
    )
    parser.add_argument("--phase", choices=("explore", "confirm", "select"), default="explore")
    parser.add_argument(
        "--candidate-definition",
        type=Path,
        help="frozen JSON definition; required for confirm/select",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20270917)
    parser.add_argument(
        "--matching-field",
        action="append",
        choices=sorted(MATCHING_FIELDS),
        help="repeat to pre-specify exact matching fields; default is scene_group then motion_bin",
    )
    parser.add_argument("--controls", type=Path, help="raw input_controls.jsonl sidecar")
    parser.add_argument(
        "--control-calibration",
        type=Path,
        help="frozen control-calibration-bundle JSON; confirm/select require it for raw control matching",
    )
    parser.add_argument(
        "--control-sampling-id",
        action="append",
        help="repeat encoder_id=input_sampling_id when the raw sidecar does not carry the identity",
    )
    parser.add_argument(
        "--data-status",
        choices=("unverified", "real_observation", "synthetic_test_only"),
        default="unverified",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if any(not path.is_file() for path in args.inputs):
        parser.error("每个输入必须是存在的 probe_summary JSONL 文件")
    if args.controls is not None and not args.controls.is_file():
        parser.error("--controls 必须是存在的 input_controls.jsonl 文件")
    if args.control_calibration is not None and not args.control_calibration.is_file():
        parser.error("--control-calibration 必须是存在的冻结文件")
    definition = None
    if args.candidate_definition:
        try:
            definition = json.loads(args.candidate_definition.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"无法读取 candidate definition：{exc}")
    try:
        candidates = validate_candidate_definition(
            definition, require_direction=args.phase in {"confirm", "select"}
        )
        probe_rows, control_receipt, calibration_bundle = _prepare_controls(
            _read_jsonl(args.inputs, args.partition), args
        )
        analysis = analyze_contrasts(
            probe_rows,
            config=ContrastConfig(
                args.bootstrap,
                args.seed,
                args.phase,
                tuple(args.matching_field)
                if args.matching_field
                else ("scene_group", "motion_bin"),
            ),
            candidate_definition=definition,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "video_summary.csv", analysis.video_rows)
    _write_csv(args.output / "contrast_summary.csv", analysis.contrast_rows)
    _write_csv(args.output / "stratum_composition.csv", analysis.composition_rows)
    receipt = dict(analysis.receipt)
    receipt.update(
        input_paths=[str(path) for path in args.inputs],
        requested_partition=args.partition,
        data_status=args.data_status,
        output_is_research_finding=False,
        controls=control_receipt,
    )
    (args.output / "analysis_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if candidates:
        (args.output / "finding_card_drafts.md").write_text(
            render_candidate_cards(analysis, candidates), encoding="utf-8"
        )
    if calibration_bundle is not None:
        (args.output / "control_calibration.json").write_text(
            json.dumps(calibration_bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    plot_receipt = (
        {"status": "disabled"}
        if args.no_plots
        else _plots(args.output, analysis, candidates, args.data_status)
    )
    (args.output / "plot_receipt.json").write_text(
        json.dumps(plot_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "contrasts": len(analysis.contrast_rows),
                "plots": plot_receipt,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
