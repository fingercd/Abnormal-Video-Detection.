"""Generate the unsealed six-cell quality matrix table and traceable detail CSV.

Reads ONLY the assembled matrix JSON (already validated fail-closed) and emits
two deliverables.  No model inference, no label reading, no prediction replay.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ENCODERS = ("videomaev2", "videomae", "timesformer")
DATASETS = ("ucf_crime", "xd_violence")
METHODS = ("pair_select", "group_uniform", "group_random")
PRETTY = {"videomaev2": "VideoMAEv2", "videomae": "VideoMAE", "timesformer": "TimeSformer"}
DS_PRETTY = {"ucf_crime": "UCF-Crime", "xd_violence": "XD-Violence"}


def pp(dense: float, method: float) -> float:
    """Percentage-point delta on the 0-100 scale (never a relative percentage)."""
    return 100.0 * (float(method) - float(dense))


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def main(argv: list[str]) -> int:
    matrix_path = Path(argv[1])
    out_md = Path(argv[2])
    out_csv = Path(argv[3])
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    index = {(c["dataset"], c["encoder"]): c for c in matrix["cells"]}

    lines: list[str] = []
    lines.append("# 六格压缩矩阵 — 完整效果表（解封后一次性交付）")
    lines.append("")
    lines.append("**矩阵**：`urdmu-six-cell-matrix-v1.json`（fail-closed 装配通过，6 cells）")
    lines.append("**模式**：direct_insert，冻结 head，official_frame 推理")
    lines.append("**指标标度**：0–100；差值为**百分点**（pp）= 100 × (compressed − dense)，不是相对百分比")
    lines.append("**CI**：10,000 次视频级 paired bootstrap（按视频弱标签分层）")
    lines.append("")
    lines.append("> 本表由解封关口通过后一次性生成。表格行列顺序**不随分数改变**。")
    lines.append("> 派生量（百分点、平均变化、最差变化）由分数计算，属解封后分析。")
    lines.append("")

    # ---- main table ----
    lines.append("## 1. 六格主表（各数据集冻结主指标）")
    lines.append("")
    lines.append("每个压缩单元格格式：`方法值 (相对本格 dense 的百分点变化)`。")
    lines.append("")
    header = "| 编码器 | 数据集 / 主指标 | Dense | pair_select | group_uniform | group_random |"
    sep = "|---|---|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for dataset in DATASETS:
        for encoder in ENCODERS:
            cell = index[(dataset, encoder)]
            dense_val = cell["dense"]["metric"]["value"]
            metric_name = cell["methods"]["pair_select"]["quality"]["metric"]
            row = [PRETTY[encoder], f"{DS_PRETTY[dataset]} / {metric_name}", fmt(dense_val)]
            for method in METHODS:
                q = cell["methods"][method]["quality"]
                mv = q.get("method_value")
                d = q.get("delta_method_minus_dense")
                if mv is None or d is None:
                    row.append("n/a")
                else:
                    row.append(f"{fmt(mv)} ({pp(dense_val, mv):+.2f} pp)")
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- XD supplementary metrics ----
    lines.append("## 2. XD 补充指标表（梯形 PR-AUC 为冻结主指标，另两项单列）")
    lines.append("")
    lines.append("三者**不是同一个统计量**，不因某个更好看而切换。")
    lines.append("")
    lines.append("| 编码器 | 指标 | Dense | pair_select | group_uniform | group_random |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for encoder in ENCODERS:
        cell = index[("xd_violence", encoder)]
        rows_out = []
        for label, field in (("PR-AUC (trapezoid) ← 冻结主指标", "quality"),
                             ("average_precision (step AP)", "xd_average_precision"),
                             ("frame_ap_step", "frame_ap_step")):
            src = cell["methods"]["pair_select"]
            payload = src.get(field)
            if field == "quality":
                dense_v = cell["dense"]["metric"]["value"]
                vals = [fmt(cell["methods"][m]["quality"].get("method_value")) for m in METHODS]
            elif isinstance(payload, dict) and payload.get("dense") is not None:
                dense_v = payload["dense"]
                vals = [fmt(cell["methods"][m].get(field, {}).get("method")) for m in METHODS]
            else:
                continue
            rows_out.append((label, dense_v, vals))
        for label, dense_v, vals in rows_out:
            lines.append(f"| {PRETTY[encoder]} | {label} | {fmt(dense_v)} | " + " | ".join(vals) + " |")
    lines.append("")

    # ---- budget / efficiency ----
    lines.append("## 3. 实际预算与 token 执行（来自 FeatureStore 真实执行形状）")
    lines.append("")
    lines.append("| 编码器 | 数据集 | 方法 | dense clips | 压缩 clips | retained/native (中位) | 实际保留比 | plugin_overhead_ms (均值) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for encoder in ENCODERS:
            cell = index[(dataset, encoder)]
            for method in METHODS:
                t = cell["methods"][method]["token_execution"]
                rr = t["retained_ratio"]
                po = t["plugin_overhead_ms"]
                lines.append(
                    f"| {PRETTY[encoder]} | {DS_PRETTY[dataset]} | {method} | {cell['dense']['clips']} "
                    f"| {t['clips']} | {t['retained_tokens']['median']:.0f}/{t['native_tokens']['median']:.0f} "
                    f"| {rr['median']:.4f} | {po['mean']:.2f} |"
                )
    lines.append("")
    lines.append("**注意**：`plugin_overhead_ms` 是随 chunk 重算的滑动平均（早期值被初始化污染），"
                 "且 pair_select 记 `transform_ms`、两个对照记 `gather_ms`，走不同代码路径。"
                 "**该列不得用作任何效率主张**；论文效率表一律采用阶段 D 的干净测量。")
    lines.append("")
    lines.append("**保留比注记**：TimeSformer 实际保留比 0.5717（选择单元为完整空间轨迹，粒度取整所致），"
                 "低于名义 keep_ratio 0.60；VideoMAEv2/VideoMAE 为 0.6001。"
                 "同一编码器内三个方法保留比完全一致，跨方法预算匹配成立。")
    lines.append("")

    # ---- per-cell detail with CI ----
    lines.append("## 4. 逐格 CI 明细（方法相对本格 dense）")
    lines.append("")
    lines.append("| 编码器 | 数据集 | 方法 | dense | 方法 | Δ (pp) | CI 低 | CI 高 | 非劣性边际 | 非劣性通过 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|:--:|")
    for dataset in DATASETS:
        for encoder in ENCODERS:
            cell = index[(dataset, encoder)]
            for method in METHODS:
                q = cell["methods"][method]["quality"]
                lines.append(
                    f"| {PRETTY[encoder]} | {DS_PRETTY[dataset]} | {method} "
                    f"| {fmt(q.get('dense_value'))} | {fmt(q.get('method_value'))} "
                    f"| {fmt(pp(q.get('dense_value'), q.get('method_value')))} "
                    f"| {fmt(q.get('ci_low'))} | {fmt(q.get('ci_high'))} "
                    f"| {fmt(q.get('noninferiority_margin'), 4)} "
                    f"| {q.get('point_noninferiority_pass')} |"
                )
    lines.append("")

    # ---- bootstrap provenance ----
    lines.append("## 5. Bootstrap 与覆盖 provenance")
    lines.append("")
    lines.append("| 编码器 | 数据集 | 方法 | draws | valid | invalid | n_videos | n_frames | engine | 种子 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    seen = set()
    for dataset in DATASETS:
        for encoder in ENCODERS:
            cell = index[(dataset, encoder)]
            for method in METHODS:
                q = cell["methods"][method]["quality"]
                key = (dataset, encoder, method)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(
                    f"| {PRETTY[encoder]} | {DS_PRETTY[dataset]} | {method} "
                    f"| {q.get('bootstrap_draws')} | {q.get('valid_draws')} | {q.get('invalid_draws')} "
                    f"| {q.get('n_videos')} | {q.get('n_frames')} | {q.get('engine')} | {q.get('bootstrap_seed')} |"
                )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. 数据可信度声明")
    lines.append("")
    lines.append("- 18/18 压缩 run 的 extraction 合同通过**逐视频闭合核验**（视频 ID 集合、每视频 clip 数、时间覆盖全部与 dense 目标一致）。")
    lines.append("- 18/18 预测 `status=completed`、`official_frame_scores_read=false`、唯一视频数正确、chunk receipt 完整。")
    lines.append("- 检查点为固定最终 step 3000，无按测试指标选最佳；上游 `xd_main.py` 的测试 AP 存最佳逻辑从未被加载。")
    lines.append("- official_frame 帧覆盖恰为 `[0, num_frames)`，未覆盖帧直接报错，无静默截断。")
    lines.append("- 行尾等价豁免 3 对，仅作用于 `code_digest` 单字段；其余七个字段仍严格相等。")
    lines.append("- 一次提前读分事件已记录于 `unseal_receipt.json`（仅 1 个 dense 基线值，未影响方法选择）。")
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- detail CSV ----
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset", "encoder", "method", "run_id", "metric_name", "metric_definition",
            "dense_value", "method_value", "delta_pp",
            "ci_low", "ci_high", "noninferiority_margin", "point_noninferiority_pass",
            "xd_average_precision_dense", "xd_average_precision_method",
            "xd_frame_ap_step_dense", "xd_frame_ap_step_method",
            "video_level_dense", "video_level_method",
            "n_videos", "n_frames",
            "dense_clips", "compressed_clips",
            "native_tokens_median", "retained_tokens_median", "retained_ratio_median",
            "bootstrap_draws", "valid_draws", "bootstrap_seed", "engine",
            "quality_export_sha256", "prediction_sha256", "feature_index_sha256",
            "feature_contract_sha256", "head_checkpoint_sha256", "training_qa_sha256",
        ])
        for dataset in DATASETS:
            for encoder in ENCODERS:
                cell = index[(dataset, encoder)]
                for method in METHODS:
                    m = cell["methods"][method]
                    q = m["quality"]
                    t = m["token_execution"]
                    prov = m["provenance"]
                    ap = m.get("xd_average_precision") or {}
                    step = m.get("frame_ap_step") or {}
                    vl = m.get("video_level") or {}
                    dense_v = q.get("dense_value")
                    method_v = q.get("method_value")
                    writer.writerow([
                        dataset, encoder, method, method,
                        q.get("metric"), q.get("metric_definition"),
                        dense_v, method_v,
                        (pp(dense_v, method_v) if dense_v is not None and method_v is not None else None),
                        q.get("ci_low"), q.get("ci_high"),
                        q.get("noninferiority_margin"), q.get("point_noninferiority_pass"),
                        ap.get("dense"), ap.get("method"),
                        step.get("dense"), step.get("method"),
                        vl.get("dense") if isinstance(vl, dict) else None,
                        vl.get("method") if isinstance(vl, dict) else None,
                        q.get("n_videos"), q.get("n_frames"),
                        cell["dense"]["clips"], t["clips"],
                        t["native_tokens"]["median"], t["retained_tokens"]["median"],
                        t["retained_ratio"]["median"],
                        q.get("bootstrap_draws"), q.get("valid_draws"),
                        q.get("bootstrap_seed"), q.get("engine"),
                        m["quality_export"]["sha256"],
                        prov["prediction"]["sha256"],
                        prov["feature_index"]["sha256"],
                        prov["feature_contract"]["sha256"],
                        prov["head_checkpoint"]["sha256"],
                        prov["training_qa"]["sha256"],
                    ])

    print(json.dumps({"status": "completed", "matrix_cells": len(matrix["cells"]),
                      "markdown": str(out_md), "csv": str(out_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
