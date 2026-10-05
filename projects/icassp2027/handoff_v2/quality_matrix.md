# 六格压缩矩阵 — 完整效果表（解封后一次性交付）

**矩阵**：`urdmu-six-cell-matrix-v1.json`（fail-closed 装配通过，6 cells）
**模式**：direct_insert，冻结 head，official_frame 推理
**指标标度**：0–100；差值为**百分点**（pp）= 100 × (compressed − dense)，不是相对百分比
**CI**：10,000 次视频级 paired bootstrap（按视频弱标签分层）

> 本表由解封关口通过后一次性生成。表格行列顺序**不随分数改变**。
> 派生量（百分点、平均变化、最差变化）由分数计算，属解封后分析。

## 1. 六格主表（各数据集冻结主指标）

每个压缩单元格格式：`方法值 (相对本格 dense 的百分点变化)`。

| 编码器 | 数据集 / 主指标 | Dense | pair_select | group_uniform | group_random |
|---|---|---:|---:|---:|---:|
| VideoMAEv2 | UCF-Crime / frame_roc_auc | 0.81 | 0.81 (+0.39 pp) | 0.81 (-0.11 pp) | 0.81 (+0.54 pp) |
| VideoMAE | UCF-Crime / frame_roc_auc | 0.81 | 0.80 (-0.16 pp) | 0.80 (-0.43 pp) | 0.80 (-0.09 pp) |
| TimeSformer | UCF-Crime / frame_roc_auc | 0.79 | 0.79 (-0.00 pp) | 0.78 (-0.33 pp) | 0.79 (+0.21 pp) |
| VideoMAEv2 | XD-Violence / frame_pr_auc | 0.76 | 0.77 (+0.32 pp) | 0.76 (-0.41 pp) | 0.76 (-0.43 pp) |
| VideoMAE | XD-Violence / frame_pr_auc | 0.70 | 0.71 (+0.61 pp) | 0.71 (+1.14 pp) | 0.71 (+1.09 pp) |
| TimeSformer | XD-Violence / frame_pr_auc | 0.77 | 0.77 (+0.68 pp) | 0.77 (+0.10 pp) | 0.77 (-0.11 pp) |

## 2. XD 补充指标表（梯形 PR-AUC 为冻结主指标，另两项单列）

三者**不是同一个统计量**，不因某个更好看而切换。

| 编码器 | 指标 | Dense | pair_select | group_uniform | group_random |
|---|---|---:|---:|---:|---:|
| VideoMAEv2 | PR-AUC (trapezoid) ← 冻结主指标 | 0.76 | 0.77 | 0.76 | 0.76 |
| VideoMAEv2 | average_precision (step AP) | 0.70 | 0.71 | 0.70 | 0.70 |
| VideoMAE | PR-AUC (trapezoid) ← 冻结主指标 | 0.70 | 0.71 | 0.71 | 0.71 |
| VideoMAE | average_precision (step AP) | 0.65 | 0.65 | 0.65 | 0.64 |
| TimeSformer | PR-AUC (trapezoid) ← 冻结主指标 | 0.77 | 0.77 | 0.77 | 0.77 |
| TimeSformer | average_precision (step AP) | 0.71 | 0.72 | 0.71 | 0.71 |

## 3. 实际预算与 token 执行（来自 FeatureStore 真实执行形状）

| 编码器 | 数据集 | 方法 | dense clips | 压缩 clips | retained/native (中位) | 实际保留比 | plugin_overhead_ms (均值) |
|---|---|---|---:|---:|---:|---:|---:|
| VideoMAEv2 | UCF-Crime | pair_select | 69364 | 69364 | 941/1568 | 0.6001 | 744081.98 |
| VideoMAEv2 | UCF-Crime | group_uniform | 69364 | 69364 | 941/1568 | 0.6001 | 181.51 |
| VideoMAEv2 | UCF-Crime | group_random | 69364 | 69364 | 941/1568 | 0.6001 | 187.05 |
| VideoMAE | UCF-Crime | pair_select | 69364 | 69364 | 941/1568 | 0.6001 | 1132458.89 |
| VideoMAE | UCF-Crime | group_uniform | 69364 | 69364 | 941/1568 | 0.6001 | 205.53 |
| VideoMAE | UCF-Crime | group_random | 69364 | 69364 | 941/1568 | 0.6001 | 200.73 |
| TimeSformer | UCF-Crime | pair_select | 138853 | 138853 | 897/1569 | 0.5717 | 1245902.20 |
| TimeSformer | UCF-Crime | group_uniform | 138853 | 138853 | 897/1569 | 0.5717 | 377.76 |
| TimeSformer | UCF-Crime | group_random | 138853 | 138853 | 897/1569 | 0.5717 | 405.80 |
| VideoMAEv2 | XD-Violence | pair_select | 145703 | 145703 | 941/1568 | 0.6001 | 1893623.40 |
| VideoMAEv2 | XD-Violence | group_uniform | 145703 | 145703 | 941/1568 | 0.6001 | 420.88 |
| VideoMAEv2 | XD-Violence | group_random | 145703 | 145703 | 941/1568 | 0.6001 | 424.02 |
| VideoMAE | XD-Violence | pair_select | 145703 | 145703 | 941/1568 | 0.6001 | 2143994.60 |
| VideoMAE | XD-Violence | group_uniform | 145703 | 145703 | 941/1568 | 0.6001 | 459.03 |
| VideoMAE | XD-Violence | group_random | 145703 | 145703 | 941/1568 | 0.6001 | 454.69 |
| TimeSformer | XD-Violence | pair_select | 291781 | 291781 | 897/1569 | 0.5717 | 2324600.87 |
| TimeSformer | XD-Violence | group_uniform | 291781 | 291781 | 897/1569 | 0.5717 | 837.67 |
| TimeSformer | XD-Violence | group_random | 291781 | 291781 | 897/1569 | 0.5717 | 895.48 |

**注意**：`plugin_overhead_ms` 是随 chunk 重算的滑动平均（早期值被初始化污染），且 pair_select 记 `transform_ms`、两个对照记 `gather_ms`，走不同代码路径。**该列不得用作任何效率主张**；论文效率表一律采用阶段 D 的干净测量。

**保留比注记**：TimeSformer 实际保留比 0.5717（选择单元为完整空间轨迹，粒度取整所致），低于名义 keep_ratio 0.60；VideoMAEv2/VideoMAE 为 0.6001。同一编码器内三个方法保留比完全一致，跨方法预算匹配成立。

## 4. 逐格 CI 明细（方法相对本格 dense）

| 编码器 | 数据集 | 方法 | dense | 方法 | Δ (pp) | CI 低 | CI 高 | 非劣性边际 | 非劣性通过 |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| VideoMAEv2 | UCF-Crime | pair_select | 0.81 | 0.81 | 0.39 | -0.01 | 0.02 | 0.0050 | True |
| VideoMAEv2 | UCF-Crime | group_uniform | 0.81 | 0.81 | -0.11 | -0.01 | 0.01 | 0.0050 | True |
| VideoMAEv2 | UCF-Crime | group_random | 0.81 | 0.81 | 0.54 | -0.01 | 0.02 | 0.0050 | True |
| VideoMAE | UCF-Crime | pair_select | 0.81 | 0.80 | -0.16 | -0.01 | 0.01 | 0.0050 | True |
| VideoMAE | UCF-Crime | group_uniform | 0.81 | 0.80 | -0.43 | -0.01 | 0.00 | 0.0050 | True |
| VideoMAE | UCF-Crime | group_random | 0.81 | 0.80 | -0.09 | -0.01 | 0.00 | 0.0050 | True |
| TimeSformer | UCF-Crime | pair_select | 0.79 | 0.79 | -0.00 | -0.01 | 0.01 | 0.0050 | True |
| TimeSformer | UCF-Crime | group_uniform | 0.79 | 0.78 | -0.33 | -0.02 | 0.01 | 0.0050 | True |
| TimeSformer | UCF-Crime | group_random | 0.79 | 0.79 | 0.21 | -0.01 | 0.02 | 0.0050 | True |
| VideoMAEv2 | XD-Violence | pair_select | 0.76 | 0.77 | 0.32 | -0.00 | 0.01 | 0.0050 | True |
| VideoMAEv2 | XD-Violence | group_uniform | 0.76 | 0.76 | -0.41 | -0.01 | -0.00 | 0.0050 | True |
| VideoMAEv2 | XD-Violence | group_random | 0.76 | 0.76 | -0.43 | -0.01 | -0.00 | 0.0050 | True |
| VideoMAE | XD-Violence | pair_select | 0.70 | 0.71 | 0.61 | 0.00 | 0.01 | 0.0050 | True |
| VideoMAE | XD-Violence | group_uniform | 0.70 | 0.71 | 1.14 | 0.01 | 0.02 | 0.0050 | True |
| VideoMAE | XD-Violence | group_random | 0.70 | 0.71 | 1.09 | 0.00 | 0.02 | 0.0050 | True |
| TimeSformer | XD-Violence | pair_select | 0.77 | 0.77 | 0.68 | 0.00 | 0.01 | 0.0050 | True |
| TimeSformer | XD-Violence | group_uniform | 0.77 | 0.77 | 0.10 | -0.00 | 0.00 | 0.0050 | True |
| TimeSformer | XD-Violence | group_random | 0.77 | 0.77 | -0.11 | -0.00 | 0.00 | 0.0050 | True |

## 5. Bootstrap 与覆盖 provenance

| 编码器 | 数据集 | 方法 | draws | valid | invalid | n_videos | n_frames | engine | 种子 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| VideoMAEv2 | UCF-Crime | pair_select | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| VideoMAEv2 | UCF-Crime | group_uniform | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| VideoMAEv2 | UCF-Crime | group_random | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| VideoMAE | UCF-Crime | pair_select | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| VideoMAE | UCF-Crime | group_uniform | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| VideoMAE | UCF-Crime | group_random | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| TimeSformer | UCF-Crime | pair_select | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| TimeSformer | UCF-Crime | group_uniform | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| TimeSformer | UCF-Crime | group_random | 10000 | 10000 | 0 | 290 | 1111808 | exact_video_pair_matrix_quadratic_form | 20260918 |
| VideoMAEv2 | XD-Violence | pair_select | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |
| VideoMAEv2 | XD-Violence | group_uniform | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |
| VideoMAEv2 | XD-Violence | group_random | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |
| VideoMAE | XD-Violence | pair_select | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |
| VideoMAE | XD-Violence | group_uniform | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |
| VideoMAE | XD-Violence | group_random | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |
| TimeSformer | XD-Violence | pair_select | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |
| TimeSformer | XD-Violence | group_uniform | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |
| TimeSformer | XD-Violence | group_random | 10000 | 10000 | 0 | 800 | 2330384 | compressed_video_score_counts_sorted_once_batched_weighted_PR_trapezoids | 20260918 |

---

## 6. 数据可信度声明

- 18/18 压缩 run 的 extraction 合同通过**逐视频闭合核验**（视频 ID 集合、每视频 clip 数、时间覆盖全部与 dense 目标一致）。
- 18/18 预测 `status=completed`、`official_frame_scores_read=false`、唯一视频数正确、chunk receipt 完整。
- 检查点为固定最终 step 3000，无按测试指标选最佳；上游 `xd_main.py` 的测试 AP 存最佳逻辑从未被加载。
- official_frame 帧覆盖恰为 `[0, num_frames)`，未覆盖帧直接报错，无静默截断。
- 行尾等价豁免 3 对，仅作用于 `code_digest` 单字段；其余七个字段仍严格相等。
- 一次提前读分事件已记录于 `unseal_receipt.json`（仅 1 个 dense 基线值，未影响方法选择）。

