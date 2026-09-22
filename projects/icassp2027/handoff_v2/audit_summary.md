# handoff_v2 / audit_summary.md

**生成时间**：2026-09-22（CST）
**生成者**：值守 agent（Claude Code）
**范围**：ICASSP 2027 六格压缩矩阵 — 来源、覆盖、预算、head QA 的公开审计摘要
**红线声明**：本文件**不含任何正式测试指标数值**。矩阵装配通过并由负责人确认解封前，本文件不引用 ROC-AUC / AP / 任何分数。

---

## 1. 审计对象

| 类别 | 数量 | 说明 |
|---|---:|---|
| dense 基线 head | 6 | 3 编码器 × 2 数据集，`checkpoints/final.pt` |
| 压缩 prediction run | 18 | 3 压缩方法 × 3 编码器 × 2 数据集 |
| 压缩 quality export | 18 | 每 run 一份，`urdmu-quality-exports-r08/<dataset>/<enc>/<method>.json` |
| 压缩 extraction contract | 18 | 每 run 一份，`compressed-test/<enc>/<dataset>/<run-id>/extraction-contract.json` |

数据来源仅限 node3 正式 manifest 与输出根 `/data2/localdisk/fotile-icassp2027-kimi-20260919-a01`。**node2 未接入、不补缺、不引用**（依据计划 §1.1）。

---

## 2. 来源核验（已逐项实测）

### 2.1 head 检查点与选择规则（A2 要求，已代码级取证）

| 核验项 | 结论 | 证据 |
|---|---|---|
| 检查点为固定最终 step，无"按测试 AP 存最佳" | **通过** | 上游 `xd_main.py` 不在 `urdmu_backend.py` 的 `SOURCE_SHA256` 与 `_DEFINITIONS` 清单内，从未被加载执行；本仓训练循环单趟，循环外唯一 `save_checkpoint` 落 `checkpoints/final.pt` 且 `step=request.steps` |
| 下游加载侧拒绝非固定 step | **通过** | `urdmu_evaluation.py` 强制 `metadata.checkpoint_selection == "fixed_final_step_no_test_selection"`、`steps==3000`、`qa.checkpoint.step==3000`、`nonzero_gradient_steps==3000` |
| 训练中唯一 `.eval()` 作用在训练数据 | **通过** | 仅对第 1 个 optimizer batch（训练数据）做一次重载一致性前向，自证字段 `fixed_batch_source = "first optimizer batch; no evaluation/selection data"` |
| 冻结合同声明与实现一致 | **通过** | `method-freeze-contract-v2.json`：`checkpoint_selection = fixed_final_step_no_test_selection` |

### 2.2 official_frame 推理一致性（A2 要求，已代码级取证）

| 核验项 | 结论 | 证据 |
|---|---|---|
| 帧覆盖恰为 `[0, num_frames)`，无静默截断 | **通过** | 窗口分数→帧分数用全量 `num_frames`；`aggregate_interval_scores` 对 `ends > num_frames` 报错，且 `contributors == 0` 的未覆盖帧直接 `raise` |
| 无插值、无 padding 旁路 | **通过** | 按半开区间 `[frame_start, frame_end)` 直填，无插值分支 |
| clip stride / 尾部残段显式 | **通过** | 每视频窗口与 `DenseSamplingPlan.sample(num_frames)` 逐一比对 `frame_start/frame_end`、`start_s/end_s` |
| 推理不计算帧级 test 指标 | **通过** | `official_frame` phase 的 `metrics` 保持 `None`，不打开帧级真值 |

### 2.3 装配器输出隔离（A4 要求，已代码级取证）

`scripts/icassp2027/assemble_urdmu_matrix.py` 全脚本唯一一条 print：
```python
print(json.dumps({"status": "completed", "cells": len(matrix["cells"]), "output": str(args.output.resolve())}, ensure_ascii=False))
```
**stdout/stderr 不含任何指标数值**。分数只写入输出 matrix 文件（交付物本身）。断言类异常消息只暴露 `dataset/encoder/path` 与 pass/fail。

---

### 2.4 预测完成态核验（A2 要求，18/18 实测）

核验方法：逐 run 读 `result.json` 与 `predictions.jsonl` / `query-chunk-receipts.json`，
统计**唯一 video_id 数**（非行数）、chunk receipt 覆盖视频数、`(video_id, clip_id)` 重复数。

**关于 schema 的实测结论**：`predictions.jsonl` 是**一 clip 一行**，不是一视频一行
（例：XD × TimeSformer × pair_select 为 582,733 行覆盖 800 个唯一视频）。
因此依据计划 §A2，**不以"恰好 800/290 行"为验收判据**，而以唯一视频数与 chunk receipt 覆盖为判据。

| 数据集 | 编码器 × 方法 | status | official_frame_scores_read | 唯一视频 | receipts | 重复 clip | 判定 |
|---|---|---|---|---:|---:|---:|:--:|
| UCF-Crime | 3 编码器 × 3 方法（9 run） | completed ×9 | False ×9 | 290 ×9 | 290 ×9 | 0 ×9 | **PASS** |
| XD-Violence | 3 编码器 × 3 方法（9 run） | completed ×9 | False ×9 | 800 ×9 | 800 ×9 | 0 ×9 | **PASS** |

**总计 18/18 PASS。** 另实测所有 XD 评分日志中 `direct_insert may only` 报错数为 **0**
（即行尾等价放行表 3 对精确覆盖全部 12 个需豁免的 run，无遗漏）。

---

## 3. 覆盖核验（A1 要求，逐视频闭合，非只比总数）

核验方法：对每个压缩 run，取其 `extraction-contract.json` 的 `coverage.per_video`，与 dense 视图
`merged/<enc>/<dense-view>/merge-contract.json` 的 `video_provenance` 做**逐视频**比对：
视频 ID 集合精确一致、无重复、无缺失、无额外；每视频 `intervals` 数等于 dense `record_count`；
`gap_frames == 0`、`coverage_ratio == 1.0`、`covered_frames <= num_frames`；clips 总数等于 dense 记录总数；
合同 `status == "ready"` 且 `test_only == true`。

| 数据集 | 编码器 | run | videos | clips | dense clips | mism | gap | cov<1 | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|:--:|
| UCF-Crime | VideoMAEv2 | group_random-0p60-r03 | 290 | 69364 | 69364 | 0 | 0 | 0 | **PASS** |
| UCF-Crime | VideoMAEv2 | group_uniform-0p60-r03 | 290 | 69364 | 69364 | 0 | 0 | 0 | **PASS** |
| UCF-Crime | VideoMAEv2 | pair_select-0p60-r03 | 290 | 69364 | 69364 | 0 | 0 | 0 | **PASS** |
| UCF-Crime | VideoMAE | group_random-0p60-r02 | 290 | 69364 | 69364 | 0 | 0 | 0 | **PASS** |
| UCF-Crime | VideoMAE | group_uniform-0p60-r02 | 290 | 69364 | 69364 | 0 | 0 | 0 | **PASS** |
| UCF-Crime | VideoMAE | pair_select-0p60-r02 | 290 | 69364 | 69364 | 0 | 0 | 0 | **PASS** |
| UCF-Crime | TimeSformer | group_random-0p60-r02 | 290 | 138853 | 138853 | 0 | 0 | 0 | **PASS** |
| UCF-Crime | TimeSformer | group_uniform-0p60-r02 | 290 | 138853 | 138853 | 0 | 0 | 0 | **PASS** |
| UCF-Crime | TimeSformer | pair_select-0p60-r02 | 290 | 138853 | 138853 | 0 | 0 | 0 | **PASS** |
| XD-Violence | VideoMAEv2 | group_random-0p60-r01 | 800 | 145703 | 145703 | 0 | 0 | 0 | **PASS** |
| XD-Violence | VideoMAEv2 | group_uniform-0p60-r01 | 800 | 145703 | 145703 | 0 | 0 | 0 | **PASS** |
| XD-Violence | VideoMAEv2 | pair_select-0p60-r03 | 800 | 145703 | 145703 | 0 | 0 | 0 | **PASS** |
| XD-Violence | VideoMAE | group_random-0p60-r01 | 800 | 145703 | 145703 | 0 | 0 | 0 | **PASS** |
| XD-Violence | VideoMAE | group_uniform-0p60-r01 | 800 | 145703 | 145703 | 0 | 0 | 0 | **PASS** |
| XD-Violence | VideoMAE | pair_select-0p60-r01 | 800 | 145703 | 145703 | 0 | 0 | 0 | **PASS** |
| XD-Violence | TimeSformer | group_random-0p60-r01 | 800 | 291781 | 291781 | 0 | 0 | 0 | **PASS** |
| XD-Violence | TimeSformer | group_uniform-0p60-r01 | 800 | 291781 | 291781 | 0 | 0 | 0 | **PASS** |
| XD-Violence | TimeSformer | pair_select-0p60-r01 | 800 | 291781 | 291781 | 0 | 0 | 0 | **PASS** |

**总计 18/18 PASS。**

- clip 数按编码器不同而不同（V2/VMA 每视频 16 帧窗口、TS 8 帧窗口），但**每个编码器的压缩侧与其 dense 侧严格相等**，这是公平比较的前提。
- 合同 schema 中 `coverage.per_video[*].complete` 字段为 `false` 是**良性字段**（区别于 dense 合同的顶层 `complete`），因为同批数据的 `coverage_ratio` 全为 1.0、`gap_frames` 全为 0、逐视频 interval 数与 dense 目标全等。依据计划 §A1，本条核验**不要求"合同时间戳晚于提取进程退出"**，只要求数据写完后才提交合同。
- VideoMAEv2 的 `pair_select-0p60-r01` / `r02` 为历史废弃 attempt，无合同、不参与矩阵；正式 run 为 **r03**（以正式 manifest 为准，且与装配器 `METHOD_RUN_XD` 硬编码一致）。
- 装配器 `METHOD_RUN_UCF` 硬编码的 run-id（V2→r03、VMA→r02、TS→r02）与上述实测存在的正式 run **逐一吻合**；UCF dense 视图命名（videomae→`dense-test-view-v2`，其余→`dense-test-view`）与 head 路径（`training-full/<enc>/formal-dense-s0`）亦吻合。

---

## 4. head QA（已随训练产物落盘）

六个 dense head 均满足（由 `urdmu_training.py` / `urdmu_evaluation.py` 强制）：
`steps == 3000`、`qa.checkpoint.step == 3000`、`nonzero_gradient_steps == 3000`、
`checkpoint_selection == fixed_final_step_no_test_selection`。
head 元数据中 `representation.backbone.code_digest` 已逐一提取（见 `lineending_equivalence.md`）。

---

## 5. 预算核验（待装配器汇总）

实际 token 保留量由 `assemble_urdmu_matrix.py` 流式汇总每个压缩 FeatureStore 的真实
native/retained token、ratio 与 clip 数后落盘。**本节在装配运行后回填**；不为赶时间预填或估算。
冻结规则：keep-ratio 0.60（保留约 60%、移除约 40%），三个方法同预算。

---

## 6. 已知未满足项（如实记录，不宣称通过）

| 项 | 事实 | 影响 |
|---|---|---|
| encoder 峰值显存 | 本论提取窗口的 GPU monitor 08:57 启动、09:00 退出，仅覆盖约 3 分钟 | 不构成本轮效率证据。已由负责人确认**不解封硬门槛**（`claims_not_established` 明确列 `peak_memory_reduction` 为未建立主张；`access_gates` 与 `execution_gates` 均不含显存要求）。效率证据改由阶段 D 候选确定后的小样本 benchmark 提供 |
| 行尾等价豁免 | 3 对 code_digest 豁免，作用域仅 `code_digest` 单字段 | 见 `lineending_equivalence.md`；未留决策文档（按当时指示），证据链完整归档 |
| node2 补充提取 | 未复核、不接入 | 按计划 §1.1 明确排除，不影响 node3 正式目录 |

---

## 7. 红线保持情况

- 正式测试分数：**未读取、未引用、未用于方法选择或检查点选择**。
- 未伪造任何指纹：豁免仅限已证明"仅行尾不同"的 code_digest 对，且只加在代码校验层，未改任何 run 的记录文件。
- 失败与历史目录全部保留（`attempt00-failed`、`r02` 等），未覆盖、未删除。
- 每个 completed run 均含完整 provenance（`predictions.jsonl`、`query-chunk-receipts.json`、`resolved.json`、`result.json`、`provenance/stages/*`）。
