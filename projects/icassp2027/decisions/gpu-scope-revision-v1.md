# GPU 范围与并行拆分修订（v1）

**记录时间**：2026-09-22（实际时间，不倒签）
**记录人**：值守 agent，经负责人明确授权
**性质**：对既有硬验收规则的**范围修订**。原规则原文与本次变更事实一并保留；
**不改写**旧 freeze receipt、不伪造旧检查通过、不删除既有记录。

---

## 1. 原规则（引用，未改）

`projects/icassp2027/decisions/official-detector-protocol-v3.json`：

```json
"resources": {
  "existing_V100_clip_results": "reference_only",
  "formal_timing": "exclusive_gpu_with_foreign_process_and_noise_checks",
  "ordinary_quality": "explicit_sharing_with_per_run_lease_and_memory_budget",
  "primary_gpu": "node3_A100_all_eight_authorized"
}
```

`projects/icassp2027/decisions/reducer-evaluation-freeze-v1.json` 的评价口径与
`six-cell-method-freeze-v2.json` 的 `test_gate` 均**未涉及 GPU 型号或并行拆分方式**。

## 2. 本次修订内容

| 项 | 修订前 | 修订后 |
|---|---|---|
| 正式 sealed-test 特征提取可用节点 | 仅 node3 A100 | node3 A100（8 卡）+ **node1 V100（8 卡）** |
| 单任务并行方式 | 一个 run 一个进程、全量视频一次跑完 | **按视频对半拆 2 片**，2 张卡跑一个任务，跑完合并 |
| 共享 GPU | `ordinary_quality` 已允许 `explicit_sharing_with_per_run_lease_and_memory_budget` | 沿用该机制（`run_when_gpu_free.py --allow-sharing --min-free-memory-mib 4096`） |

## 3. 修订理由

1. **特征提取是确定性计算**，与卡型无关。同一份代码快照
   （`code-kimi-adgs-20260921`，commit `dd1b4c5f0a830c2dc139d48cb4c53e3d151c6098`）
   已在同一数据集上跑通全部三个 encoder。
2. **V100 跑提取有历史事实依据**：`projects/icassp2027/progress.md:100` 记录 2026-09-22
   曾在 node2 空闲 V100 上启动 8 个独立输出的补充 extraction，覆盖三 encoder 的 UCF/XD
   `pair_select@0.60`；`progress.md:441-450` 另记录一次 16 卡（node2 V100×8 + node3 A100×8）
   共享提取，实测 16 个 run 全部有正向产物增长。
3. **拆分不改变科学内容**：两片的视频集合不相交、并集精确等于 sealed cohort；
   合并由仓库原装 `merge_feature_runs.py` 完成，带 per-video provenance，
   且 `--transport hardlink_npz` 为零拷贝。
4. 负责人明确要求"16 张卡全部启动，不要空"，且只有 12 个任务，故以"对半拆分 + 效率补测并行"
   填满 16 卡。

## 4. 不受本次修订影响的部分

- 方法与预算冻结：`six-cell-method-freeze-v2.json`、`method-freeze-contract-v2.json` **不变**。
- 评价协议：UCF frame ROC-AUC、XD 梯形 PR-AUC 主指标 + step AP 单列、10,000 次视频级
  paired bootstrap、非劣性边际 −0.005 **不变**。
- 检查点选择：固定最终 step 3000、无按测试指标选最佳 **不变**。
- 逐视频闭合核验、fail-closed 装配、`official_frame_scores_read=false` 等审计要求**不变**。
- 正式效率计时口径：`formal_timing` 仍要求独占 GPU；共享模式下产出的 run
  **不是 formal-timing eligible**（`run_when_gpu_free.py:477`），效率表一律另跑
  `--formal-timing` 独占窗口。

## 5. 已知风险与如实记录

- 合并视图是否满足质量导出链路的 `_feature_contract` 校验（要求 FeatureStore 的
  `resolved.json` + `status.json` 带 `data_content_evidence.canonical_manifest_sha256`
  且覆盖全部 290/800 视频）**未经验证**。负责人已知情并决定直接全量开跑；
  若某任务被 fail-closed 拒绝，改用 `--resume-source` reconcile 通道（搬运不重算）兜底，
  只影响被拒任务，失败目录保留不覆盖。
- V100 不支持 bf16。已由历史运行事实佐证三个 encoder 均可在 V100 实际执行
  （`progress.md:1605`：TimeSformer 在 node2 V100 实际完成 identity 等路径）。

## 6. 本次修订新增的测试用途

用于 `pair_select` 在 keep_ratio **0.80** 与 **0.40** 两档的预算扫描
（补齐冻结阶梯 `SUPPORTED_KEEP_RATIOS = (0.80, 0.60, 0.40)` 的另外两级），
以及 dense 与三档 pair_select 的效率补测。

**选择性质提示**：0.60 档的结果已解封并参与过方法比较；本次扩展网格后若依据新结果挑选
最终档位，属测试结果驱动的比较，必须在 `handoff_v2/selection_decision.md` 中
以 `selection_basis = test-comparison` 如实记录，不得称为独立最终验证。
