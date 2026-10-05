# handoff_v2 / selection_decision.md

**状态**：**待负责人填写**（2026-09-22 创建）。
**规则**：`selection_time` 必须填**实际时间，不倒签**；`selection_basis` 如实填写。
agent 给建议但不代替负责人承诺最终方法（议程 §C3、§9.3）。

---

## 决策背景（事实，供填写时参考）

- 质量矩阵：`urdmu-six-cell-matrix-v1.json`（6 cells，18 压缩 + 6 dense 基线）。
- 三个方法质量**不可区分**：18/18 格通过 0.5 pp 非劣性边际；所有方法间点估计差 ≤ 0.51 pp，
  全部小于方法-vs-dense 的 CI 宽度（约 ±1–2 pp）。
- `pair_select`：六格最差 −0.16 pp（三者中最稳），UCF 平均 +0.075、XD 平均 +0.537；
  但 **UCF 上平均不如 group_random**（−0.147 pp）。
- `group_random`：UCF 平均 +0.222 pp（三者中最好）、最差 −0.09 pp、实现最简；
  XD 平均 +0.183 pp（最低）、XD × VideoMAEv2 −0.43 pp。
- `group_uniform`：两数据集平均均非最好，最差 −0.43 pp，**不建议**。
- **效率完全未测**，且 `pair_select` 的选择器开销存在强负面信号（字段不可用作证据，仅为信号）。

详见 `method_recommendation.md`。

---

## 负责人填写区

```text
selected_method:
selected_method_version:            # 精确代码快照 + 配置 + keep_ratio
quality_matrix_id:                  # urdmu-six-cell-matrix-v1.json
selection_basis:                    # validation / test-comparison / other（如实填写）
selection_time:
strengths:
weaknesses:
next_action:                        # efficiency-only / equivalent-engineering / new-algorithm-version
```

---

## 选择性质声明（必须随决策一并记录）

本次选择**使用了测试分数**。依据议程 §C2 与 [R1]（scikit-learn cross-validation 文档）：

> 测试集参与参数/方法选择后，不再提供独立的最终评价；验证集与测试集职责需区分。

因此：

- 所选方法在**同一组测试数据**上的表现**不再是独立的最终验证**。
- 论文必须如实披露候选确定的**真实时间与依据**，并**不宣称**其为无选择偏差的泛化保证。
- 完整披露减轻误导，但**不消除选择偏差本身**。
- 若之后切换候选为另一方法，须**另记一次决策**，不覆盖本次记录。

---

## 已知须随决策一并处理的事项

1. TimeSformer 实际保留比 0.5717 ≠ 名义 0.60——论文报告实际值。
2. `group_random` 为单 seed（seed=0）——论文写"单 seed"，不夸大稳定性。
3. 效率表须在阶段 D 完成后按 `efficiency_protocol.md` 产生，并与质量表**绑定版本**。
4. 一次提前读分事件已记录于 `unseal_receipt.json`。
