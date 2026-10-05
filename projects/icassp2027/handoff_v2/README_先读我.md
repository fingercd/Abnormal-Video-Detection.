# README — 如何读这个压缩包

> 这是给 AI 顾问的阅读指南。真人直接看 `00_一页纸总览.md` 和 `02_实验结果/method_choice_report.html`。

---

## 这是什么

一个 ICASSP 2027 投稿项目的**完整材料包**：论文草稿（LaTeX）+ 压缩方法源码 + 已解封的实验结果 + 冻结协议 + 审计记录。

研究问题：**在弱监督视频异常检测（WSVAD）里，把视频 Transformer 的 token 砍掉 40%，检测质量会掉多少？该砍哪些 token？**

## 建议阅读顺序（约 30–60 分钟）

| 步骤 | 文件 | 目的 |
|---|---|---|
| **1（必读）** | `00_一页纸总览.md` | 实验是什么、三种方法怎么压缩、压缩比、全部指标、**以及论文草稿和实测之间的 6 处落差** |
| **2（必读）** | `01_论文正文/main.tex` + `sections/03_method.tex` + `sections/04_experiments.tex` | 现有论文叙事。注意：定量内容全是占位符，且与实测有落差（见总览 §7） |
| **3（核心代码）** | `03_压缩方法代码/token_reduction/token_selection.py` | **压缩规则的定义处**。文件顶部 docstring 写明了三档分组预算、配额公式、pair 优先级、位置语义、泄露契约 |
| **4** | `03_压缩方法代码/token_reduction/deployment.py` | 三种 rule 如何分派、identity/receipt 字段；`GroupSelectDeployment` 的构造函数即完整配置清单 |
| **5** | `02_实验结果/quality_matrix_detail.csv` | 18 行可追溯明细：指标、dense/方法值、百分点差、CI、非劣性、retained_ratio、bootstrap 配置、各产物 SHA。**所有数字的权威来源** |
| **6** | `02_实验结果/urdmu-six-cell-matrix-v1.json` | 装配后的矩阵。`cells[*].methods[*].token_execution` 是真实执行的 native/retained token 与 retained_ratio；`cells[*].methods[*].quality` 是指标；`provenance` 是全部 SHA |
| **7** | `02_实验结果/18份原始质量报告/<dataset>/<encoder>/<method>.json` | 18 份逐 run 原始导出，含 bootstrap draws/valid/invalid、engine、frame 计数 |
| **8** | `04_协议与冻结决策/six-cell-method-freeze-v2.json` | 方法冻结契约：methods、keep_ratio、insertion 层、selection_unit、test_gate、bootstrap 配置 |
| **9** | `05_核验与审计/audit_summary.md` | 来源/覆盖/预算/head QA 的审计摘要（公开版不含分数） |
| **10** | `05_核验与审计/paper_claims_checklist.md` | 论文主张边界：哪些能写、哪些不能写、为什么 |

## 目录结构

```
00_一页纸总览.md                一页纸事实总览 + 草稿与实测的落差
PROMPT_给顾问.md                本次咨询的任务说明
01_论文正文/                    LaTeX 草稿（main.tex + 6 个 section + 参考文献 + 官方模板 sty/bst）
02_实验结果/                    质量矩阵、明细 CSV、矩阵 JSON、18 份原始导出、渲染版 HTML 选型报告
03_压缩方法代码/                token_reduction 全模块 + paper 子集 + 装配/生成脚本
04_协议与冻结决策/              方法冻结合同、检测器协议、reducer 评估冻结
05_核验与审计/                  审计摘要、行尾等价证据、解封收据、主张清单、效率协议、选择记录
06_研究计划与背景/              研究计划 V2/V3、项目 README、状态总结、收尾计划
07_项目规范/                    AGENTS.md、CLAUDE.md（实验红线与工作规范）
```

## 读代码的三个要点

1. **不要按方法名字猜算法。** `pair_select` 不是"importance-based pruning"，`group_uniform` 不是"均匀采样整个序列"——它们是**分组预算**规则下的三种不同选法。真正的定义在 `token_selection.py` 的 docstring 和 `group_quotas()` / `GroupBudgetSelector`。
2. **三者的预算和配额表完全相同**（`same_budget_for_all_controls = true`）。所以质量差异只可能来自"选哪些 token"，不掺预算偏差。这是干净的对照实验。
3. **两种位置语义**：`group_uniform`/`group_random` 是精确静态 gather（保留原生位置）；`pair_select` 是逐前向动态，填充 skeleton 槽位，位置语义为 `approximate_anchor_position`。

## 读数据的三个要点

1. **指标标度**：报告用 0–100；差值一律用**百分点**（= 100 ×(压缩 − dense)），不是相对百分比。
2. **两个数据集指标不同**：UCF 是 frame ROC-AUC，XD 是梯形 PR-AUC（另有 step AP 单列）。**不可跨数据集平均成一个总分**。
3. **哪些是事实、哪些还没做**：质量数据齐全；**效率数据（显存/延迟/吞吐）完全没有**；方法-vs-方法的配对 bootstrap CI **尚未计算**。总览 §5 和 §4 已标明。

## 已知边界（不要误判为疏漏）

- 只跑了 **keep_ratio 0.60 一档**预算，没有预算阶梯。
- `group_random` 是**单 seed**。
- 六格共享同一批测试视频（UCF 三个编码器用同一 290 个、XD 用同一 800 个），**不是六次独立统计重复**。
- 候选方法的选择使用了测试分数比较，已在 `05_核验与审计/unseal_receipt.json` 和 `paper_claims_checklist.md` 中披露。
