> **跨双栏图版更新**：框架图位于第 2 页页首，技术正文仍为 4 页；修改说明见 `notes/DOUBLE_COLUMN_LAYOUT_ZH.md`。主文件为 `main.tex`。

# PairSelect — ICASSP 2027 LaTeX 作者工作稿

更新日期：2026-09-22。直接承接用户提供的 `ICASSP2027_论文咨询包_20260922.zip/01_论文正文`，保留其章节式工程和官方样式文件；重写论文内容、方法定义和结果呈现。原始压缩包未被覆盖。本轮改动的逐项记录见文末「2026-09-22 本轮改动」。

**状态：可编译作者工作稿，不是可以原样提交的终稿。** 当前 PDF 为四页技术内容＋一页声明和参考文献。已完成的中等预算结果来自原始 CSV；新预算与硬件实测缺项显示破折号。尤其需要阅读 `notes/AUTHOR_HANDOFF_ZH.md` 中的 ICASSP 2027 LLM 使用政策说明。删去工作稿提示、简单润色或仅补充 AI 声明，并不能自动满足该政策。

## 编译

Overleaf：上传本目录下全部文件，主文档设为 `main.tex`，编译器选择 pdfLaTeX。不要只上传 `main.tex`。

本地：安装常规 TeX Live / MiKTeX、Python 3，运行：

```bash
bash build.sh
```

Windows 上若 `python3` 被其他环境抢占（`which python3` 不是预期解释器），先把步骤拆开跑，并给 Python 用完整绝对路径：

```bash
"<python>" scripts/export_tables.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

其他脚本：

```bash
python3 scripts/export_submission_fields.py   # 重新导出 submissions/*.txt
python3 scripts/check_refs.py                 # 校验 cite 键与 references.bib 一致
python3 scripts/check_pdf.py main.pdf         # 可选 PDF 预检（需 PyMuPDF）
```

## 2026-09-22 本轮改动

1. 作者块从 3 人扩到 7 人（`author_config.tex`），与投稿表单一致；`Jianbo Yu` 标为通信作者。邮箱和 ORCiD 只在投稿系统填写，未印在 PDF 上。
2. 删除原 TikZ 方法图 `figures/overview.tex`，Fig. 1 改用作者提供的方法总览位图 `figures/pairselect_framework.png`（全栏宽 `figure*`）；因此 `main.tex` 不再加载 `tikz`。
3. `references.bib` 换成另一修订版的 17 条文献（补 EVAD / LAVIDA / AED-AE，去掉 STTS；带 doi/url，键名与方法本名对齐），正文引用键同步改名。相关工作中为这 3 条新增文献补了引用句，其余正文沿用原描述。
4. 新增 `submissions/`、`scripts/export_submission_fields.py`、`scripts/check_refs.py`。

## 主要文件

| 文件 | 用途 |
|---|---|
| `main.tex`、`author_config.tex` | 入口、标题、摘要、作者单位及分页边界 |
| `sections/01_...06_*.tex` | Introduction、Related Work、Method、Experiments、Discussion、Conclusion |
| `sections/07_declarations.tex` | 需作者确认的伦理、资助、利益冲突声明 |
| `references.bib` | 17 篇正文引用文献；核对来源见研究说明 |
| `figures/pairselect_framework.png` | Fig. 1 方法总览图（位图，1536×1024，全栏宽）；原 TikZ 版 `figures/overview.tex` 已删除 |
| `figures/quality_table.tex`、`efficiency_table.tex` | 两个跨栏表的结构和说明（是表格，不是图） |
| `submissions/*.txt` | 从 `main.tex` 自动导出的标题、摘要、关键词纯文本，供投稿系统粘贴 |
| `data/quality_matrix_detail.csv` | 原始已完成实验，完整保留对照数据 |
| `data/budget_results.json` | 新增 keep=0.80 / 0.40 的结果回填入口 |
| `data/efficiency_results.json` | 完整 encoder FLOPs、延迟、吞吐、显存的回填入口 |
| `data/quality_ladder_summary.csv` | 自动导出全部预算的均值、变化及可用区间 |
| `data/analytical_costs.json` | 自动重算的理论预算和主要 block 运算量 |
| `tables/*.tex` | 自动生成，不应直接修改数值单元格 |
| `notes/FORMAT_CHECK.md` | 官方规则、检查边界与编译后检查记录 |
| `notes/EVIDENCE_AND_LITERATURE.md` | 实验—代码—论断来源映射、文献核查与不可越界的解释 |
| `notes/AUTHOR_HANDOFF_ZH.md` | 下一轮修改和补测说明 |

## 回填口径

`budget_results.json` 的键是**保留率**，不是删除率。`0.80` 对应删除20%，`0.40` 对应删除60%。`quality` 使用0–100标度；`delta_pp` 和 `ci_*_pp` 使用百分点。填入数值必须同时填写 `source`。原始 CSV 中的部分指标与 CI 使用0–1标度，导出脚本已转换，不要重复乘100。

`efficiency_results.json` 的所有实测值目前均为 `null`。分别记录 latency、throughput、memory 的 batch，不能只用一个模糊的 batch 字段。保持 GPU、精度、输入、后端及完整执行边界可比；补齐 `source` 与运行元数据。理论 GFLOPs 不得填入实测列冒充完整 encoder 结果。

回填后还必须更新摘要、实验解释、结论和工作稿提示。脚本只更新数据表，不会自动宣称最优预算，也不会自动改写科研结论。
