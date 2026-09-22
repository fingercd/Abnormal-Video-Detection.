# PairSelect：跨双栏流程图版修改说明

## 结果

本版直接基于本次上传的 `icassp2027.zip` 内 `manuscript-20260922/` 修改，未使用更早的论文工程覆盖本稿。已核对外层目录和包内 `manuscript-20260922.zip` 的原始文件一致。

- 流程图置于第 2 页页首，横跨两栏：从约 86 mm 扩大至 178 mm，线性放大约 2.07 倍，保持长宽比。
- 最终为 5 页：技术内容结束于第 4 页，第 5 页仅保留原有声明及参考文献。
- 作者、单位、摘要、原始 PNG、参考文献、所有数据文件、表格数值、七个方法公式和作者声明保持不变。
- 保留现有待测实验的破折号；没有补写新的质量或硬件结果。

## 怎么改的

跨栏图独立放在 `figures/framework.tex`：

```tex
\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{figures/pairselect_framework.png}
\caption{...}
\label{fig:overview}
\end{figure*}
```

`main.tex` 在 Introduction 之后、Related Work 之前调用 `\input{figures/framework}`，使跨栏浮动体在第一页进入队列，落在第二页页首。原来 `sections/03_method.tex` 中的单栏 `figure` 已删除。只把 `\columnwidth` 改成 `\textwidth` 而不换 `figure*`，会让图片挤入相邻栏；把 `figure*` 定义放得太晚，也可能使它推迟到第三页。

原始 PNG 未重绘、未重采样、未拉伸，长宽比不变；仅裁掉底部 169/1024 px 那段烙在图里的 "Fig. 1. …" 说明文字，因为 LaTeX 图注与其重复。图注改为对局部配额、pair 成员优先级和完整轨迹适配的简洁概括；正文新增 Fig. 1 引用。

## 为跨栏图腾出的空间

提前入队质量表与效率表，并保持表格编号和数据不变；预算表采用 `[!b]`，在第二页左栏下方显示，使方法标题能直接接在框架图后。

正文只压缩重复表达：Related Work 末尾的一句定位说明、方法中的范数动机、预算讨论、硬件测量协议、Scope and Discussion 与 Conclusion。保留实验设置、指标区分、统计区间解释、跨架构限制及尚待测量的事实。Introduction 主体保留原文，仅补图号引用。

## 本地独立复核（2026-09-22 晚，本机 TeX Live 2024）

对上述"只压缩表达"的逐条比对结果：02/05/06 为等义改写，四条局限、全部 17 处引用、21.75--23.04\% 等数字均保留；04 中"Setup and evaluation"与"Direct insertion"两段是**整块搬家**（移到两张表定义之后以便浮动体提前入队），文字逐字未改，1,610/290、3,950/3,954/800、3,000 步、69,364/145,703/138,853/291,781、$+0.39/-0.16/0.00$、$+0.32/+0.61/+0.68$、AP 70.33→71.30/64.86→64.51/70.86→72.04、23.04/21.75/33.05/32.43\%、64.24\%、20 次 warm-up 与 3 次进程重复全部在位；03 的七个公式与全部数学符号未动；01 仅新增 `Fig.~\ref{fig:overview}`。

同时复核：`author_config.tex`（7 作者）、`references.bib`（17 条）、`sections/07_declarations.tex`（LLM 披露维持"author confirmation required"，未新增任何 AI 使用声明）、`data/*`、`tables/*` 数值单元格、PNG 长宽比均未改动。本机重建结果：5 页、技术内容止于第 4 页、Letter、字体嵌入无 Type 3、0 overfull、0 未定义引用，与上文记录一致。

图表与正文的间距设置为正值：

```tex
\setlength{\dbltextfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
```

没有更改 `spconf.sty`、页边距、正文或图注字号，也没有负间距挤压、跨页裁切或非等比例拉伸。

## 编译与回填

主文件仍为 `manuscript-20260922/main.tex`。在该目录运行：

```bash
bash build.sh
python3 scripts/check_pdf.py main.pdf
```

Overleaf 使用 pdfLaTeX，并将 `main.tex` 设为主文档。完整工程包保留原目录结构；仅论文 TeX 包可直接用作论文工程。外层包中的 `manuscript-20260922.zip` 已同步更新，避免误用旧稿。

## 本地检查

最终 PDF 已逐页渲染检查，并通过项目本地检查脚本：5 页，技术内容结束页为 4；Letter 页面；无未解析引用、超栏警告或过大浮动体；PDF 字体全部嵌入并子集化，无 Type 3 字体。原始位图宽度 1536 px，在 178 mm 版面宽度下约为 219 ppi；放大改善印刷尺寸，但不增加图像细节。

详细检查保存在 `notes/DOUBLE_COLUMN_LAYOUT_CHECK.json`，逐文件补丁为 `notes/double_column_layout.patch`。这是本地版式检查，不是会议系统验收，也不代表对研究或声明的全面核验。

## 沿用原图、未在本次修改的内容提醒

原图仍含 `CLIP`、`Keep top-k pairs per group`、`Reduces FLOPs, memory, and latency` 等文字。本次只改变图在论文中的大小与位置，没有重绘图内内容。这些文字应在提交前与正文的三个编码器、token/pair 配额规则和实际完成的硬件测量核对；原图内的小字为栅格内容，未对其字号作合规认证。

## 官方格式依据

ICASSP 2027 Paper Kit 明确允许插图跨两栏，并优先建议页首放置；技术内容最多四页，第五页仅允许参考文献、资助致谢和伦理合规声明；正文及图注不小于 9 pt。

来源：https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php （本次访问核对：2026-09-22）。
