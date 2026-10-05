# 给 GPT-6 Pro 的论文修改任务

请在压缩包中的 `paper/manuscript-20260922/` 修改 ICASSP 2027 PairSelect 作者工作稿。先阅读 `PACKAGE_README.md`、该目录的 `README.md` 与 `notes/AUTHOR_HANDOFF_ZH.md`，再核对 `evidence/three-video-encoders/RESULTS.md`、`evidence/dsanet/RESULTS_PUBLISHED_20260923.md` 和原始 JSON/CSV。最终给出修改后的 LaTeX、可编译 PDF、逐条改动说明，以及“每项正文数值/论断—证据文件”的核对表。作者会对实质性论证和会议 AI 使用要求作最终审阅；不要提交论文、commit 或 push。

## 需要建立的统一叙事

从摘要和引言起，把研究呈现为**对四条视觉骨干/WSVAD 系统线的统一压缩验证**，不要写“先做三个，后来补 DSANet”的开发时间线。前三条为预训练的 VideoMAEv2-B、VideoMAE-B、TimeSformer-B，分别配套**我们在对应数据集完整训练集上训练的 UR-DMU 检测头**；第四条为 CLIP ViT-B/16 视觉塔搭配 AAAI 2026 的 DSANet，使用**作者发布的 UCF/XD 检测器权重**。它们不是四个由我们从零训练的 encoder；VideoMAE 与 VideoMAEv2 属于同一家族，VadCLIP/DSANet 等也不能算多个独立 CLIP 视觉编码器。对四线都把 PairSelect 作为无需额外训练的中间 token 选择插入，固定各自 dense 检测器，不按压缩档重训。介绍 DSANet 时可称“近期 SOTA 级/高性能 CLIP WSVAD 方法”，引用 [AAAI 2026 正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/38191)，不要声称已经核实它截至今天仍绝对排名第一。

希望论文明确说：**40% 标称删除在多数设置下呈近无损的点估计表现；60% 删除时总体仅有轻微掉点，体现方法跨视觉表征与检测器的可迁移性。**请照这个用语组织论述，同时在表格和邻近正文披露实际范围与例外，不能把“轻微”偷换成“所有设置下降≤0.5 个百分点”或“均已统计证明非劣”。尤其保留 UCF 上 DSANet 的 40% 下降约 0.90 pp、60% 下降约 1.80 pp，XD 上 DSANet 60% 下降约 1.22 pp，以及 UCF 上 VideoMAE 60% 下降约 1.05 pp。可说最明显回落约 1–2 pp，整体仍属作者所称的轻微质量代价；同时诚实写明置信区间结论：DSANet×XD 的 20% 删除得到预设 −0.5 pp 非劣性支持，其余 DSANet 档没有。不要根据 test 结果重新定义最优预算。

方法和三条视频骨干的独立检测头是当前工作的可复用基础：在 Discussion/Future Work **只用一两句**提出，今后可以在这些骨干与压缩机制之上训练一个压缩感知或联合优化的 WSVAD 模型，进一步提高质量—效率权衡。这里是未来研究方向，不是已完成的训练、贡献或本轮数值来源；不要暗示当前 PairSelect 已经是学习得到的压缩模型。

## 数据与效率口径

- UCF 主指标都为 frame ROC-AUC。三条视频线与 DSANet 均有 dense/删除20%/40%/60% 的完整结果；DSANet 本轮 raw 特征主分支为 88.868 / 88.542 / 87.966 / 87.065%。作者预提特征+发布权重的旧 UCF 89.4446% 仅是独立校准身份，不能作为本轮压缩差值基线。
- XD 三条视频线的旧主表是**梯形 PR-AUC**，另有 step AP；DSANet 新主结果是 **step AP** 85.597 / 85.490 / 85.353 / 84.379%。不要把这两种 XD 指标放入一个未加区分的数字排名。建议分面表、双指标列或在文中分别论述，并保留来源和时间轴协议。若重新计算统一指标，只能使用压缩包中已审核的预测与真值，不能猜数。
- **不要把 UCF 的 AP 留空并让读者误以为没跑。**现有表若将副指标列写成“XD AP”，UCF 横杠仅表示该表头不适用；但 UCF frame AP 实际已有回执。建议把副指标明确改名为“Frame AP (step, secondary)”并填写两个数据集，或保留 XD-only 列同时在表注明确说明 UCF AP 已计算且另表可查。40% 删除时三视频骨干的 UCF AP（dense→PairSelect）是 VideoMAEv2 19.29→19.60、VideoMAE 18.82→18.99、TimeSformer 20.37→20.90；来源为 `paper/manuscript-20260922/data/quality_matrix_detail.csv` 的 UCF 行（历史字段名 `xd_average_precision_*`，其定义仍为帧级 step AP）。DSANet 的 UCF 40% AP 是 34.74→34.36，来源为 `evidence/dsanet/results_published/quality-ucf.json` 的 `coarse-0.6.frame_ap`。20%/60% 的 UCF AP 也在原始质量导出中；若表格展示全预算，请按对应回执填齐，不从 AUC 推算 AP。
- 40% 删除下，三条视频线 UCF 主指标相对 dense 约为 +0.39/−0.16/0.00 pp，XD 梯形 PR-AUC 约为 +0.32/+0.61/+0.68 pp；DSANet 相应 UCF AUC −0.90 pp、XD step AP −0.24 pp。点估计接近与 95% CI 非劣性是不同命题，正文按实际回执区分。
- 60% 删除可用“总体轻微掉点”概括，但同一段给出最明显的 1–2 pp 回落，不删除任何负面格子。不要说四个系统全都有相同方向，也不要把分类/定位 mAP 的提升当帧级 AP 的提升。
- 效率分开写：三视频线 batch32 编码器在 40% 删除时为 1.225×/1.247×/1.277×，60% 时为 1.330×/1.378×/1.470×；DSANet 的 CLIP 视觉塔 batch32 在 40% 时约 1.22×、60% 时约 1.38×。**完整视频到 CPU 分数**的 DSANet 效率只有约 0.97–1.01×，显存未下降，batch1 视觉塔可能变慢。绝不能把 encoder-only 加速写成端到端加速。VideoMAE-family 与 TimeSformer 的 clip 覆盖不同，也不能直接按 clips/s 做跨模型端到端排名。

## 修改范围与验收

同步修改摘要、引言、贡献、实验设置、结果、讨论、局限和结论，使“四条线”的术语与表格一致；按需修改 `data/` 来源和 `scripts/export_tables.py` 并重新生成表格，**不要直接手改自动生成的数值单元格**。保留方法冻结、真实 token 数、训练集范围、前期 test 接触、不同 raw 表征、置信区间和失败设置的披露。不要编造性质探针、额外训练、能耗、完整模型 FLOPs、端到端显存节省或外部方法胜出。维持 ICASSP 模板与技术正文版面，编译并逐页检查 PDF；缺少可复算来源的数字保持缺项并说明原因。最后给我一段可供作者改写的英文核心论述，同时列出可能被审稿人追问的三处最关键边界。
