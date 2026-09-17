# Token 计算节省的文献基线与主张边界（受控核查）

日期：2026-09-18。范围：冻结 `videomaev2`、`timesformer`、`vjepa2`、`videomae` 的弱监督 VAD，比较免训练与轻量可训练的 token 处理插件。仅核对一手论文及能确认归属的作者官方代码链接；这不是穷尽检索，也不是实验结果。

## 可用于决策的结论

1. **免训练视频 token 合并已被直接覆盖。**vid-TLDR 已在视频 Transformer 中以 attention-derived saliency 区分背景 token 并进行免训练合并；ToMe 与 ATS 已分别覆盖通用的免训练合并和自适应采样。因此，不能主张“首个免训练视频 token merging / pruning”，也不能把 attention 或输入自适应本身当创新点。
2. **VAD 中“异常稀疏，因而应按反向注意力压缩”的叙述也已有直接先例。**LAVIDA 明确报告 reverse-attention token compression 来应对异常的时空稀疏性。它是零样本 MLLM 体系，不是当前四个 encoder 上的同设置基线；但足以禁止把该规则、或“异常 token 稀少所以压缩”单独写成首次贡献。
3. **文献支持局部异常和背景干扰是值得检验的动机，不支持预设某个 ViT token 统计方向。**Sultani 等的弱监督设定和 STPrompt 都说明长视频/全帧中异常可局部化、背景可能主导；RTFM 还把高的*经训练 snippet feature magnitude*当作判别假设。它们都不能推出异常 ViT patch 必然高幅值、低熵、高运动或低冗余。现有任务的 C1/C2/C3、视频级重采样和反例记录仍是必要条件。
4. **较有希望且仍需实验成立的定位**是：在冻结 encoder 的在线可用内部性质中，先独立证实一个正常—异常差异，再证明该性质在相同 token 预算下优于通用合并/剪枝，同时报告四类 encoder 的实际净延迟、显存和 VAD 指标。有限检索未发现完全相同的“四 encoder、性质先验证、弱监督 VAD token 节省”组合；这不是新颖性保证，投稿前仍需做系统检索。

## 同预算基线：最小而充分的集合

| 插件路径 | 必做比较 | 为什么不可省略 | 适用与边界 |
|---|---|---|---|
| 所有路径 | Dense identity；固定时空均匀保留/池化；固定 seed 的随机保留/随机组内合并 | 分离“少 token 本身有效”与 selector/merge rule 的作用 | 每层实际 token 数、插入层和 merge 聚合完全一致；identity 还应验证数值输出不变。 |
| 免训练、以合并为主 | ToMe 相似性双边匹配；vid-TLDR attention-saliency 合并 | 前者是通用免训练合并，后者是最接近的视频 attention 显著性合并 | 以当前 encoder 真正可获得的 token/attention 实现；若取得完整 attention 才能打分，计入该成本。 |
| 免训练、以保留/动态数目为主 | ATS 或其严格等价的无参数 attention sampling；简单幅值分数 | ATS 已覆盖无训练、输入自适应采样；幅值是 VAD 文献中实际使用过的判别线索 | ATS/EViT 的 CLS-attention 变体只在 encoder 有真实 CLS 时适用；无 CLS 不能把首个 patch 伪作 CLS。动态预算须同时报均值、p90 和最坏 token 数/延迟。 |
| 若主方法主张 merge 优于 drop | drop-only 与 ToFu 的 norm-preserving merge（或当前相同分组上的等价实现） | ToFu 指出均值合并会造成特征范数/分布偏移，并比较 pruning 与 merging 的适用性 | 不是要求复现其全部训练配方；要保持 pair/group、token 数和读出一致。 |
| 轻量可训练 selector/组关系预测器 | 同预算、同插入层的 DynamicViT 风格小预测器；以及主插件的随机/均匀目标训练对照 | 已有轻量 predictor、渐进 pruning 与端到端训练先例；否则“训练一个小路由器”缺少强基线 | 只训练插件时，冻结 encoder 与统一 MIL head 预算；若端到端训练，则单列，不能与冻结轻量插件混作一条曲线。 |
| VAD 特异主张 | 规则型 reverse-attention selector（LAVIDA 概念对照）；token 幅值 selector（RTFM 概念对照） | 分别排除“反注意力保留异常”和“异常幅值”被重新命名 | LAVIDA 的 MLLM、伪异常、零样本协议与本项目不同，不能直接搬数字；RTFM 是时间 snippet 表征与训练损失，也不能直接当 patch-token 结论。均须在同一当前 backbone/head/预算下重实现或明确标为不可直接比较。 |

两个条件应同时满足，才称“同预算”：每个插入点后的实际序列长度相同（对动态方法以逐样本长度配对或报完整分布），并且保存相同的 token 身份语义/位置处理和同一输入时间覆盖。只在完整 dense 前向之后删除 token、清零 token，或用 dense attention 离线产生在线 selector，都不属于 encoder 计算节省的比较。

## 免训练与可训练插件的公平成本

免训练不等于零校准，也不等于无额外计算。每个方案应分别记录：(a) 分数/配对/聚类所用层的 QKV、attention、排序或匹配时间；(b) reducer 之后每层真实 token 数；(c) 仅模型、含 adapter、端到端三种计时；(d) allocated 与 reserved 峰值显存；(e) 固定 batch 延迟和吞吐。尤其是 attention-based selector：若为取得全 attention 禁用了融合 attention 或构造了完整 `N×N` 矩阵，它的实测成本必须计入，不能只报告 reducer 后的 FLOPs。

训练路径还要单列训练成本和监督信息：训练视频/视频级标签、正常原型或阈值拟合、selector 参数量、训练轮数、教师/蒸馏前向及后缀 head 是否反传。`direct_insert`（dense-trained head 直接接压缩特征）与 `refit_head`（同一预算重训每个方法的 head）必须分开；后者对所有方法给相同数据、seed 与训练预算。任何由异常类别、文件名、测试汇总统计或未来层 token 得到的 selector 都不能称部署可用。

## 主张措辞的红线

- 不写：首次 training-free video token merging/pruning、首次 attention-guided 视频 token 选择、首次动态 token sampling、首次 VAD reverse-attention compression、或“异常天然高幅值/高运动/低冗余”。
- 可在实验证据成立后写得更窄：提出一个由预注册 probe 确认、只用当前层状态的 token 规则；它在指定弱监督 VAD 协议和四个具体 encoder 上，以相同预算超过明确列出的通用和 VAD 特异基线，并产生实测净收益。
- 不能从 STPrompt 的“许多异常区域相对全帧较小”、Sultani 的“异常在长视频中往往短暂”、或 RTFM 的训练目标，跳到“正视频中每个 clip/patch 均异常”或任意内部 token 性质。对反向、类别特异或受场景/运动控制后消失的结果，应保留为反例而非改写规则。

## 关键一手来源（直接支持与限制）

| 来源 | 年份 | 直接支持的结论 | 不能支持的结论 |
|---|---:|---|---|
| Bolya et al., [Token Merging: Your ViT but Faster](https://openreview.net/forum?id=JroZRaRw7Eu)；[官方代码](https://github.com/facebookresearch/ToMe) | 2023 | ToMe 以相似 token 的轻量匹配做合并，可免训练应用，也报告视频 ViT 吞吐。 | 不涉及 VAD、正常/异常性质或四个目标 encoder。 |
| Fayyaz et al., [Adaptive Token Sampling for Efficient Vision Transformers](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710397.pdf)；[官方代码/项目页](https://adaptivetokensampling.github.io/) | 2022 | 参数无关、可插入的 attention-based 自适应 token sampling，评估包括视频分类；也可训练。 | 自适应 token 数会影响批处理/延迟口径；不是 VAD 证据。 |
| Choi et al., [vid-TLDR: Training-Free Token Merging for Light-weight Video Transformer](https://openaccess.thecvf.com/content/CVPR2024/html/Choi_vid-TLDR_Training_Free_Token_Merging_for_Light-weight_Video_Transformer_CVPR_2024_paper.html)；[官方代码](https://github.com/mlvlab/vid-TLDR) | 2024 | 直接已有视频 Transformer 的免训练 attention-saliency 背景 token 合并。 | 下游为视频理解/检索等，不验证异常检测或异常保持。 |
| Liang et al., [Not All Patches Are What You Need: Expediting ViTs via Token Reorganizations](https://openreview.net/forum?id=BjyvwnXXVn)；[官方代码](https://github.com/youweiliang/evit) | 2022 | 以 CLS attention 保留 attentive token、融合其余 token，并在训练中集成。 | 真实 CLS 是前提；不能无条件移植到无 CLS 或不同 attention 布局。 |
| Rao et al., [DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](https://openreview.net/forum?id=jB0Nlbwlybm)；[官方代码](https://github.com/raoyongming/DynamicViT) | 2021 | 轻量 importance predictor、分阶段 token pruning 和端到端训练是既有路线，论文报告实测吞吐。 | 是图像分类训练方案；不能以其数字说明当前冻结视频 encoder 的效率。 |
| Kim et al., [Token Fusion: Bridging the Gap between Token Pruning and Token Merging](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Token_Fusion_Bridging_the_Gap_Between_Token_Pruning_and_Token_WACV_2024_paper.pdf) | 2024 | pruning/merging 取舍及范数保持合并已被研究；普通平均 merge 可有分布偏移。 | 不提供视频或 VAD 的优越性结论。 |
| Dai et al., [No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection (LAVIDA)](https://arxiv.org/abs/2602.19248)；[官方代码](https://github.com/VitaminCreed/LAVIDA) | 2026 | 明确提出 reverse-attention token compression，以异常的时空稀疏性减少成本。 | 零样本 MLLM + pseudo-anomaly 协议不同，不能同表比较其精度/速度，也不能证明当前 encoder 中同一性质。 |
| Sultani et al., [Real-World Anomaly Detection in Surveillance Videos](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html) | 2018 | 视频级弱标签下将长视频视作 bag、片段视作 instance；论文指出异常在长未裁剪视频中通常只出现短时间。 | 视频标签不能赋予 clip/patch 真值；不提供 ViT token 压缩规则。 |
| Tian et al., [Weakly-Supervised VAD with Robust Temporal Feature Magnitude Learning (RTFM)](https://openaccess.thecvf.com/content/ICCV2021/html/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.html)；[官方代码](https://github.com/tianyu0207/RTFM) | 2021 | 高/低**训练后 snippet feature magnitude**作为 WSVAD 的显式可检验假设，且处理异常视频的 dominant negative instances。 | 不是冻结 ViT 内部 token 的观察事实；不支持将 token L2 norm 预设为异常分数。 |
| Wu et al., [Weakly Supervised VAD and Localization with Spatio-Temporal Prompts (STPrompt)](https://arxiv.org/abs/2408.05905) | 2024 | 异常可位于相对全帧小的空间区域，背景会误导全局特征；采用 patch/motion 与 VLM prompt 处理。 | 不支持“运动即异常”或一般 ViT token 分布方向；其 VLM/prompt/时序模块不可直接当本项目 reducer 的公平实现。 |

## 核查状态

已核对上述论文页面/论文 PDF 或作者链接的官方代码入口；未下载、运行或复现任何第三方实现。本文件没有产生或暗示本项目的实验数值。LAVIDA 为 2026-02 arXiv 预印本，尚应在投稿前复查其发表状态与最终版本；其余“未发现完全相同组合”的判断只覆盖本页十项近邻工作，置信度为中等。
