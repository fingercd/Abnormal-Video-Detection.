# 独立批判性核验：LAVIDA、vid-TLDR 与 ToMe

核验日期：2026-09-18。范围：对 `literature-baselines.md` 中三条决定论文定位的引用作独立复查；只查看原论文/正式出版页、arXiv 元数据和作者声明为官方的代码仓库，不运行第三方代码，也不产生本项目实验数值。这里的“支持”只指来源是否支持该文献描述，不能外推为对当前四个 encoder 的实验结论。

代码快照用于可追溯性，而非声称它们是论文发行版：LAVIDA `main` 为 `56b30589ea2c`（2026-02-25，无 tag/release）；vid-TLDR `main` 为 `4a897230c123`（2025-10-21 的 README 更新，无 tag）；ToMe `main` 为 `af95e4b1befa`（2023-03-31，仓库后来已归档）。LAVIDA 的论文版本以 arXiv v4 为准。

LAVIDA 的代码级复查定位在 [`model/LLM.py`](https://github.com/VitaminCreed/LAVIDA/blob/56b30589ea2c8a9e28e812bcd0f1282c0243b187/model/LLM.py) 的 `temporal_token_reduction`、`visual_token_sampling` 与 `token_reduction`，以及 [`train/train_para.yaml`](https://github.com/VitaminCreed/LAVIDA/blob/56b30589ea2c8a9e28e812bcd0f1282c0243b187/train/train_para.yaml)。vid-TLDR 的可执行注意力—合并插入点在其 [ViT block 实现](https://github.com/mlvlab/vid-TLDR/blob/4a897230c1231eab4a09ba631fe471df0e18114f/multi_modality/models/backbones/vit/vit.py)；ToMe 的官方 [README/benchmark 入口](https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/README.md) 固定了其 patch 支持范围。

## 裁决

| 主张 | 裁决 | 对本项目的含义 |
|---|---|---|
| ToMe 是已有、可免训练的 ViT token merging 基线，并有视频效率—质量评测 | **支持** | 不能以“训练自由 token merging”或“视频 token merging”作为新颖性主张。它不是 VAD 方法，也没有覆盖当前四个 bridge。 |
| vid-TLDR 是已有的、attention-saliency 驱动的视频 Transformer 训练自由 token merging | **支持** | 是最接近的通用视频合并概念基线；只能在当前模型真实可得 attention 的条件下重实现/比较。 |
| LAVIDA 确为 VAD 中的 reverse-attention token compression，而非同名别领域模型 | **支持（但须严格收窄）** | 它封顶“VAD + 反向注意力压缩”的宽泛新颖性说法；不能把它当作与冻结 clip ViT + 弱监督 MIL 的同设置性能或速度基线。 |
| LAVIDA 证明异常在当前四个冻结 ViT 中具有可普适利用的、区别于正常的内部 token 性质 | **已反驳** | 论文将时空稀疏性作为其 MLLM/VAD 设计前提；没有正常—异常配对的四个 encoder 内部统计，也没有证明其压缩分数跨架构、场景或 MIL readout 不变。该性质仍必须先由本项目的匹配 probe 验证。 |

## 一手证据逐项核验

| 文献与版本 | 输入 encoder、任务与训练 | 压缩信号和位置 | 质量/效率的实际评测口径 | 与本项目的实质关系 |
|---|---|---|---|---|
| Bolya et al., [Token Merging: Your ViT but Faster（ICLR 2023）](https://openreview.net/forum?id=JroZRaRw7Eu)，[官方 ToMe 代码](https://github.com/facebookresearch/ToMe) | 既有 ViT；论文涵盖图像、视频和音频任务。可直接应用于预训练模型，也可在训练时使用。官方仓库仅提供对 `timm`、SWAG、MAE 实现的 patch，不含当前四个 encoder 的适配。 | 在 Transformer 内按 token 相似度做双边匹配并合并；不是异常分数、reverse attention 或 VAD selector。 | 论文摘要明确比较吞吐与下游精度，并含视频 ViT 的吞吐—准确度折衷；官方 benchmark 示例报告固定硬件上的 images/s。没有 UCF/XD 的 frame 指标、MIL head 或当前 backbone 的净延迟。 | 是通用、强制性的同预算 merge 对照。它说明“相似/冗余”本身不是异常特有线索。该仓库已于 2025-01-01 归档，最后公开提交在 2023-03；无 release tag，引用应以 ICLR 论文为准。 |
| Choi et al., [vid-TLDR（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Choi_vid-TLDR_Training_Free_Token_Merging_for_Light-weight_Video_Transformer_CVPR_2024_paper.html)，[官方代码](https://github.com/mlvlab/vid-TLDR) | UMT 的视频/视频—文本 Transformer，用于视频—文本检索与视频问答等视频理解任务。README 明确称全部实验直接使用公开 UMT checkpoint、无额外训练。它不是 TimeSformer、VideoMAE/VideoMAEv2 或 V-JEPA 2，也不是 WSVAD/MIL。 | block 先返回 attention；代码用 attention map 得到 salient region，背景 token 作为 merge/drop 对象，再执行 saliency-aware merging。每层压缩数 `vidTLDR_r` 是显式配置。 | 正式论文比较任务质量与计算量；作者 README 提供多数据集检索/VQA 的 evaluation 命令和逐层 token reduction 配置。已核对的材料没有让其 VAD 指标、MIL readout 或当前模型的端到端延迟可直接比较。 | 对“只由 attention 区分背景并训练自由地合并视频 token”构成直接先例。若完整 attention 的提取改变了当前 encoder 的实现/融合路径，其实际成本必须纳入测量，不能移植其计算量结论。仓库无 tag；当前 README 最近更新晚于论文，文献身份以 CVPR 2024 论文固定。 |
| Dai et al., [No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection（LAVIDA，arXiv:2602.19248v4）](https://arxiv.org/abs/2602.19248v4)，[官方代码](https://github.com/VitaminCreed/LAVIDA) | 是端到端零样本 VAD，不是同名 LaViLa/视频表征模型。arXiv 元数据：首发 2026-02-22，v4 更新 2026-03-23，并标注“Accepted by CVPR 2026”；本次未找到可替代该版本的正式 proceedings 页。代码采用 Qwen2-VL、SAM2、CLIP/Q-Former 等 MLLM—分割管线，带语言 token 和密集 frame/pixel 输出；与冻结 clip encoder 后接视频级 MIL 完全不同。完整系统**要训练**：Anomaly Exposure Sampler 用分割数据构造 pseudo-anomaly，配置含 LoRA、掩码/分类损失及训练命令；“without VAD data”不是“完全不训练”。 | 论文摘要明确写 `reverse attention` token compression。官方代码进一步显示两处无梯度缩减：其一在 Qwen visual blocks 前，按相对首帧的余弦变化删除后续静态区域；其二对视觉/视频 language tokens 计算全对全距离、以局部密度选中心，再用负的中心—token 点积做加权聚合。这是“与高密度中心相反的相似度”机制，不等价于当前 ViT 任一 attention head 的 reverse attention，也不是由正常/异常标签直接计算。 | 论文摘要的质量终点是多个 VAD 基准的 frame-level 与 pixel-level 检测；它只声称压缩降低计算成本。官方 README 仍写数据准备和使用说明待补充，未提供可复现的统一 latency/throughput benchmark；代码的压缩还包含全对全 `cdist`、top-k、聚合及 padding。因此不能把其质量或“降低成本”表述转换为当前四个冻结 encoder 的实测净加速/显存结论。 | **概念重合**：在 VAD 中利用当前视频表示的背景/稀疏性压缩 token，目标兼顾质量和成本。**决定性差异**：它压缩的是 Qwen2-VL 的视觉及语言输入序列，并耦合 pseudo-anomaly、语义文本、SAM2 分割和端到端训练；本项目在固定 `videomaev2`、`timesformer`、`vjepa2`、`videomae` 内探索内部 token，之后由视频级弱标签 MIL 读出。故可作规则型概念对照和新颖性边界，不能直接抄实现、数字或称为公平同表 baseline。官方仓库没有 release/tag；当前 `main` 的可追溯代码提交来自论文发布期，版本固定应记录 commit SHA，而非声称有发行版。 |

## 对 LAVIDA “反向注意力”表述的精确结论

这个名称没有混淆：原论文摘要直接称其为基于 reverse attention 的 token compression，官方仓库亦自称该论文的 official open-source implementation。因此，“VAD 从未使用反向注意力进行 token compression”不可成立。

但它不能支持更强的转述。代码里的主压缩分数不是“异常 token 的 ViT self-attention 反向值”：先以当前样本的嵌入局部密度选中心，再对中心与 token 的负相似度做聚合；时间缩减则按静态性做硬删除。两者都没有比较正常视频和异常视频的 token 分布，也没有使用当前项目的四类 encoder 或 MIL bag 得分。这一点把可用的论证限定为“已有 VAD 特异的背景/稀疏性 token compression 思路”，不能限定为“已证实异常具有某个通用内部统计”。

## 反证与必须保留的范围限制

ToMe 的原论文报告：不经训练即可依据相似性合并 token，且定性观察到视频中对象部件可以跨帧合并。这是对“冗余或可合并 token 是异常特有性质”的直接反例：冗余同样存在于前景对象和非 VAD 视频任务。vid-TLDR 也只把 attention-salient 与背景相关联，并未把低注意力或高冗余验证为异常/正常的稳定判别。

更具体地，LAVIDA 自身的静态区域规则预设了异常相对背景的时空局部性；全屏事件、相机抖动、光照突变、拥挤运动或占据大区域的异常都超出这个前提。它的 pseudo-anomaly 训练分布也不是自然 WSVAD 中的真实异常分布。因而本项目的初步主张最多只能是待检验的条件假设：**在指定数据、场景/运动匹配、指定层与 encoder 中，某个在线可得的内部性质可能在同预算下有用。** 不能写成异常天然低冗余、必然偏离背景中心、必然高/低 attention，或把视频级标签传播为 patch 真值。

## 可执行的论文边界

- 禁止主张：首个 training-free video token merging、首个 attention-guided 视频 token 选择、首个 VAD reverse-attention compression，或“异常稀疏因此应压缩”的一般规则。
- 只有在预注册的正常—异常匹配 probe、同预算 ToMe/vid-TLDR/随机/均匀对照、以及四个实际 bridge 的净计时和显存测量均成立后，才能提出更窄的结论：所验证的规则在本项目的冻结 encoder + WSVAD/MIL 协议中有效。
- LAVIDA 的直接比较层级应标为“VAD 特异概念/规则对照”；若实现其简化版本，必须只使用当前层在线状态、重算到同一 token 预算，并把其全对全分数计算计入成本。它的 pseudo-anomaly、MLLM、SAM2 和像素监督不能进入本项目的弱监督插件比较。

## 核验限度

三条文献均真实存在，且官方仓库均可访问；ToMe、vid-TLDR 的会议版本可由正式页面确认。LAVIDA 的可核验正式记录是 arXiv v4 及其“CVPR 2026 accepted”元数据；其作者仓库尚无 release/tag 且 README 留有复现说明缺口。因此，LAVIDA 的出版状态、精确硬件/时间测量和代码—论文逐行一致性应在投稿前随最终 proceedings 再查一次。本文不把该缺口错误表述为论文不存在，也不把作者的成本主张升级为对本项目有效的速度结论。
