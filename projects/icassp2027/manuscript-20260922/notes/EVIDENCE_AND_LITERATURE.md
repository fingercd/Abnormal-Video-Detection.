# 证据与文献核查

访问日期：2026-09-22。正文15项引用；这里记录它们的用途、边界和一手来源。不是所有检索命中都当成可用证据，也没有作“首个/全部优于现有方法”的新颖性保证。

## A. 本项目证据

| 论断/数值 | 原始咨询包依据 | 本稿处理 |
|---|---|---|
| 六组 Dense/PairSelect质量与CI | `02_实验结果/quality_matrix_detail.csv`；18份原始JSON | 原CSV复制保存，脚本按单位转换 |
| 同预算固定/随机参照 | 同一CSV | 放小表；不声称范数普遍优胜 |
| XD非插值AP | CSV的`xd_average_precision_*` | 与梯形PR-AUC分开解释 |
| 6层后插入、分组、配额、pair优先级 | `03_压缩方法代码/token_reduction/token_selection.py`、`deployment.py` | 公式与代码一致 |
| 动态位置anchor近似 | `deployment.py`位置语义字段 | 不写精确位置保持 |
| Dense训练及冻结复用 | `04_协议与冻结决策/six-cell-method-freeze-v2.json`、`02_实验结果/urdmu-six-cell-matrix-v1.json` | 区分head原始训练与压缩免训练 |
| F04局部冗余、S7路线探索 | `04_协议与冻结决策/method-freeze-20260920.md`摘要 | 原效应量未提供，未编造 |
| 显存/计时范围 | `05_核验与审计/efficiency_protocol.md` | 当前仅计划，实测留空 |

输入归档及关键文件的SHA-256见 `SOURCE_MAP.json`。这份文件标记的是来源身份，不是要求未来优化必须保持二进制指纹相同。

## B. 正文引用的一手来源

1. **Sultani, Chen, Shah. Real-World Anomaly Detection in Surveillance Videos. CVPR 2018.** WSVAD、多实例排名和UCF-Crime背景。https://arxiv.org/abs/1801.04264
2. **Tian et al. Weakly-Supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning. ICCV 2021.** 异常实例被正常实例淹没的问题与特征幅度背景；它不证明本方法中间层token范数等同异常概率。https://arxiv.org/abs/2101.10030
3. **Zhou, Yu, Yang. Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection. AAAI 2023, 37(3):3769–3777.** 本项目下游head来源。https://ojs.aaai.org/index.php/AAAI/article/view/25489
4. **Tong et al. VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training. NeurIPS 2022.** 视频冗余与高比例掩码预训练；不是冻结推理同等删减比例的证明。https://arxiv.org/abs/2203.12602
5. **Fayyaz et al. Adaptive Token Sampling for Efficient Vision Transformers. ECCV 2022.** 参数自由采样的重要相关方法；信号结合注意力和value范数，不是本项目的原始block-output范数。https://arxiv.org/abs/2111.15667 ；公式核查：https://arxiv.org/html/2111.15667v3
6. **Bolya et al. Token Merging: Your ViT But Faster. ICLR 2023.** 无额外训练的merging及视频应用；不能写“现有压缩都需要训练”。https://arxiv.org/abs/2210.09461
7. **Wu et al. VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection. AAAI 2024, 38(6):6074–6082.** 冻结视觉语言表示与任务适配背景；本稿不声称复现它的质量。https://ojs.aaai.org/index.php/AAAI/article/view/28423
8. **Wu et al. Not Only Look, but Also Listen: Learning Multimodal Violence Detection under Weak Supervision. ECCV 2020:322–339.** XD-Violence来源；本项目只用视觉。https://arxiv.org/abs/2007.04687
9. **Wang et al. VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking. CVPR 2023.** 第二个VideoMAE系编码器与双掩码预训练背景。https://arxiv.org/abs/2303.16727
10. **Bertasius, Wang, Torresani. Is Space-Time Attention All You Need for Video Understanding? ICML 2021, PMLR 139:813–824.** 分解时空注意力。https://proceedings.mlr.press/v139/bertasius21a.html
11. **Rao et al. DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. NeurIPS 2021.** 可学习token重要性预测与层级缩减。https://arxiv.org/abs/2106.02034
12. **Liang et al. Not All Patches Are What You Need: Expediting Vision Transformers via Token Reorganizations. ICLR 2022.** EViT对注意/非注意token重组织；不是与本方法相同的纯gather。https://openreview.net/forum?id=BjyvwnXXVn_ ；官方作者仓库的引用核对：https://github.com/youweiliang/evit
13. **Wang et al. Efficient Video Transformers with Spatial-Temporal Token Selection. ECCV 2022:69–86.** STTS的视频选择相关工作，会议不是CVPR。https://arxiv.org/abs/2111.11591
14. **Liu et al. ESOM: Efficiently Understanding Streaming Video Anomalies with Open-world Dynamic Definitions. arXiv:2604.07772, 2026.** 开放世界流式多模态语言模型异常理解也使用token merging，故不能宽泛宣称首次将token缩减用于异常任务。正文按预印本引用，不虚构会议。https://arxiv.org/abs/2604.07772 ；https://arxiv.org/html/2604.07772v1
15. **Darcet et al. Vision Transformers Need Registers. ICLR 2024.** 高范数token可能出现在低信息背景，限制“范数就是异常显著性”的论断。https://arxiv.org/abs/2309.16588

各条只用于上述具体论点。预训练、高层video-LLM输入压缩、内部视觉encoder缩减和下游head适配不是同一设置。没有把某篇工作的标题或摘要当作本方法效果证据。

## C. 补充检索、暂不进入主文的近作

**ForestPrune（视频MLLM版本，2026）**：作者提出时空forest驱动的training-free剪枝。研究对象是video-MLLM，不直接等同本文的冻结UR-DMU设置。不要与同名的决策树ensemble剪枝混淆。https://arxiv.org/abs/2603.22911

**Adaptive Two-Stage Visual Token Pruning for Efficient Inference in Video-Language Models（2026-08）**：同时涉及冗余帧删除与保留帧内token删减，提示不能把post-hoc/training-free当作独占新颖性；删除输入帧与本文保持原clip schedule的约束不同。https://arxiv.org/abs/2608.03112

这些补充目前主要核对了摘要/方法概述，未据其数字制作比较表。广泛检索也出现了生成模型、事件相机、视频问答和KV cache等不同任务；没有为了增加引用数量而塞进正文。

## D. 公式与指标实现来源

TimeSformer主要block运算量按作者公开实现中的temporal attention、spatial attention、`temporal_fc`、MLP与CLS路径推导：
https://raw.githubusercontent.com/facebookresearch/TimeSformer/main/timesformer/models/vit.py

Joint block：`12*N*D^2 + 2*N^2*D` MAC。
Divided block：`(17*P*T+4*T+8)*D^2 + 2*D*(P*T^2+T*(P+1)^2)` MAC。
全12个主要block为6层Dense＋6层Reduced。未计入embedding、normalization、readout、selector和索引搬运；排序/搬运也不能简单用FLOPs反映。

AP与梯形PR-AUC区别依据scikit-learn官方定义核对：
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

计时的异步、预热及可重复测量依据PyTorch官方基准说明核对：
https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html

这些在线文档的当前版本可能与服务器安装版本不同。它们用于确认概念与计时原则，不代表已核实用户服务器版本或运行速度。
