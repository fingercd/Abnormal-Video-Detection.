# 局部覆盖候选的近作边界核验

核验日期：2026-09-20；来源为论文arXiv页面/全文和CVF原文。结论：局部代表覆盖与medoid选择不能单独充当本论文创新点；当前probe继续作为中立诊断和潜在对照，不因为实现成功而升级为新方法。

| 原始论文 | 核验到的相关内容 | 对当前研究的约束 |
|---|---|---|
| [CenterCLIP，SIGIR 2022](https://arxiv.org/abs/2205.00823) | CLIP视频token分段聚类并选择确定性medoid | 不声称首次在CLIP中以真实代表token压缩 |
| [vid-TLDR，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Choi_vid-TLDR_Training_Free_Token_Merging_for_Light-weight_Video_Transformer_CVPR_2024_paper.pdf) | 用attention分布锐度引导早层视频token合并 | attention/早层/免训练组合本身不是新贡献 |
| [LAVIDA，2026-02](https://arxiv.org/abs/2602.19248) | 视频异常检测内已有reverse-attention token压缩 | 不声称首次为VAD压缩token；需细分视觉encoder内与后端的作用位置 |
| [KeepAD，2026-08 v3](https://arxiv.org/html/2608.03681v3) | 异常检测ViT内已有2×2覆盖保底、额外选择、深层原型和训练适应；§3.2为每块保留一个按学习分数排序的成员 | 当前2×2选两个局部medoid并非该文相同算子，但局部覆盖保异常这一上位故事已有直接先例 |
| [S²Prune，2026-09](https://arxiv.org/abs/2609.01224) | 空间区域保底覆盖、局部结构预算分配与代表选择 | 不能靠再加一个空间保底或局部自适应预算来声称独创 |

这些研究的任务、监督和作用阶段不同，不能据此断言其已解决本项目的WSVAD两数据集问题；也不能把任务换成WSVAD就自动视作机制创新。这里未使用任何论文成绩挑模型、层或预算。

当前执行调整：保留小规模只读coverage probe，观察尺度、覆盖与后缀扰动的关系；不启动以该通用算子为论文主方法的全量提取。未来主方法必须从独立确认的正常—含异常性质及受控干预产生具体、可验证的决策差异。没有这样的证据就保持study/边界结果身份，不临时堆叠模块绕开近作。
