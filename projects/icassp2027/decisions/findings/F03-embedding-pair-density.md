# F03｜Embedding 层采样 token 对的高相似比例未通过确认

状态：`rejected_or_inconclusive`。这表示预登记的 VideoMAEv2 模型内正向确认门槛未通过，
**不是**证明效应为零或两个视频组等价；P04 token 操作停止，不把它作为 LoRA、局部合并或其他插件的依据。

研究问题是：在 VideoMAEv2 的 `embedding.output`，含异常训练视频是否具有更高的高相似
token 对比例。精确定义为：collector 固定时空采样的至多256个非零 token 中，全部非对角 token 对的
余弦相似度不低于 0.90 的比例。它不是冗余 token 的比例、局部邻接相似度，也不表示局部
token 可以安全合并。

独立确认使用此前未触及的官方 UCF-Crime train confirm 视频：32 正常、32 含异常，每视频
8 个中心窗口，统计时先在视频内等权平均 8 个 clip，再以视频为独立单位。结果导出为
`outputs/icassp2027/analysis/p04-heldout-confirm-v1-20260918-a02/analysis.json`，SHA-256
`e71abbbe51b847ca5b01b23be5760fe17d482010e21e6f676c771ed81dc65913`；receipt SHA-256 为
`ed2840ca72b1f784c74a7382d084caa5a228adbafc65d357b31312f0b8a53593`。

| 比较 | 视频数 | 点估计 | 预登记 95% CI | 结论 |
|---|---:|---:|---:|---|
| 原始：正视频−正常视频 | 32 / 32 | +0.0060656 | [−0.038622, +0.052196] | 正向区间不排除 0 |
| 原始 Hedges g | 32 / 32 | +0.0630 | [−0.437, +0.537] | 效应方向不稳定 |
| motion×brightness matched | 32 正常 / 29 正视频 | −0.0036728 | [−0.037680, +0.033041] | 条件比较反向且区间跨 0 |

matched 使用 fit-normal 冻结的三分位阈值；8 个完整 bin 按两组中较小的样本数确定固定权重，
这些较小样本数合计为23。一个
`low×mid` bin 没有正常视频，因此 3 个正视频未进入 matched 比较；覆盖仍达到预登记的每标签
至少 16 个视频门槛。它不授权重新拟合阈值、替换层、阈值、统计量或样本。

类别表只作描述，不能用于重新选择性质。10 个含异常类别中，Abuse、Arrest、Assault 相对
正常均值为正，而 Burglary、Fighting、RoadAccidents、Robbery、Shooting、Stealing、Vandalism
为负；其中部分样本仅 1–5 个视频，不能作类别显著性或机制结论。新 confirm ID 与 explore128、
旧 confirm64、neutral26 不重叠，但 source/scene 仍 unknown；ID 不重叠不证明来源独立。

观察中没有事件、空间 patch 或测试标签。即使原始点估计为正，P04 也只测量采样 token 全对的
高相似密度，不能定位可合并位置，不能推出均值合并、水平 pair 合并或实际后缀计算共享安全。
本轮没有运行 P04 token reducer、LoRA 训练、检测质量或效率比较。后续方法若继续，必须从另一条
新预登记性质开始，并重新经过观察、独立确认与同预算操作门槛。
