# F01｜中层 attention 相对更新：模型内复现，跨模型确认未通过

状态：VideoMAEv2 `confirmed`（限定本数据/输入/层）；四 encoder 共性候选 `rejected`，
指本轮预设确认门禁未通过，不是证明零效应或模型等价。
四个模型均已完成相同独立64视频的512个clip确认；TimeSformer、VideoMAE和V-JEPA2未通过
固定门禁。confirmed 不代表 actionable 或 deployable。

问题：正常与含异常的训练视频，其 attention 分支相对即时输入的更新强度是否不同？
只使用官方训练视频级标签，W路径，无事件或空间真值。完整fit探索为64正常/64正视频，
独立确认固定32正常/32正视频，每视频8窗口；统计以视频为独立单位。

探针为有效patch上的 `||U_attention||_F / ||X_pre-attention||_F`，先算每clip再等权视频均值。
固定相对深度0.5、方向正视频更高。12-block骨干用index5，V-JEPA2用index11。
TimeSformer对应spatial分支及其temporal更新后的即时输入，不能把原始比值与联合attention
骨干直接相减。精确签名和拒绝条件见上级目录的property-confirmation-v1.json。

| Encoder | 独立确认Hedges g及95% CI | 运动×亮度条件均值差及95% CI | 预冻结门禁 |
|---|---|---|---|
| VideoMAEv2 | 0.780 [0.327, 1.282] | 正；[0.002124, 0.008482] | 通过 |
| TimeSformer | 0.001 [-0.494, 0.499] | CI [-0.010270, 0.019793] | 未通过 |
| VideoMAE | 0.166 [-0.317, 0.677] | -0.000037 [-0.003348, 0.003097] | 未通过 |
| V-JEPA2 | 0.292 [-0.190, 0.808] | 0.000705 [-0.000708, 0.002128] | 未通过 |

TimeSformer点估计接近0，但区间仍宽；这里是未复现，不是等价性检验证明完全没有差异。
正常参考的分箱只用fit正常视频校准；确认只apply，分箱内按标签重抽视频，固定min组数权重。
scene/source仍unknown。探索的ratio差异部分伴随输入范数变化：分子范数单独CI跨0；
输入范数、运动、亮度、时长敏感性模型保留条件关联，不能据此声称独立异常因果机制。

可部署信号在attention执行后即可计算，成本为token维度范数/归约。它不能节省已经执行的
该分支。候选操作仅为下一步中立验证：同预算uniform、随机、高score/低score保留；
视频级ratio差异尚未证明逐token ratio能定位异常或指导最优保留。代理开销、检测质量、
净延迟和显存收益均未确认，不因本卡预先建立最终插件。

运行与导出：探索两批 `probe-20260917T193541117630Z-ef589e03` 加原唯一缺项Normal533补批，
gate和敏感性在 `outputs/icassp2027/analysis/explore-gpu-v1-20260917T204307Z/`。
确认前两模型导出在 `outputs/icassp2027/analysis/confirm-v3-20260917T213045Z/`，VideoMAE在
`outputs/icassp2027/analysis/confirm-v3-videomae-20260917T214000Z/`。每份receipt绑定输入与
经27项完整规模合成门禁测试的v3统计实现。没有修改确认方向、层或统计量来挽救主张。
V-JEPA2在 `outputs/icassp2027/analysis/confirm-v3-vjepa2-20260917T234328Z/`；
四模型只读合并在 `outputs/icassp2027/analysis/confirm-four-v3-20260917T234419Z/`，核验
共同的候选、cohort、fit校准及统计脚本摘要，独立视频数仍为64，不乘以encoder数量。

后续fit26中立操作的前三个模型结果均不支持把逐token相对更新分数直接用于保留排序：
在局部pair相同覆盖预算下，high/low均未优于paired random的pooled扰动；正常/正视频差中
之差区间跨0。这是表示扰动诊断的负结果，不能写为异常定位有效或检测质量结论。

最近工作边界：ToMe、vid-TLDR和LAVIDA已包含通用/视频/异常检测token处理思想，不能把
更新或attention优先级的简单变体直接当作首次。后续若只能得到通用压缩收益，应收缩异常特异主张。
