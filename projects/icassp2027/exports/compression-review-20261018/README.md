# 冻结压缩实验公开导出（2026-10-18）

本目录保留三个最终报告目录的原字节副本。完整结果：VideoMAEv2 38/36、CLIP-DSANet 38/36、VideoMAE+TimeSformer 52/48（提取预测条件/压缩质量比较），合计128/120，含8个dense。不是新训练或新实验。

各子目录原artifact-sha256.json可校验原报告。源运行版本由报告内科学身份与原source SHA定义，不因本次Git归档而改写。私有原始运行日志及聊天不在公开提交中。

比较时必须区分post-attention（V2/CLIP）和pre-block（VMA/TS）；CLIP post12保留CLS是结构对照。XD DSANet主指标step AP，其余主指标梯形PR-AUC；不可混用。VMA/TS uniform已由用户取消，历史已完成uniform效率保留。V2净效率未测，不能补零。

本次只归档已完成结果、代码和相关测试。单head条件的实测事实可支持描述性规律；不据此声称跨seed稳定或方法间直接显著性。请完整保留负结果。
