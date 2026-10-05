# P04 四 encoder 早期局部相似性：独立视频确认

日期：2026-09-19。机器可读合同及最终冻结状态以同名 JSON 为准。
本设计接续 F04 的完整四模型 FIT128 观察，不改变旧 F01/F02/F03 的候选、样本、结论或角色锁。

## 单一主要问题

在每个模型相对深度 0.25 的 block 输入处，含异常视频的同原生时间索引局部 token
余弦相似性均值，是否高于正常视频？固定 P04 `same_frame_local_cosine_mean`，
12-block 的 VideoMAEv2、TimeSformer、VideoMAE 使用 `block.2.input`，24-block 的
V-JEPA 2 使用 `block.5.input`；`sublayer_kind=input`、`head_id=null`。
索引为零起始，映射公式为 `ceil(relative_depth * block_count) - 1`。

选择依据是已完成 FIT128 的四模型同方向探索结果。effective rank 与 local−nonlocal
没有四模型同方向区间支持，因此不列为本次候选，也不在确认后换成其它站点。
`coverage_k4` 与 local cosine 相关，本次不增设第二个确认候选；它保留为 F04 的探索描述。

这是视频级观测问题。正视频的全部片段、patch 不因视频标签而成为异常样本；局部余弦
升高不能推出异常 token 的位置、重要性、因果或可压缩性。通过确认后仍须另行登记同预算
中立干预，不能直接锁定压缩算子、预算或 LoRA。

## 固定样本、采样和校准

使用原 UCF official-train 角色锁中剩余且未被旧 F01/F02 或 F03 确认池消耗/保留的
全部 33 个 confirm 视频：16 正常、17 含异常。ID 集合、manifest、cohort、来源哈希及
与 FIT/旧确认池的互斥关系在 JSON 和 metadata-only 回执中绑定。不得补入 select、fit
或旧 confirm 视频。source/scene/near-duplicate 来源信息未知；ID 互斥不等于镜头来源独立。

每视频固定 8 个均匀中心逻辑窗口，frame stride=2，各自沿用原生输入 16/8/64/16 帧。
使用已验证的完整窗口采样及真实 `(t,h,w)`；边界 clamp 复用如实记录，以 8 个逻辑窗口的
等权均值为视频统计量，不将重复窗口或 token 当成独立样本。确认采样身份必须与各模型
FIT 校准一致；短视频或缺失导致身份无法满足时，不能静默改变采样或删除样本。

主要控制固定为 motion × brightness 两个三分位字段。各模型分别使用其已完成 FIT128
中 64 个正常视频拟合的校准文件，不重新拟合，也不根据 confirm 覆盖合并 bins。
原 FIT controls SHA、校准 SHA、normal IDs 与 input_sampling_id 分别封存；新 confirm
controls 另有自己的 SHA。两个 controls 文件的内容与 SHA 本来就应不同。
不追加第三个控制字段来改变本次主要决策。

## 先看覆盖，再作方向判定

在任何 confirm 目标统计读取前冻结下列新设计门槛。它们用于最低覆盖和可解释性，
不是功效保证、已有旧协议要求或从 confirm 结果反推的阈值：

| 每个 encoder 的覆盖条件 | 固定门槛 |
|---|---:|
| 完整匹配分层内保留的正常视频 | 至少 12 |
| 完整匹配分层内保留的含异常视频 | 至少 12 |
| 各完整分层 `min(n_normal,n_positive)` 之和 | 至少 10 |
| 同时含两标签的完整分层数 | 至少 2 |
| 两标签均至少 2 个视频的分层所占固定质量比例 | 至少 0.80 |

12/label 保留原池中大部分视频；质量 10 要求相对于理论上限 16 有实际重叠；
至少两个完整分层避免仅一个条件支撑结果；0.80 限制 singleton 分层对估计的支配。
这些选择不能修复小样本或来源相关性。singleton 保留在原 estimand 中，报告其质量比例；
不得为取得较窄或较好的区间事后删除。

先执行 controls-only 覆盖预检，不加载 encoder 或读取 P04 目标。任一模型覆盖不足，
四模型主确认记为 `coverage_insufficient`，不启动新的 P04 GPU 确认；保留全部四模型
覆盖和原因。该停止规则使不足覆盖不会诱发更换层、指标、样本、控制、阈值或权重。
controls-only 本身也记入此次 confirm 使用记录，不能后来再把这 33 个视频称为全未使用。

覆盖全部满足后，运行原模型/只读观察/identity 的等价性和真实几何检查，再收集主要
精确签名。每模型必须完整 33 视频 × 8 窗口；缺失、非有限、身份或等价性失败属于
`measurement_incomplete`，不能用剩下的成功样本确认。

## 冻结估计量和不确定性

每模型主要估计量为完整 motion×brightness 分层内 positive-minus-normal 视频均值差，
以各分层 `min(n_normal,n_positive)` 作为固定归一化权重。按完整分层与标签分别对视频
有放回抽样，10,000 次 bootstrap，报告双侧 percentile 95% CI。阈值、完整分层、
每标签样本数和权重在重采样中固定；不将其解释为包括重新拟合分箱不确定性的区间。
按同名 JSON 的 seed 和冻结实现执行；raw Hedges g 与其区间为伴随描述，不替代主估计量。

只有四模型覆盖、身份、完整性均通过，且每模型预定 matched 差为正、95% CI 下界均
严格大于 0，才记为 `confirmed_at_frozen_scope`。四模型共享这 33 个视频，不是四份
独立重复，四个边际区间也不是联合 95% 区间。零跨区间记为 `not_confirmed` 并说明
精度限制，不能推断无差异；全部负结果与缺失理由保留。

本次确认不训练或评分 UR-DMU，不访问官方 test，不选择最终压缩机制。四模型通用机制
仍须由后续允许的中立干预和质量—效率比较支持。总体研究仍未完成。
