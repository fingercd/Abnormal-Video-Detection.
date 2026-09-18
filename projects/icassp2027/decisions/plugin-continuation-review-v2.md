# 现有 encoder 插件的继续验证审查

日期：2026-09-18。审查对象为当前固定四encoder和新UR-DMU后端；没有修改插件、层、
预算或校准资产。结论：**可以继续作为候选和对照进入冻结后端验证；目前不足以宣称已形成
异常特异的新方法、检测质量非劣或峰值显存改善。**

## 工程契约

| 检查 | 已核实的实现/证据 | 判断 |
|---|---|---|
| 插入位置 | `paper/reduction_setup.py:170`先观察真实geometry，再固定原生网络中间深度；桥输出被实际gather/merge | 插件位于encoder内部，不是输出后删feature |
| 后续序列 | `bridges/indexed.py`逐后缀层记录shape，缺任一层或VJ原位置注入即失败；四模型V100已有实际执行回执 | 可以复用；本轮另验接UR-DMU后的完整路径 |
| 时序与输出接口 | 每个原生clip仍输出一个D768/1024 pooled向量，dense输入窗口规则保持；新head只接该序列 | 结构兼容；不能据此推断分数不变 |
| TimeSformer | 完整空间轨迹保留，真实CLS不删；gate对轨迹内logits做平均并共享pair权重 | 没有用普通全局ViT reshape强行替代分离时空结构 |
| V-JEPA 2 | 后缀逐层接收原token位置索引；无人工CLS；原生RoPE算法不变 | 工程路线已有支持，不能把位置锚点近似说成精确合并位置 |
| 数据权限 | `PairMergeDeployment.__call__`只用batch形状/有效性，策略不读ID、类别、源帧绝对编号或标签；实际encode前使用clean batch | 部署不依赖标签或测试汇总统计 |
| 可训练参数 | `PairLinearGate`只有D维向量；原encoder冻结，仅gate进入optimizer，教师detach；目标为pooled相对MSE | 是表示保持校准，不是新UR-DMU任务监督训练 |
| 部署教师与状态 | 部署私有复制gate并eval/requires_grad(false)，只用当前block隐状态；无在线完整教师 | 不存在通过线上dense教师隐藏计算成本的设计 |
| 原子失败 | 不支持的batch/layout/padding显式拒绝，hooks退出移除，不回退dense后宣称缩短 | 应保留原失败处理 |

`PairMergeSpec.mass`保存来源质量与几何元数据。它不代表后缀attention或所有非线性层
严格守恒，也不证明压缩后表示等于原表示。当前保留原生adapter读出算法；尤其包含CLS
的mean pooling，其相对权重会受序列长度影响，质量影响必须通过实测回答。

## 可以复用的资产

- 四encoder现有权重、native运行环境、真实geometry和bridge，不需要为换任务后端重建。
- 固定UCF fit128/256clips的gate校准；每encoder768/1024参数、768步，非零梯度、
  参数变化和重载回执保留。它没有依赖TopKMIL，因此换UR-DMU不会在逻辑上使校准失效。
- 独立confirm的性质图、fit26中立操作及全部反例；它们不变成新后端质量结论。
- V100纯模型/adapter/clip参考计时和真实缩短回执；A100正式性能另测。

旧fit26随机对照使用逐clip种子，当前部署`paired_random`是跨clip固定mask。
两者不能沿用为同一数值结果；当前版本需要独立完成正式配对评测。

## 继续前与论文前的门槛

1. **连接门槛：**真实原视频、四native encoder、同一dense UR-DMU检查点下验证identity、
   uniform、paired_random、pair_linear；输出维数、片段数、位置回执、有限值和后端状态
   不变。两步工程head训练只检验梯度/更新/重载，不评价异常检测能力。
2. **质量门槛：**完整官方训练的dense后端固定后，才计算官方测试的直接插入差值和视频级
   CI，容忍度0.005。原pooled MSE和合成测试不能代替该结果。
3. **贡献门槛：**跨模型通用性质当前未确认，P16仅在VideoMAEv2成立；静态随机保留、
   局部两两合并、线性打分都不足以单独支撑新颖性。至少完成最相关ToMe/vid-TLDR等
   同预算方法的可适用比较与受控消融，再决定是否有可辩护的方法贡献。
4. **效率门槛：**真实后续token减半已证实；V100四模型96配置/2880sample的参考审计
   保留4个负端到端加速配置，72个非dense配置均未降低peak allocated。需要A100独占
   条件下重新测完整detector，不能主张显存改善或把共享作业walltime当插件速度。

## 当前决定

四组真实A100连接验收已经通过：两完整fit视频在每encoder的dense和三缩减路径都成功
提取，dense UR-DMU实际两步训练后严格重载，各路径使用同一冻结head且参数不变。
共348次clip提取仍只有两个独立视频，不作为检测质量结论。
[实际汇总](../../../outputs/icassp2027/control/native-urdmu-gpu-acceptance-20260918/summary.json)。
原插件/部署/梯度/位置回归103项通过，未因此修改策略、预算或校准。

不因换检测后端重写现插件，也不把它们提前命名为最终“ours”。继续完成新后端下的
工程验收和正式质量，保留global_uniform与paired_random作为必要对照。若现有候选
没有优势，结论如实保留；后续方法修订只能回到允许的开发数据，另行记录冻结版本，
不能用官方测试分数挑层或预算。

关联证据：[算法原冻结记录](reducer-evaluation-freeze-v1.json)、
[TARGET校准记录](pair-gate-calibration-v1.md)、
[独立性质确认](property-confirmation-v1.md)、
[V100审计](../../../outputs/icassp2027/analysis/formal-timing-audit-20260918/AUDIT.md)、
[新后端协议](official-detector-protocol-v2.md)。本审查未新增真实检测质量数字。
