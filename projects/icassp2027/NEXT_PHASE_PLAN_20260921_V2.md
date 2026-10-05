# 下一阶段详细计划 V2：VideoMAEv2 主验证与 64N+64A 压缩筛选

> **2026-09-21 训练范围修正：** 本文件中的 64N+64A 只用于快速筛选压缩规则。任何正式训练、
> 正式 dense baseline、正式 refit head 和正式压缩比较都必须使用对应 encoder×dataset 的全部训练集。
> 详见 [`TRAINING_SCOPE_AND_RUNTIME_ESTIMATE_20260921.md`](TRAINING_SCOPE_AND_RUNTIME_ESTIMATE_20260921.md)。
> 该规则覆盖本文中把 screen600 描述为“完整 training-side run”的旧表述；screen600 结果保留为筛选证据。

**版本：** 2026-09-21（Asia/Hong_Kong）  
**状态：** 用户已确认执行方向，等待 KimiCode 按本文件接管。  
**主目标：** 在两个数据集上分别训练一个 VideoMAEv2 检测 head，用两个数据集合计 64 个正常视频和 64 个异常视频，比较一个主要压缩方法和一个低风险备选方法。  
**两小时边界：** 两小时内完成方法实现、工程门、固定小规模筛选和可复核回执；不把两小时小筛选冒充完整论文实验或跨数据集泛化证明。

## 0. 用户已经确定的决策

以下内容已经确定，KimiCode 不得自行改回旧计划：

1. **不用共享 head。** UCF 和 XD 各训练一个 head。VideoMAEv2 是当前主验证对象；不在本轮两小时任务里为 VideoMAE、TimeSformer 或 CLIP 同时实现新方法。
2. **压缩筛选池总量固定为 128 个视频：** 两个数据集合计 64 个 normal +64 个 anomalous。默认均衡分配为：UCF 32 normal +32 anomalous，XD 32 normal +32 anomalous。
3. **不以 fit/select 的方法迁移为目标。** 本轮使用新的实验身份 idea_screen_64x64，只做固定视频池上的训练侧压缩筛选。
4. **selector 不区分正常和异常。** 压缩决策只读当前 VideoMAEv2 的 token、attention/隐藏状态和已核验坐标，不读标签、文件名、测试统计或异常类别。检测 head 仍然使用视频级 normal/anomalous 标签完成训练。
5. **主方法和备选方法必须在相同预算下比较。** 默认主压缩率为 keep_ratio=0.60；dense、uniform、主方法、备选方法都使用相同实际 token 预算。
6. **效果优先，但效果必须包含真实净加速。** 只提高训练分数而没有实际变短或端到端变快，不算成功；只变快而检测输出完全失真，也不算成功。
7. **XD 数据事实不能被删除或伪造。** 主文不必强调坏源细节，但数据协议或补充材料必须最少披露 declared=3954、accepted=3950、excluded=4。不得把 accepted3950 写成无条件完整3954。

## 1. 当前实验身份和数据合同

### 1.1 实验身份

本轮所有产物统一放入：

~~~text
outputs/icassp2027/control/codex-takeover-20260920/idea/v2-adgs-screen-20260921/
~~~

推荐 run 身份：

~~~text
dataset=ucf_crime|xd_violence
encoder=videomaev2
head_mode=direct_insert
screen=idea_screen_64x64
method=dense|group_uniform|adgs|pair_select
keep_ratio=1.00|0.60
seed=0
~~~

不要把这轮短筛选命名为 fit、select、confirm 或 official_test。已有 manifest 的历史 role 字段和原始数据身份不得删除、重命名或覆盖；本轮只是在这些资产之上建立一个明确的 training-side screen 身份。

### 1.2 128 个视频的配比

| 数据集 | normal | anomalous | 合计 | head |
|---|---:|---:|---:|---|
| UCF-Crime | 32 | 32 | 64 | UCF-V2 head |
| XD-Violence | 32 | 32 | 64 | XD-V2 head |
| **合计** | **64** | **64** | **128** | 两个独立 head |

视频选择要求：

- 只从各自 accepted train manifest 取视频；不使用 test 视频；
- UCF 和 XD 各自按 normal/anomalous 分层抽样；
- 使用固定 seed 20260921 生成 screen_manifest.json；
- 在看到 dense、ADGS 或 PairSelect 分数前锁定视频清单；
- 同一视频清单用于 dense、uniform、ADGS、PairSelect；
- 不按 clip 数量抽样，不让长视频因为切出的 clip 更多而自动获得更大权重；
- XD 只能从 accepted3950 视图取样，excluded4 不得重新混回。

如果已有可核验的 64+64 清单，应优先复用并记录其 manifest SHA；不能为了得到更好结果换视频。

### 1.3 训练 batch 配比

两个 screen head 分开训练；每个 screen head 只看一个数据集的 64 个视频。由于每个数据集的 screen 池是 32 normal +32 anomalous，screen run 使用每类 32 个 bag；完整训练的正式 run 才恢复既有 UR-DMU 的每类 64 个 bag：

~~~text
screen: 32 normal bags + 32 anomalous bags
formal full-train: 64 normal bags + 64 anomalous bags
~~~

screen bag 从该数据集的 32 normal +32 anomalous 视频池中采样，允许跨 step 重复采样，但必须记录 sampler seed 和视频来源。正式 full-train 则从该数据集完整训练集建立的样本池采样。dense 与所有压缩方法使用相同的 bag 采样序列。

本轮短筛选不把 128 个视频当成 128 个独立统计样本来制造置信区间；结果标记为 training-side screen。最终论文质量仍需按完整数据和视频级评测协议重新运行。

## 2. Head 和训练方式

### 2.1 两个 head 的身份

只建立下面两个 head：

~~~text
UCF-V2-URDMU-head
XD-V2-URDMU-head
~~~

不能把 UCF/XD 送入同一个 head，也不能把 VideoMAEv2 head 复用给 VideoMAE、TimeSformer 或 CLIP。

每个数据集的 head 只训练一次 dense baseline，然后冻结该 checkpoint，所有压缩方法走同一个 direct_insert head：

~~~text
dense          -> UCF/XD 对应 dense head -> dense score
uniform        -> 同一个对应 head -> uniform score
ADGS           -> 同一个对应 head -> ADGS score
PairSelect     -> 同一个对应 head -> PairSelect score
~~~

这样可以把 head 重新训练带来的收益和压缩方法本身分开。两小时内不做 refit_head，避免 head 数量和训练变量膨胀。

### 2.2 短筛选训练预算

正式 UR-DMU 合同仍保持已核验字段：

~~~text
optimizer = Adam
learning_rate = 1e-4
betas = (0.9, 0.999)
weight_decay = 5e-5
temporal_training_segments = 200
normal_bags_per_step = 64
anomaly_bags_per_step = 64
~~~

两小时筛选采用两级预算：

1. **screen run：** seed0、600 optimizer steps，只用于判断 ADGS/PairSelect 是否值得继续；
2. **如果主方法通过：** 再启动完整 3000 steps 的同一 head 训练，使用固定最终 checkpoint，不挑中途最佳 checkpoint。

600-step 结果不得写成正式论文质量数字；3000-step 结果仍属于固定 64+64 视频池的 training-side 结果，不能写成全数据泛化。

## 3. 主要方法：ADGS

### 3.1 方法来源

ADGS（Attention-Diverse Group Selection）复用：

- 平台 vispruner.py 的“重要 token + diverse token”思想；
- 平台 visionzip.py 的 attention importance 和 key/feature similarity 思路；
- VAD 现有 token_selection.py 的 group budget、确定性取整和 same-budget control；
- VAD 现有 deployment.py 的 verified geometry、真实 gather 和 layout receipt。

本轮采用 **drop-only**，不做 token mean 或 contextual merge。此前 VAD 的 pair-mean 已有负结果，平台 ToMe/VisionZip 的 merge 适配还涉及 mass、位置和后续布局，不适合作为两小时主线。

### 3.2 固定插入位置

只使用现有 VideoMAEv2 bridge 已核验过的单层位置：

~~~text
layer_index = 5
site = block.5.attn.output / 当前 bridge 已登记的等价位置
~~~

KimiCode 必须先读取当前 bridge receipt 确认真实 site；如果实际 receipt 名称不同，以真实 receipt 为准，不得自行把 block index 改成另一层。两小时内不做 layer sweep。

### 3.3 选择步骤

输入为当前层的 patch hidden states X 和 self-attention A。selector 不读取 label、video filename、future layer 或 dense teacher。

1. 计算 received attention importance：

   ~~~text
   importance[i] = mean over heads and query positions of A[..., query, key=i]
   ~~~

2. 按当前 VAD verified geometry 分组。VideoMAEv2 使用既有相邻 patch pair/group 表，不自行重新解释 t,h,w。
3. 每组按 keep_ratio=0.60 计算 exact quota；沿用现有 round_half_up 规则。
4. 预算的一半优先保留 attention importance 高的 token。
5. 余下预算使用 cosine farthest/diversity fill，避免所有 token 都来自同一个局部区域。
6. attention 分数相同用 native index 稳定打破平局。
7. 输出 selected indices、source coordinates、actual K、quota、importance digest 和 LayoutDelta。
8. 通过现有 indexed gather 产生真实短序列，不把结果 padding 回原长度。

默认配置：

~~~yaml
method: adgs
encoder: videomaev2
layer_index: 5
keep_ratio: 0.60
important_ratio: 0.50
diversity_metric: cosine
score_source: mean_received_self_attention
selection_scope: current_layer_only
protected_tokens: none_for_videomaev2_patch_only_layout
rounding: round_half_up
tie_breaking: native_index_ascending
~~~

如果当前 V2 bridge 没有可用的 attention weights，ADGS 不得假装使用 attention。10 分钟内无法接通时，将 ADGS 标为 blocked_missing_attention_capture，转而执行 PairSelect 备选。

## 4. 备选方法：PairSelect

备选方法直接使用已经存在并通过工程测试的 GroupBudgetSelector：

~~~yaml
method: pair_select
encoder: videomaev2
layer_index: 5
keep_ratio: 0.60
signal: member_l2_norm_max
rule: pair_select
seed: 0
~~~

PairSelect 的选择信号只来自当前 block output，不看标签，不看文件名，不看测试统计。它与 group_uniform 使用相同的 group quota、实际 K、特殊 token 规则和输出长度，因而是 ADGS 的低风险同预算备选。

必须保留 group_uniform 作为中立控制：

~~~text
dense         = keep_ratio 1.00
group_uniform = keep_ratio 0.60
ADGS          = keep_ratio 0.60
PairSelect    = keep_ratio 0.60
~~~

如果 ADGS 或 PairSelect 只比 dense 好，却不如同预算 uniform，不能把结果解释成 selector 有效；最多写成压缩率或 head 适配效应。

## 5. 两小时执行时间表

### 0–10 分钟：锁清单和输入

- 读取 AGENTS.md、本文件、当前 V2 bridge receipt 和 progress.md；
- 生成或核验 screen_manifest.json：UCF 32N+32A、XD 32N+32A；
- 记录 accepted3950 数据视图、VideoMAEv2 权重 SHA、代码 SHA、解释器和 torch/CUDA；
- 确认现有 dense head 输入 shape、readout、采样和视频/clip 映射；
- 禁止访问官方 test 标签或 test 汇总。

### 10–25 分钟：检查 ADGS 所需张量

- 读取当前 bridge 的真实 layer_index=5 和 hook site；
- 用一个 UCF normal、一个 UCF anomalous、一个 XD normal、一个 XD anomalous 做最小 forward；
- 确认 hidden、attention、coordinates 的 token 数一致；
- 确认当前 cache 是否含 attention/Q/K；
- attention 不可用时立即登记 ADGS blocked，不扩展新的 attention bridge。

### 25–60 分钟：实现 ADGS

建议新增：

~~~text
src/vadbench/token_reduction/attention_diverse_selection.py
tests/icassp2027/test_attention_diverse_selection.py
~~~

实现要求：

- 复用现有 TokenLayout、group quota 和 indexed_gather；
- 不修改现有 pair_select 的身份和历史行为；
- identity 路径必须返回数值等价输出；
- 实际 K 与 group_uniform 完全一致；
- 记录 source index、coordinates、group quota、selected index digest；
- 不使用 label、filename、test summary、future layer 或 dense teacher；
- 不新增模型参数，不引入 selector 网络，不训练 gate。

### 60–75 分钟：接入 PairSelect 备选和 uniform 控制

- 使用现有 GroupSelectDeployment；
- 不复制第二套 pair/trajectory geometry；
- 固定 seed0；
- 检查 dense、uniform、ADGS、PairSelect 的实际输出 token 数一致；
- 检查所有输出都进入同一个 UCF/XD 对应 dense head。

### 75–90 分钟：工程门

必须通过：

1. keep_ratio=1.0 identity，V2 输出 max-abs 在已登记容差内；
2. observer 开关不改变 dense 输出；
3. 输出 token 数真实减少；
4. selected indices 无重复、无越界；
5. coordinates 和 LayoutDelta 可追溯；
6. 不产生 padding 回 dense；
7. hidden/attention/selection 全部 finite；
8. 同一 seed 重跑 selection digest 一致；
9. 插件开销和序列长度均进入 timing receipt；
10. 失败时保留日志和 partial，不用旧结果冒充成功。

### 90–120 分钟：固定视频池短筛选

每个数据集分别使用一个 head，四个同预算 arm：

~~~text
UCF-V2 head: dense / uniform / ADGS / PairSelect
XD-V2 head:  dense / uniform / ADGS / PairSelect
~~~

推荐先使用 600 steps 的 screen run；如果 600 steps 已显示明显无效，停止，不为挽救方法追加未登记比例或层。若主方法通过，再启动 3000 steps 的完整 training-side run。

## 6. 通过、失败和选择标准

### 6.1 必须通过的硬门

- identity 通过；
- 同预算实际 token 数一致；
- 序列真实变短；
- selector 不读标签；
- 插件开销后端到端 latency 仍下降；
- 显存峰值不高于 dense，或至少明确记录额外开销；
- ADGS/PairSelect 的 detector output 可被对应数据集 dense head 接受。

### 6.2 主要方法通过条件

ADGS 进入主方法候选需要同时满足：

1. 相对于 dense，固定视频池的检测分数没有低于预设 -0.005 工程容忍度；
2. 相对于同预算 group_uniform，ADGS 至少不更差；
3. 插件开销后仍有正的端到端加速；
4. 实际 token 数、显存、latency 和 selection digest 有完整 receipt。

这是 training-side screen 的工程判定，不是正式泛化证明，也不替代完整数据上的视频级 bootstrap CI。

### 6.3 备选方法进入条件

如果 ADGS 在 10–60 分钟内无法取得 attention 或工程门失败，立即转 PairSelect。PairSelect 如果满足上面四个硬门，可以作为本轮主方法；ADGS 的失败原因和源码状态必须保留。

如果 ADGS 和 PairSelect 都不满足净加速，停止这轮方法扩展，报告 no_actionable_method_in_2h，不要继续扫更多层、更多 ratio 或更多 selector。

## 7. 计时口径

本轮至少记录：

1. encoder_only：VideoMAEv2 前向和压缩插件；
2. head_inclusive：压缩输出接入对应 UR-DMU head；
3. end_to_end：原始视频采样、encoder、插件、head 和输出。

计时要求：

- 独占 GPU；
- 相同输入视频、相同 clip 数和相同 batch；
- warmup 10 次，测量 50 次，CUDA synchronize；
- 报告 latency、throughput、allocated、reserved、peak memory；
- 插件自身 attention capture、cosine diversity 和 index/gather 时间单独列出；
- 不用特征提取共享任务的速度冒充正式 compression timing；
- 不把“最大 batch 变大”单独写成模型加速。

## 8. 代码、回执和目录

本轮建议新增或更新：

~~~text
src/vadbench/token_reduction/attention_diverse_selection.py
tests/icassp2027/test_attention_diverse_selection.py
work/codex-takeover-20260920/idea/run_v2_adgs_screen.py
outputs/icassp2027/control/codex-takeover-20260920/idea/v2-adgs-screen-20260921/
~~~

输出至少包含：

~~~text
screen_manifest.json
resolved_config.yaml
environment.json
model_and_code_sha.json
head_ucf_receipt.json
head_xd_receipt.json
dense_receipt.json
uniform_receipt.json
adgs_receipt.json
pairselect_receipt.json
metrics_training_screen.json
timing.json
failure_or_negative_result.md
~~~

每份 receipt 必须记录：host、开始/结束时间、命令、代码 SHA、数据 manifest SHA、模型 SHA、encoder layer/site、method、ratio、seed、actual K、token digest、head checkpoint、metrics、latency、memory、issues 和 next action。

## 9. 论文边界

本轮 64N+64A 是快速方法筛选，不是最终正式数据实验。可以用于决定 ADGS 或 PairSelect 是否值得继续，但不能把它写成：

- UCF/XD 泛化已证明；
- 正常/异常性质已经确认；
- 方法在完整数据上不掉点；
- 正式 test 已完成；
- 完整 XD3954 已无数据缺口。

论文中可以不在主叙述里展开 XD 四个坏源，但数据和复现部分必须最少披露 accepted3950/declared3954/excluded4。所有失败、负结果、ADGS blocked 或 PairSelect fallback 都进入内部 evidence summary。

## 10. 最终交接定义

KimiCode 完成这一阶段的条件是：

- screen manifest 已锁定；
- UCF/XD 各一个 V2 dense head 已生成；
- ADGS 或 PairSelect 至少一个通过 identity、真实变短和净计时门；
- dense、uniform、主方法、备选方法在同一 128 视频池和同一 head 下完成可比回执；
- 代码测试与 python -m compileall src tests 通过；
- progress.md 追加一条带时间口径的阶段记录；
- 未访问官方 test 选择方法；
- 失败与负结果没有被删除或覆盖；
- 若没有方法满足硬门，明确报告 no_actionable_method_in_2h，停止扩展。
