# 训练数据范围与时间估算（2026-09-21）

## 1. 用户确认的训练规则

本文件记录 2026-09-21 用户补充的正式规则：

> 所有正式训练都必须使用对应 encoder、对应数据集的完整训练集。64 normal + 64 anomalous 只用于训练集内部的 idea/压缩规则快速筛选，不能替代正式 full-train。

因此要区分两个阶段：

| 阶段 | 视频范围 | 用途 | 能否写成正式结果 |
|---|---|---|---|
| idea screen | 两个数据集合计 64 normal +64 anomalous；默认每个数据集 32 normal +32 anomalous | 快速比较压缩规则、检查真实变短和净加速 | 只能写 training-side screen |
| formal full-train | 每个 encoder、每个数据集的全部可用训练视频 | 训练正式检测模型、生成可用于正式比较的 checkpoint | 可以进入正式训练链，仍需完成 QA 和评测门 |

“使用全部训练集”和“每个 optimizer step 使用 64 normal +64 anomalous bags”并不矛盾：

- 完整训练集用于建立每个视频的 200-bin 特征序列、样本池和训练来源；
- optimizer 每一步仍按固定协议抽取 64 normal bags +64 anomalous bags；
- 64+64 是每步的采样预算，不是整个训练只包含 64+64 个视频；
- 每个 encoder×dataset 使用独立 head，不跨 UCF/XD 共享，也不跨 encoder 复用参数。

## 2. 当前正式训练身份

当前 active dense encoder 为：

~~~text
videomaev2
videomae
timesformer
~~~

对应正式训练组合：

~~~text
3 个 encoder × 2 个数据集 = 6 个正式训练组合
~~~

如果 CLIP 兼容性验证通过并进入主线，则增加：

~~~text
1 个 CLIP × 2 个数据集 = 2 个正式训练组合
总计最多 8 个正式训练组合
~~~

每个组合使用：

~~~text
UR-DMU
upstream commit = 40cfdf5...
optimizer = Adam
learning rate = 1e-4
betas = (0.9, 0.999)
weight decay = 5e-5
temporal segments = 200
optimizer steps = 3000
normal bags per step = 64
anomalous bags per step = 64
checkpoint = fixed final step
seed = 0（关键配置再按计划补 seed 1/2）
~~~

正式训练不能使用：

- 64+64 screen 视频池作为唯一训练来源；
- screen600 checkpoint 作为正式模型；
- UCF head 直接用于 XD；
- 一个 encoder 的 head 直接用于另一个 encoder；
- 测试集分数选择 checkpoint、压缩率或方法。

## 3. 已完成和未完成的正式训练

截至当前已有正式 full-train 回执：

| Encoder | 数据集 | 视频 | steps | QA | 状态 |
|---|---|---:|---:|---|---|
| VideoMAEv2 | UCF | 1610（800 normal +810 anomalous） | 3000 | nonzero gradient 3000、reload parity 通过 | 已完成 |
| VideoMAE | UCF | 1610（800 normal +810 anomalous） | 3000 | nonzero gradient 3000、reload parity 通过 | 已完成 |

对应回执见：

~~~text
work/codex-takeover-20260920/heads/heads_receipt-v3.json
~~~

当前尚未把以下组合完成为新的正式 full-train 回执：

~~~text
VideoMAEv2 × XD accepted3950
VideoMAE   × XD accepted3950
TimeSformer × UCF full1610
TimeSformer × XD accepted3950
~~~

XD 的 VideoMAEv2 accepted3950 特征分片已经完成覆盖，但正式训练仍需使用完整 accepted3950 视图和绑定合同；不能使用刚才的 32+32 screen 训练结果代替。

TimeSformer 需要等待它的 full-train merged view 和 role-equivalence contract 完成，再启动正式训练。

## 4. 已测到的 screen 训练时间

刚才的 64+64 screen 训练使用了每个数据集 32 normal +32 anomalous 视频，训练步数为 600：

| 数据集 | 实际开始 | 实际结束 | 用时 |
|---|---|---|---:|
| UCF screen600 retry32 | 19:44:08 | 19:46:16 | 128 秒 |
| XD screen600 retry32 | 19:46:16 | 19:48:27 | 131 秒 |

这个数字只说明小样本 screen head 很快，不能线性当成 full-train 的实际时间，因为 full-train 还要加载完整视频池、建立完整 200-bin aggregation cache 并完成更大的来源审计。

## 5. 正式 full-train 单次时间估算

目前没有把正式 UCF full-train 回执的开始/结束时间写入 heads_receipt-v3.json，因此下面是规划范围，不冒充历史实测数字。

按当前 A100 环境、3000 steps、完整 FeatureStore 读取和固定末 checkpoint 估计：

| 组合 | 完整数据准备与 aggregation | 3000 步训练与 QA | 单次规划范围 |
|---|---:|---:|---:|
| UCF × VideoMAEv2/VideoMAE | 10–30 分钟 | 10–20 分钟 | 20–50 分钟 |
| XD × VideoMAEv2/VideoMAE | 25–60 分钟 | 10–25 分钟 | 35–85 分钟 |
| UCF × TimeSformer | 15–35 分钟 | 10–25 分钟 | 25–60 分钟 |
| XD × TimeSformer | 30–75 分钟 | 10–30 分钟 | 40–105 分钟 |
| UCF/XD × CLIP | 取决于作者输入路径、crop 和兼容性 | 10–25 分钟 | 每次约 30–90 分钟 |

XD 范围更宽，原因是 accepted3950 的完整特征视图更大、分片更多，且必须做统一身份和索引检查。共享 GPU 或 NFS 读取拥塞时，时间会增加；不能把小样本 screen 的 2 分钟直接外推到正式 full-train。

## 6. 六次和八次训练的总时间

### 从零开始完整跑六次

六次组合是：

~~~text
VideoMAEv2 × UCF/XD
VideoMAE   × UCF/XD
TimeSformer × UCF/XD
~~~

如果一张独占 A100 顺序运行，包含每次的完整数据准备、3000 steps、保存重载 QA 和必要的短验收：

~~~text
保守估计：约 4–8 小时
~~~

如果使用两张相互独立、没有 foreign process 干扰的 GPU 并行：

~~~text
墙钟时间约 2.5–5 小时
~~~

当前服务器是共享 GPU 环境，正式效率计时和训练资源不能默认按理想并行估算。实际 first full-train receipt 出来后，应使用该组合的真实开始/结束时间重新校准后续 ETA。

### 当前已经有 UCF V2/VMA 正式回执的情况

当前不是完全从零开始。UCF 的 VideoMAEv2 和 VideoMAE full-train head 已经完成，因此后续至少还需要：

~~~text
VideoMAEv2 × XD
VideoMAE   × XD
TimeSformer × UCF
TimeSformer × XD
~~~

这些组合在完整视图和合同就绪后，顺序运行的规划范围约为：

~~~text
约 2.5–5.5 小时
~~~

### 如果 CLIP 进入主线

在六次 dense 训练之外，再增加：

~~~text
CLIP × UCF
CLIP × XD
~~~

总规划范围增加约：

~~~text
0.75–2 小时
~~~

这还不包括 CLIP 作者原生 crop、center/10-crop、feature readout 和已有 head 兼容性尚未通过时的额外排查。

## 7. 正确执行顺序

后续按下面顺序执行：

1. 64+64 screen 只用于比较 ADGS、PairSelect、uniform 等压缩规则；screen 结果不进入正式论文数字。
2. 根据 screen 结果决定是否保留一个主压缩方法和一个备选方法。
3. 对每个 active encoder×dataset 建立完整训练视图。
4. 使用完整训练集训练 dense baseline，3000 steps、固定最终 checkpoint、QA/reload 通过。
5. 如果采用 refit_head，压缩版本也必须使用对应 encoder/dataset 的完整训练集重新训练；不能用 64+64 训练 head。
6. 如果使用 direct_insert，压缩输出接入同一 encoder/dataset 的完整 dense checkpoint，但压缩输入和正式评测覆盖必须与 dense 对齐。
7. 训练和压缩版本的所有正式结果都必须绑定完整数据 manifest、feature contract、model SHA、代码 SHA 和 environment。

## 8. 当前修正结论

刚才已经完成的 UCF/XD screen600-retry32 训练保留为小样本方法筛选证据，不删除、不改写、不升级为正式结果。压缩生成阶段已经停止，避免继续在错误的数据范围上浪费 GPU。

后续正式训练不能再调用 screen600 的 32+32 视频池。正式训练必须回到：

~~~text
UCF: full1610
XD: accepted3950
~~~

缺少完整视图、身份合同或 QA 时，保持 blocked，不拿 screen 结果顶替。

## 9. 四卡并行安排

用户已明确要求：剩余四个正式组合尽量使用四张相互独立的 GPU 同时训练。四个目标为：

```text
VideoMAEv2 × XD accepted3950
VideoMAE   × XD accepted3950
TimeSformer × UCF full1610
TimeSformer × XD accepted3950
```

并行启动的前提是：

- 每个组合已有完整、ready、身份一致的 FeatureStore/view contract；
- 每个进程只看到一张指定 GPU（通过 CUDA_VISIBLE_DEVICES 映射为 cuda:0）；
- 四个 run 使用独立 output root、独立 run id 和独立日志；
- 不同进程不写同一个 FeatureStore index、merge contract 或 aggregation cache；
- 启动前重新核对 GPU owner、显存、foreign process 和解释器；
- 共享卡或显存不足时，宁可降低并行数，也不抢占其他项目或重复派工。

并行不会改变“完整训练集”规则，只改变墙钟时间。四个 run 的 QA、checkpoint reload、实际开始/结束时间和失败回执必须分别保存；任何一个组合失败，不用其他三个成功结果替代。
