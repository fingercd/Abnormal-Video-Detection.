# ICASSP 2027：投稿冲刺与双数据集执行计划 V3

> **2026-09-20 22:05：本文件的执行排期已暂停。** 用户收拢到只完成特征提取，当前唯一执行入口为
> [`FEATURE_EXTRACTION_PLAN_20260920.md`](FEATURE_EXTRACTION_PLAN_20260920.md)。下文保留为该裁决前的
> 研究背景和历史计划，不自动恢复其中的训练、idea、写作或打包任务。

更新日期：2026-09-20（18:14 历史现场表；20:04–20:16 当前快照已追加）。状态：**当前唯一执行计划**。

本文件接替 `RESEARCH_PLAN_V2.md` 的执行地位。V2 正文保留为历史记录，文件顶部只增加
superseded 指针，不追溯改写其中的实验身份、旧 SHA、旧结果或旧安排。`progress.md` 保留完整
历史回执，并在顶部提供当前状态和回执导航。任何旧协议 JSON、已绑定 SHA 的方法/角色文件、失败
回执和缓存都保持原位；本计划不会通过修改旧文件来制造“已冻结”或“已完成”。

本轮目标是按论文关键路径交付可审计的两数据集证据、匹配检测头、最小有效方法和真实效率表。
论文主线优先于无后继的工程扩展。所有缺失值保持 `null` 或“待完成”，训练内结果、开发诊断和
已接触测试的事实必须带身份说明，不能包装成独立泛化证据。

## 1. 当前权威、范围和不再适用的旧口径

### 1.1 权威顺序

发生冲突时按以下顺序读取：

1. 用户 2026-09-20 最新裁决及本文件的“当前执行层”；
2. 当前运行回执与各自的 resolved config、数据/代码/环境/校准指纹；
3. 未被新契约替代的稳定协议字段；
4. V2、旧桌面计划和旧任务交接文档中的历史安排。

`official-detector-protocol-v3.json`、`method-freeze-contract-v1.json` 和旧
`batch1-preregistration-20260920.md` 仍是不可变历史资产。它们可以提供已核验的 UR-DMU
结构参数或旧批次来源，但不能被当作本计划的当前模型范围、当前方法范围或当前完成状态。
例如 v3 JSON 仍含 V-JEPA 2、`paired_random`/`pair_linear` 和旧 status；这些字段不因继续
可读就自动代表当前论文主方法。需要新方法身份时，建立真实的新版本和新 SHA，不能覆盖旧绑定。

### 1.2 当前论文执行范围

| 线 | 当前角色 | 执行要求 |
|---|---|---|
| VideoMAEv2（V2） | 主 dense encoder，主方法适配对象 | UCF/XD 各自完成 native dense 提取、各自训练匹配 UR-DMU 头，并进入直接插入和效率主路径 |
| VideoMAE（VMA） | dense 基线与边界研究 | UCF/XD 各自完成提取和匹配头；不预先承诺压缩迁移成功，依据证据和资源决定是否扩展 |
| TimeSformer（TS） | dense 基线与结构边界研究 | UCF/XD 各自完成提取和匹配头；当前 full-train merge 正在运行，完成后再启动 formal head |
| V-JEPA 2 | 已退出检测实验 | 保留权重、缓存、观察、失败和工程资产；不删除、不再以其作为本轮检测完成条件 |
| CLIP 候选 | 论文主方法训练侧探索，核心适配为 V2 + 一个 CLIP | 三条候选路线均为 ViT-B/16，共享 visual 不能写成三个 encoder。当前暂定 VadCLIP（两数据集权重已就绪，仍须兼容性核验）；DSANet 可选；AnomalyCLIP 新增成绩暂停。第二个 CLIP 后端只有原视频路径可用且训练开发支持时才做补充，测试集不用于选赢家 |

三 dense encoder 与两个数据集之间没有“全体完成”屏障：任一 encoder/dataset 的完整视图和
质量门通过后，立即训练该组头并做 QA/准备评分链；正式 test 评分等待方法与比较矩阵冻结。VMA、TS 的 dense/边界证据必须保留，即使主方法
最后只在 V2 与一个 CLIP 上形成闭环。

XD 当前接受训练集为 `accepted_train=3950`，原 declared=3954，excluded=4；用户已接受该
隔离视图作为本轮完整可用训练集合。quarantine 视图和排除回执保持不可变，停止继续搜索、下载、
补 4 个文件或等待新的 3954 契约；旧修复尝试、失败证据和历史安排保留。XD 三 encoder 的头和
质量关键路径不再被这 4 个文件阻塞。

旧 VMA/TS 压缩支线已按 `user_scope_change` 处置：VMA pair 和 TS random 已实际 SIGTERM 停止且
无复发；VMA random、TS uniform/pair 已自然完成原身份，结果和 partial 保留。V2 有效消融继续
服务范数/成员偏好机制审计；VMA/TS 的三 dense 两数据集基线仍是必做路径。

### 1.3 执行角色

- `root` 负责优先级、资源/进程判断、版本决策、身份核验和最终汇总；不把旧的“Kimi 主控只调度不能读/判断”限制带入本轮。
- 所有常规执行、提取、训练、评分、计时和文档落盘由 Luna 以最高可用推理强度完成；一次任务只绑定一个责任人和一个写入范围。
- idea 训练侧探索由唯一 Astra（`gpt-6-astra`，medium）负责，范围固定为 V2 + 一个经核实的 CLIP，覆盖 UCF/XD 的训练侧闭环；不得扩展成多个共享 visual 的“encoder”或反复派发旧探索。
- `takeover_infra_dispatch` 负责 node2/node3 提取恢复、队列接力和低开销结构化巡检；node1 管理员恢复由用户延后，不作为生产门槛。它只写自己的控制目录和回执，不覆盖本计划。

## 2. 用户最新裁决落地

以下决定已经进入执行层，后续任务不得回到旧的“等全部完成再开始”流程：

1. **node1 管理员恢复延后。** 用户已裁决先用 node2、node3 继续；node1 状态记为 `user_deferred_admin_recovery`。node1 仍可经 node3 内网 SSH 登录，但逐卡 NVML 超时且无 sudo，SSH 可达不等于 GPU 可用；不承诺 24 卡都在运行。
2. **保持 TS/VMA 生产接力。** node2 已恢复 TS shard-026→030 共 5 路并有真实产物增长，继续核对自动后继；不把这项局部恢复写成 full XD 完成。VMA 新 claim 已恢复，已有片自然完成，不取消三 encoder 目标。
3. **删冗余守护、暂停无用新工作。** 只暂停已经确认没有有效后继、会重复写入或不再服务关键路径的本轮 guard/controller；不杀其他项目、不删旧缓存/结果/失败证据、不在运行中的 bash 上原地覆盖。每个终止动作必须有 PID/owner/argv、原因、替代任务和回执。
4. **接受 XD 排除视图。** `accepted_train=3950/3954` 是本轮完整可用训练集合，excluded=4 诚实披露；停止坏源搜索、下载和补 4，不等待新 3954 契约。quarantine 与排除回执不可变，旧修复失败和搜索历史保留。
5. **自然接力。** A100/V100 当前片段完成并通过增量审计后，自动领取下一 encoder/dataset 片段；不等待另外五组或两个数据集全齐。健康槽位优先领取关键路径，不能空等代理回话。
6. **idea 立即开始。** 训练侧双数据集探索立刻启动，核心适配限定 V2 + 一个 CLIP。每个具体训练侧实验配置在执行前登记数据角色、输入、版本和参数；允许根据训练探索选择或修订候选并留下新版本。最终方法必须在正式 test 前冻结，不能拿已经接触过的 test 结果调整新方法。
7. **按当前 head/merge 状态推进。** V2/VMA UCF formal head 已完成 QA；TS fit/confirm/旧 select union 已核到 1610，full merge 发布后再训练 TS head。VideoMAE 另开诊断，QA 通过不等于质量通过，不能据旧 test `0.8106` 断定非 bug。
8. **论文导向。** 停止无论文证据价值的 dev、重复校准和旧探索派发；保留历史资产、负例、失败、已看测试的时间线和反例。idea 线继续，但必须产生可审计的双数据集训练侧产物。
9. **逐项回执。** 每次报告逐项回应本节 1–8 及资源、测试、论文状态；未知写未知，不能用目录名、进程数或旧 ETA 代替。
10. **新 MD 计划。** 本文件是仓库唯一当前计划；桌面 `Codex_接管后论文执行计划_20260920.md` 与本文件一致复制，其他旧桌面文件只作历史参考。
11. **交付核心包。** 整理服务器核心代码、关键文档、真实回执索引和审阅 prompt 到桌面交付包；不把密码、token、私钥、数据凭据、大型特征或权重放入 Git/压缩包。
12. **收窄旧压缩支线。** VMA pair/TS random 已由真实 SIGTERM receipt 停止且无复发；VMA random/TS uniform/pair 自然完成的原身份保留。V2 有效消融保留，停止原因是范围裁决，不是按成绩删除反例。

## 3. 不变的实验与数据契约

### 3.1 数据、角色和训练

- UCF 主划分为官方 1610 train / 290 test；XD 的 original declared train 为 3954，本轮 `accepted_train=3950` 作为完整可用训练集合，excluded=4 单列披露，quarantine 视图不可变。XD test 为 800。
- 两个训练集都使用视频级弱标签 W。不同训练视频、近重复来源、fit/confirm/select/test 角色隔离；不同片段不能跨角色。`full1610` 是官方训练视图，不等于独立验证；select161 是训练内开发角色。
- 既有 fit/confirm/select 锁、原始 manifest、旧缓存和失败记录保持身份。`full1610` 视图由审计后的成员并集发布，不能把 `1288` 视图或旧 32-clip 特征上采样后称为完整训练。
- 后端采用 UR-DMU 的稳定参数：作者源码 commit `40cfdf5f8bebbc3b59373935f9d6ca00e2f32bbc`、200 个时间段、Adam、lr `1e-4`、weight decay `5e-5`、3000 optimizer steps、每步 64 normal + 64 anomaly bags、固定末 checkpoint。heads 代理必须核对当前实现与输入维度；不把旧 v3 JSON 中的历史方法列表或旧模型范围当作本轮方法冻结。
- 每个 encoder、dataset、seed 独立训练匹配头；不跨表示空间复用 I3D 头，也不把同维度写成同一权重。dense 头 `direct_insert` 是主比较，`refit_head` 只作独立补充，训练 budget 相同并单列身份。
- 训练内 100 视频是方法诊断和收敛检查，不是泛化证明。任何 100 视频结果都明确标记 training-internal，不能替代 1610/3950 训练视图或 290/800 测试证据。

### 3.2 方法和已接触测试的披露

- 旧 pairselect 预登记只属于旧批次历史。新 idea 线允许产生新版本；每个训练侧配置在执行前登记规则、层、预算、取整、随机种子、训练角色和版本。训练探索可以据训练侧证据选择或修订候选并保留版本；最终方法、正式层/预算和 test 格子必须在正式测试前冻结。
- 当前已接触过旧的 test290 generic 诊断范围；Arson011 字段问题属于历史 audit 修复过程，当前 official 状态只认 fresh receipt，不能把历史 generic 数字升级为 final official 分数。因此不能在论文中写“全流程从未看 test”；后续方法、层、预算、阈值和 checkpoint 不得根据 test 结果调整。
- 当前允许的主适配是“V2 + 一个 CLIP”的训练侧探索；VMA/TS 先交付 dense 基线和结构边界。压缩能否迁移由独立证据、同预算对照、质量和效率共同决定，不先承诺统一成功。
- 部署 reducer 只使用当前可得的模型状态和已冻结校准资产，不读真值、视频级标签、异常类别、标签文件名、测试汇总、未来层张量或完整 dense 教师。
- UCF 主指标为 frame ROC-AUC；XD 主指标按作者 `precision_recall_curve` 后 `auc(recall, precision)` 的梯形口径，`average_precision` 另列，不能混称。质量差值统一定义 `Δ = method - dense`；95% 配对 CI 下界 `>= -0.005` 才能写非劣，训练内点估计本身不能替代 CI。

## 4. 18:14 现场事实（事实与计划分开）

以下是 2026-09-20 18:14 现场快照，只能作为该采集时点的事实；后续以新鲜回执更新，不把它写成现在仍在运行：

| 范围 | 现场事实 | 解释/动作 |
|---|---|---|
| node1 | 没有本轮提取；GPU 整体空；GPU7 有历史 Xid48/63 隔离 | 先做 node1 健康槽位恢复与小片段增长验证；保留 GPU7 隔离 |
| node2 | 7 路 VMA；TS 为 0 extract；guard `125984` 等待 `GPU7--formal-timing` 独占，撞上 VMA | formal-timing guard 不得抢 VMA；确认冗余后按第 2 节规则停/暂停并留回执，计时另排独占窗口 |
| node3 | 8 路 V2 XD，4 路 UCF batch；fullformal CPU drivers `26801/26807` 约 20 GB RSS，尚无完成 | 先核实 CPU 合并/训练真实增长；不把 driver 存活写成 head 完成 |
| XD V2 | 有效去重 clips `496,474 / 1,164,186 = 42.65%`；test800 已齐 | 可先做测试来源审计和 V2 训练视图准备；不能等 VMA/TS |
| XD VMA | `199,145 / 1,164,186 = 17.11%`；test 尚未开始 | 当前优先训练片段/修复槽位，完成后自动转 test |
| XD TS | `325,387 / 2,330,558 = 13.96%`；连续两分钟 0 增长 | 先修 launcher/guard/队列；物理 `384,270` 含 `58,883` 重复，不能当完成进度 |
| UCF V2/VMA | merged-full 的 1610 视图已显示 completed，当前回执记录字段为 `790,271 records` | 仍需成员覆盖、表示身份和 consumer QA；records 数不替代完整审计 |
| UCF TS | merged fit1288 已完成，但相对 protocol 的 train 路径立即失败 | 修路径/视图/输入契约后单独补 full1610；不写成 full1610 完成 |
| UCF dev/test 诊断 | dev1288 头 generic test290：V2 `0.8509978553`，VMA `0.8106398037` | 这是已接触的 generic 阶段性数字；Arson011 截断字段阻塞 v13 official audit 验收，不能写成最终 official 主结果 |
| 批次 1/1b | V2 dense/uniform/random/pairselect 均 `100/100`，另有 pairfixed `55/100`；VMA dense/uniform `100/100`、random `51/100`、pairselect `0/100`；TS dense `100/100`、uniform `71/100`、pairselect `71/100`、random `0/100` | 以上是 18:14 逐臂计数，不代表继续派工；VMA random、TS uniform/pair 待 infra 按 `user_scope_change` 停止，VMA pair/TS random 不派。V2 pairfixed/1b 只隔离 pair 内成员偏好，不证明 pair 间排序；旧身份和负例保留 |

这张表只说明现场，不证明任何方法有效、任何 head 已通过 QA 或任何正式效率数字成立。

### 4.1 约 18:37 后续诊断（待处理，不是完成回执）

后续本地 SSH 工具观察补充了两项诊断，不能覆盖 18:14 快照，也不能写成恢复完成：

- node1 逐卡 NVML 查询出现超时，当前只支持“疑似驱动阻塞”的诊断；GPU7 的历史 Xid48/63 隔离不解除，node1 修复仍待 infra 回执。
- node2 只剩一条 VMA 路径，其他卡显示空闲；精确 PID、owner、argv、claim 和有效增长以新的 infra receipt 为准。TS 恢复优先在 node2 处理，不等待 node1 修复。

现场审计回执为 [`live-audit-1814.json`](../../outputs/icassp2027/control/codex-takeover-20260920/audit/live-audit-1814.json)
和 [`live-audit-1814.md`](../../outputs/icassp2027/control/codex-takeover-20260920/audit/live-audit-1814.md)。
其中记录实际远端 status/log 路径、root 18:06–18:14 本地 SSH 观察范围和两类去重扫描；它明确
标注为非全量内容审计。

### 4.2 当前快照（20:04–20:16 本机时间；不改写 18:14 历史表）

当前层以 `work/codex-takeover-20260920/heads/heads_receipt-v3.json`、
`work/codex-takeover-20260920/retire/retire_receipt_20260920.json` 和后续 infra receipt 为准：

- V2/VMA UCF `official-fulltrain-final` 头均为 1610 videos、3000 steps、3000 nonzero-gradient steps，training QA passed，exact reload parity 通过。它们先进入 QA/评分链准备，不等于已产生正式 test 分数。
- TS fit/confirm/authoritative old select union 已核到 1610，未重提缺失 149；旧 select 权威路径为 `/users/fotile/icassp2027-runs/code-2d9d1a0/outputs/icassp2027/control/development-baseline-20260918-a01/jobs/timesformer-select/extractions/dense/features/train`。较早 full merge PID8255 因 role_lock SHA 未传入而 failed before publish；当前 retry01 PID29202 正在同一源上发布 `merged-full/timesformer/dense-full-view`，TS formal head 尚未启动。
- merge/training path/fulltrain-choice 修复后的受影响回归为 101 passed，compileall 通过；这些是工程验证，不替代模型质量结果。
- node2 TS/VMA 正在生产，TS 已从 shard026 接力到 030 共 5 路并见真实产物增长；VMA 新 claim 已恢复且已有片自然完成。node3 V2 v11 回执为 48 done、8 running、22 unclaimed；A100 空卡接力逻辑已在 fake 场景通过，实际未来接力尚未触发。v11 runtime 身份以 `outputs/icassp2027/control/codex-takeover-20260920/infra/v11_runtime_receipt.json` 的 dispatcher source SHA `877bb434722b82b908b0955730959436ac78e037b75fb634ff44b309b9cc4e64` 为准，不沿用旧 v8 摘要。
- node1 标记为 `user_deferred_admin_recovery`：20:17–18 Tailscale gateway ping 11 ms、node2 SSH 成功，只有 node1 经 Tailscale banner 超时；20:31 关闭 Tailscale 后校园网 node1 SSH 成功。node1 CUDA Driver API 已枚举 8 卡，但 NVML/nvidia-smi 仍超时；SSH 可达和 CUDA 枚举都不等于 GPU 可用于生产。有效生产资源是 node2 健康卡与 node3，不能承诺 24 GPU 全部运行。
- XD 当前 accepted train 是 3950/3954，excluded=4；本轮不再修复/下载/等待四文件，三 encoder XD 头不被该问题阻塞。用户行动回执继续披露排除身份。
- VideoMAE 新诊断 receipt 未发现 invalidating implementation bug，但 pooled temporal lag-1 `0.9994–0.99998` 是可复现质量风险；token-axis `0.827` 不能用来反驳 pooled 风险。旧工程 QA、旧 generic test 或旧 test290 分数不作“非 bug”证据。CLIP 当前为 float32 engineering identity/shortening smoke，质量入口 A/B 未冻结；F06-S1 已完成当前 8 视频×2 窗口，未支持稳定主信号，停止扩量但不停止 idea。
- 已创建的 30 分钟主控 heartbeat `icassp` 只在实质变化时通知；它不新增 feeder。普通执行使用 Luna Max，idea 使用 Astra Medium；本机/应用离线时本地主控不会唤醒。

这组当前快照用于接续任务，不代表三 encoder、两数据集或论文实验整体完成；正式 test 评分仍等待
idea 收敛和方法/比较矩阵冻结。

### 4.3 20:38 fresh progress、资源情景和 9/20–9/24 关键路径

20:38 的 fresh 去重扫描记录在 `outputs/icassp2027/control/codex-takeover-20260920/infra/xd_eta_receipt_20260920T202815Z.json`；此前从 20:38 起算的旧 ETA 已过期，必须以 infra 新鲜快照重算，不能把旧 ETA 当承诺：

| Encoder | 20:38 total unique clips / target | 比例 | 20:38 旧 ETA（已过期） | 使用边界 |
|---|---:|---:|---:|---|
| V2 | `849950/1164186` | 73.0% | 约 2 h | 只适用于 V2 自身当前 aggregate rate |
| VMA | `238412/1164186` | 20.5% | 约 41.5 h（旧快照条件） | 新 claim 已恢复；不能用 V2 速率替代 |
| TS | `466999/2330558` | 20.0% | 约 38.6 h | 只能按 TS 自身 fresh rate 估算 |

资源调度分两种情景：

- 保守情景：node1 不纳入生产，使用 node2 健康卡和 node3；沿用各 encoder 自己的有效速率，VMA/TS 不借用 V2 吞吐作 ETA。GPU 槽位用于当前 shard、F07 CPU 审计和 head/merge 关键路径，不抢正式独占计时。
- 加速情景：A100 释放后立即 relay 到 TS/VMA 当前 unclaimed shard；目前 node3 GPU0 与 V2 共享 TS038 的 347 秒试验得到 TS 约 8.8 clips/s，V2 overlapping 约 37.8 clips/s。该试验不是正式计时，也不足以外推所有 A100 或 VMA；下一张卡必须等待对应 encoder 的真实 successor receipt 后再更新速率。

当前 dispatcher v15 的实际源为 `work/codex-takeover-20260920/infra/takeover_dispatcher_v15.py`，SHA
`176f014f56e2a2083ed7b50f88c5e3bf95e7e029c8122be3bf07c3f47b8b6ec3`，远端 PID `45240`；
TS/VMA 自动后继以 dispatch receipt 为准，node1 管理员恢复不进入关键路径。

| 时间（香港） | 关键路径 | 就绪/停止门 |
|---|---|---|
| 9/20 20:38–22:00 | fresh 进度、TS/VMA/V2 node2/3 接力；F07 CPU 100 视频×7 臂 norm audit 收口；论文格式与审阅包核查 | F07 完成但若三关键臂 cache identity 阻塞则不进入 N；不新增 F07 GPU 提取 |
| 9/20 22:00–23:30 | Astra 查 F07 参数/控制是否生效；F08 保持 ToFu/MLERP 既有尺度对照；F09 只做 T/U/R CPU config/identity gate | F07 不重跑100、不批准N；F09 不派 GPU；没有稳定主信号不改层/预算救结果 |
| 9/21 01:00 | 训练侧小实验结果整理与失败证据；等待 TS full merge/head fresh receipt | 不把 F07 训练内 audit 当独立确认 |
| 9/21 08:00 | 方法 Go/NoGo | 没有性质、干预和同预算边界证据就 NoGo 新两阶段主方法；deadline 不降低门槛 |
| 9/21 白天 | 若 Go：冻结 V2+CLIP 方法、层、预算、输入身份和比较格子；若 NoGo：保留 study/边界主线并停止扩展 | root 逐项批准后才生成 formal test feature/scoring job |
| 9/22 | formal test dense/方法评分、CI、XD raw→canonical 审计和短独占效率 | head ready 只代表 QA/评分链准备；不得在冻结前偷跑 test 选参 |
| 9/23 | 内部论文数字和审阅冻结 | 保留 null/blocked/负结果；作者核查主要论证与引用 |
| 9/24 20:00 HKT | CMS 页面显示的外部投稿时限 | CFP 页面显示 9/23，旧 PaperKit 日期表显示 9/16，三者冲突保留；内部截止仍按 9/23，不因 CMS 延后放宽研究门槛 |

论文格式工作区为 [`manuscript-20260920`](manuscript-20260920/)，官方 Paper Kit 和格式构建已完成
初步核查。目标为 4 页技术内容，第 5 页按最新规则只在允许范围放 references/资助/ethics；single-anonymous、
ORCID、英文、字体和文件大小按最新 CMS/官方 kit 核对。AI 只可辅助实验、图表和核查，禁止生成重要、
大部分或完整论文 section；作者必须核查事实、引用、统计和主要论证。

## 5. 依赖图、并行泳道和自动接力

核心依赖如下；每个箭头都由就绪事件触发后继，不要求 root 或代理持续占位等待：

```text
node2/3 queue健康 ─┬─> encoder×dataset shard增长 ─> 增量审计 ─> 完整视图发布
                │                                  ├─> 对应 UR-DMU head/QA ─> dense评分
                │                                  └─> 当前方法特征 ────────────┘
XD accepted/quarantine receipt ─────┘

UCF dense/head + 训练侧方法冻结 ─> test压缩特征 ─> official consumer ─> CI/论文表
                                      └──────────────> 短独占效率（含插件开销）

V2+CLIP训练配置登记 ─> 性质/干预探索 ─> root收敛与新方法正式冻结 ─> UCF/XD正式质量/效率 ─> 论文主方法
（可选第二CLIP后端） ─> 兼容性/训练开发审阅 ─> 补充材料或停止
```

| 泳道 | 负责人 | 当前动作 | 产物 | 验收 | 后继触发 |
|---|---|---|---|---|---|
| 基础设施 | `takeover_infra_dispatch` / root 判断 | node1 管理员恢复 deferred；用 node2/3 恢复 TS、清冗余 guard、维护唯一队列 | 节点/进程快照、dispatch 状态、stop/增长 receipt | owner/argv/startticks、两次有效增长、无重复 claim；node1 不作为生产门槛 | 片段完成即按优先级接力下一 encoder |
| XD 数据 | Luna 执行；infra 维护领取 | V2/VMA/TS 分离 train/test；使用 accepted=3950 与不可变 excluded=4/quarantine 视图 | per-video index、增量审计、quarantine/exclusion receipt | 成员、clip 坐标、SHA/CRC/finite、去重身份、raw→canonical 状态 | encoder accepted train ready 即训练对应 head；test ready 即准备评测 |
| UCF baseline | Luna | V2/VMA 已发布视图 QA；TS full merge 后训练；逐组训练 full1610 head | resolved config、3000-step QA、checkpoint SHA | 1610 覆盖、非零梯度/更新、保存重载、输入/权重身份 | head ready 先 QA 并准备评分链；正式 test 评分等待新方法与矩阵冻结 |
| 方法/批次 | Luna | 复核旧 7/12 产物；按 `user_scope_change` 收窄 VMA/TS 压缩支线；V2 1b 范数/成员偏好审计继续；新 idea 配置先登记、训练侧允许修订 | method manifest、rule/version/actual ratio、配对分数 | 同名单、同输入、质量/CI/失败证据齐全；正式 test 前冻结最终方法 | 完成一臂即完整性检查；V2 审计或新方法正式冻结后由 root 决定扩展 |
| idea 训练 | Astra（V2+一个 CLIP） | UCF/XD 双训练侧探索；核实 CLIP backbone/backend；允许登记后修订候选 | 配置登记、性质/干预产物、新方法冻结文件、训练日志、head/adapter QA、审阅表 | 训练角色隔离、测试暴露披露、无 test 选参、可重载；正式 test 前冻结 | root 收敛后进入 V2+CLIP 正式质量/效率主线；可选第二 CLIP 才进入补充/停止 |
| raw→canonical | Luna/数据审计 | XD dataset-only 时间轴审计 | mapping receipt、坐标/端点证据 | 唯一可核验映射；未唯一则 explicit_blocked | 只在冻结后解除 raw official AP 阻塞 |
| 效率/论文 | Luna；root 复核 | 申请短独占窗口；同步方法图、表格和审阅 prompt | timing receipt、审计 exports、桌面包 | 纯 encoder / adapter+UR-DMU / 端到端边界明确，含插件开销 | 数据齐且不再改规则后冻结数字 |

自动接力规则：

- 一个 shard 发布并通过轻量结构审计后，领取器立即从未完成集合领取下一 shard；不能因为另一个 encoder 仍在长视频上而空卡。
- A100/V100 完成当前 encoder 的有效片段后，先检查当前队列的健康 owner 和失败 claim，再接力下一 encoder；同一逻辑任务只能有一个生效领取者。
- 一个 dataset/encoder 完整视图 ready 后，head 训练立即启动；head ready 后先做 QA 和评分链准备。正式 test 评分统一等待新方法与比较矩阵冻结；特征提取不等待 head，评分不重提特征。
- TS 恢复只能以有效产物增长和新 receipt 作为完成条件；两分钟无增长必须进入诊断，不用 GPU 利用率单点猜测。node1 管理员恢复保持 deferred，不作为 node2/3 生产门槛。
- 每 30 分钟只做低开销结构化巡检：PID/owner/argv、queue/claim、最新 receipt、已发布 index、增量 clip 数、磁盘/RAM/GPU；不对全 NAS 重复 hash，不用 `ls` 文件数当成功，不把跨节点时钟相减算耗时。

## 6. 方法、CLIP 和论文决策规则

### 6.1 证据优先级

观察、匹配对照、独立确认和受控干预仍是方法门槛。已有 pairmean 负例、pairselect 探索和
VMA/TS 反例必须保留，不能从论文包删除；它们只能支持对应实验配置的结论，不能扩大为所有
合并方法都失败或所有 encoder 都有效。

新训练侧探索要显式写出：配置登记版本、登记时间、使用的数据角色、是否在任何 test 之后启动、
参数选择依据、与旧 pairselect 的关系。训练探索可以产生修订版；最终方法和正式格子在 test 前
冻结。既有正式 test 已被查看的范围在论文方法选择和局限中披露；
不允许使用 test 分数挑方法、层、预算、阈值、后处理或 checkpoint。训练内 100 的结果只回答
实现/收敛/方向问题，不回答泛化。

### 6.2 CLIP 最小闭环

1. 三条候选路线均按 ViT-B/16 记录真实 revision、visual 输出、输入预处理和 backend；共享 visual 不写成三个 encoder。
2. 主路线暂定 VadCLIP，依据两数据集权重已就绪和兼容性候选；这不是“已通过”。DSANet 可选；AnomalyCLIP 新增成绩暂停。
3. CLIP 工程 bridge 已在 UCF/XD float32 smoke 验证 `197→99`，但这只是 identity/shape/真实缩短工程证据，不能写成质量兼容。VadCLIP 原生 10-crop 与 OpenAI center 输入存在差异；质量入口须在作者协议 A（原始提取协议可核验）或新 raw representation B（新身份 + 发布 head transfer/refit）之间登记，尚未做质量 claim。
4. 在 UCF/XD 各自使用视频级训练标签 W 做训练侧闭环；不同训练视频/近重复来源角色隔离，full1610/accepted3950 与 test 角色分开。V2+CLIP 是论文主方法线，不是补充候选。
5. 只有原视频路径可复现、训练开发支持、身份和 head 可审计时，第二个 CLIP 才能作为补充；测试集不得用来选“赢家”。若原视频→CLIP 路径无法对齐，保留校准和失败证据，资源转回 V2/VMA/TS dense 与 XD。

### 6.3 质量与效率

质量比较必须匹配同 encoder、同 dataset、同 seed、同输入覆盖、同 readout 和同冻结 dense head。
每个方法报告点差、95% 视频级配对 bootstrap CI（10,000 次、分层、同一抽样 multiplicity）和
实际 token/轨迹数量。非劣判据只写成 `CI_lower(Δ) >= -0.005`；CI 跨边界写“不确定”。

正式效率使用短独占 GPU 窗口，计入插件本身的选择/合并成本，分列纯 encoder、明确 adapter+
UR-DMU 和含解码端到端；报告固定 batch、精度、warmup、同步、latency、吞吐、allocated/reserved
峰值和逐层实际 token 数。研究 hook、完整 attention dump、CPU 诊断导出关闭；共卡、多进程
日志和生产提取耗时不能进入正式速度表。减少 token 不能直接写成减少显存。

XD raw→canonical 时间轴审计是关键路径。作者梯形 AP 与 `average_precision` 分列；若 raw
时间映射仍不唯一，raw official AP 保持 explicit_blocked，不能用 generic AP、总长度切片或
模型分数反推坐标。

### 6.4 文献边界和科学主张

Astra 与 root 已核对 KeepAD（arXiv:2608.03681）、CenterCLIP（arXiv:2205.00823）和
S²Prune（arXiv:2609.01224）的相关机制。KeepAD 已有异常检测中的 2×2 coverage/rescue，
CenterCLIP 已有 medoid 代表选择，S²Prune 已有空间覆盖预算；因此当前 local-coverage probe
只能作为诊断和强对照，不能冻结为论文创新 selector，也不能把“局部覆盖”本身写成首次贡献。
详细边界见 [`findings/codex-idea-20260920/prior-art-boundary.md`](findings/codex-idea-20260920/prior-art-boundary.md)。

idea 线仍须从正常—异常性质、匹配/独立确认和受控干预产生新的可检验决策差异；文献相似性、
小样本 coverage 误差、旧 pairselect 成绩或训练内分数都不能承诺新 selector 已定型或已泛化。
没有足够证据时，保留 study/边界结果身份，停止无效全量提取和重复校准。

当前最强的下一问题是 [`F06-update-ratio-actionability.md`](findings/codex-idea-20260920/F06-update-ratio-actionability.md)：
F01 已确认的 V2 block5 relative-attention update 是否预测真实后缀压缩敏感性。F06 当前身份仍为
`engineering_probe_authorized_not_property_confirmed`；2×2 coverage 只作诊断/强对照，不冻结为新
selector。需先完成工程等价、性质确认和同实际 token 多重集的干预对照，才由 root 决定是否进入正式方法。

F06-S1 的 V2+CLIP 真实 16-window 配对干预已完成，但没有支持稳定主信号；因此停止 F06 扩量，
不是停止 idea 主线。它仍不是方法冻结或科学性质确认，不能把 `197→99` 或 ratio 重构误差写成新性质成立。

下一优先是 [`F07-cohort100-norm-audit-spec.md`](findings/codex-idea-20260920/F07-cohort100-norm-audit-spec.md)：
优先复用已付费的 7 臂、100 视频、36,470 clips 做 CPU pair-member norm audit，不新增 GPU 提取；
当前状态为 `cpu_analysis_complete_blocked_cache_identity`，结果目录 `outputs/icassp2027/idea/f07`：100×36,470×7 已完成 CPU 审计，但
pair_fixed、random_member、reverse 三关键臂 pooled 逐值完全相同，`causal_member_preference_status=blocked_cache_identity`，不能把差值写成成员范数偏好增量。N 阶段不批准；Astra 只查运行参数/控制是否生效，不重跑 100。F08
[`F08-merge-scale-boundary.md`](findings/codex-idea-20260920/F08-merge-scale-boundary.md) 只作 ToFu/MLERP
既有尺度对照，未派 GPU。F09
[`F09-temporal-first-bounded-decision.md`](findings/codex-idea-20260920/F09-temporal-first-bounded-decision.md)
是待 root 收敛的时间候选，尚未冻结、尚未启动 GPU，不写成功。

当前参考保留版 v1 不执行“低变化整轨迹全部删除”：相邻 native 时间的 reference 偶数轨迹保留，
奇数 target 按 pre-position embedding 时间差做有界选择；先过 CPU/config/identity gate，明早 08:00
严格 Go/NoGo。V2/CLIP 的 native 时间粒度、CLS、prefix/suffix 和实际 token 数分别记录，不把
V2 tubelet 与 CLIP 逐帧 attention 域强行混合。
F09 v1 的 CPU 规则/config/identity 准备已由 root 派给 Luna `takeover_temporal_gate`；尚未授权 GPU，
仍受 F07 结果和明早 Go/NoGo 门控制。

VideoMAE 进入独立 train-side diagnosis：检查权重、输入、官方 readout、pooling、head 和时间轴。
QA 通过不等于质量通过，不能据旧 test `0.8106` 直接断定非 bug；诊断完成后由 root 决定恢复生产或做最小修复。

## 7. 保留、暂停与删除边界

### 必须保留

- V-JEPA 2 权重、缓存、观察、工程验收和退出检测原因；VMA/TS 的 dense 边界和失败；pairmean、pairselect、随机/均匀对照、旧 7/12 批次和 1b 产物；旧契约、旧 SHA、旧 run、失败日志、test 暴露时间线。
- 正在写入但尚未完成的 partial、重复物理副本及其身份证据，直到 owner 审计判断可回收；不以“看起来重复”直接删除。
- node1 GPU7 隔离证据、XD excluded=4 原文件和不可变 quarantine/排除清单；旧修复失败和搜索证据保留，不再创建新的修复副本。

### 可以暂停/终止

- 已确认没有有效后继、会重复写同一逻辑任务、或仅等待已停止队列的本轮冗余 guard/controller；先核对 owner/argv/进程组/输出价值，动作后保存 termination receipt。
- 已被本 V3 关键路径取代的新增 dev、重复校准和旧探索派发；保留它们的目录、日志和结果，不删除历史。
- 无法通过现有数据/身份契约且不再服务论文主线的 CLIP 第二补充线；必须先写失败原因和转移的资源，不将其写成完成。

### 禁止

不杀其他项目或他人健康任务；不重装共享环境；不执行 reset/clean/强制 checkout；不删除旧缓存、
结果、失败证据或 VJ 资产；不覆盖运行中的 bash；不改源码、协议 JSON、旧 SHA 绑定契约来绕门。
用户已授权的研究服务器修复、同步新的研究代码快照、提取/训练调度可以执行；禁止未授权的公开
发布、生产系统部署、commit、push 和自动投稿。确需删除明确无用的新脚本时，先检查依赖和运行
状态，使用可恢复方式，并在回执说明目标绝对路径、理由和可恢复性。

## 8. 每轮交付格式和完成条件

每轮 `progress.md` 更新按以下顺序：

1. 训练：每个 encoder×dataset 的数据、视图、head、QA、checkpoint；
2. 质量：generic/official、训练内/独立测试、帧级/视频级及已接触 test 的披露；
3. 方法：版本、臂/档位完成数、实际保留率、Δ/CI、负例和下一触发；
4. XD：train/test 分开，去重 clip、发布视频、有效增量、accepted/excluded 回执和 raw mapping；
5. 资源：节点、GPU、PID、guard/extractor、owner、下一片；
6. 论文：主表、主图、审阅 prompt、缺失值和主张边界；
7. 本轮改动文件、真实命令、通过/未运行测试和未确定项。

论文交付至少需要：

- V2/VMA/TS 两数据集可追溯 dense 提取和匹配 UR-DMU 头，XD 3950/3954 隔离如实披露；
- V2 主适配与一个 CLIP（若满足条件）的新版本身份、训练侧双数据集证据、已接触 test 的过程披露；
- direct_insert 主比较、必要 refit 补充、正确的 UCF ROC-AUC/XD 梯形 AP、视频级 CI 和实际效率；
- raw→canonical 时间轴和 XD excluded=4 接受回执的独立审计，所有失败/负结果/不确定性保留；
- 服务器核心代码、关键文档、回执索引和审阅 prompt 的桌面交付包，不含凭据和大资产。

这些条件未齐之前，不能写“论文实验完成”“方法跨 encoder 成立”“XD 已完整验证”或“显存降低”。
达到条件后由 root 逐项核对导出来源和会议政策；本文件不授权自动投稿。

## 9. 当前下一步（按就绪事件执行）

1. `takeover_infra_dispatch` 继续用 node2/node3 推进 TS/VMA/V2，维护自动后继和冗余 guard 回执；node1 管理员恢复保持 deferred，不让 node1 阻塞生产。
2. XD 使用不可变 quarantine 的 accepted 3950/3954 视图作为本轮完整可用训练集合；停止四文件修复搜索/下载/等待新契约，继续三 encoder 的头和质量链。
3. V2/VMA UCF full1610 formal head 已 QA/reload 通过；TS 继续当前 retry01 full merge，发布 1610 视图后再启动 formal head，不把旧 fit1288 视图升级。
4. 完成 V2 1b fixed-member/random-member/reverse 的身份审计；VMA/TS 旧压缩支线按 retire receipt 保留原身份和反例，不再机械补齐三家。旧 pairselect 结果只写历史，新 idea 每个训练配置先登记，训练探索允许修订，最终方法在正式 test 前冻结。
5. Astra/idea 继续 V2+CLIP 训练侧主线；F06-S1 的 8 视频×2 窗口 fixed/random 已完成且无稳定主信号，停止扩量但不停止 idea。VideoMAE 诊断先检查权重/输入/readout/pooling/head/时间轴，再由 root 决定恢复或最小修复；CLIP A/B 质量入口保持未冻结。
6. head ready 先完成 QA 和评分链准备；等 idea 收敛、方法/比较矩阵冻结后统一申请正式 test 评分和短独占效率窗口。XD raw→canonical dataset-only 审计继续，未通过审计的 official AP 保持 blocked。
7. 每个 ready 事件都自动触发下游任务；任何空等、重复派发、资源撞车或无增长都要在下一轮报告中明确负责人和阻塞证据。

本计划的真实完成状态以 `progress.md` 顶部当前状态和对应 receipt 为准，不以本节文字或旧文档中的预计时间替代现场证据。
