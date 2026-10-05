# AGENTS.md — ICASSP 2027 WSVAD 正常—异常性质驱动研究

> **2026-09-21 最新 GPU 授权：** 用户在知晓其他项目占用后明确要求“直接16张卡开始提取”。允许在
> node2/node3 全部16卡共享空余显存；仍不停止或修改其他项目进程。保留已有正常writer，每卡按实际
> 可用显存限制新进程用量，不以重复堆进程冒充16卡有效产出。该授权覆盖此前“不在foreign占用卡新派”限制。

> **2026-09-21 训练范围补充：** 所有正式训练必须使用对应 encoder、对应数据集的完整训练集；
> 64 normal + 64 anomalous 视频只允许作为训练集内部的快速 idea/压缩规则筛选池，不能替代正式
> full-train。每个 encoder×dataset 使用独立检测头，不跨数据集共享；每个 optimizer step 的
> 64 normal + 64 anomalous bags 是采样预算，不代表训练只看 64+64 个视频。V2/VMA 的 UCF
> full-train head 已有通过回执；XD accepted3950 和 TimeSformer full-train head 必须在完整视图、
> 合同和 QA 通过后再训练。若 CLIP 进入主线，也必须分别使用 UCF/XD 全部可用训练视频训练，
> 不能把 64+64 screen head 当正式结果。时间估算与已知实际回执记录在
> `projects/icassp2027/TRAINING_SCOPE_AND_RUNTIME_ESTIMATE_20260921.md`。

> **2026-09-21 00:37 用户最新调整：** 每10分钟由主控直接简查提取与GPU接力，不再例行派发agent；
> 定时提示词保持简短。已有必要安全修复收尾后交主控。明早08:00有界idea安排保留，提取故障优先。

> **2026-09-20 22:26 用户最新补充：** 今晚只继续提取；9/21 08:00（Asia/Hong_Kong）起可开始有界
> idea 探索，按 `FEATURE_EXTRACTION_PLAN_20260920.md` 第 10 节执行。默认主控加一名执行代理，必要
> 方法审阅顺序切换，不恢复多代理并行。研究目标为有论据的真实加速与质量保持；必要基线公平匹配，
> 正式 test 不用于选择方法。此补充仅覆盖下方“永不自动研究”的时间范围，其他保护规则继续有效。

> **2026-09-20 22:05 最新范围覆盖：只完成特征提取。** 当前唯一执行入口是
> [`projects/icassp2027/FEATURE_EXTRACTION_PLAN_20260920.md`](projects/icassp2027/FEATURE_EXTRACTION_PLAN_20260920.md)。
> 只保留一个提取调度代理，使用 node2/node3 完成 V2/VMA/TS 的 XD accepted3950 train 与 800 test，
> 复用已有 UCF 特征；不自动启动新 head、idea、文档打包或 node1 排障。下文研究安排及 V3 的并行排期
> 暂停，保留历史身份；与本条冲突时以本条和用户最新要求为准。

## 0. 2026-09-20 当前执行覆盖

当前唯一执行计划是 [`projects/icassp2027/RESEARCH_PLAN_V3.md`](projects/icassp2027/RESEARCH_PLAN_V3.md)，
当前状态与现场回执导航在 [`projects/icassp2027/progress.md`](projects/icassp2027/progress.md) 顶部。
本节在与下方旧的 2026-09-18/19 具体排期冲突时优先；旧计划、旧回执、协议 JSON 和 SHA 绑定契约
仍保留历史身份，不能覆盖或改写。

当前论文执行固定三条 dense encoder 线：`videomaev2`、`videomae`、`timesformer`，在 UCF/XD
分别完成 dense 提取和匹配 UR-DMU head；任一 encoder/dataset ready 即独立推进，不设全体完成屏障。
V-JEPA 2 已退出检测实验，但其权重、缓存、观察、失败和退出证据必须保留。当前主适配只要求
VideoMAEv2 + 一个核实过 backbone/backend 的 CLIP；VadCLIP、DSANet、AnomalyCLIP 三条候选均为
ViT-B/16，共享 visual 不能写成三个 encoder。主路线暂定 VadCLIP（两数据集权重已就绪但仍须兼容性
核验），DSANet 可选，AnomalyCLIP 新增成绩暂停；原生 10-crop 与 OpenAI center 先做训练侧对齐。
第二 CLIP 只有原视频路径和训练开发证据都成立时才做补充。

普通执行由 Luna 最高强度完成，idea 双数据集训练侧探索由唯一 Astra medium 专职，root 负责规划、
判断和验证。XD original declared=3954、accepted=3950、excluded=4 已作为本轮完整可用训练集合，
不再等待修复、下载或新 3954 契约。node1 管理员恢复已 deferred，node2/node3 继续生产；TS 自动接力、XD accepted
3950 训练集合、A100/V100 完成片段后的跨 encoder 接力是当前关键路径。旧 pairselect 预登记属于
历史；VMA pair/TS random 已按 `user_scope_change` 真实 SIGTERM 停止且无复发，VMA random/TS
uniform/pair 自然完成的原身份保留；已有 partial、结果、失败和反例保留。VideoMAE 当前另开
train-side diagnosis，不据旧 test 断定 bug。新 idea 的每个训练配置先登记并允许根据训练侧证据
修订，最终方法和正式 test 格子在测试前冻结；既有 test290 诊断已接触的事实如实披露，禁止据
test 调方法、层、预算、阈值或 checkpoint。

20:38 fresh 进度需与 20:38 起算的旧 ETA 分开；不能用 V2 速率替 VMA/TS，也不能把 A100 shared
TS038 的 347 秒、8.8 clips/s 试验当正式计时。VMA 的独立诊断未证实 invalidating bug，但 pooled
lag-1 风险保留；F06-S1 停止扩量不等于 idea 停止。F07 CPU audit 已完成但三关键臂为
`blocked_cache_identity`，不能写正面成员偏好，N 阶段不批准；F08 既有尺度对照和 F09 T/U/R
CPU/config 仍按 V3 的 Go/NoGo 门推进。论文内部截止保持 9/23，CMS HKT 9/24 20:00 不放宽研究门槛。

根本约束仍然有效：不读取 `VAD_Idea/` 决定本轮方向；不杀其他项目、不删旧资产、不重装共享环境，
不改源码/协议 JSON/旧 SHA 以绕门。用户已授权的研究服务器修复、同步新的研究代码快照、提取/训练
调度可以执行；禁止未授权公开发布、生产系统部署、commit、push 或自动投稿。正式效率必须短独占且含插件
开销，多进程日志不入速度表；XD raw→canonical 时间轴、作者梯形 AP 与 `average_precision`
区分、`Δ=method-dense` 的 CI 下界 `>= -0.005` 和训练内 100 不等于泛化均为硬规则。

## 1. 当前任务与研究主轴

本仓库当前 active 任务：比较异常视频输入与正常视频输入在 ViT encoder 内部的 token、分布、时空关系、Q/K/V、attention、CLS/全局读出、跨层及多头行为；由真实性质决定 token 压缩、合并或相关计算共享插件，以检测质量—延迟—显存的实测权衡验证。

**不要把任务改成普通视频压缩，也不要预设“压缩会损伤异常”“异常一定高方差/低熵”“高运动就是异常”。** 每个主要探针必须有正常—异常比较、匹配对照和可检验的插件决策。没有结论前不锁定最终 selector、损失、插入层或论文标题。

不读取 `VAD_Idea/`、旧 CF-Cert-VAD 或旧 playbook 来决定本轮研究方向。这些历史文件原位保留，不删除。

用户已说明环境搭好、主流 encoder 跑通。先复用当前可工作的解释器、权重和服务器配置；历史 9 月 12 日故障不是当前状态。真实执行时补齐版本、设备、输入和日志回执，不重装整套环境。

历史入口为`projects/icassp2027/RESEARCH_PLAN_V2.md`及
`decisions/property-first-method-contract-v3.md`；V2 已由 V3 supersede。稳定的 UR-DMU 结构和
训练字段仍按已核验协议使用，但旧文件中的四 encoder、旧方法列表、旧排期和旧完成判断不能当作
当前范围。主比较继续固定 dense head `direct_insert`，`refit_head` 单列；旧冻结回执保持原身份，
不追溯改写。

## 2. 指南入口与阅读顺序

六份交接文档共同构成本轮计划，本文件是第 4 份。

| 当前工作 | 先读 |
|---|---|
| 任何首次进入项目的任务 | 本文件；[执行流程](docs/icassp2027/03_EXECUTION_RUNBOOK.md) 第 1–3 节 |
| 架构、profile、模型桥 | [02 重构说明](docs/icassp2027/02_ENCODER_SCOPE_AND_REFACTOR.md) |
| 数据分组、hook、统计及性质确认 | [01 观察计划](docs/icassp2027/01_NORMAL_ANOMALY_PROBES.md) |
| 实验调度、特征、训练、计时 | [03 执行流程](docs/icassp2027/03_EXECUTION_RUNBOOK.md)；[06 实验规则](docs/icassp2027/06_EXPERIMENT_RULES_AND_PITFALLS.md) |
| 图表、论证、论文和投稿材料 | [05 写作工作区](docs/icassp2027/05_ICASSP_PAPER_WORKSPACE.md)；文档 06 |

指南中的新目录、接口、命令是待实现设计，除非当前源码和真实命令帮助已证明存在。不能把计划写成已经完成的功能。先确认当前工作树，不只看 Git HEAD；未提交改动也是用户资产。

## 3. Active 范围与保护边界

当前论文 dense runtime ID 为 `videomaev2`、`timesformer`、`videomae`。`vjepa2` 保留为历史/观察
资产并退出检测实验；VideoMAE 与 VideoMAEv2 不宣称为截然不同的架构。CLIP 候选不是未经核验就
加入 active profile 的第五个 encoder。

唯一 active 配置源拟定为 `projects/icassp2027/profile.yaml`。论文命令只调度白名单；现有 `vadbench encoders list` 仍返回完整 catalog。未经明确决策不加入第五个模型。

**其他 encoder 不动：** 不删除、移动、改名其 adapter、配置、注册记录、checkpoint、上游 lock、环境、权重或历史结果；不因为退出 active 队列卸载依赖。全局注册表先保持原样，新增论文 profile 通过 ID 引用。共享代码确需修改时，验证 inactive 的兼容行为，并记录影响面。

不执行 `git reset --hard`、`git clean`、强制 checkout、强制覆盖、无差别删除或格式化。未经当前任务授权，不 commit、push、公开发布、生产系统部署、提交论文、覆盖用户已有数据或更改系统环境；本轮已授权的研究服务器修复、代码快照同步、提取/训练调度属于任务范围。先读当前根文件再合并此 AGENTS；保留仍然适用的用户保护规则。

不在 Git、日志、论文包写入密码、token、私钥或数据下载凭据。视频、权重、第三方代码、大型 token 数组、环境和临时构建不进入 Git。`assets.local.yaml` 只存本机路径并忽略，不存凭据。

服务器沿用现有 node2 联网准备、node3 离线计算的工作方式，实际路径以本机配置为准。不得因旧默认 `.venv` 含旧包而误导入；每次重要运行记录 `vadbench.__file__`、解释器和真实设备。

## 4. 强制研究顺序

1. 工作树保护、active profile、资产盘点和实际架构回执。
2. 固定 4 个异常视频与 4 个正常视频，验证只读探针、token 坐标和输出不变。
3. 优先运行文档 01 的 P01/P02/P04/P07/P10/P11/P13/P16；扩展到独立视频，不靠少数可视化下结论。
4. 输出正常—异常性质图与 finding cards，明确视频级、片段级或空间级证据范围。
5. 对最多两个候选性质做匹配控制和中立的最小操作实验；结果决定压缩单位和插入方式。
6. 先实现一个主插件；只有证据要求才加一个关联组件。共享同一策略接口，模型差异留在 bridge。
7. 质量、真实序列缩短、净计时、峰值显存、第二数据集和跨 encoder 确认。
8. 冻结方法与主张，从审核后的结果生成写作资产，由作者完成论文论证与核查。

实现困难时缩减可选探针、预算或模型组合，不编造成功，不跳过数据/指纹/真实变长测试。允许报告没有观察到差异，不反向寻找唯一好看的样本。

## 5. 标签与数据权限

异常视频的每个 clip 不等于异常 clip，异常帧的每个 patch 不等于异常 token。

在 `protocol.yaml` 明确 W 或 D：W 的方法开发和选择只使用训练视频级标签；D 使用独立的细粒度诊断开发标注时，必须披露它对方法选择的影响，不能称为全流程仅弱监督。官方测试时间标注不能用来挑方法、层、预算、阈值或最佳 checkpoint。

按视频/近重复来源拆分 fit、confirm、select、test；不同片段不可跨角色。正常参考统计只拟合 fit 的正常视频。高分 pseudo clip 必须标 pseudo，不能当真值。优先报告视频级分布和同场景/同运动层次的对比；有独立可靠标注再做事件内外配对。

部署 reducer 不得接收真值、视频级标签、异常类别、带标签文件名、测试汇总统计、未来层张量或完整教师在线结果。只允许当前可获得的模型状态和已冻结的校准资产。

## 6. 模型与张量语义

当前三条 dense 主线是固定 clip encoder；V-JEPA 2 的既有观察资产保留但已退出检测实验。可研究
主线模型的 Q/K/V 激活；没有跨 clip 的可复用状态就不称 decoder KV cache。不得把 HERMES 接回
主队列来凑 KV 实验。

分别记录真实 CLS token、adapter pooling、原生分类头。没有 CLS 的模型将 CLS 探针标 N/A，不人为添加 CLS 改模型。动作类别置信度不是异常分数。

区分 block 输入、LayerNorm 后、Q/K/V 的实际位置处理、attention 后输出投影、MLP 分支和残差后输出。近似均匀 timeline 不能当作真实 tubelet 的空间定位依据。保留真实 `(t,h,w)` 来源、位置索引、mask 和合并质量。

全局 ViT、TimeSformer 的分离时空布局及 V-JEPA 2 位置实现不能强行共用 reshape。桥负责结构约束，策略负责同一规则。任何不支持的位置机制或 ragged 路径应显式失败，不静默还原 dense 后宣称加速。

## 7. 工程分层

新增 `research/` 负责只读观察，`token_reduction/` 负责部署计算，`paper/` 负责本论文配置与阶段编排。不得用 `compression/` 覆盖已有 `compression.py`，也不得用 `resources/` 遮蔽已有 `resources.py`。

继续复用 `ClipBatch`、`EncoderOutput`、manifest、审计、FeatureStore、MIL 训练与严格帧级评测。不要另写一个没有 provenance 的平行评测脚本。

ratio=1/identity 首先证明与原模型输出一致；观察 hook 开关也要有数值一致性测试。清零、输出后删 feature、已算完再求平均都不能宣称 encoder 省算。正式基准关闭分析 hook、昂贵 SVD、CPU dump 和完整 attention 保存，记录插件自身开销。

V-JEPA 2 现有 wrapper 的 `no_grad` 需要在训练内部插件时显式处理；仅设置 `requires_grad=True` 不证明梯度已接通。当前已有弱监督头为 AttentionMIL/TopKMIL，不把 `TemporalSupervisedHead` 用于视频级主实验或称其为现成时序 Transformer。

## 8. 实验身份、采样和结果

每个 run 使用唯一目录、resolved config、数据/模型/代码/环境/校准指纹。失败和未完成也记录，不用旧成功文件代替此次执行。

区分 backbone、representation、training sampling、evaluation sampling 和 head 身份。现有 CLI 将 sampler 纳入 encoder fingerprint；新流程的 train-32→test-dense 和 dense-head→compressed-feature 必须有显式、窄范围的兼容契约。禁止伪造相同指纹或删除原严格检查。所有被比较方法的测试输入、时间覆盖、预处理与 readout 必须匹配。

`direct_insert` 与 `refit_head` 分开；后者为各方法提供相同 head 训练预算。方法通用不等于参数跨骨干零样本迁移。非梯度拟合统计也要标校准，不偷换为 data-free。

UCF 主指标 frame ROC-AUC，XD 主指标 frame AP；实现遵循各数据集明确协议。F1 阈值在允许的验证数据上固定。统计以视频为独立单位，不能把百万 token 当百万独立样本。关键配置补独立 head 种子；方法质量差值、置信区间和可接受容忍度写明。

速度区分纯模型、含 adapter、视频端到端；列出每层 token 数和精度、固定 batch 延迟/吞吐、allocated/reserved 显存。仅改变 batch 上限的结果单列。共享 GPU 上不并行做正式计时。

## 9. 测试、并行和交付

优先运行受影响测试；Python 改动后执行 `python -m compileall src tests`。profile/资源/注册改动补轻量 CLI 和 wheel 回归；token 变化补 identity、坐标、质量、真实 shape、梯度和泄漏测试。完整测试结果标注日期、解释器；未运行必须明确原因，不报历史数字为当前结果。

并行可分为：架构/桥、观察/统计、数据/检测基线、插件/效率、论文资产。一个集成人员负责公共 schema/profile；各任务单独 run 与工作分支/目录。不得多个进程无锁写同一 FeatureStore index 或改同一协议文件。

每轮报告：改了哪些文件；执行了哪些真实命令；哪些测试通过或未跑；产生了哪些性质证据；还有什么未确定。更新 `projects/icassp2027/progress.md`，不是到处生成互相矛盾的“最终计划”。

论文数字只能来自审核 exports，缺失值保持 null/待完成，禁止模拟数值和虚构引文。遵守当前会议 AI 使用政策，Codex 可辅助实验、图表和核查，不能代替作者写出无人核查的重要论证或自动提交论文。

## 10. 当前现场快照（ICASSP 2027 六格压缩矩阵，2026-09-22 下午实测）

> 本节只是状态快照，不是规则；有重大进展时更新日期并整节替换。权威流水账以 `projects/icassp2027/progress.md` 为准。

**压缩特征提取（18 个正式任务 = 3 压缩方法 × 3 个编码器 × 2 个数据集）**
- UCF：9 个任务全部完成（每任务 290 视频，共 2610），均持有 extraction 完成合同（10:00–11:13 发布）。
- XD：9 个任务仍在收尾，已完成 6601/7200 视频（每任务约 89%–96%）；9 个 extraction 进程实测在跑。XD 数字为视频目录数口径。
- 18 路合计 9211/9810 ≈ 94%。

**预测与质量报告**
- 六个 dense 预测全部完成（07:29–07:39）。
- UCF 九个压缩预测已全部完成：V2×3（上午，export 10:12–10:17）、VMA×3（当日 14:00 前后，export 13:54–13:56）、TS×3（进行中，完成后自动 export）。**分数一律未读取、未用于方法选择。**
- 曾失败的 6 个（VMA/TS × UCF）根因：身份核对按原始字节给代码算指纹，VMA/TS 共用的适配器包装文件 dense 侧为 LF 行尾、压缩侧为 CRLF 行尾，内容逐字节一致（diff 为空），被误判为两个代码版本；权重/骨干/处理器/配置/库版本全部一致。
- 处置（用户拍板走豁免，未留决策文档）：`compatibility.py` 的 direct_insert 校验加入**极窄放行表**，仅放行**三对**已证明"仅行尾不同"的指纹——用仓库原装哈希函数复算，68 个（VMA-XD）/135 个（TS-XD）dense 源运行结果与 head 记录值逐一相等，且与压缩侧只差该包装文件 LF/CRLF；其余字段（runtime_id/weights/preprocessing/readout/output_dim/precision/position_strategy）仍严格比对。补丁已同步远端评分快照，原版备份为 `.bak-pre-lineending-equivalence-20260922`。**2026-09-22 16:0x 更正**：本节及 CLAUDE.md、progress.md、STATUS_DATA_SUMMARY_20260922.md、PROMPT_FOR_ADVISORS_20260922.md 原写作"四对"，经从 6 个 dense head 的 `checkpoints/final.pt` 元数据实推 `representation.backbone.code_digest` 确认为**三对**：VMA×UCF `81e31440…↔30fe33c1…`、VMA×XD `9a23c5da…↔30fe33c1…`、TS×两数据集共用 `7f9a3ac6…↔e5fd0bdf…`；V2 两侧同为 `19720d4e…`，原生一致不需豁免。远端放行表实测 5 个唯一摘要、0 条残留兼容报错，与推导一致。18 路中需豁免的是 VMA×6 + TS×6 = 12 路。
- 调度器已重启（attempts 重置）：6 个失败任务自动重投并成功，export 链路随之自动恢复。
- XD 预警已解除：9 路 XD 提取出合同后，9 个压缩预测将自动启动（V2 原生指纹匹配，VMA/TS 走放行表）。

**进行中的守护**
- GPU 显存监控 08:57 启动，汇总脚本 `scripts/icassp2027/summarize_urdmu_gpu_monitor.py` 就绪；等 extraction 全部结束后按 PID/GPU 汇总峰值/均值，只覆盖观测窗口。
- node2 的 8 路补充提取按记录仍在独立输出根运行（本轮未复核）。
- 红线不变：不读正式测试分数、不用测试集选方法或检查点；失败目录保留不覆盖；每 completed run 需完整 provenance/QA。
