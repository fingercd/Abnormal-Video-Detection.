# AGENTS.md — ICASSP 2027 WSVAD 正常—异常性质驱动研究

## 1. 当前任务与研究主轴

本仓库当前 active 任务：比较异常视频输入与正常视频输入在 ViT encoder 内部的 token、分布、时空关系、Q/K/V、attention、CLS/全局读出、跨层及多头行为；由真实性质决定 token 压缩、合并或相关计算共享插件，以检测质量—延迟—显存的实测权衡验证。

**不要把任务改成普通视频压缩，也不要预设“压缩会损伤异常”“异常一定高方差/低熵”“高运动就是异常”。** 每个主要探针必须有正常—异常比较、匹配对照和可检验的插件决策。没有结论前不锁定最终 selector、损失、插入层或论文标题。

不读取 `VAD_Idea/`、旧 CF-Cert-VAD 或旧 playbook 来决定本轮研究方向。这些历史文件原位保留，不删除。

用户已说明环境搭好、主流 encoder 跑通。先复用当前可工作的解释器、权重和服务器配置；历史 9 月 12 日故障不是当前状态。真实执行时补齐版本、设备、输入和日志回执，不重装整套环境。

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

默认 active 四个 runtime ID：`videomaev2`、`timesformer`、`vjepa2`、`videomae`。前三个优先，第四个为同家族稳健性验证。VideoMAE 与 VideoMAEv2 不宣称为截然不同的架构。

唯一 active 配置源拟定为 `projects/icassp2027/profile.yaml`。论文命令只调度白名单；现有 `vadbench encoders list` 仍返回完整 catalog。未经明确决策不加入第五个模型。

**其他 encoder 不动：** 不删除、移动、改名其 adapter、配置、注册记录、checkpoint、上游 lock、环境、权重或历史结果；不因为退出 active 队列卸载依赖。全局注册表先保持原样，新增论文 profile 通过 ID 引用。共享代码确需修改时，验证 inactive 的兼容行为，并记录影响面。

不执行 `git reset --hard`、`git clean`、强制 checkout、强制覆盖、无差别删除或格式化。未明确授权，不 commit、push、部署、提交论文、覆盖用户已有数据或更改系统环境。先读当前根文件再合并此 AGENTS；保留仍然适用的用户保护规则。

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

当前四个模型是固定 clip encoder。可研究其 Q/K/V 激活；没有跨 clip 的可复用状态就不称 decoder KV cache。不得把 HERMES 接回主队列来凑 KV 实验。

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
