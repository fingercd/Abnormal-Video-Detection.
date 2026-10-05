# 03｜从已跑通 encoder 到观察、插件、benchmark 和提交的执行手册

> 前提：用户已经搭好环境，主流 encoder 已跑通；不重新 bootstrap，不无故升级 CUDA/PyTorch/Transformers。
> 本文件中的“现有命令”已按上传 `src/vadbench/cli.py` 或服务器脚本核对参数，但没有在用户服务器执行。
> `python -m vadbench.paper ...`、新 dense sampler、XD importer 等明确标为“待实现”，必须由 Codex 实现和测试后才能运行。

## 0. 先后顺序与并行关系

```text
保存当前工作树/保护清单 → 论文 active profile → 资产盘点/缺失数据补齐
                                         ├→ 正常—异常观察 → 候选性质确认
                                         └→ dense 弱监督检测基线
两条证据汇合 → 一个最小 token 操作 → 在线低成本插件 → 多模型/数据验证
→ 真实质量—效率结果 → 论文图表导出 → 人工论证与投稿核查
```

数据准备与只读探针开发可以并行。不要等重构全部完成才看第一张性质分布图；也不要在第一张图之前开发最终 selector。

## 1. 第一步：记录现在能用什么，而不是重建环境

### 1.1 建立本轮状态

按文档 02 创建 `projects/icassp2027/`、保护清单和新结果根。保存未提交改动与未跟踪文件的清单；源码快照不能只记录 Git HEAD，因为上传概览明确当前工作树有未提交修改。

用户报告的已跑通状态记为 `user_reported_ready`；找到对应真实日志后链接为 `run_verified`。历史 9 月 12 日故障留在历史记录，不作为今天失败的证据。

### 1.2 当前环境只读检查，现有命令

下例变量需指向已经跑通的解释器和工作区，不创建新环境：

```bash
export VAD_ROOT="${VAD_ROOT:-/users/fotile/VAD}"
export VAD_PY="${VAD_PY:?请设置为已跑通环境的 Python 绝对路径}"
cd "$VAD_ROOT"
test -x "$VAD_PY"
export PYTHONPATH="$VAD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

pwd
hostname
git status --short
"$VAD_PY" -c 'import sys, vadbench; print(sys.executable); print(vadbench.__file__)'
"$VAD_PY" -m vadbench --help
"$VAD_PY" -m vadbench encoders list
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader
df -h
```

现有环境分组保留；不同 encoder 可由既有隔离 runner 使用对应解释器/overlay。不要把所有依赖硬塞进一个新虚拟环境。记录 GPU UUID 和实际使用的解释器，不能只记 `cuda:0`。

**验收：**active 四个模型的加载代码、权重目录、解释器和可用 GPU 都有明确记录；未确认内容列为待确认，不改登记状态冒充成功。

## 2. 第二步：数据下载到服务器与资产分层

### 2.1 数据优先级

主数据为 UCF-Crime 和 XD-Violence（本论文先 RGB-only）。UCF 先启动闭环；XD 原始视频准备同时进行。ShanghaiTech 弱监督划分是可选扩展，不阻塞核心两集。

必须有原始视频才能研究 encoder 内 token；下载 I3D/CLIP 的预提取特征不能代替中间层观察，也不能证明视觉编码器加速。

### 2.2 先盘点已有资产

对每个数据集记录原始路径、授权来源、视频文件数量、总字节数、可解码数量和清单覆盖。上传代码包的 `data/` 仅保留占位，这是打包排除规则，不证明服务器没有数据。

四个现有 encoder 权重已能运行时不重新下载。若 revision 与登记不同，记录当前 revision；不要悄悄更换为新权重再沿用旧 fingerprint。

### 2.3 存储建议

```text
<已有数据盘>/datasets/ucf_crime/           # 原始视频，只读使用
<已有数据盘>/datasets/xd_violence/        # 原始视频，只读使用
<已有数据盘>/vad_icassp2027/features/     # pooled 特征和有限研究数组
<已有数据盘>/vad_icassp2027/runs/         # 不可覆盖运行
<工作区>/data/raw/ucf_crime               # 指向已有数据目录的软链接
<工作区>/data/manifests/icassp2027/       # 小型清单，按仓库策略管理
```

不要默认所有资产放 `/users/fotile` 家目录。创建软链接前检查已存在路径；禁止 `ln -sfn` 静默替换。确认数据盘剩余空间，再获取原始视频。

当前项目文档规定 node2 为联网出口、node3 离线执行；按现有部署事实复核后沿用。下载源、登录和许可需要合法可用入口，Codex 不得绕过访问控制或自动替用户接受未知许可。

### 2.4 下载来源与本轮核查边界

UCF 官方项目入口为 <https://www.crcv.ucf.edu/projects/real-world/>；本轮网页抓取该入口失败，不能据此声称某个视频下载镜像现在可用。优先使用用户已取得的授权视频或官方作者页面导向的可用入口。

UCF split 与测试标注的固定来源、commit、SHA-256 已在快照 `registry/datasets.yaml`，可以据此获取小文件并校验。原始视频不在该 YAML 的小文件清单中。

XD 官方项目页 <https://roc-ng.github.io/XD-Violence/> 当前列有原始视频、测试标注和多个存储入口，部分旧训练入口标为停用。先确认具体链接，再分块下载；不得宣称已实际下载完整视频。不要误下成 `V1.0 Features`。

**验收：**下载清单、SHA/来源、失败文件列表和断点记录齐全；没有“只有几十个工程视频却标 full UCF”的目录。

## 3. 第三步：导入清单、审计并锁定标签权限

### 3.1 UCF 现有命令

变量 `UCF_ROOT` 指向已存在的视频根目录。下面的 split 路径来自仓库登记；缺失时先从登记固定 URL 获取并核对摘要。

```bash
export UCF_ROOT="${UCF_ROOT:?请设置原始 UCF-Crime 视频目录}"
test -d "$UCF_ROOT"
test -f data/splits/ucf_crime/Anomaly_Train.txt
test -f data/splits/ucf_crime/Temporal_Anomaly_Annotation.txt

"$VAD_PY" -m vadbench manifest import-ucf \
  --dataset-root "$UCF_ROOT" \
  --train-split data/splits/ucf_crime/Anomaly_Train.txt \
  --temporal-annotations data/splits/ucf_crime/Temporal_Anomaly_Annotation.txt \
  --output-dir data/manifests/icassp2027/ucf_crime \
  --require-files --probe-video-info

"$VAD_PY" -m vadbench manifest validate \
  data/manifests/icassp2027/ucf_crime/train.jsonl \
  --dataset-root "$UCF_ROOT" --require-files

"$VAD_PY" -m vadbench manifest audit-ucf \
  --dataset-root "$UCF_ROOT" \
  --train-manifest data/manifests/icassp2027/ucf_crime/train.jsonl \
  --test-manifest data/manifests/icassp2027/ucf_crime/test.jsonl \
  --output outputs/icassp2027/assets/ucf-audit.json
```

先检查 importer 实际输出名字；当前约定为 train/test.jsonl。正式审计预期官方 train 1610、test 290；依据是上传 `registry/datasets.yaml`。少文件不能通过改 expected count 冒充正式全量。

导入阶段允许解析测试时间标注来建立审计数据，但研究探针、校准器和训练器不得使用这些标注。用独立角色访问权限实现，不靠注释自律。

### 3.2 必查内容

视频 ID 跨 split 重叠；同源切段/近重复；原帧数、实际 FPS 与时间长度；可变帧率处理；视频打不开；标签缺失；时间端点；空异常区间；negative 未标注是否真的正常；padding 重复帧。

UCF 的官方 1-based inclusive 端点转换为内部 `[start-1, end)`，保留原始字段。其他数据集不要套用这个转换。

### 3.3 创建研究分区，待实现

只从官方 train 划分 fit/confirm/select，按文档 01 的规则建立 explore 子集。生成 `cohort_index.jsonl`、视频 ID 锁和来源摘要；禁止 clip 级随机拆分造成同一视频两边出现。

C2 没有可靠时间标注就记 unavailable。可选人工/外部诊断标注单独存，不能混入严格弱监督训练/选择目录。

### 3.4 XD 是新能力，不是假装现成命令

上传快照中只有 UCF 专用 importer 与 official evaluator。Codex 需新增 XD 适配，至少用真实正常、异常、多区间、短视频样本验证时间轴，并与官方标注解析结果核对；为 XD 建立独立 protocol ID，保留 RGB-only。

**验收：**训练组件无法读测试真值；正常—异常探针知道自己在比较视频级组还是事件级组；两个数据集的坐标规则分别有测试。

## 4. 第四步：最小论文重构＋四个模型观察能力回执

完成文档 02 的 M0/M1。不要先改公共目录布局。

现有跨环境模型复核脚本示例：

```bash
# 在项目适用的基础解释器执行；脚本按既有 registry 调度模型环境。
# REAL_VIDEO 为合法、已存在的本地原始视频；不要使用虚构路径。
export REAL_VIDEO="${REAL_VIDEO:?请设置一个真实视频的路径}"
"$VAD_PY" scripts/server/run_native_encoder_matrix_v2.py \
  --id videomaev2 --id timesformer --id vjepa2 --id videomae \
  --video "$REAL_VIDEO" --device cuda:0
```

脚本有 GPU 可用性及输出目录保护，失败时读实际原因，不通过删保护代码强跑。已有足够新回执时不必重复四模型完整冒烟；但**新 probe bridge 的 identity 等价测试必须补做**。

回执包括：真正加载的类、block 路径、QKV 位置、CLS/pooled、tubelet 网格、采样帧、dtype、位置方案、no_grad 范围、observer 开关等价、支持压缩的位置。

**验收：**每模型 `probe_ready` 有证据；`reduction_ready` 尚可为 false，不能混为一个 ready 标志。

## 5. 第五步：4＋4 视频调试，再做正常—异常探索

### 5.1 第一批先检查什么

固定 4 个 V+ 与 4 个 V−，每个 8 窗口，用 VideoMAEv2 的 embedding 和 4 个相对深度位置采集一级探针。检查：token 图映射回正确视频位置、padding 不参与、LayerNorm 位置正确、attention 行和为 1、特殊 token 不混到空间统计。

对一段视频重复采集，结果应在固定随机种子和已声明数值容差内一致。打开 observer 与关闭 observer 的 pooled output 应一致；不同 attention 实现有数值差异时记录上界，不能把差异归因于方法。

### 5.2 正式探索的最小交付

扩到固定 explore 视频，生成文档 01 的三种表和四类图。先对比正常/异常，再看混淆因素；不要只计算全体视频共享冗余而没有类别条件结果。

先报告方向、重叠和样本量，再做候选筛选。最后输出最多两份 finding card。

### 5.3 新论文入口的命令形状，待实现

```bash
# 以下为实现目标，当前快照不存在 vadbench.paper：
"$VAD_PY" -m vadbench.paper status \
  --project projects/icassp2027/profile.yaml

"$VAD_PY" -m vadbench.paper probe \
  --project projects/icassp2027/profile.yaml \
  --suite configs/papers/icassp2027/suites/probe-pilot.yaml \
  --dry-run
```

所有新 stage 支持 dry-run，显示输入视频角色、模型、输出路径、可能访问的标签级别和成本样本数。dry-run 不加载大权重，不创建伪成功结果。

## 6. 与观察并行：建立真正能检测的 dense 基线

### 6.1 主头先简单、可解释、可复现

先固定一个 TopKMIL 或现有 AttentionMIL；使用视频级监督。主头训练稳定后，再在主 encoder 上验证第二个头即可。encoder 采用 Transformer 不代表检测头必须叫 Transformer；当前 gated attention 不是时序 Transformer。

若确实需要全 Transformer 检测头，可新增很薄的 temporal Transformer＋MIL，但它是新模块，必须所有压缩方法共享同一训练配置。不要把 head 的升级收益当作 token 插件收益。

### 6.2 现有配置只作跑通，不直接当正式最优配置

上传示例 `epochs=1, batch_size=2, 32 segments` 是参考配置。先做小规模收敛检查，再固定训练轮数、优化器、head 宽度、dropout、TopK 和 seed。当前 runner 不会自动选择最佳 checkpoint，也没有现成 scheduler/early stopping；需要时新增并测试，不在 YAML 写入未被消费的字段后假装生效。

若使用 ranking loss，要有正负 bag 的 batch 保障；否则可先使用现有 video BCE 形成稳定闭环，再单独增加平衡采样。所有压缩基线同步采用同一训练协议。

### 6.3 现有阶段命令模板

`CFG` 指向已经由 Codex 生成、校验且能执行的**完整普通实验配置**；不是 paper profile。`FEATURE_ROOT` 指 FeatureStore 根，里面有 `index.jsonl`；`HEAD_CKPT` 从训练输出 JSON 的 `checkpoint_path` 读取，不能猜文件名。

```bash
export CFG="${CFG:?完整实验配置路径}"
"$VAD_PY" -m vadbench config validate "$CFG"

# 只适用于配置指定的现有采样方式；插件/新 dense sampler 用 paper 新入口。
"$VAD_PY" -m vadbench extract -c "$CFG" --split train
"$VAD_PY" -m vadbench extract -c "$CFG" --split test

export FEATURE_ROOT="${FEATURE_ROOT:?实际 FeatureStore 根目录}"
"$VAD_PY" -m vadbench train -c "$CFG" \
  --features "$FEATURE_ROOT" --device cuda:0

export HEAD_CKPT="${HEAD_CKPT:?训练输出的实际 checkpoint_path}"
"$VAD_PY" -m vadbench predict -c "$CFG" \
  --features "$FEATURE_ROOT" --checkpoint "$HEAD_CKPT" \
  --output outputs/icassp2027/baseline/predictions.jsonl --device cuda:0

# 此步只在预先锁定的方法/评估阶段执行，不用于早期逐日调参。
"$VAD_PY" -m vadbench evaluate -c "$CFG" \
  --predictions outputs/icassp2027/baseline/predictions.jsonl \
  --dataset-audit outputs/icassp2027/assets/ucf-audit.json \
  --protocol official \
  --output outputs/icassp2027/baseline/metrics.json
```

在一轮早期工程验证中，可在非正式开发子集完成训练预测和投影测试，不读取官方 test 帧级指标。正式命令展示的是接口，不授予随时用测试集调参的权限。

### 6.4 训练稀疏取段和测试 dense 覆盖，必须显式兼容

当前 extractor 只支持均匀 segment 或真实 streaming；不支持固定 encoder 的通用 dense sliding windows。32 个中心 clip 回填整段可以满足区间覆盖，但不表示密集观察了整个视频。

新增 dense 测试协议后，训练 32-bag 与测试 dense 的采样摘要不同。当前 fingerprint 包含 sampler，head 绑定可能拒绝这种差异；不能伪造指纹。

新论文兼容层应登记：训练采样协议、允许的测试采样协议、相同 clip 定义和特征语义、实际 evaluation sampling identity。dense 与 compressed 的**测试采样必须完全相同**，而训练与测试采样可以按预先声明的方案不同。`expected_clips` 在测试阶段也必须正确处理变长序列，不能继续要求每视频只有 32 条。

### 6.5 特征存储

正式 detector 训练优先缓存 pooled feature；中间 token 只给 explore/confirm 保存有限样本。当前 FeatureStore 可能仍写入 token 数组，`feature_level=clip` 是训练读取选择，不自动等于磁盘只保存 pooled。如新增 pooled-only writer，应显式记录 readout 并测试一致性。

一次统计少量视频的解码时间、encoder 时间、存储字节和总窗口数，用实测吞吐计算每个计划任务的 GPU 小时和磁盘需求。不要凭 A100 数量预先承诺全矩阵很快。

**验收：**dense head 在开发数据上学习有效，正负标签、分数方向和帧回填正确；正式评测协议冻结；结果可追溯。

## 7. 第六步：确认性质，选择一个最小 token 操作

从 explore 最多选两条性质，固定定义和方向，用 confirm 的独立视频和第二 encoder 复核。不是所有维度都要有差异。

每条性质测试一个最小操作：例如某统计引导组内合并、关系约束代表选择、时间稳定组聚合、层间预算或头协同引导 token 选择。每次与相同 token 数、同插入位置的随机/均匀/通用策略比较。

初始预算可用最终 token 保留比 1.00/0.75/0.50，再加 0.25 做压力测试。比例只作为实验档位，不能当作已知有效率；记录每层 N，不把最终保留比当 FLOPs 比。

### 不预设训练路线

能直接计算就先规则；需要正常参照则校准；若性质清晰但在线识别昂贵，再训练小预测器。额外训练必须登记数据来源、模型、耗时和参数迁移方式。

**验收：**至少一条性质有独立证据，并在同预算操作上支持实际决策。只见正负分布差异但操作不奏效，不直接进入主表。

## 8. 第七步：实现正式插件与真实效率路径

固定一个主方法、最多一个互补组件。实现同一核心算法＋四模型桥。先检查 keep_ratio=1、质量/位置/padding、真实后续 token shape，再计时。

计时分三层：tensor-only encoder；adapter 含预处理；raw-video end-to-end 含解码和检测头。禁止从缓存 features 的 head FPS 推断视频 encoder FPS。

晚层压缩未降低整网峰值、K/V-only 未降低 MLP、路由过贵等都是实测事实；可以据此改变压缩位置或简化代理，不需要把它们当论文预设动机。

**验收：**质量约束和净速度/显存结果都满足事先定义的口径；输入自适应方案报告正常、异常及总体成本，不能只计正常视频的快路径。

## 9. 第八步：控制实验矩阵

建议的分层矩阵：

| 范围 | 实验 |
|---|---|
| 主 encoder × UCF | dense、随机/均匀、ToMe、一个视频合并强基线、ours；3–4 个预算 |
| 主 encoder × XD | 同一协议下的 dense、最强通用基线、ours；主要预算 |
| 其他 3 个 encoder | dense、最强基线、ours；先 2 个预算；两集优先形成一致结果 |
| 主 encoder 的第二检测头 | dense、最强基线、ours；固定主要预算 |
| 消融 | 去掉性质信号、随机替代同数 token、只改单层/预算、必要组件消融 |
| 关键配置多种子 | 3 个独立 head 训练种子；确定性 encoder 不重复制造“独立种子” |

direct_insert 与 refit_head 分表。探索阶段只用一个 seed，最终关键结论补多 seed；不要四模型×所有头×所有数据×所有超参一次全跑。

模型 × 配置 × 数据等组合的运行调度，使用明确的任务清单和 GPU lock。单个 FeatureStore 的 index 不能被多个进程无锁追加；按 run/encoder/method/split 分片，完成后验证合并。

## 10. 第九步：4–6 张 GPU 的分工

4 卡时：一张主模型观察和候选；一张 dense/基线特征与 head；一张第二 encoder/confirm；一张 XD 或正式效率复核。6 卡时，额外两张给第三/第四 encoder 和关键种子。

同一张卡不同时跑速度测试和其他任务。下载、解码及数据准备在 CPU/I/O 侧调度，避免重复读取造成所有 GPU 等待。DDP 不是默认选项，多个独立单卡任务通常更方便追溯；是否需要 DDP 由实际训练规模决定。

## 11. 七天阶段计划：按门禁推进，不是成功时长保证

| 阶段 | 主要工作 | 可以继续的门槛 |
|---|---|---|
| Day 0 / 9 月 17 日 | 保护清单、active profile、数据盘点、4＋4 探针 | 数据角色正确，真实中间张量和位置图可用 |
| Day 1 | 一级正负分布、dense 基线、XD 资产准备 | 有有效分布和候选，不是只见模型能跑 |
| Day 2 | 混淆对照、第二 encoder、独立确认、小操作 | 至少一条性质有任务操作价值 |
| Day 3 | 一个主插件、真实变长路径、两预算 | 质量与路由开销可接受；identity 测试过 |
| Day 4 | 两数据集、跨 encoder、消融 | 主结果范围明确，方法定义停止随意变化 |
| Day 5 | 多 seed、端到端效率、图表来源检查 | 每个核心数字都有有效 run 与协议 |
| Day 6 / 内部 9 月 23 日 | 人工论文论证、排版、引用、提交信息复核 | 完整可提交版本，不再增大模块 |

官方截止日期和页面冲突见文档 05；不要把本表当成精确到时区的官方倒计时。

如果某门禁没过，缩减 encoder 的预算数量或可选数据集，不删除反例或拿子集冒充全量。最终以真实完成范围写论文。

## 12. 最后一阶段：只从结果生成写作资产

使用 `outputs/icassp2027/exports/` 中已审核 run 生成表、曲线和 claim registry，导出到 `paper/icassp2027/`。人工写论证、核查引用和数值。缺失结果用明确占位，release 模式拒绝占位。

提交前检查：方法和超参冻结记录、官方测试访问记录、图中病例选择规则、数据使用权限、dense 与 compressed 的输入一致性、代码/权重/数据摘要、所有作者及官方投稿要求。

详细结构见 [05 论文工作区](05_ICASSP_PAPER_WORKSPACE.md)，实验风险见 [06 实验规则](06_EXPERIMENT_RULES_AND_PITFALLS.md)。

## 13. Codex 现在应执行的第一条任务

> 在当前工作树创建论文 profile 和保护清单，不修改 inactive encoder。沿用已跑通解释器与权重，检查 UCF/XD 原始视频是否已在数据盘。用固定的 4 个正视频和 4 个正常视频实现 VideoMAEv2 一级只读探针，输出真实架构回执和正常—异常分布。没有候选性质确认前，不实现最终 token selector；没有新命令实现前，不声称它已可运行。
