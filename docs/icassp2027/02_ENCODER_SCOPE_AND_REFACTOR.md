# 02｜四个 active encoder、兼容式架构重组与论文专用入口

> 本文件是 Codex 的改造说明，不表示已经修改仓库。
> 依据：上传代码快照 `VAD_code_20260917_210441.zip`，SHA-256：`42631065300b7136af0f832f70b37b35d396d9e7e90c68428b10f482cf002db1`。
> 用户当前状态：环境已搭建，主流 encoder 已跑通。以这条最新说明作为启动前提；只补本论文所需运行回执，不根据 9 月 12 日历史故障重建环境。

## 1. 重构目标和不可改变的边界

**对外是当前 ICASSP 论文专用项目，对内复用已验证的数据、模型加载、训练和评测能力。**

“只保留需要的 encoder”解释为：论文首页、配置、任务计划、默认运行和结果表只显示 active 四个；其他 encoder 的源文件、注册记录、配置、权重、环境、锁文件及已有结果留在原位，不删除、不改名、不搬进 archive。

允许大幅重组论文工作入口、研究模块、配置组织和写作目录；不为了目录好看破坏已跑通模型。旧公共 API 与 CLI 继续可用，论文流程经过新入口调用它们。

本轮不读取或采用 `VAD_Idea/` 中的研究假设；该目录作为非 active 历史资料原位保留。

## 2. 选定四个 encoder：三个核心＋一个同家族验证

| 论文角色 | runtime ID | 当前配置 | 当前 checkpoint ID | 快照输入 profile | 用途 |
|---|---|---|---|---|---|
| 主探索／主消融 | `videomaev2` | `configs/encoders/videomaev2-base.yaml` | `videomaev2-base-hf` | 16 帧，224 | 最先形成正常—异常性质图和插件闭环 |
| 跨结构验证 | `timesformer` | `configs/encoders/timesformer.yaml` | `timesformer-default` | 8 帧，224 | 分离式时空交互；有无 CLS 及 readout 要实测 |
| 不同预训练及规模验证 | `vjepa2` | `configs/encoders/vjepa2.yaml` | `vjepa2-default` | 64 帧，256 | 预测式表征路线；结构由本地模型对象确认 |
| 同家族稳健性／第四个模型 | `videomae` | `configs/encoders/videomae.yaml` | `videomae-default` | 16 帧，224 | 检验性质不是 VideoMAEv2 单个 checkpoint 的偶然现象 |

上述 profile 是上传配置，不是此处重新执行的 GPU 证据。四个 checkpoint 在快照登记表中均有明确 revision 与 verified 状态；当前运行版本由回执确认。

VideoMAE 与 VideoMAEv2 不能包装成完全不同的两类架构。论文可写“四个 encoder/checkpoint，覆盖不同预训练路线与时空 attention 结构”。核心三个优先使用前三项；第四项减少额外接入成本。

**不默认接入第五个。**只有当前服务器已有第五个 ViT 且正常前向、探针和可变 token 路径都有证据，才通过 profile 单独启用。快照中 UMT/UniFormerV2 的配置仍有 planned 字段，不能凭文件存在就加入正式结果。CLIP 不在当前四个 active 路径中，不为了凑数新增依赖。其他 CNN、VideoMamba、长视频 MLLM、流式模型均保留但不进入本论文默认队列。

如果某模型只能观察、暂不能真实压缩，登记 `probe_ready=true, reduction_ready=false`，不能在压缩主表冒充已适配。先完成至少三个真实压缩路径，再决定论文跨结构主张的范围。

## 3. 实际源码审计：必须针对这些位置改造

| 已核对源码 | 当前行为 | 本论文需要的改造 |
|---|---|---|
| `src/vadbench/compression.py` | 缓存 identity、keep_recent 等策略 | 原位保留。新 token reducer 使用不同包名，不把它改成同名目录 |
| `integrations/videomaev2.py:210–318` | 通过 observation hook 获取已有前向结果；时间轴为近似映射 | 新增选中模型桥，定位中间 block、实际 token 来源和变长路径 |
| `integrations/videomaev2_encoder.py` | 已有 `_get_encoder_layers()`、模型加载和 `_pool()` | 复用加载；为真实输出类型与池化加测试，不擅自换权重或预处理 |
| `integrations/transformers_video.py:430–478,641–775` | 通用 token 时间映射、HF 前向与 pooling | 区分 TimeSformer/VideoMAE 的布局、CLS、tubelet 和子层 |
| `integrations/foundation/vjepa2.py:40–63` | `get_vision_features` 包在 `torch.no_grad()` 中 | 观察／免训练可复用；若训练内部插件，需要新梯度路径，不能只改 requires_grad |
| `cli.py:_extraction_batches()` | 只分发 `uniform_segments` 与 `chronological_stream` | 固定 encoder 的 dense clip sampler 是待新增功能，不能把流式 sampler 当替代 |
| `data/features_dataset.py` | 按 encoder fingerprint 选择特征 | 新指纹必须包含 reducer 与校准，防止读取 dense 或别的预算缓存 |
| `engine/predict.py:153–186` | checkpoint 与 encoder fingerprint 有严格绑定 | 直接插入实验需显式兼容契约，不能假造相同 fingerprint 绕过 |
| `models/heads.py` | AttentionMIL、TopKMIL、TemporalSupervisedHead | 复用前两者；没有现成弱监督 Temporal Transformer，不把 gated attention 称 Transformer |
| `config.py` | 仅显式 defaults 深合并；encoder 顶层字段白名单 | 新论文配置由新 resolver 解析，不发明当前不存在的隐式 YAML 继承 |
| `pyproject.toml`、`resources.py` | wheel 强制包含 configs/registry/schemas/integrations；存在 `resources.py` 模块 | 保持这些目录及模块路径，禁止用同名 `resources/` 新包遮蔽它 |

以上是代码静态审计，不等于确认某个模型运行时一定触发了问题。例如 `_pool()` 对 raw rank-3 tensor 的分支需要针对实际返回类型测试；若实际返回的是标准对象，这个分支可能没有执行。先写回归测试，再决定修复。

## 4. 目标目录：论文工作面清晰，公共底座兼容

```text
VAD/
├── AGENTS.md                                # 本轮第 4 份文档，根协作入口
├── README.md                                # 改为本论文入口，链接通用框架历史说明
├── README-CN.md                             # 同上；不声称已经有结果
├── pyproject.toml / uv.lock                 # 默认保留现有环境和打包规则
├── projects/
│   └── icassp2027/
│       ├── profile.yaml                    # 唯一 active encoder 列表及阶段默认值
│       ├── protocol.yaml                   # 数据分区、标注权限、质量容忍度、预算口径
│       ├── progress.md                     # 本论文唯一当前进度；由执行者更新
│       ├── decisions/                      # 性质、算法和主张决策，不存大数组
│       ├── locks/                          # 环境/模型/数据/保护文件回执
│       └── assets.local.yaml               # 本机路径；gitignore；不含凭据
├── configs/
│   ├── encoders/                           # 25 个现有配置原位保留
│   ├── experiments/                        # 旧实验原位保留
│   ├── benchmarks/ / smoke/                # 原位保留
│   └── papers/icassp2027/
│       ├── datasets/                       # UCF、XD；不同协议分开
│       ├── observations/                   # 探针集合、采样上限、采集位置
│       ├── methods/                        # dense、基线、最终候选，不预填算法方向
│       ├── heads/                          # 同一弱监督检测头配置
│       └── suites/                         # pilot、确认、主表、消融、效率
├── src/vadbench/
│   ├── data/ / engine/ / models/            # 公共主链保留；新增能力保持旧接口
│   ├── integrations/                       # 原有所有 encoder 原位保留
│   ├── compression.py                      # 原有 cache policies，不改语义
│   ├── research/                           # 新增，只负责观察和比较
│   │   ├── cohorts.py / labels.py
│   │   ├── collectors.py / artifacts.py
│   │   ├── probes/                         # distribution / spatial / temporal /
│   │   │                                    # attention / readout / evolution
│   │   └── contrasts.py                    # 视频级统计、配对/分层对比
│   ├── token_reduction/                    # 新增，部署计算路径；不得导入真值分析模块
│   │   ├── contracts.py / registry.py
│   │   ├── identity.py / baselines.py
│   │   ├── policies/                       # 只有获证据支持的候选才建立
│   │   └── bridges/                        # 4 个 encoder 的布局/位置/算子胶水
│   └── paper/                              # 新增论文编排入口；不是另造训练框架
│       ├── __main__.py / cli.py
│       ├── profile.py / resolve.py
│       ├── stages.py / compatibility.py
│       └── export.py
├── scripts/icassp2027/                      # 薄启动脚本与一次性迁移/保护检查
├── tests/icassp2027/                        # 新能力与冻结兼容性测试
├── docs/icassp2027/                         # 这套编号 01、02、03、05、06 文档
├── paper/icassp2027/                        # 写作专用，详见文档 05
├── registry/ / schemas/ / integrations/     # 原始静态登记与上游锁全保留
├── data/ / weights/ / external/             # 原始资产与软链接原位保留
├── outputs/
│   ├── <历史目录>/                         # 原位不动
│   └── icassp2027/                          # 新运行，最好链接数据盘
│       ├── runs/                           # immutable run-id
│       └── exports/                        # 经确认的表/图/主张导出
├── docs/<历史目录>/                        # 保留；通过索引注明不是本论文入口
└── lab_anomaly/ / VAD_Idea/                 # 不进入论文执行链，原位保留
```

不一次性生成几十个空模块。先实现 profile、cohort、collector、一级 probes 和 identity bridge；按实际需求逐步拆分。避免让“重构框架”成为新的研究任务。

## 5. active profile：唯一名单，引用旧定义，不复制注册真相

下面是**待实现的新论文配置草案**，不能直接当作现有 `load_experiment()` 已支持的语法。

```yaml
schema_version: 1
project_id: icassp2027
mode: contrast_first
active_encoders:
  - id: videomaev2
    definition: configs/encoders/videomaev2-base.yaml
    checkpoint: videomaev2-base-hf
    role: discovery
  - id: timesformer
    definition: configs/encoders/timesformer.yaml
    checkpoint: timesformer-default
    role: structure_confirmation
  - id: vjepa2
    definition: configs/encoders/vjepa2.yaml
    checkpoint: vjepa2-default
    role: pretraining_confirmation
  - id: videomae
    definition: configs/encoders/videomae.yaml
    checkpoint: videomae-default
    role: family_replication
protocol: projects/icassp2027/protocol.yaml
output_root: outputs/icassp2027/runs
paper_root: paper/icassp2027
annotation_policy: weak_development
```

新 `paper.resolve` 必须：读取 profile → 解析原定义与原 checkpoint → 校验能力和资产 → 显式合并本论文配置 → 保存完整 resolved config 和摘要。未知字段或未实现 method 直接失败，不允许忽略 `reduction` 后运行 dense 却标成压缩。

现有 `python -m vadbench encoders list` 仍显示全部注册模型。新论文入口 `python -m vadbench.paper status --project ...` 只显示四个 active。后一个命令是开发目标，当前快照尚不存在。

## 6. 分开三层接口

### 6.1 EncoderBridge：只处理真实结构差异

桥负责：加载后的模型定位、token 顺序、特殊 token、时间/空间来源、位置机制、合法压缩位置、attention/MLP 计算入口。桥不负责定义“异常重要性”。

要求 `probe_available`、`reduction_available`、`supports_grad` 分开报告；不能用 `supports_training=true` 推断内部梯度未被 no_grad 截断。

优先使用 wrapper、context manager 与实例级替换；禁止永久修改上游类全局 forward，禁止把 monkey patch 留在后续 dense baseline 中。

### 6.2 ProbeCollector：只读观察

统一接收 `ProbeView`，包含 `x`、可选 Q/K/V、attention sampled rows、分支更新、token metadata。标签不进入 encoder；collector 输出后由 contrasts 按 video_id join 标签。

要求：关闭后数值等价；异常退出移除 hooks；无 CPU/GPU 张量泄漏；不把完整 attention 永久缓存到每个样本；数据量上限可配置。

### 6.3 TokenReducer：同一算法，不读标签

建议最小逻辑接口：

```python
class TokenReducer:
    def reduce(self, tokens, layout, context):
        """Return reduced tokens + layout + compact mapping + telemetry.
        context contains only currently available activations and budget.
        It must not contain labels, video names, categories, or future layers.
        """
```

`layout` 至少包含：有效位、真实原 token ID、特殊 token ID、时间和空间来源、分组质量、位置处理契约。`ReductionResult` 包含新 tokens、质量、紧凑成员映射、实际 N_in/N_out、需要的索引信息。

映射优先用 index/scatter/CSR，不默认创建 dense 的 `N×K` 指派矩阵。记录来源不等于声称深层 token 只感知该像素区域；全局 attention 后的来源坐标只是 provenance。

`context` 只提供当前可得属性：layer depth、预算、当前 token/轻量统计、允许访问的同一 block QKV。不得调用完整 dense encoder 先生成最终异常分数再决定早层压缩，除非作为明确计入成本的不同两阶段方法。

## 7. 同一插件如何跨四个 encoder

核心算法只能依赖统一张量和结构元数据，不在 policy 中写 `if model == ...: 用另一个算法`。允许桥在以下方面差异化：

- 分离式时空 attention 的合法维度；
- CLS/无 CLS 和 native pooling；
- 绝对位置、相对位置、RoPE 的正确索引；
- window/grid/trajectory 的结构约束；
- 当前模型的 n、d、head 数和 block 数。

**TimeSformer 特别要求：**随机删 token 后不能继续按原 T×H×W 盲目 reshape。若采用跨时间一致的空间分组，应显式维护每帧相同组数、时间轨迹身份与位置；证明后续算子真的在更短序列运行。若每个 block 前恢复成完整网格，须如实计入恢复和完整 MLP 成本，不能按短序列虚报全部 FLOPs。

**V-JEPA 2 特别要求：**当前 loader 使用 `get_vision_features`；必须定位实际 vision encoder，而不是 predictor 或无关 forward。64 帧 profile 是现有配置，不为赶进度未经验证改成 16 帧并继续称同一 dense baseline。

**VideoMAE 系列特别要求：**核对 tubelet 时间范围、是否存在特殊 token、最终 mean/native pooling。不要套用第一个 token 是 CLS 的图像 ViT 假设。

所有桥先通过 `IdentityReducer`、keep_ratio=1、tokens/pooled 等价、实际后续 shape、position 和 padding 测试，才做性能实验。

## 8. 特征与 checkpoint 身份：直接插入实验不能靠绕过校验

现有引擎将 head checkpoint 和特征的 encoder fingerprint 绑定。这是应保留的保护，不是要删除的“麻烦”。

新论文层定义：

- `backbone_identity`：权重、代码、预处理、读出定义。
- `representation_identity`：backbone＋reducer代码/参数/预算/校准＋精度＋位置策略。
- `sampling_identity`：源视频、取帧、窗口、stride、padding、投影协议；训练采样和评测采样分别记录。
- `training_identity`：head 配置、fit split、seed、优化日程。

feature cache 不能跨 representation 或 sampling identity 隐式复用。canonical hash 使用稳定序列化，并记录实际读到的权重及文件摘要；不只依赖 dirty Git HEAD。

两条评测轨分开：

1. **refit_head**：在每种压缩表征上分别训练 head；表征和 sampler 身份相同的情况下可复用现有严格预测入口。训练稀疏、测试 dense 时仍需下述显式采样兼容层。
2. **direct_insert**：用 dense head 评估 compressed 表征。由新 `paper.compatibility` 登记 dense 训练身份、compressed 输入身份以及不变的 backbone、输出维度、预处理和 readout，并输出独立运行记录。

兼容契约区分两种允许的变化：预先声明的 train-32→test-dense 采样变化；同一次评测中仅 reducer 改变的 dense→compressed 表征变化。训练与测试采样不必相同，但各压缩方法与 dense baseline 的**测试采样必须完全相同**。拒绝未登记的权重、维度、预处理、readout 或时间覆盖变化。不要篡改 checkpoint metadata、冒充 dense fingerprint 或全局关闭现有检查。

## 9. 数据与评测重用：新增在哪里

保留原 `VideoManifestRecord`、`ClipBatch`、`TokenTimeline`、`FeatureStore`、`PredictionRecord`、stage-v1 和 official/subset/generic 门禁。

新增研究 cohort 与 spatial token provenance 作为 sidecar/新 schema，不强迫所有 inactive 模型改输出。研究标签 sidecar 与实际模型输入分离。

固定 clip 的 dense sampler 应生成：真实取帧窗口、计分区间、有效 mask、原帧坐标。重叠窗口先按明确定义聚合成全帧分数，再形成无缺口、无重复计分的预测区间；不通过 `--allow-incomplete-coverage` 掩盖问题。

XD-Violence 需要独立 importer、标注端点规则和 evaluator protocol；可以复用通用投影及 AP 工具，但不能把 `ucf-crime/official-frameauc-v1` 改个名字就当 XD 官方协议。

## 10. 保护其他 encoder：按内容和行为验收

### 10.1 改造前建立保护清单

在当前工作树记录已跟踪、未跟踪、已修改文件；保存快照和 raw SHA-256。保护范围至少包括：

- 所有非 active encoder 的 `configs/encoders/*.yaml` 与对应 `integrations/*/upstream.lock.yaml`。
- 非 active adapter 专属源码：CNN、legacy、long_video、HERMES、VideoMamba、InternVideo2 等。
- 原 `registry/encoder-candidates.yaml`、`encoder-integrations.yaml`、`checkpoints.yaml`、环境 registry。新增论文 profile 不要求修改原表。
- 非 active 权重、external checkout、环境目录、历史 outputs。

不用 `git clean`、`reset --hard` 或“删除未引用文件”脚本。受保护清单不读取 `.env`、私钥或认证缓存。路径操作先 resolve，禁止递归跟随数据盘软链接做清理。

### 10.2 共享源码修改必须有兼容性测试

新增代码优先，不改公共契约默认语义。若必须改共享 `common.py`、`contracts.py`、FeatureStore 或 evaluator，先写兼容测试：inactive registry 可枚举、原 adapter import string 可解析、原 unit tests 不退化、schema v1 历史产物仍能读。

“只跑四个 active”不等于删除旧测试或降低其断言。对缺环境的真实 GPU 回归标记未运行，不伪称全部通过。

### 10.3 安全的迁移分期

| 阶段 | 可改内容 | 验收 |
|---|---|---|
| M0 | 文档入口、profile、保护快照、论文目录 | 旧文件不变；新入口能只列四个模型 |
| M1 | collector、cohort、一级 probes、identity bridge | 原始前向等价、坐标与标签权限测试 |
| M2 | 根据 finding card 实现一个 reducer、dense sampler | 真实 shape 缩短、正常/异常两组操作对照 |
| M3 | 直接插入兼容契约、运行导出、论文资产生成 | 独立结果可复现；没有指纹造假 |
| M4 | 必要的共享模块拆分与 compatibility shims | 完整核心回归＋保护 hash 不变 |

不要先完成 M4 才开始实验。M0/M1 就能运行关键观察。

## 11. 最低测试集合

现有测试至少覆盖：

```bash
python -m pytest tests/test_config.py tests/test_contracts.py \
  tests/test_orchestration.py tests/test_features.py \
  tests/test_features_dataset.py tests/test_predict.py \
  tests/test_ucf_crime.py tests/test_sampling.py tests/test_cli_evaluate.py \
  tests/test_transformers_video_integration.py tests/test_foundation_integration.py \
  tests/test_registry.py tests/test_integration_catalog.py \
  tests/test_packaging_resources.py tests/test_cli_lightweight.py
```

测试命令在项目既有合适解释器内执行；实际依赖不满足时记录 skip/blocked。修改 Python 后另做 `python -m compileall src tests`。

新增测试包括：active 白名单、inactive 内容保护、未知 method 失败、observer 不改数值、hook 清理、正确 CLS/tubelet 映射、padding 不参与统计、不同角色标签不能互读、keep_ratio=1 等价、mass 守恒、真实序列缩短、位置机制等价、direct_insert 合法与非法配对、sampler 与 frame projection 边界、feature cache 身份隔离、缺失结果不生成论文数字。

## 12. Codex 的完成报告

每次完成一个阶段必须报告：改了哪些文件；哪些是新增；保护清单结果；实际执行的命令和解释器；通过/失败/未运行的测试；真实 GPU 回执位置；下一阶段唯一阻塞。

不能把本方案提到的目录、命令、桥或 reducer 描述成已经存在。当前任务交付的是说明；实际实现由 Codex 在服务器代码上逐步完成。

来源：本次上传概览第 3–9 节和上述源码；TimeSformer 架构可核对 <https://arxiv.org/abs/2102.05095>；V-JEPA 2 模型卡 <https://huggingface.co/facebook/vjepa2-vitl-fpc64-256>。模型卡仅作核对，运行以本地锁定 revision 为准。

下一份：[03 从已有环境开始的执行流程](03_EXECUTION_RUNBOOK.md)。
