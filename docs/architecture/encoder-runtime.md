# Encoder 运行时架构

本文是编码器接入层的当前事实入口，描述基于 `0badc34` 的本轮改造路径。研究候选和历史验证结果分别由 [`encoder-candidates.yaml`](../../registry/encoder-candidates.yaml) 与 [`encoder-integration-matrix.md`](../progress/encoder-integration-matrix.md) 维护。

## 五类配置对象

| 对象 | 唯一职责 | 数量 / 范围 |
|---|---|---|
| `registry/encoder-candidates.yaml` | 研究全集、环境分组、注册策略、许可状态与阻塞原因 | 25 路 |
| `registry/encoder-integrations.yaml` | 运行时可发现目标、adapter import string、能力、定义文件和 smoke profile | 21 路 |
| `configs/encoders/*.yaml` | 单目标构造参数、输入 profile、输出阶段与 cache 语义 | 25 份；其中 4 份只供候选保留 |
| `registry/checkpoints.yaml` | 21 个运行目标的权重来源、revision、许可、文件清单与摘要 | 21 项 |
| `integrations/*/upstream.lock.yaml` | 25 条上游代码路线在调研截止日固定的 commit、许可证和入口 | 25 份 |

环境依赖由 `registry/encoder-environments-v2.yaml` 维护，它覆盖全部 25 个候选并将每个 ID 唯一分到四组；overlay 只覆盖模型特有的包差异。

这些文件不是相互替代关系。candidate 说明“为什么保留或阻塞”，catalog 说明“框架能发现什么”，encoder definition 说明“怎样构造”，checkpoint registry 说明“应验证哪些资产”，upstream lock 说明“上游代码身份”。一次 smoke 是否成功只能由带时间和输入身份的 result 产物回答。

## 从服务器矩阵到 adapter

```text
run_native_encoder_matrix_v2.py
  ├─ candidate policy：candidate_only / manual asset / license gate
  ├─ environment registry：选择 group Python 与 overlay
  └─ 子进程：python -m vadbench integrations smoke
       ├─ runtime catalog：选择 IntegrationRecord
       ├─ encoder definition：合并 constructor 与命令行 device
       ├─ EncoderRegistry：惰性导入并实例化 adapter
       ├─ video probe + sampling：fixed 中心 clip 或 streaming chunks
       ├─ adapter encode / init_state→encode_step→finalize
       ├─ contract + output-health 校验
       └─ smoke-v2 result.json + launcher log
```

服务器 runner 是当前唯一完整落实“四组解释器隔离”的入口。`environment.runtime=in_process|external_python` 是 catalog 的选择元数据：

- `in_process` 表示 adapter 在 runner 选定的目标解释器进程中运行。
- `external_python` 只说明该路线需要隔离 worker。通用 matrix 执行 external_python 目标前核对当前sys.prefix与registry目标环境；不匹配即blocked并指向server native runner，不再静默回落。
- `ExternalPythonFoundationBridge` 目前与本地 lazy bridge 共用加载逻辑，类本身不启动子进程。
- 无生产消费者的 long-video JSON/stdin `ExternalPythonWorker` 及command/runner别名已删除。保留 `worker=` 注入、原生loader和严格NPY sidecar；worker不再通过临时registry重复构造模型。

因此，在普通 Python 进程中直接调用通用 matrix，只能证明当前解释器中的执行结果，不能声称完成四组隔离验证。

## 注册、能力与构造

`integrations/catalog.py` 严格解析 catalog，再把 21 个 `module:attribute` 目标注册到线程安全的 `EncoderRegistry`。列举 name/spec 不解析目标；首次 `create()` 才导入模块，合并 catalog 默认参数和调用参数，并用 `validate_encoder_adapter` 检查实例能力与静态能力完全一致。

`orchestration.create_encoder_from_experiment()` 的顺序是：读取 `encoder.adapter` → 用 catalog 能力验证实验请求 → 加载对应 definition → 以 `encoder.params` 覆盖 constructor → 把本地模型和 checkout 路径绝对化 → 实例化 adapter。它在构造模型前校验实际checkpoint摘要；环境选择由共享runtime helper负责。

Adapter 家族当前包括：

| 家族 | 目标 | 主要实现 |
|---|---|---|
| TorchVision | R(2+1)D、MViTv2、Video Swin | 本地 state dict、BCTHW 预处理、hook/特征归一化 |
| PyTorchVideo | X3D、SlowFast、I3D | 本地 checkpoint、单/双路径输入、classifier 前激活 |
| Transformers | TimeSformer、VideoMAE | `local_files_only` 模型与 processor、hidden-state 输出 |
| 专用 fixed/foundation | VideoMAE V2、VideoMamba、V-JEPA2、InternVideo2 | 明确上游 loader 或兼容旧 encoder |
| long/streaming VLM | LongVU、VideoChat 系列、MA-LMM、MovieChat、StreamingVLM | projected visual、visual memory 或 decoder-context 输出 |
| HERMES | HERMES + LLaVA-OneVision | 显式 decoder KV、position IDs、原生或外部压缩与 telemetry |
| legacy | C3D | Caffe prototxt/checkpoint/blob 和 `fc6` 特例 |

## 输入、输出与状态契约

统一输入是 `ClipBatch.frames[B,T,H,W,C] uint8`，时间戳、有效帧 mask、原帧索引和 `video_id` 与像素同行。固定路径返回 `EncoderOutput`：

- `features[B,S,D]` 是检测头消费的序列；
- `pooled[B,D]` 是可选聚合，smoke 当前要求存在；
- `TokenTimeline[B,S]` 记录每个 token 的秒区间，可选记录原帧半开区间；
- `aux` 可包含附加 tensor 与元数据；写入索引/报告的元数据须可 JSON 化，tensor 使用二进制 sidecar，并记录 `feature_stage`、`sequence_source`、实现来源和必要遥测。

流式路径为每个视频调用 `init_state(video_id)`，随后逐 chunk 调用 `encode_step`。`StreamState.step_index` 必须递增，`video_id` 不得切换，下一时间戳必须前进。缓存以 `CacheView` 明确声明 `kind`、tensor、sequence axis 和 timeline；更新以 append/replace 等 `CacheUpdate` 表达。

三种状态对象不能互换：

| kind | 所有者 | 当前代表 |
|---|---|---|
| `vision_tokens` | 视觉/投影 token | 当前 catalog 没有 streaming token-cache 目标；LongVU、VideoChat-Flash 只导出 projected visual |
| `visual_memory` | Q-Former/视觉记忆 | VideoChat-Online、MA-LMM、MovieChat |
| `decoder_kv` | 因果语言模型 decoder | HERMES、StreamingVLM |

VideoMAE V2、VideoMamba 等模型内部有 attention 或 SSM 状态，不等于可跨 chunk 传递的框架 `StreamState`。

HERMES 默认 catalog 输出阶段是 `projected_visual`。该阶段特征不受历史 decoder cache 内容直接条件化，适合比较性能与缓存成本；显式选择 `decoder_contextual` 才能研究 cache 对表征和精度的影响。HERMES 的上游 `predict_and_compress` 与框架 `keep_recent` 是不同策略，双重压缩会被拒绝。

## Preflight、smoke 和产物的边界

通用 preflight 保持轻量存在性检查；真正create会校验绑定权重，server资产核验还检查checkout身份。preflight没有执行前向，不能当作smoke成功。

`run_encoder_smoke_v2` 对 fixed 目标采一个中心 clip；对 streaming 目标要求至少两个 chunk。每个输出经过 shape、dtype、有限值、timeline 单调与视频边界检查。失败按当前异常类型分成 `blocked`（缺文件、依赖、权限）或 `failed`（运行或契约错误），同时保留已经完成的 batch、输出和流式 step 摘要。

`smoke-v2` 产物包含输入视频与 batch 身份、输出健康度、环境、代码/权重来源、命令、耗时、峰值显存、Git commit/dirty 和结构化错误。标准构造路径返回生效definition和经过校验的checkpoint身份；结果使用同一对象，写入前按正式smoke schema校验。历史结果不自动复用，异常或缺身份不能补成成功。

`worker_protocol.py` 用严格 JSON envelope 加受 SHA256、dtype、shape、大小和路径约束的 `.npy` sidecar，在另一个 Python 进程中传输 `ClipBatch` 和 `EncoderOutput`。流式返回只序列化每步输出和状态/cache 摘要，不搬运模型私有 `opaque` 状态。

## 新增或恢复一个目标

1. 在 candidate registry 固定研究身份、环境组、许可和注册策略。
2. 固定 upstream lock 与 checkpoint registry；planned 资产不能写成 verified。
3. 编写 encoder definition，明确输入 profile、feature stage、cache kind 和本地路径。
4. 实现最小 adapter 特例，复用公共输出规范化；只有真实跨 chunk 状态才声明 streaming/cache 能力。
5. 加入 runtime catalog，确保 listing 不导入重依赖，并让实例能力与 catalog 一致。
6. 在四组 runner 中用目标自己的真权重和真实视频运行；fixed 至少一 clip，streaming 至少两 chunk。
7. 保存result、日志与实际环境身份后更新带日期的进度矩阵；静态catalog只保留planned/integrated/blocked，不写smoke_pass。许可不明的技术通过仍保持blocked。
