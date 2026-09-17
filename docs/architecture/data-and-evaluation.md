# 数据、特征、训练与评测架构

## 结论

参考配置的抽取互斥已修复，当前流水线统一使用实际 constructor 及绑定权重身份。模块关系是：UCF-Crime 源清单进入 canonical manifest，视频解码器产生统一的 `BTHWC uint8`，adapter 输出 `features[B,S,D] + TokenTimeline`，特征仓保存内容寻址数组，head-only runner 训练 MIL 或时序 head，预测器生成带帧区间的 JSONL，evaluator 再计算全局 frame ROC-AUC/AP。预测与评测共用 `engine.coverage` 的覆盖校验；official 模式校验 audit v2 中的官方源和 manifest 内容身份；抽取、训练、预测、评测及 benchmark 通过公共 `record_stage` 保存 config、输入、代码身份及失败状态。实现与后续验证摘要见[2026-09-11 实施记录](../progress/2026-09-11-implementation.md)。

截至 2026-09-11，服务器代码 commit 为 `0badc3435e734a841110e29d497940bfda7cc707`。`data/raw/ucf_crime` 软链接存在，但目标目录没有视频，canonical train/test manifest 也不存在。因此以下“当前实现”依据同 commit 的源码和测试；真实 UCF-Crime 的规模、容器、时间轴和指标仍待数据部署后验证。

## 当前数据流及责任边界

```text
官方 split / temporal TXT / 可选 UCA
        │  data/ucf_crime.py：解析、坐标转换、语义隔离
        ▼
video manifest JSONL
        │  data/manifest.py：记录契约、路径与 split 校验
        │  data/audit.py：官方计数、文件/容器、评测就绪性
        ▼
OpenCV 解码 + sampling.py
        │  data/video.py：BTHWC、真实帧索引、segment 元数据
        ▼
VideoEncoderAdapter / StreamingVideoEncoderAdapter
        │  contracts.py：features[B,S,D]、TokenTimeline、StreamState
        ▼
FeatureStore
        │  features.py + engine/extract.py：NPZ/NPY、索引、checksum
        ▼
FeatureDataset → MIL/temporal head → checkpoint
        │  data/features_dataset.py、tasks.py、engine/runner.py
        ▼
PredictionRecord JSONL
        │  engine/predict.py：默认严格帧覆盖
        ▼
frame labels + interval projection → micro frame AUC/AP
           data/labels.py、engine/evaluate.py、metrics.py
```

### Manifest 和真值

`VideoManifestRecord` 是数据身份的中心契约，路径必须相对 dataset root，区间统一为 0-based half-open。UCF 导入器把官方 MATLAB 1-based inclusive `[a,b]` 转成 `[a-1,b)`，并把 raw 端点保存在 annotation metadata。UCA 只生成 `scope=caption, is_anomaly=null`，不会自动变成二值异常标签。相关实现见 `src/vadbench/data/manifest.py:124`、`src/vadbench/data/ucf_crime.py:195` 和 `src/vadbench/data/ucf_crime.py:507`。

常规 manifest 校验负责类型、唯一性、相对路径和基于规范 ID/路径的 split 泄漏。真实基准应额外运行 `audit_ucf_crime_dataset`；它按 audit v2 schema 校验冻结的官方 split/temporal 来源身份、train/test manifest SHA256、官方计数、14 个类别、文件存在性、容器几何、manifest/container 冲突以及完整 290 视频的 frame-AUC 就绪性。完整文件 SHA256 只有 `deep_hash=true` 才计算，视觉近重复仍未实现。相关实现见 `src/vadbench/data/audit.py` 与 `schemas/dataset-audit-v2.schema.json`。

### 采样、解码和时间轴

`uniform_segments` 对正常长视频生成无空洞的等分区间；短于段数时复用帧以维持 instance 数。每段当前只采一个固定 clip。padding 重复最后一帧并由前缀 `valid_mask` 排除。`iter_fixed_segment_batches` 把完整 segment 的边界、实际采样帧索引、split 和视频标签都写入 batch metadata。`FeatureDataset(feature_level="clip")` 用完整 segment 边界作为最终预测区间，而不是把中心 clip 的窄区间误当作整段覆盖。相关实现见 `src/vadbench/data/sampling.py:138`、`src/vadbench/data/video.py:340` 和 `src/vadbench/data/features_dataset.py:134`。

streaming 路径按单视频、B=1 的 chunk 顺序调用 adapter，并在每个视频开始时创建新的 `StreamState`。框架 cache policy 只操作 adapter 显式暴露的 `CacheView`；HERMES 原生压缩由 adapter 自己负责。固定 clip 与流式能力均由 `EncoderCapabilities`/运行时校验约束。相关实现见 `src/vadbench/contracts.py:112`、`src/vadbench/contracts.py:641` 和 `src/vadbench/engine/extract.py:479`。

### 特征和训练

`FeatureStore` 把大数组写到内容寻址 NPZ/NPY，把 shape、dtype、字节数和 SHA256 写入 `feature-index-v1` 记录。每次读取重新校验文件 hash。特征索引按 `(encoder_fingerprint, video_id, clip_id)` upsert；默认 overwrite 是可逆的索引替换，但旧内容 blob 不会自动清理。相关实现见 `src/vadbench/features.py:308`、`src/vadbench/features.py:520` 和 `src/vadbench/features.py:636`。

`FeatureDataset` 只选择一个 encoder fingerprint，核对 feature 中记录的 split/label 与 manifest 一致，并可要求 clip index 恰为 `0..N-1`。弱监督输出视频级标签；强监督只接受显式 frame/segment 二值区间，caption/video-only 样本默认排除。`train_feature_head` 训练 Attention/Top-k MIL 或 temporal head，保存最后一次 checkpoint 及其 SHA256 sidecar。当前没有 scheduler、early stopping、best-checkpoint 选择或类别平衡 sampler。相关实现见 `src/vadbench/data/features_dataset.py:203`、`src/vadbench/tasks.py:85` 和 `src/vadbench/engine/runner.py:279`。

### 预测和评测

`predict_feature_head` 从 checkpoint sidecar 恢复 task、feature level、维度与 encoder fingerprint，并默认要求每个视频的预测区间从 0 无缝覆盖到 `num_frames`，不允许 gap、overlap、越界或重复 clip index。输出的 `PredictionRecord.ground_truth` 当前是视频级 `manifest.is_anomaly`，evaluator 不读取这个字段；逐帧真值只从 manifest 的显式时序 annotation 生成。相关实现见 `src/vadbench/engine/predict.py:132`、`src/vadbench/engine/predict.py:167` 和 `src/vadbench/data/labels.py:16`。

`evaluate_ucf_prediction_records` 将预测 interval 投影到逐帧数组，再把所有视频拼接后计算 micro ROC-AUC/AP。底层 metric 正确处理 ties，并可在单类别输入时返回 NaN 或抛错。`vadbench evaluate` 默认 official，检查匹配的 audit v2 与共享覆盖校验；subset 也要求完整覆盖。仅显式 generic 或底层通用 API 保留填值/重采样诊断。相关实现见 `src/vadbench/engine/coverage.py`、`src/vadbench/engine/evaluate.py`、`src/vadbench/metrics.py` 和 `src/vadbench/cli.py`。

## 实现状态与规范差距

| 能力 | 当前代码 | 正式基准仍需完成 |
|---|---|---|
| 官方 split/时序导入 | 已实现并测试 6/7 列格式、端点转换和泄漏拒绝 | 在真实官方文件上核对 1,610/290 和若干 `.mat` |
| 数据就绪审计 | audit v2 已检查官方源身份、manifest source hash、计数、文件、容器和 GT 几何；可选 SHA256 | 部署真实数据；增加视觉近重复审计 |
| 32 段基线 | 已实现每段一个中心 clip、固定 32 instance | 明确它与原 C3D 段内多 clip 均值的差异 |
| 弱监督训练 | Attention/Top-k、BCE、可选 ranking/正则项已实现 | 冻结主配置的 objective；实现/记录正负采样策略 |
| 强监督训练 | 显式 frame/segment target 与 ignore mask 已实现 | 提供合法的 train temporal GT；UCA 映射尚未建立 |
| 预测覆盖 | `predict`、subset 与 official evaluate 共用严格覆盖校验 | 防止显式 incomplete/generic 诊断产物混入正式结果 |
| 帧级指标 | micro ROC-AUC/AP、逐帧分数和共享覆盖门禁已实现 | normal-only 误报诊断与分桶 |
| 可追溯性 | stage v1 记录 config、输入、代码身份和失败状态；audit v2 记录官方源/manifest hash | 真实数据与真实模型运行的完整产物 |
| 真实结果 | 模型冒烟和合成闭环已有证据 | 服务器目前无 UCF 视频/manifest，尚无全量 benchmark |

## 修改方向

本轮已完成 constructor/权重身份绑定、official audit + coverage 门禁、共享阶段记录，以及 HERMES 的 32 段 `weak_mil`、`decoder_contextual`、`native_compression_mode=off` 对照配置。原问题、触发条件与验收标准保留在[架构审查](../reviews/2026-09-11-architecture-review.md)，实现状态与后续验证以[实施记录](../progress/2026-09-11-implementation.md)为准。仍待真实数据验证的重点是视觉近重复审计、UCA 映射审查、normal-only 误报诊断及正式全量结果。

## 验证依据

本页不重复维护测试计数或模型运行结论；最新验证汇总见[实施记录](../progress/2026-09-11-implementation.md)。测试主要覆盖合成数据或 fake backend；服务器真实数据为空，所以没有执行真实视频解码、官方 `.mat` 对照或全量 frame-AUC。
