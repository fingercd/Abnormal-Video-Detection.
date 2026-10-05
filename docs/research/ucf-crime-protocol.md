# UCF-Crime 数据、训练与评测协议

## 结论先行

本项目的可比主协议是：**官方视频级 1,610/290 划分；训练视频仅使用视频级标签；每视频 32 个时间段；测试分数投影回原始帧；在全部 290 个测试视频的帧上计算 micro ROC-AUC**。其中 32 段是为兼容 Sultani 等经典 WSVAD 基线而固定的项目协议，不是数据集唯一允许的采样法；改变段数可以做消融，但必须另命名 protocol。任何使用测试时间标注训练、从 clip 级重新随机划分、或用测试集拟合归一化/阈值的结果，都不能标为这一官方弱监督兼容协议。

强监督实验可以作为独立轨道，但现有公开来源要谨慎区分：UCA 是时间戳自然语言事件集，不是现成二值异常真值；截至 2026-08-31，FS-UCF-Crime Zenodo 条目只有 placeholder 文件，尚不能用于训练。

截至 2026-09-11，代码已经接通 manifest 导入与审计、视频解码、32 段抽取、冻结特征训练、标准预测 JSONL 和 frame ROC-AUC/AP 计算；这不等于正式基准已经完成。服务器的 `data/raw/ucf_crime` 软链接存在，但目标数据目录当前没有视频，`data/manifests/ucf_crime/train.jsonl` 与 `test.jsonl` 也不存在。因此仓库当前只有合成/冒烟链路证据，没有可报告的真实 UCF-Crime 全量分数。

## 1. 一手事实与来源层级

### 1.1 数据集规模与任务

[UCF 官方项目页](https://www.crcv.ucf.edu/projects/real-world/)与[原始 CVPR 2018 论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html)给出的事实是：

- 1,900 段长、未裁剪的真实监控视频，总时长 128 小时；
- 13 类异常：Abuse、Arrest、Arson、Assault、Road Accident、Burglary、Explosion、Fighting、Robbery、Shooting、Stealing、Shoplifting、Vandalism，另有正常活动；
- 可做“所有异常 vs 正常”的一般异常检测，也可做 13 类异常活动识别；
- 弱监督训练标签位于视频级，测试异常视频另有时间标注。

### 1.2 官方划分

原论文“Training and testing sets”明确写明：

| split | 正常 | 异常 | 合计 | 监督可见性 |
|---|---:|---:|---:|---|
| train | 800 | 810 | 1,610 | 视频级 normal/abnormal；主协议不可读异常时间区间 |
| test | 150 | 140 | 290 | 最终评测时读取异常时间区间；正常视频全零 |
| 总计 | 950 | 950 | 1,900 | — |

必须以官方视频身份列表导入 manifest；不要根据某个第三方特征包的文件数量反推 split。

### 1.3 官方时间标注文件

UCF 官方项目页提供 [`Temporal_Anomaly_Annotation_For_Testing_Videos.zip`](https://www.crcv.ucf.edu/projects/real-world/Temporal_Anomaly_Annotation_For_Testing_Videos.zip)，作者还在[官方代码仓](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018)保存了文本标注与训练列表。其中：

- `Txt_formate/Temporal_Anomaly_Annotation.txt` 每行形如 `video class start1 end1 start2 end2`；第二段不存在时用 `-1 -1`；
- `Matlab_formate/*.mat` 中的 `Ann` 与 TXT 保存同一组端点，例如 `Abuse028_x264` 为 `[165,240]`；
- [作者官方 MATLAB evaluator](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018/blob/master/Evaluate_Anomaly_Detector.m)执行 `GT(st_fr:end_fr)=1`，证明源坐标是 **MATLAB 1-based、两端包含**。

内部 `TemporalSpan` 使用 Python **0-based、半开** `[start_frame,end_frame)`，所以必须转换为 `[raw_start-1, raw_end)`；示例 `165..240 -> [164,240)`，共 76 帧。当前导入器已执行该转换，并在 annotation metadata 保留 raw 端点、`matlab_1based_inclusive` 与内部坐标声明。任何直接存成 `[165,240)` 或做成 `[165,241)` 的实现都与官方帧 GT 不一致；正式全数据评测前仍需直接读取若干官方 `.mat` 文件做端到端抽样对照。

来源优先级如下：官方项目页/原论文/官方标注压缩包 > 作者官方代码 > 本项目固定副本的校验值 > 第三方仓库或网盘整理。发生文件名、帧数或区间冲突时停止导入并报告，不自动覆盖。

## 2. Canonical manifest 与数据身份

实现位于 `src/vadbench/data/manifest.py`、`src/vadbench/data/ucf_crime.py`，schema 位于 `schemas/video-manifest-v1.schema.json`。公共记录类型是 `VideoManifestRecord`（兼容别名 `ManifestRecord`）、`SupervisionAnnotation`/`TemporalSpan`，划分与坐标由 `DatasetSplit`、`SupervisionScope`、`SpanUnit` 显式表示。每一行 JSONL 至少要能表达：

- 稳定 `video_id` 和相对/可重定位 `path`；
- 顶层 `split=train|test`、`category`、`is_anomaly`；`dataset=ucf_crime` 等来源信息放入 `metadata`；
- `fps`、`num_frames`、`duration_seconds` 及其探测来源；
- 时间区间及坐标单位、端点规范、annotation source/version；
- 可选的文件大小/内容指纹（放入 metadata）或其他可审计的数据身份；
- 可追溯的导入器版本和异常告警。

manifest 只保存元数据，不复制视频。服务器上 `data/raw/ucf_crime` 应是指向已有数据盘的软链接；截至 2026-09-11 该软链接已经建立，但目标目录为空，不能据此声称数据已部署。`load_manifest_jsonl` / `write_manifest_jsonl` 负责 I/O，`validate_manifest` / `validate_manifest_pair` / `assert_no_split_leakage` 负责单文件与跨 split 校验。同一规范化 `video_id`/路径跨 split、帧率非正、已知视频长度下的区间越界或区间倒置均为硬错误；视频实体是否存在只在调用 `require_files=true` 时检查。

`audit_ucf_crime_dataset` 是正式数据到位后的完整门禁入口。它输出受 `dataset-audit-v2` schema 约束的审计记录，先核对冻结的官方 split/temporal 来源身份及 train/test manifest SHA256，再核对 1,610/290 与 800/810、150/140 的计数，探测容器帧数/FPS/时长，检查 manifest 与容器的一致性，并验证 290 个测试视频是否具备 frame-AUC 所需几何信息和真值。`deep_hash=true` 才读取全文件计算 SHA256；默认不会付出这一 I/O 成本。视觉感知近重复目前始终记录为 `not_run`，不同文件名但内容相近的视频仍需另做审计。

## 3. 弱监督训练协议

### 3.1 采样

原论文在训练中把每个视频划成 32 个非重叠时间段，把每个时间段作为 MIL bag 中的 instance；原实现用每 16 帧的 C3D 特征，并对落在同一时间段内的 clip 特征取平均。论文同时明确 32 是经验设定。[原论文实验设置](https://arxiv.org/html/1801.04264#S5.SS1)与[作者的 32 段聚合脚本](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018/blob/master/Save_C3DFeatures_32Segments.m)可交叉核验。

本项目的 encoder-agnostic 规则：

1. 对 `N>=32` 帧视频构造覆盖 `[0,N)` 的 32 个无交叠时间段；`N<32` 时无法同时满足“32 个非空且互不重叠”，当前 sampler 会夹紧边界并复用部分帧，仍输出 32 个 instance，产物必须记录这一退化情况。
2. 当前 `sample_uniform_segment_clips` 每段只选择一个 clip；首批 VideoMAE V2 配置为 `clip_frames=16`、`frame_stride=2`，默认取时间段中心。段内不足所需跨度时重复最后一个真实帧，并用 `valid_mask=false` 标出 padding。
3. 固定 clip 抽取把每个 clip 写成一条 `FeatureRecord`。`FeatureDataset(feature_level="clip")` 优先读取 adapter 的 `pooled` 向量；没有 `pooled` 时才对有效 token 做均值，因此当前主配置最终每视频严格得到 32 个 instance。代码尚没有“同一段采多个 clip 后再做段内聚合”的实现。
4. 训练消融可以把 `position` 设为 `random` 并记录 seed，但验证/测试必须使用确定性位置。当前抽取 provenance 保存完整实验配置，因此能保留 position/seed；解码失败会直接中止，尚不会汇总失败视频列表，这一正式报告要求仍待实现。

采样实现与测试位于 `src/vadbench/data/sampling.py` 和 `tests/test_sampling.py`；公共入口为 `uniform_segments`、`sample_fixed_clip`、`sample_uniform_segment_clips` 和 `sample_32_segments`。`vadbench extract` 已通过 `iter_fixed_segment_batches` 接通真实视频解码、采样、adapter 和特征仓。可变帧率仍用容器报告的单一 FPS 生成时间戳，OpenCV 随机 seek 失败会作为解码错误退出；损坏尾帧和真实数据上的 VFR 对齐尚未验证，不能由单元测试推断为已解决。

### 3.2 Bag、标签与损失

- 正常视频是 negative bag，32 个 instance 均不应含异常；异常视频是 positive bag，只知道至少一个 instance 异常。
- 主配置使用 `WeaklySupervisedMILTask + AttentionMILHead`。当前默认优化目标是 bag-level BCE；`ranking_weight` 默认是 0，只有配置显式启用且一个 batch 同时含正、负 bag 时才计算 ranking loss。`TopKMILHead`、平滑项和稀疏项已有实现，但当前实验 YAML 没有启用。因此现状只能称为“32 段 Sultani-compatible 采样”，不能称为完整复现 Sultani 2018 的 ranking objective。
- encoder 冻结特征训练与 encoder 联合微调是不同实验。首批先冻结真实权重抽特征，验证全链路后再做渐进解冻。
- 当前 DataLoader 只使用 `shuffle=true`，没有正负平衡 sampler，也没有 class weight；“保持 normal/abnormal 平衡或记录采样权重”仍是正式主基线的待实现要求。不要用测试 150/140 比例调训练阈值。

原论文的 30 positive + 30 negative bag、C3D-4096D 和三层 FC 是历史复现参数，不强制所有现代 encoder 照搬；若声称“复现 Sultani 2018”，则这些差异必须逐项列出。

## 4. 测试分数与帧级指标

### 4.1 时间投影

模型输出可以是 32 段分数，也可以是更密集的 `TokenTimeline`。`evaluate_ucf_prediction_records` / `evaluate_ucf_predictions` 在评测前必须产生与原视频 `num_frames` 等长的分数向量：

- 32 段基线：对 `t in [b_i,b_{i+1})` 赋值 `score_i`；
- 任意 token/clip：优先使用记录的起止帧把分数投影到覆盖区间；重叠区间的聚合方法（mean/max/last）必须固定在配置中；
- 没有覆盖的帧不得从指标中删除。`predict_feature_head` 默认执行严格覆盖检查：要求 manifest 同时有 `num_frames`/`fps`，区间从 0 连续覆盖到 `num_frames`，且既不重叠也无空洞。这个门禁可以被 `--allow-incomplete-coverage` 关闭。
- `vadbench evaluate` 当前没有重复执行上述覆盖检查，也不会要求 dataset audit 通过。底层投影 API 对未覆盖帧使用 `fill_value=0`、对越界 interval 裁剪；若记录没有显式 interval，还可走均匀重采样兼容路径。正式结果必须先由预测阶段的严格门禁和 dataset audit 保证输入，或者先修复 evaluator 使其独立拒绝不完整输入。
- 不按固定 FPS 猜帧数。分数必须对齐解码探测到的 `num_frames`；当前投影函数会把网格外 interval 自然裁掉而不自动报错，因此 manifest/evaluator 上游的 GT 越界硬门禁不可省略。

实现由 `project_intervals_to_grid` / `project_intervals_to_frames` 负责；合成边界与官方数值样例已有测试，正式数据到位后仍须直接读取官方 `.mat` 做端到端抽样验证。

### 4.2 主指标

原论文使用 frame-based ROC 及其 AUC，并明确不以 EER 为主指标。[原论文 Evaluation Metric](https://arxiv.org/html/1801.04264#S5.SS1)

本项目固定：

1. 按官方 test manifest 顺序，对全部 290 个视频分别生成 frame score 和 frame label；
2. 将所有帧拼接后计算一个 **micro frame ROC-AUC**，以 `[0,1]` 小数保存，展示时可乘 100；
3. 同时报 frame Average Precision，便于观察异常帧稀疏时的排序质量，但 AP 不是替代官方 AUC 的主排名；
4. 不把 per-video AUC 的平均值叫作官方 AUC。全正常视频单独算 ROC-AUC 本身也未定义。

当前 `UCFFrameMetrics` 实际输出 `frame_auc`、`frame_ap`、正/负/总帧数和视频数（以及两个指标别名）。预测、subset 和 official evaluate 共用严格 frame coverage 校验；official evaluate 还要求匹配的 audit v2。normal-only false-positive、按视频长度/异常持续时间分桶和阈值化指标仍是后续诊断，不能写成现有产物。evaluate 阶段通过 stage v1 记录 config、输入和代码身份及完成/失败状态；普通 metrics/cache 事件没有被误称为独立 schema。

阈值化准确率、F1、event mAP 或 latency-aware 指标可以附加，但阈值必须由训练内划出的 validation 或预注册固定值决定，不能在 290 个 test 视频上寻优。

## 5. 强监督轨道：可用数据与限制

### 5.1 官方 test temporal annotation

它只能作为 test GT。把 140 个测试异常视频的时间区间加入训练，会直接破坏官方弱监督协议；即使仍在同一 290 视频上计算 AUC，也必须标成 transductive/leaky，不得与 WSVAD 表格并列。

### 5.2 UCA 可用，但不是二值强监督真值

[UCA/CVPR 2024 论文](https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_Towards_Surveillance_Video-and-Language_Understanding_New_Dataset_Baselines_and_Challenges_CVPR_2024_paper.html)报告：从 UCF-Crime 过滤 46 个低质量/重复等视频后，对 1,854 个视频提供 23,542 条自然语言事件描述和起止时间；标注视频时长 110.7 小时，时间精度 0.1 秒。其原则是“尽可能描述每个事件/状态变化，**无论是否异常**”。[作者官方数据/代码仓](https://github.com/Xuange923/Surveillance-Video-Understanding)按 Train/Val/Test 发布 timestamp + sentence 的 JSON/TXT，可用于核验导入器，但字段存在不等于二值异常标签。

因此安全导入方式是：

- 原样导入 `start_s/end_s/text`，保留 UCA split 与源版本；
- 另建受审计的 `text/event -> abnormal | normal | ignore` 映射；不能把所有 UCA interval 标为 1；
- 未被 caption 覆盖的间隙默认 `ignore`，除非有独立证据证明 normal；
- UCA 已移除 46 个 UCF-Crime 视频且自有 split，不能假设与官方 1,610/290 一一对齐；
- 若模型或 caption encoder在 UCA test 文本上训练，不得再把同一视频当独立 UCF test。

只有完成映射版本化、双人抽样审查、一致性统计和 train/test 身份检查后，才可用 `TemporalSupervisedTask` 做独立强监督实验。当前 `build_temporal_targets(timeline, annotations_by_video, ...)` 只把显式 frame/segment anomaly span 投影到 `[B,S]`；caption 与 video-only annotation 默认进入 `valid_mask=false`，这正是防止 UCA 被自动二值化的实现门禁。

还需显式处理一个配置默认值：`FeatureDataset`/runner 的 `assume_unannotated_is_normal` 默认为 `true`。只要某视频已有至少一条二值 temporal annotation，未命中的有效 token 就会被当作 normal。这个默认值适合“异常区间之外均为正常”的完整标注，不适合只映射了一部分 UCA caption；后者必须设为 `false`，让未覆盖位置保持 ignore。当前 `ucf_hermes_stream.yaml` 已是官方弱监督配置：32 段、16 帧、stride 2、`weak_mil`、`decoder_contextual`，并明确 `native_compression_mode: off` 作为 raw/identity 对照；它不是强监督 UCA 配置。

### 5.3 FS-UCF-Crime 截止日状态：仅 placeholder

[Zenodo record 21336651](https://zenodo.org/records/21336651)元数据称其计划提供训练视频时间标注、修订后的 test 标注和 validation split，但描述同时写明“associated paper accepted 后发布完整 annotation package”。截至 **2026-08-31**，[Zenodo API](https://zenodo.org/api/records/21336651)列出的唯一文件是 `FS-UCF-Crime_Zenodo_placeholder.md`。

所以当前规则是：可以把 DOI、版本 `0.1.0` 和检查日期写入候选来源登记；不得将 record 的 `access_right=open` 或 `status=published` 误读为完整标注已发布；不得以 placeholder 生成伪标签。未来若出现真实文件，要新建来源版本、校验 checksum 并重新审查 split。

## 6. 泄漏与污染检查表

| 风险 | 典型错误 | 必须的防线 |
|---|---|---|
| split 泄漏 | 先切成 clip 再随机 train/test | 只按官方 `video_id` 切分；manifest 以规范化 ID/路径做硬门禁，另做可选内容 hash/视觉近重复审计 |
| temporal GT 泄漏 | 用 test anomaly interval 选 clip、调 prompt 或训练 head | test annotation 只在 evaluator 打开；训练数据对象不携带可访问的 test GT |
| UCA 语义泄漏 | 所有 caption interval 设为 abnormal | 显式三态映射与版本；未映射为 `ignore` |
| 预处理泄漏 | 用全量特征拟合 mean/std、PCA、聚类字典 | 只在 train 拟合并保存统计指纹；test 只 transform |
| 阈值/早停泄漏 | 在 290 test 视频上选最佳 epoch/budget/threshold | 从 train 内固定 validation；test 每个最终配置只做一次正式评测 |
| prompt 标签泄漏 | 压缩器用真实类别或 GT 文本作 query | 使用预注册的类无关 prompt；query-aware 设置单独标注 |
| 文件名泄漏 | `Abuse...`/`Normal...` 进入模型输入或可学习特征 | 模型只接像素/允许的模态；文件名仅供 manifest 与报告 |
| cache 串样本 | 前一视频 KV/state 未 reset | 每个 `video_id` 创建新 `StreamState`；测试跨视频隔离 |
| 重复/近重复 | UCA 已指出原集含重复视频 | 记录精确 hash，并额外做可选视觉近重复审计；发现跨 split 只报告，不擅改官方 split |
| 外部预训练污染 | 基础模型训练语料可能含 UCF-Crime | 记录上游预训练数据披露；无法排除时标为 unknown，不声称从零无污染 |

## 7. 运行产物与报告模板

正式结果要求至少保存：

- Git commit、dirty 状态、实验 YAML 的内容 hash；
- train/test manifest hash、官方 annotation zip/hash、UCA/FS 来源状态；
- encoder repo/revision、权重 SHA256、许可证、precision/device；
- 视频解码失败列表、实际样本数、总帧数、frame score 覆盖率；
- seed、采样边界、clip frames/stride、head、loss 和 checkpoint 选择规则；
- frame ROC-AUC、AP、正常视频误报诊断；
- 若为 streaming：每步 cache kind、budget、token/memory 数、压缩事件、峰值显存和分段耗时。

当前实现已覆盖其中的运行身份主链：特征索引与 checkpoint 有内容摘要，audit v2 保存官方来源与 manifest identity，抽取、训练、预测、评测和 benchmark 写 stage v1 的 config、输入、代码身份以及完成/失败状态。真实数据、真实模型的完整运行产物和 normal-only 误报诊断仍待补齐，不能据此声称已有正式全量结果。最新实现/验证摘要见[2026-09-11 实施记录](../progress/2026-09-11-implementation.md)。

正式表格中的 protocol 名建议固定为：

- `ucf-crime/wsvad-official-32seg-frameauc-v1`
- `ucf-crime/temporal-uca-mapped-v1`（只有完成映射审计后）
- `ucf-crime/fs-ucf-crime-<released-version>`（当前禁止使用）

## 8. 最小协议验证命令

在 Windows 运行：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_manifest.py tests/test_ucf_crime.py tests/test_sampling.py tests/test_metrics.py
```

服务器/Linux 运行：

```bash
python -m pytest tests/test_manifest.py tests/test_ucf_crime.py tests/test_sampling.py tests/test_metrics.py
```

本页不重复维护测试计数；最新实现和验证汇总见[2026-09-11 实施记录](../progress/2026-09-11-implementation.md)。正式验收还要确认 dataset audit 恰为 1,610 train + 290 test、类别计数为 800/810 与 150/140、290 个测试容器全部可探测、异常测试视频均有 1–2 个合法区间、正常测试视频 GT 全零、预测严格覆盖每一帧，并在抽样视频上与官方 `.mat` GT 一致。当前服务器数据目录为空，这组真实数据验收尚未执行。
