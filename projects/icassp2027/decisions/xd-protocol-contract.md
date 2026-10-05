# XD-Violence 原视频导入与评测契约（官方来源核查）

核查日期：2026-09-18。本文记录作者项目、ECCV 2020 原论文和作者官方 `XDVioDet` 的协议证据；本轮未读取 `annotations.txt` 或 `gt.npy` 的事件/数组值。代码已另行实现训练身份导入与受限的 feature-grid 数学评估，原视频评测对齐尚未接通。

## 可实现的最小契约

### 1. 原视频命名与弱标签

作者 [annotation ReadMe](https://roc-ng.github.io/XD-Violence/images/ReadMe.md) 规定文件名的 `label_` 字段可包含多类 token。类 token 为 `B1` Fighting、`B2` Shooting、`B4` Riot、`B5` Abuse、`B6` Car accident、`G` Explosion；一个暴力视频可有 1–3 类。作者官方代码 [dataset.py](https://github.com/Roc-Ng/XDVioDet/blob/f846d8cfc454943ec165bd2489b71e7dc0a1064d/dataset.py#L21-L51) 不做多类学习：路径中含 `_label_A` 的样本赋 `0.0`，其他全部赋 `1.0`。

当前实现采用 manifest 的 opaque SHA `video_id`；`path` 保存卷名和成员路径，metadata 的 `archive_member` 保留原始归档相对路径及文件名，`raw_label_tokens` 保留完整标签序列。opaque ID 绑定完整成员路径（去掉可选 Videos 包装目录和扩展名、casefold），避免仅 basename 的潜在冲突；线上 reducer 不读取这些离线审计字段。`A` 是官方代码的正常判断标记；`B*`/`G` 类别不应被用作线上压缩路由。

### 2. 官方 train/test 身份

原论文明确给出 4,754 个视频：训练 3,954，测试 800；测试包含 500 暴力和 300 非暴力视频。作者项目页把 V1.0 原视频分为 5 个训练 ZIP（`0001–1004`、`1005–2004`、`2005–2804`、`2805–3319`、`3320–3954`）和一个测试视频 ZIP。因此 raw importer 的 split 契约应来自**下载卷成员身份**：五个训练卷的 `.mp4` 为官方 train，测试卷的 `.mp4` 为官方 test；不通过类别、文件名数字、音频或 test annotation 推断 split。

对于当前 RGB-only 项目，保存 `archive_volume`、archive 相对路径、完整 video ID、视频级 `weak_label` 与原始 label token；按 archive member count 验证 train=3,954、test=800 后才接受 manifest。官方 README/论文未提供另一个已核对的“原视频名→split”文本清单，故不得擅自用 lexicographic 或类别规则替代卷身份。

**训练成员 quarantine 口径（2026-09-20 增补，与代码 `official_training.py` 一致）**：两次独立审计一致确认官方 train 3954 名成员中 4 个在官方源损坏、无法修复（quarantine 隔离清单为独立文件，含成员 ID、损坏证据指针与 SHA，`schema=icassp2027.xd-official-quarantine/v1`）。官方训练视图在绑定该清单后可被接受为 `declared=3954 − quarantined=4 = accepted 3950`，契约如实记录 `declared_members` / `accepted_members` / `quarantined` / `disclosure_note`；无清单仍 fail-closed（视图不 ready）。**论文必须如实报告 XD 训练集为 3950/3954（接受/隔离）**，不得声称完整 3954 训练；UCF 1610 无此豁免，"完整"定义不变。

### 3. 时间坐标与帧率

官方 ReadMe 把每个 test 视频的一对或多对数称为 violent event 的 start/end **frame**；原论文也称对测试暴力视频标记 violent event 的 start/end frames。作者历史特征流程将视频固定为 24 FPS，使用 16-frame sliding window；其特征评测代码将每个 snippet 分数 `np.repeat(..., 16)` 再与 `gt` 比较。

以下三项在所核对的一手资料中**没有明确、可安全实现的定义**：

- frame index 是 0-based 还是 1-based；
- end frame 是包含端点还是半开区间；
- 标注坐标究竟对应原始解码帧序列，还是作者特征提取时的 24-FPS 重采样时间线。

因此 importer 不得把 UCF 的端点转换规则移植到 XD，也不得由文件名或 `gt.npy` 长度猜测。应保留原始 annotation token 对，另行在合法的冻结后评测阶段以公开的 parser/小范围人工约束确定端点和时间线；在此之前，契约状态为 `unknown_explicitly`。

### 4. XD 主 AP 的官方算法

作者 [infer.py](https://github.com/Roc-Ng/XDVioDet/blob/f846d8cfc454943ec165bd2489b71e7dc0a1064d/infer.py#L15-L22) 加载 `gt` 后调用 [test.py](https://github.com/Roc-Ng/XDVioDet/blob/f846d8cfc454943ec165bd2489b71e7dc0a1064d/test.py#L1-L34)。后者的精确实现是：

```python
precision, recall, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
pr_auc = auc(recall, precision)
```

所以作者官方 “AP” 是 **`precision_recall_curve` 后 `sklearn.metrics.auc(recall, precision)` 的梯形面积**，不是 `sklearn.metrics.average_precision_score` 的阶梯式 average precision，也不是插值 mAP。项目若选择更标准的 `average_precision_score`，必须另列为新评测口径，不能声称逐数复现作者官方 AP。

### 5. `gt.npy` 能作为 canonical truth 的范围

**对作者的预提取 feature 评测链路，可以。**官方 commit 的 `option.py` 默认 `gt` 为 `list/gt.npy`；`infer.py` 直接 `np.load(args.gt)`；`test.py` 把平铺预测向量逐项 repeat 16 后交给该二元向量。对 `gt.npy` 的后续只读 NPY header Range 检查表明它是 `<f4`、shape `(2,330,384,)`；没有读取任何 array value。因此官方 code 要求的总 feature-score 数为 `2,330,384 / 16 = 145,649`。这定义了官方 feature pipeline 的 reference truth vector，不需要由事件端点重建该 vector。

测试 feature 排序也有部分可核对的证据。官方仓库的 `rgb_test.list`、`flow_test.list` 各有 4,000 entries，`audio_test.list` 有 800；每 5 条连续 RGB entry 是同一个 video 的 5 crops，且其 base video 顺序与 800 条 audio list 完全对应。`infer.py` 的 DataLoader 固定 `batch_size=5, shuffle=False`，`test.py` 对 batch 维取均值后追加时间分数。因此官方链路的预期顺序是 800 个有序视频、每个视频由 5-crop RGB 组平均为一条时间分数序列。

**对当前 raw RGB ViT evaluator，仍然不足。**`Dataset(..., test_mode=True)` 返回原始 feature array，并不调用 `process_feat`；`model.py` 的评分路径保留 feature 时间长度。官方仓库本身没有提供可审计的每视频 `.npy` shape manifest、每个视频在 `gt.npy` 中的 slice boundary，或原始视频时间戳到该 feature grid 的映射。本文第 6 节的独立 archive metadata 审计已补齐前两项的 canonical feature-grid `T_i` 与 slice boundary；剩余缺口是原始视频时间戳、FPS/resampling、尾部 pad/drop 和 annotation endpoint 到该 grid 的映射。全局 `gt` 长度不能反推这些 raw 时序语义，也不能用 UCF 的时间规则填补。

所以实现应区分两层：`official_feature_grid` 模式可在提供了已验证 `test_list + 每视频 feature row counts + canonical gt vector` 的情况下严格复现 `repeat(16)` 与梯形 AP；`raw_video_grid` 模式在缺少上述映射时必须标为 `explicit_blocked`，不能把 canonical vector 静默重采样、按总长度切片或拿它选择任何方法。`gt.npy` 值仅可在冻结后评测阶段作为 canonical truth 读入，不能用于探针、层/预算或阈值选择。

### 6. 官方 feature archive 对每视频 `T` 的可达性

作者项目页的 visual-feature OneDrive entry 当前是 `i3d-features.zip`，大小 41,176,272,748 bytes。只读 ZIP EOCD/Zip64/central-directory 检查成功：Zip64 central directory 为 7,520,475 bytes，总元数据读取 7,586,088 bytes，低于本轮 32 MiB 限制；其中恰有 4,000 个 `RGBTest` `.npy` entries，且 compression method 为 Deflate。这与官方 `rgb_test.list` 的 4,000 crops 一致。

该 archive 对 member Range 的行为需要明确处理：它会把请求的起点延长到 archive EOF，但响应的 `Content-Range` 起点与已验证的 ZIP local-header offset 精确一致，total 也等于已核对 archive 长度。因此在 `ResponseHeadersRead` 下接受这类响应、流式读取硬上限 4,096 bytes 后立刻 dispose，仍可安全取得 NPY header，不会接收长尾 body。两个同一 test 视频的不同 RGB crop 分别直接核对为 NPY v1.0、`<f4`、shape `(360, 1024)`；每次只读 4,096 bytes，array values 不检查。

两个直接 header 的布局检查先表明全部 4,000 个 RGBTest members 的 central-directory `uncompressed_size` 满足 `128 + 4,096 × T`，且每个有序视频的五 crop size 相同。随后不再把该关系当作最终证据：对 800 个有序测试视频各选 crop 0，逐一直接验证 ZIP local header、NPY magic、NPY header length、dtype `<f4` 和 shape `(T_i, 1024)`。初次扫描 792/800 通过；余下 8 条因 scan record 没有携带 local-header offset 而失败，不是 header 不匹配。由 central directory 补回 offset 后，以同一严格 4 KiB prefix 机制重试，8/8 通过。

最终 `rgbtest-video-t.json` 含 800/800 `header_verified=true`、`verification_method='direct_npy_header'` records，包括 `video_index`、`video_id`、feature name、`T_i`、`gt_start` 和半开 `gt_end_exclusive`。其 `Σ(T_i × 16)` 为 **2,330,384**，与只读 GT header 的向量长度精确相等；canonical records SHA-256 为 `427913f7fe9108dd597fb80bc3c0a417ce76e00e3017af4f0fea7b56670690a2`。扫描完成本身不是 metadata freeze；外部 evaluator wrapper 还应冻结整个 metadata、official test list、GT header和 archive central-directory provenance。

这使 `official_feature_grid` 具备严格的每视频 canonical GT slice boundary，可在冻结后严格复现 `repeat(16)` 与官方梯形 AP。它**不**建立 raw-video timestamp、原始 FPS、24-FPS resampling、尾部 pad/drop 或 annotation endpoint 映射。`raw_video_grid` 因此仍为 `explicit_blocked`；不得把 feature-grid offsets 解释成原始视频帧坐标或用来选择方法。

### 7. 与 RGB-only importer/evaluator 的边界

作者官方代码运行在预提取 RGB/flow/audio features 上，`dataset.py` 从 feature list 读取 `.npy`；它不是 raw-video reader。论文的 24 FPS、16-frame 设定只能作为历史 feature/evaluation 对齐证据，不能自动规定四个 ViT 的输入 FPS、clip 长度或采样策略。项目的 raw RGB pipeline 应独立记录实际解码帧 index、时间戳、采样 coordinate、有效帧 mask 与预测回填；所有方法共用同一测试采样。

正式 XD 主指标应在方法冻结后，以协议确认的时间线和官方代码定义计算。任何探索、探针、校准、层/预算选择仍只使用官方 train 的视频级标签；测试时间标注只能用于冻结后的评测。

### 8. raw video → 官方 feature grid 的一手证据边界

作者原论文只明确：视觉特征提取前将视频固定为 **24 FPS**，并使用 **16-frame sliding window**。它没有规定滑窗 stride、第一窗的精确起点、是否采用连续或均匀抽帧、末尾不足 16 帧时 drop/pad/repeat 的规则，也没有给出一个 raw frame index 到 I3D row index 的公式。

作者官方 `XDVioDet` commit `f846d8c` 只提供消费 `.npy` feature 的 `dataset.py`、`infer.py`、`test.py`、list generator 和模型代码；没有 raw-video I3D extraction implementation。因此不能由官方 code 补足上述未写明的 stride/尾部规则。官方 feature grid 的 `T_i` 和 canonical GT slice 已能严格复现 feature-domain AP，但它们不能被提升为 raw-video frame timing 契约。

进一步只检索作者官方 `XDVioDet` issues/公开链接中的 `feature extraction`、`stride`、`16`、`fps`、`ground truth` 与 `annotation`：可见用户提出 extraction script/ambiguous segments 问题，但没有找到作者回复或作者链接的上游 extraction 仓库来规定 stride、起点或尾部行为。没有作者可归属的一手补充实现时，第三方 extraction scripts 不能升级这一契约。

作为有限一致性检查，从已验证 SHA-256 的第一训练卷按相对文件名 SHA-256 固定选取 5 个 `_label_A` 正常和 5 个非 `_label_A` 视频。只用 `ffprobe` 读取容器 `nb_frames`、FPS 和 duration；所有 10 个样本均报告 24/1。再对官方 RGBTrain crop-0 feature 读取最多 4 KiB prefix 并直接验证 NPY header 的 `T`、`<f4` 和 1,024 维，未读取 feature array values。候选匹配数为：`floor(num_frames/16)` 9/10、`ceil(num_frames/16)` 0/10、stride-1 valid windows 0/10。一个明确反例是 2,448 raw frames 对应 `T=152`，而 floor/ceil 都为 153。

这支持“下载的训练 raw 容器本身已报告 24 FPS、非重叠 16-frame 计数在多数样本上有一致性”的**有限描述**，也反驳把任何这三条简单公式当作全数据集契约。它不能确定采样起点、resampling 细节、末尾行为或全部视频的实际 feature extraction；不更新 `raw_video_grid=explicit_blocked`。详细的 10 条 metadata 与候选计数只保存在已忽略的 `xd_train10_raw_feature_consistency_receipt_20260918.json`。

预先 hash 选择的训练视频容器 metadata（`num_frames`/FPS）即使与 RGBTrain feature header 作比较，也最多提供有限的一致性或反例：在缺少作者 extraction stride、起点和尾部规则时，它无法唯一证明全数据集 raw frame→feature row 映射。为避免把局部巧合写成协议，本轮把这类比较明确标为有限一致性检查而非 raw mapping 证据。

解除 `raw_video_grid=explicit_blocked` 的后续证据不只依赖作者发布新代码或回复。方法冻结后可进行独立、dataset-only 的对齐审计：只比较官方 annotation 与 canonical 800-video GT slice，在预先枚举的 0/1-based、inclusive/half-open 端点及每视频 `L=16T_i` prefix 截尾候选中寻找能逐位一致的唯一规则；不读取模型分数、不做方法选择，也不据此改层、预算或阈值。只有这种唯一的标签—canonical-GT 一致性，加上独立 raw 容器 FPS/帧数证据，才能形成可核验的 raw mapping；在审计实际完成前，未知事实仍保持未知。

## 最小审计来源

| 来源 | 用途 | 已核实内容 |
|---|---|---|
| [作者项目页](https://roc-ng.github.io/XD-Violence/) | 原视频卷、训练/测试划分和官方代码入口 | 5 个训练 ZIP、1 个测试 ZIP；features 与 videos 分离。 |
| [ECCV 2020 原论文](https://roc-ng.github.io/XD-Violence/images/paper.pdf) | 数据规模、测试构成、标注单位、24 FPS/16-frame 历史流程、PRC/AP 动机 | 3,954/800、500/300、start/end frames、24 FPS、16 frames。 |
| [官方 annotation ReadMe](https://roc-ng.github.io/XD-Violence/images/ReadMe.md) | 文件名 label token 和多事件格式 | `B1/B2/B4/B5/B6/G` token；一对或多对 start/end frame。 |
| [官方 XDVioDet commit `f846d8c`](https://github.com/Roc-Ng/XDVioDet/tree/f846d8cfc454943ec165bd2489b71e7dc0a1064d) | 二元正常规则、feature score 与 `gt` 的展开、AP 函数 | `_label_A` 正常；`repeat(16)`；PRC 后梯形 `auc`。 |

被核对的官方源码文件与 SHA-256、未知项和未读取资产列在已忽略的 `outputs/icassp2027/source_checks/xd_protocol_primary_evidence_20260918.json`；feature archive 单 member 可达性回执在 `xd_feature_header_access_receipt_20260918.json`。本核查没有提供实际 test label、预测值或实验数字。
