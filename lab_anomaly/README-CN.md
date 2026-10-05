# `lab_anomaly`：保留的 VideoMAE v2 + MIL 原型

`lab_anomaly` 是仓库中保留的一条本地实验线：它把一个视频划为若干固定时长片段，逐片段独立调用 `OpenGVLab/VideoMAEv2-Base`，再用 MIL 头输出**视频级二分类**（`normal` / `anomaly`）。它不是当前 VADBench 的实验入口，也没有实现跨片段状态、KV cache、UCF-Crime 官方划分、帧级评测或 VADBench 的 manifest/provenance/产物契约。

当前基准实验请从仓库根目录的 [README-CN.md](../README-CN.md) 和 `src/vadbench/` 开始；UCF-Crime 的数据与评测规则以 `docs/research/ucf-crime-protocol.md` 为准。本目录的随机验证集只能用于本地原型检查，不能报告为 UCF-Crime 官方测试结果。

## 实际组成

```text
data/index_build.py       扫描 raw_videos，写 video_labels.csv
tool/precompute_clips.py  离线抽帧，写 NPZ 和 manifest.json
train/train_end2end.py    读取 NPZ，训练 VideoMAE v2 + MIL 二分类器
infer/rtsp_service.py     对本地视频或 RTSP 滑窗打分，写报警 JSONL/截图/可选 POST
infer/known_event_runtime.py  供外部 Python 调用的异步多流运行时
```

VideoMAE v2 在此处是无状态固定 clip 编码器。`max_clips_per_video` 只限制一个视频送入 MIL 的片段数，不能解释为长视频缓存或流式等价实现。

## 可执行的原型流程

在仓库根目录运行。依赖清单见 `lab_anomaly/requirements.txt`；首次构造编码器会通过 Hugging Face 的 `trust_remote_code=True` 加载模型，因此需事先准备可用的本地/HF 权重和网络策略。

1. 数据按 `lab_dataset/raw_videos/<label>/<可选 camera_id>/...` 放置，`normal` 是正常类；运行：

   ```powershell
   .venv\Scripts\python.exe -m lab_anomaly.data.index_build
   ```

   它会重建已扫描文件的 `label` 和 `camera_id`，标签来自首级目录名，并保留扫描不到的既有 CSV 行。CSV 格式与时间字段见 [data/readme.txt](data/readme.txt)。

2. 让预切参数与训练配置完全一致，再运行：

   ```powershell
   .venv\Scripts\python.exe -m lab_anomaly.tool.precompute_clips
   ```

   当前提交没有 `configs/precompute_clips.yaml`，而工具默认是 16 帧；已提交的训练 YAML 是 12 帧。因此应先编辑 `tool/precompute_clips.py` 的 `CONFIG`，或自行创建其固定读取的本地 `lab_anomaly/configs/precompute_clips.yaml`，使 `frames_per_clip`、`interval_sec`、`max_clips_per_video`、数据路径、`exclude_unknown` 和 `normal_label` 与 `train_end2end.yaml` 相同。`manifest.json` 会严格校验这些字段，不一致时训练会失败。

3. 检查 `configs/train_end2end.yaml` 中的路径与训练超参数，然后运行：

   ```powershell
   .venv\Scripts\python.exe -m lab_anomaly.train.train_end2end
   ```

   该入口没有训练参数 CLI；它将内置 `CONFIG` 与固定的 `configs/train_end2end.yaml` 做顶层键合并。输出包括 `checkpoint_best.pt`、`checkpoint_last.pt`、`labels.json`、`history.json` 及可选 PNG 图，位置由 `out_dir` 决定。

4. 仅在明确配置本地视频或 RTSP、checkpoint 和输出位置后使用独立服务：

   ```powershell
   .venv\Scripts\python.exe -m lab_anomaly.infer.rtsp_service --config lab_anomaly/configs/rtsp_service_example.yaml --video <视频文件> --known_checkpoint <checkpoint_best.pt>
   ```

   这会打开源、可能持续写文件并在 `api_url` 非空时发送 HTTP POST；先用本地视频验证配置。具体字段和现有限制见 [infer/README.md](infer/README.md)。

## 训练语义与限制

- 标签被折叠为二类：标签经小写化后等于 `normal_label` 的视频为 0，其余标签全为 1。原异常类别不会作为多分类目标保存。
- 每个视频按 `interval_sec` 等分，段内均匀抽 `frames_per_clip` 帧。`start_time` / `end_time` 若合法会限制处理范围；它们不是异常起止标注。
- 验证集由视频行随机切分，未做分层、摄像头/来源去重或官方 train/test 隔离。训练的 AUC 仅在验证集中同时有两类且安装 scikit-learn 时计算。
- MIL 的 ranking 支路针对视频级弱标签优化，实时服务每次只传入一个 clip embedding，因此其 ranking 分数不是帧级定位结果。

目录级细节见 `configs/README.md`、`data/readme.txt`、`tool/README.md`、`train/README.md`、`models/README.md` 和 `infer/README.md`。
