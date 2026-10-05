# `lab_anomaly/infer`

本目录服务于保留的本地二分类 checkpoint，不接入 VADBench 的 feature store、prediction schema 或 UCF-Crime 评测。

| 文件 | 当前行为 |
|---|---|
| `scoring.py` | 从端到端 checkpoint 重建 VideoMAE v2 和 MIL 头，并融合二分类 softmax 与 ranking 分数。 |
| `rtsp_service.py` | 独立进程：读取本地视频或 RTSP，滑窗打分、去抖、写 JSONL/截图，可选 HTTP POST。 |
| `known_event_runtime.py` | 可被另一个 Python 程序实例化的异步多流类；本仓库没有它的调用方。 |

## 独立服务

在仓库根运行。先用本地视频，避免无意启动 RTSP 循环或网络报警：

```powershell
.venv\Scripts\python.exe -m lab_anomaly.infer.rtsp_service `
  --config lab_anomaly/configs/rtsp_service_example.yaml `
  --video <视频文件> `
  --known_checkpoint <训练产物/checkpoint_best.pt> `
  --camera_id local-test `
  --out_dir <可写输出目录>
```

checkpoint 必须含 `encoder_cfg`、`encoder_state`、`mil_cfg` 和 `mil_state`，即 `train_end2end.py` 生成的格式。服务忽略 checkpoint 外的 `--model_name`：加载 checkpoint 中 encoder state 时，实际模型设置来自 checkpoint 的 `encoder_cfg`。

服务从视频源按 `sample_fps` 近似降采样，保留最近 `clip_len` 帧，每得到 `window_stride` 个采样帧才推理一次。每次推理只向 MIL 传一个 embedding，因此输出是滑窗级分类/报警，不是原始视频帧级异常曲线。`events.jsonl` 仅在触发报警时追加；截图位于 `snapshots/<camera_id>/`；`api_url` 非空时会 POST 同一事件 JSON。

报警先计算 `predict_fusion()`：非 normal 且概率达到 `min_known_prob`，或 ranking 分数达到 `ranking_alarm_threshold`，就被视为异常。随后服务对“已知类”使用 `known_alarm_prob`，并可让任何 fusion 异常走 `unknown_alarm` 路径；最后经过 `min_consecutive` 和 `cooldown_sec`。因此阈值的含义并非单一二分类阈值，应先离线校准再连接真实告警端点。

## 配置的实际边界

传 `--config` 时，命令行的非空值优先；但 `--camera_id` 默认 `cam0`、`--out_dir` 默认 `lab_dataset/derived/realtime`，会压过 YAML 的同名根字段。通过 CLI 明确指定这两个值。

`sampling.frame_stride`、`encoder.use_dual_stream`、`runtime.inference_use_rgb_only`、`artifacts.open_set_dir` 会进入 `ServiceConfig`，当前打分循环并未使用。`--show`、`--save_snapshot`、`--unknown_alarm`、`--treat_low_conf_as_unknown` 是单向开启开关，不能通过 CLI 将 YAML 中的 true 改为 false。

不传 `--config` 时，脚本会把文件末尾 `BUILDIN_DEFAULTS` 注入缺失参数；其中含开发机绝对路径，不能视为可移植默认配置。

## `KnownEventRuntime`

构造 `KnownEventRuntime(known_checkpoint, frames_per_clip=..., window_stride=...)` 后，对每帧 BGR 调用 `add_frame(stream_id, frame, state=None)`。每路各有一个长度为 `frames_per_clip` 的缓冲，任务写入一个最多 32 项的后台队列；满队列时该次窗口被丢弃。结果可由 `get_result(stream_id)` 获取。若传入的 `state` 是以 stream id 为键、值拥有 `vit_event` 属性的字典，后台线程会更新该属性。

该类没有停止方法或线程生命周期管理，也没有本仓库内集成点；它是待调用方自行管理的实验接口。
