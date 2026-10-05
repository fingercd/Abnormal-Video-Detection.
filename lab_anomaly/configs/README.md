# `lab_anomaly/configs`

本目录只包含两个已提交 YAML：

| 文件 | 被谁读取 | 实际用途 |
|---|---|---|
| `train_end2end.yaml` | `train/train_end2end.py` | 覆盖训练脚本的顶层 `CONFIG`。 |
| `rtsp_service_example.yaml` | `infer/rtsp_service.py --config ...` | 本地视频/RTSP 服务的示例。含占位 RTSP 凭据，不能直接使用。 |

## 训练配置

`train_end2end.py` 没有 argparse；启动时总是读取内置 `CONFIG["yaml_config"]` 指向的 `train_end2end.yaml`，然后以 YAML 的**顶层键**覆盖同名默认值。`stages` 是一个完整列表，写入 YAML 会替换内置的整个列表，不会逐项合并。

当前 YAML 的关键值是 `frames_per_clip: 12`、`interval_sec: 8.0`、`max_clips_per_video: 16` 与 `preclip_root: lab_dataset/derived/preclips`。预切片 manifest 会严格匹配下列字段：`dataset_root`、`labels_csv`、`frames_per_clip`、`interval_sec`、`max_clips_per_video`、`exclude_unknown`、`normal_label`。改变其中任一项后必须重新预切片。

`encoder_use_half` 只在 CUDA 上开启 autocast；编码器权重仍保持 FP32。`micro_batch_clips` 控制一次送入 VideoMAE 的 clip 数量；每个 stage 可以单独覆盖它。`label` 会被压成 normal / anomaly 二类，所有非 normal 标签均为 anomaly。

仓库没有 `precompute_clips.yaml`，但 `tool/precompute_clips.py` 会尝试读取该固定路径。要避免当前 16 帧预切默认值和训练 YAML 的 12 帧冲突，可以在本地创建该文件并写入与训练配置相同的预切字段，或直接改工具内的 `CONFIG`。

## RTSP 示例配置

服务可读取 YAML 或 JSON。有效的配置层级为：

```text
rtsp_url, camera_id, out_dir
artifacts.known_checkpoint, artifacts.open_set_dir
encoder.model_name, encoder.device, encoder.use_half
sampling.clip_len, sampling.frame_stride, sampling.sample_fps, sampling.window_stride
fusion.min_known_prob, fusion.known_alarm_prob, fusion.treat_low_conf_as_unknown,
fusion.unknown_alarm, fusion.ranking_alarm_threshold
outputs.save_snapshot, outputs.api_url, outputs.api_timeout_sec
runtime.loop_video, runtime.min_consecutive, runtime.cooldown_sec,
runtime.open_timeout_sec, runtime.reconnect_wait_sec, runtime.show
```

命令行中非空的参数优先。当前实现有两个容易误解的点：`--camera_id` 和 `--out_dir` 自带非空默认值，所以即使传了 `--config`，示例 YAML 的这两个根字段也会被命令行默认值覆盖；应显式通过 CLI 设定它们。`sampling.frame_stride`、`encoder.model_name`、`encoder.use_dual_stream`、`runtime.inference_use_rgb_only` 和 `artifacts.open_set_dir` 会被读入配置对象，但不参与现有打分路径。

示例的 `clip_len: 12` 应与产生 checkpoint 的 `encoder_cfg.num_frames` 相同。不要把真实 RTSP 凭据写入受版本控制的 YAML。
