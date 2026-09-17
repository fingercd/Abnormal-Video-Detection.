# `lab_anomaly/data`

这里的数据格式只服务于保留的本地原型，不是 VADBench/UCF-Crime manifest。

## 输入 CSV

`video_labels.csv` 采用 UTF-8 BOM，列为：

```text
video_id,video_path,label,camera_id,start_time,end_time,note
```

`video_path` 可相对 `dataset_root`，也可为绝对路径。`start_time` 与 `end_time` 是可选秒数；只有两者构成合法正区间时才限制采样范围，否则代码回退到整段视频。它们不是帧级异常真值。

推荐目录是 `lab_dataset/raw_videos/<label>/<可选 camera_id>/<video>`。运行：

```powershell
.venv\Scripts\python.exe -m lab_anomaly.data.index_build `
  --dataset_root lab_dataset `
  --videos_root lab_dataset/raw_videos `
  --out_csv lab_dataset/labels/video_labels.csv
```

脚本递归扫描 `mp4/avi/mov/mkv/webm/m4v`，以第一层目录名设置 `label`、第二层目录名设置 `camera_id`，并保留 CSV 中扫描不到的旧行。它会覆盖已扫描视频原先手工改过的标签；应在目录组织稳定后再运行。

## 片段与预切产物

`clip_dataset.py` 先将有效时间范围按 `interval_sec` 等分，最多取 `max_clips_per_video` 段；每段均匀采样 `frames_per_clip` 帧，返回 RGB `uint8` HWC 帧列表。OpenCV 读取失败时，读取器会复用上一帧；首帧也取不到时会用黑色 `224×224` 帧，因此预切完成不等于视频内容已经通过质量审计。

训练要求预切模式：`tool/precompute_clips.py` 写入 `preclips/<safe-video-id>/cNNNN.npz`、`manifest.json` 与 `precompute_meta.json`。NPZ 的唯一键为 `frames`，形状为 `(T,H,W,C)`、类型为 `uint8`。训练加载 manifest 后只读取 NPZ，不再打开源视频。

manifest 绑定 CSV 绝对路径、数据根、采样参数和标签过滤设置；CSV 行数、行顺序或上述参数变化都会使 `VideoClipDataset` 拒绝复用。它不验证 manifest 中的 `video_id`、`label` 与当前 CSV 内容逐项一致，因此训练前应固定 CSV，避免静默错配。
