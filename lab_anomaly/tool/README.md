# `lab_anomaly/tool`

`precompute_clips.py` 是当前唯一的数据准备工具。它读取 CSV 中未被过滤的视频，按 `clip_dataset.clip_specs_for_row()` 的规则抽帧，写 `.npz` 和 `manifest.json`，供端到端训练只读缓存使用。

从仓库根运行：

```powershell
.venv\Scripts\python.exe -m lab_anomaly.tool.precompute_clips
```

该脚本没有命令行参数。配置来源依次是文件内 `CONFIG`，再用固定路径 `lab_anomaly/configs/precompute_clips.yaml`（文件存在且已安装 PyYAML 时）覆盖。当前仓库没有这个 YAML；默认值为 16 帧、8 秒、最多 16 段、20 个进程，且默认不覆盖既有 NPZ。训练 YAML 目前是 12 帧，因此不应直接按默认值预切。

在运行前，令预切和训练的 `dataset_root`、`labels_csv`、`frames_per_clip`、`interval_sec`、`max_clips_per_video`、`exclude_unknown`、`normal_label` 完全相同。推荐先显式检查 `configs/train_end2end.yaml`，再在本地的 `precompute_clips.yaml` 写入相同字段。随后删除或改用不含旧缓存的新 `preclip_out_dir`；`skip_existing: true` 只检查文件是否存在，并不检查它是否来自当前视频内容。

此工具会解码所有入选源视频、创建/覆盖输出目录中的产物，并按配置启动多进程。不要在服务器数据目录或正式 VADBench 数据上直接运行。VADBench 的特征与清单应使用 `vadbench` 命令，而非此脚本。
