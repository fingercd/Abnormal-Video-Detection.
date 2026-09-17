当前训练只使用 train_end2end.py：
  python -m lab_anomaly.train.train_end2end

它要求预先生成的 NPZ + manifest.json；不会从原视频临时切片。
旧 embedding/KMeans/OCSVM 训练脚本已经不存在，run_train_known_classifier.ps1/.bat 会直接报“已弃用”。

训练配置没有 CLI 参数，读取文件内 CONFIG 和 configs/train_end2end.yaml。
预切片的 frames_per_clip 等参数必须与训练配置一致；当前默认预切是 16 帧，提交的训练 YAML 是 12 帧，未对齐会被 manifest 检查拒绝。

详细说明见 README.md。该目录的随机视频级验证不等于 UCF-Crime 官方评测。
