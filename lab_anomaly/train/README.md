# `lab_anomaly/train`

当前训练入口只有 `train_end2end.py`。它以预切 RGB 帧训练 `VideoMAE v2 + MIL` 二分类模型：normal 标签为 0，其余标签全部压为 anomaly 标签 1；可选的 ranking 分支仍只使用视频级弱标签。

## 运行条件与命令

训练只接受已预切的 `preclip_root/manifest.json`，不会回退到原始视频现读。先完成 `data/index_build.py` 和 `tool/precompute_clips.py`，并保证 manifest 参数与 `configs/train_end2end.yaml` 一致，然后在仓库根运行：

```powershell
.venv\Scripts\python.exe -m lab_anomaly.train.train_end2end
# 或：.\lab_anomaly\train\run_train_end2end.ps1
```

入口没有训练参数 CLI；修改 `configs/train_end2end.yaml` 或文件内默认配置。脚本以视频行随机切训练/验证集，非 normal/normal 两类都存在时才使用平衡 batch sampler，使每个常规 batch 含一条各类视频。最后一个不足 batch 可能只有一个类别，此时 ranking loss 为 0。

## 阶段与产物

每个 stage 先冻结 backbone，再解冻最后 `unfreeze_blocks` 个识别出的 Transformer block；优化器在每个 stage 重新创建。冻结阶段可把 backbone 置为 eval 模式，并以 `torch.no_grad()` 编码；CUDA 且 `encoder_use_half` 为真时使用 AMP + GradScaler。

`out_dir` 下会写：

```text
checkpoint_best.pt       按验证 accuracy 选择，不按 AUC
checkpoint_last.pt
labels.json              固定 normal/anomaly 映射
history.json
plot_loss_curves.png, plot_acc.png          仅 matplotlib 可用时
confusion_matrix_val.png                    仅 matplotlib 可用且验证集非空时
```

checkpoint 同时包含 encoder 和 MIL 的 state/config，供 `infer/scoring.py` 和 `infer/rtsp_service.py` 加载。它不包含数据集版本、CSV hash、代码 revision、采样 provenance 或可复现实验 run 信息，不能当作 VADBench 产物。

`plot_training_log.py` 只解析旧的文本日志格式；当前训练应直接使用 `history.json`。`run_train_known_classifier.*` 已明确弃用，不能运行。

## 评测边界

验证 accuracy/AUC 是视频级随机 holdout 指标。没有官方 split、按摄像头分组、分层、帧级标签投影或测试集覆盖校验；它不适合用于 UCF-Crime 的正式 frame ROC-AUC/AP 报告。
