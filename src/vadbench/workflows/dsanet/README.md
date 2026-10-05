# DSANet 三阶段入口

这个包把 `work/dsanet-extension-20260923-r01/` 中已执行过的三个脚本接入可安装的 `vadbench` 源码。提取、评分和质量导出仍分别调用原有算法；此改动没有启动或新增模型实验，也不改历史工作目录中的脚本与回执。

原 CLIP/DSANet 环境是 Python 3.9。该环境运行 `extract`/`score` 时应在仓库根目录直接执行 `scripts/icassp2027/dsanet_legacy.py`，无需设置 `PYTHONPATH`；脚本通过私有包名装载同一套解析器和执行模块，绕过当前 `vadbench` 顶层注册中不受 Python 3.9 支持的 `dataclass(slots=True)`。准备 PyTorch、torchvision、OpenCV、Pillow、NumPy；`extract` 另需本地 OpenAI CLIP 源码和 ViT-B/16 权重，`score` 另需 DSANet 作者源码及对应数据集的固定 checkpoint。`quality` 使用 Python 3.10+ 环境的模块命令，调用 `vadbench.workflows.quality_comparison` 和 `vadbench.workflows.urdmu_quality_export`。Python 3.10+ 也可对三个阶段统一使用 `python -m vadbench.workflows.dsanet`。这些帮助菜单不加载模型或 GPU。

下面每一步中的路径是必需输入，需替换为本机真实路径。每个输出目录必须是尚不存在的新目录；提取阶段只有在同一身份的已有目录上才能用 `--resume`。以下以 UCF dense 为例，压缩档在提取命令增加 `--keep-ratio 0.8`、`0.6` 或 `0.4`，并使用独立输出目录。

```bash
python scripts/icassp2027/dsanet_legacy.py extract \
  --manifest /path/manifests/ucf_test.jsonl \
  --clip-repo /path/openai-clip --weights /path/ViT-B-16.pt \
  --output /path/features/ucf-dense --gpu 0

python scripts/icassp2027/dsanet_legacy.py score \
  --dataset ucf --upstream-src /path/DSANet \
  --manifest /path/manifests/ucf_test.jsonl \
  --features /path/features/ucf-dense \
  --checkpoint /path/model_ucf.pth \
  --output /path/scores/ucf-dense

python -m vadbench.workflows.dsanet quality \
  --dataset ucf --scores-root /path/scores \
  --truth /path/sealed_truth/ucf.jsonl \
  --checkpoint-origin published --output /path/quality/ucf
```

`extract` 默认使用 `src/vadbench/token_reduction/bridges/clip.py` 当前实现：Python 3.9 启动脚本自动传入其绝对路径，Python 3.10+ 模块命令直接导入它，两者都在 `resolved.json` 记录同一 bridge 文件 SHA256。只有重放历史代码身份时才传 `--bridge /path/clip_bridge.py`；该路径应指向要重放的实际 Python 文件。提取读取含 `video_id`、`raw_path`、`role`、`crop_ids` 的 JSONL manifest，输出 `features/*.npy`、`receipts/*.json`、`resolved.json` 和 `summary.json`。`score` 读取同一 manifest、提取回执和固定权重，输出逐视频 `predictions.jsonl` 与 `summary.json`；此阶段不接收或读取帧级真值。`quality` 在评分全部封存后读取 `scores-root/<dataset>-{dense,0.8,0.6,0.4}/` 和该数据集的帧级真值，输出帧区间 JSONL 与 `quality.json`。XD 改用 `--dataset xd`、XD manifest、XD checkpoint 和 XD 真值。

来源分别是历史 `extract_clip_features.py`、`score_dsanet_fixed.py`、`export_dsanet_quality.py`。质量导出继续遵守 UCF 290 视频、XD 800 视频覆盖要求及原有 frame ROC-AUC、PR-AUC、AP 实现；正式 test 分数只在 `quality` 阶段读取。这个入口不包含训练、benchmark 或多任务调度。
