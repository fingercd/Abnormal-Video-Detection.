# Encoder 四组隔离环境 v2 实施与复核记录

> 首次实施：2026-09-03
>
> 只读复核：2026-09-11
>
> 服务器根：`/users/fotile/VAD`

## 这份文档记录什么

本文记录服务器四组环境的落地事实和带日期的验证证据。环境的声明配置以 [`registry/encoder-environments-v2.yaml`](../../registry/encoder-environments-v2.yaml) 为准；25 路候选与 21 路运行目标的区别见 [`encoder-integration-matrix.md`](encoder-integration-matrix.md)；代码如何选择环境、加载 adapter 和写 smoke 产物见 [`encoder-runtime.md`](../architecture/encoder-runtime.md)。当前实现和后续验证汇总见 [`2026-09-11-implementation.md`](2026-09-11-implementation.md)。

`smoke_pass` 是一次既有运行的结论，不是环境或 catalog 的永久属性；catalog 只保存 `planned`、`integrated`、`blocked` 的静态登记状态。2026-09-11 的复核检查了代码版本、路径、解释器和已有产物，并由全量测试重新校验 8 条登记权重；没有重新执行模型 forward，其余大权重没有在本轮全部重算摘要。

## 四组环境的声明与 2026-09-11 实况

| 组 | 服务器路径 | registry 固定版本 | 2026-09-11 只读复核 |
|---|---|---|---|
| `classic-video-v2` | `.encoder-envs/v2/classic-video-v2` | Python 3.10.20；Torch 2.3.0+cu121；TorchVision 0.18.0；Transformers 4.37.2；PyTorchVideo 0.1.5 | Python 3.10.20、Torch 2.3.0+cu121、CUDA 12.1 可导入 |
| `foundation-video-v2` | `.encoder-envs/v2/foundation-video-v2` | Python 3.10.20；Torch 2.8.0；TorchVision 0.23.0；Transformers 4.57.3 | Python 3.10.20、Torch 2.8.0、CUDA 12.9 可导入 |
| `visual-vlm-v2` | `.encoder-envs/v2/visual-vlm-v2` | Python 3.11.15；Torch 2.5.1+cu124；TorchVision 0.20.1+cu124 | Python 3.11.15、Torch 2.5.1+cu124、CUDA 12.4 可导入 |
| `stream-kv-v2` | `.encoder-envs/v2/stream-kv-v2` | Python 3.11.15；Torch 2.5.1+cu124；TorchVision 0.20.1+cu124 | Python 3.11.15、Torch 2.5.1+cu124、CUDA 12.4 可导入 |

七个模型还使用独立 overlay：`videomaev2`、`videomamba`、`longvu`、`videochat_online`、`videochat_flash`、`streaming_vlm`、`hermes_llava_ov`。overlay 的包版本和路径只在 environment registry 维护，本文不复制完整清单。

服务器 runner [`scripts/server/run_native_encoder_matrix_v2.py`](../../scripts/server/run_native_encoder_matrix_v2.py) 才负责按目标选择组内 Python、拼接 overlay `PYTHONPATH`、设置离线 cache 目录并启动子进程。catalog 中的 `environment.runtime` 本身不会切换解释器；通用 integration matrix 的默认 runner 也不会自动选择这里的四组环境。

## 2026-09-03 迁移结果

迁移建立四个新环境与 `external-v2`/`weights-v2` 写入边界，保留旧环境只读。Conda clone 后对 foundation 环境恢复 28 个 CUDA 动态库软链；visual/stream 组补齐 OpenCV 和音频动态库依赖。classic 的 decord wheel 元数据警告、foundation 从种子环境继承的 InternNav/Habitat 无关依赖冲突均保留在当次 `pip-check` 产物中，因此不能把四组环境表述为 `pip check` 全绿。

当次 native smoke 使用 `data/smoke/mlvu-surveil-8.mp4`（SHA256 `5c7dd43429c5e556de67489920a799af8fdb614a089ab52c04b1c3b044703963`）在 CPU 跑得：

| 结果类别 | 数量 | 含义 |
|---|---:|---|
| `smoke_pass` | 14 | 目标自己的代码、权重和真实视频 forward 通过 |
| `blocked_license` | 2 | VideoChat-Online、StreamingVLM 技术前向通过，但许可状态未闭合 |
| `manual_required` | 5 | C3D、InternVideo2、VideoChat、MA-LMM、MovieChat 缺人工资产或完整依赖链 |
| `unregistered` | 4 | UniFormerV2、UMT、InfiniPot-V、MuKV 只在候选表中 |

重构后的历史矩阵 `outputs/refactor/final-native-72faada/matrix-v2.json` 仍保存 14 个 `smoke_pass` 和 7 个 `skipped`（5 个 `manual_asset_missing`、2 个 `license_blocked`），文件 SHA256 为 `6f550410dc14bbe487143366f2af8638200b0babed39d4feed55d312e38dfea1`。原迁移矩阵 `outputs/environment-migration-v2/native-smoke-matrix.json` 的 SHA256 为 `0c5b6ae5e3bd50c07badaff0b574dfa04269f3b4fc428e6056c793ea67b27a47`。两者都是历史证据，不替代新运行。

## 2026-09-11 服务器复核边界

服务器位于分支 `qzt/refactor-vadbench-simplification`、commit `0badc3435e734a841110e29d497940bfda7cc707`，检查时工作树干净。四组 Python/Torch 均可导入，磁盘可用约 495.6 GB，高于 registry 的 400 GiB 门禁。Windows 全量测试为 366 passed、9 skipped；node3 全量测试为 374 passed、1 skipped。服务器测试重新计算并匹配 R(2+1)D、MViTv2、Video Swin、I3D、X3D、SlowFast、VideoMAE V2 和 HERMES 共 8 条权重摘要，但没有执行这些模型的 forward。

UCF-Crime 软链解析到 `/users/fotile/datasets/UCF-Crime`，但目标目录文件数为 0，标准 train/test manifest 也不存在，因此不能开始真实 UCF benchmark，也不能从本次文档复核推导任何 AUC/AP 结论。精简后的复核证据见 [`server-doc-audit-2026-09-11.json`](../evidence/server-doc-audit-2026-09-11.json)。

## 复现和更新要求

1. 用 `manage_encoder_envs_v2.py verify` 核对路径、marker、保护目录和磁盘门禁。
2. 用 `fetch_encoder_assets_v2.py` 与 `prepare_encoder_overlays_v2.py` 准备本地资产；不得让 adapter 隐式联网。
3. 用 `run_native_encoder_matrix_v2.py` 执行真权重 smoke。流式目标至少两 chunk；许可未闭合的目标只有显式传入 `--include-license-blocked` 才运行，结果仍降级为 `blocked_license`。
4. 每次新运行保存 commit、dirty 状态、环境 marker、overlay marker、视频身份、结果和 launcher log。只有这次新产物可以更新历史运行证据；不得把它写回 catalog 的静态状态。
