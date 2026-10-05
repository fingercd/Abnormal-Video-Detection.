# 服务器 v2 编码器运行手册

## 当前 node2 连接路线（2026-09-26 验证）

先从公网登录 node3，再在 node3 内通过 `ibnode2` 登录 node2；这条路线已实测成功，Tailscale 故障不再阻塞 node2 维护。第二跳使用批处理模式和严格主机密钥核对：

```bash
ssh -p 12345 fotile@121.196.228.153
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes ibnode2
```

node2 的 `/users` 已核实为来自 `ibnode3:/users` 的 NFS 共享目录；node2 的 `/data2/localdisk` 是本地盘，node3 经 `/data2` 挂载读取。当前六个旧资产根的实际位置与旧路径兼容软链见[WAVAD 资产说明](wavad-assets.md)。连接成功不等于调度器、writer 或 GPU 作业状态已核验。

服务器既有 `.venv` 内仍有早期已安装包。使用源码工作区时先执行 `export PYTHONPATH=/users/fotile/VAD/src`，再运行 `python -m vadbench`；可用 `python -c "import vadbench; print(vadbench.__file__)"` 确认路径。v2服务器脚本和隔离benchmark已通过公共helper显式选择当前源码，无需修改受保护环境。

> 适用工作区：`/users/fotile/VAD`
> 最后按代码与服务器只读观察核对：2026-09-11

本手册只描述当前四组隔离环境和原生 encoder smoke 的实际运行路径。它不把历史 smoke、环境导入成功或数据软链接存在，表述为 UCF-Crime 全量训练、评测或 GPU 性能结果。

## 当前可用边界

`registry/encoder-environments-v2.yaml` 是环境路径、受保护环境、磁盘下限和四组模型归属的唯一配置来源：

| 组 | Python | 受管前缀 | 典型目标 |
|---|---|---|---|
| `classic-video-v2` | 3.10.20 | `.encoder-envs/v2/classic-video-v2` | R(2+1)D、X3D、I3D、VideoMAE |
| `foundation-video-v2` | 3.10.20 | `.encoder-envs/v2/foundation-video-v2` | VideoMAE V2、VideoMamba、V-JEPA 2 |
| `visual-vlm-v2` | 3.11.15 | `.encoder-envs/v2/visual-vlm-v2` | LongVU、VideoChat 系列 |
| `stream-kv-v2` | 3.11.15 | `.encoder-envs/v2/stream-kv-v2` | HERMES、StreamingVLM |

受保护环境是 `.venv`、`.venv-hermes` 及两个既有 Conda 环境。v2 工具只能从它们复制或克隆，不能把它们作为 smoke 的解释器。模型的注册、许可证、资产和上游锁分别以 `registry/encoder-candidates.yaml`、`registry/checkpoints.yaml`、`registry/encoder-integrations.yaml` 与 `integrations/*/upstream.lock.yaml` 为准；不要在命令行重复手填版本、权重路径或 overlay 依赖。

截至本次观察，四个 v2 Python/Torch 环境都能导入且版本符合 registry；机器有 8 张 A100 PCIe 40 GiB，项目所在文件系统可用约 461.6 GiB，高于 400 GiB 下限。这只是某一时刻的资源快照，不能据此认定 GPU 可供作业使用。`data/raw/ucf_crime` 已解析到 `/users/fotile/datasets/UCF-Crime`，但当时目录无视频、默认 train/test manifest 也不存在，因此不能运行真实 UCF-Crime 流程。

历史 `outputs/refactor/final-native-72faada/matrix-v2.json` 记录的是 CPU 下 14 个技术 smoke 通过、7 个跳过的旧尝试；它不是本次代码或当前 GPU 的新结果。

## 先做只读预检

在任何创建环境、复制 overlay、下载资产或调用 GPU 前，先确认工作区和实际资源。以下命令不修改项目逻辑、权重或环境；`inventory_runtime.sh` 会逐个尝试导入候选解释器中的包，并将单个解释器导入失败标在输出中。

```bash
cd /users/fotile/VAD
git status --short --branch
git rev-parse HEAD
git diff --check
bash scripts/server/inventory_runtime.sh
python scripts/fetch_upstreams.py --project-root /users/fotile/VAD --verify-only
```

最后一条只核验传统 `external/<id>` checkout 的 Git commit；v2 的独立 checkout 在 `external-v2/`，它的身份还必须从资产审计 JSON 中检查。`inventory_runtime.sh` 对 `conda env list` 和单个 Python 的导入使用容错继续机制，不能只看 shell 的退出码，必须查看每一段 JSON 中的 `import-error:*` 字段。

下列两个命令不下载或创建环境，但会写入指定的审计 JSON。使用本次运行专属的输出目录，避免覆盖既有证据：

```bash
run_id="preflight-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/server/manage_encoder_envs_v2.py verify \
  --output-root "outputs/operations/${run_id}"
python scripts/server/fetch_encoder_assets_v2.py \
  --output-root "outputs/operations/${run_id}"
```

`verify` 只有所有组核验成功才返回0；资产核验存在missing、manual_required或摘要不符时返回非0并保留JSON。需要只核验单个可运行目标时显式传`--id`。

## v2 的写入式运行流程

只有完成预检、确认工作树与证据目录后才依次执行以下步骤。每一步都应使用新 `run_id`，并保留前一步 JSON 作为后一步的输入证据。

```bash
cd /users/fotile/VAD
run_id="encoder-v2-$(date -u +%Y%m%dT%H%M%SZ)"

# 仅在 v2 前缀尚未建立且确认允许创建时执行；会克隆/复制种子环境并写 marker。
python scripts/server/manage_encoder_envs_v2.py bootstrap \
  --output-root "outputs/operations/${run_id}"

# 复核既有资产；不加 --execute 时不会联网下载。
python scripts/server/fetch_encoder_assets_v2.py \
  --output-root "outputs/operations/${run_id}"

# 仅为有注册 overlay 的模型复制已知可用包；会创建或更新 overlay marker。
python scripts/server/prepare_encoder_overlays_v2.py \
  --output "outputs/operations/${run_id}/overlay-matrix.json"

# 先用一个已许可、已登记目标确认运行路径。真实 GPU 前应由作业调度/用户占用信息确认设备归属。
python scripts/server/run_native_encoder_matrix_v2.py \
  --id videomaev2 \
  --video data/smoke/mlvu-surveil-8.mp4 \
  --device cuda:5 \
  --output-root "outputs/encoder-v2/${run_id}"

# 汇总 25 个候选的状态；它读取已有 matrix，不会重新前向。
python scripts/server/consolidate_encoder_v2_results.py \
  --run-root "outputs/encoder-v2/${run_id}" \
  --output "outputs/operations/${run_id}/native-smoke-matrix.json"
```

`bootstrap` 不是纯检查：它可能创建 v2 前缀、cache、overlay 目录，并对新环境执行已知运行时修复；即使已有 marker，也会重新应用这些修复。它用 protected-environment fingerprint 防止改动旧环境，并在预计剩余空间低于 400 GiB 时终止。不要为了日常 smoke 反复执行它。

按项目约定，`fetch_encoder_assets_v2.py --execute` 在 node2 执行；代码只检查许可证状态为 `verified` 且资产源为 Hugging Face，并未强制核对节点身份。最终下载目标以 `registry/checkpoints.yaml` 的 `local_path` 为准（不保证全部在 `weights-v2/`），临时文件与缓存位于 `.cache-v2/`。它拒绝覆盖已有目标。node3 应只运行已同步且 SHA256 已在审计 JSON 中显示为 `verified` 的资产。下载先在staging校验registry摘要，通过后才发布到最终目录；缺失或未知checkout revision不能标为verified。

`prepare_encoder_overlays_v2.py` 不是包管理器。它按代码中固定的来源，复制少数已知工作的目录到 `.encoder-envs/v2/overlays/<id>`，并经 `PYTHONPATH` 叠加到对应解释器。遇到非空且没有 `.overlay-v2.json` marker 的 overlay 会拒绝继续；不要手工清空、覆盖或把通用 pip 升级混入该目录。

## 原生矩阵的状态、退出码和证据

| 命令 | 正常输出位置 | 退出码语义 | 必须检查的字段 |
|---|---|---|---|
| `manage_encoder_envs_v2.py snapshot-old` | `<output>/old-envs-before.json` | 无异常为 0 | 每个 protected prefix 的 fingerprint |
| `manage_encoder_envs_v2.py bootstrap` | `<output>/environment-matrix.json`、`old-envs-after.json` | 无异常为 0 | `groups[].status`、protected fingerprint |
| `manage_encoder_envs_v2.py verify` | `<output>/environment-matrix.json` | 任何组缺失或版本不匹配返回非0 | `protected_unchanged` 与每组 `status` |
| `fetch_encoder_assets_v2.py` | `<output>/asset-matrix.json`、`manual-download-manifest.json` | 仅所选项全部verified时为0 | `files[].match`、code revision、`items[].status` |
| `prepare_encoder_overlays_v2.py` | 指定 `--output` | 完成返回 0 | `items[].status`、overlay fingerprint |
| `run_native_encoder_matrix_v2.py` | `<output>/matrix-v2.json`、`<output>/<id>/launcher.log`、`<output>/<id>/**/result.json` | 只有所选项全部smoke_pass为0；skip/blocked/failed均非0 | 每项 `status`、`technical_status`、`exit_code`、result/log path |
| `consolidate_encoder_v2_results.py` | 指定 `--output` | 只依据指定run中所选项是否全部成功，无固定数量 | 25 项 `status`、`attempt.matrix_path`、时间与环境身份 |

runner 先检查 GPU 已用显存；`cuda:N` 超过 1024 MiB 会在启动前退出。这个一次性阈值检查不能排除检查后的竞态，也不说明进程所属用户，所以 GPU 作业仍需在启动前后检查调度分配、进程和 `nvidia-smi`。`--include-license-blocked` 只允许做技术核验；即使前向成功，结果仍会重写为 `blocked_license`，不能进入已许可通过集。

每个可接受的 `smoke_pass` 至少应能定位到同一 run 的 `matrix-v2.json`、目标 `result.json` 与 `launcher.log`，并在 result 中核对输入视频身份、输出 shape/dtype、环境 marker、权重/上游身份、缓存遥测和退出码。汇总器只接受显式`--run-root`，历史目录通过`--history-root`仅列出路径，不参与成功判断。runner拒绝已有输出目录，只读本次确定的result路径，且进程必须正常退出。

## 数据链接与真实 UCF-Crime

数据目录只能通过显式软链接接入，视频不复制进仓库。首次接入且确认 `data/raw/ucf_crime` 不存在时：

```bash
bash scripts/server/link_ucf_crime.sh /users/fotile/datasets/UCF-Crime
```

该脚本要求目标是 `/users/` 下已存在的目录，拒绝替换既有 canonical link；若旧 `data/ucf_crime` 链接到不同目标会以 4 退出。运行后先用 `readlink -f`、文件数量和 manifest 导入/验证确认数据完整，再执行任何真实训练或评测。软链接存在不证明数据存在，更不证明官方 1,610/290 split、帧信息、时间标注和泄漏审计已经通过。

## 旧 bootstrap 与辅助脚本

`bootstrap_offline.sh` 与 `bootstrap_hermes_offline.sh` 是旧 `.venv` / `.venv-hermes` 的离线恢复脚本：它们从 `wheels/` 用 `pip --no-index` 安装固定包，并以 `--system-site-packages` 建立环境。它们会修改受保护环境，不能作为 v2 日常预检、v2 bootstrap 或 smoke 的替代。只有在从可信离线 bundle 恢复历史环境、并且该恢复操作已明确纳入变更时才使用。

`install_bundle.sh` 用于接收 `.incoming/` 中的 Git bundle、external/wheels tar 包和 smoke 视频；它会重建/切换工作树、解压文件并更新 remote，属于恢复操作而非诊断命令。`fetch_upstreams.py` 在不带 `--verify-only` 时会 clone/fetch/checkout `external/`，只允许在 node2 获取冻结上游时执行。

`create_smoke_video.py` 创建合成视频，`create_pipeline_smoke_fixture.py` 创建合成特征、manifest 和预测；二者只能检查视频/训练/评测链路，不能替代真实权重 smoke 或 UCF-Crime 结论。`inventory_runtime.sh` 是环境巡检入口，`link_ucf_crime.sh` 是唯一的 canonical 数据链接入口。

## 当前操作约束

- node2 负责网络获取；node3 不应反复尝试联网。node2 不可达时，先固定并校验本地资产，再按离线传输流程同步，同时记录偏离。
- 固定工作区、代码提交、registry revision、视频 SHA256 与本次输出目录；不要复用无时间和环境身份的旧 `matrix-v2.json`。
- 缓存目录、权重、external checkout、视频和模型输出都在 Git 忽略边界内。不要将它们加入提交。
- native smoke 只证明一个真实视频上的加载、前向和输出契约；它不证明 UCF-Crime 指标、缓存压缩收益、吞吐横比或 GPU 峰值显存结论。

## 2026-09-12 验证边界

当前CPU真实矩阵14路通过，GPU前向仍未完成。foundation环境中现存cuDNN子库需要主机缺少的GLIBC_2.27，不能通过修改搜索路径当作修复；本轮未更换系统/受保护环境。GPU操作前还要重新确认空卡。失败benchmark会在结果目录的benchmark-runtime下保留request/response/stdout/stderr和case/group上下文。实际结果见[当前状态](../progress/current-status.md)。
