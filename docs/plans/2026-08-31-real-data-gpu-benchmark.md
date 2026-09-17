# UCF-Crime 真实数据与 A100 性能阶段实施计划

## 目标与成功标准

在已交付的 VADBench 上接入完整 UCF-Crime，证明官方 `1610/290` 划分、视频容器和 temporal GT 均可用于正式 frame ROC-AUC/AP；随后在同一空闲 A100 上串行运行 VideoMAE V2、HERMES raw decoder-KV 与 HERMES native predict，报告同步后的吞吐、峰值/稳态显存和缓存遥测。最终用真实特征完成 `train → predict → evaluate`，并明确区分完整 benchmark、真实小样本和纯性能实验。

成功必须同时满足：

- `vadbench manifest audit-ucf` 返回 `passed=true`、`evaluation_readiness.ready=true`；
- 视频文件计数为 1,900，train/test 为 `1610/290`，normal/anomaly 为 `800/810` 和 `150/140`；
- 290 条 test manifest 均含 `num_frames`、`fps`；140 个异常视频各有 1–2 个合法 frame span，150 个 normal 为 0 span；
- A100 benchmark 使用 CUDA 同步、warmup/repeat、同一物理空卡并写出符合 `performance-result-v1` schema 的结果；
- HERMES raw 是 `native=off + external identity`，native predict 至少实际调用并应用一次压缩；
- checkpoint 推理产生 290 视频完整、无 gap/overlap/越界的 PredictionRecord，再计算 frame ROC-AUC/AP；
- 当前功能分支、本地、GitHub 与 node3 commit 一致，工作树干净。

## 当前权威状态（2026-08-31）

- 本地与 GitHub 分支：`feat/video-encoder-benchmark-framework`；本阶段代码已达到 `208 passed, 1 skipped`。
- 官方协议文件已固定在 `registry/datasets.yaml`：
  - `Anomaly_Train.txt`：SHA256 `ea91e03de7581bfa6b139e663ebb7511ab1651c33427f85741d3f497a2027744`；
  - `Temporal_Anomaly_Annotation.txt`：SHA256 `3b9542413f2ed9e94f73bf0488c151b3d7d595a8d2ad30f524e9243a2ae2a17c`。
- 不要求视频存在时，官方文件已确定性生成 `1610/290` manifest；类别与 video/frame annotation 数量正确。
- 当前本地审计为 `present=0, missing=1900`，因此 `passed=false`；290 条测试记录尚无容器 `num_frames/fps`。
- node3 本轮公网、Tailscale 和 node2 跳板均不可达；这是外部线路状态，不允许用猜测替代数据/GPU核查。
- 正确服务器数据入口固定为 `/users/fotile/VAD/data/raw/ucf_crime`，不得继续使用旧的 `data/ucf_crime`。

## 关键技术决策

1. **数据通过审计后再训练。** `import-ucf --require-files --probe-video-info` 负责生成 enriched manifest，`audit-ucf` 负责官方计数、文件、容器与 evaluation readiness；两者缺一不可。
2. **PredictionRecord 禁止降级。** 正式评测使用 feature store 中完整 segment frame ranges；每视频 clip index 唯一、区间连续、无重叠并覆盖 `[0,num_frames)`，不允许 uniform resample/fill-value 掩盖缺口。
3. **性能 case 串行驻留。** benchmark plan 逐 case 创建 adapter、运行、释放并清 CUDA cache，避免两份 HERMES 同时占用 A100。
4. **raw/native 明确分离。** HERMES raw 设置 `native_compression_mode=off`；native predict 单独设置 `predict`，两者的 external policy 均为 `identity`。
5. **精度语义分轨。** `projected_visual` 在 decoder/cache 前，只能做性能/状态遥测；缓存—精度曲线必须使用 `decoder_contextual`，因为其当前 chunk hidden state受前一 chunk压缩后 KV 条件影响。
6. **系统路线与缓存消融分开。** VideoMAE 与 HERMES 的架构、预处理和 token 语义不同，只能做系统级吞吐对照；缓存收益只在同一 HERMES 基座 raw/native 间计算。

## 任务 1：恢复 node3 并规范数据软链

**涉及文件：**

- `scripts/server/link_ucf_crime.sh`
- `/users/fotile/VAD/data/raw/ucf_crime`
- `/users/fotile/datasets/UCF-Crime`

**实施：**

1. 用 `server-login` 自动探测 node3 公网/Tailscale，每条线路最多一次；不绕过 host key。
2. 读取 node3 Git HEAD、工作树、磁盘和 GPU 进程归属。
3. 检查旧 `/users/fotile/VAD/data/ucf_crime`：仅当它解析到同一数据目标时保留兼容链接；创建 canonical `data/raw/ucf_crime`，不覆盖其他目标。
4. 用 Git bundle/已有离线部署流程把 node3 更新到 GitHub 分支 HEAD。

**验收：**

- `readlink -f /users/fotile/VAD/data/raw/ucf_crime` 指向实际数据目录；
- node3 `git status --short --branch` 无 tracked 改动；
- 记录 8 张 A100 的 PID、用户、显存和命令，只有空卡可进入任务 4。

## 任务 2：生成 enriched manifest 并完成正式数据审计

**运行：**

```bash
python -m vadbench manifest import-ucf \
  --dataset-root data/raw/ucf_crime \
  --train-split data/splits/ucf_crime/Anomaly_Train.txt \
  --temporal-annotations data/splits/ucf_crime/Temporal_Anomaly_Annotation.txt \
  --output-dir data/manifests/ucf_crime \
  --require-files \
  --probe-video-info

python -m vadbench manifest audit-ucf \
  --dataset-root data/raw/ucf_crime \
  --train-manifest data/manifests/ucf_crime/train.jsonl \
  --test-manifest data/manifests/ucf_crime/test.jsonl \
  --output outputs/dataset-audits/ucf-crime.json
```

**验收：**

- 审计 `passed=true`；文件 present/missing 为 `1900/0`；
- `evaluation_readiness.ready=true`；
- 默认 `deep_hash=false`，报告明确 near-duplicate `not_run`；只有用户明确承担全文件 I/O 时再运行深哈希。

## 任务 3：冻结真实特征并训练检测头

先用明确的真实小样本验证，再决定是否跑全量：

```bash
python -m vadbench extract \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --split train

python -m vadbench train \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --features outputs/ucf-videomaev2-weak/features
```

小样本必须包含 normal、anomaly train，以及有/无异常 test；报告 `protocol=smoke-subset`，不得引用为完整 UCF-Crime AUC。全量只有在 1,900 文件审计通过、磁盘/时长可接受时运行。

## 任务 4：运行 A100 性能矩阵

**运行：**

```bash
python -m vadbench benchmark \
  -c configs/benchmarks/video-encoder-smoke.yaml \
  --video data/smoke/mlvu-surveil-8.mp4 \
  --device cuda:0 \
  --warmup 1 \
  --repeat 5 \
  --output outputs/benchmarks/a100-video-encoders/performance.json
```

实际空卡编号替换 `cuda:0`。默认正式建议 warmup 3/repeat 10；HERMES 成本过高时最低 1/5，但必须保留配置。

**验收：**

- 每 repeat 有同步后的 decode/preprocess/encoder/wall time；
- 显存包含 allocated/reserved 的 baseline、peak、steady；
- 报告 frames/s、source-video-seconds/s，且记录 sample FPS/实际覆盖秒数；
- native predict 每 repeat 至少一次 `called=true, applied=true`；
- raw/native 采样指纹相同；VideoMAE/HERMES 如预处理或任务不同，`comparison.comparable=false` 并列出原因；
- 结果通过 `schemas/performance-result-v1.schema.json`。

## 任务 5：checkpoint 推理与 frame AUC/AP

```bash
python -m vadbench predict \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --features outputs/ucf-videomaev2-weak/features \
  --checkpoint outputs/ucf-videomaev2-weak/training/checkpoints/final.pt \
  --manifest data/manifests/ucf_crime/test.jsonl \
  --output outputs/ucf-videomaev2-weak/predictions/predictions.jsonl

python -m vadbench evaluate \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --predictions outputs/ucf-videomaev2-weak/predictions/predictions.jsonl \
  --manifest data/manifests/ucf_crime/test.jsonl \
  --output outputs/ucf-videomaev2-weak/evaluation/metrics.json
```

`predict` 默认 strict coverage；正式评测禁止 `--allow-incomplete-coverage`。

**验收：**

- 恰好 290 个预测 video ID；
- 固定 32 段路线每视频 `clip_index=0..31`，frame ranges 100% 覆盖；
- frame score 数量等于每视频 `num_frames`；
- 输出全局 micro frame ROC-AUC/AP，并附完整 config、checkpoint SHA、encoder fingerprint 和 split hash。

## 任务 6：证据、提交与同步

- 把 dataset audit、A100 performance、真实 subset/full metrics 的紧凑摘要写入 `docs/evidence/`；大 JSON/NPZ 留在 `outputs/`。
- 每个功能运行相关测试、ruff、compileall 后使用中文 Conventional Commit 并 push 当前分支。
- node3 通过 bundle/离线方式同步到同一 HEAD；最终审计本地、GitHub、node3 三处 commit 和权重 SHA。

## 风险与停止条件

| 风险 | 处理 | 是否阻塞最终目标 |
|---|---|---:|
| node3 两条线路不可达 | 每个 Goal turn 各探测一次并继续本地可执行工作；第三次连续且无其他进展才按 blocked 规则处理 | 阻塞 GPU/服务器数据证据 |
| 数据目录为空或多一层归档目录 | audit 明确 missing；修正软链到直接包含 15 个标准目录的根 | 阻塞真实 manifest/AUC |
| A100 全部被占用 | 不抢卡、不杀任务；等待空卡或用户调度 | 阻塞 GPU 性能，不阻塞数据审计 |
| HERMES native 未触发 | 降低 smoke KV budget 或延长 stream；结果要求 fail closed | 阻塞 native 性能结论 |
| projected_visual 精度无差异 | 标为 perf-only；改用 decoder_contextual 后再训练同一 head | 阻塞缓存—精度曲线 |
| 全量抽取成本过高 | 先真实 stratified subset，报告范围；不能用 subset 冒充 full benchmark | 可能阻塞完整 AUC，但不阻塞真实小样本 |

