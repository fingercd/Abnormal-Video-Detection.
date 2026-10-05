# 当前命令与实验流程

服务器既有 `.venv` 内仍有早期已安装包。使用源码工作区时先执行 `export PYTHONPATH=/users/fotile/VAD/src`，再运行 `python -m vadbench`；可用 `python -c "import vadbench; print(vadbench.__file__)"` 确认路径。v2服务器脚本和隔离benchmark已通过公共helper显式选择当前源码，无需修改受保护环境。

核对日期：2026-09-11；源码：`0badc34`。以下命令在仓库根目录执行；`python` 指已安装 VADBench 的选定解释器。模型运行还要匹配[服务器环境与 overlay](server.md)。当前服务器 UCF-Crime 数据和默认 manifest 未到位，真实数据流程是数据准备完成后的操作步骤。

## 命令发现与预检

```bash
python -m vadbench --help
python -m vadbench doctor --project-root .
python -m vadbench encoders list
python -m vadbench integrations list
python -m vadbench config validate configs/experiments/ucf_videomaev2_weak.yaml
python -m vadbench config validate configs/experiments/ucf_hermes_stream.yaml
```

`encoders list` 与 `integrations list` 当前列出 21 个运行目标；25 个研究候选在 `registry/encoder-candidates.yaml`。CLI 部分帮助文案仍写“25项”，以实际 JSON 与 catalog 为准。配置校验通过只说明结构合法；环境、真实资产与能力协商是后续步骤。

| 命令族 | 实际用途 |
|---|---|
| `doctor` / `config validate` | 依赖发现、路径权限、实验结构 |
| `encoders list\|inspect` | 惰性注册项、能力与构造入口 |
| `integrations list\|preflight\|smoke\|matrix` | 接入目录、资产预检和单项/矩阵冒烟 |
| `manifest import-ucf\|validate\|enrich\|audit-ucf` | 官方导入、单清单校验、容器补全与全数据审计 |
| `weights list\|verify\|fetch` | checkpoint 登记、文件校验与显式许可证下载 |
| `extract` / `train` / `predict` / `evaluate` | 冻结特征和检测头链路 |
| `smoke` / `benchmark` | 真实前向验证与独立性能实验 |

详细参数直接读取 `python -m vadbench <command> --help`。跨环境批量原生冒烟优先使用服务器 v2 launcher；通用 `integrations` CLI 不会自动完成整套 v2 环境选择。

## 数据准备

正式流程先准备已登记的官方 split/TXT，存放路径与 `registry/datasets.yaml` 对齐。源视频接入 `data/raw/ucf_crime`，不要将视频复制进 Git 工作树。

```bash
python -m vadbench manifest import-ucf \
  --dataset-root data/raw/ucf_crime \
  --train-split data/splits/ucf_crime/Anomaly_Train.txt \
  --temporal-annotations data/splits/ucf_crime/Temporal_Anomaly_Annotation.txt \
  --output-dir data/manifests/ucf_crime \
  --require-files --probe-video-info

python -m vadbench manifest audit-ucf \
  --dataset-root data/raw/ucf_crime \
  --train-manifest data/manifests/ucf_crime/train.jsonl \
  --test-manifest data/manifests/ucf_crime/test.jsonl \
  --output outputs/dataset-audits/ucf-crime.json
```

`import-ucf` 可从 temporal TXT 推导 test split；`manifest validate` 检查单份清单，不等于完整官方数据审计。已有 manifest 可用 `manifest enrich <path> --dataset-root <root> --output <new-path>` 补容器信息。

正式验收要求 train/test 为 1,610/290、数据审计通过、帧数/FPS 完整、测试时间端点正确且数据泄漏检查完成；默认审计不会做全部内容 hash 或视觉近重复识别。坐标转换、训练标签可见性和验收缺口见[协议](../research/ucf-crime-protocol.md)。

## 冻结特征、训练、预测和评测

以下为已修复的 VideoMAE V2 弱监督流程；数据、所选环境与权重就绪后执行。抽取先校验实际绑定资产，失败会留下独立stage记录。

先选定本次运行目录，使用同一 encoder/sampler 配置分别抽取 train/test；目录示例中的 `example-vmae` 应换成新的实验名。

```bash
python -m vadbench extract \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --split train --output outputs/example-vmae/train

python -m vadbench extract \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --split test --output outputs/example-vmae/test

python -m vadbench train \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --features outputs/example-vmae/train/ucf-videomaev2-weak/features \
  --output outputs/example-vmae/head

python -m vadbench predict \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --features outputs/example-vmae/test/ucf-videomaev2-weak/features \
  --checkpoint outputs/example-vmae/head/checkpoints/final.pt \
  --manifest data/manifests/ucf_crime/test.jsonl \
  --output outputs/example-vmae/predictions/predictions.jsonl

python -m vadbench evaluate \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --predictions outputs/example-vmae/predictions/predictions.jsonl \
  --manifest data/manifests/ucf_crime/test.jsonl \
  --dataset-audit outputs/dataset-audits/ucf-crime.json \
  --output outputs/example-vmae/evaluation/metrics.json
```

这里 `extract --output` 是运行根，程序仍会在其后添加配置中的 `run_name`。`train --output` 则直接是训练目录；`predict/evaluate --output` 是文件路径。不要把同一个输出参数按同一种路径层级理解。

`--limit-videos` 和 `--max-steps` 可用于工程冒烟。小样本需要同步使用对应 manifest 并含训练所需类别，产物标为 subset/smoke；不能将其指标称为完整 UCF-Crime 结果。`--max-steps` 仅限制训练步数，不会创建或自动平衡数据子集。

`predict` 默认 strict coverage；正式测试不用 `--allow-incomplete-coverage`。`evaluate` 默认official，要求完整覆盖与匹配的v2数据审计；`--protocol subset`允许已命名的测试子集但仍拒绝gap/overlap。`generic`仅保留显式诊断用途。结果中的run、encoder和checkpoint身份不能混用。

### HERMES 配置的用途

`ucf_hermes_stream.yaml` 现与VideoMAE一样使用32段、每段16帧、stride=2、Attention MIL与显式BCE训练设置。HERMES逐段传递decoder状态，默认raw/off并导出`decoder_contextual`。这是在同一组稀疏采样片段上的状态对照，不应描述成逐帧连续流式处理。

连续视频流仍可把sampler显式设为`chronological_stream`；它的chunk数量和时间覆盖是另一种采样协议，需要单独命名。强监督任务必须另有合法的训练时序标签；未标注区间是否normal要在配置中明确，不能从UCA caption自动推断。

## 真实权重冒烟

单模型 CLI 只在所选 Python、依赖、checkout 和权重都已准备好后运行；node3 四组环境的标准入口与参数在[服务器手册](server.md)。

```bash
python -m vadbench smoke \
  -c configs/experiments/ucf_videomaev2_weak.yaml \
  --video data/smoke/mlvu-surveil-8.mp4 --device cpu \
  --output outputs/example-smoke/videomaev2.json

python -m vadbench smoke \
  -c configs/experiments/ucf_hermes_stream.yaml \
  --video data/smoke/mlvu-surveil-8.mp4 --device cpu --chunks 2 \
  --native-compression-mode off \
  --output outputs/example-smoke/hermes-raw.json
```

HERMES raw 对照需要原生模式 `off` 且外部 `identity`；只写 `identity` 不会关闭默认原生 `predict`。原生 `--kv-size` 统计选中的视觉 token，protected prompt token 额外计入总 KV。smoke v2 的 `smoke_pass/blocked/failed` 分别返回 0/2/1；单模型CLI和matrix默认拒绝已有输出，请每次使用新路径。

## 性能实验

当前node3的cuDNN子库缺少主机GLIBC_2.27支持，GPU前向未通过。下面的GPU命令用于兼容用户态环境和空卡就绪后的验证；不要仅凭CPU冒烟成功执行全量GPU实验。

`configs/benchmarks/video-encoder-smoke.yaml` 包含 VideoMAE 固定 clip、HERMES raw、HERMES native predict 三个 case。执行前选择空闲 GPU、检查模型对应运行环境，并为结果使用新文件。

```bash
python -m vadbench benchmark \
  -c configs/benchmarks/video-encoder-smoke.yaml \
  --video data/smoke/mlvu-surveil-8.mp4 \
  --device cuda:0 --warmup 1 --repeat 5 \
  --output outputs/example-performance/performance.json
```

`cuda:0` 是待替换的设备示例，benchmark CLI 本身不查询 GPU 归属。默认benchmark为每个case选择registry指定的Python/overlay，并在子进程内加载、解码、计时和测显存；无需将不兼容依赖装到同一Python。运行结果仍须实际验证，不能由smoke推导性能结论。

原生 predict case 要求每 repeat 至少一次压缩调用和实际应用。读取结果时检查 `comparison`、`accuracy_eligibility`、实际采样、计时与 cache 子项；现有默认 projected-visual 计划是性能实验。跨模型 token/s 只作各自诊断。细节与当前统计限制见[架构](../architecture/current-system.md)。

## 权重与来源

`weights fetch` 的参数顺序为 `weights [--registry PATH] fetch <checkpoint_id> <path> --accept-license <license>`；校验使用 `weights verify <checkpoint_id> <path>`。它不能替代全部来源类型的 v2 资产流程，具体来源以 registry 和[服务器手册](server.md)为准。下载仅在有外网且授权的节点进行。

## 验证

```bash
python -m pytest
python -m ruff check src tests scripts/fetch_upstreams.py
python -m ruff format --check src tests scripts/fetch_upstreams.py
python -m compileall src tests
git diff --check
```

本轮已实施代码修复和职责收敛，验证范围及当前结果见[当前状态](../progress/current-status.md)。
