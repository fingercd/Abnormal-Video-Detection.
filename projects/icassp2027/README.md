# ICASSP 2027 项目入口

本项目研究正常与异常视频在编码器内部的 token、时空关系、attention 和读出性质，并验证由此设计的 token 缩减在检测质量、延迟和显存上的实际影响。服务器统一代码入口是 `/users/fotile/VAD`，Python 包与 CLI 仍叫 `vadbench`；大型视频、特征和实验导航从 `/users/fotile/VAD/assets` 进入。当前物理落点、兼容路径与核验范围见[WAVAD 资产说明](../../docs/operations/wavad-assets.md)。

## 当前代码与实验入口（2026-09-26）

- 通用数据、encoder、训练和评测能力保留在 `src/vadbench/`。token 缩减和 encoder bridge 在 `src/vadbench/token_reduction/`；`clean_encoder_batch` 位于 `src/vadbench/data/batches.py`，旧导入兼容。[DSANet 三阶段入口](../../src/vadbench/workflows/dsanet/README.md)现已提供 extract、score、quality；原 Python 3.9 环境通过 `scripts/icassp2027/dsanet_legacy.py` 运行 extract/score，Python 3.10+ 可使用模块命令。增量测试 17 项通过，但新入口尚未重跑正式模型实验。旧 run code 保持历史身份。本轮进一步把提取、恢复、合并、通用检测、质量和效率实现迁至 `workflows/`，公共身份与许可分别位于 `data`、`engine`，UR-DMU 后端位于 `integrations/detectors/urdmu/`；旧导入兼容。论文冻结检查和正式阶段编排留在 `paper/`。见[当前架构](../../docs/architecture/current-system.md)。
- 本目录保存论文 [profile](profile.yaml)、[protocol](protocol.yaml)、[冻结决策](decisions/)、[研究回执](progress.md)、[论文交接](handoff_v2/README_先读我.md) 与 [manuscript](manuscript-20260922/README.md)。旧 [V3 研究计划](RESEARCH_PLAN_V3.md)、[特征提取计划](FEATURE_EXTRACTION_PLAN_20260920.md) 和 [V2 计划](RESEARCH_PLAN_V2.md) 保留各自日期与决策身份，不再用“唯一执行计划”概括当前状态。
- 2026-09-26 已将经验证的本地源码整合到远端 `/users/fotile/VAD`：456 个文件新增或更新、216 个语义一致文件保留原字节、0 冲突。整合后回归 **191 passed、2 skipped**，`compileall`、encoder catalog、paper status 已通过；跳过项来自当前 Transformers 未导出 `VideoMAEImageProcessorPil`。源码与 dirty patch 备份位于 `assets/provenance/pre-integration-remote-*`。没有改 Git 历史、commit 或 push。

## 从哪里开始

| 要做的事 | 入口 |
|---|---|
| 看研究进度、冻结与历史运行 | [progress](progress.md)、[决策](decisions/)、[交接材料](handoff_v2/README_先读我.md) |
| 找 UCF/XD 视频、特征、实验和代码快照 | [资产说明](../../docs/operations/wavad-assets.md)、[正式特征绑定](organization/formal_feature_bindings.jsonl) |
| 核对数据及评测规则 | [protocol](protocol.yaml)、[实验规则](../../docs/icassp2027/06_EXPERIMENT_RULES_AND_PITFALLS.md) |
| 看旧框架或服务器操作 | [框架操作流程](../../docs/operations/workflows.md)、[服务器手册](../../docs/operations/server.md) |
| 阅读论文文件 | [manuscript README](manuscript-20260922/README.md)、[paper workspace](../../paper/icassp2027/README.md) |

[整理计划](organization/README.md)保留设计依据与原始盘点；当前完成项见 [执行状态](organization/EXECUTION_STATUS.md)。2026-09-26 已完成六个旧资产根的物理收拢，旧路径改为兼容软链；原视频新副本与归档源副本均保留。`progress.md` 按日期保留执行历史，以最新条目及对应回执为准。

## 服务器只读入口

以下命令已在整合后的远端源码上执行过。沿用现有解释器，不重装环境：

```bash
cd /users/fotile/VAD
export PYTHONPATH=/users/fotile/VAD/src
PY=/users/fotile/VAD/.encoder-envs/v2/foundation-video-v2/bin/python
"$PY" -m vadbench encoders list
"$PY" -m vadbench.paper status --project projects/icassp2027/profile.yaml
```

`encoders list` 展示完整 catalog，论文 active 模型以 profile 为准。`status` 仅报告配置和路径可用性，不证明权重加载、GPU 空闲、run 完成或测试质量。node3 公网入口转内部 `ibnode2` 的 SSH 已验证可用，具体命令见[服务器手册](../../docs/operations/server.md)；连接成功本身不代表调度器或 GPU 任务状态。

## 数据、特征与证据边界

UCF-Crime 的默认入口 `data/raw/ucf_crime` 已指向 `assets/datasets/ucf_crime/official_raw_20260918/raw`，1,900 个视频与来源一致；旧 897 子集别名 `ucf_crime_legacy_897` 保留。XD-Violence 的默认入口 `data/raw/xd_violence` 已指向 `assets/datasets/xd_violence/official_raw_accepted3950_20260920/raw`，其中 4,750 个视频及一个辅助文件与来源一致。XD 训练身份仍区分 declared 3,954、accepted 3,950 和 excluded 4，不能把 64+64 筛选池当正式 full-train。原始来源副本在 `archive/datasets` 保留；不能因新默认入口就删掉旧副本。

[正式特征绑定](organization/formal_feature_bindings.jsonl)覆盖 6 个 dense test 和 18 个压缩正式 test 视图。24 个合同 SHA、完整 index SHA 和每个 index 首项 blob 的 SHA/zip CRC 样本已通过；回执位于服务器 `assets/catalog/formal_feature_qa-20260926-r02.jsonl`。未核对全部 blob，也未重跑 encoder 或检测头。正式测试分数不能用于重新选择方法、预算、阈值或 checkpoint；UCF 与 XD 仍按各自时间轴和评测协议解释。

旧基础重构的只读观察资料仍见[观察计划](../../docs/icassp2027/01_NORMAL_ANOMALY_PROBES.md)、[架构说明](../../docs/icassp2027/02_ENCODER_SCOPE_AND_REFACTOR.md)与[执行流程](../../docs/icassp2027/03_EXECUTION_RUNBOOK.md)。观察阶段的 cohort 用训练 manifest 与 JSONL sidecar 固定窗口和角色；训练弱标签不得注入部署 reducer，测试真值只在正式评价阶段使用。原 manifest、视频采样、权重验证、FeatureStore、MIL、predict 和 official evaluator 的具体调用仍以当前源码及相应运行回执为准。
