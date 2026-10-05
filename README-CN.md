# WAVAD / ICASSP 2027

**当前服务器代码入口是 `/users/fotile/VAD`，Python 包与命令仍叫 `vadbench`。** WAVAD 是项目名称；仓库同时保存通用视频异常检测框架，以及 ICASSP 2027 关于正常—异常性质、token 缩减和检测质量的研究。大型资产通过 `/users/fotile/VAD/assets` 浏览，实际根位于 node2 的 `/data2/localdisk/fotile-wavad/`，node3 经 `/data2` 共享挂载读取。具体路径与验收范围见[资产与路径说明](docs/operations/wavad-assets.md)。

## 当前状态（资产整理 2026-09-26，代码重构 2026-09-30）

- 通用框架保留 encoder catalog、manifest、FeatureStore、检测头和预测评测链。论文压缩代码在 `src/vadbench/token_reduction/`，项目配置、实验协议与冻结回执在 `projects/icassp2027/`。公共特征身份已移至 `data/feature_contracts.py`，预测许可位于 `engine/compatibility.py`；提取、恢复、合并、通用检测、质量和效率实现已移至 `workflows/`，UR-DMU 后端位于 `integrations/detectors/urdmu/`。旧 `paper` 路径保留同一实现的兼容导入，项目冻结和正式阶段规则继续留在 `paper/`。当前边界见[架构说明](docs/architecture/current-system.md)。
- [DSANet 提取、评分、质量导出入口](src/vadbench/workflows/dsanet/README.md)已接入包内，增量测试 17 项通过；旧 Python 3.9 环境经 `scripts/icassp2027/dsanet_legacy.py` 运行提取和评分，新入口尚未重跑正式模型实验。
- 经验证的本地源码已整合到服务器 `/users/fotile/VAD`：456 个文件新增或更新、216 个语义一致文件保留原字节、0 个冲突；未改 `.git` 历史，未 commit 或 push。远端原始代码和 dirty patch 备份位于 `assets/provenance/pre-integration-remote-*`。整合后的远端回归为 **191 passed、2 skipped**，两项跳过与当前 Transformers 未导出 `VideoMAEImageProcessorPil` 有关；`compileall`、encoder catalog 和 paper status 检查已通过。这些结果不等于所有旧运行已重新验证。
- UCF-Crime 1,900 个视频（103,795,576,492 B）与 XD-Violence 4,750 个视频及 1 个辅助文件（85,869,780,149 B）已复制到新根，`rsync --checksum` 复核无差异。默认 `data/raw/ucf_crime`、`data/raw/xd_violence` 已指向新副本；原来源副本保存在 `/users/fotile/VAD/archive/datasets`，旧路径为兼容软链。
- 六个旧资产根已完成物理收拢：四个 `/data2/localdisk/fotile-icassp2027-*` 根移入 `assets/storage/`，两个共享 `/users/fotile` 根归档到 `/users/fotile/VAD/archive/`；旧路径改为兼容软链。原有 314 个导航链接及两节点各 96 个身份锚点的迁移后检查均通过。正式 24 组特征 index 的完整 SHA 已通过，blob 只做每组首项抽样；具体范围见[资产说明](docs/operations/wavad-assets.md)。

## 先看哪里

| 任务 | 入口 |
|---|---|
| 论文当前代码、实验与边界 | [ICASSP 项目入口](projects/icassp2027/README.md)、[profile](projects/icassp2027/profile.yaml)、[protocol](projects/icassp2027/protocol.yaml) |
| 数据、特征和实验具体位置 | [WAVAD 资产说明](docs/operations/wavad-assets.md) |
| CLIP–DSANet提取、评分、质量导出 | [三阶段命令与既有环境兼容](src/vadbench/workflows/dsanet/README.md) |
| 历史执行回执与研究决策 | [progress](projects/icassp2027/progress.md)、[V3 研究计划](projects/icassp2027/RESEARCH_PLAN_V3.md)、[论文交接](projects/icassp2027/handoff_v2/README_先读我.md) |
| 框架用法与模型运行 | [操作流程](docs/operations/workflows.md)、[架构](docs/architecture/current-system.md)、[服务器手册](docs/operations/server.md) |
| 论文文件 | [论文工作区](paper/icassp2027/README.md) |

[资产整理计划](projects/icassp2027/organization/README.md)保留设计依据，实际位置和完成项见[执行状态](projects/icassp2027/organization/EXECUTION_STATUS.md)与[资产说明](docs/operations/wavad-assets.md)。[2026-09-11 架构审查](docs/reviews/2026-09-11-architecture-review.md)保留为历史记录。

## 服务器只读检查

沿用服务器已有的解释器与源码，不为查看配置安装新环境：

```bash
cd /users/fotile/VAD
export PYTHONPATH=/users/fotile/VAD/src
PY=/users/fotile/VAD/.encoder-envs/v2/foundation-video-v2/bin/python
"$PY" -m vadbench encoders list
"$PY" -m vadbench.paper status --project projects/icassp2027/profile.yaml
```

`encoders list` 显示完整模型 catalog；论文 active 范围由 profile 指定。`status` 检查配置与路径，不证明权重已加载、GPU 可用或正式实验通过。执行抽取、训练和评测时还需按照具体 run 的配置、环境和数据协议核对身份。

## 框架与历史资料

VADBench 用统一的视频身份、时间轴、特征和评测契约组织固定 clip 编码器与长视频模型。VideoMAE V2 是独立 clip 前向，不存在跨 clip decoder KV cache；HERMES 的缓存语义属于语言模型 decoder。`lab_anomaly/` 保存旧版训练、评分和 RTSP 原型，见[旧版说明](lab_anomaly/README-CN.md)。

2026-09-11 的框架统计、CPU smoke、安装说明和当时缺失数据的判断，保留在[原实施记录](docs/progress/2026-09-11-implementation.md)及[框架状态档案](docs/progress/current-status.md)。它们不能替代本轮已复制数据、已整合源码和正式实验回执。新开发环境的通用安装步骤见[服务器手册](docs/operations/server.md)，既有 node2/node3 环境无需重装。

原始视频、权重、特征 blob、第三方 checkout、环境和大型运行产物不进入 Git。本仓库代码使用 MIT License；第三方代码、权重和数据遵守各自的许可与来源约束。
