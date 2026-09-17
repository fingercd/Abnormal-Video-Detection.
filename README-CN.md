# VADBench：视频表征与 UCF-Crime 基准框架

VADBench 用统一的视频身份、时间轴、特征和评测契约，比较固定 clip 表征与长视频状态/缓存路径。项目主代码在 `src/vadbench/`，当前重点是把已经接入的模型变成可复现的实验链路。

- **VideoMAE V2 Base**：固定 clip 编码器，每个 clip 独立前向，没有跨 clip KV cache。
- **HERMES + LLaVA-OneVision-Qwen2-0.5B**：流式 VLM，复用语言模型 decoder KV。`projected_visual` 在 decoder 前；研究缓存对表征/精度的影响要使用 `decoder_contextual`。

[English](README.md) · [文档总入口](docs/README.md) · [操作流程](docs/operations/workflows.md) · [服务器操作](docs/operations/server.md) · [架构与问题](docs/reviews/2026-09-11-architecture-review.md)

## 已实现与已验证的范围

本页按基于`0badc34`的本轮改造整理，最终验证日期为2026-09-12。服务器与本地都位于 `qzt/refactor-vadbench-simplification`；可变状态及证据统一维护在[当前状态](docs/progress/current-status.md)。

| 能力 | 当前边界 |
|---|---|
| 注册与统一接口 | 25 个研究候选，21 个运行 catalog 条目；输入 `BTHWC uint8`，输出 `features[B,S,D]` 与 `TokenTimeline` |
| 固定/流式抽取 | 统一生效配置与权重身份；固定/流式共用采样，参考配置均使用32段弱监督 |
| 数据与检测头 | 官方 split 导入、容器补全、数据审计、冻结特征、弱监督 MIL 与显式时序监督 |
| 推理与评测 | `train → predict → evaluate`；预测/评测共用覆盖检查，official模式要求匹配的v2数据审计 |
| 冒烟与性能 | 单模型 smoke v2、矩阵、warmup/repeat 性能 runner；冒烟耗时不直接用于模型排名 |
| 真权重CPU验证 | 本轮14路重新通过；两个参考encoder各完成真实视频2段抽取。许可/人工资产/仅候选状态保持 |
| 完整 UCF-Crime 实验 | 尚未完成。服务器数据目标为空，默认 manifest 缺失，不能报告正式 AUC/AP |

2026-09-11 已按审查修复抽取、结果误用和正式协议门禁，统一配置/权重身份与阶段记录，删除重复 worker 路径。实施范围与验证见[本轮改造](docs/progress/2026-09-11-implementation.md)；[审查报告](docs/reviews/2026-09-11-architecture-review.md)保留基线问题。

## 安装与轻量检查

Python 3.10–3.12，在仓库根目录操作。下面安装框架与测试依赖；真实编码器还需要其固定上游、权重和独立运行环境。

Windows：

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev,train,video]"
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m vadbench doctor
```

Linux 新开发环境：

```bash
uv sync --extra dev --extra train --extra video
uv run python -m pytest
uv run python -m vadbench doctor
```

HERMES依赖以四组环境登记和固定上游revision为准，已删除不匹配该运行栈的通用`hermes`安装extra。已部署的 node3 使用既有离线环境，按[服务器操作](docs/operations/server.md)检查。`doctor` 只检查依赖是否可发现与目录权限，不证明模型可加载、权重通过校验或 GPU 空闲。

## 从哪里开始

1. 读取[当前状态](docs/progress/current-status.md)，确认数据和目标模型是否具备运行条件。
2. 用[操作流程](docs/operations/workflows.md)核对配置、导入与审计数据，再抽取 train/test 特征、训练检测头、生成预测和评测。
3. 模型接入与缓存实验先读[编码器运行机制](docs/architecture/encoder-runtime.md)及[UCF-Crime 协议](docs/research/ucf-crime-protocol.md)。两套主实验均使用32段、相同弱监督head/loss设置；HERMES在同一组采样片段之间传递decoder状态。原生缓存预算消融另行配置。
4. 需要继续修改代码时，从[架构审查的优先级与验收条件](docs/reviews/2026-09-11-architecture-review.md)选取独立功能。

`lab_anomaly/` 保留旧版训练、评分与 RTSP 相关代码；VideoMAE V2 包装器已迁入主包，旧模块仅重导出同一实现；部分旧应用入口仍只属于原型。范围见[旧版说明](lab_anomaly/README-CN.md)。

## 存储与来源

`configs/` 保存实验与模型参数，`registry/` 保存候选、运行 catalog、环境和 checkpoint 元数据，`integrations/` 保存上游锁，`schemas/` 保存已有的版本化数据契约。具体职责见[架构说明](docs/architecture/current-system.md)。

原始视频、模型权重、特征 blob、第三方 checkout、环境和大运行产物不进入 Git。输出位置由配置或各命令的输出参数决定；示例实验默认是 `outputs/<run_name>/`，详细路径语义见[操作流程](docs/operations/workflows.md)。

本仓库代码使用 MIT License。第三方代码、权重与数据遵循各自许可证；例如锁定的 VideoMAE V2 Base 权重为 CC-BY-NC-4.0。下载前核对登记来源、revision、许可证与 SHA256，node3 使用已有离线资产。
