# 文档入口与维护边界

最初的文档核对以 2026-09-11 读取的 `ibnode3:/users/fotile/VAD` 为事实基线，Git commit 为 `0badc34`。当时本地相同提交且受跟踪源码无修改，因此按模块分工通读相同版本的源码、测试、schema 和配置；服务器实况与历史结果直接从服务器读取。

## 按任务阅读

| 要解决的问题 | 入口 |
|---|---|
| 项目目标、安装 | [README-CN](../README-CN.md) |
| 当前有什么资产、验证到哪里 | [当前状态](progress/current-status.md) |
| 导入、抽特征、训练、推理与评测 | [操作流程](operations/workflows.md) |
| 离线环境、资产、overlay、服务器运行 | [服务器操作](operations/server.md) |
| 模块职责、配置与产物关系 | [当前架构](architecture/current-system.md) |
| 数据、训练与帧评测实现 | [数据与评测](architecture/data-and-evaluation.md) |
| adapter、worker 与缓存语义 | [编码器运行机制](architecture/encoder-runtime.md) |
| 接下来改什么、如何验收 | [架构审查与建议](reviews/2026-09-11-architecture-review.md) |
| UCF-Crime 实验的硬约束 | [数据协议](research/ucf-crime-protocol.md) |
| 为什么选这些模型 | [截至 2026-08-31 的研究调研](research/video-encoder-survey-2026-08-31.md) |
| 模型变体、来源、登记状态 | [来源审计](research/native-encoder-source-audit-2026-08-31.md)与[接入矩阵](progress/encoder-integration-matrix.md) |
| 四组环境及历史迁移证据 | [环境说明](progress/encoder-environment-v2.md) |
| 旧训练/RTSP 模块 | [lab_anomaly](../lab_anomaly/README-CN.md) |

## 当前事实、规范和历史记录

代码行为以执行入口、对应测试及 schema 为据；用户的目标和实验约束仍是验收要求。尚未实现的要求列为缺口，不能把要求改写成“已经实现”。

- `operations/` 只维护当前操作方法；参数完整定义交给对应 `--help`，模型与环境细项引用 YAML，不在每份文档重复。
- `architecture/` 解释当前行为与边界；`reviews/` 保留基线建议，实施状态与验证统一到[本轮改造](progress/2026-09-11-implementation.md)。
- `progress/current-status.md` 是可变状态入口；历史计划不再用“当前 Goal”决定后续任务权限。
- `research/` 中的学术/外部来源保留原截止日。本轮核对仓内实现，未重查所有论文、模型卡或标注发布状态。
- `docs/evidence/` 原有 JSON 是不可回填的历史证据。本轮新增[服务器文档核对证据](evidence/server-doc-audit-2026-09-11.json)，记录观察范围与限制；大运行结果仍在服务器 `outputs/`。

## 历史决策

| 文档 | 用途 |
|---|---|
| [2026-08-31 框架计划](plans/2026-08-31-video-encoder-benchmark-framework.md) | 初始接口与实验边界、已落地及待完成项 |
| [2026-08-31 25 路接入计划](plans/2026-08-31-25-video-model-integration.md) | 从全候选登记改为候选/运行 catalog 分离的决策 |
| [2026-09-03 四组环境计划](plans/2026-09-03-four-group-encoder-environments.md) | 隔离环境、overlay、资产复用与保护边界 |
| [2026-09-03 精简计划与结果](plans/2026-09-03-vadbench-code-simplification.md) | 历史重构范围、提交、净删减与验证证据 |
| [旧进度入口](progress/2026-08-31-current-progress.md) | 历史阶段说明及当前入口链接 |

本地已有未跟踪的真实数据/GPU计划草案保持原样，其可用内容已纳入当前操作流程；其中旧服务器状态与提交/推送安排不作为本轮事实或授权。

## 更新范围

本轮核对仓库维护的 Markdown 与说明性 `readme.txt`，涵盖根目录、`docs/` 和 `lab_anomaly/`。`LICENSE`、依赖清单、训练日志、历史 JSON 证据、二进制文件和第三方 checkout 不改写。首次文档核对保留上述边界；用户随后授权的代码改造见[实施记录](progress/2026-09-11-implementation.md)，其中包含实际代码与schema变化。
