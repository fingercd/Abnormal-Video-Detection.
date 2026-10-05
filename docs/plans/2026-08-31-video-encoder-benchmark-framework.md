# 视频编码器基准框架：初始计划与当前落地映射

初始决策日期：2026-08-31。2026-09-11 按 `0badc34` 整理为历史决策记录；当前命令见[操作流程](../operations/workflows.md)，状态见[当前进度](../progress/current-status.md)。

## 目标与保留的设计决定

以 UCF-Crime 为首个数据基准，通过注册机制接入独立 clip 编码器与真实跨 chunk 状态路径，解耦采样、编码、压缩、训练、评测与产物。

1. 公共输入 `BTHWC uint8`，公共特征 `features[B,S,D]`；时间轴保留原视频坐标。
2. 固定与流式接口分开；VideoMAE V2 是无状态 clip 对照，HERMES 状态属于 LLM decoder KV。
3. `vision_tokens`、`visual_memory`、`decoder_kv` 分别声明；原生策略与通用 CachePolicy 分别记录。
4. 弱监督仅消费训练视频级标签；官方测试时间标注仅用于评测，UCA caption 不自动二值化。
5. 固定 revision、权重校验、数据身份和运行证据；大资产不进入 Git。

## 已落地的职责

| 初始任务 | 当前实现位置 | 验收边界 |
|---|---|---|
| 配置、注册与轻量诊断 | `config.py`、`registry.py`、`doctor.py` | 结构/能力与资产运行检查仍分阶段执行 |
| 编码器、时间轴、stream/cache 契约 | `contracts.py`、`compression.py` | 模型族实际输出语义由 adapter 说明 |
| UCF 导入与采样 | `data/`、相关 schemas/tests | 坐标与合成测试已覆盖；真实完整数据本轮缺失 |
| VideoMAE V2/HERMES 接入 | `integrations/` | 历史真实权重证据存在，后续扩展见25路决策 |
| 特征与运行产物 | `features.py`、`artifacts.py` | 当前 run/metrics/cache JSON 没有各自独立 schema，追溯仍需完善 |
| MIL/显式时序监督、预测和评测 | `models/`、`tasks.py`、`engine/` | 冻结 head 链路存在；不等于全 encoder 联合微调已接入 CLI |
| CLI 与性能计划 | `cli.py`、`benchmark*.py` | 已包含后续增加的 enrich/audit/predict/benchmark |
| 离线部署与验证 | `scripts/server/`、`docs/evidence/` | 当前日常运行以 v2 环境为准 |

实现细节集中到[当前架构](../architecture/current-system.md)与[数据/评测](../architecture/data-and-evaluation.md)，本计划不再复制函数清单和完整命令参数。

## 历史验收与尚未完成部分

2026-08-31 的部署记录包含 167 项服务器测试、两套真实权重 CPU 冒烟和合成特征训练/评测闭环，原始 JSON 保持原样。2026-09-04 之后的代码与验证基线见[精简结果](2026-09-03-vadbench-code-simplification.md)。

初始目标中的完整 UCF-Crime 训练/测试、真实特征精度、GPU 性能与缓存精度消融尚不能据这些证据宣称完成。当前服务器数据空缺、HERMES 示例监督类型不同，以及配置/指纹/覆盖门禁缺口均列入[审查建议](../reviews/2026-09-11-architecture-review.md)。

初始计划中的分支、部署和提交安排属于当时执行背景；不要求重建旧分支，不授予当前任务额外的提交、推送或实验运行权限。
