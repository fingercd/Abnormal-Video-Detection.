# 25 路视频模型接入：决策与落地结果

初始计划日期：2026-08-31。2026-09-11 根据 `0badc34` 校正。本文件保留接入目标与架构决策，逐路线明细统一到[接入矩阵](../progress/encoder-integration-matrix.md)与[来源审计](../research/native-encoder-source-audit-2026-08-31.md)。

## 目标与范围

让不同视频模型/VLM 消费共同的真实视频输入，经 adapter 导出明确阶段的表征与时间轴，并用真实 checkpoint 验证。接入门槛不要求模型是纯视觉 encoder，但必须说明其实际执行的上游模块和输出阶段。

当时阶段只验加载、前向、状态与产物，不要求异常标签、检测头训练、AUC/AP 或缓存收益。这是该阶段的验收范围，并非限制项目后续训练、性能与缓存研究。

## 从 25 项 catalog 到候选/运行分离

原始计划要求全部 25 项进入运行 catalog。后续来源审计发现有路线缺少可校验的目标 checkpoint，因此当前实现改为：

- `registry/encoder-candidates.yaml` 保留 25 个研究候选与来源/许可/资产状态；
- `registry/encoder-integrations.yaml` 登记 21 个运行目标，包括尚需人工资产的目标；
- UniFormerV2、UMT、InfiniPot-V、MuKV 保留候选配置与上游锁，不注册不存在的可运行 adapter；
- 人工资产缺失和许可阻塞均保留明确状态，不能用同族替代权重或 mock 充当原生通过。

因此 `encoders list` / `integrations list` 的正确当前数量为 21；文件名和部分帮助中的“25”不代表实际 registry 数量。

## 已落地的技术决定

1. 统一 ClipBatch / EncoderOutput / StreamStep，模型私有预处理和输出变换位于 adapter 内。
2. TorchVision、PyTorchVideo、Transformers 等共享后端复用实现，特殊上游保留最小 adapter。
3. 依赖兼容路径使用当前进程，冲突路径依赖服务器选择独立 Python；代码中仍并存 JSON/stdin 与 NPY sidecar 两类 worker，隔离和传输边界见编码器运行机制。
4. 列表与 catalog 检查保持轻量；真实模型只在构造/执行阶段加载。
5. smoke v2 记录身份、shape、dtype、有限值、时间轴与 stream/cache 遥测；运行耗时只作诊断。
6. 环境按四组 v2 与模型 overlay 部署，原有按模型零散环境的计划已被[四组环境决策](2026-09-03-four-group-encoder-environments.md)替代。

当前 worker 状态生命周期与真实输出阶段见[编码器运行机制](../architecture/encoder-runtime.md)。不能因为统一输出 shape 或第二 chunk 成功，就推导跨模型的全部语义已等价。

## 验收结果与剩余项

已有服务器历史结果为 14 路已许可 smoke 通过、2 路技术验证但许可阻塞、5 路人工资产缺失、4 路仅候选。2026-09-11 重新读取并哈希了历史矩阵，自身内容仍一致；本轮没有重新运行 14 路前向。

可关闭原生压缩的 `off` 对照与按配置开启原生 HERMES 的验证都可能出现在不同历史记录中；最终行为由每次配置和 telemetry 决定。不能把原计划“统一关闭压缩”当作所有现存结果的事实。

后续接入继续要求目标自己的代码/权重、许可与离线环境证据，缺失项见[矩阵](../progress/encoder-integration-matrix.md)。结果复用、成功退出码和身份绑定的实现问题见[架构审查](../reviews/2026-09-11-architecture-review.md)。

本计划中的历史服务器修改、提交与推送约定不构成后续任务授权；使用当前分支与当前用户指令。
