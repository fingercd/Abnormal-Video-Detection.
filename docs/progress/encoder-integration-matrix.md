# 25 路视频模型/VLM 接入矩阵

> 目录事实对齐 commit：`0badc3435e734a841110e29d497940bfda7cc707`
>
> 历史真权重证据：2026-09-03
>
> 服务器只读复核：2026-09-11

## 如何读这张表

25 是研究候选数，来源是 [`registry/encoder-candidates.yaml`](../../registry/encoder-candidates.yaml)；21 是可由框架发现的运行目标数，来源是 [`registry/encoder-integrations.yaml`](../../registry/encoder-integrations.yaml)。四个 `candidate_only` 目标没有进入运行 catalog，也不能传给当前 native runner。

`registered_local` 表示 registry 记录了本地代码/权重路线，`awaiting_manual_asset` 表示保留 adapter 定义但资产未闭合，二者都不等于本轮 forward 成功。运行 catalog 的静态状态只有 `planned`、`integrated`、`blocked`；最后一列只复述 2026-09-03 已保存的 smoke 证据，2026-09-11 未重跑模型。运行时契约与状态流见 [`encoder-runtime.md`](../architecture/encoder-runtime.md)，实现/后续验证汇总见 [`2026-09-11-implementation.md`](2026-09-11-implementation.md)。

## 分类、运行目录和历史证据

| # | ID | 环境组 | 候选登记 | 运行 catalog / 静态状态 | 导出阶段 / 状态种类 | 2026-09-03 证据 |
|---:|---|---|---|---:|---|---|
| 1 | `r2plus1d_18` | classic | `registered_local` | 是，`integrated` | `pooled` / 无状态 fixed | `smoke_pass`（CPU；另有 GPU 1 记录） |
| 2 | `x3d` | classic | `registered_local` | 是，`integrated` | `pooled` / 无状态 fixed | `smoke_pass`（CPU） |
| 3 | `mvitv2` | classic | `registered_local` | 是，`integrated` | `pooled` / 无状态 fixed | `smoke_pass`（CPU） |
| 4 | `slowfast` | classic | `registered_local` | 是，`integrated` | `pooled` / 无状态 fixed | `smoke_pass`（CPU） |
| 5 | `c3d` | classic | `awaiting_manual_asset` | 是，`blocked` | `fc_features` / 无状态 fixed | `manual_asset_missing` |
| 6 | `i3d` | classic | `registered_local` | 是，`integrated` | `pooled` / 无状态 fixed | `smoke_pass`（CPU） |
| 7 | `timesformer` | classic | `registered_local` | 是，`integrated` | `last_hidden_state` / 无状态 fixed | `smoke_pass`（CPU） |
| 8 | `video_swin` | classic | `registered_local` | 是，`integrated` | `backbone_tokens` / 无状态 fixed | `smoke_pass`（CPU） |
| 9 | `videomae` | classic | `registered_local` | 是，`integrated` | `last_hidden_state` / 无状态 fixed | `smoke_pass`（CPU） |
| 10 | `videomaev2` | foundation | `registered_local` | 是，`integrated` | `observed_backbone` / 无状态 fixed | `smoke_pass`（CPU） |
| 11 | `uniformerv2` | foundation | `candidate_only` | 否 | 计划 `backbone_tokens` | `unregistered`：目标权重链接失效 |
| 12 | `umt` | foundation | `candidate_only` | 否 | 计划 `backbone_tokens` | `unregistered`：目标权重链接失效 |
| 13 | `internvideo2` | foundation | `awaiting_manual_asset` | 是，`blocked` | `backbone_tokens` / 无状态 fixed | `manual_asset_missing`：gated 权重与 checkout |
| 14 | `videomamba` | foundation | `registered_local` | 是，`integrated` | `backbone_tokens` / 无状态 fixed | `smoke_pass`（CPU reference selective scan） |
| 15 | `vjepa2` | foundation | `registered_local` | 是，`integrated` | `backbone_tokens` / 无状态 fixed | `smoke_pass`（CPU） |
| 16 | `longvu` | visual VLM | `registered_local` | 是，`integrated` | `projected_visual` / 无持久状态 long | `smoke_pass`（CPU 视觉路径） |
| 17 | `videochat` | visual VLM | `awaiting_manual_asset` | 是，`blocked` | `projected_visual` / 无状态 fixed | `manual_asset_missing` |
| 18 | `videochat_online` | visual VLM | `registered_local` | 是，`blocked` | `visual_memory` / streaming | 技术前向通过，`license_blocked` |
| 19 | `videochat_flash` | visual VLM | `registered_local` | 是，`integrated` | `projected_visual` / 无持久状态 long | `smoke_pass`（CPU 视觉塔/projector） |
| 20 | `ma_lmm` | visual VLM | `awaiting_manual_asset` | 是，`blocked` | `visual_memory` / streaming | `manual_asset_missing` |
| 21 | `moviechat` | visual VLM | `awaiting_manual_asset` | 是，`blocked` | `visual_memory` / streaming | `manual_asset_missing` |
| 22 | `streaming_vlm` | stream KV | `registered_local` | 是，`blocked` | `decoder_contextual` / decoder KV streaming | 技术前向通过，`license_blocked` |
| 23 | `infinipot_v` | stream KV | `candidate_only` | 否 | 计划 decoder KV | `unregistered`：许可和目标 checkpoint 未闭合 |
| 24 | `hermes_llava_ov` | stream KV | `registered_local` | 是，`integrated` | `decoder_contextual` / decoder KV streaming；当前主配置 raw/off | `smoke_pass`（CPU、两 chunk） |
| 25 | `mukv` | stream KV | `candidate_only` | 否 | 计划 decoder KV | `unregistered`：许可和 MuKV 专用资产未闭合 |

环境组全名均带 `-v2`。完整路径和版本见 [`encoder-environment-v2.md`](encoder-environment-v2.md)。

## 当前代码能保证什么

- catalog 加载时严格校验字段、相对 YAML 路径、能力组合、streaming 至少两 chunk，以及 `integrated` 目标不能引用 `planned` checkpoint；运行成功/失败不写入 catalog。
- 注册与列举只保存 import string，不导入 Torch、Transformers 或上游代码；实例化时才加载 adapter，并检查实例能力是否与 catalog 一致。
- fixed smoke 取一个确定性中心 clip；streaming smoke 创建独立 `StreamState`、至少处理两 chunk，并校验 step 单调性、cache kind 与输出健康度。
- `smoke_pass` 需要实际 result JSON。单元测试、配置可解析、权重目录存在或 preflight 通过都不能单独计为真权重 PASS。
- VideoChat-Online 的状态对象属于 `visual_memory`；HERMES 与 StreamingVLM 才声明 decoder KV。LongVU/VideoChat-Flash 当前导出 projected visual token，不声明流式 KV。

## 真权重 PASS 门禁

1. 使用目标自己的固定上游代码和 checkpoint，记录 revision、许可、文件大小与 SHA256。
2. 使用真实视频 forward；fixed 至少一个 clip，streaming 至少两个连续 chunk。
3. 通过 adapter 输出契约、时间轴、有限值和 cache 语义校验。
4. 通过四组环境 runner 启动，记录实际 Python、base/overlay marker、命令、日志和退出码。
5. 许可不明的技术成功仍是 `blocked_license`；mock、随机权重、其他模型 alias、旧产物复用均不能升级为新 PASS。
