# 代码精简：决策、边界与执行结果

计划日期：2026-09-03；执行结果日期：2026-09-04；2026-09-11 文档复核基线：`0badc34`。本任务已经执行，以下是历史记录，不是重新运行的待办计划。当前问题与改进建议见[架构审查](../reviews/2026-09-11-architecture-review.md)。

## 历史目标与边界

在 `qzt/refactor-vadbench-simplification` 上，以 `a40ae2b` 为基线，保持行为并减少无效防御、签名猜测、旧服务器入口和重复集成/编排代码。保留 UCF split/端点、真实权重身份、sidecar 路径/shape/dtype/hash、原子写、stream/cache 语义和服务器保护边界。

最终采用单次固定 worker 调用、统一输出规范化、smoke v2 单路径和四环境 v2 工具；删除 4 个未注册声明型 adapter 与 6 个旧服务器入口。两个仍用于旧环境离线恢复的 bootstrap 脚本被保留，并未按初始删除候选表全部删除。

最初“每次生产代码净删”“合计至少删 1,500 行”等量化约束是这次精简任务的范围，不能提升为之后修复正确性问题的通用规则。当前实现应优先满足真实需求、可信实验与可读性。

## 执行结果（保留 2026-09-04 原有数字）


### 9.1 代码量

主要代码完成提交为 `72faada`，最终格式提交为 `407dd66`。相对基线 `a40ae2b`：

| 范围 | 新增 | 删除 | 净变化 |
|---|---:|---:|---:|
| `src/ + scripts/` | 559 | 2,236 | **-1,677** |
| `tests/` | 510 | 114 | +396 |
| `src/ + scripts/ + tests/` | 1,069 | 2,350 | **-1,281** |

生产代码超过至少净删 1,500 行的门禁；新增测试用于锁定单次调用、边界错误和真模型兼容，代码与测试合计仍净删 1,281 行。

| 指标 | 基线 | 完成后 | 变化 |
|---|---:|---:|---:|
| `src/vadbench` Python 行数 | 24,484 | 23,178 | -1,306 |
| `integrations` Python 行数 | 10,972 | 10,101 | -871 |
| `scripts/server` 行数 | 1,491 | 1,284 | -207 |
| `src/vadbench` Python 文件 | 65 | 61 | -4 |
| server 工具文件 | 16 | 10 | -6 |

粗略语法计数从 `if=2362 / defensive-if=811 / raise=1154 / try=263` 降至 `2235 / 764 / 1137 / 249`。这些数字不是质量目标；主要收益来自整段删除多签名重试、旧入口、重复序列化和声明型模块，而不是把异常换名。

### 9.2 主要结果

- Windows sidecar 从 14 个失败恢复为全绿；NPY 1/2/3 header、hash、shape、dtype、size 和 path 安全继续覆盖。
- `smoke` 只保留一条 v2 执行路径；不再隐藏关闭压缩，stream telemetry 不再丢失，失败返回非零退出码。
- foundation、legacy 和 long-video worker 均改为明确单次调用；删除 frames/batch/x/pixel_values/opaque 等参数名猜测和 `TypeError` 后重试。
- 删除 4 个不在运行 catalog 的纯声明 adapter，同时保留它们的 candidate config、upstream lock 和状态。
- 删除 32 个无仓内消费者的兼容别名；协议文档明确承诺的 `ManifestRecord` 保留。
- 删除 6 个被 v2 覆盖的服务器入口；两个仍承担离线种子重建的 bootstrap 脚本保留。
- UCF split/坐标、feature fingerprint、权重 license/revision/hash、worker sidecar、cache kind/axis、原子写和服务器 no-overwrite 门禁未删除。

### 9.3 验证

- Windows：`366 passed, 9 skipped`。
- node3：`374 passed, 1 skipped`。
- node3 定向与全量 `compileall`、`git diff --check` 通过；Ruff lint 在 Windows 通过。
- 最终真权重矩阵：`outputs/refactor/final-native-72faada/matrix-v2.json`：
  - 14 `smoke_pass`；
  - 5 `manual_asset_missing` 与 2 `license_blocked` 按预期跳过；
  - 4 `candidate_only` 不进入 21 路运行 catalog；
  - 汇总仍为 14/2/5/4。
- foundation 专项：`outputs/refactor/foundation-6e0660b/`，VideoMamba `[1,1,192]`、V-JEPA2 `[1,8192,1024]`。
- long-video 专项：`outputs/refactor/long-video-signatures/` 与 `outputs/refactor/long-video-signatures-streaming/`；LongVU、VideoChat-Flash 通过，VideoChat-Online、StreamingVLM 技术前向通过且继续保持许可阻塞。

StreamingVLM 在合法结果写完后的进程退出期出现 `libgcc_s.so.1 must be installed for pthread_cancel to work` 并返回 `-6`。用未改动的 `c00d6eb` 代码快照、同一解释器/overlay/权重也复现 `exit 134`，证明它是当前 stream-kv 环境问题，不是本次代码回归；结果中没有把该退出码隐藏。最终 14 路 PASS 矩阵不包含许可证阻塞的 StreamingVLM。
