# handoff_v2 / efficiency_protocol.md

**状态**：**预写协议**（2026-09-22）。候选方法尚未由负责人确定，本文件的方法清单在决策后回填。
**写入时点要求**：依据议程 §D2，benchmark manifest 必须在**测量之前**落盘；本文件即为该 manifest 的文字部分。
**边界**：效率测量**不需要标签**，**不运行任何正式 test 指标计算**，**不读任何测试 manifest**。

---

## 1. 为什么必须先写协议

- 避免"按结果是否好看来挑样本/挑计时边界"（议程 §D2、§D4）。
- 保证所有方法享有**同样的工程设置**（议程 §D3）。
- 让论文效率表可复现、可审计。

---

## 2. 必做范围（议程 §D1）

| 项目 | 要求 |
|---|---|
| 编码器 | VideoMAEv2、VideoMAE、TimeSformer（全部三个） |
| 每编码器至少测 | **dense** 与**最终候选** |
| 强烈建议同测 | group_uniform 与 group_random（否则结论仅限 dense 与候选两者，不能声称候选效率优于未测对照） |
| 覆盖条件 | 正式实验实际使用的输入形状、batch size、precision、token 插入位置 |
| 数据集分别测的条件 | 两数据集的上述条件若不同则分别测；计算工作负载完全相同时可共享一组 encoder 核心测量，但**必须明确适用条件**；解码/端到端耗时涉及数据差异，不能自动合并 |

---

## 3. 小样本起步方案（议程 §D2，可直接采用）

| 项目 | 起步方案 |
|---|---|
| 输入来源 | 两数据集各 **12 个非测试视频**；按时长**短/中/长各 4 个**；**不参考测试分数选样本** |
| clip 选择 | 每视频按固定位置规则取 **4 个有效 clip**；目标每数据集 **48 个**；不足时记录实际数量与原因 |
| 工作负载 | **优先复现正式提取配置**；额外 batch size 扫描**不是本轮必需** |
| warm-up | 每配置先固定 **20 次**，保证初始化/编译不混入稳态计时；未稳定则记录并统一调整策略 |
| 重复 | 每配置 **3 次独立进程重复**，使用相同固定输入；保存原始记录，汇总**中位数和波动**，不只报最快一次 |
| GPU | **独占**同一型号、可比状态的 GPU；不与正式提取抢同卡资源 |
| 标签 | 不使用；不运行正式 test 指标计算 |

**扩展判据**：若数据相关分支、动态 token 数或变长 batch 导致样本未覆盖实际形状，须补齐代表性工作负载。
**命名纪律**：小样本测得的峰值只能叫"该工作负载下的峰值"，**不能**叫"完整测试集的最大峰值"。

---

## 4. 测什么：必须是真正执行 token 压缩的 encoder 提取路径（议程 §D3）

**必须包含**：压缩之前的计算 → 选择器 → token gather/scatter → 后续层 → 最终 readout。
**不能只测**被缩短的 attention。
**明确排除**：UR-DMU 读取预提取特征时的显存**不能**替代 encoder 的显存。

---

## 5. 显存分口径记录（议程 §D3 + [R6][R7]）

三个口径**必须分开记录，不得混用**：

| 口径 | 含义 | 取法 |
|---|---|---|
| `peak_allocated` (GiB) | tensor 实际占用 | `torch.cuda.max_memory_allocated()` |
| `peak_reserved` (GiB) | 缓存分配器管理的内存 | `torch.cuda.max_memory_reserved()` |
| 进程总占用 (GiB) | `nvidia-smi` 口径 | 另测，不替代前两者 |

**测量窗口纪律**：
1. 独立进程 warm-up 并 **CUDA synchronize**；
2. 调 `torch.cuda.reset_peak_memory_stats()`；
3. 测规定的完整前向窗口；
4. 再读峰值。

该统计**只能覆盖 reset 之后的窗口**。稳态峰值与冷启动/加载峰值**分别标注**，不得混用，
**不得**通过减去权重占用来夸大总显存降幅。

---

## 6. 计时处理异步执行（议程 §D3 + [R6]）

- **GPU 核心时间**：用正确同步的 **CUDA events**（`torch.cuda.Event`），记录 elapsed_time。
- **端到端时间**（含解码、传输、序列化）：另测，墙钟区间边界**必须同步**。
- **禁止**仅用未同步的 CPU 计时（可能测到提交时间而非执行时间）。
- **选择器开销已计入总耗时**；单独拆解可用另一轮 profile，但**不得把重度 profile 的计时混进主表**。

---

## 7. 所有方法同样工程设置（议程 §D3）

下列各项对 dense 与全部压缩方法**必须相同**，任何偏差都要记录：
dtype / 输入 / batch size / 缓存策略 / 预处理 / 计时边界 / 输出存储策略。

**明令禁止**：候选关日志而 dense 开重日志；候选用优化实现而对照仍用低效实现而不说明。

---

## 8. 计算公式与报告纪律（议程 §D3）

```text
峰值显存降低比例   = 1 - compressed_peak / dense_peak
同工作负载加速比   = dense_latency / compressed_latency
实际 token 保留比  = sum(retained_tokens) / sum(native_tokens)
```

- **同时报告原始 GiB 和延迟值**，不能只报百分比。
- **0.60 是保留约 60%、移除约 40%**；**不能**据此直接推导"显存减少 40%"或"耗时减少 40%"。

---

## 9. 最低交付表（议程 §D1）

| 编码器 | 工作负载 / 数据域 | 方法和版本 | 实际保留比 | 峰值 allocated GiB | 峰值 reserved GiB | encoder ms/clip | 吞吐 clips/s | 相对 dense 显存变化 | 相对 dense 加速 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

所有测量必须对应**实际运行的版本**。此前 3 分钟 monitor 保留为历史运行记录，
**不用于补测结果，也不冒充原整轮提取峰值**。

---

## 10. 现有工具与已知缺口

| 项 | 状态 |
|---|---|
| `scripts/icassp2027/benchmark_frozen_reducers.py` | 存在（669 行）。默认 dry-run，`--execute` 才真正跑；**只读 fit-only manifest、不读任何 test manifest/帧标注/分数/head/FeatureStore**；`--dense-only` 是显式的更窄路径，clip 级 `end_to_end` 口径为 decode + adapter encoding + pooled readout，**不是 UR-DMU detector benchmark** |
| `vadbench/paper/efficiency.py` | 已实现议程 §D3 要求的口径：`CudaTimingRuntime`（synchronize / `reset_peak_memory_stats` / `max_memory_allocated` / `max_memory_reserved`）、CUDA event 计时、`measure_scope` 的 warm-up 与 repeat 同步 |
| **缺口 1** | 该脚本每次只接受**一个** `--fit-video-id`，需要写一个**驱动循环**跑完 12 视频 × 3 进程重复 × 4 方法 × 3 编码器 × 2 数据集 |
| **缺口 2** | 需确认三种冻结 reducer（`pair_select` / `group_uniform` / `group_random`）在该脚本中的可用入口；`pair_linear` 需要 fit128 校准，不是本轮冻结候选 |
| **缺口 3** | 正式提取用的是 `--cuda-memory-fraction 0.5` 与 `--processor-tensor-type np`（TimeSformer）等设置，补测**必须复现**这些配置，否则不可比 |

---

## 11. 已知不可用的数据（重要）

| 数据 | 为什么不能用 |
|---|---|
| FeatureStore `plugin_overhead_ms` | 随 chunk 重算的**滑动平均**（同 chunk 内恒定、跨 chunk 单调收敛 3,335,345 → 1,895,034 → 18,998），早期值被初始化污染；且 `pair_select` 记 `transform_ms`、两个对照记 `gather_ms`，**走不同代码路径**，正是议程 §D3 禁止的"工程设置不一致" |
| 08:57–09:00 GPU monitor | 只覆盖约 3 分钟；保留为历史运行记录，不作为任何效率主张依据 |

---

## 12. 止损（议程 §D4）

1. **先确定候选，再测对应版本的效率。**
2. 效率不达预期时，**先检查测量是否公平、压缩是否真正缩短执行维度、选择器开销是否抵消收益**。
3. **不得**为得到好看的数字而修改计时边界或只保留有利输入。
4. 实际显存收益很小就如实写小；吞吐没有改善就不写端到端加速。
5. 必要时由负责人依据完整质量—效率证据**重新作工程选择，记录第二次决策**，不让 agent 自动反复搜索。
