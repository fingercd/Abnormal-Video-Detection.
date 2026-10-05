# handoff_v2 / lineending_equivalence.md

**生成时间**：2026-09-22（CST）
**作用**：归档"代码指纹行尾等价"豁免的**已有证据**与**确切作用域**，供论文方法学审查与后续审计引用。
**不含任何正式测试指标数值。**

---

## 1. 事实陈述

两个编码器（VideoMAE、TimeSformer）共用的一个适配器包装源码文件
`src/vadbench/integrations/transformers_video.py`
在两次不同快照中**内容完全相同、仅行尾不同**：dense 训练侧为 LF，压缩提取侧为 CRLF。

`_verified_code_digest` 按**原始字节**计算代码指纹，因此同一个文件在两种行尾形态下得到两个不同摘要，
被 `validate_compatibility` 判为"两个代码版本"，导致 direct_insert 兼容校验失败。

**这不是科学差异**：模型权重、transformers 骨干、processor、模型配置、依赖库版本、runtime 配置全部一致。

---

## 2. 豁免的确切作用域（已代码级取证）

放行表定义于 `src/vadbench/paper/compatibility.py`：
`_LINE_ENDING_EQUIVALENT_CODE_DIGESTS`（frozenset）

生效路径：`_same_direct_insert_contract()` ← `validate_compatibility()` 的 **direct_insert 分支**。

| 字段 | 处理方式 |
|---|---|
| `code_digest` | **唯一可被豁免的字段**，且必须是放行表中枚举的成员对 |
| `output_dim` | 严格相等，不等即 `return False` |
| `precision` | 严格相等 |
| `position_strategy` | 严格相等 |
| `backbone.runtime_id` | 严格相等 |
| `backbone.weights_digest` | 严格相等 |
| `backbone.preprocessing` | 严格相等 |
| `backbone.readout` | 严格相等 |

- `refit_head` 模式走 representation 整体全等，**不放行**。
- 放行表是**固定枚举集合**、按成员判定，不是"任意 LF/CRLF 通用规则"。
- 放行表**无传递闭包**：`sha256:30fe33c1…` 同时与两个摘要配对，但这两个摘要之间不构成豁免对（行为更保守，非更宽松）。
- 全仓 live 代码中放行表仅一处引用；`work/` 下的同名副本均为冻结归档，非 live 路径。

---

## 3. 放行表内容（3 对，逐字）

```
("sha256:81e3144071f0a2c0b3f7696a0c88c8853d7f225be7ef2a92ec458f7503e4f579",  # dense: VMA × UCF
 "sha256:30fe33c1cedd8fe065b2985514549b196241dfc51ebc37fa00025b5e5aada264"), # compressed

("sha256:7f9a3ac6b5528cf0d02281a35a7d2093bf927de70421c1f57fc78f8ed3c9c4de",  # dense: TS × UCF+XD
 "sha256:e5fd0bdf4679f16622b5b3d3ea6dcf74ed0a6a1139607f3424e0cc515a879eed"), # compressed

("sha256:9a23c5da46de06dd166b314207b85d72822048c6f3d25c492fd763d8e65bc999",  # dense: VMA × XD
 "sha256:30fe33c1cedd8fe065b2985514549b196241dfc51ebc37fa00025b5e5aada264"), # compressed
```

### ⚠️ 历史文档计数更正

本仓多份文档（`AGENTS.md`、`CLAUDE.md`、`projects/icassp2027/progress.md`、
`STATUS_DATA_SUMMARY_20260922.md`、`PROMPT_FOR_ADVISORS_20260922.md`）此前写作"**四对**"。
2026-09-22 由 6 个 dense head 的 `checkpoints/final.pt` 元数据实推
`representation.backbone.code_digest`，确认正确数量为**三对**。上述文档均已更正并保留更正痕迹。

---

## 4. 豁免覆盖范围推导（从 head 元数据实推，非类比）

训练侧 digest 取自 `training-full/<enc>/<dataset-head>/checkpoints/final.pt` 的
`metadata.representation.backbone.code_digest`：

| 编码器 × 数据集 | dense 侧 code_digest | 压缩侧 code_digest | 是否需豁免 |
|---|---|---|---|
| VideoMAE × UCF-Crime | `81e31440…` | `30fe33c1…` | ✅ 豁免对 1 |
| VideoMAE × XD-Violence | `9a23c5da…` | `30fe33c1…` | ✅ 豁免对 2 |
| TimeSformer × UCF-Crime | `7f9a3ac6…` | `e5fd0bdf…` | ✅ 豁免对 3 |
| TimeSformer × XD-Violence | `7f9a3ac6…`（与 UCF 相同） | `e5fd0bdf…` | ✅ 同一对 |
| VideoMAEv2 × UCF-Crime | `19720d4e…` | `19720d4e…` | ❌ 原生一致 |
| VideoMAEv2 × XD-Violence | `19720d4e…` | `19720d4e…` | ❌ 原生一致 |

**结论**：18 路正式 run 中，**12 路**（VideoMAE×6 + TimeSformer×6）走豁免，**6 路**（VideoMAEv2）原生匹配。
放行表的 3 对精确覆盖全部 12 路，无遗漏、无多余。

**实测佐证**：远端放行表含 5 个唯一 digest 字面量（`30fe33c1`、`7f9a3ac6`、`81e31440`、`9a23c5da`、`e5fd0bdf`），
VideoMAEv2 的 `19720d4e` **不在**表中；所有 XD 评分日志 `direct_insert may only` 报错数 = **0**。

---

## 5. 原始证据索引（既有，未重做）

| 证据 | 位置 / 内容 |
|---|---|
| 两版本文件去行尾差异后对比 | `diff` 为空，0 行不同 |
| 同一文件 LF/CRLF 双形态哈希 | 分别等于 dense 侧 / 压缩侧记录值 |
| XD-VMA dense 训练视图复算 | 全部 **68** 个源运行均等于 head 记录值 `9a23c5da…` |
| XD-TS dense 训练视图复算 | 全部 **135** 个源运行均等于 head 记录值 `7f9a3ac6…` |
| 复算函数 | 仓库原装 `_verified_code_digest`（非自定义哈希） |
| 补丁原版备份 | `compatibility.py.bak-pre-lineending-equivalence-20260922`（远端快照 `code-kimi-adgs-20260921`） |
| 适用 run 清单 | 12 路：VideoMAE × {UCF,XD} × {pair_select, group_uniform, group_random} + TimeSformer × {UCF,XD} × {pair_select, group_uniform, group_random} |

依据计划 §1.4：**证明已经覆盖的对象不重复折腾；未覆盖对象不能靠类比补齐。**
原快照保持只读；后续行尾治理在新副本中进行。

---

## 6. 论文方法学审查的预期质疑与应答要点

| 质疑 | 应答要点 |
|---|---|
| "为什么两个不同哈希被判为同一份代码？" | 不是"判为同一份代码"，而是**显式声明**这对摘要对应的源码内容在规范化行尾后逐字节相同；豁免是白名单枚举，不是通用规则 |
| "会不会掩盖真实的代码差异？" | 不会：豁免只放开 `code_digest` 一个字段，权重/骨干/processor/配置/库版本/output_dim/precision/position_strategy 仍逐一严格相等；且 diff 为空是实测而非推断 |
| "Python 不是会规范化行尾吗？为何还影响哈希？" | 影响的是**指纹计算**（按原始字节），不影响**执行语义**。参考 Python Language Reference, Lexical analysis, Physical lines |
| "未来新差异是否自动放行？" | 否。新差异必须先定位实际差异，证明仅行尾不同后才可加入；未证明前保持失败 |

---

## 7. 未做的事（如实记录）

- **未留决策文档**：当时按负责人指示不留（"C不需要这么研究，直接去跑推理，这个是完全合理的，不必记录"）。本文件即为事后补的等价证据归档，时间戳如实标注，不倒签为冻结时点的决策。
- **未统一远端快照行尾**：原快照 `code-kimi-adgs-20260921` 保持只读，防止原记录指向的代码被改写。
- **未扩展豁免到任何未证明差异**：放行表自建立以来未新增成员。
