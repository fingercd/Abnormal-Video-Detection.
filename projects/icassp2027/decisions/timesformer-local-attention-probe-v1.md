# TimeSformer 局部 attention 观察口径 v1

日期：2026-09-18。状态：只读观察实现与单元测试完成，真实 native 验收及新观察仍待执行。
这不是性质候选、压缩方法或确认结论，也不改写已运行的 `code-f25d79e` 探针。

## 原生域与独立单位

使用 bridge 通过实际 patch/分离布局验证得到的 `B,T,P`，其中 `P=H×W`。不从张量长度猜布局。

| 分支 | 原生 execution groups | 每组 key 域 | 每 clip 归约 |
|---|---|---|---|
| temporal | `B×P` | 固定空间位置的 `T` 个时间 token | `P` 组等权平均 |
| spatial | `B×T` | 当前帧的 `P` 个 patch＋1个临时 CLS | `T` 组等权平均 |

临时 frame-local CLS 保留在 spatial 的真实 key/query 域，不能删除后悄悄改变概率定义。
每个 head、每个 source clip 只输出一组统计；`B×P` 和 `B×T` 不是独立视频。
这里不建立全局 attention 矩阵，也不把临时 CLS 当作唯一全局 CLS 的注意力。
TimeSformer 的全局 P13 在本实现中仍为 `unavailable/global_cls_aggregation_not_implemented`；
这与 V-JEPA 2 因确实没有 CLS 而 `not_applicable` 的含义不同。

## 统计量

原设备上按既有固定规则选 query，默认最多每 native group 32 行，保留该组全部 key。
对本轮原生 `T=8,P=196`，temporal 为每组8行、每clip1568个原生query行；spatial为每组32行、
每clip256行。这些计数不是唯一全局token数，也不代表encoder序列缩短。

概率必须有限、非负，原生行和在既有 `atol=rtol=2e-3` 范围内等于1；随后除以实际行和，
与global collector使用相同统计公式。对每个group、head，使用 `K` 个局部key：

- P10：query行的归一化熵 `−sum(p log p)/log K` 的均值，以及top-min(4,K)质量均值。
- P11：先在已采样query内平均得到incoming分布，再计算归一化熵、top-min(4,K)质量和Gini。
- 对group统计等权平均回source clip；后续仍先等权平均8个clip到video，再以video为独立单位。

`K=1`时没有可用的归一化熵；padding/ragged、未知layout、错误shape/CLS元数据或非法概率
明确保留N/A，不使用仅剩的有效group伪造完整clip统计。

每行保留可机器读取的 `native_domain`、`groups_per_clip`、`local_key_count`、
`queries_per_group`、`native_query_rows`、`temporary_cls_policy` 与 `group_aggregation`。
`num_valid_tokens`在此指局部key数，`query_count`指跨group累计的原生query行数。
真实encoder各层token数仍来自architecture/execution回执。

## 验收与解释边界

合成测试覆盖B=2的不同分布、真实P-major/time-fast坐标、K=5非均匀手算、query采样、
容差内行重归一化、非法概率/padding/CLS元数据、单key域、原生group到clip/video ID映射、
参数/缓冲区/梯度与hook清理。global P10/P11复用同一公式并保持原有语义。
定向attention-only suite只挂probability hooks；仍保留native、observer、identity三次前向与
输出一致性检查，避免为未请求的token统计计算SVD。

新增真实运行必须在独立目录记录新代码与本口径版本，并先完成native权重/真实fit视频验收。
跨模型只比较各自真实域内正常—含异常的效应及不确定性，不能相减绝对entropy，不能把更低
entropy或更高incoming质量直接解释为异常token或可删除token。任何方法决策仍需独立确认与干预。
