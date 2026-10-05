# VideoMAE 的统一早期性质检查：fit 探索补充

日期：2026-09-19。状态：`registered_fit_check_not_independent_confirmation`。

目的：补齐第四个 encoder 对 F04 所涉及的同定义 P02/P04 早期统计的观察。四个 encoder
最终必须共用同一种压缩机制；这次补充不为 VideoMAE 单独设计算法，也不把 F04 升格为确认。
VideoMAE 与 VideoMAEv2 属同家族，这一结果是稳健性补充，不增加独立架构数量的主张。

固定输入为原 fit128：64 正常、64 含异常视频，每视频 8 个中心窗口。沿用原角色、原始内容
SHA、cohort/manifest 与权威训练视图，不读取新的 confirm/select/test 统计。原生 VideoMAE
窗口为16帧，frame stride=2；不把它与其它模型的不同原生窗口称为相同时间覆盖。

本次在读取结果前固定如下观察范围：

- `encoder=videomae`、`method=identity`、`partition=fit`、`role=explore`。
- `depths=[0.25]`，对应12个 block 的 `block.2`；主要对照站点是 `block.2.input`。
  其余由桥按固定注册规则产生的 embedding/分支站点同样保留，不按结果删选或更换主站点。
- `probes=[P02,P04]`、`max_tokens=256`、`max_queries=32`、`max_records=128`。
  三个主 encoder 既有/在运行的更大探索网格仍保留；四模型之间只在该同定义切片上作此项对照。
- 主要描述有效秩和同原生时间索引的局部/非局部 cosine；normalized rank仅为同一指标缩放。
  local−nonlocal若计算，必须先在同一视频内对相同8clip的均值作差，不能对置信区间相减。
- 每clip继续原模型、只读hook、identity三次前向及真实坐标一致性门禁；任何不支持项明确N/A。

完成全部128×8之前不解释结果。随后只做视频级 fit/explore 统计，10,000次 bootstrap、
预设seed20260919；控制校准使用 VideoMAE 自己的 fit-normal 窗口，保留原始与匹配估计量、
全部N/A及反向结果，不从中挑一个更有利的 head/层。该补充并不独立于已用于探索的128个视频。

输出以完整新run及输入/脚本/代码/模型/校准指纹封存。此项不训练 head、reducer 或LoRA，
不改变已确认/被拒绝的F01/F02/F03，也不消耗剩余33个未用确认视频。

若四者无法支持同一后续机制，记录具体的模型/统计/范围差异；不得将每模型另一套规则拼接为
一个方法。未来任何候选仍须另行预登记并经过独立视频确认和同预算中立干预。
