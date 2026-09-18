# F04｜早期层结构差异：探索记录

日期：2026-09-19。状态：`exploratory_not_confirmed`。

在同一批 UCF-Crime fit 视频中，VideoMAEv2 与 TimeSformer 的第 3 个 block 输入
（`block.2.input`，零起始索引）出现方向一致的组间差异：含异常视频的采样 token
激活范数、中心化有效秩较低，空间邻近 token 的余弦相似度较高。加入输入亮度标准差
匹配后，这些探索区间仍不跨零。局部相似度减去同一时间索引下非邻近相似度的增量也为正。

VideoMAE 的同家族补充现已完成，支持下述有效秩、局部相似度及局部相对非局部增量的探索方向；
它只采集预登记的早期 P02/P04 切片，没有新增范数证据。V-JEPA 2 仍待完整结果。

这些结果来自方法开发所用的 fit 数据，经过全网格探索后才形成当前检查，未进行多重比较
校正，不能称为独立确认、四 encoder 通用性质、异常 token 定位或可压缩性证据。

## 数据与统计身份

- 每模型 128 个原 fit 视频：64 正常、64 含异常；每视频 8 个固定中心窗口。
  先得到 clip 统计，再在视频内等权平均。bootstrap 以视频为单位，重复 10,000 次。
- TimeSformer 保持 8 帧原生输入，VideoMAEv2 保持 16 帧原生输入，frame stride 均为 2。
  两者的同名站点是各自原生网络的 block 输入，不把不同窗口解释为相同时间覆盖。
- VideoMAEv2 使用既有 Windows GPU fit 观察，本次已重新计算视频级统计；TimeSformer 使用
  node3 A100 的完整 fit 观察。不能将前者描述为本轮 A100 重新采集。
- 基础匹配为各 encoder 自己的 fit-normal motion/brightness 三分位分箱，完整分层固定
  权重为 `min(n_normal, n_positive)`。原始组间区间对应 Hedges g；matched 区间对应
  加权的原始正减正常均值差，两种估计量不能混用。
- 额外的输入反差诊断使用已有 `brightness_std`：每 clip 的亮度标准差先在 8 clips 内平均，
  再只用该 encoder 的 64 个正常视频拟合三分位阈值。它是事后敏感性检查，不替换原分析
  的冻结分箱，也不代表场景、纹理或因果控制。

基础匹配在两模型均保留 64/64 视频、9 个完整分层，较小标签样本数合计均为 56。
加入亮度标准差后，TimeSformer 保留 59 正常/56 正视频、20 个分层、权重质量 40；
VideoMAEv2 保留 59/52、19 个分层、权重质量 39。单边分层的排除改变了估计人群，
不能将两套点估计的变化全部归因于新增变量。

## 指标含义与完整探索范围

有效秩使用至多 256 个采样 patch token，中心化后计算 Gram 谱能量的归一化熵指数。
`effective_rank_normalized` 只是同一有效秩除以其最大可能值，不作为另一条独立性质。

P04 的 local/nonlocal 均基于实际 `(t,h,w)`：固定同一原生时间索引，空间曼哈顿距离为 1
的非零 token 对属于 local，距离大于 1 的属于 nonlocal。这里的时间索引对 VideoMAEv2
对应 tubelet，对 TimeSformer 对应原生帧布局，不能等同于带异常真值的空间区域。

两模型共有 384 个签名。按“两模型各自 raw/matched 四个 95% CI 均不跨零且同方向”
描述性筛查，22 条满足条件，均集中在 block 2 附近的输入/输出及相关站点：P01 10 条、
P02 2 条、P04 6 条、P16 4 条。所有未满足项仍保留；该筛查未登记候选或选定部署层。

其中 4 条 P16 `activation_norm_median` 在两模型的每个视频上都与同站点 P01 完全相同，
它们不能构成额外的 P16 证据。本记录与 F01 的相对 attention 更新指标具有不同定义，
不改变 F01 原有确认身份。

## 第 3 个 block 输入处的敏感性结果

下表全部为加入亮度标准差后的 matched 原始差值及探索性 95% CI，不是 Hedges g。

| 指标 | TimeSformer | VideoMAEv2 |
|---|---:|---:|
| token RMS 范数中位数 | −0.06629 [−0.08659, −0.04600] | −0.04849 [−0.06084, −0.03610] |
| 中心化有效秩 | −1.00054 [−1.81124, −0.19403] | −2.30959 [−3.41130, −1.21069] |
| 同时间索引的 local cosine | +0.03146 [+0.01464, +0.04818] | +0.05569 [+0.03963, +0.07169] |
| 每视频 local−nonlocal cosine | +0.01425 [+0.00319, +0.02503] | +0.01988 [+0.00824, +0.03138] |

最后一行先在同一视频内对相同 8 clips 的两个均值作差，再对这个视频级差值进行组间分析。
没有用两个独立置信区间相减构造区间。全部 20 个共有且无 head 的可比站点都已计算并保留，
其中只有 4 个早期站点的原始 Hedges g 区间在两模型都为正；其余站点存在单模型残留或区间跨零。
因此，局部相似增量并非仅由所有相似度等量增加构成，但也不是纯去均值操作或因果解释。

对最初 22 条筛入签名，完整保留 13 个异常类别的既有对比及逐个排除正类类别的点值敏感性。
该操作保持 64 个正常视频不变，没有新 bootstrap 或独立验证主张；不能按类别挑选较好的结果。

![两个encoder的早期层fit探索图](../../../../outputs/icassp2027/control/cross-encoder-early-structure-20260919/early_structure_fit_diagnostic.png)

## VideoMAE 同家族补充（2026-09-19）

本次遵循[预登记fit切片](../videomae-early-structure-fit-check-v1.md)，仍是相同128个fit视频，
不消耗独立确认视频。node3原生VideoMAE完成1024窗口、12 blocks、16帧、224²、D768、
mean readout。所有窗口的真实Conv3d flatten及8×14×14网格通过，root按权威视频帧数和
8个中心区间独立重建16,384个源帧索引；observer/identity的features与pooled最大绝对差均为0。
三个Q/K/V模块hook未触发，6个对应P02/P04签名保留unavailable。静态receipt仍有
`layout=unverified`、`probe_ready=false`；实际坐标证据来自独立的`flatten_verified`和源帧门禁，
不能把整个观察接口说成全可用，也不把该run当作新的梯度或缩短路径验收。

完整统计有80个可用签名、10,240行video×signature值，每项均为64正常/64含异常视频、
每视频8窗口；另6个不可用签名保留。原motion×brightness匹配均保留64/64视频、9个完整
分层、min-count质量56。root独立重建全部80个Hedges g与80个matched点及覆盖，最大差
`1.7431e-14`，下载表SHA与原worker输出绑定一致。

三模型精确共有且head为空的50个P02/P04签名完整保留；其中8条的三个模型raw/matched
共六个95%区间同方向且不跨0：block.2.input有效秩及其normalized缩放、4个早期站点的
local cosine、block.2.input的nonlocal cosine及coverage_k4。它们存在尺度重复和站点相关性，
不能计为8条独立性质。其它42条均保留，未从这张表选择部署层或方法。

下表只列预先固定的主站点`block.2.input`，使用原两字段匹配的原始正减正常均值差，
与前文两模型的三字段敏感性表分开；两种匹配估计的人群可能不同。

| VideoMAE指标 | matched差值及探索性95% CI |
|---|---:|
| 中心化有效秩 | −2.16031 [−3.31216, −1.01161] |
| 同时间索引local cosine | +0.05583 [+0.03926, +0.07244] |
| 同时间索引nonlocal cosine | +0.04176 [+0.02523, +0.05774] |
| 每视频local−nonlocal cosine | +0.01408 [+0.00523, +0.02314] |

最后一项的raw Hedges g为+0.45347 [+0.10813, +0.83170]。新派生诊断保留VideoMAE全部
8个可配对无head站点、两种匹配共16行；5个站点与旧两模型均可比，其中4个早期站点的raw
及原匹配区间在三模型都为正，MLP残差前分支的TimeSformer区间跨0。不能推广为任意站点规律。
加入本模型64个fit-normal视频的平均brightness_std三分位后，主站点matched增量为
+0.01375 [+0.00625, +0.02132]，保留59正常/52含异常、19分层、质量39。它是事后fit敏感性，
并非因果控制或纯去均值；本次未重算VideoMAE有效秩的三字段敏感性。

![三个encoder主站点的fit探索](../../../../outputs/icassp2027/control/cross-encoder-early-structure-20260919/three-encoder-a02/early_structure_three_encoder.png)

新图全部使用原motion×brightness匹配；三个模型不混池。PNG/SVG由真实审核CSV生成并实际
查看，9组点/区间绑定输入SHA。VideoMAE是同家族补充，三者使用相同fit视频且窗口不同，
不是三次独立数据确认。V-JEPA 2、剩余独立确认和中立压缩干预仍是后续门槛。

## 限制与下一步门槛

当前数据只支持含异常视频与正常视频的组间描述。source/scene 元数据仍 unknown；相同 decoded
分辨率 320×240、帧率 30 及已有控制，不足以排除所有内容或来源差异。没有事件或 patch 真值。

F03 的 embedding 全 token 高相似比例假设仍未通过原预登记确认。本卡的站点和统计定义不同，
将来若推进必须另行预登记并取得新的独立确认，不能改写 F03 或沿用其确认样本作为新证据。

V-JEPA 2 的完整fit观察仍在执行。TimeSformer局部attention补充已完成，局部head方向不一致，
不能与本卡的token结构统计合并为一个全局attention规律。
剩余未用于原两轮确认的 UCF confirm 容量仅为 33 个视频（16 正常/17 正视频）。后续确认方案
必须在读取这些视频的目标统计前固定假设、匹配与覆盖门槛，不能因结果改变分箱或降低要求。

通过独立确认后，仍须用同预算中立干预检验当前层可用的局部统计是否真能指导压缩。
在此之前，不确定 token 单位、selector、插入层、预算或 LoRA 配置，也没有插件质量/效率结论。

## 可复核产物

所有路径相对仓库；大型结果保持 ignored，不进入 Git。

- `outputs/icassp2027/control/timesformer-fit128-descriptive-20260918/`：完整网格、筛查、类别表与来源核验。
- `outputs/icassp2027/control/input-contrast-sensitivity-20260919/a02_verified/`：输入反差诊断。
  `sensitivity_results.csv` SHA-256：
  `43bb7d10b1b5a0ff42298dd84d7db140fde1d4df13307faa19016bab4cae74fa`。
  初版未绑定产物曾被覆盖，已另存说明且未用作证据；a02 的原文件、输入、脚本和哈希清单均保留。
- `outputs/icassp2027/control/local-excess-similarity-20260919/`：20 个共同站点的配对差值诊断。
  `results.csv` SHA-256：
  `2e16404c448a07a794ec2496f2ec4a5c69c6c9991f458f9fbd7321f45acda22b`。
- `outputs/icassp2027/control/cross-encoder-early-structure-20260919/`：已渲染并查看的 PNG/SVG、图数据回执、
  88 个 matched 点值/覆盖的独立重建，以及 40 个派生差值的独立点值与 Hedges g 核对。
  最大数值差分别为 `4.44e-16`、`2.50e-15`；这些复核不产生新的确认结论。
- `outputs/icassp2027/control/confirmation-capacity-20260919/`：只读元数据容量审计；未读取剩余确认统计。

- `outputs/icassp2027/control/videomae-early-structure-descriptive-20260919/`：完整VideoMAE统计与
  全50个三模型共同比较。contrast SHA `cd51d436938cb54b9366054893fe05fede82905fdadaa0258dd5eb271f492c21`；
  shared表 SHA `fa771f51e52e9c7a1f531e881dfdb40c0eb8dddb79d5be68b091eb05f44b933b`。
- `outputs/icassp2027/control/videomae-early-structure-integrity-20260919/`：全1024条几何/parity/源帧
  及80项统计点独立复核。architecture审计receipt SHA `8bb61cc9dbf7f5af5e5e15b19cbf5748d844475fb1b5cc65c3911f7122e885f0`。
- `outputs/icassp2027/control/videomae-local-excess-fit-20260919/a02_verified/`：完整8站点派生诊断，
  results SHA `802f9dbd6a240519c8a3d2311eae8a96fba4cd671c53c74528b9c2af8bb1f1e9`。
  root重建16个matched点及8个raw g/均值差，最大差`1.7764e-15`。a01漏raw统计及部分绑定，
  原样保留并标记不完整，不作为完成诊断；a02独立新目录绑定全部输入前后SHA、线程设置及真实版本。
- `outputs/icassp2027/control/cross-encoder-early-structure-20260919/three-encoder-a02/`：新增三模型图、
  figure-data、输入/输出SHA与视觉检查回执。初次使用项目解释器缺matplotlib，未安装依赖；
  改用先前已工作的本机pytorch解释器完成渲染。
