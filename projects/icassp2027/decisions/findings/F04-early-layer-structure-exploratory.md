# F04｜早期层结构差异：探索记录

日期：2026-09-19。状态：`exploratory_not_confirmed`。

最新确认状态为 `coverage_insufficient`：预登记的剩余33视频 controls-only 检查已完成，
四模型均未满足预定匹配覆盖门槛，因此没有启动P04目标确认。下面的FIT发现保持探索身份；
这次停止不是“局部相似性差异不存在”的检验结果。详见本卡末尾的确认记录。

四个encoder的完整fit结果现已齐备。在固定相对深度0.25的输入站点，含异常视频组的
local cosine和`uniform_representative_coverage_k4`更高；原始、motion×brightness匹配及
加入输入亮度标准差的敏感性区间均保持同方向。它们是相关的相似度描述，不是独立性质。

**有效秩下降与local−nonlocal额外增量未在V-JEPA 2复现：对应区间跨0。** 因此当前证据
不支持“四encoder共同低秩下降”或“共同具有额外局部优势”的说法，也不能排除整体方向性
或场景等因素。保留完整50个共有签名及全部负向、跨零和缺失结果。

以下保留两模型与VideoMAE补充阶段的原表，并给出四模型修订。全部来自相同128个fit视频，
经过探索网格后形成，未做多重比较校正；不是独立确认、异常token定位、因果或压缩性证据。
VideoMAE与VideoMAEv2属于同家族；不同原生输入窗口和处理阶段不被当成完全等价。

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

## 两模型第3个block输入处的原敏感性结果

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

两模型阶段的原图保留于[早期探索图](../../../../outputs/icassp2027/control/cross-encoder-early-structure-20260919/early_structure_fit_diagnostic.png)。

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
并非因果控制或纯去均值；该补充阶段未计算VideoMAE有效秩三字段敏感性，后续补齐见下节。

前三模型阶段的原图保留于[三模型探索图](../../../../outputs/icassp2027/control/cross-encoder-early-structure-20260919/three-encoder-a02/early_structure_three_encoder.png)。

新图全部使用原motion×brightness匹配；三个模型不混池。PNG/SVG由真实审核CSV生成并实际
查看，9组点/区间绑定输入SHA。VideoMAE是同家族补充，三者使用相同fit视频且窗口不同，
不是三次独立数据确认。四模型更新见下节；独立确认和中立压缩干预仍未完成。

## 四模型完整结果与V-JEPA 2的限制（2026-09-19）

V-JEPA 2的新run完整完成128视频、1024个逻辑窗口：805,888条available记录对应787个
可用签名，12,288条unavailable记录对应12个缺失签名；CPU分析为10,000视频bootstrap。
root复核全部1024条真实几何/parity记录：64帧、256²、32×16×16网格、8192 tokens、D1024、
24 blocks；65,536个源帧索引均按冻结中心采样独立重建，observer/identity的features与pooled
最大绝对差均为0。SDPA未返回原生概率，四个probability sites缺失；P10/P11/P13保留NA，
且模型无CLS，不把context张量当概率或制造CLS结论。本次不构成新的梯度或缩短路径验收。

两个短视频（Normal_Videos155_x264、RoadAccidents069_x264，一正常一含异常）各有2个
边界clamp造成的重复输入窗口。因此1024个逻辑窗口对应1020个按video内源帧索引去重的
窗口。这里的去重依据是源帧索引，不是像素内容；统计单位始终是128个视频，
保持原定每视频8个逻辑区间权重，没有因结果删除短视频。

全部100,736行video×signature值有限；各主统计64/64，原匹配9分层、min-count质量54
（其余三模型为56）。root独立重建787个raw g及787个matched点和覆盖，最大差4.6088e-14。
下载的四个主统计文件均与completed worker SHA绑定；一次误取的VideoMAE controls被保留并
隔离，正确VJ controls SHA为`5158d640…1776e`，未用错误controls计算任何VJ敏感性。

按已冻结的相对深度对齐后，四模型共有的50个P02/P04、head为空的签名全部保留。只有5条
在四模型raw及原matched共8个区间均为正：input、mlp.pre_norm.input、norm2、output的
local cosine，以及input的coverage_k4。它们也通过新三字段敏感性的方向检查，但站点相关，
不能称5条独立确认。`coverage_k4`是在至多256个采样token中，取4个均匀位置的代表，计算
每个非零token与代表集合的最大余弦，再平均；它不是覆盖百分比、四token压缩实验或重构误差。

V-JEPA 2主站点为`block.5.input`（零起始），对应12-block模型的`block.2.input`。下表
均为matched原始正减正常均值差；两/三字段匹配的人群不同，不能直接归因于新增变量。

| V-JEPA 2指标 | 原motion×brightness匹配及95% CI | 加亮度标准差后及95% CI |
|---|---:|---:|
| 中心化有效秩 | +0.03542 [−1.11367, +1.19177] | +0.63262 [−0.25021, +1.51103] |
| local cosine | +0.00744 [+0.00398, +0.01086] | +0.00668 [+0.00399, +0.00942] |
| coverage_k4 | +0.00380 [+0.00176, +0.00588] | +0.00407 [+0.00245, +0.00566] |
| 每视频local−nonlocal cosine | +0.00305 [−0.00604, +0.01246] | −0.00172 [−0.00824, +0.00487] |

VJ有效秩的raw g为+0.05231 [−0.28528,+0.40589]；local−nonlocal的raw g为+0.05047
[−0.29623,+0.38720]，也不支持前三模型的共同下降或额外局部增量。local显著而nonlocal
区间跨0，不等于二者差异显著；配对差值已实际计算，没有对两个区间相减。

新的四模型输入反差诊断保留全部50签名×4模型×两种匹配共400行。三字段分别保留：
Time59/56、20分层/质量40；V2及VMA59/52、19分层/质量39；VJ61/54、21分层/质量39。
本次统一补齐VMA的rank三字段敏感性，不改变原阶段表；诊断使用新的固定seed序列，旧原始
和两字段区间仍保留原身份。新二字段点与原主统计完全一致，三字段仍为post-hoc fit诊断。

VJ local−nonlocal诊断另外保留全部41个可配对站点×两种匹配=82行，含20个与Time/V2
相对深度可比站点，其中5个也属于本次VMA切片。主站点结果跨0不代表所有VJ站点或深度
都没有差异；完整表保留且未另挑一个VJ层来挽救主切片结论。root重建全部派生点与覆盖，
最大差1.9568e-15；四模型400行匹配点独立复核最大差2.2205e-15。

![四模型固定相对深度的完整fit比较](../../../../outputs/icassp2027/control/cross-encoder-early-structure-20260919/four-encoder-a03/early_structure_four_encoder.png)

图中b是零起始的原生block编号；只画原两字段matched差值，不混入三字段人群。VJ的rank和
local−nonlocal跨零区间完整显示。四个模型共享同一组视频，图不构成四份独立统计样本。

## 限制与下一步门槛

当前数据只支持含异常视频与正常视频的组间描述。source/scene 元数据仍 unknown；相同 decoded
分辨率 320×240、帧率 30 及已有控制，不足以排除所有内容或来源差异。没有事件或 patch 真值。

F03 的 embedding 全 token 高相似比例假设仍未通过原预登记确认。本卡的站点和统计定义不同，
将来若推进必须另行预登记并取得新的独立确认，不能改写 F03 或沿用其确认样本作为新证据。

跨模型以既定相对深度对齐，不按同编号block直接匹配。已完成的四模型native后层回执显示：
VideoMAEv2、TimeSformer、VideoMAE为12 blocks，V-JEPA 2为24 blocks。原探针公式
`ceil(relative_depth × block_count) − 1`使0.25分别对应前三者的block.2和V-JEPA 2的block.5；
0.5/0.75/1.0同理分别为5/8/11与11/17/23。保持实际层号、完整site后缀及sublayer/domain，
不会把TimeSformer局部attention与全局attention混池，也不因VJ结果改变主切片深度。
该映射只核验结构元数据，不证明处理阶段语义完全等价；当前VJ全部1024条architecture
回执已确认24-block映射一致。映射回执为`outputs/icassp2027/control/`
`cross-encoder-depth-alignment-20260919/alignment.json`（SHA adfc2cc8…cd15f）。

四模型的当前fit性质探针与统计均已完成。TimeSformer局部attention补充中的局部head方向不一致，
不能与本卡的token结构统计合并为一个全局attention规律。
剩余未用于原两轮确认的 UCF confirm 容量仅为 33 个视频（16 正常/17 正视频）。后续确认方案
必须在读取这些视频的目标统计前固定假设、匹配与覆盖门槛，不能因结果改变分箱或降低要求。

后续若推进，优先评审一个主性质（固定相对深度输入处的local cosine）；coverage_k4只作相关
辅助描述，不把当前跨零的rank/local−nonlocal登记成第二条四模型共同主性质。此为就绪度建议，
当前尚未注册或执行新确认。33视频下匹配稀疏与单例风险必须先纳入冻结的覆盖门槛；不足时
报告不可评估，不因确认结果合bin、换层、换指标、删control或降低门槛。

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

- `outputs/icassp2027/control/vjepa2-fit128-descriptive-20260919/`：完整VJ四文件原字节镜像、
  正确controls/resource/worker回执及root-mirror-verification；contrast SHA
  `1418b655291bd40a78df1a0c1b269da18b6b3416ed0dd882afef815217ed0057`，video SHA
  `f03a421cae4030f648a3b88536ec0cbf5035599a54a90d3a7a87f485d6655024`。
- `outputs/icassp2027/control/vjepa2-fit128-integrity-20260919/`：全几何、parity、源帧与787点复核；
  geometry receipt SHA `fda27651bd7b9b70c11ed4ea08d6b87a44fe44f76aad0f42f7b43f4fd9641ba3`。
- `outputs/icassp2027/control/relative-depth-comparison-20260919/a05_four_completed/`：四模型全
  union及50共有签名长表，模型、实际site、相对深度、状态、点、区间和来源分别保留。
- `outputs/icassp2027/control/four-encoder-input-contrast-20260919/`：400行完整敏感性，results SHA
  `f30a3ae819b00010bc68f261fe6ebfc7715db71d29d2ceb493e0e8cafd440d26`。
- `outputs/icassp2027/control/vjepa2-local-excess-fit-20260919/a01_verified/`：41站点82行派生诊断，
  results SHA `d978b0e232477cd7b5b1615002b47b419a249df182556d94dee25402e2fd1584`。
- `outputs/icassp2027/control/cross-encoder-early-structure-20260919/four-encoder-a03/`：12点/区间
  绑定的PNG/SVG及实际视觉检查；PNG SHA `61554cb715c26df6205ecafac48af7c271b984d3c557115375c61dc4aca1cd2e`。

## 新独立确认的覆盖停止记录

[冻结合同](../p04-four-encoder-confirmation-v1.md)在读取原confirm池最后33个视频的像素、
controls或目标统计前固定单一local cosine性质、四模型.25输入站点及覆盖条件。
本次真实执行仅计算原生8窗口的运动/亮度controls，应用各自FIT正常64校准。

| encoder | 匹配保留正常/含异常 | 完整分层 | min-count质量 | 非singleton质量/总质量 | 覆盖通过 |
|---|---:|---:|---:|---:|---|
| VideoMAEv2 | 14/15 | 7 | 14 | 11/14 = 0.785714 | 否 |
| TimeSformer | 14/15 | 7 | 13 | 9/13 = 0.692308 | 否 |
| V-JEPA 2 | 16/15 | 8 | 13 | 8/13 = 0.615385 | 否 |
| VideoMAE | 14/15 | 7 | 14 | 11/14 = 0.785714 | 否 |

各模型均通过至少12/label、质量至少10、至少2完整分层的前三项门槛；均未达到最后一项
预定0.80比例。“非singleton”在此指分层内两标签各至少2个视频。没有把0.785714四舍五入
成0.80，没有事后删分层、调门槛、改分箱或换样本。

完整四组264窗口已产出，输入33视频SHA及13个产物SHA封存；root独立重建132个视频的
controls均值与分箱、四组覆盖和27,456个源帧索引，均值最大差2.7756e-17。
完成回执SHA `b8320d17155e4b10b3c7eca9bf97f3ca0a784944cf6f232005a96fdf6aad8a06`。
仅用node3既有CPU原生环境、nice15、两线程、CUDA=-1，worker exit0，max RSS217,648 KiB；
它不是模型计时。原始镜像与独立复核见
`outputs/icassp2027/control/p04-four-encoder-confirmation-v1/`。

[机器可读结局](../p04-four-encoder-confirmation-v1-outcome.json)明确effect/CI保持null、
target未执行/读取、没有新GPU确认或LoRA启动。33视频的input-controls已被使用，之后不能
再称其完全未见；原角色锁未改变。新增独立验证样本需要先解决现有角色约束，不能偷偷复用
旧确认池、把FIT探索改称确认或把当前原型作为ours。全研究仍在进行，四组dense基线继续。
