# F09-v0：整轨迹删除草案，运行前因静态信息风险撤回

本草案未派GPU、未作为论文方法。架构审阅指出：将低变化轨迹在全部时间删除，没有保留时间代表，会丢掉全部静态内容。当前版本改为相邻时间对保留完整reference、仅压target，见同目录F09-temporal-first-bounded-decision.md。保留本文件记录运行前修订，不将其身份写成已试验。

登记日期：2026-09-20；时区Asia/Hong_Kong。状态：`candidate_specified_not_confirmed`。F07现成cohort100审计并行继续；F06停止扩量，不换符号复活。只保留一个新时间候选和一个有条件的两阶段版本，不先认定二者有效。

## 决策及新颖性边界

建议先验证一个能保留两种骨干原生attention域的简单规则：**用同坐标的早期跨时间变化，选择保留完整的空间pair轨迹；如F07证实成员范数偏好有增量，再在后层对保留pair选一员。** 第一步实际删除空间轨迹，并不把多个CLIP帧合成一个attention序列，也不宣称它是原生跨帧token合并。低变化只被当作待检验冗余信号，不等同于正常、背景或无用。

这只是最小可实施干预，不是已成立的论文新机制。下面近作已覆盖上位结构：

- [FrameFusion，ICCV 2025](https://arxiv.org/abs/2501.01986)：相邻帧对应token相似性合并，随后importance pruning；“先时间再重要性”不能独立作为创新点。
- [CenterCLIP，SIGIR 2022](https://arxiv.org/abs/2205.00823)：视频分段token聚类与代表选择，CLIP时序冗余并非新问题。
- [TempMe](https://arxiv.org/abs/2409.01156)：逐步组合邻近clips并作跨帧token合并。不能直接把这一改过时序交互的模型当作原生独立逐帧CLIP。
- [ToMe](https://arxiv.org/abs/2210.09461)已在视频ViT使用相似token合并；[VidToMe，CVPR 2024](https://arxiv.org/abs/2312.10656)已有跨帧合并及分块处理，但其视频编辑attention改造不是本研究保留原生域的直接等价操作。

本候选的具体限制是各模型的原生attention域、CLS、来源坐标和单pass执行都不变，跨时间只计算选择信号；这是工程与实验身份差别，不自动等于足够的新颖性。若只有通用压缩收益，应写成既有思想的WSVAD适用性/边界研究，不包装“首次时间冗余压缩”。

## 架构约束

| 模型 | 真实输入/attention | 本候选允许的操作 |
|---|---|---|
| V2 | 16源帧，tubelet2，真实8×14×14网格；联合时空ViT | 依据真实(t,h,w)保留完整pair轨迹，随后仍为原联合attention |
| VideoMAE | 原生tubelet与联合ViT，具体参数按其回执 | 当前只保留扩展可能，不新增本轮实验线，不称为另一种2D架构 |
| TimeSformer | 分离时间/空间attention及其原生布局 | 不强行纳入主插件，不套V2 reshape |
| OpenAI CLIP ViT-B/16 | 每帧197序列、独立CLS和attention；16帧作为batch | 跨帧只算分数；每帧仍独立执行缩短后的attention，CLS始终保留 |

相同16源帧不等于相同native时间粒度：V2为8个tubelet，CLIP为16帧。原时间支持/中心分别记录，原始分数不跨encoder混池。所有状态只在一个固定输入clip内生效，处理下个clip时清空；不构造跨clip缓存，不称decoder KV cache。

## 第一阶段T：当前clip内的时间信号，约80% patch预算

为避免V2时间位置编码本身制造变化，信号取**位置编码加入前的真实patch embedding**：V2 patch_embed输出，CLIP conv1输出。不能事后从经过非线性block的hidden减位置向量，假装恢复内容。bridge分别转换为已验证的`[T,H,W,D]`，不得猜reshape。

对空间位置p的embedding向量作通道L2归一化，定义：

`d_p = mean_t (0.5 * ||normalize(e[t,p]) - normalize(e[t-1,p])||²)`。

这是相邻native时间方向差的描述，等范数下等价于余弦距离；差向量写法使完全重复embedding的分数精确为0。zero/nonfinite embedding要记录，不声称来自正常/异常含义。算法不读取labels、文件名、head分数或后层tensor。

同一时间的水平相邻两patch构成原生pair；pair轨迹的分数为两个位置d的均值。14×14网格有98个水平pair，按native pair顺序划成19组各5pair，尾组3pair。每完整组保留4条pair轨迹，尾组保留2条，选较高时间变化者；因此每个native时间保留78pair=156patch。

tie按预先固定的局部uniform优先序处理，使完全静态的embedding退化为同预算uniform，不随机抖动。uniform控制每5pair保留预登记均匀4项，尾组保留两端；random控制在同组/同quota内用seed20260920选择。三者保留数、局部覆盖约束和轨迹完整性相同。

信号在embedding阶段已经得到，立即只保存小型pair分数/索引。**执行gather固定在block2输入，即完成blocks0–1后。** 这样不重跑prefix；原位置内容保存在实际选出的当前hidden里，按原生顺序gather。第一阶段不平均/复制跨帧feature，不补零，也不把已跑完的dense再删后处理称省算。

## 第二阶段N：仅F07有增量才启用

F07需先证明max-norm成员相对fixed和random-member有训练内增量且无明显标签组/头输出反转；若F07失败或未完成，N阶段不进入GPU候选，不为凑两层硬加。

N执行于完整block5之后、block6输入之前。T保留下来的每条水平pair轨迹仍有两成员；在当前hidden上按原生时间逐pair保留L2范数较大的一员（tie原生左员），CLS不参与。CLIP可在不同帧选择不同pair成员，但各帧数量相同，只在各自序列内gather，不拼接attention域。V2按当前live indices映射回真实(t,h,w)，不能套原dense flat索引。

这沿用成员偏好问题，但当前0.50条件预算不同于cohort100的0.60组预算；F07通过不直接证明这个组合有效，仍须当前干预。N-neutral为固定成员，N-random为同pair固定seed成员；不引入F06比值或新网络。

## 实际数量与公平对照

| 位置/arm | V2 patch tokens | CLIP每帧tokens（含CLS） |
|---|---:|---:|
| dense/完整identity，全部层 | 1568 | 197 |
| 第一阶段后（blocks2起），T/U/R单阶段 | 1248 | 157 |
| 两阶段的blocks2–5 | 1248 | 157 |
| 第二阶段后（blocks6–11） | 624 | 79 |

第一阶段实际patch比例是156/196≈0.795918；第二阶段条件比例是78/156=0.5；两阶段最终patch比例是78/196≈0.397959。乘积只对同一patch域成立，不能把CLIP的CLS算进同一比例，也不能把0.8×0.5误标成0.6。主probe只这一个组合点，不扫层/预算。

先跑T单阶段、uniform单阶段、random单阶段，均有完全相同的逐层路径。如果N获准，再跑以下factorial表：

| 标签 | 早期轨迹选择 | 后期成员选择 | 用途 |
|---|---|---|---|
| U/U | uniform | fixed | 双阶段中立结构基线 |
| T/U | 时间分数 | fixed | 仅时间信号激活 |
| U/N | uniform | norm | 仅成员性质信号激活 |
| T/N | 时间分数 | norm | 组合 |
| R/R | 同quota随机 | 同pair随机 | 同路径随机对照 |

五臂逐层实际token路径完全一致；所谓“仅时间/仅性质”指关闭另一阶段的**信号**、保留中立缩减来匹配计算。必须这样标注，不能假装它们只有一次操作。纯单阶段T的80%结果另列，不能与最终40%的组合直接宣称增益。第一阶段信号开销、第二阶段norm/gather开销均计入，不能只用最终token数匹配计算。

## 4+4训练视频最小干预

每个数据集固定4正常+4含异常视频、每视频两个窗口、每窗同16源帧，两encoder共享。UCF从既有允许fit名单按固定seed/ID选择，XD从accepted3950可用训练成员选择并按电影/来源去重；先冻结ID/原roles，不能挑看到有利分数的样本。已有2+2可复用但不能为追结果换人；若需要新增两个正视频，优先不重复已有来源，来源未知则披露。只把这批当探索，不消耗/冒称独立confirm。

各arm保存原input hash、当前stage实际tensor dtype、embedding分数、阶段indices/原坐标、逐层shape、pooled小向量、identity/dense重复。主响应为与dense同输入最终pooled绝对MSE/cosine；合法冻结后端可用时另列训练侧logit漂移，不调用test、不重训头。每视频等权汇总；normal/positive的时间分数、丢弃轨迹分数和扰动分别报告，不能把视频标签当patch真值。

输入motion/brightness按共同原16帧计算，匹配或分层只作探索性控制；4+4不足以宣称场景/相机运动完全匹配。特别检查低运动正视频是否出现相对uniform/random的大幅损伤，但低运动正视频仍不能直接称静态异常clip。另用一个预声明重复帧fixture验证分数为0、tie回退uniform；fixture只验工程，不作为真实性质证据。

## 截止与止损建议（由root最终排卡）

- **2026-09-20 22:00前：**固定配置/ID/接口，完成CPU shape、budget、静态tie、跨帧域隔离测试；F07继续独立完成。若两stage需要改骨干attention语义或在线双跑dense，立即NoGo该实现。
- **23:30前：**完成第一阶段T/U/R的4+4双集配对表。identity/真实shape不过，或T不优于同路径U/R并在任一模型/标签组明显反向，停止T扩展；不换层、改分数、加运动网络或扫预算救结果。小样本不证明总体无效，但不足以晋级。
- **2026-09-21 01:00前：**仅在T与F07均有明确训练侧增量时完成factorial小表。组合不优于两种单信号臂，或增益仅因计算路径不匹配，停止组合；不强行保留两阶段。
- **08:00最终Go/NoGo：**缺少任何必要证据就NoGo新两阶段主方法，不把工程成功升级为独立确认。Go须明确主规则、性质范围、输入/后端身份和质量验证计划；正式test仍必须先冻结方法与格子，4+4的MSE不替代检测质量/非劣CI。可以同步撰写已审dense、性质、负结果和方法边界部分，不能为了开始写论文而填入未验证方法数字。

若截止前F07仍无可靠增量，第二阶段直接NoGo；第一阶段即使有通用收益，也只按时间冗余基线/研究结果定位，不将其变成异常性质创新。不同压缩率的正式测试属于Go后冻结格子，不在探索里扫描全部比例挑最好。

## 给Luna的有界实现任务

写`work/codex-takeover-20260920/idea/temporal_pair_trajectory.py`纯tensor规则、`run_temporal_pair_probe.py`实验runner及小型测试，复用S1原视频/加载/readout/receipt。一个worker独占这些文件，其他人不改；不改公共协议、旧结果或已运行snapshot。

规则函数只接收已验证`[T,H,W,D]`preposition embedding和实际pair表，返回78条空间pair索引及分数；不能收labels/文件名。T stage在模型instance的block2 pre-hook内按实际hidden gather。CLIP的16帧保持batch域，绝不把它们reshape成同一序列；V2从真实tubelet来源展开索引。保留每stage live-original-index映射。

若N获准，再用block6 pre-hook对当前pair成员做batched gather。现有公共CLIP bridge只支持一次静态1D gather并检查native长度，不可关掉其检查来容纳第二阶段；在实验runner中单独实现明确长度/映射的instance级staged hooks，identity时仍验证native结果。F07、正式提取和其它项目不受影响。

验收：两个模型的单pass计数为一次native forward；hook无变更时输出parity；T/U/R counts与预算表完全一致；N索引只引用当前存活成员；每帧独立CLS/attention域；所有阶段保留真实位置来源；下一clip没有遗留状态；dtype/权重/input/code SHA齐全。GPU统一由infra在node2/3分配，本文件不自动启动作业。
