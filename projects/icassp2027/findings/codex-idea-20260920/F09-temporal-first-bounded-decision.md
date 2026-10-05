# F09-v1：保留时间reference的两阶段备选与截止决策

日期：2026-09-20，Asia/Hong_Kong。状态：`specified_not_dispatched_not_confirmed`。F07完整cohort100 CPU审计并行继续；F06不再扩量。本文是运行前架构审阅后的v1，撤回的整轨迹删除v0保留在同目录，没有GPU结果被改写。

**建议只保留一个新时间候选：相邻native时间对中，reference保留全部patch，只在target中压缩与reference相近的空间pair。** 第二阶段仅在F07有可靠成员范数增量时启用。先做小实验，不预认两阶段有效，不靠新网络或层/预算扫描救负结果。

## 架构与静态内容

V2/VideoMAE是实际tubelet(t,h,w)到联合时空ViT，不能当2D帧任意reshape；TimeSformer是分离时空attention，本轮不强拉进主插件。CLIP是每帧独立的197-token序列与CLS。跨帧比较产生选择分数，但不能把不同帧tokens拼成同一个attention。

v0按时间分数删除整条空间轨迹，会把静态内容在所有时间一起删掉；相似性本身不支持这种操作，因此在任何运行前撤回。v1在每对native时间中保留一个完整reference，压缩仅发生在target，至少保留该位置的一个原始时间代表。但target的CLS不能直接看到reference的patch，故仍然是近似、仍可能损失静态异常信息，不能宣称异常保留保证。

所有状态只存在于一个固定输入clip，下一clip清空；无跨clip持久复用，不叫decoder KV cache。固定16源帧全部读取，不引入超过原clip的未来帧。

## 文献边界

[FrameFusion，ICCV2025](https://arxiv.org/abs/2501.01986)已有相邻帧对应token相似性合并后再importance pruning；[CenterCLIP](https://arxiv.org/abs/2205.00823)已有视频分段聚类/代表token；[TempMe](https://arxiv.org/abs/2409.01156)已有组合邻近clip和跨帧合并；[ToMe](https://arxiv.org/abs/2210.09461)已有视频ViT合并，[VidToMe](https://arxiv.org/abs/2312.10656)已有跨帧合并与分块处理。

“先时间冗余，再性质压缩”上位结构不新。本候选保留原生attention域、保留reference、直接作用于视觉encoder并做同路径因子对照，但这些工程约束不自动构成论文创新。没有独立异常相关性质与可操作增量，就按通用基线/WSVAD边界研究定位。

## T阶段：分数、位置与预算

1. 同一个16帧clip：V2实际8个tubelet；CLIP实际16帧。分别将相邻native时间划成(0,1),(2,3)…，偶数为reference，奇数为target。两个模型native时间支持不同，必须记录来源帧，不混原始分数或声称tubelet等于单帧。
2. 信号取**位置加入前**的真实patch embedding：V2 patch_embed输出；CLIP conv1输出。bridge提供真实`[T,H,W,D]`及坐标，不从非线性hidden减位置向量冒充内容。对每个reference/target位置p计算`d_p=0.5*||normalize(e_target,p)-normalize(e_ref,p)||²`。完全相同embedding产生精确0；zero/nonfinite状态显式记录。
3. 每个水平相邻patch对为一个pair，两位置d取均值。14×14共98pair，按native顺序分成19组各5pair、尾组3pair。**target每完整组保留3pair，尾组保留2pair**，较高变化者优先；reference保留全部98pair。完整组uniform成员为局部pair索引[0,2,4]、tie优先序[0,2,4,1,3]；尾组uniform为[0,2]、tie优先序[0,2,1]。按score降序稳定排序，完全相同score时退化为U，不设可调阈值。
4. 分数在embedding时已可得，只保存小型分数/索引。实际gather在**block2输入**，即原生blocks0–1只运行一次后；gather当前hidden，位置内容及真实来源索引一起保留。不开完整dense教师来决定mask。

高变化不等于异常，低变化不等于无用。相机运动可能抬高全局分数，重复纹理也影响排序；输入motion/brightness及低运动正视频的结果必须报告，不能把视频级标签当异常token真值。

## CLIP单pass执行

16帧作为batch运行patch/position/ln_pre和blocks0–1一次；随后拆成两个**互不重叠的帧子batch**：8个reference保留197 tokens，8个target保留119 tokens，各自调用原生suffix blocks。每帧只沿自身一条路径计算，不重算prefix。最后用原生ln_post(CLS)@proj，按原帧顺序拼回16×512并在clip内平均。

不能将119-token target padding回197后称节省，也不能合并两个子batch的attention序列。分组增加kernel launch/调度开销，最终要实测净时间；小probe先核每帧前向计数和identity，不用FLOPs代替延迟。

V2在block2将4个reference tubelet各196patch、4个target各118patch按真实(t,h,w)gather为1256长度，继续原生联合attention。各时间存活数不同，禁止reshape成统一二维网格。原生mean readout未改，但reference来源token在fc_norm之前的均值权重为784/1256≈0.624204，target为472/1256≈0.375796；第二阶段后仍是392/628与236/628。由于联合attention已经混合上下文，这不是原始reference帧的因果信息贡献比例。T/U/R计数完全相同，不悄悄加新的pooling修复。

CLIP的最终16帧readout仍各占1/16，reference和target各占1/2；没有将reference token/feature输送给target attention，只用其embedding算mask。T单阶段中，reference帧原生前向完全未改，其每帧512维读出应与dense在数值噪声内一致；target帧读出可变，必须分别报告，不能只用clip均值掩盖误差抵消。V2没有这一reference不变性质，因其joint attention会受到target删除影响。N启用后CLIP reference也会被后层压缩，不再要求它与dense一致。

## N阶段：F07通过后才启用

完整block5后，在block6输入前，对当前仍完整的每个水平pair按当前hidden L2范数选择一员（tie原生左员），CLS保留。成员随native时间可以不同，但不引用已被T删掉的token。索引由current-live→original映射生成，不复用原dense flat索引。

这是F07成员偏好机制，0.50条件预算不同于cohort100的0.60组预算，所以F07通过也不能替代组合实测。F07无可靠增量或到截止未完成，N不派GPU；不复活F06、不新增网络。

## 数量与公平因子表

| 路径 | V2序列长度 | CLIP reference帧 | CLIP target帧 |
|---|---:|---:|---:|
| dense/完整identity | 1568 | 197 | 197 |
| T/U/R单阶段的blocks2–11 | 1256 | 197 | 119 |
| 两阶段的blocks2–5 | 1256 | 197 | 119 |
| 两阶段的blocks6–11 | 628 | 99 | 60 |

第一阶段patch比率`(196+118)/(2*196)=0.8010204`；第二阶段每个存活pair留一员，条件比率0.5；最终0.4005102。乘积仅指同一patch域，CLS另算。CLIP记录每帧长度向量，不以平均长度代替二次attention成本。

第一轮仅T/U/R单阶段，同层、同target quota：T为真实时间分数，U为局部uniform，R为同组固定seed随机。三者reference都全保留，均走相同CLIP子batch调度。

R的每个native reference/target对使用预存的组内随机成员，seed=20260920加native时间对索引；随机mask不读视频ID或标签，跨clip固定。保存具体索引而非仅seed。所有最终gather索引按原生顺序排序，避免F06的组遍历顺序偏离。

若N获准，再跑U/U、T/U、U/N、T/N、R/R：第一字母为target选择，第二字母为后层成员选择；U成员为固定左员，N为max-norm，R为固定seed成员。五臂逐层计数完全一致。“仅时间/仅性质”明确指只启用一个**信号**，另一阶段仍中立缩减匹配计算；纯单阶段T另列，不能拿最终80%对最终40%宣称组合增益。

## 4+4训练视频

每数据集4正常+4含异常视频，每视频2个固定中心窗口，每窗同16原帧，两encoder共用manifest。UCF使用允许fit成员；XD仅accepted3950并按来源/电影去重。已有smoke可复用，新增ID先按固定seed及角色/来源规则选定，不按目标分数挑样本。不占confirm角色，不访问test。这里4+4明确为每集8视频。

各arm记录input/weights/source SHA、dtype、embedding分数、stage indices/原坐标、逐层shape、pooled小向量、dense重复/identity。主响应为pooled绝对MSE/cosine，合法固定后端可用时另列训练侧logit漂移；不训练新头，不用test选预算。

视频等权报告正常/含异常组，保留全部反向结果。原16帧motion/brightness用于探索匹配/敏感性，4+4不足以确认场景/相机控制。低运动正视频不是已标注静态异常clip。重复帧fixture核score=0、tie→uniform及reference全保留；只验实现，不冒充真实性质。

## 截止与止损（root按资源最终调度）

- **9月20日22:00：**冻结ID/配置，完成CPU预算、静态tie、live索引和CLIP域隔离测试；如必须合并attention域、双跑prefix或padding回dense才跑通，NoGo实现。
- **23:30：**完成T/U/R双集小表。T无同时优于U/R的稳定线索，或任一模型/标签组明显反向，停止扩展；小样本不证明总体无效，但不足晋级。不换层、阈值、score或加运动网络。
- **9月21日01:00：**仅T与F07均有增量才完成两阶段因子小表。组合不优于两种单信号臂，或增益来自不同路径/预算，停止组合，不硬保留两层。
- **08:00最终Go/NoGo：**缺关键证据则NoGo新主方法，保留已审dense/性质/负结果写作。Go不能把4+4当独立确认；正式test前须完成方法/格子冻结、输入/后端匹配和必要质量证据。不同压缩率正式测试放在Go后冻结格子，不在本probe扫比例选赢家。

F07无可靠增量时N直接NoGo；T只有通用收益时作时间冗余基线，不称异常性质创新。论文可同步写已核数据/协议/负结果/方法边界，数字只来自审核exports。

## Luna任务与验收

一个worker负责`work/codex-takeover-20260920/idea/temporal_reference_pair.py`、`run_temporal_reference_probe.py`及小测，其它人不改；复用S1加载、原视频、readout与receipt，不改公共协议、旧snapshot或FeatureStore。

纯规则函数接收已验证preposition embedding和pair表，按相邻native时间返回reference完整mask、target59pair mask；不接labels/文件名。V2用instance级block2 hook和真实live坐标；CLIP明确调用原生模块的prefix/suffix并分reference/target子batch。完整identity须与encode_image对齐，不暗改attention。

最小接口可固定为`temporal_pair_mask(embedding_thwd, mode, seed)->keep_mask_thw, pair_scores`，mode只为T/U/R，真实坐标和source时间支持由bridge绑定；输出mask的reference行全true、target行每组满足quota。runner按该mask维护live-original-index；不把mask直接乘hidden冒充删token。首批logical batch=1 clip，CLIP的B维明确对应16帧，避免多个clip混在同一时间评分域。

N获准后再加block6操作：V2从各时间的实际pair映射gather，CLIP两个子batch分别做本帧内batched gather。每stage保存当前长度和来源索引。公共CLIP bridge当前只支持一次静态gather，不能关闭其长度保护来硬套第二阶段；实验runner持有明确阶段状态。

验收：每帧/clip一次prefix、原生attention域/CLS不变；计数与表一致；真实位置保留；clip间reset；只用当前/早期状态；选择开销与两个子batch调度单列。完整identity及T单阶段CLIP reference输出沿用S1 float32 max-abs 1e-5门槛并记录dense重复；分组schedule本身若不过，先调查，不自动放宽阈值。GPU由infra统一分配node2/3，此文件不自动开工。
