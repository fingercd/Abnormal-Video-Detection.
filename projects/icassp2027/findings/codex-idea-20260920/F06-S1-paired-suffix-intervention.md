# F06-S1：当前层信号与后缀压缩扰动的配对小实验

2026-09-20，配置登记。当前状态：`specified_pending_root_dispatch`。先运行已有工程样本的配对干预，不默认扩大样本、预算或层数。不是论文方法冻结，不访问官方test。

测试暴露披露：项目此前已经接触旧test290诊断，V3现场表也保留这些历史数字。本次新配置登记发生在该事实之后；本研究问题取自旧训练侧F01与当前原视频工程回执，不使用上述test分数确定模型、层、预算、方向或停止条件。不能写成全流程从未查看test。

## 为什么现在做这一项

V2与CLIP已有真实原视频hook和grid工程回执，足以推进“同输入、同预算、只改后缀token”的干预。它回答当前最关键缺口：F01全局更新比值是否有实际计算决策价值，而不是继续从局部相似性直接推出selector。

真实V2回执的8窗口均与共享manifest的16帧索引完全一致，grid为8×14×14、1568patch、无CLS，hook输出差为0；U/X标量可由导出的U、X范数复算，最大绝对误差约1.4e-8。比值范围约0.25190–0.26096。这只是4个fit视频的工程观察，不继承F01独立确认。

当前V2跑在classic-video-v2（torch2.3）与code-e30c3ab-v4，主控另核到transformers4.37.2及临时easydict；生产V2使用foundation/overlay。当前JSON未绑定checkpoint SHA，也未记录transformers/easydict字段；需要下一run绑定或独立补充回执，不能说生产特征等价已证明。不要改写旧JSON。可先在明确独立研究身份下做同环境成对干预；扩展前优先用生产loader完成一个共享clip的dense及X/U对照。若生产loader不能复用，不重装环境，把等价状态保留未验证。

CLIP UCF/XD float32回执验证197→99、投影维512，identity最大差分别8.583e-6/6.914e-6。现有F06数值是一个视频的两个8帧窗口合并为一批后计算的单个比值；不是每个共享16帧clip一条。其`shared_with_v2_probe=true`只可解释为工程中心规则相同，不能当作源帧完全相同。旧half抖动不进入本次成功样本或数值门槛。

## 第一批固定输入

- UCF：既有4个fit视频和`probe_shared_window_manifest.json`的8窗口，逐行使用已冻结的16个frame indices，不独立重采样。
- XD：沿用`xd-smoke-ids.json`的4个官方train视频；先生成新共享16帧manifest（stride2、中心0.25/0.75、与UCF同一采样公式），V2/CLIP消费同一文件。旧8帧CLIP输出不冒充新输入。
- 共8个视频、16个窗口、两个encoder。每个窗口单独输出一条信号/干预记录；不把同视频两窗口拼成一个clip。
- 不换坏源、不访问test、不消费confirm。四个XD成员都已能解码，不以另4个坏源修复阻断本实验。
- CLIP固定OpenAI center预处理、ViT-B/16 float32，逐帧`ln_post(CLS)@proj`后在当前16帧内算术均值作为clip读出；不做额外L2归一化，不与VadCLIP10crop预提特征/已训练head混用。
- V2固定当前已核checkpoint/原生预处理，明确真实dtype和读取路径；每个encoder内部所有arm复用同一已预处理输入。两encoder预处理并不宣称像素张量完全相同。

## 信号与干预必须按因果顺序执行

1. block5 pre-hook读取完整patch的X（第一LN之前）；block5 attention模块hook读取U。
2. 完成block5全部attention、MLP和residual。
3. 在block6输入处gather保留token，后续blocks6–11都接收短序列。
4. 原生readout输出pooled。只在离线分析中与dense参考比较。

V2可用`indexed_gather(depth=5)`的block5-output hook；CLIP现有bridge是block-input编辑，因此应为`capture_depth=6, keep_indices=...`，另在block5装独立只读X/U hook。不能在CLIP`capture_depth=5`之前拿同层尚未算出的U决定压缩，也不能把block6的U误叫F01。

当前两种中立选择均不依赖s。每个reduced forward仍需自行采到block5 X/U，并与同输入dense的prefix统计核对，证明将来信号能在同一次执行中到达决策点；部署不允许先完整dense再重新跑compressed。离线dense参考多跑仅是研究测量。

CLIP每clip的U/X在16帧全部有效patch上先汇总平方和再开根相除，排除每帧CLS；不要平均逐帧ratio。V2按1568patch计算。额外记录U_rms、X_rms、各帧/片段真实维度、gamma/LayerScale所在位置。

## 固定arm与预算

| arm | 操作 | V2 suffix tokens | CLIP每帧suffix tokens |
|---|---|---:|---:|
| dense_a / dense_b | 同输入重复dense，记录数值噪声 | 1568 | 197 |
| identity | block6前按完整真实索引gather | 1568 | 197 |
| fixed_local_half | 每2×2空间块按行优先保留前2格（上排），全部时间使用同空间mask | 784 | 99（98patch+CLS） |
| random_local_half | 每2×2块固定seed保留2格；空间mask跨时间共享，与fixed同预算 | 784 | 99（98patch+CLS） |

固定臂复用现有CLIP中立`fixed_local_keep_indices`的前2格定义，不将它与原coverage诊断的对角棋盘混写。随机seed=20260920；一次生成49个空间块的成员索引并保存mask及SHA，两个encoder使用同一2D选择，V2只按其8个tubelet时间重复，CLIP按16帧重复。所有索引按native顺序排序，CLS原样保留，不均值合并、不位置重写。无norm排序、无local-coverage最优化、无多层或预算扫描。

## 最小输出

每个encoder×clip×arm记录：输入manifest SHA/帧索引、预处理tensor SHA或相同输入对象回执、checkpoint与源码SHA、环境、实际dtype、block5 X/U范数与s、真实选择indices SHA、blocks6–11逐层shape、pooled shape、pooled向量（D768或D512的小数组）和失败状态。

主响应`E=mean_d((z_arm−z_dense_a)^2)`，不除以X、U或dense范数。次响应为`1−cos(z_arm,z_dense_a)`；另记dense读出范数。CLIP以每窗口16帧平均后的向量计算主响应，逐帧MSE可作描述，不能替代clip主响应。不训练头、不算帧AP/AUC、不用饱和概率代理敏感性。

两个encoder的原始s尺度和绝对MSE分别分析，不直接比较大小、不混池；只有对应的关系方向与同encoder对照增量可比较。

每视频等权聚合2窗口，固定与随机两臂分别保留；总体敏感性描述可取两臂误差均值，但必须同时给出各臂。4视频/集只形成完整配对表，不拟合显著性结论或报告“确认有效”。

## 验收与停止

| 条件 | 处理 |
|---|---|
| 输入帧/预处理/权重不一致，或者V2 checkpoint身份无法绑定 | 停止比较并修复身份；不拿旧缓存填缺 |
| 信号采自错误站点、在干预后计算或跨两个窗口混合 | 工程失败；不进入关系分析 |
| identity无法复现dense，或后缀未真实收到784/99 | 工程失败，保持原始失败回执 |
| float32 dense重复max-abs或identity max-abs超过原工程1e-5门槛 | 调查实现/精度；不临时放宽阈值继承half噪声 |
| s的样本变化不超过重复运行噪声、或所有干预MSE不超过10×dense重复MSE | 当前量测缺乏可识别敏感性；不自动扩样本或调整预算救结果 |
| 真实配对表完整且扰动高于噪声 | root据表决定一次有界训练探索；不把8视频标为科学确认 |

同时核查prefix信号：压缩在block6才发生，dense和reduced的block5统计应在重复运行误差内一致；否则先排查hook污染/输入漂移，而非解释成压缩影响了过去。

## 只有root明确推进后，才扩大到一次16视频/集探索

建议上限为每集8正常+8含异常、每视频2窗口，共32窗口/集；两个encoder共用来源和输入，仍只一个层、一个预算、两种中立arm。既有样本可纳入fit探索，额外成员按既有roles和source groups预先冻结；XD同电影片段不能跨角色，UCF来源未知须披露，不以ID不同断言source独立。不自动使用64确认池。

这批数据只比较`log(s)`和`−log(X_rms)`两个等参数单变量模型预测绝对MSE的能力，并报告常数预测器；按视频留一（LOVO），同视频所有窗口一起留出，模型/标准化仅拟合其余视频。正方向假设预先固定，不按结果翻转规则。可加入同样的motion/brightness协变量做对称敏感性比较，不将log(s)、log(U)、log(X)共同回归后声称独立系数。标签仅用于正常—含异常分组与视频级汇总，不进入部署决策。

若s不优于norm-only/常数预测，或者关系在两个encoder/数据集间反向，停止晋级；区间宽就记不确定，不自动开更大样本。只有观察与干预增量同向且可复核，才另立独立确认与同K多重集shuffle-budget实验，届时冻结阈值。当前不实现或宣称自适应预算插件。

## Luna实施责任

一个执行者负责新`work/codex-takeover-20260920/idea/run_f06_suffix_intervention.py`及必要小型测试；复用现有V2加载/geometry/indexed路径和CLIP bridge，不改公共协议、旧SHA或旧回执。脚本必须先打印resolved config，再由infra统一分配node2/3资源执行；不自行新建feeder。首批每encoder有界上限为8视频×2窗口，失败即留回执，不续跑全量。

运行只作诊断，不抢正式独占计时资源。按当前12.66秒V2工程回执只能粗判规模很小，不能据此承诺科学run时长；工程walltime含decode/hook/summary，不进入速度表。先保存首1视频耗时，再报告剩余估计和实际结束回执。
