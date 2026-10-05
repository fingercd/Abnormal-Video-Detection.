# F06｜中层attention相对更新是否预测压缩敏感性

日期：2026-09-20。状态：`engineering_probe_authorized_not_property_confirmed`。这是主控批准的下一优先可操作性问题，尚不是方法。当前只给已有2正常+2含异常视频工程probe增加固定站点统计，不默认扩到32/64视频，不新增GPU生产线。

核心问题：F01在V2确认过的clip级更新比值能否预测同输入压缩的后缀扰动，还是仅为输入范数相关的分组现象？逐token排序已有负结果，本次不再假定它定位异常token。

## 真实F01绑定

原冻结文件为`projects/icassp2027/decisions/property-confirmation-v1.json`：`relative_attention_update`，相对深度0.5；V2固定layer_index=5，site=`block.5.attn.output`，P16=`attention_update_to_input_norm_ratio`。旧确认数据为UCF官方训练、独立64视频，每视频8固定窗口；该独立性和结论不能转移到本轮新输入或CLIP。

原采集代码`src/vadbench/research/collectors.py`计算有效patch上的全局Frobenius比值：

`s = ||U_attention[patch]||F / ||X_before_attention[patch]||F`。

X为原block输入、第一LayerNorm之前；U为attention模块返回的输出（含投影，按实际模块定义）。不是LN之后的X，不是block输出减输入，不是逐token ratio的均值。若模型有外置LayerScale/gamma/DropPath，原U是否包含它由真实路径确定并写回执，不悄悄重定义旧签名。eval模式下仍须记录路径。

CLIP ViT-B/16同为12层，预声明取resblock index5；X为block输入，U为MultiheadAttention返回tensor，经[N,B,D]→[B,N,D]转换。只在patch上计算，排除CLS；CLS输出是否变化作为后缀读出的一部分。这个对应关系是相对深度/模块语义对应，不代表两个模型训练目标或输入时间覆盖等价。coverage探针仍在block2输入，不拿它代替F01中层。

CLIP原生处理单帧，若一个固定clip含F帧，clip比值在全部F×196 patch上汇总平方范数后相除，不以某中心帧或帧比值均值替代；逐帧比值可单独描述。只跑一帧的smoke明确为frame-level工程，不声称完成clip级对应。clip级部署需要在固定输入clip内获得全部帧的当前层状态，不依赖跨clip未来帧；实际分批/缓冲开销须计入。

为核对公式，同时记录`U_rms=||U||F/sqrt(Npatch*D)`、`X_rms`和ratio，以及Npatch/D、dtype、site、模型/权重指纹、输入帧索引和preprocess身份。零范数或非有限输入显式记unavailable，不改统计定义。

## 最小工程步骤

1. 复用已排队的4视频×2固定窗口，沿同一次dense forward采上述三个数。hook开关下最终输出数值等价，站点shape和触发次数正确。该样本不检验正常—含异常差异。
2. 在上述通过后，复用同输入干预接口，固定在block5输出后做静态真实索引选择；原block5的attention和MLP均已付费，仅后缀block6–11缩短。先只验dense/identity/一个固定50%覆盖选择的后缀形状与输出，选择索引独立于s；这一步不把50%锁为方法预算。
3. 工程回执后由主控决定是否值得进入限定训练探索。没有真实原视频CLIP路径、当前probe不具备精确站点、或无法使用同输入运行时，不扩样本、不用预提特征伪装完成。

## 敏感性目标和机械相关防护

主敏感性目标为最终原生pooled读出的每维绝对平方误差`mean_d((z_compressed−z_dense)^2)`；不除以X、U或s中任何逐样本分母。余弦漂移作次要描述，dense输出范数独立记录；若使用固定head，再记未阈值化logit绝对变化，不能只看饱和概率。不同encoder的绝对误差不混池。

不会宣称ratio在完整(U,X)之上包含额外信息：它是两者的确定函数。需要检验的是ratio是否比仅输入norm、仅分支norm或仅输入motion/brightness更实用地预测后缀敏感性。训练探索允许分别拟合这些简单预测关系，独立确认冻结模型、方向与误差口径；不得同时把log(s)、log(U)、log(X)放进回归再声称独立系数，因为三者精确线性相关。

正常—含异常比较继续以视频为独立单位、同源隔离、匹配输入统计，不能把clip标签视为事件真值。s与误差的关系也按视频聚合/视频bootstrap，不用海量clip制造显著性。若ratio关联仅由input norm解释，不将F01包装成新的计算决策。

## 若关系存在，怎样检验一个实际决策

仅作为未来有条件的干预定义：当前层s较高的clip给较多后缀token预算，较低者较少；同一个固定覆盖选择器，不增加token评分网络。阈值与两档预算只用允许fit数据冻结。当前不执行预算网格，不按测试表现找阈值。

最关键对照是在同批clip上打乱s对应的预算分配，并保持完全相同的实际K多重集、特殊token处理、同一suffix/选择器。这样总token数和N²项理论计算均严格相同；只匹配平均保留率不足以匹配attention计算。部署阈值从fit冻结，不在测试批次排序或读取汇总统计。变长/按长度分组的调度开销单独实测，不能以padding后的dense执行冒充节省。

dense未来输出只作离线训练侧参考，不是部署输入。真正节省来自已观测层之后更短的attention/MLP，当前层的attention计算不能算作已节省。

## 与近作的关系和停止条件

[KeepAD](https://arxiv.org/abs/2608.03681)含局部覆盖和异常原型驱动选择，[CenterCLIP](https://arxiv.org/abs/2205.00823)使用代表token聚类，[S²Prune](https://arxiv.org/abs/2609.01224)按局部结构分配空间预算。本问题关注无需异常分数的全局更新统计能否标记真实压缩敏感性，决策单位是clip级后缀计算量，而非以局部高风险分数给token排序。上述区别只是研究问题不同；动态计算和更新范数已有相关研究，尚不能宣称机制独创。

预期方向为s更高对应固定压缩下更大输出扰动，因而较多后缀预算应分给高s输入；这是待检验假设，不是已知异常性质。两模型同为负关系也不自动授权反转规则。

停止或不晋级：s相对input norm单独没有预测增量；V2/CLIP关系反向；两数据集关系不稳；同实际K多重集下按s分配不胜打乱；或者测得开销抵消节省。置信区间不足时记不确定，不反选层、方向或新预算挽救。没有增量就保持“新机制未成立”，不预设异常token易损。
