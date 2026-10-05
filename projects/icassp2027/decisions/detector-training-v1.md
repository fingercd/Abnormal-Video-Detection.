# 检测头与质量比较预算 v1（正式测试前）

该协议固定本轮检测头基线，不冻结尚未验证的压缩方法。所有方法先使用同一个简单头，避免把
更换检测头与token操作收益混在一起。

- 原UCF train1610的完整角色锁保持seed20260918：fit1288用于拟合，select161用于视频级
  验证，confirm161不进入训练。确认性质所用64视频仍属于confirm，不改变角色。
- 训练与验证每视频32个完整原生输入窗口，中心沿全视频均匀分布、frame_stride=2；对齐规则
  复用已验证的uniform_full sampler。推理使用dense窗口和完整帧覆盖，禁止旧segment内不足
  帧数的截短做法。遇真实短视频先显式处理契约，不通过删视频或假padding掩盖。
- 头为现有TopKMIL，`k=3`、单个线性分类器、dropout=0、视频级BCE；encoder冻结。
  它不是时序Transformer，也不接受训练事件区间。输入维数使用真实encoder pooled维度。
- 初始统一预算为20 epochs、batch16、AdamW、learning_rate=0.001、weight_decay=0。
  预先固定使用最后一个epoch，不从官方测试挑epoch或seed。若允许的select视频级损失显示
  明确工程异常/未收敛，须在接触测试前统一修订并记录，不能只增加某方法预算。
- 主seed为0，关键配置补seed1/2。各方法使用相同数据、头、训练预算及种子；缓存可共享的
  仅是相同数据/采样/表征身份的真实特征，不伪造fingerprint。
- `direct_insert`使用不可改写的dense头；`refit_head`在各方法自己的特征上按相同预算重训。
  插件是否有训练与检测头训练分开报告。小预测器的训练预算待操作证据后另行冻结。
- 每次训练必须核对fit/source身份、loss有限、梯度有限且有真实参数更新、checkpoint回读、
  样本计数和推理帧覆盖。历史工程小样本成功不能代替本次真实回执。
- 官方测试仅在方法/预算冻结及数据审计通过后使用；UCF主指标frame ROC-AUC、XD官方定义
  为PR曲线梯形面积。相对dense质量容忍度0.005，报告点差、成对视频bootstrap区间与seed差异。
  点估计达到要求但区间不足时，不写成已证明保质。没有测试指标时保持null。

现有整体controller的engineering模式附带独立预测覆盖检查，可使用原confirm中一个按固定
哈希选择的工程canary；仅核对输入身份和覆盖，不读取其分数/标签关系来选择方法或检测头。
完整fit训练本身的身份与该工程预测目标分开，源checkpoint不会绑定未来正式评测样本。
原官方train派生的validation视图须明确保留原始split=train及select/工程角色，不能冒充官方val。
