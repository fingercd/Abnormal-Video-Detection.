# VideoMAEv2 关系压缩正式测试冻结（2026-10-13）

用户要求“开始测试吧，用gpu0123每个gpu一个任务，依次完成”。本轮仅测试已通过真实QA的VideoMAEv2。
新轮次不改写旧六格冻结文件；没有异常特有机制主张，不要求额外性质筛选。

UCF、XD各：dense identity，attention_topk / visionzip_internal / uniform × block 6/9/12 × keep0.50/0.30，
共38个完整数据集任务。每条件只在attention residual后、norm2/MLP前压一次。
VisionZip输出dominant/contextual固定4:1、原参考target+assigned-mean公式，最终原生pooling不改。
不做动态/静态分组、不以L2幅值排序，不根据test改变方法、层、比例、head、seed或checkpoint。

每数据集独立复用已验收的full-train UR-DMU dense s0 final.pt（3000 steps，64normal+64anomaly/step，200训练bins），
本轮不重训head。测试保留完整native dense序列，clip_frames16/frame_stride2/window_stride16/short_policy stride1_if_needed；
UR-DMU使用已验证的exact all-key query chunk=256，不变成200-bin测试或局部序列推理。
训练/测试身份、权重、实际preprocessing/readout、precision和采样的原有严格检查继续有效；只允许reducer维度变化。

UCF完整290视频，frame ROC-AUC；XD完整800视频，沿用原raw→canonical作者prefix16Ti真值口径，
梯形PR-AUC主列，step average precision另列。复用原审计和真值SHA，不修改时间轴。
Delta=method-dense，10000次视频弱标签分层配对bootstrap、95% CI；下界>=-0.005才支持质量非劣。
全部配置预先冻结并完整报告，不以正式测试分数选择最优配置继续调参。

GPU0/1/2/3每卡最多一个任务，完成提取和评分后接下一项。GPU0共享原PID1827服务，
遵循既有共享授权，不停止/修改其进程；单任务CUDA allocator上限0.25（约10GiB），至少留2GiB余量。
GPU1/2/3启动时应空闲，出现未知进程时不发新任务；不换到其他GPU。
共享条件只作质量测试，不把这些运行耗时或监控数值作为正式独占效率结果。

所有任务先初始化SwanLab在线run，再启动远端nohup任务；本机控制器持续上传EVENT及日志，
任务完成回读云端状态/完成标量。原始失败目录和cloud CRASHED保留，失败时停止新派、不盲目重试。
每任务独立FeatureStore、输出和PID，不并发写同一index。不覆盖/删除旧视频、权重、历史回执。

QA证据是2026-10-13训练侧两视频工程验证（18配置、36 B2前向、6identity、真实1568→784/470），
只能作为实现与预算验收，不冒充异常性质证据、检测质量或正式速度结果。
