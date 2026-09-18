# ICASSP 2027 当前进度

更新日期：2026-09-18。当前状态：**完整研究 goal 进行中；仅有 VideoMAEv2 特定性质的独立确认，尚无跨四 encoder 的通用规律或插件有效性结论。**

## 最新执行回执（UTC 2026-09-18 04 时）

- 正式UCF评测、同seed head重复训练/缓存评分、视频配对质量导出和完整视频detector计时入口
  已实现。统一集成 **97 passed / 240.18秒**，本机Python3.11、Torch2.13.0+cpu、NumPy2.4.6；
  Ruff、compileall通过。来源核验绑定独立fit1288/select161、真实checkpoint内部metadata、
  梯度/参数更新/reload QA以及sealed test；repeat评分只复用同表征/采样的完整test缓存，
  secondary使用同seed dense head。还没有正式模型评分结果。
- node3的VJ dense、V2 global_uniform、VJ global_uniform分别在232、726、20个完整训练
  视频后因外部GPU争用被guard停止。旧输出与中断回执保留；没有停止其他用户的进程。
  新续跑接口只复制严格核验的完整视频，partial整视频重算，新run重新训练固定预算的head。
  source数据/采样/模型/校准/精度及数组SHA全部核验，原resolved、索引和数组来源均有记录。
  非identity还与当前真实bridge/layout/indices逐项核对完整执行回执，且保持原microbatch=8。
  完整resume回归 **22 passed / 99.16秒**，独立审计复现的回执篡改已被拒绝；真实服务器续跑待验收。
- 已定位长dense视频的索引瓶颈：旧FeatureStore每clip会重读并重写整个视频索引。
  paper extractor现用每64 clips一个原FeatureStore分片，完整视频只合并发布一次；不改核心
  FeatureStore格式、模型输出或缓存语义指纹。这是提取工程优化，不计作插件推理加速。
- V2 dense的fit/validation已完整完成；最后精确canary快照为2097/3350 clips，head尚未开始。
  canary为53606帧的训练派生视频，耗时不能误报为head训练缓慢。其他运行状态继续以guard
  和逐阶段回执核对，不将未最终汇总的失败数写成已验证零失败。
- 四模型正式clip计时已准备在同一node2 GPU3顺序执行。修正了V2原生backbone实际返回
  `[B,D]`的接口假设，保留真实native pooled readout；新冻结源`dc61652`，guard16503。
  旧guard65292仍在waiting时已安全停止，原基线未动；尚无正式计时数字。
- XD当前raw训练metadata为ready2800、ordinary pending1150、source-corrupt4；四坏成员全部
  保留原fit角色并隔离。已另行冻结名称组隔离视图1598身份：fit1267/confirm167/select164，
  当前available830/pending768；不按可用性重选，不称为完整官方train3954。保留一个真实34帧
  fit视频，不静默padding或删除。XD第二数据集验证范围为V2与Time；UCF四模型主矩阵不变，
  见[XD范围决定](decisions/xd-validation-scope-v1.md)。raw test坐标门禁仍待完整原视频。
- XD测试ZIP采用严格4路分块恢复，已验证前缀加分块约4.28GB/10.80GB；未发布ready。
  新恢复程序用流式SHA和打开ZIP期间的逐成员CRC校验，损坏ZIP/路径越界合成测试已通过。
  先前控制文件缺失判断已纠正：实际文件全部保留，旧进程是网络截断退出，新程序继续使用
  已验证分块。一次公网/备用线路中断后已恢复连接，没有据失联推定服务器任务成功或失败。

## 此前执行回执（UTC 2026-09-18 02 时）

- 元数据修复已冻结为 `a710ce4` 并部署。TimeSformer 在 node2 V100 实际完成 identity、
  global_uniform、paired_random、pair_linear 各两个 fit 视频 ×32 clips 的写入/读取，均64
  records、runtime reference SHA 与 paper identity 一致；每路径两步 TopKMIL 的非零梯度、
  参数更新、磁盘重载完全一致 QA 通过。dense representation 与旧 `96e4631` 完全相同。
  成功 control 为 `timesformer-roundtrip-a710ce4-20260918T141500Z`，最初单视频不满足
  detector 训练契约的 `140000Z` 失败仍保留，不能混为成功。
- Time 完整基线已在 node2 GPU1 重启（guard106181 / child106211）。12个后续方法任务
  已启动跨节点独立协调器，首个 V2 global_uniform 在 node3 GPU4 实际运行。协调器仅在
  无 compute PID、空闲显存检查和共享 GPU lease 通过时领取任务；同一任务原子领取一次。
  其他三条既有基线继续，无正式测试分数或 head 完成主张。
- TARGET 四encoder的768步校准与部署 preflight 全部通过，四模型均768步非零有限梯度、
  权重真实变化、主干无梯度、磁盘reload最大误差0。三epoch平均相对pooled MSE：
  V2 0.164965/0.113355/0.098779；Time 0.054734/0.046339/0.044095；
  VMA 0.003742/0.002302/0.002003；VJ 0.014988/0.012835/0.012312。
  这些是fit校准损失，不是检测质量或独立泛化结果。原始receipt归档及摘要位于
  `outputs/icassp2027/analysis/target-calibration-complete-20260918T015500Z/`。
- 同一次采集保留了两台主机各自时钟：node2 UTC01:54:50、node3 UTC01:47:16，约差7分34秒。
  不能用不同主机的 wall-clock timestamp 相减作为运行耗时。最新dense train完整视频分片：
  V2 1236/1288、VMA566/1288、VJ123/1288；Time新run当时仍metadata准备。部分分片单列，
  未终结运行的 failed_count=0 不等同于已验证零失败。完整回执在
  `outputs/icassp2027/control/next-method-queue-a710ce4-20260918/receipts/`。

## 此前执行回执（UTC 2026-09-18 01 时）

- TimeSformer `96e4631` np 基线的 train1288、validation161 均因 FeatureStore 单条元数据
  超过 2048 项上限而写入失败，0 records，未训练成功。完整 runtime 重复嵌入每条记录是
  原因。已核对 owner/PID/命令树，只停止该 child50907；保留原失败运行及外部停止回执。
  下方 00 时的“运行中”是此前快照，不代表该基线成功。其余三条基线仍继续。
- 修复将完整 runtime 保存在每个提取 run 的 `resolved.json`，记录内只保留短摘要及文件 SHA
  引用。读取时核验引用路径、SHA、实际 representation/sampling 内容和身份，不提高元数据
  上限，不改变 encoder 数值、采样、语义指纹或冻结预算。每个提取 run 在首个视频失败后停止，
  controller 在每个角色完成前不会进入下个角色，避免系统性失败继续消耗完整数据集。
  2026-09-18 本地 `.venv/Scripts/python.exe` 定向综合测试 **61 passed / 85.30 秒**，
  `compileall src tests`、Ruff、`git diff --check` 通过；真实 GPU 写入/读取验收及新运行待执行。
- TARGET 原生服务器校准的 V2、Time、VMA 均完成 768 steps，各自三种 reducer 的 B1/B2/
  重复 B2、真实后续层长度、context 清理及 dense 恢复检查通过；V-JEPA2 仍进行中。
  本机校准仅作复现，不能替代 TARGET 权重或根据质量择优挑运行。
- XD 在方法和预算冻结后完成 dataset-only 标签投影审计：预登记的 10 个不同变换中，唯一
  全部 800 个 feature-grid slice 匹配的是 0-based 半开区间、起点0、裁到 `[0,16T_i)`。
  独立实现复核一致。原视频 N/FPS/PTS 与预登记 suffix 族仍待完整 test 原视频，raw mapping
  保持未通过；未读取模型预测或性能分数。

## 此前执行回执（UTC 2026-09-18 00 时）

- 4条完整dense基线实际运行：V2在node2 GPU3、TimeSformer新版np在node2 GPU1、VideoMAE
  新版np在node2 GPU7、V-JEPA2在node3 GPU3。新np开关在真实GPU完成8批B8的像素/pooled
  逐位一致性核验。旧VideoMAE因外部GPU contention被guard中止；旧Time经PID/命令树核对后
  停止并保留superseded回执，未触及其他作业。V-JEPA2微实验64/64，真实输出维数1024，
  batch8峰值allocated约5.21GB/reserved约7.44GB（工程回执，非正式网络速度）。
- 固定fit128、256clips、3epochs的真实gate校准已启动，本机数据执行前完整SHA核验。
  V2完成768/768，三epoch平均relative MSE为0.164982/0.113354/0.098786，768个非零有限
  gate梯度步、主干无梯度、dense/reduced实际读出parity及磁盘reload误差均0；Time/VMA也已
  exit0，V-JEPA2继续运行。所有结果仍是校准QA，不是检测质量。
- 已接通global_uniform、静态seed0 paired_random和pair_linear的原生部署context、严格
  校准加载与同一FeatureStore/controller；不包装adapter、不修改model.forward，保留原生
  backbone身份，reducer单独绑定代码/权重/校准指纹。75项定向集成测试通过，66.58秒；Ruff
  通过。真实固定mask和完整质量/净速度验证仍须完成。
- XD后续两个卷提前EOF造成size mismatch，原part保留；独立clone严格206续传已分别恢复
  train2805和test_videos并出现真实字节增长，原train3320 writer继续。无错误CRC改写或完整
  数据集ready主张。
- 四模型pair gate真实RTX4060工程训练已在冻结 `8c35f62` 顺序完成，均exit0。每模型只训练
  w[D]两步，主干eval/requires_gradFalse且grad全None；identity与zero-gate/mean最大差0，
  真实磁盘checkpoint回读输出最大差0。V-JEPA2 native mean工程readout与该clip的adapter
  pooled最大差0，其他三模型直接用actual adapter pooled。它们不是完整校准或检测质量结论。
  回执：`D:/PythonProject/icassp2027-runs/code-8c35f62/outputs/icassp2027/control/pair-active-local-20260918T000425Z/`。
- 新pairmean fit26操作在同一A100配置下已运行。首两模型的mean−pairedrandom pooled
  relativeL2视频均值差：V2 +0.205829 [0.170041,0.243300]，Time +0.024723 [0.006196,0.042251]。
  因此不把均值冻结为免训练方法，保留其为消融和训练gate的初始状态。候选免训练路线继续
  采用局部随机单边保留；静态seed版本须按自身身份验证，不能与原逐clip seed pilot混称。
- 服务器独立micro定位到Time/VMA旧processor将NumPy列表转Torch tensor的CPU成本。
  同一输入改为processor返回NumPy后from_numpy，像素tensor和pooled逐位相同，batch8构造
  时间约2.0→0.46秒、3.6→1.2秒。新增显式processor_tensor_type，默认pt保持不变，paper
  请求可显式np并进入verified constructor；未改resize/normalize、权重或系统环境。
  相关adapter、controller、extractor和deployment集成52 passed/44.37秒，Ruff通过。
- node2 重新核验后存在独立空闲 V100，已在不同 UUID 租约启动 V2/TimeSformer/VideoMAE
  完整fit1288/select161的dense基线，冻结 `cd39a5d`，各自独立FeatureStore。64clip实测确认
  batch8/FP32可运行，约0.201/0.653/0.812秒每clip（含读取与adapter，不是纯模型正式计时）。
  这些是启动及工程吞吐事实，head/质量仍待真正完成；V-JEPA2全量尚待专用微基准。
- 原neutral26的四模型现均52/52完成。前三模型已审查结果不支持P16/P07高低排序优于局部
  随机保留；V-JEPA2也同方向，完整统计正在归档。均值/线性加权pair原型已实现，核心审查
  修复了真实坐标变化时的缓存误用、padding绕过和FP16零门舍入差异。首次真实训练工程验证
  尚待运行，不能把合成测试当作插件已有效。新pair pilot只四个forward/clip，单例训练QA另行执行。
- 新pair核心、只读CLI和engineer训练CLI定向测试为34 passed；另核心/runtime/index controls
  为39 passed、1 CUDA skip，提取context身份与兼容检查32 passed。数字有重叠，不相加。
- 四模型独立确认现已全部完成。V-JEPA2 run `probe-20260917T213446427749Z-6b75566a`
  为512/512、64/64、exit0；P16 g=0.292 [-0.190,0.808]、P07 g=-0.192 [-0.743,0.278]，
  原始和运动×亮度匹配条件都未通过。两条四encoder共同性质均 `not_confirmed`，仅V2的
  P16通过模型内固定门禁。全部导出绑定同一候选/校准/64视频/审核v3脚本，见
  `outputs/icassp2027/analysis/confirm-four-v3-20260917T234419Z/`，不把4模型计作256独立视频。
- UCF 官方原始视频 1900/1900 下载和成员 CRC 核验已完成；原始 train1610 的 metadata 已全部
  生成。固定角色锁 seed20260918 保持 fit1288、confirm161、select161。原测试严格审计另有
  5 个标注终点超出解码帧数（3 个 +1、2 个 +2），不把下载完成写成 frame metric audit 通过。
  依据作者官方 evaluator 的 source-end clamp，已选定明确的 decoded-frame 派生协议，详见
  [边界决策](decisions/ucf-source-end-reconciliation.md)；原始标注与严格失败回执继续保留。
- 完整 train 中存在 2 个长度小于 127、但大于 64 的视频。V-JEPA2 的 64 帧 stride2 无法容纳，
  正在实现显式 `stride1_if_needed`，默认 strict 不变；baseline/插件共享政策并绑定采样指纹，
  不丢视频、不复制帧。其他三个模型的本轮训练窗口不受该边界影响。
- neutral26 已完成 VideoMAEv2、TimeSformer 各 52 clips。TimeSformer 初次在模型移入 CUDA 时
  OOM，失败原样保留；同配置先初始化 CUDA context 后复跑 52/52 成功。无法据此确定初次
  失败的唯一原因。VideoMAE/V-JEPA2 的剩余固定 pilot 顺序执行，尚未完成的统计不汇总。
- 主确认 V-JEPA2 仍在本机真实 CUDA 采集，最近读数 463/512 clips；其结果尚未进入规律结论。
  已完成三模型的结果继续使用冻结候选和审核后的 v3 统计脚本，不更换方向、层或签名。
- 完整 dense 头训练协议已固定于 [训练预算](decisions/detector-training-v1.md)：TopKMIL k3，
  20 epochs、batch16、AdamW 0.001、weight decay 0，最后 epoch、seed0；关键结果补1/2。
  正在准备完整提特征与训练 QA。中立 pilot 每 clip 包含多个重复前向，其耗时不能直接当作
  单次 dense 抽特征速度或净插件收益。
- 本轮数据边界、短视频采样与训练 QA 集成验证：79 passed，70.67 秒，本地
  `.venv/Scripts/python.exe`。训练 QA 默认不改变旧 runner；paper 训练显式开启，失败的
  checkpoint 标记 `trained_pending_qa`，只有非零有限梯度、实际参数更新和回读 logits
  一致后发布 completed 状态。controller 已贯通 train/validation/evaluation 的同一 short policy。
- XD 第一卷 1004 个有效，第二卷 996 个通过、4 个 CRC 失败隔离；卷级保持
  `partial_with_crc_failures`。后续卷继续下载核验。官方 feature-grid 的 800 个 Ti header
  均已直接验证并冻结，但 raw-video 到官方坐标的严格映射尚未成立，没有正式 XD AP。

## 最近核验（UTC 2026-09-17 20:45）

- A100单clip真实adapter中立操作已覆盖四模型，两种分数各自的identity/均匀/随机/高分/低分
  路径均执行成功，真实后缀长度减半；identity误差0或VideoMAEv2的3.58e-7，低于固定容差。
  单例中全局高低分排序未优于均匀/随机的pooled扰动，不作VAD质量推断。
- 下一步已固定 [26视频中立pilot](decisions/neutral-intervention-v1.md)，服务器独立根已26/26
  大小与SHA核验。新增局部相邻pair覆盖对照，主索引始终int64，另有8192以上索引+fp16 score
  回归以避免舍入。完整CLI合成测试实际经过26视频/52clip采样、role lock与失败记录路径。
  最后集成41 passed、1 CUDA-only skip；compileall/Ruff通过。
- VideoMAEv2配置的`use_half=true`不能用来推断实际精度：当前实现的真实参数与观察输出都是
  float32，已查源码和运行回执确认。本轮没有改变该既有forward行为，所有比较记录实际dtype。

- 三个encoder的独立确认已完成并使用经独立审查的v3脚本分析。P16仅VideoMAEv2在当前
  32/32确认及matched条件下复现，g=0.780 [0.327,1.282]；TimeSformer和VideoMAE未通过。
  P07也未通过预定的原始+matched联合条件。两条“四encoder共同性质”主张尚不能得到本轮证据支持，
  负结果记录于 [F01](decisions/findings/F01-relative-attention-update.md) 和
  [F02](decisions/findings/F02-midlayer-temporal-change.md)，不改方向/层挽救结果。
- V-JEPA2主确认在任何本模型确认效果被读取前改为已有完整数据的本机CUDA：服务器上传实测
  约410KB/s，剩余粗估108.6分钟。原等待数据的server PID31073已核对命令/状态后取消，
  尚未执行GPU，上传继续。当前本机run为`probe-20260917T213446427749Z-6b75566a`，
  仍在采集，不能把其未完成项计入确认。冻结候选内容没有变化。

- VideoMAEv2确认采集512/512完成，run `probe-20260917T205806968233Z-e21841be`。
  TimeSformer正在同一冻结64视频上采集；尚未读取确认效应来改变候选。
  V-JEPA2已排入服务器“64个final内容校验→空闲GPU租约→确认采集”的依赖队列。
- 跨encoder输入控制校准只用64个fit正常视频，严格区分8/16/64帧；有效资产为
  `outputs/icassp2027/control/control-calibration-cross-encoder-v2-20260917T213000Z/`。
  先前统一16帧且误用旧sampler的资产已标superseded，未用于确认。更正路径与原16帧探索的
  512个正常clip控制量完全一致。确认统计脚本尚在独立审查，未经审查不输出确认结论。
- XD新增仅训练视频弱标签的专用导入，完整3954训练与800测试身份分离；固定name-derived
  来源分组、seed202709，不随下载可用性改变角色。官方feature-grid PR梯形面积实现已有
  合成验证，但拒绝provisional Ti metadata；800个直接NPY header核验与原视频坐标映射
  尚未全部解决，不能写成XD原视频正式评测已接通。
- XD第一卷1004/1004原视频已在node2完成流式解压、CRC、大小与SHA核验，11,924,807,590
  字节、0part。其余卷下载/验证继续，非完整XD完成。
- 新XD导入/feature-grid数学门禁/中立固定预算索引工具，本次集成检查为54 passed、1 skipped
  （默认CPU解释器无CUDA，GPU-score/CPU-layout测试跳过）；compileall src tests、Ruff与
  diff检查通过。中立工具不是最终插件，无检测质量或净加速主张。

- 完整128探索数据已通过union gate。UTC20:55冻结两个中层候选（P16分支相对更新为正、
  P07同坐标时间变化为负），详见 [确认定义](decisions/property-confirmation-v1.md)。
  探索发现经过视频级分箱bootstrap复核，但不称为确认规律。原between-bin CI保留并纠正解释。
- 四模型已具备实际序列缩短与冻结主干内部参数梯度的工程证据：最新GPU实权重
  TimeSformer为1569→785，VideoMAE为1568→784，V-JEPA2在index11后8192→4096；
  三者identity最大差0且leaf梯度有限非零。V-JEPA2较早index3测试在8GB/70%上限下OOM，
  原失败保留；index11是单独运行。VideoMAEv2此前CPU实权重1568→784证据保持原身份。
- node3显式进程级`torch.backends.cudnn.enabled=False`后，原V-JEPA2真权重GPU验证通过，
  没有重装环境或换权重。服务器正式dense/plugin比较将固定同一backend，不和本机计时混合。

- 主探索 CUDA 两批均退出 0：`probe-20260917T193541117630Z-ef589e03` 完成 127×8，
  原唯一缺项 Normal_Videos533 的补充批完成 1×8。冻结完整候选为正常/异常各64视频；
  正在审核 union、采样和环境身份，再做视频级统计。CPU复测单独保存，不能增加独立样本数。
- 本地真实权重 GPU observer/identity 已覆盖四 encoder，features 与 pooled 最大差值全部0。
  最新 TimeSformer、VideoMAE、V-JEPA2 run 分别为
  `observer-validation-20260917T204241101526Z-5367fe39`、
  `observer-validation-20260917T204306840733Z-9e72f1a4`、
  `observer-validation-20260917T204341966395Z-1c81ced0`；V-JEPA2 实际网格32×16×16。
  它们在本地冻结 `4a2a3bb`、torch2.8.0+cu126 overlay 顺序执行，CUDA allocator 上限70%，
  未更改原profile精度或已有CPU环境。这不是检测质量或压缩加速结论。
- 服务器 V-JEPA2 CPU verification、旧VideoMAEv2 debug64和TimeSformer/VideoMAE debug128
  均完成。旧GPU守候任务曾取得空闲GPU3后退出2，错误原因正在核验，不标成成功。
- node2 官方UCF verified根的最近快照为1172个final、35,837,614,767字节，仍在下载；
  XD第一卷完成并有SHA，后续卷仍在下载，CRC/展开未完成。时间化状态见
  `outputs/icassp2027/runtime-status/20260917T204000Z/status.json`。
- 检测head工程QA已经确认真实非零有限梯度、参数更新、同seed重复训练权重一致，以及3999帧
  完整预测覆盖。5epoch训练loss下降而validation loss上升，应视为小样本过拟合证据，
  不能作为主检测质量或可训练压缩插件的结论。

## 当前持续执行记录

用户要求上传服务器、维护本地 Git，并在完成正常—异常观察、四 encoder 复核、免训练与可训练
简单插件的真实比较后统一交付。未完成研究之前不将工程进展标成研究完成。

- 本地工作分支：`qzt/icassp2027-evidence`，从已验证的 `25f9593` 开始。
- 已将 `25f9593` 经校验后的增量 Git bundle 上传，并建立服务器独立工作区
  `/users/fotile/VAD-icassp2027`。原 `/users/fotile/VAD` 的 156 个受保护源码/配置文件摘要未变。
- 已复用服务器原有 foundation 环境和 VideoMAEv2 overlay，完成一段真实视频的 CPU observer /
  identity 验证，features 与 pooled 最大绝对差均为 0。run：
  `observer-validation-20260917T155850443123Z-7e6ae0e7`。
- 用户已确认主指标容忍度为 **0.5 个百分点**，`protocol.yaml` 记录 `quality_tolerance=0.005`。
  用户离开期间不再提问，实际待答事项集中在
  [questions-for-return.md](decisions/questions-for-return.md)。
- UCF 现有 node2 下载任务 PID 112405 保持运行；最近盘点约 509/1900 个文件完成，总字节约
  16.6%。这是当时进度，不是当前完成证明；依同节点字节变化继续核实，不重复写入同一分块。
- node2/node3 当时均无完全空闲 GPU，node1 已知线路不可达。继续做受限 CPU 工程验证、数据
  准备与代码检查，不占用他人的 GPU、不停止用户已有服务。
- 正在补齐四模型真实几何、固定数据角色与候选、输入控制量、完整窗口采样和 pooled-only
  检测链。局部测试通过不等于四模型真实实验或插件结论通过。

### 2026-09-18 持续运行补记

- `9e1edc9` 和 `799670a` 已通过增量 bundle 上传；服务器主工作区现为 `799670a`。
  真实 CPU 探针使用独立冻结工作区 `/users/fotile/icassp2027-runs/code-799670ab`，
  正在运行 8 个视频、每视频 8 个片段。最近回执为 7 个片段完成，运行尚未结束。
- 固定 debug 视频是 Abuse005/018/025/040 与 Normal_Videos022/037/043/083。
  四个异常视频均属于 Abuse，因此这一工程小样本不能支持异常普遍性质。
- 本地限定镜像 8/8 下载完成，共 67,777,819 字节；大小与 SHA-256 均对应服务器冻结身份，
  无残留分块。它不是完整 UCF 数据集，服务器原下载继续运行。
- 本次进一步修正 TimeSformer embedding 的原始 batch 语义、分离 attention 的不可用标记、
  P16 的分支边界，以及 head checkpoint 的不可变训练身份。训练前强制比对真实 fit manifest
  摘要；一个已训练 checkpoint 可在严格许可下评估多个目标缓存，不回写 checkpoint。
- 当前本地解释器 `.venv/Scripts/python.exe` 执行 `python -m pytest tests/icassp2027 -q`：
  **134 passed**；`python -m compileall -q src tests`、受影响文件 Ruff 与 `git diff --check` 通过。
  此结果是工程测试，不是四 encoder 的真实质量或速度结论。

### 后续实测与在运行任务（2026-09-18）

- 本地 `243444d` 冻结工作区的 VideoMAEv2 debug8 已完成全部 64 个片段，退出码 0。
  run 为 `probe-20260917T174831982685Z-1f7ce815`，输出 51,136 条可用统计和 3,072 条
  not-applicable 记录。它仍是 8 个独立视频；四个正视频都属于 Abuse，不能泛化为异常规律。
- 服务器 `243444d` 真权重、真视频验证完成：TimeSformer run
  `observer-validation-20260917T174326293687Z-40667352`，VideoMAE run
  `observer-validation-20260917T174601576537Z-281e9f4f`。两者 native/observer/identity
  features 与 pooled 最大差均为 0；缺失 attention 统计如实记录，不补造。
- 真实 VideoMAEv2 的单次中间层固定索引实验确认 `1568 → 784` 后缀序列：block index 3 后
  gather，block 4–11 实际接收 `[1,784,768]`，pooled 仍为 `[1,768]`。identity 最大误差
  `3.5762786865234375e-07`；冻结主干后内部 leaf 参数梯度为 `-0.002514377236366272`。
  这是变长及梯度工程证据，不是最终 selector、检测质量或净加速结论。
- V-JEPA2 新增显式冻结主干内部插件梯度入口，默认 no_grad 推理不变。真实小模型验证插件
  梯度能穿过冻结后缀；实权重梯度验证尚未完成。原 `243444d` 的实权重 CPU observer 验证
  已以 2 线程、nice 15 启动，控制目录 `cpu-vjepa-verify-20260917T175520Z`。
- 最近独立 GPU 盘点仍无可用卡。node3 持久观察守候任务 `gpu-observer-queue-20260917T174607Z`
  正在等候，只有无计算 PID 且显存低、持有本任务租约时才运行；发生外部争用则终止自己的进程。
  这不是已完成的 GPU 实验或计时。
- 正式 explore128 与 confirm64 的冻结镜像共 9,980,051,375 字节（9.295 GiB），正在按原候选
  补齐本地镜像。服务器全量 UCF 原下载保持运行；未修改候选、原下载队列或其分块。
- 本地 debug8 的近重复候选审计未找到满足固定规则的 pair；9 帧摘要无法排除所有同源或
  不同裁剪，因此 source group 仍 unknown。接触表可见 Normal_Videos083 的结束卡，保留此
  内容异质性记录，不为获得更好结果临时删片。
- 检测控制器已接通真实 pooled 抽取、独立 validation store、TopKMIL BCE 与预测；新增
  验证缓存的采样策略核对、source group 冲突拒绝与 profile/assets 解析。
  本次相关测试为 **164 passed**，包括 validation 非空的 toy 集成测试；真实 head 训练待执行。

### 实际执行补充

- 代码 `6ab8781` 已做本地提交并上传服务器主工作区；模型运行使用独立 checkout。
  本地真实检测工程 run `paper-detection-20260917T181257007323Z-886655a8` 已启动，
  train 4 视频 ×32、validation 2 视频 ×32 的 pooled 特征已完整生成，无抽取失败。
  两个额外 debug 视频用于 dense 预测覆盖检查，主干冻结、TopKMIL BCE 5 epochs、seed 0；
  整个划分均来自原 fit 调试集，只用于工程核验，不参与性质选择或正式质量结论。
- 正式 explore 目标仍为预先冻结的 128 视频。核对发现实际服务器锁 seed 是 `20260918`，
  以原锁为准；全部 1610 个训练 ID 的角色和全部 explore/confirm 候选与本地重建一致。
  lock SHA-256：`53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59`。
- 已按实际落盘身份冻结第一计算批 127 视频（1016 片段）；34 个有服务器 SHA 比对，
  93 个绑定云端 fid/预期大小与本地实算 SHA，不混淆两种证据。唯一未齐候选为
  `Normal_Videos533_x264`。run `probe-20260917T182551484047Z-0a3dcc04` 在本地 CPU 4 线程运行，
  max_tokens=256、max_queries=32。剩余一条将单独补齐，128 视频齐备前不选择性质。
- debug8 的完整统计核验：64 clip 与 controls 的 ID、标签、分区全一致；799 个可估计签名，
  P13 无 CLS 的 48 项不填补。运动×亮度仅覆盖两组匹配，所有 scene unknown；没有从此小样本
  锁定任何性质。分页图已实际渲染并检查，图内明确标 debug、视频样本数和 site/head。
- **工程记录偏差**：绘图子任务在服务器 `code-6ab8781` 的 classic debug 探针运行期间改写了
  已跟踪的 `scripts/icassp2027/analyze_probes.py`（仅图标题）。完整差异、时间、SHA 已存
  `outputs/icassp2027/incidents/plot-only-source-mutation-20260917T181849Z/`。核验模型/collector/
  bridge/stages 源码均等于 HEAD；仍不把该工作区称为运行期间完全不可变，也不把该次运行当作
  正式确认。保留现场，未追溯改写回执；后续临时绘图脚本放 ignored 控制目录。**本地正式探索
  和本地检测工程 checkout 没有此改写，保持不变。**
- XD 官方六个视频 ZIP 已用匿名 HTTP Range 审核目录：4754 个视频、1 个目录，无路径逃逸
  成员；压缩 79.57 GiB、展开 80.06 GiB。后续核实：node2 的 curl HEAD 返回 403，但标准
  urllib + 本次匿名 CookieJar 的 GET Range 0-0 返回 206；不能把 HEAD 的结果误写成数据不可达。
  本机初次归档 worker 误用 Windows PowerShell 5.1，缺少 HttpClientHandler；改为已验证的
  PowerShell 7 后实际写入约 7.03 GB。因服务器直接下载路径已验证，已停止这组自己的本机
  下载进程，保留两份 part/身份记录作备份，正在安排 node2 独立下载；UCF 原下载不变。
  此状态不表示 XD 数据已齐，也未读取测试事件标注。

以下保留上一轮基础重构的已完成记录。

### 本机 CUDA、输入与缓存核验

- 本机实际存在 RTX 4060 Laptop 8 GiB，驱动 566.24。已在独立
  `.encoder-envs/local/torch280-cu126-py311` 加入官方 PyTorch 2.8.0+cu126 / torchvision 0.23.0+cu126，
  沿用 Python 3.11 与其余现有依赖；原 `.venv` 解释器及 Torch/vision RECORD 摘要未变。
  CUDA matmul 和真权重 V2 observer/identity 均通过，后者四项误差均为 0，run
  `observer-validation-20260917T193123231246Z-aa43acc8`。
- CUDA 主探索 run `probe-20260917T193541117630Z-ef589e03` 使用同一冻结 127 视频第一批和原始
  观察预算；CPU run 保留为另一运行条件的重复检查，不合并成更多独立样本。主探索选择 CUDA
  是依据可用计算资源，在查看正式性质统计前固定；原生精度与库版本分别保留。
- 唯一余项 Normal_Videos533 已从 Quark 完成，且 SHA 与官方 ZIP 独立补取完全相同：
  `7aa51dd1d758b0135fea20810507044671489ab647d8fca21bdc2fb6a8680618`。
  两个运行条件各排入同版本的 8 clip 补齐任务，依赖对应 127 视频批完成，不重复合并。
- 真实检测工程训练已完成：128 train clip、64 validation clip、249 dense evaluation clip；
  5 epochs / 10 steps，参数确实更新，有限非零梯度与 checkpoint 校验通过。同种子完全重跑
  的 head 参数差为 0，原 checkpoint SHA 未变；两视频 3999 帧覆盖完整且无重叠或缺口。
  训练损失从 0.6178 到 0.3596，验证损失从 0.6954 到 0.8272：出现小样本过拟合，不能作为
  泛化、插件有效性或正式检测质量结论。正式测试仍未访问。
- 修复实际缓存身份缺口：有效 constructor、已加载模型/encoder config、processor 配置均参与
  摘要；真实 `torch.dtype` 单独规范化，其他未知值仍严格拒绝。真权重加载实测改变 pooling
  会改变表示身份、恢复则还原。旧工程缓存保持原身份，不重标或伪装成新版本。
- 已实测原 decoder 对近邻帧重复 seek 的开销。近邻顺读在 debug8 的全部 RGB bytes/SHA、
  重复/乱序/长跳边界上与旧实现完全相同；生产实现仍对长跳 seek，异常时回到原目标 seek。
  实测 decode-only 配对中位约 7.24 倍，**不是插件、GPU 或端到端加速**；新采样身份绑定
  decoder 源码 SHA。运行中的旧冻结 probe 未修改。相关定向测试 68 项通过，最终受影响
  video/extraction 24 项通过；身份/检测相关 39 项通过；原生 verifier 17 项通过。
- 本机另外三份 active 权重也已通过原 registry 校验。TimeSformer 的小型预处理配置从现有
  服务器复制并验证指定 SHA，其来源与官方模型主体分开记录；没有编造官方 snapshot 文件。
- 官方 UCF 单 ZIP 的 192 个候选路径/大小全部与冻结计划相符。补齐中间证书链后，node2 使用
  原 certifi 根锚、主机名和证书校验的临时链路已实际通过；没有新增根 CA 或更改系统 TLS。
  新 `UCF-Crime-official-verified` 正按官方 CRC/SHA 复用或独立下载 1900 个视频，旧 Quark
  任务保持原路径。错误地在 node3 启动的一次零 payload 失败已单独保留，不混入成功结果。

此前基础重构范围是已确认的 M0 与 M1 工程基础：论文入口、配置、身份契约、cohort、只读 collector、
一级统计和 identity 桥。具体 reducer、dense sampler、XD 协议、正式 detector 实验和论文数字
依照研究顺序留待后续阶段，不在本轮以占位实现冒充完成。

## 本轮变更

| 范围 | 实际完成内容 | 验证边界 |
|---|---|---|
| 论文入口 | profile/protocol、路径示例、status/probe/verify CLI、严格字段解析与 dry-run | 原有全量 catalog 与 CLI 不变；只有 profile 中的 active ID 可由论文入口调度 |
| 观察与标签 | cohort 的视频/同源分区检查、W 标签限制、collector 与分析标签 join 分离 | 不允许官方 test 进入开发 cohort；输入 encoder 的 metadata 已剥离标签与文件名 |
| 一级探针 | P01/P02/P04/P07/P10/P11/P13/P16 的有限统计，缺失或不适用时明确标记 | 仅描述统计实现与工程回执；没有正常—异常 finding card 或统计结论 |
| 模型桥 | 四模型已知 block/site 定位；TimeSformer 分离时空 site；V-JEPA 2 encoder/predictor 边界及 no_grad 说明 | 其他三模型使用真实库的随机小模型验证模块定位；真权重几何和完整采集验证尚未运行 |
| VideoMAEv2 | 真实 Conv3d tubelet 来源、实际展平顺序、native attention 行采样、identity 与 observer 等价验证 | 真实视频、真权重、本地 CPU、一个无标签 clip；不作为速度或异常性质证据 |
| 实验身份 | backbone/representation/sampling/training 身份、缓存键、显式兼容声明校验 | 现有 predictor 的严格指纹保护未改；跨表征/采样的预测执行仍未接入 |
| 工程保护 | 原始文件摘要快照、只读保护检查、测试、wheel 回归与使用说明 | 工程改造未修改其他 encoder、注册表、原公共源码、权重或环境；随后按用户要求进行本地 Git 归档，见下节 |

新增主要文件位于 `src/vadbench/paper/`、`research/`、`token_reduction/`、
`tests/icassp2027/`、`scripts/icassp2027/`、`configs/papers/icassp2027/`、
`projects/icassp2027/` 和 `paper/icassp2027/`。
已有文件只调整根 `README.md`、`README-CN.md` 的导航及 `.gitignore` 的本机路径忽略规则。

## 最新真实回执

Run ID：`observer-validation-20260917T150359001546Z-cceee66f`。

- [架构与等价回执](../../outputs/icassp2027/runs/observer-validation-20260917T150359001546Z-cceee66f/architecture_receipt.json)
- [运行摘要](../../outputs/icassp2027/runs/observer-validation-20260917T150359001546Z-cceee66f/summary.json)
- [统计输出](../../outputs/icassp2027/runs/observer-validation-20260917T150359001546Z-cceee66f/probe_summary.jsonl)
- [本轮锁定记录](locks/refactor-20260917.json)

解释器：`D:/PythonProject/VAD/.venv/Scripts/python.exe`；实际导入
`D:/PythonProject/VAD/src/vadbench/__init__.py`；设备 CPU。
环境为 Python 3.11.15、PyTorch 2.13.0+cpu、Transformers 5.16.1；版本与来源同时记录在
同 run 的 stage provenance。没有安装或升级模型环境。

输入为已有真实视频 `data/smoke/mlvu-surveil-8.mp4` 的固定 clip，权重使用已有
`weights/videomaev2-base-hf`。核心 safetensors 摘要与登记匹配：
`ebffa1874066ea227330016e58a848e9e2bb1ff5605746459bded1122a42176d`。
模型实际实现文件摘要、源码工作树摘要、输入视频摘要、采样帧、处理后的 shape 均随 run 保存。

| 项目 | 实测结果 |
|---|---|
| block 数量 / 每头数量 | 12 blocks / 12 attention heads |
| tubelet / patch | 2 帧 / 16×16 |
| 网格与特征 | 8×14×14；features `[1,1568,768]`，pooled `[1,768]` |
| CLS | 无；P13 标记 not_applicable |
| observer features / pooled 最大绝对差 | 0 / 0 |
| identity features / pooled 最大绝对差 | 0 / 0 |
| attention 来源 | native eager、dropout 前概率；有限 query、完整 key 轴 |
| 当前有效范围 | 工程等价性和真实几何；`reduction_ready=false` |

注意：此原生模型使用函数式 `linear` 计算 Q/K/V，qkv 模块本身不被调用。
桥记录实际投影路径并观察 native attention；不把从未触发的 qkv hook 写成采集成功。
几何坐标属于预处理后的网格，不宣称已有原视频像素级定位。

## 验证与命令

本轮使用上述本地解释器，执行了新能力测试及文档 02 第 11 节列出的核心兼容测试。
最终结果：**182 passed，1 skipped**，其中新能力测试 64 项全部通过。
跳过项为 Windows 创建测试符号链接权限不足；不是模型运行失败。
最后一轮结果及完整命令见：

- [最终测试日志](../../outputs/icassp2027/refactor/20260917T141225Z/final-tests.log)
- [最终测试命令与退出状态](../../outputs/icassp2027/refactor/20260917T141225Z/final-tests.json)
- [保护检查结果](../../outputs/icassp2027/refactor/20260917T141225Z/protection-final.json)
- [原工作树基线](../../outputs/icassp2027/refactor/20260917T141225Z/baseline.json)

验证命令要点：

```powershell
.venv/Scripts/python.exe -m pytest tests/icassp2027
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m ruff check src/vadbench/paper src/vadbench/research src/vadbench/token_reduction tests/icassp2027 scripts/icassp2027
.venv/Scripts/python.exe -m vadbench.paper verify --project projects/icassp2027/profile.yaml --encoder videomaev2 --video data/smoke/mlvu-surveil-8.mp4 --device cpu
uv build --wheel --offline --out-dir outputs/icassp2027/refactor/20260917T141225Z/wheel-final
.venv/Scripts/python.exe scripts/verify_wheel_resources.py outputs/icassp2027/refactor/20260917T141225Z/wheel-final/vadbench-0.1.0-py3-none-any.whl
.venv/Scripts/python.exe scripts/icassp2027/verify_paper_wheel.py outputs/icassp2027/refactor/20260917T141225Z/wheel-final/vadbench-0.1.0-py3-none-any.whl
```

真实运行在进程内设置离线模式和 4 个 CPU 计算线程；不改变系统配置。
wheel 检查从临时安装/解包目录导入，验证旧资源与新 paper profile/suite 的打包后解析，
没有把“源码目录可导入”当成 wheel 成功。

保护检查覆盖 316 个原有源码、配置、登记和文档文件，原始内容变化仅有计划内的三项入口文件。
大型权重、视频、环境、external 和历史 outputs 不在本轮写入范围；没有递归哈希所有大型资产。
Windows 缺少创建符号链接权限，原有 symlink 测试跳过，不能计作已通过。
GPU 验证未运行；服务器盘点时八张 GPU 均有作业，本轮未抢占或改动这些进程。

## 本地 Git 归档（2026-09-17）

用户随后明确授权“本地做好 git”。工作分支为 `qzt/icassp2027-foundation`，采用两次提交：

1. `ff74605`：保存本轮之前已有的基础框架、配套文档和测试，内容按重构前 baseline 核对。
2. 随后的 ICASSP foundation 提交：本轮新模块、论文配置与入口、测试，以及 README 导航和本机路径忽略规则。

拆分原因是新模块依赖此前未提交的 `hashing.py`、`resources.py`、`videomaev2_encoder.py`
和公共框架改动，仅提交新增目录会得到不完整的版本。原工作树中的这些改动保持原内容。
本节与第二次提交一起归档，提交号以 `git log -2 --oneline` 为准。

只做本地提交，不 push。视频、权重、环境与 outputs 继续忽略；`VAD_Idea/` 和两个根目录临时
修复脚本保持原位、未纳入本次提交。测试与运行回执中的原 Git commit/dirty 标记是运行当时事实，
不追溯改写；源码内容摘要可以用于核对提交后的相同实现。

## 下一阶段

下一项工作是固定训练集中的 4 个含异常视频与 4 个正常视频及其 cohort/source 分组，运行
`probe-pilot`。本地尚未建立该真实 cohort 与对应 UCF 训练 manifest，dry-run 会如实报告缺失。
服务器存在视频资产不等于该研究分区已经审计和冻结。

随后扩展正常—异常分布与独立确认；在取得可操作性质前继续保留 identity。
其他三模型的真实几何/只读等价回执、正式 GPU 验证和质量容忍度仍需在相应阶段落实。
正式结果和论文数字仍为空。
