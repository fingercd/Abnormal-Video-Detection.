# ICASSP 2027 当前进度

更新日期：2026-09-19。当前状态：**完整研究 goal 进行中；仅有 VideoMAEv2 特定性质的独立确认，尚无跨四 encoder 的通用规律或插件有效性结论。**

当前位于`RESEARCH_PLAN_V2.md`的**第2/7步：开发期dense基线、性质证据与方法冻结**。
第1步四组native工程验收通过；第3步正式完整train三seed、最终TF/LoRA比较及论文结果尚未完成。

## 当前运行快照（本轮接手后的最新状态）

- node3最近实际快照epoch1789745433：完整视频分片V2 fit179/select97、Time fit184/select104、
  VMA fit263/原select79、VJ fit89/select70，原fit/select分母为1288/161；视频长短不等，不能作为计算完成率。
  原VMA select在完整79视频边界暂停；唯一handoff监视器已按Time完成/VJ失败终态派发resume guard17046。
  新resume正在复制/恢复原完整分片，当前目录16个完整shard不能算作16个新增视频；保留原完整身份和旧79分片。
  其余七路native进程仍存活。完整dense总量2,923,485 clips；UR-DMU开发头尚未开始3000步训练。
- Time原128个fit视频、64正常/64含异常、8窗口共1024 clips的完整观察已完成：
  `code-f25d79e/outputs/icassp2027/runs/probe-20260918T132416783598Z-94424de8`，guard13203 exit0。
  已重新核对完整cohort、raw SHA、几何/parity和controls；code-deda86e的10,000次视频级
  fixed-min-count matched bootstrap已完成，输出`timesformer-fit128-video-bootstrap-20260918-a01`。
  独立审计确认1,119一级签名=1,095可用+24 N/A，140,160有限video行、128视频每个8clip；
  control仅用Time自己的64个fit-normal拟合。全族/跨深度方向混合，不能概括为全局通用规律，
  也不排除特定site的待确认候选。Time旧P10/P11缺少局部attention实现，不等于科学阴性。
- VJ原观察guard13204随后取得GPU6，但在CUDA初始化前因剩余磁盘预算不足退出；没有probe数据。
  已按原56GiB完整dense预算加1GiB VMA resume副本预留修复容量缺口，不降低15GiB安全余量。
  VJ新观察将采用修正后的probability hook：native Transformers4.57.3默认仅返回context，
  旧fallback会误报token族；现已修正为明确P10/P11/P13 unavailable，不强制切换生产attention后端。
  tiny fixture明确请求output_attentions以核验真实第二项；本地两组受影响测试共39通过，Ruff/compileall通过；冻结0192fad已上传，native也39 passed。
- A100 dense-only clip计时a02四模型均已完成，guard28110 exit0；独占GPU6、FP32、cuDNN/TF32关闭。
  独立审计已核验B1/B8、warmup5/repeats30、纯模型/adapter/单clip解码口径；没有计入UR-DMU后端，
  不能将其称为完整视频detector质量或完整长序列端到端效率。完整审核数字另行归档。
  a01首个Conv3d的cuDNN loading失败原样保留；其之前UR-DMU四例GPU operator equivalence已通过，
  含D768/D1024及明确synthetic repetition的N1025/Q128，max abs差最高8.94e-08、权重未变。
- Time局部attention只读实现已完成本地36项验证；native a01/a03含VJ的组合测试因classic环境无该类失败，
  a02 foundation组合35通过/1失败，后者已由旧f25同环境复现为VJ tiny fixture缺少output_attentions=True。
  a03实际Time相关28项通过；独立a04已用真实Time预训练权重和两个fit视频完成B1/B2 source clip归约、
  hooks/identity parity及参数不变验收；模型前后SHA均3ad0137b…eef9d，CUDA未初始化。上述失败保留。
- XD三尾卷已CRC解压，测试raw800齐备，训练仍缺4个原始坏成员。旧9b4e1f2全800审计最终794通过/6失败，
  六个失败均为原冻结规则拒绝的精确QT chapter metadata诊断，不改写旧结果。新1b6bd26完整800复审
  b01因launcher缺PYTHONPATH退出；b02已在node2真实运行，PID132797，最近590/800通过、0失败、0复用。
  仍需完成全部PTS/24fps/OpenCV解码及sealed门禁，不能提前宣布新增0598/0599/0600完整通过。
- XD四坏源的官方OneDrive页面可访问下载按钮，但工具无法把浏览器下载定向到D盘；等待用户完成已提出的
  临时下载目录设置或提供直接原始文件链接，没有启动约15GB到C盘的下载。UCF研究不等待该人工步骤。
- 磁盘门禁实测缺口2,460,884,992 bytes（仍预留完整56GiB dense、2GiB XD、4GiB probe、1GiB resume
  副本及15GiB安全余量）。按用户既有条件清理授权，只移除了未被active配置或可读计算进程引用的
  `weights/videochat-flash/model.safetensors`：4,143,085,560 bytes，SHA8d7599e4…70b475，单硬链接、非symlink。
  其配置、源码、tokenizer、training_args、锁/缓存及历史产物保留；该inactive模型以后重跑须先恢复权重。
  初次审计因sshd的proc权限有限而停止，复核仅为低内存SSH传输进程后a02执行；未绕过系统权限。
  回执`/users/fotile/icassp2027-runs/control/inactive-flash-weight-cleanup-20260918-a02/receipt.json`，
  清理后实际free=79,340,277,760 bytes。原TopKMIL coordinator仍保持研究协议hold，
  未读取官方模型测试分数。可操作性质、同源TF/LoRA、正式完整train三seed和最终质量矩阵尚未完成，Goal active。

## 新观察恢复与科学审查（2026-09-19）

- 新冻结`0192fadcfcb6f4c0a677aeca65730681e0a89b51`，bundle SHA
  `c492d95db06ed0f2ff17630a6cb4d6ea6a36adee573e20e085f31fe3341f3564`，native39项全部通过。
  不改变生产VJ后端/forward；仅缺失原生概率时明确缺失，不把context挂在probs site上报token证据。
- 在该工作区`outputs/icassp2027/control/cross-encoder-fit-explore-20260918/recovery-0192fad`
  已创建唯一新队列：VJ guard24800完整8探针，Time guard24836仅P10/P11，仍为原128 fit、每视频8窗口。
  原cohort/manifest/plan小元数据按SHA复制到新项目内以满足resolver路径合同；原视频和角色没有改变。
  node3 epoch1789746696：Time已获GPU6，真实run`probe-20260918T155022550042Z-94a7dc2c`
  26/1024 clips，VJ仍等待lease；不读取或解释未完成子集。
- 旧VMA select resume当时仍只在复制原79个完整视频，当前已见的62个目录全部属于原79集合，
  没有开始新的视频提取。核验owner/argv/startticks、原完整source SHA及当前copy语义后，唯一新monitor24916
  经原guard cancel停止该重复复制，guard17046随后cancelled/exit-15且native18448退出；原79完整shards保留。
  两项新观察都completed后，monitor将从原79唯一source用0192新batch writer恢复完整select；若观察失败则
  停在needs_review供修复，不自动切走研究卡。其它七路dense不受本次让卡影响。
- 新旧VMA resume已有真实CPU兼容验收：adapter/native model/processor摘要、representation、semantic runtime、
  encoder fingerprint均一致。原真实reader等价18组RGB/mask/索引/时间戳验证、64数组single/batch writer
  等价回执也绑定到新handoff plan；不把实现Git不同混为表示不同，也不放松严格resume消费门禁。
  只有原source完整视频可复用；新copy目录的partial记录未被当作source。受影响的只是I/O实现与调度。
- Time完整fit分析审计在ignored `outputs/icassp2027/control/timesformer-fit128-descriptive-20260918/`，
  analysis receipt SHA`6df22038…441b53`、video summary SHA`3c624157…a3e713`、contrast SHA`de66a737…812fb`。
  weak CI继续解释为Hedges g，matched CI为motion×brightness固定min-count的raw delta；全网格未多重校正。
  正在对全部共享签名做完整描述和类别敏感性，不据少数显著cell锁定层、head、方法或直接进入LoRA。

## 已审核的A100固定clip dense基线（2026-09-18）

同一fit视频Abuse005_x264；GPU6独占guard、FP32、cuDNN/TF32关闭；每scope warmup5、timed30。
下表为B8同步wall中位数，单位秒，不能比较为同样时长的视频吞吐（各encoder原生窗口不同）。

| encoder | 纯模型＋pooled | adapter | clip解码＋adapter | 峰值allocated/reserved GiB（解码scope） |
|---|---:|---:|---:|---:|
| VideoMAEv2 | 0.31076 | 0.77835 | 4.29336 | 2.375 / 2.518 |
| TimeSformer | 0.22175 | 0.59163 | 1.58709 | 1.144 / 1.484 |
| V-JEPA 2 | 5.62937 | 9.59445 | 18.39982 | 4.599 / 5.926 |
| VideoMAE | 0.29916 | 1.04538 | 2.34982 | 2.448 / 2.522 |

B1/B8全scope、CUDA event诊断、P10–P90及输入/模型/代码指纹在ignored
`outputs/icassp2027/control/a100-dense-baseline-review-20260918/`。UR-DMU不在本计时边界，
完整detector和整视频dense序列效率仍待实际匹配head完成；本表没有检测质量或压缩加速结论。

## 第2步继续：优先性质观察、纠正探索统计（2026-09-18）

- 新增只读诊断：对原V2 fit128的5个固定站点做每clip尺度比值，再平均8clip到video；
  10,000次新matched统计，全部10项与类别描述保留。embedding的IQR/median及variance/median²
  仍为负，Block9 input的IQR/median为正；其它多个站点区间跨0。不能以一个全局尺度或一个
  “异常高范数/高方差”规则概括，更不支持直接norm-topk。它是posthoc fit敏感性诊断，非确认。
  数据和图在ignored `outputs/icassp2027/control/v2-scale-sensitivity-20260918/`；图已实际查看。
- Time的P10/P11旧N/A经源码审查是未实现局部group归约，而不是模型没有attention性质。
  现已补齐只读temporal/spatial原生key域统计，按真实B×P/B×T分组还原每source clip、每head，
  不伪造global矩阵，不重复视频样本；P13全局CLS仍明确unavailable。定义见
  [TimeSformer局部attention口径](decisions/timesformer-local-attention-probe-v1.md)。
  attention-only观察现在只挂probability hooks，继续三次前向parity而不额外计算无关SVD。
  本地相关36项测试、Ruff、compileall通过；真实native验收待执行，当前f25运行未改动。

- 用户询问阶段、数据、四encoder进度及fit/select、开发头含义，已明确解释当前仍在找可操作性质，
  并行建立开发期原encoder＋UR-DMU基线；没有最终方法或官方frame质量。fit1288训练/参考拟合，
  select161开发比较，confirm161独立性质确认；冻结后再联合官方train1610训练正式头。各encoder
  的UR-DMU独立训练，且同维不代表同特征空间；主direct_insert的各方法共用同encoder/seed的dense头。
- 为避免两主模型观察再等待全部select提取完成，已在VMA select的`Normal_Videos548_x264`
  5264个native clips完整发布后，通过原guard cancel让出GPU6，未影响其他7路。暂停前核验guard/native
  owner、argv、startticks和resolved SHA，保存79个完整视频index的SHA/条数；完整分片不被删除或改写。
  唯一交接monitor9980位于`code-2d9d1a0/outputs/icassp2027/control/`
  `research-priority-handoff-20260918-a01`。两组已登记probe guard终态后，将以原2d源码/classic环境、
  同full-select authority、相同np预处理/微批/采样，通过`resume_source`在新目录恢复全部161视频，
  未完整的最后视频重新提取，不把旧partial输出当完成。额外复制预算及剩余dense/raw/probe/15GiB
  余量已实时预检。该调整改变排队优先级，不改变实验身份或方法预算。
- 对V2既有fit128的全847个一级签名完成只读盘点，v1摘要的CI对象解释不够明确，因此保留v1并写v2：
  原weak CI是Hedges g，raw delta只是另一个点估计；legacy matched按complete groups等权重抽样，
  原分析是1000次而非本轮要求的10000次。全量阴性和N/A保留，没有从计数直接推广候选。
- 新公共matched入口改为`fixed_min_count_video_stratified_v1`：固定完整strata与min(nN,nP)权重，
  在每个bin×label内按独立video重抽原样本量，point与CI同一估计器；报告unknown/one-sided排除、
  retained、min-count mass、bin/singleton构成及CI对应量。全退化bootstrap仅保留point和覆盖，CI=null。
  已有F01/F02/F03确认使用原v3冻结统计，不被本次修改重算或改写；新结果仍是未校正的fit探索。
  本地contrasts＋controls共25项通过，包含16种联合draw的独立精确枚举CI、相同bin差但bin内有方差、
  顺序不变性、one-sided/unknown和单bin/退化边界；Ruff、compileall及diff check通过。
  新代码`deda86ed0a23354d11ca3a402e059df227ad69a0`已以bundle
  `b2f02a4804b54dfdaed141c32053fef66b3ca0eca4ca13fb15a1e1e43542aa71`安全上传到独立工作区；native也25 passed。
- 已实际按新统计重算旧V2 fit128，产物`outputs/icassp2027/analysis/`
  `v2-fit128-video-bootstrap-20260918-a02`：867,328原始probe行、128视频64/64、每video8 clips，
  完整847一级签名中799有可用video值、48 N/A保留；10,000 bootstrap，无候选选择或新确认结论。
  原probe实际是Windows Torch2.8 CUDA观察，本次仅本机CPU重算，不称A100新观察。
  四个输入文件前后SHA匹配旧封存gate；独立verification receipt SHA为
  `ba1d800b5aaaa9a6f33b66812805495b2ba5dd087cb40226d8f2d448cab7a43a`。
  a01后台生命周期退出且未产出，保留stopped_without_receipt；a02最终CLI成功marker与全部产物门禁
  通过，但PowerShell supervisor未保存Python exit code（null），不改写为0。初始a02只监测launcher，
  随后追加独立descendant守护；实际分析树约4.4GB、可用RAM约6GB，未触发3GB/6GB门禁。
- 用户再次明确要求专责修复XD四坏源，已交给独立agent执行：寻找准确原始副本、最多2GiB新临时
  下载预算、逐成员size/CRC/SHA及解码验证后输出独立receipt，再由root接正式inventory。
  不覆盖旧raw/归档，不改变期望CRC，不以重编码或近似剪辑冒充修复，不启动重复writer。
- XD成员级复查已排除当前客户端下载截断：原zip总长及外层SHA与冻结身份一致，四个压缩段均完整；
  一个deflate报错，另三个解压长度正确且可播放，但CRC不符。429,389,964B分类payload仅在新control，
  不接入raw/inventory。服务器没有找到第二份归档或历史可恢复分块。
- 浏览器真实检查纠正了“没有可访问来源”的判断：作者OneDrive `1005-2004.zip` 页面可见下载按钮，
  只是node2的同share请求跳登录。已给专责agent扩大本地D盘临时预算至20GiB（实测空余223.15GiB），
  服务器仍保持2GiB预算；计划浏览器本地取得原zip，再只传核验通过的4成员。
  当前WebBridge拒绝下载路径控制和浏览器设置页，下载BUTTON也没有可直接读取的DOM href，故未点
  约15GB下载到空间较少的C盘。已向用户请求将该任务浏览器下载目录临时设为
  `D:\PythonProject\VAD\outputs\icassp2027\control\xd-four-bad-repair-20260918\browser-source-a01`
  或提供直接原始文件链接；等待回复期间UCF任务继续。未读取或导出cookie/token，未改永久设置。

## 本轮继续：观察入队与实测写入开销（2026-09-18）

- Time/VJ观察使用`code-f25d79e/outputs/icassp2027/control/`
  `cross-encoder-fit-explore-20260918-a01`，每模型原fit128、64正常/64含异常、8窗口，
  共1024 clips。authoritative plan/cohort/manifest SHA分别为`6fe7ad49…64d`、
  `6766efa3…3d5`、`742ec3d4…6fb`。runner重新绑定完整UCF authority、逐原视频SHA、
  真实native sys.prefix、模型窗口及原角色；源代码和脚本均冻结记录。只选GPU6/5/7/2等待lease，
  allocator fraction0.5，至少24GiB空闲；正式计时继续独占guard14261等待，未抢占任何健康任务。
- 两个探索统计入口已核查：仅分析完整run，分别用各自原生窗口的fit-normal控制拟合motion/brightness，
  不复制V2的16帧阈值；Time不支持的分离attention统计及VJ无CLS项保持N/A，保留全部阴性。
  public matched分组统计只能作探索描述；后续确认仍需预登记且按视频在bin内bootstrap，不能回写成确认。
- 新的真实64条FeatureStore写入CPU诊断绑定Time select中已经完整发布的`Abuse012_x264`首块。
  读前后源tree SHA均`0ea4d232…23b5`；独立诊断目录只写637,059 bytes。64次逐条upsert导致
  2,016次旧FeatureRecord解析、64次索引发布；共享node3高负载下wall为3.661秒，仅定位工程开销，
  不能推算正式速度或总提取占比。node2只读SSH超时未重试，r01启动路径错误与有效r02分开保留。
- 基于该证据新增`FeatureStore.write_many`：同一锁内一次读/发布index，继续使用原blob、schema、
  fingerprint与单条upsert语义。提取按micro-batch提交，完整视频恢复按64条提交，块上限仍64。
  批内重复、后项无效、blob/index失败及跨块失败均不能发布部分video/root；不可达blob保留。
  首轮回归发现恢复路径尚用旧私有接口，已修正；最终提取/恢复/常规store联合81 passed，
  新边界负例加UR-DMU训练/评分43 passed，Ruff、compileall及diff check通过。
  新冻结`030c86f28b37646ded5b81e9ce0a3507a58ec3af`已上传，bundle SHA为
  `579295746ceca1b2ef4b0ee6efdc16bc1d99a5718443ac0ee5d6df1e3e9d9772`。
  native同范围114 passed/2 skipped；两个skip是默认位置没有作者源码，显式使用现有固定
  `URDMU_UPSTREAM_DIR`后单独2 passed，包括真实更新和精确重载，未安装环境或复制外部源码。
  原生真实64数组按single/batch8/batch8/single交错重放，四次数组及除created_at外全部记录语义一致；
  源tree SHA前后不变，诊断只写2,397,041 bytes。batch8的旧记录解析从2016降至224、index发布64降至8；
  wall为single3.220/3.249秒与batch0.830/1.859秒，保留共享CPU波动，不能报为正式端到端加速。
  回执在`code-030c86f/outputs/icassp2027/control/feature-batch-native-20260918-a01/real-replay/receipt.json`，
  本地镜像SHA`083a40314dfac824302fef743b9bbf7bffc819c7d24cc481800b125f7ed5605e`。
  现有八路运行仍保持2d9d1a0，未原地换代码或重启健康提取。
- 只读审查确认现有resume仅接受同一完整身份的一个来源，不能把健康fit任务直接拆到额外GPU。
  本轮不新建跨store assembler，不把工程子集拼成完整训练，后续释放卡先用于已登记的观察与计时。
- 四个XD坏源的补充只读检查已有明确结论：官方项目页的AliyunDrive训练分享
  `https://www.aliyundrive.com/s/6UquaxKpKTm`在真实浏览器显示“文件违规，该文件已禁止访问”，
  无目录或成员可核验。CUA连接层不可用后，复用本机已安装Kimi WebBridge并只打开本任务标签，
  未登录、下载、转存或读取凭据；官方OneDrive匿名登录限制及HF镜像缺失结论继续保留。
  训练四缺口没有被这次可见性检查修复，仍需可访问且可匹配冻结size/CRC的原始副本。

## 新执行任务接手（2026-09-18）

- 用户选择的现有目录方案已生效：在 `D:/PythonProject/VAD` 接续
  `d555ca8ed2d761f73935359d1fa18ca1da486f1f`，本任务 ID 为
  `01a0b40b-ae65-7222-ae63-6c2a5f3f28bd`。已按 `GOAL_NEXT_TASK.txt` 全文建立 active Goal，
  不设 token 预算，不再等待旧 worktree 或启动确认。下方“待创建/待审阅”均为历史记录。
- 首次 `git status --short` 只有受保护的 `VAD_Idea/` 和两个根目录临时脚本未跟踪。
  本轮未读取旧方向、未重装环境、未访问官方模型测试分数。
- 经公网线路实连 node3，复核四组 native UR-DMU 验收 aggregate 及逐模型 receipt SHA，
  aggregate 仍为 `16fa0a7a56b7fe4084e2fee2f4633ca0db6da7d87792ca2eaf016929adbcaa6c`，
  guard completed/exit0。UCF1610 训练视图合同 SHA 与交接一致；不重跑已完成验收。
- 实时发现五条旧 train32/TopKMIL 原型作业仍运行。核对 owner、完整 argv、进程启动标识、
  guard/child 后，保存现有索引摘要并通过其 guard 的 `cancel` 文件有序停止，保留所有产物。
  请求回执：服务器 `/users/fotile/icassp2027-runs/control/goal-resume-20260918/`
  下 `superseded-train32-request.json`；五项随后均 confirmed cancelled/SIGTERM/exit -15，
  child 均已退出，见同目录 `superseded-train32-result.json`。未停止无关用户或服务。
- XD 实时复核：train2805 writer 已退出且传输 progress=completed；test105262、train3320
  83455 均仍为原 fotile writer，继续运行，无重复启动。传输完成仍需 CRC/SHA、解压和
  raw 坐标审计；四个源坏成员尚未据此宣布修复。共享 `/users` 剩余约 149 GiB。
- 当前实现重点为严格绑定原 fit1288/select161 的 UR-DMU 开发期 dense 提取、固定预算
  训练和视频级开发评分；confirm 及官方 test 不并入该训练。旧 TopKMIL 契约不削弱。
  提取/训练接口已实现：`--development-role fit|select` 与 `--run-mode development`。
  仍强制完整角色、3000 steps、64+64 bags 和200bin；拒绝confirm、任意子集及来源变更。
  本机 `.venv/Scripts/python.exe` 的提取/训练测试分别10、21项通过，Ruff和compileall通过。
  这31项是工程测试，尚未声称开发训练基线已完成。
- 性质审查保留 P16/P07 的拒绝及操作负结果；正在核对 P04 局部冗余的未使用确认样本。
  新的[P04确认计划](decisions/p04-heldout-confirmation-v1.md)已在读取新结果前登记：
  VideoMAEv2单一embedding冗余统计、剩余confirm97中哈希固定64、原fit-normal匹配阈值，
  失败即停止该性质干预。在独立确认和受控干预通过前，不实现或训练论文压缩规则/LoRA。
- 完整开发fit/select四模型预检共2,923,485 clips。现有FeatureStore分配估算约52GiB，
  另留4GiB缓存/日志；XD当前剩余归档及raw写入约49.48GiB，预计保留约32GiB磁盘余量，
  后续以实际增长复核。此为容量预算，不是实测提取速度或正式效率数字。
- 提取/训练入口及P04预登记已本地提交为 `2d9d1a0`，bundle SHA
  `a953133b0f1183b657063b6eb2093e4a92ff6398019223a5e6cbc626526349a5`。
  服务器新冻结工作区为 `/users/fotile/icassp2027-runs/code-2d9d1a0`，native compileall、
  原角色consumer和P04的64视频/512行cohort物化均通过。首次旧主仓库缺bundle前置提交，
  在任何GPU启动前失败；随后从实际持有866674c的冻结仓库建立新工作区，没有修改运行源。
- **八卡实际运行已开始**：GPU0–3分别为V2/Time/VMA/VJ完整fit dense，GPU4–6为
  V2/Time/VMA select dense；GPU7先P04独立确认，再接VJ select。普通共享任务均持物理卡
  lease、allocator fraction=0.5，启动时至少24GiB可用显存，不作为正式计时。
  主control为新工作区下 `outputs/icassp2027/control/development-baseline-20260918-a01`。
  七条dense已产生真实完整视频shard；fit完成后各自执行seed0的固定3000步开发头训练，
  尚无开发质量或正式frame指标。
- P04首次在GPU7设置CUDA allocator预算时OOM，发生于探针前、未产生任何P04统计；保留
  `a01`失败。重新核验空余并显式初始化设备的 `development-slot7-20260918-a02` 已运行，
  同一冻结样本与设置不变。最近进度为155/512 clips、19/64完整视频，尚不解释部分结果。
- UR-DMU开发评分与全key/value精确query分块推理已实现。原完整attention默认不变，分块需
  显式启用并保留原作者位置先验实际广播；它是等价执行优化，不计作视觉插件创新。
  ROOT `.venv` 联合后端/分块/评分/训练验证49项通过，随后真实fit身份修正的评分12项通过；
  Ruff与compileall通过。目标GPU数值等价仍待验证，未据CPU测试宣称长视频显存达标。
  新评分拒绝同维但异encoder、不同采样、缺失select或不完整训练QA；官方test路径在新v3
  方法冻结合同建立前明确拒绝，旧原型冻结回执不能解锁它。
- XD三尾卷已完成归档恢复。已核实并保留旧失败status后，按独立ZIP大小/SHA/成员CRC证据
  更新监督器的恢复标记；train2805现在515/515 raw-ready。原监督器因test卷合法`videos/`
  目录条目而失败，其日志保留；新唯一监督器已按路径/类型/重复/软链接负例验证后恢复，
  正在实际解压test，随后train3320。800 raw坐标auditor继续原实例等待，没有重复启动。
- 四个源坏成员仍未解决，公开raw镜像也缺失相同四项。已向用户询问可访问的作者原始备份/
  官方网盘入口；UCF继续进行。完整XD3954仍不能声明ready，未用重新编码或部分数据替代。
- 随后三尾卷全部raw-ready：train2805=515、test=800、train3320=635视频，恢复监督器正常
  退出。新的receipt区分801个ZIP entry与800个视频；consumer已提交`9b4e1f2`，本地
  30 passed/1 skipped（无ffprobe），服务器native联合raw audit/backend/精确分块48 passed，
  包含真实FFprobe/OpenCV测试。SHA绑定上传到新冻结工作区后，核实并终止仅等待旧receipt的
  auditor20108，保留旧记录；新CPU auditor25629实际开始全800 raw坐标审计，最近17条通过。
  它不读取模型分数。监督器完成后磁盘约86GiB可用；4个坏训练成员仍令完整3954不可用。
- P04的a02随后在160/512处失败：profile中的旧UCF根目录缺`Normal_Videos551_x264`。
  并非官方完整视图缺文件。a03只修正输入根，按完整authority逐SHA核对固定64个文件后重跑，
  原样本/签名/阈值不变；旧partial保留且不进入确认统计。a03 run为
  `probe-20260918T110111264470Z-a9e80f00`。未读取任何partial性质值来调整规则。
- P04主量精确定义为“至多256个固定采样非零token的全部非对角对中，cos≥0.90的比例”，
  不是局部邻接或冗余token比例。分析复用旧v3的标签内视频重采样与固定min-count匹配权重；
  通用analyzer的等权匹配组bootstrap不用于本次门禁。新的有源SHA绑定wrapper已通过合成
  完整/篡改/partial输入测试，待完整真实run后执行。
- dense提取新增每视频复用原OpenCV reader，采样、microbatch和FeatureStore语义不变。
  ROOT受影响提取/严格resume42项通过；服务器两个固定fit视频、8/16/64帧、18组的RGB、
  帧索引、mask、timestamps、metadata均逐位相等。共享CPU下耗时方向不一致，不宣称加速；
  当前健康长作业继续原2d9d1a0冻结版本，新实现留给后续运行，不为未证实收益反复重启。
- **P04的64视频/512片段确认已完整完成，预登记门槛未通过。** 原始高相似token对比例差
  为+0.0060656（95%CI [−0.038622,+0.052196]），Hedges g=0.0630
  [−0.437,+0.537]；motion×brightness匹配差为−0.0036728
  [−0.037680,+0.033041]，保留32正常/29正视频。完整输入/权重/角色/采样身份审核通过，
  主点值由独立CSV重算一致，未使用a02 partial。
  见[F03 finding card](decisions/findings/F03-embedding-pair-density.md)和
  [审核图](../../outputs/icassp2027/analysis/p04-heldout-confirm-v1-20260918-a02/audit/p04_video_level_confirmation.png)。
  `analysis.json` SHA为`e71abbbe51b847ca5b01b23be5760fe17d482010e21e6f676c771ed81dc65913`。
  按原登记停止P04干预，不以此启动LoRA；不把未确认解释为零效应或等价。
- 当前A100已转为四fit＋四select dense提取。原encoder dense-only三层clip计时入口已新增，
  与旧四原型入口分开记录，保留原fit/role SHA和FP32、B1/B8、warmup5/repeat30；
  显式native环境核对同时验证解释器和sys.prefix。15项计时接口/统计测试通过。
  它不读取旧gate校准、不是UR-DMU完整detector计时，A100独占执行尚未完成。

## 用户七点澄清后的最新执行决定

- 用户最新明确倾向A，并优先考虑LoRA。计划及Goal正文已补充：保持同一性质驱动的
  TF压缩规则/预算，冻结ViT基础权重和后端，只训练ViT内部LoRA；增加dense同预算LoRA
  对照并实测训练成本，不能用LoRA代替压缩的性质依据。尚未实现或运行LoRA训练。
- 用户明确授权解释七个问题后直接新建任务并设Goal，先前的“等待审阅”状态已更新。
  新任务实际ID/Goal以创建回执为准。新增[性质先行与两版同源约束](decisions/property-first-method-contract-v3.md)：
  统一后端设计但各encoder独立权重，D768/1024由原输入时序卷积映射到512。
- 当前随机保留是对照，线性加权＋MSE是工程原型/对照，**不预定为论文两版方法**。
  必须先确认真实可操作性质，再决定同源的A训练适应或B规则/小网络选择关系，不能先造网络
  再解释性质。先实跑dense基线，再免训练、再可训练；方法开发不用官方test分数。
- 2026-09-18约17:11香港时间，三条XD writer完整命令/owner确认仍运行。按已入账校验分块：
  train2805卷14.678/14.930GB（98.3%）、test卷8.812/10.805GB（81.6%）、train3320卷
  11.134/13.526GB（82.3%）。同node2最近27.86分钟速率约0.386/0.336/0.411 MB/s，
  线性剩余约11/99/97分钟；保守按剩余大文件传输2–3小时，不含后续校验/解压/800 raw
  坐标审计，也不包含4个源损坏文件的修复。ETA不是完整XD已就绪的承诺。
- UCF完整可用，后续dense基线、性质与免训练操作不必等待XD；可训练插件必须在性质/
  免训练机制确认后才进入相应训练，不能把“下载好再训练”代替方法依据。

## 当前审阅门禁与连接验收

- **四组A100真实工程闭环已全部通过**，guard-a02最终completed/exit0；原记录中的
  “其余三组正在运行”已成为历史快照。每encoder两步UR-DMU训练、七组件参数变化和
  严格重载通过，同一冻结head接四条路径均输出有限且时序长度一致。四组每路径20/42/
  20/5 clips，合计348次提取仍只有两个独立fit视频，不是正式质量结果。
  [汇总回执](../../outputs/icassp2027/control/native-urdmu-gpu-acceptance-20260918/summary.json)
  已从服务器取回并逐SHA核验。新任务仍等待用户审阅计划后创建。
- 用户最新要求先解释新旧情况、交付计划，确认后才新建任务并设Goal执行。
  [新计划](RESEARCH_PLAN_V2.md)、[拟用Goal正文](GOAL_NEXT_TASK.txt)、
  [交接清单](NEXT_TASK_HANDOFF.md)已经准备；目前未创建新任务、未修改当前Goal文字，
  未启动新的正式训练或评分。当前继续的是已授权的小规模连接验收和原数据恢复。
- 新后端/data/extraction实现已提交本地Git为`866674c`并上传同名clean服务器checkout。
  联合首轮51项通过；新增raw内容绑定后提取器6项通过；LF/CRLF修复后的backend＋训练
  30项通过。原插件/部署/校准梯度/位置路径回归103项通过；Ruff、compileall通过。
- 原UR源码及两套checkpoint已转入服务器，native foundation CPU严格加载两套均通过。
  处理了Windows源码CRLF与上游Git blob LF摘要差异，实际执行仍是五份精确上游定义，
  没有改网络/损失。原native环境未安装或升级。
- 小规模A100实际guard-a02 PID24912、chain25347，控制目录为
  `code-866674c/outputs/icassp2027/control/native-urdmu-acceptance-20260918T085000Z`。
  首次guard参数5秒低于其30秒下限，在启动child前被拒绝；原记录保留，修正为30秒
  后显式a02启动。最近VideoMAEv2已经真实完成20clips×4路径、UR两步训练/重载和
  同一冻结head的有限输出/shape/权重不变检查；其余三组顺序运行，未报全完成。
- 新UCF1610逐成员CRC/SHA/size/container probe审计和完整view consumer复验已通过。
  contract SHA为`47467e6c88497fbb8f81bc7e2be59cc9e7ed0b6076cdf1548f0e1fe1be5c6998`。
  XD官方3954 inventory实际blocked，available2800/missing1154，不生成完整训练合同。
- 主机RAM充足；共享磁盘最近剩余约183GiB。没有删除任何旧encoder资产。
  [插件继续审查](decisions/plugin-continuation-review-v2.md)明确区分工程可继续与论文
  新颖性/检测质量尚未成立，保留峰值显存未降等负证据。

## 当前正式口径与执行（UTC 2026-09-18 07 时）

- 用户在预训练权重调查后的最新决定：每个ViT encoder只配一个固定检测后端，允许为
  公开预训练encoder训练匹配的dense头后冻结比较。当前4种配对均采用UR-DMU；
  原前三encoder为主，VideoMAE为同家族补充。CLIP多检测后端仅为未采用的建议，
  不进入active profile。四个encoder公开pinned权重文件已实时核对非gated及LFS SHA。
- 新UR-DMU backend根代理实际复验10 tests passed/5.33秒。新的完整dense训练特征入口
  已实现，复用原extractor/FeatureStore/严格resume与采样，工程subset单独标记，未调用
  detector训练或评分；控制器5 tests passed/10.24秒，包括真实NPZ发布/失败不发布合同。
  这些是工程验证；完整训练view和新后端正式GPU训练仍未完成。
- 用户已确认官方划分主实验、直接插入为主、重训头和XD隔离子集为补充，无指定WSVAD
  论文，执行者核验选择UR-DMU。正式后端预算、来源和门禁见
  [协议v2](decisions/official-detector-protocol-v2.md)。下方历史TopKMIL/1288/1598及队列
  “启用”记录描述当时状态；当前旧64项正式评分队列保持research hold，未产生官方模型分数。
- 四encoder均为通用视频预训练表示：V2/VMA/VJ为自监督权重，Time为K400动作分类权重。
  旧弱监督训练仅作用于新增TopKMIL；新UR-DMU必须在各自dense表示上重新训练。
  公开I3D检测权重不能凭维数相同当作已适配四encoder。
- 原作者两套UR-DMU权重已取得，weights_only安全读取均34 tensors/6,492,929元素，
  strict load无缺失/多余key，现有CPU环境合成输入和重载检查通过。正式可复用loader正在
  接入；原作者公开特征尚未取得，不能声称复现了论文分数。
- node3八卡已实际启动本轮工程任务，显式共享的六个child使用0.5 allocator fraction；
  既有其他用户进程保留。新主训练将使用完整dense序列及原作者200bin聚合，旧32片段
  产物不冒充同样时间覆盖。正在核算真实官方训练集提取量，安排有序切换。
- V100四encoder clip计时已全部完成并独立重算96配置/2880samples，9216项数值一致。
  2184份真实缩短/执行回执通过；4项负端到端加速及72个非dense配置峰值allocated未降
  全部保留。该结果是V100参考，不替代A100主计时和完整detector质量/效率结果。
- XD当前三下载卷仍未ready。最新检查test已有686个校验分块+1.229GB前缀，train2805有
  128块+11.616GB前缀；train3320旧PID132714因BrokenPipe/503重试耗尽退出，原253块
  和7.116GB前缀保留。核实错误、源SHA、旧进程退出及无重复writer后，新v2 PID83455
  续用同一已校验存储；test105262/train2805 105261仍在运行。无新完整卷发布主张。

## 最新执行与资源决定（2026-09-18）

- 用户删除goal后已按原任务恢复，现有进程和产物没有重启或清空。用户随后明确指定后续使用
  node3全部八张A100；正在落实显式共卡的普通质量调度与独占正式计时的区分，见
  [资源决定v2](decisions/resource-allocation-v2.md)。已启动V100作业收尾，不再追加新V100 GPU任务。
  此处记录授权与实施方向，不冒称另外三张A100已经启动本轮进程。
- XD训练证据、raw坐标、重复头、缓存评分与质量导出已完成集成，修复了独立审核发现的
  ready manifest绑定、底层PTS证据、晚期raw变化和cached input evidence断链。原篡改
  反例在不改独立封存SHA的条件下均拒绝；正式head合同为v2，逐行对照冻结上游证据。
  ROOT联合验证 **131 passed、1 skipped / 434.68秒**；本机缺少FFprobe的真实小视频测试
  跳过，但node3已有 **25项原生测试通过**，包含真实32帧视频的FFprobe4.4/OpenCV双解码。
  Ruff及`compileall src tests`通过。验证不能替代真实XD模型结果。
- XD raw auditor已在node3以PID20108等待完整test卷，等待时不持CPU锁、无seal；真正审计
  才取得共享CPU lease。快照与解释器已核验，规则先登记。训练视图仍为830/1598 available，
  新绑定回归如实blocked，不生成head合同、不选择未齐confirm中的canary。
- V2 seed2重复头已完成。最近源任务快照中，V2 uniform训练特征完成、验证约125/161；
  VMA全部特征完成，head最终回执待出；Time仍在工程覆盖阶段。VJ四配置仍在提取特征。
  官方UCF test的总原帧数实读为1,111,808；密集窗口数为V2/VMA各69,364、Time138,853、
  VJ17,233，未读取模型分数。预计完成时间依据实际剩余工作更新，不以代码完成量代替实验进度。

## 最新执行回执（UTC 2026-09-18 05 时）

- **V2 dense 已完成固定20 epoch/1620 steps**，每步梯度非零，两个参数张量变化，
  snippet/video logits磁盘重载误差均0。最终checkpoint SHA为
  `cc039fcf673c07b855f19d2c43aff19f5d40e96073f2898b484efaebfa69a717`。
  全部fit/select/canary与最终controller result完成；这不是官方测试集质量结果。
- head I/O优化已冻结为`f3f9c9a`并上传新clean checkout。四encoder各自原生CPU环境用真实
  2视频×32clips、两epoch进行cached/uncached验收，loss/state/logits逐位一致；各模式
  梯度、更新与重载QA通过，读取128→64，CUDA不可见且未初始化。权威汇总SHA为
  `2ed1b3189e1623547a8d5393d3daea3699f609bead1509977b76ec23fc3e6c25`。
- 64项队列在node2/node3实际启用。最初两个V2 repeat/a01因控制参数中的manifest根目录
  少了`/manifests`而在训练前失败，保留原失败；修正路径后显式登记a02，实际strict loader
  与双host参数/依赖复验通过。最新coord为node2 PID11612、node3 PID19950。V2 seed1/a02
  已完成20 epoch/1620非零梯度steps及QA，参数在CPU；seed2/a02顺序接力，尚无完成主张。
  24组配对质量导出另有等待控制器（node2 PID139208、node3 PID1091），共用每host CPU锁。
- GPU3正式clip计时已实际运行。V2、Time完成并独立审核了全部1440次正式sample、4608个
  汇总数值，真实后半6–11层token分别为1568→784、1569→785。纯模型实测加速范围
  V2 1.265–1.298×、Time 1.297–1.318×；adapter和含解码的clip路径收益较小。
  V2 B1的三种缩减方法clip端到端均略慢；两模型全部36个非dense配置peak allocated均略增。
  负结果完整保留。这里不包含MIL head和完整视频聚合，不代替完整detector计时或质量结论。
- XD测试下载旧PID109671在同一分块两次Timeout后退出，431个已验证分块仍保留；新PID129785
  复核并续用，未发布完整ZIP。两慢训练卷经独立11项测试及服务器验收后，逐个核owner/argv/FD
  切换为四路请求，新PID130957/132714；原part分别保留11.616GB/7.116GB，未覆盖或删除。
  新程序还绑定原CD精确字节SHA，逐成员CRC后才create-only发布。数据仍非ready。
- XD训练materialization和评测/重复头/质量导出已具备本地实现，真实训练视图仍为
  830/1598 available，完整167 confirm的最短时长canary规则已预先冻结，当前不选canary。
  独立审核发现ready manifest与上游receipt缺少内容绑定，以及raw坐标consumer未追查部分
  探针证据/晚期原文件变化；均已在合成负例复现，正在修正。未生成正式XD head合同、未开始
  XD训练或模型评分，也不把尚未审核通过的实现当成已完成的第二数据集验证。

## 最新执行回执（UTC 2026-09-18 05 时前）

- 四条中断运行的真实 native GPU 复用验收全部通过：每项两个完整视频、64条特征，复制
  阶段零 encoder 调用，setup一次真实forward单独记录。旧source全文件SHA/大小/mtime
  前后不变；当前bridge、部署回执、runtime、采样和完整1288数据身份全部匹配。汇总SHA为
  `3b5877f181ee8ff9affc17f83c4e037dde36095efa55aaede65e86097202517b`。
- 四个新run已在node3真实运行：V2 uniform GPU1/guard19272，VJ uniform GPU2/19273，
  VJ dense GPU3/19275，VJ paired-random GPU4/19307。原VJ paired-random也是外部GPU
  争用导致原guard中止，保留55完整视频与partial，不冒充健康完成。首批正式复用逐条核对了
  新resolved SHA、旧index快照和数组来源；VJ learned另在GPU0/guard2795启动。
- 两节点source训练协调器已恢复（node2 PID100288、node3 PID971），原失败与superseded
  关系均有回执，node2 GPU3计时预留不变。V2、Time、VMA的fit1288/select161特征均完整；
  V2的3350 clip工程canary已完成，head目录已建立。该runner在末尾才写history与QA，
  因而没有head文件不能判定尚未开始训练；完成与否仍只依据最终QA/检查点。
- 64项UCF官方评测/重复头/缓存评分的两节点真实parser与依赖审查通过，16个seed0
  controller尚无完整head完成回执。队列暂未启用，等待head读取优化的新冻结源及native
  验收；没有提前访问模型测试评分。完整视频效率控制也已部署并保持禁用等待依赖。
- head训练增加仅paper显式开启的pooled序列内存复用。首次仍完整校验FeatureStore SHA
  和序列，全部成功才缓存，返回独立数组防止后续修改。原通用入口默认关闭，token路径
  与多worker缓存显式拒绝。两epoch小规模对照的loss、最终参数和logits逐位一致，
  load_bundle次数从16减为8。包含既有UCF来源/质量导出的联动验证为
  **72 passed / 199.66秒**，本机Python3.11、Torch2.13.0+cpu；服务器验收待执行。
  这是训练I/O优化，不计入插件推理收益，也不修改当前运行中的冻结代码。

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
