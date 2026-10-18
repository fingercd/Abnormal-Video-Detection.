## 2026-10-18 研究交接与Git归档（不启动新实验）

用户明确要求保存代码、文档、完整结果到新Git分支并交付ZIP与中文研究咨询prompt；已另明确授权推送当前公开origin的新分支。分支qzt/encoder-compression-handoff-20261018；范围为V2/CLIP/VMA/TS三轮128个提取预测条件、120项压缩质量比较及相关历史证据。原实验身份、冻结SHA与数值均不改，uniform取消保持，不恢复已完成goal。

公开导出目录projects/icassp2027/exports/compression-review-20261018保留三个最终报告原字节副本（17个核心原SHA核验通过）；聊天与原始运行日志只进入本地交付ZIP，不公开提交。交付根work/handoff-encoder-compression-20261018，包含中文prompt、阅读入口、逐文件SHA/原路径清单、排除/缺档清单及ZIP验收回执。引用ChatGPT对话仅取回用户原问题及助手content-reference占位，完整旧回复缺失如实披露。

本次python -m compileall -q src tests通过；纯结构/资源mock/原报告门10项测试通过（Python3.12.12，2026-10-18），没有执行模型、训练或质量重算。保留其他未跟踪个人材料与临时脚本；Git提交/推送结果以交付GIT_STATE和push-receipt为准。分析需分清post-attention与pre-block、XD stepAP与梯形PR、固定head条件与跨seed结论；prompt要求多agent（可用时）做实际数据处理、近期文献/协议核查和条件式研究规划。

## 2026-10-17T08:44:50.033146+08:00 全部完成：用户调整后52/52预测、48/48质量、156+156效率与MD/CSV/JSON及原云端最终核验

本goal按用户后续明确取消uniform及允许8卡共享的最新范围完成，不把原goal旧文本76/72误报为全部做完。原76行/72质量行与24取消记录完整保留，52活动提取预测全部完成，48配对质量全部只计算一次，活动clips8394113/8394113=100%。24uniform无正式extraction-contract/prediction输出，未重启或重算；取消前已完成的72encoder+72pipeline uniform效率组保留历史。原228+228效率组全部真实计算/审核/云FINISHED，按用户范围活动156+156全部有效，不因正式质量/速度挑格。

最终严格审计通过：278个实际源文件SHA全匹配；52个原远端run exit0、原Swan URL逐一一致且原任务PID不再运行；24取消uniform无正式输出。4个原fulltrain head及training_qa实际SHA匹配，UCF1610/XDaccepted3950；2encoder各36条件classic-native原回执SHA与云端qa/completed=1核验，逐层真实input/MLP shapes、pre6/9/12与signal5/8/11、TimeSformer8帧/空间轨迹/原backend均通过，未重做工程gate或模型实验。全部52+48原云端成功及48主dense/method/delta全精度回读一致，原efficiency会话456计数终态有效。

报告最终目录work/preblock-vma-ts-20261015-r01/reports/final-active52/，包含FINAL_REPORT.md、quality-efficiency.csv（48行）、final-results.json、encoder/pipeline-efficiency-active.csv各52汇总行、artifact-sha256.json、source-audit-reconciliation.json、cloud-verification.json。7个核心文件SHA逐一实际匹配；最终报告原Swan s2esfubw（https://swanlab.cn/@ryfrcd693/VAD-PreBlock-VMA-TS-20261015/runs/s2esfubw）FINISHED且report/completed=1、formal52、quality48均API确认。first最终源审计因UCF历史契约旧路径消失而真实失败；在现存/users/fotile/icassp2027-runs/project/outputs/icassp2027/control/official-fulltrain-20260918-r01/ucf-view/contract.json找到原SHA47467e6c...逐字节一致副本后，原科学路径/契约SHA不改，以build_active_final_report_r02复用同s2esfubw成功，旧失败源和日志保留，未重建契约或重算模型/指标。最终原件路径与实际副本关系完整披露。

主CI结论{'inconclusive': 33, 'supported': 13, 'inferior': 2}：13支持非劣、33证据不足、2确认劣于−0.5pp界限。VideoMAE-UCF12格全部inconclusive；VideoMAE-XD5支持/5不足/2inferior；TS-UCF3支持/9不足；TS-XD5支持/7不足。两inferior均VMA-XD VisionZip pre9，remove50主PR68.544978%（Δ−1.755101pp，CI[−2.551950,−0.967874]pp），remove70主PR67.584105%（Δ−2.715974pp，CI[−3.779466,−1.678597]pp）。同配置两dataset主CI均支持的为TS Attention pre9/r50与TS VisionZip pre9/r50，但XD完整视频倍率各0.899851x/0.913119x，反而更慢；没有同一配置两dataset同时满足主CI非劣且两dataset观测pipeline均>1。三轮速度倍率只是描述性结果，无速度显著性或多配置校正；单head seed0不称泛化。XD stepAP补充下降全部保留，不能因主梯形PR点值接近dense声称全指标保持。

八卡共享阶段未观察到OOM，全部保留原运行日志CUDA OOM/OutOfMemory/reducing batch检索0匹配；其它项目1827、26335、11616、25402/25403、后续32113等不停止、不改。两本项目生产主控163748/CPU143152均自然退出（队列all_predictions_completed/completed），没有重启正常任务；原key-attention-40定时任务实际toml status=PAUSED，不再运行，不归档聊天。未commit/push/公开发布/投稿。最终completion-audit-active52-20261017.json记录完整审计与核心SHA，goal已由工具确认complete（累计执行164697秒，约45小时45分钟）；无剩余必需实验或交付。

## 2026-10-17 01:49 HKT 八卡共享稳定推进36/52、32/48；持续无OOM，已出8项VisionZip质量

上轮用户授权后的8个生产slot0..7均持续真实产出，本轮完成一次实测30分钟窗口及随后2小时对原live主控163748/CPU143152的有界等待。36/52=69.23077%预测、32/48=66.66667%质量已完成；24uniform仍用户取消。32个原quality-preview结果SHA复用/回传，API逐项核dense/method/delta与本地全精度一致；当前6supported、26inconclusive、0确认劣于margin，不据正式质量改方法/head/位置/预算。快照6147178/8394113clips=73.23201%，每task一次。当前8任务均live、两主控stderr空，无OOM或卡重投；主控实际多次自然接力到VisionZip pre9/pre12。其他项目GPU进程32113等共享出现也只记录、不拦启动、不停止，按最新用户授权执行。

八路30分钟完成视频clip吞吐窗口1817.317秒：GPU0 VMA-UCF VisionZip pre6/r70 15.3242clips/s；1 VMA-XD pre6/r50 16.1276；2 TS-UCF pre6/r50 17.9715；3 TS-XD pre6/r50 20.8852；4 VMA-XD pre6/r70 16.7731；5 TS-UCF pre6/r70 27.7475；6 VMA-UCF pre9/r50 16.9216；7 TS-XD pre6/r70 25.1453，合计156.8961clips/s。仅共享生产统计、完成视频边界口径，不冒充独占encoder或方法加速倍率，不把0/1/2/3与其他时段混作严格速度对照。依据实测encoder/dataset/method/budget和原独占encoder位置倍率仅作为未知CPU/GPU占比的规划上下界，再按8slot当前内存偏好模拟剩余队列、包含merge/head/报告云端尾部，eta-eight-shared-20261016-2340.json给出10/17 08:22至17:10 HKT规划范围。该范围不是CI或保证，新OOM/其他项目负载/网络可延后；不能简单把旧四卡ETA除2。

已完成的8个VisionZip配置均主CI未支持非劣。特别保留XD补充反向结果：VMA-XD VisionZip pre6/r50主梯形PR71.522772%、Δ+1.222693pp，但CI[−0.676154,+3.151267]pp，inconclusive；stepAP64.840107%→58.009829%，下降6.830278pp。pre6/r70主PR70.196888%、Δ−0.103191pp、CI[−2.314961,+2.156831]pp，inconclusive；stepAP降至53.622280%，下降11.217827pp。不能因主梯形点值接近dense就省略stepAP下降或声称质量保持。全部保留在预览MD/CSV/JSON，不更换冻结主指标、不重算bootstrap、不利用该结果停掉其他保留配置。

效率组件和最终报告限制补明确：encoder-only仅用固定UCF训练正常视频Normal_Videos001_x264、B1/B8各三进程，XD质量行复用同encoder时序，并非另测XD encoder；pipeline才按两数据集各16完整训练视频独立测量。只补说明并更新派生artifact SHA，原156+156活动效率数值和原456测量/原云身份不变。最终报告门5项纯结构测试通过，尚未执行全量最终源核验/生成正式最终报告；goal active，未commit/push/恢复旧定时任务。

## 2026-10-16 23:09 HKT 用户明确授权直接共享启动：已真实8卡产出、29/52预测与25/48质量，无OOM

最新用户要求“直接启动就行爆显存就换卡 不爆”覆盖此前由主控设置的未知PID/24GB空闲保护门。已绑定user-shared-gpu-policy-20261016.json SHA6c8f5c9ae9357c9571b1cd71308b5d49d04c4f1ec5c9da2085de6c1abfe5e5df：共享其他项目进程是允许的，其他PID及空闲显存只观察记录、不再阻止启动；不杀/修改其他项目、不改方法/head/采样/预算/B8。OOM时仅处理本项目失败项，先核原PID真正退出并保留已完成视频及原Swan，随后换卡，禁止原进程未退出就重投或同卡盲目循环。当前尚无OOM，不宣称已经做过OOM恢复实测。

新controller-r14/PID163748（formal-r02-user-shared-eight-slot-r18），CREATE_BREAKAWAY_FROM_JOB|CREATE_NO_WINDOW，接管四原logger和原Swan、不重启它们；新增GPU4/5/6/7已实开而非只写slots。GPU4 VMA-XD VisionZip pre6/r70、GPU5 TS-UCF VisionZip pre6/r70、GPU6 VMA-UCF VisionZip pre9/r50、GPU7 TS-XD VisionZip pre6/r70。5/7按已测更低显存的TimeSformer先排，以减少OOM机会，不改变任何科学配置或按分数选择。四新增均通过原兼容preflight、持续完成真实clips，8个logger均live、stderr空；最新free MiB为0:4100、1:27475、2:28659、3:28659、4:15263、5:1317、6:15191、7:1317，新增5/7各实际占约2GB并未OOM。其他项目所有PID保留；当前为8个有效本项目任务、每卡一个，不能将共享生产吞吐计为独占效率基准。

此前20:47/20:59/21:34三项VisionZip被旧GPU门exit22拦下，日志没保存瞬时PID，故不能回溯断言是其他项目还是上下文清理；22:35实查1/2/3已空。原三个remote目录只有swanlab-url.txt，无run.sh/PID/output，逐项证明从未GPU执行，精确保留空目录为.preserved-prelaunch-guard-20261016-2238后，新wrapper_r02按固定三身份manifest SHA06490a8176dca70dfcb33a2a0792079a0033f08048db1bce3ad518efd5ee5a23及运行时absence再查，仅复用alplvlm9/3keufwre/qgevlkzm第一次执行，未新建Swan替代或重复提取。恢复后原1/2/3任务持续真实clips。旧门已完全撤去，新版资源探测仅记录；3项新共享策略测试通过，旧资源等待策略测试作为历史不改写。

正式29/52=55.76923%，质量25/48=52.08333%，clip快照4722263/8394113=56.25684%，每task一次。全部24项Attention现已完成提取预测及配对质量，首项VisionZip VMA-UCF pre6/r50也完成：ROC80.256678%、Δ−0.292064pp、CI[−2.034780,+1.683535]pp，inconclusive，原Swan41lhhz1m FINISHED、结果SHA706f78bb615497e6ac4bb4de8977285ad9c92f2a4455b17cf1cbfd31f511f。25份原preview均复用/回传并API核三主点，当前6supported、19inconclusive，未据分数改保留配置。原24uniform仍用户取消、不计完成；48质量分母不变。

CPU主控143152继续；当前8并发刚运行，不将旧4卡ETA简单除2，后续按8并发实测clip吞吐重算。最终报告门代码/源审计尚未最终执行，活动156+156原独占效率结果保持不变；全scope52/48及最终报告云端尚未齐，goal active。未commit/push/恢复旧定时任务，其他项目无修改。

## 2026-10-16 18:33 HKT 跨turn存活与实际接力已验证；24/52预测、20/48质量，完整报告严格门已准备但未提前生成

上轮已启动的detached主控164200、CPU主控143152及4原logger在本轮起点全部以原PID存活，跨turn不再出现上一组无traceback消失现象；不据此宣称已穷尽原外部退出根因。对原live进程做一次有界30分钟等待后，TS-XD Attention pre9/r70自然完成291781clips及固定head、原Swan成功，GPU3真实补到TS-UCF pre12/r70；VMA-UCF pre12/r70也自然完成，GPU0真实补到TS-XD pre12/r70。4slot仍0/1/2/3，GPU5/7其他项目VLLM不动；原nm66pa10在GPU2持续首次提取，没有新建另一个Swan替代旧身份。两主控stderr空，无通信失败/自动GPU重投，24uniform保持取消。

本轮正式24/52=46.15385%，CPU质量20/48=41.66667%，已完成20个都重新以API回读dense/method/delta与本地原SHA核验一致，原18不重算bootstrap；新增两项为原blr299ak/hk0kw3oz。TS-XD pre9/r70主梯形PR76.343447%、相对dense−0.312396pp，95%paired CI[−1.089844,+0.452796]pp，inconclusive；stepAP70.838508%→71.807443%。VMA-UCF pre12/r70主ROC81.563261%、相对dense+1.014519pp，CI[−0.829626,+2.881710]pp，仍inconclusive。当前4supported、16inconclusive，不把正点差或小负点差等同非劣。quality-preview/CSV/JSON已更新20，实际clips快照3881833/8394113=46.24471%，每task一次。四当前任务为TS-XD pre12/r50、VMA-XD pre12/r70、TS-UCF pre12/r70、TS-XD pre12/r70；VisionZip后续原完整24格仍待接力，未据正式分数调位置/预算/head。

已准备build_active_final_report.py及5项纯结构测试，真实--check-only仍明确ready=false；没有提前建立“最终成功”云run或写出最终报告。严格门要求全部活动52预测和48质量、原24uniform全取消；核实际48原quality JSON SHA/seed0/10000/margin、全部52+48 fresh云端状态与主点、原效率456云端和活动156+156审核SHA，再只读核原绑定源文件及四head/两truth的实际SHA，canonical truth符号链接另做resolve一致性，随后才组装MD/CSV/JSON并在线保存。所有数字仅查表复用，代码没有提取、模型、head训练或bootstrap调用；尚未跑最终全量源核验，不能将纯测试/门代码当已经完成验收。Markdown真实换行及不完整/取消任务拒绝门已测试。

研究披露措辞已补准确：uniform排除发生在用户查看旧VideoMAEv2/CLIP正式结果之后，这是明确用户授权的后验范围变更，不应写成原完整三方法预注册矩阵或完全盲选。保留Attention/VisionZip全部原位置/预算、未根据当前新encoder质量调参数；不能凭此推断本轮未测uniform效果。只修说明、活动效率组件artifact SHA清单同步更新，所有实验数值不变，原冻结scope/campaign SHA不改。

本轮5项报告门测试通过，compileall src tests exit0；效率活动156+156组件原始审核/原云FINISHED仍有效。整套全量结果和最终报告尚未完成，goal active。主ETA仍按当前四生产slot、未测VisionZip正式吞吐与网络/共享资源的不确定性保留10/17晚上至10/18上午规划，不宣称精确截止。未commit/push/恢复旧定时任务，其他项目不动。

## 2026-10-16 17:55 HKT 四卡原身份接力稳定、22/52预测与18/48质量；效率活动156+156已全审核并云端核验

本goal turn有实质完成：原efficiency r4ef7f3u经已GET核实的原internal CUID恢复后，原存储标量efficiency/completed=1、encoder228、pipeline228均可API回读；原会话FINISHED。efficiency-original-cloud-final-verification.json保留fresh API三点及raw tail；没有重跑任何模型/计时。audit_pipeline_results.py将228原pipeline JSON逐一绑定原456queue、group_completed EVENT SHA、真实PID、head、原16完整训练视频/采样、原GPU5独占前后、warmup与内存；同时把原228encoder字节重新绑定同一parent/queue/events，全部通过。效率活动scope只过滤用户取消的uniform，156encoder+156pipeline已审核；原72+72uniform组原位保留不重测。reports/efficiency-active-scope-20261016输出MD/两CSV/JSON/SHA清单，是已完成效率组件，不是整套最终质量报告。

真实净效率限制明确：VideoMAE encoder B1范围1.0049–1.5291x、B8 1.0448–1.5960x；TimeSformer B1 1.0048–1.5099x、B8 1.0352–1.6323x。原完整视频pipeline的VideoMAE UCF范围0.8524–1.1818x、XD0.9805–1.0364x；TimeSformer UCF0.9637–1.2495x、XD0.7891–1.0442x。小于1的反例保留，不能把encoder省时或CPU生产打包提速称作整体必然加速。pipeline原native CPU路径、FeatureStore落盘/权重加载/SHA预查排除、共享CPU/存储和页缓存未控制限制完整保留，三次描述性配对中位数不冒充速度显著性。

本地旧observers在换turn后消失且无traceback，原因尚未完全证实；远端原GPU/PID均继续正常，不能以本地PID消失重投GPU。已用支持的CREATE_BREAKAWAY_FROM_JOB|CREATE_NO_WINDOW启动独立观察器171992/178396/179852/25784，原PID/start_ticks/log inode/EVENT前缀和原Swan全部匹配；采用公网12345原线路（新一次hostname/业务检查已恢复ibnode3）。备用node3路由此前banner超时，原失败记录保留，不无限重试。新detached controller-r12/PID164200以scope SHA约束52/48，只管0/1/2/3四slot；quality controller-r02/PID143152走同公网并独立运行，实际完成3项后又完成第18项，没有重复原14个bootstrap。跨turn持久性下一轮继续验证，不假报已穷尽本地退出根因。

GPU5与7在原效率完成后被其他项目新VLLM TP进程25402/25403占用（yangrunze，各35974MiB），已真实映射owner/command并保护，不能继续宣称六卡有效产出。GPU5曾先建立原Swan nm66pa10后SSH失败，fresh原线路明确证实remote run/PID/output全部不存在；失败观察器已自然CRASHED，没有任何GPU提取。对这个唯一已证实未启动项保留原nm66pa10，以独立first-launch wrapper再核原路径不存在、只复用原cloud身份、所有step严格后移并明确原失败已恢复；在GPU2原VMA-XD pre12/r50自然完成后，主控已真实将其首发到GPU2，原nm66pa10、相同科学配置、完整preflight通过并持续新clips，不是重提已计算任务。wrapper两项纯日志步数测试通过，源码已先在线保存；不修改原冻结formal tracker/helper/四pair执行bindings，质量短poll独立_r02版本。恢复中被绑定门拒绝的本地helper变更已从原release精确字节恢复，相关控制器失败均在GPU派发之前，证据保留。

正式完成22/52=42.30769%，质量18/48=37.5%；累计3722920/8394113clips=44.35156%，只统计活动scope，每task一次。四任务继续：TS-XD Attention pre9/r70已291781clips、尚需global merge/固定head；TS-XD pre12/r50；VMA-UCF pre12/r70；VMA-XD pre12/r70首发原nm66pa10。尚未运行VisionZip正式格，未因分数改顺序或跳配置；24uniform保持用户取消，取消不计完成。

18原质量结果均API回读主dense/method/delta与本地全精度一致，collect只复用/回传原quality-preview，不重复bootstrap、不覆盖quality.json；预览MD/CSV/JSON已改52/48活动分母并明确用户排除，当前4supported、14inconclusive、0确认劣于margin。新增VMA-XD pre9/r70主梯形PR68.603887%（dense70.300079%），Δ−1.696191pp、CI[−3.006130,−0.469188]pp；虽CI低于0，但仍跨−0.5pp，判inconclusive，stepAP64.840107%→61.880926%。VMA-UCF pre12/r50 ROC81.194278%、Δ+0.645537pp、CI[−1.244695,+2.593239]pp，inconclusive。TS-UCF pre12/r50 ROC79.426400%、Δ+0.746334pp、CI[−1.195827,+2.882448]pp，inconclusive。VMA-XD pre12/r50主PR71.186301%、Δ+0.886222pp、CI[−0.062788,+1.863315]pp，支持非劣但不证明高于0，补充stepAP反而64.840107%→63.992633%，不能称全面提升。原Swan分别erb3qs34/c5i6qlf4/pe4aoexo/q4wlqlrb均FINISHED，原SHA索引完整；不据这些正式结果调参。

eta-active-scope-20261016-1755.json按26pending与四项实际剩余clips、各encoder/dataset既有耗时区间重算：当前只有四有效生产卡，活动整套含报告/云端规划10/17 17:02至10/18 07:21 HKT，用户表达以10/17晚上至10/18上午为主。VisionZip优化后全test生产吞吐尚未测，CPU/I/O共享与网络仍有不确定性，区间不是CI，不能保证最早边界。28质量调度测试+2日志offset测试通过，compileall src tests exit0。全scope52/48/156+156及最终全量报告仍未齐，goal active；未commit/push/恢复旧定时任务，其他项目和旧资产不动。

## 2026-10-16 17:00 HKT 用户明确取消后续uniform：活动范围52预测/48质量，24项未启动格真实取消

用户明确要求“我觉得可以不用做uniform听我的，之后相关uniform的不用做了”，覆盖此前三方法执行范围。已创建user-scope-20261016-no-uniform.json（SHA ba513144248751835cab0b1bfb65b1a1c1ed2a7870e654abc25d192d4115df0d），引用原scientific campaign-r02 SHA不改；精确排除24个uniform任务，保留原76/72行历史，不把取消伪报完成。活动范围为2encoder×2dataset×2method×3位置×2预算=48压缩，加4dense共52提取预测、48质量。原quality queue24个uniform也全为pending，现已标cancelled_user_scope_change；14个completed原结果完整复用。用户授权范围变化已执行，不能因旧goal文本仍写76而恢复uniform。

仅在核对精确本地controller7532命令后停止该调度器以原子更新队列，没有停止任何GPU/非uniform原logger。新controller-r09.py/PID94240以manifest SHA约束活动范围并接管4个原logger/原Swan，后续只调度Attention/VisionZip；真实观察到TS-UCF pre12/r50原Swan完成后自动补到VMA-UCF pre12/r70。24个uniform均未启动、无GPU进程需要杀。其他项目、旧VideoMAEv2/CLIP uniform历史结果及原协议不删不改。

效率原228encoder+228pipeline已全算完；pipeline-results-r01.tar.gz SHA0357eb4474ad7436ae17d86155cc67ae1457c8141a010a0ea5595a439cb8e2dd、原456queue SHAe25b78a77088144b6a6daa4d369e5cc72855bfa11ffb8685830364c2525943b7回传精确；原228pipeline原始值/成员/输入采样/head/GPU独占/逐组EVENT SHA纯审核通过，76描述性汇总产生但标cloudpending不当最终验收。已算uniform效率72encoder+72pipeline保留历史，不重跑；活动Attention/VisionZip+dense各156encoder/pipeline组可直接审核复用。

原u90yf3ez同云身份补传核验已取得真实新证据：原二进制内test/completed=1/297、failed=0/208、clips69364/294经API可查，fresh API state FINISHED。public slug POST未真正恢复终态，使用已GET核原内部cuid的同run POST后state转RUNNING且此前保存在服务端的点可见；不据此断言所有后端根因，原完成时间/指标未伪造，不重算GPU或bootstrap。u90-cloud-verification.json为新独立回执，旧失败回执原位保留，待主控安全接纳后补GPU7。此前直接PUT RUNNING被400拒绝，未改变状态；脱敏失败证据保留。全局SDK/凭据未修改。

后续只继续Attention/VisionZip，head、位置、比例、采样、原科学SHA不变。goal active，最终交付按用户新活动范围52/48/156+156，历史76/72/228+228及取消记录同时披露；不把明确取消当失败或计入完成。未commit/push/恢复旧定时任务。

## 2026-10-16 16:30 HKT ETA更新：计算实际20格结束、正式入账18格；原身份云端/补位恢复中

本次用户询问预计完成时间，按当前53项pending、3项仍提取和2项计算结束但尚未主控入账重新核算。正式队列18/76、质量14/72；另VMA-XD Attention pre9/r70原Swan rgtsdfdm已FINISHED且test/completed=1，待接管入账；VMA-UCF pre12/r50原计算exit0/完成EVENT存在，但u90yf3ez恢复后云端缺completion标量，暂不计正式完成。累计clips按每logical task一次为3369790/12268319=27.46741%，不叠加copy/resume前缀。原TS-XD pre9/r70、VMA-XD pre12/r50、TS-UCF pre12/r50继续；公网传输故障后已使用可信node3备用线路恢复原PID/原Swan观察，没有重投GPU。主控自动补位当前held，不能描述为五卡均持续产出。

原GPU5效率228 encoder+228完整视频组均已原始计算结束，但原r4ef7f3u云端completion/counter新标量未入库，原始archive审核/最终云端确认未齐；GPU5尚未加入生产。只读诊断确认新scalar已实际写入本地.swanlab且通过SDK step检查，异常位于持久化后上传/后端链，具体根因未定，不能假报已恢复。已有完成结果/身份不变，不重算ML或为了绕云端另开run。

ETA派生记录eta-planning-20261016-1630.json：按各encoder/dataset已有实测范围和剩余clips，五卡补位恢复后的整套规划仍以10/18白天至晚上为主、保守10/19上午；如果GPU5原效率审核/原云端核验结束并获空闲加入，第六卡条件窗口约10/18凌晨至晚上，不能提前承诺10/17结束。包含76格提取预测、72质量、456效率组审核、最终报告/云端；VisionZip/uniform完整优化任务吞吐未测、CPU/I/O共享和通信恢复仍带不确定性，区间不是统计CI。当前goal保持active，未缩小范围/调正式方法/head/预算，未commit/push/恢复旧定时任务。

## 2026-10-16 14:00 HKT 16/76预测、12/72质量；五卡接力已实开，整套规划10/18为主、保守10/19上午

主控已更新controller-r07/PID174560（formal-r02-execution-r07），生产slots0/1/2/3/7，每卡一任务。新GPU7启动VMA-UCF Attention pre12/r50，原GPU6有foreign PID11616/20670MiB，不使用；GPU4 foreign26335、GPU0保护1827保留。依据用户“尽可能优化”及AGENTS既有全16空闲资源授权，只在实查GPU7无任何PID/40326MiB free后加入，启动器再次查UUID/空PID/free>=24576MiB；原四GPU任务/五个既有Swan不变。GPU5仍原效率独占，未提前占用。node3实测CPU约48.98%、iowait0.084%、load13.25/32逻辑核，未因扩1卡过度占满CPU。

Future pending四pair供数都已放行r03动态原生8/16分组，原B1无重复np路径直接复用，避免r02对TS多pad16的倒退。四原真实训练门均exact并FINISHED：mxd81hnh（VMA-UCF）、0vc5htvq（TS-UCF）、l0rfzkj6（TS-XD）、4oveb7dm（VMA-XD）。B8纯prepare倍率1.44175/1.48859/1.47779/5.36050；B1约1.015/1.035/1.008/3.717，不把这些当整任务或encoder加速。r03候选SHAa9675c99aeecd1fe63565da5a3be6f14cf255558d136f9b88be1a8cf542e0240；4pair release bundle2a50434ac31d2cb03fb9807843e6d919a01c72bb4622d9705c250832365c9fa5，原科学campaign/source/head/方法/预算/采样SHA不改。现有两个VMA-XD继续已经上线的r02，不为换小版本再次重启。

新增track_preblock_online.py对新formal先online+单次nohup，截断长tail后只读短SSH poll；初始launcher255只尝试原URL/PID身份探测，不重放launcher，认证/主机指纹错误立即hold。3项fake-poll分块/终态/日志inode测试通过；实际GPU3原TS-XD pre9/r50自然完成后，新TS-XD pre9/r70已经由主控r06以此链真实启动并持续输出，Swan d28eiow2。GPU7新任务u90yf3ez同链有效。旧4个恢复链源SHA/PID/start_ticks/原Swan绑定完整，未因传输故障重提GPU。

原TS-XD pre9/r50至此真正完成global merge、原固定head预测及原Swan c4upi8u3 FINISHED，正式16/76=21.05263%。新增第12个CPU质量结果原Swan toeurlgw FINISHED，原resultSHA33d22444c6441d4f6ebb61044358fe51affb37422aa720dff56c9af6dd505c13回传与原EVENT、云端主三点一致，原其他11结果仅验SHA复用。TS-XD pre9/r50主梯形PR dense76.6558428832%、method77.1697329324%、Δ+0.5138900493pp、95%paired CI[+0.0540231222,+0.9879924836]pp，supported；stepAP70.8385083610%→72.0089727788%，Δ+1.1704644179pp。与既有TS-UCF同pre9/r50 supported组成首个同encoder/同配置两dataset主CI均支持非劣，但仅单head seed0/多配置未校正，UCF CI包含0，不能称两dataset均证实显著提升。总计3supported、9inconclusive、0确认劣于margin；方法/head/预算无依据分数调整，预览MD/CSV/JSON更新12项、不重bootstrap/不覆盖quality.json。

收尾瓶颈更正：此前仅last clip progress时把TS-XD称为“head收尾”过早；只读审计证实其先在CPU global merge，非O(V*N)读全局index。800个video索引共1.226GB，约4202B/clip；video publish和global各一轮NPZ SHA+解压，再进入head读取。原atomic_write_jsonl还全量积累JSON strings/join/bytes。新streaming merger仅候选，真实r01小例3weaofow全index/arrays exact却0.8904x（更慢）；减少冗余resolve的r02在独立copy完整Normal924 6749clips+Shooting032 1355clips共8104上也全index/arrays exact/source unchanged，但总merge29.5015s对原27.8288s，仅0.9433x。global单段13.3358→11.5422s，RSS610076→368368KiB，per-video额外校验导致总耗时倒退。两失败性能反例/代码/Swan原结果保留，未上线此merger，没有跳过任何原校验或覆盖运行中index。实际不缺RAM，不能把降内存写成提速。r02仅系统调用计数降低，23synthetic tests通过不替代真实吞吐结论。

14:00累计2840091/12268319clips=23.14980%，每logical task一次。五项：GPU0 VMA-XD pre9/r50 120246/145703（82.53%），GPU1 r70 62790/145703（43.09%），GPU2 TS-UCF pre9/r70 125489/138853（90.38%），GPU3 TS-XD pre9/r70 24573/291781（8.42%），GPU7 VMA-UCF pre12/r50 528/69364（刚启动）。两个VMA-XD最近1079.2s、已排除copy前缀的实际新clip速率12.8697/12.9800clips/s，旧约2.6–2.7；仅生产统计非正式算法倍率。效率原encoder228/228、pipeline214/228、恢复Swan r4ef7f3u仍RUNNING，待终态才可释放GPU5。

ETA依据eta-planning-20261016-1400.json：55pending+5running，按各自encoder/dataset实测wall/新VMA生产吞吐及原merge/prediction尾部，规划余172.7–258.9 GPU小时；当前5卡理想34.5–51.8h，再留负载尾部/报告/云验约4–9h，实际目标范围10/18 04:30–10/19 02:50 HKT。对用户简报“10/18白天—晚间，保守10/19上午”；若GPU5原228+228全部完成/审核/原云端FINISHED后加入六路，有利情形可提前到10/17深夜—10/18上午，但不是承诺。VisionZip/uniform优化后整task吞吐未完整实测、共享CPU/I/O/其他项目负载和网络仍不确定，不用CPUprepare倍率直接除整套时长。全76/72/456、最终报告/云端尚未齐，goal active；未commit/push/恢复旧定时任务。

## 2026-10-16 13:14 HKT 通信观察链恢复原四PID/原五Swan；优化remove70已进入新提取，新增三pair CPU门通过

13:02前后四条formal和原效率的持久SSH同时Connection reset，Windows logger退出，observer142864按安全规则hold四slot；并非GPU任务失败。一次node3原PID核验确认bash12250/12240/27487/21226及效率父10074/child10329均仍live、无exit-code，未重投GPU或重复benchmark。此类故障不因进度观察中断重启任务；原已故意取消的旧VMA h7xmqfsw/v884d11z云端CRASHED/test_failed=1也如实留存，不混成当前运输故障的恢复对象。

新增transport-recovery-r01/reattach_stream.py，仅短SSH只读/proc与原log、原URL及exit/queue；PID+start_ticks/owner/完整command marker和log inode固定，递归tracking链精确匹配所有已上传EVENT prefix，只补新events；每轮传输max3、认证/主机失败立即停，不启动/停止任何远端任务。synthetic5项及原TS-UCF真实189个EVENT prefix只读验证通过。五个原Swan均resume must、ONLINE恢复原身份；实际源、provenance、recovery先上传，结束仍须原FINISHED+completion1。API回读13:10五run全部RUNNING：VMA z7qo6cbv/rgtsdfdm、TS c4upi8u3/bdgmc4w0、eff r4ef7f3u，prefix全部匹配；失败scalar恢复置0，历史失败事件及原日志保留。观察层故障原因未归咎具体网络/代理。

新observer143868/controller-r05，formal-r02-transport-r05，精确支持新tracker源SHA/新logger创建时间而保留原task started时间/行内expected_snapshot/source和Swan；四logger161688/101140/165092/166368，效率logger160248，质量主控102460均live。实际remote任务PID未改变。当前status_snapshot递归组合旧tracking+恢复新events，已保留旧前缀max计一次，不把reattach/复制当新吞吐。

真实复用检查：优化r70已越过191个原完整video/17690clips进入新GPU提取（13:14之前已发表247video/25380clips）；r50仍逐数组独立copy原611video/87078clips，检查时514video/70208clips已发表，未宣称全部copy结束。原TS-UCF是1小时Normal_Videos_924_x264长视频，PID27489实际R、3秒CPU205ticks并持续读取，当前完整block索引已有13499clips；video级progress在publish前不跳，不能因几分钟无event判挂。原TS-XD已291781clips提取完成，正常固定head收尾，未重启。

同时已有相同r02候选在其他三pair的真实训练CPU exact门（串行，不建model/GPU）全通过并FINISHED：VMA-UCF h336zw27、TS-UCF r0nnn6up、TS-XD 7fhb614l。实际acceptance原SHA分别6adbb92bce73b80e7ccdcf252692f401e888c04486db1c3bba7d945392048662、4db3e6e3a92fde6b74edc2267c066f3a10328117d8b1a54c37916dfb4b70080c、690e690c4185b75179a7825dbaa7d288ea93dd88f83bb0a77769c753b5baa578，均对原completed EVENT结果SHA核验。B8纯prepare倍率1.4356/1.2434/1.2390；TS B1反而0.5170/0.4978，原因候选固定16帧分组对native8帧多pad。结果如实保留，不把它推广为全面加速；这三pair尚未接入正式新执行，继续改进有据的执行打包（非方法选择），现有TS不动，原r02候选/真实门不覆盖。

13:14正式15/76、质量11/72；累计2709338/12268319=22.08402%，每logical task一次。效率当前原encoder228/228、pipeline208/228（恢复prefix+新events口径），原会话仍RUNNING未终态；GPU5未挪生产。完整76/72/228+228、最终报告与全部云端未齐，goal active。科学冻结/head/预算/采样不改，旧V2/CLIP不重跑，其他项目/已完成及失败资产保护；未commit/push/恢复旧定时任务。

## 2026-10-16 13:00 HKT 执行提速已上线：两个VMA-XD原writer安全替换，104768clips保留并逐视频严格复用

用户明确要求多智能体优化慢进度，并进一步允许“可以的话杀掉，换最新优化提取”。本turn完成实际代码/验证/替换，不是仅提出计划。三名代理分别只读定位CPU路径、审计队列关键路径、实现最小CPU候选；root整合并经独立审阅。只读证据：VMA-XD已完成格13.91–14.86h、约占剩余GPU小时60%；816条原完整视频效率事件中encoder+H2D16.65%、head0.14%、其余83.20%（不能等同全部packing）。20秒relay不是小时级拖延来源；CPU全节点未满。工作量本轮12268319clips，旧V2仅4086273，不能把范围差异忽略。

新增optimization-r01/numpy_packing_candidate.py只把原HF最终容器改np→torch.from_numpy；numpy_dedup_candidate.py在同batch按真实frame_indices校验重复RGB后，仅对unique帧运行完全相同原HF逐帧resize/crop/normalize，并还原原B,T顺序。无跨batch/cache，不动adapter原constructor/默认pt/权重、head、方法/层/预算/采样/真值。真实CPU门分别原Swan ftjy2gtw（packing）和berxx8zk（dedup）ONLINE先于计算，均FINISHED/qa/completed=1回读核验；各27case、训练2normal+2anom、B1/B8首中尾/存在的tailpartial、每路径交替2重复及B8双worker重复，全RGB/indices/valid lengths/keys/dtype/shape/tensor逐元素相等，结果SHA与原completed event绑定。Packing acceptance SHA8603cdda5b4713bbc817ebb5cce9e8e6219c8306721806c6d1ac4c34bd586ea9；dedup SHA017bf2b81a1c155f2f289855921afed44b2c2a8e56f66188a769ecbd82051b78。B8纯prepare：packing原3.0428→10.3953clips/s=3.4164x；dedup原3.0735→15.2514=4.9623x。此为共享CPU准备阶段，不是encoder/完整提取或算法正式加速倍率。

preblock_resume.py为独立狭窄PreBlock复用实现，实际PreBlockDeployment类型及完整frozen reducer/每层真实shape回执严格校验；仅copy，不hardlink/伪装PairMerge。除绑定exact门的spec.cpu_pipeline差异外，spec/microbatch/encoder/paper/data/semantic runtime保持严格；继承原copy_video逐样本/原视频SHA/每数组SHA、shape、dtype、finite、readout、runtime-reference检查；仅完整发表video复用，partial整视频重提。Synthetic packing5项、dedup10项、resume17项通过，compileall src tests通过。另上线前对两个真实源103380条原metadata几何回执核验通过（不是数组copy验收）。原全局resume/科学快照未改。

冻结执行bundle execution-optimization-r03.tar.gz SHA466fb7adcdb578453e1184daefc49ac66c6ffe3664ba9dcd3853162229038ec4；execution-config SHA265eed471ed3c7c80c590dc7b725e289c45b6503945bdd4e8b066c0307eae3f0；候选SHAeb55d8d759b017df8f487257ed0071d49defa99b8418e84f31225869b01de4c8。含原科学tar、实际wrapper/controller/候选、原在线门acceptance/events/cloud与测试。remote先验bundle SHA/所有proof及实际导入pipeline路径/SHA、候选实际execution_identity；原campaign-r02 SHA16cec7a9…/formal-source-r02原字节不改。独占效率实验仍原路径，不以新生产提速替换已测228+228数字。

先只停止本机observer161388（完整记录cmd/原queue/所有logger，未递归终止子进程），再按原runner PPid+owner+完整argv精确找到VMA-XD pre9/r50 python23713、r70 python27300，SIGTERM后确认实际writer消失、原runner exit143，才将两完整旧task目录原样rename至同output_root/preserved-execution-r02/<task>。r50保留611完整视频/87078clips；r70保留191/17690；合计104768clips。原PID/Swan/日志/失败终态和未发表partial全保留，replacement-receipt.json SHA77c4dfed3c6525040c7953c1da5b3c7ffea9c7e19ff5269fad81a7e55ee9e1d4。没有删资产/覆盖旧run或重跑已完成logical格。

新observer142864，controller-r04在formal-r02-execution-r04；原TS-XD logger148884/原Swan c4upi8u3、TS-UCF logger121412/原Swan bdgmc4w0完整接管、未重启。新VMA-XD两个logger135316/142700，remote bash12250/12240、python12252/12242，GPU0/1；已通过原head SHA与backbone code_digest9a23c5da…严格preflight、真实resume_lineage已生成，首次检查各已有33个完整video/3838clips经原copy验证发表。此时仍在copy阶段，不把复用clips当新GPU吞吐、也不宣称全部104768已验完。status_snapshot按每logical task取新attempt与preserved完整前缀max只计一次，并从新吞吐中扣除旧前缀，防止复制速度冒充GPU速度。

12:58本机快照正式15/76、质量11/72、累计2675761/12268319clips=21.81033%；encoder228/228、pipeline207/228，原效率父10074仍live/GPU5独占测量继续，未提前挪用。已授权的后续提速继续有界验证，TS和UCF尚未放行新candidate；其他项目1827/26335等保护、旧V2/CLIP不重跑、未commit/push/恢复旧定时任务。完整目标76/72/228+228、最终报告/终态云端仍未齐，goal active。

## 2026-10-16 12:37 HKT 15/76预测、11/72质量；完整视频效率204/228逐组审计通过

本turn先核实用户询问的生产吞吐与CPU占用，而非擅自执行提速。node3公网只读采样12:32：32逻辑CPU，3秒整体busy27.6643%、iowait0.1572%、steal0，load 7.52/7.48/7.21，没有逻辑核在该窗口平均超过80%；MemAvailable2022694072kB。该证据不支持“别人把整机CPU占满”的说法，但不排除短时争用/局部限制。原VMA-XD约2.77–2.87clips/s，当前两配置2.65/约2.7，持续偏慢而非突然大跌；慢CPU准备路径及分阶段耗时为原有证据，不能未经隔离实测断言全部耗时均来自单个张量打包函数。用户只问原因和前后速度，没有批准切换CPU路径或GPU5加入正式生产；本次没有进行此两项变化。

主控161388、CPU质量102460、效率logger103812均本机live。node3原效率父10074实查/proc存在、queue running；audit_completed_pipeline.sh对204个已完成pipeline组逐一通过：每组16完整冻结训练视频、video SHA/帧数/clips、对应head SHA、原采样、目标block与前一signal block、真实变短token及组前空GPU/组后仅自身PID，结果SHA匹配原queue。只是204/228中间审计，不冒称228全完或Swan终态成功；encoder228/228既有审核复用。

12:37快照正式15/76=19.73684%（4dense+11压缩）、质量11/72=15.27778%。新增两项：VMA-XD Attention pre6/r70主PR71.0012893033%，对dense70.3000788652%，Δ+0.7012104381pp、95%CI[-1.2102812206,+2.6608263804]pp，inconclusive；补充stepAP64.8401074310%降至57.6441098513%，不能称全面提升。VMA-UCF pre9/r70主ROC80.0209500511%，Δ-0.5277914420pp、CI[-2.7304471361,+1.4592837537]pp，inconclusive。原Swan h878fifb/3zaxtr6u及原结果SHA/云端主三点已在上一执行段核验；已有结果复用不重bootstrap。总计2supported、9inconclusive、0确认劣于margin，所有质量结论仍单head/多配置未校正。

累计2640537/12268319clips=21.52322%，每task一次、结束video下界。GPU0 VMA-XD Attention pre9/r70为14610/145703=10.03%；GPU1 VMA-XD同pre9/r50继续；GPU2 TS-UCF pre9/r70为62643/138853=45.11%；GPU3 TS-XD pre9/r50为265051/291781=90.84%。原四卡继续、未重启/重投；GPU5原效率单任务继续，保护其他项目。完整76/72/228+228、最终报告和云端终态仍未齐，goal active；未commit/push、未恢复旧定时任务。

## 2026-10-16 09:21 HKT 13/76预测、9/72质量：TS-UCF pre9/r50首次在UCF主ROC支持非劣

本turn为确认live的持续执行与有界等待：TS-UCF Attention pre9/r50完成全部138853clips、完整预测与原Swan成功后，由主控161388把GPU2补到VMA-UCF pre9/r70。原四路任务持续输出，旧Swan身份和正式source-r02/campaign-r02不变，未重投/重启、未据分数调整冻结网格。恢复主控写队列稳定、CPU质量102460和效率logger103812均live。

新增第九项TS-UCF Attention pre-block9/remove50：原dense frameROC-AUC78.6800663081%，method80.0865229571%，Δ+1.4064566489个百分点；95% paired-video CI[−0.4125422792,+3.4748940875]个百分点，下界>=−0.5pp，原noninferiority=supported。CI仍包含0，不能声称已证实显著提升；单head seed0/多配置未校正限制保留。同一配置TS-XD还在提取，不能提前说跨dataset均支持。方法/位置/预算/head/采样/真值仍原冻结值，seed0/10000draw没有修改。

质量原Swan tpe6r5xj FINISHED/quality_completed=1，主三点API回读与本地匹配。原结果SHA9561f661ca2f71d251a42762c8959fbaa61c5c0fedcb581cda8df259f31c34e2回传校验，其他8结果原SHA复用、不重复bootstrap。预览MD/CSV/JSON更新9项，不覆盖quality.json。当前主指标2supported（VMA-XD pre6/r50及TS-UCF pre9/r50）、7inconclusive、0确认劣于−0.5pp界限；前者stepAP下降−2.9883pp的反向补充仍明确保留，不能概括全指标提升。

09:21正式13/76=17.10526%（4dense+9压缩），质量9/72=12.5%，encoder228/228、完整视频效率177/228=77.63158%。累计2308164/12268319clips=18.81402%，每task一次、结束video口径。四卡继续VMA-XD pre6/r70、VMA-XD pre9/r50、VMA-UCF pre9/r70、TS-XD pre9/r50；VisionZip/uniform正式格子仍按原顺序排队，未因已有支持非劣格就跳过。效率GPU5原会话RUNNING待全部228及终态核验；全76/72/228+228、最终报告/云端未齐，goal active。其他项目与旧资产不动，未commit/push/恢复旧定时任务。

## 2026-10-16 07:17 HKT 12/76预测、8/72质量；TS-XD pre6/r70主PR下降，仍未支持非劣

本turn为核验live的持续工作/有界等待：TS-XD Attention pre6/r70完成全部291781clips后经历正常固定head预测收尾，未因观察时间较长重启。必要一次node3原PID检查：bash1975/python1977仍live，原formal-source-r02/原Swan mh8iia8r命令不变，Python RSS约7.9GB、GPU3约616MiB，表明已进入固定head阶段而不是GPU提取失败；随后自然完成并云端成功，GPU3由主控161388自动补到TS-XD pre9/r50。其他原任务未重投、源/方法/head/预算不改。

第八项质量TS-XD Attention pre-block6/remove70：冻结主梯形frame PR-AUC dense76.6558428832%，method75.6858562160%，Δ−0.9699866671个百分点，95%视频配对CI[−1.8967412284,−0.0324245474]个百分点。CI整体为负，但相对−0.5pp容忍界限仍跨界，原noninferiority=inconclusive：未支持非劣，亦未确认损失超过容忍界限。不能把“相对0的下降方向”与“确认劣于−0.5pp界限”混为一谈。补充stepAP dense70.8385083610%、method70.5280995569%，差−0.3104088041pp，仅点值，不另作CI。

原质量Swan rtvcfckf FINISHED/quality_completed=1，API主三点与本地全精度一致；原结果SHA c90063aca2be33f1f5f3feb1fcbdb06862df587bc920c8db86b135dd2690e958回传匹配。其余7项原SHA保持，collect仅复用、没有重复bootstrap。预览MD/CSV/JSON共8项已更新，补充明确“不能因主指标改善而省略stepAP下降或称全指标提升”。只改派生报告措辞，未改指标/冻结源；compileall src tests exit0。

当前正式12/76=15.78947%（4dense+8压缩），质量8/72=11.11111%（主指标1supported、7inconclusive、0确认劣于容忍界限），encoder228/228，完整视频效率156/228=68.42105%。累计2102465/12268319clips=17.13735%，每task一次、结束video口径。四卡继续VMA-XD pre6/r70、VMA-XD pre9/r50、TS-UCF pre9/r50、TS-XD pre9/r50；CPU主控102460与效率logger103812正常。主控恢复已自然完成并补位多个原/新任务，队列写入稳定，无新WinError5；原身份/行内tracking/source约束有效，其他项目未动。全76/72/228+228、最终报告及终态云端尚未齐，goal active；未commit/push/恢复旧定时任务。

## 2026-10-16 05:26 HKT 11/76预测、7/72质量：首次主梯形PR支持非劣，但同组stepAP下降，不能称全面提升

本turn持续验证原live任务并等待自然完成：第四个dense VMA-XD已完整145703clips/800视频预测并云端成功，四dense至此全ready；GPU1按原队列补VMA-UCF Attention pre9/r50，该项完整完成后补到VMA-XD pre9/r50。原长任务VMA-XD pre6/r50也完成145703clips+预测+原Swan hi0l3k2h成功，GPU2补TS-UCF pre9/r50。所有动作由恢复主控161388完成，原身份/正式source-r02/campaign-r02保持；没有重投GPU、没有改冻结配置或依据分数调整队列。

新增VMA-UCF Attention pre9/remove50主frameROC-AUC：dense80.5487414931%，method80.5039537124%，Δ−0.0447877806个百分点，95%视频配对CI[−2.1216258574,+1.8945156626]个百分点，inconclusive。点值很接近不等于已证实非劣。原质量Swan s80vf370 FINISHED，原结果SHA f11f54141c69a86e1421003d90b1b979b3a506fe24f59be11461d8f05a672530回传验证。

新增VMA-XD Attention pre6/remove50：冻结主frame梯形PR-AUC dense0.7030007886523076（70.3000788652%）、method0.7209091822113581（72.0909182211%），Δ+1.7908393559个百分点，95% paired-video CI[+0.3319367101,+3.2753465512]个百分点。按预登记margin−0.5pp，原noninferiority=supported；单次未校正CI在0以上，但不能扩展成跨配置/跨seed或全面质量提升。关键反向补充：同组stepAP由64.8401074310%降到61.8518277814%，点差−2.9882796495个百分点，未给stepAP计算CI。UCF同配置也仍未支持主ROC非劣，因此当前没有同一非平凡配置两dataset主CI同时支持的证据。保持原作者梯形PR主指标与stepAP单列，不能挑有利指标来宣称胜出。

该VMA-XD质量原Swan6qqsllhk FINISHED/quality_completed=1；API回读dense/method/delta与本地全精度匹配。原quality-preview.json SHA ace755377b2ab140f235a6a19f1931c692b36da5e119ccfcd4fa501d35b23d16校验一致。800视频/2330384帧、正常300/弱正500，paired seed0/10000次有效draw，无无效样本。静态复核原PRCurve实现确为全局同分阈值分组+PR原点线性梯形积分，stepAP为分组矩形积分，未发现需要修正的代码；未额外重算正式指标、未换主指标。已有7结果只验SHA复用，MD/CSV/JSON预览和stepAP独立表已更新。

当前正式11/76=14.47368%（4dense+7压缩），质量7/72=9.72222%（1主指标supported、6inconclusive），encoder228/228，完整视频效率137/228=60.08772%。累计1923173/12268319clips=15.67593%，每task一次，当前未结束video不计。四路为VMA-XD pre6/r70、VMA-XD pre9/r50、TS-UCF pre9/r50、TS-XD pre6/r70，正常自动接力；CPU主控102460/效率logger103812继续，原效率会话仍RUNNING待228全部及终态核验。主控原子队列恢复后稳定，其他项目及旧资产不动。最终76/72/228+228、完整最终报告/云端仍未齐，goal active；未commit/push/恢复旧定时任务。

## 2026-10-16 00:41 HKT 恢复主控已完成真实补位；8/76预测、5/72质量、完整流程效率102/228

恢复后的controller161388已实际观察原TS-UCF pre6/r70自然完成、核验原Swan成功、把GPU3补到TS-XD pre6/r70；不是仅看见PID就宣称恢复。原logger103052/原Swan p2945py2完成后复用结果，新任务logger154108正常启动；新local tracking目录formal-r02-recovery-r03/tasks/...，行内expected_snapshot仍formal-source-r02，原campaign-r02 SHA/方法/head/预算/保护命令不改。其余三个原GPU任务保持旧PID/旧Swan继续，没有重新提取或重投。主控恢复后连续写队列超过1小时，无新增PermissionError，旧源/故障记录/接管审计均保留。

新增第五项质量为TS-UCF Attention pre-block6/remove70：原dense frameROC-AUC78.6800663081%，压缩77.0134477467%，Δ−1.6666185614个百分点，95% paired-video CI[−5.0112354039,+1.9256533795]个百分点。原决策inconclusive，点值超出−0.5pp容忍界限，CI仍不能确认劣于界限，更不能称非劣。seed0/10000draw、对应正式head和相同输入/真值SHA不变；没有根据该结果改参数、切指标或停止计划内其他配置。

原质量Swan od69syin FINISHED/quality_completed=1，API三点回读与本地一致；原结果SHA ddf94d521aaef24b0a669657a1db556aaac28e7d2ac72adc773b58d91a8466e1回传验证。其余4个结果验SHA复用，不重跑bootstrap。预览MD/CSV/JSON已更新5项，时间显示改为HKT并保留原UTC字段；未覆盖任何最终quality.json。

当前正式8/76=10.52632%，质量5/72=6.94444%，encoder228/228、完整视频效率102/228=44.73684%；累计1498864/12268319clips=12.21735%，每task仅计一次、已完成video口径。四路为VMA-XD dense136894/145703、VMA-XD pre6/r50为102747/145703、VMA-XD pre6/r70为44896/145703、TS-XD pre6/r70为6114/291781。效率原父10074/本地103812继续，CPU质量主控102460完成第五项后等待匹配。完整目标76/72/228+228、最终报告及云端终态还未齐，goal active；未commit/push、未改其他项目、未恢复旧定时任务。

## 2026-10-15 23:37 HKT 本地调度器写队列瞬时失败已恢复；四原GPU/PID/Swan完整接管、没有重投

本turn观察到原本地controller PID125064退出，stderr明确为os.replace(queue.tmp,queue.json)的Windows WinError5/PermissionError，最后正常队列时间22:54:09 HKT。具体持锁者未定位；不是GPU计算失败或模型错误。queue.json与遗留queue.tmp逐任务状态一致，没有未登记新任务。只读核验四个原本地logger128876/87480/114528/103052均live，完整命令/原provenance/输出路径一致；node3四原bash PID26832/2548/1633/29847均live、无exit-code，日志继续产出，原Swan分别pr54ynzc/hi0l3k2h/w5r5goko/p2945py2未改变。因此没有重启、SIGTERM或重投任何GPU任务。

新增纯调度恢复controller-r03.py，保持原campaign-r02 SHA和formal-source-r02、GPU保护/显存上限/全部76格不变：--resume按行尊重tracking_directory/swanlab_source/expected_snapshot，核对原logger命令+创建时间、原Swan link、原campaign/provenance后接管；已完成任务严格复用原云端成功，未知/失败只hold原slot，绝不重放。新增controller_state_io.py仅针对已观察PermissionError对原子replace做最多10次有界重试（最大约6.5秒），旧JSON在成功前保持不动，失败临时文件保留可恢复。不吞其他错误。

新主控PID161388，日志work/preblock-vma-ts-20261015-r01/formal-r02-recovery-r03/controller.stdout/stderr；ADOPT_ORIGINAL四条逐一打印原ID，队列controller_pid与source SHA已更新。未来新任务的local tracking目录在recovery-r03下，但四个活跃行仍使用原formal-r02路径，远端tasks-r02/模型source-r02不变。controller-recovery-r03.json/tar.gz绑定旧stderr、新源/IO helper/原队列备份/原PID核验/接管日志；audit_controller_recovery.py确认四行六个身份字段及Swan对象完全不变，7个已完成状态不丢失。故障注入测试2passed：临时锁两次失败后成功、永久锁10次后抛出并保留旧队列/完整临时状态；compileall src tests exit0。恢复后实际30分钟持续写队列成功、新主控及原loggers均live，未更改任何科研配置/源码快照或已完成质量结果。

23:37正式仍7/76、质量4/72、encoder228/228；累计1433349/12268319clips=11.68334%，原四任务继续自然推进。完整视频效率父10074/本地logger103812正常接力，CPU质量主控102460正常等待匹配；没有因观察器退出新建Swan身份或重复bootstrap。目标76/72/228+228与最终报告/终态云端未完成，goal active；未commit/push/恢复旧定时任务，其他项目进程与旧资产保留。

## 2026-10-15 21:55 HKT 首个XD配对完成；7/76预测、4/72质量、完整流程效率79/228

本turn为实质进展与确认live的有界等待：VMA-UCF Attention pre6/r70完成完整69364clips+预测+云端后GPU0补到VMA-XD同配置；TS-XD Attention pre6/r50完整291781clips+预测+Swan成功后GPU3补到TS-UCF pre6/r70。正式7/76=9.21053%，质量4/72=5.55556%，encoder228/228、完整视频效率79/228=34.64912%；累计1289173/12268319clips=10.50815%，单task计一次。正常旧PID/原Swan身份继续、没有重投/重启，没有修改冻结方法或其他项目。

新增VMA-UCF pre6/r70主frameROC-AUC：dense80.5487414931%，压缩79.7548360770%，Δ−0.7939054161个百分点，95%视频配对CI[−2.9324303084,+1.6815758966]个百分点；非劣仍inconclusive。原head/采样/真值/实现SHA门通过，点值超过−0.5pp界限但CI不能确认劣于界限，更不能称支持非劣。不据该分数改预算/位置或切换主指标。

首XD质量是TS-XD Attention pre6/r50：主作者口径梯形frame PR-AUC dense0.7665584288317776（76.6558428832%），压缩0.7655602667912158（76.5560266791%），Δ−0.0009981620405618008（−0.0998162041个百分点），95% CI[−0.006560589259434912,+0.004676768825352917]，即[−0.6560589259,+0.4676768825]个百分点。margin−0.5pp，CI下界仍越界，noninferiority=inconclusive，不能把小点差说成非劣已证实。补充step AP单列：dense70.8385083610%，压缩70.9037622542%，差+0.0652538932pp，未为stepAP另算CI、不替换主指标/结论。两个定义在并列分数下可以明显不同；未为解释差值重复读取正式真值或重算评测。

原XD质量Swan smya5sd8 FINISHED/quality_completed=1，并API回读dense/method/delta与本地完整精度匹配。首XD原quality-preview.json SHA230be0b8079cb5b11988d8393cd4e19c8ad36b4b8cd43ec12463c1d530b63ca5回传一致，seed0/10000 paired-video draws不变。运行中只必要一次查原CPU子30187：99.8%单核、RSS约3.46GB，节点可用内存约1.96TB，证明正常CPU计算而非观察超时后重启；随后自然完成。其他三质量原结果SHA仍一致，仅收集复用、没有重复bootstrap或覆盖quality.json。

reports/preview-20261015/MD/CSV/JSON已更新四项真实结果，新增XD补充stepAP独立表和未给stepAP做CI的限制，不混报PR与stepAP。给原有纯合成单测补明确断言：同分阈值toy例梯形PR=.75、stepAP=.5，说明二者不是别名；tests/icassp2027/test_urdmu_quality_export.py当前3passed，compileall src tests exit0。没有修改实际指标实现/冻结源或重新计算正式分数。

四路正式提取继续（VMA-XD dense、VMA-XD pre6/r50、VMA-XD pre6/r70、TS-UCF pre6/r70）；CPU质量主控102460等待后续匹配，效率父10074/本地logger103812继续GPU5逐组独占测量。当前四项主CI均inconclusive，不扩大为全部方法结论；余68配对尚未评测，VisionZip/uniform正式矩阵仍按原冻结队列等待。完整76/72/228+228、最终报告及终态云端核验仍未结束，goal active；未commit/push、未恢复旧定时任务。

## 2026-10-15 18:24 HKT TS-XD dense及TS-UCF首压缩完成，第二项配对质量仍未支持非劣

本turn持续工作并对原live任务做有界等待，期间TS-XD dense完整提取291781clips+预测+Swan FINISHED后自动补到TS-XD Attention pre6/r50；TS-UCF Attention pre6/r50完整提取138853clips+预测+云端完成后GPU0自动补到VMA-UCF Attention pre6/r70。没有重启正常任务或修改冻结矩阵。正式5/76=6.57895%，质量2/72=2.77778%，encoder228/228、完整视频效率47/228=20.614%；累计946825/12268319clips=7.71764%（每task一次、结束video口径）。四路当前进度VMA-XD dense69645/145703、VMA-XD Attention6r50为37562/145703、TS-XD Attention6r50为126875/291781、VMA-UCF Attention6r70为4528/69364。

新增TS-UCF Attention pre-block6/remove50（前block5完整输出后、1569→785）的真实配对主frameROC：dense0.7868006630812754（78.6800663081%），method0.7812979816906978（78.1297981691%），Δ−0.005502681390577603（−0.5502681391个百分点），95% CI [−0.03004511970691046,+0.0219722930596632]，即[−3.0045119707,+2.1972293060]个百分点。非劣margin仍−0.5个百分点，点值已超容忍界限，但CI仍跨界/跨0，原决策inconclusive：不能称支持非劣，也不能称已确认劣于界限。单正式head seed0/paired-video bootstrap seed0/10000不变；不据该分数调参、换head或删预定配置。

原质量Swan5ivp441n FINISHED/quality_completed=1；API回读dense/method/delta三点与本地结果匹配。原quality-preview.json SHA216878c25611c8fbca054dfdeb2be28c706895e162e54ad4d0b43b0f24c91292回传验证，首VMA结果原SHA仍一致，仅复用不重算bootstrap。reports/preview-20261015/QUALITY_PREVIEW.md及CSV/JSON已按两项真实结果更新，余70格不补旧成绩。尚无XD配对分数；三项dense中只有已配对的UCF两条读过帧真值。

效率父10074及在线会话继续；audit_completed_pipeline.sh对已结束31组逐组核验每组16完整冻结训练视频、成员/原视频SHA/全部native clips/帧数/head SHA/采样、真实token/目标/信号block、GPU前空后仅自身PID、raw结果SHA，全部通过，之后自然推进至47组。不是把raw视频到score计时冒充含FeatureStore落盘的正式提取总耗时；全节点CPU/存储共享、SHA预读可能暖页缓存的限制保留。全会话仍RUNNING，未提前报终态成功。4.6TB空闲存储及充足inode已核验，无需清理其他项目。

主控125064、CPU质量102460、效率logger103812均继续，质量队列完成两项后等待后续匹配。下一项预计是VMA-UCF pre6/r70与稍后的TS-XD pre6/r50；具体以实际完成+云端门为准，不预报质量。完整76/72/228+228、最终报告与终态云端仍未全完成，goal active；未commit/push、未恢复旧定时任务、没有改其他项目或已完成V2/CLIP资产。

## 2026-10-15 15:41 HKT 首个正式配对质量完成：VMA-UCF Attention pre6/r50，CI未支持非劣

本goal turn为实质进展：首个压缩提取/完整预测/在线终态核验完成，CPU质量主控自动匹配对应dense，原compare_prediction_runs已真实跑通一次；没有重复bootstrap或修改冻结方法。正式3/76=3.94737%，质量1/72=1.38889%，encoder228/228，完整视频效率21/228。累计608117/12268319clips=4.95681%，单task计一次、当前未完成video不计；status-snapshot已按实际更新。

首质量为videomae×UCF、Attention在block6之前（block5完整输出后）移除50%，1568→784token，原正式UCF head SHA9c47365f…、相同完整290test视频/采样/真值SHA门均通过；encoder不是VideoMAEv2，不与旧模型分数混比。Dense frameROC-AUC=0.8054874149305954（80.5487414931%），method=0.8029191175454538（80.2919117545%），Δ=−0.0025682973851416206（−0.2568297385个百分点）；95%视频配对CI=[−0.021502819585984777,+0.017640679723698326]，即[−2.1502819586,+1.7640679724]个百分点。非劣界限−0.5个百分点；点值在容忍范围内，但CI下界越界，原决策inconclusive，不能称已证明非劣，也不能称确认劣于界限。单head seed0/分层paired-video bootstrap seed0/10000draw保留，不据此调参/停跑预定配置。

质量原Swan3nrtocm7 FINISHED/quality_completed=1终态已核验；另用API回读quality/dense、method、delta，三项与本地结果在1e-12内一致。远端quality-preview.json原SHA c95b751a93bd6b53a9264b0b642249fe4860f160129353a1ca890d1ea84e8a10，经scp原字节回传吻合；quality-results-index.json及quality-cloud-points-verification.json记录绑定。collect_quality_previews.py只收集完成且云端成功原结果，已有则验SHA复用，不重算；build_quality_preview.py仅生成reports/preview-20261015/QUALITY_PREVIEW.md、quality-available.csv/json，明确partial_not_final，其他71格不填旧成绩，不覆盖quality.json。

该配置独占GPU常驻encoder+原adapter/readout/plugin的三轮配对中位倍率为B1 1.37004×、B8 1.41112×；这不等于完整视频流程加速，也不能抵消质量CI未支持非劣的事实。同配置全流程三轮尚未齐，不填虚构值。

主控已自动把GPU0从该已完成压缩任务补到TS-UCF Attention pre6/r50（6324/138853），GPU2继续VMA-XD同配置（9086/145703）；GPU1原VMA-XD dense40966/145703，GPU3原TS-XD dense274160/291781。四卡正常、不重启、不改原身份；前两个UCF dense及首压缩三条预测成功保留。CPU主控102460完成首项后等待后续匹配，效率父10074/GPU5原会话继续。全部76/72/228+228、最终报告和终态云端未全完成，goal active；未commit/push/恢复旧定时任务。

## 2026-10-15 15:22 HKT 两个UCF dense全完成，首个压缩任务94.27%，完整流程效率19/228

本turn属于实质进展与已核验等待：VMA-UCF和TS-UCF先后完整提取+预测并云端FINISHED/test_completed=1，正式2/76=2.63158%；主控自动补位GPU0到VMA-UCF Attention pre6/remove50、GPU2到VMA-XD同配置，不重启正常旧任务。first-dense-provenance-audit.txt已核验首dense完整69364clips、全部native windows、query256全key、对应head SHA、prediction与feature contract三文件SHA；不打开帧真值。存储预算只读检查data2剩4.6TB/inode余2.81亿，无需清理其他资产。

15:22累计567948/12268319clips=4.62939%，每task只计一次（完成video口径）。GPU0 VMA-UCF Attention pre6/r50：65389/69364clips=94.269%、275/290视频；GPU1 VMA-XD dense：37562/145703=25.780%、324/800；GPU2 VMA-XD Attention pre6/r50：6186/145703=4.246%、54/800；GPU3 TS-XD dense：250594/291781=85.884%、743/800。node3原bash PID分别4422/26832/2548/26825均/proc存在、无exit-code，主控125064与原Swan loggers正常；保护其他项目，没有重投GPU。

Encoder228/228原始审核与预览不变；完整视频效率19/228=8.333%，原父10074确认live，Swan r4ef7f3u仍在线RUNNING，未冒称整套终态。CPU质量主控102460正常等待，质量0/72：首压缩尚未预测完整，不能提前读真值报分数/非劣。

按各自实际生产clips/logger wall（非独占速度），VMA-UCF压缩约10.04clips/s，VMA-XD dense2.87、压缩2.77，TS-XD dense19.17；首VMA-UCF压缩剩余提取估计15:27–15:32，另需预测/配对CPU评测；TS-XD dense提取15:49–16:16；VMA-XD dense今日23:12–明日07:03，VMA-XD压缩明日01:50–12:19。规划范围来自同任务余量/吞吐乘0.75–1.5，不是统计CI，也不含未测预测尾部。TS压缩/VisionZip/uniform生产吞吐尚未测，整套可靠ETA仍未知，不能承诺当日全完。完整目标76/72/228+228/报告及云端仍active。

## 2026-10-15 13:11 HKT Encoder独占效率228/228完成；完整视频效率与四路正式提取继续

上一goal turn已实质启动四路正式提取与76格接力，不是无进展。本turn在不重启任何正常GPU任务、不改冻结方法/head/预算的前提下，完成独占效率代码、计划冻结、CPU质量自动接力，并两次对确认live的原PID做20/45分钟有界等待；主提取持续产出。

新增src/vadbench/paper/pre_block_efficiency.py：228个encoder组（2encoder×19配置×B1/B8×3独立进程），每组20warm/40实测，GPU常驻的原生输入，含原adapter/mean读出/插件信号选择merge与执行回执，不含解码/CPU预处理/H2D。GPU5每组前后只允许自身PID，保存raw wall/CUDA stream/allocated/reserved；共享生产吞吐不当正式速度。另228个完整流程组（2encoder×2dataset×19×3），每组固定16完整训练视频（8normal/8anomalous、按ID首8、全native dense），分别2973/2354/5952/4718clips。含连续解码、原HF CPU管道、传输、encoder、各自正式head的exact query256全key推理及原帧均值投影；不含权重加载、SHA检查、FeatureStore落盘和最终结果文件写入。原视频SHA在计时前核验可能预热页缓存，不称冷盘/冷启动；CPU/存储共享、不冒称全节点独占；完整时序head为离线非因果。协议EFFICIENCY_PROTOCOL_R01.md已固定，不按速度/正式分数筛选配置。

冻结efficiency-plan-r01.json SHA2e0e30adbd76726f7f7368aee80c51cf42205b0c54efb264158effff1e50e619，456个唯一组在测前列齐；efficiency-source-r01.tar.gz SHA47fcd0decaa1cda2100ac83c4db8ff821cb91d48935b952413906d7ee2c89cb2，原adapter字节/bridge/selector与formal-r02一致。效率在线会话r4ef7f3u先建后算，本机logger103812、node3父10074，GPU5单子进程自动接力；其他项目1827/15743/15744/26335/4592/4593保留。源/父控/计划/协议云端bundle b071867ac8a8e750aefcf75d8f84323e05b9d875d467313048e93adf79efcd3d，本地Git HEAD/dirty状态补存，不commit/push。

已真实完成encoder228/228=100%，云端summary回读encoder_groups_completed=228；但会话仍RUNNING，不能称整套效率终态已核验。原始228份结果从服务器打包回传，SHA daf122debd40b5d97c4a2d3896ad7645f2b9a95640a8c662dc8c79ab4042618d，两端匹配。audit_encoder_results.py逐组核验全部job集合、每组40wall+40stream样本、3独立进程重复、实际token/目标block/信号block、GPU前无进程/后只有原自身PID、显存正值与plan SHA，228全部通过；聚合76个encoder×方法×预算×batch格到encoder-efficiency-preview.csv/json，含全部raw groups。描述性三轮配对wall倍率：VMA B1约1.0049–1.5519×、B8约1.0444–1.6424×；TS B1约1.0048–1.5511×、B8约1.0352–1.6485×。这不是完整流程加速、显著性结论或质量保持结论，未据此选配置。

完整视频效率已完成2/228，当前第三组TS-XD VisionZip pre6 keep0.5 R1在跑（16视频已8个）；仍须等待全部228及终态云端核验。原结果/每video分阶段耗时、帧数与clips、frame score字节SHA均保留。针对新增效率与bootstrap入口测试5passed，compileall src tests当前exit0；前一turn bridge+quality13passed/9local-version-skip记录保留，不混报为本轮完整测试。

CPU质量主控102460已实际启动，quality-control-r01/queue.json保持waiting_for_matched_predictions；只在method和对应dense均预测完成且Swan成功时启动一个CPU worker，CUDA_VISIBLE_DEVICES为空，每项先建在线会话，再由原compare_prediction_runs跑一次seed0/10000配对CI，独立quality-preview.json严格复用，不覆盖quality.json。质量尚0/72，第一项正式预测匹配后才能验证真实质量入口；不把等待队列说成质量完成。

13:11 HKT正式完成仍0/76；clips已241635/12268319=1.96959%，每task仅计一次、当前未结束video不计。四原提取任务/原Swan身份均正常，没有重启；当前source-r02不变。各自从logger启动到当前的生产吞吐下界约VMA-UCF11.12、VMA-XD2.82、TS-UCF13.24、TS-XD18.88 clips/s（含数据/写盘/启动，不是独占计时）。同任务剩余提取时长以0.75–1.5倍作规划范围：UCF-VMA今日13:24–13:36，UCF-TS14:17–15:22，XD-TS15:19–17:27，XD-VMA今日22:51–明日08:31；不含尚未实测的预测收尾，范围不是统计CI。VMAXD使用原默认pt处理器路径明显较慢，如实记录，未为赶进度偷换身份或重启。压缩生产配置尚无实测速率，整套完成ETA不硬承诺；status-snapshot.json/status-history.jsonl已保存各task实测与未知项。目标仍active，76/72/228+228/报告/云端全验收未完成。

## 2026-10-15 11:47 HKT 原环境门通过，四卡正式dense已实际提取、76格自动接力启动

上一turn属于实质进展：定位原提取环境和constructor差异、恢复原摘要，而非重复状态。本turn恢复classic-video-v2环境完成真实native和B8 pipeline门：TimeSformer m2y5f5uo、VideoMAE k6dxrcxk均在线FINISHED且qa/all_completed=1云端回读；两个dataset各18配置、ratio1三个位置、四独立fulltrain head消费、原生RGB/processor精确一致和持续B8桥均通过。TS无需重跑已成功部分。VMA额外失败4ffmxeee已保留：原因是显式processor_tensor_type='pt'在canonical constructor中不等同于原来省略该键，虽实际默认同pt但摘要不同；最终按原构造省略该键（request None），精确复现XD9a23c5da…，不伪造/放宽检查。原VMA UCF=np 81e31440…与TS两dataset=np 7f9a3ac6…也全部原值精确匹配。

真实新source-r02压缩快照SHA aae666274c3ddd37e24e925e3827cff1cb3b49966504af78f8ad37226826dd81；campaign-r02 SHA16cec7a9cba5ca1505d36a1bb8be6815948bbce72864fe93b5285db7002b584e。FORMAL_FREEZE_R02.md/method-freeze-r02.json及四条原环境gate原始SHA均服务器验证。VMA原receipt实际路径classic-classic-gate-videomae-r02（launcher替换产生双前缀），如实绑定原路径、不移动重命名；回传四文件字节SHA均吻合。旧source-r01/失败输出不改，新output根preblock-vma-ts-20261015-r01-r02。

主控本地PID125064已实际启动四卡接力，队列按原逻辑76格，只有预测完成+云端FINISHED/test_completed=1才记完成。GPU0 VMA-UCF（bash26840/python26842，emwifjsk），GPU1 VMA-XD（26832/26835，pr54ynzc），GPU2 TS-UCF（26810/26812，pb88ijf6），GPU3 TS-XD（26825/26827，p4j06tic）；四个正式预检已通过并已写真实特征。每卡一个本项目任务、显存上限0.5，0/1/2既有1827/15743/15744完整命令+owner+PID保护、至少24GiB空闲才派发；现场原进程保留，新增约3120/3120/2020/2020MiB。未知占用、失败或网络异常只hold该slot核查原PID/Swan身份，不盲重投。track目录/原Swan来源/expected_snapshot已逐行写queue，独立formal-r02/controller.stdout与tasks-r02原PID/日志可恢复。

11:47 HKT快照：正式完成0/76，质量0/72，效率0/228+0/228；四任务已完成视频的clips分别1294/69364、433/145703、1511/138853、2838/291781，单task计一次共6076/12268319=0.04953%。当前视频仍写入的clips未纳入，是保守下界。主指标尚无新结果，未读旧成绩冒充。仅启动数分钟，压缩配置尚无生产吞吐，整套ETA保持unknown，待实际分encoder/dataset/method测速再估，不套CLIP或foundation门速率。

并行于GPU生产（主控独立，未派agent）准备CPU质量入口scripts/icassp2027/run_pre_block_quality.py：先核验配对预测/head/sampling/encoder与artifact SHA，再打开真值；复用原compare_prediction_runs，独立quality-preview.json、存在则严格身份复用，绝不覆盖quality.json。共享compare_prediction_runs仅增加可选bootstrap_seed参数，历史默认20260918不变，本goal显式0与冻结一致；UCF ROC、XD梯形PR、stepAP单列。此新入口尚未对正式预测调用，质量snapshot上传/接力和228+228独占效率仍待实现，不能称goal完成。本机受影响测试13passed/9skipped，compileall src tests通过；9skip为本机4.53.2 VMA不同于审核版本，server实际classic完整门已通过。未改正式分数选参、未head训练、未动V2/CLIP旧成果、未commit/push/恢复定时任务；goal active。

## 2026-10-15 11:36 HKT 正式预检拦截与原环境身份恢复（正式0/76）

Goal仍active，仅VideoMAE/TimeSformer；76提取预测格、72质量、228 encoder+228完整视频计时范围未变。foundation环境的真实训练侧native gate和B8持续部署gate只是工程证据，不能算正式完成或替代对应原head环境。pipeline云端bokx6g25/i5kd3on4均FINISHED；首次未启动GPU的568mscov/mkdx81r7失败保留。

正式source-r01四dense尝试均在特征提取前被原direct_insert兼容门拒绝，backbone code_digest不一致；dvqg3wpx/dgfm63l6/ygja56yb/htt9l1h2失败云端及原目录/exit1保留，无正式clip产出、不覆盖失败目录。queue已改compatibility_recovery_classic_runtime_gate_required；正式完成0/76、clips0/12268319。没有伪造摘要或修改兼容白名单。

已定位并CPU元数据复算：原提取是classic-video-v2（torch2.3.0+cu121、Transformers4.37.2、timm0.9.12），而非前期工程gate的foundation（2.8.0/4.57.3/1.0.28）。旧adapter原文件仍在code-e30c3ab-s6p，SHA2ff1f9e0…匹配；独立compat-candidate-r01仅恢复该文件/原解释器，不安装环境、不改共享源码。实际constructor用原processor_tensor_type：VMA UCF=np精确复算81e31440…，VMA XD=pt精确复算9a23c5da…，TS=np精确复算7f9a3ac6…，均与各正式head逐字相等，无需新增豁免。证据classic-candidate.txt、classic-np-candidate.txt、old-constituent.txt、diff-runtime.txt。旧resolved仍在，但顶层及全部constituent的index/特征已清理，4dense需新提。

pre_block桥已新增读取审核过的4.37.2 VideoMAE原生forward AST支持：直接取原生dropout前概率，VisionZip同LN输入额外算原K，不改原attention。TS原生forward AST与先前一致。补丁尚未在classic真实权重上完成训练侧门，不能提前宣布通过。本机回归10passed/9skipped（本机4.53.2 VMA仍不在审核版本），compileall src tests通过。下一步在恢复原环境做必要native/pipeline门，再冻结新source/campaign，用新attempt输出目录恢复四dense及72项自动接力。生产吞吐未知，不给未经测量ETA；其他项目1827等未改动，未commit/push、未恢复旧定时任务。

## 2026-10-15 11:11 HKT VideoMAE/TimeSformer真实层前桥及四head消费门通过

本goal上一turn为实质进展（76格范围、4head元数据、pre_block桥草案及合成测试）；本turn完成4份training_qa与result/checkpoint匹配，均passed/completed、正式fulltrain1610/3950且seed0。_load_source原严格检查通过，保留训练阶段实际manifest SHA及完整样本数，不依赖已清理旧cache/view目录授权。模型真实路径/config确认VMA16帧tubelet2、TS8帧divided_space_time、224/16patch。旧4个dense合并合同SHA均仍匹配，但顶层index与features文件已清理，只剩contract/status/resolved；来源constituent可用性仍最后核对，若无有效索引/原特征则4个dense全部重提，不复用历史分数伪装对照。合同历史计数VMA UCF69364/XD145703，TS UCF138853/XD291781；新提取仍按实际完整manifest重算，不能用同一clip分母。

服务器原生训练侧检查已实跑且云端成功：VideoMAE36ewe307 FINISHED/qa_completed=1、TimeSformer oe6arjmy FINISHED/qa_completed=1。每encoder从各自四个正式head的全训练清单中固定每dataset2normal+2abnormal，一视频一真实原采样clip（不拼不同视频成虚构时序），前层信号与pre6/9/12 ×3方法×2预算在两dataset共36条件检查通过；三个ratio1位置数值一致性通过、所有真实target输入及MLP长度已核验，2个对应固定UR-DMU head分别消费dense+18配置的训练features并独立单视频得有限frame分数。真实VideoMAE支持1568→784/470，TS支持1569→785/473的完整轨迹缩短，原position/attention/backend/mean读出不变，未打开正式测试真值。结果已保存videomae/timesformer-native-gate-result.json。

首次两条检查在第一原生卷积处共同出现CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED，尚未触及压缩，原hdhkcbuh/4anjr1eu CRASHED云端失败记录/exit1/PID消失已核验后，针对已定位执行配置只修一轮：采用项目现有cudnn=False、FP32/TF32off，dense/压缩一致，未重装依赖、未切换attention backend、未改压缩规则。新source-r03与旧snapshot/failed dirs均保留；r01/r02源archive区别含开跑前的逐视频head消费检查整理，未曾混拼正式视频。新gate实际场景完成，不能用旧本机9skip代替。

正式完成仍0/76、未启动72项test矩阵；queue状态native_gates_passed_dense_reuse_final_audit。接下来接入已存在的PooledExtractionSpec/FeatureStore和严格UR-DMU评测，避免另写无provenance pipeline；包装已验收的pre-block桥为一次附加、每batch清信号的部署接口，正式前固定身份、源码、采样和4dense处理结论。当前无新生产吞吐，不给CLIP式两小时承诺。没有重跑V2/CLIP、没有head训练/正式分数调参、没有动其他项目或恢复定时任务，goal active。

## 2026-10-15 10:51 HKT 新goal启动：仅VideoMAE/TimeSformer层前压缩，76格

用户明确授权建立新goal并执行，仅videomae/timesformer；不重跑videomaev2/CLIP，不重训head。已创建work/preblock-vma-ts-20261015-r01/queue.json：76逻辑格（72新压缩+4个dense复用核验/重提）、72质量、228encoder+228完整流程效率；正式完成0，当前asset_audit_and_bridge_implementation。原工作树修改与旧完成成果已保留；未启动正式GPU/CPU评测，未恢复旧定时任务、未commit/push。

P0已只读登录node3公网，核验四个checkpoint实际存在并算SHA/读取元数据：VMA UCF9c47365f…、XD4b273e63…，TS UCF2b55818b…、XD85586f89…；均metadata run_mode=formal/status=completed_training/seed0/3000steps，UCF available1610、XD3950。实际采样身份确认VideoMAE clip16/frame_stride2/window_stride16，TimeSformer clip8/frame_stride2/window_stride8，两者head输入768/mean读出，不把TS默认或V2采样强套。完整训练QA/训练manifest及dense资产文件仍待逐项配对核验，不提前宣布4个dense可复用。资产记录asset-audit.json含原metadata和原生类源码；foundation-video-v2 Python3.10.20/torch2.8.0环境可用。现场GPU1/2/3/5空闲，GPU0原1827、GPU4/6/7其他项目保留，本轮未占卡；实际派发前重查。

新增src/vadbench/token_reduction/bridges/pre_block.py草案，source为target前一完整block，目标pre-block6/9/12分别source5/8/11。原attention及backend不改；VideoMAE无法直接取融合attention概率时同LN输入/原权重额外计算必要Q/K并计入成本；TimeSformer观察前层原生空间attention概率及实际QKV，空间score跨帧聚合/共享轨迹选择，VisionZip通过沿特征轴打包时间轨迹复用同一策略，逐帧应用同一merge，不平均混掉时间。桥一次附加可多batch使用，每forward清空信号，避免上一轮逐forward源码审计开销。

关键结构审计：HF TimeSformer suffix用image_size/patch_size写死14的reshape，70%空间保留59不能被14整除。新桥仅在已SHA审核的suffix block实例中，将H×W的逻辑槽分解改为S_retained×1，原有temporal/spatial attention及MLP操作不变，绝不改config/位置embedding、不把1×59称作物理图像网格。原14×14来源坐标关系仍由trajectory/representative映射解释。预算明确按空间轨迹：196→98/59，保留8个时间位置，因此TS总tokens预期1569→785/473；VideoMAE1568→784/470。需真实checkpoint训练侧证明，不把草案说成已生产通过。

新增tests/icassp2027/test_pre_block_bridge.py，CPU合成原生模型单测10passed/9skipped；9skip因本机Transformers4.53.2与已审核服务器VideoMAE新源码不同，保护门拒绝，不扩大SHA白名单以迁就单测。TimeSformer本机原生函数与server审计一致，其identity、两预算/三方法、重复前向清状态、真实target input/MLP长度/恢复通过；compileall src tests通过。服务器原生权重训练侧检查仍未跑，必须先SwanLab online再做，之后才正式冻结/派发。当前没有新吞吐，不能按旧CLIP两小时承诺ETA。

下一步：核验四份training_qa与完整训练清单、查旧dense合同/预测位置及实际SHA是否可复用；完成server训练侧有界native gate和对应head消费，再冻结真实方法/源版本与接力。旧152格四encoder计划不执行，唯一范围以本次76格queue及用户新goal为准。

## 2026-10-15 02:26 HKT DSANet 三方法38项、完整质量及两层效率最终交付完成

38/38=100%完整提取+预测+云端成功，UCF19/XD19；4105577/4105577编码图像100%（19×69634+19×146449，非VideoMAEv2 clips单位）。36/36单次主指标视频级配对CI与两branch补充ROC/stepAP/梯形PR点值、原生五tIoU mAP齐全，36个唯一质量云端URL；38个唯一预测URL。所有比较实际head/manifest/crop/采样/桥策略/真值SHA门通过、bootstrap_seed0/10000保留，未按正式分数改参数，不重算旧preview或最终quality.json。

两层正式效率皆齐：encoder114组=19配置×B1/32×3独立PID；完整流程114组=两dataset×19×3独立PID，每组完整固定16个训练视频，共1824次完整视频测量。均GPU5独占前/后PID/UUID验证、20warm40测量的encoder原始样本与完整视频三轮/阶段/p50/p95/吞吐/allocated/reserved/插件信号选择gather耗时保留；CPU/存储共享边界披露。encoder Swan w2iyxnvr、pipeline mv2zg766 FINISHED/efficiency_completed=1回读。每次源版本相同，完整流程OpenAI视觉与DSANet自带clip独立命名加载且训练侧身份核验通过。效率不是仅共享生产吞吐。

最终报告仅复用38预测/36质量及已完成228计时组，未运行新模型/重bootstrap。目录work/dsanet-three-method-20261015-r01/reports/final-20261015/：FINAL_REPORT.md（36主指标及全部CI、两branch补充帧指标、原生五tIoU、完整效率和局限）、quality-results.csv、final-results.json（完整原结果/所有Swan links及来源SHA、未知作者seed/未校正/未测null）、efficiency-results.csv/json（76个三轮汇总格及全部原始重复样本）、artifact-sha256.json、cloud-verification.json、swanlab.json。最后汇总Swan fgsl8mtg FINISHED/report_completed=1与38/36/114/114摘要回读；客户端上传完成7文件12.6MB、本地SHA与CSV36/76行数已核验。SDK未提供云端保存文件逐SHA读取接口，明确不伪称云端文件byte-hash回读。

真实质量结论：dense UCF ROC-AUC88.867606%；XD主stepAP85.596471%，XD补充梯形PR-AUC85.596286%。36配置共14支持非劣、7确认劣于界限、15证据不足；UCF6/18支持（均block12结构对照），XD8/18支持（block12六格+Attention6r50+VisionZip9r50）。12/14支持来自最后attention后仅patch操作、不再影响CLS的结构对照；没有取得两dataset同时支持非劣的非平凡前层压缩格。单作者权重/训练seed未知、未多配置校正，不称多seed泛化或方法间显著性。

真实效率结论：所有B1压缩净倍率约0.2474–0.2950×（变慢），B32约0.8797–1.3255×依配置不同；完整视频UCF三轮配对中位倍率约0.8987–1.1105×、XD0.9131–1.0874×，不据最好点称普遍显著端到端加速。source guard/上下文/hooks开销已在net wall中，不把GPU组件节省冒充净效率；完整时序头非因果，来源时长比只为离线处理。所有负例保留、能耗/FLOPs/F1无证据保持null。

completion-audit.json及final-remote-audit.txt已完成范围级复核：远端38份sealed scores/summary逐SHA再验，36份preview存在；本goal GPU/CPU/原controller/logger已结束，GPU进程表只剩他人任务，1827及R5 13429/13430继续正常。两个启动前busy失败只在无PID/无输出证据后开新attempt，正常旧任务未重启；元数据65字符SHA错误以独立分析绑定显式纠正、旧记录保留，无全局门豁免。源码四个byte-exact本地annotated tree tags核验，未commit/push；不动用户已有/无关改动、不停其他项目。

key-attention-40在本机toml仍PAUSED已复核，本轮未创建定时任务，不归档聊天。当前queue状态completed_report_cloud_verified，无后续自动训练/新模型/额外实验。所有必要任务已完成，不再给ETA；最终报告云端链接https://swanlab.cn/@ryfrcd693/VAD-DSANet-Compression-20261015/runs/fgsl8mtg。

## 2026-10-15 01:40 HKT DSANet 独占encoder全114组完成、完整视频效率实跑及R5保护恢复

上一goal turn为实质进展（四卡接力、单次CPU及encoder计时）；本轮复核encoder Swan w2iyxnvr FINISHED/efficiency_completed=1，114/114组完整原始wall/CUDA事件/allocated/reserved与plugin分项已复用保存encoder-results-reused.json；3个独立PID、同训练输入SHA、B1/32、19配置各20warm40measure满足预算，无GPU或bootstrap重跑。B1 dense中位5.192418ms，Attention6r50 18.031462ms、VisionZip6r50 20.489110ms、uniform6r50 18.531887ms；配对三轮倍率约0.2950/0.2555/0.2807，明显变慢。B32 dense83.680786ms，同三格75.228518/75.050327/72.810993ms，配对倍率1.1211/1.1177/1.1398。是当前实现的净wall（包含源校验/上下文/hooks/额外QK/选择），不能抹去B1反例或用局部CUDA节省代替净成本。

新增benchmark_pipeline.py，GPU5独占空卡核验后在原在线run mv2zg766启动。DSANet保留其原clip包，OpenAI视觉包用独立名称装载原始同一源码/权重（无全局替换/无dtype补丁），每dataset每独立进程先对已通过P1的固定训练feature和对应发布head CPU分数做数值身份检查。UCF/XD首轮检查均通过。边界完整打开训练原视频→原16帧采样/作者单crop/尾段→B32 CLIP及原插件→对应固定DSANet→真实coarse/class CPU分数，无特征落盘；每dataset固定8normal+8abnormal完整16视频，19配置×3独立进程，预设114组/1824测量。统一先完整读取16个原视频文件作暖文件策略，每配置首视频另做未计时预热；组前后PID/UUID独占门、CPU/存储共享如实保留。当前66/114完整流程组已产出，原在线logger108044/远端runner3580及其当前原进程实活核验，未宣布全部效率完成。

生产接力两次启动前busy门拒绝（AttentionXD12r70 l7auuq3v、VisionZipXD9r50 a4ycvu8q）。均核验远端目录只有Swan URL，无run.sh/PID/输出、exit22，无GPU计算，保留失败回执和队列。原controllers96816/115772已退出；恢复只采用新attempt目录并接管其他正常/成功logger，未重启计算。第二次核实实际GPU1/2新占用PID13429/13430均fotile、cwd CVPR-JY2、精确R5_E17/E18命令，自01:21:38 node时间启动，各约11GiB；根据用户已有全16卡共享空余显存授权，新controller91388用精确PID/owner/cmd/cwd名单及free≥16GiB/B32派发，不停其他项目、不永久放行未知PID。queue-before-recovery-r02/r03、recovery-r02/r03及gpu-share-r03保留，所有实际task tracking_directory权威，不按旧默认目录猜。

当前28/38完整预测云端核验，24/36单次主指标CI/两branch补充点指标/原生定位质量核验。最近图像库存2608249/4105577=63.5294%（采集稍早于当前队列）。UCF18质量齐全；XD首Attention6r50 stepAP85.956976% vsdense85.596471%，delta+0.360505pp CI[-0.276952,+1.040418]支持非劣但不证明提升；6r70=84.203485% CI delta[-2.920881,+0.260363]非劣不足；9r50=85.435173% CI[-0.572433,+0.271602]不足；9r70=84.487239% CI[-2.133651,-0.010546]不足但支持下降；12r50接近dense、支持非劣，结构对照。stepAP主与梯形PR补充不互换，不据这些test结果改变方法/预算。实际XD Attention生产median约246.13图/s、VisionZip243.27（目前样本少；共享工时不作效率）。

重要执行源版本已用fingercd/noreply身份annotated tags绑定实际源码tree，不创建branch commit、不改主index、不push。最初Git autocrlf正规化tag保留为记录；新formal-source-r01-raw-v2、quality-source-r04-raw-v2逐blob与archive原字节核验，encoder/pipeline原tag字节本来一致。source-tags.json有全部tree/tag/archive SHA与正规化差异，不误把脏代码当HEAD。第三方export源码当前逐py SHA记录native-dependency-source-sha.json，原导出没有.git，不编造Git revision。

最终汇总脚本build_final_report.py已实现但未发布假最终结果：严格要求38预测/36质量/114encoder/114pipeline云端、重复PID/输入/权重/源码/tag/全配置预算门；已验证当前partial被拒绝。只复用既有preview/计时，导出MD/CSV/JSON及SHA并在线记录、回读，显式区分ROC/stepAP/梯形PR单位；尚待全部数据完成再执行。compileall src/tests及本轮work脚本通过。

ETA按实际XD方法剩余队列及完整流程每组约23–28s更新：38预测暂HKT02:00–02:35；全部质量、余下完整流程效率、报告及云端核验暂02:30–03:30。R5共享/I/O、未测XDuniform及后续独占窗口不确定；当前任务不是被暂停或blocked，goal active；旧定时任务仍PAUSED，聊天未归档。

## 2026-10-15 01:04 HKT DSANet 四卡接力、单次CPU质量与独占encoder计时实产

上一goal turn为实质进展（原生训练门通过、两项新dense启动）；本轮接管两个原在线logger，无GPU重启，新增唯一生产接力controller96816并按完成后同dataset dense ready派任务。四卡0/1/2/3每卡一个任务，GPU0仅放行现场核实的yangrunze PID1827及完整uvicorn gpu_service命令；free>=16GiB、冻结B32限制本项目显存，未知PID拒绝，不停他人。当前16/38完整提取+预测+FINISHED云端核验，42.11%；最近读取累计1431658/4105577编码图像=34.87%，每task只计一次。GPU任务/原Swan身份与tracking_directory/expected_snapshot均在queue.json。两dense已成功，UCF主coarse ROC-AUC88.867606%，XD主仍待其压缩配对CPU输出，不从另一轮指标套用。

新增DSANet quality-one入口，仅完成且匹配dense的单taskCPU评测：保持对应head/输入采样/实际bridge+selector SHA、真值SHA门；主指标UCF ROC-AUC/XD stepAP用原compare_paired_quality显式bootstrap_seed0、10000视频级分层配对；两branch的ROC/stepAP/梯形PR-AUC点值以及原生5tIoU定位mAP分别保留，补充点值不冒充已有CI。dense补充点值/mAP通过输入/真值/模块SHA缓存复用，独立quality-preview.json和main-ci.json，不覆盖quality.json、不重复全矩阵bootstrap。68项相关单元测试通过，compileall src/tests通过。CPU主控117348（quality_controller_r02.py）串行，当前9/36成功云端结果，原CPU不重投。

启动CPU前出现元数据错误：原campaign UCF真值SHA被cat无换行尾部的右花括号拼成65字符，qm38khb3失败且在帧标注解析/bootstrap之前被严格SHA门拦住，原PID32767已退出/exit1，无preview/无CI计算。实际原真值文件hash重新核验为3fdf2faf…，未变化。原campaign.json/原SHA/原失败保留；新增analysis-campaign-r01.json指向原campaignSHA并显式记录这一字段的metadata correction，仅CPU使用此新绑定，预测/方法/预算/采样/权重不变，无全局门豁免。质量加载器在CPU-only薄入口装载同一源码，绕过无关encoder registry所需配置文件；三个仅help的缺打包依赖失败保留，未算模型实验/未改旧源快照。

已完成UCF Attention6r50=88.518695% delta−0.348912pp CI[-1.215277,+0.408242]非劣证据不足；6r70=87.190058% delta−1.677549 CI[-3.453344,-0.197448]非劣证据不足但支持下降；9r50=88.747443% delta−0.120163 CI[-0.832396,+0.542642]证据不足；9r70=87.912143% delta−0.955463 CI[-2.326395,+0.181884]证据不足；12两档约88.8676%且支持非劣，符合仅patch操作已无后续跨token影响CLS的结构对照，不是selector优越证据。VisionZip6r50=88.203585% delta−0.664021 CI[-1.605750,+0.145344]证据不足；6r70=86.407404% delta−2.460202 CI[-4.359152,-0.937348]确认劣于−0.5pp界限。单作者权重、未多配置校正，不据上述测试分数改正式队列/方法。

GPU5在已授权node3资源内原本空闲且严格empty-PID门通过，独立encoder计时在线logger114352已启动，19配置×B1/32×3独立进程=114组，20warmup/40同步实测、交错顺序。使用原formal-source-r01桥/策略，net wall包含构造/附加hooks/QK/selector/gather/readout所有实际成本；plugin组件为另行带时间书签的原forward AST副本，不混入net样本，分项边界明确，不将分项相加假冒净时延。每组前后验证独占GPU UUID只出现自己的CUDA PID；第一重复38组完成，剩两独立重复实跑，未宣布全效率完成。完整原视频两dataset×19×3×16视频效率尚待实现/执行，仍为goal必需，不能只以encoder结果完成。

现场生产提取完整任务median UCFdense约294.81图/s、Attention362.89、VisionZip356.04；XDdense242.28图/s（均共享生产吞吐，不入正式效率表，XD压缩未测）。按剩余18个XD压缩配置以dense速率仅作代理且允许共享/I/O/初始化差异，当前UCF批次暂HKT01:10–01:25，38项提取预测约02:00–02:45；包含尚未验证的完整流程效率、全部CPU、最终报告及云端核验暂03:00–05:00，非承诺，下一轮按实际分dataset/method更新。node3系统时钟比本地/工具UTC转HKT慢约数分钟，吞吐用同机elapsed，ETA用工具/本地HKT，不混加时钟。无定时任务恢复、无head训练、无据test调参、无commit/push，goal保持active。

## 2026-10-15 00:43 HKT DSANet 原生门通过，两项新dense正式启动

本goal上一轮为实质进展（P0及桥/CLI实现）；本轮完成原生训练侧有限验收并启动真实正式提取。原生检查r03 SwanLab 91v233oa FINISHED/qa_completed=1云端回读核验，18个配置全部真实post-attention shape197→99/60、三处identity与第12层CLS输出通过预先使用的FP32 rtol1e-4/atol1e-5门，B32检查通过，两数据集对应发布head各消费dense+三方法的4个训练视频，8个head检查全部成功。训练侧共每dataset2normal+2abnormal、首16帧固定输入，未读取test真值。

两次已定位修复及反例保留：r01错误把合成单测bitwise要求套到真实FP32运行（native连续自身亦有约4.8e-6差异），失败8t14u16q CRASHED云端核验；r02完成18个CLIP格子后，同进程缓存OpenAI CLIP遮蔽DSANet自带clip，导致DSANet文本分支dtype不匹配，失败原记录保留。r03按历史生产模式以独立进程评分解决，不改DSANet权重、dtype处理或方法。未无限扩展检查。代码回归86项通过、compileall src/tests通过；新bootstrap_seed参数默认保留旧20260918，未来DSANet单task入口显式seed0，不能把旧结果追溯改种子。

formal-source-r01.tar.gz及campaign.json两份SHA远端核验后只读展开；冻结三方法/blocks6,9,12/remove50,70、B32/workers2、原test crops/采样、原CLIP SHA和对应head SHA，38项ID不变。正式数据分母为19×(69634+146449)=4105577张测试编码图像，不是VideoMAEv2的4086273clips。源码快照/各源码SHA固定，未commit/push，不碰已有资产。

GPU1 UCFdense远端CUDA PID17645、本地原在线logger126560；GPU2 XDdense PID17644、原logger133524；均已ps/UUID/持续写入核验。最近读回UCF41/290视频11635图，XD70/800视频7754图，合计19389/4105577≈0.4723%；正式完整提取+预测+成功云端完成0/38，不能将部分产出算完成。GPU0本轮暂未启动（保护其他用户1827），GPU3暂待对应dense ready后的压缩接力。原两个GPU/在线身份不重启。queue.json记录每行tracking_directory/swanlab_source/expected_snapshot及真实slot/PID。

后续继续实现四卡单任务接力与DSANet单配置quality-preview/原生定位输出，38预测结束不自动重算整套bootstrap；独占效率及报告仍为目标必要部分。首批只有初始化后短窗口，不从视频计数或旧模型速度给本轮整套ETA；按实际图像吞吐/后续方法补齐。旧40分钟heartbeat仍暂停，本轮未新建定时任务。

## 2026-10-15 00:26 HKT DSANet 三方法 goal 已激活，P0 完成、P1 接入中

用户明确授权按 DSANET_THREE_METHOD_COMPRESSION_PLAN_20261015.md 建立 goal 并执行。本轮38项固定head提取预测、36项单次质量、计划规定独占效率及最终云端报告均在目标内；未启动head训练、未恢复旧heartbeat、未commit/push。新campaign为work/dsanet-three-method-20261015-r01，queue.json状态p1_implementation_not_formally_frozen，正式完成0/38、正式GPU任务0。既有工作树/未跟踪资产已记录并保护，根A.md不存在，不据缺失文件猜Git授权。

P0只读node3公网核验成功。原CLIP环境Python3.9.23/torch2.0.1+cu117可复用；CLIP权重SHA5806e77c…、DSANet UCF cc674979…与XD78317c34…当前文件SHA核验一致。UCF290/XD800 test、UCF1610/XD3950 train manifests当前全部原视频存在。重要crop澄清：既有test manifest为UCF crop[5]、XD crop[0]（每时刻1图），train为0–9共10crop；本轮保持历史测试协议，不把计划中泛称多crop解读为擅自改10crop测试。资产全文回执work/dsanet-three-method-20261015-r01/asset-audit.json，未读取正式质量分数。GPU1/2/3空闲；GPU0其他用户PID1827占约12GB，GPU4/6/7亦有其他项目，不停止/修改任何foreign任务。现场资源不等于永久独占预订。

P1新增clip_post_attention.py，通过已审计native block/attention AST身份限制，调用原native attention（不改need_weights/backend），同ln1输入额外计算patch-only Q/K选择信号，CLS不进入selection softmax/匹配/预算、native attention保留CLS，单次post-attention residual/pre-MLP缩短。复用原relation_selection三策略；DSANet CLI新增method/block/.5/.3，旧PairSelect预算/身份保留。新关系身份记录桥/selector SHA及执行shape。合成单元验证identity、真实MLP/suffix长度、最后block CLS预期与异常恢复；首次测试发现本机Python3.12+空type_params导致预设AST哈希未正规化，修正哈希生成后34项受影响单元测试通过（含旧CLIP/DSANet轻量入口），python -m compileall -q src tests通过。此为合成代码测试，不是原生checkpoint训练侧验收，不能提前宣布bridge已通过正式门。

下一步：冻结有限训练侧2normal+2anomalous输入，先SwanLab online再做原生模型检查、对应DSANet消费/身份核验；检查通过后才冻结正式实现并接力38项。单task质量入口/调度与效率尚待完成。当前没有本轮真实吞吐，不以旧VideoMAEv2或旧DSANet时间承诺本轮ETA。

## 2026-10-14 17:52 最终交付：38项及质量报告全部完成

38/38=100%提取+完整预测完成（UCF19/19、XD19/19），4086273/4086273clips=100%；36/36压缩配置单次配对质量结果全部完成，38份独立预测云端成功回执、36份独立质量成功回执已汇总。最后XDuniform12remove70原Swan wsfe36x1 FINISHED，原计算/记录未重跑；质量r36 Swan jsofr41c FINISHED/quality_completed=1核验。主作者梯形PR-AUC76.128493%（dense76.181075%），delta−0.052583pp，95%CI[-0.118150,-0.001173]pp，支持非劣（下界≥−0.5）但CI低于0支持微小下降，不能写成提升/无下降。stepAP70.314135%（dense70.333010%）单列。最终quality-result.json已保存，原head/采样/真值SHA门通过、独立preview、未覆盖finalquality.json。

最终报告复用36份原质量JSON，没有重算旧bootstrap/GPU任务。目录work/key-attn-formal-20261013/reports/final-20261014/：FINAL_REPORT.md（全18格双数据集主指标与判定/全部CI/XDstepAP/解释边界）；quality-results.csv（36配置完整数值）；final-results.json（38预测回执与云端链接、36质量源SHA及源输入SHA绑定、campaign/head/data/代码身份、失败恢复说明）；artifact-sha256.json；cloud-verification.json。汇总过程中修正dense在campaign内实际名为identity，以及早期恢复回执链接需从原events/controller.log恢复的问题，均仅汇总校验，不改变质量数值/重算实验。最终核对38个唯一预测run URL、36个唯一质量run URL；最终报告Swan oqyvuz18 FINISHED/report_completed=1、38预测/36质量摘要核验，文件上传已完成。当前controller8队列状态completed_report_cloud_verified。

最终非劣统计：Attention UCF2/6、XD3/6；VisionZip UCF2/6、XD4/6；uniform UCF2/6、XD2/6。共15/36支持非劣、2/36确认劣于界限（XDuniform6r50/70）、19/36证据不足。XD attention/VisionZip第9层两档CI高于0；UCF对应CI跨0；单headseed、未多配置校正，不将点估计排序当方法间显著差异，不据正式分数调冻结矩阵。

正式独占净延迟/插件开销/峰值显存未测量，最终报告相关字段明确null，不能用共享提取吞吐或执行管线工程加速替代压缩方法效率。当前交付是完整冻结质量矩阵与provenance，未宣称质量—效率论文结论已成立。旧失败、协议、指纹及finalquality.json原位保留；不改其他项目，不新增训练/benchmark/QA。

已按用户完成条件将key-attention-40定时任务设为PAUSED（停止每40分钟唤醒），未归档聊天。最终完成时间17:52 HKT，不再提供预计时间范围。未commit/push；仅progress.md为本轮跟踪文档修改，其他用户资产不动。
## 2026-10-14 17:05 定时汇报：全部clips写入完成，待最后预测

37/38=97.37%完整提取+预测+云端确认（19UCF/18XD），累计4086273/4086273clips=100.00%，按每任务仅计一次。最后XDuniform12remove70已800/800、145703clips全写入，尚未extracted/completed最终回执，不提前将38项计为完成。原在线logger136892、原GPU0 PID2003仍正常，node3公网单次只读17:01:31 fotile/Rl，无新错误。GPU1/2/3任务已完成，无待派任务；已停止的旧调度器不重启（避免重复quality），本heartbeat持续观察原任务，无GPU/其他项目变更。

r01–r35已核验指标保持，没有新质量结果、不重算。最后XDuniform12r70必须等prediction_completed及云端FINISHED后再单次CPU比较，复用原head/采样/真值SHA门和独立preview。UCF三方法全部质量齐全，XDattention/VisionZip/其他uniform齐全；最后XDuniform12r50=76.176378%主梯形PR-AUC delta−.004697pp CI[-.029657,+.019638]pp支持非劣；stepAP70.344682%单列。冻结方法/head/grid不改。

最后35kclips在近39m28s完成，当前ETA HKT原任务后处理/预测/记录17:15–17:40；整套38+最终报告仍暂今日18:00–23:00，最终CPU/bootstrap、复用报告整理和云端核验仍须完成，后处理/共享/通信不确定。只剩原任务收尾，勿以clips100%宣称整个任务完成。无源码/训练/扩展QA/新CPU任务。
## 2026-10-14 16:25 定时汇报：复用结果收尾接管

37/38=97.37%完成（19UCF/18XD），累计4051128/4086273clips=99.14%。最后GPU0 XDuniform12remove70原PID2003、原在线logger136892正常，无新增错误，node3公网单次只读16:22:03：693/800、110558clips，fotile/Rl。GPU1/2/3任务均完成，无待派任务，不额外开重复计算/不动其他项目。

已核实旧调度器在38项完成后默认进入quality阶段重算全部compare_prediction_runs，这与用户最新“不重复计算、复用已有结果”冲突。仅停止已无待派任务的本地主控138236（command核验为controller8），未停止/重启最后GPU进程或原SwanLab logger。queue-before-quality-reuse-detach.json保留原队列；当前queue状态test_running_heartbeat_observed_quality_reuse，保留原task row/PID，后续heartbeat观察原logger的cloud完成并更新行，最终报告复用各独立preview。不会把主控停止误报为GPU故障，也不恢复旧重复评测入口。

r35 XDuniform12remove50 CPU已完成并SwanLab piogx9ye FINISHED/quality_completed=1核验，quality-result.json保存：主作者梯形PR-AUC76.176378%（dense76.181075%），delta−0.004697pp，95%CI[-0.029657,+0.019638]pp，支持非劣但CI跨0不确认提升。stepAP70.344682%（dense70.333010%）单列。r01–r34既有指标保持，原head/采样/真值SHA门与独立preview/final不覆盖，冻结方法/head/grid不改。现在35个压缩配置质量结果已齐，仅最后XDuniform12r70质量待预测后执行一次。

近40m13s最后XDuniform12r70约16.04clip/s，剩35145clips，ETA HKT今日17:10–17:40（含预测/记录收尾）；整套38+最终报告仍暂18:00–23:00，最后视频/预测/上传/报告核验不确定。调度观察非正式效率，无源码/训练/扩展QA/新CPU任务。
## 2026-10-14 15:45 定时汇报

主控138236与最后一个在线logger正常，无新增错误。XDuniform12remove50完整预测并云端mdtovj8j FINISHED，37/38=97.37%完成（19UCF/18XD），累计4012423/4086273clips=98.19%。node3公网单次只读15:41:50：最后GPU0 XDuniform12remove70 529/800 71853clips PID2003，fotile/Rl。GPU1/2/3本轮GPU任务均已完成，无待派任务，不额外开重复任务；其他项目不动，最后正常计算不重启。

新完成XDuniform12remove50启动单次原consumer在线CPU r35 local140528，原head/采样/真值SHA门，独立preview不覆盖final，预建日志，当前待回收后续读取不重算。r01–r34已有指标不变（r17 recovered回执），冻结方法/head/grid保持。UCF质量全部回收，XD只剩uniform12两项待质量回收；最终整理必须复用既有preview避免重复bootstrap，38后原调度器quality阶段为旧默认重复计算入口，收尾时应拦截旧入口并保留正常GPU/logger，单独复用结果完成最终导出。

近40m4s最后XDuniform12r70实际约15.39clip/s，按剩余73850clips ETA HKT今日17:15–18:00（含预测/记录收尾范围）；整套38+最终报告仍暂今日18:00–23:00，0待派1在跑，最后视频长度/后处理/共享/通信及报告核验不确定，观察速度非正式效率。无源码/训练/扩展QA。
## 2026-10-14 15:05 定时汇报

主控138236/controller-r03-recovery8与两在线logger正常，无新错误/新prediction_completed，36/38=94.74%完成（19UCF/17XD），累计3959585/4086273clips=96.90%。node3公网单次只读15:01:46实际slot：GPU0最后XDuniform12remove70 305/800 34858clips PID2003 Sl（持续产出正常等待）；GPU1 XDuniform12remove50 756/800 129860clips PID6346 Rl，均fotile。GPU2/3任务已完成、无待派，不额外开重复工作。正常任务/其他项目不动，无新CPU评测。

上一轮r34 XDuniform9remove70 CPU已完成并SwanLabnzi52ca7 FINISHED/quality_completed=1核验、quality-result.json保存：主作者梯形PR-AUC75.458880%（dense76.181075%），delta−0.722195pp，95%CI[-1.048517,-0.410340]pp。非劣证据不足（CI跨−0.5），但整个CI低于0支持本配置/当前headseed质量下降；不能称CI跨0，也不能确认整个区间低于−0.5。stepAP69.797228%（dense70.333010%）单列。r01–r33指标保持（r17 recovered回执），原head/采样/真值门/独立preview不覆盖final，冻结方法/head/grid不改。UCF全部质量齐全，XD仅剩12层两项待完成。

新GPU0 XDuniform12r70首39分钟约14.84clip/s含初始化，比旧共享卡12层快，但未完成整段仍宽估；GPU1近轮約9.8clip/s。ETA HKT今日GPU0 17:15–18:20；GPU1 15:45–16:10；GPU2/3已完成。整套38+最终报告仍暂今日18:00–23:00，0待派2在跑，最后配置后段/共享/通信及最终报告核验不确定，调度观察不入正式效率。无源码/训练/扩展QA。
## 2026-10-14 14:25 定时汇报：最后任务已恢复启动

新增UCFuniform12remove50/70（y4s89olg/4ghjucb3）及XDuniform9remove70（9auje3qc）完整预测云端FINISHED，36/38=94.74%完成（19UCF/17XD）。最后XDuniform12remove70在GPU2启动前因foreign R3→R4 PID换代被Unknown门拦住，原bktbnqz1 CRASHED保留；远端无新taskPID/无GPU计算已核验。只读核验GPU1/2新R4 PIDs29679/29680均fotile、完整配置R4_E13/E14，GPU2空闲29343MiB/GPU3空闲40326MiB，旧R3名单备份，运行allowlist1/2更新为已核实新PID，GPU3恢复原空卡严格门；free12GiB/allocator.25与owner/command门不变。仅停止held126548，新主控138236/controller-r03-recovery8接管唯一正常logger/GPU1任务不重启；从resume输入剔除已证明未启动失败行，新attempt最后XD12r70分派GPU0 local136892，preflight通过。保护1827与所有其他项目，无模型/方法/seed/采样/精度或代码快照更改。

当前累计3902612/4086273clips=95.51%（新GPU0恢复后9/800、1418clips计入）。实际GPU0最后XDuniform12remove70已开始实产9/800、1418clips；GPU1 XDuniform12remove50 682/800 106327clips PID6346；GPU2/3本轮任务已完成且无待派GPU任务，不额外开重复任务。两个在线logger正常。node3公网只读14:21:47+后续full-command/PID确认，slot权威以当前queue为准。

新三项分别单次原consumer在线CPU：r32 UCFuniform12remove50 local105580；r33 UCFuniform12remove70 local132964；r34 XDuniform9remove70 local93332，原head/采样/真值门、独立preview不覆盖final、预建日志，r32/r33均已完成并SwanLab afiumgzg/1fsjdor9 FINISHED/quality_completed=1核验，quality-result.json保存。UCFuniform12remove50 ROC-AUC80.689946%，delta+.057988pp，95%CI[-.036016,+.147242]pp；12remove70 ROC-AUC80.646241%，delta+.014283pp，95%CI[-.173193,+.176559]pp，两项支持非劣但CI跨0不确认提升。r34 XDuniform9remove70仍CPU bootstrap remote2699，后续读取本次不重算。r01–r31已核验指标保持，冻结方法/head/grid不据正式分数调参。

ETA HKT今日GPU0最后XD12r70 18:00–19:30（新GPU/共享窗未稳，参考已测XD第12层范围）；GPU1 15:40–16:20；GPU2/3已完成。整套38+最终报告仍暂今日18:00–23:00，0待派2在跑，新R4共享与最后配置/通信不确定。没有以堆进程冒充四卡有效产出，不重跑已完成任务。无源码/训练/扩展QA。
## 2026-10-14 13:45 定时汇报

主控126548及四在线logger正常，无新错误/新prediction_completed，33/38=86.84%完成（17UCF/16XD），累计3864681/4086273clips=94.58%。node3公网单次只读13:41:37实际slot：GPU0 UCFuniform12remove70 257/290 62002clips PID21338；GPU1 XDuniform12remove50 580/800 80437clips PID6346；GPU2 UCFuniform12remove50 extracted全69364，预测待回执 PID30129；GPU3 XDuniform9remove70 793/800 142442clips PID20790，全fotile/Rl。无本轮新补位/新CPU，full提取不提前计pred完成；正常任务/其他项目不动。

上一轮r31 XDuniform9remove50 CPU已完成并SwanLab55vzs0dq FINISHED/quality_completed=1核验、quality-result.json保存：主作者梯形PR-AUC75.838740%（dense76.181075%），delta−0.342336pp，95%CI[-0.575540,-0.139129]pp。非劣证据不足（区间跨−0.5界限）；但整个CI低于0，支持本配置/当前headseed的质量下降，不能写成CI跨0或已确认超过0.5。stepAP70.042634%（dense70.333010%）单列。r01–r30其他指标保持、原head/采样/真值SHA与独立preview不覆盖final；冻结方法/head/grid不改，不据分数调参。

近39m59s UCFuniform12r70约15.07clip/s、XDuniform12r50约11.59、XDuniform9r70約14.0；按剩余clips和预测收尾ETA HKT今日GPU0 14:00–14:20；GPU1 15:30–16:20；GPU2/3 14:00–14:20。整套38+报告仍暂今日18:00–23:00，1待派4在跑，最后XDuniform12r70接力/共享/通信不确定，调度观察非正式效率。无源码/训练/扩展QA。
## 2026-10-14 13:05 定时汇报

主控126548与四在线logger正常，无新错误。XDuniform9remove50完整预测并云端0qan4xuz FINISHED，GPU0自动补UCFuniform12remove70；33/38=86.84%完成（17UCF/16XD），累计3743343/4086273clips=91.61%。node3公网单次只读13:01:38实际slot：GPU0 UCFuniform12remove70 119/290 25838clips PID21338；GPU1 XDuniform12remove50 405/800 52621clips PID6346；GPU2 UCFuniform12remove50 199/290 45639clips PID30129；GPU3 XDuniform9remove70 689/800 108809clips PID20790，全fotile/Rl；正常任务不重启/其他项目不动。

上一轮r29 XDuniform6remove70已完成并SwanLabgpbitl8e FINISHED/quality_completed=1核验、quality-result.json保存：主作者梯形PR-AUC73.265207%（dense76.181075%），delta−2.915868pp，95%CI[-3.562302,-2.283542]pp，inferior：整个区间低于−0.5，确认本配置/当前headseed超过允许下降幅度；stepAP68.148919%（dense70.333010%）单列。新完成XDuniform9remove50已启动原consumer单次在线CPU r31 local133600，原head/采样/真值SHA门，独立preview不覆盖final，当前待结果后续读取不重算。r01–r28/r30既有指标保持，冻结方法/head/grid不据分数调整。

近38m5s XDuniform12r50约14.0clip/s、UCFuniform12r50约9.38、XDuniform9r70约11.6；新UCFuniform12r70启动28min约15.2含初始化但共享差异，宽估。ETA HKT今日GPU0 14:05–14:45；GPU1 15:10–16:00；GPU2 14:00–14:35；GPU3 14:05–14:45。整套38+最终报告仍暂18:00–23:00，1待派4在跑，最后XDuniform12r70需空位接力且未完整实测/共享/通信不确定。调度观察非正式效率，无源码/训练/扩展QA。
## 2026-10-14 12:25 定时汇报

主控126548及四在线logger正常，无新错误。新增XDuniform6remove70（w9r0zwsa）与UCFuniform9remove70（r1hntmne）完整预测云端FINISHED，GPU1/2自动补XD/UCF uniform12remove50。32/38=84.21%完成（17UCF/15XD），累计3637665/4086273clips=89.02%。node3公网单次只读12:23:33按实际slot：GPU0 XDuniform9remove50全800/800、145703clips，后处理/预测待回执 PID25680，不提前计完成；GPU1 XDuniform12remove50 220/800 20704clips PID6346；GPU2 UCFuniform12remove50 114/290 24209clips PID30129；GPU3 XDuniform9remove70 586/800 82316clips PID20790，全fotile/Rl。正常任务未重启/其他项目不动。

新完成两项单次原consumer在线CPU：r29 XDuniform6remove70 local117164；r30 UCFuniform9remove70 local21172，原head/采样/真值SHA门与独立preview，预建日志不覆盖final，r30已完成并SwanLab k980aiwv FINISHED/quality_completed=1核验：UCFuniform9remove70 ROC-AUC80.671635%，dense80.631958%，delta+0.039677pp，95%CI[-1.161744,+1.191163]pp，非劣证据不足；原结果quality-result.json保存。r29 XDuniform6remove70仍CPU bootstrap remote652，后续读取本次不重算。GPU0预测未完成暂不启动对应质量。r01–r28已核验指标保持，冻结方法/head/grid不据正式分数改。

第12层uniform启动窗实产较慢：XD约9.2clip/s（37min含初始化）、UCF约9.6（42min），XDuniform9r70近轮約10.8。ETA HKT今日GPU0收尾12:40–13:00；GPU1 16:15–17:15；GPU2 14:00–14:45；GPU3 14:15–15:15。整套38+最终报告修订暂今日18:00–23:00（旧16–23下界不再最新），2待派4在跑，未派XDuniform12r70还须等下个slot接力，按当前完整前缀/共享速度不能维持16点最早收尾。后续速率/共享/通信仍不确定，调度观察非正式效率。无源码/训练/扩展QA。
## 2026-10-14 11:44 定时汇报

主控126548及四在线logger正常，无新增错误/新prediction_completed，30/38=78.95%完成（16UCF/14XD），累计3527905/4086273clips=86.34%。GPU1 XDuniform6remove70、GPU2 UCFuniform9remove70均已extracted全部clips，尚在预测收尾，不提前计完成或启动CPU质量。node3公网一次只读11:40:35：GPU0 XDuniform9remove50 688/800 108640clips PID25680；GPU1 XDuniform6remove70 extracted145703 PID23034；GPU2 UCFuniform9remove70 extracted69364 PID27572；GPU3 XDuniform9remove70 418/800 54532clips PID20790，全fotile/Rl。无本轮新补位，自动接力保持，正常任务/其他项目不动。

r01–r28质量指标保持不变，复用已有结果不重算，不因变化小静默。Attention/VisionZip质量已全部回收，最后XDVZ12r70=76.291015%主梯形PR-AUC，delta+.109939pp，CI[-.523987,+.720684]pp非劣证据不足；stepAP71.957733%单列。uniform已完成UCF6r50/70/9r50均非劣证据不足、XD6r50=74.865774% delta−1.315301pp CI[-1.631973,-1.002609]pp为inferior，不调整冻结方法/head/grid。

近37m50s XDuniform9r50约15.5clip/s、XDuniform9r70約10.76，按各自剩余clip和预测收尾ETA HKT今日GPU0 12:35–13:00；GPU1/2 11:50–12:15；GPU3 14:15–15:10。整套38+最终报告仍暂今日16:00–23:00，4待派4在跑，后层uniform/共享/通信不确定，调度观察不入正式效率。无源码/训练/扩展QA/新CPU任务。
## 2026-10-14 11:04 定时汇报

主控126548及四在线logger正常，无新错误/新prediction_completed，不启动新CPU、不重算。30/38=78.95%完成（16UCF/14XD），累计3421569/4086273clips=83.73%。node3公网单次只读11:02:45按实际slot：GPU0 XDuniform9remove50 541/800 73466clips PID25680 Sl（有持续产出正常等待）；GPU1 XDuniform6remove70 732/800 122683clips PID23034；GPU2 UCFuniform9remove70 199/290 45639clips PID27572；GPU3 XDuniform9remove70 278/800 30115clips PID20790，其余Rl全fotile。无本轮新补位，自动接力保持，正常任务/其他项目不动。

最后r27 XDVZ12remove70 CPU已完成并SwanLab6edusn8x FINISHED/quality_completed=1核验，quality-result.json保存：主作者梯形PR-AUC76.291015%（dense76.181075%），delta+0.109939pp，95%CI[-0.523987,+0.720684]pp，下界略低于−0.5，非劣证据不足，不四舍五入放行。stepAP71.957733%（dense70.333010%）单列。至此attention与VisionZip全部质量已回收。均原head/采样/真值门与独立preview/final不覆盖，r01–r26/r28旧指标保持。uniform已完成UCF6r50/r70/9r50均非劣证据不足，XD6r50为inferior；其余待完整任务，不据分数调整冻结格子。

近39m4s XDuniform9r50約14.7clip/s、XDuniform6r70約13.1、UCFuniform9r70約10.4；XDuniform9r70首48分钟约10.4含初始化，仍共享波动。ETA HKT今日GPU0 12:35–13:15；GPU1 11:45–12:15；GPU2 11:50–12:20；GPU3 14:10–15:20。整套38+报告仍暂今日16:00–23:00，4待派4在跑，后层uniform/共享/通信不确定，调度观察非正式效率。无源码/训练/扩展QA。
## 2026-10-14 10:24 定时汇报

主控126548及四在线logger正常，无新错误。最后XDVZ12remove70（j1590jcp）与UCFuniform9remove50（qzr2na9z）完整预测并云端FINISHED，GPU2/3自动补UCFuniform9remove70、XDuniform9remove70；30/38=78.95%完成（16UCF/14XD），累计3307362/4086273clips=80.94%。node3公网一次只读10:23:41按真实GPU slot：GPU0 XDuniform9remove50 331/800 39018clips PID25680；GPU1 XDuniform6remove70 637/800 91943clips PID23034；GPU2 UCFuniform9remove70 101/290 21201clips PID27572；GPU3 XDuniform9remove70 45/800 5534clips PID20790，全fotile/Rl。正常任务不重启/其他项目不动。Attention/VisionZip全部提取+预测已完成，uniform收尾仍剩8项。

r26 XDuniform6remove50已有CPU结果云端zy3xgikk FINISHED/quality_completed=1核验（此前用户即时问进度已报告），本轮保存quality-result.json：主作者梯形PR-AUC74.865774%（dense76.181075%），delta−1.315301pp，95%CI[-1.631973,-1.002609]pp，inferior：整个CI低于−0.5，确认为本预定配置/当前headseed超过允许下降幅度，不只是证据不足。stepAP69.361571%（dense70.333010%）单列。新完成两项单次在线CPU：r27 XDVZ12remove70 local129728；r28 UCFuniform9remove50 local95700，原head/采样/真值门、独立preview不覆盖final、预建日志，r28已完成并SwanLab x3lmuuh3 FINISHED/quality_completed=1核验：UCFuniform9remove50 ROC-AUC80.324850%，delta−0.307108pp，95%CI[-0.875000,+0.148325]pp，非劣证据不足，quality-result.json已保存。r27 XDVZ12remove70仍CPU bootstrap remote14261，后续读取本次不重算。冻结方法/head/grid不据正式分数调整。

实际近轮XDuniform9r50約14.14clip/s、XDuniform6r70約12.69；新UCFuniform9r70启动32min约10.88clip/s，XDuniform9r70启动9min尚不稳。ETA HKT今日GPU0 12:35–13:20；GPU1 11:45–12:20；GPU2 11:45–12:30；GPU3 14:20–15:40。整套38+最终报告修订暂今日16:00–23:00（旧14–23下界不再最新），4待派4在跑，剩余第12层uniform及共享/通信不确定，不能早于慢XD9任务和后续XD12波收尾。调度观察非正式效率，无源码/训练/扩展QA。
## 2026-10-14 09:44 定时汇报

主控126548和四在线logger正常，无新错误。XDuniform6remove50完整预测并云端64j9fqtl FINISHED，GPU0自动补XDuniform9remove50；28/38=73.68%完成（15UCF/13XD），累计3187596/4086273clips=78.01%。node3公网单次只读09:40:38按实际slot：GPU0 XDuniform9remove50 19/800 2480clips PID25680；GPU1 XDuniform6remove70 439/800 59175clips PID23034；GPU2 XDVZ12remove70已800/800全部145703clips PID16714，仍后处理/预测待回执，不提前计完成；GPU3 UCFuniform9remove50 199/290 45639clips PID23042，全fotile/Rl。正常任务未重启/其他项目不动。

新完成XDuniform6remove50单次原consumer在线CPU completed-metrics-r26 local119392，原head/采样/真值SHA门，独立preview不覆盖final、预建日志，当前待结果，后续读取本次不重算。GPU2虽所有clips已写但prediction未完成，不提前启动其质量评测。r01–r25已核验指标保持，冻结方法/head/grid不改。

近38m20s XDuniform6r70约12.81clip/s、UCFuniform9r50约8.40（共享下实速低于启动猜测）；新XDuniform9r50启动仅3分钟不足稳定。ETA HKT今日GPU0 12:30–13:30；GPU1 11:40–12:20；GPU2收尾10:00–10:20；GPU3修订10:35–11:00。整套38+最终报告仍暂14:00–23:00，6待派4在跑，后层uniform/共享/通信不确定，调度观察不入正式效率。无源码/训练/扩展QA。
## 2026-10-14 09:04 定时汇报

主控126548/controller-r03-recovery7及四在线logger正常，无新错误/新prediction_completed，不启动新CPU、不重算已有结果。27/38=71.05%完成（15UCF/12XD），累计3097765/4086273clips=75.81%。node3公网单次只读09:02:18按真实GPU slot：GPU0 XDuniform6remove50 751/800 127704clips PID31504；GPU1 XDuniform6remove70 275/800 29722clips PID23034；GPU2 XDVZ12remove70 743/800 125117clips PID16714；GPU3 UCFuniform9remove50 122/290 26326clips PID23042，全fotile/Rl。共享R3名单恢复后无新增拦截，没有本轮新补位，正常任务/其他项目不动。

r01–r25指标不变，复用已核验SwanLab结果（r17 recovered回执）。UCFuniform6remove50/70=80.055400/79.753894%，delta−0.576558/−0.878063pp，CI[-1.214343,+0.013139]/[-2.374078,+0.466921]pp，两项非劣证据不足，CI跨0不声称显著劣化。XDuniform尚无完成质量。VisionZip/attention原指标及判定保持，冻结方法/head/grid不改。

按近轮实际增量，XDuniform6r50约14.3clip/s、XDVZ12r70约9.3；新XDuniform6r70约12.1/UCFuniform9r50约10.7含启动，两新共享任务速率弱于早期猜测，修订ETA HKT今日GPU0 09:35–10:00；GPU1 11:45–12:45；GPU2 09:55–10:30；GPU3 10:10–10:50。整套38+报告仍暂今日14:00–23:00，7待派4在跑，后层uniform/共享/通信不确定，调度速率不是正式效率。无源码/新训练/扩展QA。
## 2026-10-14 08:24 定时汇报：R3共享进程换代及GPU表更正

新增UCFuniform6remove50/70完整预测并云端9m3hjpmt/at2th655 FINISHED，27/38=71.05%完成（15UCF/12XD）。下一项XDuniform6remove70启动前因foreign PID换代被Unknown门拦住，原i3am5sxr CRASHED保留；remote新任务无PID无GPU计算已核验。只读确认GPU1/2/3新CVPR-JY2 R3_E09/E10/E11 PIDs8903/8904/8905均fotile、完整cmd核实，GPU1/3空闲约23.5GiB；依据原9/21共享空余显存授权更新运行allowlist，保留R2名单备份，严格PID+owner+command/free>=12GiB/allocator.25门不变。仅停止held主控82184，新主控126548/controller-r03-recovery7直接接管两个旧正常logger；resume输入保留原tracking，仅剔除未启动失败行后新attempt。GPU1补XDuniform6remove70（local92456）、GPU3补UCFuniform9remove50（119772），preflight通过，旧两GPU未重启/其他项目不动。

本轮通过nvidia-smi UUID/PID及queue slot发现此前简报GPU表沿用任务显示顺序并非实际GPU号，明确更正：当前GPU0 XDuniform6remove50 PID31504（638/800 92292clips）；GPU1新XDuniform6remove70 preflight；GPU2 XDVZ12remove70 PID16714（671/800 101974clips）；GPU3新UCFuniform9remove50 preflight。任务统计/质量结果不受表内GPU标号错误影响，旧报告原样保留并以本更正为准。新两项preflight按0clips计，后续从actualslot读取。累计2983162/4086273clips=73.00%。四在线logger现正常。

新完成两项单次在线原consumerCPU：r24 UCFuniform6remove50 local122876/remote23710；r25 UCFuniform6remove70 local127008/remote23727，原head/采样/真值门通过、独立preview不覆盖final、预建日志，两项已完成并SwanLab q16ai8rs/96imaelv FINISHED/quality_completed=1核验，quality-result.json保存。UCFuniform6remove50 ROC-AUC80.055400%，delta−0.576558pp，95%CI[-1.214343,+0.013139]pp；6remove70 ROC-AUC79.753894%，delta−0.878063pp，95%CI[-2.374078,+0.466921]pp。两项点估计下降均超过0.5pp且CI下界低于-0.5，非劣证据不足；CI仍跨0不声称显著劣化。不重复计算。r01–r23已核验指标保持，冻结方法/head/grid不改。

ETA HKT今日按真实GPU号：GPU0 09:30–10:20；GPU1新XD6uniform70 11:30–12:30；GPU2 09:45–10:45；GPU3新UCF9uniform50 09:40–10:30。新R3共享任务及新配置缺稳定速率窗口范围宽；原XDuniform6r50近轮约15.8clip/s/XDVZ12r70约9.7，不跨任务互代。整套38+报告仍暂今日14:00–23:00，7待派4在跑，后层uniform/共享/通信不确定，调度观察非正式效率。无源码/训练/扩展QA。
## 2026-10-14 07:44 定时汇报

主控82184与四在线logger正常，无新增错误/新prediction_completed，故不启动新CPU/不重算。25/38=65.79%完成（13UCF/12XD），累计2883414/4086273clips=70.56%。node3公网单次只读07:40:26：GPU0 XDVZ12remove70 566/800 78221clips PID16714；GPU1 UCFuniform6remove50 221/290 56549clips PID27131；GPU2 XDuniform6remove50 410/800 53057clips PID31504；GPU3 UCFuniform6remove70 197/290 45419clips PID28018，全fotile/Rl。没有本轮新补位，自动接力继续，不重启正常任务/不动其他项目。

上一轮r22/r23均已回收原CPU结果并云端FINISHED/quality_completed=1核验，保存各自quality-result.json。r22 XDVZ12remove50（Swan zm0rur6r）：主作者梯形PR-AUC76.414071%（dense76.181075%），delta+0.232995pp，95%CI[-0.111887,+0.563082]pp，支持非劣但跨0不确认提升；stepAP71.318831%（dense70.333010%）单列。r23 XDVZ9remove70（jw84q9lp）：主作者梯形PR-AUC77.541321%，delta+1.360245pp，95%CI[+0.750563,+1.954626]pp，支持非劣且区间高于0，限本预定配置/当前headseed未多比较校正；stepAP73.215057%单列。原head/采样/真值SHA门一致、独立preview不覆盖final、不据分数调方法。其余指标保持。

近39m55s各自观察速率XDVZ12r70约9.63clip/s、UCFuniform6r50约12.57、XDuniform6r50约13.75、UCFuniform6r70约13.55；启动窗后的实测速率不同于首波估计，按剩余clips修订ETA HKT今日GPU0 09:45–10:45；GPU1 08:10–08:35；GPU2 09:45–10:30；GPU3 08:20–08:50。整套38+最终报告仍暂14:00–23:00，9待派4在跑，后层uniform未完整实测/共享/通信不确定，调度观察不入正式效率。无源码/训练/扩展QA。
## 2026-10-14 07:04 定时汇报

主控82184和四在线logger正常，无新增错误。新增UCFVZ12remove70（yi3e0lyo）、XDVZ12remove50（bfc1vems）、XDVZ9remove70（irt083dk）完整预测并云端FINISHED，GPU1/2/3分别自动补UCF uniform6remove50、XD uniform6remove50、UCF uniform6remove70。25/38=65.79%完成（13UCF/12XD），累计2764843/4086273clips=67.66%。node3公网单次只读07:00:31：GPU0 XDVZ12remove70 421/800 55150clips PID16714；GPU1 UCFuniform6remove50 123/290 26438clips PID27131；GPU2 XDuniform6remove50 215/800 20125clips PID31504；GPU3 UCFuniform6remove70 45/290 12962clips PID28018，全fotile/Rl；正常任务未重启、其他项目不动。三路正式uniform已开始实产，不调整冻结队列。

新三项分别单次原consumer在线CPU：r21 UCFVZ12remove70 local117032；r22 XDVZ12remove50 local89504；r23 XDVZ9remove70 local104080，原head/采样/真值SHA门、独立preview不覆盖final、预建日志避免tail竞态，r21已完成并SwanLab4n28z1kl FINISHED/quality_completed=1核验：UCFVZ12remove70 ROC-AUC80.906818%，dense80.631958%，delta+0.274860个百分点，95%CI[-1.036945,+1.510633]个百分点，非劣证据不足，quality-result.json已保存。r22/r23仍CPU bootstrap remote3954/3944，后续读取本次不重复计算。r01–r20已核验指标保持不变，不据正式分数选方法/层/预算。

近轮XDVZ12r70约11.39clip/s；首波uniformUCF6r50/70约17.1/17.5clip/s、XD6r50約14.1含启动，仅初始调度观察不是正式效率。ETA HKT今日GPU0 09:30–10:30；GPU1 07:50–08:20；GPU2 09:40–10:30；GPU3 08:00–08:40。整套38+最终报告仍暂14:00–23:00，9待派4在跑，uniform后层尚未完整实测/共享/通信不确定。无源码/新训练/扩展QA。
## 2026-10-14 06:24 定时汇报

主控82184与四在线logger正常，无新增错误/新prediction_completed，不重复CPU计算，22/38=57.89%完成（12UCF/10XD）。累计2652913/4086273clips=64.92%。node3公网单次只读06:20:22：GPU0 XDVZ12remove70 262/800 27705clips PID16714；GPU1 UCFVZ12remove70 204/290 52881clips PID18024；GPU2 XDVZ12remove50 789/800 140487clips PID28220；GPU3 XDVZ9remove70 793/800 142442clips PID27863（短暂Dl但近轮有持续产出，IO等待不判故障），其余Rl全fotile。没有本轮新补位，自动接力保持，不重启正常任务/不动其他项目。

r01–r20已核验质量指标不变，复用已有结果（r17以recovered云端核验为准），不因本轮无变化静默。VisionZip已支持非劣UCF9两项、XD6r50/XD9r50；UCF6两项/UCF12r50/XD6r70证据不足。XD9r50主作者梯形PR-AUC77.185239%，delta+1.004163pp，CI[+.639717,+1.365171]pp，本预定配置当前seed支持提升未多比较校正。stepAP分开，方法/head/grid冻结不改。

近39m55s XDVZ12r70约8.99clip/s、UCFVZ12r70约9.39、XDVZ12r50约14.15、XDVZ9r70约13.57。按剩余clips ETA HKT今日GPU0修订10:00–11:00（旧09:00–10:30不再最新，实际短窗后速度较慢）；GPU1 07:00–07:25；GPU2 06:40–07:00；GPU3 06:35–06:55。整套38+最终报告仍暂今日14:00–23:00，12待派4在跑，uniform未完整实测及共享/通信不确定。调度观察不入正式速度，无源码/新训练/扩展QA。
## 2026-10-14 05:44 定时汇报

主控82184和四在线logger正常，无新增错误。UCFVZ12remove50完整预测并云端93cj0aza FINISHED，GPU0自动补XDVZ12remove70；22/38=57.89%完成（12UCF/10XD），累计2542520/4086273clips=62.22%。node3公网单次只读05:40:27：GPU0 XDVZ12remove70 54/800 6186clips PID16714；GPU1 UCFVZ12remove70 146/290 30396clips PID18024；GPU2 XDVZ12remove50 683/800 106597clips PID28220；GPU3 XDVZ9remove70 691/800 109943clips PID27863，全fotile/Rl。正常任务未重启，其他项目不动。

上一轮r19 XDVZ9remove50已完成SwanLab aidmw2nv FINISHED/quality_completed=1核验，r19quality-result.json保存：主作者梯形PR-AUC77.185239%（dense76.181075%），delta+1.004163个百分点，95%CI[+0.639717,+1.365171]个百分点，支持非劣且区间高于0，限本预定配置/当前headseed配对评测，未多比较校正/不外推。stepAP72.079782%（dense70.333010%）单列。新UCFVZ12remove50已启动单次原consumer在线CPU r20 local21948，原head/采样/真值门、独立preview不覆盖final，预建日志避免tail竞态，r20已完成并SwanLab tyf6n9fi FINISHED/quality_completed=1核验：UCF ROC-AUC80.751865%，dense80.631958%，delta+0.119908个百分点，95%CI[-0.647867,+0.880487]个百分点，非劣证据不足；原结果保存r20quality-result.json不重算。方法/head/grid保持冻结。

近40m2s UCFVZ12r70约9.87clip/s、XDVZ12r50约14.23、XDVZ9r70约14.52；新XDVZ12r70仅12分钟启动窗，不能据前50臆断稳定速率。ETA HKT今日GPU0 09:00–10:30；GPU1 06:55–07:30；GPU2 06:35–07:15；GPU3 06:35–07:10。整套38+最终报告仍暂14:00–23:00，12待派4在跑，剩余uniform/共享/通信不确定，观察速度不入正式效率。无源码/训练/扩展QA。
## 2026-10-14 05:04 定时汇报

主控82184与四在线logger正常，无新增错误。XDVZ9remove50完整预测并云端cd5mspq9 FINISHED，GPU1自动补UCFVZ12remove70；21/38=55.26%完成（11UCF/10XD），累计2431444/4086273clips=59.50%。node3公网一次只读05:00:25：GPU0 UCFVZ12remove50 229/290 57259clips PID4423；GPU1 UCFVZ12remove70 24/290 6688clips PID18024；GPU2 XDVZ12remove50 534/800 72403clips PID28220；GPU3 XDVZ9remove70 550/800 75060clips PID27863，全fotile/Rl；正常任务不重启、其他项目不动。

新完成XDVZ9remove50单次原consumer在线CPU completed-metrics-r19 local103540，沿用实际head/采样/真值SHA门和独立preview，不覆盖final，当前待结果，后续读取不重复计算。新CPU launcher先touch本任务日志再nohup，避免已确认r17的tail先于文件创建竞态；不改评测算法/快照。r01–r18既有指标复用不变，r17使用recovered云端回执。冻结方法/head/grid不调整。

近39m33s UCFVZ12r50约9.68clip/s、XDVZ12r50約14.07、XDVZ9r70約11.04；新UCFVZ12r70仅12分钟启动窗约9含初始化，ETA参考同层UCF实测保持宽。HKT今日GPU0 05:30–05:55；GPU1 06:50–07:40；GPU2 06:45–07:30；GPU3 07:00–08:00。整套38+最终报告仍暂今日14:00–23:00，13待派4在跑，后续uniform/后层共享/通信不确定，观察速度不入正式效率。无源码/新训练/扩展QA。
## 2026-10-14 04:24 定时汇报

主控82184及四在线logger正常，无新增错误、无新prediction_completed，故不启动新CPU评测/不重算已有结果。20/38=52.63%完整提取+预测+云端确认（11UCF/9XD），累计2332826/4086273clips=57.09%。node3公网单次只读04:20:52：GPU0 UCFVZ12remove50 164/290 34294clips PID4423；GPU1 XDVZ9remove50 778/800 136328clips PID11529；GPU2 XDVZ12remove50 331/800 39018clips PID28220；GPU3 XDVZ9remove70 383/800 48855clips PID27863，全fotile/Rl。没有本轮新补位，自动接力保持，正常任务及其他项目不动。

上一轮r18 XDVZ6remove70已完成SwanLab cbjmr255 FINISHED/quality_completed=1核验，r18quality-result.json保存：主作者梯形PR-AUC76.120104%（dense76.181075%），delta−0.060971个百分点，95%CI[-0.558620,+0.443500]个百分点，下界低于−0.5，非劣证据不足。stepAP70.750245%（dense70.333010%）单列。复用原CPU结果/原head采样真值门，独立preview不覆盖final。此前r01–r17指标不变（r17采用recovered-cloud-verification核验）。冻结方法/head/grid保持。

近40m25s各自观察速率UCFVZ12r50约9.68clip/s、XDVZ9r50约10.63、XDVZ12r50约12.77、XDVZ9r70约10.58。ETA HKT今日GPU0 05:25–06:00；GPU1 04:45–05:00；GPU2 06:45–07:45；GPU3 07:00–08:00。整套38+最终报告仍暂今日14:00–23:00，14待派4在跑，后层VZ/uniform、共享/通信不确定，调度观察不是正式效率。无源码/训练/扩展QA。
## 2026-10-14 03:43 定时汇报

主控82184及四在线logger正常，R2共享名单恢复后无新错误。UCFVZ9remove70（oxv1yurs）与XDVZ6remove70（z3xt7jjv）完整预测云端FINISHED；GPU0/2自动补UCF/XD VZ12remove50，正常任务不重启、其他项目不动。20/38=52.63%完成（11UCF/9XD），累计2226970/4086273clips=54.50%。node3公网单次只读03:40:27：GPU0 UCFVZ12remove50 34/290 10829clips PID4423；GPU1 XDVZ9remove50 693/800 110558clips PID11529；GPU2 XDVZ12remove50 72/800 8053clips PID28220；GPU3 XDVZ9remove70 237/800 23199clips PID27863，全fotile/Rl。

上一轮r16 XDVZ6remove50已完成并SwanLab z3r6ka0o FINISHED/quality_completed=1核验：主作者梯形PR-AUC76.579646%（dense76.181075%），delta+0.398571个百分点，95%CI[-0.021895,+0.861495]个百分点，支持非劣但CI跨0不确认提升。stepAP70.727539%（dense70.333010%）单列，r16quality-result.json已保存。新完成两项各单次在线CPU原consumer：r17 UCFVZ9remove70 local94812；r18 XDVZ6remove70 local111096，原head/采样/真值门，独立preview不覆盖final，r17原CPU已完成，但tail抢在后台重定向建日志前启动而漏收EVENT；只读取原PID22951已完成日志、resume原SwanLab9if3ulhp，未重算，云端现FINISHED/quality_completed=1核验，保留原CRASHED记录和recovered-cloud-verification.json。UCFVZ9remove70 ROC-AUC82.166465%，delta+1.534508个百分点，95%CI[-0.301308,+3.552810]个百分点，支持非劣但CI跨0不确认提升，quality-result.json保存。r18 XDVZ6remove70仍CPU bootstrap（remote22955），下一轮读取本次不重复计算。方法/head/grid冻结不变。

ETA HKT今日GPU0 05:20–06:00；GPU1 04:40–05:15；GPU2 07:00–08:30；GPU3 07:00–08:00。近轮XDVZ9r50約10.3clip/s，XDVZ9r70新共享约10clip/s含启动；新第12层UCF约8.8clip/s含20分钟启动窗口/XD短窗速率未稳，完整dense前缀工作多，不能套第6层速率。整套38+最终报告改为暂今日14:00–23:00（旧12–22不再最新），14待派4在跑，余后层VZ和uniform完整前缀、共享/通信不确定。观察吞吐不入正式效率表，无源码/训练/扩展QA。
## 2026-10-14 03:03 定时汇报：共享进程换代门恢复

新增XDVZ6remove50完整预测并云端pkbcftuc FINISHED，18/38=47.37%完成（10UCF/8XD）。GPU3随后XDVZ9remove70启动前被Unknown process门拦住，原attempt云端uj7zvk1o CRASHED保留；远端该新任务无PID/无计算，不是GPU重跑。只读核验CVPR-JY2已换R2进程：GPU1/2/3 PID3414/3415/3416均fotile，完整配置R2_E05/E06/E07，显存13.6/13.6/11.6GiB，GPU3空闲28751MiB；符合既有9/21共享空余授权。更新运行allowlist三项（原20:20名单备份保留），仍严格PID+owner+command、12GiBfree及0.25allocator门。仅停止held主控115884，82184新controller-r03-recovery6直接接管旧三正常logger；resume输入保留原tracking目录，仅移除已证明未启动的失败行重派新attempt。GPU3新XDVZ9remove70已通过preflight，正常旧GPU/logger未重启、其他项目不动。没有改模型/方法/精度/seed/代码快照。

当前累计2131785/4086273clips=52.17%（GPU3恢复后2/800、136clips计入，不重复已完成）。node3公网只读03:01:06及恢复后localpoll：GPU0 UCFVZ9remove70 199/290 45639clips PID24838；GPU1 XDVZ9remove50 606/800 86259clips PID11529；GPU2 XDVZ6remove70 789/800 140487clips PID6314；GPU3新XDVZ9remove70已开始产出2/800、136clips，local108580。四logger均存活。单次新的CPU在线completed-metrics-r16 XDVZ6remove50 local122452/remote28444，原head/采样/真值门通过，独立preview不覆盖final；结果待回收，后续读取不重算。此前r01–r15指标保持。

ETA HKT今日GPU0 03:35–04:00；GPU1 04:35–05:15；GPU2 03:15–03:35；GPU3新任务06:30–07:45（新共享R2竞争窗口尚不足，宽估）。整套38+报告暂12:00–22:00，16待派4在跑，后层VZ/uniform及共享/通信不确定；近轮XDVZ9r50约10.5clip/s、XDVZ6r70约12.3，UCF9r70约16含启动，调度速率非正式计时。不根据分数改冻结格子，无源码/训练/扩展QA。
## 2026-10-14 02:23 定时汇报

主控115884及四在线logger正常，无新增错误。UCFVZ9remove50完整预测并云端xw67906e FINISHED，GPU0自动补UCFVZ9remove70；17/38=44.74%完成（10UCF/7XD）。GPU3 XDVZ6remove50已提取全部145703clips，仍在预测，不提前计completed；累计2036329/4086273clips=49.83%。node3公网一次只读02:20:25：GPU0 UCFVZ9remove70 24/290 6688clips PID24838；GPU1 XDVZ9remove50 442/800 59819clips PID11529；GPU2 XDVZ6remove70 693/800 110558clips PID6314；GPU3 XDVZ6remove50已extracted PID8583，全fotile/Rl。正常任务未重启，其他项目不动。

新完成UCFVZ9remove50已启动单次原consumer在线CPU completed-metrics-r15 local120476，独立preview且保持head/采样/真值SHA，r15已完成并SwanLab jqc8n6xy FINISHED/quality_completed=1核验：UCF ROC-AUC81.555593%（dense80.631958%），delta+0.923635个百分点，95%CI[−0.204326,+2.135831]个百分点，支持非劣但CI跨0不能声称提升已确认。原结果保存r15quality-result.json。其余r01–r14已核验指标复用不重算；GPU3尚在预测，不启动XDVZ6r50质量评测。冻结方法/head/grid不改，不据正式分数挑参数。

近40min XDVZ9r50约12.19clip/s、XDVZ6r70约13.73；新UCFVZ9r70启动8分钟约14含初始化，不作稳定速率。ETA HKT今日GPU0 03:30–04:00；GPU1 04:20–05:00；GPU2 03:10–03:45；GPU3预测收尾02:30–02:50。整套38+报告暂今日12:00–22:00，17待派4在跑，后层VZ/uniform及共享/通信不确定，观察吞吐不入速度表。无源码/新训练/扩展QA，仅进度和授权CPU评测。
## 2026-10-14 01:43 定时汇报

主控115884及四logger正常，无新增错误。UCFVisionZip6remove70已完整预测并云端ruq0ic7a FINISHED，GPU1自动补XDVisionZip9remove50，正常GPU未重启/其他项目不动。16/38=42.11%完成（9UCF/7XD），累计1921483/4086273clips=47.02%。node3公网单次只读01:40:25：GPU0 UCFVZ9remove50 199/290 45639clips PID16697；GPU1 XDVZ9remove50 282/800 30555clips PID11529，短暂Dl但progress持续；GPU2 XDVZ6remove70 562/800 77618clips PID6314；GPU3 XDVZ6remove50 737/800 123474clips PID8583，其余Rl，全fotile。

上一轮r13 XDattention12remove70已完成并SwanLab skfpv2al FINISHED/quality_completed=1：主作者梯形PR-AUC75.904954%（dense76.181075%），delta−0.276121个百分点，95%CI[−1.269023,+0.672717]个百分点，非劣证据不足。stepAP72.091499%（dense70.333010%）单列，不能用AP上涨替代主指标判定。原结果保存r13quality-result.json。新UCFVZ6remove70已单次在线原consumerCPU completed-metrics-r14 local110352，独立preview，原head/采样/真值门，r14已完成并SwanLab177wxrpe FINISHED/quality_completed=1核验：UCFVZ6remove70 ROC-AUC80.185463%，dense80.631958%，delta−0.446494个百分点，95%CI[−2.746208,+1.699475]个百分点，非劣证据不足。已保存r14quality-result.json；不重复已有评测、不覆盖final。Attention六配置×双数据集主质量已全部回收，方法/head/grid不改。

近40min各自速率：UCFVZ9r50约15.4clip/s、XDVZ6r70约14.1、XDVZ6r50约11.7；新XDVZ9r50启动38分钟约13.4含初始化。ETA HKT今日GPU0 02:10–02:35；GPU1 04:00–05:00；GPU2 03:00–03:45；GPU3 02:15–02:45。整套38+报告仍暂今日12:00–22:00，18待派4在跑，后层VZ/uniform、共享/通信未测仍不确定；观察吞吐不入正式速度表。无源码/训练/扩展QA变化。
## 2026-10-14 01:03 定时汇报

主控115884与四logger正常，无新错误。XD attention12remove70完成并云端xhjzoy44 FINISHED，GPU0自动补UCF VisionZip9remove50；15/38=39.47%完整提取+预测+云端确认。GPU1 UCFVZ6remove70提取全部69364clips，尚在预测（不提前计completed）；累计1792064/4086273clips=43.86%，每任务仅计一次。node3公网单次只读01:00:28：GPU0 UCFVZ9remove50 29/290 8687clips PID16697；GPU1 UCFVZ6remove70已extracted PID30950 Rl；GPU2 XDVZ6remove70 354/800 43814clips PID6314 momentary Dl（进度持续，IO等待不判故障）；GPU3 XDVZ6remove50 655/800 95366clips PID8583 Rl。其他项目不动、正常任务不重启。

上一轮r11 XDattention12remove50已完成SwanLab flu5kz37 FINISHED/quality_completed=1：主作者梯形PR-AUC76.224890%（dense76.181075%），delta+0.043815个百分点，95%CI[-0.548459,+0.633443]个百分点。CI下界低于-0.5，非劣证据不足，不因点值微升放行。stepAP71.459211%（dense70.333010%）单列。原结果保存r11quality-result.json。新完成XD12remove70单次在线CPU completed-metrics-r13，local103536/remote8609，原head/sample/truth门通过，独立preview不覆盖final；当前待结果，后续读取本次不重复计算。UCFVZ6remove70预测尚未完成不启动CPU评测，其余已核验指标不变。

近轮XDVZ6remove50约10.74clip/s，XDVZ6remove70约14.11clip/s；新UCFVZ9启动窗约14clip/s含初始化仅粗估。ETA HKT GPU0今日02:10–02:50；GPU1预测收尾01:10–01:25；GPU2 03:00–04:00；GPU3 02:20–03:00。整套38+报告暂今日12:00–22:00，19未派4在跑，后层VZ/uniform及共享/通信不确定。冻结格子不改；无源码/新训练/扩展QA，仅进度记录和授权CPU评测。
## 2026-10-14 00:23 定时汇报

主控115884及四在线logger正常，无新错误。新增XD attention12remove50（ko0i24ru）与UCF VisionZip6remove50（9m7ydqll）完整预测并云端FINISHED核验，自动GPU1/2补UCF/XD VisionZip6remove70，不重启正常旧任务、不动其他项目。14/38=36.84%完成，累计1673495/4086273clips=40.95%。node3公网单次只读00:20:53 GPU0 XD attention12remove70 770/800 132905clips PID13324；GPU1 UCF VisionZip6remove70 151/290 31290clips PID30950；GPU2 XD VisionZip6remove70 85/800 10302clips PID6314；GPU3 XD VisionZip6remove50 513/800 69868clips PID8583，全fotile/Rl。

新完成两项各自单次原consumer CPU在线质量评测：completed-metrics-r11 XD12remove50 local121200/remote8970；r12 UCFVisionZip6remove50 local63588/remote8979。PAIRED_QUALITY_IDENTITY_GATE_PASSED，原head/采样/真值SHA与campaign保持，独立quality-preview不覆盖最终quality.json。r12已完成并SwanLab fey1q91y FINISHED/quality_completed=1核验：UCF VisionZip6remove50 ROC-AUC80.490497%，dense80.631958%，delta−0.141461个百分点，95%CI[−1.794929,+1.474373]个百分点，非劣证据不足。原结果保存quality-result.json。r11 XD12remove50仍在CPU bootstrap，下一轮读取既有任务，不重复计算。既有r01–r10指标不变；冻结格子不调整。

近轮XDattention12r70约13.2clip/s、XDVZ6r50约11.0clip/s；新UCFVZ6r70约16.7clip/s含启动；新XDVZ6r70尚只有13分钟启动窗，用范围不作稳定速率。ETA HKT GPU0今日00:40–01:00；GPU1 01:00–01:30；GPU2 02:50–04:00；GPU3 02:10–03:00。整套38+报告仍暂今日12:00–22:00，20待派4在跑，后层VZ/uniform、共享/通信仍不确定。仅进度观察不是正式计时，无源码改动/新训练/额外QA。
## 2026-10-13 23:43 定时汇报

主控115884及四在线logger正常，无新错误/新prediction_completed，故不重复CPU评测；12/38=31.58%完整提取+预测+云端确认。累计1558897/4086273clips=38.15%。node3公网单次只读23:40:26：GPU0 XD attention12remove70 669/800 100805clips PID13324；GPU1 UCF VisionZip6remove50 278/290 66396clips PID20273；GPU2 XD attention12remove50 774/800 134502clips PID8544；GPU3 XD VisionZip6remove50 350/800 43131clips PID8583，全部fotile/Rl。无本轮新补位，仍自动接力；其他项目和正常任务不动。

与23:01:10比较近39m16s：attention12 XD GPU0约13.70clip/s、GPU2约10.91；VisionZip6 UCF约16.28、XD约11.04。按各自剩余clips更新ETA HKT：GPU0 10/14 00:30–01:05；GPU1 10/13 23:50–10/14 00:05；GPU2 10/14 00:00–00:20；GPU3 10/14 02:15–03:00。仅共享环境进度速率，不是论文速度。整套38+报告仍暂10/14 12:00–22:00，22未派4在跑，未測后层VisionZip/uniform及共享/通信不确定性不变。

已核验指标r01–r10不变：UCF dense80.631958%，attention6remove50/70=80.553156/80.184568，9remove50/70=81.403313/82.130803，12remove50/70=80.678065/80.713885。XD主作者梯形PR-AUC dense76.181075%，6remove50/70=76.640234/75.980119，9remove50/70=77.175098/77.501657；stepAP单列70.333010、71.073361、71.009467、71.981071、73.158876。UCF9两项及XD6r50/XD9两项支持非劣，XD9两项CI高于0仅本配置/seed未多比较校正；其余CI非劣证据不足。冻结格子不调整，无源码/训练/额外QA。
## 2026-10-13 23:03 定时汇报

主控115884及四个在线logger存活，短轮询无新增错误。UCF attention b12remove70完整预测并云端FINISHED确认（t9dex86h）；GPU3自动补位首个XD VisionZip b06remove50，正常GPU任务均未重启。12/38=31.58%完成，累计1436559/4086273clips=35.16%。node3公网一次只读检查23:01:10：GPU0 XD attention12remove70 504/800 68532clips PID13324；GPU1 UCF VisionZip6remove50 132/290 28034clips PID20273；GPU2 XD attention12remove50 689/800 108809clips PID8544；GPU3 XD VisionZip6remove50 186/800 17121clips PID8583，均fotile/Rl，其他项目不动。

新完成UCF12remove70由原consumer单次CPU在线completed-metrics-r10，local101080/remote18282，PAIRED_QUALITY_IDENTITY_GATE_PASSED，保持head/采样/真值SHA，独立preview不覆盖最终quality.json；结果已完成并云端1qyasal4 FINISHED/quality_completed=1核验：UCF ROC-AUC80.713885%，dense80.631958%，delta+0.081927个百分点，95%CI[-1.620429,+1.706851]个百分点；非劣证据不足，不能因点估计微升判通过。原结果保存quality-result.json，不重复计算。此前已核验指标不变。仅做本轮简查/新完成质量计算，没有新方法、训练或扩展验收。

按近轮clip增量和VisionZip首波实际速率，ETA HKT GPU0 10/14 00:40–01:20；GPU1 10/14 00:00–00:40；GPU2 10/14 00:00–00:45；GPU3 10/14 02:10–03:20。VisionZip首波粗观测UCF约12.4clip/s、XD约10.3clip/s含启动，不是正式效率；attention12近轮约GPU0 11.6、GPU2 8.6clip/s，按dataset/method分开不互代。整套38+最终报告暂仍10/14 12:00–22:00，未测后层VisionZip/uniform、共享负载和通信有不确定性，余22待派、4在跑。冻结格子不因正式分数调整。
## 2026-10-13 22:23 定时汇报：退出记录竞态恢复

XD b09remove50远端完整预测已完成，本机poll在进程恰好退出时/proc存在但ps/cmdline消失，set-e空错误导致logger标失败、主控暂停新派；并非GPU计算失败。修复该外部进程退出竞态：只对仍有owner+非空cmdline的活进程执行严格身份检查，已结束时仍必须看到真实completed事件才能成功；不放宽未知活PID门。修复resume让仍在运行的在线logger直接被新调度器观察，不重启它们。主控115884，最新controller-r03-recovery5/queue.json；只补接失败XD50原SwanLab755pv25s并云端成功，旧三正常GPU/在线logger未重启。8项局部控制/轮询测试及compileall通过；无模型、方法、采样、精度、预算或GPU提取变更。

11/38=28.95%已完整预测+云端确认，累计1341358/4086273clips=32.83%。GPU1自动进入首项正式VisionZip UCF block6remove50（1/290 88clips），GPU0 XD12remove70 334/800（40580clips），GPU2 XD12remove50 618/800（88090clips），GPU3 UCF12remove70 282/290（67901clips）。其余已验证共享进程全部保留，GPU仍每卡一个本项目任务。新XD9remove50指标单次CPU在线completed-metrics-r09，本机105324，原head/采样/真值门及原比較器，独立preview；已完成并核验SwanLab qprk6p3q FINISHED/quality_completed=1。XD b09remove50主作者梯形PR-AUC=77.175098%，dense76.181075%，delta=+0.994022个百分点，95%CI[+0.491079,+1.484264]个百分点，支持本预定配置/当前headseed的提升及非劣；不作多配置校正后或跨seed结论。stepAP=71.981071%（dense70.333010%）单列。原结果已保存completed-metrics-r09/quality-result.json，不重复计算。

ETA HKT GPU0 10/14 00:35–01:20，GPU1新VisionZip 10/14 00:10–01:10（正式新方法尚缺完整任务实测，范围较宽），GPU2 10/14 00:10–01:00，GPU3 22:40–23:00；整套38+报告仍初步10/14 12:00–22:00，共享竞争和未测方法不确定。保持冻结格子，不根据分数调方法。

**21:43新增指标已完成：** XD b09remove70主作者梯形PR-AUC=77.501657%，dense76.181075%，delta=1.320582个百分点，95%CI[0.543034,2.051094]个百分点，区间高于0，本配置本headseed/预定配对评测支持提升（不外推其他条件，也不按该分数修改后续格子）。stepAP=73.158876%（dense70.333010%）单列。SwanLab8qoah3ng已FINISHED/quality_completed=1核验，原评测10kbootstrap，同一head/采样/真值SHA，独立preview不覆盖最终quality.json。

## 2026-10-13 21:43 定时汇报

主控26992及四个在线logger正常，短连接无新增错误。XD b09remove70已完成完整预测+云端核验并自动补GPU0 XDb12remove70；总10/38=26.32%，累计1243461/4086273clips=30.43%。远端21:40:19 GPU0 XD b12remove70 32/800（3748clips）；GPU1 XD b09remove50 789/800（140487clips）；GPU2 XD b12remove50 462/800（62790clips）；GPU3 UCF b12remove70 188/290（37440clips）。既有共享PID保留，未重启正常GPU，仍每卡一个本项目任务。

新完成XD9r70已单次CPU在线评测completed-metrics-r08，本机112588/远端28245，实际head/采样/真值SHA及原比较器一致，独立preview，不覆盖最终quality.json。当前尚在计算全配对结果，先记待完成，下一次读取已有文件不重算。此前UCF六项和XD三项指标不变，不据正式分数改方法。

ETA HKT GPU0 10/14 00:50–01:45，GPU1 22:05–22:30，GPU2 10/14 00:10–01:00，GPU3 22:40–23:15；第12层共享卡速度慢于初期，按剩余clip量修订。整套及最终报告仍初步10/14 12:00–22:00，其他项目负载及未跑配置可能继续影响；不把此范围当硬保证。此次无源码变化、不扩展验收。

## 2026-10-13 21:03 定时汇报

主控26992及四个在线logger均正常，无新错误/通信中断，无新prediction_completed；不重复计算指标。任务9/38=23.68%，累计1150806/4086273clips=28.16%。远端21:00:19：GPU0 XDb09remove70 745/800（125652clips）；GPU1 XDb09remove50 705/800（114945clips）；GPU2 XDb12remove50 331/800（39018clips）；GPU3 UCFb12remove70 84/290（17898clips）。四卡仍各一个本项目任务，同项目没有堆进程；已核实的其他项目进程均保留。CVPR29603显存涨到约16.7GiB，其他共享卡29600/29602约10.9GiB，仍有安全余量，未改变其进程。后续补位按已验证共享名单和12GiB空闲/本任务0.25 allocator门。

既有指标不变，UCF frameROC-AUC dense80.631958、b6r50 80.553156、b6r70 80.184568、b9r50 81.403313、b9r70 82.130803、b12r50 80.678065%；XD作者梯形PR-AUC dense76.181075、b6r50 76.640234、b6r70 75.980119%，stepAP70.333010/71.073361/71.009466%，不混用口径、不据正式分数调配置。

按本次实际clip速率与共享竞争更新ETA HKT：GPU0 21:30–22:00，GPU1 22:00–22:35，GPU2 10/14 00:15–01:15，GPU3 22:15–23:00。整套及最终报告仍初步10/14 12:00–22:00；第12层几乎保留全部encoder前缀且共享计算，速度比早期第6/9层条件慢，不拿初期吞吐强套。此次不改源码、不扩展验收、没有新实验或GPU重投。

## 2026-10-13 20:23 定时汇报：共享占用与两项UCF完成

本轮检查期间UCFb09remove70和b12remove50均完成完整预测及云端核验，自动补位正常，总9/38=23.68%；累计最新已接回clip计数1074257/4086273=26.29%。当前GPU0 XDb09remove70为669/800（100805clips），GPU1 XDb09remove50 655/800（95366clips），GPU2新XDb12remove50 239/800（23499clips），GPU3新UCFb12remove70 6/290（1294clips）。主控26992健康，未重复启动GPU任务。

发现19:59起CVPR-JY2新增GPU1/2/3进程29600/29602/29603及GPU5进程29604。只读核对owner=fotile、完整CVPR-JY2 runtime.cli train/config/输出身份；未停止或修改它们。GPU1/2/3空闲约22–23GB，沿用用户AGENTS 9/21明确共享空余显存授权，后续tracker用gpu-share-allowlist.json极窄已验证PID/owner/命令标记名单，仍禁止未知PID、要求至少12GiB free、本任务allocator0.25，GPU0旧1827规则不变。提取/模型/方法/精度/数据/head冻结未改。新GPU3补位已通过名单正常启动；共享训练可能降低速度，正式独占效率仍不可据此宣称。相关3项记录门单测和compileall通过，只做必要执行修复、不扩展benchmark。

新UCFb09remove70 ROC-AUC=82.130803%，delta=1.498845个百分点，CI[-0.220130,3.322102]个百分点，supported非劣但不显著优于dense；SwanLab mmhqtpr1 FINISHED/quality_completed核验。新UCFb12remove50 ROC-AUC=80.678065%，delta=0.046107个百分点，CI[-1.166994,1.204065]个百分点，非劣inconclusive；SwanLab450rbdbh FINISHED/quality_completed核验。均独立preview、原比较器和实际head/采样/真值SHA门，不占GPU，不覆盖最终quality.json，不据分数改格子。

ETA HKT GPU0 21:30–22:15、GPU1 21:40–22:30、GPU2 23:20–10/14 00:30、GPU3 22:00–22:45。因新共享计算占用及最后层条件较慢，整套38+报告范围暂放宽到10/14 12:00–22:00（原10:00–19:00不再当最新承诺），下轮用实际剩余clip速率修订；未跑VZ/uniform的开销仍未知。

## 2026-10-13 19:43 定时汇报

主控26992和四个在线logger均存活，短连接轮询正常、无新报错，四原PID1965/27889/31858/653继续推进，不重启、不扩展验收。7/38任务完成=18.42%，远端19:41:07累计932178/4086273clips=22.81%；相比19:00快照新增145141clips（约40分48秒四卡合计59.3clips/s，只作现场提取统计，非正式纯encoder效率）。没有新prediction_completed，不重复指标CPU评测。GPU0 XD b09remove70为422/800（55506clips），GPU1 XD b09remove50 440/800（59278clips），GPU2 UCF b09remove70 275/290（65389clips），GPU3 UCF b12remove50 188/290（37440clips）。GPU2瞬时Dl仅表示读写等待；新进度持续增长，不按单次状态判挂或停止。

既有指标不变：UCF主frameROC-AUC dense80.631958、b6r50 80.553156、b6r70 80.184568、b9r50 81.403313（%）；XD主作者梯形PR-AUC dense76.181075、b6r50 76.640234、b6r70 75.980119（%），stepAP70.333010/71.073361/71.009466（%）。已预定的UCFb9r50和XDb6r50支持非劣但CI跨0，不称稳定提升，其余压缩非劣未定。没有用正式分数改方法/预算/检查点。

ETA HKT GPU0 21:30–22:10，1 21:20–22:00，2 19:50–20:10，3 20:20–20:55；整套38+报告仍初步10/14 10:00–19:00，未知后续VZ/uniform/depth和云端延迟继续保留。保持每卡一个任务自动接力。

**19:03新增XD50指标已完成：** 主作者梯形PR-AUC=76.640234%，dense76.181075%，delta=0.459159个百分点；95%CI[-0.060628,1.061184]个百分点，按-0.5pp界supported非劣，区间跨0不宣称显著优于dense。stepAP=71.073361%（dense70.333010%），口径单列。SwanLab sv0oa6fq已核验FINISHED及quality/completed=1，独立preview/本地completed-metrics-r05/quality-result.json，不改冻结方法，不占GPU。

## 2026-10-13 19:03 定时汇报

短连接轮询后本机主控26992及四个在线记录进程均正常，未再发现本轮通信失败。新增XD b06 remove50完整预测+云端完成，总7/38=18.42%；实际远端19:00:19累计787037/4086273clips=19.26%。四张卡已全部进入r05优化版：GPU0 XD b09remove70 224/800（21217clips）；GPU1 XD b09remove50 244/800（24611clips）；GPU2 UCF b09remove70 116/290（25350clips）；GPU3新补UCF b12remove50 6/290（1294clips）。真实PID1965/27889/31858/653均Rl或等待态正常，其他项目1827未动。原四项第一波现已全部完成；各卡按各自队列自动补位。

新完成XD50已启动单次CPU在线指标对照completed-metrics-r05（本机83828、远端13362），复用原compare_prediction_runs及head/采样/真值SHA门，独立preview不覆盖最终quality.json，尚在计算/等待最终CI与云端核验，不重复GPU。此前UCF dense80.631958%、b6r50 80.553156%、b6r70 80.184568%、b9r50 81.403313%；XD主作者梯形PR-AUC dense76.181075%、b6r70 75.980119%，stepAP70.333010%/71.009466%，保持两口径区分。没有依据正式分数调整任何冻结格子。

当前四项ETA HKT：GPU0 21:30–22:30、GPU1 21:20–22:20、GPU2 19:50–20:20、GPU3 20:20–21:10（新启动，尚不能按初始低吞吐线性外推）；整套及报告仍初步10/14 10:00–19:00，依据实际clip吞吐更新，剩余VZ/uniform/depth组合与云端延迟不确定。此次未扩展验收、未修改源码。

## 2026-10-13 18:40 定时汇报及短连接恢复

再次发现长SSH日志连接中断，本机97040 held_after_failure，但四GPU计算均正常。改为每20秒短SSH只读轮询原日志，保留源行游标、原PID和SwanLab身份，最多3次传输重试，认证/指纹失败立即停止；新formal launcher只启动一次nohup后退出SSH，再短轮询原日志，不重复启动GPU。r03恢复尝试补路径时缺tracking_directory，取消其本机记录进程并保留目录，未动GPU；r04补齐原路径/实际snapshot，主控26992、最新controller-r03-recovery4/queue.json。11项局部测试及compileall通过；不加工程benchmark、不改提取/评分算法或模型。

本次多项已收尾：完整预测+云端确认6/38=15.79%，累计本地最新计数721764/4086273clips=17.66%。已完成UCF dense/b06remove50/b06remove70/b09remove50及XD dense/b06remove70。自动接力后GPU0 XD b09remove70 16/800（2032clips）；GPU1 XD b09remove50 44/800（5269clips）；GPU2 UCF b09remove70 13/290（3159clips）；GPU3原XD b06remove50 793/800（142442clips）。预测/云端完成后马上补位，不等待全体。上述计数为18:40附近本机已接回原日志，非模型训练采样预算。

新增指标均复用原CPU评测、实际head/采样/真值SHA不变，独立preview、不覆盖最终quality.json、无GPU重跑。UCF block9 remove50 frameROC-AUC=81.403313%，dense80.631958%，delta=0.771355个百分点，95%CI[-0.275440,1.873450]个百分点，按预定-0.5pp非劣界supported；区间跨0，不宣称显著优于dense。SwanLab5hqbxfzx FINISHED/quality_completed核验通过。XD dense主作者梯形PR-AUC=76.181075%，block6remove70=75.980119%，delta=-0.200957个百分点，CI[-0.838616,0.455831]个百分点，非劣inconclusive；stepAP分别70.333010%/71.009466%，不与梯形PR-AUC混称。SwanLab lb2yzg6a FINISHED/quality_completed核验通过。原UCF50/70成绩不变，不按这次正向成绩选择/修改后续冻结格子。

按现有clip速率，新GPU0/1 XD任务预计21:30–22:40/21:20–22:30，GPU2 UCF预计19:45–20:20；原GPU3XD50收尾预计18:55–19:15（HKT）。整套38+最终报告仍初步预计10/14 10:00–19:00；其余method/depth未全实测，通信恢复耗时及指标计算可能推迟，下次继续修订。key-attention-40保持ACTIVE，每40分钟主动报告。

**新增完成指标：** 已按用户定期指标请求计算UCF AttentionTopK block6 remove70，frame ROC-AUC=80.184568%，dense仍80.631958%，delta=-0.447390个百分点；95% CI=[-2.491389,1.546806]个百分点，非劣inconclusive。完整290视频、10000配对bootstrap同原口径，独立preview文件，不改最终quality.json、不重跑GPU。SwanLab zetywh1x FINISHED及quality/completed=1已核验；本地completed-metrics-r02/quality-result.json。保持冻结格子继续，不因这次分数改配置。

## 2026-10-13 用户要求每40分钟汇报

已创建本聊天heartbeat `key-attention-40`（ACTIVE），每40分钟主动汇报，包含任务完成比例、累计clip提取比例、已完成指标、每卡及整套预计结束时间。用户明确要定期汇报，因此不沿用“无变化则静默”；仍保持简查，不扩展验收。新完成且匹配dense ready时复用原评测CPU在线计算，SwanLab完成核验，不据正式分数改方法。

初始快照远端17:41:28：3/38任务完成=7.89%，累计598200/4086273clips=14.64%；GPU0 UCFb09remove50为109/290（22804clips），GPU1XDdense756/800（129860clips），GPU2XDb06remove70为704/800（114405clips），GPU3XDb06remove50为735/800（123039clips）。四PID12608/31729/16460/31727 Rl，当前主控97040继续自动补位。按实际clip速度，当前四项预计分别18:35–18:55、18:25–18:45、18:20–18:45、18:45–19:10（HKT）；完整矩阵及报告初步预计10/14 10:00–19:00，后续每次按实际各方法/层/数据集速率修订。尚未实测全部剩余配置，网络和云端记录可能推迟，不把两fit视频倍率套全体。

## 2026-10-13：17点连接恢复与接力

用户简查进度时发现本机记录连接全部退出，旧调度器119980 held_after_failure；实际远端三XD原PID31729/31727/16460持续运行，UCF b06 remove70已自然完成完整提取与预测，GPU计算并未失败。SSH exit255及代理超时历史保留，不按本机断连重投GPU。

修复记录恢复的实际路径：跳过已完成任务；按queue行内tracking_directory读取原事件；保存并复用原SwanLab身份，用最后一条真实事件定位重接游标；接回新版时校验其实际r05 snapshot及task-id，不误按旧snapshot拒绝。新增后续SSH keepalive，未改环境或认证。恢复本机controller97040，最新queue `work/key-attn-formal-20261013/controller-r03-recovery2/queue.json`，原四个SwanLab run保持同ID。UCF b06 remove70已补齐记录并云端完成，三XD已接回持续上传，GPU0自动启动下一冻结项UCF AttentionTopK block9 remove50、真实PID12608。原XD三个GPU进程和其他项目均未停止。

远端17:19:20最新发布：UCF block9 remove50 6/290（1294clips）；XD dense727/800（121326clips）；XD block6 remove50 705/800（114945clips）；XD block6 remove70 653/800（94901clips）。38项为3完成、4在跑、31未派。已完成三项为UCF dense/b06remove50/b06remove70，均完整290视频并完成预测。此次只恢复通信及接力，不扩展验收，不读取新正式质量分数。相关7项控制/记录门单测通过，compileall src tests及改动脚本通过；GPU提取快照、采样、精度、方法、head不变。

## 2026-10-13：按用户要求读取已完成UCF两项正式指标

用户明确要求“之前提取特征结束的直接测试一下指标”，允许本次消费已完成预测；原冻结配置/方法/head仍不依据test调参。单次CPU运行复用原`compare_prediction_runs`，预先核对两实际head SHA及evaluation_sampling_fingerprint相同、两prediction_completed与真值原SHA；不重跑GPU、不改主队列。输出独立`quality-preview-20261013-r01.json`，不覆盖最终quality.json。共享评测使用压缩视频pair矩阵，一次已完成原10000配对bootstrap，不需另开实验。

UCF完整290视频、1111808帧（正84338、负1027470）：dense frame ROC-AUC=0.806319575743（80.631958%），AttentionTopK block6 remove50=0.805531563091（80.553156%），delta=method-dense=-0.000788012651（-0.078801个百分点）。95% CI=[-0.014189957200,0.012303887799]，noninferiority=inconclusive；点差满足-0.005预算，但CI下界未达门槛，不能宣称已证明非劣。bootstrap_seed=20260918，10000/10000有效，0无效。附属frame stepAP dense=0.192917278299、method=0.199852912015；video AUC不是UCF主指标。

SwanLab run3i3rdyam已核验FINISHED及quality/completed=1，指标与完整事件/源SHA在线保存；本机`work/key-attn-formal-20261013/completed-metrics-r01/quality-result.json`。这是本聊天首次读取这两项正式帧级成绩，后续只报告预先冻结格子，不由这些分数改层/预算/checkpoint。四卡生产未中断。

## 2026-10-13 用户最新执行要求：简查、加速、持续接力

不再扩展或重复工程验收；已通过r05供数优化用于后续任务，原正常任务继续。每次简查四个PID、进度和末尾报错，真实故障才处理。远端15:43:12：本轮UCF dense及UCF b06 remove50已全部完成预测；原XD dense586/800（82316/145703 clips）、XD b06 remove50为561/800（77266/145703 clips）仍运行。下一项新版GPU0 UCF b06 remove70为19/290（5682 clips），GPU2 XD同条件40/800（4577 clips），新两PID16462/16460均Rl，约运行5:45。两旧XD31729/31727未重启，四卡各一项生产；38项为2完成、4在跑、32待接力。新两项从实际启动起含加载粗计约16.5/13.3 clips/s，仅现场进度口径，不是正式效率数字。

应用在更新原20分钟heartbeat时返回该automation已不存在，未擅自重建；本机后台队列119980仍独立接力，不影响已在跑的任务。最新queue在controller-r03-optimized2，GPU0/2行内waiting_execution_acceptance是启动时留下的调度状态，实际已通过门禁并运行，以远端PID和新EVENT为准；GPU1/3原在线logger由新调度器直接观察，不重复初始化。定时检查项是否被用户删除未知，不猜测原因。

## 2026-10-13：按GPU 0/1/2/3启动VideoMAEv2正式关系压缩矩阵

用户明确要求每GPU一个任务依次完成。新[正式冻结](key_attention/FORMAL_FREEZE_20261013.md)锁定VideoMAEv2×UCF/XD：每集dense加18个attention_topk/visionzip_internal/uniform×block6/9/12×删除50/70%条件，共38个GPU提取+评分任务；不重训head，不按test分数改配置。使用完整UCF290与XD800、原sealed manifest/审计/作者时间轴真值；UR-DMU full-sequence exact query-chunk256，后续自动CPU质量导出及10000配对视频bootstrap。

已验证两份现有head的原生backbone身份与当前encoder精确相同（code_digest 19720d4e…），full-train源清单1610/3950仍由final.pt与训练stage原SHA绑定。修复资产迁移后原声明路径与当前物理路径的匹配，并解除对已清理UCF训练缓存/view目录的不必要依赖；未修改检查点、旧SHA或历史协议。新轮次目录保留全部失败记录，只核实后SIGTERM本轮早期XD PID28946/28948；未动其他项目进程。

当前成功执行轮为`/users/fotile/VAD/work/key-attention-formal-20261013-r03`，输出在`/data2/localdisk/fotile-wavad/experiments/icassp2027/key-attention-20261013-r03`。代码快照`f6fa8ca`，tag `key-attn-formal-v4-20261013`，campaign SHA `0709def30dc2d5d66be164079eac6564442ac337e89500d9ffda534cd953c80d`。本机后台队列PID110448；远端GPU0 UCF dense PID31731、GPU1 XD dense PID31729、GPU2 UCF AttnTopK block6 remove50 PID31725、GPU3 XD对应条件PID31727，均有真实视频完成事件。GPU0按既有共享授权保留其他用户服务PID1827（约12GB），本轮各进程allocator限制0.25约10GiB；1/2/3每卡一任务，不占其他卡。共享运行不入正式速度表。

[SwanLab项目](https://swanlab.cn/@ryfrcd693/VAD-Key-Attention-20261013)：四个当前run分别为 qzw8qfyv、vlqduu7k、m79i0tux、3aj98wey，均回读RUNNING及真实进度标量。服务偶发HTTP504，SDK自动重试/恢复，完整本地events和远端日志保留，任务结束回读确认后才派下一项。GPU任务全结束后自动质量导出；配对消费端额外核对实际head SHA和evaluation_sampling_fingerprint相同后才打开真值。当前无新AUC/AP结果。

本地新流程/提取/评分/质量回归首组83 passed；迁移修复后的相关43 passed，最近训练authority相关21 passed、配对质量身份门12 passed；compileall src tests通过。队列依赖本机后台控制器和网络保持运行，远端当前任务由nohup保护；失败停止新派，不盲目重复启动。
**连接恢复补充：** 后续公网SSH流发生exit255中断，原本机控制器110448安全停止新派，四个远端nohup PID31725/31727/31729/31731经重新连接核实仍在运行并继续生成特征（并非计算失败）。新增只读reattach控制器PID81924接回原PID与原SwanLab run，没有重投GPU任务；云端旧CRASHED保留为通信中断历史，恢复时追加transport/recovered标量。SwanLab API仍间歇504/超时，部分上传恢复，完整事件保存在本地及远端；后续接力等待当前任务计算完成与云端核验。当前状态以`work/key-attn-formal-20261013/controller-r03-recovery1/queue.json`及远端task.json为准，不按断开的SSH会话判断GPU任务已终止。


**13:38 HKT进度及守护：** 四路原计算继续推进，恢复控制器PID81924仍在运行，本地接回事件已更新至UCF dense 170/290（35300 clips）、UCF AttnTopK b06 remove50 155/290（32555 clips）、XD dense 259/800（27216 clips）、XD对应AttnTopK 248/800（25652 clips）。这是特征提取计数，不是质量结果；38个任务尚未全部完成。SwanLab仍偶发上传失败并自动重试，XD dense日志已有Upload recovered；最终云端成功尚需结束后核验。按用户请求已创建本聊天每20分钟heartbeat（automationId=key-attention，ACTIVE），主控检查真实进程、接力冻结队列和补传，实质变化/故障/完成时通知，完成全部任务与云端核验后停止守护。

**13:58 HKT heartbeat：** 公网只读复核四个原PID仍Rl，恢复控制器81924及四个记录进程均存活；GPU0保留1827，无重复本项目任务。远端时钟13:54:42，最新发布计数UCF dense199/290（45639 clips）、UCF AttnTopK188/290（37440 clips）、XD dense312/800（35872 clips）、XD AttnTopK299/800（33823 clips）。尚无任务完成，不派重复任务；SwanLab仍自动重试，三路记录日志出现Upload recovered。方法/线程/batch/采样均未调整，等待原队列接力；本轮未读取正式质量分数。

## 2026-10-13 14:20 HKT：用户授权后续任务优化，原四项不变

新增有界CPU供数：一个连续解码器保留游标/32帧尾部，native处理器按真实帧索引去重并恢复原BCTHW，两批预取与GPU推理重叠；不改原采样、microbatch8、模型精度、reducer或head。默认旧路径不变，新版仅按独立execution配置启用，代码/provenance与原冻结分开绑定。50项相关单测及compileall通过。

后续任务的本机tracker新增pending-execution门禁；当前controller/PID及四个原GPU进程未重启，原任务继续跑。`work/key-attn-formal-20261013/pending-execution.json`目前preparing，新任务在门禁处等待，不重复计算。用户明确要求单GPU实验及四进程加速核验：工程验收先做四路CPU供数，GPU2原PID31725释放后才做单卡native/optimized完整提取写入对照；验收/云端成功后才放行后续，真实四卡聚合收益另核验，单卡不冒充四卡结论。

工程r01因历史fit manifest的num_frames为空在CPU阶段退出，未分配CUDA；保留失败目录及SwanLab CRASHED云端回执。r02改用实际OpenCV probe，remote `/users/fotile/VAD/work/key-attention-optimization-20261013-r02`，本机tracker PID50780，SwanLab run n99346d2。复用原role-locked两份fit视频及SHA；RGB/帧索引/处理器pixels逐元素相同。四路CPU测试旧42.18–42.89 clips/s，新workers1为70.95–72.47、workers2为64.06–78.01；按两次中位耗时选择workers1/prefetch2，CPU聚合约1.69x，不是GPU正式效率。r02现等待GPU2原任务自然结束，GPU吞吐/pooled一致性及云端验收尚未完成。

**14:40 HKT优化候选更新：** 为继续减少重复CPU工作，r04加入单worker限定的32帧已处理像素尾部缓存，视频切换清空。63项单测与compileall通过；新候选commit90f2a04、tag key-attn-cpu-pipeline-v2-candidate-20261013，source SHA `5d0ba4209df5578b0bed9836f77b110ea63ec791171043d26ef315c4ded26543`，原冻结及native adapter/sampler/video源码未改。r02等待GPU的本工程PID5092经命令与GPU归属核实后以user_scope_upgrade SIGTERM（未分配CUDA），CPU证据/CRASHED云端回执保留；四个正式原PID未动。

r04本机tracker76924、远端benchmark19076、SwanLab run29ryhefg，新增缓存RGB/源帧/处理器pixels逐元素一致，max_abs=0。四CPU并发中位聚合旧42.58、新worker1无缓存73.52、worker2无缓存79.57、新worker1带像素尾缓存85.68 clips/s；选择worker1/prefetch2/deduplicate+temporal_cache，约2.01x CPU侧，不是GPU端到端。GPU2仍由原任务31725运行；r04等待其自然释放再做四方法单GPUnative/optimized完整提取写入/pooled对照。pending-execution仍preparing，下一任务等待验收，不盲目放行。

本机只读验收放行watcher85160（日志在optimization-r03/promoter.log/err，读取最新pending字段已指向r04）将等待r04云端成功回执后下载两个小JSON，核对像素/pooled/吞吐/配置/云端/快照SHA，原子设ready；不启动GPU。当前真实四GPU新版聚合吞吐尚未验证。最近远端时钟14:35:30，原UCF dense284/290（68309clips），UCF AttnTopK267/290（64252clips），两XD原任务仍在跑，未读正式分数。

## 2026-10-13 15:44 HKT：优化验收通过并恢复四卡生产

UCF dense、UCF AttentionTopK b06 remove50均完成完整290/69364clips提取和预测，原SwanLab云端完成通过。原XD dense PID31729、XD b06 remove50 PID31727及原在线记录进程103016/84860未停止/重启。GPU0/2曾因本轮严格验收等待较久，是执行层门禁导致空卡，不是原计算失败。

r04的dense/AttentionTopK输出bitwise相同，VisionZip初次bitwise gate失败max_abs1.43e-6，失败目录和云端CRASHED保留。补做独立固定输入native/native对照：diagnostic-r01被原B1布局契约拒绝（脚本用错B8，未放宽契约），diagnostic-r02 separate B1 setup通过；SwanLab5xr8gik1 FINISHED且qa/completed=1核验成功。Abuse旧/旧max1.907e-6、旧/新2.146e-6；Normal旧/旧1.550e-6、旧/新1.669e-6，relativeL2<=4.453e-7、RGB/pixels exact。源码原VisionZip FP32 scatter_add_未改，CUDA归并本身非bitwise稳定。仅VisionZip采用独立原生重复波动界：native_noise>0且<=1e-5、cross<=min(1e-5,2*native_noise+1e-7)、relativeL2<=1e-6；另三方法仍bitwise exact，不是一律放宽，不使用正式分数。

r05正式工程验收SwanLab pk85cmf7已云端FINISHED、qa/completed=1。同GPU2完整两fit视频83clips、原提取/FeatureStore写入/校验、两次反向顺序对照：dense2.030x、AttentionTopK1.806x、VisionZip1.617x、Uniform1.794x，中位1.800x；四CPU供数聚合1.688x。最终选择workers2/prefetch2/deduplicate=true/temporal_cache=false（按这次实测选择，不预设cache必更快）。RGB/pixels exact，dense/AttentionTopK/Uniform pooled exact；VisionZip native/native及native/new同为max1.669e-6、relativeL2 4.370e-7，通过上述窄界。独立数值回执及config/acceptance/cloud/source各SHA绑定，源码/采样/精度/reducer/head未改。

pending-execution已ready；采用r05快照`/users/fotile/VAD/work/key-attention-optimization-20261013-r05/snapshot`，source SHA32dabf9b51d98d0b2f97e12e576c6144fce1b9a96f234e7ed471bdab84e89683。本机调度器升级为119980，最新队列`work/key-attn-formal-20261013/controller-r03-optimized2/queue.json`：直接接管原XD在线进程，仅刷新两项尚未初始化SwanLab/尚未发GPU的等待tracker；源GPU进程没有被重放。新GPU0 UCF b06 remove70 PID16462，新GPU2 XD同条件PID16460均真实运行、native19720d4e身份门通过并生成视频完成事件。远端15:39:19为UCF6/290（1294clips）、XD8/800（908clips）；GPU0其他项目1827保留。当前四卡生产为两旧XD+两新任务，真实四个全部新版的聚合收益仍待后续接力验证。最新GPU/像素/头/帧采样与原冻结保持一致；不读取正式AUC/AP。

## 2026-10-13：Key/Attention 单点压缩消融开始实施

用户批准[新方案](key_attention/README.md)：AttentionTopK、VisionZip encoder-internal adaptation与Uniform同预算对照，block 6/9/12各单次，删除50%/70%（keep0.50/0.30）。本轮不使用L2排序、不做动态/静态分组，不改旧冻结合同。已建立qzt/key-attn-ablation-20261013分支并校正仓库局部提交邮箱为用户要求的noreply身份。

首阶段native VideoMAEv2的post-attention/pre-MLP桥、精确预算算法、不可变部署身份、fit-only QA及本机SwanLab在线转发入口已实现；VideoMAE/TimeSformer尚未接入。当前本地相关单元回归79 passed，compileall通过，保留既有numexpr版本提示；尚未据此宣称真实模型质量/速度。node3公网连接已核验，空闲GPU可用于QA；不修改其他项目进程。真实两视频×两窗口QA及云端核验待执行。

**本轮后续实测：** r06在node3空闲A100 GPU1完成2个fit视频×2窗口、全部18压缩条件，共36次B=2压缩前向；6个capture-on identity与退出后dense恢复的pooled最大绝对误差均0。真实token由1568变784/470，当前层MLP和后续层shape正确；独立K投影对full-QKV参考最大误差2.9564e-05。18份membership/mass回读核验通过。本地最终受影响回归84 passed、compileall通过。代码`91fc9bb`，tag `key-attn-qa-v4-20261013`；[SwanLab](https://swanlab.cn/@ryfrcd693/VAD-Key-Attention-20261013/runs/c2u9erwg)已回读FINISHED及完成/计数标量，作业退出0且GPU释放。摘要见[key_attention/qa-result-20261013.json](key_attention/qa-result-20261013.json)。

r01–r05启动/验收失败目录均保留：打包资源、Windows SSH行尾、既有overlay/cuDNN-off环境入口、Python3.12/3.10 AST空字段差异均已定位；有在线启动的失败均保留CRASHED记录。成功轮复用原baseline环境钩子，不重装环境，不修改权重，不读正式test分数。该阶段仅证明V2工程路径通过，不代表AUC/AP或净加速达标；VMA/TS适配、新正式特征/评分许可与正式质量效率矩阵仍未完成。

## 2026-09-26：经node3内网跳转node2完成6根物理归档

## 2026-10-05：框架基线合并准备与验收

按用户明确授权，将 GitHub main 的 `ed9626e` 缓存清理合入 `qzt/vad-framework-pairselect-20261005`，解决唯一 `.gitignore` add/add 冲突：保留框架完整忽略规则，接纳29个历史Python缓存的移除。当前受跟踪Python缓存为0，无未解决冲突。框架重构及PairSelect README/论文证据已在 `e5a0b2a` 提交；两个旧远端分支均无独有提交并已清理，本地历史保留。

本次合并结果在2026-10-05使用 `C:/Users/lenovo/anaconda3/envs/pytorch/python.exe` 验证：`python -m compileall -q src tests` 通过；工作流边界/身份兼容、打包资源、token选择/部署、特征合并、DSANet入口、CLIP桥与UR-DMU质量导出共76项测试全部通过（64.40秒）；基础CLI帮助、encoder目录、paper status及DSANet帮助均退出0。未重跑模型实验，历史结果身份保持。此前当前框架协议回归54 passed、3 skipped（Windows符号链接权限）保留其验证范围。后续通过PR保留提交历史合入main，并以 `vad-framework-baseline-20261005` tag固定最终主分支提交；GitHub最终状态以PR与tag为准。

## 2026-09-30｜通用工作流重构完成并同步服务器

通用特征身份/许可归入data/engine，提取/恢复/合并/检测/质量/效率归入workflows，
UR-DMU后端归入integrations.detectors.urdmu；11个旧paper入口保留同一实现的兼容导入。
共享ReductionDeployment/ReductionExecutionContext接入实际部署；profile三主线与
assets/experiments/icassp2027/runs对齐，输出挂载与配置输入的路径约束分开。
修复恢复回执与每层token/耗时字段不一致，源回执和所有身份检查保留。

本地受影响分组509通过/7跳过；服务器290通过/1跳过，实际Linux软链接边界通过。
compileall与独立wheel回归通过；71文件增量经预期SHA校验及备份后同步。
先前中断的大范围运行不计入通过数。正式论文实验与旧资产未改。
详见[当前架构](../../docs/architecture/current-system.md)与
[验收、跳过原因和回执](organization/EXECUTION_STATUS.md)。


用户指出替代线路后，公网进入node3再`ssh ibnode2`实测成功，原Tailscale故障不再阻塞。完整目录/软链接扫描覆盖四个data2根和两个users根，无外跳相对软链接；60份正式合同/配置/head元数据未见父目录相对路径。node2主特征根含27938202个常规文件条目，合并视图实测大量硬链接。因此采用同盘原子目录交换：四个data2根归入`/data2/localdisk/fotile-wavad/storage/<原名>`；共享users的运行根与旧数据源归入`/users/fotile/VAD/archive/{icassp2027-runs,datasets}`。原六个路径均变兼容软链，常规文件与硬链接不复制不改写。archive已忽略Git。

迁移前已验证中断回滚和独立超时SIGCONT救援；只在维护窗口短暂暂停本项目空队列/hold/monitor守护及非计算子进程，成功窗口node2约16.56秒、node3约19.34秒，均恢复。一次早期保护检查因未识别monitor的tr子进程中止并恢复，未移动目录，回执保留。两节点各96个锚点与314条旧导航均通过，6根inode保持；物理位置映射3455条、missing=0。回执在`assets/catalog/physical-consolidation-20260926.json`、`physical-paths-20260926.jsonl`及provenance的physical-relocation/physical-anchors/maintenance-pause文件。旧raw来源保留为归档备份，未删除历史实验或改科学结果。当前状态见 [执行状态](organization/EXECUTION_STATUS.md)。

## 2026-09-26：WAVAD源码与数据入口实际整合，特征/实验完成分层编目

最终增量验证补充：服务器17项通过；原CLIP Python3.9.23通过薄启动器导入同一extract/score实现和核心CLIP桥，质量模块在原classic Python3.10.20成功导入。回执为 `assets/provenance/organization-final-tests-20260926.log`、`dsanet-legacy-runtime-20260926.json`、`dsanet-quality-runtime-20260926.json`；未加载模型或新增正式实验。两份DSANet作者权重SHA也已复核，加上六个UR-DMU正式head，共8个检测checkpoint校验通过。

用户授权开始整理后，新建共享资产根 `/data2/localdisk/fotile-wavad` 和 `VAD/assets`。两数据集累计189665356641 B复制完成且rsync checksum无差异，默认raw入口切到UCF1900/XD4750完整视频目录，保留旧UCF897历史入口和原始来源。源码候选672文件经SHA核验后提升到远端VAD：456新增/更新、216保持原字节、0并发冲突；原源码、dirty patch与状态已备份，未改Git历史。回归191 passed、2 skipped，compileall及catalog/status通过。token_reduction去掉对paper的反向依赖，新增DSANet extract/score/quality框架入口和9项本地测试，资产完整性3项测试通过；远端增量验证另记回执。

建立139条特征入口和175条实验/来源导航，314条解析检查通过；24份冻结正式测试索引（11.94GB）与合同SHA全部匹配，首blob抽样SHA/CRC通过。运行目录深度编目收集3419个回执目录、174个显式run ID，48目录绑定冻结矩阵，未把目录数写成成功实验数。六个UR-DMU正式checkpoint完整SHA复核通过。本地重复源码ZIP及两份过时接管文档已删除。当前特征/run仍由链接引用旧物理位置，node2今日SSH超时使跨节点活跃writer复核与全量物理迁移尚未完成；没有停止原作业或改写旧实验回执。详见 [执行状态](organization/EXECUTION_STATUS.md) 和 [资产操作说明](../../docs/operations/wavad-assets.md)。

## 2026-09-25：整理计划按明确代码、特征、数据集落点重写

根据用户反馈，重新核对 node2/node3 的物理挂载、数据集入口、dense/压缩/预算扫描/DSANet 特征和正式/开发/失败运行，更新 [organization/README.md](organization/README.md) 及四份配套文档。明确以 `/users/fotile/VAD` 放代码、`/data2/localdisk/fotile-wavad/` 放大型数据/特征/实验并由 `VAD/assets` 导航，按完整 run 分批迁移和保留旧路径兼容链接。综合本地相关 Markdown 与正式结果，新增 [实验与文档重编](organization/04_实验与文档重编.md)。已删除两份过时且无独立证据的本地接管/阶段计划；项目源码 ZIP 删除被执行策略拦截，仍在原位。服务器未迁移、未停作业、未删资产；全量文件级盘点和迁移 QA 待执行。

## 2026-09-25：WAVAD 与 ICASSP 资产整合计划（仅规划）

新增 [organization/README.md](organization/README.md) 及资产清单、目标架构、实施验收三份配套文档。依据本地工作树和 node3 只读目录/进程盘点，记录旧框架与论文运行快照分离、数据和特征散布在 `/users/fotile` 与 `/data2`、部分守护进程仍在等现状。尚未完成全量资产清单，未改远端文件、移动数据、停止作业、读取测试分数或改动框架源码；各阶段均待执行验收。

## 2026-09-24：补齐通讯作者单位上标

按用户确认，将 `manuscript-20260922/author_config.tex` 中 Jianbo Yu 的上标由 `\ddagger` 改为 `\dagger,\ddagger`，对应复旦大学与通讯作者身份。原 `main.pdf` 被占用，首次原路径编译无法写入；改用独立输出目录，两次 pdfLaTeX 编译成功，五页 PDF 保存为 `manuscript-20260922/updated_pdf/author_affiliation/qi.pdf`。首页渲染及文本核对通过，无未定义引用；保留既有约 1.92pt / 0.69pt 纵向警告。旧路径 PDF 未更新，请使用此次新文件。

## 2026-09-24：删除第五页普通致谢

按用户要求删除 `sections/07_declarations.tex` 中的普通资源致谢句及 Acknowledgments 标题，保留伦理声明。两次 pdfLaTeX 编译成功；五页 PDF 的第五页已渲染与文本核对，仅保留伦理声明和参考文献。无未定义引用；纵向轻微警告约 1.92pt / 0.69pt，未见裁切。原路径 `qi.pdf` 被其他程序占用，覆盖与移动均被 Windows 拒绝；新版保存为 `manuscript-20260922/updated_pdf/qi.pdf`，`main.pdf` 同为新版。原目录 `qi.pdf` 仍为旧版，不能误用。旧文件备份在 `outputs/pairselect_drawio/paper_before_remove_acknowledgments_20260924/`。

## 2026-09-24：按第一作者姓氏重命名论文 PDF

按用户要求，将当前五页论文 `manuscript-20260922/main.pdf` 重命名为 `manuscript-20260922/Qi.pdf`，对应第一作者 Zitong Qi 的姓氏。重命名前后 SHA-256 一致，论文内容未变。

## 2026-09-24：新增用户提供的 PairSelect 概念总览图

按用户要求将竖版概念图重建为 `figures/pairselect_graphical_overview.drawio`：第一页包含 205 个可单独修改的对象（文字、曲线、箭头、配对圆点、虚线圈、模型图标及四段原始照片），第二页明确标为原图参考页并完整嵌入原 PNG。可编辑页字体、渐变及抗锯齿仍有细微差异，不宣称逐像素一致。为保证论文图细节保真，论文 PNG 使用参考页对应的原始像素，仅裁外侧空白；嵌入原图字节、论文裁剪像素和两个 draw.io 副本均核对一致。

新概念图位于第一页右栏 Fig. 1，原流程图仍在第二页并顺延为 Fig. 2；更新引言图引用及浮动体声明位置。图注明确分数曲线为 schematic，未作为新实验结果。保留已有正文、讨论结论、表格和实验数据。两轮排版各运行两次 `pdflatex -interaction=nonstopmode -halt-on-error main.tex`；PDF 五页、技术正文结束于第四页，五页均已渲染检查，无未定义引用或横向溢出，保留约 1.92pt 纵向警告但未见遮挡或裁切。draw.io 结构检查 0 错误，原生文字溢出 0；有意叠放的图元仍产生重叠告警。旧 PDF 与改动前源文件保存在 `outputs/pairselect_drawio/paper_before_graphical_overview_20260924_182304/`，回执与编辑说明在同级目录。未运行模型实验，未提交或发布。

## 2026-09-24：第 5、6 节按用户确认文本替换

仅替换讨论与结论正文，使用用户逐句确认的两段英文；仅作 LaTeX 百分号、乘号、连接号和表格引用转换。其余正文、方法、数据、表格和现有图不改。论文目录内两次 pdfLaTeX 编译成功，PDF 五页，技术正文结束于第四页；逐页渲染检查未见重叠或裁切，无未定义引用。balance 留有约 1.36pt 纵向警告。更新工程包为 `PairSelect_sections56_20260924.zip`。

## 2026-09-24：按用户要求覆盖论文流程图（补 DSANet）

将当前确认的原布局 draw.io（仅修复 Patch Embedding，并在三处 encoder 列表追加 DSANet）覆盖到 `manuscript-20260922/figures/pairselect_framework.png`，同时放入同名 `.drawio` 源文件。新图由 draw.io 渲染器 2× 导出，裁去内嵌图注，保留现有 LaTeX caption 和论文正文。两次 `pdflatex -interaction=nonstopmode -halt-on-error main.tex` 成功，`main.pdf` 仍为五页；已检查第二页图及全篇渲染。无未定义引用或横向溢出，balance 仍有 1.36pt 纵向警告。旧 PDF、图及框架 tex 已备份到 `outputs/pairselect_drawio/paper_before_dsanet_figure_20260924/`。未改动模型、表格或实验数据。

## 2026-09-24：按用户附件替换论文叙述

摘要、四系统 token 表、实验协议、效率段和结论按用户提供问答的英文替换稿更新；同步投稿摘要和表格生成器。主文删除 DSANet 独立完整视频测速段落及相关引用，缓存差异和未验证数值等价说明移到 `notes/REBUILD_HANDOFF.md`，原始证据与质量数值保留。运行两次 `pdflatex -interaction=nonstopmode -halt-on-error main.tex`，五页 PDF 逐页渲染检查完成；无未定义引用或横向溢出，balance 留有 1.14pt 纵向警告但未见文字重叠或裁切。`python -m compileall -q src tests` 通过，表格生成器语法及四行计数核对通过，四段替换文案与附件逐字核对通过。未运行模型实验。

## 2026-09-24：论文共同一作标识

在 `manuscript-20260922/author_config.tex` 中为 Zitong Qi 与 Yu Ji 添加同一上标 `1`，作者栏注明两人对本文贡献相同；单位及通讯作者标识未改。用本机 pdfLaTeX 重新生成五页 `main.pdf`，首页渲染核对通过。

## 2026-09-24：四系统论文 Table 3 证据复核与合表

只读核对 `manuscript-20260922/evidence_snapshot/evidence/dsanet/`：UCF/XD 各有 dense、删除 20/40/60% 的 CLIP–DSANet 正式质量回执，同数据集四档的 checkpoint 与测试 manifest 一致；原始 batch-32 编码器计时及完整视频独占计时均保留。Table 3 用同一张表按预算列示四系统的编码器效率，CLIP 主表只选 XD 训练输入的一组，与三视频编码器的数据集来源一致；已删去 B1、显存及完整视频两列，完整视频结果只在正文中说明。每列只在同一系统的四档内标粗最优。导出脚本保留 20 条原始编码器记录，主表展示 16 行；未运行模型、未修改原始证据或质量数据。

## 2026-09-23：DSANet 发布权重四档评分与质量/效率完成（用户范围覆盖）

用户改为不训练检测头、每数据集只用一份作者发布权重。node3 的八卡评分在本机时间23:06:08约5ms窗口同时开始并完成：UCF四档各290视频，XD四档各800视频；各数据集四档权重SHA和测试manifest一致，评分时不读取真值。旧本地full-train head尝试原位保留但不用于本轮结果；旧等待本地head评分的守护PID31830按归属核对后SIGTERM，其他项目进程未改。

正式质量：UCF新raw dense帧AUC 88.868%，删除20/40/60%为88.542/87.966/87.065%；XD新raw dense step AP 85.597%，三档为85.490/85.353/84.379%。10,000次视频级bootstrap的主指标抽样全部有效，预设CI下界≥−0.5pp仅XD删除20%得到支持；其余档保留“未证实”。原生定位五阈值mAP、两分支所有质量指标和原始回执见 [`RESULTS_PUBLISHED_20260923.md`](../../work/dsanet-extension-20260923-r01/RESULTS_PUBLISHED_20260923.md)。UCF此前作者预提特征+发布权重89.4446%是独立身份，不能与本轮新raw特征算压缩差值。

编码器batch32三档倍率UCF为1.096/1.222/1.381×、XD为1.093/1.221/1.383×；batch1三档均略慢。检测头GPU分数计时、完整视频到CPU分数同卡三独立进程计时均已完成。端到端倍率UCF三档为0.968/0.991/0.978×、XD为1.008/1.008/1.014×，完整流程峰值显存未随压缩下降；不得把编码器收益写成完整视频收益。所有正式端到端配置各用对应数据集相同16条完整训练视频，覆盖、帧数、checkpoint SHA已复核。条件能耗/F1与不完整算子覆盖的总GFLOPs按缺项原因标注，不填伪数。

## 2026-09-23 node3本机18:23：UCF尾部接力清单工具就绪但未执行迁移

UCF轻片还未完成，所有八卡仍有本轮提取PID。为可能的UCF尾部负载均衡备好一次性 `work/dsanet-extension-20260923-r01/plan_ucf_tail_rebalance.py`：仅在旧四片writer均退出、轻片2/3有完整completion且回执覆盖预定成员时，才会按已完成回执排除视频、把尚未提取的整视频按原文件字节量贪心分成四份新的唯一manifest；不移动/覆盖旧特征。当前对真实在跑的shard0做安全门试验，脚本按预期拒绝 `old UCF shard0 writer is still alive`，目标目录不存在，四个UCF提取PID均未被改变。**没有启动新任务、没有重分配任何视频。** 轻片完成后的接力需再次根据现场CPU/存储和剩余回执决定，不能把计划清单当作已执行结果。

## 2026-09-23 node3本机18:20：第二条超长UCF训练视频完成

UCF shard1 从258推进至261/403，`Normal_Videos308_x264` 完成回执已核实：原始976503帧，冻结16帧组产生61032行，10个crop文件全部存在，总计1,249,936,640字节；本地复制回执为 `work/dsanet-extension-20260923-r01/qa/normal308-receipt.json`。这说明此前长时间无video计数增长确为一个完整9小时源视频的提取耗时，不是作业失败。UCF四片同期296/261/354/336，XD465/411/424/534；八路进程仍在执行后续完整训练视频，head尚未启动、无正式检测分数。

两条超长视频完成后的第二次只读剩余量快照已保存到 `work/dsanet-extension-20260923-r01/qa/remaining-train-bytes-r02.json`：UCF四片约14.92/15.00/1.87/2.47GB待处理原视频，XD四片约9.80/10.43/10.63/8.47GB。前者失衡仍显著；继续保留当前八路、等较轻片真正完成后才决定是否有必要按**未开始的整视频**接力到空卡。

## 2026-09-23 node3本机18:14前后：剩余原视频字节量确认UCF尾部失衡

只读剩余字节盘点 `work/dsanet-extension-20260923-r01/qa/remaining-train-bytes-r01.json` 按固定manifest和现有完成回执计算：当时UCF四片剩余112/145/60/84视频，原视频字节约15.94/23.70/3.41/4.22GB；XD四片约10.78/11.56/10.63/9.65GB。UCF前两片的剩余大头包括 `Normal_Videos308` 8.67GB与其他较大正常视频。原视频字节仅是工作量proxy，编码复杂度/帧数不同，不能换算确定ETA。当前不停止运行中的完整视频；若后两片先完成且前两片仍有大量**未开始**视频，再以CPU/存储负载与完整回执判断是否对空闲GPU做无覆盖、无重复的整视频接力。没有据此缩减全训练集或改test配置。

## 2026-09-23 node3本机18:12：首条超长UCF训练视频完成并核对十crop

UCF shard0 从258推进至272/403，`Normal_Videos307_x264` 已正式写完：原始628020帧，按冻结16帧组生成39252行；10个crop特征文件全部存在，总计803,882,240字节。该视频完成回执是单独结果，不读取test分数。GPU1的更长 `Normal_Videos308_x264`仍在运行；UCF shard1暂维持258/403且进程/GPU活动存在。同期UCF shard2/3为326/317，XD四片393/345/404/472。训练head仍未满足1610/3950全量门槛。

## 2026-09-23 node3本机17:55前后：完整训练视频字节量与静态分片失衡已核实

当前八路提取均有存活进程和GPU/CPU活动；UCF shard0/1短时不增视频数不是卡死，它们正在处理正常长视频307/308：分别5.48GB/8.67GB、628020/976503帧，代表帧加十crop计算量远大于普通视频。只读全训练集stat清单见 `work/dsanet-extension-20260923-r01/qa/train-size-inventory-r01.json`：UCF1610原视频合计95.81GB，4个分片按视频数均分却按字节为31.05/31.69/16.26/16.81GB；XD accepted3950合计74.64GB，4片18.86/18.65/18.82/18.31GB。UCF前两片未来可能成为尾部瓶颈，当前不能按已完成视频比例线性推ETA。保持现有视频身份和8路生产运行；等轻片真实完成、GPU释放后再以剩余完整视频和CPU/存储负载决定是否值得有界接力，不覆盖旧特征或重复写同一视频。

## 2026-09-23 node3本机17:49：UCF refiner调度总预算在训练前对齐作者源码

复核DSANet原始 `ucf_train.py` 发现refiner的WarmCosine总预算为 `epochs × (len(normal_loader)+len(anomaly_loader))`，而本轮待启动新入口原先写为 `epochs × 实际每epoch优化步数`；两者学习率轨迹不同。已在四head尚未启动时修正：UCF保留作者的调度总预算，实际迭代仍取覆盖完整数据的两loader较长者并循环较短loader，XD调度与实际loader长度原本一致。resolved训练回执将并列记录`optimizer_steps_per_epoch`和`refiner_schedule_batches_per_epoch/total_iters`，不把调整后的训练称作者checkpoint的精确复现。远端新训练脚本已同步并通过Python语法检查；8个特征提取作业未受影响，未读取正式测试分数。

## 2026-09-23 node3本机17:47：训练继续，四head独立完整性门已补验证

八路训练提取继续推进：UCF分片219/220/228/219，XD208/184/218/254，PID全部存活。head入口尚未启动。新增 `verify_four_heads.py` 在四份final完成后核对本轮训练CSV**实际**行数16100/39500、唯一路径、CSV SHA、seed234/235数据集和固定epoch身份、参数总数/可训练数、最终权重SHA与文件大小；四份head QA未通过则八路评分不会派发。合成损坏权重测试1项通过，并核验损坏时不会写出ready回执。node3等待守护PID `16331/1318/31830`均仍存活，测试分数未读取。

## 2026-09-23 node3本机17:45前后：head参数与检查点身份纳入正式回执

在四个正式DSANet head尚未启动前，训练入口 `train_dsanet_final.py` 增加实际加载模型的总/可训练/冻结参数量到resolved config和完成回执；最终固定epoch权重写完后回执记录文件字节数与SHA-256。评分守护新增 `verify_four_heads.py` 强制对四head逐一核对：UCF16100/XD39500十crop CSV行数和SHA、训练seed及数据集、10个固定epoch、无test选模标志、参数量一致、final文件SHA/大小一致、两seed文件身份独立。只有四head QA写出`head-qa.json`后才会释放主8卡同步评分。旧评分等待守护PID `31739`已按归属核对SIGTERM，替换为PID `1318`，训练作业和另外两名守护未中断。

接力后八个训练分片仍有新增回执：UCF216/219/219/207，XD194/184/204/243；四head/评分尚未启动。此前发布检查点的文件大小只作参考，本轮参数/存储主表应以新固定final回执为准。

## 2026-09-23 node3本机17:40前后：训练特征早期QA通过

训练进行中只读抽样QA `work/dsanet-extension-20260923-r01/qa/partial-train-20260923-r01.json` 已通过：UCF四片累计760个唯一视频、XD累计656个唯一视频；每一片已产出的video ID均属于该片固定的全训练集来源，无跨片重复或额外成员。每片选首末各3个视频回执、十crop全核对：48个回执的帧组数/原帧索引/512维FP32特征形状正确，480个 `.npy` 文件SHA与回执一致。该检查不读测试分数；它是运行中的抽样QA，不替代所有1610/3950视频的最终全量CSV验收。

本地四份冻结来源JSONL按`video_id`只读交叉核对：UCF train/test集合交集0，XD accepted train/test集合交集0。这是视频ID层面的检查，不推断官方数据中是否存在同源电影或近重复片段。

## 2026-09-23 node3本机17:38前后：按数据集独立接力完整训练，保留八卡评分屏障

发现旧训练守护错误地把UCF与XD四片的完成合成全局8片屏障，会在UCF先ready时闲置其GPU并推迟两套head。现改成每个数据集独立审核4份分片completion/对应PID退出→生成该数据集16,100或39,500条十crop CSV→立刻在该数据集对应两张GPU启动seed234/235 full-train head；另一数据集未ready不阻塞此阶段。正式八路主score仍等待四head均完成后统一释放，满足用户的八张A100同时推理要求。旧训练守护PID `31738`核对后SIGTERM，换新版PID `16331`；8个在跑提取PID未被中断，head/score守护 `31739/31830`保持原状态。

更换后只读复核8路提取均在进展：UCF各分片161/184/188/161、XD各分片165/132/131/176，全部尚未完成。训练CSV、正式head与评分都尚未启动，未读test分数。

## 2026-09-23 node3本机17:33：八路训练继续，四head与复核评分门槛已核实

当前训练提取八PID全部存活且持续前进：UCF分片126/163/164/140，XD分片143/107/102/145；8卡仍一项本轮提取任务/卡。改进后的后续守护PID `31738/31739/31830` 已远端`bash -n`并实测存活。四head的seed234/235训练均只能在对应完整CSV之后发起；八路主评分须四head全退出并有final后同时释放，seed235四格复核须主评分PID全部退出后才能在GPU0/1/4/5发起。此时CSV、正式head、score均未出现，前期八路测试特征仍是sealed且尚未读分。

本轮将step AP质量导出扩成显式 `--seed` 和 `--tiers`，并要求同一dense/压缩比较的score run全部为completed、视频数正确、测试manifest SHA一致、head checkpoint SHA一致；score原始文件与变换后区间文件都保留SHA。两项导出辅助测试通过。头部计时入口新增16条固定训练视频的独占特征→DSANet GPU测量；完整视频基准把CPU解码/裁剪、H2D、视觉encoder、D2H、时序聚合与head分别计时，尚缺final checkpoint而未执行。

## 2026-09-23 node3本机17:30前后：独立DSANet head种子已接入后续调度

当前八路训练特征提取PID仍存活且各自推进，UCF分片最新约87/145/130/107，XD约119/101/94/126（各分母见下节）。为满足关键配置独立head种子，本轮在完整训练特征及CSV通过后并行启动UCF seed234/235（GPU0/1）、XD seed234/235（GPU4/5）；每个head仍对应自己的完整数据集，不跨数据集或压缩档共享参数，压缩档均使用本数据集对应的dense-trained固定final head。主八格只绑定预设seed234。四份final均完成后才同时释放八路主评分，避免GPU1/5与seed235训练同卡重叠；主八格完成后另外四路seed235 dense/删除40%测试头推理，用于训练变异，不能从两seed挑更高分写主表。

三个未运行到下游阶段的旧守护进程已核对PID后SIGTERM并用新版脚本重启，当前守护PID为 `31738/31739/31830`；其中map/复核守护再补上主八路评分PID全部退出后才释放GPU的检查，旧PID `31740`已按归属核对停止。8个训练作业未中断。新增 `score-seed235-r01` 只做四个复核格。远端 `code/status.sh` 已纳入两个seed及复核评分状态。此时四个正式head和八路评分均**尚未启动**，没有新检测质量数值。

## 2026-09-23 node3本机17:26：八路完整训练提取仍在产出

`status.sh`再查8个训练分片均有存活PID和新增视频回执：UCF shard0–3分别58/107/87/72，XD shard0–3分别82/84/70/93；服务器watch-head、watch-score、watch-map三个等待进程均存活。八路test特征仍为封存完成，不存在当前正式head训练或评分。完整训练集门槛没有跳过。实时节点时钟偏差见下节。

为避免最终“端到端加速”缺口，新增仅在训练视频上使用的 `benchmark_dsanet_head.py` 和完整视频 `benchmark_dsanet_pipeline.py`：分别准备记录DSANet特征到分数GPU时延、从打开视频到分数的wall时延，并对后者分开记录解码/裁剪、H2D、encoder含选择、D2H、聚合与head时间。两个入口已通过Python语法检查，尚缺固定final权重和独占GPU，**未运行、没有数值**。

## 2026-09-23 node3 时钟勘误：旧远端时间戳是服务器本机时间

本轮17:22:00（node3远端显示）同步比对宿主时钟09:31:31 UTC，node3约慢9分半；`timedatectl`显示node3 NTP未启用/未同步。下方所有由node3 `date -Is/-Ins`写出的16:26:39、17:12:45等启动时间，以及远端作业日志时间，都应读作**node3本机时间**，不能当作已与宿主校准的香港时间。并发屏障内的相对窗口（约39ms/100ms）和各脚本`time.perf_counter()`测得的持续时间仍有意义。本轮不修改共享服务器时间、不追溯改写原始回执；跨主机绝对时间汇总必须附时钟偏差。

## 2026-09-23 17:20 DSANet 完整训练特征持续进行，解码优化已做训练侧核验

本轮接力复核 node3：8个训练提取PID `8563/8566/8596/8597/8598/8599/8600/8601` 均实测存活、各绑GPU0–7，约7分钟后UCF各分片26/47/57/49视频，XD各分片36/57/35/51视频；GPU/存储无错误，`/data2`可用约4.8TB。此前8路test特征已封存，不重复读取分数。后续守护PID32047/32049/23101仍在，训练全量CSV、固定final检查点和质量尚未出现。当前CPU负载约45（32核），运行稳定。

为了核对CPU解码是否值得中途切换，仅在UCF/XD各一条**训练视频**比较现有逐帧`read()`与`grab()+retrieve()`取代表帧：两条视频171/196组的选中图像逐像素完全一致，局部解码wall倍率分别`1.382×`/`1.090×`。但十crop预处理与视觉编码仍占全流程时间，收益有限且仅覆盖两条视频；保留已经冻结并运行的提取代码，不中断、不混用新实现。证据在服务器 `smoke/decode-comparison-r01.json`，不是正式端到端效率。

本轮进一步检查将来完整pipeline基准脚本的计时边界，确保视频打开与元信息读取落在wall计时内；脚本尚待head完成后才运行。Goal继续active，不将测试特征完成等同于八项质量实验完成。

## 2026-09-23 17:13 DSANet 测试特征八格封存，训练特征八卡同时开工

`test-campaign-r02` 八档raw CLIP特征均已完成并通过 `check_eight_features.py` 封存，回执同步本地 `work/dsanet-extension-20260923-r01/sealed_test_features.json`：UCF四档各290视频/69,634图像，XD四档各800视频/146,449图像；同数据集dense/compressed逐视频帧索引、crop及特征shape匹配。UCF290/290原始帧数与官方帧真值精确相同；XD800/800原始帧完整覆盖已封存的canonical前缀。封存标记 `test_scores_read=false`，迄今未读取或导出正式分数。

完整训练集特征campaign在17:12:45.941 HKT释放8卡屏障，8个PID实际开始落在约100ms内。GPU0–3各处理UCF1610的一个分片（403/403/402/402视频），GPU4–7各处理XD accepted3950的一个分片（988/988/987/987视频），每个视频十crop。启动后8/8进程实测存活，已出现早期完成回执；输出为 `/data2/localdisk/fotile-icassp2027-dsanet-extension-20260923-r01/train-features-r01/`。两数据集标签和训练列表完整核验，UCF800正常、XD2046正常。训练分片全齐后才生成10crop CSV、训练两套DSANet固定final检查点并同时启动8路评分；守护进程已就位，任何缺失停下游。

两数据集3进程的encoder独占计时完成，新增AP引擎及受影响旧评测回归测试 `55+23` 项通过，`compileall src tests`通过。完整质量、定位mAP、DSANet head及视频端到端效率仍待正式运行；无需把现有旧作者权重分数当本轮dense。

## 2026-09-23 17:06 DSANet CLIP 视觉编码器独占计时双数据集三重复完成

node3 GPU0（UCF）和GPU1（XD）在本轮对应测试特征任务结束后分别独占该卡做短计时；每个数据集3个独立模型加载进程，16条固定训练视频（8正常+8异常）、FP32/TF32-off、batch1/32/128，dense与三个压缩档各20次预热、40次CUDA同步计时。源数据和3份原始回执位于 `work/dsanet-extension-20260923-r01/benchmark/{ucf,xd}-r00/r01/r02.json`；摘要为 `{ucf,xd}-summary.json`。计时边界只含准备好的GPU图像→CLIP视觉塔pooled输出，**包含动态选择和gather**，不含解码/传输/DSANet head。三次中位效率如下：

| dataset | batch | del20 | del40 | del60 |
|---|---:|---:|---:|---:|
| UCF | 1 | 0.953× | 0.956× | 0.932× |
| UCF | 32 | 1.096× | 1.222× | 1.381× |
| UCF | 128 | 1.101× | 1.238× | 1.403× |
| XD | 1 | 0.919× | 0.962× | 0.924× |
| XD | 32 | 1.093× | 1.221× | 1.383× |
| XD | 128 | 1.105× | 1.231× | 1.402× |

batch1变慢已保留；选择器单独测量约0.45–0.57ms。显存allocated/reserved峰值在40%/60%档未稳定下降，受前六层dense和静态权重限制，不建立显存节约主张。每一档另保留实际每层token、principal-block解析GFLOPs、p50/p95、原始40样本与3进程倍率。完整视频端到端测量待固定DSANet head就绪。

UCF测试特征与既有官方帧真值长度290/290精确相等；XD目前已完成的639视频均满足官方canonical前缀可由原视频完整覆盖，无短于真值的记录。UCF/XD新固定训练入口分别完成一step训练样本smoke；分片提取和resume smoke通过，正式完整训练仍待XD测试特征封存后自动启动。

## 2026-09-23 16:56 DSANet UCF四档测试特征完成，XD四档仍在跑

正式 `test-campaign-r02` 中UCF dense/keep0.8/0.6/0.4四路均完成：每路290视频、69,634图像，单路记录约1233–1241秒；它们是8卡共享节点上的**生产提取wall时间**，不能冒充独占加速。XD四路约603–614/800，仍在GPU4–7运行。UCF四路保留共同manifest/权重/bridge摘要。8路完成后守护PID2731才做逐视频帧索引和特征shape封存，封存通过后才自动同时启动8卡训练特征分片；守护PID32047在8个完整train分片与CSV齐全后启动UCF/XD固定final DSANet训练；PID32049等两head通过后启动8路正式head评分；PID23101随后跑作者五档定位mAP。所有自动阶段失败时停止下游，保留失败目录。

训练侧工程验证：UCF正常/异常训练样本DSANet前向、反向及188个有限梯度张量通过；UCF和XD各使用训练特征完成新固定final训练入口的单步smoke，均不建立正式checkpoint；使用发布权重在各一条训练视频上完成新评分入口smoke，无test真值访问。训练feature CSV的单视频十crop收据验证通过。这些不构成正式检测成绩。

GPU0从UCF dense完成后转为独占encoder效率测量：已有一轮UCF训练图像的B1/B32/B128四档、20 warmups+40同步测量回执；另两个独立进程重复在运行。B1压缩可能更慢，B32/B128中高档净加速的方向已见，但三重复和XD/端到端尚未完成，不将此写为最终论文结论。batch128与每路worker1继续遵照高CPU负载现场配置。

当前文件改动：新增独立DSANet执行脚本、CLIP动态PairSelect桥及其测试；共享视频级质量比较器新增独立step AP bootstrap模式，保留原梯形PR-AUC定义。相关55项本地测试通过，`python -m compileall -q src tests`通过。论文未修改，未读取正式测试分数。

## 2026-09-23 16:27 DSANet 八卡同时启动测试特征提取，检测头和评分待完成

用户明确要求8张A100同时开始、每卡一个实验。node3 `ibnode3` 上 `test-campaign-r02` 于16:26:39.016 HKT释放屏障，8个PID实际开始时间在约39毫秒内：GPU0–3为UCF dense/keep0.8/0.6/0.4，GPU4–7为XD相同四档。运行根 `/users/fotile/icassp2027-runs/dsanet-extension-20260923-r01/test-campaign-r02/`，启动后8/8进程存活，已各处理约4–5个视频。旧 `test-campaign-r01` 因dense两路启动脚本空数组错误形成失败attempt；其余6路由主控核对PID后SIGTERM，失败日志及部分输出原位保留，不能混入正式结果。

现有来源清单UCF train1610/test290、XD accepted train3950/test800，测试CSV全清单UCF为crop5、XD为crop0。UCF `Anomaly_Train.txt` 本身包含800个normal，准备脚本的首次重复追加被数量门拦住，修正后的4份清单写入 `manifests-r02/`。CLIP逐图像动态PairSelect bridge已补，真实权重训练视频identity最大误差约`1.49e-6`，三档suffix长度158/119/79；UCF/XD各一个训练视频十crop提取冒烟通过。本地3项CLIP桥测试通过。

八路正在执行raw视频解码与CLIP特征推理，**并非完整测试评分**。完整训练集DSANet dense检测头及固定final checkpoint、八格score与质量、独占效率仍待做；未读取正式测试分数。node3启动前32核CPU负载约70，八路先统一batch128、每路1个decode worker，依据生产吞吐再决定是否调整；其它项目进程未停止或修改。计划和实时约束见 [DSANET_CLIP_EXTENSION_PLAN_20260923.md](DSANET_CLIP_EXTENSION_PLAN_20260923.md)。

## 2026-09-23 DSANet 强 WSVAD 扩展计划完成，尚未启动新实验

用户要求新增一个SOTA级CLIP WSVAD模型，并明确具体选择由主控决定。本轮读完当前PairSelect论文、现有CLIP部署/对齐回执及DSANet官方论文和实现，选定DSANet（AAAI 2026）。详细计划见 [DSANET_CLIP_EXTENSION_PLAN_20260923.md](DSANET_CLIP_EXTENSION_PLAN_20260923.md)：UCF/XD各dense、删除20/40/60%共8次完整test；原生DSANet检测器、对应完整训练集dense训练与固定final；质量/定位mAP/视频级CI，以及batch1/32、端到端、显存、FLOPs和插件开销均有验收项。

node3公网只读核验确认DSANet两套权重和CLIP底座SHA与台账一致；历史UCF89.4446为官方特征+官方权重旧回执，不是本轮raw复现。测试CSV首部UCF为crop5、XD为crop0，训练多crop不能误写成test十crop平均；完整CSV分布仍列为执行前审计。XD原生训练脚本使用test AP选模并回载，正式新训练须隔离该路径。原视频16帧组内取样协议仍未完全恢复，计划规定有界对齐后冻结明确raw身份并完整训练。DSANet源码目录没有.git，历史revision未冒充现场Git验证。

本轮只新增计划并更新此进度入口，未修改论文/实验源码、未启动提取/训练/测试、未commit或push。计划中的预算token/GFLOPs为解析预期，所有新质量/效率实测仍待执行。

## 2026-09-23 论文已回填三档质量与 batch=32 效率，PDF 通过布局核查

经用户明确允许，`projects/icassp2027/manuscript-20260922` 已完成结果回填。没有新增或删除结果表：既有质量表填满 dense、删除20/40/60%三档；既有效率表填入 batch=32 的 token、principal-block GFLOPs/降幅、每批延迟、吞吐和相对dense加速。论文正文不展示batch8/16对比，只在效率表标题和实验设置中声明固定batch=32口径。

摘要、Introduction、Experiments、Scope和Conclusion已同步：中等预算batch32吞吐提升为V2 `1.225×`、VMA `1.247×`、TS `1.277×`；激进预算为`1.330×/1.378×/1.470×`。质量表完整保留UCF VMA高压缩下降和XD梯形PR-AUC/step AP方向差异，不据test选择最优预算。硬件结果限定为准备好的GPU输入到pooled encoder输出，包含动态选择/gather/deployment校验，排除解码、processor、传输、检测头和写盘。

`scripts/export_tables.py`已增加batch32数据一致性检查；`data/batch32_efficiency_results.json`绑定实测值和来源；12项新增质量由审核exports导入`data/budget_results.json`。`submissions/abstract.txt`同步为119词。PDF重新编译为5页Letter（4页技术正文+1页声明/参考文献），`lasttechnical`在第4页；17条引用全部解析，无missing/uncited、Overfull、undefined reference或LaTeX warning。Poppler逐页渲染复核表格、正文、图和参考文献，无裁切/重叠；最终文件为`projects/icassp2027/manuscript-20260922/main.pdf`。

本轮验证：`python scripts/export_tables.py`、`scripts/check_refs.py`、`python -m py_compile scripts/export_tables.py`、`python -m compileall -q src tests`、`git diff --check`通过；PDF metadata为5页、Letter、未加密。仍待作者确认伦理/资助/冲突声明和按会议政策实质性作者撰写核查，未commit、push或投稿。

## 2026-09-23 batch=16/32 快速扩展复测完成：三编码器、三预算全部加速

用户确认论文主要 batch 口径为8，并要求最快补测 batch16/32。node3 三张空闲 A100 并行完成：TimeSformer/VMA/V2 分别使用GPU2/3/4，无同卡共驻计算进程。每个encoder固定同一条正常和一条异常XD训练视频；每个视频按batch大小确定性均匀采样16或32 clips；同一模型实例下比较dense与PairSelect keep=.80/.60/.40。FP32、TF32-off、cuDNN-off、输入、权重、原生forward和pooling均匹配；processor/H2D在计时外，动态选择、gather、deployment context和执行校验在计时内。每个配置预热20次，每条视频正式计时20次，合计40样本；执行顺序按视频和batch轮换。

batch16加速（.80/.60/.40）：TS `1.096/1.257/1.438×`，VMA `1.080/1.182/1.311×`，V2 `1.061/1.209/1.311×`。batch32：TS `1.111/1.277/1.470×`，VMA `1.088/1.247/1.378×`，V2 `1.095/1.225/1.330×`。每格两条视频的方向均一致。结合此前batch8结果，中高压缩三encoder均加速；VMA/V2 keep=.80从batch8近似持平变为batch16/32明确加速。独立报告为 `work/batch-scaling-20260923/RESULTS.md`，原始文件在其`results/`。本轮`verify_results.py`核验脚本SHA、输入receipt、40样本和逐视频方向通过；`compileall -q src tests`通过。论文未修改。

## 2026-09-23 本轮计算完成：12/12 预算质量报告、三编码器效率实测，论文等待允许

本轮约定的计算已经完成。VideoMAE×XD keep=0.40 恢复索引后合并，keep=0.80 只补提缺失17视频/3871clips；两个最终视图均覆盖800视频/145703clips，路径为 `compressed-budgetsweep-r01/videomae/xd/{0p40,0p80}/merged/pair_select-{tier}-r05-resumed`。全部12项新增预算预测及质量导出完成，主控 `campaign-state.json` 为 `completed`。原有dense、keep=0.60、六个full-train head保持原身份，没有重训或全量重提。

独立交付目录为 `work/result-review-20260923-r01/`，主报告 `RESULTS.md`、结构化汇总 `results.json`，包含12项新质量结果、原中等预算对照、12种效率配置及来源。远端69份小型证据文件已逐一与本地SHA核对一致。质量导出验证290/800视频、指标/单位和delta一致；计时逐项验证3独立进程×4预算×2batch×16视频×3正式样本，每encoder1152个样本。Luna max只读复算与汇总逐项一致，没有单位或汇总错误。

主要观察：新增12格中11格点估计满足-0.5百分点容忍度，5格配对95%CI下界达标；UCF VideoMAE keep=0.40下降1.053百分点。XD VideoMAE keep=0.40的梯形PR-AUC增加0.776百分点，但step AP下降1.394百分点，两种定义必须分开报告。

效率使用同一A100 GPU6、FP32、固定16个XD训练视频（8正常+8异常），只测准备好的GPU输入到encoder pooled readout，包含选择/gather/部署开销。TimeSformer batch1在keep=.8/.6/.4加速1.019/1.145/1.308倍；VideoMAE与VideoMAEv2的batch1均比dense慢。batch8在keep=.6/.4的吞吐倍率分别为V2 1.116/1.194、VMA 1.141/1.268、TS 1.228/1.422；VMA/V2 keep=.8接近持平。batch1和batch8峰值allocated/reserved均未明显下降。FLOPs仅为已核实的principal-block解析值，不宣称完整encoder实测FLOPs。累计 `plugin_overhead_ms` 未用于任何单次耗时表。

过程中保留了V2首次加载失败（遗漏既有tokenizers shim）和最后补片评分身份失败；前者补上现有依赖路径后在r02完成3次有效重复，后者识别merge已声明的显式pt/默认pt等价后完成预测，未覆盖旧文件、未伪造指纹。VMA/V2新预算与旧路径的数值等价仍未证明，按用户明确决定接受其复用并保留说明。

自用户要求“先讲清楚、再允许改论文”以来，论文目录内容摘要核对完全不变；未回填新指标、未重编PDF。暂停前的四处说明/表结构修改已在独立包 `changes-before-permission.md` 列出。后续论文修改须等待用户明确允许。本轮 `python -m compileall -q src tests`、新增脚本py_compile通过，导入工具4项测试通过。

## 2026-09-23 当前交付边界：完成计算并解释，论文等待后续明确允许

用户在短暂暂停并讨论计划后要求先完成全部剩余计算与独立结果说明，解释清楚后才允许修改论文。现已恢复提取/合并、预算评分和效率调度；本阶段禁止继续写入 `manuscript-20260922/` 或重编 PDF。独立结果包输出至 `work/result-review-20260923-r01/`，其中记录最新指示后的论文文件摘要，并单列暂停前已发生的四处说明/表结构修改。主控负责执行整合，简单只读效率口径核对已委派 Luna max；不恢复先前的非 Luna 执行代理。

## 2026-09-23 用户决定：复用预算扫描特征，完成 12 项评分与小样本效率测试

用户明确要求复用现有 keep=0.40/0.80 特征，取消将运行环境/代码指纹数值等价验收作为评分前置条件；该决定不等于已证明数值等价。保留真实特征身份、旧失败目录、视频覆盖、时间轴、权重、固定 head 与 SHA 来源检查。旧 dense/keep=0.60 主矩阵保持原身份。

现场更正：TimeSformer 四项新预算预测已完成，可复用。VideoMAE×XD keep=0.40 的 800 视频、145703 clips 已落盘，需恢复汇总索引和合同，无需重提；keep=0.80 缺 17 视频、3871 clips，正按原 pt/classic 路径独立补片。两个最终视图计划为 `compressed-budgetsweep-r01/videomae/xd/{0p40,0p80}/merged/pair_select-{tier}-r05-resumed`。

主控已启动专用评分控制器：node3 公网线路，PID 28162，GPU4，日志 `/users/fotile/icassp2027-runs/budget-resume-20260923/campaign.log`。它复用已有 TimeSformer 预测，其余写入新的 `urdmu-official-predictions-budgetsweep-r02-reuse`，12项质量导出写入 `urdmu-quality-exports-budgetsweep-r02-reuse`。仅对此预算扫描进程放行 code_digest 配对差异，真实 representation 不改写，接受差异回执作为 export 来源；不修改通用校验源码或旧行尾等价表。新输出目前仍在运行，不能据此记为完成。

效率短测在 GPU6 独占该卡执行，固定 XD 训练集16视频（8正常+8异常，同三encoder），测试 dense/keep=0.80/0.60/0.40；batch1延迟/显存、batch8吞吐、20次预热和3独立进程。统一 benchmark 环境与历史质量提取环境的区别须披露；只测量实际执行范围，不将 principal-block 解析 FLOPs 冒充完整 encoder 实测 FLOPs。首轮 runner PID 26894，目录 `efficiency-20260923-r01`，等待实际结果。

本地本轮 `python -m py_compile` 两个专用评分脚本、`python -m compileall -q src tests` 已通过。数值结果和论文表格待真实导出后回填。

## 2026-09-22 20:1x 预算扫描派工定稿：8 对卡 16 GPU 并行（负责人最新口径）+ 19:23 全灭事故根因与处置

**负责人最新调度口径**（覆盖 serial 串行方案）：12 个任务（3 encoder × 2 数据集 × pair_select keep 0.80/0.40，sealed test），16 张卡两两成对 → 同时起跑 8 个任务，成对空闲即接剩余 4 个；过程中持续记录效率数据（显存、加速、FLOPs 等）。

**19:23 全灭事故（12/12 rc=2，总耗时 1.6 秒）根因**：`run_when_gpu_free.py` 的 verify_worktree 拒绝启动——`code-kimi-adgs-20260921` 快照有 69 个 tracked 改动（行尾豁免补丁及后续工作），日志原文 "gpu guard refused to start: tracked worktree changes are present"。该串行运行未走 `go.sh` 的快照 fork 步骤（`code-bsw-r01` 从未创建；远端 sweep 目录当时也缺 `do_snapshot.sh`/`update_plan.py`）。

**处置**：①失败现场 24 个半片目录 + merges + progress.json 完整归档至 `state/archive-failed-20260922-1923/`，未删除未覆盖；②新派工 `pair_worker.py`（一对卡一个进程，NAS mkdir 原子锁认领整任务，两半片并行→自动 merge→认领下一任务；`.failed` 半片永不覆盖；`--reclaim-task` 人工确认后回收死 claim）；③配套 `start_pairs.sh`（每节点 4 对：0-1/2-3/4-5/6-7）、`monitor_gpu.sh`（双节点 20 秒级 GPU 显存/利用率/PID 采样，效率证据只覆盖监视窗口）、`status_pairs.sh`（只读进度聚合，不含任何测试分数）；④本地 py_compile / bash -n 通过。

**现场实测**：node3 GPU3/6 全空，0/1/2/4/5/7 有他项目小进程（GPU5 有 4GB/100% 占用）；node1 8×V100 全空；node2 GPU6 被占 15.7GB 不纳入。16 卡 = node3 8×A100 + node1 8×V100（沿用 gpu-scope-revision-v1.md 范围）。跳板机 ibmnode 链路间歇性超时/重置，bringup（同步脚本→fork 快照→更新 plan→node1 可达性核验）改由带重试的后台任务执行。

**效率数据记录口径**：监视器全程采显存/利用率；runner progress 记录每半片 wall-clock；FeatureStore `reduction_execution` 提供实测保留比与 per-layer token 数（FLOPs 解析计算）；正式延迟/吞吐表仍须后续独占 `--formal-timing` 窗口（共享卡不 eligible，质量不受影响）。

## 2026-09-22 17:00 六格矩阵装配成功并解封（M1/M2 达成）

**执行依据**：`C:\Users\lenovo\Desktop\课件\ICASSP2027_next_steps_v2_20260922.md`（v2 计划）。负责人对三个关口问题的批示：①完整显存证据**不是**解封硬门槛（依据 `reducer-evaluation-freeze-v1.json` 的 `claims_not_established` 明确列 `peak_memory_reduction` 为未建立主张，且 `access_gates` 四项 / `official-detector-protocol-v3` 的 `execution_gates` 七项均不含显存要求）；②**不补开** GPU monitor；③解封与阶段 B **全程自主，做完一次性汇报**。

**主线实测（收官）**：
- XD 九份 extraction 合同全部落盘（最后一份 TS group_random，status=completed、800 视频、291781 clips、failures 为空；合同作为数据写完后的最后提交标记，符合 v2 计划对"合同晚于进程退出"的修正）。
- XD 九个压缩预测全部 `completed`；XD 九份质量报告全部 `available`（16:35 最后一份落盘）。
- **18/18 压缩质量报告齐备**（UCF 9 + XD 9）+ 6 个 dense 基线。矩阵相关进程全部正常退出，调度器无失败。
- `assemble_urdmu_matrix.py`（fail-closed）运行成功：`{"status":"completed","cells":6}`，rc=0。本地与远端快照脚本 SHA 一致（`f38384c4…`）。

**A1 逐视频闭合核验：18/18 PASS。** 对每个压缩 run 的 `coverage.per_video` 与对应 dense 视图 `video_provenance` 逐视频比对：视频 ID 集合精确一致、无重复/缺失/额外；每视频 `intervals` == dense `record_count`；`gap_frames==0`；`coverage_ratio==1.0`；`covered_frames<=num_frames`；clips 总数 == dense 记录总数；合同 `status==ready` 且 `test_only==true`。clip 数按编码器不同（V2/VMA 69364/145703，TS 138853/291781），但每个编码器压缩侧与 dense 侧严格相等。已核实装配器 `METHOD_RUN_UCF`/`METHOD_RUN_XD` 硬编码的 run-id（V2→r03、VMA→r02、TS→r02；XD V2 pair_select→r03）与实测存在的正式 run 逐一吻合；V2 的 `pair_select-0p60-r01/r02` 为历史废弃 attempt，无合同、不参与矩阵。

**A2 预测核验：18/18 PASS。** `result.json` `status==completed` 且 `official_frame_scores_read==false`；`predictions.jsonl` 实测为**一 clip 一行**（例 XD TS pair_select 582,733 行覆盖 800 唯一视频），故按 v2 计划不以行数为判据，而以唯一视频数与 chunk receipt 覆盖为判据；`(video_id, clip_id)` 无重复；所有 XD 评分日志 `direct_insert may only` 报错数 == 0。

**代码级四项核验（派子代理只读取证）**：①检查点为固定最终 step 3000，上游 `xd_main.py` 的"按测试 AP 存最佳"逻辑从未被加载（`xd_main.py` 不在 `SOURCE_SHA256` 与 `_DEFINITIONS` 清单内），训练循环单趟、唯一存盘点 `checkpoints/final.pt`，下游加载侧强制 `checkpoint_selection=="fixed_final_step_no_test_selection"` 与 `steps==3000`；②official_frame 帧覆盖恰为 `[0,num_frames)`，未覆盖帧直接报错，无插值/padding/静默截断旁路；③装配器全脚本唯一 print 只输出 status/cells/output，stdout/stderr 不含任何指标数值；④行尾豁免仅作用于 `code_digest` 且仅在 direct_insert 路径，`refit_head` 不放行，放行表无传递闭包（更保守）。

**三对行尾豁免的实推（含此前的计数更正）**：本行及 14:20 条目原写"4 对"，系计数错误。由 6 个 dense head 的 `checkpoints/final.pt` 元数据实推 `representation.backbone.code_digest`：VMA×UCF `81e31440…`↔压缩 `30fe33c1…`、VMA×XD `9a23c5da…`↔`30fe33c1…`、TS×两数据集共用 `7f9a3ac6…`↔`e5fd0bdf…`；V2 两侧同为 `19720d4e…` 原生一致不需豁免。18 路中 12 路（VMA×6 + TS×6）走豁免、6 路（V2）原生匹配。远端放行表实测含 5 个唯一 digest、`19720d4e` 不在表中。该更正已同步 AGENTS.md、CLAUDE.md、STATUS_DATA_SUMMARY_20260922.md、PROMPT_FOR_ADVISORS_20260922.md。

**一次提前读分事件（如实记录，未重跑改写）**：我为核对 quality export 的 JSON schema 以确认装配器字段，调试打印输出了 UCF × VideoMAEv2 × pair_select 的 `metric.dense_value`（frame ROC-AUC 唯一一个 dense 基线值）。暴露范围仅此 1 个 dense 基线值，未读取任何压缩方法值、其他格子或 XD 指标；该值三方法共用，不指示任何压缩方法优劣，未影响方法选择。已记入 `handoff_v2/unseal_receipt.json` 的 `early_score_exposure`，并已向负责人披露。此后所有检查只打印结构字段。

**两个必须报告的数据质量发现（非失败，但是论文纪律问题）**：
1. **TimeSformer 实际 token 保留比 0.5717**，低于名义 keep_ratio 0.60（V2/VMA 为 0.6001）；成因是 TS 选择单元为完整空间轨迹、粒度取整。同一编码器内三方法保留比完全一致，跨方法预算匹配成立。论文须报告实际值。
2. **FeatureStore `plugin_overhead_ms` 不可用作效率证据**：该字段是随 chunk 重算的滑动平均（同 chunk 内恒定、跨 chunk 单调收敛 3335345→1895034→18998），早期值被初始化污染；且 pair_select 记 `transform_ms`、两个对照记 `gather_ms`，走不同代码路径。论文效率表一律采用阶段 D 干净测量。

**解封后一次性交付（`projects/icassp2027/handoff_v2/`，另同步至远端 `$B/handoff_v2/`）**：`audit_summary.md`（来源/覆盖/预算/head QA，公开版不含分数）、`lineending_equivalence.md`（三对豁免证据与作用域）、`unseal_receipt.json`（矩阵摘要、有效规则版本、关口读数、提前读分事件）、`quality_matrix.md`（六格主表 + XD 补充指标 + 预算 + CI + bootstrap provenance）、`quality_matrix_detail.csv`（6 dense + 18 compressed 可追溯明细）、`method_recommendation.md`（五问回答）、`selection_decision.md`（待负责人填写）、`efficiency_protocol.md`（阶段 D 预写协议）、`paper_claims_checklist.md`（七项方法学自查）。新增 `scripts/icassp2027/emit_quality_matrix.py`。

**阶段 B 核心结论（详见 method_recommendation.md）**：三个方法质量**不可区分**——18/18 格通过 0.5 pp 非劣性边际，所有"方法−dense"百分点差落在 [−0.43, +1.14] pp，方法间点估计差 ≤0.51 pp，全部小于方法-vs-dense 的 CI 宽度（约 ±1–2 pp）。pair_select 六格最差 −0.16 pp（三者中最稳）、两数据集平均皆正（UCF +0.075、XD +0.537），但 **UCF 上平均不如 group_random**（−0.147 pp）；group_random UCF 平均最好（+0.222 pp）且最差仅 −0.09 pp。质量候选建议 pair_select 但必须写"效率待确认"。方法-vs-方法配对 bootstrap CI 待补（议程 §B2 许可先交点估计），在它出来前不得写"统计显著优于"。

**红线保持**：正式分数在装配+审计通过后才用于分析；失败目录全部保留未覆盖；未伪造任何指纹；node2 未接入；10,000 次 paired bootstrap 未减少未改定义。

---

## 2026-09-22 14:20 全面收尾状态与今日事件归档

node3 实测：UCF 九路压缩"提取→预测→质量报告"全链路闭环（9/9 exports available，TS 最后一份 14:09 前落盘，UCF 压缩侧正式完成）；XD 九路提取 6788/7200=94.3%（V2 2240、VMA 2354、TS 2194，各 /2400），单 run 约 1 视频/分钟，预计 60–90 分钟内出 extraction 合同；已完成预测 15/24（6 dense + 9 UCF 压缩）。调度器 13:51 重启后 6 个曾失败任务全部自动恢复：根因为 VMA/TS 共用的适配器包装文件 dense 侧 LF、压缩侧 CRLF 行尾差异，被按原始字节计算的代码指纹判为两个代码版本——文件内容 diff 为空（0 行不同），且用仓库原装 _verified_code_digest 复算 XD-VMA 全部 68 个、XD-TS 全部 135 个 dense 源运行均等于 head 记录值，与压缩侧仅差该文件行尾；model/processor/configs/libs/weights 全部一致。按用户拍板走窄豁免（未留决策文档）：compatibility.py 的 direct_insert 校验加入仅 3 对已证明"仅行尾不同"的指纹放行表，runtime_id/weights/preprocessing/readout/output_dim/precision/position_strategy 仍严格比对；原版备份 compatibility.py.bak-pre-lineending-equivalence-20260922，未改动任何运行中任务的记录与数据。**2026-09-22 16:0x 更正**：本条原写"4 对"系计数错误，正确为三对（由 6 个 dense head 的 final.pt 元数据实推 code_digest：VMA-UCF `81e31440`↔`30fe33c1`、VMA-XD `9a23c5da`↔`30fe33c1`、TS 两数据集共用 `7f9a3ac6`↔`e5fd0bdf`；V2 两侧同为 `19720d4e` 原生一致）；18 路中 12 路走豁免、6 路原生匹配；远端放行表 5 个唯一摘要、XD 评分日志 0 条兼容报错。同一更正已同步 AGENTS.md、CLAUDE.md、STATUS_DATA_SUMMARY_20260922.md、PROMPT_FOR_ADVISORS_20260922.md。GPU 显存 monitor 已于 09:00 退出（541 行日志止于 01:00:13Z），显存峰值只覆盖 08:57–09:00 约 3 分钟，效率审计显存项不完整，建议对预测阶段补开。node2 本轮 SSH 连接超时，8 路补充提取未复核。新增文档：STATUS_DATA_SUMMARY_20260922.md（数据深度总结）、NEXT_STEPS_PLAN_20260922.md（收尾操作计划）、PROMPT_FOR_ADVISORS_20260922.md（外部顾问提示词）。正式分数未读取、未用于方法选择；失败目录全部保留；红线保持。

## 2026-09-22 09:12 最新 clip 现场

xtract_count=18 仍保持。当前 compressed clip records 约 1,046,227/2,582,304=40.5%，其中 UCF 541,896/832,743=65.1%、XD 504,331/1,749,561=28.9%；scheduler 六 dense jobs completed、exports=0，compressed contract 尚未生成。计数继续增长，未见失败。

## 2026-09-22 09:09 最新 clip 写入量

node3 正式 18 路仍有 xtract_count=18。当前已写 compressed clip records 约 1,026,830 / 2,582,304 = 39.8%：UCF 约 531,099/832,743=63.8%，XD 约 495,731/1,749,561=28.3%。clip 进度低于视频目录进度，符合 XD 长视频尾部的实际计算量特征。scheduler 六个 dense job 仍 completed、exports=0；尚无 compressed contract。

## 2026-09-22 09:18 最新闭合量

逐视频 clip-count 对齐复核：UCF V2 199/290 三路、VMA 195–199/290、TimeSformer 187–188/290；XD V2 338–353/800、VMA 338–364/800、TimeSformer 315–331/800。合并为 UCF 1753/2610=67.2%、XD 3075/7200=42.7%、全部 18 路 4828/9810=49.2%。已有目录均闭合且无 partial/extra，尚无正式 compressed contract 或 scoring/export。

## 2026-09-22 09:15 GPU monitor 汇总入口准备

新增并同步 scripts/icassp2027/summarize_urdmu_gpu_monitor.py，本地 compileall 与远端 classic runtime py_compile 均通过。它在 monitor 结束后按 PID/GPU 汇总显存采样的峰值、均值、采样区间和命令，并明确 observed_peak 只覆盖 monitor 启动后的观测窗口，不冒充完整进程峰值。当前不读取、不修改运行中的任务。

## 2026-09-22 09:05 最新闭合量

最新逐视频核对：UCF V2 188/290 三路、VMA 188–199/290、TS 159–187/290；XD V2 328–334/800、VMA 325–343/800、TS 297–313/800。汇总为 UCF 1666/2610=63.8%、XD 2917/7200=40.5%、全部 18 路 4583/9810=46.7%。已有目录均为完整 clip 覆盖，仍无 extraction contract 或 compressed scoring/export。

## 2026-09-22 09:02 最新闭合视频进度

逐视频 clip-count 对齐复核显示：UCF VideoMAEv2 三路均 188/290，VideoMAE 为 188–197/290，TimeSformer 为 155–178/290；XD VideoMAEv2 为 324–331/800，VideoMAE 为 322–335/800，TimeSformer 为 292–309/800。合并为 UCF 1641/2610=62.9%、XD 2880/7200=40.0%、全部 18 路 4521/9810=46.1%。所有已有目录仍为完整视频 clip 数，无 partial/extra；18 路仍未发布 extraction contract，compressed scoring/export 尚未启动。

## 2026-09-22 08:57 GPU 进程效率监视器启动

已在 node3 启动低开销 monitor（PPID=1，远端 PID 23427），日志为 /users/fotile/icassp2027-runs/codex-takeover-20260920/logs/urdmu-matrix-gpu-monitor-r08.csv。它每 20 秒记录 extraction、UR-DMU scoring 和 quality export 进程的 GPU UUID、PID、显存和命令，直到 scheduler/评分进程全部退出；不干预任务。用于后续六格矩阵的峰值显存、运行时段和吞吐审计，不能把 monitor 启动前的显存峰值写成已观测值。

## 2026-09-22 08:56 矩阵脚本远端同步验证

ssemble_urdmu_matrix.py 已同步至当前 node3 代码快照；本地/远端 SHA 均为 38384c4ea5175bbff696c5e8be314ef46366d862bc171eb2ff299b118d9837d，远端 classic runtime py_compile 通过。它尚未被 scheduler 调用，等待正式 compressed quality exports 完成后手动装配矩阵。

## 2026-09-22 08:53 六格矩阵装配入口完成

新增 scripts/icassp2027/assemble_urdmu_matrix.py，本地 compileall 通过。它要求 18 份 post-freeze quality export 全部 status=available 且覆盖 290/800 视频，核对 dense view、prediction、compressed extraction、head checkpoint/training QA、method contract 和 freeze receipt 的 SHA；并流式汇总每个 compressed FeatureStore 的实际 native/retained token、ratio、clip 数和 plugin overhead。缺证据会 fail-closed，不会把未完成 run 或测试分数缺失填成完成。待远端 exports 齐全后运行并生成正式矩阵。

## 2026-09-22 08:55 当前提取范围与闭环计划说明

正式压缩 test extraction 的范围不是重复训练视图：UCF sealed test 为 290 个视频、XD sealed test 为 800 个视频；三条 encoder × 两数据集 × 三个冻结方法（pair_select 主方法、group_uniform/group_random 同预算 controls）共 18 个 method runs，即 9810 个 encoder×dataset×method 视频实例，预计 2,582,304 个 compressed clip records。dense test views 已完整发布并复用，不再重复提取；六个 full-train head 也已完成 QA，不再重训。全量 290/800 是 paired video bootstrap 和官方 coverage 的必要条件，64+64 screen 或抽样子集不能替代正式 test。node2 的 8 个任务只是独立接力/故障备用，不扩展科学范围，也不覆盖 node3 正式目录。计划保持：node3 18 路 contract 完成 → 逐 run provenance/coverage audit → scheduler 自动启动 compressed direct_insert prediction → 18 份 paired quality export（UCF ROC-AUC、XD 梯形 AP 与 step AP、video-level、Δ/CI）→ 汇总实际 retained token/clip ratio、插件/端到端效率和六格矩阵；当前不安排 refit_head，除非主线闭环后仍有预算且独立补充所需。

## 2026-09-22 08:50 六格 dense head 训练指标与 QA 复核

已直接读取六个正式 head 的 	raining-steps.jsonl、
esult.json 和 	raining_qa.json。六格均正好 3000 steps，所有 loss 分量 finite，每步均为 64 normal + 64 abnormal bags，
onzero_gradient_parameters=34 且 QA 记录 
onzero_gradient_steps=3000；六格 	raining_qa.status=passed、strict reload missing/unexpected keys 为空、reload xact_equal=true、max_abs_difference=0.0，六个 official_model_scores_read=false。训练目标从约 1.11 起步并降到各自低位；VideoMAEv2×UCF 的尾部 loss 中位数约 0.0237、最终 0.0205，高于其他 head 且有少量有限尖峰，但没有 NaN/Inf、梯度或 reload 异常，先作为训练侧诊断记录，不据此读取测试或改方法。训练文件不产生正式 UCF/XD AUC/AP；正式质量指标仍只在冻结后的 evaluation 阶段导出。

## 2026-09-22 08:45 提取 ETA（按真实 clip 速率）

最新现场仍有 node3 18 路、node2 8 路 extraction 运行。node3 形式 run 按 clip 数约完成 29.0%（UCF 约 45.1%，XD 约 21.3%），按已闭合视频目录约 40.3%；XD 长视频尾部使 video% 不能直接换算完成时间。以各进程实际运行时长和当前 clip 写入速率外推，node3 正式压缩特征 extraction 大约在今日 14:30–16:00 HKT 完成，区间受剩余视频长度和共享 I/O 影响，暂不承诺单点时间。随后 scheduler 才能启动 18 路 compressed official_frame prediction；按 dense scoring 已见运行量，质量 export 还需数小时，预计直接质量闭环在傍晚至晚间完成，待第一批 extraction contract 发布后再收窄 ETA。

## 2026-09-22 08:35 按视频身份核对的压缩提取进度

远端按每个视频的 dense clip 数逐目录核对：当前已有 shard 目录的每个视频均已写满该视频应有的 clip，未见 partial/extra 身份。node3 18 路正式提取的完成比例为：UCF VideoMAEv2 50.3–52.1%（146–151/290），UCF VideoMAE 51.0–56.6%（148–164/290），UCF TimeSformer 39.7–44.1%（115–128/290）；XD VideoMAEv2 35.4–36.5%（283–292/800），XD VideoMAE 34.1–36.5%（273–292/800），XD TimeSformer 31.4–32.9%（251–263/800）。按三种方法合并统计，UCF 为 1285/2610=49.2%，XD 为 2490/7200=34.6%，六格全部合并为 3775/9810=38.5%。当前仍在继续提取，尚未发布 extraction contract，也未开始压缩质量评分。

## 2026-09-22 08:30 当前六格压缩评测状态

六格 dense UR-DMU `official_frame` prediction 已全部完成：UCF/XD × VideoMAEv2/VideoMAE/TimeSformer 均有 `result.json`、`predictions.jsonl` 和 query-chunk receipt，且 `official_frame_scores_read=false`。node3 上 18 个正式压缩 extraction 仍在写入 shard，覆盖三 encoder × 两数据集 × `pair_select/group_uniform/group_random@0.60`；当前尚无压缩 `result.json` 或 `extraction-contract.json`，因此质量 scheduler 尚未启动压缩 prediction 或 export。node2 另有 8 个独立补充 extraction（六格 `pair_select` 加 V2-UCF 两个同预算对照）持续产出，输出根与 node3 隔离。当前未读取正式 test 分数做方法选择；下一门仍是压缩 extraction 合同完成后自动评分并导出 UCF ROC-AUC、XD 梯形 AP/step AP 和 paired bootstrap CI。

## 2026-09-22 08:06 node2 prediction watcher

node2 补充 extraction 已启动独立 watcher `/users/fotile/icassp2027-runs/code-takeover-20260920/eval/launch_node2_official_matrix.py`（PPID=1）。它等待 node2 的完整 native extraction contract/status 后，在 node2 空闲卡上运行 `official_frame` direct_insert prediction，并将结果与已完成的六格 dense prediction 配对导出质量文件；不会覆盖 node3 的 output root，也不改变冻结方法。
## 2026-09-22 08:03 node2 额外提取接力

node3 上 18 个压缩 extraction 继续运行，未覆盖完整视图。为利用用户授权的 16 卡范围且不触碰 node3 现有目录，已在空闲 node2 V100 上启动 8 个独立输出的补充 extraction：三 encoder 的 UCF/XD `pair_select@0.60`，以及 VideoMAEv2 的 UCF `group_uniform/group_random`。输出根为 `/data2/localdisk/fotile-icassp2027-kimi-20260919-a01/compressed-test-node2`，run-id 带 `node2-r01`，每个任务使用独立日志和完整 sealed manifest/contract。node2 任务已开始写入 shard；它们只在完整 contract 发布后才可进入 scoring，不会覆盖 node3 结果。
## 2026-09-22 07:40 六格 dense official prediction 完成、压缩提取继续

远端 scheduler 已为六个 encoder×dataset 格子完成 `official_frame` 的 dense `direct_insert` UR-DMU prediction。六个 run 的 `result.json` 与 `resolved.json` 均为 `completed`，`official_frame_scores_read=false`，使用 exact all-key query-chunk attention（query chunk 256）；UCF/XD 的 prediction JSONL、query receipt、feature contract 和 head checkpoint provenance 均已落盘。当前没有任何正式质量分数用于方法选择。

三种冻结压缩方法的 18 个 sealed-test extraction 仍在 node3 以 PPID=1 运行：VideoMAEv2/VMA/TimeSformer × UCF/XD × `pair_select`, `group_uniform`, `group_random`。当前没有压缩 extraction `result.json`，所以 scheduler 尚未启动压缩 prediction，也尚未生成 paired quality export。其 detached scheduler 继续等待每个原生 extraction contract 完成；早先失败 run 保留为失败证据，不复用其目录。
## 2026-09-22 07:33 官方 prediction/压缩矩阵接力（未读分数）

六格 dense test view 已全部发布为 ready merged view：UCF 的 VideoMAEv2、VideoMAE、TimeSformer 与 XD accepted3950 的三条 encoder 均有完整目标列表和 merge contract；合并视图允许多个独立 shard fingerprint，但 representation/sampling 与行级 identity 仍由合同核验。六格正式 head 仍保持此前 3000 steps、64+64 bags、3000 nonzero-gradient、exact reload QA 通过。

方法已冻结在 `six-cell-method-freeze-v2`；兼容 official_frame gate 的 `method-freeze-contract-v2.json` 当前 SHA 为 `50dadbdd4a0201f275f75b4ab439de03504cd73bcfa594457208133bbe724bfc`，绑定 markdown、selection receipt、scope 和训练侧证据。没有 test 分数参与方法选择。

当前远端有 18 个独立压缩特征提取进程（3 encoder × 2 dataset × dense-free candidate/control 三方法），V2 使用 foundation + VideoMAEv2 overlay，VMA/TS 使用 classic 环境；UCF 修正后的数据根为 `UCF-Crime-official-verified`。早先环境/目录错误 run 和日志均保留，当前仍未生成质量分数。

官方 UR-DMU prediction scheduler 已 detached 运行，状态文件为 `/data2/localdisk/fotile-icassp2027-kimi-20260919-a01/urdmu-official-matrix-r08-state.json`。UCF V2/VMA dense prediction 已完成；UCF TS 与 XD 三个 dense prediction 正在运行。下一步由同一 scheduler 等待各压缩 extraction contract 完成，再运行 direct_insert prediction，并生成 UCF frame ROC-AUC、XD 梯形 PR-AUC 与单列 step average precision 的 paired video bootstrap export。当前未读取、未用于选择的官方测试分数保持空白。

本阶段新增评测实现：`urdmu_evaluation.py` 支持显式 SHA 绑定的 merged test view contract，并保留 native sealed extraction contract 路径；`urdmu_quality_export.py` 提供 XD 作者梯形 AP 与 separate average-precision 导出。受影响代码已 `compileall`；本机未运行 pytest，因为当前解释器没有 pytest 模块。
# ICASSP 2027 当前进度

## 2026-09-21 23:25 六格 dense 基线接力与 quarantine-aware head 修复

当前六格目标继续按 `videomaev2/videomae/timesformer × UCF/XD` 执行。UCF 三个
`official-fulltrain-final` 头已有 1610 videos、3000 steps、非零梯度和 reload QA 回执；XD
VideoMAEv2 accepted3950 合并视图已发布并通过合同验收：run-id
`xd-v2-full-20260921-r02`，3950 videos、1,018,483 clips、`complete=true`，合同 SHA
`d60438982d201a064cce9533e80cd19bdc79f70f2d942f66b379be4702cb3250`。VideoMAE×XD 与
TimeSformer×XD 的完整 merge 仍在 node3 运行，分别使用 accepted3950 的 68/135 个已审计逻辑来源，
合同尚未发布；不得把目录数量当作 ready。

VideoMAEv2×XD 第一次正式 head 在 23:11 因 quarantine-aware authority receipt 只从顶层读取
`accepted_members`，把 accepted3950 错判成 declared3954，保留为失败回执。已修复
`src/vadbench/paper/urdmu_training.py`：优先读取 `quarantine_exemption.accepted_members`，兼容旧的
顶层字段；本地受影响回归 `39 passed`，`compileall src tests` 通过，并已同步到远端代码快照。新的
独立重试输出 `training-full/videomaev2/xd-formal-dense-s0-r02` 已于 23:22:08 在 GPU2 启动，
现已完成 aggregation/训练和 QA：3950 videos、3000 steps、3000 nonzero-gradient、exact reload
（`max_abs_difference=0.0`），checkpoint SHA `35f320f3f34e3a82e5d7e2eef70e509104d7383e84dbe48ed47c2111fde20d99`，
training QA SHA `759e2f8b87e1bf2d16c4c1c1073c6f86fd81769d9f2badcea60ce55124904b32`。旧失败目录和日志保留，不覆盖。

三个 XD watcher 已改为读取各自 merge run-id 子目录的合同；VMA/TS ready 后会分别自动启动独立
formal head。当前尚未开始 XD test 评分；测试集只能在方法冻结合同、数据审计和 dense/压缩矩阵冻结后进入
evaluation。现阶段继续保留 accepted=3950/declared=3954/excluded=4 身份，不使用 screen600 或
64+64 结果代替正式 full-train。

新增 UR-DMU prediction-only 质量导出入口 `scripts/icassp2027/export_urdmu_quality.py` 与
`vadbench.paper.urdmu_quality_export`：它对 dense/method 两份完整预测 JSONL 和独立 truth JSONL 做
共同 frame partition，对 UCF 使用 frame ROC-AUC、对 XD 使用作者梯形 PR-AUC，并附 weighted step AP、
视频级辅助 ROC-AUC、10,000 次视频配对 bootstrap 和源文件 SHA。当前仅通过合成测试（2 passed）；
真实 test 导出仍等待六格 dense head、test truth seal、method-freeze-v2 和同覆盖压缩预测。

六格范围合同已登记为 `decisions/six-cell-scope-contract-v1.json`（SHA
`eac12d9248b4970e1d3528af30efc14a3a9d90ee7c7558965215b1cd9d6d7e25`）。它绑定六个格子、训练角色门、
accepted3950 身份、UR-DMU 预算和正式指标，但明确标记为 `active_scope_not_official_test_freeze`；
训练侧 selector/layer/budget 选择完成后仍需单独发布 method-freeze-v2。当前旧方法冻结文件之间存在
历史候选范围差异，不能直接冒充新的六格最终冻结合同。

为解决旧 evaluator 方法集合与当前三 encoder scope 的冲突，已登记新的六格方法冻结合同
`decisions/six-cell-method-freeze-v2.json`（当前 SHA `4874ce4fe3f82179d9bafaa6efe859125e21cdd3bd26a590827485d6cd3db658`）。
它在官方测试前冻结 dense、同预算 group_uniform/group_random、pair_select、0.60 budget、中间 verified
layer、direct_insert 主模式和可选 budget-matched refit；实际 retained ratio/geometry snap 仍必须由
每格 receipt 写出。该合同不读取 test 分数，也不替代 raw-coordinate/test audit 门。

训练侧选择回执为 `decisions/six-cell-method-selection-receipt-v2.json`（SHA
`260c9d26a80d5d796ab6cea2149b284c8f4f0bc1f224927135e69537e015f442`），固定 pair_select 与
group_uniform/group_random 同预算 0.60、中间 verified layer、seed0，并保留 pair-mean 和跨 encoder
小口径不复现证据。

VideoMAE×XD merge 已在 00:15 HKT 发布 ready：run-id `xd-vma-full-20260921-r02`，3950 videos、
1,018,483 clips，合同 SHA `5dca3e8b8d24aeb0f402ba0a9ead084fd575ad66915c8087e03ca2696bdf33cc`；对应
formal head watcher 已在 GPU3 启动。TimeSformer×XD 使用已补齐来源的 `xd-ts-full-20260922-r02`
继续合并；TimeSformer UCF test 290 视图也在独立合并，尚未进入评分。

VideoMAE×XD formal head 已通过完整 QA：`training-full/videomae/xd-formal-dense-s0`，3950 videos、
3000 steps、3000 nonzero-gradient、exact reload（`max_abs_difference=0.0`），checkpoint SHA
`4b273e634b61a9187ab8c64856f489ff97485b117742a249204df2cd57d1a90e`，training QA SHA
`ed05f79c965d51f31608e0743abd37a373fb6828244c65c11249233528401fa7`。当时已完成
5/6，随后 TimeSformer×XD 接力完成并写入下方 dense receipt。

TimeSformer×XD r03 merge 已于 05:24 HKT 发布 ready：3950 videos、2,038,777 clips、`complete=true`，
合同 SHA `3ed55c0560a62f08b4f111415054937a90a2654d8df71e987b45ef32c86048c9`。TS formal watcher 已
于 05:24:49 在 GPU4 启动，训练输出为 `training-full/timesformer/xd-formal-dense-s0`；至此六格
dense head 都已具备对应完整视图并进入/完成 head 链，TS 的 3000-step QA 已通过。

六格 dense receipt 已在远端生成并镜像到 `work/codex-takeover-20260920/heads/six-cell-dense-receipt-20260922.json`，
SHA `cf8b42828ffc4b0ff93f31b85b1bde66fd8464ee1daa7ea11f8e39b09a8e7a18`。该 receipt 已逐格验证 view
contract ready、video coverage、3000 steps、3000 nonzero gradients 和 exact reload；TS receipt 的
checkpoint/QA 也已纳入后续验收，当前下一门是 method-freeze-v2 后的压缩特征和 prediction export。

V2×XD 的冻结方法 test feature 生成已启动：foundation-video-v2 + cudnn-off + VideoMAEv2 overlay 环境，
`pair_select@0.60`、`group_uniform@0.60`、`group_random@0.60(seed0)` 三路独立 output/log，分别运行于
GPU2/5/6。前两次环境错误在模型加载前失败并保留；r03 已进入真实 encoder 前向，尚未读取 GT 或写质量分数。

TimeSformer×XD dense head 已完成 QA：`training-full/timesformer/xd-formal-dense-s0`，3950 videos、
3000 steps、3000 nonzero-gradient、exact reload（`max_abs_difference=0.0`），checkpoint SHA
`85586f8959cebfdf280cf777cf912f354ae1a54280392608347bf40c7e05cf69`，training QA SHA
`5f7e1fcc400ccbb9853fd71794e1ce73738d0c4ce84d5460d9d9d9c81cff7f75`。三 encoder×两 dataset 的
六格 dense baseline 现已全部通过 head QA；正式压缩和 test evaluation 仍未开始。

TimeSformer×XD 首次 merge 在 9/21 23:44 因部分 `nativerep` 分片当时尚未发布
`resolved.json/status.json` 而 fail-closed；随后只读复核确认当前 135 个选定逻辑来源均已
`status=completed`、`resolved/index/status` 齐全，缺口是时序而非数据身份问题。已用同一 accepted3950
video list、同一 authority/role-lock 合同重启 `xd-ts-full-20260922-r03`（此前 r02 脚本将 001–009
写成两位编号，已修复并保留旧失败日志）；TS watcher 已切换到该新 run-id 子目录，待合同发布后
自动启动 head。VideoMAE merge 仍在原
`xd-vma-full-20260921-r02` 运行。

为补齐 UCF 六格评测基础设施，已从 sealed UCF test manifest 生成 290-video ID 清单
`control/ucf-test-290-video-ids-20260922.txt`（SHA `8d9cb84d872e3df0fdc0211e0a8c210683daf5d243ea399fb9faca87fcf85dbb`），
并生成独立 frame-truth JSONL `control/ucf-test-frame-truth-20260922.jsonl`；未读取模型分数。TimeSformer
UCF test 的 dense slice（dense + slice01–05）正在合并到 `merged/timesformer/dense-test-view`，
合同发布后才进入冻结后的 scoring。

XD 的 800-video sealed raw-coordinate receipt 已只读获取并验证，使用 canonical metadata 与 GT 生成
`work/xd-evaluation-20260922/truth.jsonl`（800 videos，truth receipt 标记 `model_scores_read=false`，
output SHA `538e764805db263efcf72f356fd5e0150b6f7f9dfaa6908c8f77155a2b58e330`）。新增
`scripts/icassp2027/make_xd_truth_from_seal.py` 支持从 sealed receipt 生成 canonical prefix truth；
UR-DMU 质量导出合成测试仍为 `2 passed`，真实预测尚未读取。

TimeSformer UCF test merge 已发布 ready：290 videos、138,853 records、`complete=true`，合同 SHA
`5015cc388b1bd679c06464149877fb3c25cb715b8f5cea26a71c0645adf0ea44`。这只完成 test FeatureStore
基础设施，不代表已读取模型分数；正式 scoring 仍等待 method-freeze-v2 与统一 UR-DMU prediction
export 门通过。

## 2026-09-21 训练范围修正与 ADGS screen 收尾

用户明确：64 normal +64 anomalous 只用于训练集内部的压缩规则筛选；所有正式训练必须使用对应
encoder×dataset 的完整训练集。刚完成的 `screen600-retry32` 仅使用每个数据集 32 normal +32 anomalous
视频，UCF/XD 两个训练回执已通过，但保留为 training-side screen，不升级为正式结果。

screen 的压缩生成在产生有效结果前已停止。小样本数据入口的同设备硬链接核对 r02 已通过：UCF/XD
各抽查 1 个正常和 1 个异常视频，source/destination inode、SHA、features/pooled 数值和严格读取/200-bin
汇总均通过；回执为 `outputs/icassp2027/control/codex-takeover-20260920/idea/v2-adgs-screen-20260921/canary_receipt-r02.json`。

正式训练规则回执与时间估算见 [`TRAINING_SCOPE_AND_RUNTIME_ESTIMATE_20260921.md`](TRAINING_SCOPE_AND_RUNTIME_ESTIMATE_20260921.md)。
现有正式回执中 UCF VideoMAEv2 与 VideoMAE 已完成 full1610/3000 steps/QA；XD accepted3950 和
TimeSformer 的正式 full-train 仍需使用完整视图、合同和 QA，不得使用 screen600 模型替代。

## 2026-09-21 15:15–18:20 idea_screen_64x64（V2 ADGS 筛选）中点交接

按 `NEXT_PHASE_PLAN_20260921_V2.md` 执行。已完成：ADGS selector 与测试（本地+服务器快照各 16 passed）；
128 视频 screen manifest 锁定（sha f6f36fd6…，UCF/XD 各 32N+32A，seed 20260921，XD 逐视频特征完整）；
工程门 11/11 通过（identity/observer/token 数/attention 捕获/digest 确定性/缓存逐值一致，见
`outputs/icassp2027/control/codex-takeover-20260920/idea/v2-adgs-screen-20260921/gates_receipt.json`）。
未达成：dense head 600 步训练尚未产出——dense 缓存重打包为 native store 的初版太慢（逐 clip 重写约 2.2 clips/s），
已按用户指示改为 /data2 同设备硬链接 + 1N+1A canary 先行方案，代码改到一半。
环境教训：node3 系统 LD_LIBRARY_PATH 的 cudnn 9.5 会击碎 torch 2.8（须 env -u + cudnn-off hook + overlay PYTHONPATH）；
禁止 pkill -f 杀远程任务（会误杀自身 wrapper）。另如实披露：用户要求保留的 PID 28421 临时目录被我一次
`rm -rf heads` 误删。完整现场、路径、剩余改动与执行顺序见
[`HANDOFF_ADGS_SCREEN_20260921.md`](HANDOFF_ADGS_SCREEN_20260921.md)，交接 Codex 继续。

## 2026-09-21 08:00–10:01 idea 首轮：F09 工程门通过，尚未成为方法

提取健康先验：node2/node3 16卡持续运行；没有为了idea停提取。F09 reference-preserving时间候选已完成
CPU规则/真实前向smoke，使用两个固定的训练侧正常窗口（UCF、XD），没有test、没有检测分数、没有训练。
31个窄测试通过：V2/CLIP identity、真实序列长度、native坐标、T/U/R quota/tie/seed、CLIP prefix一次+
reference/target分离suffix、无跨帧attention、无padding回dense、无double-prefix，以及真实部署规则分支。

预训练CPU smoke结果：CLIP reference帧在T/U/R中的max-abs为0；V2 identity max-abs约2.4e-7/3.6e-7。
相对dense pooled MSE：CLIP UCF T/U/R=0.00290/0.00333/0.00263，XD=0.00164/0.00350/0.00320；
V2 UCF=0.00223/0.00717/0.00499，XD=0.00699/0.00729/0.00727。两窗口内V2的T优于U/R，CLIP方向跨数据集不一致；
这只支持接口和候选排序的下一步，不支持质量、不掉点或GPU净加速主张。CPU时间也不是正式速度测量。
完整receipt：`outputs/icassp2027/control/codex-takeover-20260920/idea/f09-cpu-engineering-r01-root.json`。

方法边界核查：FrameFusion已做相邻帧对应token相似合并再importance pruning；TempMe已做跨连续clip渐进合并；ToFu/MLERP已做最大范数平均校正；
Eventful Transformers已展示跨输入变化token gating和稀疏更新。因此“时间冗余优先”本身不是创新；能否形成WSVAD可辩护贡献取决于异常/正常匹配证据、
同预算基线、训练侧质量和实际端到端收益。下一步只保留T/U/R同路径训练侧小表；F07历史缓存因producer dispatch未生效仍不能作为成员范数因果证据。

### 05:29 简查

16卡持续增长，无空卡、重复writer或停滞；TS100与VMA52/53已自动接力并产出，无新增故障。

### 05:21 简查

16卡全部增长，无空卡、重复writer或停滞；TS098/099已自动接力且有产物，无新增故障。

### 05:11 简查

16卡全部增长，无空卡、重复writer或停滞；VMA50/51、TS095–097已正常接力并有产物。
本轮仅检查live run，不全量重扫、不派agent，无新增故障。

### 05:00 简查

16卡均持续增长，无空卡、重复writer或停滞；调度状态正常更新，无需额外通知。

### 04:50 简查

16卡全部增长，无空卡或重复writer；TS093/094已自动接力且有产物，无新增故障。

### 04:40 简查与覆盖更新

16卡均持续产出，无空卡或重复writer。node2 04:38:43–04:39:18去重覆盖：VMA train667,563/test0；
TS train1,313,912/test0。条件完成窗口仍为今天12:00–18:00，无新增故障。
回执`infra/direct-check-20260921-0439.json`；未改源码、未派agent、未中断任务。

### 04:30 简查

16卡均有writer，既有run全部增长；node3 GPU6刚切换VMA49，已对启动后的产出作补充核验。
无重复writer或新增故障，未全量重扫、未派agent。

### 04:20 简查

16卡全部增长，无空卡或重复writer；TS087–091已自动接力并有产物。无新增故障，未全量重扫或派agent。

### 04:10 简查：已登记恢复全部完成

16卡继续增长，无空卡或重复writer；四个暂停分片恢复及VMA19原生pt替代均已通过调度的目标/身份
校验并完成，资源已续派。整体数据集仍未提完，不将局部恢复完成写成全任务完成。
node2 04:09:43–04:10:09去重覆盖：VMA train635,907/test0；TS train1,236,212/test0。
条件完成窗口仍为今天12:00–18:00。回执`infra/direct-check-20260921-0410.json`。

### 04:00 简查

16卡均有增长，无空卡、重复writer或停滞；node2 GPU6已自动接VMA47并产出。
恢复队列状态稳定，无新增故障，未全量重扫或派agent。

### 03:50 简查

16卡全部增长，无空卡或重复writer；VMA28恢复已完成，释放资源已自动接上新分片。
VMA30恢复仍正常运行，其余已核恢复和原生pt替代均完成。无新增故障，不另发通知。

### 03:40 简查与覆盖更新

16卡均有增长，无空卡或重复writer。node2 03:39:13–03:39:35去重覆盖：VMA train603,879/test0；
TS train1,150,535/test0。条件完成时间仍在今天12:00–18:00，无新增故障。
回执`infra/direct-check-20260921-0339.json`；未重扫V2内容或中断提取。

### 03:31 简查

VMA19原生pt替代分片已完成，调度alias已更新，早期np试跑仍保留作历史。分片集中结束期间依次
观察到正常交接：TS079/080、VMA42/43均已新产出；其他run持续增长，无重复writer。
node2 GPU3随后也进入新一轮接力，未因交接瞬时无compute进程而重启正常任务。

### 03:21 简查

初次采样碰到node3 GPU2分片交接，随后复核已自动接TS078，480个NPZ已落盘，恢复16卡全有writer。
其他任务继续增长、无重复writer；这是正常交接，未手动重启任务或派agent。

### 03:11 简查与覆盖更新

16卡继续增长，无空卡或重复writer；VMA29恢复已完成并由调度接续其他分片，其余恢复正常。
node2 03:10:18–03:11:12文件位置去重：VMA train576,219/test0；TS train1,066,872/test0。
条件完成窗口仍为今天12:00–18:00，无新增故障。回执`infra/direct-check-20260921-0311.json`。

### 03:00 简查

16卡均有增长，无空卡、重复writer或停滞；node3 GPU2已接TS074并产出3688个NPZ。
调度状态持续更新，未全量重扫或派agent，无需额外通知。

### 02:50 简查

16卡均持续产出，无空卡、重复writer或零增长。TS070/071/072/073已自动接上完成分片释放的卡，
且均有新特征文件。未全量重扫、未派agent、无新增故障。

### 02:40 简查与覆盖更新

16卡全部持续增长，无空卡或重复writer。node2 02:39:43–02:40:03去重覆盖：VMA train541,516/test0，
剩余622,670；TS train1,004,222/test0，剩余1,326,336。相对02:10条件估计仍在今天12:00–18:00范围。
回执`infra/direct-check-20260921-0240.json`；覆盖不替代最终统一表征及内容验收，无新增故障。

### 02:29 简查

16卡继续产出，无空卡、重复writer或零增长。VMA28修正后的恢复已在node3 GPU7实际启动，
已有4696个NPZ；TS069已接上GPU2并产出。至此四个暂停分片的恢复均已进入执行或完成状态，
未提前宣称尚在运行的恢复已完成。无新增故障，不另发通知。

### 02:20 简查

16卡全部有writer且全部持续增长，无空卡、重复writer或零增长任务。v22状态更新至02:19:42，
恢复队列状态正常；未重扫全量、未派agent、未中断任务，无需额外通知。

### 02:10 直接简查与半小时覆盖更新

16卡均有真实writer且当前run全部增长，无空卡或重复writer。node2 02:09:47–02:10:09去重覆盖：
VMA train504,823/test0；TS train954,446/test0。两者仍含待最终身份验收的保留历史源，不能以文件
覆盖替代统一表征完成。相对01:34窗口条件估计仍在今天12:00–18:00范围，无新增故障。
回执：`infra/direct-check-20260921-0210.json`。本轮无源码修改、无新agent、无GPU任务中断。

### 02:01 简查

直接检查仍为16卡、无重复writer；同一run的14路都有增长，另两路已接TS067/068并有产出。
关闭VMA28上次Bash预启动失败遗留的非终结claim（日志与目录保留、未产生目标数据），已实际验证
该条恢复重新具备领取资格，不会永久停在queued；正常提取未中断。

### 01:54 直接简查

两台各8卡均有本项目extractor，未见同卡重复writer；当前16条run都有已有产物。VMA19 native-pt
和VMA29恢复继续增长，VMA30恢复已接上node2 GPU5；VMA28等待自然空卡。后续直接检查复用轻量
`direct_shared_watch_root.py`：读取实时GPU/进程，仅在node2本地统计当前live run并比较上次增长，
不做全量重扫、不派agent。正常无变化不另发通知。

### 01:49 恢复修正的实际产出回执

TS053已完成14967并发布调度alias。VMA29 pt-r02已产出4186个NPZ，encoder fingerprint与原partial
完全一致；VMA19 native-pt已有326个新NPZ，其representation与生产pt配置一致。VMA28/30仍排队，
未提前记完成。回执：`infra/recovery-nativept-verified-root-20260921-0149.json`。

## 2026-09-21 01:40 恢复入口兼容性与完成判定修正

直接复核发现服务器旧Bash在nounset模式下把空PROCESSOR_ARGS数组视为未绑定，VMA28的r02启动
因此在GPU前退出；已改为只向非空视频参数数组追加TS所需参数，VMA不加该参数。失败保留，未动已有
正常writer。同时修正恢复条目未继承顶层protocol/training-contract SHA导致TS053虽完整却无法通过
调度验收的问题：实际14967条TS053目标已在新校验器下通过，不绕过SHA检查。
v22已运行于node2 PID120158，SHA`2227913ae25c7013eb3324eb713012a2434cbe3d63db762305f0bd0eea4e7dd6`；
新增退出日志优先于残留claim、僵尸PID不当作运行进程的处理。行为回归、实际TS053契约校验、编译通过。
控制器切换只影响派工，不停止正常提取；新的VMA恢复和019原生pt替代继续按空卡调度，尚未提前记完成。

## 2026-09-21 01:34 简查与一致性补齐

v21状态更新至01:32:18，TS053恢复继续运行，VMA28/29/30修正后的pt恢复排队等待自然空卡。
01:33:38–01:34:13文件位置去重覆盖：VMA train465,016/test0；TS train891,212/test0。
该覆盖数尚不等于全量身份验收完成，物理重复与失败目录仍保留、不重复计数。

小范围resolved核验确认早期A100 VMA019试跑使用np表征，而原有VMA25与新共享VMA32均使用默认pt，
representation fingerprint不同。未将两者强行视为等价，也未改写旧fingerprint：已将019加入独立
native-pt完整重提队列（14,955 clips，旧试跑保留），在自然空卡上执行。最终VMA数据验收必须使用
该统一配置的替代源，不能仅凭旧文件覆盖宣布完成。其他正常分片继续，未重扫V2、未派agent。

## 2026-09-21 01:21–01:30 直接修复恢复配置，正常分片继续

发现VMA28/29/30恢复任务failed_exit且0条产出：旧partial为原生默认pt预处理，恢复脚本强加np，
严格encoder fingerprint校验正确拒绝复用。失败目标和旧partial均保留，未改写任何指纹。
先将3条失败恢复从活动预约隔离，释放卡自动接上普通VMA37/38/39；其余任务继续。
主控部署v21（node2 PID93097，SHA`ed556f27466db5b583d9b5c4150e628b812f5b474662bf557ad196ac685d9f67`）：
VMA恢复不再注入processor参数，沿旧默认pt；TS保持np。三个恢复进入新的独立pt-r02目标，继续严格
验证source SHA/契约，等待自然空卡。并修正failed结果被未改名claim遮蔽造成的预约占用，启动时接续
已运行的TS053恢复，控制器切换没有停止extractor。相关失败预约回归及compileall通过。
这三条恢复修正尚待实际执行通过，不能提前记为完成；16卡按普通与恢复分片继续续派。

## 2026-09-21 01:10 简查：自动接力已实际发生

主控直接核对v20 PID48090状态更新至01:08:15：TS056正常完成14,994 clips，node3 GPU4已自动
领取TS053独立恢复run，01:07:40启动且producer_seen=true。其余15条已核run相对01:03均继续增长，
没有重复派agent或全量重扫。VMA三条partial保持恢复队列，正常提取与后继按统一调度继续。

## 2026-09-21 01:03 实测：16/16 GPU 已提取并持续产出

主控直接安装并启动v20共享调度，node2 PID48090，源码SHA
`6e4ae835c5978a8662c243252da9608b75645d0fef30cd1730505f2aad8d0ae4`。
按用户最新授权共享空余显存，不停止其他项目；逐卡排除本项目已有writer/等待器，避免重复堆任务。

| 节点 | VideoMAE | TimeSformer | 本项目工作卡数 |
|---|---|---|---:|
| node2 / V100 | GPU2、4、6、7 | GPU0、1、3、5 | 8 |
| node3 / A100 | GPU0、1、2、5、6 | GPU3、4、7 | 8 |

node2统一计数窗口01:01:59–01:03:01，16个对应run全部有正向NPZ增长，实测约61.90秒；
合计VMA约16.93 clips/s、TS约50.60 clips/s。这是共享生产短窗，不是论文正式计时或稳定吞吐保证。
01:03:01–01:03:28全量去重：VMA train437,525/test0，剩余726,661；TS train822,145/test0，
剩余1,508,413。按短窗粗推VMA约12小时、TS约8.3小时，整体仍按今天12:00–18:00条件窗口规划。
V2已完成覆盖与内容审计。服务器每30秒检查续派，分片完成后立即领取下一片，直到train/test齐备。

共享普通launcher使用4096MiB空闲门槛和受限CUDA fraction（V1000.12、A1000.2）；现有正常任务
保持原配置。四个v18启动失败claim已保留并补终结后缀，恢复队列不会被它们永久挡住。
v19恢复行为窄测试及本地`compileall -q src tests`通过；已核实际16卡GPU进程和两次产物增长。
完整独立回执：`infra/shared16-verified-root-20260921-0103.json`。
当前没有另派执行agent，后续主控直接简查；自动head禁用，明早08:00研究安排不改变提取优先级。

## 2026-09-21 最新授权：16 GPU 共享提取

用户在了解其他项目占用后再次明确“直接16张卡开始提取”。当前允许node2/node3全卡共享空余显存，
不停止其他项目，已有正常writer保留。00:40附近主控实测我方7卡均为VMA（node2 GPU1/4/5；
node3 GPU0/1/2/6），TS暂无活跃writer，node3 GPU4空闲；其余卡由其他项目使用。
因此旧的下午完成估计需按新部署后的实际吞吐重算。当前执行者只收尾恢复安全修复及16卡实际上线，
后续主控直接简查、不例行再派agent。此条是用户最新范围，不代表16卡已全部启动成功。

## 2026-09-21 00:37 最新巡检方式：主控直接简查

用户要求不再每轮派agent、缩短prompt。唯一心跳已更新为简短提示，每10分钟由主控直接检查，
明早08:00 idea保留。当前执行代理仅收尾已经开始的恢复安全修复，不再派新巡检。
root直接读取调度状态00:35:24：PID33082仍在更新，V2 78/78完成；TS train52片done、
VMA train22片done，其余按实际producer/覆盖核验，unknown不等于运行成功。
最新完整clip计数仍为00:22：VMA407,669/1,164,186，TS800,871/2,330,558，两个test仍0。

## 2026-09-21 00:31 巡检：恢复配置已接入，首次派工前修正验证缺口

已读 `infra/takeover_infra_receipt_20260921T003148Z.json`：v18 PID33082加载四条显式恢复配置，
当前4 queued/0 inflight，没有获准空卡。root独立审查发现恢复启动预约未纳入跨循环slot保留、
失败固定等待24小时，以及launcher读取顶层契约字段/TS53空列分隔的问题，已要求唯一代理在首次
恢复执行前修正并补慢启动、重复claim、失败回收及四条命令参数的行为验证。不能将“配置已加载”
写成“恢复运行已成功”。既有正常提取保持，不新开其他代理。

node2新focused计数窗口00:22:02–00:22:18：VMA train407,669/test0，剩余756,517；
TS train800,871/test0，剩余1,529,687。相对23:43窗口速率约18.99/24.89 clips/s，
条件剩余约11.1/17.1小时；新恢复任务尚未计作已完成，物理重复不计unique进度。

## 2026-09-21 00:22 里程碑：V2 全量内容审计通过

root独立读取node2审计日志的终结JSON：00:17:48 completed，78个FeatureStore、1,164,186条记录/
bundle、2,328,372个数组成员全部检查，issues为空；train1,018,483/test145,703与已核覆盖一致。
校验包含FeatureStore加载/完整性、shape/dtype/nbytes、数值有限及index行数一致性，耗时约35.3分钟。
单独的`.final.json`文件未生成，但日志末行有完整终结JSON且进程已退出，不能继续等待不存在的输出。
独立回执：`infra/v2-content-audit-final-root-verified-20260921.json`。这是特征完整性通过，不是检测
质量结论，不启动新head。当前唯一代理继续完成VMA28/29/30与TS53的可执行恢复队列及VMA/TS新计数。

## 2026-09-21 00:12 巡检：资源边界修复已核实

已读 `infra/takeover_infra_receipt_20260921T000804Z.json`，错误node3调度实例残留的新任务是四个
违规writer来源；仅停止本项目 VMA30/29/28 与 TS53 的确认launcher/child，原partial保留。
root本轮独立NVIDIA查询已确认四个child PID13274/13404/13362/13561消失，外部任务及原VMA22/24
仍在。node2现唯一v17 PID15306，SHA `2ed6ad52da24d16e32cc3e36389006b90d3111d2c2b57cd30795bb979726cbeb`，
带ibnode2启动门禁，自动head入口已禁用。alias身份保持。下一轮需把四个partial纳入合法恢复队列，
不得永久跳过；仅使用获准自然空卡，不在外部占用卡上重开。
V2低优先级内容审计00:08:04已到790,000条，日志暂未见问题，最终结果尚未发布。

## 2026-09-21 00:02 巡检：资源归属与调度范围修复进行中

root 独立读取 node3 GPU 列表（该节点时钟为9/20 23:53:23，勿与本机时间混算）：在既有外部
占用的GPU7/5上分别出现新增PID13274/13404，GPU3/4既有VMA及外部占用旁新增13362/13561。
已要求唯一执行代理核实项目归属、启动先后和launcher来源，仅停止确认违规新派入的本项目进程，
保留外部任务与先前正常producer。候选根因为此前误在node3启动的dispatcher残留子进程，尚待回执确证。
root源码复核显示v16的free_gpus按所有compute UUID排除忙卡，不能未经核实改成仅按显存放行。
同时发现v16仍含auto-head调用，与当前范围不符；下一最小版本须加入本机ibnode2启动硬门禁并禁用
本阶段自动head派工。当前修复尚未验收，不能写成已经完成。
另要求fresh读取V2内容审计，前两轮60000是23:44旧观测，不能当作持续进展。

## 2026-09-20 23:50 巡检：VMA/TS 覆盖更新，V2 内容审计运行中

规范回执 `infra/takeover_infra_receipt_20260920T154425Z.json`：node2 23:43:26–23:43:41计数，
VMA train363,646/test0，总剩余800,540；TS train743,190/test0，总剩余1,587,368。
物理重复文件分别2,218/61,532，均不计入unique完成量。相对22:54同机窗口吞吐约14.04/29.08
clips/s，条件剩余约15.84/15.13小时（从23:43起算），没有把新卡短窗当长期保证。
V2覆盖不再重扫；内容审计在node2以foundation解释器、单CPU、nice19/ionice3运行，PID121809，
23:44:25已检查60,000条，验证SHA、shape/dtype/nbytes和finite，最终结果仍待完成。
首次错误默认解释器在导入阶段即失败且未读数据，失败保留，已换既有正确解释器继续。
本轮唯一代理正在核逐卡真实进程/空卡接力，含TS48完成后的后继；不新增head/idea或其他代理。

## 2026-09-20 23:41 里程碑：V2 XD 提取写完，覆盖验收通过

已核 `infra/takeover_infra_receipt_20260920T153748Z.json`：node2 本地23:37:31–23:37:48验收，
train 3950/3950视频、1,018,483/1,018,483 clips；test 800/800视频、145,703/145,703 clips。
物理NPZ与唯一clip数一致，缺失视频与非完成run列表为空；末片00 result/status completed，index
与records均16,224、failures为空。该步骤尚未做全数组shape/finite深查，不能写成全部内容审计通过；
后续先复用既有检查回执，确有缺口才安排低优先级内容审计，不重提、不影响主队列。

node2唯一v16 dispatcher PID99341继续TS/VMA接力；错误残留node3 dispatcher PID24817已停止，
正常extractor保留。VMA019与TS025均通过调度专属alias映射到真实完整run，row为done，原partial
与成功数据不变，alias不是merge输入。后续巡检不再重复全扫已确认的V2覆盖。

## 2026-09-20 23:31 巡检：V2 作业收尾，调度 alias 已隔离

已读 `infra/takeover_infra_receipt_20260920T152911Z.json`：node2 v16 alias dispatcher PID99341，
源码 SHA `2dfcfd4512f25d5e5ff5ed53302fe7144fbe40800850c0299f364d98d6dd71c3`，23:29:11 状态为
V2 78/78 done、无running/unclaimed。正在做 V2 最终 unique/video 覆盖和末片契约一致性核验，
尚不以调度状态替代最终数据验收。
VMA19 的兼容标记已移出提取数据树，改为调度专用 `dispatcher-run-aliases.json`，绑定实际019的
manifest/result/status/resolved/index/contract SHA，并明确不是 feature alias 或 merge 输入。
正常writers未停止。下一步把同一机制用于已完整恢复的TS025，消除队列遗留partial状态，保留原partial。

## 2026-09-20 23:21 巡检：新增 A100 接力与全量计数

规范回执 `infra/takeover_infra_receipt_20260920T150000Z.json` 的 node2 本地计数窗口为
22:53:39–22:54:39（该主机时间），按 video_id/clip_index 去重并包含 TS025 独立恢复 run：
V2 train 1,014,057 + test 145,703，剩余 4,426；VMA train 322,319 + test 0，剩余 841,867；
TS train 657,476 + test 0，剩余 1,673,082。该快照不是 23:21 的即时计数。
后续 dispatcher 23:10:51 为 V2 77/78 done，仅 train0 在跑。新增 A100 VMA25/GPU0、
TS48/GPU2、VMA26/GPU6 均已核实际 extractor 和产出；外部进程保持保护。

发现 VMA019/19 零填充命名差异导致重复领取，已停止确认属于本项目的重复 producer，保留成功019
及重复尝试证据，调度 row 已转 done。当前兼容 alias 位于19/status.json；下一轮正在核验它的
调度专属身份及下游合并隔离，不能把无features的alias当独立成功提取结果。
VMA 慢速存在 CPU/GPU 资源竞争线索，尚不能单凭外部进程存在确证因果。按上一小时混合资源吞吐
条件外推 VMA/TS 尚需约16.5/18.6小时（从22:54起算）；新增卡的短窗速率另列，未混入该估计。

## 2026-09-20 23:00 巡检：V2 收尾与吞吐复核

上轮规范回执 `infra/takeover_infra_receipt_20260920T144301Z.json` 记录 dispatcher 74/78 done、
4 个 V2 分片仍在跑、unclaimed=0；尾片局部计数剩余 12,875，不替代全量去重验收。
VMA019 已核 completed、14,955 条，TS038 成功回执保留。VMA22/24 在同机 574 秒窗口仅约
1.92/1.93 clips/s，低于先前 VMA019 约 5.4 的单片粗估；不能继续直接按旧速率线性外推。
本轮唯一代理正在 node2 本地更新全量去重覆盖、核查 launcher/CPU/IO 差异，并依据剩余量和实际吞吐
调整自然空卡接力。此前 9/21 12:00–18:00 仍是有条件规划窗口，待新资源配置下的总量/速率复核。

## 2026-09-20 22:50 巡检：TS025 缺口已补齐

已核对 `infra/takeover_infra_receipt_20260920T143327Z.json`：TS025 独立恢复 run 的 result/status
均为 completed，index、NPZ 与目标均为 14,998；复用旧完整视频 2,649 clips，新解码 12,349 clips。
原 partial 未覆盖，attempt01 与 canonical done 回执已发布。最终聚合需显式包含该恢复 run，按 clip
去重旧 partial，不可因恢复目录名称不同而遗漏。VMA22/24 持续增长，未进入 pending，未见重复领取。
唯一执行代理已开始下一轮低开销巡检及适时总覆盖更新。共享/独占卡状态须按当前 GPU 进程映射，
不沿用旧资源描述；当前报告不等于三模型全量验收完成。

## 2026-09-20 22:40 巡检：A100 两个接力已核实

VMA22（node3 GPU4，extractor24797）与 VMA24（GPU3，extractor27499）真实提取已启动，
上轮回执分别记录 2233/2007 NPZ；root 本轮独立 NVIDIA 查询仍见两进程各占 3120MiB。
此前 `producer_seen=false` 属 canonical/compat 跨节点观察差异，不能据此杀掉正常 producer。
TS025 独立恢复 run 的 attempt01 上轮为 9451/14998；本轮原 extractor 已退出，正在核验最终成功/失败
回执，尚不标完成。原 partial、失败 attempt00 均保留。VMA019 也正在核验结束状态。
唯一执行代理已复用开展本轮检查，没有新增并行代理。规范回执目录为 `outputs/icassp2027/control/codex-takeover-20260920/infra/`。

## 最新巡检补充：同一执行 agent 每 10 分钟核查接力

用户要求确保无缝接力。已将唯一心跳 `icassp` 改为每小时 00/10/20/30/40/50 分唤醒，复用
`takeover_infra_dispatch`，不新建并行代理。主控已立即唤醒一轮：核查 node3 VMA22/GPU4、VMA24/GPU3
在 dispatcher 中 `producer_seen=false` 的实际状态，以及 TS025 resume；claim 不算启动成功。
刚读到 dispatcher 状态更新时间 22:25:56、PID45240；V2 72/78 done、6 running、unclaimed=0，
node2 有新分片续派记录。上述为状态回执，A100 两个后继仍待实际进程/增长核实。
明早 08:00 idea 保留，与巡检顺序安排，提取故障优先。

## 2026-09-20 22:26 用户补充：明早开始有界 idea 探索

今晚仍集中提取，9/21 08:00（香港时间）起按当前计划第 10 节启动 idea 探索；默认只用一名执行代理。
唯一心跳 `icassp` 已调整为整点/半点检查，08:00 首次到时开始，应用不可运行时在恢复后补执行。
复用已有证据，单方向长时间无进展则查原始论文与官方实现的可证实局限；目标为真实加速和质量保持，
必要时加入同预算基线，正式 test 不用于挑方法。第 10 节规定阶段止损与中午结论，不承诺必有正结果。
这条新授权覆盖下文较早的“提取完成后等待研究指令”，其他暂停的打包/写作任务保持暂停。

## 2026-09-20 22:05 当前范围：仅特征提取

22:12 ETA 更新：实际 A100 VMA 为 shard-019（此前 018 启动因已有 claim 被拒绝），单片首末文件跨度粗算约 5.4 clips/s；TS038 约 8.7 clips/s。按 V2 今晚收尾、约 7 张获准 A100 及时接力且 V100/CPU/存储保持有效吞吐，三模型 XD train+test 暂估 9/21 12:00–18:00 写完，校验另留 1–3 小时。此为有条件规划，非完成回执；精确采集/时钟边界见 `infra/eta-planning-20260920-2213.json`。

用户要求收拢并按新文档执行。唯一入口：
[`FEATURE_EXTRACTION_PLAN_20260920.md`](FEATURE_EXTRACTION_PLAN_20260920.md)。
只保留一个提取调度代理，其他辅助任务已中断；node2/node3 正常提取继续，node1 不参与。
VMA 新 claim 正常继续。新 head/idea/打包不再自动启动；既有 TS 特征合并可自然结束。

上一轮约 21:57 收到的 unique clip 快照（精确采集时刻待 infra 回执补齐，非本次重新扫描）：
V2 train 898,328 + test 145,703 = 1,044,031 / 1,164,186（89.7%）；
VMA train 270,579 + test 0 = 270,579 / 1,164,186（23.2%）；
TS train 566,671 + test 0 = 566,671 / 2,330,558（24.3%）。
TS025 partial 仍须合法补齐；当前计数不替代最终身份、覆盖和文件完整性验收。
本次仅编写执行文档、同步入口和进度，不修改提取源码，不重跑 Python 测试。

以下内容均按其原始时间阅读，不得覆盖本节的最新范围。

> ## 当前状态与回执导航（2026-09-20 18:14 现场快照；18:37 后续诊断待处理）
>
> 当前唯一执行计划：[`RESEARCH_PLAN_V3.md`](RESEARCH_PLAN_V3.md)。
> [`RESEARCH_PLAN_V2.md`](RESEARCH_PLAN_V2.md) 已标记 superseded，正文和历史回执保留；本文件
> 下方旧段落也全部按各自采集时点阅读，不能覆盖 V3 的当前范围。
>
> 现场事实来源：root 18:06–18:14 本地 SSH 工具汇总回执
> [`live-audit-1814.json`](../../outputs/icassp2027/control/codex-takeover-20260920/audit/live-audit-1814.json)
> / [`live-audit-1814.md`](../../outputs/icassp2027/control/codex-takeover-20260920/audit/live-audit-1814.md)。
> 桌面 `KimiCode_交接Codex_状态报告_20260920.md`、`KimiCode_提取核查与后续执行安排_20260920.md`
> 和 `04-投稿冲刺执行计划-20260920.md` 仅作为较早历史来源；18:14 快照不是持续运行承诺。
> 新状态以 root/infra 后续 receipt、PID/owner/argv/startticks 和两次有效产物增长复核为准。
>
> ### 当前执行范围
>
> - 三个 dense encoder 固定为 VideoMAEv2、VideoMAE、TimeSformer；UCF/XD 各自完成 dense
>   提取和匹配 UR-DMU head，任一 encoder/dataset ready 即独立进入下一步，不设全体完成屏障。
> - V-JEPA 2 已退出检测实验；权重、缓存、观察和失败/退出证据保留，不删除、不作为完成条件。
> - 当前主适配要求为 VideoMAEv2 + 一个核实过 backbone/backend 的 CLIP，训练侧探索是论文主
>   方法线。三条候选路线均为 ViT-B/16，共享 visual 不能写成三个 encoder；主路线暂定
>   VadCLIP（两数据集权重已就绪，仍须兼容性核验），DSANet 可选，AnomalyCLIP 新增成绩暂停。
>   CLIP 原生 10-crop 与 OpenAI center 输入先做训练侧对齐，不能写成已通过；可选第二 CLIP
>   才是补充。双数据集训练侧探索由 Astra 专职，常规执行由 Luna 最高强度完成，root 负责规划、
>   判断和验证。
> - 旧 pairselect 预登记属于历史；V2 四主臂与 V2 1b fixed-member/random-member/reverse
>   继续用于范数/成员偏好机制审计。V2 1b 只隔离 pair 内成员偏好，不证明 pair 间排序。VMA
>   pair/TS random 已真实 SIGTERM 停止且无复发，VMA random/TS uniform/pair 自然完成的原身份保留；
>   已有 partial/结果/负例全部保留。
>
> ### 当前快照（20:04–20:16 本机时间；独立于 18:14 历史表）
>
> - XD 本轮接受训练集合为 `accepted_train=3950/3954`，excluded=4；quarantine 视图不可变，用户已要求停止搜索、下载、补 4 和等待新的 3954 契约。三 encoder 的 XD head/质量链不再被四文件阻塞；旧修复失败和排除历史继续保留。
> - UCF V2/VMA formal `official-fulltrain-final` 头均已完成 1610 videos、3000 steps、3000 nonzero-gradient steps，training QA passed、exact reload parity 通过。当前只进入 QA/评分链准备，尚未产生新的 formal test 分数。
> - TS fit/confirm/authoritative old select union 已核到 1610，未重提 149；旧 select 权威路径为 `/users/fotile/icassp2027-runs/code-2d9d1a0/outputs/icassp2027/control/development-baseline-20260918-a01/jobs/timesformer-select/extractions/dense/features/train`。较早 full merge PID8255 因未传 role_lock SHA 在 publish 前失败；当前 retry01 PID29202 正在 `/merged-full/timesformer/dense-full-view` 发布，TS formal head 尚未启动。当前 heads 回执在 `work/codex-takeover-20260920/heads/heads_receipt-v3.json`。
> - merge/training path/fulltrain-choice 修复后受影响回归 **101 passed**，compileall 通过；这是工程验证，不是模型质量结论。
> - node2 TS/VMA 继续生产，TS 已从 shard026 接力到 030 共 5 路并有真实产物增长；VMA 暂缓领取新分片，已有片自然完成。node3 V2 v11 回执为 48 done、8 running、22 unclaimed；A100 空卡接力逻辑仅在 fake 场景通过，实际未来接力尚未触发。v11 源身份以 `outputs/icassp2027/control/codex-takeover-20260920/infra/v11_runtime_receipt.json` 的 dispatcher source SHA `877bb434722b82b908b0955730959436ac78e037b75fb634ff44b309b9cc4e64` 为准，不沿用旧 v8 摘要。
> - node1 状态为 `user_deferred_admin_recovery`：20:17–18 Tailscale gateway ping 11ms、node2 SSH 成功，只有 node1 经 Tailscale banner 超时；20:31 关闭 Tailscale 后校园网 node1 SSH 成功。node1 CUDA Driver API 已枚举 8 卡，但 NVML/nvidia-smi 仍超时；SSH/CUDA 枚举不等于 GPU 可生产。有效资源是 node2 健康卡和 node3，不能承诺 24 GPU 全部运行。receipt：`work/codex-takeover-20260920/network/tailscale-ssh-check-20260920.md`。
> - VMA diagnosis receipt 未发现 invalidating implementation bug，但 pooled temporal lag-1 `0.9994–0.99998` 可复现，保留质量风险；token-axis `0.827` 不能反驳 pooled 风险。QA 通过不等于质量通过，不能据旧 test `0.8106` 断定非 bug。已有 VMA 分片自然完成，暂缓新分片领取，不取消三 encoder 目标。
> - 旧 VMA pair/TS random 已按 `user_scope_change` 真实 SIGTERM 停止且无复发；VMA random/TS uniform/pair 自然完成的原身份保留；V2 有效消融保留。停止回执：`work/codex-takeover-20260920/retire/retire_receipt_20260920.json`。
> - CLIP float32 engineering smoke 已验证 `197→99`，但原生 VadCLIP 10-crop 与 OpenAI center 表征未对齐，质量入口 A/B 未冻结；不写 quality claim。F06-S1 的 V2+CLIP 16-window 已完成，未支持稳定主信号，停止扩量但不停止 idea。F07 CPU 100视频×7臂 norm audit 进行中；F08 仅 ToFu/MLERP 既有尺度对照、未派 GPU；F09 reference-preserving v1 已由 root 派 CPU/config/identity 准备，未授权 GPU。
> - heartbeat `icassp` 只在实质变化时通知，不新增 feeder；普通执行 Luna Max、idea Astra Medium。当前 head ready 先 QA/准备评分链，正式 test 评分等 idea 收敛及方法/矩阵冻结后统一进行。

> ### 20:38 fresh progress 与 9/20–9/23 关键路径
>
> - fresh 去重：V2 `849950/1164186 = 73.0%`，VMA `238412/1164186 = 20.5%`，TS `466999/2330558 = 20.0%`。从 20:38 起算的旧 ETA（V2 约2h、VMA 约41.5h、TS 约38.6h）已经过期，等待 infra fresh snapshot；不能用 V2 速率替代 VMA/TS。
> - dispatcher v15 实际 SHA `176f014f56e2a2083ed7b50f88c5e3bf95e7e029c8122be3bf07c3f47b8b6ec3`、PID `45240`，receipt 在 `outputs/icassp2027/control/codex-takeover-20260920/infra/takeover_dispatch_receipt.json`。node2/node3 继续，node1 不进入生产资源计划。
> - node3 A100 GPU0 与 V2 共享 TS038 的真实短试验为 347s、TS 约8.8 clips/s；不是正式计时，不外推所有 A100 或 VMA。释放 A100 后立即 relay，但下一 encoder 必须有自己的真实 successor receipt。
> - VMA diagnosis receipt `outputs/icassp2027/control/codex-takeover-20260920/infra/videomae_quality_resume_receipt.json`：未发现 invalidating implementation bug，pooled temporal lag-1 `0.9994–0.99998` 可复现但不足以否定 readout；token-axis `0.827` 不能反驳 pooled 风险。新 claim 已恢复，质量风险保留。
> - F06-S1 的 V2+CLIP 16-window 已完成；未支持稳定主信号，停止 F06 扩量但不停止 idea。F07 CPU 100×36,470×7 已完整审完，结果目录 `outputs/icassp2027/idea/f07`；但 pair_fixed/random_member/reverse pooled 逐值完全相同，`causal_member_preference_status=blocked_cache_identity`，不能写成员范数偏好正面，N 阶段不批准；Astra 查参数/控制且不重跑100。F08 仅 ToFu/MLERP 既有尺度对照、未派 GPU；F09 reference-preserving v1 继续 T/U/R CPU/config，未冻结、未启动 GPU。
> - 9/21 08:00 做严格方法 Go/NoGo；9/21 白天若 Go 才冻结 V2+CLIP 方法/层/预算/矩阵，9/22 做 formal test/CI/独占效率，9/23 内部冻结。CFP 页面显示 9/23，CMS 页面显示 HKT 9/24 20:00，旧 PaperKit 显示 9/16；内部截止不放宽。
> - 论文格式 scaffold 在 `projects/icassp2027/manuscript-20260920/`，目标 4 技术页，第 5 页按规则放 references/资助/ethics；single-anonymous、ORCID、作者核查和 AI 禁止生成重要/大部分/完整 section 的约束保留。

> ### 18:14 现场事实
>
> | 线 | 当前快照 | 处理 |
> |---|---|---|
> | node1 | 无本轮提取；GPU 空；GPU7 有历史 Xid48/63 隔离 | 修复健康槽位并以两次有效增长验收，GPU7 继续隔离 |
> | node2 | 7 路 VMA；TS 为 0 extract；guard `125984` 等待 GPU7 formal-timing 独占并撞 VMA | 不抢 VMA；确认冗余后按 receipt 停/暂停，正式计时另排独占窗口 |
> | node3 | 8 路 V2 XD、4 路 UCF batch；fullformal CPU drivers `26801/26807` 约 20 GB RSS，未完成 | 核实真实增长，不把 driver 存活写成 head 完成 |
> | XD V2 | `496474/1164186 = 42.65%`；test800 已齐 | 可先准备 V2 训练视图和测试审计，不等其他 encoder |
> | XD VMA | `199145/1164186 = 17.11%`；test 未开始 | 继续训练片段，ready 后自动进入 test |
> | XD TS | `325387/2330558 = 13.96%`；两分钟无增长 | 先恢复 launcher/guard/队列；物理 `384270` 含 `58883` 重复，不能作进度 |
> | UCF V2/VMA | merged-full 1610 回执显示 completed，records 字段 `790271` | 仍须成员覆盖、身份和 consumer QA |
> | UCF TS | merged fit1288 完成；train 路径立即失败 | 修路径后单独补 full1610，不能升级为完成 |
> | 阶段性 test 诊断 | V2 `0.8509978553`、VMA `0.8106398037`，generic、dev1288 head | Arson011 截断字段阻塞 v13 official audit；已看 test 必须披露，不能再据 test 调方法 |
> | batch1/1b | 18:14 逐臂：V2 dense/uniform/random/pairselect 100，pairfixed55；VMA dense/uniform100、random51、pairselect0；TS dense100、uniform71、pairselect71、random0 | 计数不等于继续派工；V2 pairfixed/1b 只隔离 pair 内成员偏好，不证明 pair 间排序；保留身份和反例，按 V3 收窄范围 |
>
> XD 本轮使用 accepted_train=3950/3954 的不可变 quarantine 视图，excluded=4 诚实披露；不再搜索、下载或补 4。XD raw→canonical
> 时间轴审计仍在关键路径；作者梯形 AP 与 `average_precision` 分列。
>
> F06 当前入口为 [`F06-update-ratio-actionability.md`](findings/codex-idea-20260920/F06-update-ratio-actionability.md)：
> F01 的 V2 block5 relative-attention update 是否预测后缀压缩敏感性。当前身份是
> `engineering_probe_authorized_not_property_confirmed`；2×2 coverage 只作诊断/强对照，不冻结为新 selector。

> ### 约 18:37 后续诊断（待处理）
>
> node1 逐卡 NVML 查询出现超时，当前只能标记为疑似驱动阻塞，不能写成 node1 已修复。node2
> 只剩一条 VMA 路径、其他卡空闲；精确状态以 infra 新 receipt 为准，TS 恢复优先在 node2
> 处理，不等待 node1。该诊断不改变 18:14 事实，也不代表计划已恢复。
>
> ### 自动接力与安全边界
>
> A100/V100 当前片段完成并通过增量审计后立即领取下一 encoder/dataset 片段；一个视图 ready
> 立即训练对应 head，head ready 先 QA/准备评分链；正式 test 评分等 idea 收敛及方法/矩阵冻结，不等其他组。巡检只读取结构化进度、PID/owner/argv、
> claim、最新 receipt 和增量产物，不全 NAS 重复 hash、不用 `ls` 数量代替成功、不用跨节点时钟
> 相减算耗时。只暂停确认为本轮冗余且无有效后继的 guard/controller，并记录停止理由、替代任务
> 和回执；不杀其他项目、不删旧缓存/失败证据、不重装环境、不改旧协议 JSON/SHA、不 commit/push。
>
> 质量差值定义为 `Δ=method-dense`；95% 视频级配对 bootstrap CI 下界 `>= -0.005` 才能称
> 非劣。训练内 100 视频不是泛化证据。正式效率必须短独占并包含插件自身开销，多进程日志不入
> 速度表。当前计划完整说明见 V3；本段事实更新后再在此处追加 receipt，不重写历史段落。

更新日期：2026-09-19。当前状态：**完整研究 goal 进行中；仅有 VideoMAEv2 特定性质的独立确认，尚无跨四 encoder 的通用规律或插件有效性结论。**

当前工作：**利用已有128视频观察形成并试验免训练候选。1288视频全量提取与关联开发训练/评分已按用户要求暂停。**
四组native工程验收已有回执；匹配UR-DMU正式基线、两个数据集的原版/压缩版完整比较仍未完成。

2026-09-19用户明确收拢顺序：保留现有全量缓存，先从现有FIT观察找到方法，之后再比较
UCF/XD测试集上的检测与压缩指标。LoRA及冻结谁/联合训练等选择放到免训练结果之后。
不修改旧角色锁，不用测试分数选择方法，也不把开发头/额外验证层当作找方法的必经前置。

已执行暂停：四个GPU guard均以SIGTERM正常终止各自作业组，native与guard原PID身份均已
确认退出；评分等待controller及Time迁移监控也已停止，没有创建评分case。保留完整视频缓存：
VideoMAEv2 **648**、TimeSformer **561**、VideoMAE **660**、V-JEPA 2 **498**。暂停前已发布的
完整视频索引inode/device/size/mtime逐一保持不变，缓存与旧partial未删除；正在写入的partial
仍按未完成产物处理。原job receipt可能仍留运行时的running字段，当前状态以guard取消回执和
本次暂停回执为准，不追溯改写旧运行记录。
回执：`outputs/icassp2027/control/pause-dense-fit-user-focus-20260919-a01/receipt.json`。

V-JEPA 2的FIT128注意力补观察已在暂停前自然完成1024/1024片段，exit0；完整核验与性质统计
尚待处理。下方各运行快照为注明时点的历史记录，不代表四个全量提取作业仍在运行。

2026-09-19 用户再次强调：最终四个encoder必须使用同一种压缩机制，TF/LoRA两版也同源。
已补入执行计划和方法约束；模型适配仅限布局/坐标/位置等原生约束，不允许各骨干另换评分或算子。
当前F04已包含四模型完整fit观察；绝对local cosine方向一致，但rank下降与local−nonlocal额外增量
未在VJ复现。仍无新独立确认或压缩有效性结论，不能直接确定统一方法。

## V-JEPA 2 注意力观察修复（2026-09-19；原固定八视频原生验收通过）

新增研究用 `SDPAQueryRowObserver`，在所选原生 attention 模块进入 SDPA backend 时读取
实际 post-RoPE Q/K，只重建均匀采样 query 对完整 K 的 float32 softmax 行。
原 backend 仍只调用一次、原返回对象保持不变；不切换 eager，不改变模型配置，退出后恢复
registry 的 local/global 状态。当前只支持 eval、无 mask、非 causal、零 dropout、非 GQA
路径，其余条件显式拒绝；重建显式关闭外层 autocast，并保留源码 SHA、实际 scale、dtype、
Q/K/V shape、query 索引和坐标等证据。

P10/P11 继续使用既有统计定义，但新增独立 reconstructed site；原生概率缺失记录保留。
P13 在已验证无 CLS 的模型上标为不适用，不再随 SDPA 是否返回概率改变语义。
本地定向验证已覆盖只读输出/梯度/参数与 RNG、registry 恢复、非零位置旋转、完整 K、
source JSON、旧 native missing 隔离和 TimeSformer 分离布局。本地小模型或单元测试
不构成正常—异常性质证据，不替代真实观察开关的数值一致性回执，旧冻结观察不追溯改写。

本轮 `.venv/Scripts/python.exe -m pytest` 合并执行 research、TimeSformer attention、runtime
layouts、SDPA observer、probe stage/streaming、bridges、V-JEPA SDPA stage 八个受影响测试文件，
结果为 **66 passed in 9.96s**。真实安装版 tiny V-JEPA 测试采用 Transformers 5.16.1、
torch 2.13.0+cpu、2层/2头/N8；observer 与 identity 的 features/pooled 最大绝对差均为0，
原 registry、RNG、参数/buffer 和 hook 清理验证通过。`compileall -q src tests`、受影响文件
ruff 与 `git diff --check` 均通过。
这些验证没有启动新确认分区、训练 LoRA 或读取官方测试数据。

原生验收已在独立 `code-6630829` 冻结目录完成，沿用原debug8的64行cohort和八行train
manifest，按字节镜像且SHA不变；仍为fit/debug/train，并非新确认样本。初次增量bundle因
服务器主仓库缺少e82cd7d前置提交而未通过校验，保留失败身份后改用完整历史bundle，成功创建
detached工作树，原origin tracked工作树未变。代码上传与预检记录均在
`outputs/icassp2027/control/vjepa2-sdpa-native-pilot-20260919-a01/`。

本次native PID1533/startticks1348257281、guard PID1452/startticks1348257229，使用GPU6
UUID `GPU-71a55fac-63a9-8504-5651-4c72fdf1f2b1`，启动时无其他compute PID。实际导入
torch2.8.0/Transformers4.57.3及新冻结src；2线程、interop1、cuDNN/TF32关闭、allocator.5。
固定四层b5/11/17/23、每层64 query对完整8192 K，P10/P11/P13-only，原八视频×八窗口，
共192次baseline/observer/identity前向。run为`probe-20260919T022308454825Z-7f4ccde1`。
epoch1789784926全64 clips产物审计通过：全部四层Q/K/V为[1,16,8192,64]、重建矩阵
[1,16,64,8192]，实际scale=.125，native/reconstruction均float32。每clip的observer与identity
features/pooled最大绝对差全部为0；4096个输入帧索引、query IDs与网格/tubelet坐标、
模型权重/源码来源均核对。64个shard与最终JSONL逐条一致：重建P10/P11 available20480，
原生概率unavailable512、P13无CLS not_applicable256；native与guard正常退出，未见tensor dump。
回执`audit-receipt-a01.json` SHA为`5636a509da0ace369be7c1fba0eb39405fb0d172e1b5dee8a3db55ca76e3c445`。
本轮实测产物266,128,171 B（253.7996 MiB），接近256 MiB预算，不能沿用此前低于128 MiB的
估算扩展；下一轮FIT补观察使用独立NAS命名空间及6 GiB输出预算。这里的峰值allocated
1,834,341,888 B、reserved2,092,957,696 B仅为含观察器的工程容量。
本次是工程观察验收，即使guard标记idle策略可计时，也没有正式计时请求或论文效率数字。

在八视频全验收通过后，原FIT128的V-JEPA注意力补观察已真正启动。原cohort为1024行/128
视频/1024唯一clips，保持fit/explore/train、八窗口、原采样和query32/token256/record128；
cohort/manifest/plan SHA原样保持，新suite只请求P10/P11/P13，并保留新来源身份。
数据根仍为原`UCF-Crime-official-verified`，不使用debug路径代替，也不创造新确认分区。

新研究工作树为`/data2/localdisk/fotile-icassp2027-vjepa-sdpa-fit-20260919-a01/code-6630829`，
与heads/scoring的24 GiB命名空间完全分开。a01/a02部署稿在静态审查被拒，均未执行；
实际执行a03，复用已验收pilot wrapper并绑定全部输入/源码摘要，NAS原子写/rename及
真实CLI dry-run通过。预检分别核root15 GiB+64 MiB、NAS15 GiB+6 GiB，实测root free
58,488,651,776 B、NAS free5,612,703,318,016 B；6 GiB为规划预算，不是物理quota。
预检SHA`41f34c6cd4b853af8035daa1bd64834641c180f058f0d384de4a15b0466ae9b2`。

实际run为`probe-20260919T023815898253Z-e49df322`，guard PID14010/startticks1348347976、
native PID14097/startticks1348348049，GPU6、原foundation环境和同一backend/thread/cap规则。
epoch1789785515实际核验进程身份、NAS导入路径与首clip已发布，仍运行中；全1024 clips的
完成、与旧FIT逐clip frame-map对齐、性质比较尚未完成。控制材料为
`outputs/icassp2027/control/vjepa2-sdpa-fit-reobserve-20260919-a01/a03/`，不得重复启动。
epoch1789785719首consumer审计通过，快照57/1024 clips、7/128视频，两个PID身份仍live。
仅首clip `Abuse005_x264:segment-00`已逐项与旧FIT核对：完整sampling/source_frame_indices及
11项几何字段一致，实际query32/fullK8192、四parity均0；project/run/shard均为NAS device44、
owner fotile，与/users不同device。首shard为2,076,253 B；不能将首clip审计扩大为1024全部通过。
回执`a03/first-consumer-review.json` SHA为
`df44d0cdba59ce123e3d4e58ff2b4002578a812c17b4a0fa19762c987fd4d6eb`。

## 当前运行快照（2026-09-19；各项以其注明的采集时点为准）

- epoch1789784305新快照`baseline-live-20260919-a08/snapshot.json`：VideoMAEv2 498/1288、
  TimeSformer498/1288（497复用、1个新完成）、VideoMAE547/1288、V-JEPA 2 448/1288。
  四native owner/argv/startticks一致，均extract_dense；root free59,271,155,712 B，
  RAM available1,424,325,664,768 B。下方a07保留为此前时点，不能据视频计数换算计算完成率。
- Time新完成视频已单独核对为Normal_Videos307_x264：628,020帧@30fps，78,502条完整
  dense索引坐标、首尾窗口及end_anchored通过；所有记录无reused_from，78,502个新唯一inode
  均nlink1且不与oldpartial重叠。旧partial的489个block index/31,264 clips及31,753个文件
  stat/索引SHA前后未变、原top index仍不存在。本轮只核metadata与stat，不重读NPZ payload；
  原64 clips的bitwise证据仍仅覆盖64，不能将31,264条声明SHA相等扩大为新payload验证。
  回执`time-first-new-complete-20260919-a01/receipt.json` SHA为
  `0ae6f863334aa3b4628d08e6f097948a3608d86fed3f90b992c1e935df5d2d3c`。
- 2026-09-19T02:19:31Z只读复核评分controller PID24721/startticks1347808693：owner、argv
  匹配，state=S；completed=[]、needs_review=[]，四组waiting=live。四FIT仍running，
  training_status/run_dir为空，尚无score cases或质量结果。回执
  `development-select-scoring-20260919-a01/poll-receipts/controller-poll-20260919T021931Z.json`
  SHA256为`0685e8c4e44410c926397f2e38477223a3ac01291cd4a4d59ab97f2b3ec3c379`。
  指定远端poll脚本不存在，因此改用等价只读检查并在回执标明；未重启或修改队列。
- 此前node3只读测量epoch1789780308：完整fit视频分片为VideoMAEv2 498/1288、
  TimeSformer 497/1288、VideoMAE 498/1288、V-JEPA 2 407/1288；四条native的owner/argv/startticks
  均匹配，仍处于dense提取。Time的497个视频为已复用完成的原完整缓存，其余视频继续编码。
  回执为`outputs/icassp2027/control/baseline-live-20260919-a07/snapshot.json`；
  root可用60,821,815,296 B、RAM available1,425,347,682,304 B，各目录扫描并非原子快照。
  视频长短不同，这些计数不是计算完成率；UR-DMU 的 3000 步开发训练尚未开始。
  VideoMAE此前停留497的原因已核实：Normal_Videos307含628,020帧、约5.815小时，需39,251
  个dense窗口。epoch1789767169时块index已有28,880条记录（含当前块，非完整视频）；
  同PID/startticks间隔132秒CPU增加15,751 ticks、wchar增加67,287,953 B，明确仍在写入。
  `baseline-live-20260919-a04/vma-live-progress-verification.json`保存核验，不据此重启或报告模型速度。
- VideoMAEv2、TimeSformer 的 development-select 均已完成 161/161 视频（80正常/81含异常）。
  独立审计逐一验证 65,862 / 131,814 个 blob、数组 SHA/shape/dtype/finite、原生窗口和视频完整覆盖，
  无缺失、重复或外来视频。结果在 `outputs/icassp2027/control/development-select-completion-audit-20260919/`。
  这些是开发基线特征，不是检测质量、官方测试或已训练后端。
- VideoMAE select 已在空出的 GPU4 恢复，唯一新控制目录为
  `code-0192fad/outputs/icassp2027/control/research-priority-handoff-20260919-a03`；
  node3 epoch1789757320已核验guard/native正常完成、根index发布、161分片无partial，共65,862 clips。
  其中79视频/40,244 clips为严格复用，82视频/25,618 clips为新提取；未将复用写成新增计算。
  extraction contract SHA `bf26ad197459d8e0e11863a0ea8ea9c52374955435b7259e337c080544ce18dd`。
  全量独立审计已通过：161视频（80正常/81含异常）原select角色、原生dense窗口和数据指纹一致；
  65,862个blob SHA及每项features(1,768)/pooled(768,)的<f4/shape/finite均通过，无缺失或重复。
  回执`outputs/icassp2027/control/videomae-select-completion-audit-20260919/receipt.json`，
  SHA `586ac40a037f9bf5b66d68436e4e0114fa0e1705cce2dc089c309a8b0d8b978f`。
  审计使用CPU flock、CUDA=-1、两线程nice15，实际max RSS 1,135,532 KiB；未运行模型或评分。
  原receipt的wait_seconds实际含审计时长，旁置root-review明确字段局限，未修改原回执。
  旧等待监视器24916已核验退役，旧源与已取消的重复复制产物保留。
- V-JEPA 2 select 的独立a03任务现已完成并通过全量审计：161视频（80正常/81含异常）、
  16,400原生dense clips（每视频3–2,217），根index已发布，原native20861及dispatcher24884已退出。
  逐blob SHA、features(1,1024)/pooled(1024,)的<f4/shape/finite、原角色/manifest/model/sampling
  均通过；四个SELECT现全部完成。这不是检测头训练或测试分数。
  回执镜像`outputs/icassp2027/control/vjepa2-select-completion-audit-20260919/remote-a02-receipt.json`，
  SHA `1d7824da94cf2e4ba93d264a5f8c13597d17706b7f5953a15b08f7e9b40721b5`。
  CPU审计nice15、CUDA=-1、两线程、max RSS292,172 KiB。原lock_wait_seconds混用epoch和monotonic，
  无法还原真实等锁时长；旁置timing-field-companion保留说明，原回执未改、未重复运行payload。
  同时钟的审计walltime为14.1407秒，仅为缓存核查用时，不能作模型效率数字。
- TimeSformer 主观察及补充 P10/P11 观察均为完整 128 fit 视频 × 8 窗口，并已完成 10,000 次
  视频级 bootstrap。补充观察有480可用签名；统计与语义审查见下节，不构成统一压缩规则。
- V-JEPA 2完整8探针观察已完成128视频/1024逻辑窗口，guard24800 exit0；CPU a03 worker2160
  随即完成10,000视频bootstrap、exit0，父monitor13537也已正常结束。787可用+12不可用签名完整保留。
  全量几何/parity/源帧及787项统计点复核通过，四文件原字节镜像与worker输出SHA一致；科学更新见下节。
  后续只读源码检查明确：P10/P11未取得概率是SDPA返回None的工程观测缺口，不能把missing-sites
  直接解释成hook从未执行或模型没有attention；真实CLS类P13因VJ无CLS应语义上N/A。
  可在实际post-RoPE Q/K边界旁路重建有限query、完整keys的数学概率；必须标重建来源并重新做
  数值/几何/生命周期验证。当前尚未实现或运行此扩展，未回填旧结果或改变P04候选/确认门槛。
  只读依据见`vjepa2-sdpa-observer-feasibility-20260919/README.md`。
- VideoMAE同128个fit视频的早期P02/P04切片已完整完成：1024窗口，GPU5 guard exit0。
  CPU队列a02随即完成10,000次视频级bootstrap，worker28963与monitor8874均正常退出，
  13次资源采样、max RSS 600,162,304 B、无资源中止。原a01退役记录及a02运行身份保留。
  统计、几何与输出等价性完成独立复核，全部缺失项保留；科学结果见下节及更新的F04。
- 四个 fit 的新旧 native 表示/采样兼容盘点已经完成；TimeSformer 已通过正式 handoff 并由
  新 native 接管，具体状态见下；其他fit没有迁移。单独的inactive权重备份已完成4,957,392,176 B传输与完整SHA验证，
  verified-stream-a03回执绑定本地/远端同一SHA90e6a81a…5b0aa；服务器原文件仍保留，未据此删除权重。
  本次迁移已再次核验容量门禁、源完整视频及新进程接管；完整提取与训练仍未完成。
  受限`resume_transport=hardlink_npz`现已实现：仅identity、完整视频、同文件系统的两成员NPZ，
  保留原SHA/身份/shape/dtype检查，独立重建索引/lineage并记录共享inode；默认copy不变。
  本机相关测试94 passed/3 skipped，skip均为Windows真实symlink权限；隔离拒绝分支通过。
  compileall、限定五文件Ruff及diff-check通过。实现验收时未用于真实缓存；旧helper原样保留，
  实际迁移采用下面独立的新control和重新审核的门禁。
  实现已冻结为`e82cd7d`并上传独立工作树；Linux classic环境96项通过，唯一VJ旧reducer测试因
  该环境没有VJEPA2Model类失败，随后在现成foundation环境单独1项通过。合计97个独立测试
  已在各自适用环境通过，Linux真实symlink负例不再跳过；原失败日志保留，未重装环境。
  node3最初新建synthetic NPZ的文件系统canary证明link共享inode且字节不变；真实源首clip格式
  样本符合两成员约束，后续全量核验见下。初期426完整视频的Time metadata扫描
  S=3,827,105,792 B，其中完整NPZ候选H=2,269,626,368 B；新journal、
  三层索引、lineage和目录开销另计，不能简单按S−H宣称迁移容量够用。
  八个普通dense store精确共2,923,485 clips，原始数组payload17.055GiB不是总占盘上界。
  最终采用root修正的serializer-a03：四个真实resolved原字节与远端新鲜SHA一致，补齐
  declared_readout及三层各3个浮点字段宽度；原获取时间无据则null，不沿用手写时间。
  三层JSON逻辑包络24,830,642,412 B、逐index 4KiB模型24,912,494,592 B；旧a01/a02保留。
  两native环境真实NPZ格式+其链接的zlib raw-DEFLATE bound得到D768≤6,672B、D1024≤8,722B；
  对应4KiB数据块模型8/12KiB。XFS实查inode512B、block4096B、noquota；目录/extent/日志
  分配仍另计，不把逻辑字节界称为完整磁盘保证。初步容量预检只读取完整源index/stat，
  后续payload验收和真实迁移见下。
  两项已结束FIT探针及CPU分析的剩余写入已核验为0；未来工作仍需另预算，15GiB safety保持。
  后续固定497个完整视频/309,593 clips的全NPZ核验已通过：每个blob完整SHA、ZIP恰好两成员、
  fp32/shape/nbytes/finite、features与pooled相等、前后inode/mtime/size及索引SHA均一致；
  309,593个独立inode实际分配2,536,185,856 B。完整源生产进程仍运行，未冻结整个root。
  回执`time-fit-hardlink-source-validation-20260919/receipt.json`绑定逐视频摘要SHA
  `ef28fba1a40afc110cf321c3462601d05f30c8f97475d1743e63bddbb0c53056`，CPU worker已正常退出。
  新e82代码在classic CPU加载现有权重后，representation/semantic runtime/encoder fingerprint
  与源完全相符，未初始化CUDA。数据摘要在该身份检查中沿用旧值，实际extractor仍须重新核对。
  四SELECT及旧冻结probe的terminal证据已单独绑定；零未来写入仅适用于这些已完成阶段。
  最新剩余写入模型required=65,315,596,288 B、free=65,830,965,248 B，差515,368,960 B；
  已含15GiB safety、2GiB heads/scores、1GiB XD及512MiB transient。此为有余量的规划模型，
  并非已执行迁移；切换前还要新鲜检查，旧源全部保留。旧source与新official的模型microbatch
  都是8；此次提取优化是reader复用和批量索引写入，不是将模型batch从1改为8。
  一次实际格式canary已通过：Assault038的17 clips先按字节复制至隔离source，再做hardlink/copy，
  标准FeatureStore消费者的数组/dtype/覆盖及新索引一致；真实源字节/index/inode/nlink未变。
  包含worker输出实际2,117,632 B，小于64MiB；receipt SHA
  `77774955547d3a74c966e6c3e0916e73a7bd1b0b90427da986e5d6ae33072e82`。
  正式迁移control为`dense-batch-migration-20260919-timesformer-hardlink-a01`，plan SHA
  `7b00352edaabfa42fd4ce3cd3b250ff6494ebc0fbffb2c02fa63deda27eaa55c`；四helper与具体plan
  完成独立审阅。实机preflight-only通过，required65,065,486,336 B、free65,475,231,744 B，
  所有reserves之后余409,745,408 B；真正取消前后又重新校验容量和全部497个完整index。
  epoch1789775072旧guard1以SIGTERM结束自身slot，native24420退出；原源完整/partial文件均保留。
  epoch1789775095新monitor确认接管：monitor26697/t1347304509、native29605/t1347307375，
  同GPU1 UUID、classic torch2.3.0+cu121、allocator0.5、identity/batch8，仍为非正式计时质量任务。
  原Normal_Videos307只有部分block，需重新提取。epoch1789775208–5223的新native两次只读核验
  中身份稳定、CPU增加1494 ticks，rchar增加12,926,554,088 B，正读取UCF原视频进行数据身份检查。
  此时新target尚未发布任何完整分片，不能把进程接管写成497视频已经复用完成或完整fit已完成。
  epoch1789775502已真实复用81个完整视频；首视频Abuse002的108 clips完成独立消费核验：
  新旧数组/坐标/encoder fingerprint一致，实际GPU提取resolved的spec/data evidence与源一致，
  两路径确实同device/inode、nlink=2，源index SHA不变，108条journal及新runtime引用正确。
  回执`first-real-reuse-verification-a02.json`；初版只读核验脚本误把既有relative runtime引用
  当绝对路径而断言失败，已按实际`base=extraction_run,path=resolved.json`修正并保留失败记录，
  没有修改生产格式或重启任务。81是复用完成数，不是新编码或1288完整fit完成数。
  更新到epoch1789775626：target已有119个完整视频，119个均属于固定已验源集合，集合外为0；
  monitor/native身份稳定，状态仍为`resume_running/extract_dense`。快照为
  `hardlink-resume-native-20260919/handoff-progress-snapshot-a01.json`。
  全部固定497视频/309,593 clips现已复用完成，并通过metadata-only整体复核：原source index与
  target lineage SHA仍等冻结plan；坐标、采样、fingerprint、数组声明及runtime引用一致；
  309,593个唯一共享inode当前nlink均2，dev/inode/size/mtime与journal一致。当前stat+声明SHA
  重建的滚动摘要仍等此前全量payload验收。此次没有再读/解压NPZ，不能写成重复全payload审计。
  receipt SHA `def18eb57eae79951d0319bff94a568075388aabef2416b866555464bf8c599e`，目录
  `time-fit-hardlink-complete-reuse-audit-20260919/`；CPU审计退出0，原提取进程未受干预。
  Normal_Videos307已开始真实重新编码：其628,020帧对应78,502个原生dense窗口；前64 clips
  与旧partial的数组逐元素相等（max abs=0），64对NPZ字节SHA也相同，但为不同inode且没有reuse标记。
  这只证明前64窗口的真实重新编码等价，完整长视频尚未完成；回执`reencoded-prefix-equivalence-a01.json`。
- **已纠正训练容量漏项，并落实双卷存储。** 检查实际`urdmu_training._aggregate`后发现，
  原单卷remaining模型没有单列四份float32 `[1288,200,D]` 的`bags.npy`；纯payload共
  3,429,171,200 B，包含header/4KiB与16MiB规划余量后新增3,445,964,800 B。
  因而不能继续把旧2GiB heads/scores预留当作完整训练容量保证。修正的单卷实测规划曾缺
  3,091,933,184 B；原门槛与历史回执保留，未通过下调15GiB safety掩盖缺口。
  已在NFS `/data2/localdisk/fotile-icassp2027-20260919-a01` 创建本项目owned0700目录，
  device44与`/users` device64768分开，实际memmap与fsync/atomic rename canary通过。
  在四job仍extract_dense且原heads不存在时，以create-only symlink将未来heads映射至该卷；
  原训练函数的`Path(output_root).resolve()/run_id`使aggregation、checkpoint、QA和result均落此处。
  未移动既有数据、重启native、修改训练代码或删除权重。独立路径/预算审查通过。
  双卷snapshot epoch1789778921：root required60,859,205,632 B/free61,203,390,464 B，
  原15GiB safety、2GiB heads/scores、1GiB XD、512MiB transient均保留；secondary required
  21,699,575,808 B低于24GiB规划预算，实际free5,612,705,415,168 B。规划预算不等于文件系统配额预留。
  证据集中于`outputs/icassp2027/control/aggregation-budget-repair-20260919/`；后续需继续检查两卷空间。
- **开发SELECT评分已接续排队，但尚未产生质量结果。** 现有`score_urdmu.py`支持严格的
  `development_video/direct_insert`；已固化四份完整SELECT的真实路径/contract/manifest SHA，
  仍将actual trained run保留null，待各自真实3000步、seed0、dense_reference、完整QA全部通过。
  代码中的`official_frame`仍在读测试来源前拒绝执行；计划第94段已纠正“开发评分未接线”的旧描述。
  对最长SELECT形状做了单次合成算子容量验收：Time `[1,17737,768]` 使用classic torch2.3.0，
  VJ `[1,2217,1024]` 使用foundation torch2.8.0；query_chunk256始终保留全K/V，输出finite、
  shape正确、权重不变。实际allocated/reserved为745,642,496/803,209,216 B及
  127,467,008/142,606,336 B。该数据仅为工程head+合成输入的容量证据，不是质量或正式速度结果。
  GPU4 guard与两个native均exit0；回执在`urdmu-select-shape-capacity-20260919/native-a02/`。
  新评分controller24721/t1347808693已通过实机CPU预检并启动，唯一control为
  `code-e82cd7d/outputs/icassp2027/control/development-select-scoring-20260919-a01`，plan SHA
  `410ac89a6370321f58bcc587b19d2bfa6613944ccbafce331b8494459e01a308`。epoch1789780215进程身份
  匹配，四case均等待live fit、needs_review=[]、score case目录0；等待时不申请GPU。
  完成head后才逐case冻结真实artifact，经GPU4 lease串行运行；query_chunk256、workspace2GiB、
  allocator0.5及cuDNN/TF32关闭固定。head与score大输出落新NFS卷，控制器/原生wrapper均重新检查
  实时空间和静态来源；旧start/intent拒绝重入，不自动重试。正式测试、额外seed与方法选择不在队列内。
- XD 测试原视频800份的新坐标审计已sealed并通过独立proof和consumer检查，详情见下节。
  训练仍有4个原始坏成员未修复；有界公开镜像检索没有找到可用原始payload。
  官方OneDrive浏览器下载仍待已提出的D盘目录设置/直接原始文件链接，尚未启动约15GB到C盘的下载。
  UCF研究继续推进。官方模型测试分数未读取，最终TF/LoRA方法与完整三seed质量矩阵尚未完成。
  当前CUA surface已做一次只读复核，仍无browser surface且返回nodeRepl.fetch失败；没有新下载途径，
  未再次搜索镜像或启动大下载，人工D盘目录问题仍待答复。新记录为
  `xd-four-bad-repair-20260918/capability-review-20260919/current-surface-recheck.md`。

## 四模型性质观察完成：保留异质性（2026-09-19）

最新推进：已在读取剩余33个confirm的像素、controls与目标统计之前冻结
[P04四模型独立确认合同](decisions/p04-four-encoder-confirmation-v1.md)及同名JSON，
JSON SHA `936c5af50864fff0a7b9975d8985fdb626731a87eb2fa8a546cc1cb4699f0210`。
唯一primary为相对深度.25 block input local cosine，不把coverage_k4增列为第二候选。
权威元数据证明33=16/17、原confirm角色及与FIT128/两旧64确认池ID互斥；最短485帧，
264个VJ64/stride2窗口均有效，16个边界clamp、无复用。不把未知来源分组称为近重复已排除。
四份FIT正常64校准的SHA/normal IDs/采样身份已绑定，root实际重算四native采样digest一致。
新覆盖门槛为每标签至少12、min-count质量至少10、至少2个完整分层、非singleton分层质量比例至少.8。
先执行controls-only；任一模型覆盖不足即报告coverage_insufficient，不启动新P04 GPU确认。
真实controls-only现已完成，四模型均在最后一项覆盖门槛失败：V2与VMA为11/14=.785714，
Time为9/13=.692308，VJ为8/13=.615385；其余三项覆盖条件均通过。
已按合同停止新P04 GPU目标确认，没有放宽0.80、删singleton或改变bins。性质本身仍未确认，
本次不是效应为零的证据。[结局JSON](decisions/p04-four-encoder-confirmation-v1-outcome.json)
与[F04](decisions/findings/F04-early-layer-structure-exploratory.md)保留完整表和null效应/区间。
root核验13个产物SHA，独立重建所有controls均值/分箱、四组覆盖与27,456源帧索引，最大差2.7756e-17。
回执SHA `b8320d17155e4b10b3c7eca9bf97f3ca0a784944cf6f232005a96fdf6aad8a06`；四组各33×8窗口，
33输入视频SHA、帧坐标、校准与代码绑定齐全。native worker27500/controller27483已正常exit0，
nice15、两线程、CUDA=-1、max RSS217,648 KiB，36次监测没有资源停止。
执行来自独立冻结`code-ab338f5`，既有baseline工作树未改。初版工具缺项及a02标量metadata错误
均在真实读取前由root修正，a03通过本地/原生环境合成边界和合同校验；未执行两份旧草案。
首次部署仓库无前置commit、首次启动误用不存在的/usr/bin/python3均在读取数据前失败，
旧记录保留；新a02部署/启动各只执行一次，没有重复controls进程。
原33视频仅controls已消耗、target仍未读；新增独立验证视图的角色规则修订已向用户询问，
当前只做可用样本元数据审计，未创建或运行新cohort。旧角色锁及原3个保护文件不变。
该有界审计在1288个原fit中排除了明确已用FIT128及其已知探索/校准子集，留下1160个角色上
候选（576正常/584含异常）。没有完整跨本地/远端使用账本，因此不能称全局未使用或已证明独立；
不据此自动重新划角色或启动确认。回执在`additional-property-validation-feasibility-20260919/`。
随后补审三个指定远端根的一级probe resolved/cohort及初始debug8，未发现额外原fit使用ID，
候选池不变；其他历史根、非probe手动运行和删除的run仍未覆盖，未知边界保留。

- 当前完整VJ原生run为`code-0192fad/outputs/icassp2027/runs/probe-20260918T163038291972Z-a4fc7f17`，
  805,888条available/12,288条unavailable；分析输出`vjepa2-fit128-video-bootstrap-20260919-a02`。
  原生24 blocks，.25实际b5；全1024条架构记录的真实32×16×16几何、65,536源帧索引和
  observer/identity零差值通过独立复核。SDPA未返回概率，P10/P11/P13缺失保留，且没有CLS。
- 1024是逻辑窗口数。Normal_Videos155与RoadAccidents069各有2个边界clamp复用，按video内
  源帧索引去重后共1020窗口；一正常一含异常。仍按原8逻辑区间取video均值，以128视频bootstrap，
  不把窗口或token当独立样本。CPU分析64次采样的最大RSS 5,259,964,416 B，无资源中止。
- 787项raw g与787项matched点/覆盖已独立重建（最大差4.6088e-14）。VJ原匹配64/64、9分层、
  质量54；其它三个模型质量56。原始完整导出和错取后隔离的VideoMAE controls均保留，正确VJ
  controls SHA5158d640…1776e已与原worker绑定，未用错取文件生成VJ诊断。
- 四模型主切片共有50个P02/P04无head签名，全部保留。仅5条在四模型raw/原matched和新
  三字段敏感性均为正：4个早期站点的local cosine与input的coverage_k4；它们相互相关，
  不能计成5条独立性质。coverage_k4是4个均匀代表的平均最大余弦，不是四token压缩实验。
- VJ主站点原matched rank +0.03542 [−1.11367,+1.19177]、local−nonlocal +0.00305
  [−0.00604,+0.01246]均跨0。因此前三模型的rank下降和额外局部增量未得到四模型支持，
  不改换VJ层来挽救结论。local cosine仍为+0.00744 [+0.00398,+0.01086]，coverage_k4
  为+0.00380 [+0.00176,+0.00588]；这还不能证明何种token可以压缩。
- 新输入反差诊断完整保留50签名×4模型×两种匹配=400行。VJ三字段local +0.00668
  [+0.00399,+0.00942]、coverage_k4 +0.00407 [+0.00245,+0.00566]仍正，rank与local−nonlocal
  仍跨0。三字段VJ保留61/54、21分层/质量39，不能与原64/64人群混淆。
  VJ全41个local−nonlocal站点的82行诊断亦保留；root对400/82行点与覆盖的独立重建均通过。
- [F04完整更新](decisions/findings/F04-early-layer-structure-exploratory.md)保留历史2/3模型原表、
  原F01/F03身份及剩余33个confirm边界；四模型图显示VJ跨0区间，已实际查看并通过独立审查。
  该FIT阶段结束时尚未注册新确认；随后完成单一local cosine预登记，并得到上述coverage停止结果。
  未执行P04目标确认，未选择selector/预算或启动LoRA。

## VideoMAE完整观察与三模型探索证据（2026-09-19）

- 预登记的同家族补充已完成，80可用签名、6不可用签名，10,240行video×signature统计；
  独立样本为128视频（64/64），每项8窗口。全1024窗口真实8×14×14几何与16,384源帧索引
  通过独立复核，observer/identity输出最大差均为0。Q/K/V模块hook未触发，对应NA保留，
  不称接口全可用，也不把本次只读观察当作新梯度或变长验收。
- root核验下载输出SHA，重建全部80 raw Hedges g和80原matched点及覆盖，最大差1.7431e-14。
  三模型共有50个head-free签名全部保留；8条满足三个模型raw/matched六CI同方向且不跨0，
  包含同一rank的normalized缩放与相关站点，不能称8个独立性质。
- VideoMAE主站点block.2.input原motion×brightness matched差：有效秩−2.16031
  [−3.31216,−1.01161]，local cosine +0.05583 [+0.03926,+0.07244]。这支持原两个模型
  的部分探索方向，仍是相同fit视频上的同家族补充，没有新增独立确认或四模型通用结论。
- 补齐全8站点的每视频local−nonlocal诊断：主站点raw Hedges g +0.45347
  [+0.10813,+0.83170]；原matched +0.01408 [+0.00523,+0.02314]；加入fit-normal亮度
  标准差分箱后+0.01375 [+0.00625,+0.02132]。三字段保留59/52视频、19分层、质量39，
  与原64/64、9分层、质量56人群不同。不能把它解释为因果控制、纯去均值或压缩性证明。
- 正式诊断仅采用独立新目录`videomae-local-excess-fit-20260919/a02_verified`；a01漏raw统计
  和部分执行绑定，原样保留并标不完整。a02绑定输入前后SHA、真实解释器/版本和导入numpy前
  的线程环境，原8站点16行全部保留。root重建派生raw/matched点，最大差1.7764e-15。
- [F04](decisions/findings/F04-early-layer-structure-exploratory.md)已加入完整范围、负/缺失项和
  三模型原匹配图。PNG/SVG已实际渲染查看，数值来自审核CSV；VJ仍待完成，未注册候选或启动LoRA。

## 跨encoder相对层深度核验（2026-09-19）

只读复核已完成的四模型native后层执行回执，SHA与原aggregate绑定一致。VideoMAEv2、
TimeSformer和VideoMAE三个12-block模型的0.25/0.5/0.75/1.0对应2/5/8/11；24-block
V-JEPA 2对应5/11/17/23（均零起始）。
保留原观察公式和主切片0.25；未来比较保留实际层号、site后缀、sublayer和attention domain，
不直接按block编号混为同深度，不根据结果换层。这里只读结构元数据时未读取未完成VJ的性质统计。
`outputs/icassp2027/control/cross-encoder-depth-alignment-20260919/alignment.json`绑定15个VJ
和其余三模型的完整后层执行回执；现已由VJ当前全部1024条probe架构记录核验自身block_count一致。

## 完整结果比较与第四组select审计准备（2026-09-19）

- 已完成相对深度对照表的真实三模型回归，最终仅采用ignored
  `relative-depth-comparison-20260919/a04_verified`。join key去除实际block编号，仅以既定
  相对深度和完整site后缀等字段对齐；实际层号、site和来源仍逐模型保留。Time的non-block
  embedding及temporal/spatial域独立保留，缺失值为not_in_export/空数值，不填0。
- 同128个video→label映射、64/64和每项8窗口已核对。7800行union含全部head-free一级对比，
  300行共有长表对应原三模型50个签名×3模型×2估计量；原status/point/CI/estimand/reason
  均逐项不变。12-block b2与24-block b5对齐、相同b5不同相对深度不合并等元数据用例通过
  生产函数验证。所有输入前后SHA及输出SHA绑定，未新增bootstrap或读取VJ未完成统计。
  Time补充的per-head attention仍在独立export中，不混入这张head-free表，也不改写旧run的NA。
- a01–a03保留为开发尝试：早期实际编号残留在key、遗漏non-block及报告字段不足均已纠正。
  当前a04回执SHA `f439d7382da8476e50e291a5f86fac6724a6cb1a433d4a9c6ddf9154dfb8a5a8`。
  待VJ全run与统计审计完成后，才另建绑定其真实产物的新manifest运行，不假装已有第四模型结果。
- VJ select的全量CPU审计已排入唯一新a02队列（monitor15535），等待source job、guard成功完成
  和非空根index三条件。验证固定64帧/stride2/window64、D1024及指定manifest/role-lock/authority
  SHA；原native20861/guard24884未改动。旧waiting器21761在精确核验且无审计子树后退役，
  原文件和错误monitor JSON保留。当前还没有VJ select审计通过结论。
- Time复制空间静态审查确认每clip目标只新增一份NPZ（内部features/pooled两个成员），
  三层索引不另复制blob；仍有原source保留、lineage及重复metadata增量。源码不足以证明更紧
  allocated-byte上界，因此未下调现预算、未启动迁移，报告在`time-fit-copy-space-review-20260919/`。

## 提取迁移准备与当前资源门禁（2026-09-19）

- 一次性 TimeSformer fit 迁移控制器已补齐取消前preflight、写cancel前身份/阶段复核、
  termination_pending/终态处理、同目录重入和真实native接管监督。原始helpers保留，
  r01新文件由root再次py_compile通过；独立审查的6项隔离状态分支通过，未作为实机迁移证据。
  位置：`outputs/icassp2027/control/dense-batch-migration-20260919-timesformer/revisions/`。
- 仍未生成真实迁移plan、取消任何fit或启动新fit。控制器保留旧保守容量公式；上线前必须
  根据当前实测预算修订并重新绑定SHA。取消只保证已有完整视频可复用，不保证精确停在视频之间。
- 独立容量v2使用原8个dense store（VJ select为专用a03）、停止的VMA副本和当前a03副本、
  两项probe实际run目录；全部同容量池且无祖先重叠。57GiB dense、4GiB VJ probe、1GiB VMA
  probe各扣除已分配空间；保留1GiB XD修复、15GiB安全余量及2×Time source+512MiB迁移预留。
  实测free=72,867,971,072 B，required=77,843,406,848 B，缺4,975,435,776 B，未通过门禁。
  错误v1曾把jobs控制目录当probe数据目录、比较动态df字段，已保留并纠正；只采用
  `outputs/icassp2027/control/time-fit-current-capacity-20260919/snapshot-v2.json`（SHA eab4f75a…8ade6）。
  本轮没有清理权重；原fit继续运行，备份完成及完整SHA验证前不删除服务器原文件。
- node3 epoch1789755416的VJ/VideoMAE观察为554/732个窗口（各1024）；VJ guard/native身份匹配，
  VideoMAE随后也再次核验三层实际进程存活。所有结果仍等待完整封存与CPU分析。

## TimeSformer补充attention完整描述（2026-09-19）

- 全量保留 spatial/temporal × 4深度 × 12 heads × 5指标 × weak/matched 共960行对比；
  fit-only、未多重校正、不选择head。weak区间对应Hedges g，matched区间对应原始差值。
- 语义判定分别处理outgoing与incoming：P10只有entropy下降且top-4 mass上升才称集中；
  P11还要求gini上升。反向才称分散，其余uncertain。不能把不同指标的符号直接混为方向，
  也不能将P11的三个相依描述计作三个独立性质。
- raw与matched共同支持的模式局限于部分head：spatial layer2的outgoing集中2/分散1、
  incoming集中3；spatial layer5 outgoing分散2；spatial layer8 incoming集中1；
  temporal layer8 outgoing集中3/incoming集中5；temporal layer11 incoming集中1。
  全部96个domain×depth×head组合及uncertain结果保留，未据此选择层或形成候选。
- spatial是在每帧197 keys、32 sampled queries上的局部分布，temporal是每空间轨迹8 keys/
  8 queries上的局部分布，再按原生group等权汇总。它们不能直接与V2的全局attention数值混池，
  不提供全局CLS或异常token结论。产物与语义修订在
  `outputs/icassp2027/control/timesformer-attention-descriptive-20260919/`。
  root已从960行CI独立重建全部96项语义分类及8×18汇总计数，完全一致。
- 实际CPU分析exit0、66次资源采样，峰值RSS 3,628,580,864 bytes，未触发资源停止。
  这是分析资源回执，不是encoder正式效率数字。

## 最新完成回执（2026-09-19）

- Time局部attention补充已完整完成，guard24836 exit0：128视频、1024clips、491,520条available
  P10/P11记录（480签名×1024），run为`code-0192fad/outputs/icassp2027/runs/`
  `probe-20260918T155022550042Z-94a7dc2c`。完成后VJ guard24800已获得GPU6，native14237
  正在新身份下检查输入；实际owner/argv/startticks已核对。尚未解释未完成VJ数据。
- XD新规则全800审计已sealed，800新探测、0复用、0raw timing失败，仅`b0-half_open-origin0`
  满足精确坐标门禁。seal SHA
  `ac2ba2fd66899ebe0bdc1ab15c048f146d901b4cbd8425682551577986c8d21b`。
  独立复核重新计算800当前raw的SHA/CRC/bytes、receipt及其三份artifact的SHA，并对保存的全帧
  PTS/OpenCV结果重算时序条件；六个QT分类索引恰为0140/0141/0142/0598/0599/0600，全部通过。
- `verify_raw_coordinates`叶子consumer也通过800 manifest/raw rows；这只验证数据坐标，不是新方法
  的完整评测授权。首次报告脚本访问不存在的`.hashes`属性，旧尝试保留；新v2回执成功。
  全过程未读模型评分或GT值。原9b的794/6结果不改写，XD四个训练坏源仍缺失。
  产物在`code-1b6bd26/outputs/icassp2027/control/xd-raw-audit-20260918-b02`及同级
  `...-independent-proof`、`...-coordinate-consumer-check-v2`；本地摘要在
  `outputs/icassp2027/control/xd-four-bad-repair-20260918/b02_full800_seal_audit_20260919.md`。

## 两模型早期层的探索证据（2026-09-19）

- 新增[F04探索卡](decisions/findings/F04-early-layer-structure-exploratory.md)，状态明确为
  `exploratory_not_confirmed`。384个共享签名完整筛查后22条满足两模型raw/matched四CI同方向，
  都位于早期block2附近；P02 normalized只是同一rank缩放，4条P16 activation norm与P01每视频
  完全重复，不能计为额外性质。原F01/F02/F03确认身份不变，没有新确认或插件决策。
- 加入每视频8clip平均亮度标准差的fit-normal三分位控制后，block.2.input的有效秩差为
  Time −1.0005 [−1.8112,−0.1940]、V2 −2.3096 [−3.4113,−1.2107]；local cosine差为
  +0.03146 [+0.01464,+0.04818]、+0.05569 [+0.03963,+0.07169]，均为matched原始差值。
  新匹配保留Time59/56与V2 59/52视频，估计人群已变化；这是事后fit敏感性，不是因果控制。
- 进一步对全部20个共同且无head的站点逐video计算local−nonlocal cosine，不从两个CI相减。
  block.2.input的三字段matched增量为Time +0.01425 [+0.00319,+0.02503]、V2 +0.01988
  [+0.00824,+0.03138]；只有4个早期站点的raw g区间在两模型均为正，其余保留模型/站点差异。
  这不决定哪个token可丢弃。local指同原生时间索引、真实空间坐标曼哈顿距离1的采样token对。
- 两个诊断均保留10,000视频bootstrap、输入/脚本/输出SHA与覆盖；root独立重建88个匹配点值及
  40个派生组间点值/g，最大差4.44e-16/2.50e-15。三联PNG/SVG已实际查看；图明确标fit探索、
  不同原生窗口与匹配人数，不作正式论文方法图。证据和图均由F04列出ignored路径及SHA。
  初版输入反差诊断在修复前曾同路径覆盖，已明确无法恢复，未作为证据；只使用a02_verified快照。
- 只读cohort元数据审计确认原confirm161（80N/81P）中，两轮64 cohort互不相交，尚余33视频
  （16N/17P，ID集合SHA e71da9d0…16b54）。没有读取其统计或重新分配角色；source/scene分组未知。
  未来假设、校准及覆盖门槛必须在读取这些目标统计前冻结，不能用小样本结果调整门槛。
- 新CPU分析监视器最终a03 PID13537，绑定Time/VJ两个原观察guard并按各自真实成功流水分析。
  control为`code-0192fad/outputs/icassp2027/control/observation-analysis-0192fad-20260919-a03`；
  输出仍分别名为`timesformer-local-attention-fit128-20260919-a02`与
  `vjepa2-fit128-video-bootstrap-20260919-a02`。旧a01/a02在waiting且无分析子树时已核验退役，
  原文件和回执保留。新CLI在worker同进程执行，取得CPU lease后要求20GiB可用RAM，5秒采样
  worker+monitor RSS；16GiB/4GiB阈值仅停止自有分析。该守护不代表分析已完成，尚待真实产物。
- node3最新完整快照epoch1789748471：Time attention800/1024、VJ待lease；七路dense仍活：
  V2 fit214/select149、Time fit226/select149、VMA fit306、VJ fit131/select78；VMA原select79保留。
  被取消的VMA duplicate-copy receipt仍写running是陈旧状态，其guard cancelled/native退出才是实际状态。
- 后续启动预算再次实测缺715,378,688 bytes，依原条件授权只清理inactive HERMES的
  `weights/hermes-llava-ov-0.5b/model.safetensors`（1,787,445,680 bytes，SHA
  `07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e`），
  active配置/可读计算进程未引用，非symlink且单硬链接。配置、tokenizer、源码、登记、缓存保留，
  该inactive模型未来需恢复权重。回执`control/inactive-hermes-weight-cleanup-20260919-a01/receipt.json`，
  free由75,936,354,304增至77,721,411,584 bytes；没有把磁盘清理称为释放显存。

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




## 2026-09-22 08:30 当前六格压缩评测状态

六格 dense UR-DMU `official_frame` prediction 已全部完成：UCF/XD × VideoMAEv2/VideoMAE/TimeSformer 均有 `result.json`、`predictions.jsonl` 和 query-chunk receipt，且 `official_frame_scores_read=false`。node3 上 18 个正式压缩 extraction 仍在写入 shard，覆盖三 encoder × 两数据集 × `pair_select/group_uniform/group_random@0.60`；当前尚无压缩 `result.json` 或 `extraction-contract.json`，因此质量 scheduler 尚未启动压缩 prediction 或 export。node2 另有 8 个独立补充 extraction（六格 `pair_select` 加 V2-UCF 两个同预算对照）持续产出，输出根与 node3 隔离。当前未读取正式 test 分数做方法选择；下一门仍是压缩 extraction 合同完成后自动评分并导出 UCF ROC-AUC、XD 梯形 AP/step AP 和 paired bootstrap CI。
## 2026-09-22 08:30 当前六格压缩评测状态

六格 dense UR-DMU `official_frame` prediction 已全部完成：UCF/XD × VideoMAEv2/VideoMAE/TimeSformer 均有 `result.json`、`predictions.jsonl` 和 query-chunk receipt，且 `official_frame_scores_read=false`。node3 上 18 个正式压缩 extraction 仍在写入 shard，覆盖三 encoder × 两数据集 × `pair_select/group_uniform/group_random@0.60`；当前尚无压缩 `result.json` 或 `extraction-contract.json`，因此质量 scheduler 尚未启动压缩 prediction 或 export。node2 另有 8 个独立补充 extraction（六格 `pair_select` 加 V2-UCF 两个同预算对照）持续产出，输出根与 node3 隔离。当前未读取正式 test 分数做方法选择；下一门仍是压缩 extraction 合同完成后自动评分并导出 UCF ROC-AUC、XD 梯形 AP/step AP 和 paired bootstrap CI。


## 2026-10-16 用户授权的 AutoAI/VAD 存储清理（不改实验）

用户明确要求清理 AutoAI 无用生成权重与 VAD 老旧特征。只在 node3 的 /users 分区清理经核验的旧生成物：DSANet 20260923 head-training-r01 的 ucf-seed235、xd-seed234、xd-seed235 三个 completed run 各 epoch-01 至 epoch-09.pt，共27个中间训练checkpoint、32967780636 bytes；原固定最终轮选择契约、3个final.pt SHA均核对通过，final.pt、epoch-10.pt、全部配置/逐轮JSON/完成回执/日志及未完成ucf-seed234完整保留。AutoAI只删除llm-policy-20260930/pytest-related下test_*旧测试临时文件1081247040 bytes，不删真实实验、输入、数据库、报告或登记Qwen原始/量化权重。实际删除7864文件，逻辑34049027676 bytes（31.71 GiB），分区可用由354155212800升至388219510784 bytes，df约330G→362G（共享盘同期写入会影响差值）。

精确清单与删除流水账：/users/fotile/work/storage-cleanup-20261016-autoai-vad/{manifest.json,deletion-journal.jsonl,completed.json}；本地回执work/storage-cleanup-20261016-autoai-vad/completed.json。node2/node3未见候选文件的本用户进程/打开文件引用；清理前后四个正式提取PID11922/20824/27794/28618及start_ticks一致，两个既有LLM服务11616/26335继续。没有停止/重启实验，没有改campaign/head/源/输出/Swan身份；/data2当前约75GiB输出未触碰。/users下历史大头实际为checkpoint而非旧特征，未以“旧”名义删可追溯特征、失败记录或上游权重。已删中间epoch无法再用于对应轮恢复；最终模型和最后一轮恢复点仍在。未commit/push；仅记录运维，不运行ML实验。
