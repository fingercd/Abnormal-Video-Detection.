# 新任务交接清单（七点澄清后按授权移交）

本文件准备新任务的执行交接，不表示新任务或新Goal已经建立。用户于2026-09-18在审阅后
提出七点澄清，并明确允许解释后直接新建任务并以Goal执行。必须先读
`decisions/property-first-method-contract-v3.md`：旧paired_random/pair_linear降为工程/对照，
最终免训练/可训练方法必须从确认性质推导并同根同源，不以当前两原型直接冒充论文方法。

用户最新进一步选择A，优先LoRA：固定同一性质压缩规则，冻结ViT base和检测后端，
训练ViT内部低秩适配参数。要有dense同预算LoRA控制及梯度/更新/重载/合并身份验证。
不新增selector网络，也不预先锁rank或损失。旧A/B并列说明以此新选择更新。

## 工作目录与入口

- 原工作区：`D:/PythonProject/VAD`，分支`qzt/icassp2027-evidence`。
- 新实现冻结提交：`866674c2040e49190c98656c5e70be88e678ba5b`；后续计划文件提交另记。
- 先读`AGENTS.md`、`RESEARCH_PLAN_V2.md`、`progress.md`、`decisions/official-detector-protocol-v2.md`。
- 新Goal正文：`GOAL_NEXT_TASK.txt`。新任务创建后按用户明确授权设立该Goal，不缩为准备脚本。
- 不读`VAD_Idea/`等旧方向决定本轮研究；原未跟踪的该目录及两个根临时脚本保持原样。
- 新worktree不含ignored大资产。复用原工作区已存在的`.venv`和outputs，服务器路径为实际
  资产来源；不要因worktree缺weights/outputs就重新下载所有模型或重建环境。

## 新后端实现与验证

`paper/urdmu_backend.py`按五份上游Git blob SHA加载原类/损失，仅转换六处硬编码CUDA常量。
原仓库commit40cfdf5；LF与CRLF均只经明确换行归一化后校验，raw文件摘要另存。
原官方两checkpoint已weights_only＋strict加载。新`official_training.py`为完整官方训练
视图中心producer/consumer；`official_extraction.py`只负责完整dense提取；
`urdmu_training.py`真实FeatureDataset读取、一次200bin缓存、固定原损失训练和最终QA。

**尚未实现的新接口：UR-DMU正式test评分/quality export/完整detector计时。** 旧
`evaluation.py`、`quality_export.py`等正式controller仍绑定TopKMIL/20epoch/fit1288，
不能直接使用或删来源校验来冒充新后端。新任务需增加窄范围适配，复用已有预测记录、
帧级评测和quality纯统计核心；`_feature_contract`的真实dense采样核验已经可共享。

根代理本机：新模块首轮51项联合测试通过；补充来源内容绑定后的提取器6项通过；
LF/CRLF修复后的backend＋训练30项通过。现有插件/部署/梯度/位置回归103项通过。
Ruff和compileall通过。它们不能替代服务器真实四encoder闭环或正式质量结果。

## 服务器固定路径

- 新代码：`/users/fotile/icassp2027-runs/code-866674c`，已clean checkout、复制既有assets.local、compileall。
- bundle SHA：`59e93a82e86b54471d32db64a71deb2c2e7abee8a25d5bc1c195565226ca34f2`。
- UCF fulltrain view：`/users/fotile/VAD-icassp2027/outputs/icassp2027/control/official-fulltrain-20260918-r01/ucf-view/contract.json`。
- contract SHA：`47467e6c88497fbb8f81bc7e2be59cc9e7ed0b6076cdf1548f0e1fe1be5c6998`。
- training manifest同目录`official-fulltrain.jsonl`，SHA `30878c1a07e07c2be0493f017f40289c4329ec16912fcd640660b5e1988274b5`。
- 新独立UCF audit实际1610成员CRC/SHA/size/container probe全通过，consumer复验通过。
- XD official3954 inventory同control的`xd-view/blocked-inventory.json`；当前可用2800，缺1154，不能生成完整训练合同。
- 固定gate：`/users/fotile/icassp2027-runs/code-5e95107/outputs/icassp2027/control/target-runtime-gate-20260918T120000Z/calibration/<encoder>`。
- 模型原生环境从`/users/fotile/VAD`的environment registry解析，不猜解释器和overlay。
- 原UR源码服务器本地快照：`/users/fotile/icassp2027-runs/vendor/UR-DMU-local-40cfdf5`。
  五份源码逐byte等于Git blob；跨OS带来的Git index stat标记需与实际diff区分，不通过
  修改源码或strict=False规避真实权重差异。

## 运行中的工作与禁止重复派发

旧TopKMIL正式评分/重复队列保持research hold；旧source coordinator已经退出，
旧PID可能复用，不要按历史PID杀进程。健康旧32clip提取产物保留为工程证据，
不冒充新完整dense输入。原VJ dense/random被旧独占guard中断的失败回执保留，不复活旧队列。

当前XD下载使用三个独立writer：test105262、train2805 105261、train3320 83455是最近
启动ID，必须重新核对`/proc`完整命令/owner与进度后使用，不把这些数字当永远有效。
ROOT维护的poll脚本位于ignored `outputs/icassp2027/control/poll_xd_current_20260918T074000Z.sh`。
下载源与凭据不进入Git；旧partial、CRC失败和分块证据保留。raw test auditor最近为
node3 PID20108，等待完整test卷，不访问模型分数。

真实四配对小规模验收的脚本在本机
`outputs/icassp2027/control/native-urdmu-gpu-acceptance-20260918/`；计划control为
`code-866674c/outputs/icassp2027/control/native-urdmu-acceptance-20260918T085000Z`。
验收只用Assault038_x264与Normal_Videos157_x264两个旧fit短视频，四方法完整dense序列、
UR两步工程训练及同一冻结head的四路径。**四组实际全部完成、guard-a02 exit0**。
本地summary.json已按aggregate与各receipt SHA逐份复验；aggregate SHA为
`16fa0a7a56b7fe4084e2fee2f4633ca0db6da7d87792ca2eaf016929adbcaa6c`。
四组每条方法各20/42/20/5 clips，合计348次提取，仍仅两个独立视频。
开始新任务时核对实际回执，不重复这项已完成工程验收；正式模型测试分数当前未访问。

## 资源与交付边界

node3 RAM最近可用约1323GiB，node2约99GiB；共享/users最近剩余183GiB，需阶段预算。
未删除旧encoder资产。只在实际容量不足、路径和引用均核实后考虑用户授权的条件清理，
不得影响active四模型环境或其他任务。资源快照在上述native acceptance本地control。

旧V100参考计时四模型96配置/2880samples及2184份执行回执已审计；4个负端到端配置和
72个非dense峰值allocated未降保留。计划中的A100完整detector效率仍待执行。

用户确认计划后：创建新的VAD任务并携带本计划、Goal正文及本交接；确认新任务已接手并
设Goal后，再停止旧任务的自动Goal续行，避免双重调度。不要把旧研究Goal标为“完成”来迁移。
