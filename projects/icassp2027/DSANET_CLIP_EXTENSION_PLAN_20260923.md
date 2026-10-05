# DSANet 强 WSVAD 检测器扩展计划：完整测试集四档质量与效率评测

> **2026-09-23 用户执行范围覆盖：**不再训练 DSANet 检测头。UCF/XD 各固定使用一份作者发布的
> DSANet checkpoint，对本轮已封存的同协议 raw CLIP dense/删除20%/40%/60% 特征直接评分；
> 八格内不挑 seed 或 checkpoint。下文关于本地 full-train head、两 seed 及其门槛的安排保留为
> 原计划历史，不再阻挡本轮评分。2026-09-20 UCF 作者预提特征+发布权重的复现结果只作独立
> 校准参考，不能与本轮 raw 特征计算压缩差值；XD 作者预提特征尚无同等级本地回执。

日期：2026-09-23（Asia/Hong_Kong）  
状态：**用户覆盖后的发布权重八格评分、质量/定位导出和编码器/检测头/完整视频效率已完成；结果见 [`RESULTS_PUBLISHED_20260923.md`](../../work/dsanet-extension-20260923-r01/RESULTS_PUBLISHED_20260923.md)。下文“本地完整训练”执行叙述是旧范围历史。**  
选择：**DSANet，AAAI 2026，CLIP ViT-B/16 + DSANet 原生检测器。**  
最低正式矩阵：**UCF-Crime、XD-Violence 各 dense / 删除20% / 删除40% / 删除60%，共8次完整测试集推理。**

## 1. 结论与本轮目标

采用 DSANet 作为新增的强 WSVAD 方法，检验 PairSelect 插入其 CLIP 视觉编码器后，能否在较强检测基线上保持质量并获得真实计算收益。用户已明确具体选哪个模型无所谓，因此不再把三个候选逐一跑 test 后选高分者。

DSANet 论文为 *Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment*，发表于 AAAI 2026。作者报告 UCF-Crime AUC 89.44%、XD-Violence AP 86.95%；它是本轮选定的近期 SOTA 级 WSVAD 方法。该称谓指有公开论文、官方实现和较强公开成绩的代表方法，不宣称已穷尽截至2026-09-23的全部工作并证明其绝对排名第一。[AAAI 正式论文](https://ojs.aaai.org/index.php/AAAI/article/view/38191)、[官方实现与成绩](https://github.com/lessiYin/DSANet)。

本实验保留 DSANet 原生时序建模、正常性建模、文本与分类分支。不能把“CLIP 特征接 UR-DMU”写成 DSANet，也不能用作者预提取特征测出的检测头速度代表 CLIP 编码器加速。

实际执行按以下依赖推进：输入协议与部署核验 → CLIP 动态 PairSelect 接入 → 双数据集四档测试**特征**提取 → 完整dense训练与冻结 → 八格检测头推理/质量评分 → 独占效率测量 → 结果审计与论文增补建议。先提测试特征不读取测试分数；训练检测头仍只用完整训练集。当前论文和旧实验身份保持原有状态。

### 2026-09-23 现场执行补充

- **时钟注记（2026-09-23晚复核）：**node3的墙上时钟约落后宿主时钟9分半，`timedatectl`显示NTP未启用。以下由node3 `date`产生的启动/运行时间均为**未与宿主校准的node3本机时间**；相对持续时间以进程单调计时为准。不要为了对齐回执改系统时钟或改写旧日志。
- 目标node3 `ibnode3`：8张A100各绑定一个任务。`test-campaign-r02` 的node3本机时间16:26:39.016释放启动屏障，八个进程的本机开始回执落在16:26:39.098–16:26:39.137（约39毫秒窗口）。GPU0–3对应UCF dense/keep.8/.6/.4，GPU4–7对应XD相同四档；启动后8个PID均存活并已开始处理视频。
- 服务器输出根：`/users/fotile/icassp2027-runs/dsanet-extension-20260923-r01/test-campaign-r02/`。首次 `test-campaign-r01` 因dense两路启动脚本空数组错误没有完成8路启动；6个本轮进程已按PID核对后SIGTERM，失败日志/部分输出保留，不作为结果。
- 特征入口在同一冻结源码、FP32与batch128下运行。GPU0训练视频短测的batch128吞吐高于batch32/64；节点当时32核CPU负载约70、CPU空闲约6%，八路各用1个解码worker以控制竞争。此设置是生产提取配置，不替代后续GPU独占正式计时。
- 完整来源清单已经核验：UCF train1610/test290、XD accepted train3950/test800；官方测试CSV全清单显示UCF全部crop5、XD全部crop0。清单SHA及每路源身份位于服务器 `manifests-r02/summary.json` 和结果根 `resolved.json`。
- 新CLIP bridge的batch内动态PairSelect通过本地3项测试；真实权重训练帧smoke的dense/identity最大绝对差约`1.49e-6`，三档suffix长度158/119/79。UCF、XD各1个训练视频的十crop提取冒烟完成。以上不是正式质量或效率结果。
- 16:26:39 HKT启动的 `test-campaign-r02` 已8/8完成：UCF每档290视频/69,634图像，XD每档800视频/146,449图像。服务器 `sealed_features.json` 的状态为 `ready_for_fixed_head_scoring`、`test_scores_read=false`；UCF290/290原始帧数与官方评测时间轴精确一致，XD800/800均覆盖官方canonical前缀。两数据集的单路提取wall约20.6分钟和45.7分钟，为8卡共享CPU/存储生产工时，不是独占效率。
- 训练完整视图为UCF1610、XD accepted3950，十crop共16,100/39,500特征行。`train-features-r01` 于node3本机时间17:12:45.941释放第二个8卡屏障，GPU0–3四分片UCF、GPU4–7四分片XD；8/8训练提取PID已实测存活。所有正式检测头必须等对应完整训练特征收齐后才启动。
- 训练侧UCF/XD各一条真实视频的ten-crop提取、分片/resume、模型前反向、固定训练入口单步smoke均通过；发布权重仅用于训练视频上的评分入口shape冒烟。不会将这些smoke结果作为正式分数或检查点。
- 两数据集各3个独立进程的准备好GPU图像→视觉encoder计时已完成，覆盖batch1/32/128、20次预热、40次同步测量。batch32三档倍率UCF `1.096/1.222/1.381×`、XD `1.093/1.221/1.383×`；batch1各档略慢，显存高压缩档未稳定下降。原始记录和摘要在本地 `work/dsanet-extension-20260923-r01/benchmark/`；完整pipeline与检测头计时仍待做。
- 四个dense DSANet head的后续任务已登记：两数据集各seed234/235，均只在本数据集完整训练特征上训练固定final。主八格绑定seed234；seed235仅在dense/删除40%各跑一次用于独立训练变异核查。为保持每卡一个本轮任务，四head全完成后才同步启动主八路head评分，主八路退出后再启动四路seed235复核评分。当前只是调度入口，四head和评分均尚未产生数值。
- 训练特征接力按数据集独立：UCF四片完成并退出即可生成UCF CSV、启动UCF两seed，不等待XD四片；XD同理。只有**正式八路评分**保留四head全部ready的联合屏障。2026-09-23 node3本机17:38前已用新版守护PID `16331`替换原全局8片等待脚本，8个提取作业未中断。
- 四head评分门新增文件级核验：每个实际加载模型的总/可训练/冻结参数量、固定epoch final文件字节数与SHA由本轮训练记录；独立检查UCF16100/XD39500训练CSV身份、两seed非同一权重、无test选模，再发布 `head-qa.json`。该回执未出现前不得启动8路正式head评分；此规则在当前等待守护中已实现，仍待真实full-train验证。

## 2. 读完现有论文后的判断

已阅读当前工作树中的 `manuscript-20260922/main.tex`、Introduction、Related Work、Method、Experiments、Scope、Conclusion，以及质量表、README和近期 progress。当前论文主张是 frozen detector 上的 training-free token selection，压缩发生在12层编码器的前6层之后，不改变视频观察覆盖和既有 dense 检测头。

| 现有事实 | 对新增实验的要求 |
|---|---|
| 三编码器均配 UR-DMU，主要验证同一检测器族 | 新增完整 DSANet 检测器，补充跨检测器证据 |
| UCF dense AUC 分别为80.63、80.55、78.68 | 在强基线上验证质量保持；不能以这些分数归因编码器必然较弱，特征与训练协议也有影响 |
| XD 现有主表是梯形 PR-AUC，另有非插值 AP | DSANet 的公开 AP 与其官方实现对齐，两个定义分别保留 |
| 原有 batch32 表只计准备好的GPU输入到encoder输出 | 新增端到端和检测器阶段计时，避免把encoder收益扩大为整个系统收益 |
| 当前已有前期接触 test 和部分预算运行环境不一致的披露 | 新线保留这段历史；本轮严格固定实现、精度和三档预算，不能宣称整个项目重新变成完全盲测 |
| 当前限制包括外部压缩方法尚未比较 | DSANet 扩展不能自动补齐 ATS/ToMe 对照，仍需诚实保留限制 |

评价目标是“强检测器 + 压缩”的质量—效率权衡。只有本地 dense 也达到有说服力的质量，才进一步主张“在高质量检测系统上验证”。加入一个强方法的架构本身，不保证本地新输入协议达到论文分数。

## 3. 服务器资产与现有能力：事实和未知分开

### 3.1 2026-09-23 15:48 HKT 起的只读核验

实际连接：`ssh -p 12345 fotile@121.196.228.153`，目标主机回显 `ibnode3`。使用 BatchMode 与有限连接超时，没有安装、启动GPU计算或修改远端文件。

公共资产根：

```text
/data2/localdisk/fotile-icassp2027-kimi-20260919-a01/clip-assets
```

| 资产 | 当前证据 | 本轮用途 |
|---|---|---|
| DSANet UCF/XD 权重 | 两个文件存在，重新计算SHA与历史台账一致 | 作者发布检查点校准、输入兼容性核验 |
| VadCLIP UCF/XD 权重 | 两个文件存在 | 保留既有资产，本轮不增设第二模型 |
| 视频异常识别版 AnomalyCLIP | 官方weights目录为空；历史为ShanghaiTech自训替代 | 不纳入本轮矩阵；不能称三个方法均已具备双数据集正式模型 |
| OpenAI CLIP ViT-B/16 | 文件存在，重新计算SHA一致 | 固定视觉底座 |
| `clip-vad-v1-gpu/bin/python` | 路径存在；历史环境为Python3.9.23/torch2.0.1+cu117 | 后续先实测导入与CUDA，不重装共享环境 |
| DSANet历史UCF校准 | 本地回执记录89.4446 AUC；远端校准日志存在 | 是旧“官方特征+官方权重”证据，不是本轮raw重跑 |
| DSANet XD全量复现 | 权重存在，未发现相同等级的已完成校准回执 | 明确待验证，86.95仅是作者报告值 |

核实SHA：

```text
dsanet/weights/model_ucf.pth
cc674979901e2ed7153161e2cd4888ecacd25966e18caaba8472d993aedd47e0
dsanet/weights/model_xd.pth
78317c34fb8806d16bffe432946dd1b4cb6c6f3cc79ab29151163f6093febb39
openai-clip/weights/ViT-B-16.pt
5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
```

DSANet 历史登记源码revision为 `eb335b23fd6f01810bcd176c948c10348764a504`。现场 `git rev-parse` 返回“not a git repository”，因此只称其为台账revision，不能称本轮已由Git确认。实施前对实际源码文件做内容摘要，连同本地未提交修改形成独立快照；不伪造Git身份。

现场A100 GPU5/6显示0 MiB，其他卡有占用；这是时间点快照，不是预留。`/users` 所在文件系统约剩687G，`/data2` 约剩4.8T。启动前重新核对GPU进程归属与可用磁盘。

### 3.2 输入协议的重要更正

远端 DSANet CSV 的行数减去表头为：UCF train 16100、test 290；XD train 39540、test 800。训练行数包含多crop，不能当作独立视频数。当前论文正式训练身份仍为 UCF 1610、XD accepted3950；XD 原作者3954视频与本地排除4个视频的差异必须披露。

测试清单首部显示 **UCF为 `__5.npy`，XD为 `__0.npy`**。VadCLIP 的 `crop.py` 对应：crop0为340×256 resize后的224×224中心裁剪；crop5为该中心裁剪的水平翻转。因此：

- 不能把“训练资产有10种crop”解释成“正式test必须10-crop平均”。
- 正式执行前逐行检查完整CSV的crop分布；首部抽查不能代替全清单审计。
- UCF/XD各自冻结测试crop规则，四档之间完全一致。
- OpenAI默认保持长宽比的resize/center-crop，与作者先resize到340×256的协议不同，不能互换。

历史raw对齐记录还未确定作者每个16帧组究竟取哪帧、是否组内平均。只有维数相同或cosine较高，不能证明与作者特征同身份。该缺口同时影响DSANet，因为它官方复用VadCLIP特征。[DSANet数据准备说明](https://github.com/lessiYin/DSANet#-data-preparation)

### 3.3 代码现状

可复用：`src/vadbench/token_reduction/bridges/clip.py`、`scripts/icassp2027/probe_clip_train.py`、现有FeatureStore、manifest、时间轴评估与质量导出思路。

历史CLIP工程smoke证明的是固定局部选择、`197→99`、512维读出与identity一致性；当前bridge使用共享 `index_select` 索引，**还不是本计划的逐样本动态PairSelect三档部署**。`score_urdmu.py` 等专用入口也不能原封不动当DSANet评分器。本计划中的新入口均须实施后检查真实 `--help`，不提供假装已经可用的运行命令。

## 4. 模型与数据协议：先固定，再运行

### 4.1 训练与发布检查点的角色

正式新增主线遵守仓库完整训练约定：**CLIP视觉骨干冻结；用对应数据集全部可用训练视频的dense特征，分别训练DSANet原生可训练模块。** UCF用1610个视频；XD用accepted3950个视频。保留正常性模块、文本适配器和原生损失，不替换为UR-DMU，不跨数据集共用检测头。

每个数据集只训练dense检测器；训练结束后固定其检查点，dense与三档压缩共用它。压缩本身不训练、不finetune，不为20/40/60分别重训检测器。这仍是 `direct_insert`，训练free指压缩步骤。

作者发布检查点及历史89.4446分数作为独立的发布模型参考。它们不能替代本地raw完整训练回执；尤其不能把作者3954训练身份写成本地3950身份。若后续明确采用纯发布checkpoint实验，须单列实验身份与上游检查点选择历史，不能冒充本计划的本地full-train结果。

### 4.2 raw输入路线与有界对齐

优先用已部署作者源码、发布特征和训练视频恢复作者原提取协议；这项调查以一个工作时段、最多2小时为上限，超时不继续猜测和比test成绩。

| 路线 | 成立条件 | 表述 |
|---|---|---|
| A：作者协议可追溯 | 能确定解码、frame indices、每行聚合、crop、normalize、精度及读出，并在训练视频逐行比对 | DSANet按可追溯作者表征训练/评估；仍说明本地训练集与固定final checkpoint的差异 |
| B：新raw协议（A不可证实时默认） | 使用下列显式协议并全量训练dense DSANet | DSANet在本地raw CLIP表征上的复现，不声称作者特征字节等价 |

路线B预设，需在任何新test前写入冻结文件：

1. 使用当前已核实的原视频根、完整训练/测试清单与实际解码帧序号；不改变已有raw→canonical映射。
2. 以连续16帧为一个组，stride=16；代表帧为组中下中位帧，完整组取零基索引7，末尾不足16帧取实际剩余帧的下中位帧。每组保存代表帧索引、组区间和valid长度。
3. 每帧先按作者 `crop.py` resize340×256，再224×224 crop、CLIP normalize；训练默认沿用10个crop身份，UCF测试crop5、XD测试crop0，完整CSV审计有不同时先记录差异再冻结。
4. 每个代表帧/crop调用一次CLIP视觉塔，保留native `CLS→ln_post→proj` 512维输出，不擅自新增特征L2 normalize。
5. 默认视觉提取FP32、TF32关闭，检测器FP32，特征存储FP32；四档完全相同。若训练侧确定需用其他精度，则在test前统一修订并补identity，不能只给compressed换精度。
6. 代表帧取法是本轮新协议，不称已恢复作者16帧取样。作者口径要求完整16帧组时，评测适配器明确投影/截取其支持区间；全帧canonical口径另外覆盖尾帧，二者分开报告。

对齐和工程验证只使用训练视频。64 normal +64 anomalous仅允许筛查或验证，不能作为正式full-train。

### 4.3 训练细节与禁止test选检查点

已读远端 `ucf_option.py`、`xd_option.py` 与 `xd_train.py`。默认配置参考如下，最终以冻结源码和resolved config为准：

| 项目 | UCF | XD |
|---|---:|---:|
| visual_length / feature_dim | 256 / 512 | 256 / 512 |
| temporal visual_layers / attn_window | 2 / 8 | 1 / 64 |
| classes_num | 14 | 7 |
| 原生batch-size | 64 | 96 |
| 初始学习率 | 7e-5 | 1e-5 |
| 固定epoch预算 | 10 | 10 |
| DNP_use / num_prototypes / decoder_depth | true / 16 / 8 | true / 16 / 8 |
| text_adapt_until / t_w | 3 / 0.1 | 1 / 0.6 |
| score temperature | 5.0 | 1.0 |
| 主seed / 复核seed | 234 / 235 | 234 / 235 |

保留原生优化器、参数分组、损失权重、学习率调度；不能把UR-DMU的3000步、200bins直接搬来。UCF优化器和完整训练循环需实施前继续核实，当前不声称已经逐行审计。

执行勘误（node3本机2026-09-23晚）：UCF作者 `ucf_train.py` 的refiner WarmCosine总长度按 `epochs×(len(normal_loader)+len(anomaly_loader))` 设置，实际优化循环按两loader较短者迭代。本轮为保证完整训练集每轮覆盖，实际循环改为较长者并循环较短loader，**调度总长度仍采用作者表达式**；两个数都写入训练resolved回执，不能说整个训练过程与作者逐步一致。XD原始总长度即`epochs×len(train_loader)`，与全数据轮次一致。修正在四head正式启动前同步。

**已确认XD原脚本每epoch访问官方test、按AP保存最佳并回载继续训练。** 本地新训练入口应复用模型/损失定义，但移除test loader、测试真值读取、按test保存、按test回载四条路径，固定第10个epoch最终检查点。仅删除打印不足以隔离test。UCF入口同样审计。两个optimizer与两个scheduler的续训状态必须完整保存；续训不改变数据/seed身份。

训练视频级类别标签用于DSANet原生任务，仍是视频级弱监督；应披露其类别语义监督不同于二元UR-DMU设置。推理可以使用固定类别词表，但不得让视频真值类别或带标签文件名进入选择器或决定某个视频的prompt。

至少保证每个epoch的真实unique video覆盖回执，检查正常/异常不平衡与 `drop_last` 是否导致视频长期遗漏。主seed产生8格主结果，复核seed至少完成dense和40%两数据集4格；两seed不择优，固定234为主表，另报变异。复核只需复用四档特征重跑检测器，不必重提CLIP。

## 5. CLIP内部PairSelect接入

### 5.1 固定规则

- 视觉底座：ViT-B/16，224×224，14×14=196个patch、1个CLS，hidden768、12blocks、输出512维。
- 插入点：完成blocks索引0–5后，在block索引6输入前选择；前6层保持197个token，后6层执行缩短后的序列。旧smoke的索引2不是此次正式层。
- 只压缩空间patch，CLS始终保留；每个图像/crop独立选择，不能跨帧或跨batch共享某个样本的动态索引。
- 水平相邻patch成对；按原生坐标顺序，每5对组成最多10token的组。配额沿用 `floor(r*n+0.5)`。
- 比较当前第6层输出的L2范数，使用论文的pair-member优先顺序，稳定处理并列值，选后恢复原生顺序。不平均、不重标定向量。
- CLIP绝对位置已在前缀加入，gather携带原内容/位置；不重复添加位置，不套用TimeSformer轨迹取整或人为锚位。作为CLIP bridge特性明确写入方法扩展。
- 选择、索引构造和gather均在GPU；不能把token送CPU排序后在计时中遗漏往返开销。

### 5.2 预算表与计算量预期

以下数字是**依照当前规则推导的目标，不是运行结果**。196个patch为19个10token组加1个6token尾组；特殊CLS不进入删除率分母。

| 删除比例（名义） | keep_ratio | 保留patch | 后6层总token（含CLS） | 实际patch删除率 | 12层principal-block GFLOPs/图像 | principal-block理论降幅 |
|---|---:|---:|---:|---:|---:|---:|
| dense | 1.00 | 196 | 197 | 0% | 34.895 | 0% |
| 20% | 0.80 | 157 | 158 | 19.898% | 31.327 | 10.224% |
| 40% | 0.60 | 118 | 119 | 39.796% | 27.816 | 20.287% |
| 60% | 0.40 | 78 | 79 | 60.204% | 24.272 | 30.442% |

计算使用 `C(N,D)=12ND²+2N²D` MACs/block，D=768，1MAC=2FLOPs；四档均为 `2×[6C(197,768)+6C(K,768)]`。这些值排除embedding、LayerNorm、激活函数、readout与selector，不是完整CLIP或DSANet的总FLOPs，更不是实测加速倍率。完整成本与selector需单独统计。

### 5.3 必过验证

1. 同输入同精度下native dense、keep=1 identity、关闭hook三者输出一致；FP32初始容差 `atol=1e-5, rtol=1e-4`，同时保存max-abs/relative误差，不据test调容差。
2. 用真实权重在两数据集训练视频验证197→158/119/79；逐层保存shape回执，证明后6层确实缩短，不是mask/清零或最终输出删维。
3. CLS唯一且未删除，选择索引无重复越界，batch内动态选择正确、稳定排序、尾组配额正确。
4. 原始512维读出、crop轴、时间行数、padding mask与检测器输入长度保持契约。
5. 连续调用dense/reduced不残留hook或错误复用mask；精度、train/eval状态与随机性可追溯。
6. 置换标签/文件名而像素保持相同，选择不变；选择器不可读测试标注、类别、未来层或额外dense教师输出。

## 6. 正式质量矩阵和运行顺序

### 6.1 最低必须交付8格

| Run逻辑ID | 数据集 | 测试视频数 | 删除比例 | dense检测器检查点 |
|---|---|---:|---:|---|
| dsanet-ucf-dense | UCF | 290 | 0% | 固定UCF final |
| dsanet-ucf-del20 | UCF | 290 | 20% | 同一UCF final |
| dsanet-ucf-del40 | UCF | 290 | 40% | 同一UCF final |
| dsanet-ucf-del60 | UCF | 290 | 60% | 同一UCF final |
| dsanet-xd-dense | XD | 800 | 0% | 固定XD final |
| dsanet-xd-del20 | XD | 800 | 20% | 同一XD final |
| dsanet-xd-del40 | XD | 800 | 40% | 同一XD final |
| dsanet-xd-del60 | XD | 800 | 60% | 同一XD final |

合计4360个video-setting推理记录，底层独立测试视频仍为1090。身份相符的本轮dense运行可复用；旧预提取特征上的head-only测试不能充当本轮raw dense，更不能代替四档视觉前向。共享一次视频解码用于四档可以节省离线成本，但记录为共享解码的提取作业，不拿它计独立端到端延迟。

冻结后按数据集推进四档，可并行不同GPU；一个数据集不必等另一集完成。每个FeatureStore分片单writer，合并后先QA再评分。全部质量格必须报告，不能读到dense分数低就换方法，也不能读到某档下降就删掉该档。

复核seed增加4次head推理。推荐同配额 `group_uniform` 与 `group_random` 在40%各做双数据集，共4次额外完整推理，检验新增CLIP下范数选择是否优于简单参考；它们独立于用户要求的8格，不能挤掉8格或显存/端到端指标。外部ToMe/ATS不在本轮最低交付范围内。

### 6.2 输入与预测验收

- 每档unique video集合恰为290/800，无重复、漏项、额外项；不以npy数或预测行数替代视频覆盖。
- 同数据集四档的代表帧、crop、时间区间、有效长度逐视频逐行一致；记录decoded frame数与encoder调用数。
- XD绑定现有raw→canonical变换；不同视频各自重置时间轴，不能只把所有分数拼接后截断到GT总长。
- 原生16倍repeat与canonical覆盖分别输出明确身份；末尾不足16帧、视频FPS/帧数差异、process_split边界都需测试。
- 特征与分数有限，dtype/维数正确；checkpoint、源码、权重、manifest、representation和预算指纹齐全。
- 写完所有结果后发布completion receipt；失败attempt保留，新attempt只补缺失分片，不覆盖旧成功资产。

## 7. 完整质量指标：每项都有去向

DSANet官方 `ucf_test.py` / `xd_test.py` 计算ROC-AUC、`average_precision_score`和定位mAP。因此XD作者表中的AP应按**非插值average precision**核对，不能用现有论文的梯形PR-AUC替代。[官方UCF评测源码](https://github.com/lessiYin/DSANet/blob/main/src/ucf_test.py)、[官方XD评测源码](https://github.com/lessiYin/DSANet/blob/main/src/xd_test.py)。实施时以本地固定源码SHA为权威，网页main仅作交叉核查。

| 类别 | 每档必交指标 | 口径 |
|---|---|---|
| UCF主指标 | frame ROC-AUC | 全部290视频；与冻结原生score分支对应 |
| XD主指标 | frame AP，非插值 | 全部800视频；绑定sklearn实现与版本 |
| 补充二元质量 | 两数据集均给frame ROC-AUC、非插值AP、梯形PR-AUC | 三列不同名称，不混写AP/AUC |
| 分支输出 | coarse与hierarchical/refined两套原生分支指标 | 主表预先固定coarse prob1；另一分支完整保留，不取两者最大 |
| 时序定位 | mAP@tIoU=0.1/0.2/0.3/0.4/0.5及均值 | 原生类别词表、正常类处理、proposal/NMS/阈值固定；不要只留一个平均值 |
| 质量保持 | 每项绝对值、method−dense差值（百分点） | 同预测身份与同时间轴进行配对 |
| 不确定性 | 主指标与差值的95% CI | 10000次按视频标签分层的paired video bootstrap |
| 非劣性 | CI下界与−0.005比较 | 内部0–1分数；−0.005等于−0.5百分点，不是−0.005百分点 |
| 训练稳定性 | seed234与235的dense/40%结果 | 主seed不变；视频bootstrap不声称涵盖训练seed变异 |
| 失败/退化 | 丢视频、NaN、OOM、未覆盖帧、各档下降 | 如实记录，不能只导出成功或正增益 |

bootstrap每次抽取视频后重组其全部帧，dense/reduced使用同一抽样序列；重复抽中视频保持重复实例，定位指标的video ID需要独立副本以免合并。保存seed、有效重采样数及退化样本处理。六个压缩主比较的区间默认是逐项95% CI，不据此宣称联合95%保证。

F1/precision/recall/FPR属于有阈值指标：若纳入，须在允许训练开发数据上预先固定阈值，或直接沿用有明确来源的作者固定阈值，并披露视频级校准不等于帧级最优阈值。不能在test扫阈值报best-F1。当前无合法阈值时，交付表仍保留这些字段并标 `not_applicable_no_frozen_threshold`；它不阻塞无阈值质量主表。原生定位mAP所需的固定proposal参数则必须冻结并完成。

通过条件分开解释：点估计下降≤0.5pp只是点估计达标；95% CI下界≥−0.5pp才是本计划预设非劣性支持。CI跨边际时报告“尚不能确认”，不能把“显著性不足”解释成已证明等价。

## 8. 完整效率指标：同时测encoder、检测器和端到端

### 8.1 三种计时边界

| 边界 | 包含 | 必交 |
|---|---|---|
| 视觉encoder | GPU输入→CLIP embedding、12blocks、selector、gather、native readout | latency、吞吐、token/FLOPs、显存 |
| 检测器 | 特征输入→DSANet时序、文本、DNP及实际执行分支→分数 | latency、显存、文本cache冷/热状态 |
| 完整视频pipeline | 打开原视频→解码→crop/normalize→H2D→encoder→聚合→DSANet→CPU分数 | wall time、video延迟、source-video秒/墙钟秒、端到端speedup、系统资源 |

端到端排除用于评测的GT加载和bootstrap，不默认写完整特征缓存；若离线作业写盘，另外记录“含FeatureStore写盘”总工时、字节数和I/O。冷启动包括模型加载与文本缓存生成，稳定推理另报，二者不能混合。

DSANet现场forward在 `DNP_use=true` 时确实调用refiner，eval含静态文本cache。计时默认保留原生调用路径，不能只在compressed路径删分支。固定静态文本缓存可以双方同等启用，但视频条件融合部分仍须计算；缓存是否正确失效需验证。

### 8.2 标准实验设置

1. 主硬件node3同一张A100-40GB，正式计时该卡无其他计算进程；记录GPU UUID、驱动、CUDA、torch、cudnn、时钟/功率设置、运行时温度与利用率。共享GPU提取日志只作生产监控。
2. encoder主表测batch=1和batch=32；batch指单帧crop图像数，明确与视频编码器clip batch不同。相同batch、同输入、同精度、同attention后端、同runtime比较四档。
3. 从每个数据集训练集固定16视频（8正常+8异常），按短/中/长时长覆盖，视频内确定性取帧；两数据集均评估其实际test crop。无test质量参与样本挑选。
4. 每配置20次warm-up，至少40个正式GPU同步测量，覆盖全部16视频；3个新进程重复，四档次序按进程轮换。使用CUDA events并在边界同步；wall-clock部分使用同步后的高精度计时器。
5. 端到端使用固定训练视频完整长度，warm-up与正式视频分离；四档各3次独立进程重复，交错次序。同等OS缓存条件；无法安全清理共享缓存时标warm-file-cache，不伪称冷盘。
6. 固定batch若dense OOM，四档统一降到预设下一档，保留OOM回执；batch32缺失须给原因并补共同可运行batch，不把compressed较大batch的吞吐冒充同batch加速。
7. 可选测最大可运行batch和V100泛化，但单列，不能替代A100固定batch主表。主矩阵完成前不扩大到更多压缩比例或层。

### 8.3 指标清单与计算

| 指标 | 单位/算法 | 防止遗漏或误报 |
|---|---|---|
| 每批/每图延迟 | ms，mean/std、p50、p95 | 同步测量；保留原始样本 |
| encoder吞吐 | crop-images/s、采样帧/s | 多crop时同时给调用倍数，不含糊称原视频FPS |
| 全视频时间 | s/video及全测试生产总wall time | 正式独占样本与共享生产工时分开 |
| 实时能力 | source-video秒 / wall秒，及倒数RTF | 说明使用完整视频或何种窗口，DSANet离线时序不称因果流式 |
| speedup | `T_dense/T_reduced` | encoder、head、端到端各自相同边界 |
| 延迟下降 | `1−T_reduced/T_dense` | 与吞吐提升百分比区分 |
| 吞吐提升 | `Q_reduced/Q_dense−1` | 1.25×对应吞吐+25%、延迟−20% |
| 峰值显存 | allocated/reserved MiB | encoder和完整pipeline分别测，batch1/32全部四档 |
| 显存节约 | `1−M_reduced/M_dense` | 保留为零或负收益，不能只给FLOPs |
| 显存基线 | 模型载入后allocated及工作集增量 | 权重不减少，前6层dense可能决定峰值 |
| 实际token | 每层token、patch保留比、序列N/K | 不用名义删除率替代实测shape |
| 理论成本 | principal-block MACs/GFLOPs | 与第5节解析式一致，明确1MAC=2FLOPs |
| 完整成本 | encoder、head、总pipeline可计算术GFLOPs | 列profiler未支持算子、selector/LN等漏算；混合解析补账有明细 |
| 插件开销 | norm、排序、索引/gather合计ms及占比 | 独立profiling；净时延测量也包含插件，不读取旧累计overhead字段 |
| 参数与存储 | 总/可训练参数量、checkpoint MB、feature GB | token selection不会减少权重，明确无权重压缩 |
| CPU/I/O | peak RSS、解码/预处理/传输/写盘时间、读写字节 | 分阶段时间可能重叠，不强行相加伪造wall time |
| 能耗（条件项） | GPU采样功率积分J/video、J/源视频秒 | 仅独占且采样充分才给数值，否则保留字段+不可测原因 |

显存用独立测量阶段：模型加载、warm-up后同步并reset peak stats，记录测量前baseline及 `max_memory_allocated/max_memory_reserved`；不同预算用新进程隔离历史allocator峰值。NVML/nvidia-smi进程显存作为旁证，低频采样峰值不能代替分配器峰值。延迟正式测量中不开profiler、完整attention dump或分析hook。

FLOPs、token删除率和实测speedup是三个不同结果。按Amdahl关系检查合理性：未加速部分包括前6层、解码、预处理和检测器，端到端收益通常受其限制。任何数值不符合分阶段时间时先审计，不解释为神奇加速。

## 9. 分阶段实施与交付

| 阶段 | 工作 | 完成依据 | 后续依赖 |
|---|---|---|---|
| P0 固定资产 | 环境实测、源码SHA、两权重、完整CSV/crop分布、raw清单、磁盘/GPU | asset_receipt、完整身份 | P1/P2 |
| P1 表征冻结 | 有界恢复A，否则冻结B；训练侧逐帧与crop/时间轴审计 | representation.json、训练侧误差/覆盖回执 | P3提取 |
| P2 压缩接入 | 复用PairSelect规则，支持CLIP逐样本动态gather | identity、shape、索引、无泄漏测试通过 | P4压缩提取 |
| P3 全量dense训练 | 两数据集完整dense特征；DSANet final epoch10；两个seed | train coverage、finite loss、final SHA、无test访问 | P4检测器评分 |
| P4 八格质量 | 主seed完整raw四档→预测→QA→封存→评分 | 8/8主格完成，指标齐全；复核seed另外4格 | P6汇总 |
| P5 效率 | 独占A100四档batch1/32、完整pipeline、显存/FLOPs/开销 | 同边界原始计时、GPU占用证据、资源表 | P6汇总 |
| P6 审计与解释 | 质量与效率合表、失败/差异说明、图表导出建议 | 溯源可复算，缺项有机器可读原因 | 论文修改另按实际授权范围 |

P1与P2为逻辑独立工作，可由主控按依赖推进；默认不增设多个agent。多GPU任务按实时资源并行，提取允许沿用已授权共享空余显存，但正式计时必须独占。不得停止他项目、重装环境或把单卡多进程数量当作有效多卡产出。

需实现/适配的职责边界如下，文件名为建议，**当前不声称存在**：

- CLIP部署：扩展已有 `bridges/clip.py` 与可复用selection接口，避免复制一份偏离论文的选择规则。
- raw提取：给现有manifest/FeatureStore补CLIP单帧crop及source-frame支持，保留所有来源信息。
- DSANet训练和评分适配：建议 `train_dsanet_backend.py`、`score_dsanet.py`，复用官方定义及项目QA，不改上游原目录冒充原版。
- 质量导出：复用现有视频级配对bootstrap和严格时间轴检查，新增原生mAP与分支字段，避免平行无provenance脚本。
- benchmark：同模型实例、相同前向路径测四档，独立输出同步GPU计时、wall time、显存和源码身份。

代码实现阶段运行受影响测试，Python改动后运行 `python -m compileall src tests`；重点覆盖CLIP identity、逐样本索引、DSANet模型加载、两数据集评估、16帧尾部、raw→canonical、禁止训练读取test、恢复训练状态。当前仅文档改动，无须运行GPU测试或把历史测试数字当本轮结果。

## 10. 时间与磁盘预算：先短测再报ETA

不能拿VideoMAE/TimeSformer提取速度或旧CLIP工程smoke秒数套算新任务。CLIP单帧特征、10-crop训练、DSANet完整训练和不同NAS吞吐会显著改变成本。

实施P0/P1后用训练侧小批次测三类速率：raw解码+各crop编码，dense特征级DSANet完整训练step，四档测试路径完整pipeline。保存每视频总帧数、特征行数、crop数，再估算：

```text
V_train_calls = sum_video(groups_video * train_crop_count)
V_test_calls  = sum_video(groups_video * test_crop_count)
T_train_extract ≈ V_train_calls / measured_dense_pipeline_rate
T_test_extract ≈ sum_r(V_test_calls / measured_pipeline_rate_r)
T_head_train ≈ epochs * actual_steps_per_epoch * measured_step_time
S_feature ≈ feature_rows_all_crops * 512 * bytes_per_value + index_overhead
```

两个数据集分别计算，train ten-crop与test单crop不能混用倍数。预留至少30%时间和磁盘余量；并行按实测NAS/CPU饱和情况计算，不把GPU数直接乘进吞吐。先交预计区间与主要瓶颈，再给剩余ETA，不承诺未测出的“几小时完成”。

若临近内部截止，优先P0–P6最低8格及完整核心指标；缩减可选第二seed之外的扩展对照/第二硬件，不减少测试视频、不漏60%档、不省略显存或端到端。无法按时完成就明确未完成，不能把SOTA参考值填入本地dense格。

## 11. 输出结构、可复算表与缺失处理

建议新根目录，实施时检查不存在再创建，不覆盖旧资产：

```text
work/dsanet-extension-20260923-r01/               # 本地报告与小型证据
  assets/asset_receipt.json
  freeze/representation.json
  freeze/experiment.json
  freeze/code_manifest.json
  qa/identity_and_shapes.json
  qa/coverage.json
  results/quality.csv
  results/efficiency.csv
  results/metric_status.json
  results/seed_check.csv
  RESULTS.md

/users/fotile/icassp2027-runs/dsanet-extension-20260923-r01/
  training/{ucf,xd}/seed-{234,235}/
  inference/{ucf,xd}/{dense,del20,del40,del60}/
  quality-exports/
  benchmark/
  logs/
```

大特征优先放已核实的 `/data2` 资产空间下新独立结果根，具体目录在磁盘核验后绑定；服务器路径只是建议，尚未创建。

`quality.csv`至少包含：dataset、model/representation/run ID、head checkpoint SHA、seed、ratio nominal/actual、video/frame/feature-row counts、frame映射协议、score分支、AUC/AP/PR-AUC、五个tIoU mAP及均值、dense差值、CI、非劣性状态、来源路径。

`efficiency.csv`至少包含：hardware UUID、dtype/backend、batch与crop单位、计时边界、warm/cold状态、token各层数量、解析/完整GFLOPs、latency mean/std/p50/p95、throughput、speedup、end-to-end wall、allocated/reserved baseline/peak、RSS、插件ms、源视频时长、参数/文件大小和证据路径。

每项都有 `status = measured | analytical | not_applicable | failed | pending` 与reason。待测值使用null，不写0，不把理论值标measured。必交质量、显存、吞吐、延迟、端到端或成本账本仍pending/failed时，不能宣布本扩展完整完成；条件项能耗/F1可带明确原因交付。

主结果表建议：

| 数据集 | 配置 | AUC | AP(step) | PR-AUC(trap) | 主指标Δ+95%CI | mAP均值 | encoder speedup B32 | B1 p95 | E2E speedup | peak allocated/reserved |
|---|---|---|---|---|---|---|---|---|---|---|
| UCF/XD分别4行 | dense/20/40/60 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

完整导出另保留全部分支、各tIoU值和各seed，不能因为论文版面小而删除原始指标。

## 12. 论文如何接纳新证据

完成后可把检测器范围扩展为“UR-DMU与DSANet两类检测器”，视觉底座新增CLIP ViT-B/16；VadCLIP/DSANet/AnomalyCLIP共享visual不能算三种新增encoder。

质量表增加DSANet在两个数据集的dense/三档结果，XD明确step-AP与旧梯形PR-AUC差异。效率表标CLIP batch单位为图像crop，不能按clips/s直接给它和16帧视频编码器排名。若报告pipeline提升，必须来自本轮端到端计时。

保留raw协议、full-train范围、作者检查点与本地final策略差异、历史test暴露和seed数。出现低dense、60%质量下降、batch1变慢、显存不降，均保留并收窄主张。不能通过重跑test换层、换crop、换温度或挑seed解决。

“SOTA上的压缩”需要强dense复现证据支撑。最终若只有DSANet架构但本地分数明显偏离公开水平，应写“DSANet架构扩展”，说明偏差与训练/输入协议限制，不能把公开89.44/86.95贴到本地实验上。

## 13. 完成清单

- [ ] DSANet双数据集资产、实际源码/环境、输入crop和时间轴身份固定。
- [ ] 作者表征对齐状态有结论；不完整时新raw协议明确且全训练集训练完成。
- [ ] CLIP动态PairSelect三档真实缩短，identity/CLS/读出/无泄漏测试通过。
- [ ] UCF1610、XD3950完整dense训练；固定final checkpoint，禁止test选模。
- [ ] 主seed8/8完整测试推理与QA完成；四档共享每数据集同一检测器。
- [ ] AUC、step-AP、梯形PR-AUC、原生分支与五档定位mAP齐全。
- [ ] 主指标与dense差值、10000次视频bootstrap CI、非劣性判定齐全。
- [ ] 复核seed的dense/40%独立训练及预测回执完成并单列。
- [ ] 同A100独占batch1/32延迟、吞吐、插件开销、每层token/成本账本齐全。
- [ ] 完整pipeline延迟、速度倍率、峰值allocated/reserved与CPU/I/O记录齐全。
- [ ] 冷/热启动、文本cache、输入crop倍数、理论/实测区别已披露。
- [ ] 失败、缺项与条件指标都有状态；报告没有未标注的空格或模拟数值。
- [ ] 审核导出与实验身份可追溯；更新progress；论文增补建议基于本轮实际结果。

## 14. 证据导航

本地现有来源（相对于本文件）：

- [当前论文](manuscript-20260922/main.tex)、[方法定义](manuscript-20260922/sections/03_method.tex)、[实验与计时口径](manuscript-20260922/sections/04_experiments.tex)、[质量表](manuscript-20260922/tables/quality_rows.tex)。
- [当前进度](progress.md)、[完整训练范围](TRAINING_SCOPE_AND_RUNTIME_ESTIMATE_20260921.md)。
- [历史CLIP资产台账](../../work/codex-takeover-20260920/clip/asset_inventory.md)、[raw对齐决策草案](../../work/codex-takeover-20260920/clip/clip-quality-integration-decision-v1.md)。
- [DSANet历史UCF校准回执](../../work/clip-assets-scripts/receipt-dsanet.json)、[三条检测路线协议历史记录](../../work/clip-assets-scripts/PROTOCOLS.md)。旧协议中的初步特征解释应以本轮CSV/crop源码核验和后续全清单审计更正。
- [CLIP bridge现有实现](../../src/vadbench/token_reduction/bridges/clip.py)、[PairSelect三档定义](../../src/vadbench/token_reduction/token_selection.py)。

外部一手依据核查日期：2026-09-23。主要采用AAAI正式论文、DSANet官方仓库和评测源码，引用已置于对应结论旁。未依据聚合排行榜推定绝对SOTA，也未把作者数字当作本轮运行结果。
