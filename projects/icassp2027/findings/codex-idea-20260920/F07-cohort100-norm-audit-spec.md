# F07：优先复用cohort100核查成员范数选择的增量

日期：2026-09-20。状态：`cpu_analysis_specified_pending_dispatch`。这是停止F06扩量后的下一优先问题，不是结束idea研究。优先级高于新GPU提取：现成7臂足以单独核查pair内范数偏好，先把已付费的证据用完整。

## 已真实核到的资产

node3公网线路只读读取了7臂的result、extraction-contract、resolved_reducer和首行index，回执为`work/codex-takeover-20260920/idea/cohort100-identity-read.txt`。根目录：

`/data2/localdisk/fotile-icassp2027-kimi-20260919-a01/batch1/videomaev2/`

| run | 角色 |
|---|---|
| b1-dense | dense参考 |
| b1-pairselect-060 | max-norm成员优先 |
| b1b-pair-fixed-060 | 同pair排序，native首成员 |
| b1b-pair-random-member-060 | 同pair排序，固定seed成员 |
| b1b-pair-reverse-060 | 同pair排序，min-norm成员优先 |
| b1-uniform-060 | group覆盖对照 |
| b1-random-060 | group随机对照 |

7臂均声明completed、100视频/36,470 clips、官方train工程子集；各index路径为`<run>/features/train/index.jsonl`。sampling fingerprint均为`823cb5cd9f4645aae685725e1e7a56ec81173dfc83f68bbb3a4a0d36f2b64dc8`。压缩点depth5，实际941/1568=0.60012755，topology SHA相同，indexed实现SHA相同。首条Abuse008的source content SHA和frame区间一致。缓存是pooled_only，不含中间token；不能从这些文件重建pair内原始范数或hidden。

这些仅是入口审计，不替代全部36,470 clip键的配对。b1为v7、1b为v10，selection/deployment源码SHA不同；必须核窄diff与同rule CPU parity，不能凭预算一样宣称只有成员偏好不同。

角色限制：现存contract的`development_role`和`original_role_lock`均为null，只有`engineering-subset-of-official-fulltrain`身份。因此本次按用户授权分析既有100训练视频，不能把它们补写成原来锁定的fit或held-out confirm。Luna应导出100个ID供后续新实验避开；若需要映射历史roles，只读原始锁并保留来源，不改旧contract。formal dense head若在full1610上训练，这100的head诊断明确为训练内。

## 主问题和比较身份

H：在同group quota、pair表、pair间norm排序及位置处理下，max-norm成员偏好能否比固定成员和随机成员更好保持dense表示及冻结dense头输出？先验证这一有限问题，不把它扩为异常token定位、pair间norm排序有效或跨模型泛化。

主比较预先固定为pairselect减pair_fixed、pairselect减pair_random_member；reverse只作方向对照。uniform/group_random不能代替同结构成员消融。0.60预算每5pair组保留6成员：有一对保留两员，其余pair才体现成员偏好；不能简化为“每pair一员”的旧0.50 S7方法。

使用全部已完成100训练视频，不根据结果挑子集。统计每视频等权，每视频内clip等权。全部正常/含异常组和总体均保留，弱标签只离线汇总；不计算新test、不把训练内头分数称泛化。

## Luna实现规格

责任文件：`work/codex-takeover-20260920/idea/analyze_cohort100_members.py`及同目录运行/回执；不修改公共src、协议、原FeatureStore或旧指纹。CPU流式分析，最多2线程；不抢GPU、不重提、不创建平行评测体系。

1. SHA绑定7个extraction-contract、resolved和index；检查status/成员集合。用`(video_id,clip_id)`配对所有行，逐行核frame_start/end、start_s/end_s、actual stride、source content SHA、source split、sampler身份、readout、dtype。发现不匹配就报阻塞差异清单，不能按最短长度zip、补齐、截断或静默取交集。
2. 比较v7/v10的`token_selection.py`和`deployment.py`，确认pairselect的已有分支未被逻辑改变。CPU固定输入（含非相等范数、相等范数tie、batch1/2）比较v7/v10同pairselect输出及真实选择indices。索引表、quota、hidden、seed相同，旧module在隔离Python进程加载；不覆盖sys.modules去伪造同版本。如果差异超出新增成员分支，停止因果增量主张并报告。
3. 按index中的`arrays.pooled.path/key`读取已有NPZ，核SHA/shape/finite；优先逐视频处理，避免全NAS扫描或全部数组留内存。先1视频估读取耗时，再完成既有100。每arm输出clip级绝对MSE、cosine drift、norm ratio，以及下面的径向/角度分解；保存逐视频小表与输入绑定，不下载/提交大型数组。
4. 如现有严格head加载/输入契约支持，复用已核UCF V2 formal dense head（3000步、checkpoint SHA以heads最新回执为准）在相同视频、相同原有200段聚合、同eval模式下做CPU前向。只测dense与各arm未阈值化logit的配对漂移，不重训/选checkpoint，不加AUC/AP/F1选方法。若只能取得sigmoid概率，明确标概率及饱和比例，不用它冒充raw logit；不要通过logit(p)近似或clamp补造。head/consumer不能合法接入时保持unavailable，表示审计照常交付，不绕原严格检查。
5. 结果以视频配对bootstrap（10,000，seed20260920）给主差值区间，标签分层抽视频；同一视频的所有arms和clips共同进入一次抽样。CI不是独立confirm，标training-internal audit。给正/负组、整体以及全部反向结果；不做token级显著性。

## 表示误差的可解释分解

对dense向量x和arm向量y，D为维数，逐clip验证：

`MSE = (||y||−||x||)^2/D + 2||x||||y||(1−cos(x,y))/D`。

第一项是最终读出的径向差，第二项是角度差；逐视频平均后汇报，不混淆“范数差”和“全部表示误差”。零范数时cosine不可用，保留原始MSE并显式记录，不能静默跳过。

该分解不用新模型/标签，适合直接复用缓存。但最终pooled的径向误差小不意味着中间层norm没有因果影响：内部尺度变化仍可能经过非线性/残差变成最终方向变化。不能只靠这张表判定合并的内部机制。

## 停止与后继

- 身份/代码parity不通过：先保留可审计差异，不把不匹配数据当消融，不默认重提100视频。
- pairselect对fixed或random-member的主要MSE差区间未低于0：不确认成员norm增量；若反向则记录该配置的负结果，不换层/预算救结果。
- 只有总体获益、含异常组明显反向或head漂移恶化：不晋级为异常检测方法。
- 只有方向/尺度变动但固定head未改善：可以记录表示机制，不声称检测有效。
- 两个主同结构比较都有一致训练内增量，且没有组间/头输出反转，才考虑在V2+CLIP、UCF+XD另行登记小规模性质与干预验证；已有100不当独立确认，当前不自动派新GPU。

第二候选仅保留“均值合并的尺度因素诊断”（见F08），不在F07结果前另造新selector。本文件登记在读取cohort100表示/头结果之前；已读仅状态、身份和首行provenance，没有读取这7臂的训练评分或新test。
