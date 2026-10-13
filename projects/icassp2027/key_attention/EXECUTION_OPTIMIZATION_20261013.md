# 2026-10-13 后续任务的执行优化与验收

用户授权优化后续任务，并要求单GPU对照及四进程加速核验。原四个GPU任务继续原代码，不重启；原正式冻结、campaign SHA、采样、microbatch8、精度、方法、head与测试权限不变。

## 改动与门禁

- 一个连续解码线程保留游标和32个源帧的尾部缓存，减少dense重叠窗口的重复seek/解码。
- native VideoMAEImageProcessor按实际源帧去重处理，再恢复原BCTHW；不是换resize/归一化算法。
- 有界预取两批，CPU准备与主线程GPU前向重叠。模型、hooks、reducer和唯一FeatureStore writer仍在主线程。
- 默认提取路径保持旧行为。新路径只通过独立execution配置启用，配置、快照、验收及云端回执各自SHA绑定。
- 已运行本机controller无需重启；后续新tracker进程读取`work/key-attn-formal-20261013/pending-execution.json`。preparing时仅等待，不发GPU任务；只有真实验收accepted及云端成功才能ready。已运行四个tracker不受改动影响。

## 验收范围与已知结果

复用原role-locked fit的Abuse005_x264和Normal_Videos022_x264及原SHA，没有读取正式测试分数。两视频共83个dense clips；RGB、源帧索引和native处理器像素逐元素一致。两次四路CPU并发测量：

| 每路预处理workers | 四路聚合clips/s，两次实测 | 中位总耗时 |
|---|---|---|
| 旧串行路径 | 42.89、42.18 | 7.807 s |
| 新路径1 worker | 70.95、72.47 | 4.630 s |
| 新路径2 workers | 64.06、78.01 | 4.719 s |

当前选择1 worker、prefetch2，CPU侧约1.69x；这仅是这两个fit视频的供数测量，不是GPU端到端、四卡正式吞吐或论文效率结果，不宣称全局最快。

单GPU验收等待GPU2原PID31725自然结束；有任何未知GPU占用即不启动。之后native/optimized反向顺序各重复两次，跑原完整提取、FeatureStore写入/校验，覆盖dense与三个压缩方法，比较保存的pooled逐元素一致。要求CPU四路speedup>1.05、四方法GPU speedup中位>1.05且任何方法不低于0.95，云端完成回读成功，才放行。正式四个新版进程启动后还需核对聚合吞吐；单卡通过不等于四卡收益已证实。

本地51项受影响单测通过，compileall src tests及改动脚本通过。工程r01在CPU阶段因历史fit manifest num_frames为空退出，未分配CUDA；失败及SwanLab CRASHED保留。r02使用实际OpenCV probe，SwanLab run n99346d2、remote `/users/fotile/VAD/work/key-attention-optimization-20261013-r02`，完整日志与CPU结果保留；GPU验收尚待完成。

效率测量处于其他任务并行的共享节点，不能进入论文独占速度表。输出一致性或吞吐门失败时保留所有证据，继续诊断，不篡改旧身份或用测试质量选择执行参数。

## 后续候选：跨批处理器像素尾部缓存

在上述已验收CPU路径上进一步减少跨批重复预处理：保留32个源帧的已处理像素，视频切换时清空，仅允许单预处理worker，仍调用同一native处理器处理新帧。r04将旧路径、1/2 worker无像素缓存、1 worker带像素缓存四路CPU并发逐一对照，再由选定配置做GPU2对照。r02只读CPU结果保留，原等待GPU阶段以user_scope_upgrade停止，未分配CUDA。此项新增真实视频/GPU验收尚未通过，不能据单测宣称加速。63项相关单测及compileall通过；自动放行watcher只在验收和云端均成功后发布ready，不启动任何GPU任务。
