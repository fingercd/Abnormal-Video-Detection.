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

## 已通过的r05与生产接力

r04严格bitwise门揭示VisionZip原FP32 CUDA scatter_add_的原生重复波动，并非输入像素变化。独立native/native四次对照已云端验收：Abuse旧/旧1.907e-6、旧/新2.146e-6；Normal旧/旧1.550e-6、旧/新1.669e-6，relative L2<=4.453e-7。仅VisionZip用原生重复波动界（noise>0且<=1e-5；cross<=min(1e-5,2*noise+1e-7)；relativeL2<=1e-6），其他方法仍逐元素一致；不改算法、预算、head或原冻结。独立诊断SwanLab5xr8gik1 FINISHED；诊断-r01用错B8 setup被原B1契约拒绝，修正脚本而未弱化契约，失败均保留。

r05完整单GPU2提取/FeatureStore写入对照：dense2.030x、AttentionTopK1.806x、VisionZip1.617x、Uniform1.794x，GPU中位1.800x；四CPU供数1.688x。最终按本次CPU四路中位选择workers2、prefetch2、deduplicate=true、temporal_cache=false，候选像素缓存未默认启用。VisionZip完整83clip旧/旧和旧/新均max_abs1.669e-6，relativeL2 4.370e-7；其他三方法pooled bitwise exact，RGB/pixels全部exact。SwanLab pk85cmf7云端FINISHED且qa/completed=1已核验。source32dabf9b51d98d0b2f97e12e576c6144fce1b9a96f234e7ed471bdab84e89683及acceptance/config/cloud/独立数值回执各SHA绑定。

当前新版GPU0/2已经生产，原GPU1/3仍正常旧代码；不重启原任务。本机调度器119980仅接管旧logger，刷新未发GPU的等待tracker；本机新tracker显式PYTHONPATH=repo/src以读取标准库门禁模块，不安装或更改共享环境。用户最新要求检查保持简单，后续只简查PID、进度、错误与接力，不重复或扩展benchmark。当前四个全部新版的聚合收益尚未实测；不能把单卡/CPU并发结果说成四GPU成功。
