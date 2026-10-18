# VideoMAE / TimeSformer 独占效率协议 R01

冻结日期：2026-10-15。仅覆盖本轮两个 encoder，不改变已经冻结的三方法、pre-block6/9/12、移除50%/70%、各自正式head或采样。

- Encoder：2 encoder ×19配置 ×B1/B8 ×3个独立进程 =228组。每组20次warmup、40个实测样本，输入已经按原processor准备并常驻GPU；计入原adapter、mean读出、插件信号/选择/merge/gather及实际执行回执开销，不计视频解码、CPU预处理、H2D或权重加载。保存wall和CUDA stream原始毫秒、p50/p95、吞吐、allocated/reserved；不把stream时间称作纯kernel时间。
- 完整流程：2 encoder ×2 dataset ×19配置 ×3独立进程 =228组。每组16个完整训练视频，按video_id排序取8 normal+8 anomalous；四套输入分别2973/2354/5952/4718个原生dense clips，固定成员与原视频SHA，不按test效果选片段。完整视频不截取、不做200-bin聚合。
- 完整流程计时包含：OpenCV连续解码、原生HF CPU预处理workers2/prefetch2、H2D、encoder/读出/插件、特征D2H、正式固定head与全key的query256精确推理、原帧区间均值投影。模型/head加载、原视频SHA检查、FeatureStore落盘及最终结果文件序列化不计入，所以不是“整条正式提取命令的总用时”。
- 每组正式计时前20次encoder B8 warmup和1次head warmup；模型输入/帧分数数值及覆盖检查保留。视频文件SHA在该视频计时前核验，可能预热系统页缓存；不称冷盘/冷启动吞吐。系统页缓存未清空，CPU/存储为共享资源，不宣称全节点独占。
- Node3 GPU5须在每个独立PID开始/结束时无其他CUDA进程；未知占用时停止该效率队列并保留已有结果，不停止外项目。FP32、TF32关闭、cuDNN关闭、原生attention backend不改，单PID显存上限0.5。
- 每个配置三次独立进程，顺序预先用seed0打乱；首个VMA dense B1 R0先验证真实入口。全部456组列表已冻结在efficiency-plan-r01.json，SHA 2e0e30adbd76726f7f7368aee80c51cf42205b0c54efb264158effff1e50e619。
- 源代码快照efficiency-source-r01.tar.gz SHA47fcd0decaa1cda2100ac83c4db8ff821cb91d48935b952413906d7ee2c89cb2；原提取/预测仍运行formal-source-r02，不改、不重启。效率使用相同原adapter字节、bridge/selector和固定head。
- 全队列在SwanLab在线会话建立后才启动；每组保存原PID、原始日志、结果SHA、独占GPU前后回执。质量真值和正式测试视频均不进入此效率队列。不根据测得速度或正式质量选择或删除配置；OOM/失败/负加速如实保留。