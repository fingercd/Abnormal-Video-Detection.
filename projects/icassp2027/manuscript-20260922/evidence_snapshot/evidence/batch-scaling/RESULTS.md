# PairSelect batch scaling 快速复测

测试日期：2026-09-23。运行节点：node3，A100 40GB。TimeSformer、VideoMAE、VideoMAEv2 分别在 GPU 2、3、4 上运行，三张卡均无其他计算进程。

## 结论

batch=16 和 batch=32 下，三个 encoder 的 keep=0.80、0.60、0.40 均比相同条件下的 dense 快。每格使用一条正常训练视频和一条异常训练视频；每条视频每个配置预热 20 次、正式计时 20 次，共 40 个正式样本。两个视频上的加速方向逐格一致。

本测试只比较准备好的 GPU 输入到 pooled encoder 输出，包含当前 PairSelect 的动态选择、gather、deployment context 和执行校验；不包含解码、processor、CPU→GPU、检测头和写盘。各 encoder 内使用同一模型实例、输入、FP32、TF32-off、cuDNN-off 和原生前向路径。执行顺序按视频和 batch 轮换。

## 吞吐和相对 dense 加速

| Encoder | Batch | Dense clips/s | keep=.80 | 加速 | keep=.60 | 加速 | keep=.40 | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TimeSformer | 8 | 35.78 | 39.01 | 1.091× | 43.92 | 1.228× | 50.88 | 1.422× |
| TimeSformer | 16 | 34.94 | 38.30 | 1.096× | 43.92 | 1.257× | 50.23 | 1.438× |
| TimeSformer | 32 | 34.42 | 38.23 | 1.111× | 43.97 | 1.277× | 50.59 | 1.470× |
| VideoMAE | 8 | 27.91 | 27.91 | 1.000× | 31.83 | 1.141× | 35.40 | 1.268× |
| VideoMAE | 16 | 27.78 | 30.01 | 1.080× | 32.84 | 1.182× | 36.40 | 1.311× |
| VideoMAE | 32 | 27.58 | 30.02 | 1.088× | 34.40 | 1.247× | 38.03 | 1.378× |
| VideoMAEv2 | 8 | 27.47 | 27.50 | 1.001× | 30.64 | 1.116× | 32.78 | 1.194× |
| VideoMAEv2 | 16 | 27.08 | 28.73 | 1.061× | 32.74 | 1.209× | 35.51 | 1.311× |
| VideoMAEv2 | 32 | 27.43 | 30.04 | 1.095× | 33.60 | 1.225× | 36.49 | 1.330× |

batch=8 数字来自先前的 16 视频、3 独立进程正式测量；batch=16/32 是本轮快速扩展，两条训练视频、每格40个正式样本。两类测量的输入视频数量不同，因此 batch=16/32 用来确认批量扩展趋势，不替代 batch=8 的主测证据。

## 核验

- 三个远端 `result.json` 均为 `completed`，脚本 SHA 与本地一致。
- 每个 batch×配置有 40 个正式计时样本，wall-clock 和 CUDA event 同时保留。
- 相同 encoder、视频和 batch 下，各配置使用同一 prepared-input SHA。
- 每格的两条视频都显示相同的加速方向；最小加速为 VideoMAEv2 batch16 keep=.80 的 1.061×。
- `python work/batch-scaling-20260923/verify_results.py` 通过；`python -m compileall -q src tests` 通过。

原始结果位于 `results/{encoder}/result.json`。本轮没有修改论文。
