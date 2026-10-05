# DSANet 发布权重 × 本轮 raw CLIP 四档结果

执行：2026-09-23 晚；汇总：2026-09-24 00:00（Asia/Hong_Kong）。状态：八格完整测试评分、质量与定位导出、编码器/检测头/完整视频效率均完成；不使用本轮本地训练检测头。

## 身份与范围

- UCF-Crime 固定作者发布 `model_ucf.pth`，SHA-256 `cc674979901e2ed7153161e2cd4888ecacd25966e18caaba8472d993aedd47e0`；XD-Violence 固定 `model_xd.pth`，SHA-256 `78317c34fb8806d16bffe432946dd1b4cb6c6f3cc79ab29151163f6093febb39`。发布 checkpoint 的原训练 seed 未知；本轮每个数据集只使用这一份权重，不选 seed、不训练。
- 八格使用同一冻结 raw 视频协议的 dense / 删除20% / 删除40% / 删除60% CLIP 特征。UCF test 290 视频、每档 69,634 图像；XD test 800 视频、每档 146,449 图像。每数据集四档的测试 manifest、权重 SHA 一致，预测阶段未读取真值。
- 八个评分进程在 node3 `ibnode3` 本机时间 23:06:08 的约 5 ms 窗口内开始。node3 时钟相对宿主约慢 9.5 分钟，服务器原时间戳未改写。
- 此处的 dense 是**本轮 raw CLIP 特征**。2026-09-20 作者预提 UCF 特征 + 发布权重的 89.4446% AUC 是独立校准身份，不能用于下面的压缩差值。XD 作者预提特征没有同等级的本地复现回执。

## 完整测试集质量

主分支为 DSANet `coarse`；UCF 主指标是帧级 ROC-AUC，XD 主指标是非插值 step AP。差值 = 压缩档 − 本轮 dense，单位为百分点。95% CI 用 10,000 次按视频配对、按弱标签分层的 bootstrap；所有主指标比较均为 10,000/10,000 有效抽样。预设非劣性门槛是 CI 下界 ≥ −0.5 个百分点。

| 数据集 | 删除率 | 主指标 % | 相对 dense 差值 pp | 95% CI pp | 非劣性 | 原生定位 mAP 均值 % |
|---|---:|---:|---:|---:|---|---:|
| UCF | dense | 88.868 | — | — | 基准 | 13.17 |
| UCF | 20% | 88.542 | −0.326 | [−0.760, +0.056] | 未证实 | 13.26 |
| UCF | 40% | 87.966 | −0.902 | [−2.072, +0.215] | 未证实 | 11.41 |
| UCF | 60% | 87.065 | −1.803 | [−3.725, −0.003] | 未证实 | 10.12 |
| XD | dense | 85.597 | — | — | 基准 | 28.83 |
| XD | 20% | 85.490 | −0.106 | [−0.377, +0.173] | **支持** | 29.38 |
| XD | 40% | 85.353 | −0.243 | [−1.003, +0.496] | 未证实 | 29.13 |
| XD | 60% | 84.379 | −1.217 | [−2.713, +0.252] | 未证实 | 30.47 |

两数据集的另一分支、ROC-AUC、step AP、梯形 PR-AUC、五个 tIoU 阈值的定位 mAP 和原始 CI 见 [`quality-ucf.json`](results_published/quality-ucf.json)、[`quality-xd.json`](results_published/quality-xd.json)、[`map-ucf.json`](results_published/map-ucf.json)、[`map-xd.json`](results_published/map-xd.json)。XD 的 step AP 与梯形 PR-AUC 各保留自身定义，不能互换。定位 mAP 与帧级 AP 衡量不同目标，不能用定位上升掩盖帧级质量下降。
本轮没有依据这些 test 数值修改压缩预算、层、crop、score 分支或 checkpoint。

## 效率

编码器计时为准备好的 GPU 图像到 CLIP pooled 输出，包含动态选择与 gather；node3 A100 40GB、FP32、TF32 关闭，3 个独立进程，20 次预热、40 次同步测量。完整视频计时为打开训练视频到 DSANet coarse/class 分数回传 CPU，不写特征缓存；每数据集固定 16 个完整训练视频，batch32，3 个独立进程，四档交错顺序，同一数据集固定 GPU（UCF GPU0、XD GPU4）。两数据集同时使用不同 GPU，CPU/存储仍为共享资源；OS 文件缓存为暖状态。完整视频倍率是每轮 `T_dense/T_tier` 的三轮中位数。

| 数据集 | 删除率 | encoder B1 倍率 | encoder B32 倍率 | 完整视频倍率 | 16 视频 wall 中位数 s | 完整流程峰值 allocated MiB |
|---|---:|---:|---:|---:|---:|---:|
| UCF | dense | 1.000× | 1.000× | 1.000× | 25.74 | 1639.7 |
| UCF | 20% | 0.953× | 1.096× | 0.968× | 26.53 | 1639.7 |
| UCF | 40% | 0.956× | 1.222× | 0.991× | 26.60 | 1639.7 |
| UCF | 60% | 0.932× | 1.381× | 0.978× | 26.78 | 1639.7 |
| XD | dense | 1.000× | 1.000× | 1.000× | 28.16 | 1624.7 |
| XD | 20% | 0.919× | 1.093× | 1.008× | 27.96 | 1624.7 |
| XD | 40% | 0.962× | 1.221× | 1.008× | 28.04 | 1624.7 |
| XD | 60% | 0.924× | 1.383× | 1.014× | 27.74 | 1624.7 |

编码器 batch32 的 principal-block 解析成本为 dense 34.89、删除20% 31.33、删除40% 27.82、删除60% 24.27 GFLOPs/图像；这不是完整 encoder/head/pipeline 的 GFLOPs。selector 本身约 0.45–0.57 ms。完整流程峰值显存没有随压缩稳定下降；16 视频的 CPU 峰值 RSS 约 2.6–2.7 GiB。检测头准备好 GPU 特征到 GPU coarse/class 分数的 16 视频中位 p50 为 UCF 7.612 ms、XD 6.946 ms，详见 [`head-ucf-scores.json`](results_published/head-ucf-scores.json)、[`head-xd-scores.json`](results_published/head-xd-scores.json)。

编码器 batch32 的 UCF 吞吐为 dense 397.3、三档 436.8/486.2/550.6 图像/s；XD 为 391.4、428.6/478.0/541.4 图像/s。相应 UCF 每批 p50 为 80.55、73.25/65.82/58.12 ms，XD 为 81.75、74.66/66.95/59.10 ms。详细 p95、allocated/reserved 显存与原始40次同步测量见 [`ucf-summary.json`](benchmark/ucf-summary.json)、[`xd-summary.json`](benchmark/xd-summary.json)及其绑定的三份原始文件。发布 checkpoint 文件大小为 UCF 775,264,773 B、XD 760,552,038 B；按相同上游 DSANet 构造的模型参数量分别为 193,751,619 与 190,074,947，压缩不改变这些权重。单档测试特征目录占盘约 UCF 143 MB、XD 302 MB（含回执与索引）；完整流程计时不写这些缓存。

完整视频中，dense 的 16 视频来源时长为 UCF 1589 s、XD 1573.5 s；解码/预处理阶段分别约 17.55 s、21.31 s，远大于视觉编码阶段约 7.47 s、6.03 s。因此本轮**没有可信的端到端显著加速主张**，batch1 编码器还略慢。八卡同时跑的完整视频记录在 [`pipeline-concurrent-r02`](results_published/pipeline-concurrent-r02/) 中，只作并发生产回执；正式倍率采用 [`pipeline-exclusive-r02`](results_published/pipeline-exclusive-r02/) 的同卡三轮结果。此前仅到 detector logits 的计时为历史尝试，不列入正式结果。

按上述固定16视频，dense 完整流程的来源时长/墙钟时间约为 UCF 61.7×、XD 55.9×；这表示离线处理速度，DSANet 仍是非因果时序检测器，不称流式实时延迟。八档完整测试特征的共享节点生产提取 wall 约为 UCF 每档 20.6 分钟、XD 每档 45.7 分钟，见 [`sealed_test_features.json`](sealed_test_features.json)；它们是多卡并发、含写盘的工时，不能当作独占倍率。

## 验收与限制

- 八份评分 summary 均为 `completed`，每数据集四份 checkpoint SHA 与 manifest SHA 唯一；评分文件 SHA 与质量、定位导出逐项一致。两数据集各 12 份正式完整流程计时均为 16 个相同视频、相同原始帧数与特征行数，checkpoint SHA 与评分相同。
- 视觉 encoder 的 batch32 吞吐优势不能推断为完整视频优势。完整流程仍受视频解码、裁剪、传输和检测头固定成本限制；并发 CPU/存储也可能造成小幅倍率波动。
- 计划中的 GPU 能耗为条件项，本轮没有与每视频边界同步的可靠高频功率积分，状态为 `not_measured`；F1 需要预先冻结验证阈值，本轮无此契约，状态为 `not_applicable`。完整模型 FLOPs 未获算子覆盖完整的 profiler 回执，状态为 `analytical_partial`，仅报告上述 principal-block 成本。没有用零或估计值填这些缺项。
- 原代码的 `run_published_analysis.sh` 第一次尝试因 Python 3.9 无法导入当前质量模块而失败；最终质量计算用 Python 3.10.20 和与工作树逐字节相同的 `metrics.py`、`quality_comparison.py`、`urdmu_quality_export.py` 隔离副本完成。评分/效率仍使用原 CLIP 环境 Python 3.9.23、torch 2.0.1+cu117。失败日志原位保留。
- 第一批完整视频并发计时 `pipeline-concurrent-r01` 的 dense 入口因 Bash 空数组展开失败，另外六档曾完成但不构成八格对照；修正后另起 `pipeline-concurrent-r02` 八格全部完成，旧输出没有覆盖。独占 `pipeline-exclusive-r01` 只到 detector logits，补齐实际分数读出和 CPU 回传后另起 `r02` 作为正式结果。
