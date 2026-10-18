# VideoMAEv2 UCF/XD 冻结38项实验最终质量报告

生成时间：2026-10-14T17:51:56.800573+08:00

## 完成范围

38/38提取与预测任务完成，38份成功云端回执；36/36压缩配置完成单次配对质量评测，全部云端FINISHED。累计4,086,273 clips（19×69,364 UCF＋19×145,703 XD）。

固定dense head、direct_insert；移除50%/70%，插入block 6/9/12。没有依据正式测试分数改配置。

UCF主指标frame ROC-AUC，XD主指标作者梯形PR-AUC；数值单位%，括号为method−dense的百分点差值。非劣要求差值95%CI下界≥−0.5个百分点。

Dense：UCF ROC-AUC 80.631958%；XD梯形PR-AUC 76.181075%，XD step AP 70.333010%。

## 主指标总表

| 方法 | 层 | 移除比例 | UCF ROC-AUC（Δ）与判定 | XD梯形PR-AUC（Δ）与判定 |
|---|---:|---:|---|---|
| Attention | 6 | 50% | 80.553（-0.079） / 非劣证据不足 | 76.640（+0.459） / 支持非劣 |
| Attention | 6 | 70% | 80.185（-0.447） / 非劣证据不足 | 75.980（-0.201） / 非劣证据不足 |
| Attention | 9 | 50% | 81.403（+0.771） / 支持非劣 | 77.175（+0.994） / 支持非劣；CI高于0 |
| Attention | 9 | 70% | 82.131（+1.499） / 支持非劣 | 77.502（+1.321） / 支持非劣；CI高于0 |
| Attention | 12 | 50% | 80.678（+0.046） / 非劣证据不足 | 76.225（+0.044） / 非劣证据不足 |
| Attention | 12 | 70% | 80.714（+0.082） / 非劣证据不足 | 75.905（-0.276） / 非劣证据不足 |
| VisionZip（本轮视频适配） | 6 | 50% | 80.490（-0.141） / 非劣证据不足 | 76.580（+0.399） / 支持非劣 |
| VisionZip（本轮视频适配） | 6 | 70% | 80.185（-0.446） / 非劣证据不足 | 76.120（-0.061） / 非劣证据不足 |
| VisionZip（本轮视频适配） | 9 | 50% | 81.556（+0.924） / 支持非劣 | 77.185（+1.004） / 支持非劣；CI高于0 |
| VisionZip（本轮视频适配） | 9 | 70% | 82.166（+1.535） / 支持非劣 | 77.541（+1.360） / 支持非劣；CI高于0 |
| VisionZip（本轮视频适配） | 12 | 50% | 80.752（+0.120） / 非劣证据不足 | 76.414（+0.233） / 支持非劣 |
| VisionZip（本轮视频适配） | 12 | 70% | 80.907（+0.275） / 非劣证据不足 | 76.291（+0.110） / 非劣证据不足 |
| uniform | 6 | 50% | 80.055（-0.577） / 非劣证据不足 | 74.866（-1.315） / 确认劣于非劣界限；CI低于0 |
| uniform | 6 | 70% | 79.754（-0.878） / 非劣证据不足 | 73.265（-2.916） / 确认劣于非劣界限；CI低于0 |
| uniform | 9 | 50% | 80.325（-0.307） / 非劣证据不足 | 75.839（-0.342） / 非劣证据不足；CI低于0 |
| uniform | 9 | 70% | 80.672（+0.040） / 非劣证据不足 | 75.459（-0.722） / 非劣证据不足；CI低于0 |
| uniform | 12 | 50% | 80.690（+0.058） / 支持非劣 | 76.176（-0.005） / 支持非劣 |
| uniform | 12 | 70% | 80.646（+0.014） / 支持非劣 | 76.128（-0.053） / 支持非劣；CI低于0 |

## 全部差值置信区间

| 数据集 | 方法 | 层 | 移除% | Δ，百分点 | 95%CI，百分点 | 非劣判定 |
|---|---|---:|---:|---:|---|---|
| ucf_crime | attention_topk | 6 | 50 | -0.0788 | [-1.4190, +1.2304] | inconclusive |
| ucf_crime | attention_topk | 6 | 70 | -0.4474 | [-2.4914, +1.5468] | inconclusive |
| xd_violence | attention_topk | 6 | 70 | -0.2010 | [-0.8386, +0.4558] | inconclusive |
| ucf_crime | attention_topk | 9 | 50 | +0.7714 | [-0.2754, +1.8734] | supported |
| xd_violence | attention_topk | 6 | 50 | +0.4592 | [-0.0606, +1.0612] | supported |
| ucf_crime | attention_topk | 9 | 70 | +1.4988 | [-0.2201, +3.3221] | supported |
| ucf_crime | attention_topk | 12 | 50 | +0.0461 | [-1.1670, +1.2041] | inconclusive |
| xd_violence | attention_topk | 9 | 70 | +1.3206 | [+0.5430, +2.0511] | supported |
| xd_violence | attention_topk | 9 | 50 | +0.9940 | [+0.4911, +1.4843] | supported |
| ucf_crime | attention_topk | 12 | 70 | +0.0819 | [-1.6204, +1.7069] | inconclusive |
| xd_violence | attention_topk | 12 | 50 | +0.0438 | [-0.5485, +0.6334] | inconclusive |
| ucf_crime | visionzip_internal | 6 | 50 | -0.1415 | [-1.7949, +1.4744] | inconclusive |
| xd_violence | attention_topk | 12 | 70 | -0.2761 | [-1.2690, +0.6727] | inconclusive |
| ucf_crime | visionzip_internal | 6 | 70 | -0.4465 | [-2.7462, +1.6995] | inconclusive |
| ucf_crime | visionzip_internal | 9 | 50 | +0.9236 | [-0.2043, +2.1358] | supported |
| xd_violence | visionzip_internal | 6 | 50 | +0.3986 | [-0.0219, +0.8615] | supported |
| ucf_crime | visionzip_internal | 9 | 70 | +1.5345 | [-0.3013, +3.5528] | supported |
| xd_violence | visionzip_internal | 6 | 70 | -0.0610 | [-0.5586, +0.4435] | inconclusive |
| xd_violence | visionzip_internal | 9 | 50 | +1.0042 | [+0.6397, +1.3652] | supported |
| ucf_crime | visionzip_internal | 12 | 50 | +0.1199 | [-0.6479, +0.8805] | inconclusive |
| ucf_crime | visionzip_internal | 12 | 70 | +0.2749 | [-1.0369, +1.5106] | inconclusive |
| xd_violence | visionzip_internal | 12 | 50 | +0.2330 | [-0.1119, +0.5631] | supported |
| xd_violence | visionzip_internal | 9 | 70 | +1.3602 | [+0.7506, +1.9546] | supported |
| ucf_crime | uniform | 6 | 50 | -0.5766 | [-1.2143, +0.0131] | inconclusive |
| ucf_crime | uniform | 6 | 70 | -0.8781 | [-2.3741, +0.4669] | inconclusive |
| xd_violence | uniform | 6 | 50 | -1.3153 | [-1.6320, -1.0026] | inferior |
| xd_violence | visionzip_internal | 12 | 70 | +0.1099 | [-0.5240, +0.7207] | inconclusive |
| ucf_crime | uniform | 9 | 50 | -0.3071 | [-0.8750, +0.1483] | inconclusive |
| xd_violence | uniform | 6 | 70 | -2.9159 | [-3.5623, -2.2835] | inferior |
| ucf_crime | uniform | 9 | 70 | +0.0397 | [-1.1617, +1.1912] | inconclusive |
| xd_violence | uniform | 9 | 50 | -0.3423 | [-0.5755, -0.1391] | inconclusive |
| ucf_crime | uniform | 12 | 50 | +0.0580 | [-0.0360, +0.1472] | supported |
| ucf_crime | uniform | 12 | 70 | +0.0143 | [-0.1732, +0.1766] | supported |
| xd_violence | uniform | 9 | 70 | -0.7222 | [-1.0485, -0.4103] | inconclusive |
| xd_violence | uniform | 12 | 50 | -0.0047 | [-0.0297, +0.0196] | supported |
| xd_violence | uniform | 12 | 70 | -0.0526 | [-0.1182, -0.0012] | supported |

## XD step AP（补充指标，不替代主指标）

| 方法 | 层 | 移除% | step AP，% |
|---|---:|---:|---:|
| attention_topk | 6 | 70 | 71.0095 |
| attention_topk | 6 | 50 | 71.0734 |
| attention_topk | 9 | 70 | 73.1589 |
| attention_topk | 9 | 50 | 71.9811 |
| attention_topk | 12 | 50 | 71.4592 |
| attention_topk | 12 | 70 | 72.0915 |
| visionzip_internal | 6 | 50 | 70.7275 |
| visionzip_internal | 6 | 70 | 70.7502 |
| visionzip_internal | 9 | 50 | 72.0798 |
| visionzip_internal | 12 | 50 | 71.3188 |
| visionzip_internal | 9 | 70 | 73.2151 |
| uniform | 6 | 50 | 69.3616 |
| visionzip_internal | 12 | 70 | 71.9577 |
| uniform | 6 | 70 | 68.1489 |
| uniform | 9 | 50 | 70.0426 |
| uniform | 9 | 70 | 69.7972 |
| uniform | 12 | 50 | 70.3447 |
| uniform | 12 | 70 | 70.3141 |

## 解释边界

- 所有CI是各配置相对dense的视频级配对区间；未做多配置校正，只有一个head seed。不是不同压缩方法之间的直接配对CI。
- CI低于0支持下降，但只有整个CI低于−0.5个百分点才确认劣于预设非劣界限；跨−0.5时判证据不足。
- UCF和XD不能混用指标；XD step AP不能替代作者梯形PR-AUC。
- 未完成正式独占净延迟、插件开销和峰值显存基准；这些字段为null。共享提取吞吐、CPU执行管线工程加速不能作为方法加速倍数。
- 第12层保留较长完整前缀，质量接近dense不等于显著净加速。
- 历史失败目录、原协议与指纹均保留；r17恢复原云端身份，无CPU/GPU重算。
- 旧默认quality阶段被停止，最终报告复用36份原preview；未覆盖任何final quality.json。

## 证据文件

- final-results.json：38份预测回执、36份质量源文件SHA及云端链接、原数据/head绑定和完整数值。
- quality-results.csv：36项完整配置/主指标/差值/CI/判定/step AP。
- 最终报告云端：[SwanLab](https://swanlab.cn/@ryfrcd693/VAD-Key-Attention-20261013/runs/oqyvuz18)。
