# VideoMAE / TimeSformer 活动52格最终报告

用户在查看旧VideoMAEv2/CLIP正式结果后明确取消后续uniform：48项Attention/VisionZip压缩＋4dense；48项配对质量；活动156 encoder＋156完整视频效率组。此为后验范围变更，不是原完整三方法预注册矩阵；uniform历史保留，不把取消记为完成。

固定对应完整训练集head、direct_insert，不重训；pre-block6/9/12，信号来自同次前向前一完整block。

## 架构、预算与固定检测头

| Encoder | 实际blocks | 每clip输入帧 | 原始tokens | 移除50%后 | 移除70%后 | 预算单位 |
|---|---:|---:|---:|---:|---:|---|
| VideoMAE | 12 | 16 | 1568 | 784 | 470 | tubelet patch |
| TimeSformer | 12 | 8 | 1569 | 785 | 473 | 196条空间轨迹保留98/59条，8帧及原CLS保留 |

pre6/9/12均在前一完整block5/8/11计算结束后、目标block的LN/attention之前；target block的attention和MLP接收真实短序列。比例按预定整数预算取整。对应adapter readout保持原合同（mean），不把保留CLS偷换成只读CLS。

每encoder/dataset独立固定UR-DMU检测头，UCF1610个完整训练视频、XD3950个完整可用训练视频；复用原seed0 head，不是64+64 screen head，不重新训练或跨数据集共享head。

## 全部质量与效率

UCF主ROC-AUC；XD主梯形PR-AUC。分数单位%，Δ及CI为百分点；倍率小于1代表更慢。

| Encoder | 数据集 | 方法 | pre层 | 移除% | Dense | 压缩 | Δ pp | 95%CI pp | 非劣 | encoder B8倍率 | 完整视频倍率 |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| timesformer | ucf_crime | attention_topk | 6 | 50 | 78.6801 | 78.1298 | -0.5503 | [-3.0045,+2.1972] | inconclusive | 1.3838 | 0.9637 |
| timesformer | ucf_crime | attention_topk | 6 | 70 | 78.6801 | 77.0134 | -1.6666 | [-5.0112,+1.9257] | inconclusive | 1.6323 | 1.0325 |
| timesformer | ucf_crime | attention_topk | 9 | 50 | 78.6801 | 80.0865 | +1.4065 | [-0.4125,+3.4749] | supported | 1.1935 | 1.0487 |
| timesformer | ucf_crime | attention_topk | 9 | 70 | 78.6801 | 79.7943 | +1.1142 | [-1.2235,+3.3672] | inconclusive | 1.2810 | 1.1728 |
| timesformer | ucf_crime | attention_topk | 12 | 50 | 78.6801 | 79.4264 | +0.7463 | [-1.1958,+2.8824] | inconclusive | 1.0403 | 1.1678 |
| timesformer | ucf_crime | attention_topk | 12 | 70 | 78.6801 | 79.5405 | +0.8604 | [-1.1208,+2.9947] | inconclusive | 1.0575 | 1.1667 |
| timesformer | ucf_crime | visionzip_internal | 6 | 50 | 78.6801 | 79.2025 | +0.5224 | [-1.3282,+2.8396] | inconclusive | 1.3768 | 1.1647 |
| timesformer | ucf_crime | visionzip_internal | 6 | 70 | 78.6801 | 78.8402 | +0.1601 | [-2.0540,+2.7721] | inconclusive | 1.6239 | 0.9999 |
| timesformer | ucf_crime | visionzip_internal | 9 | 50 | 78.6801 | 80.1778 | +1.4977 | [-0.1001,+3.2017] | supported | 1.1818 | 1.2326 |
| timesformer | ucf_crime | visionzip_internal | 9 | 70 | 78.6801 | 79.6135 | +0.9334 | [-1.2392,+3.0328] | inconclusive | 1.2780 | 1.2495 |
| timesformer | ucf_crime | visionzip_internal | 12 | 50 | 78.6801 | 79.4043 | +0.7242 | [-0.6385,+2.2477] | inconclusive | 1.0352 | 1.2204 |
| timesformer | ucf_crime | visionzip_internal | 12 | 70 | 78.6801 | 79.7180 | +1.0379 | [-0.4612,+2.6351] | supported | 1.0553 | 0.9968 |
| timesformer | xd_violence | attention_topk | 6 | 50 | 76.6558 | 76.5560 | -0.0998 | [-0.6561,+0.4677] | inconclusive | 1.3838 | 0.8658 |
| timesformer | xd_violence | attention_topk | 6 | 70 | 76.6558 | 75.6859 | -0.9700 | [-1.8967,-0.0324] | inconclusive | 1.6323 | 1.0362 |
| timesformer | xd_violence | attention_topk | 9 | 50 | 76.6558 | 77.1697 | +0.5139 | [+0.0540,+0.9880] | supported | 1.1935 | 0.8999 |
| timesformer | xd_violence | attention_topk | 9 | 70 | 76.6558 | 76.3434 | -0.3124 | [-1.0898,+0.4528] | inconclusive | 1.2810 | 0.9161 |
| timesformer | xd_violence | attention_topk | 12 | 50 | 76.6558 | 77.3135 | +0.6576 | [+0.1761,+1.1501] | supported | 1.0403 | 0.8865 |
| timesformer | xd_violence | attention_topk | 12 | 70 | 76.6558 | 76.8038 | +0.1479 | [-0.5043,+0.8281] | inconclusive | 1.0575 | 0.7891 |
| timesformer | xd_violence | visionzip_internal | 6 | 50 | 76.6558 | 77.0555 | +0.3997 | [-0.0787,+0.8924] | supported | 1.3768 | 0.9149 |
| timesformer | xd_violence | visionzip_internal | 6 | 70 | 76.6558 | 76.3878 | -0.2681 | [-1.0384,+0.5262] | inconclusive | 1.6239 | 0.8551 |
| timesformer | xd_violence | visionzip_internal | 9 | 50 | 76.6558 | 76.8835 | +0.2276 | [-0.1636,+0.6331] | supported | 1.1818 | 0.9131 |
| timesformer | xd_violence | visionzip_internal | 9 | 70 | 76.6558 | 76.0716 | -0.5842 | [-1.2361,+0.0633] | inconclusive | 1.2780 | 0.8697 |
| timesformer | xd_violence | visionzip_internal | 12 | 50 | 76.6558 | 76.8595 | +0.2036 | [-0.1948,+0.6176] | supported | 1.0352 | 1.0037 |
| timesformer | xd_violence | visionzip_internal | 12 | 70 | 76.6558 | 76.5216 | -0.1343 | [-0.6964,+0.4495] | inconclusive | 1.0553 | 1.0442 |
| videomae | ucf_crime | attention_topk | 6 | 50 | 80.5487 | 80.2919 | -0.2568 | [-2.1503,+1.7641] | inconclusive | 1.4111 | 1.0032 |
| videomae | ucf_crime | attention_topk | 6 | 70 | 80.5487 | 79.7548 | -0.7939 | [-2.9324,+1.6816] | inconclusive | 1.5960 | 1.0141 |
| videomae | ucf_crime | attention_topk | 9 | 50 | 80.5487 | 80.5040 | -0.0448 | [-2.1216,+1.8945] | inconclusive | 1.2139 | 0.9794 |
| videomae | ucf_crime | attention_topk | 9 | 70 | 80.5487 | 80.0210 | -0.5278 | [-2.7304,+1.4593] | inconclusive | 1.2978 | 1.0289 |
| videomae | ucf_crime | attention_topk | 12 | 50 | 80.5487 | 81.1943 | +0.6455 | [-1.2447,+2.5932] | inconclusive | 1.0629 | 1.1818 |
| videomae | ucf_crime | attention_topk | 12 | 70 | 80.5487 | 81.5633 | +1.0145 | [-0.8296,+2.8817] | inconclusive | 1.0675 | 0.8524 |
| videomae | ucf_crime | visionzip_internal | 6 | 50 | 80.5487 | 80.2567 | -0.2921 | [-2.0348,+1.6835] | inconclusive | 1.4315 | 1.1306 |
| videomae | ucf_crime | visionzip_internal | 6 | 70 | 80.5487 | 79.1287 | -1.4200 | [-3.9462,+0.9825] | inconclusive | 1.5880 | 0.9730 |
| videomae | ucf_crime | visionzip_internal | 9 | 50 | 80.5487 | 80.8462 | +0.2975 | [-1.2619,+1.9088] | inconclusive | 1.1815 | 0.9950 |
| videomae | ucf_crime | visionzip_internal | 9 | 70 | 80.5487 | 80.2419 | -0.3068 | [-2.2432,+1.5066] | inconclusive | 1.2859 | 1.0406 |
| videomae | ucf_crime | visionzip_internal | 12 | 50 | 80.5487 | 81.2332 | +0.6844 | [-0.9148,+2.3917] | inconclusive | 1.0448 | 0.9839 |
| videomae | ucf_crime | visionzip_internal | 12 | 70 | 80.5487 | 81.4637 | +0.9150 | [-0.7597,+2.6814] | inconclusive | 1.0716 | 1.1549 |
| videomae | xd_violence | attention_topk | 6 | 50 | 70.3001 | 72.0909 | +1.7908 | [+0.3319,+3.2753] | supported | 1.4111 | 1.0053 |
| videomae | xd_violence | attention_topk | 6 | 70 | 70.3001 | 71.0013 | +0.7012 | [-1.2103,+2.6608] | inconclusive | 1.5960 | 0.9970 |
| videomae | xd_violence | attention_topk | 9 | 50 | 70.3001 | 69.8897 | -0.4104 | [-1.2914,+0.4741] | inconclusive | 1.2139 | 0.9851 |
| videomae | xd_violence | attention_topk | 9 | 70 | 70.3001 | 68.6039 | -1.6962 | [-3.0061,-0.4692] | inconclusive | 1.2978 | 1.0084 |
| videomae | xd_violence | attention_topk | 12 | 50 | 70.3001 | 71.1863 | +0.8862 | [-0.0628,+1.8633] | supported | 1.0629 | 1.0138 |
| videomae | xd_violence | attention_topk | 12 | 70 | 70.3001 | 70.8725 | +0.5724 | [-0.4958,+1.6811] | supported | 1.0675 | 1.0009 |
| videomae | xd_violence | visionzip_internal | 6 | 50 | 70.3001 | 71.5228 | +1.2227 | [-0.6762,+3.1513] | inconclusive | 1.4315 | 1.0183 |
| videomae | xd_violence | visionzip_internal | 6 | 70 | 70.3001 | 70.1969 | -0.1032 | [-2.3150,+2.1568] | inconclusive | 1.5880 | 0.9805 |
| videomae | xd_violence | visionzip_internal | 9 | 50 | 70.3001 | 68.5450 | -1.7551 | [-2.5520,-0.9679] | inferior | 1.1815 | 0.9965 |
| videomae | xd_violence | visionzip_internal | 9 | 70 | 70.3001 | 67.5841 | -2.7160 | [-3.7795,-1.6786] | inferior | 1.2859 | 0.9858 |
| videomae | xd_violence | visionzip_internal | 12 | 50 | 70.3001 | 71.1481 | +0.8481 | [+0.1226,+1.6232] | supported | 1.0448 | 1.0035 |
| videomae | xd_violence | visionzip_internal | 12 | 70 | 70.3001 | 70.9725 | +0.6724 | [-0.2113,+1.5674] | supported | 1.0716 | 1.0364 |

## XD step AP补充（不另算CI、不替换主指标）

| Encoder | 配置 | dense stepAP% | method stepAP% |
|---|---|---:|---:|
| timesformer | attention_topk pre6 remove50 | 70.8385 | 70.9038 |
| timesformer | attention_topk pre6 remove70 | 70.8385 | 70.5281 |
| timesformer | attention_topk pre9 remove50 | 70.8385 | 72.0090 |
| timesformer | attention_topk pre9 remove70 | 70.8385 | 71.8074 |
| timesformer | attention_topk pre12 remove50 | 70.8385 | 70.9586 |
| timesformer | attention_topk pre12 remove70 | 70.8385 | 70.0652 |
| timesformer | visionzip_internal pre6 remove50 | 70.8385 | 71.4109 |
| timesformer | visionzip_internal pre6 remove70 | 70.8385 | 71.1645 |
| timesformer | visionzip_internal pre9 remove50 | 70.8385 | 71.1910 |
| timesformer | visionzip_internal pre9 remove70 | 70.8385 | 70.8258 |
| timesformer | visionzip_internal pre12 remove50 | 70.8385 | 70.3973 |
| timesformer | visionzip_internal pre12 remove70 | 70.8385 | 69.8275 |
| videomae | attention_topk pre6 remove50 | 64.8401 | 61.8518 |
| videomae | attention_topk pre6 remove70 | 64.8401 | 57.6441 |
| videomae | attention_topk pre9 remove50 | 64.8401 | 63.4570 |
| videomae | attention_topk pre9 remove70 | 64.8401 | 61.8809 |
| videomae | attention_topk pre12 remove50 | 64.8401 | 63.9926 |
| videomae | attention_topk pre12 remove70 | 64.8401 | 63.2350 |
| videomae | visionzip_internal pre6 remove50 | 64.8401 | 58.0098 |
| videomae | visionzip_internal pre6 remove70 | 64.8401 | 53.6223 |
| videomae | visionzip_internal pre9 remove50 | 64.8401 | 62.8718 |
| videomae | visionzip_internal pre9 remove70 | 64.8401 | 61.5007 |
| videomae | visionzip_internal pre12 remove50 | 64.8401 | 64.5564 |
| videomae | visionzip_internal pre12 remove70 | 64.8401 | 63.8613 |

## 解释边界

- UCF历史训练契约原工作目录已移位，原SHA不变；最终审核使用保留项目中同SHA原契约副本，原路径及副本路径完整记录在JSON，未重建契约或修改协议身份。
- 单head seed0；视频级分层配对bootstrap seed0/10000，非劣margin0.5pp。CI下界须≥−0.5pp才支持非劣；点值小损失不是证明。
- 多配置未整体校正；没有方法间直接配对显著性，不依据test分数选位置/比例/权重。
- 净encoder含adapter/readout/plugin、resident输入；pipeline含16完整训练视频原native CPU准备/H2D/encoder/D2H/固定head/frame projection，不含FeatureStore落盘、权重加载与源SHA预查。
- Encoder-only计时使用固定UCF训练正常视频Normal_Videos001_x264输入，B1/B8分别测量；同encoder的XD表行复用该encoder时序，并非另测XD encoder。完整视频pipeline则按UCF/XD各自16个完整训练视频分别测量。
- 效率为三独立进程配对中位倍率；GPU独占，CPU/存储共享、OS页缓存未控制；倍率不是速度显著性。CPU生产优化吞吐单列，不替换原native基准。
- 原所有228+228效率组与uniform历史完整保留；本报告仅按明确用户范围保留活动156+156，保留两方法全部原固定位置和预算，不按分数只报告最佳格；uniform排除发生于用户查看历史模型正式结果之后，必须披露为后验范围变更，不能据此推断新encoder上的uniform质量。
- 最终报告上传终态以本地cloud-verification.json为准，只有verified=true才算整套交付完成。
