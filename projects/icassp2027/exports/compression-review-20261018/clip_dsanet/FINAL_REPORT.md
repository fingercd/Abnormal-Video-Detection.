# CLIP ViT-B/16 + DSANet 冻结三方法压缩报告

生成时间：2026-10-15T02:19:07.122899+08:00

## 完成范围

38/38完整提取与固定发布检测头预测、36/36单次配对质量及云端核验完成。工作量4,105,577张编码图像，测试每时刻UCF crop5、XD crop0各一图；没有改成训练manifest的10crops。
独占GPU5完成114组encoder计时与114组完整原视频计时，三次独立进程、交错配置；完整流程共1,824次完整训练视频测量。没有依据test分数调整方法、层、预算、checkpoint或评分分支。

UCF主指标frame ROC-AUC；XD沿用DSANet历史核验的非插值step AP。梯形PR-AUC单列，不与step AP或上一轮VideoMAEv2主指标混用。主分支coarse；refined、三种帧级点指标、五个tIoU定位mAP完整保留在JSON。
CI为seed0、10000次视频级按弱标签分层配对，Δ=method−dense；容忍度0.5pp。补充点指标没有另算CI，不冒充主指标CI。单发布权重身份、作者训练seed未知，未多配置校正、未做方法间直接配对显著性。

## 本轮新dense基线

| 数据集 | frame ROC-AUC % | 非插值step AP % | 梯形PR-AUC % | 主指标 |
|---|---:|---:|---:|---|
| ucf | 88.867606 | 34.742314 | 34.729202 | ROC-AUC |
| xd | 94.979738 | 85.596471 | 85.596286 | step AP |

## 全部质量配置

| 数据集 | 方法 | 层 | 移除% | 主指标% | Δ pp | 95%CI pp | 判定 |
|---|---|---:|---:|---:|---:|---|---|
| ucf | attention_topk | 6 | 50 | 88.518695 | -0.348912 | [-1.215277, +0.408242] | 非劣证据不足 |
| ucf | attention_topk | 6 | 70 | 87.190058 | -1.677549 | [-3.453344, -0.197448] | 非劣证据不足 |
| ucf | attention_topk | 9 | 50 | 88.747443 | -0.120163 | [-0.832396, +0.542642] | 非劣证据不足 |
| ucf | attention_topk | 9 | 70 | 87.912143 | -0.955463 | [-2.326395, +0.181884] | 非劣证据不足 |
| ucf | attention_topk | 12 | 50 | 88.867604 | -0.000003 | [-0.000010, +0.000005] | 支持非劣 |
| ucf | attention_topk | 12 | 70 | 88.867607 | +0.000001 | [-0.000007, +0.000010] | 支持非劣 |
| ucf | uniform | 6 | 50 | 85.921779 | -2.945827 | [-6.151394, -0.691495] | 确认劣于界限 |
| ucf | uniform | 6 | 70 | 81.625777 | -7.241829 | [-10.643097, -4.506777] | 确认劣于界限 |
| ucf | uniform | 9 | 50 | 87.934489 | -0.933117 | [-2.930421, +0.698337] | 非劣证据不足 |
| ucf | uniform | 9 | 70 | 84.896396 | -3.971211 | [-6.165454, -2.123038] | 确认劣于界限 |
| ucf | uniform | 12 | 50 | 88.867605 | -0.000002 | [-0.000008, +0.000005] | 支持非劣 |
| ucf | uniform | 12 | 70 | 88.867604 | -0.000002 | [-0.000009, +0.000004] | 支持非劣 |
| ucf | visionzip_internal | 6 | 50 | 88.203585 | -0.664021 | [-1.605750, +0.145344] | 非劣证据不足 |
| ucf | visionzip_internal | 6 | 70 | 86.407404 | -2.460202 | [-4.359152, -0.937348] | 确认劣于界限 |
| ucf | visionzip_internal | 9 | 50 | 88.647951 | -0.219656 | [-0.946087, +0.426651] | 非劣证据不足 |
| ucf | visionzip_internal | 9 | 70 | 87.642020 | -1.225587 | [-2.671629, -0.014884] | 非劣证据不足 |
| ucf | visionzip_internal | 12 | 50 | 88.867604 | -0.000002 | [-0.000009, +0.000004] | 支持非劣 |
| ucf | visionzip_internal | 12 | 70 | 88.867606 | -0.000000 | [-0.000007, +0.000006] | 支持非劣 |
| xd | attention_topk | 6 | 50 | 85.956976 | +0.360505 | [-0.276952, +1.040418] | 支持非劣 |
| xd | attention_topk | 6 | 70 | 84.203485 | -1.392987 | [-2.920881, +0.260363] | 非劣证据不足 |
| xd | attention_topk | 9 | 50 | 85.435173 | -0.161298 | [-0.572433, +0.271602] | 非劣证据不足 |
| xd | attention_topk | 9 | 70 | 84.487239 | -1.109232 | [-2.133651, -0.010546] | 非劣证据不足 |
| xd | attention_topk | 12 | 50 | 85.596469 | -0.000002 | [-0.000009, +0.000005] | 支持非劣 |
| xd | attention_topk | 12 | 70 | 85.596470 | -0.000001 | [-0.000008, +0.000006] | 支持非劣 |
| xd | uniform | 6 | 50 | 85.612363 | +0.015892 | [-1.054397, +1.223827] | 非劣证据不足 |
| xd | uniform | 6 | 70 | 79.800782 | -5.795689 | [-7.777264, -3.971390] | 确认劣于界限 |
| xd | uniform | 9 | 50 | 83.535955 | -2.060516 | [-3.616175, -0.574572] | 确认劣于界限 |
| xd | uniform | 9 | 70 | 77.494503 | -8.101969 | [-11.000465, -5.395594] | 确认劣于界限 |
| xd | uniform | 12 | 50 | 85.596470 | -0.000002 | [-0.000009, +0.000006] | 支持非劣 |
| xd | uniform | 12 | 70 | 85.596470 | -0.000001 | [-0.000008, +0.000006] | 支持非劣 |
| xd | visionzip_internal | 6 | 50 | 85.581248 | -0.015224 | [-0.991442, +0.929775] | 非劣证据不足 |
| xd | visionzip_internal | 6 | 70 | 83.654824 | -1.941647 | [-3.711893, -0.164259] | 非劣证据不足 |
| xd | visionzip_internal | 9 | 50 | 85.628732 | +0.032261 | [-0.464948, +0.605720] | 支持非劣 |
| xd | visionzip_internal | 9 | 70 | 84.514577 | -1.081894 | [-2.190179, +0.090486] | 非劣证据不足 |
| xd | visionzip_internal | 12 | 50 | 85.596468 | -0.000003 | [-0.000010, +0.000003] | 支持非劣 |
| xd | visionzip_internal | 12 | 70 | 85.596471 | +0.000000 | [-0.000006, +0.000007] | 支持非劣 |

非劣统计：{"inconclusive": 15, "supported": 14, "inferior": 7}。CI低于0支持下降，但只有整个CI低于−0.5pp才确认劣于界限。

## 补充帧级点指标（不另算CI）

| 数据集 | 方法/层/移除% | coarse ROC-AUC% | coarse stepAP% | coarse 梯形PR-AUC% | refined ROC-AUC% | refined stepAP% | refined 梯形PR-AUC% |
|---|---|---:|---:|---:|---:|---:|---:|
| ucf | attention_topk/b6/remove50 | 88.518695 | 34.827611 | 34.812266 | 88.518696 | 34.826548 | 34.812275 |
| ucf | attention_topk/b6/remove70 | 87.190058 | 32.332867 | 32.312385 | 87.190056 | 32.331105 | 32.312397 |
| ucf | attention_topk/b9/remove50 | 88.747443 | 33.931120 | 33.918571 | 88.747445 | 33.930918 | 33.918588 |
| ucf | attention_topk/b9/remove70 | 87.912143 | 32.634153 | 32.620913 | 87.912141 | 32.634048 | 32.620919 |
| ucf | attention_topk/b12/remove50 | 88.867604 | 34.742275 | 34.729184 | 88.867605 | 34.741671 | 34.729202 |
| ucf | attention_topk/b12/remove70 | 88.867607 | 34.742287 | 34.729200 | 88.867608 | 34.741644 | 34.729201 |
| ucf | uniform/b6/remove50 | 85.921779 | 33.212180 | 33.197483 | 85.921780 | 33.211974 | 33.197544 |
| ucf | uniform/b6/remove70 | 81.625777 | 25.052247 | 25.039979 | 81.625777 | 25.052276 | 25.039975 |
| ucf | uniform/b9/remove50 | 87.934489 | 35.593830 | 35.577344 | 87.934488 | 35.593738 | 35.577289 |
| ucf | uniform/b9/remove70 | 84.896396 | 28.490105 | 28.475486 | 84.896396 | 28.490115 | 28.475527 |
| ucf | uniform/b12/remove50 | 88.867605 | 34.742301 | 34.729187 | 88.867604 | 34.741630 | 34.729201 |
| ucf | uniform/b12/remove70 | 88.867604 | 34.742308 | 34.729192 | 88.867605 | 34.741677 | 34.729210 |
| ucf | visionzip_internal/b6/remove50 | 88.203585 | 33.772942 | 33.755667 | 88.203587 | 33.773063 | 33.755781 |
| ucf | visionzip_internal/b6/remove70 | 86.407404 | 31.410546 | 31.398924 | 86.407404 | 31.410571 | 31.398825 |
| ucf | visionzip_internal/b9/remove50 | 88.647951 | 33.631032 | 33.618590 | 88.647951 | 33.630921 | 33.618615 |
| ucf | visionzip_internal/b9/remove70 | 87.642020 | 32.367915 | 32.354663 | 87.642020 | 32.367837 | 32.354682 |
| ucf | visionzip_internal/b12/remove50 | 88.867604 | 34.742304 | 34.729189 | 88.867604 | 34.741664 | 34.729187 |
| ucf | visionzip_internal/b12/remove70 | 88.867606 | 34.742292 | 34.729189 | 88.867607 | 34.741635 | 34.729203 |
| xd | attention_topk/b6/remove50 | 95.136461 | 85.956976 | 85.956780 | 95.136461 | 85.956976 | 85.956780 |
| xd | attention_topk/b6/remove70 | 94.537319 | 84.203485 | 84.203241 | 94.537319 | 84.203484 | 84.203241 |
| xd | attention_topk/b9/remove50 | 94.969831 | 85.435173 | 85.434983 | 94.969831 | 85.435173 | 85.434983 |
| xd | attention_topk/b9/remove70 | 94.644669 | 84.487239 | 84.487016 | 94.644669 | 84.487239 | 84.487016 |
| xd | attention_topk/b12/remove50 | 94.979737 | 85.596469 | 85.596285 | 94.979737 | 85.596469 | 85.596285 |
| xd | attention_topk/b12/remove70 | 94.979737 | 85.596470 | 85.596285 | 94.979737 | 85.596470 | 85.596285 |
| xd | uniform/b6/remove50 | 95.036670 | 85.612363 | 85.612156 | 95.036670 | 85.612363 | 85.612156 |
| xd | uniform/b6/remove70 | 92.280373 | 79.800782 | 79.800443 | 92.280373 | 79.800781 | 79.800443 |
| xd | uniform/b9/remove50 | 94.318855 | 83.535955 | 83.535703 | 94.318855 | 83.535955 | 83.535703 |
| xd | uniform/b9/remove70 | 91.617718 | 77.494503 | 77.494008 | 91.617718 | 77.494502 | 77.494008 |
| xd | uniform/b12/remove50 | 94.979737 | 85.596470 | 85.596284 | 94.979737 | 85.596469 | 85.596284 |
| xd | uniform/b12/remove70 | 94.979737 | 85.596470 | 85.596285 | 94.979737 | 85.596470 | 85.596285 |
| xd | visionzip_internal/b6/remove50 | 95.060717 | 85.581248 | 85.581040 | 95.060717 | 85.581247 | 85.581040 |
| xd | visionzip_internal/b6/remove70 | 94.372227 | 83.654824 | 83.654568 | 94.372227 | 83.654824 | 83.654568 |
| xd | visionzip_internal/b9/remove50 | 95.034063 | 85.628732 | 85.628546 | 95.034063 | 85.628732 | 85.628546 |
| xd | visionzip_internal/b9/remove70 | 94.678911 | 84.514577 | 84.514351 | 94.678911 | 84.514577 | 84.514351 |
| xd | visionzip_internal/b12/remove50 | 94.979737 | 85.596468 | 85.596283 | 94.979737 | 85.596468 | 85.596283 |
| xd | visionzip_internal/b12/remove70 | 94.979737 | 85.596471 | 85.596286 | 94.979737 | 85.596471 | 85.596286 |

## 原生定位mAP（不替代主帧指标）

| 数据集 | 配置 | 五个tIoU阈值与mAP%，按原作者返回值 | mean mAP% |
|---|---|---|---:|
| ucf | dense | 0.1: 19.9341；0.2: 15.0461；0.3: 13.5551；0.4: 9.7685；0.5: 7.5332 | 13.1674 |
| xd | dense | 0.1: 41.1737；0.2: 34.3703；0.3: 28.4237；0.4: 22.2636；0.5: 17.9372 | 28.8337 |
| ucf | attention_topk/b6/remove50 | 0.1: 21.1139；0.2: 15.8386；0.3: 12.7414；0.4: 10.2711；0.5: 7.4891 | 13.4908 |
| ucf | attention_topk/b6/remove70 | 0.1: 12.9723；0.2: 9.5930；0.3: 8.2832；0.4: 5.3287；0.5: 4.8409 | 8.2036 |
| ucf | attention_topk/b9/remove50 | 0.1: 20.3605；0.2: 15.7866；0.3: 13.1153；0.4: 9.8262；0.5: 8.3251 | 13.4827 |
| ucf | attention_topk/b9/remove70 | 0.1: 19.4509；0.2: 16.2265；0.3: 12.5201；0.4: 9.2608；0.5: 5.2873 | 12.5491 |
| ucf | attention_topk/b12/remove50 | 0.1: 19.9341；0.2: 15.0461；0.3: 13.5551；0.4: 9.7685；0.5: 7.5332 | 13.1674 |
| ucf | attention_topk/b12/remove70 | 0.1: 19.9341；0.2: 15.0461；0.3: 13.5551；0.4: 9.7685；0.5: 7.5332 | 13.1674 |
| ucf | uniform/b6/remove50 | 0.1: 14.6685；0.2: 11.4347；0.3: 9.4959；0.4: 7.6136；0.5: 3.5521 | 9.3530 |
| ucf | uniform/b6/remove70 | 0.1: 11.0433；0.2: 7.8999；0.3: 6.5330；0.4: 4.3865；0.5: 3.3877 | 6.6501 |
| ucf | uniform/b9/remove50 | 0.1: 15.9110；0.2: 12.6172；0.3: 10.3448；0.4: 6.8626；0.5: 4.1834 | 9.9838 |
| ucf | uniform/b9/remove70 | 0.1: 13.1799；0.2: 10.0719；0.3: 8.1628；0.4: 6.5626；0.5: 5.1092 | 8.6173 |
| ucf | uniform/b12/remove50 | 0.1: 19.9341；0.2: 15.0461；0.3: 13.5551；0.4: 9.7685；0.5: 7.5332 | 13.1674 |
| ucf | uniform/b12/remove70 | 0.1: 19.9341；0.2: 15.0461；0.3: 13.5551；0.4: 9.7685；0.5: 7.5332 | 13.1674 |
| ucf | visionzip_internal/b6/remove50 | 0.1: 20.3550；0.2: 14.7088；0.3: 11.7926；0.4: 9.2755；0.5: 6.5257 | 12.5315 |
| ucf | visionzip_internal/b6/remove70 | 0.1: 14.0800；0.2: 9.9796；0.3: 8.2177；0.4: 5.6549；0.5: 4.6758 | 8.5216 |
| ucf | visionzip_internal/b9/remove50 | 0.1: 19.9853；0.2: 16.5166；0.3: 11.5235；0.4: 9.1422；0.5: 7.6905 | 12.9716 |
| ucf | visionzip_internal/b9/remove70 | 0.1: 19.7510；0.2: 16.7928；0.3: 13.1233；0.4: 9.9780；0.5: 6.3115 | 13.1913 |
| ucf | visionzip_internal/b12/remove50 | 0.1: 19.9341；0.2: 15.0461；0.3: 13.5551；0.4: 9.7685；0.5: 7.5332 | 13.1674 |
| ucf | visionzip_internal/b12/remove70 | 0.1: 19.9341；0.2: 15.0461；0.3: 13.5551；0.4: 9.7685；0.5: 7.5332 | 13.1674 |
| xd | attention_topk/b6/remove50 | 0.1: 41.7223；0.2: 35.1759；0.3: 29.6900；0.4: 23.1431；0.5: 18.5765 | 29.6616 |
| xd | attention_topk/b6/remove70 | 0.1: 42.3938；0.2: 36.2989；0.3: 30.4717；0.4: 24.6300；0.5: 18.9460 | 30.5481 |
| xd | attention_topk/b9/remove50 | 0.1: 40.8835；0.2: 35.1174；0.3: 28.8276；0.4: 22.8212；0.5: 17.7591 | 29.0818 |
| xd | attention_topk/b9/remove70 | 0.1: 40.0240；0.2: 34.5894；0.3: 28.2382；0.4: 22.1027；0.5: 17.6239 | 28.5156 |
| xd | attention_topk/b12/remove50 | 0.1: 41.1737；0.2: 34.3703；0.3: 28.4237；0.4: 22.2636；0.5: 17.9372 | 28.8337 |
| xd | attention_topk/b12/remove70 | 0.1: 41.1737；0.2: 34.3703；0.3: 28.4237；0.4: 22.2636；0.5: 17.9372 | 28.8337 |
| xd | uniform/b6/remove50 | 0.1: 42.0352；0.2: 36.1115；0.3: 29.8567；0.4: 23.7795；0.5: 19.0312 | 30.1628 |
| xd | uniform/b6/remove70 | 0.1: 39.5272；0.2: 35.0871；0.3: 29.7576；0.4: 23.9924；0.5: 19.0970 | 29.4923 |
| xd | uniform/b9/remove50 | 0.1: 42.6729；0.2: 36.0185；0.3: 30.4184；0.4: 23.8673；0.5: 18.7362 | 30.3426 |
| xd | uniform/b9/remove70 | 0.1: 40.0466；0.2: 35.6677；0.3: 29.8928；0.4: 24.3495；0.5: 19.6055 | 29.9124 |
| xd | uniform/b12/remove50 | 0.1: 41.1737；0.2: 34.3703；0.3: 28.4237；0.4: 22.2636；0.5: 17.9372 | 28.8337 |
| xd | uniform/b12/remove70 | 0.1: 41.1737；0.2: 34.3703；0.3: 28.4237；0.4: 22.2636；0.5: 17.9372 | 28.8337 |
| xd | visionzip_internal/b6/remove50 | 0.1: 42.2192；0.2: 36.0586；0.3: 29.5183；0.4: 23.9194；0.5: 18.8477 | 30.1126 |
| xd | visionzip_internal/b6/remove70 | 0.1: 42.6177；0.2: 36.3786；0.3: 30.3941；0.4: 24.3628；0.5: 19.1148 | 30.5736 |
| xd | visionzip_internal/b9/remove50 | 0.1: 40.8114；0.2: 34.9988；0.3: 28.5561；0.4: 22.8239；0.5: 17.9211 | 29.0223 |
| xd | visionzip_internal/b9/remove70 | 0.1: 40.5316；0.2: 34.7715；0.3: 28.8110；0.4: 23.0256；0.5: 17.8697 | 29.0019 |
| xd | visionzip_internal/b12/remove50 | 0.1: 41.1737；0.2: 34.3703；0.3: 28.4237；0.4: 22.2636；0.5: 17.9372 | 28.8337 |
| xd | visionzip_internal/b12/remove70 | 0.1: 41.1737；0.2: 34.3703；0.3: 28.4237；0.4: 22.2636；0.5: 17.9372 | 28.8337 |

## 效率完整表

| 边界 | 输入/数据集 | 方法 | 层 | 移除% | Batch | 延迟中位数 | 三轮配对倍率中位数 | peak allocated MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|
| encoder | shared_CLIP_UCF_training_input | attention_topk | 6 | 50 | 1 | 18.0315 ms | 0.2950× | 606.50 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 6 | 70 | 1 | 19.3490 ms | 0.2670× | 604.05 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 9 | 50 | 1 | 20.3346 ms | 0.2550× | 606.96 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 9 | 70 | 1 | 20.3476 ms | 0.2531× | 604.80 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 12 | 50 | 1 | 20.7449 ms | 0.2508× | 607.26 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 12 | 70 | 1 | 20.6061 ms | 0.2520× | 605.27 |
| encoder | shared_CLIP_UCF_training_input | identity | — | — | 1 | 5.1924 ms | 1.0000× | 603.93 |
| encoder | shared_CLIP_UCF_training_input | uniform | 6 | 50 | 1 | 18.5319 ms | 0.2807× | 605.33 |
| encoder | shared_CLIP_UCF_training_input | uniform | 6 | 70 | 1 | 18.0323 ms | 0.2923× | 604.51 |
| encoder | shared_CLIP_UCF_training_input | uniform | 9 | 50 | 1 | 18.9751 ms | 0.2736× | 604.57 |
| encoder | shared_CLIP_UCF_training_input | uniform | 9 | 70 | 1 | 19.0149 ms | 0.2794× | 604.69 |
| encoder | shared_CLIP_UCF_training_input | uniform | 12 | 50 | 1 | 19.2448 ms | 0.2698× | 605.04 |
| encoder | shared_CLIP_UCF_training_input | uniform | 12 | 70 | 1 | 19.1149 ms | 0.2694× | 605.45 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 6 | 50 | 1 | 20.4891 ms | 0.2555× | 606.50 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 6 | 70 | 1 | 20.3439 ms | 0.2555× | 604.51 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 9 | 50 | 1 | 21.4085 ms | 0.2483× | 607.14 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 9 | 70 | 1 | 21.2854 ms | 0.2476× | 604.69 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 12 | 50 | 1 | 22.0487 ms | 0.2474× | 607.61 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 12 | 70 | 1 | 21.8820 ms | 0.2490× | 605.45 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 6 | 50 | 32 | 75.2285 ms | 1.1211× | 964.51 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 6 | 70 | 32 | 64.3586 ms | 1.2809× | 956.56 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 9 | 50 | 32 | 84.3702 ms | 0.9918× | 979.42 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 9 | 70 | 32 | 79.3155 ms | 1.0550× | 981.33 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 12 | 50 | 32 | 93.8824 ms | 0.8913× | 988.23 |
| encoder | shared_CLIP_UCF_training_input | attention_topk | 12 | 70 | 32 | 92.9494 ms | 0.9003× | 996.22 |
| encoder | shared_CLIP_UCF_training_input | identity | — | — | 32 | 83.6808 ms | 1.0000× | 953.46 |
| encoder | shared_CLIP_UCF_training_input | uniform | 6 | 50 | 32 | 72.8110 ms | 1.1398× | 1001.30 |
| encoder | shared_CLIP_UCF_training_input | uniform | 6 | 70 | 32 | 63.2331 ms | 1.3255× | 974.80 |
| encoder | shared_CLIP_UCF_training_input | uniform | 9 | 50 | 32 | 82.1075 ms | 1.0228× | 976.73 |
| encoder | shared_CLIP_UCF_training_input | uniform | 9 | 70 | 32 | 76.9540 ms | 1.0899× | 980.99 |
| encoder | shared_CLIP_UCF_training_input | uniform | 12 | 50 | 32 | 91.6286 ms | 0.9135× | 992.18 |
| encoder | shared_CLIP_UCF_training_input | uniform | 12 | 70 | 32 | 90.6539 ms | 0.9231× | 1005.20 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 6 | 50 | 32 | 75.0503 ms | 1.1177× | 966.39 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 6 | 70 | 32 | 66.0179 ms | 1.2585× | 975.07 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 9 | 50 | 32 | 84.7103 ms | 0.9784× | 987.96 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 9 | 70 | 32 | 79.8664 ms | 1.0362× | 980.58 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 12 | 50 | 32 | 94.5345 ms | 0.8797× | 1003.19 |
| encoder | shared_CLIP_UCF_training_input | visionzip_internal | 12 | 70 | 32 | 94.0829 ms | 0.8873× | 1004.97 |
| pipeline | ucf | attention_topk | 6 | 50 | 32 | 24.1399 s_per_16_complete_videos | 0.9804× | 1654.45 |
| pipeline | ucf | attention_topk | 6 | 70 | 32 | 24.2965 s_per_16_complete_videos | 0.9597× | 1643.49 |
| pipeline | ucf | attention_topk | 9 | 50 | 32 | 25.3112 s_per_16_complete_videos | 0.9351× | 1654.45 |
| pipeline | ucf | attention_topk | 9 | 70 | 32 | 25.5521 s_per_16_complete_videos | 0.9122× | 1643.49 |
| pipeline | ucf | attention_topk | 12 | 50 | 32 | 26.3613 s_per_16_complete_videos | 0.8987× | 1654.45 |
| pipeline | ucf | attention_topk | 12 | 70 | 32 | 25.4777 s_per_16_complete_videos | 0.9298× | 1643.49 |
| pipeline | ucf | identity | — | — | 32 | 23.6679 s_per_16_complete_videos | 1.0000× | 1643.49 |
| pipeline | ucf | uniform | 6 | 50 | 32 | 21.2723 s_per_16_complete_videos | 1.1105× | 1643.49 |
| pipeline | ucf | uniform | 6 | 70 | 32 | 23.7921 s_per_16_complete_videos | 0.9948× | 1643.49 |
| pipeline | ucf | uniform | 9 | 50 | 32 | 23.8313 s_per_16_complete_videos | 0.9931× | 1643.49 |
| pipeline | ucf | uniform | 9 | 70 | 32 | 23.3442 s_per_16_complete_videos | 1.0139× | 1643.49 |
| pipeline | ucf | uniform | 12 | 50 | 32 | 22.9503 s_per_16_complete_videos | 1.0322× | 1643.49 |
| pipeline | ucf | uniform | 12 | 70 | 32 | 25.4282 s_per_16_complete_videos | 0.9249× | 1643.49 |
| pipeline | ucf | visionzip_internal | 6 | 50 | 32 | 24.6237 s_per_16_complete_videos | 0.9612× | 1654.54 |
| pipeline | ucf | visionzip_internal | 6 | 70 | 32 | 23.8070 s_per_16_complete_videos | 0.9942× | 1643.49 |
| pipeline | ucf | visionzip_internal | 9 | 50 | 32 | 23.3439 s_per_16_complete_videos | 1.0139× | 1654.54 |
| pipeline | ucf | visionzip_internal | 9 | 70 | 32 | 23.3536 s_per_16_complete_videos | 1.0135× | 1643.49 |
| pipeline | ucf | visionzip_internal | 12 | 50 | 32 | 24.2151 s_per_16_complete_videos | 0.9774× | 1654.54 |
| pipeline | ucf | visionzip_internal | 12 | 70 | 32 | 23.5982 s_per_16_complete_videos | 1.0030× | 1643.49 |
| pipeline | xd | attention_topk | 6 | 50 | 32 | 26.3031 s_per_16_complete_videos | 0.9285× | 1638.69 |
| pipeline | xd | attention_topk | 6 | 70 | 32 | 26.1557 s_per_16_complete_videos | 1.0033× | 1627.64 |
| pipeline | xd | attention_topk | 9 | 50 | 32 | 26.3046 s_per_16_complete_videos | 0.9819× | 1638.69 |
| pipeline | xd | attention_topk | 9 | 70 | 32 | 27.0348 s_per_16_complete_videos | 0.9554× | 1627.64 |
| pipeline | xd | attention_topk | 12 | 50 | 32 | 25.4235 s_per_16_complete_videos | 0.9200× | 1638.69 |
| pipeline | xd | attention_topk | 12 | 70 | 32 | 25.6313 s_per_16_complete_videos | 0.9320× | 1627.64 |
| pipeline | xd | identity | — | — | 32 | 25.8285 s_per_16_complete_videos | 1.0000× | 1627.64 |
| pipeline | xd | uniform | 6 | 50 | 32 | 26.1235 s_per_16_complete_videos | 0.9508× | 1627.64 |
| pipeline | xd | uniform | 6 | 70 | 32 | 26.3019 s_per_16_complete_videos | 0.9765× | 1627.64 |
| pipeline | xd | uniform | 9 | 50 | 32 | 26.8446 s_per_16_complete_videos | 0.9503× | 1627.64 |
| pipeline | xd | uniform | 9 | 70 | 32 | 25.2048 s_per_16_complete_videos | 0.9131× | 1627.64 |
| pipeline | xd | uniform | 12 | 50 | 32 | 24.7303 s_per_16_complete_videos | 0.9209× | 1627.64 |
| pipeline | xd | uniform | 12 | 70 | 32 | 23.5491 s_per_16_complete_videos | 0.9671× | 1627.64 |
| pipeline | xd | visionzip_internal | 6 | 50 | 32 | 23.8221 s_per_16_complete_videos | 0.9648× | 1638.69 |
| pipeline | xd | visionzip_internal | 6 | 70 | 32 | 23.5883 s_per_16_complete_videos | 1.0874× | 1627.64 |
| pipeline | xd | visionzip_internal | 9 | 50 | 32 | 24.9852 s_per_16_complete_videos | 1.0580× | 1638.69 |
| pipeline | xd | visionzip_internal | 9 | 70 | 32 | 26.1973 s_per_16_complete_videos | 1.0090× | 1627.64 |
| pipeline | xd | visionzip_internal | 12 | 50 | 32 | 25.9144 s_per_16_complete_videos | 0.9582× | 1638.69 |
| pipeline | xd | visionzip_internal | 12 | 70 | 32 | 26.5337 s_per_16_complete_videos | 0.9720× | 1627.64 |

## 解释与记录边界

- 第12层仅改变patch、保留CLS，之后没有跨token block；近dense质量是结构对照，不证明selector优越。
- encoder倍率用准备好的GPU输入到原pooled输出的净wall时间，包含原实际桥构造、hooks、额外Q/K、selector/gather和readout。GPU事件时间及分项原始样本另存；plugin分项只覆盖信号/选择/gather，不把分项之和冒充全净时延。
- DSANet为本项目历史核验的强发布基线，不据本轮称其当前全球SOTA。两官方测试集此前已用于历史/VideoMAEv2实验；本轮按用户预先冻结的完整矩阵执行，不称全研究过程test-blind，也不把不同head/输入/指标的跨模型数字直接归因于encoder优势。
- 完整流程从打开固定完整训练视频到真实coarse/class CPU分数，保持原16帧采样/crop/尾段与对应固定head；不写特征缓存。decoder、传输、encoder及head/readout阶段、每视频p50/p95和所有三轮原始值在JSON。
- GPU窗口按启动和各组前后CUDA PID/UUID核验，无停止其他项目；CPU与存储仍和生产共享，微小倍率受这些资源影响。暖文件策略及每配置训练视频预热固定，不把共享生产吞吐填进效率表。
- 原完整视频时序检测头非因果。来源视频时长/处理wall倍率仅表示离线处理速度，不称流式实时延迟。当前实现逐forward源校验和hook构造也计入净时间；batch1变慢或完整流程无收益均保留，不另改实现再挑有利数字。
- 原预测campaign的UCF真值SHA有cat无换行导致的前缀右花括号。严格门在首次CPU帧标注解析/bootstrap前拒绝。原记录保留，独立analysis binding引用原campaign SHA并记录纠正；实际真值文件、模型、输入、预测未改变。
- 失败原生检查/元数据错误、日志和云端失败回执保留。XD Attention12 remove70及VisionZip9 remove50被busy GPU保护门在启动前拒绝；无PID/无GPU输出核验后另起独立attempt。后续核实GPU1/2的CVPR-JY2 R5进程，依据已有共享空余显存授权使用精确PID/owner/命令/cwd名单；不停止它们，不放行未知PID。未重跑已计算任务，原成功/正常logger继续复用。最终汇总复用preview与既有计时，不重算bootstrap或GPU。
- 源码用字节等价annotated tree tags与SHA归档固定，未把当前脏工作树误称为HEAD版本；行尾正规化的初始tag保留为记录，采用后续byte-exact tag。未commit、push、公开发布或投稿。
- 未测能耗、完整FLOPs保持未测，不填估算数值。

[独占encoder云端回执](https://swanlab.cn/@ryfrcd693/VAD-DSANet-Compression-20261015/runs/w2iyxnvr)；[完整流程云端回执](https://swanlab.cn/@ryfrcd693/VAD-DSANet-Compression-20261015/runs/mv2zg766)。全部38预测和36质量链接、源SHA、完整CI和原始效率样本见final-results.json。
