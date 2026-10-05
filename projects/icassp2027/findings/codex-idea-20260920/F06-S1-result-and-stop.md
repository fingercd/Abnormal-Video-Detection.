# F06-S1结果：工程通过，当前不建议扩大该方向

2026-09-20，独立CPU审阅。研究负责人建议：**不将F06扩大到16视频/集，不实现按s分配后缀预算的selector。** 这批数据没有提供跨encoder、跨数据集、跨中立mask一致的可操作线索。每格仅4个选定训练视频，不能据此证明总体无效；结论是当前不足以优先投入下一批GPU，而不是显著性否定。

当前只有8个视频身份、每视频2窗口，在两个encoder重复运行。不是32个独立视频，也不继承旧F01独立确认；没有官方test或检测头评分。XD使用accepted3950训练集合，本实验不等待4坏源恢复，node1不参与。

## 工程与身份审计

读取`f06_s1_videomaev2.json`、`f06_s1_clip.json`及merged/paired_summary，独立从保存的pooled向量重算绝对MSE、cosine和max-abs。16个共享窗口的完整源窗口字段在两encoder间逐项一致，每窗16帧、stride2；merged与各独立receipt完全一致，paired summary逐行没有发现不一致。

| 项目 | VideoMAEv2 | CLIP |
|---|---|---|
| 每encoder窗口/视频 | 16窗口/8视频 | 16窗口/8视频 |
| dense_a、dense_b、identity的blocks6–11 token数 | 全部1568 | 全部197 |
| fixed/random的blocks6–11 token数 | 全部784 | 全部99（含CLS） |
| dense重复最大绝对差 | 0 | 1.90735e-6 |
| identity最大绝对差 | 4.76837e-7 | 1.66893e-6 |
| identity最大MSE | 5.27827e-15 | 5.71614e-14 |
| 所有arm的prefix标量最大差 | 0 | 2.38419e-7 |
| 实际hook dtype | float32 | float32 |

相对于worker的float32导出，CPU float64重算MSE最大差分别为8.44e-9/4.86e-9，cosine最大差约2.38e-7，属计算精度差异。identity的cosine漂移可能有约1e-7负值，不能当作负的真实距离收益。

信号来自每次arm自身forward的block5输入和attention输出，压缩发生于完整block5之后。当前实现检查没有显示dense教师用于reduced决策。prefix在各arm间保持一致/数值噪声内；不是把压缩后的block6信号冒充F01。CLIP的prefix现在为每窗口`[196,16,768]`，修复了旧工程回执把两个8帧窗口合并成一个比值的问题。

本次V2实际解释器为production foundation，torch2.8、videomaev2 overlay transformers4.56.1，easydict真实module路径已记录（模块没有版本字段）。权重`model.safetensors` SHA为`ebffa1874066ea227330016e58a848e9e2bb1ff5605746459bded1122a42176d`；模型代码/config SHA也绑定。constructor虽含`use_half=true`，实际hook dtype为float32，不凭参数名宣称half实验。该环境身份不倒推旧classic smoke与生产特征数值等价。

CLIP实际torch2.0.1 float32，权重SHA为`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；原生center预处理身份不变，不是VadCLIP10crop后端质量评测。V2和CLIP记录了不同probe源码SHA，需要保留各自脚本版本；不能因同一脚本名就称完全相同源码。共享核心mask已通过实际索引核对，不依靠文件名证明。

## 随机mask确实共享，但有一项预登记偏离

两个encoder的49个2×2块都实际保留2格；去掉CLIP的CLS索引偏移后，V2全部8个tubelet时间片与CLIP的空间索引**成员及顺序完全一致**。V2没有重复或越界索引，首次relative-index错误已隔离在`f06_s1_videomaev2_mask_bug_attempt.json`，状态`partial_failures`，错误为indices重复；未混进成功表。

成功随机arm使用2×2组遍历顺序，**不是预登记要求的全局native升序**；fixed是升序。当前两个全局ViT的位置内容已在hidden内，后缀无重新注入位置，因而该共同排列在理想计算下具有排列等变性；但本轮没有额外跑sorted-vs-unsorted数值对照，不能默认为完全一致。

保留本次真实顺序及结果，不改旧receipt。鉴于本方向暂不扩大，不要求全部重跑；如果未来重用该随机arm进入论文/后继方法，应将新版本排序并做每encoder一个共享窗口的顺序parity，单独留回执。此项不是把无一致性结果改写成“工程错误导致失败”的理由。

## Bad.Boys与输入变化

提供的早期`probe-xd-fit-gpu7-20260920.json`和S1都使用`Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0`；在这些实际文件中未观察到`00-26-51_00-26-51`成员替换。若某条历史日志有不同文字，应另按日志身份保留，不能据此推定运行时换了视频。

早期CLIP每窗口8帧，S1每窗口16帧，中心采样起点随span改变，这是登记过的输入版本变化；旧CLIP窗口不能与S1直接配对。S1内部V2/CLIP源窗口完全一致。

## 视频级描述，不作显著性判断

每视频先等权平均两个窗口。下表为4个视频的Spearman描述值；`inverse norm`为`−log(X_rms)`，没有拟合模型、p值或置信区间，也不把两个encoder的原始MSE混池。

| Encoder/训练集 | s与fixed MSE | s与random MSE | inverse norm与fixed MSE | inverse norm与random MSE | s与inverse norm |
|---|---:|---:|---:|---:|---:|
| V2/UCF | 0.0 | +0.6 | +0.6 | 0.0 | −0.8 |
| V2/XD | 0.0 | 0.0 | −1.0 | −1.0 | 0.0 |
| CLIP/UCF | 0.0 | −0.4 | 0.0 | −0.4 | +1.0 |
| CLIP/XD | +0.4 | +0.8 | +0.2 | +0.4 | +0.8 |

主要观察：V2的fixed臂在两集均没有正向排序线索；CLIP的random臂跨集反向。CLIP/UCF中s与inverse norm排序完全相同，没有展示新的排序信息。只有CLIP/XD局部正向，不能挑这一格晋级。inverse norm本身也不稳定，不改为反范数selector。

视频平均绝对MSE范围为：V2/UCF fixed 0.03046–0.09774、random 0.04150–0.05841；V2/XD fixed 0.02379–0.04983、random 0.01078–0.03498；CLIP/UCF fixed 0.01069–0.07124、random 0.00870–0.04669；CLIP/XD fixed 0.00574–0.01583、random 0.00519–0.00995。扰动远高于重复噪声，当前不足来自关系不一致，而不是信号/输出精度不足。

标签组描述同样不支持“异常特别易损”的预设：V2/UCF含异常组扰动较大，但V2/XD及CLIP两集的含异常组平均扰动较小。每组只有2视频，不能作总体结论；全部正负方向保留。标签来自已登记smoke名单，仅用于汇总，没有进入选择函数。

## 收敛决定与后继边界

建议root停止F06的规模扩展和adaptive-budget实现，归档为“clip级已确认关联尚未转换为跨模型/数据集可操作规则”的有界诊断。它不推翻旧F01，也不证明所有压缩信号无效。

不为挽救这条线换层、翻转方向、挑mask、单独留下CLIP/XD或增加新网络。若后续重新启动，必须有独立的新训练侧依据和新配置版本，不能把这批已读结果称未触及确认。接下来的研究资源应先整合已有训练侧性质和V2成员消融的可用结果，缺乏新机制证据时明确保留缺口，而不是继续扩样本碰碰运气。

审计代码：`work/codex-takeover-20260920/idea/audit_f06_s1.py`；结果：同目录`f06-s1-independent-audit.json`，包含所有输入SHA、16个encoder×video描述行、环境/权重身份及偏离。实际执行该CPU脚本及`python -m compileall -q src tests ...`，退出0；本审阅未启动GPU或正式测试评分。`issues=[]`表示已执行的结构/算术一致性检查通过，不表示没有上述协议偏离或论文方法成立。
