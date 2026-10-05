# F05｜既有训练观察的尺度与合并诊断复核

日期：2026-09-20。状态：`posthoc_existing_train_diagnostic`。本次真实执行CPU分析，但没有新增独立视频、像素观察或检测评测，不作为新性质确认。

结论：旧均值合并相对于同pair随机保留造成的V2最终表示扰动，在正常和含异常视频中都存在；旧早层local cosine差异又与视频级范数明显关联。因此当前不能把均值合并负结果写成“异常信息特别易损”，也不能把F04直接翻译成“高范数token更重要”。

## 数据和执行

脚本：`work/codex-takeover-20260920/idea/audit_existing_train.py`。输出：同目录`existing-train-audit.json`，绑定两个输入CSV的SHA-256。

2026-09-20实际命令：

```text
python work/codex-takeover-20260920/idea/audit_existing_train.py
python -m compileall -q src tests work/codex-takeover-20260920/idea/audit_existing_train.py
```

两条命令退出码均0。解释器为`C:/Users/lenovo/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe`，NumPy 2.4.3，CPU；本次没有模型加载、GPU任务或环境安装。compileall是语法检查，不等同于单元测试或数值验收。仅新增研究目录脚本及卡片，公共src没有修改。

## 配对表示扰动

来源为已有train26 neutral pair-mean导出的逐视频结果：13正常、13含异常，每视频两clip。每个视频计算`pairmean relative_l2 − pairedrandom relative_l2`，视频内已有等权clip均值。10,000次视频bootstrap，seed=20260920；不把clip或token当独立样本。

| Encoder | 标签组 | 平均配对差 | 探索性95%区间 | 正差视频 |
|---|---|---:|---|---:|
| VideoMAEv2 | 正常 | +0.211927 | [+0.156803,+0.273902] | 13/13 |
| VideoMAEv2 | 含异常 | +0.199730 | [+0.154912,+0.246071] | 13/13 |
| TimeSformer | 正常 | +0.037708 | [+0.017812,+0.057295] | 10/13 |
| TimeSformer | 含异常 | +0.011739 | [−0.017580,+0.037844] | 9/13 |

正差表示均值合并的最终pooled表示相对dense更偏离。这既不是检测性能，也不证明两标签组效应相等。未作新的matched比较、差中之差检验或因果论断。

## F04 的尺度敏感性

读取既有V2 fit128、block2输入的每视频local cosine与activation_norm_median；P16同名norm重复项排除，保持128个独立视频。两者Pearson相关为−0.730381。

事后固定线性描述模型的标签系数：调整motion/brightness三档分类变量后为+0.045905；再加入视频级范数中位数后为+0.012016。此处没有bootstrap区间，不称统计显著或消除混杂。范数本身是模型内部状态，可能是关联过程的一部分，调整并不构成因果控制。

该诊断只说明局部方向相似与尺度共同变化，不能评判pair内选大范数成员是否有效。下一步仍需要同pair排序/预算、仅改变成员选择的消融；新局部覆盖探针同时记录欧氏与cosine敏感性，避免仅靠角度相似度命名冗余。

尚未确定：V2+CLIP、UCF+XD上的同源性质；局部代表规则是否保持后缀表示/检测质量；插件净开销。没有测试集参与本次分析。

## 追加的固定站点检查：不支持直接假设MLP输出更冗余

同一CPU脚本追加读取block2的既有P04值，未扫描/选择新层。正常/含异常两组local cosine均值在MLP的LayerNorm输出分别为0.55108/0.58524，在MLP分支残差相加前输出为0.46575/0.49497。结果不支持“MLP分支输出方向比输入更相似”的假设；没有测量配对token幅值或实际共享误差，因此也不证明所有分支共享无效。

不据此开新GPU候选。MLP共享且保留原residual的算子上位结构也已有[ToMeSD，CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Bolya_Token_Merging_for_Fast_Stable_Diffusion_CVPRW_2023_paper.pdf)：其在分支计算前后执行merge/unmerge，再加回跳连。这个思路不能只因应用于VAD就称为新机制。
