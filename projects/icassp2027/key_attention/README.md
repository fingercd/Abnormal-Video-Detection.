# Key / Attention 单点压缩消融（2026-10-13）

用户已批准开始实施；这是独立于旧 L2 PairSelect 的新开发轮次，旧协议与结果不改写。
本目录方案不是旧 test 冻结合同的修订，也不自动授予新方法正式 test 访问。

## 已确认矩阵

- 两种候选：`attention_topk`、`visionzip_internal`；对照：`uniform`。
- 12-block 模型在第6/9/12 block（index 5/8/11）的 attention residual 后、norm2/MLP 前单次压缩。
- 删除50%/70%，即保留0.50/0.30；nearest-half-up取整。
- 共9个方法×位置结构、18个压缩条件；每encoder×dataset另设共享dense。
- 不分动态/静态，不按hidden L2排序，不继承旧pair/group quota。
- ToMe仍为候选资料，不进入本轮矩阵；标准单轮不能删除70%。

## 算法与来源

算法参考本机 `D:/PythonProject/压缩框架/MLLM-Comp-Platform`：
`mllm_comp/compressors/encoder_level/attention_topk.py` 和
`mllm_comp/compressors/image/visionzip.py`。具体参考文件SHA随source-provenance.json记录。
此处为独立batched PyTorch适配，不运行时依赖该VLM工程。

- AttentionTopK：原生softmax probability按head/query平均，得到每个key token收到的attention，stable降序选K，恢复原索引顺序。
- Uniform：原生patch序列linspace→round→int64，不计算attention/key信号。
- VisionZip：固定dominant/contextual **4:1**（本轮获批方案，不是外部默认27:5）；预算内nearest-half-up分配并各至少1。
  dominant按相同received attention选；remaining按native顺序、整数prefix stride取context targets；
  mean-head原生key经余弦归一化后，将非target剩余token分到最相似target。
  数值沿参考实现 `target + mean(assigned non-targets)`，**不是全组mass均值**；FP32累加后转回原dtype。
  输出按source group最小索引排序，保存representative、source→output和mass。
- 原生最终pooling不改，不额外mass加权或比例注意力修正；表示偏移作为direct_insert待检验现象。
  mass用于来源审计，不表示本方法数学上保持了dense均值。

## 工程边界与验收

当前首阶段仅接入审计过的native VideoMAEv2，无CLS，原生mean pooling。
VideoMAE与TimeSformer尚不宣称已接通；后者需完整空间轨迹及分离时空attention适配，失败必须显式报告。
最后block只缩短MLP，不能声称减少已完成attention的计算。

真实key来自同次attention输入的K权重切片独立投影（原生functional QKV无法直接module hook）；
记录额外开销并与full-QKV切片核验。Attention直接观察原生probabilities，不重算QK，不切换backend。
native forward按AST验证；unknown forward/bias/mask/checkpoint配置失败关闭。
context只允许一次eval forward，退出恢复instance forward与hooks，不改全局模型实现。

QA先使用旧锁定fit集合中的Abuse005与Normal022，各uniform8窗口取1/6，batch=2、seed=0。
这些是工程样本，**不是完整训练、正式检测质量或效率基准**。所有样本必须train且role lock为fit。
比对dense、3处capture-on identity、18压缩条件、退出后dense恢复；FP16 pooled parity预设atol/rtol=1e-3，
独立K与full-QKV参考预设atol/rtol=5e-3；记录实际误差，不依结果修改容差。
源数1568时保留784或470（实际70.0255%删除），同预算对照精确匹配。

## 后续正式协议

QA通过后才进入训练侧更大固定cohort，之后生成独立新方法冻结合同与正式提取/评分许可。
主比较direct_insert，复用相同encoder×dataset×seed的已验收dense head；不重训、不跨数据集共享。
UCF train1610/test290；XD accepted3950/test800（declared3954/excluded4保留）。
UCF frame ROC-AUC；XD作者梯形PR-AUC主列、step AP另列；10,000次视频分层配对bootstrap95% CI。
Delta=method-dense，CI下界>=-0.005方可支持非劣；不根据正式test回改方法、层、比例、阈值或checkpoint。
正式效率短时独占且计入全部插件开销，分encoder/adapter/end-to-end；此QA的host计时不入速度表。

## 记录

代码、配置和原始来源SHA按独立快照保存；重要版本Git tag，旧资产不删除。
SwanLab由已有本机登录的online控制器实时接收远端EVENT上传，不向服务器复制凭据或更改共享环境。
云端完成后回读状态和完成标量；失败保留本地日志、远端PID/回执并补传，不盲目重复启动。

## 第三方声明

参考框架许可为MIT，Copyright (c) 2026 Hao Wu (吴浩)。参考选择/合并实现遵循以下许可：

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
