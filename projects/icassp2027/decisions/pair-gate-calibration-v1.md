# 单线性 gate 校准预算 v1

本轮校准用于检验用户要求的最小训练插件，不预设它会优于随机保留。均值合并已在固定
fit26的四模型上增加了表示扰动，因此均值只保留为零初始化状态和消融，不定为免训练主方法。
不加入第二个网络、额外损失或根据测试结果改变层、预算、初始化。

训练资产使用原完整explore128的相同128个fit身份，正常/正视频各64。实际输入文件以官方
verified根的SHA为准；本机已有副本与该128项来源摘要、大小均一致，每次执行重新核验字节。
manifest SHA为`7dc3092e63294c65cc8896292cf4f1fea9d95cbaa3f7c5cb15153aa5435b64ac`，
cases SHA为`18210e83bca96fab275cfa3c12e22d746cdc07eb0272b62d7dc54e750c6f2d5c`。
完整role lock仍是`53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59`。
每视频从均匀8个完整窗口固定取index1和6，共256个clip；按实际native帧数、stride2与显式
`stride1_if_needed`采样。只有视频级标签可留在审计侧，不进入打分器、损失或权重。

每个encoder分别学习一个无bias向量`w[D]`。当前层每个水平相邻pair的两个线性分数经过
softmax得到加权均值，训练与部署同式。TimeSformer沿完整时间轨迹平均分数，并共享同一
pair权重；V-JEPA2采用左成员原位置作为明确近似anchor。位置算法保持原生实现。

固定相对深度0.5，patch保留数量减半；12层模型在index5之后，V-JEPA2在index11之后。
冻结主干所有参数且保持eval；仅优化w，初始化为0。预算为3 epochs、batch1、AdamW、
learning rate0.001、weight decay0、seed0。局部CPU generator产生每epoch完整随机排列，
不按类别排序训练；使用最终epoch，不看官方测试选checkpoint。

唯一目标是逐clip的actual adapter pooled相对平方误差：

```text
loss = sum((pooled_reduced - stop_gradient(pooled_dense)) ** 2)
       / sum(pooled_dense ** 2)
```

dense teacher仅在fit校准时使用，可缓存detach后的CPU小向量；部署没有teacher输入。
V-JEPA2显式梯度入口和公共掩码池化要分别核查dense、reduced与actual adapter readout一致。
各encoder保持自己的原生pooled定义，不能把分类概率或任意token均值当作另一种读出。

每次训练记录全部输入帧与像素摘要、真实设备/dtype、代码/权重/校准身份、逐步loss/梯度、
参数更新、主干无梯度以及磁盘checkpoint回读。任何失败保留，不发布completed。
本机校准与服务器推理的库/设备身份分别记录；正式使用前须核对权重、层、几何、读出和
必要的运行时数值一致性，不伪造相同代码指纹，也不宣称跨encoder参数零样本迁移。

这份预算不等于训练已完成或VAD有效性已确认。质量仍按相同完整检测头预算、同一测试输入
和0.005容忍度检验；失败时如实报告，不追加复杂组件来改变这次预先固定的比较。
