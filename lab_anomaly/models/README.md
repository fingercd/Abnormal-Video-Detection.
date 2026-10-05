# `lab_anomaly/models`

本目录实现保留原型的固定 clip 编码与视频级 MIL 头。

## `vit_video_encoder.py`

此文件只为旧训练/推理代码重导出类；唯一实现位于 `vadbench.integrations.videomaev2_encoder`。`VideoMAEv2Encoder` 从 Hugging Face 以 `trust_remote_code=True` 加载 `OpenGVLab/VideoMAEv2-Base`（或配置的同类模型）。输入可以是处理后的 `(B,C,T,H,W)` tensor，或 RGB `uint8` HWC 帧嵌套列表；后者经模型 image processor 后在必要时转成 `(B,C,T,H,W)`。输出为 float32 `(B,D)` embedding，`D` 来自模型 config，Base 常见为 768。

`pooling=auto` 优先 `pooler_output`；否则取 `last_hidden_state` 的第一个 token。该选择依赖上游 remote-code 返回结构，当前目录没有锁定上游 revision 或对不同 Transformers 版本的端到端测试。其 meta-tensor 修复也会重建位置编码/CLS token；这是一条兼容性补丁，不能视为上游原生权重等价性保证。

`freeze_backbone()` 冻结所有 backbone 参数；`unfreeze_last_n_blocks()` 通过若干可能的模块路径或启发式寻找 Transformer block，仅解冻末尾 N 个。它不实现跨 clip state 或缓存复用。

## `mil_head.py`

`MILClassifier` 输入 `(B,N,D)` 和可选有效位 mask，输出视频级 `(B,C)` logits。两种聚合：

- `attn`：attention pooling 后线性分类；
- `topk`：先得到每 clip logits，再按训练标签（训练）或最大 logit（推理）选 top-k 平均。

可选 `anomaly_scorer` 为每个 clip 产生 `(B,N)` 的 sigmoid 分数。它只是排序损失的学习支路；不会自动转为帧级得分，也没有与时间标注对齐的校准。

`masked_softmax()` 假定每行至少一个有效 clip；全 false 行会对全 `-inf` 做 softmax，产生 NaN。训练代码通过只选择 `num_clips > 0` 的视频来避免该情形。

## `ranking_loss.py`

`mil_ranking_loss()` 使用异常 bag 最大分数应高于正常 bag 最大分数的 hinge 项，加上异常 bag 的稀疏与相邻 clip 平滑项。正/负 bag 数不一致时循环配对；任一类为空则返回不参与反传的零值。它针对视频级弱监督，而非官方 UCF-Crime 帧级 GT。
