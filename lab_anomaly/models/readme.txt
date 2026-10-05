模型事实：
- VideoMAEv2Encoder：固定 clip 的无状态编码器，输出每个 clip 一个 embedding；旧文件只重导出 VADBench 内的唯一实现。
- MILClassifier：把一个视频的多个 embedding 聚合为 normal/anomaly 二分类 logits。
- ranking_loss：仅使用视频级正常/异常 bag 的弱监督排序损失。

它没有视觉/语言 KV cache、跨视频状态、帧级标签投影或 VADBench adapter 接口。
模型加载使用 Hugging Face trust_remote_code=True，实际兼容性取决于本地 Transformers 与上游模型代码。
详细说明见 README.md。
