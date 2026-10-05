# 六格实验范围合同 v1

本文件绑定本轮最终实验的六个格子：`videomaev2 / videomae / timesformer × UCF-Crime / XD-Violence accepted3950`。
它是范围和门禁合同，不替代训练侧选择完成后发布的 method-freeze-v2，也不单独授权读取官方测试分数。

每个格子必须使用对应 encoder 和对应数据集的完整训练视图，训练独立 UR-DMU dense head，固定 3000
optimizer steps、每步 64 normal + 64 anomalous bags 和最终 checkpoint。UCF 使用完整 1610 train；XD
使用 accepted3950，同时披露 declared3954 与 excluded4。旧的 screen600、64+64 视频池和 derived1598
训练视图都不能替代正式 full-train。

方法只在 `fit/confirm/select` 训练角色中登记和选择。当前范围保留 dense identity、global uniform、paired
random 和 pair-select；pair-mean 的负证据保留为 rejected。最终层、预算、seed、head mode、校准资产和
实际 selector 版本必须在正式 test 前写入新的 method-freeze-v2，测试集只用于冻结后的 evaluation。

正式导出必须按 UCF frame ROC-AUC、XD 作者梯形 frame AP 和单列 sklearn `average_precision` 报告，附上
video-level 辅助指标、`Δ=method-dense`、按视频分层配对 bootstrap 95% CI、实际 token/clip 保留率、端到端
和插件开销、吞吐、延迟、峰值显存，以及 manifest/view/head/code/environment SHA。CI 下界未达到 `-0.005`
时只能报告为 inconclusive 或失败，不能写成非劣。
