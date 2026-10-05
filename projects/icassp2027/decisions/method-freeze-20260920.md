# 方法冻结契约 v1（2026-09-20）

**冻结决定**：用户于 2026-09-20 明确批准（"解锁，这个肯定是要看的"）。本契约冻结 ICASSP2027 WSVAD 主实验的方法范围，冻结后方可执行 official_frame 官方测试集评分。冻结后新增方法须另立契约版本并披露其对方法选择的影响。

## 1. 冻结的方法范围

- **Encoder**（冻结主干，全程不微调）：VideoMAEv2-B、VideoMAE-B、TimeSformer-B。V-JEPA 2 已按吞吐门槛判定退出检测实验（2026-09-19 用户终裁），不在本契约范围。
- **检测后端**：UR-DMU（作者源码 40cfdf5 的双记忆+时序模块），dense 头为基线，development 预算 3000 步、64+64 bags、固定 seed、末位 checkpoint。
- **压缩插件候选（性质驱动，免训练主线）**：**选择类算子**——邻接 token 对按实测性质保留一员，不修改表示本身（典型形态：对内范数选择 pair-select；预算档 50% 起，阶梯含 75%/25%）。
- **对照**：同预算均匀/随机选择对照，同一冻结头；可选等预算 refit-head 补充列。
- **合并类算子（如 pair-mean）已被证据否决**，不属于本方法（见 §2 证据）。
- TF/LoRA 可训练版本：与免训练版同机制同族，本轮默认不做；若做，另立补充契约且同预算对照。

## 2. 性质证据摘要（property-evidence 回执指针）

- F04 四模型观察（fit128，1 万次视频级 bootstrap，motion×brightness 匹配）：异常窗口早期层局部空间冗余更高（local cosine 正向差，4/4 encoder 方向一致，V2/VMA 效应量最大）。
- S7 探索（v0 诊断口径，exploratory 身份）：
  - 合并算子负结果：pairmean held-out AUC 0.583 < 随机 0.750（videomaev2，50% 预算）——均值合并的分布偏移不可忽略；
  - 选择算子在 videomaev2 达预登记门槛（0.833 vs 0.750，dense 0.903）；
  - 跨 encoder 在 12 视频口径未一致复现（videomae 饱和、timesformer 反转）——正式对比须在 select161/test290 规模重测，如实报告。
- 证据产物：`work/s7-idea-a01/`（finding cards）、NAS `v0-explore/`、F04 观察回执。

## 3. 评测纪律

- UCF 主指标 frame ROC-AUC（XD 为 frame AP，另议）；视频级仅辅助。
- F1 类阈值只在允许的验证数据上固定；测试集只评测，不用于选方法/层/预算/checkpoint。
- 全部评测记录 encoder/权重/checkpoint sha/视图契约 sha/帧覆盖率；缺测保 null。

## 4. 生效

本文件 SHA-256 由 `method-freeze-contract-v1.json`（机器可读版，含性质证据回执指针与本文件 SHA）绑定；official_frame 代码门校验该 JSON 存在且 SHA 匹配后放行。冻结前已产生的 test290 特征提取（sealed-test 角色）合法，评测零访问原则维持到本契约生效。
