# PairSelect 四系统重建版：论断—证据核对表

本文件与本次重新生成的 PDF、TeX 工程对应。由于上条回复所链接的四系统成品在本次运行环境中不存在，本次从用户上传的 `VAD_PairSelect_four_systems_20260924_r02.zip` 重新构建；不是旧三系统 PDF 改名，也不是可承诺逐字节等同于上一条链接文件的副本。

## 数据来源

工程 `evidence_snapshot/` 包含 62 份原包证据文件。`data/numeric_provenance.csv` 包含 340 条数值读取与换算记录，每条列明论断定位、原始路径、字段、值、换算与 SHA-256。`data/evidence_sha256.csv` 列明快照校验值。表格通过 `scripts/export_tables.py` 自动生成，未手工填写实验数值。

| 论文论断／表格 | 原始证据（相对 evidence_snapshot） | 字段／核对方式 | 保留的定义或范围 |
|---|---|---|---|
| 四系统的 encoder/head 身份 | `PACKAGE_README.md`；`PROMPT_FOR_GPT6PRO.md`；`evidence/dsanet/RESULTS_PUBLISHED_20260923.md` | 训练来源与发布权重身份；DSANet 质量 JSON 的 `checkpoint_origin=published` | 三条预训练视频骨干＋分别训练的 UR-DMU；CLIP＋作者 DSANet 权重，非四编码器从零训练 |
| 三视频系统 40% 质量及 UCF step AP | `evidence/three-video-encoders/references/quality_matrix_detail.csv` | 对应 dataset/encoder 的 `method=pair_select`；`dense_value`、`method_value`、历史 `xd_average_precision_*` 字段 | 历史字段名前缀 xd 不表示 UCF 没有 AP；统一乘 100 |
| 三视频系统 20%/60% 完整质量 | `evidence/three-video-encoders/quality/{dataset}/{encoder}/0p80_pair_select.json`、`0p40_pair_select.json` | `metric.{dense_value,method_value,ci_low,ci_high}`；`frame_ap_step.{dense,method}` | 文件里的 0p80/0p40 是保留率 |
| DSANet UCF 所有预算 | `evidence/dsanet/results_published/quality-ucf.json` | `results.coarse-{0.8,0.6,0.4}.frame_roc_auc` 与 `.frame_ap` | Dense 为本轮 raw CLIP 特征；不用旧 89.4446% 校准分数 |
| DSANet XD 所有预算 | `evidence/dsanet/results_published/quality-xd.json` | `results.coarse-{0.8,0.6,0.4}.frame_ap` 与 `.frame_pr_auc` | 主指标 step AP；与梯形 PR-AUC 分列显示，不混排 |
| 40% 在多数配置近无损 | 表 2 与上述质量源 | 八组主指标差值约 −0.90 至 +0.68 pp；三视频骨干最明显主指标退化约 −0.16 pp | 是点估计描述，非所有配置已证明统计等价 |
| 60% 较明显的三处回落 | DSANet 两个 quality JSON；`quality/ucf_crime/videomae/0p40_pair_select.json` | DSANet/UCF −1.8028 pp；DSANet/XD −1.2173 pp；VideoMAE/UCF −1.0534 pp | 所有负面格子保留 |
| XD 双指标方向差异 | `quality/xd_violence/videomae/0p40_pair_select.json` | 主指标＋0.7762 pp；step AP −1.3942 pp | 不能把其中一个上升改写成所有指标上升 |
| 表 1 实际 token 数 | `server_evidence/code-bsw-r01/src/vadbench/token_reduction/token_selection.py`；`server_evidence/dsanet-extension-20260923-r01/code/clip_bridge.py` | 分组取整、TimeSformer 对齐、CLIP CLS 保留；导出脚本复算 | VMAE/v2：1568→1254/941/627；TS：1569→1233/897/561；CLIP：197→158/119/79 |
| 表 3 三视频骨干 B32 | `evidence/batch-scaling/results/{encoder}/result.json` | `summary.32.{dense,keep_0.80,keep_0.60,keep_0.40}` 的延迟、吞吐、speedup | 编码器 GPU 输入到 pooled 输出，不含视频解码或 head |
| 正文 B1 与显存限制说明（表 3 不展示） | `evidence/three-video-encoders/efficiency-summary.json` | `encoders.{encoder}.budgets.{ratio}.batch_sizes.1` | B1 单独测量；不把 B1 显存当 B32 显存 |
| 表 3 CLIP B32 | `evidence/dsanet/benchmark/xd-summary.json`，绑定三份原始 benchmark JSON | `results.batch32-keep{None,0.8,0.6,0.4}` | 主表仅展示 XD 训练输入，与三视频编码器的输入数据集一致；UCF 原始回执仍保留，不将两组计时说成逐字节相同 |
| 表 3 FLOPs | 导出脚本 `flops()`；CLIP benchmark 的 `principal_block_gflops_per_image` | 标准 12 block 的解析主要 MACs×2 | 不是完整 encoder/head/pipeline 的 profiler FLOPs |
| 正文 DSANet 完整视频速度（表 3 不展示） | `evidence/dsanet/results_published/pipeline-exclusive-r02/{dataset}-{budget}-r{0,1,2}.json` | `total_wall_seconds`，逐轮 dense/reduced 后取中位数 | 完整视频到 CPU scores，16视频，三次进程；不是显示耗时中位数之比；UR-DMU 没有对应实测 |
| 完整视频约 0.97–1.01× | 同上 | UCF 三档约 0.968/0.991/0.978；XD 约 1.008/1.008/1.014 | 没有显著端到端加速主张 |
| 显存未随压缩下降 | B1 summary 与 pipeline JSON | allocated/reserved 或完整流程 peak_gpu_allocated_bytes | 未生成虚假的显存节省 |
| 将来训练压缩感知模型 | Discussion and Future Work | 明确使用 future 语态 | 不是本轮训练或实验成果 |

## 复现

```bash
cd manuscript-20260922
bash scripts/build.sh
python3 scripts/check_refs.py
```

导出将验证每个质量值的 Dense 一致性、clip 数量匹配及 CLIP 解析运算量的一致性。对原始数值不重新做模型推理；完整置信区间和原始统计来源留在数据文件中。

## 作者需确认

本次恢复聚焦文件可用性与四系统结果，不代表会议已接受其格式、方法或声明。保留了用户原来的跨栏位图；其中 `top-k pairs`、固定 N≈1.5K、显存收益等图中文字仍须与正文的 token 配额、CLIP 每图197个初始token和实测显存结果同步。不可将图中的泛化示意当成额外实验。

伦理审批/豁免、基金、利益冲突、所有作者单位与会议 AI 使用要求需作者按实际情况确认；不因本地成功编译而视为自动满足。原包的 README 和手交说明作为历史材料保留于快照，当前编译以本次主目录 README 为准。
