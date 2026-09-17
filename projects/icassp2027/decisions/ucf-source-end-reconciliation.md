# UCF-Crime source-end reconciliation：decoded-frame 决策

状态：**已在访问任何模型测试分数前采用 decoded-frame 口径；须完成 dataset-only 审计才可评测。**
该决定来自原始源码与真实容器边界，与模型效果无关。原 annotation 和严格审计的失败记录保留。

## 已核实的一手事实

作者官方 [MATLAB evaluator](https://github.com/WaqasSultani/AnomalyDetectionCVPR2018/blob/8aa957cdaa7eac821e07115fe63920c3b07da59d/Evaluate_Anomaly_Detector.m) 不是以实际解码 frame count 建 GT。它：

1. 写死 `fps=30`；
2. 定义 `Actual_frames=round(VideoReader.Duration*30)`；
3. 建立 `GT=zeros(1,Actual_frames)`；
4. 对每一段 annotation 执行 `st_fr=max(raw_start,1)` 与 `end_fr=min(raw_end,Actual_frames)`；
5. 用 MATLAB `GT(st_fr:end_fr)=1` 写标签，即 **1-based、两端包含**；
6. 把最后一个 C3D score 复制到 `Actual_frames`，作为历史 C3D evaluator 的 score 尾部扩展。

因此，作者明确提供了 annotation source-end 的截断规则；没有提供“将任意现代 raw-video 模型 score 尾部扩展”的一般规则。当前 sealed audit 已确认有 5 个 annotation end 超过实际 decoded `N`（3 个超过 1、2 个超过 2），且容器 `count_frames` 与 OpenCV 顺序解码一致。这是 decoded-frame timeline 与作者 `round(Duration*30)` timeline 的真实不一致，不能删除视频、改 source annotation 或写成 strict audit 已通过。

## 本项目采用的窄范围 reconciliation 契约

本项目使用一次性、全方法一致的 `source_end_reconciled_decoded_v1`：

```text
raw_start, raw_end         # immutable source endpoints; 1-based inclusive
N_author = round(duration_seconds * 30)  # historical diagnostic only
N_decode = verified decoded frame count
start_effective = max(raw_start, 1)
end_effective = min(raw_end, N_decode)
internal half-open span = [start_effective - 1, end_effective)
```

若 `end_effective < start_effective`，该 source span 在 decoded timeline 无有效支持，必须显式记录为 `empty_after_source_end_reconciliation`，不能静默扩张为一帧。原 raw endpoint、`N_author`、`N_decode`、start/end clamp 是否触发、最终 internal span 与 annotation SHA 都要随 run receipt 保存。

作者源码支持按评估 timeline 截断 annotation end。当前选择的 timeline 是经过实际解码核验的帧，而不是历史的固定 30 FPS 长度；实际视频含 25–30 FPS，因此不把 `N_author` 的固定 30 FPS 假定加入现代 raw-video 主指标。`N_author` 仅保留为历史差异诊断。这不改写原 source：只在带独立指纹的派生 manifest 中建立有效区间。所有被比较方法使用同一个 `N_decode`、同一个有效 mask 与同一 reconciliation，不允许只给某个方法尾部 padding 或只对某类视频截断。

## 不应混淆的替代口径

- **作者历史 C3D reproduction**：应按其 `N_author=round(duration*30)` timeline 和最后 score 扩展复现；不能与当前 decoded-frame metric 混称同一口径。
- **decoded-frame micro metric**：使用上面的 narrow reconciliation；它是项目的明确评测契约，不宣称逐位等同于作者 MATLAB score timeline。
- **严格无 reconciliation audit**：继续报告 5 个越界 source endpoint，作为 source/raw 审计事实；不能因选择 v1 而删除该异常记录。

## 放行前的 dataset-only 审计

先执行 dataset-only 审计：固定 annotation SHA、视频容器 digest、`N_author`/`N_decode` 和有效 mask，统计每类 clamp 数、空 span 数及 source-end delta 分布。只有该 receipt 完整且方法已冻结，才允许所有方法共用此有效 mask 产生测试指标。模型效果、selector、层、预算、阈值、checkpoint 与 head seed 不参与该协议决定。

作者 evaluator 的源码 commit、文件 hash 与精确行级证据位于已忽略的 `outputs/icassp2027/source_checks/ucf_author_evaluator_end_contract_20260918.json`。本草案没有读取当前 annotation 值、GT arrays 或模型分数。
