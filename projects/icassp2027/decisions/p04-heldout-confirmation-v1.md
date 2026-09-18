# P04 保留确认：VideoMAEv2 embedding 局部冗余

日期：2026-09-18。本文在读取任何新样本的 P04 统计前冻结。它只定义一个模型内、弱视频标签的观察确认；不冻结 token 操作、插入层、预算或论文方法。

## 已有依据与范围

已完成的 VideoMAEv2 `fit/explore` 数据为 128 个视频（正常/正视频各 64，每视频 8 窗口）。在
`outputs/icassp2027/analysis/explore-gpu-v1-20260917T204307Z/candidate_review.md` 中，embedding 输出的 P04
`redundancy_fraction` 为正：Hedges g=0.529901，95% CI [0.215806, 0.867088]；以 fit-normal 标定的
motion×brightness 分箱后，正减正常为 +0.0567345，95% CI [0.0251304, 0.0907075]。

这只是探索结果。Explosion 与 Robbery 类别层的方向相反，原审查明确将它标为 `not promoted`，因此
不能写成广泛的早期合并规律。此前四 encoder 的确认只读取了 P16/P07：
`outputs/icassp2027/analysis/confirm-four-v3-20260917T234419Z/analysis.json` 及其四个 source analysis
没有 P04 行；本计划不会重新读取或复用已执行的 confirm64 P04 值。

**唯一主统计签名**：

```text
encoder_id=videomaev2
site=embedding.output
probe_id=P04
statistic_name=redundancy_fraction
token sample=collector 的固定时空样本（max_tokens=256）
redundancy_similarity=0.90
clip aggregation=每视频 8 个中心窗口等权均值
contrast=weak-positive minus normal
expected direction=positive
```

P04 的同层 `same_frame_local_cosine_mean` 只作为机制诊断与完整性检查，**不是第二个候选、共主指标或
通过门槛**；不得在它与 `redundancy_fraction` 之间按新数据择优。采集时仍保留该 P04 行，方便判断主统计
是否由局部邻接而非全局 token 对驱动。

## 新的未使用确认样本

来源锁是
`outputs/icassp2027/assets/frozen192-verified-20260917T200000Z/frozen-partition-lock.json`，其 SHA-256 为
`53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59`。旧 confirm64 cohort 是
`outputs/icassp2027/assets/confirm64-plan-20260917T205233Z/cohort.jsonl`，其 cohort SHA-256 为
`bf6d30676b63488648c3fbd5f71b941a58d2d80b660fc722e0b1f0db66b18c14`，其 ID-set SHA-256 为
`5367dcc5e33d33c45bba5dd1a6bbda8a78daac28482e441d00e777d242cefdc5`。

原 `confirm161` 有 80 正常、81 正视频；删除上述已执行 64（32/32）后，未触及池仍有 48 正常、49 正视频。
从每个标签池按 UTF-8 `SHA256("p04-heldout-confirmation-v1:20260918:" + video_id)` 升序取前 32 个；同一
label 内 hash 相同再按 `video_id` 升序。选中清单按 `(weak_label, video_id)` 排序后的规范文本
`weak_label<TAB>video_id<TAB>selection_sha256`（行间 LF、末尾无 LF）SHA-256 是
`3596e2ad7ce60fdd03fe6ac308482d2fe564fce7e3f64310d7da3900d6d3ad60`。

| 标签 | 选择的 video_id（共 32） |
|---|---|
| 正常（0） | Normal_Videos011_x264, Normal_Videos039_x264, Normal_Videos052_x264, Normal_Videos089_x264, Normal_Videos092_x264, Normal_Videos093_x264, Normal_Videos094_x264, Normal_Videos102_x264, Normal_Videos108_x264, Normal_Videos221_x264, Normal_Videos227_x264, Normal_Videos254_x264, Normal_Videos328_x264, Normal_Videos347_x264, Normal_Videos389_x264, Normal_Videos395_x264, Normal_Videos425_x264, Normal_Videos482_x264, Normal_Videos506_x264, Normal_Videos528_x264, Normal_Videos551_x264, Normal_Videos565_x264, Normal_Videos578_x264, Normal_Videos607_x264, Normal_Videos628_x264, Normal_Videos635_x264, Normal_Videos639_x264, Normal_Videos664_x264, Normal_Videos699_x264, Normal_Videos713_x264, Normal_Videos757_x264, Normal_Videos792_x264 |
| 含异常（1） | Abuse001_x264, Abuse016_x264, Abuse020_x264, Abuse022_x264, Abuse035_x264, Arrest003_x264, Arrest029_x264, Assault015_x264, Assault018_x264, Assault052_x264, Burglary058_x264, Burglary078_x264, Burglary084_x264, Burglary091_x264, Fighting002_x264, Fighting011_x264, Fighting013_x264, Fighting026_x264, RoadAccidents007_x264, RoadAccidents088_x264, RoadAccidents142_x264, Robbery002_x264, Robbery004_x264, Robbery011_x264, Robbery082_x264, Robbery107_x264, Shooting030_x264, Stealing018_x264, Stealing022_x264, Stealing024_x264, Stealing100_x264, Vandalism025_x264 |

完整 ID、选择规则、规范选择摘要和派生条件以同名 JSON 为机器可读权威；每项选择 hash 可由其中的
UTF-8 前缀规则复算。新 cohort/manifest 生成时必须逐项核对：
角色均为 `confirm`、官方 split 均为 `train`、弱标签与 manifest 一致、8 个 `segment-00` 至 `segment-07`
均存在；禁止替补缺失文件。它与 explore128、旧 confirm64、neutral26 无 ID 重叠；不动 fullfit1288 或
select161 的检测器训练/选择身份。

## 采集、控制与统计

使用同一 native 预处理、每视频 8 个 `uniform_segment_centers_full_clip` 窗口、`frame_stride=2`，只把
视频级弱标签在 collector 后 join。禁止读取事件时间、patch 标签、文件名类别或官方 test 标注；异常类别只作
确认后描述，不能改变统计签名或通过判定。

motion 与 brightness 的阈值沿用只由 fit-normal 标定的
`outputs/icassp2027/control/control-calibration-cross-encoder-v2-20260917T213000Z/control_calibration_cross_encoder_v2.json`。
确认运行产生的 `input_controls.jsonl` 必须同 native frame/sampling identity 匹配该 calibration；若不匹配或任一
视频控制缺失，则 matched 结果为 unavailable，不重新拟合阈值。`scene_group` 仍为 unknown，本确认不构成
scene/source matched、事件级或因果证据。

统计单位为视频。每个视频先等权平均 8 clips，正常/正视频各自重采样 10,000 次，seed `20260918`；主比较报告
正减正常均值差、Hedges g 与 percentile 双侧 95% CI。matched 比较在固定 motion×brightness 完整 bin 内分别
按标签重采样，以 `min(n_normal,n_positive)` 固定权重汇总；记录保留数、bin 组成、掉落数和 CI。

状态规则如下：

1. `confirmed_model_specific` 仅当原始主比较和 matched 比较都为正、两者 95% CI 都不含 0，且每标签至少保留
   16 个视频；否则为 `rejected_or_inconclusive`。
2. 类别方向相反、区间宽、scene unknown 或局部 cosine 与主统计不一致，都原样写入 finding card；它们不会触发
   替换 statistic、阈值、层、样本或方向。
3. 通过也只支持 VideoMAEv2 在上述采样与输入下的模型内关联。TimeSformer、VideoMAE、V-JEPA2 是之后的结构
   复核对象，不在本次门槛中预先承诺同向或方法通性。
4. 失败即停止 P04 干预，不以全局/局部排序、pair mean、LoRA 或检测分数补救；通过也仍需一个独立、同预算的
   中立操作验证才可能讨论 token 操作。

## 可执行入口和预期输出

同名 suite 已按当前 `vadbench.paper resolve_probe` schema 写入。主调度者生成并 SHA 绑定其中的 cohort 与
64-video manifest 后，可用：

```powershell
$VAD_PY -m vadbench.paper probe `
  --project projects/icassp2027/profile.yaml `
  --suite projects/icassp2027/suites/p04-heldout-confirmation-v1.yaml `
  --device cuda:0
```

该入口复用 `src/vadbench/paper/stages.py` 的只读 observer/identity、
`src/vadbench/research/collectors.py` 的 P04，以及 `scripts/icassp2027/analyze_probes.py` 的视频级
bootstrap 和控制 join。预期 run 输出为 `probe_summary.jsonl`、`input_controls.jsonl`、architecture receipts、
progress/summary；后续分析输出 `video_summary.csv`、`contrast_summary.csv`、`stratum_composition.csv`、
`analysis_receipt.json`、控制 receipt 和 finding-card draft。新 run 目录必须唯一，不能覆盖旧 confirm export。

当前缺口仅是把本冻结 ID 清单物化为 cohort/manifest 并生成对应 candidate-definition JSON；上述缺口不包含
新 reducer 或新压缩策略。本计划不执行 GPU、不读新结果，也不授权检测器评分。
