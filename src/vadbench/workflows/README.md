# 可复用实验流程

这里保存不需要导入 `vadbench.paper` 或读取 ICASSP 项目配置的执行实现。
项目负责选择数据、模型、参数和冻结协议，工作流接收已经明确的输入与身份。

| 模块 | 输入与职责 |
|---|---|
| `extraction.py` | adapter、manifest、采样/表示身份 → pooled FeatureStore、分片与回执 |
| `feature_resume.py` | 核验完整视频分片后恢复；复制或显式硬链接运输 |
| `feature_merge.py` | 合并或选择特征视图，校验成员、内容、身份和分片覆盖 |
| `detection.py` | 已声明特征配对 → 通用 MIL 训练、预测与评价 |
| `quality_comparison.py` | 配对质量差值、bootstrap 与声明的容忍度 |
| `urdmu_quality_export.py` | 既有预测与真值的对齐、质量导出 |
| `efficiency.py`、`video_efficiency.py` | 固定边界的耗时、显存及视频执行统计 |
| `dsanet/` | CLIP–DSANet 的原生 NPY 提取、评分、质量导出 |

公共特征身份在 `vadbench.data.feature_contracts`；显式预测兼容许可在
`vadbench.engine.compatibility`。UR-DMU 作者后端和精确分块推理在
`vadbench.integrations.detectors.urdmu`。这些实现不依赖论文流程模块。

## 调用方式

现有通用 CLI `vadbench extract/train/predict/evaluate` 继续使用基础 engine。
需要 pooled 特征身份与断点恢复时，使用这里的 Python API；论文正式任务的入口
仍由 `scripts/icassp2027/` 和 `vadbench.paper` 负责审核冻结协议后调用这些实现。
DSANet 的命令见 [DSANet README](dsanet/README.md)。

```python
from vadbench.data.feature_contracts import RepresentationIdentity, SamplingIdentity
from vadbench.workflows.extraction import PooledExtractionSpec, extract_pooled_features
from vadbench.workflows.detection import DetectionConfig, train_detector, predict_detector
from vadbench.workflows.feature_merge import FeatureRunMergeRequest, run_feature_merge
```

## 历史兼容与新运行

原 `vadbench.paper.extraction` 等已迁移模块只保留导入兼容入口，旧、新路径返回
同一个模块对象，旧脚本的类型检查和模块 patch 仍作用于同一实现。新代码使用上表路径。
旧产物中的 `paper_identity`、`paper_detector`、`icassp2027.*` schema 名称及身份
散列规则继续保留；代码迁移不重命名或改写已有特征、checkpoint、合同和结果。
三对已审核的行尾等价摘要仍是固定白名单，没有扩大为忽略代码身份。

新运行必须使用实际代码/模型/数据身份生成自己的回执，不能因兼容导入而复用旧运行的
代码指纹。历史精确重放继续使用归档中的原执行快照。

`projects/icassp2027/profile.yaml` 的默认新运行根为
`assets/experiments/icassp2027/runs`。API 或脚本显式指定的 `output_root` 继续有效；
历史目录和资产导航不会因修改默认值自动移动。
