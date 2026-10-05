# WAVAD：视频异常检测实验框架与 PairSelect

WAVAD 是一个可复用的视频异常检测研究框架，Python 包与命令行名称为 `vadbench`。它连接视频清单、编码器、特征缓存、弱监督检测头、帧级评价与效率测量，保留数据、权重、代码和配置的身份信息，使结果能够追溯到实际运行。

本仓库也提供论文 **PairSelect: Training-Free Token Selection for Efficient Weakly Supervised Video Anomaly Detection** 的实现、实验协议、作者工作稿和结果证据。下面先介绍框架，再说明如何阅读论文、定位代码和核对实验效果。

## 1. 框架组织

主要流程是：**原始视频 → manifest 与数据审计 → encoder → FeatureStore → 检测头 → 帧级预测与评价**。Token 缩减介入 encoder 内部；质量比较与效率测量使用各自明确的输入和执行边界。

| 目录 | 职责 | 阅读入口 |
|---|---|---|
| `src/vadbench/data/` | 视频清单、采样、审计、特征存储与表示身份 | [特征身份](src/vadbench/data/feature_contracts.py) |
| `src/vadbench/integrations/` | 编码器加载、预处理与统一输出 | [当前架构](docs/architecture/current-system.md) |
| `src/vadbench/engine/` | 基础提取、训练、预测、覆盖检查与兼容许可 | [预测许可](src/vadbench/engine/compatibility.py) |
| `src/vadbench/workflows/` | 可复用的提取、恢复、合并、检测、质量与效率流程 | [工作流 API](src/vadbench/workflows/README.md) |
| `src/vadbench/integrations/detectors/` | 上游检测器适配 | [UR-DMU 后端](src/vadbench/integrations/detectors/urdmu/backend.py) |
| `src/vadbench/token_reduction/` | Token 选择、合并、部署与模型内部结构适配 | [PairSelect 规则](src/vadbench/token_reduction/token_selection.py) |
| `src/vadbench/research/` | 中间层观察与性质分析 | [观察计划](docs/icassp2027/01_NORMAL_ANOMALY_PROBES.md) |
| `src/vadbench/paper/` | 项目配置、冻结检查、数据权限与正式任务编排 | [项目入口](projects/icassp2027/README.md) |

重构后的公共工作流不依赖 ICASSP 项目配置；论文层确定模型、数据、参数和冻结协议，再调用公共实现。原 `vadbench.paper` 中已迁移模块保留兼容导入，历史合同与指纹不因迁移而重写。通用 CLI 继续使用 `vadbench extract/train/predict/evaluate`；断点恢复、特征视图合并等 Python API 见工作流文档。

## 2. 论文研究什么

PairSelect 在保持视频输入覆盖与检测头不变的条件下，减少视觉 encoder 后半段的 token 计算。它在 12 层 transformer 的第 6 层之后插入选择操作：局部分组获得固定配额，根据当前激活的 L2 范数决定成员优先级，保留原向量并恢复索引顺序，不在选择点平均或缩放向量。范数是选择信号，不是异常概率。

VideoMAE、VideoMAEv2 和 CLIP 使用相邻空间 patch 配对；TimeSformer 保留或删除完整空间轨迹，以适配时空分离结构。CLIP 保留 CLS。视频骨干的配额槽位使用近似空间锚点，不能解释为所有动态选择都保留精确原始空间位置。

论文在 UCF-Crime 与 XD-Violence 上评估四套系统：

| 视觉 encoder | 检测器来源 | 后半段 token 数：dense / 删除 20% / 40% / 60% |
|---|---|---|
| VideoMAEv2-B | 对应 encoder、对应数据集的完整 dense 训练特征上训练 UR-DMU | 1568 / 1254 / 941 / 627 |
| VideoMAE-B | 对应 encoder、对应数据集单独训练 UR-DMU | 1568 / 1254 / 941 / 627 |
| TimeSformer-B | 对应 encoder、对应数据集单独训练 UR-DMU | 1569 / 1233 / 897 / 561 |
| CLIP ViT-B/16 | 作者发布的对应数据集 DSANet 权重 | 197 / 158 / 119 / 79 |

视频骨干的数量按 clip 计，CLIP 按图像计；有 CLS 时已计入。`keep_ratio=0.8/0.6/0.4` 是保留率，对应名义删除 20%/40%/60%。TimeSformer 受宽度对齐约束，实际删除率为 21.41%/42.83%/64.24%。

**Training-free** 指插入和改变压缩算子无需额外训练。已有检测器仍有其训练来源；各预算复用对应系统的 dense 检测头，不重新拟合压缩专用 head。

## 3. 如何阅读论文与实验效果

建议按以下顺序阅读：

1. **先读论文整体。** 打开 [论文 PDF](projects/icassp2027/manuscript-20260922/main.pdf)，先看摘要、方法与实验。[文稿 README](projects/icassp2027/manuscript-20260922/README.md) 说明工作稿来源与构建方式。
2. **看表 1，理解实际预算。** 用实际 token 数判断后六层减少了多少计算；前六层仍是 dense，输入帧覆盖不变。
3. **看表 2，判断检测质量。** 在同一系统、同一数据集内比较 dense 与各预算，再看配对差值和置信区间。UCF 主指标是 frame ROC-AUC；XD 的三条视频系统以梯形 PR-AUC 为主，DSANet 以 step AP 为主。两种 PR 积分定义分别阅读。
4. **看表 3，判断编码器效率。** 这是 batch size 32、XD 训练输入上的编码器测量，包含选择与 gathering，不包含视频解码和检测头。解析 FLOPs 只覆盖主要 transformer blocks。
5. **核对原始证据。** [论断—证据表](projects/icassp2027/manuscript-20260922/notes/CLAIM_EVIDENCE_MAP.md) 将主张映射到结果文件、字段和测量范围。

当前工作稿报告：40% 名义删除时，四系统 batch-32 编码器加速约 **1.22–1.28×**，多数质量点估计接近 dense；60% 时约 **1.33–1.47×**。质量回落也保留：40% 时 DSANet/UCF AUC 下降约 0.90 个百分点；60% 时 DSANet/UCF、DSANet/XD 与 VideoMAE/UCF 主指标分别下降约 1.80、1.22、1.05 个百分点。完整数字见[质量表](projects/icassp2027/manuscript-20260922/tables/quality_rows.tex)与[效率表](projects/icassp2027/manuscript-20260922/tables/efficiency_rows.tex)。

“接近 dense”描述点估计，不表示全部配置通过统计等价检验。DSANet 完整视频流程的证据约为 0.97–1.01×，不能据编码器表格宣称同等端到端加速；当前证据也不支持普遍显存下降。

## 4. 从论文找到代码

| 想理解的内容 | 阅读顺序 |
|---|---|
| 分组、配额、成员范数优先级与对照 | [token_selection.py](src/vadbench/token_reduction/token_selection.py) → [规则测试](tests/icassp2027/test_token_selection.py) |
| 插入层、真实变长执行与部署 | [deployment.py](src/vadbench/token_reduction/deployment.py) → [indexed bridge](src/vadbench/token_reduction/bridges/indexed.py) → [部署测试](tests/icassp2027/test_member_rule_deployment.py) |
| CLIP patch 选择、CLS 与位置处理 | [CLIP bridge](src/vadbench/token_reduction/bridges/clip.py) → [bridge 测试](tests/icassp2027/test_clip_bridge_s1.py) |
| 特征提取、恢复与完整视图合并 | [工作流说明](src/vadbench/workflows/README.md) → [extraction.py](src/vadbench/workflows/extraction.py) → [feature_merge.py](src/vadbench/workflows/feature_merge.py) |
| UR-DMU 训练、固定 head 推理与正式门槛 | [训练入口](scripts/icassp2027/train_urdmu_backend.py) → [评分入口](scripts/icassp2027/score_urdmu.py) → [后端适配](src/vadbench/integrations/detectors/urdmu/backend.py) |
| CLIP–DSANet 提取、评分、质量导出 | [DSANet 使用说明](src/vadbench/workflows/dsanet/README.md) |
| 质量差值、bootstrap 与帧级导出 | [quality_comparison.py](src/vadbench/workflows/quality_comparison.py) → [urdmu_quality_export.py](src/vadbench/workflows/urdmu_quality_export.py) |
| 编码器与完整视频计时边界 | [编码器 benchmark](scripts/icassp2027/benchmark_frozen_reducers.py) → [完整视频 benchmark](scripts/icassp2027/benchmark_frozen_video_detector.py) |

当前重构代码用于理解和扩展实现；核验历史运行时，还应查看文稿证据中的执行代码快照。当前 Git 版本和历史实验代码身份不能自动视为相同。

## 5. 实验协议与结果在哪里

| 材料 | 入口与用途 |
|---|---|
| 项目配置 | [profile.yaml](projects/icassp2027/profile.yaml)：三条视频 encoder 的正式编排；CLIP–DSANet 使用独立工作流 |
| 范围、冻结协议与历史 | [项目 README](projects/icassp2027/README.md)、[decisions](projects/icassp2027/decisions/)、[progress.md](projects/icassp2027/progress.md) |
| 四系统汇总 | [four_systems_results.json](projects/icassp2027/manuscript-20260922/data/four_systems_results.json) |
| 原始证据 | [evidence_snapshot](projects/icassp2027/manuscript-20260922/evidence_snapshot/)：质量、区间、benchmark 与历史代码 |
| 数值来源与校验 | [numeric_provenance.csv](projects/icassp2027/manuscript-20260922/data/numeric_provenance.csv)、[evidence_sha256.csv](projects/icassp2027/manuscript-20260922/data/evidence_sha256.csv) |
| 图表生成 | [export_tables.py](projects/icassp2027/manuscript-20260922/scripts/export_tables.py)：从已有证据生成论文表格，不运行模型 |

早期计划与进度是日期化记录；论文四系统结论以对应文稿证据为准。UCF 使用 1610 个训练视频和 290 个测试视频；XD 使用通过完整性检查的 3950 个训练视频和全部 800 个测试视频。UR-DMU 每步的 64 normal + 64 anomalous bags 是完整训练集中的采样预算。

正式测试标注与分数不得用于选择方法、预算、层或 checkpoint。新增训练、评测与小实验必须按项目规则在线记录到 SwanLab，保存配置、seed、代码版本、数据/权重身份、命令和日志，并核验上传。

## 6. 使用与复现

Python 包要求 **Python 3.10–3.12**。模型依赖、权重、数据和第三方后端按 encoder 准备；已有工作环境应直接复用。环境与常用操作见 [README-CN](README-CN.md) 和[操作指南](docs/operations/workflows.md)。

在已准备的环境中，以下命令查看入口，不启动实验：

```bash
python -m vadbench --help
python -m vadbench encoders list
python -m vadbench.paper status --project projects/icassp2027/profile.yaml
python -m vadbench.workflows.dsanet --help
```

源码尚未安装时先将仓库 `src` 加入 `PYTHONPATH`。`status` 只报告配置与路径可用性，模型加载和实验成功需有真实回执。DSANet 历史 Python 3.9 环境使用[专用启动器](scripts/icassp2027/dsanet_legacy.py)，参数见其工作流说明。

仅重建已有结果表格与论文 PDF：

```bash
cd projects/icassp2027/manuscript-20260922
bash scripts/build.sh
```

构建需要 Python 3、TeX Live/MiKTeX 和 BibTeX；也可在 Overleaf 打开 `main.tex`。重新运行模型还需要原始视频、固定权重、上游代码、清单和运行合同。结果快照能用于核对论文数字，不能替代完整实验资产。服务器资产布局与核验范围见[资产地图](docs/operations/wavad-assets.md)。

## 7. 许可证与论文状态

仓库代码采用 [MIT License](LICENSE)。视频数据、第三方实现和权重遵循各自许可；大型视频、权重、特征和完整运行输出不进入 Git。

当前提供 PairSelect 作者工作稿与证据导航，不声明论文已被会议录用。作者信息、图文一致性、伦理与会议声明的待核对项见[论断—证据表](projects/icassp2027/manuscript-20260922/notes/CLAIM_EVIDENCE_MAP.md)。
