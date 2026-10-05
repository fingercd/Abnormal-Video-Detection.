# PairSelect 四系统论文改写证据包

生成日期：2026-09-24（Asia/Hong_Kong）。用途：供作者与 GPT-6 Pro 核对并改写 ICASSP 2027 工作稿；不是可直接投稿的最终文件。

## 从哪里开始

1. 读根目录 `PROMPT_FOR_GPT6PRO.md`。
2. 编辑 `paper/manuscript-20260922/`，以其 `main.tex` 为入口。
3. 用 `evidence/three-video-encoders/RESULTS.md` 和 `evidence/dsanet/RESULTS_PUBLISHED_20260923.md` 核对四条系统线的质量、速度和边界。原始 JSON/CSV 同目录或 `server_evidence/` 下。
4. `MANIFEST_SHA256.csv` 列出 ZIP 内所有其他文件的字节数与 SHA-256。`SERVER_SOURCE_ARCHIVE_SHA256.txt` 绑定从 node3 下载的原服务器归档。

## 来源与范围

- `server_evidence/code-bsw-r01/`：node3 上三条视频骨干预算/效率运行引用的 VADBench 源码快照。它是**运行快照**，不是当前本地未提交工作树，也不包括模型权重或视频。
- `server_evidence/dsanet-extension-20260923-r01/code/`：本轮 DSANet/CLIP 特征、评分、质量和计时的服务器执行脚本。`analysis_runtime/` 是最终 Python 3.10 质量导出实际使用的隔离指标模块。`score-published-r01/` 含八格原始预测；`analysis-published-r02/` 含 10,000 次视频级 bootstrap 的区间输入与最终结果；`pipeline-exclusive-r02/` 是到 CPU 分数的正式完整视频计时。
- `server_evidence/official_dsanet_source/`：服务器保存的 DSANet 作者源码、README、LICENSE 与数据清单，不含作者发布权重。正式论文来源仍需引用 [AAAI 2026 论文](https://ojs.aaai.org/index.php/AAAI/article/view/38191)及其官方仓库。
- `paper/`：本机截至打包时的最新 LaTeX 作者工作稿。它尚未加入 DSANet 结果；不要把当前 PDF 误当四系统完成版。
- `evidence/three-video-encoders/`：三视频骨干的预算复用、质量与效率回执；排除了其中重复打包的内部 ZIP，但原始 JSON、CSV 与报告均保留。
- `evidence/dsanet/`：本地复核后的 DSANet 发布权重结果与编码器/检测头/完整流程摘要。`evidence/batch-scaling/` 是三视频骨干 batch16/32 补测。
- `project_docs/`：当前计划、进度和实验规则。

## 必须保留的事实边界

前三条视觉骨干是**预训练 encoder + 本项目分别训练的 UR-DMU head**；第四条是 CLIP ViT-B/16 + **作者发布 DSANet head**。PairSelect 本身不训练。可以把四条写成统一研究设计，不应称我们从零训练了四个 encoder 或声称 DSANet 检测头是本项目训练的。

40% 删除在多数配置点估计接近 dense；60% 删除可按作者选定措辞称“总体轻微掉点”，同时展示最大回落：DSANet×UCF 约 −1.80 pp、DSANet×XD 约 −1.22 pp、VideoMAE×UCF 约 −1.05 pp。统计非劣性只在满足预设置信区间门槛的格子成立，不能用“轻微”替代此判定。

XD 旧三视频骨干主表为**梯形 PR-AUC**，DSANet 新主结果为**step AP**。两者不可混作同一定义直接排名；DSANet 本轮新 raw 特征与旧作者预提特征也不能混成一个压缩对照。纯编码器 batch32 加速不能被改写成完整视频加速；DSANet 完整视频倍率约 0.97–1.01×，峰值显存未随压缩下降。

**UCF frame AP 也已计算。**如果论文某表的辅指标列仅标作“XD AP”，UCF 横杠表示该列范围限定在 XD，不表示 UCF 没跑。需要在同表展示时，把列名改为跨数据集的 `Frame AP (step, secondary)` 并从原始结果填入 UCF 行；40% 删除的三视频骨干 UCF 值为 19.29→19.60、18.82→18.99、20.37→20.90。`evidence/three-video-encoders/references/quality_matrix_detail.csv` 的历史字段名带 `xd_`，但 UCF 行有有效的帧级 AP 值。切勿用 AUC 数值或其他数据集的 AP 代填。

为控制体积，包内不含原视频、原始大特征数组、模型权重、共享环境或凭据。上述资产的 SHA、来源清单和必要路径保留在回执中。完整模型 FLOPs、能耗和未冻结阈值的 F1 无可靠实测值，不得补造。
