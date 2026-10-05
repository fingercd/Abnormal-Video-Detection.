# WAVAD 与 ICASSP 2027 资产整合计划

代码重构的实际模块边界、兼容入口和新运行默认值见[当前架构](../../../docs/architecture/current-system.md)，复用方法见[工作流 API](../../../src/vadbench/workflows/README.md)。本页与编号文档中的早期目标模板保留设计来源。

状态：**主要整理已完成，2026-09-26 更新**。源码、默认数据入口和6个旧物理资产根均已切换/归档，旧路径保留兼容软链接。四个data2根同盘放入assets/storage，两个users根同盘放入VAD/archive，以保留硬链接与日志文件描述符关系。具体完成项与验证边界见 [EXECUTION_STATUS.md](EXECUTION_STATUS.md)；下方保留9月25日的设计依据。源码项目ZIP及两份过时接管文档已清理。

## 结论与目标

以 `/users/fotile/VAD` 作为 WAVAD 的**统一代码和导航入口**，继续使用现有 `vadbench` Python 包与 CLI，暂不为改名重写导入路径。论文已经验证有通用价值的能力进入框架；论文特定的协议、冻结决策、图表和投稿材料留在 `projects/icassp2027/`。压缩方法保持一个相对独立的 `src/vadbench/token_reduction/` 包，通过小型契约接入 encoder、特征抽取和评测，使同一方法可换 encoder 或数据集，也不让方法读取论文标签与测试信息。

目标是把大型视频、权重、特征和实验物理归入 node2 的 `/data2/localdisk/fotile-wavad/`，node3 通过 `/data2` 挂载读取；`/users/fotile/VAD/assets` 指向该根。代码与小型清单留在 `/users/fotile/VAD`。迁移按数据集/完整 run 分批，源路径在所有引用与 QA 通过前保留。`/users` 所在文件系统只读查看时剩余约 717 GB，`/data2` 剩余约 4.7 TB；迁移前重新测量总字节与空闲量。具体每一类代码、特征和数据集的目录见 [02](02_目标架构与数据规范.md)。

执行次序是：资产普查与保护 → 确定权威源码 → 框架接口整合 → 特征与数据集目录规范化 → 全部实验和文档建索引 → 回归验收 → 切换默认入口。旧目录在验证完成前保持可用。详细分工与门槛见 [03_实施与验收.md](03_实施与验收.md)。

## 目前查到的事实和范围

| 对象 | 2026-09-25 只读观察 | 整理方向 |
|---|---|---|
| 旧框架根 | `/users/fotile/VAD`；远端 Git 工作树有未提交改动；代码中已有 `token_reduction/` 和论文脚本，但比本地及运行快照旧 | 保留原根；先逐文件确定源码身份，再整合 |
| 论文运行 | `/users/fotile/icassp2027-runs` 有 `project/`、大量 `code-*` 快照、`codex-takeover-20260920/`、`dsanet-extension-20260923-r01/` 等 | 每个实验保留原目录与身份，建立运行目录索引；不按文件名猜“最新即权威” |
| 大型特征 | `/data2/localdisk/fotile-icassp2027-kimi-20260919-a01` 含 dense/压缩特征视图；DSANet 训练特征在另一 `/data2/localdisk/fotile-icassp2027-dsanet-extension-20260923-r01` 根 | 建规范特征 catalog，复用完整视图；只在必要时复制，先校验原始 blobs |
| 原始数据 | `/users/fotile/datasets/` 有 UCF 多套经核验目录、XD 原视频及归档；`VAD/data/raw/ucf_crime` 已是链接 | 按 dataset/split/version 登记唯一推荐视图，其他版本保留并注明角色 |
| 文档 | 本地 `docs/icassp2027/`、`projects/icassp2027/`、manuscript 与服务器快照并存 | 只设一个当前导航；协议、冻结回执、历史计划、论文材料分层索引 |
| 仍在运行 | node2 有提取调度器，node2/node3 有旧 coordinator/monitor/watcher；仅凭进程名不能证明写入已结束 | 每个源根单独核查 writer、打开文件及完成回执后才能迁移 |

这是**初步根目录盘点，不是“全部实验已盘完”**。完整普查的范围、清单字段和漏项检查见 [01_资产清单与分类.md](01_资产清单与分类.md)。node2 与 node3 共享 `/users`，但执行前仍要核对各节点实际挂载和本地盘路径。

## 冻结的保护原则

1. 旧实验的原始 `resolved`、日志、checkpoint、特征、预测和失败回执保持原样；整理只增加索引和派生视图。不得用一个成功结果覆盖失败或 partial 目录。
2. 保留所有非论文 encoder、历史 V-JEPA 2 资产、旧缓存策略 `compression.py`、第三方 checkout 和已有环境。新 token 压缩包不得遮蔽 `compression.py`、`resources.py`。
3. UCF 与 XD 各自的原始时间轴、split、排除记录及评分协议独立；XD accepted 3950/declared 3954/excluded 4 必须在 catalog 中并列。训练 64+64 是筛选池，不是正式 full-train。现有 `VAD/data/raw/ucf_crime` 指向 `/users/fotile/datasets/UCF-Crime`，与名为 `UCF-Crime-official-verified` 的目录不同，先核验身份再切换。
4. 特征按 `encoder × dataset × split × representation × sampling × method × budget × source run` 登记。dense、压缩、CLIP/DSANet 十 crop 和不同 readout 不合并成一个“同名特征”。
5. 正式 test 的特征、预测和真值分开授权与存放；整理阶段只核对文件身份和覆盖，不据测试分数挑方法、预算或 checkpoint。
6. 不把大数据、模型权重、密钥或运行产物提交到 Git；本计划不授权 commit、push、删除旧目录或停掉其他进程。

## 本套文档

- [01_资产清单与分类.md](01_资产清单与分类.md)：现场根目录与全量盘点字段。
- [02_目标架构与数据规范.md](02_目标架构与数据规范.md)：逐类代码、旧特征和数据集的明确目标目录与迁移方式。
- [03_实施与验收.md](03_实施与验收.md)：分阶段执行、切换与回退的门槛。
- [04_实验与文档重编.md](04_实验与文档重编.md)：全部论文运行的归档分类、文档综合结论与删减记录。
- [document_inventory.jsonl](document_inventory.jsonl)：本地相关 Markdown 的逐文件标题、SHA 与处置分类，数量以清单实际行数为准。
- [EXECUTION_STATUS.md](EXECUTION_STATUS.md)：实施回执、当前目录与仍待完成的工作。

## 先要解决的一个命名问题

用户称框架为 WAVAD；当前仓库、Python 包与命令均使用 `VAD`/`vadbench`。本计划把 **WAVAD 作为项目对外名称**，不自动更改 Python 包名、已有命令或历史 fingerprint。若之后需要真正将包名改成 `wavad`，应作为单独兼容性迁移处理，不能夹在资产整理中。
