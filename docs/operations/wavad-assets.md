# WAVAD 服务器资产与路径（2026-09-26）

本页记录已执行的 WAVAD / ICASSP 2027 资产整理状态。代码入口是 `/users/fotile/VAD`，大型资产根是 node2 的 `/data2/localdisk/fotile-wavad/`；node3 经 `/data2` 共享挂载可见。同一代码仓库里的 `assets` 链接指向大型资产根。两节点的 `/users` 也已核实共享：node2 的该路径由 `ibnode3:/users` NFS 提供。Python 包和命令仍叫 `vadbench`，没有进行包名迁移。

## 路径与角色

| 路径 | 角色与当前状态 |
|---|---|
| `/users/fotile/VAD` | 已整合的框架与论文源码；项目 profile、协议和小型文档在 Git 工作树内 |
| `/users/fotile/VAD/assets` → `/data2/localdisk/fotile-wavad/` | 大型资产导航入口；在 node2 创建，node3 通过共享挂载可见 |
| `assets/datasets/ucf_crime/official_raw_20260918/raw` | 1,900 个 UCF-Crime 原视频，103,795,576,492 B |
| `assets/datasets/xd_violence/official_raw_accepted3950_20260920/raw` | 4,750 个 XD-Violence 原视频及 1 个辅助文件，85,869,780,149 B |
| `/users/fotile/VAD/data/raw/ucf_crime` | 已切换至上列 UCF 新副本的默认入口；`ucf_crime_legacy_897` 保留旧 897 子集别名 |
| `/users/fotile/VAD/data/raw/xd_violence` | 已切换至上列 XD 新副本的默认入口 |
| `assets/storage/` | 四个原 `/data2/localdisk/fotile-icassp2027-*` 根的真实物理落点；另有两个指向共享 `VAD/archive/` 的统一入口 |
| `/users/fotile/VAD/archive/` | `icassp2027-runs` 与 `datasets` 两个旧 `/users/fotile` 根的真实归档位置；目录被 Git 忽略 |
| `assets/features/`、`assets/experiments/` 及相关导航 | 规范导航链接，解析到 `storage` 或 `VAD/archive` 的实体；旧来源路径为兼容软链 |
| `assets/provenance/pre-integration-remote-*` | 远端整合前的原始代码与 dirty patch 备份，保留可追溯来源 |

UCF 与 XD 的复制经过 `rsync --checksum` 对源、目标复核，无差异。原始来源副本现保留在 `archive/datasets` 作为备份，没有对重复 raw 做危险去重。XD 的版本名 `accepted3950` 指训练集合已接受 3,950 条；原始声明 3,954、排除 4 条的身份和 test 800 仍应从协议、manifest 与运行合同分别读取。不要从目录名推断所有成员角色。

## 六个旧根的物理收拢

2026-09-26 六个根均完成同文件系统 `renameat2` 原子交换。下面的“旧路径”仍在，现为指向新实体的兼容软链；不要删除旧路径或把软链误当第二份实体。

| 旧路径 | 现有实体或统一导航 |
|---|---|
| `/data2/localdisk/fotile-icassp2027-kimi-20260919-a01` | `/data2/localdisk/fotile-wavad/storage/fotile-icassp2027-kimi-20260919-a01` |
| `/data2/localdisk/fotile-icassp2027-dsanet-extension-20260923-r01` | `/data2/localdisk/fotile-wavad/storage/fotile-icassp2027-dsanet-extension-20260923-r01` |
| `/data2/localdisk/fotile-icassp2027-20260919-a01` | `/data2/localdisk/fotile-wavad/storage/fotile-icassp2027-20260919-a01` |
| `/data2/localdisk/fotile-icassp2027-vjepa-sdpa-fit-20260919-a01` | `/data2/localdisk/fotile-wavad/storage/fotile-icassp2027-vjepa-sdpa-fit-20260919-a01` |
| `/users/fotile/icassp2027-runs` | `/users/fotile/VAD/archive/icassp2027-runs`；统一入口 `assets/storage/users-icassp2027-runs` |
| `/users/fotile/datasets` | `/users/fotile/VAD/archive/datasets`；统一入口 `assets/storage/users-datasets` |

两个 `/users` 根留在共享文件系统的同盘归档，保留约 277 万历史文件的 inode、hardlink 关系及正在打开的日志/锁文件描述符；没有跨盘复制这些根。六根统计的 **30,784,751 个常规文件条目**包含 hardlink 重复名字，不表示独立实验数或视频数。

迁移后，node2 与 node3 各自的 96 个锚点 `stat`（dev/inode/size/mtime/nlink，部分小文件另验 SHA）以及原有 314 个导航链接检查均通过，六个旧根的 inode 身份保持。`assets/catalog/physical-consolidation-20260926.json` 的状态为 `completed`；`assets/catalog/physical-paths-20260926.jsonl` 记录 3,455 条路径、`missing=0`；`assets/catalog/current.json` 已链接当前目录。迁移时只对核实过的旧空队列/hold/monitor 守护及安全子进程短暂 `SIGSTOP`，node2 为 16.56 秒、node3 为 19.34 秒，随后均 `SIGCONT`；没有停计算作业、动其他项目进程或改科学结果。

## 导航与核验范围

大型资产根下已有 **139 个特征导航链接**，以及 **175 个实验、数据源和代码快照导航链接**。六根物理收拢后，这 314 个规范链接可解析到 `storage` 或 `VAD/archive` 的实体；它们仍只是入口数，不能当作独立 run 的数量。例如 `assets/features/video_vit/ucf_crime/test/videomaev2/pair_select/keep_060/run` 与 `assets/features/clip_dsanet/xd_violence/test/clip_vit_b16/pair_select/keep_040/run` 均是当前实际导航；预算目录使用 `keep_060`、`keep_040` 这样的命名，不采用旧设计表中的 `keep_0.60`。

`inventory-20260925-r02` catalog 登记 **6 个数据源候选、81 个特征入口、101 个运行系列目录、42 个代码快照**，保留首次盘点身份；“运行系列”不等于每次run，候选数据源也不等于默认版本。[资产整理计划](../../projects/icassp2027/organization/README.md)和[目标架构](../../projects/icassp2027/organization/02_目标架构与数据规范.md)说明分类原则，实际位置以本页和物理迁移回执为准。

进一步的 `assets/catalog/run-history-20260926-r02/` 已记录3419个含回执的目录、174个显式run ID，其中48个目录绑定冻结六格矩阵。扫描也包括旧框架输出、源码快照中的outputs、开发与失败尝试；这些数目不能当成独立成功实验数。`assets/catalog/current.json` 是当前清单的机器可读导航，`models-20260926.jsonl` 保存模型入口与checkpoint核验状态。

[正式特征绑定](../../projects/icassp2027/organization/formal_feature_bindings.jsonl)逐项绑定 **6 个 dense test + 18 个压缩正式 test** 视图，其记录中的旧绝对路径通过兼容软链继续可读。服务器回执 `assets/catalog/formal_feature_qa-20260926-r02.jsonl` 显示 24/24 组的合同 SHA、**完整 index SHA** 和各 index 首个引用 blob 的 SHA/zip CRC 抽样通过，`issues=[]`；参与完整 index 校验的文件合计 11,940,730,830 B。该检查没有逐个重新计算所有 blob，也没有重新运行特征提取、预测或评测。旧失败、partial 和正式结果各保留原身份，不能因新链接而合并。

## 代码整合与命令

经验证的本地源码已提升到远端 `/users/fotile/VAD`：456 个文件新增或更新，216 个语义一致文件保留原字节，0 冲突；`.git` 历史未改，也没有 commit 或 push。远端整合后回归 **191 passed、2 skipped**，`compileall`、encoder catalog 和 paper status 已完成；两项跳过来自服务器 Transformers 版本未导出 `VideoMAEImageProcessorPil`。这验证了当前整合源码的一组回归，不为每个历史 run 重新出具结果。

`src/vadbench/token_reduction/` 已是独立压缩子包；`clean_encoder_batch` 位于 `src/vadbench/data/batches.py`，旧导入仍兼容。本轮代码重构已将通用提取/恢复/合并、检测、质量和效率实现迁至 `workflows/`；特征身份在 `data/feature_contracts.py`，许可在 `engine/compatibility.py`，UR-DMU 后端在 `integrations/detectors/urdmu/`。旧 `paper` 路径只保留兼容导入，论文冻结与阶段规则保留原职责。当前 profile active 为 V2/VMA/TimeSformer，默认新输出是 `assets/experiments/icassp2027/runs`；显式输出根和旧运行路径不变。详见[当前架构](../architecture/current-system.md)。新增 `python -m vadbench.workflows.dsanet extract|score|quality` 三阶段入口，默认复用框架CLIP bridge，并调用已有质量评价；使用说明与输入/输出见 [DSANet入口](../../src/vadbench/workflows/dsanet/README.md)。原run code目录保留为历史身份，新入口未重跑正式模型实验。

旧CLIP Python3.9环境通过 `python scripts/icassp2027/dsanet_legacy.py extract|score` 使用同一实现；现代Python3.10+用模块命令。真实3.9.23的提取/评分模块、核心桥，以及3.10.20的质量模块导入均通过，最终增量测试17项通过。帮助菜单不加载模型；尚未由新入口重跑正式模型结果。

下列只读命令已在远端当前源码上取得回执，使用既有环境，不安装依赖：

```bash
cd /users/fotile/VAD
export PYTHONPATH=/users/fotile/VAD/src
PY=/users/fotile/VAD/.encoder-envs/v2/foundation-video-v2/bin/python
"$PY" -m vadbench encoders list
"$PY" -m vadbench.paper status --project projects/icassp2027/profile.yaml
```

`status` 的路径检查不等于权重加载或GPU运行验收。node3公网SSH再通过内部`ibnode2`登录node2已验证，命令见[服务器手册](server.md)。迁移前node2 dispatcher为`pending=0, running=0, done=233, recovery_done=28`；恢复后的独立复核仍为233 done、0 pending/running/failed，node3旧协调器保持held_research_protocol。回执为`assets/provenance/control-resume-verification-node{2,3}-20260926.json`，其他日期的作业状态仍需现场读取。

## 继续查阅

- [项目入口](../../projects/icassp2027/README.md)：论文代码、协议、正式特征身份及研究资料。
- [项目进度](../../projects/icassp2027/progress.md)：历次研究回执，顶部旧整理记录保留原日期。
- [框架操作流程](workflows.md)与[服务器手册](server.md)：通用命令、环境和路径注意事项；其中早期资源快照须按原日期理解。
- [论文交接](../../projects/icassp2027/handoff_v2/README_先读我.md)：正式实验与论文证据导航。
