# WAVAD 整理执行状态（2026-09-30）

本页是这次整理的当前状态；同目录编号文档保留设计与分类依据。代码、默认数据入口以及6个旧资产根的物理归档均已完成。四个data2根已在新资产根的storage中，两个共享users根已在VAD/archive中；旧路径是兼容软链接，分类导航继续沿用。原始数据备份和冻结证据保留，没有以删除历史数据来简化目录。

## 2026-09-30 代码重构与验收

本轮已完成通用实现迁移、部署契约收敛、profile/输出路径对齐，并同步到服务器
`/users/fotile/VAD`。源代码增量共71个文件，逐文件预期SHA检查通过后写入，备份位于
`assets/provenance/refactor-20260930-r01/source-before/`；没有覆盖并发修改。

- `paper` 的11个原实现模块移至公共层：特征身份进入 `data/feature_contracts.py`，
  检测许可进入 `engine/compatibility.py`；提取、恢复、合并、通用检测、质量比较和效率
  进入 `workflows`；UR-DMU后端与精确分块推理进入 `integrations/detectors/urdmu`。
  旧路径是同一模块对象的兼容导入。通用实现无 `paper` 导入，论文冻结门槛保留在项目流程层。
- 压缩包新增共享 `ReductionDeployment` / `ReductionExecutionContext` 契约，PairMerge、
  GroupSelect、ADGS与提取接口共用它；单次token操作的 `TokenReducer` 契约保持独立。
  方法公式、预算、随机行为和真实变长路径未改；未把只读观察桥当压缩能力。
- `profile.yaml` active为VideoMAEv2、VideoMAE、TimeSformer；新运行默认
  `assets/experiments/icassp2027/runs`。输出专用路径解析允许该资产挂载，拒绝 `..`
  和挂载内软链接外跳；配置输入继续受原有路径约束。V-JEPA2全局接入与历史资产保留。
- 恢复回归发现原执行回执已经包含每层token计数和插件耗时，而恢复比较仍按旧字段集合。
  已严格核对每层token计数/形状/位置，单独校验有限非负耗时，保留源回执原字节；
  完整恢复、再次恢复与篡改拒绝测试通过。未忽略代码/权重/特征SHA或放宽身份许可。

当前验收均绑定本轮源码：

| 范围 | 实际结果 |
|---|---|
| 本地6组受影响回归 | 509 passed、7 skipped、0 failed，516项；覆盖特征、接口、质量、正式流程、XD与CLI |
| 本地跳过项 | 6项Windows软链接权限限制、1项缺少einops的外部UR-DMU检查 |
| Linux服务器回归 | 290 passed、1 skipped、0 failed；Windows跳过的软链接边界在此执行通过 |
| Linux跳过项 | 新增外部作者模型类路径检查未找到其默认位置的固定UR-DMU checkout；后端与分块推理仍有本地相关单测 |
| 编译与包 | 本地/远端compileall；独立wheel的可移植导入、内置资源、encoder CLI与DSANet帮助检查通过 |
| 实际运行配置 | node3解释器为既有foundation-video-v2环境，源码来自 `/users/fotile/VAD/src`；默认输出解析到 `/data2/localdisk/fotile-wavad/experiments/icassp2027/runs` |

早先1006项大范围运行在中断前没有完成回执，未计入以上通过数。最终采用分组受影响回归，
每组独立保存退出码、JUnit与源码摘要；不是声称整个仓库全部测试已通过。没有重跑正式
论文模型实验，也没有改写已有特征、checkpoint、合同、指标或历史执行快照。

回执入口：`assets/provenance/refactor-20260930-r01/` 下的 `source-update.json`、
`verification.json`、`remote-regression.xml/log`、`local-verification-summary.json` 与
`final-audit.json`；本地回执在 `work/refactor-20260926/`（沿用任务开始时目录名）。
文档清单共120条，其中4份本地工作区参考原文已按原SHA归档到服务器
`reference-documents/`，清单的 `server_path` 指明实际位置；其余文档沿用仓库路径。
当前模块说明见[架构文档](../../../docs/architecture/current-system.md)和
[可复用工作流](../../../src/vadbench/workflows/README.md)。

## 2026-09-26 资产整理已执行

| 对象 | 当前落点与真实回执 |
|---|---|
| 框架源码 | `/users/fotile/VAD`；经预检提升 672 文件候选中的 456 个新增/更新文件，216 个相同或仅行尾不同的文件保留远端原字节，0 个并发冲突。`.git`、环境、权重与旧执行快照保留 |
| 原始数据 | `/data2/localdisk/fotile-wavad/datasets/`；UCF 1900 视频、103795576492 B；XD 4750 视频及1辅助文件、85869780149 B。复制后 rsync checksum 比较无差异，切换时再次核对全部相对成员与大小 |
| 默认数据入口 | `/users/fotile/VAD/data/raw/ucf_crime`、`xd_violence` 指向新完整副本；旧 UCF897 子集入口为 `ucf_crime_legacy_897`。原始来源目录仍保留 |
| 大资产入口 | `/users/fotile/VAD/assets` 指向 `/data2/localdisk/fotile-wavad`，node3 可通过现有共享挂载访问 |
| 已迁移物理根 | 四个旧`/data2/localdisk/fotile-icassp2027-*`实体同盘归入`assets/storage/<原根名>`；`/users/fotile/icassp2027-runs`和`datasets`实体同盘归入`/users/fotile/VAD/archive/`。源路径保留兼容链接。`assets/storage/users-icassp2027-runs`、`users-datasets`提供统一入口 |
| 特征导航 | 139 条链接，含 24 个冻结六格正式测试视图、6个 full-train 候选视图、12个预算扫描候选、CLIP–DSANet 测试与训练分片，以及原始发现入口；链接不复制或改写 feature bytes |
| 实验/来源导航 | 175 条实验系列、数据源和代码快照链接。连同特征入口共314条已在 node3 验证解析正确 |
| 物理切换验收 | 两节点各96个文件锚点的dev/inode/size/mtime/nlink及小文件SHA一致，各314原有导航一致；6个根inode保持。覆盖30784751个常规文件条目（含硬链接重复名称），没有逐个复制这些文件 |
| 正式特征身份 | 6个 dense test +18个 keep0.60 压缩视图：合同 SHA、完整 index SHA、各 index 首个 blob 的 SHA/zip CRC 均通过。24份 index 共11940730830 B；没有重新检查每个 blob 的所有数值 |
| 检测头 | 6个UR-DMU正式final和2个DSANet作者发布checkpoint均重新核对完整SHA；模型入口和校验状态在 `assets/catalog/models-20260926.jsonl` |
| 运行编目 | `assets/catalog/run-history-20260926-r02/` 登记3419个含结果/状态/配置回执的目录，观察到174个显式run ID，48个目录绑定冻结矩阵。包括旧框架输出、论文控制目录、代码快照内outputs和开发/失败尝试，未将目录数当独立实验数 |
| 文档 | 根中英文README、项目README已重写为当前入口；`docs/operations/wavad-assets.md` 汇总路径。两份过时接管/阶段计划和本地 `projects/icassp2027.zip` 已删除，冻结合同、负结果、论文交付包与证据保留 |

## 第一阶段代码接入和验证边界（2026-09-26）

压缩方法继续集中在 `src/vadbench/token_reduction/`，方法、模型桥和论文编排维持现有边界；将公用 `clean_encoder_batch` 从论文模块移到 `data/batches.py`，消除了 token_reduction 对 paper 的反向依赖，旧导入仍兼容。没有为目录整洁修改选择公式或重写旧 code digest。

新增 `vadbench.workflows.dsanet` 的 extract/score/quality 三阶段入口，默认复用框架 CLIP bridge，显式 `--bridge` 用于重放历史实现；score 不接收帧级真值，quality 调用现有帧级评价与配对统计。训练和 benchmark 的历史原脚本通过实验/代码快照入口保留，新包没有声称覆盖所有历史调度器。

验证证据：源码候选在本地和服务器各通过49项针对性测试；主源码提升后服务器回归191 passed、2 skipped（当前Transformers未导出VideoMAEImageProcessorPil）。最终增量在服务器17项通过，包括资产完整性、DSANet入口、旧环境启动器和既有质量映射；compileall、encoder catalog/paper status通过。真实Python3.9.23环境已成功导入提取/评分模块和同一核心CLIP bridge，Python3.10.20环境已成功导入质量模块；没有加载模型或重跑正式结果。旧3.9环境需用 `scripts/icassp2027/dsanet_legacy.py` 薄启动器，现代3.10+环境用模块命令；两者复用同一套实现。

## 回执在哪里

服务器所有整理回执由 `assets/catalog/current.json` 导航，主要包括：

- `catalog/dataset-cutover-20260926.json`：复制、成员、大小、旧路径与新路径。
- `catalog/formal_feature_bindings-20260925.jsonl`、`formal_feature_qa-20260926-r02.jsonl`：24份正式输入绑定与核验。
- `catalog/asset_views-20260925-r01.jsonl`、`experiment_views-20260925-r01.jsonl`、`asset_navigation_qa-20260926.json`：所有新链接的源/目标与核验。
- `catalog/run-history-20260926-r02/{runs.jsonl,summary.json,repeated_run_ids.jsonl}`：运行级回执目录与扫描范围；跳过blob、逐视频块和依赖目录，保留扫描边界。
- `provenance/source-promotion-20260926.json`：文件级切换和实际内容摘要；`pre-integration-remote-*` 保存提升前源码、dirty patch和status。
- `provenance/promoted-source-tests-20260926.log`：服务器191通过、2跳过回执。
- `provenance/organization-final-tests-20260926.log`、`dsanet-legacy-runtime-20260926.json`、`dsanet-quality-runtime-20260926.json`：最终17项与两套真实解释器的导入/帮助验证。
- `catalog/physical-consolidation-20260926.json`、`physical-paths-20260926.jsonl`：6根真实迁移及3455条原路径→物理位置映射，missing=0。
- `provenance/physical-relocation-{data2,users}-20260926-r01.json`、`physical-anchors-after-node{2,3}-20260926-r01.json`：原子切换、原inode和双节点验收。
- `provenance/maintenance-pause-node{2,3}-20260926-r02.json`：成功窗口中仅暂停已核实的旧VAD守护及非计算子进程；node2约16.56秒、node3约19.34秒，已恢复。独立超时恢复与中断回滚均用任务专用测试目录验证。
- `provenance/control-resume-verification-node{2,3}-20260926.json`：恢复后复核通过；node3旧协调器仍为held_research_protocol，node2仍为233 done、0 pending/running/failed。

## 保留的兼容关系与验证边界

1. 旧路径必须继续作为兼容软链接存在，以便冻结配置、相对blob路径和运行回执可读；它们不再是散落的物理目录。六根使用同盘原子目录交换，保留常规文件inode、硬链接与已打开文件描述符关系；没有跨盘重新复制约3078万条目。
2. 原始数据来源副本保存在`archive/datasets`作为备份；规范raw副本在`assets/datasets`。未删除这些数据备份，也未对所有历史blob重跑内容QA；正式24组索引及既有校验回执仍按各自范围解释。
3. node2的Tailscale线路故障已通过用户指定的路线绕开：先公网进入node3，再`ssh ibnode2`。两节点/users共享同一实体已由挂载和inode核对。后续集群操作可沿用此路线，不再把原线路超时写成不可抵达。
4. 归档是资产组织，不是新的科学实验。历史run仍使用冻结代码/环境身份；未改变测试数值、失败状态、方法选择和论文限制。
