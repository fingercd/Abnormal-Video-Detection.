# 架构审查与修改建议（2026-09-11）

当前分层方向适合项目目标：统一视频/时间轴契约、惰性注册、adapter 隔离上游、冻结特征后独立训练检测头，以及显式区分视觉 token、视觉记忆和 decoder KV。建议保留这些边界，先修复影响运行、实验身份和成功判定的缺口，再扩展模型数量或开展完整性能/精度实验。

审查基线为服务器 `ibnode3:/users/fotile/VAD` 的 `0badc3435e734a841110e29d497940bfda7cc707`，本地相同提交。本文保留文档审查时的基线发现。用户随后授权的修复与职责收敛已实施，逐项状态及验证见[本轮改造](../progress/2026-09-11-implementation.md)；下文代码行号和“当前”均指最初的审查基线。P1 表示应在下一次相关正式实验前修复，P2 表示后续有明确使用需求时处理。完整数据未就绪是实验前提，不是算法结论。

## 1. P1：修复参考配置的特征抽取阻塞

**已复现。** `cli._extract` 同时传入 definition 的 `checkpoint.local_path` 和实验的 `encoder.checkpoint`；`compute_encoder_fingerprint` 明确要求二者互斥。两个参考 YAML 都具有这组字段，实际模型构造成功后也会在创建抽取引擎时失败。

证据：[CLI](../../src/vadbench/cli.py) 第 633–654 行、[抽取引擎](../../src/vadbench/engine/extract.py) 第 217–220 行、[指纹函数](../../src/vadbench/features.py) 第 119–134 行。现有[CLI 抽取测试](../../tests/test_cli_extract.py) 返回空 checkpoint definition，避开了这个组合。

本轮复现使用真实 CLI、参考实验/definition 和真实抽取引擎，仅替换模型构造以避免加载权重，得到 `exit_code=2`、`checkpoint and checkpoint_id are mutually exclusive`，尚未进入视频解码。记录见[核对证据中的 reproductions](../evidence/server-doc-audit-2026-09-11.json)。

**建议：** 明确“本地资产内容 hash 优先、只有无本地路径时才使用不可变 ID”的单一路径；登记 ID 仍可作为独立 provenance 元数据保存。不要放宽互斥校验来掩盖调用错误。

**验收：** 两个参考 definition 的 path+ID 组合都能走过真实 CLI/引擎初始化；权重内容改变会改变指纹。随后用至少一个真实视频完成抽取并读回索引，不能只验证 fake adapter。

## 2. P1：成功结果必须属于本次执行且进程正常结束

**已复现。** [v2 native runner](../../scripts/server/run_native_encoder_matrix_v2.py) 第 159–200 行允许复用已有输出目录，递归寻找 `result.json`，从文件状态决定成功；第 237 行只看 `failed` 数量，没有要求子进程退出码为 0。

离线复现预置旧 `smoke_pass`，让本次子进程返回 17 且不写新结果。runner 仍生成 `status=smoke_pass, exit_code=17`，主命令返回 0。即使使用全新目录，写完成功结果后的进程崩溃也可能出现“结果成功、进程失败”的矛盾。

[汇总器](../../scripts/server/consolidate_encoder_v2_results.py) 第 33–103 行还会从历史文件中优先选技术成功，且把恰好 14 个通过作为返回 0 的条件。它不证明当前 code/env/video 身份与旧运行一致。

**建议：** 每次只接受明确的本次 result 路径；拒绝复用非空 run 目录，或严格按唯一 run_id 建子目录。将进程退出码、result schema、run/config/code/video/asset/environment 身份作为成功条件；历史复用显式标为 `reused`，先比对完整身份。汇总数量由本次选中的目标和状态计算，不硬编码 14。

**验收：** 旧成功+本次非零、本次写成功后崩溃、全 skipped、缺 result、身份变化五种情况都不得成为本次成功；正常成功仍保留完整证据。

## 3. P1：在正式 evaluator 入口落实协议与覆盖门禁

**已复现边界。** `predict` 默认检查完整、连续、无重叠的帧覆盖，但 [CLI evaluate](../../src/vadbench/cli.py) 第 581–600 行直接调用底层 evaluator。底层[投影](../../src/vadbench/engine/evaluate.py) 第 121–172 行允许空洞填 0、越界裁剪；官方 290 视频与数据审计也不是 evaluator 前置条件。

本轮以两个各 4 帧的视频、每视频仅 1 帧预测运行实际 record evaluator，结果将其余 6 帧补零并返回 AUC/AP。这个行为可用于通用实验，但产物没有足够信息阻止其被误认为完整官方协议结果。现有两视频 CLI 合成测试验证了通用入口，并没有覆盖正式协议门禁。

**建议：** 增加明确的 protocol 选择；官方模式在计算前校验 manifest/audit 身份、官方视频集合/计数、预测覆盖、区间边界与 split。复用当前预测覆盖实现，通用/subset 模式仍可用，但产物必须标明范围、覆盖率和输入 hash。禁止以填零或均匀重采样修补正式结果。

**验收：** 不完整覆盖、遗漏视频、非官方集合、越界/重叠与错误 split 在正式模式失败；显式 subset 模式正常输出且不能冒充完整 UCF-Crime。真实数据到位后与官方 `.mat` 做样例帧标签对照。

## 4. P1：让配置、实际构造与特征身份使用同一份解析结果

[orchestration](../../src/vadbench/orchestration.py) 第 73–107 行只把 definition constructor、`encoder.params` 和 `encoder.device` 传给 registry；顶层 `encoder.model/checkpoint/precision` 不会自动改变模型、权重或 dtype。registry 还会合并自己的 default kwargs。当前 YAML 写下的内容、实际参数和 provenance 不是同一份对象。

抽取指纹已经包含整个 `streaming` 段，**包括 compression**；问题不是漏掉所有压缩配置。具体风险是：实际 `params.model_path` 可与用于哈希的 definition checkpoint 路径脱节、registry defaults 未作为统一解析结果落盘，以及改无效顶层字段可能只改变指纹却不改变执行。

**建议：** 用一个集中解析函数返回 adapter 类型、最终 kwargs、实际 checkpoint 身份、预处理/精度和两类缓存参数。adapter 构造、指纹、日志均消费该结果。删掉无效字段或显式映射；不要新增另一套平行配置格式。区分不影响数据身份的设备位置与会影响结果的 precision/feature stage。

**验收：** 改 dtype、权重、feature stage、原生/外部 policy 时，实际构造与指纹一致变化；仅重定位同内容资产时身份不变。记录中的权重 hash 必须属于真正加载的路径。

## 5. P1：为公平比较提供实际可训练的实验配置

当前两个参考 YAML 的任务不同：[VideoMAE](../../configs/experiments/ucf_videomaev2_weak.yaml) 是 weak MIL，[HERMES](../../configs/experiments/ucf_hermes_stream.yaml) 是 temporal supervised，却都指向官方训练视频级 manifest。后者在数据到位后仍会因没有显式训练时序标签而排除全部样本。

VideoMAE YAML 没有 `training` 段；[runner](../../src/vadbench/engine/runner.py) 默认 batch=2、epoch=1、lr=1e-3、无 validation，当前 task 默认 bag BCE、ranking 权重 0、普通 shuffle，最后保存 final checkpoint。它是工程默认值，不能称完整 Sultani 2018 复现。`32` 仅说明本项目的段采样兼容。

**建议：** 首先把主对照统一为官方 train/test、相同 bag/feature level、相同 head/loss 和明确的训练参数。暂不为没有来源的强监督标签造数据。若目标仍是冻结特征比较，保留这一简单流程；需要缓存精度消融时固定同一 HERMES 基座、`decoder_contextual` 读出和相同训练配置，只改变压缩策略/预算。

如使用 UCA 的部分语义映射，还必须显式设置 `assume_unannotated_is_normal=false`，因为 runner 默认 true 会把未覆盖部分加入 normal loss。模型选择、阈值、epoch 和预算选择全部留在训练内 validation，不能使用官方 test。

**验收：** 配置在训练前核对 supervision；弱监督主对照可用真实 normal/anomaly 小样本完成 `extract → train → predict → evaluate`，记录全部差异和 subset 范围。正式全量实验仍等数据与审计就绪。

## 6. P1：把运行追溯贯穿全部阶段

现有 FeatureStore/checkpoint 内容 hash、原子写和 sidecar 校验值得保留。缺口在于各阶段的身份字段不同：[extract](../../src/vadbench/cli.py) 第 658–667 行只记录 manifest 路径与视频数；evaluate 写裸 metrics/NPZ；benchmark 缺少统一失败状态。固定 run_name 或重复输出还可能覆盖先前 provenance。

**建议：** 用小型公共记录/辅助函数统一 stage 的 `running/completed/failed`、code/dirty、解析后配置、manifest/checkpoint/父产物 hash 与环境。路径采用可重定位逻辑路径，绝对路径只作诊断。已有 JSON schema 的产物继续校验；run、普通 metrics、cache telemetry 补齐清晰的版本契约。无需引入通用调度平台。

**验收：** 任一阶段异常退出都保存失败阶段与输入身份；同路径不同 manifest 内容不会复用旧结果；每个指标能追溯到预测、训练 checkpoint、特征与数据版本。

## 7. P1：收紧资产、checkout 与环境核验的成功语义

[资产脚本](../../scripts/server/fetch_encoder_assets_v2.py) 第 98–110 行在无法取得 checkout HEAD 时也可能标 `verified`；第 145–162 行下载后直接移动到最终路径，状态为 `downloaded_pending_registry_hash`，尚未按 registry hash 校验。后续 no-overwrite 又会阻止覆盖这个未验收目标。

环境 `verify` 和资产核验中的单项 `missing/version_mismatch/manual_required` 通常不会令命令返回非零；node2 获取是项目操作规定，资产脚本本身未实现主机强制门禁。操作文档已如实说明这些边界。

**建议：** staging 完成后先检验 revision/文件清单/size/SHA256，通过后才发布到最终目录。没有 Git metadata 的离线源码需要可核对的归档内容 manifest，未知不能写成 verified。区分“审计完成”与“所选目标可运行”，为自动化提供显式 required-targets 检查和稳定退出码。

**验收：** 错 hash、缺 revision、环境不匹配不能成为可运行状态；失败资产不占用已验证目录；人工缺失项仍保留具体原因。

## 8. P2：将性能测量放在实际执行模型的运行时

当前 v2 smoke launcher 选独立 Python/overlay，但 [benchmark plan](../../src/vadbench/benchmark_plan.py) 在单进程逐 case 构造模型，没有复用相同运行时选择。父进程 CUDA allocator 的峰值也不能代表外部 worker 的显存。`accuracy_comparable` 是采样与读出条件判断，不是模型精度或上游预处理完全一致的证据。

**建议：** 沿用既有运行环境选择，让每个 case 在实际模型进程内完成 warmup/repeat、同步、显存与失败记录，再统一汇总。分别报告加载、IPC、解码、adapter 执行和源视频覆盖范围；把系统级对照与同基座缓存消融区分开。

**验收：** worker 的实际设备与显存进入结果；同一 case 的单独运行和矩阵运行统计一致；不同 adapter 的 resize/crop/normalization 与 feature stage 都可追溯。

另一个具体重复在 `long_video/base.py` 的 `ExternalPythonWorker`：它可把帧/状态转成 JSON list，通过 stdin 交互；`worker_protocol.py` 则已有 NPY sidecar 协议。优先统一实际使用的外部执行路径，移除被替代的 JSON tensor 传输；不要仅因类名含 ExternalPython 就认为隔离已经发生。大帧批传输应保留 shape/dtype/hash/path 门禁，模型私有状态保持在持久 worker 内。

catalog 还保存静态 `smoke_pass` 历史结论，容易与实现就绪状态混淆。将实现/资产/许可状态与带日期的运行证据分开，在显示时再关联，而不是在多份 YAML 和测试中重复固定通过数量。

## 9. P2：收敛旧版依赖，修复仍在使用的旧流程

主 VideoMAE V2 adapter 仍引用 `lab_anomaly.models.vit_video_encoder`，但 wheel 配置只打包 `src/vadbench` 和一个 catalog。源码 checkout 的测试不能证明独立 wheel 安装能加载该模型。

**建议：** 把仍被主框架使用的编码器包装移入主包；旧版应用调用同一实现。当前只承诺 repository checkout 的操作方式，若需要发布包，再补配置/registry 资源打包和干净目录安装验证。不要为此次文档任务删除整个旧目录。

旧原型还有几个独立问题，是否继续维护取决于它是否仍是实际需求：

| 问题与依据 | 建议与验收 |
|---|---|
| `clip_dataset.py:181–207` 只对比预切参数与视频数；同数 CSV 重排/改标签可能错配旧 NPZ | manifest 按稳定 video ID/相对路径与标签绑定；重排不改变映射，内容变化明确拒绝 |
| `precompute_clips.py:20–25` 默认16帧，训练 YAML 为12帧；预切 YAML 未提交 | 共享一个明确配置来源；默认预切与训练参数一致 |
| `train_end2end.py:132–139,649,721–727` 随机验证切分且 class weights 使用包含验证标签的 all_vi | 仅保留为原型，或改为显式 split 与 train-only 统计后再讨论正式结果 |
| `video_reader.py:56–68` 解码失败复用前帧/黑帧 | 显式报告失败/替换及计数，不把损坏视频静默当正常样本 |
| `rtsp_service.py:227–267` 默认 CLI 值覆盖 YAML；部分字段未接入 | 明确配置优先级，删除无效字段并验证实际生效 |
| `known_event_runtime.py:85–92,167–173` 队列满丢窗口，无停止生命周期 | 有实际实时服务需求时补停止/回收与丢帧统计 |

对应文档已改成[旧版实际能力说明](../../lab_anomaly/README-CN.md)，不再宣称旧 embedding/聚类/伪标签等已删除链路可用。

## 10. P2：其他应有边界的精确补强

- `PredictionRecord.ground_truth` 当前存视频标签，不是逐片段真值。当前 evaluator 不消费该字段，所以不直接污染现有 AUC；建议用 `video_label` 或显式 scope 避免新消费者误用。
- overlay fingerprint 主要取文件路径/大小，未覆盖同尺寸内容漂移；已有 overlay 也可能直接复用。改成内容 manifest，并把它绑定到 run identity。
- v2 GPU 检查只有瞬时显存阈值，无法替代进程归属或调度分配。沿用集群分配机制，不自建复杂 GPU 调度器。
- `link_ucf_crime.sh` 的目标检查依赖 `/users/` 文本前缀；应先解析绝对路径再核对目录边界。`install_bundle.sh` 恢复 tar 也应检查成员路径与目标，日常实验不使用恢复脚本。
- normal-only 误报、覆盖率、按类别/时长分桶尚未完整实现；在主协议可靠后补诊断，保持 micro frame AUC 为主指标。

编码器结果写入还有三处需要一并核对：`integration_matrix._read_existing_success` 和 `write_smoke_result_v2` 也只按成功状态处理旧结果，不能只修服务器外层；`_coerce_runner_result` 可为缺少状态的外部返回值默认赋予成功，应该拒绝不完整返回；smoke 的部分身份读取会将解析失败降级为 unknown/空对象，登记路线应将必要身份缺失记为失败，并在写入前检验完整结果 schema。这些均为静态调用路径发现，本轮未新增真实模型实验来证明其影响范围。

## 建议执行顺序

1. **恢复可信执行：** 特征抽取 path/ID 冲突 → 本次成功/失败判定 → evaluator 官方协议门禁；每项独立验证。
2. **固定实验身份：** 生效配置与 fingerprint → 阶段追溯 → 资产/环境验证。
3. **完成真实最小实验：** 数据/manifest 就绪 → 同监督与同 head 的小样本闭环 → 独立运行时性能测量 → 全量正式结果。
4. **按需求维护旧应用与诊断：** 仅在仍有训练/RTSP 使用者时修复相关原型问题。

## 证据范围与验证

本轮以 147 个受跟踪的项目源码/脚本/测试文件为审查范围，按模块通读项目实现并核对相关测试、YAML、上游锁与 schema；不把第三方依赖仓库逐行审查纳入“全部项目源码”。服务器受跟踪树与本地基线相同，运行环境、数据路径和既有产物直接从服务器读取。

Windows 全量测试为 366 passed/9 skipped，node3 为 374 passed/1 skipped；这说明现有测试通过，不排除上述未覆盖组合。另完成抽取冲突、评测空洞和旧成功结果三个离线复现，详见[本轮证据](../evidence/server-doc-audit-2026-09-11.json)。没有开展新的真实模型前向、GPU benchmark、UCF 数据导入或训练。
