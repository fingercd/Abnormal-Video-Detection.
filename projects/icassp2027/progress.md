# ICASSP 2027 当前进度

更新日期：2026-09-18。当前状态：**完整研究 goal 进行中；尚无经确认的正常—异常规律或插件有效性结论。**

## 当前持续执行记录

用户要求上传服务器、维护本地 Git，并在完成正常—异常观察、四 encoder 复核、免训练与可训练
简单插件的真实比较后统一交付。未完成研究之前不将工程进展标成研究完成。

- 本地工作分支：`qzt/icassp2027-evidence`，从已验证的 `25f9593` 开始。
- 已将 `25f9593` 经校验后的增量 Git bundle 上传，并建立服务器独立工作区
  `/users/fotile/VAD-icassp2027`。原 `/users/fotile/VAD` 的 156 个受保护源码/配置文件摘要未变。
- 已复用服务器原有 foundation 环境和 VideoMAEv2 overlay，完成一段真实视频的 CPU observer /
  identity 验证，features 与 pooled 最大绝对差均为 0。run：
  `observer-validation-20260917T155850443123Z-7e6ae0e7`。
- 用户已确认主指标容忍度为 **0.5 个百分点**，`protocol.yaml` 记录 `quality_tolerance=0.005`。
  用户离开期间不再提问，实际待答事项集中在
  [questions-for-return.md](decisions/questions-for-return.md)。
- UCF 现有 node2 下载任务 PID 112405 保持运行；最近盘点约 509/1900 个文件完成，总字节约
  16.6%。这是当时进度，不是当前完成证明；依同节点字节变化继续核实，不重复写入同一分块。
- node2/node3 当时均无完全空闲 GPU，node1 已知线路不可达。继续做受限 CPU 工程验证、数据
  准备与代码检查，不占用他人的 GPU、不停止用户已有服务。
- 正在补齐四模型真实几何、固定数据角色与候选、输入控制量、完整窗口采样和 pooled-only
  检测链。局部测试通过不等于四模型真实实验或插件结论通过。

### 2026-09-18 持续运行补记

- `9e1edc9` 和 `799670a` 已通过增量 bundle 上传；服务器主工作区现为 `799670a`。
  真实 CPU 探针使用独立冻结工作区 `/users/fotile/icassp2027-runs/code-799670ab`，
  正在运行 8 个视频、每视频 8 个片段。最近回执为 7 个片段完成，运行尚未结束。
- 固定 debug 视频是 Abuse005/018/025/040 与 Normal_Videos022/037/043/083。
  四个异常视频均属于 Abuse，因此这一工程小样本不能支持异常普遍性质。
- 本地限定镜像 8/8 下载完成，共 67,777,819 字节；大小与 SHA-256 均对应服务器冻结身份，
  无残留分块。它不是完整 UCF 数据集，服务器原下载继续运行。
- 本次进一步修正 TimeSformer embedding 的原始 batch 语义、分离 attention 的不可用标记、
  P16 的分支边界，以及 head checkpoint 的不可变训练身份。训练前强制比对真实 fit manifest
  摘要；一个已训练 checkpoint 可在严格许可下评估多个目标缓存，不回写 checkpoint。
- 当前本地解释器 `.venv/Scripts/python.exe` 执行 `python -m pytest tests/icassp2027 -q`：
  **134 passed**；`python -m compileall -q src tests`、受影响文件 Ruff 与 `git diff --check` 通过。
  此结果是工程测试，不是四 encoder 的真实质量或速度结论。

### 后续实测与在运行任务（2026-09-18）

- 本地 `243444d` 冻结工作区的 VideoMAEv2 debug8 已完成全部 64 个片段，退出码 0。
  run 为 `probe-20260917T174831982685Z-1f7ce815`，输出 51,136 条可用统计和 3,072 条
  not-applicable 记录。它仍是 8 个独立视频；四个正视频都属于 Abuse，不能泛化为异常规律。
- 服务器 `243444d` 真权重、真视频验证完成：TimeSformer run
  `observer-validation-20260917T174326293687Z-40667352`，VideoMAE run
  `observer-validation-20260917T174601576537Z-281e9f4f`。两者 native/observer/identity
  features 与 pooled 最大差均为 0；缺失 attention 统计如实记录，不补造。
- 真实 VideoMAEv2 的单次中间层固定索引实验确认 `1568 → 784` 后缀序列：block index 3 后
  gather，block 4–11 实际接收 `[1,784,768]`，pooled 仍为 `[1,768]`。identity 最大误差
  `3.5762786865234375e-07`；冻结主干后内部 leaf 参数梯度为 `-0.002514377236366272`。
  这是变长及梯度工程证据，不是最终 selector、检测质量或净加速结论。
- V-JEPA2 新增显式冻结主干内部插件梯度入口，默认 no_grad 推理不变。真实小模型验证插件
  梯度能穿过冻结后缀；实权重梯度验证尚未完成。原 `243444d` 的实权重 CPU observer 验证
  已以 2 线程、nice 15 启动，控制目录 `cpu-vjepa-verify-20260917T175520Z`。
- 最近独立 GPU 盘点仍无可用卡。node3 持久观察守候任务 `gpu-observer-queue-20260917T174607Z`
  正在等候，只有无计算 PID 且显存低、持有本任务租约时才运行；发生外部争用则终止自己的进程。
  这不是已完成的 GPU 实验或计时。
- 正式 explore128 与 confirm64 的冻结镜像共 9,980,051,375 字节（9.295 GiB），正在按原候选
  补齐本地镜像。服务器全量 UCF 原下载保持运行；未修改候选、原下载队列或其分块。
- 本地 debug8 的近重复候选审计未找到满足固定规则的 pair；9 帧摘要无法排除所有同源或
  不同裁剪，因此 source group 仍 unknown。接触表可见 Normal_Videos083 的结束卡，保留此
  内容异质性记录，不为获得更好结果临时删片。
- 检测控制器已接通真实 pooled 抽取、独立 validation store、TopKMIL BCE 与预测；新增
  验证缓存的采样策略核对、source group 冲突拒绝与 profile/assets 解析。
  本次相关测试为 **164 passed**，包括 validation 非空的 toy 集成测试；真实 head 训练待执行。

### 实际执行补充

- 代码 `6ab8781` 已做本地提交并上传服务器主工作区；模型运行使用独立 checkout。
  本地真实检测工程 run `paper-detection-20260917T181257007323Z-886655a8` 已启动，
  train 4 视频 ×32、validation 2 视频 ×32 的 pooled 特征已完整生成，无抽取失败。
  两个额外 debug 视频用于 dense 预测覆盖检查，主干冻结、TopKMIL BCE 5 epochs、seed 0；
  整个划分均来自原 fit 调试集，只用于工程核验，不参与性质选择或正式质量结论。
- 正式 explore 目标仍为预先冻结的 128 视频。核对发现实际服务器锁 seed 是 `20260918`，
  以原锁为准；全部 1610 个训练 ID 的角色和全部 explore/confirm 候选与本地重建一致。
  lock SHA-256：`53aacbd89d221ff036625927ff5eccee8286704652bf436b093cd4ea89d1fc59`。
- 已按实际落盘身份冻结第一计算批 127 视频（1016 片段）；34 个有服务器 SHA 比对，
  93 个绑定云端 fid/预期大小与本地实算 SHA，不混淆两种证据。唯一未齐候选为
  `Normal_Videos533_x264`。run `probe-20260917T182551484047Z-0a3dcc04` 在本地 CPU 4 线程运行，
  max_tokens=256、max_queries=32。剩余一条将单独补齐，128 视频齐备前不选择性质。
- debug8 的完整统计核验：64 clip 与 controls 的 ID、标签、分区全一致；799 个可估计签名，
  P13 无 CLS 的 48 项不填补。运动×亮度仅覆盖两组匹配，所有 scene unknown；没有从此小样本
  锁定任何性质。分页图已实际渲染并检查，图内明确标 debug、视频样本数和 site/head。
- **工程记录偏差**：绘图子任务在服务器 `code-6ab8781` 的 classic debug 探针运行期间改写了
  已跟踪的 `scripts/icassp2027/analyze_probes.py`（仅图标题）。完整差异、时间、SHA 已存
  `outputs/icassp2027/incidents/plot-only-source-mutation-20260917T181849Z/`。核验模型/collector/
  bridge/stages 源码均等于 HEAD；仍不把该工作区称为运行期间完全不可变，也不把该次运行当作
  正式确认。保留现场，未追溯改写回执；后续临时绘图脚本放 ignored 控制目录。**本地正式探索
  和本地检测工程 checkout 没有此改写，保持不变。**
- XD 官方六个视频 ZIP 已用匿名 HTTP Range 审核目录：4754 个视频、1 个目录，无路径逃逸
  成员；压缩 79.57 GiB、展开 80.06 GiB。后续核实：node2 的 curl HEAD 返回 403，但标准
  urllib + 本次匿名 CookieJar 的 GET Range 0-0 返回 206；不能把 HEAD 的结果误写成数据不可达。
  本机初次归档 worker 误用 Windows PowerShell 5.1，缺少 HttpClientHandler；改为已验证的
  PowerShell 7 后实际写入约 7.03 GB。因服务器直接下载路径已验证，已停止这组自己的本机
  下载进程，保留两份 part/身份记录作备份，正在安排 node2 独立下载；UCF 原下载不变。
  此状态不表示 XD 数据已齐，也未读取测试事件标注。

以下保留上一轮基础重构的已完成记录。

此前基础重构范围是已确认的 M0 与 M1 工程基础：论文入口、配置、身份契约、cohort、只读 collector、
一级统计和 identity 桥。具体 reducer、dense sampler、XD 协议、正式 detector 实验和论文数字
依照研究顺序留待后续阶段，不在本轮以占位实现冒充完成。

## 本轮变更

| 范围 | 实际完成内容 | 验证边界 |
|---|---|---|
| 论文入口 | profile/protocol、路径示例、status/probe/verify CLI、严格字段解析与 dry-run | 原有全量 catalog 与 CLI 不变；只有 profile 中的 active ID 可由论文入口调度 |
| 观察与标签 | cohort 的视频/同源分区检查、W 标签限制、collector 与分析标签 join 分离 | 不允许官方 test 进入开发 cohort；输入 encoder 的 metadata 已剥离标签与文件名 |
| 一级探针 | P01/P02/P04/P07/P10/P11/P13/P16 的有限统计，缺失或不适用时明确标记 | 仅描述统计实现与工程回执；没有正常—异常 finding card 或统计结论 |
| 模型桥 | 四模型已知 block/site 定位；TimeSformer 分离时空 site；V-JEPA 2 encoder/predictor 边界及 no_grad 说明 | 其他三模型使用真实库的随机小模型验证模块定位；真权重几何和完整采集验证尚未运行 |
| VideoMAEv2 | 真实 Conv3d tubelet 来源、实际展平顺序、native attention 行采样、identity 与 observer 等价验证 | 真实视频、真权重、本地 CPU、一个无标签 clip；不作为速度或异常性质证据 |
| 实验身份 | backbone/representation/sampling/training 身份、缓存键、显式兼容声明校验 | 现有 predictor 的严格指纹保护未改；跨表征/采样的预测执行仍未接入 |
| 工程保护 | 原始文件摘要快照、只读保护检查、测试、wheel 回归与使用说明 | 工程改造未修改其他 encoder、注册表、原公共源码、权重或环境；随后按用户要求进行本地 Git 归档，见下节 |

新增主要文件位于 `src/vadbench/paper/`、`research/`、`token_reduction/`、
`tests/icassp2027/`、`scripts/icassp2027/`、`configs/papers/icassp2027/`、
`projects/icassp2027/` 和 `paper/icassp2027/`。
已有文件只调整根 `README.md`、`README-CN.md` 的导航及 `.gitignore` 的本机路径忽略规则。

## 最新真实回执

Run ID：`observer-validation-20260917T150359001546Z-cceee66f`。

- [架构与等价回执](../../outputs/icassp2027/runs/observer-validation-20260917T150359001546Z-cceee66f/architecture_receipt.json)
- [运行摘要](../../outputs/icassp2027/runs/observer-validation-20260917T150359001546Z-cceee66f/summary.json)
- [统计输出](../../outputs/icassp2027/runs/observer-validation-20260917T150359001546Z-cceee66f/probe_summary.jsonl)
- [本轮锁定记录](locks/refactor-20260917.json)

解释器：`D:/PythonProject/VAD/.venv/Scripts/python.exe`；实际导入
`D:/PythonProject/VAD/src/vadbench/__init__.py`；设备 CPU。
环境为 Python 3.11.15、PyTorch 2.13.0+cpu、Transformers 5.16.1；版本与来源同时记录在
同 run 的 stage provenance。没有安装或升级模型环境。

输入为已有真实视频 `data/smoke/mlvu-surveil-8.mp4` 的固定 clip，权重使用已有
`weights/videomaev2-base-hf`。核心 safetensors 摘要与登记匹配：
`ebffa1874066ea227330016e58a848e9e2bb1ff5605746459bded1122a42176d`。
模型实际实现文件摘要、源码工作树摘要、输入视频摘要、采样帧、处理后的 shape 均随 run 保存。

| 项目 | 实测结果 |
|---|---|
| block 数量 / 每头数量 | 12 blocks / 12 attention heads |
| tubelet / patch | 2 帧 / 16×16 |
| 网格与特征 | 8×14×14；features `[1,1568,768]`，pooled `[1,768]` |
| CLS | 无；P13 标记 not_applicable |
| observer features / pooled 最大绝对差 | 0 / 0 |
| identity features / pooled 最大绝对差 | 0 / 0 |
| attention 来源 | native eager、dropout 前概率；有限 query、完整 key 轴 |
| 当前有效范围 | 工程等价性和真实几何；`reduction_ready=false` |

注意：此原生模型使用函数式 `linear` 计算 Q/K/V，qkv 模块本身不被调用。
桥记录实际投影路径并观察 native attention；不把从未触发的 qkv hook 写成采集成功。
几何坐标属于预处理后的网格，不宣称已有原视频像素级定位。

## 验证与命令

本轮使用上述本地解释器，执行了新能力测试及文档 02 第 11 节列出的核心兼容测试。
最终结果：**182 passed，1 skipped**，其中新能力测试 64 项全部通过。
跳过项为 Windows 创建测试符号链接权限不足；不是模型运行失败。
最后一轮结果及完整命令见：

- [最终测试日志](../../outputs/icassp2027/refactor/20260917T141225Z/final-tests.log)
- [最终测试命令与退出状态](../../outputs/icassp2027/refactor/20260917T141225Z/final-tests.json)
- [保护检查结果](../../outputs/icassp2027/refactor/20260917T141225Z/protection-final.json)
- [原工作树基线](../../outputs/icassp2027/refactor/20260917T141225Z/baseline.json)

验证命令要点：

```powershell
.venv/Scripts/python.exe -m pytest tests/icassp2027
.venv/Scripts/python.exe -m compileall -q src tests
.venv/Scripts/python.exe -m ruff check src/vadbench/paper src/vadbench/research src/vadbench/token_reduction tests/icassp2027 scripts/icassp2027
.venv/Scripts/python.exe -m vadbench.paper verify --project projects/icassp2027/profile.yaml --encoder videomaev2 --video data/smoke/mlvu-surveil-8.mp4 --device cpu
uv build --wheel --offline --out-dir outputs/icassp2027/refactor/20260917T141225Z/wheel-final
.venv/Scripts/python.exe scripts/verify_wheel_resources.py outputs/icassp2027/refactor/20260917T141225Z/wheel-final/vadbench-0.1.0-py3-none-any.whl
.venv/Scripts/python.exe scripts/icassp2027/verify_paper_wheel.py outputs/icassp2027/refactor/20260917T141225Z/wheel-final/vadbench-0.1.0-py3-none-any.whl
```

真实运行在进程内设置离线模式和 4 个 CPU 计算线程；不改变系统配置。
wheel 检查从临时安装/解包目录导入，验证旧资源与新 paper profile/suite 的打包后解析，
没有把“源码目录可导入”当成 wheel 成功。

保护检查覆盖 316 个原有源码、配置、登记和文档文件，原始内容变化仅有计划内的三项入口文件。
大型权重、视频、环境、external 和历史 outputs 不在本轮写入范围；没有递归哈希所有大型资产。
Windows 缺少创建符号链接权限，原有 symlink 测试跳过，不能计作已通过。
GPU 验证未运行；服务器盘点时八张 GPU 均有作业，本轮未抢占或改动这些进程。

## 本地 Git 归档（2026-09-17）

用户随后明确授权“本地做好 git”。工作分支为 `qzt/icassp2027-foundation`，采用两次提交：

1. `ff74605`：保存本轮之前已有的基础框架、配套文档和测试，内容按重构前 baseline 核对。
2. 随后的 ICASSP foundation 提交：本轮新模块、论文配置与入口、测试，以及 README 导航和本机路径忽略规则。

拆分原因是新模块依赖此前未提交的 `hashing.py`、`resources.py`、`videomaev2_encoder.py`
和公共框架改动，仅提交新增目录会得到不完整的版本。原工作树中的这些改动保持原内容。
本节与第二次提交一起归档，提交号以 `git log -2 --oneline` 为准。

只做本地提交，不 push。视频、权重、环境与 outputs 继续忽略；`VAD_Idea/` 和两个根目录临时
修复脚本保持原位、未纳入本次提交。测试与运行回执中的原 Git commit/dirty 标记是运行当时事实，
不追溯改写；源码内容摘要可以用于核对提交后的相同实现。

## 下一阶段

下一项工作是固定训练集中的 4 个含异常视频与 4 个正常视频及其 cohort/source 分组，运行
`probe-pilot`。本地尚未建立该真实 cohort 与对应 UCF 训练 manifest，dry-run 会如实报告缺失。
服务器存在视频资产不等于该研究分区已经审计和冻结。

随后扩展正常—异常分布与独立确认；在取得可操作性质前继续保留 identity。
其他三模型的真实几何/只读等价回执、正式 GPU 验证和质量容忍度仍需在相应阶段落实。
正式结果和论文数字仍为空。
