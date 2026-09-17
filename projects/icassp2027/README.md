# ICASSP 2027 项目入口

本轮基础重构把论文配置、只读观察和 token 操作契约接到现有 VADBench。
active 列表仅由 `profile.yaml` 管理；普通 `python -m vadbench encoders list` 继续返回完整 catalog。
已补充完整中心窗口与 dense 采样、显式检测身份兼容入口，以及不带选择规则的外部索引干预工具。
当前没有最终 selector、经真实数据验证的压缩方法、XD 专用评测结果或论文数字。

## 运行入口

使用目标模型已经跑通的 Python，示例中的 `python` 需替换为该解释器。

```bash
python -m vadbench.paper status --project projects/icassp2027/profile.yaml
python -m vadbench.paper probe --project projects/icassp2027/profile.yaml --suite configs/papers/icassp2027/suites/probe-pilot.yaml --dry-run
python -m vadbench.paper verify --project projects/icassp2027/profile.yaml --encoder videomaev2 --video data/smoke/mlvu-surveil-8.mp4 --device cpu --dry-run
```

上述命令不加载权重、不创建输出。`status` 的 probe-ready 为 null，含义是本命令不核实运行状态；
真实状态以当前源码对应的验证回执为准。缺失数据和权重在 dry-run 的 blockers 中列出。
未知字段、inactive encoder、未实现 method 和开发阶段的 test 角色请求会失败。

去掉 `verify` 的 `--dry-run`，会校验登记权重摘要，复用原 adapter 和预处理，在一个真实 clip
上分别运行 native、observer、identity 三次前向。输出写到独立的
`outputs/icassp2027/runs/observer-validation-<时间>-<唯一ID>/`，包括架构回执、统计与 stage-v1 来源。
失败也保留独立失败回执。这是工程验证，不能用其运行时间宣称加速，也不为输入视频补造标签。

四个桥已实现各自原生几何，真实权重的完整验证优先完成了 VideoMAEv2。其他三模型已用实际库的
小配置验证结构路径，服务器真权重验证继续进行。TimeSformer 显式区分 temporal/spatial，
V-JEPA 2 只定位 encoder、保留 wrapper 的 no_grad 限制。未知位置或布局会明确失败。

## 路径与数据权限

按需将 `assets.example.yaml` 复制为 `assets.local.yaml`，填写已有视频与权重路径。
本机文件被 Git 忽略，只接受 `dataset_roots`、`checkpoint_paths`，不放凭据。
相对路径基于仓库根目录；非标准位置的 profile 使用 `--root` 明确根目录。

默认 `protocol.yaml` 使用 W：开发只用训练视频级标签，正常参考统计只在 fit 拟合。
用户已在正式测试前将 quality_tolerance 固定为 `0.005`，即最多下降 0.5 个百分点，
同时报告差值置信区间。
`paper.compatibility` 提供显式身份及兼容声明校验。`paper.detection` 通过绑定实际 checkpoint
与目标缓存的 typed permit 执行已声明的采样/表征配对；原 predict 的默认严格指纹检查不变。
对应工程测试通过不代表已经产生真实 UCF 检测结果。

执行 `probe` 前需要真实训练 manifest，以及只读的 cohort JSONL sidecar。
suite 中 `manifest` 指向原有 VideoManifestRecord 格式；cohort 每行对应一个固定窗口，例如：

```json
{"video_id":"actual_manifest_id","clip_id":"actual_manifest_id:segment-00","official_split":"train","partition":"fit","role":"debug","weak_label":0,"label_source":"video_weak","source_id":"actual_source_group"}
```

示例仅说明格式，不是运行数据。每个入选视频应包含 `segment-00` 至 `segment-07` 共八行，
与示例 suite 的八个均匀窗口一致。实际弱标签必须与训练 manifest 一致；没有可靠同源信息时
保留未确定状态，不声称已完成近重复审核。`source_frames` 可进一步冻结每个窗口的真实帧索引。
同一视频与同源分组不能跨 fit/confirm/select，test 视频和 W 路径的时间真值会被拒绝。

collector 仅接收张量和 token 几何。进入 encoder 的 ClipBatch 会剥离标签、类别、视频路径并使用
不含类别的临时 ID；统计生成之后才按原始 clip ID join 分析标签。
每 clip 保留统计和来源，不永久保存完整 attention/token 数组。

## 模块边界与验证

- `src/vadbench/research/`：cohort、标签 join、有限只读采集及一级统计。
- `src/vadbench/token_reduction/`：布局、质量和 CSR 来源映射、identity 及实例级模型桥；不导入标签模块。
- `src/vadbench/paper/`：严格 profile/suite 解析、运行、失败回执与身份兼容声明。
- 原 manifest、视频采样、权重验证、FeatureStore、MIL、predict 和 official evaluator 保持原接口。

VideoMAEv2 几何从真实 Conv3d 输出验证展平顺序，保留每个 tubelet 的原采样帧。
空间坐标属于预处理后的网格；不宣称已建立原图像素级定位。部分 padding tubelet 排除在统计外。
P10/P11 使用 native pre-dropout attention 的有界 query 抽样与完整 key 轴，P11 明确标为抽样估计。
无 CLS 时 P13 为 not_applicable。时间变化按已知 tubelet 位置比较，跨视频解释还需匹配实际时间间隔。

```bash
python -m pytest tests/icassp2027
python -m compileall src tests
python scripts/icassp2027/check_protection.py --baseline <本轮baseline.json> --allow-changed .gitignore --allow-changed README.md --allow-changed README-CN.md
```

保护检查比较原始字节摘要，不递归读取数据、权重、环境或历史研究目录，不修改文件。
完整当前测试和真实运行回执见 [progress.md](progress.md)。
