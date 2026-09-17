# 四组 Encoder 隔离环境：迁移决策

决策日期：2026-09-03；2026-09-11 根据源码与服务器实况复核。当前操作命令集中在[服务器手册](../operations/server.md)，版本与路径以 `registry/encoder-environments-v2.yaml` 为准。

## 目标与边界

在保留原有 Python 环境、checkpoint 与 checkout 的前提下，为不兼容依赖建立四组独立环境和少量模型 overlay：classic-video-v2、foundation-video-v2、visual-vlm-v2、stream-kv-v2。

- node2 负责网络获取，node3 使用已同步资产；node2 不可达时的本地冻结/离线上传必须记录。
- 新环境从受保护种子克隆/复制，原种子环境保持只读；v2 bootstrap 与旧 `.venv` 恢复脚本分开使用。
- 25 路候选与 21 路运行 catalog 分离；人工资产可登记但缺失时阻塞，无目标 checkpoint 的路线只保留候选。
- 许可未明确的目标即使技术前向成功仍保持 blocked；不用随机或其他模型权重替代。
- 创建/下载前按 registry 的空间预算检查；当前配置要求保留 400 GiB。

## 当前落地

| 产物 | 当前作用 |
|---|---|
| `encoder-candidates.yaml` | 25 路候选、资产/许可/登记状态 |
| `encoder-integrations.yaml` | 21 路 adapter 运行元数据 |
| `encoder-environments-v2.yaml` | 4 组前缀、种子、overlay、保护根与磁盘下限 |
| `scripts/server/*_v2.py` | 环境、资产、overlay、native runner 与历史汇总 |
| `.encoder-envs/v2/` | 新运行环境与 overlay，Git 忽略 |
| `external-v2/`、`weights-v2/`、`.cache-v2/` | 新增离线资产/缓存；已有已校验权重可继续引用原目录 |

初始迁移逐路重新执行真实 checkpoint smoke，形成 14/2/5/4 的历史结果。2026-09-11 观察确认四个 Python/Torch 环境可导入、版本匹配 registry；本轮没有重建环境或重跑模型。

## 后续维护

旧迁移结果不能替代新代码的原生验证。每次执行保存新的 run 目录、code/config/video/asset/environment 身份及退出码，读取每项状态。现有汇总器优先选择历史技术成功，不会完整绑定本次身份；overlay fingerprint 也尚未覆盖文件内容。这些缺口已进入[架构审查](../reviews/2026-09-11-architecture-review.md)。

版本明细与历史证据见[环境记录](../progress/encoder-environment-v2.md)，本文件不再复制完整安装命令或逐路依赖版本。
