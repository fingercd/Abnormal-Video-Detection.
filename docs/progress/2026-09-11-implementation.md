# 2026-09-11 实施：可信实验与重复职责收敛

本轮基于`0badc34`和既有文档更新工作树实施。未创建commit或push；具体运行的代码摘要和服务器备份位置记录在运行产物中。

## 已完成

| 职责 | 现在的唯一入口与行为 |
|---|---|
| 模型构造与资产 | resolve_encoder_config → create_encoder_from_experiment；参数绑定实际权重，校验后构造 |
| 文件摘要 | hashing.sha256_file；特征、checkpoint、训练、smoke和worker复用 |
| 预测/评测覆盖 | engine.coverage.validate_frame_coverage；不维护两套gap/overlap规则 |
| 官方数据身份 | audit v2校验冻结协议源、官方集合/标注、manifest hash；旧v1保留历史契约 |
| smoke结果 | schema validator/read/write；拒绝缺字段、旧成功复用和非零退出成功 |
| 运行环境 | environment_registry公共resolver与env builder；server与benchmark共用 |
| 性能执行 | benchmark_runtime单case子进程，在实际模型进程中完成计时和显存测量 |
| 阶段追溯 | record_stage独立保存running/completed/failed、config/input/code身份 |
| VideoMAE实现 | 主包videomaev2_encoder唯一实现；旧lab模块仅重导出 |
| 配置/运行状态 | 静态catalog只用planned/integrated/blocked；运行成功保存在独立证据 |

删除无生产消费者的JSON/stdin视频列表worker及command/runner别名、旧成功选择分支、临时registry重复构造和重复SHA实现。新增helper用于合并真实调用者，未增加通用调度框架。

## 行为变化

- encoder.model/precision以及无消费者的encoder fingerprint别名不再静默接受；模型参数写params，checkpoint是实际registry选择。
- 两套主实验均为32段、16帧、stride2、Attention MIL/BCE。HERMES逐段复用状态，默认decoder_contextual与raw/off；连续流是单独采样协议。
- evaluate默认official，必须提供通过的v2 audit。subset保留完整覆盖，generic明确标诊断；不同run/encoder/checkpoint预测不能混用。
- 单模型和矩阵默认拒绝已有结果。汇总器使用--run-root，不再从历史中挑成功或硬编码14。
- 权重下载先校验staging再发布，未知checkout/环境不匹配失败。overlay内容hash不符时需重新核对，不能静默复用旧marker。
- wheel包含内建配置、registry、schemas与upstream locks；已在无源码cwd验证资源与配置解析。

## 验证与限制

最终Windows为400 passed/9 skipped，node3为408 passed/1 skipped；本地Ruff、双平台compileall及离线wheel安装/资源检查通过。14条CPU真权重smoke全部通过；两个参考encoder各完成实际视频2段抽取。

VideoMAE V2存储特征为`[1568,768] float32`，HERMES为`[3136,896] float16`。HERMES CPU的这次16帧×2段抽取耗时约3.2小时，仅为工程验证，不据此比较GPU吞吐。实际输入为MLVU视频，标签是明确标注的工程占位，未做训练或精度评测。

GPU验证仍未完成：cuDNN子库缺少主机GLIBC_2.27支持，最终8卡均被其他任务占用。无效的library override尝试已从代码和registry移除；失败诊断保留。详见[当前状态](current-status.md)与[机器可读证据](../evidence/implementation-validation-2026-09-12.json)。

完整UCF-Crime视频仍未就绪，因此本轮不声称完成全量异常检测精度。旧RTSP/独立训练原型不重写；主框架对旧包的反向依赖已经解除。普通metrics/cache事件独立schema、正常视频误报分桶等属于后续诊断工作，现有结果不把它们写成已完成。

## To Do核对

- [x] 核心修复、职责收敛和文档对齐。
- [x] 双平台测试、编译、资源打包检查。
- [x] 14路CPU真实冒烟及两路真实抽取。
- [ ] GPU前向复核：受兼容运行库和空闲GPU限制，未标为成功。
- [x] 最终文档已同步，源码和文档哈希核对通过。
