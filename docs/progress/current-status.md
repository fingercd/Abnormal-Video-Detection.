# 当前状态（2026-09-12）

主框架代码修复与重复职责收敛已落地。CPU真实权重及真实抽取验证通过；GPU前向复核尚未完成，完整UCF-Crime视频仍未到位。

## 代码与服务器

- 工作区：`ibnode3:/users/fotile/VAD`；本地为 `D:/PythonProject/VAD`。
- 保留分支：`qzt/refactor-vadbench-simplification`；基线commit：`0badc3435e734a841110e29d497940bfda7cc707`。
- 本次改动尚未commit/push；交付源码摘要：`7d277f87b5f86a2c1a8b72b2476f757ee01c542180e0095eb6f88c869419d87e`。运行证据同时保留各次执行时的代码摘要，不能只用旧commit代表修改后的代码。
- node3的`.venv`仍装有早期包；源码命令先设置`PYTHONPATH=/users/fotile/VAD/src`。v2 launcher和benchmark已自动选择当前源码与模型环境，受保护环境未被改写。
- UCF数据链接仍指向`/users/fotile/datasets/UCF-Crime`，当前文件数0；默认train/test manifest不存在。两份冻结的官方split/temporal TXT已同步到`data/splits/ucf_crime/`，它们不能代替视频。

## 最终验证

| 检查 | 结果与边界 |
|---|---|
| Windows全量pytest | 400 passed，9 skipped；87.37秒 |
| node3全量pytest | 408 passed，1 skipped；33.45秒 |
| 代码质量 | Windows Ruff check/format通过；双平台compileall与diff检查通过；node3未安装Ruff |
| wheel | 离线构建、临时安装、脱离源码cwd导入、内建资源和配置解析通过 |
| 真权重CPU矩阵 | 14条全部smoke_pass；VideoMAE V2、VideoMamba在最终保留代码上另行复核通过 |
| VideoMAE V2真实抽取 | 2段，每段存储特征`[1568,768] float32`，索引/数组读回通过 |
| HERMES真实抽取 | 2段，每段`[3136,896] float16`，decoder_contextual、raw/off，保存缓存事件 |
| GPU前向 | 未完成；此前尝试失败，最终启动被空卡检查拒绝 |

Windows跳过6条服务器资产检查、2条当前账户无权限创建symlink的检查和1条只适用于最小环境的检查；node3仅跳过最小环境检查。真实抽取使用已有MLVU视频，manifest中的必填标签明确为工程占位，未用于训练或评测，不能作为UCF-Crime精度证据。

## 已收敛的职责

模型配置与实际资产通过统一resolver绑定；文件SHA由一个helper计算；预测与正式评测共用coverage；smoke只使用一个schema校验/读写入口；server与benchmark共用环境选择；CLI执行阶段共用stage记录。VideoMAE包装迁入主包，旧lab模块仅重导出；没有生产消费者的JSON/stdin视频传输和隐式旧成功复用已删除。

静态catalog只使用planned/integrated/blocked；25候选、21运行项保持。带日期的运行结果与静态状态分离，详见[接入矩阵](encoder-integration-matrix.md)。

## 未完成项

1. **GPU复核：** node3为glibc 2.17，现有cuDNN 9.10 CNN子库要求`GLIBC_2.27`；调整路径或绝对路径加载均未解决。无效的库覆盖代码已移除，基础/受保护环境未修改。还需要兼容用户态环境，以及重新核对的空闲GPU；最后检查8张卡均有其他用户任务。
2. **完整UCF-Crime：** 视频和canonical manifest未就绪，因此没有全量训练、AUC/AP或缓存精度收益结论。
3. **后续需求：** 旧RTSP/独立训练原型未重写；普通metrics/cache独立schema与额外误报分桶诊断不在本轮完成范围。

实现与To Do见[实施记录](2026-09-11-implementation.md)和[计划](../plans/2026-09-11-experiment-correctness-and-consolidation.md)；原始路径、模型shape、资产身份、失败诊断和验证范围见[本轮证据](../evidence/implementation-validation-2026-09-12.json)。
