# 四系统文件重新生成说明

上一条回复的成品路径在当前环境不存在，本次从已上传 r02 源包重新建立四系统稿，并用新的英文文件名交付。

- 32 行质量记录包含四系统×两数据集×Dense及三预算；UCF AUC/AP 均有数据。
- XD 梯形 PR-AUC 与 step AP 分开，DSANet 压缩对照为同一 raw CLIP Dense。
- 表三跨双栏，展示四个系统共 16 行编码器结果；原始 20 条编码器记录和 8 格完整流程数据仍保留在导出数据中，完整流程结果由正文单独说明。
- 原始快照 62 份；数值来源记录 340 条；表格自动生成。
- PDF 共5页，技术内容结束于第4页；第5页为声明与参考文献。主图仍为双栏。
- 17项已引用参考文献均解析，无超栏/未定义引用。字体嵌入，无Type3。

这是一份源包重建的作者工作稿，不是对之前未保存成品的逐字节恢复。原图文字及作者声明仍需按 `notes/CLAIM_EVIDENCE_MAP.md` 确认。

## 文件

`main.tex` 是主文档，`main.pdf` 是已编译文件；运行 `bash scripts/build.sh` 可以从证据快照重新导出数据与编译。

历史 `data/` 源文件仍保留，但当前表格只由 `scripts/export_tables.py` 读取 `evidence_snapshot/` 生成，输出 `data/four_systems_results.json`、`data/quality_ladder_summary.csv` 和 `data/numeric_provenance.csv`。

## 2026-09-24 按用户提供的问答修订

摘要、token 表、实验协议、编码器效率和结论采用附件中的替换稿。主文删除 DSANet 完整视频测速讨论，原始证据保留。

### 缓存复现说明（从实验协议段移入）

The 20%/60% VideoMAE-family caches include accepted tensor-construction/runtime differences from the original caches; numerical equivalence was not established.

此验证状态未改变，正文不声称所有预算的实现细节完全一致或已验证数值等价。
