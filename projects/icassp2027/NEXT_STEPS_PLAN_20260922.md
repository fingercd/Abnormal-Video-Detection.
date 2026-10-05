# 下一步操作计划（ICASSP 2027 压缩矩阵收尾）

**制定时间**：2026-09-22 14:20 CST
**适用窗口**：从现在到六格矩阵装配完成并进入分数门
**原则**：全自动推进 + 人工只在"关口"核验；任何一步失败先留证据再处置，不覆盖旧目录；正式分数在矩阵装配并通过审计前一律不读。

---

## 阶段 0：现在（等待期，无需操作）

- XD 九个提取继续跑（当前 94.3%，预计 60–90 分钟出合同）。
- 预测/导出调度器（PID 13601）待命，合同一出现即自动启动对应预测。
- 本阶段唯一建议动作：**决定是否补开 GPU monitor**（见阶段 6）。当前 monitor 已于 09:00 退出，显存峰值只覆盖 08:57–09:00。
- 可选：趁等待复核 node2 的 8 路补充提取（本轮 SSH 超时未复核）。

## 阶段 1：每出现一份 XD extraction 合同（预计 10–30 分钟内开始）

对每个新合同执行一次核验（一条 SSH 即可）：

```bash
ssh -p 12345 fotile@121.196.228.153 "B=/data2/localdisk/fotile-icassp2027-kimi-20260919-a01; \
for e in videomaev2 videomae timesformer; do for r in \$(ls \$B/compressed-test/\$e/xd); do \
P=\$B/compressed-test/\$e/xd/\$r; [ -f \$P/extraction-contract.json ] && python3 -c \"import json;d=json.load(open('\$P/extraction-contract.json'));print('\$e/\$r','videos=',d.get('videos'),'clips=',d.get('clips'),'status=',d.get('status'))\"; done; done"
```

通过条件：`videos=800`、`status=ready`、clips 与 dense 视图同量级。
**逐视频闭合核对**（目录口径之上的硬核验）：对每个已出合同的 run，比较每视频目录 clip 数与 dense 目标（`merged/<enc>/xd-dense-test-view/merge-contract.json` 的 `video_provenance[*].record_count`），应全部相等、无 partial/extra。可用 `work/` 下既有核对脚本模式，或让我代跑。
预期：调度器至多在 45 秒后自动启动该 run 的预测（4 并发上限内）。

## 阶段 2：XD 预测启动后 5 分钟内（过闸确认）

- 检查新评分日志无 compat 报错：
  `grep -l 'ValueError: direct_insert' /users/fotile/icassp2027-runs/codex-takeover-20260920/logs/urdmu-official-matrix/score-xd_violence-*.log`（应无输出）。
- 日志字节数应持续增长；`urdmu-official-predictions-r08/xd_violence/<enc>/<method>/` 下出现 `provenance/` 与 `query-chunk-receipts.json`。
- **若再现 direct_insert 报错**：说明放行表漏了某一对。处置：取该 head 与压缩 run 的完整 backbone 指纹 → 用仓库 `_verified_code_digest` 复算其 dense 训练视图全部源运行 → 确认仅差包装文件 LF/CRLF → 把该对加入 `compatibility.py` 放行表 → 同步远端快照 → 重启调度器（attempts 随之重置）。**未证明前不得加表。**

## 阶段 3：XD 预测完成确认

每个 run 满足：`result.json` 中 `status=completed`、`official_frame_scores_read=false`、`predictions.jsonl` 存在且非空。共 9 个。

## 阶段 4：XD 九份质量报告

- 位置：`urdmu-quality-exports-r08/xd_violence/<enc>/<method>.json`，`metric.status=available`。
- 串行导出、每份预估 5–15 分钟，总计 1.5–2.5 小时；**调度器在此期间不响应新轮询属正常**，不要重启、不要误判卡死。
- 只确认存在与 status，不读取分数值。

## 阶段 5：六格矩阵装配（fail-closed 门）

前置：18/18 压缩报告 `available` 且覆盖 290/800。
```bash
# 远端：python scripts/icassp2027/assemble_urdmu_matrix.py（待 exports 齐全后运行）
```
该脚本逐格核对 dense view、prediction、compressed extraction、head checkpoint/training QA、method contract 与 freeze receipt 的 SHA，并流式汇总每个压缩 FeatureStore 的实际 native/retained token、ratio、clip 数与插件开销；缺证据即 fail-closed。产物 = 正式六格矩阵 JSON。

## 阶段 6：收尾审计（矩阵就绪后）

1. **GPU/效率**：若阶段 0 补开了 monitor，extraction 与预测全部结束后运行 `summarize_urdmu_gpu_monitor.py`，按 PID/GPU 汇总峰值/均值/采样区间；报告须明确"峰值只覆盖 monitor 启动后的观测窗口"。若未补开，在审计文档中显式声明该缺口。
2. **node2 复核**：8 路补充提取的最终状态与覆盖（网络恢复后）。
3. **文档**：把 §5 结果写回 `projects/icassp2027/progress.md` 并刷新 `AGENTS.md`/`CLAUDE.md` 快照。
4. **（可选）治理**：把远端快照 `code-kimi-adgs-20260921` 的包装文件统一为 LF，防止未来再从该快照录入 CRLF 形态指纹。**必须在全部提取结束后做**，并重新编译校验。

## 阶段 7：论文数字门（唯一允许读分数的时点）

矩阵装配通过 + provenance/coverage 审计通过后，才可读取 18 份报告的分数进行方法选择；此前任何分数读取都视为违规。读取后按计划生成 paired bootstrap CI 汇总与显存/延迟/吞吐对比表。

---

## 附 A：故障应对手册

| 故障 | 判据 | 处置 |
|---|---|---|
| 兼容校验再现 | 新 `score-*.log` 出现 `ValueError: direct_insert` | 按阶段 2 流程证明并加表 + 重启调度器；未证明则保持失败，留证据 |
| 某预测 rc≠0 且 attempts 已用尽 | state.json 中 `status=running` 但进程不存在 | 修复后重启调度器（attempts 重置）或按同样命令手动补跑 |
| 调度器无响应 >10 分钟 | 无 score/export 进程、state 不更新 | 查是否阻塞在导出子进程（查看 `export-*.log` 是否在长）；导出中属正常，否则重启 |
| 提取进程消失且无合同 | 进程数 <9 且对应 run 无 contract | 保留现场，查该 run 日志；用新 run-id 重试，不覆盖旧目录 |
| 磁盘紧张 | `df -h /data2/localdisk` <10% free | 当前 4.8T free，风险低；若紧张先清理历史保留副本 |

## 附 B：本阶段成功判据（DoD）

- [ ] XD 九份 extraction 合同齐备且逐视频闭合核对通过
- [ ] XD 九个压缩预测 completed，且全程无 compat 报错
- [ ] XD 九份质量报告 available
- [ ] `assemble_urdmu_matrix.py` 产出 18/18 六格矩阵
- [ ] monitor 汇总或显式声明显存观测窗口缺口
- [ ] progress.md 与 AGENTS/CLAUDE 快照刷新
- [ ] 全程无测试分数读取、无目录覆盖、无伪造指纹
