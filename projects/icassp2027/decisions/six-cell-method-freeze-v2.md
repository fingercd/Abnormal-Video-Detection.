# 六格方法冻结合同 v2

本合同在正式 test 分数读取前冻结六格压缩比较的方法范围。它以六格 scope contract 为范围锚点，
不复用旧的 derived1598 XD scope，也不把旧 TopKMIL controller evaluator 当作当前 UR-DMU 结果。

训练侧选择登记的规则是：在训练角色中确定中间 verified native block、`keep_ratio=0.60` 和
基于当前 token/trajectory 的 max-member-L2 pair-select。正式比较同时运行 dense identity、同预算
`group_uniform` 和同预算 seed0 `group_random`。VideoMAEv2/VideoMAE 按相邻 native pair member
选择，TimeSformer 按完整 spatial trajectory 选择；TimeSformer 的 geometry snap 和实际 retained
ratio 必须由每次 receipt 写出，不能用目标 0.60 代替。

主 head 模式为 `direct_insert`，`refit_head` 只有在同一完整训练集、同一预算和独立 QA 都完成时才作为
补充列。测试只在本合同、六格完整 dense head、UCF sealed test audit、XD raw-coordinate seal 和
同输入覆盖检查全部通过后执行。测试结果不能回改 selector、层、预算、阈值、seed 或 checkpoint。

质量导出分别使用 UCF frame ROC-AUC、XD 作者梯形 PR-AUC 和另列的 step average precision，所有方法
相对 dense 的差值都按视频分层配对 bootstrap 10,000 次并报告 95% CI。CI 下界低于 `-0.005` 时不能
写成质量非劣；效率必须同时计入 reducer 和 head 的真实开销。
