这是保留的 VideoMAE v2 + MIL 本地原型说明入口。

旧的 embedding 提取、KMeans、OCSVM、伪标签和“已知异常分类器”脚本已不在当前目录；请不要按旧命令运行。

当前唯一训练入口是：
  python -m lab_anomaly.train.train_end2end

训练前需要：
  1. python -m lab_anomaly.data.index_build
  2. python -m lab_anomaly.tool.precompute_clips
  3. 保证预切片参数与 configs/train_end2end.yaml 一致。

完整且以代码为准的说明见 README-CN.md。当前正式 UCF-Crime/VADBench 实验不从本目录进入。
