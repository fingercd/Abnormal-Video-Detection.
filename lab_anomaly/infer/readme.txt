当前推理只适用于 train_end2end.py 输出的端到端 checkpoint。

独立服务：
  python -m lab_anomaly.infer.rtsp_service --config <yaml> --video <file> --known_checkpoint <checkpoint_best.pt>

它先处理本地视频或 RTSP，再以单个 clip embedding 做 MIL 打分；写出的 events.jsonl 是报警记录，不是逐帧预测。
open_set_dir、KMeans、OCSVM、双流和 frame_stride 都不是当前有效推理功能。

known_event_runtime.py 是供外部 Python 程序调用的后台多流类，本仓库没有 web/services 或 moniter 集成。
详细字段、报警副作用与配置限制见 README.md。
