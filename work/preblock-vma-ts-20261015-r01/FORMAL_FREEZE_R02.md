# VideoMAE / TimeSformer pre-block frozen test matrix — 2026-10-15

Only VideoMAE and TimeSformer; never rerun VideoMAEv2 or CLIP. UCF290/XD800 each.
76 conditions: four fresh dense references plus72 compressed full dataset extraction/prediction tasks.
Three methods attention_topk / visionzip_internal / uniform, pre-block6/9/12, keep0.5/0.3.
Signals from the previous complete block5/8/11 in the same forward; never use future-layer tensors or truth.
VideoMAE: native16frames/stride2/window16, patch-only1568->784/470, native mean pooling.
TimeSformer: native8frames/stride2/window8, same spatial trajectories across all frames,196->98/59,1569->785/473tokens.
TimeSformer suffix Sx1 is logical slot packing only; absolute source positions untouched, temporal/spatial attention operations native.
Frozen formal independent dataset heads seed0/3000steps/64normal+64anomaly; no head refit.
FP32, TF32off, cuDNN disabled consistently, original native attention backend; B8 CPU workers2/prefetch2, no frame/pixel dedup.
UCF frameROC-AUC primary; XD trapezoid PR-AUC primary, step AP supplementary.72 paired video-stratified10000draw CI seed0, NI margin0.005.
All metrics only after matching dense and method predictions sealed. Preview once and reuse; do not overwrite old quality.json.
All tasks online SwanLab before compute; inspect original PID on transport failure, never blindly replay GPU.
Full encoder228 groups +full pipeline228 groups on exclusive GPU; no speed claims from shared extraction throughput.
Prior test exposure exists; do not claim whole research test-blind. No score-based method/budget/position/weight changes.
Native gates and CPU pipeline gates are training-side engineering evidence, not anomaly-specific property claims.

Execution correction before any official feature exists: use the original classic-video-v2 runtime (torch2.3.0+cu121, transformers4.37.2, timm0.9.12) and byte-identical original adapter from code-e30c3ab-s6p. No changes to shared environment or compatibility whitelist. VMA UCF explicit processor_tensor_type=np; VMA XD original constructor omits this key (effective default pt); TS both explicit np. Each reproduces the corresponding original head backbone digest exactly. The earlier four source-r01 preflight failures are preserved, not overwritten or counted. New data output root ends in -r02.
Classic native/pipeline gates succeeded online: VideoMAE k6dxrcxk and TimeSformer m2y5f5uo. The VideoMAE receipt directory actually has classic-classic prefix due a launcher naming substitution; preserve the original path, do not rename or reinterpret it.
