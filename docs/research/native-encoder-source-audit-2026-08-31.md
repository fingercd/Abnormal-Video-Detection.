# 25 路编码器原生来源审计（2026-08-31）

## 文档边界

本审计的外部来源核验截止于 2026-08-31，保留当日上游仓库、模型入口、许可证与资产风险判断；后文的“代码同步”只对照 2026-09-11 仓库和既有服务器产物，没有重新联网判断上游是否变化。

每条路线有四个彼此独立的结论：存在一手来源、具有可下载 checkpoint、进入本项目运行 catalog、在某次真实视频 forward 中通过。前一项不能推出后一项。catalog 只记录静态 `planned`、`integrated`、`blocked` 登记状态；`smoke_pass`、失败和许可阻塞属于带日期的运行证据。当前分类的机器可读入口是 [`encoder-candidates.yaml`](../../registry/encoder-candidates.yaml)，逐项历史证据见 [`encoder-integration-matrix.md`](../progress/encoder-integration-matrix.md)，运行时关系见 [`encoder-runtime.md`](../architecture/encoder-runtime.md)。

## 来源、目标资产与当前落点

| ID | 截止日的一手路线 | 固定的目标与导出位置 | 2026-09-11 静态登记 / 主要门禁 |
|---|---|---|---|
| `r2plus1d_18` | [TorchVision](https://github.com/pytorch/vision) | K400 R(2+1)D-18；classifier 前 pooled | 运行 catalog；本地权重 `verified` |
| `x3d` | [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) | X3D-S K400；classifier 前 pooled | 运行 catalog；本地权重 `verified` |
| `mvitv2` | [TorchVision](https://github.com/pytorch/vision) | MViTv2-S K400；classifier 前 pooled | 运行 catalog；本地权重 `verified` |
| `slowfast` | [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) | SlowFast-R50 8x8 K400；双路径 classifier 前 pooled | 运行 catalog；本地权重 `verified`，不能误记成 PySlowFast 4x16 `.pkl` |
| `c3d` | [facebookarchive/C3D](https://github.com/facebookarchive/C3D) | Sports-1M Caffe；`fc6` | 运行 catalog 但 `blocked`；官方权重人工获取，服务器还缺 Caffe |
| `i3d` | [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo) | I3D-R50 K400 8x8；classifier 前 pooled | 运行 catalog；本地权重 `verified` |
| `timesformer` | [TimeSformer](https://github.com/facebookresearch/TimeSformer)、[HF 模型](https://huggingface.co/facebook/timesformer-base-finetuned-k400) | Base K400；`last_hidden_state` | 运行 catalog；固定 HF revision 与本地摘要，CC-BY-NC-4.0 |
| `video_swin` | [TorchVision](https://github.com/pytorch/vision) | Swin3D-T K400；backbone tokens | 运行 catalog；本地权重 `verified` |
| `videomae` | [VideoMAE](https://github.com/MCG-NJU/VideoMAE)、[HF 模型](https://huggingface.co/MCG-NJU/videomae-base) | Base 预训练 encoder；`last_hidden_state` | 运行 catalog；固定 HF revision，CC-BY-NC-4.0 |
| `videomaev2` | [VideoMAE V2](https://github.com/OpenGVLab/VideoMAEv2)、[HF 模型](https://huggingface.co/OpenGVLab/VideoMAEv2-Base) | Base encoder；observed backbone tokens/pooled | 运行 catalog；代码 MIT、权重 CC-BY-NC-4.0 分别登记 |
| `uniformerv2` | [UniFormerV2](https://github.com/OpenGVLab/UniFormerV2) | B/16 K400 backbone/pooled | `candidate_only`；目标 model-zoo 链接返回 404，没有可校验权重 |
| `umt` | [unmasked teacher](https://github.com/OpenGVLab/unmasked_teacher) | UMT-B backbone tokens | `candidate_only`；官方 Issue 记录权重链接失效 |
| `internvideo2` | [InternVideo](https://github.com/OpenGVLab/InternVideo) | Stage2 1B 224p f4 vision encoder | 运行 catalog 但 `blocked`；gated 权重 403，checkout 和资产待人工准备 |
| `videomamba` | [VideoMamba](https://github.com/OpenGVLab/VideoMamba) | Tiny t16 K400；backbone tokens | 运行 catalog；官方 checkpoint 已固定；SSM 内部状态不是跨 clip state |
| `vjepa2` | [V-JEPA 2](https://github.com/facebookresearch/vjepa2)、[HF 模型](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | ViT-L fpc64 256；vision features | 运行 catalog；固定 HF revision 与本地摘要 |
| `longvu` | [LongVU](https://github.com/Vision-CAIR/LongVU)、[HF 模型](https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B) | Qwen2 7B 的 SigLIP/DINO/SVA 视觉路径；`projected_visual` | 运行 catalog；当前 adapter 不把文本生成或常规 decoder cache 当作其输出 |
| `videochat` | [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything) | VideoChat-7B projector visual | 运行 catalog 但 `blocked`；登记目标不可访问，不能用 VideoChat2 代替 |
| `videochat_online` | [VideoChat-Online](https://github.com/MCG-NJU/VideoChat-Online)、[HF 模型](https://huggingface.co/MCG-NJU/VideoChatOnline-4B) | 4B ViT feature + Pyramid Memory Bank；`visual_memory` | 运行 catalog 但许可阻塞；仓库根目录没有明确 LICENSE，不能写成 decoder KV |
| `videochat_flash` | [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash)、[HF 模型](https://huggingface.co/OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448) | 2B res448 视觉塔/projector；`projected_visual` | 运行 catalog；只加载视觉塔与 projector，关闭 LLM 压缩 |
| `ma_lmm` | [MA-LMM](https://github.com/boheumd/MA-LMM) | saved model + LAVIS/InstructBLIP/Vicuna；Q-Former `visual_memory` | 运行 catalog 但 `blocked`；Google Drive 资产和多依赖链待人工准备 |
| `moviechat` | [MovieChat](https://github.com/wenhaochai/MovieChat) | MovieChat-vicuna；short/long visual memory | 运行 catalog 但 `blocked`；目标 checkpoint、base model 与多许可证链未闭合 |
| `streaming_vlm` | [StreamingVLM](https://github.com/mit-han-lab/streaming-vlm)、[HF 模型](https://huggingface.co/mit-han-lab/StreamingVLM) | 8B 四分片；decoder-context features 与显式 decoder KV state | 运行 catalog 但许可阻塞；技术路线不能因权重 `verified` 自动升级 |
| `infinipot_v` | [InfiniPot-V](https://github.com/aiha-lab/InfiniPot-V) | Qwen2/2.5-VL continual KV compression | `candidate_only`；缺明确 LICENSE 和可校验目标 checkpoint |
| `hermes_llava_ov` | [HERMES](https://github.com/haowei-freesky/HERMES)、[LLaVA-OV 0.5B](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) | projected visual 或 decoder-context features；显式 decoder KV | 运行 catalog；最小官方基座、代码和权重分别固定 |
| `mukv` | [MuKV](https://github.com/IMBALDY/MuKV) | LLaVA-OneVision 0.5B base + multi-grained decoder KV | `candidate_only`；缺明确 LICENSE 和 MuKV 专用增量资产 |

精确 repo commit、模型 revision、文件列表和 SHA256 不在本文重复，分别以 `integrations/<id>/upstream.lock.yaml` 与 [`registry/checkpoints.yaml`](../../registry/checkpoints.yaml) 为准。

## 需要保持的语义结论

- 固定 clip 模型即使内部使用 Transformer 或 SSM，也没有本项目定义的跨 chunk state。VideoMAE/VideoMAE V2 的预训练重建 decoder 也不是语言模型 decoder KV。
- LongVU 与 VideoChat-Flash 当前接入只导出 projected visual；其论文中的视觉 token 压缩不能自动表述为本项目已经提供 streaming token cache。
- VideoChat-Online、MA-LMM、MovieChat 管理视觉特征或 Q-Former memory，属于 `visual_memory`。
- HERMES 与 StreamingVLM 明确涉及语言模型 `decoder_kv`。HERMES constructor 默认可导出 `projected_visual` 用于性能对照；当前 32 段 `weak_mil` 配置显式选择 `decoder_contextual` 且 `native_compression_mode=off`，使 raw 对照的当前 chunk 表征受历史 decoder KV 条件化。
- `verified` checkpoint 只表示 registry 中已有固定摘要及相应资产曾被核验；它不能覆盖未闭合的代码许可证，也不能证明今天的环境和 forward 仍通过。

## 已有服务器证据与当前复核

2026-09-03 的隔离环境运行得到 14 个 `smoke_pass`；VideoChat-Online 与 StreamingVLM 的两 chunk 技术前向通过但保持 `blocked_license`；5 路缺人工资产；4 路未注册。它们是历史运行证据，不能回写为 catalog 状态。2026-09-11 仍能读取两份对应矩阵且摘要匹配，四组解释器可导入；本轮没有重新运行模型 forward。

2026-09-11 的 node3 全量测试重新计算并匹配 8 条权重：R(2+1)D、MViTv2、Video Swin、I3D、X3D、SlowFast、VideoMAE V2、HERMES。其余登记权重没有在本轮全部重新哈希。服务器复核证据见 [`server-doc-audit-2026-09-11.json`](../evidence/server-doc-audit-2026-09-11.json)。

## 解除阻塞的规则

只有目标自己的合法官方资产到位、revision/文件/摘要/许可均登记、目标环境可复现，并重新完成真实视频 forward 后，才能更新该路线的运行证据；catalog 静态状态则只反映登记和可运行性。不得用同系列另一模型、随机权重、仅配置加载、mock 或旧产物复制来替代。实现与后续验证汇总见[2026-09-11 实施记录](../progress/2026-09-11-implementation.md)。
