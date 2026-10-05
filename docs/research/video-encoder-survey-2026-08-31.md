# 视频编码器与长视频缓存路线调研（2026-08-31）

## 结论先行

首批真实权重链路选择 **VideoMAE V2 Base + HERMES/LLaVA-OneVision-Qwen2-0.5B**。前者提供成熟、固定 clip、无跨 clip 缓存的 ViT 对照；后者是当前候选中能在最小官方支持基座上直接操作 **语言模型 decoder KV cache**、且无需为缓存策略额外训练（training-free）的流式方案。两者共同覆盖本项目最重要的实验轴：独立 clip 表征与有状态流式缓存。

需要纠正一个术语：第二类候选大多不是“另一个 video encoder”，而是“视觉编码器 + projector/Q-Former + 因果语言模型 + 记忆/缓存策略”的视频 VLM 系统。LongVU 压缩视觉 token，MA-LMM/MovieChat 管理视觉记忆，HERMES/InfiniPot-V/MuKV 才直接处理或存取 decoder KV；这三者不能混写成同一种 cache。

本报告采用一手来源：论文正式页/arXiv、作者项目页、官方仓库与官方模型卡。研究结论仅说明架构与可行性，不把 VQA 论文上的准确率外推为 UCF-Crime 分类效果。

本报告的外部检索截止日仍是 2026-08-31。2026-09-11 只把仓内实现状态对齐到 commit `0badc3435e734a841110e29d497940bfda7cc707`，没有重新检索论文或模型卡。当前 25 路候选中有 21 路进入运行 catalog；catalog 的静态登记状态为 `planned`、`integrated` 或 `blocked`，不保存运行 `smoke_pass`。2026-09-03 的历史服务器证据仍是 14 路真权重 smoke 通过、2 路技术通过但许可阻塞、5 路等待人工资产、4 路仅保留候选。逐路状态见[接入矩阵](../progress/encoder-integration-matrix.md)，实现与后续验证汇总见[2026-09-11 实施记录](../progress/2026-09-11-implementation.md)。

## 1. 三种“压缩/缓存”必须分开

设视觉编码器输出为 `V`，projector/Q-Former 输出给语言模型的视觉 token 为 `Z`，因果语言模型第 `l` 层自注意力缓存为 `(K_l, V_l)`：

| 层级 | 被保存或压缩的对象 | 发生阶段 | 主要收益 | 不能据此宣称什么 |
|---|---|---|---|---|
| `vision_tokens` | 帧、patch token、`V` 或 `Z` | LLM prefill 之前 | 降低视觉编码/投影后的输入长度和 prefill 计算 | 不能宣称已有 decoder KV cache 复用 |
| `visual_memory` | 跨帧/跨 chunk 的视觉特征或记忆 token | 视觉编码与 LLM 之间 | 保留长时视觉上下文，控制进入 LLM 的 token 数 | 即使它在 Q-Former cross-attention 中充当 key/value，也不是 decoder 自注意力 KV |
| `decoder_kv` | 因果 LLM 每层历史 token 的 key/value | prefill/逐 token 解码期间 | 避免重复前缀计算，支持流式追加、淘汰、合并或检索 | 不代表视觉 tower 本身支持缓存；也不等同于“缓存原始 token” |

LongVU 论文明确把帧删除、文本引导的帧内降采样和跨帧空间 token 剪枝放在视觉 token 阶段；MA-LMM 的 memory bank 是 Q-Former 的 key/value；InfiniPot-V 则把 input-vision compression 与 KV-cache compression 分成不同阶段讨论。参见 [LongVU 论文](https://proceedings.mlr.press/v267/shen25j.html)、[MA-LMM 论文](https://arxiv.org/abs/2404.05726) 与 [InfiniPot-V NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/caef5f5e658aa1f7565f063a2cd99726-Paper-Conference.pdf)。

### VideoMAE V2 为什么没有本项目所说的 KV cache

VideoMAE V2 的主体是对一个给定视频 clip 做双向自注意力的 ViT encoder；其“dual masking”是预训练时分别减少 encoder 和重建 decoder 所处理的 token。这里的 decoder 是像素重建模块，不是自回归语言模型 decoder，也没有可跨 clip 传入的 `past_key_values`。[论文](https://arxiv.org/abs/2303.16727) 和 [官方实现](https://github.com/OpenGVLab/VideoMAEv2) 都把它定义为 masked autoencoder/下游视频表征学习器。

因此，本项目把 VideoMAE V2 定义为：`supports_streaming=false`、`cache_kind=none`。相邻 clip 即使重叠，仍各自完整前向；若未来人为复用中间 token，那是新增算法，不能算上游原生能力。

## 2. 固定 clip 视频编码器候选

下表中的“无原生跨 clip cache”不是缺点判断，而是对公开接口的准确描述。输入长度通常由配置决定。当前 smoke profile 为各目标固定的原生近似输入，例如 X3D 13 帧、I3D/TimeSformer 8 帧、SlowFast/Video Swin 32 帧，不能把 smoke 结果当作统一采样的模型比较；正式 UCF 编码器对照必须另行固定可比较的采样协议，或把输入差异列为显式变量。

| 编码器 | 主干/表征 | 官方代码与权重状态 | 原生跨 clip 状态 | 对 UCF-Crime 的定位 |
|---|---|---|---|---|
| C3D | 3D ConvNet，clip 级向量 | [ICCV 2015 论文](https://openaccess.thecvf.com/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)，[官方旧版代码](https://github.com/facebookarchive/C3D) | 无；旧 Caffe 栈 | 历史复现参考；不作为新框架首批依赖 |
| I3D | 2D 卷积膨胀为 3D，RGB/光流双流可选 | [CVPR 2017 论文](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)，[DeepMind 官方仓库](https://github.com/google-deepmind/kinetics-i3d) | 无 | UCF-Crime 文献常见特征基线；后续应补作可比锚点 |
| R(2+1)D | 将 3D 卷积分解为空间 2D + 时间 1D | [CVPR 2018 论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)，[TorchVision 权重接口](https://pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html) | 无 | 依赖轻、容易接入；适合验证注册与公平性，不是首选上限模型 |
| SlowFast | 慢速语义路径 + 快速运动路径 | [ICCV 2019 论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)，[官方 PySlowFast](https://github.com/facebookresearch/SlowFast) | 无 | 强经典时空基线；预处理与双路径输入需专用 adapter |
| X3D | 沿时空、宽度、深度渐进扩张的高效 3D CNN | [CVPR 2020 论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.html)，[PyTorchVideo model zoo](https://pytorchvideo.readthedocs.io/en/latest/model_zoo.html) | 无 | 小模型吞吐基线，适合速度—AUC Pareto 前沿 |
| TimeSformer | 空间/时间分解注意力的纯 Transformer | [ICML 2021 论文](https://proceedings.mlr.press/v139/bertasius21a.html)，[官方仓库](https://github.com/facebookresearch/TimeSformer) | 无 | 纯 ViT 历史对照；依赖和上游维护老于 VideoMAE V2 |
| Video Swin | 3D shifted-window 层次 Transformer | [CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html)，[官方仓库](https://github.com/SwinTransformer/Video-Swin-Transformer) | 无 | 局部窗口计算稳健，列为第二批高优先级 |
| MViTv2 | 多尺度层次 Transformer、pooling attention | [CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.html)，[SlowFast 官方代码库](https://github.com/facebookresearch/SlowFast) | 无 | 多尺度对监控小目标可能有价值；需固定具体视频 checkpoint 后再接入 |
| VideoMAE | 高遮罩率视频 MAE + ViT encoder | [NeurIPS 2022 论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html)，[官方仓库](https://github.com/MCG-NJU/VideoMAE) | 无 | 已成熟的自监督基线；由 V2 作为首批代表 |
| VideoMAE V2 | 双遮罩预训练、ViT-B/L/H/g | [CVPR 2023 论文](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html)，[官方仓库](https://github.com/OpenGVLab/VideoMAEv2)，[Base 模型卡](https://huggingface.co/OpenGVLab/VideoMAEv2-Base) | 无 | **首批固定 clip 真实权重基线**；pinned HF forward 默认池化为 `[B,D]`，本项目仅在 private hook 成功时额外观察 token 序列 |
| UniFormerV2 | 给 image ViT 增加局部时序模块与全局多尺度融合 | [ICCV 2023 论文](https://arxiv.org/abs/2211.09552)，[官方仓库](https://github.com/OpenGVLab/UniFormerV2) | 无 | 第二批；可检验 image-pretrain 与原生 video-pretrain 的迁移差异 |
| UMT | 以 unmasked teacher 蒸馏视觉语义的视频基础模型 | [ICCV 2023 论文](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Unmasked_Teacher_Towards_Training-Efficient_Video_Foundation_Models_ICCV_2023_paper.html)，[官方仓库](https://github.com/OpenGVLab/unmasked_teacher) | 无 | 多模态预训练候选；部署栈重于 VideoMAE V2 |
| InternVideo2 | 统一 masked video token 重建、跨模态对比与 next-token 任务 | [ECCV 2024 论文](https://arxiv.org/abs/2403.15377)，[官方 InternVideo 仓库](https://github.com/OpenGVLab/InternVideo) | 分类 encoder 路径无；VLM decoder 另论 | 高上限后续候选；必须固定子模型，不能把整个系列当一个 encoder |
| VideoMamba | 状态空间模型用于视频理解 | [ECCV 2024 论文](https://arxiv.org/abs/2403.06977)，[官方仓库](https://github.com/OpenGVLab/VideoMamba) | 公开分类接口未承诺跨 clip state | 线性序列建模值得比较，但“内部是 SSM”不自动等于可复用流状态 |
| V-JEPA 2 | 自监督视频 world model/预测表征 | [论文](https://arxiv.org/abs/2506.09985)，[Meta 官方代码与模型](https://github.com/facebookresearch/vjepa2) | 官方 clip encoder 接口无本项目式 cache | 新一代表征候选；先完成基线后再评估许可证、输入成本与领域迁移 |

### 调研优先级在当前代码中的落点

1. **VideoMAE V2 Base** 已作为固定 clip 主基线进入 catalog。唯一实现位于主包 `vadbench.integrations.videomaev2_encoder`，旧 `lab_anomaly` 模块只重导出；adapter 从 pinned backbone hook 选择最丰富的兼容 token 序列，hook 不可用时回退为 `[B,1,D]`，两者均记录 `sequence_source`。
2. **I3D、X3D、Video Swin** 已进入 catalog 并有历史 smoke PASS；MViTv2、SlowFast、R(2+1)D、TimeSformer、VideoMAE 也已形成固定 clip 对照组。
3. **VideoMamba、V-JEPA 2** 已进入 catalog 并有历史 smoke PASS；InternVideo2 因 gated 权重和 checkout 等待人工资产。
4. **UniFormerV2、UMT** 因没有可校验目标权重只保留在候选 registry，当前不是可运行 adapter。

## 3. 长视频/VLM 压缩与缓存候选

| 系统 | 管理层级 | 在线/离线与核心机制 | 是否真正 decoder KV | 开源与本项目判断 |
|---|---|---|---|---|
| LongVU | `vision_tokens` | 离线看到整段视频；DINOv2 去冗余帧，文本 query 决定保留高分辨率帧，再按跨帧相似度剪空间 token | 否；生成阶段虽可使用常规 `use_cache`，论文贡献不在 KV 管理 | [ICML 2025 论文](https://proceedings.mlr.press/v267/shen25j.html)、[官方代码/权重](https://github.com/Vision-CAIR/LongVU)。3B/7B、本地 demo 文档给出较高显存门槛；列为后续 token 压缩对照 |
| VideoChat（2023） | 固定视觉 token + learnable interface | 视频 foundation model 与 LLM 的端到端短视频接口，无持续长时状态 | 否 | [论文](https://arxiv.org/abs/2305.06355)、[官方 Ask-Anything](https://github.com/OpenGVLab/Ask-Anything)。作为 VLM 编码接口历史基线，不作为缓存基线 |
| VideoChat-Online | `visual_memory` | 在线视频的 Pyramid Memory Bank，以多尺度时序记忆保留过去信息；需要 offline-to-online instruction tuning | 否 | [CVPR 2025 论文](https://arxiv.org/abs/2501.00584)、[项目页](https://videochat-online.github.io/)。比原始 VideoChat 更贴近本项目，但训练与数据工程更重，列为后续 |
| VideoChat-Flash | `vision_tokens` + LLM 内视觉上下文 | HiCo 先做 clip 级时空压缩，再在 LLM 不同深度做 progressive visual dropout，并配套 short-to-long 训练 | 不提供本项目式可复用流式 decoder KV；不能只归为纯 encoder 压缩 | [论文](https://arxiv.org/abs/2501.00574)、[官方仓库](https://github.com/OpenGVLab/VideoChat-Flash)。适合作为分层视觉上下文压缩，不与 HERMES 的有状态 KV 直接横比 |
| MA-LMM | `visual_memory` | 在线逐帧写入 memory bank；相邻相似特征选择并平均，使 memory 长度受控；memory 作为 Q-Former cross-attention 的 key/value | **否**，不是 LLM decoder KV | [CVPR 2024 论文](https://openaccess.thecvf.com/content/CVPR2024/html/He_MA-LMM_Memory-Augmented_Large_Multimodal_Model_for_Long-Term_Video_Understanding_CVPR_2024_paper.html)、[官方代码](https://github.com/boheumd/MA-LMM)。非常适合验证“视觉记忆压缩”，但需单独 cache kind |
| MovieChat | `visual_memory` | Atkinson–Shiffrin 风格短时/长时记忆；将 dense frame token 通过相似性合并为 sparse memory | 否 | [CVPR 2024 论文](https://arxiv.org/abs/2307.16449)、[论文声明的官方代码入口](https://github.com/rese1f/MovieChat)（现重定向到维护者新用户名）。长视频记忆代表；旧多仓依赖使首批部署风险较高 |
| StreamingVLM | `decoder_kv` | 真流式；保留 attention sink、短窗口近期视觉 token、长窗口近期文本 token；用重叠短 chunk 的 SFT 对齐训练/推理注意力模式 | **是** | [ICLR 2026 论文](https://arxiv.org/abs/2510.09608)、[官方代码](https://github.com/mit-han-lab/streaming-vlm)。机制干净但不是 training-free，首批不承担额外 SFT 风险 |
| InfiniPot-V | `decoder_kv` | continual KV compression；达到阈值后以 temporal-axis redundancy 与 value norm 做 query-agnostic、training-free 的原地压缩，硬限制长度 | **是** | [NeurIPS 2025 论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/caef5f5e658aa1f7565f063a2cd99726-Paper-Conference.pdf)、[公开仓库](https://github.com/aiha-lab/InfiniPot-V)。仓库明确称为不含全部论文组件的研究复现，且根目录未见许可证，不能作为首批稳健集成 |
| HERMES | `decoder_kv` | training-free；按浅/中/深层分别建模 sensory/working/long-term memory，配合跨层平滑与位置重编号；固定每层 KV 预算 | **是** | [ACL 2026 论文](https://arxiv.org/abs/2601.14724)、[官方代码](https://github.com/haowei-freesky/HERMES)。官方支持 LLaVA-OneVision 0.5B，适合 **首批真实流式冒烟** |
| MuKV | `decoder_kv` | 将历史 KV 表示压成 patch/frame/segment 多粒度，结合 attention 与频率信号；查询时半层次检索相关 KV | **是** | [CVPR 2026 论文](https://arxiv.org/abs/2605.22269)、[官方代码](https://github.com/IMBALDY/MuKV)。官方默认支持 LLaVA-OneVision 0.5B；但当前仓库无 LICENSE 文件，进入主树前必须获得/确认授权 |

### 重要可比性边界

- LongVU、MovieChat、MA-LMM 的收益主要发生在 prefill 前或视觉记忆层；HERMES、StreamingVLM、InfiniPot-V、MuKV 发生在 decoder KV。它们应分别与各自的 `identity` 对照比较，不能只按“最终 token 数相同”得出机制优劣。
- LongVU 的一部分压缩依赖文本 query。UCF-Crime 分类若使用“是否异常/哪类犯罪”提示，可能把类别先验引入视觉选择。公平设置应使用固定、类无关 query，并另做 query-aware 消融。
- HERMES、MuKV 和 InfiniPot-V 的论文任务是 VideoQA/视频理解，不是 VAD。把缓存状态或压缩后 token 接到二值/时序分类头是本项目提出的适配，效果属于待验证假设。
- 速度至少拆成视频解码、vision tower、projector、LLM prefill、缓存更新、head 和端到端墙钟时间；只报告生成 TTFT 无法回答 VAD 批处理吞吐问题。

## 4. 首批选型与真实冒烟定义

### 4.1 VideoMAE V2 Base：固定 clip 对照

选择依据：

- 官方实现、HF Base 权重和下游分类路径齐备；本项目 registry 已固定 `OpenGVLab/VideoMAEv2-Base` revision。
- 官方 HF forward 的稳定公开输出是池化 `[B,D]`；本项目保证将其规范成 `[B,1,D]`，并机会性地从 pinned private backbone hook 观察 `[B,S,D]`。该 hook 不是上游稳定 API，token-to-frame timeline 只是均匀近似，必须在产物中记录。
- 没有跨 clip cache，能作为压缩实验的明确零状态对照。

冒烟成立的最低证据：真实权重 + 真实视频解码出的 16 帧；输出有限值的 `[1,S,D]`（允许稳定 fallback `S=1`）；记录 `sequence_source`、近似时间轴策略、模型 revision、dtype/device、耗时和峰值显存。mock 权重或随机 tensor 只能算单元测试。

### 4.2 HERMES + LLaVA-OneVision-Qwen2-0.5B：流式 KV 对照

严格说这是“0.5B LLaVA-OneVision 基座上的 HERMES cache policy”，不是名为“HERMES-0.5B”的独立 encoder。官方 README 列出的最小模型正是 [`llava-hf/llava-onevision-qwen2-0.5b-ov-hf`](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf)，并公开 `kv_size`、`sample_fps` 和 `llava_ov_0.5b` 路径。

选择依据：

- 直接研究 decoder KV，和用户拟开展的 KV 压缩工作同层；该缓存策略无需额外训练，可先隔离推理机制。
- 0.5B 是官方支持范围，下载、显存和服务器冒烟风险小于 7B/32B。
- 可以在同一 adapter 中运行 `identity` 与 HERMES policy，避免模型/采样变化冒充压缩收益。

这是一项**双轨工程冒烟选型**，不是“UCF-Crime 最佳模型”判断：HERMES 的公开评测是 VideoQA/流式理解，不是 UCF-Crime MIL；真实 VAD 精度、吞吐和迁移稳定性必须由本项目实验给出。

当前 HERMES 主实验配置已与固定 clip 基线统一为 32 段、16 帧、stride 2，并使用 `weak_mil`。为使缓存消融的表征真正受历史 decoder KV 影响，配置明确选择 `decoder_contextual`；`native_compression_mode: off` 加框架侧 `identity` 是 raw 对照，HERMES 原生 `predict`/`static_pseudo` 另作独立消融。`projected_visual` 仍可用于性能/状态观测，不能代替缓存—精度结论。

冒烟必须连续处理至少两个真实 chunk，并证明：状态跨 chunk 增长或被预算策略更新；输出 chunk 时间范围递进，而同一 chunk 内多个 token 的近似时间戳允许相等（整体单调不减）；`identity` 与 HERMES 两次运行的采样、基座、精度和 head 配置一致；保存每步 KV token 数、淘汰/聚合数、峰值显存与耗时。仅调用一次普通 `generate(use_cache=True)` 不足以证明 HERMES 已接通。

## 5. 当前接入完成后仍需验证的机制

1. **LongVU：** 当前 adapter 已能导出官方视觉路径的 `projected_visual` 并有历史 smoke PASS，但尚未形成 query-independent 与 query-aware 压缩的受控消融。后续应分别报告 DINOv2 去冗余、SVA 和 prompt 影响。
2. **VideoChat-Online：** 当前已按 Pyramid Memory Bank 注册为 `visual_memory`，两 chunk 技术前向曾通过；许可证没有闭合，因此仍为 blocked，也不能升级成 `decoder_kv`。
3. **VideoChat-Flash：** 当前只加载视觉塔与 projector、关闭 LLM 压缩并导出 `projected_visual`。HiCo、progressive visual dropout 和 short-to-long 训练仍未成为本项目可比较的压缩实验。
4. **MA-LMM/MovieChat：** adapter facade 已存在，但真实资产链未到位；不得把 facade 单测表述为原生 visual-memory 前向通过。
5. **StreamingVLM/MuKV/InfiniPot-V：** StreamingVLM 有历史技术前向但许可阻塞；MuKV 与 InfiniPot-V 仍是 `candidate_only`。只有许可与专用资产闭合后，才能形成 HERMES 之外的 decoder-KV 精度消融。

## 6. 建议实验矩阵

| 实验轴 | 固定项 | 变化项 | 必报结果 |
|---|---|---|---|
| 编码器基准 | 官方 split、32 段、同一 head/seed | VideoMAE V2/I3D/X3D/Video Swin | frame ROC-AUC、AP、参数/权重 revision、吞吐、峰值显存 |
| HERMES 机制 | 0.5B 基座、真实帧、同一 chunk/FPS | identity vs HERMES；KV budget | AUC/AP、KV token 曲线、压缩率、cache update 耗时、端到端 FPS |
| 视觉 token 压缩 | 同一 VLM/head 与 decoder cache | identity vs LongVU 子策略 | prefill token、vision/projector/prefill 分段耗时、AUC/AP |
| 视觉记忆 | 同一输入与监督 | FIFO vs adjacent merge vs PMB | memory 长度、历史召回距离、AUC/AP、吞吐 |
| 监督方式 | 同一 encoder/split | video-level MIL vs 合法 temporal supervision | frame AUC/AP；标签来源、映射规则与覆盖率 |

## 7. 风险与尚未证明的事项

- **许可证：** pinned VideoMAE V2 代码是 MIT，但 Base 权重模型卡为 CC-BY-NC-4.0；pinned HERMES 代码是 MIT，LLaVA-OneVision 0.5B 权重登记为 Apache-2.0。代码与权重必须分别审查。InfiniPot-V 和 MuKV 当前根目录许可证信息不完整，不能仅凭 README badge 或论文开源声明推定可再分发。
- **安装包边界：** 当前 `VideoMAEv2Adapter` 运行时导入仓库根目录的 `lab_anomaly.models.vit_video_encoder`，而 wheel 配置只打包 `src/vadbench`。源码 checkout 中可运行不等于安装后的 wheel 可运行；在迁移 wrapper 或调整打包并加入 wheel 安装验证前，VideoMAE V2 不能声称具备独立 wheel 部署能力。
- **上游漂移：** 所有实际下载必须固定 commit/HF revision 和 SHA256；表中“官方仓库可用”不等于任意日期的 `main` 可复现。
- **HERMES requirements 移植：** 官方 LLaVA 路线使用 Python 3.12、独立 requirements 和 `flash-attn --no-build-isolation`；其 transformers/FlashAttention/CUDA 组合可能与主框架环境冲突。服务器应使用 pinned checkout 和隔离环境，真实冒烟前不能用主环境单测替代依赖验证。[HERMES README](https://github.com/haowei-freesky/HERMES)
- **硬件：** LongVU 官方 quick start 写明本地 demo 需要约 40 GB GPU，训练使用 64 张 H100-96G；不适合首批服务器低风险冒烟。[LongVU README](https://github.com/Vision-CAIR/LongVU)
- **任务错配：** VQA 中保住答案 token 不等于保住稀疏异常片段。压缩策略必须以 UCF-Crime 帧级指标和历史距离分桶重新验证。
- **强监督来源：** UCA 的时间段描述普通事件和异常事件，不能直接充当二值异常区间；详见 `docs/research/ucf-crime-protocol.md`。

## 8. 可追溯性与调研截止

- 调研截止：2026-08-31（Asia/Hong_Kong）。
- 仓内实现同步：2026-09-11；只对照本地/服务器相同 commit 和已有产物，没有重新做外网来源核验，也没有新跑模型 forward。
- 论文接收状态以论文正式页或作者官方仓库当日信息为准；未来更新应保留本报告日期并新建修订记录。
- 本报告未进行跨论文准确率排名，因为输入帧数、分辨率、LLM、prompt、数据与硬件均不可直接比较。
- Agent Reach 的 Exa 免费端在检索后段触发额度限制；缺失方向改用 arXiv 官方 API、GitHub API 和作者仓库核验。此限制不影响上表关键候选的来源追溯，但不应把本表解释为所有 2026 年工作的穷尽清单。
