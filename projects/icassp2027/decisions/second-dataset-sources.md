# XD-Violence 第二数据集：原始 RGB 来源与协议核查

核查日期：2026-09-18。目的仅为判断第二数据集的可用性；本轮未登录、未读取 cookie、未接受网盘条款、未转存、未下载或解压任何文件。

## 决策结论

XD-Violence 是有价值的第二数据集候选，但**当前只具备“官方原始视频入口已核实”的状态，尚不具备自动下载/夸克中转的放行条件**。

- 原始视频的官方入口是作者的 [XD-Violence 项目页](https://roc-ng.github.io/XD-Violence/)，不是页面下方的 I3D RGB&Flow、VGGish 等预提取特征。项目页把二者分为独立的 `V1.0 Videos` 和 `V1.0 Features` 下载区。
- 当前主线应是 **RGB-only**：只解码官方 `V1.0 Videos` 的图像帧供现有 ViT encoder 使用；不接入音频、VGGish、I3D RGB/flow，也不把作者的音视频融合 AP 与本项目结果放在同一比较中。
- 官方原始下载是 Baidu Netdisk、AliyunDrive 和 OneDrive；未列出夸克链接。核查到的公开搜索没有返回一个同时满足“标明 XD-Violence 原始视频、可追溯发布者、可公开打开”的夸克候选，故没有可转存的夸克链接，更没有大小或分卷的可靠数据。
- 官方页没有给出明确的数据许可证或第三方影片/YouTube 素材的再分发许可。原论文说明其视频来自 91 部电影和 YouTube；这使“官方页面提供下载”与“可在未核对条款时自动下载/再同步”是不同的问题。开始大规模下载前，应由数据所有者/项目负责人确认研究使用资格和当前存储服务的访问条款；不可由 agent 代为接受未知条款。

## 官方 RGB 原始视频入口

| 入口 | 实际工具核验 | 可得到什么 | 大小/分块与访问状态 |
|---|---|---|---|
| [官方项目页](https://roc-ng.github.io/XD-Violence/) | 可公开读取；页面标注 ECCV 2020、作者与项目代码。 | `V1.0 Videos` 下的原始训练/测试视频入口；`V1.0 Features` 下另列 VGGish、I3D RGB&Flow。 | 页面未给总字节数、压缩格式或校验和。 |
| [官方 AliyunDrive 训练视频分享](https://www.aliyundrive.com/s/6UquaxKpKTm) | 对官方链接的无 cookie `HEAD` 请求返回 HTTP 204；未读取文件列表或点击下载。 | 官方页标为 `Training Videos`。 | 公开落地页可达的初步信号；文件名、分卷和大小均未从无 cookie 响应获得。 |
| 官方 OneDrive 训练视频（项目页逐项链接） | 训练分块之一无 cookie `HEAD` 为 HTTP 200，读取分享页面（非下载）得到 `FileLeafRef: 1-1004.zip`。 | 官方页依次标为 `Training Videos_0001-1004`、`1005-2004`、`2005-2804`、`2805-3319`、`3320-3954`。 | 可确定共有 **5 个训练 zip 分块**，覆盖 3,954 个训练视频。第一块分享页实际可打开；未得到字节数。 |
| 官方 OneDrive 测试视频与标注 | 测试视频分享链接无 cookie `HEAD` 为 HTTP 200；后续只读页面请求曾返回 HTTP 503，未重试、更未下载。测试 annotations 为官方静态文本。 | `Test Videos` 和 [Test Annotations](https://roc-ng.github.io/XD-Violence/images/annotations.txt)。 | 测试视频的无认证访问可能受服务瞬时状态影响；大小未知。下载前需重新检查。 |
| 官方 Baidu Netdisk | 官方页列出测试视频与提取码，并把训练视频条目明确标为 disabled。 | 不能当作完整、稳定的原始视频路径。 | 本轮未登录或输入提取码。 |

上述 OneDrive 文件名和 5+1 的分块结构来自项目页的显示文字；一条训练分享页的文件名从实际 HTTP 响应中交叉确认。网页没有公开报告各压缩包大小，因而不能据此估算服务器所需空间或下载时长。下载开始前应先获取每卷的元数据和校验信息，并在服务器可用容量基础上为压缩包与解压后的原始视频保留空间。

## 准确的 RGB-only 任务协议

以作者的 [ECCV 2020 原论文](https://roc-ng.github.io/XD-Violence/images/paper.pdf) 和官方 [annotation ReadMe](https://roc-ng.github.io/XD-Violence/images/ReadMe.md) 为准：

| 项目 | 已核实定义 | 对本项目的含义 |
|---|---|---|
| 数据规模与类别 | 4,754 个未裁剪视频、217 小时；2,405 个暴力、2,349 个非暴力。暴力类为 Abuse、Car Accident、Explosion、Fighting、Riot、Shooting；一个暴力视频可有 1–3 个类别。 | 这是 violence VAD，不是 UCF-Crime 的同分布副本；分析可按类别分层，但线上 reducer 不得读取类别。 |
| 官方划分 | 训练 3,954；测试 800，其中测试为 500 暴力、300 非暴力；两个划分都含六类事件。 | 训练/确认/选择必须仅从官方训练视频按视频 ID 划分；不能把官方测试时间标注用于方法选择。 |
| 标签粒度 | 训练是视频级弱标签；测试暴力视频有人工平均后的事件起止**帧**标注。官方 ReadMe 说明 `annotations.txt` 每行是文件名，其后为一对或多对 start/end frame，例如两个事件使用四个数。 | 用官方测试标注只做冻结后的帧级评测/机制复核；端点开闭和 0/1-based 语义必须以导入时的小样本视频核验，不能沿用 UCF 的转换。 |
| 主评测 | 原论文使用帧级 precision-recall curve 及 AP，理由是正类不平衡；该页的 `annotations.txt` 是对应测试时序标签。 | 项目应把 XD 的主指标记录为全测试帧的 AP，与当前 RGB-only 预测覆盖一致；不能改报 UCF 的 frame ROC-AUC 作为 XD 主指标。 |
| 原作者的视觉特征流程 | 原文为 I3D 的 RGB 或 flow 特征，统一 24 FPS、16-frame 滑窗；默认实验融合 I3D RGB + VGGish。 | 这只说明原始基准的历史实现，不能替代 ViT 原视频解码。项目的 `RGB-only` 路径应明确自身的采样、帧率处理、clip 覆盖和预测回填，不声称复现原融合设置。 |

## 与 UCF-Crime 的独立性价值和边界

作者论文将 UCF-Crime 列为 1,900 个、128 小时、CCTV 场景、9 类且无音频；将 XD-Violence 列为 4,754 个、217 小时，来源含电影、体育、游戏、手持/CCTV/车载相机等，6 类且有音频。对于本项目的 RGB-only 跨数据集确认，这提供了**明显不同的来源场景、类别定义和规模**，因而可检验 token 规则是否只适合 UCF 的监控场景。

这不是已证明的逐视频去重保证。XD 的论文说明数据部分来自电影和 YouTube，UCF 的来源与第三方素材是否存在重合，本轮没有官方交叉名单可核验；不能把“两个官方数据集”写成“已证明无近重复”。下载后应至少以文件名/时长/感知哈希进行跨集近重复审计，并将发现的重合视频或同源片段排除出跨数据集泛化论断。

## 夸克分享核查

已实际检索以下窄查询：`"XD-Violence" "pan.quark.cn/s/"`、`"XD Violence" "pan.quark.cn/s/"`、`"XD-Violence 数据集" "夸克网盘"`、`"XDViolence" "夸克网盘"`。没有得到可归因于作者或官方项目、且明确为**原始视频**的夸克链接；结果中出现的无关夸克链接未采用。故本轮的状态是 **unknown / no verified Quark share**，不是“夸克不存在”的全网断言。

不应使用只包含 `I3D`、`VGGish`、`RGB.npy`、`RGB.zip` 等特征的网盘包作为原始视频替代物。若后续有人提供夸克链接，最小验收条件是：链接可在无 cookie 状态打开；落地页明确文件/分卷、大小和是否需登录/提取码/付费；发布者能回溯到官方作者或获得其明确授权；内容确为官方 train/test 原视频而非特征；许可证和服务条款允许预期的研究下载与存储。任一项未知则保持不下载。

## 下一步前的待决条件

1. 由项目负责人确认对官方页面所分发的含电影/YouTube素材的原始视频具有研究使用资格，并确认不需要 agent 接受未展示的条款。
2. 在正式下载时重新检测每个官方链接、读取每卷大小/哈希，并据此重新评估 443 GiB 空间是否同时容纳压缩包、解压视频和 UCF；本轮不以未知体积批准排程。
3. 下载完成后先验证官方训练/测试计数、annotation 文件名匹配和视频可解码性，再建立项目自己的 RGB-only manifest；测试 annotations 保持冻结后使用。

### 直接来源

- [XD-Violence 官方项目页](https://roc-ng.github.io/XD-Violence/)：作者维护的原始视频/特征下载区与代码链接。
- [Wu et al., ECCV 2020 原论文 PDF](https://roc-ng.github.io/XD-Violence/images/paper.pdf)：规模、来源、划分、标注与 AP 协议。
- [官方 annotations.txt](https://roc-ng.github.io/XD-Violence/images/annotations.txt) 与 [官方 ReadMe](https://roc-ng.github.io/XD-Violence/images/ReadMe.md)：测试事件起止帧的文件格式。
- [作者官方 XDVioDet 代码仓库](https://github.com/Roc-Ng/XDVioDet)：将项目页列为 dataset/features 来源，但该仓库使用的是特征流程，不能替代原始视频入口。
