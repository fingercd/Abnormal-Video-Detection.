# XD-Violence 第二数据集：原始 RGB 来源与协议核查

核查日期：2026-09-18。目的仅为判断第二数据集的可用性；本轮未登录、未读取 cookie、未接受网盘条款、未转存、未下载或解压任何文件。公开官方研究数据的查找与必要下载已获用户授权；这里不把未展示许可证泛化为额外的用户确认要求。

## 决策结论

XD-Violence 的官方压缩包预算已核清：5 个训练卷加 1 个测试卷共 **85,432,842,576 bytes（79.57 GiB）**。node2 的 `/users` 数据盘实测可用 **465,854,005,248 bytes（433.86 GiB）**，为该压缩包总量的约 5.45 倍；因此只存放官方压缩包的空间预算低于 120 GiB，**可用**。ZIP 目录元数据已进一步确认完整解压为 80.06 GiB，二者同驻为 159.63 GiB；与 UCF 下载和实验产物的实际共存安排仍由根节点统一调度。

- 原始视频的官方入口是作者的 [XD-Violence 项目页](https://roc-ng.github.io/XD-Violence/)，不是页面下方的 I3D RGB&Flow、VGGish 等预提取特征。项目页把二者分为独立的 `V1.0 Videos` 和 `V1.0 Features` 下载区。
- 当前主线应是 **RGB-only**：只解码官方 `V1.0 Videos` 的图像帧供现有 ViT encoder 使用；不接入音频、VGGish、I3D RGB/flow，也不把作者的音视频融合 AP 与本项目结果放在同一比较中。
- 官方原始下载是 Baidu Netdisk、AliyunDrive 和 OneDrive；未列出夸克链接。本机无 cookie 实测 6 个 OneDrive 卷均返回附件元数据并支持真实的单字节 Range；因此无需为技术可达性另寻夸克镜像。
- 官方页没有显示独立许可证文本，且论文说明素材来自电影和 YouTube；这限制了再分发主张，但不改变用户已授权的“从官方公开研究入口下载”范围。若某个后续服务页面出现必须点击接受的新条款或要求登录，本 agent 只记录其原文事实，不代为接受或登录。

## 官方 RGB 原始视频入口

| 入口 | 实际工具核验 | 可得到什么 | 大小/分块与访问状态 |
|---|---|---|---|
| [官方项目页](https://roc-ng.github.io/XD-Violence/) | 可公开读取；页面标注 ECCV 2020、作者与项目代码。 | `V1.0 Videos` 下的原始训练/测试视频入口；`V1.0 Features` 下另列 VGGish、I3D RGB&Flow。 | 页面未给总字节数、压缩格式或校验和。 |
| [官方 AliyunDrive 训练视频分享](https://www.aliyundrive.com/s/6UquaxKpKTm) | 对官方链接的无 cookie `HEAD` 请求返回 HTTP 204；未读取文件列表或点击下载。 | 官方页标为 `Training Videos`。 | 公开落地页可达的初步信号；文件名、分卷和大小均未从无 cookie 响应获得。 |
| 官方 OneDrive 训练视频（项目页逐项链接） | 将每卷官方分享 URL 加 `download=1` 后，无 cookie `HEAD` 得 HTTP 200、`Content-Disposition: attachment`、`Accept-Ranges: bytes`；不读取 body 的 `Range: bytes=0-0` 对每卷均得 HTTP 206 和 1-byte body。 | 官方页依次标为 `Training Videos_0001-1004`、`1005-2004`、`2005-2804`、`2805-3319`、`3320-3954`。 | 共 **5 个训练 zip 分块**，覆盖 3,954 个训练视频；合计 **74,628,109,565 bytes（69.50 GiB）**，每卷明细见下表。 |
| 官方 OneDrive 测试视频与标注 | 将官方测试分享 URL 加 `download=1` 后，无 cookie `HEAD` 得 HTTP 200、附件及 `Accept-Ranges: bytes`；`Range: bytes=0-0` 得 HTTP 206 和 1-byte body。测试 annotations 本轮未读取。 | `Test Videos` 和 [Test Annotations](https://roc-ng.github.io/XD-Violence/images/annotations.txt)。 | 测试视频卷大小 **10,804,733,011 bytes（10.06 GiB）**，支持断点续传。此前的 503 是分享页瞬时响应，不影响本次附件头/Ranges 核验。 |
| 官方 Baidu Netdisk | 官方页列出测试视频与提取码，并把训练视频条目明确标为 disabled。 | 不能当作完整、稳定的原始视频路径。 | 本轮未登录或输入提取码。 |

上述 OneDrive 文件名和 5+1 的分块结构来自项目页的显示文字；6 卷的体积来自无 cookie `HEAD` 的 `Content-Length`，并各以 `Range: bytes=0-0` 的 206/1-byte 响应验证。没有输出分享 URL 的签名参数、重定向 URL、cookie 或任何测试标注。下载配置中的完整 URL 只存在于已忽略的运行脚本，未写入 Git 跟踪文件或本报告。

| 卷名 | HTTP `Content-Length` (bytes) | GiB | 本机 Range 验证 |
|---|---:|---:|---|
| `train_0001_1004` | 11,858,055,997 | 11.04 | 206，`bytes 0-0/11858055997`，body 1 byte |
| `train_1005_2004` | 14,911,939,983 | 13.89 | 206，`bytes 0-0/14911939983`，body 1 byte |
| `train_2005_2804` | 19,401,894,490 | 18.07 | 206，`bytes 0-0/19401894490`，body 1 byte |
| `train_2805_3319` | 14,929,726,173 | 13.90 | 206，`bytes 0-0/14929726173`，body 1 byte |
| `train_3320_3954` | 13,526,492,922 | 12.60 | 206，`bytes 0-0/13526492922`，body 1 byte |
| `test_videos` | 10,804,733,011 | 10.06 | 206，`bytes 0-0/10804733011`，body 1 byte |
| **总计** | **85,432,842,576** | **79.57** | — |

### ZIP 展开预算与路径安全：已核实

每卷以默认 `System.Net.Http.HttpClient` 分别读取 65,557-byte 尾部、56-byte Zip64 EOCD 和 central directory；随后由 Python `struct` 在本地离线解析。六卷均为 Zip64，所有 central directory 为 66,142–143,649 bytes；单卷实际元数据读取最大 **209,262 bytes**，总计 **1,042,023 bytes**，均远低于每卷 16 MiB 上限。没有关闭证书验证、修改系统代理、绕过 TLS、登录、读取测试 annotations 或读取任意视频 member body。

| 卷名 | members | 展开 bytes | 展开 GiB | 非视频成员 | 绝对路径 / `..` |
|---|---:|---:|---:|---|---:|
| `train_0001_1004` | 1,004 | 11,924,807,590 | 11.11 | 无 | 0 / 0 |
| `train_1005_2004` | 1,000 | 15,012,640,817 | 13.98 | 无 | 0 / 0 |
| `train_2005_2804` | 800 | 19,527,135,368 | 18.19 | 无 | 0 / 0 |
| `train_2805_3319` | 515 | 15,018,252,853 | 13.99 | 无 | 0 / 0 |
| `train_3320_3954` | 635 | 13,615,209,362 | 12.68 | 无 | 0 / 0 |
| `test_videos` | 801 | 10,870,237,075 | 10.12 | 1 个目录 | 0 / 0 |
| **总计** | **4,755** | **85,968,283,065** | **80.06** | **1 个目录** | **0 / 0** |

成员数与官方 3,954 训练 + 800 测试视频相符：测试卷多出的 1 个 member 是目录；没有发现非视频常规文件、绝对路径或路径遍历成员。压缩包 79.57 GiB 与完整解压数据 80.06 GiB 同时驻留需要 **159.63 GiB**。按当时快照，本机 D 盘可用 252.24 GiB，余约 92.61 GiB；node2 `/users` 可用 433.86 GiB，余约 274.23 GiB。该结果只核算 XD-Violence 自身，根节点应与既有 UCF 下载和实验产物一并安排实际空间预算。安全回执保存在已忽略的 `outputs/icassp2027/source_checks/xdv_zip_directory_receipt_20260918.json`；客户端差异回执在同目录的 `xdv_zip_client_diagnosis_20260918.json`。

### node2 联网回执（只读）

未重试失败的本地 Tailscale 路径。改走已验证的 **node3 公网入口 → node3 内 `ssh -oBatchMode=yes node2`**：node3 的 helper 预探测成功（约 2 秒），node3→node2 SSH 成功，因而 node2 的磁盘信息可读。node2 以 curl 对 6 个官方 OneDrive 卷分别发送无 body `HEAD`，每个返回 **HTTP 403**（curl exit 22）。

该 curl HEAD 结果不能外推为 node2 的正常 Range GET 被拒绝。随后在同一路径上，用默认 TLS 验证的 Python 标准 `urllib.request` + 本次请求专用的内存 `CookieJar`，对第 1 训练卷只发出一次 `GET Range: bytes=0-0`：收到 **HTTP 206**、`Content-Length: 1`、`Content-Range: bytes 0-0/11858055997`，并读取 1 byte（hex `50`，ZIP 的首字节）。没有登录、持久 cookie、代理/UA 修改、TLS 绕过或大文件下载。

因此，准确结论是：node2 的**curl HEAD 方法**被官方服务拒绝，但经匿名 `urllib` 的单字节 Range GET 已证实可读取官方 ZIP 数据。该可达性事实不等同于完整六卷下载已经启动或完成；下载调度仍由根节点决定。安全回执保存在已忽略的 `outputs/icassp2027/source_checks/xdv_node2_urllib_range_probe_20260918.json`。

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

不应使用只包含 `I3D`、`VGGish`、`RGB.npy`、`RGB.zip` 等特征的网盘包作为原始视频替代物。若后续必须使用夸克（例如官方 OneDrive 从 node2 持续不可达），最小验收条件是：链接可在无 cookie 状态打开；落地页明确文件/分卷、大小和是否需登录/提取码/付费；发布者能回溯到官方作者或获得其明确授权；内容确为官方 train/test 原视频而非特征。若页面要求接受新条款或登录，记录事实后停止该路径。

## 下一步前的待决条件

1. 实际下载应使用已验证的匿名 `urllib + CookieJar` Range GET 路径，而不是被服务拒绝的 curl HEAD 判定；不重试已失败的本地 Tailscale 路径。
2. 根节点安排时按 **79.57 GiB 压缩包 + 80.06 GiB 解压数据 + 现有 UCF/实验产物** 复算当前可用空间；本轮不启动或干预下载。
3. 下载完成后先验证官方训练/测试计数、annotation 文件名匹配和视频可解码性，再建立项目自己的 RGB-only manifest；测试 annotations保持冻结后使用。

### 直接来源

- [XD-Violence 官方项目页](https://roc-ng.github.io/XD-Violence/)：作者维护的原始视频/特征下载区与代码链接。
- [Wu et al., ECCV 2020 原论文 PDF](https://roc-ng.github.io/XD-Violence/images/paper.pdf)：规模、来源、划分、标注与 AP 协议。
- [官方 annotations.txt](https://roc-ng.github.io/XD-Violence/images/annotations.txt) 与 [官方 ReadMe](https://roc-ng.github.io/XD-Violence/images/ReadMe.md)：测试事件起止帧的文件格式。
- [作者官方 XDVioDet 代码仓库](https://github.com/Roc-Ng/XDVioDet)：将项目页列为 dataset/features 来源，但该仓库使用的是特征流程，不能替代原始视频入口。
