# UCF-Crime 官方原视频归档：限定可达性核查

核查日期：2026-09-18。范围是为现有 UCF 下载寻找**官方原视频**替代源；不改动、不暂停或重启当前 Quark 下载，不读取测试事件标注，不改 `data_root` 或任何 `icassp2027-runs/code-*` 冻结目录。

## 结论

作者/CRCV 仍公开提供官方单卷原视频归档 [UCF_Crimes.zip](https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip)，其来源页是 [Real-world Anomaly Detection in Surveillance Videos](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)。本机匿名默认 TLS 的单字节 Range GET 返回 HTTP 206 与 ZIP 首字节，证明该官方 raw archive 当前公开且可断点读取。归档大小为 **102,957,372,377 bytes（95.89 GiB）**。

node2 的默认 TLS 最初不能使用该 archive：经 node3→node2 的匿名标准库 `urllib + 内存 CookieJar` Range GET 在默认验证下报 `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`，没有读取 body。现有 `foundation-video-v2` Python 以 `certifi.where()` 建立严格 context（`check_hostname=True`、`CERT_REQUIRED`）也报同一 issuer 错误。

进一步诊断确认这是服务端未交付完整中间链，而不是不可信根：Windows 默认验证的 `SslStream` 仅在 `SslPolicyErrors.None` 时导出公开 leaf 与一个中间证书，不导出根或私钥；node2 用现有 certifi CAfile 作为**唯一**信任锚、该中间证书作为 `openssl verify -untrusted` 输入，验证通过。随后仅以 `certifi + 已验证中间证书` 构造临时、进程内 CAfile，仍保持 hostname 与 `CERT_REQUIRED`，对 archive 的一次 `GET Range: bytes=0-0` 返回 HTTP 206、1 byte、ZIP 首字节 `50`。

因此，node2 已有一个严格验证的、每进程临时的官方下载链路；它不安装证书、不改系统 trust store/proxy、不新增根 CA、不降级 TLS，也不等同于已经开始下载。实际调度仍由根节点决定。

## 官方性与数据身份

CRCV 官方页面将该文件明确标为 `Dataset`，同时说明 archive 中的 `UCF_Crimes` 包含：

- `Videos`：完整视频数据，共 16 个自解释子目录，包含 13 个异常类、动作检测的正常视频，以及 anomaly-detection 训练/测试正常视频目录；
- `Anomaly_Detection_splits`：官方 anomaly-detection train/test partitions；
- `Action_Recognition_splits`：动作识别的四折划分。

该页面与原论文对 UCF-Crime 的描述相符：1,900 个长、未裁剪的真实监控视频，约 128 小时，13 类异常。它是原视频归档，不是 I3D/C3D/CLIP 特征、caption、截图或伪标注，因而与当前 1,900 文件的官方协议身份匹配。

Chen Chen 的 [数据页](https://www.crcv.ucf.edu/chenchen/datasets/) 另外提醒该 zip 中的 `Anomaly_Train.txt` 有损坏版本，正确的文件在项目页提供。因此，即使未来使用该官方 archive，也应按项目已固定的 1,610 train / 290 test 协议和当前经过审计的 split 清单建 manifest，不能盲目采用 archive 内未核验的训练列表。

## 实测回执

| 位置与 client | 请求 | 结果 | 解释 |
|---|---|---|---|
| 本机，默认 `System.Net.Http.HttpClient` | `GET Range: bytes=0-0`，最多读 1 byte | HTTP 206；`Content-Range: bytes 0-0/102957372377`；读到 1 byte `50` | 官方归档公开、是 ZIP、支持 Range。 |
| node3 公网入口 → node2，默认 `urllib.request + CookieJar` | 同一单字节 Range GET | `URLError`；TLS 本地 issuer 证书无法验证；0 body bytes | node2 默认 TLS 链路当前不能使用该源；没有用不安全绕过。 |
| node3 公网入口 → node2，`foundation-video-v2` Python + `certifi.where()` | 同一单字节 Range GET；`check_hostname=True`、`CERT_REQUIRED` | 同一 issuer 证书验证错误；0 body bytes | 现有 certifi bundle 也不能验证该路径；不添加根证书或降级校验。 |
| node3 公网入口 → node2，`foundation-video-v2` Python + 临时 `certifi + verified intermediate` CAfile | `openssl verify` 先通过，再作同一单字节 Range GET；`check_hostname=True`、`CERT_REQUIRED` | HTTP 206；`Content-Range: bytes 0-0/102957372377`；读到 1 byte `50` | 严格验证的进程内链补齐；未新增根或修改系统。 |

## ZIP 目录、冻结 192 计划与单成员可行性

本机对官方 ZIP 只读取 EOCD、56-byte Zip64 EOCD、central directory，以及待补成员的 104-byte local-header 元数据；总计 **370,517 bytes**，远低于 16 MiB 上限。archive 为 Zip64，central directory 为 304,800 bytes，含 1,981 members：1,950 个视频、20 个目录和 11 个 `.txt` 文件。所有成员展开总量是 **104,888,659,590 bytes（97.69 GiB）**，没有绝对路径或含 `..` 的 member。没有读取任何 `.txt` 内容，尤其没有读取或使用 archive 内的损坏 split。

1,950 个视频成员中，`z_Normal_Videos_event` 的 50 个正常视频属于 archive 同时携带的 action-recognition 内容；去除它后，余下 **1,900** 个视频与 anomaly-detection 官方协议相符。冻结计划 [frozen-candidate-files.json](../../outputs/icassp2027/data-plan/20260917T180000Z/frozen-candidate-files.json) 的 192 条 `official_path` 均在 `UCF_Crimes/Videos/` 下找到，192 条 `mirror_size_bytes` 与 ZIP 中逐成员 `uncompressed_size` **全部精确相等**，也全部通过相对路径安全检查。

唯一待补的 explore 视频 `Normal_Videos533_x264` 在 archive 中的精确官方 member 是 `Training_Normal_Videos_Anomaly/Normal_Videos533_x264.mp4`：展开大小 662,473,473 bytes、CRC32 `acfd9bbb`、compression method `8`（Deflate）、compressed size 657,001,186 bytes、local-header offset 73,117,930,621、payload offset 73,117,930,725。该 offset 由单独读取 104 bytes 的 ZIP local header 得到，没有读取 payload。

因此，就 archive 格式和官方 Range 支持而言，可以实施“只取该 member 的 local header + 精确 compressed payload range、再以 raw Deflate 解压”的补齐流程，而无需传输完整 95.89 GiB archive。本轮只证明 metadata 和位置，未读取 compressed payload，也没有改变当前下载调度。上述临时严格 CA chain 已将 node2 的官方单字节 Range GET 验证为可行；实际单成员或全 archive 下载仍须由根节点明确调度。

只核验了这一项官方主归档候选，未采用陌生镜像、网盘搬运或特征包，也没有测试官方时间标注链接。完整安全回执在已忽略的 `outputs/icassp2027/source_checks/ucf_crcv_official_source_receipt_20260918.json`、`ucf_crcv_zip_metadata_receipt_20260918.json`、`ucf_crcv_zip_parts_20260918/final_metadata.json`、`ucf_crcv_frozen192_comparison_20260918.json` 与 `ucf_node2_verified_intermediate_probe_20260918.json`；它们不包含 archive URL 参数、cookie 或认证数据。公开 leaf/intermediate PEM 仅保存在同目录 `ucf_tls_chain_20260918/`，不含私钥。
