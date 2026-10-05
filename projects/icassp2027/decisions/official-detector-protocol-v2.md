# 官方划分与原作者后端：正式实验口径 v2

> **2026-09-20 修订指针**：`official-detector-protocol-v3.json` 已发布——相对 v2 的唯一实质差异是
> `datasets.xd_violence.encoders` 加入 `videomae`（用户 2026-09-20 14:1x 指令：XD 三 encoder 全做，
> 控制面 README 有记录；videomaev2/timesformer 不变）。v2 JSON 保留原位、字节不变；
> XD 相关运行请改用 v3（路径 + SHA 见控制面 MANIFEST v12 增量）。本文件其余内容继续描述 v2 决策。

2026-09-18，用户已回答三个实质问题：没有指定异常检测论文，由执行者选择并核验；
**官方 train/test 划分为主，隔离子集为补充；冻结 dense 检测头的直接插入为主，重训头为补充。**
这项决定发生在本项目访问官方模型测试分数之前。它取代旧工程队列的主次安排，
不改写已有锁、校准或运行回执。机器可读配置见同目录
[official-detector-protocol-v2.json](official-detector-protocol-v2.json)。

用户随后参考预训练WSVAD调查，进一步明确创新必须位于视觉ViT encoder内部，
多个检测后端不是主要验证维度，并明确允许“公开ViT预训练权重＋为每encoder训练一次
同一固定检测后端，再冻结比较”。据此保留四种encoder—UR-DMU配对：VideoMAEv2-B、
TimeSformer-B、V-JEPA 2-L为主，VideoMAE-B为同家族补充；没有采用此前提出的
CLIP多后端建议，没有增加第五encoder。每种encoder只采用UR-DMU这一种后端算法。
不同dataset/seed的匹配参数属于同一配对的重复实验，不算增加检测后端架构。

四个pinned encoder仓库的当前公开文件已通过HF API检查，均非private、非gated，
weight大小/LFS SHA与本项目登记一致。它们是公开预训练encoder，不是四套现成的
WSVAD任务权重；原作者UR-DMU的I3D权重不能替代为每个新encoder训练的匹配后端。

选用固定版本 UR-DMU 的完整网络与原损失，保留双记忆、时序 attention、不确定性分支和
分类器。采用 Adam、lr=1e-4、weight decay=5e-5、3000 optimizer steps、200段、每类64个bag。
新 encoder 的输入维数按真实768/1024适配，分别从原初始化训练；不把I3D官方checkpoint
直接接到另一个表示空间。作者代码会按test选best，本项目预先固定最后3000步checkpoint，
明确披露这一差异。三个固定seed均保留，不挑最佳seed。

UCF主实验使用1610个官方训练视频和290个测试视频，覆盖四个既定encoder。
XD主实验使用3954个官方训练视频和800个测试视频，按已确定的可支持输入范围验证
VideoMAEv2和TimeSformer。完整训练视图是原fit/confirm/select身份的并集，保留原角色
供溯源；损坏或缺失视频不得静默删除后仍称完整官方训练。XD1598名称组隔离视图保留为
稳健性补充，不能代替主实验；官方划分本身可能包含相关来源，需在局限中披露。

先在每个encoder的完整dense训练特征上训练UR-DMU，再冻结同seed检查点，分别输入
dense、均匀缩减和两种候选插件的测试特征。该配对检验直接插入的质量变化。
重训头使用相同网络、原损失、训练步数和seed，单独报告。现有32片段工程特征不等于
原作者的dense序列，不能上采样成200段冒充覆盖；正式特征先完整提取，再按作者规则
做200个时间bin的平均，短序列空bin沿用选取已有片段的规则。

原作者UCF10crop/XD5crop的I3D特征与公开checkpoint复核是独立参考。
新四encoder沿用已验证的单crop原生预处理，所有比较方法完全一致；这是采用作者后端的
新encoder实验，不能称逐数复现原论文。UCF原作者的尾帧裁剪与本项目已审计raw帧协议
也分开记录；XD使用已预登记、仍待完整raw800验证的作者16Ti前缀坐标。

已取得两官方checkpoint并完成初步CPU安全加载；合成forward通过不代表原论文分数复现。
原始特征入口当前有登录/可访问性限制，继续查证合法公开来源。作者参考复核尚未完成，
正式新后端训练和测试也未完成。旧TopKMIL评测队列保持暂停，不自动转为正式主结果。

候选插件保持原冻结策略和UCF fit校准，未经测试重新选择层或预算。
随机保留及线性加权本身尚不足以建立新颖性；后续需结合真实质量、A100效率和已有压缩
方法比较，决定论文能支撑的主张。0.005质量容忍度、视频为独立统计单位、负结果保留
等要求不变。参考[V100计时审计](../../../outputs/icassp2027/analysis/formal-timing-audit-20260918/AUDIT.md)
已显示峰值显存没有下降，不能沿用“token减半即显存减半”的推论。
