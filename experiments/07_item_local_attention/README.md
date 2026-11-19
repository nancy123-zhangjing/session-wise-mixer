# 商品内局部注意力探索

2026-10-21追加验证已完成：固定两组best25及各自SID，在同一验证集完成beam32/256四格，无重训或测试集评估。B的NDCG@10相对提升由3.27%缩至1.59%，HR@10由3.56%缩至1.07%；新增0.296389 GPU小时且已释放。见[搜索宽度诊断](beam_validation_20261021/REPORT.md)。以下7.428056 GPU小时为此前训练交付的独立历史账本，本轮计入后的07累计为7.724444 GPU小时。

## 当前方案：交替联合训练

用户已要求按05完整交替优化方式修改，本页下方固定SID设置仅保留作执行历史，不再是当前方案。当前配置为[joint_alternating_seed2024.json](../../configs/07_item_local_attention/joint_alternating_seed2024.json)：推荐器阶段训练推荐器与新增local，Tokenizer阶段冻结它们并通过DIGER-ST把推荐损失送回码本。量化码本和SID会随联合训练更新，文本特征与OPQ仍固定。两组共享起点的联合训练、唯一正式测试及本地归档已完成；商品内注意力组相对对照组测试NDCG@10提高0.425%、HR@10提高0.279%，未达3%初筛。详见[结果报告](joint_alternating/REPORT.md)、[联合训练入口与证据](joint_alternating/README.md)和[总验收](joint_alternating/FINAL_ACCEPTANCE_20261020_r3.json)。

全部8个GPU作业已释放，包含此前固定SID尝试、失败/中止、短测及两组正式训练，累计7.428056 GPU小时，未超过后续批准的28小时上限；详见[范围变更](joint_alternating/SCOPE_OVERRIDE_20261020.json)与[最终账本](joint_alternating/ALLOCATION_FINAL_R2.json)。335件归档文件和14份实际CPU权重/状态回读均通过；未追加训练或重复测试。

## 要回答的问题

同一商品的4个SID经过查表后、压缩为一个历史商品表示之前，增加一层内容相关的注意力，能否改善DiffGRM推荐效果？原来的item_mlp和Decoder已经存在交互，本轮只检查额外注意力的价值。

## 固定设置

- 复用03阶段Beauty PSE + DiffGRM基线 `diff_formal_pse_4x256_repo_repro_v6_bf16`，不重训原版。
- 原PSE/OPQ/PQ量化资产和4×256正式SID固定；推荐器embedding、Encoder和Decoder从头训练，seed2024。
- 历史embedding `[B,L,4,256]` → `[B×L,4,256]` → 1层4头局部残差注意力 → 原形状 → 原item_mlp。新增dropout为0，输出投影零初始化；PAD的残差更新归零。
- 仅商品内部交互；不跨历史商品，不更新全局码本，不改变Decoder、OCN、损失或CPD。
- 配方从03阶段三层证据配置和锁定源码默认值重建；batch64、累积16、BF16、LR0.01、warmup10000、最多200轮；第20轮起每5轮验证，warmup后patience5；按 `0.8*NDCG@10+0.2*HR@10` 选模，验证beam32、测试beam256。
- 一套配置、一个新训练；单H100合计最多8 GPU小时，含smoke及评估。达到预算仍未完成时保留断点并明确报告。

## 预先约定的判断

最终测试NDCG@10相对原版提升至少3%，且HR@10不下降，视为值得复验的初步信号。没有提升也正常结束，不追加搜索。此标准不是统计显著性检验。

基线测试HR@10为0.07490050529893127，NDCG@10为0.04391882613008758；完整结果来自03阶段results.csv。

## 证据边界

历史runtime commit和原始完整解析配置未记录，资产锁来自迁移快照；本轮保留这两项缺口。同seed不能证明与历史模型严格同初始化，也不能排除新增参数和优化路径的影响。全部指标为SID-level，存在SID碰撞，不与06的item-level排榜。本轮不直接验证连续生成或解码提速，也不替代05的共享起点码本对照。

## 当前进展：累计28 GPU小时已批准，准备修复后H100验收与两组训练

用户已选择先修复，再确定必要对照和新增预算。开发引擎已提交`8a48a30`，仅在OCN teacher评分中禁用autocast缓存，继承外层精度。原8项模型检查、新增4项CPU BF16梯度/更新/真实保存恢复回归全部通过，见[本地验收](execution_acceptance/repair_planning_20261020/LOCAL_REPAIR_ACCEPTANCE.json)。H100修复与正式训练尚未验收，远程旧执行副本和84轮状态均保留，未提交新GPU。

[修复后计划](REPAIR_PLAN.md)只新增两组同修复、同seed、共有参数/随机状态相同的从头训练，用注意力开关作唯一差异；新A作为主要基线，03仅作背景。用户已批准先最多0.5 H100 GPU小时验收测速，整个Goal累计上限由8调整为28 GPU小时，旧3.1314小时计入；通过梯度、恢复、共同起点、实测预算及合法平台策略检查后自主推进正式两组。[授权记录](execution_acceptance/REPAIR_BUDGET_AUTHORIZATION_20261020.json)。下方8小时/不重训基线的内容均为旧实验历史协议，以本段修复后协议为当前范围。

## 诊断历史：OCN/BF16梯度路径问题

CPU已读到完整84轮/8,736次更新的断点，测试尚未开始。恢复核验发现Adam参数组90项仅有50份状态，缺少四层Decoder的40个注意力/MLP线性参数状态。锁定原版与交互版的CPU合成对照都能复现BF16＋OCN缺梯度；teacher打分禁用autocast缓存的候选修改在四层CPU模型中恢复了全部梯度、首次loss不变。实际冻结源码没有修改，未启动r2或其他GPU作业。

这不能证明未知历史runtime的03正式训练也有同一问题。修复后从84轮继续会混合训练行为，需要重新确定同修复实现下的对照和预算；已向用户询问范围选择。详细证据与未应用补丁见[诊断记录](execution_acceptance/ocn_gradient_diagnosis_20261020/README.md)。目前不能称为完整训练验收通过或Goal完成。

## 平台中断与恢复准备（未提交r2）

第一正式Job `nancy-stage07-item-local-formal-20261020-0045` 已被平台自动回收并为CANCELLED，04:07:54 Pod删除。原因是当前公共训练空间的 `gpu < 40% for 3h` 回收规则，历史五分钟GPU采样平均约16.1%；不是训练代码异常，也不是观察超时。见[终态](execution_acceptance/FORMAL_CANCELLED_STATUS_20261020.json)、[事件](execution_acceptance/FORMAL_CANCELLED_EVENTS_20261020.json)、[当前规则](execution_acceptance/RECOVERY_RESOURCE_POLICY_20261020.json)。

训练CSV已到85轮，验证85中断；完整恢复轮次仍待CPU读取last.pt确认。两次短测和第一正式Job累计3.131388889 GPU小时，剩余4.868611111小时，见[接续前账本](execution_acceptance/RECOVERY_PRIOR_ALLOCATION_AUDIT_20261020.json)。先核对并保全断点，接续同一run，不重训基线或从头开始。下方RUNNING及20轮验证均为较早历史快照，不能作为当前状态。

## 执行与产物（第一正式Job历史快照）

入口为 `scripts/train/run_item_local_attention.py`，实验配置为 `configs/07_item_local_attention/item_local_seed2024.json`。本轮独立执行副本与输入保存在个人共享目录 `experiment1_stage07/`，历史主项目只读复用；确切run_id、版本、路径、作业、计时和运行状态保存在本目录后续验收收据中。

完成必须有：有效训练曲线、原规则验证选模、最佳权重重载、最终一次测试、输入/版本身份、作业终态及GPU释放、本地必要产物核验。代码或smoke通过不代表正式训练完成。

当前状态：16项本地/远程CPU检查通过，无跳过；真实GPU短测完成32batch/2update，参数更新和最佳权重重载通过。初次短测在训练前因平台WORLD_SIZE=1触发旧入口误判DDP而失败，已修复子进程启动环境，模型/配置不变。唯一正式Job `nancy-stage07-item-local-formal-20261020-0045` 继续单H100训练。执行根提交`8b7b9af`，引擎`1dc2a0c`，具体版本、预算与作业收据见execution_acceptance。`engine_changes.patch.gz`保存相对锁定底座`39e94e6`的完整引擎差异，字节身份见SOURCE_BINDING收据。

2026-10-20首次验证进度：同一正式Job仍[RUNNING](execution_acceptance/FORMAL_STATUS_FIRST_VAL_20261020.json)，第20轮首次完整验证HR@10为0.059607387202074855、NDCG@10为0.040013885470594035、weighted_score为0.04393258621668883，耗时212.7286秒。[快照检查](execution_acceptance/first_validation_20261020/SNAPSHOT_CHECK.json)核对了12项有限指标、别名、加权分数和实际解析配置；验证beam32、无batch截断，训练计时覆盖1–20轮、每轮1652batch。该结果仍在warmup内，是早期验证，不与03最终test直接判断提升；完整训练、最终选模和测试仍待完成。

两次短测已终止且Pod删除，按Scheduled至SuccessfulDeletePod计算合计388秒、0.107778 GPU小时；这不是结算金额，也不包含仍运行的正式Job。见[短测资源账本](execution_acceptance/SMOKE_RESOURCE_ACCOUNTING_20261020.json)。本地Job等待命令曾因900秒观察窗口结束而超时，已重新确认原Job仍RUNNING；不得把观察超时当作训练失败或重启依据。

## 训练结束后的CPU交付检查

以下工具已准备，不代表正式训练或交付已通过。先确认正式Job终态和GPU释放，再在已有CPU桥接上运行打包器，不更新正在训练的执行副本。

- `scripts/analysis/pack_item_local_delivery.py`：复核12件运行输入字节，打包最佳权重、完整last状态、曲线、配置、三件轻量输入及冻结源码，产生archive和SHA收据。
- `scripts/analysis/verify_item_local_archive.py`：下载后先验证包和manifest，再向全新本地目录安全提取、逐件核对字节。它只证明产物字节对应。
- `scripts/analysis/audit_item_local_delivery.py`：在CPU上核对原配方、输入身份、完整曲线、验证选模、一次最终测试、权重和恢复状态内容。它不代替平台资源释放证据。
- `scripts/analysis/audit_item_local_resources.py`：从各个Job的原始终态和单Job事件收据重算分配用量。未提供正式终态和删除事件时，只能给出短测小计。

GPU执行源码仍固定为`8b7b9af`/`1dc2a0c`；这些后续CPU工具单独记录版本，不同步进运行中的执行checkout。最终是否完成以真实产物和分项核验为准。

07阶段SHA绑定的收据和快照通过`.gitattributes`保留原始字节，避免Windows/Linux检出时自动改换行。[本地Git字节核对](execution_acceptance/GIT_BYTE_PRESERVATION_20261020.json)确认了30件暂存证据与工作文件一致、24份原有JSON的值未改，以及12项引用SHA仍匹配；这只涉及归档，不改变当前训练配置和执行源码。
