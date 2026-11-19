# R35三组冻结续训：逐项交付验收清单

**2026-10-30 10:33增量签收：** 原A/B/C均SUCCEEDED；四次已知GPU分配（含短测）已核资源释放，总计78,518秒＝21.810555556 GPU小时，结算仍未核实。证据为`final_lifecycle_r1/{STATUS,EVENTS,RELEASE,GPU_LEDGER}.json`及独立`ACCOUNTING_REVIEW.md`。C五关口、R35训练终态和共同网格best R23有实际证据，test R35/R23预测已生成，尚未签收其逐样本复算。最终归档已成功pack（6,846,443,520字节、3,405成员、63不可覆盖状态、外部依赖已在远端核SHA），当前仅下载进行中；不把pack成功或本地环境准备当成整包到达、成员验收、66状态检查或实际恢复。后续仍须完成C分支及跨组统计、独立复核、全部恢复、图表报告、C离线诊断补充、Git交接和跟踪停用。以下旧签收条目按各自时间理解。

**当前结论：部分通过，最终验收待完成；不能宣布Goal完成。** 本清单于2026-10-29整理，对照[Goal九节全文](GOAL_OBJECTIVE.md)及H1的R29/R32/R35及时通知补充。它是待后续签收的工作清单，不是最终PASS收据。

**16:50增量签收：** B整包下载及43成员独立解压已通过，4份真实逐样本指标审计与2份用户聚类bootstrap已通过，并已向用户报告；B实际新模型恢复仍待最终验收。H1 R32完整验证的来源SHA及协议已核并及时报告，独立逐样本复算/测试仍待完成。下文原快照保留时间语义，新增签收以本段及末表的新证据为准。

**16:57增量签收：** `C_R11_REPLAY_EVIDENCE.json`核齐真实来源SHA和已执行的重放门槛：全71,657条Top5一致、复算NDCG5差0，通知已发送。仅R8/R11通过，不能代替R14/R17/R20和最终全轨迹验收。

**17:28增量签收：** H1 R35训练完成已即时通知并记录`training_R35`。`H1_TRAINING_R35_EVIDENCE.json`核本次6,237条R更新连续至24,255、全程T200，以及终点不可覆盖完整状态收据和实际checkpoint SHA。R35完整验证、最终选模、测试、完整状态实际恢复仍待，不能将训练边界通过扩大为模型或整个Goal验收。已审查最终备份说明`BACKUP_USAGE.md`，命令准备不等于归档已经执行。最新平台状态`STATUS_20261029T172500.json`确认A/C运行、B成功；台账按查询开始时刻保守计8.5475 GPU小时，非最终时长或结算。

**18:07增量签收：** `H1_R35_STAGE_EVIDENCE.json`（SHA `0c8ccbb09c3b0ef69c70f3aca2d37ea59d3e76f24d032b7cae32b4c3107b75f9`）核齐R35全71,657条验证的来源与协议、六节点共同网格的真实summary/选模/所选权重SHA，最终best=R32；R35 NDCG@5=0.0104455471，比R32回落2.28%。已通知`validation_R35`。A当前执行固定R35及best R32测试；逐样本独立复算、配对区间和最终恢复仍待。H1 best不是R35，后续须按用户补充要求增加best R32−旧H0 R32的CPU配对比较，不能拿固定R35比较代替。

**18:19增量签收：** C R14重放关口通过，71,657条Top5全同、复算NDCG5差0，来源SHA已核并即时报告，见`C_R14_REPLAY_EVIDENCE.json`。C已进入R15/T160；R17/R20及最终全轨迹仍待，不宣称bitwise找回旧状态。A测试进行中，未签收新测试指标。

**19:10快照及A导出增量：** `OBSERVATION_20261029T191029.json`记录A训练及R29/R32/R35验证已结束、验证best=R32，固定R35和best R32两份测试预测均已完成，`independent_audit_pending=true`；尚未签收A测试指标、配对区间或模型恢复。`A_RELEASE_EVENTS.json`及`STATUS_20261029T190527.json`确认A已成功退出并释放；`A_EXPORT_PACK_RECEIPT.json`记录1,341,849,600字节、70个payload文件的单臂完整状态包，当前下载中，不能把远端打包成功当作本地下载/解包通过或全Goal备份完成。C已到R17/T160，R17验证快照为18,016/71,657，R8/R11/R14关口已通过，R17/R20仍待。

**原始核验范围（保留历史时间）：** 最初仅核对现存本地文件、实际收据及小量来源SHA，没有查询平台、下载、解包、重跑训练/推理/bootstrap或模型恢复。当时进度依据`LIVE_LATEST.json`的**08:22:43 UTC**快照，资源依据**08:05 UTC、分钟精度**快照，下载依据`B_RESUME_DOWNLOAD.log`。这些历史快照不再代表当前状态；本次文档增量只读取本地新证据，B后续已完成的真实下载、解包及独立指标验收按其专门收据列明。

状态含义：**已通过**仅覆盖所列证据范围；**待完成**表示工作尚在推进；**证据不足**表示不能靠摘要、工具源码或合成检查替代实际收据。

**19:37增量签收：** A固定R35及验证best R32与三个旧模型共5份测试原始预测CPU复算通过，6项用户配对区间完成，已在C结束前单独报告。`early_a_cpu_r2/analysis/COMPLETE.json` SHA `301fd4519a5db9d1685acf456221c471fcc352584c041ccc2da8789a04b40256`；43成员轻量包本地核SHA通过。旧R23/R26验证在该提前流程只核选模来源，正式本地A将通过版本化兼容入口复用已接受审计。完整模型下载和真实恢复尚未签收。

**19:42增量：** `C_R17_REPLAY_EVIDENCE.json`核齐R17真实源SHA及执行门槛，71,657条Top5全同、复算NDCG5差0，已即时报告；C接着进行最后40T及R18—R20，R20冻结门槛仍待。

**19:59增量签收：** A完整单臂包已下载完成，`A_DOWNLOAD_BYTES_VERIFIED.json`记录19:51:46完成整包及内嵌manifest SHA核验；`A_EARLY_TRANSFER_VERIFICATION.json`支持新鲜目录`outputs/stage09_restore/h_freeze35_20261029_early_A`内70个payload全部安全解包/落盘SHA通过。`final_analysis/A/analysis_report.json`于19:56:30生成、status=`passed`，SHA为`66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937`：11份审计为**9份本次真实复算＋2份已验收历史验证审计复用**，另有4份配对区间。A/R23、R26验证缺groups的兼容仅复用已锁SHA的旧审计，明确`recomputed_in_this_run=false`；5份test本次均实际复算，与提前CPU结果一致。程序exit0、报告/11审计/4比较及CSV/Markdown来源SHA已核，正式分析的独立交叉复审仍在进行。**这仍不是全Goal归档、63份immutable＋3份last状态字段验收或关键模型/双优化器/scheduler/RNG实际恢复；这些步骤尚未执行。** 下文早期“下载中/正式审计待”的带时间快照保留，此段和更新后的当前表为准。

## 一、接手、历史底座及输入绑定

- **已通过：真实起点、数据和PSE的输入审计。** `INPUT_AUDIT.json`、`INPUT_AUDIT_ASSETS.json`、`INPUT_AUDIT_COVERAGE.json`记录3个起点、A的旧R20祖先、8个数据文件、12个PSE资产及原始源码绑定；输入审计只做状态读取，不冒充新模型恢复。
- **已通过：科学底座与恢复扩展分开。** `EXECUTION_PACKAGE.json`绑定结果为`verified_committed_package`，提交`e4ca4b32920c95ef7190ad475d18e6f31fa3d276`；保留3029份历史文件，新增5份扩展。执行manifest SHA为`9e50d158a68f20232119edee42fae6f5c42a518a27d889b835144aa33b51f28b`。`GIT_INITIAL_STATUS.txt`保留初始改动范围，不能把当前开发HEAD替换历史科学底座。
- **已明确边界：原H0 R20完整权重缺失。** `../H0_R20_REMOTE_CHECK_20261029.json`及输入审计支持从C自己的R5重放，不支持声称逐位找回旧R20。
- **待最终复核：** 最终包中原始源码/配置/数据绑定与执行manifest一致，历史和其他任务改动未被覆盖。此处不把旧报告的PASS扩大为本轮产物PASS。

## 二、三组对象与实际训练完成情况

| 分支 | 认证起点与目标 | 当前证据支持 | 尚欠 |
|---|---|---|---|
| A / H1-Freeze20-R35 | H1 Freeze R26，18,018R/200T、phase33；新增9R至R35/T200 | 24,255R/200T、best=R32；平台SUCCEEDED、exit0并释放；70个payload下载/新鲜提取、11审计（9新＋2历史复用）/4区间及20:02独立交叉复审均通过 | 全状态字段/冻结指纹与真实模型恢复、纳入全Goal完整备份 |
| B / H0-Freeze32-R35 | H0自身best R32，22,176R/360T、phase41；新增3R至R35/T360 | 24,255R/360T；平台SUCCEEDED、exit0并释放；早期单臂包43个payload下载/解包验字节通过，`final_analysis/B/analysis_report.json`的4份真实复算及2份bootstrap通过 | 最终全部检查点状态核验与真实CPU模型恢复、纳入全Goal完整备份 |
| C / H0-Freeze20-R35 | H0 R5，3,465R/0T、phase5；重放至R20/T200，验收后冻结至R35/T200 | 五关口及phase5—24全部20阶段重放轨迹通过；21:34:17固定读数为R23第152/693次更新、累计15,398R/200T，尚无R23完整验证；R20 pre-freeze字节已核 | 冻结续训至R35全程、余下5份完整验证/选模/测试、全部状态字段及真实模型恢复 |

不得把保留Mixer的A写成H0，也不得通过关闭A的Mixer冒充C。B为不同起点、不同冻结时点的补充基线，不能作为纯交互效应对照。

## 三、恢复、时钟、学习率与训练设置

- **已通过：本地小型CPU回归。** `CPU_TESTS.json`及其原始`CPU_TESTS.log`记录40项通过，包含16项真实小模型controller、12项mock runner、12项轨迹/顺序检查；另有`PREFLIGHT_CPU_TESTS.json`的7项入口检查。这些不是完整KuaiRec正式模型验收。
- **已通过：三个真实父状态的CPU装载。** `cpu_restore_r2/CPU_RESTORE_AUDIT.json`及A/B/C各两次`*_load*.json`支持真实模型/双优化器/scheduler/计数及CPU RNG装载，未执行训练、未恢复CUDA RNG；不能替代最终新检查点恢复。
- **已通过：真实H100短验收。** `preflight_r1/{A,B,C}/report.json`、`preflight_r1/report.json`、`supervisor.json`及`H100_ACCEPTANCE_AUDIT.json`支持磁盘恢复后下一步一致性、冻结隔离、C实际40T后进入R、BF16训练及FP32生成诊断不改变训练状态。三个短测均回到认证父状态，短测bundle拒绝正式恢复。
- **待正式全程核验：** 原35R余弦日程继续而非重启；45阶段编号与种子/顺序保留；A/C最终T200、B最终T360；跳过T不更新优化器或T噪声时钟。须核正式训练日志、全部状态与指纹，不能仅凭短测通过签收整段训练。
- **C重放至冻结关口已通过，冻结后全程仍待完成：** `C_R20_REPLAY_FREEZE_EVIDENCE.json`已核phase5—24全部20阶段的原日志prefix SHA、逐更新整数时钟/数量/顺序及轨迹收据，loss和gradient_norm的最大/平均绝对差均为0，低于原锁定容差`1e-6 + 1e-4*abs(reference)`及`1e-6 + 1e-3*abs(reference)`。R8/11/14/17/20五节点各71,657条Top5全同、独立复算NDCG5差0，满足原身份/一致率门槛；R20冻结接受标记及pre-freeze字节已核，21:04已进入R21/T200。此为工程重放关口通过，不是缺失旧R20权重的bitwise恢复，也不是新checkpoint的实际模型恢复。

## 四、三卡并行、预算与资源

- **已通过：受控提交与短测释放。** `FORMAL_DRY_RUN.json`、`FORMAL_SUBMISSION.json`及`FORMAL_STATUS_INITIAL.json`支持唯一A/B/C三个独立单H100作业；短测`PREFLIGHT_EVENTS.json`/`PREFLIGHT_STATUS.json`支持219 GPU秒、成功退出及释放。`BUDGET_ESTIMATE.json`和`EXECUTION.md`记录测速、保存/传输/恢复开销、25%余量及用户批准**累计30 GPU小时，含短测**，不是30小时另加0.5小时。
- **已通过：B释放。** `B_RELEASE_EVENTS.json`、`STATUS_20261029T1605.json`支持B退出0、16:03:28删除Pod、16:03:38末次JobTerminated。Scheduled至末次终止为**6,968秒**；Scheduled至平台finished为6,949秒，平台created至finished为6,956秒，口径分开。
- **历史运行快照：** 08:05:00 UTC保守分钟快照含219秒短测及A/C各7,050秒，累计21,287秒=5.913055556 GPU小时；该时点预算余24.086944444小时。17:24:02北京时间的后续快照累计8.5475 GPU小时。两者均为已被新证据更新的历史观察，不是实时或最终余量。
- **已通过：A释放。** `A_RELEASE_EVENTS.json`及`STATUS_20261029T190527.json`支持A退出0、19:03:01删除Pod并记录末次JobTerminated；Scheduled至末次终止为**17,731秒=4.925277778 GPU小时**，Scheduled至平台finished同为17,731秒，平台created至finished为17,738秒=4.927222222小时，各时间口径不能等同已核实结算。
- **21:45已归档台账快照：** `GPU_LEDGER.json`按`STATUS_20261029T214546.json`的查询开始时刻**13:45:43.777156 UTC（21:45:43.777156北京时间）**向下取整。短测219秒＋B 6,968秒＋A 17,731秒＝已完成24,918秒；C仍运行，观察到27,493秒。累计**52,411秒=14.558611111 GPU小时**，该快照下30小时额度余**55,589秒=15.441388889小时**。C随后仍消耗，此为时点算术而非当前可用余量；`cumulative_is_final=false`、`all_goal_jobs_released=false`，结算尚未核实。
- **待完成：** C终态及释放、全部失败尝试与最终累计预算重算；排队/0GPU文件入口不计GPU，不能用训练脚本seconds替代分配时长。费用按已记录1点/GPU小时估算，**平台结算尚未核实**，不可写成实际已结算费用。

## 五、完整验证、候选协议与选模

- **已锁定协议：** `PROTOCOL.md`、实际执行manifest及冻结源码；KuaiRec，seed2024、历史最多50件、目标5件、1层Encoder/4层Decoder、hidden256；BF16训练micro256×4；FP32/关闭TF32、beam32/batch32评估。最终仍须逐份summary与原始预测核实相同协议。
- **部分已完成：** 新增全验证A R29/R32/R35、B R35、C R8/R11/R14/R17/R20已完成。A正式本地分析已对R20/R29/R32/R35验证实际复算，对R23/R26验证复用已验收旧审计，全部绑定原始来源SHA，20:02独立交叉复审通过。B新R35及旧R32本地复算通过；C五个关口通过重放检查，不扩大为最终C分析通过。
- **待全部齐备：** 新增14份完整验证目标为A29/32/35，B35，C8/11/14/17/20/23/26/29/32/35，每份71,657样本/4,733用户。当前9/14份已有完成证据，其余C23/26/29/32/35五份待完成；复用A20/23/26及B32须保持锁定旧源SHA。
- **A选模来源已核、C待完成：** A/C共同R20/23/26/29/32/35按验证单列表NDCG5选best，精确平局取早；A六节点选模已完成，**best=R32**，来源/所选权重SHA见`H1_R35_STAGE_EVIDENCE.json`。固定R35作为主比较，重点A−C。B只并列旧R32与新固定R35，不称“从共同网格选出的best”。
- **待完成：** 同时报单列表HR5/Recall5、原聚合HR/Recall/NDCG10/20/50及候选覆盖，保持评分、去重、碰撞消歧、PAD/历史过滤。聚合@50不能称为一份50商品列表。

## 六、独立测试、复算与配对区间

- **B下载、解包及独立指标已通过。** `B_DOWNLOAD_BYTES_VERIFIED.json`、`B_EARLY_TRANSFER_VERIFICATION.json`支持614,389,760字节单臂包及43个payload文件的字节验证；`final_analysis/B/analysis_report.json`与`EXECUTION_RECORD.json`记录R32/R35的validation/test四份逐样本复算通过、0不匹配，及两份2,000次/seed2024/95%用户聚类配对bootstrap通过。测试NDCG5相对旧R32为−1.7444%，95%差值区间跨0；验证相对−3.1986%，区间全负。16:47:58已向用户报告。远端`job_result`保留的`independent_audit_pending`是生成阶段状态，不能覆盖本地后续通过收据；真实模型恢复仍待。
- **A单臂字节、正式指标及独立交叉复审已通过，真实模型恢复仍待。** `A_DOWNLOAD_BYTES_VERIFIED.json`及`A_EARLY_TRANSFER_VERIFICATION.json`核齐1,341,849,600字节、70个payload的下载/独立解包；整包SHA为`da875205f9635ea59d6b47f752c32c916b51083c7e9ca75ac964bb9925b93eea`。`final_analysis/A/analysis_report.json`记录11审计（9新＋2历史验证复用）和4配对区间，5份test本次均实际复算；报告来源及产物SHA和20:02交叉复审已通过。早期CPU流程已另行完成5test/6区间并提前报告；正式4区间与早期6区间是各自规定范围，不能混计为10个不重复比较。模型包receipt中的`metric_audit=false`只描述打包时范围；`model_restore=false`、`full_goal_backup=false`仍成立。
- **用户新增验证时间分段诊断已通过并已报告。** `temporal_validation_r32_20261029/COMPLETE.json`、`VERIFY.json`核三份正确模型（旧H0 best R32/T360、旧H1 R20/T200、冻结H1 R32/T200）的37指标按原始保存的商品ID/真值复算，保留旧边界`1598745441188500`微秒、前/后段38,644/33,013样本和4,011/3,827用户，95个跨界目标块仍按首目标归段；6项分段配对各12指标、2,000次/seed2024/95%用户聚类，主口径为样本均值。旧两模型分段NDCG复现误差0，37指标加权全量最大误差`2.22e-16`，独立复核通过。冻结R32后段对自身R20为+14.46%、对旧H0 R32为+8.77%，两区间均正，用户已收到主结果。两段均参与过完整验证选模；这是事后单seed诊断，不是新的独立未来测试，也不确定唯一适应机制或Mixer因果。
- **固定R35 A−B提前补充配对已通过，仅保存供最终报告使用。** `early_ab_comparison_20261029/COMPLETE.json`状态`passed`：复用已接受的4份分支审计，原预测上完成test/validation两份各37指标、2,000次/seed2024/95%用户配对聚类区间；82项输入/实现SHA后核不变，独立review通过，未重复独立审计或GPU推理。测试NDCG5 A高5.9055%，差值区间`[+0.00005819,+0.00092402]`；验证A高20.9746%，区间`[+0.00141017,+0.00221615]`。这不是相同冻结时点的Mixer纯因果对照；`all_ready=false`、`goal_complete=false`，不替代A−C主比较或最终all-ready绑定。
- **剩余测试与复算待完成：** 三组固定R35完整测试目标各53,194样本/4,153用户；A两份及旧起点测试已实际复算，B真实复算已通过；C固定R35及必要best尚待。A/C best若不同且不能复用，同协议各最多新增一次。旧A20/26及B32有效测试预测复用，不重复GPU推理；C R5只做验证比较，不补未授权测试。A正式分析交叉复审已通过；不能把已通过的A/B范围扩大为三组完整分析通过。
- **部分通过、最终all-ready仍待：** 固定R35 A−B的test/validation补充配对已提前通过；A−C主比较、C相对自身起点的适用split比较及A/C共同验证best补充尚待。仍采用用户配对聚类bootstrap 2,000次、seed2024、95%区间、样本平均；最终须绑定或复用已通过的A−B结果并对精确同模型比较去重，不能把提前两pair算作最终全组分析完成。
- **必须保留解释限制：** 单seed；R20冻结点源于先前H1观察；C是受控重放而非旧R20逐位恢复；B的冻结时点、T360及较低R32学习率不同。A优于B不能直接证明交互有效，A−C结果也不能写成多seed稳定收益；正负结果均按协议交付。

## 七、H1及其他分支及时报告

| 必须报告的事件 | 当前状态 | 证据与下一签收条件 |
|---|---|---|
| H1 R29全验证 | **已记录发送；正式本地复算及交叉复审通过** | 原阶段通知及stage evidence保留；19:56正式分析、20:02交叉复审通过，不是测试或最终Goal完成 |
| H1 R32全验证 | **已记录发送；正式本地复算及交叉复审通过** | `H1_R32_STAGE_EVIDENCE.json`核齐全71,657条、来源SHA及协议；NDCG5=0.0106897301；16:50阶段通知、18:07选模、19:56正式审计、20:02交叉复审分别有记录 |
| H1 R35训练完成 | **17:28:21已记录发送** | `NOTIFICATIONS.json`的`training_R35`、`H1_TRAINING_R35_EVIDENCE.json`核6,237新增R/24,255总R/T200和完整训练边界；此通知先于R35验证及测试，不等其他分支 |
| H1 R35全验证/最终best | **18:07:17已记录发送；正式本地复算及交叉复审通过** | 阶段验证/选模及所选权重SHA已核，best=R32；19:56正式本地审计、20:02交叉复审通过，与训练完成及测试通知分开 |
| H1固定R35及必要best测试独立核验 | **已完成并提前报告** | 5份test实际复算、6项配对区间和43成员轻量证据通过；19:35–19:37已报告旧R20/Freeze26/H0 R32对比，C仍待 |
| 其他组完成及时报告 | **部分通过：C R20关口亦已及时报告** | `NOTIFICATIONS.json`有B训练、B R35验证及16:47:58的`B_test_independently_audited`，并保留C R8/R11/R14/R17通知；21:04已在commentary告知C R20通过、进入R21/T200，证据为`C_R20_REPLAY_FREEZE_EVIDENCE.json`；后续节点仍待 |
| 用户新增时间分段主结果 | **已验收并已向用户报告** | `temporal_validation_r32_20261029/{COMPLETE,VERIFY}.json`支持旧边界、3模型及分段统计；编码端只读比较另有`ENCODING_FIXED_CHECK.json`；A−B提前补充当前只保存供最终报告使用，不混作此项用户通知 |

**heartbeat仍应保持：** `NOTIFICATIONS.json`记录`09-h1`、5分钟及去重事件；本次未查询或更改自动化状态。只有整个Goal最终通过后才停用，须留实际停用回读证据；仅H1或B结束不能停。

## 八、全部检查点、本地完整备份与实际恢复

- **待实际签收的完整覆盖数：63份不可覆盖检查点＋3份滚动last。** 当前实现对应A14、B6、C43；已包括各自`R35_training_boundary.pt`，C另有`R20_pre_freeze.pt`。原输入方案的59槽是较早最低口径，不能当作实际完整覆盖总数，也不能漏掉额外分界文件。
- **待完成：** 每R/每T（含跳过T的游标边界）逐一核“文件名—SHA—字段—R/T计数—阶段/下一恢复位置”；推荐器、码本/Mixer/SID、双优化器动量、scheduler、Python/NumPy/Torch/CUDA RNG、源/配置/数据绑定全部完整。best/last不能替代历史节点。
- **C R20冻结边界字节已通过：** `C_R20_REPLAY_FREEZE_EVIDENCE.json`核`R20_pre_freeze.pt`为79,642,340字节，SHA `686388225489d3d264fd1a25edd0bf69ba2171358b2845f54aed9fa7a5775e2b`；收据为phase25/update0、13,860R/200T，并列出model/alignment/双优化器/scheduler/RNG/binding字段。此处只签收原始字节及收据绑定，实际模型/优化器装载仍待最终统一恢复。
- **B早期单臂交付部分通过：** 完整包与43个payload的本地下载/安全解包验字节已通过，真实分析已读取complete/selection/coverage等绑定并完成4份复算/2份bootstrap。上述通过不等于全部状态字段与张量恢复验收；B真实模型、双优化器/scheduler/RNG恢复仍待最终独立执行。
- **A单臂字节、正式分析及交叉复审已通过：** 70个payload的实际下载/新鲜安全解包已完成，正式分析已读取complete/selection/coverage及相关checkpoint的路径/字节SHA并通过9新审计＋2历史复用、4配对区间，20:02交叉复审亦通过。分析器对checkpoint的SHA绑定不等于`torch.load`后的模型、双优化器/scheduler/RNG完整恢复；**全Goal的63＋3状态字段检查和关键实际恢复仍尚未执行**。单臂包不是全Goal完整归档。
- **编码端两端点的实际只读比较已通过，不是模型恢复。** `temporal_validation_r32_20261029/ENCODING_FIXED_CHECK.json`以CPU读取已验SHA的旧H1 R20及冻结R32状态，14张编码端张量（含基础码本/Mixer/item slots/SID）逐字节相同，10,728个商品SID变化0，完整T优化器亦相同、T计数始终200；推荐器85张状态张量均改变，R累计增加8,316次，推荐器内部SID embedding属于可训练R参数。该检查未构建/恢复网络或执行前向，不替代最终模型、双优化器/scheduler/RNG装载与63＋3全状态验收。
- **待完成：** 正式完整模型/配置/源码/日志/预测/报告的最终归档，整包及每成员SHA、安全新鲜解压、容量与所有文件覆盖核验。`backup_delivery.py`、`export_branch.py`、`download_archive.py`的合成小检查均不得计为实际正式备份通过。
- **待完成：** 从独立解压目录逐份检查全部状态字段/形状/计数；三组起点、C R20、三组R35与选中best实际构建模型并恢复双优化器/scheduler/RNG，必要的旧A20/23/26状态按实际选择复用，同文件可去重。CPU检查须明确路径重定位和CUDA RNG未实际恢复，不能冒充CUDA精确续训。
- **外部依赖必须明示：** 最终工具当前保留data/PSE及部分历史预测为逐SHA验证的外部依赖；如未全部自包含，报告不得称“单包独立可恢复”，必须同时交付可定位、已验字节的依赖清单。早期B包更不包含完整运行源码和父/祖先/共享资产。

## 九、代码、最终图表报告、交接与结项

- **已通过准备范围：** 本轮科学扩展已提交`e4ca4b3`，执行包和提交/差异单独绑定；后续本地后处理提交不覆盖运行副本。原始核验时本地HEAD为`ac7c73d`，此为历史读取值，本次未据此重报当前HEAD；最终提交仍待后续交付完成。唯一运行根见`EXECUTION.md`/`INSPIRE.md`。
- **阶段报告飞书交付已通过：** 20:39的[`FEISHU_READBACK_AUDIT.json`](../progress_report_20261029/FEISHU_READBACK_AUDIT.json)确认初稿正文/标题/21张表全部单元格、4张图引用和图注回读一致；文档为[三轮实验阶段报告初稿](<feishu-link>)。本次交付只覆盖当前阶段报告，不代表C结果、最终全包、模型恢复或整个Goal完成。
- **待完成：** 真实三组R训练loss、实际R学习率及同坐标全验证NDCG5曲线；必须显示A历史20/23/26、B32、C重放及各自冻结点。`plot_results.py`和历史解析图不是本轮最终曲线。
- **待完成：** 本轮`REPORT.md`、固定R35测试/补充best及B32/35表、重要设置/起点/冻结点/更新次数表、全部checkpoint覆盖、完整本地备份与实际恢复收据、最终GPU时长/估价/结算边界；不能由单份summary替代原始证据。
- **待完成：** 更新`docs/guides/PROJECT_HANDOFF.md`、本轮执行/资产入口，必要文件Git commit及清晰交付索引；只提交本任务文件，保留历史和无关改动。结项前核没有新seed/Decay/beam实验或R35以外训练。
- **最终签收顺序：** A/B/C及评估齐备 → 原始预测/区间和图表验收 → 所有checkpoint与完整归档/真实恢复 → 全部资源终态释放/最终台账 → 报告与Git/交接 → heartbeat实际停用并回读 → 根线程按实际Goal状态结项。本清单不会自动更改Goal。

## 本次核验记录与后续签字栏

**22:11 C冻结端点检查通过：** `c_frozen_state_spotcheck_20261029/COMPLETE.json`（SHA `88ef17aad0caf867b79c15548176c0c94bc07131818e0db3d990c943dfc202c2`）绑定确切脚本、请求和实际RESULT，五文件SHA及结果已独立复核。CPU只读R20_pre_freeze/R21/R22：两个后续端点的14份alignment张量、商品SID及T优化器（3张量、33标量）逐字节不变，T200/DIGER200/400保持；R分别新增693/1,386次更新，85份推荐器参数均改变。检查点和收据前后SHA一致。这是三个保存状态的比较，不是模型或优化器实际恢复，也不代表全部63＋3状态检查已完成。

原始核验曾重新核对H100短测已有证据包的116,633字节整包SHA及15份已落盘成员实际SHA，并核对当时4项通知的来源SHA；该数量只描述当时快照。19:59更新仅读取A下载/提取收据、实际解压manifest/归档大小、正式报告及11份审计/4份比较和CSV/Markdown产物SHA，未重跑下载、解包、CUDA/CPU恢复或指标/bootstrap。CPU原始日志、真实父装载、H100短测、helper合成检查、B真实分析、A的9新审计＋2历史复用及其待完成交叉复审严格分开。

| 后续签收项 | 当前结论 | 实际新证据/签收时间 |
|---|---|---|
| A训练/全部验证/测试与提前通知 | **单臂字节、正式分析及20:02交叉复审通过；真实恢复待** | 提前通知已记录；best=R32；70个payload已下载/新鲜提取；正式报告SHA `66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937`，11审计（9新＋2历史验证复用）及4区间；早期5test/6区间另行保留 |
| B真实下载/解压/复算及R32补充比较 | **该范围通过，实际模型恢复另待** | `B_DOWNLOAD_BYTES_VERIFIED.json`、`B_EARLY_TRANSFER_VERIFICATION.json`；`final_analysis/B/analysis_report.json`及`EXECUTION_RECORD.json`：4审计无不匹配、2份2,000次bootstrap；16:48已报告 |
| C全轨迹/R20门槛/冻结训练与全部评估 | **五关口及20阶段重放轨迹通过；21:34:17为R23/u152、15,398R/200T，其余待** | `C_R20_REPLAY_FREEZE_EVIDENCE.json`与`EXECUTION.md`的21:30—21:34固定读数；R23完整验证未完成，完成计数仍9/14；余下冻结训练/选模/测试及真实恢复仍待 |
| 固定R35主A−C、补充A−B、best及bootstrap | **A−B两pair提前通过；A−C、共同best和最终all-ready待完成** | `early_ab_comparison_20261029/COMPLETE.json`核test/validation两pair、82项来源后核不变及独立review通过；目前保存为最终补充，未宣称全组完成 |
| 用户时间分段诊断与编码端只读比较 | **该新增范围通过，主结果已通知；不替代实际模型恢复** | `temporal_validation_r32_20261029/{COMPLETE,VERIFY,ENCODING_FIXED_CHECK}.json`；3模型×37指标、6分段pair及两真实checkpoint编码端相等检查通过 |
| 63不可覆盖＋3last完整检查、关键模型实际恢复 | 待完成 | 待填 |
| 最终完整归档/成员/外部依赖及独立解压恢复 | 待完成 | 待填 |
| 四次已知GPU分配最终台账、全部释放与费用边界 | 部分通过 | 短测、B及A已释放；21:45:43查询开始快照累计14.558611111 GPU小时，C仍运行并继续消耗，最终台账及结算待核 |
| 最终图表/报告/Git交接/heartbeat停用 | **阶段飞书初稿已交付，其余最终项待完成** | 20:39飞书回读收据通过；完整三组图表/报告、最终Git交接和heartbeat停用仍待 |

更新时应保留本次进度快照的时间含义，新增真实证据后再改变对应结论；不得因文件存在、进程exit0、helper通过或单臂结束而把整张清单改为PASS。

**20:02交叉复审通过：** 已独立核58项输入/实现SHA、11份审计与4项区间、六节点选模及真实本地checkpoint字节/val-test指纹；两份test各37指标与早期CPU完全一致，三项重叠NDCG5比较/CI完全一致，少数辅助浮点末位差不超过2.60e-18。A整包、70payload与B旧报告/工具绑定通过。此复审未实际加载或恢复模型，不扩大此前签收范围。

**20:39阶段报告飞书交付：** `../progress_report_20261029/FEISHU_READBACK_AUDIT.json`状态`passed`、时间`2026-10-29T12:39:11.462969+00:00`，SHA `3a7bcc156b9d7a5699ed969a9f80f426b3efdde48b3455519c3ab6bfdfdf2bcc`。已核本地源XML SHA `e3d65809c9b436394d35178797f0798364647ac11402f32c7a642f0294010d48`和原始回读JSON SHA `0cac19033901cf0742b323904292c41497b9f8717494310c98ea468ecb1a477c`与收据一致。21表、4图引用/图注及全部源文本回读通过；此为截至阶段进度的用户报告交付，**不是全Goal完成**。20:40:07本地进度快照只确认C R20训练完成/验证中，未签收R20 gate或任何未来训练/恢复结果。本次只改当前状态和新增交付记录，带时间的旧快照原文保留。

**21:04 C R20增量签收：** `C_R20_REPLAY_FREEZE_EVIDENCE.json`（SHA `e5d9c79eeb955432db1ad09d4cd489f0393d7ec57dbd7eebc36b451350a0058d`）已核R20完整验证/原预测身份、五关口、phase5—24全部20阶段原日志prefix/顺序SHA及loss/grad零偏差，冻结接受标记与79,642,340字节pre-freeze文件亦已核。用户已在commentary获知R20通过及R21/T200进度。20:40历史进度的固定来源是`OBSERVATION_20261029T204007.json`，不再以会更新的`LIVE_LATEST.json`代指该快照。新增完整验证计数为9/14；C剩余冻结训练/5份验证/选模测试、最终跨臂分析、全Goal完整备份及真实模型恢复均待完成，本次不宣布Goal完成，也未执行任何新恢复或指标计算。

**21:30—21:38新增诊断与提前配对签收：** 用户要求的时间分段主结果已报告，`temporal_validation_r32_20261029/COMPLETE.json` SHA `35af6d222835f0f985629cc2976a3fb5db03912bf4d301c6ddca319096876392`，`VERIFY.json` SHA `554a0be6a3ab00a6a0a99282a507ac1fd79e40aacf299771ef552bdb983a4277`；编码端只读比较SHA `e4caacd6598ade6d63720aa107f2e32baa4861c8cc1c3cf200ca65a4bb5e7010`。固定R35 A−B两份提前补充的`COMPLETE.json` SHA `3eec39e2a4b2b00a2ba1b3f1da6ab0e1db02995927dd41377ef86e8323c8101e`，82项来源后核与独立review通过，现保存供最终报告复用。C在21:34:17为R23/u152、15,398R/200T，R23完整验证未完成；9/14完整验证计数不变。以上均不替代A−C主对照、最终all-ready、完整归档、63＋3检查和关键模型实际恢复，不宣布Goal完成；本次只读收据并更新本清单，没有重跑统计或模型检查。
