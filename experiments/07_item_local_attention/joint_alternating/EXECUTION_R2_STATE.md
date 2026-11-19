# 07交替联合训练 r2 接续状态

最新记录：2026-10-20最终交付（Asia/Shanghai）。本文件保留历史进展，后文带时间的记录仅代表当时状态；以下最终记录优先。

- **实验及交付已验收。** 两组均45轮完成、最佳epoch25、各一次正式测试；B相对A测试NDCG@10 +0.424972%、HR@10 +0.279018%，未达3%初筛。全8个GPU作业已释放，累计26741秒=7.428055556 GPU小时。最终结果见`REPORT.md`，总索引`FINAL_ACCEPTANCE_20261020_r3.json`（SHA `199683917b01a933920b595baf30155031a7948758808908eab6181dff8ed23f`）。
- 本地`outputs/stage07_item_local_joint/delivery_20261020_final_r3/`已下载原始分卷并还原335件文件；archive SHA `cc60b6f1e317e10c0f0cc9fee7614be37ff092d66c8c827a02802d7eeae5b98a`，`verified/BYTE_VERIFICATION.json`为`BYTE_RESTORATION_VERIFIED`。`LOCAL_CHECKPOINT_READBACK_20261020_final_r3.json`（SHA `56b04052c206a5404297a461f3b387dede2deb07f47b2ba3affab03359056b5f`）实际CPU读取14份权重/状态并通过；`curves_20261020_final_r3/`保存真实曲线及输入/目视检查收据。两组最佳SID均10651 unique、2171/12101碰撞商品、721组、最大组40，见`BEST_SID_STATISTICS_20261020_final_r3.json`。13件原始输入已在远端核对并保存身份，数据/cache字节未打包；不声称完成异地重训恢复。不追加训练、测试或GPU。

- **真实成对产物审计r3已通过。** `JOINT_PAIR_AUDIT_20261020_r3.json`为`JOINT_PAIR_ARTIFACT_AUDIT_PASS`，SHA `89ef854c9aef7abc7ed8c9dc0ba357e13e92680cffa7de2c3a3114d5954e4822`。在仅将点积操作数按TF32格式舍入、norm保留原FP32的独立CPU复演中，四份最佳/末轮checkpoint的193616个码位在同一舍入模式下全部精确匹配；原FP32/FP64差异及r2失败收据保留。此证据不能还原历史CUDA标志或证明GPU内核逐位复演。训练权重、SID、测试结果和冻结训练代码均未改。后处理工具及6项CPU检查见`postprocess_tools_r3/`，独立复核见`FINAL_REVIEW_PRE_BACKUP_R3.json`。
- 远端完成包保留在`experiment1_stage07/joint_delivery_package_20261020_final_r3/`。下方带时间记录均为历史，尤其“审计未通过”“备份未完成”“Goal尚未完成”已由本页最终交付记录取代。

- 两组真实训练及唯一正式测试均已完成。B在20:23:06完成测试、formal wrapper成功，最佳epoch25；测试HR@5/10为0.05598533292/0.08035594509，NDCG@5/10为0.03868086625/0.04656315936。相对A的NDCG@10仅+0.424972%、HR@10+0.279018%，未达3%初筛。原始完成快照`progress_snapshots/item_local_formal_completed_20261020.json`、独立算术`RESULTS_COMPARISON_R2.json`，报告已补实际结果但交付仍未完成。
- B平台SUCCEEDED，20:23:56 SuccessfulDeletePod；Allocated 6903秒。最新完整清单8个作业全部终态，全部Pod删除收据已纳入`ALLOCATION_FINAL_R2.json`，总计26741秒=7.428055556 GPU小时，旧失败/中止/短测全部计入28小时上限。25份平台/授权/账本收据已按SHA核对上传并解开在远端`delivery/resource_evidence_20261020_r2/`，本地索引`RESOURCE_EVIDENCE_INDEX_R2.json`。
- 20:34历史记录：原始失败收据`JOINT_PAIR_AUDIT_20261020_2024.json`为“SID is incompatible with checkpoint codebooks”。CPU逐份实读四checkpoint后发现A最佳/末轮各1/4个码位、B各3/4个码位与CPU FP32/FP64最近邻不同（每份48404码位），最大距离gap约1.046e-4；不能简单按普通FP32误差放宽阈值。`SID_PRECISION_DIAGNOSIS_20261020_2030.json`保留全部失败位置与权重SHA。该问题已由上方r3量化算术检查处理，原失败记录不删除。

- 20:03:36实读：B完成逻辑epoch39及第7次全量验证，进入epoch40；截至39共30R/3120次推荐器更新、9T/54次码本更新。第39轮验证212.55秒，HR@10=0.0669409292134329、weighted=0.051688314521003004，仍低于第25轮0.05423174543616992。已观测验证最高分仍为25，最后一次验证在45轮后；尚无B正式测试。快照见`progress_snapshots/item_local_after_epoch39_20261020.json`。本转进为同一活跃作业的验证等待并获得新完整验证证据；尚需末轮训练/验证、最终测试、释放及全部交付核验，未追加训练。

- 19:48:54实读：B完成逻辑epoch32及第6次全量验证，进入epoch33；截至32共25R/2600次推荐器更新、7T/42次码本更新。最近Tokenizer阶段30的推荐损失独立梯度探针0.2523708940，更新/冻结/SID检查正常。第32轮验证222.43秒，HR@10=0.0684165809596208、weighted=0.05229283831752072，低于25轮0.05423174543616992；已观测验证最高分仍为25，最终checkpoint及测试尚待完成。快照见`progress_snapshots/item_local_after_epoch32_20261020.json`，下一次验证39，原配方/截止时间不变。本转进属于验证等待并获得新的完整验证证据。
- 本轮补齐根仓库代码差异归档：`root_joint_runtime_changes.patch`与`ROOT_CODE_DIFF.json`，明确base2fc97b6→target842ff2a、scripts/configs/tests范围8文件、补丁211469字节，SHA d115759fa4fc0e25c85c05b4a21685b4f2e3c9fe934bd33d455458d4bf5a47b3。这是提交区间的运行/检查代码差异，不是仅local开关的消融差异，未应用补丁或做恢复测试。已有3份engine差异链均保留。两文件已复制并核对SHA到远端`delivery/source_diff_20261020_1943/`，见`ROOT_CODE_DIFF_UPLOAD_R2.json`；会被最终packer收集，未修改冻结code/engine。实际本地权重回读、完整备份及B测试仍待执行。

- 19:33:55实读：B完成epoch25完整验证及epoch26第6个Tokenizer阶段，进入epoch27；截至26共20R/2080次推荐器更新、6T/36次码本更新。第26轮7.10秒，推荐损失独立梯度探针0.2816061080，更新/冻结/SID检查正常。第25轮验证220.83秒，HR@10=0.07150203461074095、NDCG@10=0.04991417263544923、weighted=0.05423174543616992。见`progress_snapshots/item_local_after_epoch25_20261020.json`。
- 完整B验证CSV已下载，19:35实读远端SHA与本地一致（c1ad3dc98e9ff2efaf070b1704ea4baa3f95b5e8266b7348ea22830d0d2c3c57）。19:36独立复核确认只含0/5/12/19/25的完整val组、无test，重算weighted一致（1e-7容差），截至25的CSV首次最大候选为25。同期A/B验证比较：B的NDCG@10相对+3.273892%、HR@10+3.562176%；duplicate@10增加，不称改善。见`progress_snapshots/item_local_validation25_review_20261020.json`及对应原始CSV。这只是中间验证信号，不是最终checkpoint验收或3%测试初筛达标。下一次验证为32，本转进为验证等待并取得新的完整验证与原始CSV比较证据；Goal仍需B最终测试和全部交付。

- 19:24:14实读：B完成逻辑epoch22（第5个Tokenizer阶段），进入epoch23。截至22共17R/1768次推荐器更新、5T/30次码本更新。第22轮7.12秒，独立推荐损失梯度探针0.3228918910，1个活跃码本参数张量改变、102个冻结参数张量未变，SID最近邻及映射检查通过；最新完整验证仍是epoch19，下一次在25之后。见`progress_snapshots/item_local_after_epoch22_20261020.json`。本转进为验证等待并完成交付运行说明`postprocess_tools_r2/POSTPROCESS_RUNBOOK.md`，已核对工具参数与实际部署位置；远端工具直接复用delivery/，本地大包及还原放outputs/下。未执行最终打包或实际正式权重回读，未改训练代码或追加GPU。

- 19:19:13实读：同一B组作业完成逻辑epoch19及第四次全量验证，进入epoch20；截至19共15R/1560次推荐器更新、4T/24次码本更新。第18轮Tokenizer的推荐损失独立梯度探针0.3685347736，更新/冻结/SID检查正常；第19轮90个活跃参数张量改变、13个冻结参数张量未变，并使用刷新后的同一SID。本次验证224.27秒，HR@10=0.06747752984841032、weighted=0.0516828951333272；只是训练中的验证结果，不是最终测试结论。快照见`progress_snapshots/item_local_after_epoch19_20261020.json`。下一次验证在epoch25后，仍按原配方继续；未增卡或重启。本转进为验证等待并获得新完整验证证据，同时新增`REPORT.md`阶段报告草稿，A数据已核对、B最终结果明确待填。完整备份与实际权重回读仍待两组结束，本地可用空间本轮检查约40.7GiB。

- 19:04:35实读：B完成逻辑epoch12及联合更新后的首次全量验证，已进入epoch13；截至12共10R/1040次推荐器更新、2T/12次码本更新。本次验证213.3205秒，HR@10=0.06698564593301436、weighted=0.05094987003682674；上次epoch5的weighted为0.04994123336140118。两者是训练中的验证变化，不是相对A的最终测试提升。见`progress_snapshots/item_local_after_epoch12_20261020.json`。本转进对同一RUNNING作业进行了验证等待并取得新的完整验证证据，未改配方、未增卡，Goal未完成；下一次全量验证在epoch19后。

- 18:58:30实读：B完成逻辑epoch10（8R、2T），epoch11约98%；第10轮Tokenizer 6.93秒，直接推荐梯度独立探针0.3081280291，1个活跃参数张量改变、102个冻结参数张量未变，SID映射检查通过。推荐梯度数值来自首batch的独立推荐损失探针，实际码本优化使用复合损失，不能把全部码本变化归因于推荐损失。下一次全量验证在epoch12后。原始观察及时间估计见`progress_snapshots/item_local_status_20261020_185830.json`。按当前速度、跑满45轮估计训练/验证/最终测试还需约80—100分钟（约20:20—20:40结束），不含最终下载归档与验收；不是完成保证。当前仅B占用1张H100，A已完成释放。本次实时quota再次确认可申请1/2/4/8卡规格，但冻结runner及恢复链路仅支持单进程/单卡，未新增申请。

- B最新：18:53:04确认第7轮推荐器也已完成104次更新（90活跃张量改变、13冻结张量未变、梯度有限，SID与第6轮刷新后相同），进入第8轮。至此正式首个rec→tok→rec切换完成，累计6R/624次推荐器更新、1T/6次码本更新。第6轮8.04秒、第7轮125.59秒，均未超分项预算。见`progress_snapshots/item_local_after_first_alternation_20261020.json`；下方18:50所述“第7轮尚未完成”是已被此新证据取代的历史快照。下一次全量验证在epoch12后。

- B最新：18:50:27实读完成5R及首次T（逻辑epoch6），进入epoch7推荐器；平台随后确认同一Job仍RUNNING。第6轮码本6次更新、推荐CE直接梯度0.3262924552，1个活跃码本张量改变、102个冻结张量未变，梯度有限、最近邻SID刷新与全库映射通过。原始及复核见`progress_snapshots/item_local_after_first_tokenizer_20261020.json`、`item_local_first_tokenizer_review_20261020.json`；平台回执`FORMAL_ITEM_LOCAL_STATUS_AFTER_FIRST_TOKENIZER_R2.json`。第7轮尚未完成，不提前宣称完整rec→tok→rec正式循环已核验。
- B epoch5完整验证用时212.04秒，HR@10为0.06649376202、weighted为0.04994123336；这是码本首次更新前的暖身验证，不能归因于联合码本更新。下一次完整验证在逻辑epoch12之后，继续观察同一Job，不改变配方/选模/截止时间。

- B最新：18:37:30实读已完成初始全量验证及第1轮推荐器，进入第2轮。首轮1652 batch/104次更新、142.33秒；90个活跃参数张量（含6个局部注意力张量）均改变、梯度有限，13个冻结张量未变。初始验证215.33秒，HR@10与weighted和A的epoch0逐值一致。见`progress_snapshots/item_local_first_epoch_review_20261020.json`及原始观测；首次Tokenizer阶段尚未到达，仍在5轮推荐器暖身。
- A结果已补充独立CPU元数据/CSV/选中SID/完整phase审查，见`control_final_r2_20261020/CONTROL_RESULT_REVIEW.json`。最佳epoch25的SID碰撞率17.94066606%（2171/12101），区别于末轮17.92413850%。未替代最终两臂实际权重回读和完整交付。

- control已完整完成45个逻辑轮（35R、10T），验证选择epoch25，唯一测试HR@5/10为0.05446496445/0.08013236149，NDCG@5/10为0.03809946608/0.04636611617；SID-level。见`control_final_r2_20261020/LIVE_COMPLETION_OBSERVATION.json`。平台SUCCEEDED，18:17:50 Pod删除，分配7892秒。
- 已释放历史总分配19838秒（5.510555556 GPU小时），见`ALLOCATION_AFTER_CONTROL_R2.json`。第二组仍计入同一个28 GPU小时上限。
- item_local已提交：`nancy-stage07-item-local-joint-formal-item-local-20261020-r1-r2`。最新平台RUNNING，18:28:53 Scheduled，子进程18:29:03开始；18:32实读确认初始化完成、正在epoch0完整验证。两组实际formal共有基础参数/RNG/初始SID三项SHA完全一致，见`FORMAL_PAIR_INITIALIZATION_MATCH_R2.json`。收据见`FORMAL_ITEM_LOCAL_SUBMISSION_R2.json`、`FORMAL_ITEM_LOCAL_STATUS_AFTER_SUBMIT_R2.json`、`FORMAL_ITEM_LOCAL_EVENTS_SCHEDULED_R2.json`。接续必须观察同一个Job，不重复创建。
- 实测第二组预算为9717秒（训练/验证/其他8643.02秒、测试1073.93秒），原9150/1200/10350秒窗口未改。真实预算及CPU guard均通过，源/输入/包环境与两组短测一致；独立数值审查通过。证据在`formal_preflight_item_local_r2_20261020/`。提交前18:26:58实时清单确认control释放后无同族活动作业或重复B。
- 后续：观察同一个B Job的完整交替训练及唯一测试；之后才做pair审计、总账、完整备份与本地真实CPU回读、曲线和报告。A完成或B开始均不代表Goal完成。本转进完成真实第二组提交与正式共同起点验证，属于progress。
- 用户问是否能申请多卡：当前Live配额有1/2/4/8 H100，但冻结入口明确要求exactly one visible GPU及single process。已解释多卡单模型需要分布式适配；本轮只剩B，保持已运行的单卡协议。没有把一般性询问当作追加实验或临时改DDP的指令。

以下为历史记录，从2026-10-20 16:07开始：

- 当前授权：同seed2024、同暖启动的交替联合训练两组，关闭全局Mixer，仅切换商品内局部注意力。累计28 GPU小时，旧消耗全部计入。不再使用最初固定SID方案。
- 代码：根 `842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1`，引擎 `1c879252f6395d6e64462511823c3929b9c42c00`。远程冻结根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261020_item_local_joint_r2`。另一任务在本地experiment1修改05 long；本轮不提交共享索引、不向冻结副本同步其变更。
- 两组真实H100短测已通过；共同初始状态/RNG、rec→tok→rec更新与冻结、推荐损失到码本梯度、SID刷新及阶段边界恢复见 `SMOKE_R2_PROCESS_RECEIPT.json`。短测不提供推荐质量结论。
- CPU重核对已通过，见 `formal_preflight_r2_20261020/FORMAL_CONTROL_CPU_REVALIDATION_V2.json`。先前98/110项包差异来自采集顺序：`accelerate`导入暴露12项`setuptools/_vendor` metadata；相同生产导入上下文下两组H100与CPU的110项完全一致。未修改环境或冻结训练代码。
- 第一正式Job已提交且RUNNING：`nancy-stage07-item-local-joint-formal-control-20261020-r1-r2`，单H100，平台Scheduled为16:06:18。原始提交、状态、事件在同目录`FORMAL_CONTROL_*_R2.json`。尚未有正式质量结论。
- 配方：05 `refined_joint`完整交替协议，最多45逻辑轮；不是05最新共享起点24次Tokenizer更新＋3轮恢复的四组短对照。batch64、acc16、BF16、val batch8；完整数据，未截断正式训练或验证。
- 对照组预计9653秒、注意力组9762秒；每个Job运行一个完整模型，进程上限10440秒、平台上限10620秒、训练截止前为最终评估预留1200秒。测算有短测外推限制，不保证实际耗时。不计划切片或额外负载规避平台回收。
- 正式前累计已释放11946秒（3.318333GPU小时），见 `ALLOCATION_AFTER_SMOKE_R2.json`。新正式分配从16:06:18另行累计；第二组需在第一组资源释放后，用实际总账更新启动参数。
- 下一步：观察同一对照Job及持久formal状态，勿重复提交；完成后保全结果与终态/Pod删除事件，再运行注意力组。最后远程CPU pair审计、分卷备份、本地逐字节恢复与实际权重/状态回读、资源账本、报告全部完成，才可结束Goal。

16:20更新：平台仍为RUNNING，已完成推荐器暖身第1—3轮，每轮1652 batch、104次优化器更新，累计312次；loss为3.59889、3.56028、3.52555。当前Tokenizer梯度为0符合暖身阶段安排；第5轮结束后才进入首个Tokenizer阶段。首轮训练147.37秒、完整初始验证240.89秒；按完整计时外推，点估计仍能覆盖本次截止时间，但继续使用原25%额外余量会略超训练时间窗，不能再称该保守界通过。见 `first_full_epoch_review_r2_20261020/TIMING_REVIEW.json`，下一次在第5轮及首个Tokenizer阶段后复核。

交付准备：远程`delivery/`已保存并核对pair auditor和packer字节；本地新增14份必要checkpoint的CPU回读工具及7项fixture正/负检查已通过并经独立审查，真实formal产物尚待完成后验收。工具源码、测试记录和哈希保存在`postprocess_tools_r2/`，不是冻结训练代码变更。03精确历史背景和比较限制保存在`HISTORICAL_BACKGROUND.json`。

16:33观测及随后复核：首个真实Tokenizer阶段（逻辑第6轮）已完成6次更新，推荐CE对码本的梯度范数0.3288477；1个活跃码本参数张量实际改变，96个冻结张量未变，最近邻SID与全库映射检查通过。随后第7轮推荐器完成104次更新，84个活跃张量均改变，Tokenizer保持冻结；已进入第8轮。平台随后仍为RUNNING，见`FORMAL_CONTROL_STATUS_AFTER_FIRST_TOKENIZER.json`。这是交替训练链路证据，还不是推荐效果结论。

完整计时复核保存于`first_tokenizer_review_r2_20261020/`：6个已完成推荐器轮次平均150.21秒，首次Tokenizer阶段9.48秒，两次全量验证平均243.97秒。全程点估计加500秒其他开销约7804秒；用各类阶段已观测最大耗时外推约8401秒，均低于9150秒训练窗口，但原25%额外余量公式仍为9630秒，不能称保守界通过。继续同一个有截止时间的Job，在下一次完整验证后复核，不改配方或截止时间，不提交第二组。

曲线工具也已准备并独立只读审查：`postprocess_tools_r2/render_joint_curves.py` SHA256为`a7c4b5bf3966644fd3e988972269f7d9b59e769a59924b73a78f605ef29c2b37`，同字节副本已核对在远程`delivery/`。其合成CSV测试不是真实结果；正式备份验收后才画推荐器loss、验证HR@10/NDCG@10，Tokenizer阶段断开，测试分数不参与选模或曲线。

只读CLI提示：`--json job status <Job名> --workspace 分布式训练空间`必须显式带workspace，否则本地回执工具会因无结构化结果而退出；这不影响远程训练。

16:49更新：同一control Job已完成逻辑第12轮及第三次完整验证，进入第13轮；最近平台状态仍RUNNING。第10轮的第二个Tokenizer阶段也通过更新/冻结/SID检查，累计Tokenizer更新12次。第三次验证244.54秒，前三次范围240.89—247.05秒。保存了`progress_snapshots/control_after_epoch12_20261020.json`和原始验证CSV；epoch0/5/12加权分数依次0.04868982、0.04909101、0.05094588，当前CSV首次最大值在epoch12，尚未核验最终最佳checkpoint；CSV没有测试行。

第二组的待执行预算方法见`SECOND_FORMAL_TIMING_RULE.md`，预先按control完整运行的最慢各类阶段、实测额外开销及paired smoke倍率复核；该文件不是启动通过收据。下一次完整验证预计在逻辑epoch19之后，继续观察同一Job即可。当前转进完成了新的验证证据核对和第二组复核方法记录，未提交新GPU、未修改冻结训练代码，Goal仍未完成。

17:04更新：control已完成逻辑epoch19及第四次全量验证，进入epoch20；平台原始回执仍RUNNING，见`progress_snapshots/control_platform_after_epoch19.json`。epoch18的第四个Tokenizer阶段完成6次更新（累计24次），直接推荐梯度0.3768847，更新/冻结/SID检查通过。epoch19推荐器完整1652 batch用时130.10秒，本次完整验证214.76秒；验证HR@10为0.06734338、加权分数0.05119140，高于epoch12的0.05094588，尚无测试结果。原始观测见`progress_snapshots/control_after_epoch19_20261020.json`。

本次转进属于对已确认RUNNING作业的验证等待，并取得新完成的阶段/验证证据。期间CPU日志通道曾重新发现缓存Notebook地址，随后读取成功，平台训练没有重启。未新增GPU或改训练配置；下一次完整验证在逻辑epoch25之后。本地可用空间约41.28GiB（本轮检查时）；必要的最终产物验收仍待两组完成。

17:22更新：control已完成epoch25及第五次全量验证，随后完成epoch26 Tokenizer并进入epoch27；同一Job仍RUNNING，见`validation25_review_r2_20261020/PLATFORM_STATUS.json`。保存了截至epoch25的完整计时、阶段审计和验证CSV，`TIMING_AND_PHASE_REVIEW.json`核对20轮推荐器/2080次更新、5个Tokenizer阶段/30次更新的更新、冻结、直接推荐梯度与SID一致性均有效；此计数明确截至25，不包括后来的26。epoch25验证HR@10为0.06904262、NDCG@10为0.04833184、加权分数0.05247400，是当前CSV首次最大值；最终最佳checkpoint尚未验收，测试CSV仍为0行。

本次完整计时复核：推荐器平均150.56秒，Tokenizer平均8.68秒，验证平均239.56秒。全程训练/验证点估计加500秒为7772.68秒，各类已观测最大值外推加500秒为8514.69秒，均小于9150秒训练窗口；原1.25额外余量公式仍为9590.85秒，不能改称保守界已通过。继续原有有界Job，不改截止时间。下一次完整验证在epoch32后。

第二组CPU启动核对工具已准备、做AST/--help检查并经独立只读审查；保存在`formal_preflight_item_local_r2_20261020/`，远程同字节副本在`delivery/joint_item_local_formal_cpu_guard_v2.py`，SHA256=`9c268ca2ccdf8759129c6f94d410eef60233f2526706babb0fba1b1d1f40ac9e`。它读取未来真实`FORMAL_BUDGET_ITEM_LOCAL_LIVE_V2.json`、检查control正式完成和item_local尚未启动、13件输入、完整配置及生产导入后的环境；未执行远程guard，未产生PASS，也未启动第二组。平台释放、实时规则和实际累计分配仍由父任务独立核对后再运行此工具。

17:39更新：control已完成epoch32及第六次全量验证，进入epoch33，平台仍RUNNING。快照见`progress_snapshots/control_after_epoch32_20261020.json`，平台回执见`control_platform_after_epoch32.json`。最近第30轮Tokenizer推荐梯度0.2501751、更新/冻结/SID检查通过；第31/32轮完整推荐器耗时146.62/139.71秒，本次完整验证218.17秒。epoch32验证HR@10为0.06846130、加权分数0.05222444，略低于epoch25的0.05247400；沿用原选模、早停和最大45轮规则，不因波动改配方。下一次完整验证在epoch39后，最终测试尚未启动。

本转进为对同一个明确RUNNING Job的验证等待，取得了新的阶段及完整验证证据；未启动第二组、未使用新增GPU，Goal仍未完成。

17:54更新：同一control已完成epoch39及第七次全量验证，进入epoch40；完整快照见`progress_snapshots/control_after_epoch39_20261020.json`。第38轮是第9个Tokenizer阶段，累计54次更新，直接推荐梯度0.1839899、更新/冻结/SID映射检查通过；第39轮推荐器1652 batch用时167.47秒，完整验证239.05秒。epoch39验证HR@10为0.06783526、加权分数0.05219104，当前CSV最大值仍为epoch25的0.05247400。未完成最终选模重载或测试。用户补充另一任务正在修改本地experiment1；已重新核对本地差异，保留共享目录的05 long、研究文档等改动，本轮冻结训练副本不接收这些文件。

18:06更新：control完成44轮，正在45轮（约39%）；第42轮为第10个Tokenizer阶段，总计60次更新，直接推荐梯度0.1868374，更新/冻结/SID映射检查通过。最后一次全量验证、最终选模重载和唯一测试尚待完成；第二组未提交。期间用户请求梳理FAVE论文，进行只读原文/官方代码分析，未改变当前实验协议。

第二组预算计算工具`.runtime/stage07/prepare_joint_item_local_budget_v2.py`已完成独立只读审查，SHA256=`a553646d9738d14c969bda7f1c21b162b67e1148a3a1cd7ab451eeb2bf5d2725`；仅本地准备，尚未上传或对真实control执行。需要完整control wrapper、真实COMPLETE/唯一test、包含control实际Pod释放的最新累计账本、实时policy、已定timing rule及原始`execution_acceptance/GPU_SMOKE_RESULTS_20261020.json`。严格按既定35/10/8及9150/1200/10350秒门计算；缺证或超界不写passed预算。之后仍需执行已有item_local CPU guard及实时资源核对，才可提交第二组。已把control实际使用的`FORMAL_BUDGET_CONTROL_LIVE_V2.json`下载至`formal_preflight_r2_20261020/`，旧预算文件不作为第二组新的准入。

18:42更新：同一B组Job仍运行，完成推荐器第1—3轮、进入第4轮。三轮耗时142.33/133.76/135.60秒；第3轮90个活跃张量均更新、13个冻结张量保持不变，梯度有限。初始SID保持不变符合暖身安排，首个Tokenizer阶段尚未到达。原始快照见`progress_snapshots/item_local_after_epoch3_20261020.json`。本转进为对已确认RUNNING作业的验证等待，并补充A结果的独立元数据/CSV/SID核验，不是整对实验完成。继续观察同一Job；没有改配方或新增GPU。一次观察命令误写远程脚本路径，仅查询返回127，随后正确路径读取成功；训练未受影响、未重启。
