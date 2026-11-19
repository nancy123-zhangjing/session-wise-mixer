# 后续Goal实现状态

## 当前状态：2026-10-16

**89项训练、最终评估、完整本地备份与代表状态CPU恢复检查均已验收，8张H100已释放；用户已接受保留未核实结算说明的费用计算口径，本地产物索引和全部验收均已通过，交付已提交`aec2307`（未push）。当前无需再执行本Goal；Goal工具状态以当前任务状态为准。** 下方14:57及以前是历史快照，不是当前运行状态或重提指令。

本轮[最终交付总验收](execution_acceptance/final_delivery_20261016/FINAL_DELIVERY_STATUS.json)已通过全部要求，见[Git交付核查](execution_acceptance/final_delivery_20261016/FINAL_GIT_DELIVERY_AUDIT.json)；交付提交`aec2307`，未push。

| 项目 | 当前实证与边界 |
|---|---|
| 固定训练预算 | 64搜索＋12复验＋10正式＋3旧协议配对，89/89成功、队列worker为0；全部[89项/740个元数据文件](execution_acceptance/completed_metadata_precision_v2_batch20/AUDIT.json)及[178份best/last完整状态](execution_acceptance/checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_22.json)已核对 |
| 代码与数据 | 冻结训练`bbaa45b`/SHA256 `b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907`；恢复supervisor单独绑定`37ef406`/578f；原输入、划分、缓存、配置与89项预算不变，旧数值故障版本28项不参与选模 |
| 锁定配置 | SASRec配置13：d256/L1/H1、dropout0.3、LR1e-4、batch256、3,900,672参数；eSASRec配置19：d512/L1/H4、dropout0.3、LR1e-3、batch128、K1024、10,945,536参数 |
| 最终选模 | 各配置先按2024/2025/2026验证均值锁定，再以正式3031–3035的验证均值选择SASRec＋FullCE；验证NDCG@10为0.05552036对0.05151127，测试前封存。[全部89项/原始时序独立审计](execution_acceptance/final_delivery_20261016/FINAL_SELECTION_AND_ORDER_AUDIT.json)通过 |
| 正式测试 | 两模型各5新seed，旧协议SASRec为3031–3033，共13项、每项22,363条；12,101真实商品全库、不额外屏蔽历史。13项逐样本重算通过。SASRec HR@10/NDCG@10 `0.072280±0.001294 / 0.041515±0.000845`；eSASRec `0.064920±0.001759 / 0.039181±0.000795`，均为样本标准差；[六项指标、原始seed和配对差值](execution_acceptance/final_delivery_20261016/summary.json) |
| 导出与测速 | [CPU交付89/13/10](execution_acceptance/final_delivery_20261016/CPU_DELIVERY_PASS.json)、[10份正式权重及全部商品向量重载](execution_acceptance/final_delivery_20261016/FINAL_EXPORT_AUDIT.json)、[12组/2,400次最终测速重算](execution_acceptance/final_delivery_20261016/FINAL_BENCHMARK_AUDIT.json)均PASS。时序与冻结源码证明自有训练/评估排空后固定GPU0逐个测量；没有连续进程追踪或kernel profile，不声称外部负载不存在或逐kernel精度已核实 |
| 平台终态与释放 | `nancy-stage06-20261016-8h100-supervisor-v3`于15:14:19 SUCCEEDED、15:14:40 Pod删除，剩余节点0；[终态](execution_acceptance/resource_accounting_20261016/SUPERVISOR_V3_TERMINAL_PLATFORM_20261016.json)与[事件](execution_acceptance/resource_accounting_20261016/SUPERVISOR_V3_TERMINAL_EVENTS_20261016.json)均保留。全部Goal GPU已释放，不影响非Goal Notebook或CPU桥接，不需重提GPU |
| 资源与费用 | [五次Goal作业审计](execution_acceptance/resource_accounting_20261016/FINAL_RESOURCE_AUDIT.json)核对实际分配33.2203 GPU小时、平台runtime32.9347 GPU小时；按已核实费率折算分别为33.2203/32.9347点券，均是计算金额。个人日账单含非Goal Notebook，实际Goal结算无法可靠拆分，`actual_goal_settlement_verified=false`。用户已直接回复“接受，以此完成Goal”，[确认收据](execution_acceptance/final_delivery_20261016/FEE_ACCEPTANCE_20261016.json)接受上述计算口径作为费用交付，费用不再是阻塞 |
| 完整本地备份 | [全量审计](execution_acceptance/final_delivery_20261016/FINAL_BACKUP_LOCAL_AUDIT.json)PASS：9分卷/1,300文件、14,409,004,551解压字节，178份best/last摘要与独立payload ledger一致；412份文件已安全提取到`outputs/stage06_continuous/final_delivery_20261016/verified/`，含10exports、13逐样本结果、锁定输入/27份源码及代表SASRec13/3031状态，其余seed完整状态保留在已验证分卷内 |
| 代表状态本地恢复 | 独立副本`recoverable_primary/`通过[原消费者CPU恢复审计](execution_acceptance/final_delivery_20261016/LOCAL_RECOVERY_TRANSACTION_AUDIT.json)：model、Adam、scheduler和CPU RNG加载；CUDA RNG仅保留/校验字节，未在GPU执行；COMPLETE保留，没有启动训练 |
| 本地复用入口 | [已验证本地产物索引](execution_acceptance/final_delivery_20261016/LOCAL_ARTIFACT_INDEX.json)列出10exports、13逐样本结果、178份checkpoint分卷成员及SHA、首选SASRec13/3031和`recoverable_primary/`；使用索引的本地路径，不将原远端导出位置视为本地路径 |
| 总验收与提交 | 本轮全部验收通过，费用计算交付已获用户接受，交付提交`aec2307`（未push）；保留实际结算未核实说明。当前无需再执行本Goal，工具状态以当前任务状态为准 |

两项300轮封顶搜索为eSASRec29和SASRec28，最佳分别255、294轮，均不在Top-3；保留预算限制，不追加搜索或宣称所有候选充分收敛。SASRec/eSASRec容量、架构和训练目标同时不同，当前结果不是单因素架构消融；06为连续查询向量检索基线，尚未实现给定白盒Decoder自动学习适配Encoder或列表生成。

最新[10.16周报](<feishu-link>)已生成并回读。当前无需再执行本Goal；下一研究为尚未实施的固定推荐器编码对照，需要另行确定实验范围。不重跑历史04/05，不修改冻结训练源码。

## 2026-10-16 14:57及以前执行快照（历史）

14:57：新Job和8卡队列RUNNING；64次搜索完成，SASRec Top-3为13、16、01，6次复验中2次完成、4次运行；eSASRec复验6/6完成并按三seed验证均值锁定配置19，正式seed3031、3033完成，其余3次运行。合计74完成、7运行，后续8项未展开。74项/617个元数据文件及148份完整状态已[分批核对](execution_acceptance/completed_metadata_precision_v2_batch17/AUDIT.json)，见[完整状态](execution_acceptance/checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_19.json)。两模型全32次搜索的Top-3、eSASRec三seed均值选择与正式5个配置/seed绑定已[独立核对](execution_acceptance/completed_metadata_precision_v2_batch16/DEPENDENCY_AUDIT.json)。eSASRec29与SASRec28达到300轮上限，最佳分别255、294轮，均未进Top-3；预算限制保留。训练b240、supervisor578f与缓存冻结，继续原89项；无失败，最终模型族选择和测试仍锁定。

eSASRec配置19为512维/1层/4头、dropout0.3、LR0.001、batch128、负样本1024，参数量10,945,536；三个搜索/复验seed训练耗时15.19/14.40/25.69分钟（118/112/200轮），包含验证及保存。当前3个正式seed与4个SASRec复验各占一卡；不增加预算外实验，不把此耗时混作推理测速。

13:21：唯一新Job `nancy-stage06-20261016-8h100-supervisor-v3` RUNNING，qb-prod-gpu406，正在增强GPU gate。服务器Linux 11项CPU检查零跳过及8项中断状态的16份完整payload检查通过；38项既有完成结果保留，未重跑。外部supervisor版本`37ef406`/578f单独绑定，训练worker仍为`bbaa45b`/b240冻结代码，测试锁定。尚待实际恢复队列和后续51项（含依赖未展开25项），不能标记Goal完成。

13:10：precision-v2 Job已FAILED并释放（平台13:02:43删除Pod）。调度器空command身份误判、清理超时；38项完整状态CPU通过，8项中断、18项未启动，测试锁定。保持训练计算源码b240不变，单独版本化外部supervisor及CPU恢复脚本已实现、本地11项检查通过；尚待Linux CPU验收及新GPU gate，未重提作业。见[证据与保留范围](execution_acceptance/scheduler_failure_20261016_01/README.md)。下文运行记录是此前快照。

12:57快照：新Job在线RUNNING；34项技术完成、8项运行、22项待启动。其中32项/260个元数据文件已[独立审计](execution_acceptance/completed_metadata_precision_v2_batch04/AUDIT.json)，34项/68份best/last完整状态已[CPU审计通过](execution_acceptance/checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_05.json)。原故障配置17按原设置完成53轮、最佳第3轮后早停50，逐轮末尾最大范数6.6774，完整状态通过。尚无Top-3/正式配置、最终测试或89项完成证明，Goal继续。旧作业与诊断资源窗口已核对，新Job资源监测已保存；点券折算不冒充平台结算。

11:54：修复提交`bbaa45b`已通过远端36项CPU gate、增强GPU gate与初始独占测速；新Job `nancy-stage06-20261016-8h100-precision-v2` RUNNING，原89项队列已启动。新4项训练梯度对照、35轮回归及原始测速独立核对通过；首个技术完成任务的完整checkpoint也已独立审计。不把这些写成89项完成，测试仍锁定。继续先核实新Job，不重提。见[提交收据](execution_acceptance/PLATFORM_SUBMISSION_20261016_PRECISION_V2.json)、[GPU收据独立核对](execution_acceptance/gpu_precision_v2_20261016_01/GPU_GATE_AUDIT.json)。

11:38：候选修复已落地（CUDA注意力统一FP32 Math，外层BF16），本地73项CPU gate零跳过；固定3批梯度校准、514步重放及35轮单卡压力检查通过，两个诊断Job均结束释放。新源码 `b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907` 待独立服务器CPU及增强GPU gate。将从头完成原定89项，不接续旧版本结果；旧消耗单列。下文为故障发生时快照。

完整Goal包括单节点8H100 smoke测速、89次有效训练、最终评估与交付。执行版本`4d7c466`通过初始CPU/GPU gate，但8卡Job已因搜索配置17数值异常FAILED并释放。队列停止，28项技术完成、1项失败、7项中断、28项待启动；完成记录有效性待审，测试仍锁定。独立1卡诊断已复现原BF16路径异常并结束释放，统一精度修复尚未完成。见[故障与证据](NUMERICAL_FAILURE.md)。不得将数值故障作为基础设施故障重试，不混用修复前后源码结果。

- 模型/单卡训练恢复、8worker队列、故障对账、独占测速、13项最终测试及CPU交付审计入口均已实现。
- 当前72项CPU测试零跳过：模型18、队列9、交付8、数据37；全部326,533条缓存独立核验通过。[CPU gate](execution_acceptance/local_cpu_20261016_01/CPU_GATE.json)。
- 两模型各真实数据3次CPU更新、22,363条完整验证/独立重算、商品向量与portable recommender导出重载通过。[集成收据](execution_acceptance/local_cpu_20261016_01/cpu_integration.json)。不计正式89次，没有测试集推荐指标。
- 当前源码SHA256：`7bff9f19bfef7ef8ec4ee9b058a6ff6cf7baeea3f469c5dc3f87c1cf25031ccc`；checkpoint为不可变epoch快照加原子CURRENT事务，恢复完整优化器及RNG，两处中断故障注入通过。
- 接着提交本轮明确文件、Git bundle同步独立执行副本及原数据、远端CPU gate和磁盘检查，再在线选配额/镜像提交一个8卡Job。无需push，不改变共享环境。
- GPU产物落盘和最终测速结束后退出Job释放资源，CPU完成导出、89项曲线/选模及13项逐样本独立审计、图表和报告。平台实际账单/释放另核实，程序估算不冒充已确认费用。

以下为上一开发时点快照，其中“尚未实现”等待办已被上述当前状态取代，保留用于追溯。

## 2026-10-15开发快照（历史）

2026-10-15。完整Goal仍为模型/CPU验收、单节点8H100 smoke测速、89次有效训练、最终评估与交付；当前没有申请GPU、没有任何正式推荐结果，不能标记Goal完成。

## 已有实测证据

- 分支 `codex/continuous-baseline-8h100`；原始908项tracked文件哈希和两个历史引擎状态保存在被忽略的 `.runtime/continuous_goal_20261015/initial_state.json`。代码尚未提交。
- 参考作者源码锁定及三份未改写网络源码、Apache-2.0许可已保存；见SOURCES.md和upstream_sources.json。
- SASRec＋FullCE/eSASRec模块、单卡训练器、epoch边界完整恢复、全库评估、缓存/训练/评估/导出/测速命令入口已实现。
- 16项模型CPU测试实际通过，零跳过；含作者网络前向及梯度对照、合成样本拟合、遮罩与两模型逐位恢复。
- 6项队列依赖/选模逻辑测试通过，零跳过；这是调度逻辑测试，不是64/89次训练完成证据。
- 37项已有数据测试在Python3.12复查通过，零跳过。
- 新张量缓存 `outputs/stage06_continuous/goal_20261015_cache` 已独立逐条比对全部326,533条原JSONL，PASS；旧数据未重生成或覆盖。
- Inspire CPU连接已实测；既有项目环境Python3.10.20、Torch2.6.0+cu118，在CPU实例不暴露GPU。本地独立环境为Python3.12.14、Torch2.6.0+cpu、NumPy2.2.6，不改共享环境。

## GPU前必须继续完成

1. 完成与审查端到端作业入口：CPU gate→两模型有界GPU smoke/恢复→8卡并发补位→独占初始测速→89项队列→冻结验证选择→13项测试→导出/独占最终测速→独立指标审计与汇总。
2. 当前 `campaign.run_queue` 已有8worker、UUID绑定、依赖展开、完整结果核验与验证选模封存；尚未实现作业后处理、完整资源监测/费用归集及基础设施故障的授权对账重试入口。不得将dry-run当作作业运行。
3. 对队列恢复/失败路径做审查：失效PID不能证明平台Job终态；跨节点重启必须先从平台核实旧Job，最多两个基础设施重试；被接管worker的终止路径需要一并处理。
4. 当前GPU smoke代码已写但尚未执行；需要审查smoke启动前CPU gate、实际设备/UUID、步数/时间上限、恢复与BF16阈值。
5. 补齐最终逐样本独立审计、5seed均值/标准差、旧协议配对差值、曲线和完整交付审计；命令入口存在不等于已验收。
6. 完成所有执行源码后运行CPU gate并提交；源码指纹包含continuous模块、相关训练入口、checks及tests。正式训练期间新增/修改这些文件也会使指纹改变，因此必须在申请GPU前补齐后处理代码。
7. 核验远端代码/数据/磁盘、打包已验收数据与缓存并同步已提交版本，不擅自重生成旧协议或覆盖远端已有目录。只使用一个8卡H100 Job。

验收报告应保存原始日志和来源指纹；每次代码修改后，旧gate只能证明当时版本，不能冒称当前最终版本已通过。
