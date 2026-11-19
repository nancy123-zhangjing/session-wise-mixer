# 2026-10-16：进程身份误判及有界恢复

## 当前结果：恢复与全部89项训练已完成，GPU已释放

本次恢复保留修复后38项完整成果，8项中断任务从完整状态继续，其余按原固定清单执行；训练计算源码始终为`bbaa45b`/SHA256 `b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907`，外部supervisor单独为`37ef406`/578f。最终64搜索、12复验、10正式、3旧协议配对全部成功，89/89、worker为0，旧数值故障版本的28项不进入选模。

- 元数据归档已到**批次20，89项/740个文件**，[独立审计](../completed_metadata_precision_v2_batch20/AUDIT.json)PASS；完整payload已到**批次22，89项/178份best/last**，[本地收据与元数据交叉核对](../checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_22.json)PASS。原01–22完整状态和01–20元数据批次均保留；当前无须再创建空审计批次，不执行历史段落的“下一批20/18”命令。
- 最终锁定SASRec配置13、eSASRec配置19；依据正式3031–3035的验证均值、测试前封存选择`sasrec_fullce`。[验证选择/全部89项/原始时序](../final_delivery_20261016/FINAL_SELECTION_AND_ORDER_AUDIT.json)、[CPU交付89/13/10](../final_delivery_20261016/CPU_DELIVERY_PASS.json)、[最终12组/2,400次测速重算](../final_delivery_20261016/FINAL_BENCHMARK_AUDIT.json)、[10份正式导出权重与商品向量重载](../final_delivery_20261016/FINAL_EXPORT_AUDIT.json)均已PASS；下方两个后处理工具“正式产物尚未审计”仅是14:03以前的准备记录。
- `nancy-stage06-20261016-8h100-supervisor-v3`于15:14:19 SUCCEEDED，15:14:40 Pod删除，平台剩余节点0；[终态](../resource_accounting_20261016/SUPERVISOR_V3_TERMINAL_PLATFORM_20261016.json)、[事件](../resource_accounting_20261016/SUPERVISOR_V3_TERMINAL_EVENTS_20261016.json)确认本Goal 8卡已释放。不重提GPU，不停止既有非Goal Notebook或CPU桥接。
- 有效远端产物在`goal_20261016_precision_v2/code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3/`；旧失败pipeline和恢复记录继续保留。[完整本地备份审计](../final_delivery_20261016/FINAL_BACKUP_LOCAL_AUDIT.json)已PASS：9分卷/1,300文件、14,409,004,551解压字节，178份best/last摘要与独立ledger一致；412份可用文件安全提取至`outputs/stage06_continuous/final_delivery_20261016/verified/`，含10exports、13逐样本结果、锁定输入/27份源码及代表SASRec13/3031完整状态。其余seed完整状态仍保存在已验证分卷中，不重复下载。
- 代表状态独立副本`recoverable_primary/`通过[本地原消费者CPU恢复检查](../final_delivery_20261016/LOCAL_RECOVERY_TRANSACTION_AUDIT.json)：model、Adam、scheduler和CPU RNG均加载，CUDA RNG仅字节保全/校验、未执行；COMPLETE保留，没有启动训练。无需再执行历史`--finalize-only`或训练恢复指令。
- [五次Goal作业资源审计](../resource_accounting_20261016/FINAL_RESOURCE_AUDIT.json)核实分配33.2203 GPU小时、平台runtime32.9347 GPU小时，按费率折算分别同值点券。用户已直接回复“接受，以此完成Goal”，[确认收据](../final_delivery_20261016/FEE_ACCEPTANCE_20261016.json)接受上述计算口径作为费用交付，费用不再是阻塞；仍保留`actual_goal_settlement_verified=false`，不冒称已确认结算。实验、完整本地备份及代表状态CPU恢复检查均已验收；本地路径、首选SASRec13/3031、恢复目录和178份状态的分卷成员/SHA见[已验证本地产物索引](../final_delivery_20261016/LOCAL_ARTIFACT_INDEX.json)。本轮全部验收通过，交付提交`aec2307`（未push），当前无需再执行本Goal；Goal工具状态以当前任务状态为准。下一研究为尚未实施的固定推荐器编码对照，需要另行确定实验范围。

时序/冻结源码支持自有训练与评估排空后固定GPU0依次测速；没有连续进程追踪或kernel profile，不能进一步声称外部负载不存在、所有推理kernel精度或全部系统I/O已证明。

本轮[最终交付总验收](../final_delivery_20261016/FINAL_DELIVERY_STATUS.json)为`ALL_REQUIRED_EVIDENCE_PASS_WITH_USER_ACCEPTED_FEE_BOUNDARY`，明确提交范围见[Git交付核查](../final_delivery_20261016/FINAL_GIT_DELIVERY_AUDIT.json)。

## 14:03及以前的工具准备、恢复与观察记录（历史）

以下状态只描述各自时点；“尚未验收正式产物”“继续执行入口”“下一批”均已被上方最终收据取代，不能据此启动训练或重复CPU交付。

最终CPU交付后还须执行两个独立后处理审计（目前均未验收正式产物）：[测速原始样本重算](../audit_benchmark_measurements.py)的最终模式为`--benchmarks <campaign/final_benchmarks> --campaign <campaign> --report <新收据>`，限定冻结配置、seed3031及89项训练完成，重算p50/p95/吞吐；[十份正式导出重载](../audit_formal_exports.py)使用`--root <冻结code> --campaign <campaign> --cache <锁定cache> --report <新收据>`，须先有CPU_DELIVERY_PASS，逐张量/逐商品向量核对并用8条验证历史检查query与全库logits一致。二者位于冻结训练目录外，不修改或更新GPU运行源码。[测速工具CPU验收](../BENCHMARK_AUDITOR_CPU_GATE_20261016.json)用已存在的2400条初始样本，[导出工具CPU验收](../EXPORT_AUDITOR_CPU_GATE_20261016.json)仅用旧版三次更新的CPU集成产物，均不代表最终测速/十份正式导出已完成。最终测速的独占窗口、GPU UUID和实际kernel边界仍须另核实。

14:03已将这两个独立审计脚本同步至`goal_20261016_precision_v2/delivery/`，服务器字节哈希和Python3.10语法检查通过，训练源码仍冻结。见[独立部署回读](../POSTPROCESS_AUDITORS_DEPLOYMENT_20261016.json)，源码提交`9a2ddf9`；该收据只证明工具准备就绪，正式产物仍未审计。

13:24：新Job的GPU gate与实际8项恢复已通过[独立核对](gpu_gate_supervisor_v3/GPU_GATE_AUDIT.json)；35轮压力检查的所有loss/范数行与此前候选诊断一致。SASRec配置09从221→225轮、配置16从77→79轮；eSASRec配置26从5→21轮等，均保留旧曲线前缀。该观察不代表89项已完成。原调度Job分配窗口4,783秒/约10.6289 GPU小时，含38项有效成果与被中断工作，并非全是无效消耗；[分配窗口折算](ALLOCATION_AUDIT.json)不冒充平台结算。

继续执行入口：当前Job为`nancy-stage06-20261016-8h100-supervisor-v3`，pipeline为冻结`code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3/`，旧pipeline保留。观察脚本在本地`.runtime/continuous_goal_20261015/observe_supervisor_v3.py`。后续CPU完整状态审计使用`delivery/audit_checkpoint_payloads_supervisor_v3.py`，共享已审计74项的ledger，下一批编号20，不能覆盖原01–19。元数据已由独立`delivery/archive_completed_supervisor_v3.py`归档至批次17、74项，下一批命令参数为`--batch 18`（同时传`--base`），无新完成项时不创建批次；继承全部all_completed_task_ids，并保存pipeline_relative及resume_events。两者均在冻结训练目录之外，不更新GPU作业源码。本地下载后可用`.runtime/continuous_goal_20261015/local_receipt_tools.py extract <元数据批号>`安全解包，运行原元数据独立审计，再用`crosscheck <完整payload批号>`交叉核对已下载ledger、收据与元数据；原始摘要可能相差一批，缺少交集之外的审计不能冒称已验证。14:52已独立核实全部64次搜索的Top-3、eSASRec配置19的三seed验证均值与5个正式seed绑定，见[真实依赖审计](../completed_metadata_precision_v2_batch16/DEPENDENCY_AUDIT.json)；SASRec须等6次复验全部完成后才能锁配置。最终GPU结束后仍用冻结训练目录的`run_continuous_campaign.py --finalize-only`处理此新pipeline；独立重算、导出、报告和费用/释放核实都尚待完成。

13:21：Linux CPU 11项零跳过通过，实际进程组清理分支已运行；38项完整结果及8项中断best/last状态保全核验通过。新唯一8卡Job `nancy-stage06-20261016-8h100-supervisor-v3`已RUNNING、正在GPU gate，源码及数据保持原样。下文“尚未提交”是恢复设计时记录；当前以[提交收据](PLATFORM_RECOVERY_SUBMISSION.json)、[CPU回读](RECOVERY_READBACK_AUDIT.json)及项目交接顶部为准。

13:02平台Job `nancy-stage06-20261016-8h100-precision-v2` FAILED，13:02:43已删除Pod。38项完成、8项被中断（原队列把其中1项错误标为failed）、18项未启动；后续25项尚未展开，测试仍锁定。

调度器为刚启动的`search__esasrec_26__seed2024`记录了PID 78687、start_ticks 220637418和空command。它随后按包含command的整字典判断身份，误判该进程失败；当时Popen没有退出码，该进程随后正常写完5轮，loss和梯度有限。清理又因身份不等跳过该进程，`wait(timeout=10)`覆盖了最初错误。证据见本目录原始平台日志、事件、队列身份与run/curves快照。没有把它归为基础设施故障或再次梯度溢出。

恢复边界：

- 保留原pipeline和原始字节；38项完整best/last均已CPU核验，不因调度层缺陷宣布其训练无效。
- 训练计算仍使用独立冻结目录的`bbaa45b`与源码SHA `b240275a...`，模型、优化器、精度、数据、seed、预算和选模逻辑均不变。
- `scripts/train/continuous_queue_supervisor.py`是单独哈希和提交的外部进程管理入口，**不修改冻结训练目录，不改写其源码身份，也不绕过任何训练/测试门槛**。工作流保存原训练版本和新supervisor/deployment身份。
- 子进程退出以Popen.poll为准；身份只用PID、启动时间、主机及boot ID，命令行作为预期argv保存。清理只作用于稳定身份匹配的自有进程组，错误单列并保留触发错误。
- CPU准备脚本须先核实平台终态和Pod已删除；完成项复制为不可变硬链接，8项中断状态从完整checkpoint恢复，CSV/JSON独立复制，旧目录不变。新pipeline使用独立名称，attempt_1明确记录supervisor缺陷恢复，不占用“基础设施重试”名额，不追加配置。
- 新入口通过本地测试后还须通过服务器Linux CPU测试、真实状态检查和新Job GPU gate；目前尚未重新提交。

本地测试中的89项补位是明确的CPU逻辑fixture，不是89次模型训练或验收结果。Linux进程组停止与实际训练恢复仍以服务器收据为准。
