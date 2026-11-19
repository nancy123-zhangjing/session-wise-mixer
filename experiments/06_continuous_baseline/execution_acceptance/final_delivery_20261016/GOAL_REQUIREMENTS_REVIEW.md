# 06 后续 Goal：逐项要求与完成证据审阅

审阅快照：2026-10-16 16:28（Asia/Shanghai；纳入最终总核查及首个92文件交付commit）。审阅对象为用户原 Goal 的八节要求及其最新澄清，不以现有 PASS 文件反向缩小完成标准。

原文：C:/Users/nancy/.codex/attachments/98b43b1e-1197-4638-b7bc-28346ea4988f/goal-objective.md；本次重新读取，SHA256 为 44ae1b2a62f9fe628ef8f81c483a72138c82745eaee759d7cf253d4f6f780678。

最新用户澄清：F/FEE_ACCEPTANCE_20261016.json记录原问题、直接人类回答“接受，以此完成Goal”及16:20确认时间；本审阅核对该记录引用的原Goal SHA与resource审计SHA均匹配，两种实测用量/计算额相同，未核实结算的false字段仍保留。

**总体状态：原Goal加最新用户澄清已无实质未完成项，最终总核查与首个92文件的显式Git交付均已通过。** 模型与 CPU/GPU 验收、89 项有效训练、验证选模封存、13 项最终测试、独立指标重算、10 份正式导出、初始与最终测速均有对应实证。本 Goal 的 GPU 已释放。完整九分卷、1,300个成员和178份状态的本地独立校验已通过，412个可用交付文件实际落地；代表运行事务目录也已通过真实 CPU 恢复消费者验收。**Goal专属实际结算仍无法证实，actual_goal_settlement_verified=false；用户已明确接受保留这一说明、以实际时长乘费率的两种计算口径完成费用交付。** 此确认改变费用交付的验收边界，不把计算金额或个人项目聚合账单变成已核实结算。F/FINAL_DELIVERY_STATUS.json记录要求证据全部通过且未解决要求为空；F/FINAL_GIT_DELIVERY_AUDIT.json记录首个交付commit aec2307bc4e5abf79b7ed22df8f1150b4ddfaedf、92个逐项审查文件、hash/暂存范围/whitespace及受保护字节检查通过，未push。本次随后仅同步这些完成状态说明；不声称未知的后续commit或Goal工具已经complete，最终工具状态以父任务操作为准。

本审阅读取了原 Goal、计划/协议、模型/训练/恢复/指标/测速/调度实现、相关测试和当前收据；核对证据的样本数、版本、覆盖范围与阶段顺序。未重新训练、未启动 GPU、未执行网络查询，也未因文件名含 PASS 就默认其覆盖所有要求。下文“已证明”均限定于所列证据范围；临时下载进度不作为完成依据。

路径约定：A 为 experiments/06_continuous_baseline/execution_acceptance/；F 为 A/final_delivery_20261016/。正式冻结训练为 commit bbaa45b515a15912a04665721e0a5c43d64dc9e9，源码集合 SHA256 为 b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907；外部 supervisor 为 commit 37ef4065546c3fb13397a4d9cdc0d3f29d1462df、SHA256 578fdcaed74a47ac470abc393af62fa29e9bc2bd676fbe97533e1a49ad98dd91。

## 一、读取现状，固定执行计划

| 原要求 | 实际证据及检查内容 | 状态与覆盖边界 |
|---|---|---|
| 阅读指定现状，保留第一阶段记录 | docs/guides/PROJECT_HANDOFF.md、INSPIRE.md、docs/guides/EXPERIMENT_WORKFLOW.md、06 README/PLAN/ACCEPTANCE 与 data_protocols.json；PLAN 保留第一阶段范围，另增后续 Goal、数值修复及调度恢复计划 | 已记录。第一阶段禁止 GPU 的历史边界未被冒作本 Goal 的现行限制；④⑤⑥未混为重做历史04/05。 |
| 固定模型、搜索、调度、预算和验收条件 | PLAN.md、configs/06_continuous_baseline/search_manifest.json、SOURCES.md；生成器 scripts/continuous/protocol.py 与冻结清单一致，模型测试检查清单和锚点 | 已证明。后续收据以真实执行版本绑定，不用当前报告 commit 冒充训练版本。 |
| 不重复准备数据，不改映射/划分/输入哈希 | 第一阶段 acceptance/data_verification.json、data_manifest.json；后续本地/远端 CPU gate 及 cache 验证；各训练 binding、完整状态审计 | 已证明身份链。数据 manifest 为 6ecbe2768c31a6c212146ac8d3f01a45bc72cfacd0794166fa4730d121a0fdfa，cache manifest 为 9d229a275d0c476e9c8d64a7797a0d2d1eada535cfe5ad81dc8118a518ab5519。未以重新准备覆盖旧目录。 |
| 主训练 full，legacy 仅配对 | data_protocols.json、实际89项任务和曲线审计；full 131,413 个目标、legacy 105,668；验证/测试各22,363 | 已证明。共同12,101件商品；legacy仅最终3个SASRec匹配seed，不参与搜索或新模型选定。 |

数据的第一阶段独立验证从原始交互推导全部合法位置，旧协议还抽取执行原旧分割方法；不只比较生成器自报数量。留出包含原数据中的冷目标，未为了指标过滤；leave-one-out不是全局时间划分或线上结论。

## 二、模型、训练评估接口与 CPU 检查

| 原要求 | 实际证据及检查内容 | 状态与覆盖边界 |
|---|---|---|
| SASRec＋FullCE | scripts/continuous/model.py：因果注意力、可学习商品表、输入输出共享；forward取末位历史隐藏向量，再与 weight[1:] 点积；FullCE标签为target−1 | 已证明实现符合。SOURCES.md明确为RecTools PreLN/GELU变体，不声称逐字复现2018 TensorFlow SASRec。 |
| eSASRec真实来源 | SOURCES.md、upstream_sources.json、reference_upstream/、许可证；作者代码 commit 2b039927ceb4c7654131d7ac7c43ea124b49d240，RecTools0.13.0 commit 6ffd25bf3ca8517c8bd1a7207d6b0e64c8e1912a | 已证明来源和适配有记录。LiGR sigmoid门控残差＋pre-LN SwiGLU、FFN4d无bias；CPU测试执行锁定作者类的真实前向及梯度对照。不是给普通SASRec改名。 |
| sampled softmax与同协议监督 | model.loss：每目标均匀有放回采样256/1024个真实商品，无logQ、不排除重复或偶然正例；展开前缀每行只监督一个下一商品 | 已证明。不同于作者通常的多位置监督和论文固定配置，差异已披露；不能直接比较FullCE与sampled训练loss数值。 |
| 左PAD、因果/补齐遮罩、末位唯一打分、PAD排除 | model.py、data.py与模型测试：真实查询不看PAD键或未来；数据边界保证末位非PAD；全库分数只对12,101件真实商品；采样ID从1开始 | 已证明。CPU检查含未来不变性、padding不变性、单商品历史前后向有限、共享权重及PAD梯度。 |
| Adam、裁剪、早停、最佳checkpoint | trainer.py：Adam(.9,.98)、裁剪1并拒绝非有限梯度、常数scheduler、每轮完整验证、patience50、最多300轮；按验证NDCG@10优先、HR@10平局处理 | 已证明代码及实际曲线吻合。不是按测试集或训练loss选最佳权重。 |
| 保存加载与完整恢复 | checkpoints.py原子CURRENT事务；trainer.py保存模型、Adam、scheduler、Python/NumPy/CPU/CUDA RNG、epoch/updates/早停/曲线/binding | 已证明状态内容与恢复接口。epoch内中断会重做未提交epoch；不声称保存任意更新步。 |
| 独立验证、最终测试、导出、测速入口 | scripts/train/run_continuous.py及campaign/delivery/checks；训练函数不接受test数据；test入口要求封存选择 | 已证明接口分离。测试文件提前作为锁定输入存储，不等于搜索时计算测试推荐指标。 |
| 真实CPU模型验收、零跳过 | A/local_cpu_20261016_02/CPU_GATE.json：73项＝37数据＋19模型＋9队列＋8交付；A/remote_cpu_20261016_02/CPU_GATE.json：对应36项模型/队列/交付，另核验锁定cache | 已证明实际执行，零失败/错误/跳过。含作者梯度、合成拟合、指标独立排序、两模型逐位epoch恢复、事务中断反例。数据37项没有代替模型测试。 |

本地CPU环境为Python3.12.14/PyTorch2.6.0+cpu/NumPy2.2.6；服务器CPU gate为Python3.10.20/PyTorch2.6.0+cu118/NumPy2.2.6。CUDA数值策略统一改为注意力区域FP32 Math、外层BF16；CPU作者方程对照保持原dtype。

## 三、版本、环境与执行授权

| 原要求 | 实际证据及检查内容 | 状态与覆盖边界 |
|---|---|---|
| 公共科研项目、分布式训练空间、先CPU再GPU | INSPIRE.md、PLATFORM_SUBMISSION_20261016_PRECISION_V2.json、远端CPU gate、scheduler_failure/PLATFORM_RECOVERY_SUBMISSION.json及部署manifest | 已证明执行入口/版本有绑定。复用既有grm-exp环境，未见升级共享Conda环境的改动。GPU inventory记录H100、驱动595.58.03；数据、磁盘容量与缓存先验收。 |
| 运行期间源码固定 | CPU gate的源文件SHA、每run来源、queue与部署manifest、F/FINAL_PRESERVATION_AUDIT.json | 已证明冻结集合27文件当前仍逐项匹配b240；训练修复使用独立新目录，外部supervisor独立提交，不热更新worker源码。 |
| 独立run_id、代码/配置/数据/环境/Job/GPU来源 | run.json、任务queue、pipeline、部署及平台提交收据、COMPLETE与receipt；A/completed_metadata_precision_v2_batch20/AUDIT.json核对740文件 | 已证明有效89项身份链。旧pipeline、原数值版本和中断attempt另存；没有覆盖原结果。 |
| 保护用户改动和两个引擎 | F/FINAL_PRESERVATION_AUDIT.json：初始908文件中906字节不变，允许变更仅.gitattributes和INSPIRE.md；两个根README原字节不变；两引擎HEAD/status与初始一致；FINAL_GIT_DELIVERY_AUDIT另检查首次交付后的91个保护文件 | 已证明保全快照及首次交付后的保护字节不变。DiffGRM为39e94e6…、genrec为b6c8b41…且均clean；92文件提交未包括两个受保护根README或三个无关draft目录。 |
| 只提交本轮、不批量add、不push | F/FINAL_GIT_DELIVERY_AUDIT.json列出92个显式审阅路径及commit aec2307bc4e5abf79b7ed22df8f1150b4ddfaedf；独立清单匹配工作字节、cached路径精确一致、whitespace检查通过，no_stage_all_or_directory_pathspec_used=true、push_performed=false | **首次交付已证明。** 本审阅另以实际git确认该HEAD、92文件数、无保护README/draft路径、提交whitespace检查无输出，index为空。随后仅同步有限完成状态说明，由父任务再显式提交；此代理未stage/commit，不提前声称后续commit身份。 |

## 四、单节点八卡资源与调度

| 原要求 | 实际证据及检查内容 | 状态与覆盖边界 |
|---|---|---|
| 最多同时一个8H100节点、不额外扩节点 | 五Job平台事件/规格和A/resource_accounting_20261016/FINAL_RESOURCE_AUDIT.json | 已证明Goal分配窗口不重叠，最大同时8GPU。数值诊断的两个单卡Job也在其他Goal Job释放后执行；不与既有非Goal Notebook混算。 |
| 八个独立单卡worker，不做DDP | campaign.py及外部supervisor；GPU gate/queue/run来源与8个不同UUID | 已证明。每worker显式CUDA_VISIBLE_DEVICES=GPU UUID、内部cuda:0、独立模型/优化器/RNG/目录；验证在其绑定设备运行。 |
| 初始0—3 SASRec、4—7 eSASRec；完成即补位/跨族接手 | next_task及持久化队列/独立依赖审计；GPU gate 10任务在8卡完成并有替补dispatch | 已证明调度与实际设备隔离。不是整批等最慢任务；未为填满卡新增配置或seed。 |
| 全32后Top-3、全6复验后锁配置 | completed_metadata_precision_v2_batch19/DEPENDENCY_AUDIT.json、F/FINAL_SELECTION_AND_ORDER_AUDIT.json及原始queue | 已证明。两模型依赖可各自推进，不因另一模型未完成而阻塞，也不提前用局部完成者选模。 |
| CPU/cache/线程/输出隔离和best/last | 缓存先在CPU准备并核验；trainer将张量预载到设备；每worker2线程、无DataLoader多进程；每run独立CSV/performance，主队列汇总；checkpoint仅保留当前best/last事务 | 已证明设置和实际产物组织。资源监测包含CPU、显存、磁盘和保存耗时；并非完整NFS流量或所有kernel效率审计。 |

## 五、④ GPU smoke 与初始测速

| 原要求 | 实际证据及检查内容 | 状态与覆盖边界 |
|---|---|---|
| GPU0/1两模型先短测、全库验证、BF16、保存/重载/恢复 | A/scheduler_failure_20261016_01/gpu_gate_supervisor_v3/smoke/GPU_GATE.json与GPU_GATE_AUDIT.json；两模型基座20更新、恢复至21；每次验证22,363条 | 已证明真实CUDA执行。初始short gate独立标smoke，不算正式效果；未先启动其他正式训练worker。 |
| 预设步数/时长上限、8卡短并发 | 原计划smoke最多200更新/10分钟、并发每worker最多20更新/总10分钟；实际基础20更新、并发2更新/任务；另外35整轮数值压力检查上限360秒，实际155.06秒 | 已证明有界。35轮/17,990更新是明确单列的数值回归，不偷算正式训练或调参，也不冒充最初20步smoke。 |
| BF16数值正确性 | gate固定负例、关闭dropout、训练模式梯度对FP32：4组最大relative L2约0.00516、最小cosine约0.999987；35轮原失败配置loss/梯度有限 | 已证明这些形状与有界检查。不是所有配置/所有步的数学正确性证明；实际89项有限值与完成另由曲线/payload补充。 |
| 初始独占测速与同步 | A/gpu_precision_v2_20261016_01/INITIAL_BENCHMARK_AUDIT.json：修复后b240、64维smoke权重，12组/2400样本；原始初始bench在pipeline内保留 | 已证明初始测量及独立算术核对。不是最终锁定256/512维配置的速度；不混用旧默认注意力版本初始值。 |
| 统一batch1/128、三组件、预热重复、p50/p95/吞吐/显存 | benchmark.py：warmup50/repeats200，query与resident-query全库Top20用CUDA event且等待end；端到端用同步wall clock，含prepared CPU tensor H2D和Top20 D2H | 已证明测量范围，原始200样本/组保存。排除JSON/CPU序列预处理、模型加载和网络；throughput字段实际单位为历史查询/秒，不是20个输出商品/秒。 |
| 根据实测做时间/GPUh/点券估算 | A/gpu_precision_v2_20261016_01/budget_projection.json保留89次、51—300轮与最大形状插值估计，给出时间、GPUh与点券范围及不确定性 | 已记录预测，不作为真实运行或已结算金额。未承诺八倍加速；最终实际使用另有资源账本。 |

## 六、⑤ 超参数搜索与复验

| 原要求 | 实际证据及检查内容 | 状态与覆盖边界 |
|---|---|---|
| 共同设置 | protocol.COMMON、冻结manifest、89项run配置及曲线：history50、FFN4d、Adam(.9,.98)、clip1、smoothing0、constant LR、max300、每epoch全验证、patience50、验证NDCG@10选best | 已证明；无为失败静默降低batch、长度、LR或轮数。 |
| 指定搜索空间 | dim64/128/256/512；层1/2/4；头1/2/4；dropout.1/.2/.3/.5；LR1e-4/3e-4/1e-3；wd0/1e-5/1e-4；batch128/256；eSAS K256/1024 | 已由生成器、冻结清单与运行配置核对；没有途中扩空间。 |
| 每模型32个唯一配置和两个锚点 | search_seed20261015；每模型32个config hash唯一；锚点64/1/1和256/2/4其余参数完整；两族全部search seed2024 | 已证明64次实际完成，清单不受并行完成先后改变。 |
| Top-3复验与tie rules | 各自32次按验证NDCG@10、HR@10、较小参数量、固定trial_id排序；各Top-3新增2025/2026，共12次；三seed平均按同顺序锁定 | 已独立重算。SAS Top-3为13/16/01，最终13；eSAS为16/19/08，最终19。 |
| 不计算搜索/复验测试推荐指标 | trainer不接受test；test入口validate_test_unlock；封存训练结果哈希；F/FINAL_SELECTION_AND_ORDER_AUDIT.json验证原始mtime和阶段顺序 | 已证明本冻结执行路径和归档时间顺序；不以“目录有test文件”错误推断已经泄漏，也不只凭操作者声明无泄漏。 |
| 封顶未收敛如实报告，不加轮/扩搜 | summary.json和REPORT：SAS28、eSAS29达300轮，最佳分别294、255；两者均不进Top-3 | 已证明预算执行。不能声称全部配置充分收敛或搜索达到全局最优。 |

## 七、⑥ 正式训练、旧协议对照与最终评估

| 原要求 | 实际证据及检查内容 | 状态与覆盖边界 |
|---|---|---|
| 两锁定模型各5新seed；legacy三匹配seed | queue、训练CSV、独立选模审计：两族3031—3035共10；SAS legacy3031—3033共3，均继承锁定配置13 | 已证明共13个真实运行。不是从搜索seed挑最好测试seed，也不要求两族串行训练。 |
| 验证决定最终基线并先封存 | F/FROZEN_SELECTION.json与FINAL_SELECTION_AND_ORDER_AUDIT.json：五seed验证NDCG@10均值SAS .05552036368337769，eSAS .051511270979504635，选择sasrec_fullce | 已独立重算，training_results hash匹配89结果；原服务端mtime为最后训练→封存→13测试→最终顺序测速→GPU完成。 |
| 真实商品全库、PAD排除、无历史屏蔽/抽负例 | evaluate、score_catalog、stable_top_items与独立prediction审计；13份均22,363条、同一原始test truth SHA，候选12,101 | 已证明。全库点积与确定性score降序/ID升序tie规则，不是 sampled-negative evaluation；06 item-level不与03—05 SID-level直接并榜。 |
| 13项独立逐样本指标重算 | F内13份formal/legacy audit：读取原始留出标签与保存的Top20实际商品列表，复核身份、候选合法性、rank一致性，独立求HR/NDCG@5/10/20并聚合NLL | 已证明逐样本算术与身份。没有第二套模型重新生成13组全部logits；Goal要求独立指标重算，不据此增设再启GPU门槛。 |
| 五seed原始值、均值和标准差；legacy配对 | test_results.csv、summary.json、LOCAL_FINAL_METRIC_AUDIT.json与REPORT：10正式原始行、6指标、ddof=1；3legacy原始行及六指标配对差值 | 已证明聚合；跨Python版本最大差2.17e-19在明确容差内。不是均值标准误，未声称统计显著性。 |
| legacy的曝光/更新/耗时与归因边界 | REPORT旧协议表、summary.legacy_paired、training_results.csv：完整/旧目标131413/105668，batch256每轮514/413更新，三个seed实际总更新/曝光/时间均列出 | 已证明配对与报告；HR@10配对均增.00406922，NDCG@10增.00224263，但数据利用量与总更新同时变化，不能归因架构。 |
| 训练loss、验证全库NLL、best epoch与更新量 | 89条curves/CSV、FORMAL_TRAINING_DETAILS.csv的13条训练细节，REPORT含best与总轮、更新、loss/NLL/耗时；两张五seed曲线PNG/PDF | 已证明数据保存和报告。FIGURE_VISUAL_AUDIT.json实际目视检查两PNG，PDF仅哈希记录、未单独渲染；不冒称PDF逐页视觉验收。 |
| 最终锁定模型独占重新测速、参数与配置 | FINAL_BENCHMARK_AUDIT.json：固定代表seed3031、两族12组/2400样本；FINAL_SELECTION_AND_ORDER_AUDIT及冻结supervisor证明自有worker/评估排空后GPU0顺序调用 | 已证明本Goal的顺序窗口、配置和算术。SAS13为256/1/1、3,900,672参数；eSAS19为512/1/4、10,945,536参数。无连续进程trace或kernel profile，不能独立证明无任何外部负载或逐kernel dtype。 |

最终六项测试均值均为SASRec较高；SAS/eSAS在B1端到端p50分别约1.2517/1.2363ms，B128约1.4222/1.5666ms，不写成SAS始终更快。容量、loss和结构同时不同，结果不是单因素结构消融；也没有实际SID延迟对照、ANN/超大商品库或列表生成实验。

## 八、固定预算、异常处理与交付

| 原要求 | 实际证据及检查内容 | 状态与覆盖边界 |
|---|---|---|
| 89次有效预算、无追加配置机会 | 有效queue89＝64/12/10/3、全部succeeded；初始和增强gate、两诊断、数值失效版本及调度恢复分别记录 | 已证明有效成果89。物理运行和消耗超过单次无故障89训练，不能据有效计数抹去失败成本。 |
| 实现错误停受影响任务、统一修复、不混版本 | NUMERICAL_FAILURE.md、故障queue/平台事件：旧数值版本28技术完成、1失败、7中断；新目录统一b240从原固定清单重跑，未接旧7状态/混入旧28 | 已证明失效范围、重跑范围和消耗有报告。不是基础设施自动重试；未静默改变超参以通过。目标为原89有效成果，没有把旧28当新增调参机会。 |
| 调度故障的恢复与终态先核实 | scheduler_failure/平台终态及13:02:43 Pod删除；38项完整状态保留，8项中断由独立CPU检查后在新pipeline实际恢复；外部supervisor单独版本化 | 已证明，不因瞬时观测超时重提。8项attempt_1明确为supervisor缺陷恢复，不冒占/伪装成基础设施重试次数。 |
| 基础设施同配置自动重试最多2次 | manifest固定max_infrastructure_retries=2；reconcile_terminal_job要求平台终态、原Job ID和明确基础设施原因，拒绝仅凭PID消失重跑；本轮上述两种故障不分类为infra | 已证明规则实现和实际分类。没有实际需验收的第三次infra自动重试；不把代码故障包装成可无条件重试。 |
| 持久化任务清单和来源 | queue/run/COMPLETE、receipt、CURRENT、resume_events、config/source/data hash、Job/PID/UUID及pipeline来源；元数据740文件与payload ledger交叉 | 已证明89项结果位置与状态。不存在以口头“跑完”替代正式清单的情况。 |
| 保留全部尝试 | A/resource_accounting_20261016/OLD_ATTEMPTS_RETENTION_20261016_02.json重新核对两旧队列36和46个已启动run、best/last大小、receipt/CURRENT元数据SHA及不可变目标存在 | 已证明这些旧尝试仍保留，八项中断也覆盖。**未在此收据重新hash/加载所有旧payload；新九分卷不是旧失败状态的完整独立备份。** 不扩大存在性核对范围。 |
| 监测GPU/显存/CPU/I/O、合理线程 | 旧/新resource_attempts、v2平台CPU/磁盘/网络采样、supervisor资源窗口及独立审计；新节点34条训练采样，CPU均值约7.99核、限流增量0；八worker保存阶段比约0.9%—5.8% | 已证明有真实监测并据窗口评估瓶颈。不是完整NFS流量/全程kernel效率；未发现人为GPU空转负载或自动增节点的实现。 |
| 全配置、来源、曲线、评估与导出交付 | F/CPU_DELIVERY_PASS.json、independent_campaign_audit、89 metadata/178 payload、本地CSV/曲线；10正式导出逐tensor/全向量检查，8条validation query/全库logits精确一致 | 已证明远端产物与轻量副本；完整本地字节备份另由FINAL_BACKUP_LOCAL_AUDIT核验。10份导出含模型配置/权重/binding、输入表含PAD、输出表不含PAD，商品映射行号明确；实际本地10份导出和13份预测均已落地。 |
| 可恢复完整状态 | 178份服务器CPU payload接受检查；F/LOCAL_RECOVERY_TRANSACTION_AUDIT.json对SAS13/3031/attempt0，从已核验两分卷提取10文件，在独立目录重建CURRENT目标，实际运行原repair_committed、strict模型/Adam/scheduler及Python/NumPy/CPU RNG消费者 | **代表run恢复已证明。** CUDA RNG只保留byte SHA，未在GPU执行；COMPLETE保留、training_started=false。不称89目录均已恢复，不删除完成标记续训。 |
| 独立完成审计和完整本地灾备 | F/FINAL_BACKUP_LOCAL_AUDIT.json实际PASS：9分卷、1,300成员、14,409,004,551解压字节；所有178状态SHA与独立payload ledger及89任务结果匹配；412文件安全提取，10份导出和13份预测提取后重新核对文件SHA | **已证明。** 本审阅重新读取verifier实现并核对其SHA与收据一致、manifest SHA一致，抽查实际412文件/10个四文件导出目录/13份预测/代表best与last存在。未重复全量hash；其余88运行完整状态在已逐成员核验的分卷内，没有全部解包或逐目录恢复，不扩大为89次CUDA续训演练。 |
| GPU小时和实际费用 | FINAL_RESOURCE_AUDIT：5个Goal Job实测分配33.2202777778 GPUh；平台running口径32.9347222222 GPUh；已核实费率1点/GPUh，CPU/内存0；F/FEE_ACCEPTANCE_20261016.json绑定该审计与原Goal，保存原问题和用户直接回答 | **费用交付已按最新用户澄清接受。** Goal专属实际结算仍无法证实，actual_goal_settlement_verified=false；33.2202777778/32.9347222222点均为相应GPUh乘费率的计算金额。个人项目48.27含既有非Goal Notebook，不能可靠分离；用户接受保留此限制，不等于账单核实。 |
| 更新06报告与项目交接 | REPORT.md、README.md、IMPLEMENTATION_STATUS.md、PROJECT_HANDOFF.md、PLAN与故障/恢复说明已更新；F/FINAL_DELIVERY_STATUS.json绑定证明和文档hash，未解决要求为空；首个92文件交付commit由FINAL_GIT_DELIVERY_AUDIT核验 | **已完成本轮报告与首次交付保存。** 旧快照注明被取代，报告保留六指标/seed、legacy、loss/NLL、速度/容量、消耗与研究边界；费用接受不写成实际结算核实。随后只同步最终总核查/Git已完成的状态说明及相应文档hash，不修改训练代码或实验结果。 |
| 及时释放自有8卡、不影响其他作业 | SUPERVISOR_V3_TERMINAL_PLATFORM/EVENTS：15:14:19 SUCCEEDED，15:14:40 Pod删除，剩余node_count0；FINAL_RESOURCE_AUDIT五窗口已结束 | 已证明Goal GPU释放。既有非Goal单卡Notebook及零费CPU桥接不属于要停止的资源。当前CPU交付不需重提GPU。 |

## 完成判定与最小剩余动作

本审阅不建议增加训练、搜索、正式测试或GPU测速。没有发现必须用新GPU才能补齐的研究验收缺口；有界数值证据、CPU恢复范围和测量噪声边界均应如实陈述，不通过追加无关实验“消除”表述限制。

1. **已完成：** 九分卷全量本地验证及412文件安全提取，收据FINAL_BACKUP_LOCAL_AUDIT.json实际PASS；代表事务恢复也已通过。实际提取根为outputs/stage06_continuous/final_delivery_20261016/verified/，独立恢复目录为同级recoverable_primary/。本审阅核对verifier/manifest哈希和实际目录计数，不重复全量hash、测试或训练。
2. **费用交付边界已接受：** F/FEE_ACCEPTANCE_20261016.json保存人类直接回答“接受，以此完成Goal”，接受实际时长乘平台费率的两口径计算以及“未核实结算”说明。actual_goal_settlement_verified仍false；33.2203/32.9347点不能改称实际结算。按原Goal加最新用户澄清，费用交付不再构成实质未完成项，无需继续追查未提供的逐Goal账单来阻止完成。
3. **最终总核查与首个交付commit已完成：** FINAL_DELIVERY_STATUS记录所有要求证据通过、unresolved_required_items为空；FINAL_GIT_DELIVERY_AUDIT绑定92个显式审阅文件与aec2307bc4e5abf79b7ed22df8f1150b4ddfaedf，工作字节/暂存路径/whitespace及保护检查均通过，未push。父任务随后只显式提交有限完成状态说明，并刷新其文档hash；不由此修改实验事实或训练版本。本审阅未stage/commit。

实验、独立评估、完整本地备份与代表恢复均有实证，费用交付的现有限制也已由用户明确接受；最终总核查与首个92文件交付commit已完成，依原Goal和最新澄清未发现实质未完成要求。此文件只记录已核实交付事实，不声称Goal工具状态已经complete；最终状态以父任务的Goal工具操作为准。实际结算未核实这一事实持续保留，不因交付或Goal完成而改为已核实。仅同步完成状态说明不需要新GPU。
