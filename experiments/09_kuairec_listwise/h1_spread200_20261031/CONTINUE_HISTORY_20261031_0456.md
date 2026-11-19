# 本轮实验继续位置

2026-10-31 01:46。Goal active；唯一正式训练已启动，尚未完成验收。不要重复提交，不改code目录。

**01:55增量**：实时RUNNING，新日志R3665/T20（R6新增200R步），首个8192诊断已完成。五个交付脚本已齐备：`analyze_d.py`、`temporal_d.py`、`process_d.py`、`archive_delivery.py`、`cold_restore.py`；全部--help通过，归档/全新解压真实小样例通过`delivery_accept_qighk8tc/`。尚未对完整D产物运行，不能当作科学验收完成。下文脚本未齐备的旧文字应以本增量为准。C盘空余26.3GB，最终封存前再次核量。

**02:04增量**：作业仍RUNNING，R4065/T20（R6第600步）。`report_d.py`中文科学报告和四类PNG/SVG/PDF工具已齐，本地matplotlib可用。`EVIDENCE_PLAN.md`列逐项最终验收。白名单协议/证据/六脚本已上传远端root/acceptance_tools_r1.tar.gz，双方SHA`ea38e2b1ed75055f135d494b16461208d41a1402c877fe4627a8335acdb09e16`；这是准备包而非最终通过报告。实际环境GPU torch2.6.0+cu118/H100，CPU torch2.6.0+cpu。

**首完整R检查**：R6全部693更新已完成，R4158/T20，首R后8192诊断进行中。真实713行新训练前段已下载并逐行独立重建batch SHA/样本量、R/T计数、原35R余弦LR、T LR、原400T噪声幅度和温度，全部通过`EARLY_LOG_ACCEPTANCE.json`（prefix_only，不是终点）。`analyze_d.py`增加完整噪声幅度/温度检查；工具包r1为较早准备副本，最终备份应另包实际执行版本。

**02:10增量**：首R后8192诊断完成（271.6s），正式作业已发布R6边界、继续phase7。真实远端receipt已回读并另存`EARLY_R6_CHECKPOINT_RECEIPT.json`：79,930,096字节，SHA`312e51e3bc6af680d37a53dfcc851e698af62ab915e8b2355b457289b633f033`，双优化器/scheduler/RNG/binding字段齐全；这是收据检查，未冒充独立冷恢复。当前为等待持续的唯一正式作业，不是阻塞或终态。

**早期预算复核**：正式作业实时RUNNING。693次真实R平均1.2387s/步，20T平均1.2133s/步；使用已完成较慢的8192诊断284.733s线性估全部986798评估样本，未加余量16.7483h，加25%余量20.9354h，低于23.9h硬上限。`FORMAL_EARLY_BUDGET_CHECK.json`只是第一周期吞吐估计，未改预算/科学要求，也不是结算或完成承诺。

**02:38增量（最新）**：同一正式作业持续RUNNING；R7已完整发布边界R4851/T20，R8也完成693步，真实训练日志终点R5544/T20。首份完整验证已开始，日志`evaluation_progress validation processed1024/71657 seconds35.76`，对应`runs/H1-Spread200-R35/evaluations/validation_R08/attempt1/`。预计约40多分钟，但以完成收据为准；gzip预测仍在写时不得当作完整结果或运行完整终点分析。当前没有故障、没有新增资源、没有新的科学结果结论。

**03:21增量（最新）**：R8完整验证已完成并发布R8边界，正式作业进入phase9第二个20T块，最新T40/R5544。已下载首份完整record/summary/predictions到`first_validation_R08/`，两文件SHA与记录一致；71657条/4733用户，H/seed2024/FP32/beam32/R5544/T20正确。真实独立CPU复算passed、0不一致、最大误差4.81e-8，NDCG@5=0.0062536551、HR@5=0.0285666439、Recall@5=0.0058305539，评估2510.28s。`audit.json`和`ACCEPTED.json`明确只验一节点，非终点/Goal通过。`analyze_d.py`在最终归档数据SHA完全相同、core/数据绑定一致时复用这份已接受审计，避免重复复算；其他新节点仍正常审计。源checkpoint尚待最终备份和冷恢复。

**03:45增量（最新）**：同一正式作业持续推进，第二20T块完成；R9完整训练及8192固定诊断已完成（诊断288.51s），发布边界R6237/T40并进入R10。下一完整验证R11。没有故障/新增作业/预算调整，仍为正常训练等待。最终报告生成后须根据实际SID/碰撞/位移、前后时间段及分组结果补写清楚的定量解释与证据边界，并审阅四类图表；生成器的默认文字本身不替代研究判断或最终验收。

**04:14增量（最新）**：同一正式作业持续RUNNING；R10完整边界R6930/T40已发布，R11训练693步结束R7623/T40，第二份完整验证正在进行（3008/71657条，103.57s）。路径`evaluations/validation_R11/attempt1/`；必须等完成收据，不能分析正在写的gzip。早期R8接受结果仍是唯一完成的独立验证节点。没有重启、新增资源、科学要求或预算变化。

**04:55增量（最新）**：R11完整验证已完成，第三20T块到T60/R7623，正在其8192诊断。R11正式record/summary已下载到`saved_validation_R11/`并核对SHA及71657条/H/seed2024/FP32/beam32/R7623/T40；summary SHA`d244faf6609ba60790ac59088ce3e1a6828cc6085731957767654848020bd736`。生产记录NDCG@5=0.0057987429、HR@5=0.0261942308、Recall@5=0.0053114142，评估2468.43s。`SOURCE_CHECK.json`明确只完成来源/头信息检查，逐样本预测独立复算留最终统一审计；不要把它写成R11独立审计通过。R8已接受审计与R11元数据检查的证据等级不同。早期指标没有触发调参或选模，后续正常等R12—14/下一完整验证R14。

## 当前事实

- CPU九项、原父状态、204800条T输入、30个R顺序通过。真实H100短测通过，原R5锁定128条Top5一致、前20T数值配对、下一T/R精确恢复、参数隔离通过。
- 短测`nancy-s09-d-preflight-1031-r1`已SUCCEEDED，创建至结束142秒保守记账。正式`nancy-s09-d-1031-r1`于01:43:36分配单H100 qb-prod-gpu042；平台时限23.9h，进程86040s。无自动重试/通知。
- 正式实查已到R3465/T20，phase5，400T原退火时钟和原T学习率正确；正在块后8192固定诊断。尚无正式完整验证，不宣称收益。
- 远端root：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1`；正式输出`runs/H1-Spread200-R35`，日志`jobs/formal/train.log`和`status.json`。父/代码/命令SHA见DEPLOYMENT.json。
- CPU入口`nancy-s09-d-prep-1031-r2` RUNNING（1CPU/4GiB）；r1已STOPPED。r2从约01:25起23h自动停止，预计足够19h流程及多数收尾，如临近须处理，不能误认旧Notebook。

## 工具调用

- 实际CLI Python：`C:/Users/nancy/AppData/Roaming/uv/tools/inspire-skill/Scripts/python.exe`。
- 始终走本目录`inspire_compat.py --json --account sii-shared ...`，只适配实时workspace列表迁移，其他资源API仍为原CLI；需网络/缓存权限时require_escalated，已有用户Goal明确授权。
- notebook exec的受控命令末尾加` # end`，吸收Windows SSH额外回车。保持JSON模式防止平台连接URL/token出现在输出。不把调用失败当作远端作业停止。
- CPU共享入口可tail/cat/scp，不重复旧推理/统计。单次读取新日志末尾，不扫描全部旧输出。
- `job status/instances/events --tail N`看平台；共享盘`jobs/formal/status.json`与训练日志看实际进度。终态才核算实际时长和释放状态。故障先查已有job，禁止重复创建。

## 后续工作

1. 确认首个T块及R6健康；按真实吞吐复核预算，不压缩协议。正式硬上限外另留180秒释放；短测142＋formal86040＋180=86362秒<86400。
2. 训练完成应为R24255/T200、40个新增阶段、20990个新增更新。验证十节点，best仅共同六节点，最多两个测试。确认`complete.json`及`test_predictions_complete.json`、jobs/formal/result.json。
3. 新的CPU分析脚本`analyze_d.py`已写好并检查--help，尚未实际使用（没有预测可用）。从本地新完整D run输入，核对计数/顺序/选模、独立复算所有新预测，复用原A已接受审计，再作A-D 2000用户配对bootstrap。不可将脚本存在称为分析完成。
4. 时间诊断用原冷热门分析`h_freeze35_20261029/cold_popularity_audit_20261030`的已保存shared_arrays/train_item_counts及已接受旧结果，不重算旧统计。新D35与A35验证比较，切点1598745441188500、前38644后33013，旧H0 best单列。可复用rank/gains/shift_bootstrap等函数，但旧main会重算旧模型，不能直接调用main。
5. 审核每块前/20T后/第一个R后8192诊断，完整验证可提取不重复推理；20个catalog边界及initial、SID变化/碰撞/槽使用/位移/Mixer幅度。旧H1 R35/T400仅读现有验证曲线。
6. 新源码、全部阶段checkpoint/优化器/scheduler/RNG/双时钟、日志/预测/诊断完成归档，本地逐文件SHA、独立解压、关键checkpoint真实CPU冷恢复（不作参数更新）。报告科学结果和证据边界。
7. 实时核验新GPU实例释放、停止本轮CPU r2，保留远端科学产物。只有上述均有通过证据才complete Goal；结果不正面无需加实验。

## 已知约束

- 目前launcher的`jobs/formal`目录exist_ok=False；如实际发生需要续跑，应新增明确attempt日志支持，保留旧attempt并绑定最新状态，先查job终态及预算，不直接重启同launcher或删目录。
- 若真正需要从滚动last续跑，还要先审核training.jsonl是否存在checkpoint之后的已记录更新。当前runner为追加日志，重放可产生重复计数；不能删改原日志或直接当作完整验收。应保留各attempt原始日志并核对重放、建立有来源的规范日志审计后再判断。现在没有失败，不为假设修改封存训练代码。
- 不能在正式code/scripts中加Python文件，否则严格执行绑定会变化；新分析脚本只能在本目录或远端root的独立分析目录。
- 所有原实验、无关未跟踪文件、旧监控和旧Notebook保持。禁止git add .。没有Feishu写入、对外消息授权。
