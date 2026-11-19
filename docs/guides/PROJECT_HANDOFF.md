# 项目交接摘要：Semantic ID × 生成式推荐

09冻结续训最终收尾已验收（2026-10-30 12:40）：三组实验与五模型冷热门诊断、完整备份和独立恢复已交付；原0GPU Notebook已由平台独立复查确认为STOPPED，09-h1跟踪已删除。全部已授权GPU和CPU入口均已释放，无需继续训练、统计或资源查询。最终Git/Goal签收记录见[闭环收据](../../experiments/09_kuairec_listwise/h_freeze35_20261029/FINAL_DELIVERY_CLOSURE.json)；[主报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/REPORT.md)及[五模型诊断报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/cold_popularity_audit_20261030/FINAL_REPORT.md)保持已审核原字节。下方CPU pending描述均为历史快照。

09本轮结果与后处理归档已交付（2026-10-30 12:02）：内容提交563f8db，411文件后处理tar与Git bundle独立恢复通过；主实验/冷热门报告、6.85GB运行包、66状态/9关键文件实际恢复与GPU释放均已验收。仅原0GPU文件Notebook因CLI连接重置尚未确认停止，已请用户Chrome协助；Goal及09-h1继续用于这一项收尾，不需重跑任何科学工作。见[闭环记录](../../experiments/09_kuairec_listwise/h_freeze35_20261029/FINAL_DELIVERY_CLOSURE.json)。下方早先的“Git/后处理待完成”描述已由此条和实际备份收据补齐。

09冻结续训最终科学交付（2026-10-30）：三组R35、全部预定验证/测试及独立复算完成。主对照同R20冻结A−C固定35测试NDCG5 +8.48%，95%用户区间[+0.00025060,+0.00115133]；共同best A32/C23补充+11.33%。H1验证R32最佳、R35回落；B从旧best32晚冻到35无改善。单seed、冻结点源于H1观察及选模不确定性边界保留。[主实验报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/REPORT.md)与[交付索引](../../experiments/09_kuairec_listwise/h_freeze35_20261029/DELIVERY_INDEX.md)是最新入口，下方历次进度保留时间语义。

新增冷启动/热门诊断也已补齐C公平R32和原选best23并独立审核、交付：[完整五模型报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/cold_popularity_audit_20261030/FINAL_REPORT.md)。冷目标增加不是固定粗分层下的主要组成解释（公平对照约0.31%）；暖mid内部变化主导，H1训练head份额较低但少数商品集中更高。冻结后续训缓解集中，后段仍领先；未改数据切分或评估协议。

最终运行tar 6,846,443,520字节及3,405成员核SHA，63不可覆盖+3last全部检查、9文件10关键角色真实CPU恢复通过；数据/PSE/历史预测是已验证外部依赖，包非自包含。已授权4项GPU分配累计21.810556小时、全部释放，结算未核实。后处理独立归档、Git以及原0GPU CPU入口关闭的真实收尾证据见本轮FINAL_DELIVERY_CLOSURE；不要把该入口此前的网络失败误认为GPU仍运行，也不要在闭环收据产生前替Goal宣告完成。

09连接恢复并进入最终交付（2026-10-30 10:33）：用户确认Chrome正常后，原CLI查询在未修改连接配置的情况下恢复，原因未确定。A/B/C均SUCCEEDED；C于05:00:43删除GPU Pod、05:00:50同次收尾完成。四次已知分配含短测累计21.810555556 GPU小时，30小时内，结算未核实。原0 GPU CPU入口已恢复，C真实R35/T200、24,255次R更新，best依原共同网格为R23，R35/R23测试预测已生成。C R32/R35验证来源与SHA已核并报告，完整逐样本复算、C冷热门补充、全包下载/实际恢复仍待；不可宣布Goal完成。最终6,846,443,520字节tar已生成并开始下载，见[执行记录](../../experiments/09_kuairec_listwise/h_freeze35_20261029/EXECUTION.md)、[归档收据](../../experiments/09_kuairec_listwise/h_freeze35_20261029/FINAL_PACK_RECEIPT.json)及[释放审计](../../experiments/09_kuairec_listwise/h_freeze35_20261029/final_lifecycle_r1/ACCOUNTING_REVIEW.md)。下方连接受阻记录仅描述旧时间段。

09第二次连接受阻（2026-10-30 02:34）：恢复后连续三轮原Job查询仍ECONNRESET，现有代理curl重置、直连无法解析。三模型冷热门初版已完成，独立可推进的交付准备已核；C后续及最终验收需要恢复原入口，按规则标记Goal受阻，不能宣布完成或重提训练。最后成功快照02:05:36为C R32验证22,016/71,657，当前状态未知。见[本次阻塞记录与续接步骤](../../experiments/09_kuairec_listwise/h_freeze35_20261029/CONNECTION_BLOCKER_20261030T0234.md)；C诊断补充、最终测试/归档/真实恢复与释放仍待，跟踪保留。

09冷启动与推荐集中诊断（2026-10-30）：用户授权的三模型CPU初版已完成并独立复核，见[报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/cold_popularity_audit_20261030/REPORT.md)。未作为训练目标的验证目标占比从7.71%增至12.41%，但固定五类的组成分解不能主要解释H1相对H0的后段收益减弱；H1更集中于少数训练中频商品，不是更多推训练head。冻结续训后后段集中度36.88%→27.74%、NDCG5相对旧H0+8.77%，但时间差仍存在。C固定R32和原规则best的更公平补充仍待，不据此改数据或训练。最后成功文件观察02:05为C R32完整验证22,016/71,657，之后入口与平台查询再次重置；不得据此重提GPU。原三组Goal、最终测试/备份/恢复验收继续，细节与报告去重状态见本轮EXECUTION、诊断STATUS和NOTIFICATIONS。

09观察恢复（2026-10-30 00:59）：平台认证及原Job查询恢复，原0 GPU CPU Notebook确认停止后恢复同一实例及官方连接；未重启或重提GPU。A/B已完成，C完成R29训练（20,097R/200T），当时完整验证50,016/71,657。C的R23/R26全验证NDCG@5为0.00889765/0.00854835，同轮次A高8.22%/21.25%；源文件SHA、协议和完整状态收据已核并通知，独立逐样本复算/配对区间及C最终测试待完成。00:53成功查询的GPU台账快照为17.683056/30小时，非最终结算。此前连接阻塞已解除，完整Goal仍待C终点、全包和实际恢复验收。见[执行记录](../../experiments/09_kuairec_listwise/h_freeze35_20261029/EXECUTION.md)。下方受阻记录保留历史时间语义。

09冻结续训连接受阻（2026-10-29 22:49）：连续三个Goal轮次查询原C Job均失败，代理连接重置、直连无法解析域名；当前运行状态未知。最后成功文件观察22:12为C R23训练完成、验证47,008/71,657；A/B既有结果已完成并释放。C R20重放及R21/R22冻结端点真实CPU比较通过，仍不能替代最终恢复。完整Goal、原Job和30 GPU小时上限保持，先恢复观察再取原结果，禁止据查询失败重提；C主对照、全包、63＋3状态/关键模型恢复及最终交接未完成。见[受阻记录与精确续接步骤](../../experiments/09_kuairec_listwise/h_freeze35_20261029/CONNECTION_BLOCKER_20261029T2249.md)。

09时间分段补充（2026-10-29 21:30）：用户要求的H1冻结R32 vs旧H0 best验证前后段已完成，仅CPU复用预测；[报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/temporal_validation_r32_20261029/REPORT.md)给出原时间边界、37指标、6项配对区间。冻结R32对H0前段+27.16%、后段+8.77%；对自身R20前+13.84%、后+14.46%，各NDCG5区间均正。后段更大K收益不明确；旧R20后段−4.97%本次区间跨0。真实检查点编码端/T优化器/SID完全一致，R内部SID embedding及推荐器更新；这是端点读取比较，不是最终实际恢复。支持继续优化推荐器改善整体适应，不能证明只解决时间泛化或Mixer因果；两段用于选模、单seed边界保留。固定R35 A−B补充CPU区间也已完成（测试+5.91%，验证+20.97%），见[补充报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/early_ab_comparison_20261029/REPORT.md)；A−C主对照继续，21:34:17 C到R23/T200，下一完整验证待完成。未新增GPU推理、未修改训练/选模。

09冻结续训最新进度（2026-10-29 21:04）：C已完成R20完整验证和冻结关口，进入R21/T200。71,657条Top5与原H0 R20全部一致，复算NDCG@5=0.007728139395127303；phase5—24全部20段loss/梯度范数记录最大差0。独立R20完整状态及字节SHA已核，见[关口证据](../../experiments/09_kuairec_listwise/h_freeze35_20261029/C_R20_REPLAY_FREEZE_EVIDENCE.json)，仍不宣称旧权重逐位恢复或新模型最终恢复完成。连接已通过原CPU Notebook官方refresh恢复，无重复GPU提交。A/B已经成功释放；20:57预算快照累计13.749166667/30 GPU小时，C仍继续消耗。剩余C至R35、验证/测试、A−C对比和63+3完整备份/实际恢复继续按原Goal完成。下方带日期段均保留各自历史时间语义。

09阶段报告已交付（2026-10-29 20:39）：[飞书初稿](<feishu-link>)已用用户身份创建并全文回读，21张表的全部文本单元格及4图引用/图注与本地稿一致。内容串起KuaiRec数据、首轮J/H四组、泛化/塌缩诊断、Decay/Freeze R26、当前A/B/C；[本地稿与交付收据](../../experiments/09_kuairec_listwise/progress_report_20261029/DELIVERY.md)可追溯。C在20:40:07已完成R20训练（13,860R/200T），全验证进度30,016/71,657；尚无R20关口通过证据，不能写已进入冻结。A/B实际新模型恢复及63+3完整全包仍待统一验收。飞书交付不代表实验Goal完成；不重复训练、下载或发起飞书授权。

09冻结续训最新进度（2026-10-29 20:02）：H1验证best=R32。固定R35及best R32测试已独立CPU复算，连同3旧模型共5份审计、6项用户配对区间通过，已按要求先报告、不等C。H1 R32测试NDCG@5=0.00943079，较旧H0 best R32的0.00860706高9.57%，差值95%区间[+0.00039119,+0.00128379]；较Freeze R26高2.99%，区间跨0。H1固定R35=0.00895634，有回落。A/B GPU均已释放，C的R8/R11/R14重放关口通过，19:42补充R17也全71,657条Top5一致、NDCG差0，R20门槛和统一冻结主对照仍待。先读[H1测试报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/H1_TEST_REPORT.md)、[阶段报告](../../experiments/09_kuairec_listwise/h_freeze35_20261029/REPORT.md)和[逐项验收](../../experiments/09_kuairec_listwise/h_freeze35_20261029/FINAL_DELIVERY_AUDIT.md)。A完整1.34GB包及70成员独立解包已核SHA，正式本地A的11审计(9新复算+2明确历史复用)/4配对及独立交叉复审已通过；早期CPU结果包43成员亦已核，实际模型恢复仍待最终全包。正式本地A及最终all-ready使用analyze_results_compat.py：仅复用已验历史R23/R26审计来兼容旧summary缺groups，原helper/core/B绑定不改。C完成后做固定R35主A−C和共同网格best补充比较。Goal仍active，累计30 GPU小时授权、19:54快照12.700278小时且C继续消耗，非最终结算；不要重提已有Job或因H1报告停用09-h1 heartbeat。 20:00 C快照为R19/T200；A下载/本地分析handles均已正常结束，不重复下载或推理。

当前进行中（2026-10-29）：09三组冻结续训已获批本轮累计30 GPU小时（含短测0.0608333）。A=H1 Freeze R26→R35/T200；B=旧H0 best R32→R35/T360；C=H0 R5按原交替设置重放至R20/T200、核验后冻结至R35。三组独立单H100作业均已实际开始；H100短测的C前40T＋4R与原日志逐项差0，但完整重放和新验证/测试尚未完成。科学底座保留1c42366，新增恢复源码e4ca4b3、短测入口be24824；禁止将当前HEAD覆盖远端执行副本。先读[本轮执行与唯一作业](../../experiments/09_kuairec_listwise/h_freeze35_20261029/EXECUTION.md)、[GPU台账](../../experiments/09_kuairec_listwise/h_freeze35_20261029/GPU_LEDGER.json)与[原Goal](../../experiments/09_kuairec_listwise/h_freeze35_20261029/GOAL_OBJECTIVE.md)。H1每次R29/R32/R35完整验证须及时单独报告，R35训练完成及测试独立核验完成也分别先报告，不等C；heartbeat 09-h1每5分钟跟踪，按NOTIFICATIONS去重。不能把H1阶段完成当整个Goal完成，也不能重复提交已有三任务。下面已交付的实验均是历史，不与本轮费用混算。

最新补充测试（2026-10-29）：原Decay、Freeze两个R26测试任务已完成，无重提/重训；统一53194条测试、4153用户、beam32/FP32。R20/Decay/Freeze测试NDCG@5=0.00794367/0.00456691/0.00915702，相对R20为−42.51%/+15.27%；Freeze差值95%用户聚类区间[+0.00083371,+0.00159351]。三份独立复算、三组bootstrap、本地结果包SHA验收通过，两GPU已释放，新增0.974167 GPU小时，估0.97点、结算未核实。见[测试报告](../../experiments/09_kuairec_listwise/h1_recovery_test_20261029/REPORT.md)。Freeze仍保留Mixer，不能据此证明交互优于H0；下面“未测试”的表述仅指原续训验收时点。

当前追加实验（2026-10-29）：H1两臂R20→R26训练、17份独立复算、三份用户聚类bootstrap、曲线及本地备份/CPU恢复均已完成，最终逐项审计已通过，本轮实验交付完成，见[最终审计](../../experiments/09_kuairec_listwise/h1_recovery_20261028/FINAL_DELIVERY_AUDIT.md)。完整验证NDCG@5：Decay=0.00615930（较R20−34.28%），Freeze=0.01036453（+10.59%，差值95%区间[0.00064178,0.00131981]）。Freeze保留已学到的码本交互，只停止后续编码更新，不等于H0。单seed验证结论不代表测试泛化。两臂均R18018，Decay T280、Freeze T200；本轮GPU全部释放，累计6.556667 GPU小时，估6.56点、结算未核实，低于8.5小时上限。先读[本轮报告](../../experiments/09_kuairec_listwise/h1_recovery_20261028/REPORT.md)、[交付索引](../../experiments/09_kuairec_listwise/h1_recovery_20261028/DELIVERY_INDEX.json)和[执行记录](../../experiments/09_kuairec_listwise/h1_recovery_20261028/EXECUTION.md)，不要重复提交训练。新增恢复源码`8b5acc8`配合认证历史底座；新归档不重复包含数据/PSE，恢复依赖已在索引与CPU收据绑定。下方原09四组、旧三支建议均为历史，不与本轮混算。

核对日期：2026-10-28。用途：新对话接手研究时的第一入口，不替代各阶段报告和原始证据。

09最终交付（2026-10-28）：四组35R/400T、42次完整验证、四组统一测试及固定子集beam32/256均已完成；12份最终预测与best/R35完整验证已独立复算，6份用户配对bootstrap完成。测试单列表NDCG@5：J0/J1/H0/H1=0.005795/0.004605/0.008607/0.007944；H1相对H0为−7.71%，不能用验证best+5.07%宣布测试提升。H1/J1后期码字塌缩已由实际码本复算确认，后续优先做编码更新稳定性诊断，尚未追加调参。两暖身、四run、源码/资产及两侧包完整本地SHA与独立恢复通过，所有13个Job的GPU已释放；累计41.198889 GPU小时，估41.20点，未核实结算。先读[最终报告](../../experiments/09_kuairec_listwise/REPORT.md)、[结构化结果](../../experiments/09_kuairec_listwise/SUMMARY.json)、[交接](../../experiments/09_kuairec_listwise/HANDOFF.md)。以下为有时间标记的历史快照，不是当前待办。

09历史收尾快照（17:55）：四组35R/400T及42份完整验证已验收，原训练GPU已释放；best J0/J1/H0/H1分别R20/R5/R32/R20。四份完整run已下载、SHA核验、独立恢复，J1/H1 CPU模型前向通过。最终评估当前J0/J1/H0 r1与H1 r2在跑；H1原r1因节点不可用失败已释放，保留原产物，r2只改任务/输出名，沿用冻结best及全部科学设置。H1后期真实SID塌缩已确认（9600→2551，碰撞商品12.91%→91.41%），只读诊断与优化候选已更新，未启动调参。Goal仍active，最终测试复算/侧包/完整报告未完成。**接续先读[收尾进度与r2路径](../../experiments/09_kuairec_listwise/formal_execution_20261027/FINAL_DELIVERY_PROGRESS_20261028.md)**；现有侧包helper写死H1 r1，需要受控适配，不能把失败结果纳入正式分析。以下旧快照保留时间语义。

09收尾连接（15:37）：原CPU Notebook已从STOPPED恢复到RUNNING，当前`hpc-compute716`、0 GPU/4 CPU/16 GiB；官方SSH读取和SCP实际下载源锁通过SHA核对，连接入口可用于后续验收与备份。15:37平台剩余运行时间14110秒，后续使用前仍需核实状态。未重启GPU，正式模型验收/备份尚未完成。见[CPU文件入口验收](../../experiments/09_kuairec_listwise/formal_execution_20261027/CPU_FINAL_DELIVERY_READY_20261028T1537.json)。

09最新完整验证与资源（15:24）：原两卡Job仍RUNNING，四组R32完整验证结束，统一71657条、22176R/360T、seed2024、beam32/FP32及相同数据绑定。单列表NDCG@5依次0.005461/0.000650/0.008920/0.003558；J1−J0为−88.10%，H1−H0为−60.11%，聚合NDCG@50差值−92.17%/−43.82%。H0本轮回升，两个交互版继续下降，原因尚未确定。摘要协议及算术经Python和PowerShell两条实现核对，未逐样本复算。完整验证38/42次，快照时J在R35、H在R33，均400T；最终best未固定、测试未开始。截至15:24:11累计分配35.011389 GPU小时，总承诺仍59.759722小时，未超75小时授权；近三小时worker聚合GPU采样均值59.16%，不是逐卡或平台判定保证。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)、[R32核对](../../experiments/09_kuairec_listwise/formal_execution_20261027/R32_VALIDATION_OBSERVATION_20261028.json)和[资源复核](../../experiments/09_kuairec_listwise/formal_execution_20261027/RESOURCE_WINDOW_3H_AUDIT_20261028T152411.json)。以下快照保留各自时间语义。

09最新完整验证与进度（13:39）：原两卡Job仍RUNNING，四组R29完整验证结束，统一71657条、20097R/320T、seed2024、beam32/FP32及相同数据绑定。单列表NDCG@5依次0.005773/0.001609/0.008361/0.004953；J1−J0为−72.13%，H1−H0为−40.76%，聚合NDCG@50差值−75.27%/−17.19%。J1连续三次验证下降、平均聚合候选数降至9.6681件，是当前设置下后期退化的观察，原因尚未确定，不能替代最终best测试。协议与算术经主任务和独立审计核对，未逐样本复算。完整验证34/42次，快照时J在R32、H在R30，均360T；剩R32/R35验证边界，最终选模与测试未开展。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[R29核对](../../experiments/09_kuairec_listwise/formal_execution_20261027/R29_VALIDATION_OBSERVATION_20261028.json)。以下快照保留各自时间语义。

09最新进度与资源（13:24）：13:23原两卡Job仍RUNNING，J0/J1已完成R29验证并在R30/R31、均360T；H0/H1在R29验证、20097R/320T，进度53024/66016条，共71657条。最新四组齐全结果仍R26，最终测试未启动。近3小时180个worker聚合GPU采样均值60.732639%，低于40%为4个孤立点，回收规则未变；不据此保证逐卡或平台内部判定。截至13:24:32累计分配31.023056 GPU小时，非最终时长或结算；总承诺仍59.759722小时，未超75小时授权。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[资源复核](../../experiments/09_kuairec_listwise/formal_execution_20261027/RESOURCE_WINDOW_3H_AUDIT_20261028T132432.json)。以下快照保留各自时间语义。

09最新完整验证与进度（12:00）：原两卡Job仍RUNNING，四组R26完整验证结束，统一71657条、18018R/280T、seed2024、beam32/FP32及相同数据绑定。单列表NDCG@5依次0.005079/0.002903/0.008763/0.006438；J1−J0为−42.85%，H1−H0为−26.53%，聚合NDCG@50差值分别−36.93%/−14.85%。仅中间摘要，未确定波动原因或最终收益；协议与算术经主任务及独立审计核对，未逐样本复算。完整验证30/42次；快照时J已完成R29训练、H在R27/R28，均320T，属于下一周期。最终best尚未固定，测试未启动。11:58的本地TransportError后只读查询已恢复，原GPU未重启；观察器保留并继续核对。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[R26核对](../../experiments/09_kuairec_listwise/formal_execution_20261027/R26_VALIDATION_OBSERVATION_20261028.json)。以下快照保留各自时间语义。

09最新进度与资源（11:26）：11:23原两卡Job仍RUNNING，四组均18018R/280T并验证R26，J0/J1/H0/H1进度分别59008/64000/34016/45024条，共71657条；最新完整结果仍R23，完整验证26/42次，最终测试尚未启动。近3小时180个worker聚合GPU采样均值60.03125%，低于40%共9点、最长连续3点；回收规则未变，不能保证逐卡或精确连续时长。截至11:26:16累计分配27.080833 GPU小时，尚非最终时长或结算；总承诺59.759722小时，仍在75小时授权内。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[资源复核](../../experiments/09_kuairec_listwise/formal_execution_20261027/RESOURCE_WINDOW_3H_AUDIT_20261028T112616.json)。用户补答Jupyter能打开，无需继续操作；备份阶段仍须重新验证CPU文件通道，不能把01:32恢复证据当作当前连接证明。以下快照保留各自时间语义。

09最新完整验证与进度（10:15）：原两卡Job仍RUNNING，四组R23完整验证结束，统一71657条、15939R/240T、seed2024、beam32/FP32及相同数据绑定。单列表NDCG@5依次0.006183/0.005020/0.008834/0.006016；J1−J0为−18.81%，H1−H0为−31.90%，聚合NDCG@50差值分别−4.99%/−17.86%。H组配对差由R20正向转负，H1覆盖上升但质量下降；仅中间验证波动，未确定原因，不提前作机制或最终收益结论。摘要协议与算术经主任务及独立审计核对，未逐样本复算。完整验证26/42次，快照时J在R26、H在R24，均280T，属于下一周期；最终best尚未固定，测试未启动。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[R23核对](../../experiments/09_kuairec_listwise/formal_execution_20261027/R23_VALIDATION_OBSERVATION_20261028.json)。以下快照保留各自时间语义。

09最新进度与资源（09:28）：原两卡Job仍RUNNING，四组均完成R23训练、15939R/240T并验证中；J0/J1/H0/H1当前验证分别46016/49024/20000/30016条，共71657条。每臂目标35R/400T，最新完整结果仍R20，完整验证22/42次，最终best和测试尚待完成。近3小时180个worker聚合GPU采样均值59.91%，低于40%共17点、最长连续3点，不代表精确连续低利用率时长或逐卡情况；回收规则未变。截至09:26:54累计分配23.101944 GPU小时，尚非最终时长或结算；总承诺59.759722小时，仍在75小时授权内。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[资源复核](../../experiments/09_kuairec_listwise/formal_execution_20261027/RESOURCE_WINDOW_3H_AUDIT_20261028T092654.json)。以下快照保留各自时间语义。

09最新完整验证与进度（08:28）：原两卡Job仍RUNNING，四组R20完整验证已结束，统一71657条、13860R/200T、seed2024、beam32/FP32及相同数据绑定。单列表NDCG@5依次0.006955/0.005564/0.007728/0.009372；J1−J0为−20.01%，H1−H0为+21.27%，聚合NDCG@50差值分别−10.00%/+9.04%。四组自身主要指标均较R17回升，仍只是中间验证，不作稳定或最终收益结论。摘要协议及算术由主任务和独立审计核对，未逐样本复算或测试。计划内完整验证22/42次；快照时J0/J1在R22/R23、H0/H1在R21，均240T，属于下一周期。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)及[R20核对](../../experiments/09_kuairec_listwise/formal_execution_20261027/R20_VALIDATION_OBSERVATION_20261028.json)。以下为保留原时间语义的历史快照。

09最新进度与资源（07:25）：07:23原两卡Job仍RUNNING，J0/J1/H1均13860R/200T并验证R20，H0在R20训练、13601R/200T。最新完整四组结果仍为R17，最终测试未启动。近3小时180个worker聚合采样GPU均值60.16%，最长连续低于40%为2点；回收规则未变，不能推断逐卡或精确连续时长。截至07:25:49累计分配19.065833 GPU小时，尚非最终时长或结算；包含正式完整上限和评估预留的总承诺仍59.759722小时，未超75小时授权。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)及[资源复核](../../experiments/09_kuairec_listwise/formal_execution_20261027/RESOURCE_WINDOW_3H_AUDIT_20261028T072549.json)。最终曲线的PNG/CSV绘制命令已具备，待全部材料恢复后执行并在REPORT中展示，不新增绘图实现。

09最新完整验证与进度（06:49）：原两卡Job仍RUNNING，四组R17完整验证已结束，统一71657条、11781R/160T、seed2024、beam32/FP32及相同完整数据绑定。单列表NDCG@5依次0.005036/0.004182/0.007687/0.007948；J1−J0为−16.96%，H1−H0为+3.39%，聚合NDCG@50差值分别−5.25%/+4.90%。四组自身主要指标均较R14回落，仍只是中间验证，不代表最终收益。摘要协议及算术已由主任务和独立审计核对，未逐样本复算或测试。计划内完整验证18/42次；当前J在R20、H在R18，均200T，属于下一周期。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)及[R17核对](../../experiments/09_kuairec_listwise/formal_execution_20261027/R17_VALIDATION_OBSERVATION_20261028.json)。以下快照保留各自时间语义。

09最新进度与资源（05:34）：原两卡Job仍RUNNING，J0/J1完成R17训练并验证中，H0/H1正在R17，均160T；每臂目标35R/400T，最新完整结果仍R14。近3小时180个worker聚合GPU采样均值60.54%，重新读取的Job回收规则仍为低于40%持续3小时，不能据此保证逐卡或平台内部判定。截至05:34:29累计分配15.354722 GPU小时，最终结算未核实；全部已结束用量、正式完整上限与评估预留合计59.759722小时，仍在75小时授权内。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[资源复核](../../experiments/09_kuairec_listwise/formal_execution_20261027/RESOURCE_WINDOW_3H_AUDIT_20261028T053429.json)。以下旧快照保留原时间语义。

09当前阶段及待执行分析：05:03原两卡Job仍RUNNING，四组R14完整验证结束，已进入下一周期，J0/J1在R16、H0/H1在R15，均160T。新增本地CPU SID统计入口及5项合成检查已通过，真实最终诊断未执行；最终模型恢复后须按[分析命令第7节](../../experiments/09_kuairec_listwise/formal_execution_20261027/FINAL_ANALYSIS_COMMANDS.md)输出四组best/last的SID碰撞统计，不能拿初始14.03%代替训练后状态。该工具未上传，也未修改训练或既有final验收helper。见[准备收据](../../experiments/09_kuairec_listwise/formal_execution_20261027/CHECKPOINT_SID_DIAGNOSTICS_PREPARATION.json)。

09最新完整验证：R14四组均71657条、9702R/120T、beam32/FP32及相同数据绑定，单列表NDCG@5依次0.005792/0.005001/0.008307/0.008571。J1−J0为−13.66%，H1−H0为+3.18%，轮次间仍有波动；只是中间验证，未逐样本复算、未固定最终best或测试，不代表稳定交互收益。计划内完整验证已完成14/42次。完整原始摘要、协议及差值已保存；见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[R14核对](../../experiments/09_kuairec_listwise/formal_execution_20261027/R14_VALIDATION_OBSERVATION_20261028.json)。以下运行快照均保留各自时间语义。

09资源更新：03:43原Job仍RUNNING，J0/J1的R14验证各11008/71657条，H0/H1在R13/R14训练，均120T。03:44最近3小时查询返回180个一分钟worker聚合采样，GPU均值59.77%，最长连续低于40%为2个采样点；这不保证逐卡利用率或平台内部回收判定。累计分配口径截至03:44:26约11.6864 GPU小时，尚非最终时长或结算；已结束用量＋正式完整上限＋评估预留共59.759722小时，仍在75小时内。见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[资源复核](../../experiments/09_kuairec_listwise/formal_execution_20261027/RESOURCE_WINDOW_3H_AUDIT_20261028T034426.json)。

09上一轮进度快照：02:18原两卡正式Job仍RUNNING，四组均为累计R11/7623R更新/80T，正在完整验证；最新完整结果仍R8，测试未开展。02:16自启动约2.49小时的平台worker聚合GPU平均58.34%，尚非完整3小时或逐卡检查。已整理方法报告草稿，最终结果与备份待填；见[执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)。

09观察入口：01:53已启动本地只读watch（当前执行会话`67403`），每5分钟写`outputs/stage09_delivery/goal_20261027_r1/OBSERVE_STAGE09_LATEST.json`和追加JSONL；首轮J0/J1在R11验证、H0/H1在R10、均80T。先确认已有观察会话及快照时间，避免重复启动；任何过期快照都不是当前存活证明。GPU训练独立于本地观察程序，终态后仍需最终验收。见[启动收据](../../experiments/09_kuairec_listwise/formal_execution_20261027/LOCAL_OBSERVER_START_20261028.json)。

09上一轮完整验证快照：28日01:32，同一两卡Job仍RUNNING，四组R8完整验证已结束，均进入R9、累计80T；当时J0/J1的R更新5976/5939，H0/H1为5587/5698。R8四份完整71657条验证的beam32/FP32/数据绑定和5544R/40T计数已核对，尚未独立逐样本复算；正式测试未开展。CPU文件通道已恢复并实际SCP下载/SHA验收，最终外围备份helper已部署但未打包。没有重启训练或修改科学源码。见[最新执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[连接恢复记录](../../experiments/09_kuairec_listwise/formal_execution_20261027/CPU_TRANSFER_RECOVERY_20261028.md)；下方启动和故障记录为历史快照。

## 09当前进度：四分支正式训练已启动

四组正式交替联合训练已实际开始：`nancy-stage09-four-arm-20261027-r1`于北京时间2026-10-27 23:46:38在`qb-prod-gpu441`获配2张H100。GPU0运行J0/H0，GPU1运行J1/H1，各进程只见一张卡、无DDP。四次ready、实际两卡UUID和两对initial完整状态一致性核验通过；首个更新快照中四组均完成40T，正在R6。J0/J1累计R更新3793/3788，H0/H1为3606/3607。此为运行快照，尚无R8完整验证或正式测试结果。

原始创建/调度收据、`FORMAL_READY_*.json`、`FORMAL_INITIAL_PAIR_*.json`及`FORMAL_PROGRESS_FIRST.json`位于`experiments/09_kuairec_listwise/formal_execution_20261027/`。四分支继续使用冻结科学源码`1c423662d9a25a926328ee6466c30c0f2857b4f6`；调度与资源审计提交`021d349`，不替换运行代码。每臂最终35R/400T，四组best全部固定前不测试。

准备、暖身和资源诊断7个已结束作业累计13535 GPU秒（3.759722小时）；当前两卡占用另计，尚未释放。正式Job平台上限24小时（48 GPU小时），最终评估另留8 GPU小时；连同已用共预留59.759722小时，低于75小时授权。训练及验证短测外推17.7–18.5小时，含25%余量22.2–23.2小时；须以完整验证和实际长期利用率继续校正，不能当作完成保证。平台首次5分钟快照包含初始化，平均34.83%、最后52%；这不是3小时合规证明。

J/H共同5R暖身均已成功释放、完整CPU验收并在本地独立SHA恢复。R1资源诊断仍保留退出失败记录，R2资源诊断正常退出并通过独立等价审计。四组最终模型备份、最终测试及完整交付尚待完成。不要重复提交暖身、短测或正在运行的正式Job。 详见[09执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)和[资源台账](../../experiments/09_kuairec_listwise/preparation_20261027/GPU_USAGE_LEDGER.json)。以下为历史快照。

## 最新资产变更：06/07远端副本已清理，09获准改用global_user

- 09已进入正式共同暖身：北京时间20:35左右，`nancy-stage09-warmup-j-20261027-r1`（`qb-prod-gpu002`）和`nancy-stage09-warmup-h-20261027-r1`（`qb-prod-gpu146`）均RUNNING，各1张H100，无DDP，平台上限分别2.15/2.4小时。见[当前状态](../../experiments/09_kuairec_listwise/formal_execution_20261027/STATUS_START.json)、[J创建](../../experiments/09_kuairec_listwise/formal_execution_20261027/J_CREATE.json)、[H创建](../../experiments/09_kuairec_listwise/formal_execution_20261027/H_CREATE.json)及[J日志](../../experiments/09_kuairec_listwise/formal_execution_20261027/j_LOG_START.json)、[H日志](../../experiments/09_kuairec_listwise/formal_execution_20261027/h_LOG_START.json)。TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果；四分支尚未启动，不重复提交暖身。
- 656 GPU秒只是已结束的准备作业用量；两个暖身已开始新增消耗，Scheduled分别为北京时间20:35:08、20:35:13，最终分配秒数仍为null，见[资源台账](../../experiments/09_kuairec_listwise/preparation_20261027/GPU_USAGE_LEDGER.json)。用户75小时授权已提交于`b1ba3ec`，Goal工具现已恢复`active`。后续先检查既有暖身与完整状态；对应5R及验证完成、共享状态验收后才启动两条同架构分支，当前四分支均未启动。分支提交前须核对平台3小时GPU利用率低于40%的回收风险，不能依据刚启动时的均值下结论。

- 当前预算授权见[75 GPU小时批准](../../experiments/09_kuairec_listwise/preparation_20261027/BUDGET_APPROVAL.json)及[已批准资源计划](../../experiments/09_kuairec_listwise/h100_preflight_r2/APPROVED_RESOURCE_PLAN.json)；J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。

- 用户明确授权后，06 SASRec/eSASRec及07商品内交互的11个远端运行/备份目录已清空；本地21个备份分卷约19GB重新SHA全过，正式及beam诊断报告保留。08此前已清空。05、09、共享输入/环境及源码依赖保留；后文06/07远端路径均为历史快照，不能继续当现存入口。
- 清理前合计约51G的旧目录已移除，但原`experiment1_reorganized`的136KB写入复验仍报`Disk quota exceeded`。06在10月16日已遇过相同主项目配额错误，当时换到global_user后继续；主项目盘点8.2G不是额度上限，实际配额规则仍未核实。详见[清理与存储记录](../../experiments/09_kuairec_listwise/preparation_20261027/REMOTE_CLEANUP_20261027.md)。
- 用户最新同意09使用`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/`，覆盖此前仅原目录的限制。只部署09新资产，不迁移原项目；1MiB探针、代码/数据包及BGE四片/整包/11文件远端回读均已通过，[BGE收据](../../experiments/09_kuairec_listwise/assets_preparation_20261027/BGE_REMOTE_TRANSFER_VERIFIED.json)。原目录扩容未完成；单H100内容编码已实际读写新根，完整Goal容量仍待验收。
- 远端32条真实文本CPU probe已通过：编码5.4594659秒、5.86138件/秒，保守CPU全量估算40.4383分钟。随后内容编码Job `nancy-stage09-bge-encode-20261027-r1`已SUCCEEDED，日志确认10728件、exit0；仅冻结BGE推理，不是推荐训练。编码占用135秒，后续H100验收R1占266秒、R2占255秒；三个准备Job均已释放，已终态的准备作业合计656 GPU秒（0.182222 GPU小时），准备阶段原7200秒上限下剩6544秒（历史）。最新已批准累计75 GPU小时（270000秒），包含已终态作业656秒，269344秒为两暖身分配前余额；当前新增占用和实际余额待平台终态事件核算。
- CPU共同PCA/PSE及独立审计已完成：[PSE资产清单](../../experiments/09_kuairec_listwise/assets_preparation_20261027/PSE_MANIFEST.json)、[独立审计](../../experiments/09_kuairec_listwise/assets_preparation_20261027/PSE_ASSET_AUDIT.json)确认`toy=false`、10728件、码本`[4,256,64]`，最近邻SID及误差重算一致。9513个唯一SID，1505件商品位于多商品碰撞桶（14.0287%），最大桶84件；这是商品可区分性的限制，不是工程失败。正式阶段按用户最新要求以2卡分别跑J/H共同暖身、4卡分别跑四个独立分支，每模型单卡、无DDP；累计75 GPU小时预算已批准，见[09执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)。
- [R1验收](../../experiments/09_kuairec_listwise/h100_preflight_r1/report.json)和[R2验收](../../experiments/09_kuairec_listwise/h100_preflight_r2/report.json)的四臂及各自supervisor均passed，真实BF16梯度、R/T隔离、恢复、共同初始化、SID刷新及无目标五商品生成通过。[R1指标独立复算](../../experiments/09_kuairec_listwise/h100_preflight_r1/METRIC_RECOMPUTE.json)仅覆盖每臂4条beam2短验收样本，不是正式推荐质量。R2微批256、有效batch1024、推理batch32，科学模型、beam、评价定义不变；正式训练不接短验收状态。
- 共同18文件、94,105,600字节资产tar已在`outputs/stage09_restore/goal_20261027_r1/`独立恢复PASS，见[恢复收据](../../experiments/09_kuairec_listwise/assets_preparation_20261027/GENERATED_ASSETS_LOCAL_RESTORE.json)，不含正式模型。[R2预算](../../experiments/09_kuairec_listwise/h100_preflight_r2/BUDGET_ESTIMATE.json)估后续基础58.1855 GPU小时，含25%余量和已用量累计72.9141 GPU小时；2卡暖身、最多4卡分支预计墙钟15.868/19.835小时（基础/含余量，不含排队）。展示单价1点/GPU小时仅作未结算估算；用户已批准累计75 GPU小时上限，J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。

## 当前代码入口：09 KuaiRec列表DiffGRM实现（2026-10-27）

- 此前预算等待期间，本地`83a1b9c`补齐离线分析的冷目标分组来源核验（12项CPU分析测试通过），以及[恢复与最终验收流程](../../experiments/09_kuairec_listwise/h100_preflight_r2/RECOVERY_AND_FINAL_AUDIT.md)。新版analysis仅离线处理结果，未同步远端；训练/代表checkpoint恢复仍用已验收的冻结主版本`1c423662d9a25a926328ee6466c30c0f2857b4f6`，不能因更新分析脚本意外改变训练源码binding。用户现已批准累计75 GPU小时，J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。

- 用户已启动09执行Goal：seed2024的J0/J1/H0/H1四组；原先最多2 GPU小时的准备授权已由最新累计75 GPU小时授权覆盖，包含已终态作业656秒，269344秒为两暖身分配前余额；当前新增占用和实际余额待平台终态事件核算。按已批准多卡方案推进：2卡J/H共同暖身、最多4卡四个独立分支，每模型单卡、无DDP；当前J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。
- 新增`scripts/listwise_diffgrm/`和`configs/09_kuairec_listwise/base.json`：J0/J1为20码共同去噪，H0/H1为逐商品外层＋四码DiffGRM内层；两对分别关闭/开启05全局Full Mixer，全部保留交替联合训练。直接加载原DiffGRM/PSE类，旧引擎和历史入口未改。
- 同架构共享5R暖身，随后按10×(40T更新＋3R epoch)实施；属于新数据初始预算，未证明收敛。新目录真实item级五目标评价与旧SID单目标结果不能直接比高低。
- 初版55项CPU验收通过，覆盖原单商品退化兼容、真实四层Decoder BF16、ST推荐梯度、阶段参数隔离、完整状态恢复、两路线微型完整runner及目录碰撞/指标；[代码验收收据](../../experiments/09_kuairec_listwise/code_acceptance_20261027/CPU_ACCEPTANCE.json)绑定初版源码hash。后续91项执行检查和真实BGE编码进展见本页最新记录，不能将初版CPU验收当作H100推荐训练验收。
- [实现与命令](../research/onerec_listwise/IMPLEMENTATION.md)、[逐项设置来源](../research/onerec_listwise/IMPLEMENTATION_REFERENCES.md)、[设计方案](../research/onerec_listwise/MODEL_EXPERIMENT_PLAN_20261027.md)。下方“尚未实现”记录描述此前阶段，不能覆盖本段当前状态。
- 当前执行进展与预算边界见[09执行记录](../../experiments/09_kuairec_listwise/EXECUTION.md)。平台当前可达性、真实资产和GPU验收以该记录及对应收据为准，不把前面的55项CPU验收扩大为正式通过。
- 09最新存储许可与执行根见本页顶部；此前主项目配额错误及失败上传收据保留作历史证据。执行源码91项CPU检查、预算工具14项检查、真实BGE/PSE资产、R1/R2四臂H100验收及共同资产本地独立恢复均已通过。三个准备GPU Job已释放，累计656秒；最新累计75 GPU小时预算已批准，269344秒为两暖身分配前余额；当前新增占用和实际余额待平台终态事件核算。采用R2效率设置，J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。

## 当前数据入口：KuaiRec五商品消费续接已完成全量准备（2026-10-27）

- 用户已锁定本轮仅数据准备：复用并核验作者KuaiRec 2.0全包及文本，仅用big_matrix，本地CPU，不训练、不生成embedding、不拟合PCA/OPQ/PSE、不修改DiffGRM。原始12,530,806条事件经去重968,004条、隔离22条真实时间并列后，得到11,562,780条有效无歧义观察。四条文本断行无损修复。
- 明确任务是**人工会话内五商品消费续接**，不是恢复OneRec真实请求ID。观察间隔>1800秒分段，段首连续非重叠五件块，目标商品互异且均已播放、跨度≤600秒；历史取严格过去最近5–50条已播放记录，坏目标块的既往观察仍可进入以后历史。主版本不筛总播放≥60秒，仅给训练集可选标记。
- 全局事件时间80%/90% nearest-rank切点为北京时间2020-08-28 18:06:07.593、2020-08-31 21:49:45.624；跨切点整会话不贡献目标。最终834,294个样本：训练709,443、验证71,657、测试53,194。保留并标记冷启动和缺文本；完整静态目录可用性是离线假设，文本未证实为历史快照。
- 正式产物为`outputs/kuairec_listwise_v1/run1/`，`run2/`为独立全量复跑；两轮样本文件、映射和统计一致。33项CPU测试、全量不变量与分位排名检查、从原始ZIP独立重建32用户全部3,693个合格样本、三集合CPU loader均通过。数据与模型权重不进入Git。
- 后续两种列表架构必须共用此数据。模型实现及@10/20/50候选聚合现见顶部09代码入口；正式对照训练尚未开展，不把数据/代码验收当作模型收益证据。[数据报告](../research/onerec_listwise/kuairec_preparation/REPORT.md)、[脚本与复现入口](../research/onerec_listwise/kuairec_preparation/README.md)、[固定配置](../../configs/kuairec_listwise_v1.json)。下方“尚未锁定数据”的表述为此前研究历史。

## 当前入口：08退役，转向OneRec定义的请求级列表生成

- 用户已确认学长的session-wise指一次请求生成一批商品。原08仍是单目标下一商品预测中的历史侧码字交互，不能作为该任务的实验结果。
- 按用户要求，08专属代码/配置/测试、引擎模块、本地模型及恢复包、远端四个执行/评估目录已删除；01–07、共享原始数据、通用推理缓存修复及08轻量结果保留。下方08完成与恢复段落只是历史快照，旧路径不能再使用。详见[退役与清理凭据](../../experiments/08_session_codeword/RETIREMENT_20261026.md)。
- 本轮没有启动GPU。回归共35项，34通过、1项Linux专用检查在Windows跳过。
- 用户明确要求尽量保留item-wise设置：新方案保留PSE＋DiffGRM＋交替联合训练，只适配列表必需接口，再比较Full Mixer开/关；此前默认改AR的建议已撤回。尚未实现或训练。先解决请求级数据和商品表示：Beauty无请求字段、仅日精度时间戳，未来K商品只能作为代理任务。见[具体方案](../research/onerec_listwise/PLAN_20261026.md)、[数据集选型](../research/onerec_listwise/DATASET_SELECTION_20261026.md)及[本地数据审计](../research/onerec_listwise/BEAUTY_DATA_AUDIT_20261026.json)。
- 数据选型尚未锁定：优先审核RecFlow真实请求，但其没有文本/图像内容，最终展示顺序和目标覆盖待核；MIND-small有文本和请求分组但顺序已打乱，作为集合任务备选。OneRec是按请求整体质量筛选，不要求每件都effective_view；“高质量请求观看列表”与“正反馈子列表”应分别统计。新数据未下载。
- 后续用户接受合理人工session划分，上一条为较早状态。现已CPU审计RecFlow四日408.88万行/80.21万请求，并核查KuaiRec全包及512用户观察记录。当前建议先用KuaiRec明确分段的消费列表探索，RecFlow后续验证真实请求；尚未正式锁定训练协议或启动GPU。RecFlow存在最终排序语义未明、零时长、镜像两日期同文件等问题；KuaiRec需去重、修复4条文本断行并使用适合中文的编码器。详见[最新数据审核与方案](../research/onerec_listwise/MANUAL_SESSION_REVIEW_20261026.md)。

## 08历史快照：追加正式测试已完成（2026-10-25，现已退役）

- 用户明确授权按原验证规则选出的三组best统一测试；原训练Goal的测试锁定记录保留为历史。本次独立读取baseline R15/T24、no_context R30/T54、axial R35/T60，不重训、不刷新SID、不按Top20/50或测试表现重选。
- 同一Beauty测试集22,363人，beam256、batch8，HR/NDCG@5/10/20/50全部完成，逐用户Top50候选已保存，原解码器首批整数等价、远端及本地CPU独立复算、29件本地下载字节校验均通过。三组模型和对应SID前后哈希未变。
- 测试HR@10/NDCG@10：baseline `0.074230/0.044581`，no_context `0.073738/0.043778`，axial `0.073648/0.043653`。axial仅HR@5略高于baseline，其余七项低于baseline；本轮不支持当前实现带来整体测试收益，仍限单seed、SID级证据。见[完整测试报告](../../experiments/08_session_codeword/test_best_20261025/REPORT.md)及[机器汇总](../../experiments/08_session_codeword/test_best_20261025/SUMMARY.json)。
- 三张单H100评估作业均已释放并删除，本轮保守占用`0.403790 GPU小时`，按展示单价估算约0.40点、结算未核实；与旧训练及10月24日验证补评账本分开。远端独立目录为`experiment1_session_codeword/topk_test_20261025_r1/`，本地为`outputs/stage08_session_codeword/topk_test_20261025_r1/`；新测试交付绑定位于报告目录的`DELIVERY_BINDING.json`。
- 以下10月24日训练/恢复段落属于较早交付快照，其中“测试锁定”不代表本次未获授权。旧基础包与补充包保留原字节，新测试不回写旧收据；本页此后的版本由本次测试交付单独绑定。

## 08历史快照：训练与恢复曾通过（2026-10-24，模型及包现已删除）

- 仅seed2024的baseline/no_context/axial，固定35R/3640次R更新、60次T更新，Full Mixer统一开启；从共享E0与pre-05推荐器开始，不接05/07最终模型。验证beam256，测试锁定。
- 三组正式35R/60T及完整验证已完成，CPU审计核45个配对阶段、24个全验证节点与六份实际保存状态；启动时真实14份输入、Linux107项、Beauty初始化与三臂H100 gate也已通过。本地11分卷共714,991,805字节、1995成员及14份输入逐字节通过，两个独立Git源码恢复、六份best/last真实模型及优化器/RNG/历史CPU前向均通过；未刷新SID、重训或重演GPU指标。完整交付以本轮外置`FINAL_DELIVERY_BINDING.json`与补充包配套验收，不能仅看单项PASS。
- 固定R35的HR10/NDCG10：baseline `0.099048/0.059208`、no_context `0.098511/0.059202`、axial `0.099674/0.059860`。axial的NDCG10相对两组约+1.10%/+1.11%，但HR5对baseline约−1.30%；同规则选best后baseline R15仍优于axial R35。因此只有单seed验证集小信号，尚无稳定质量收益证据。见[结果报告](../../experiments/08_session_codeword/REPORT.md)、[交接与恢复](../../experiments/08_session_codeword/HANDOFF.md)。
- 全部5份GPU分配已核实释放：生命周期口径7.755278 GPU小时，保守观测占用7.949226小时，短验收308.158秒；均在16/0.5小时上限内。按平台展示单价估算约7.95点，结算未核实。正式最终账本为`allocations/ledger_final_20261024_r1.json`；本轮没有追加seed、beam或测试。
- 独立执行根`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_session_codeword/goal_20261024_session_codeword_v1`；本地运行收据`experiments/08_session_codeword/execution_20261024/`，大文件与控制工具`outputs/stage08_session_codeword/`。实际双仓库版本、Job及验收进度见[执行状态](../../experiments/08_session_codeword/execution_20261024/EXECUTION_STATE.json)。
- 本地完整交付根为`outputs/stage08_session_codeword/delivery_20261024_final_r1/`：`package/`为基础分卷包，`restored/goal_20261024_session_codeword_v1/`为全新恢复，`verification/`为实际恢复/审计收据，最终补充ZIP绑定本次项目文档和Windows长路径恢复wrapper；补充说明覆盖基础包内早期直接调用命令。首次Windows Git长路径失败目录保留；过程仅改恢复子进程配置，不改冻结训练源码或系统Git设置。
- 原工作区HEAD和其他任务改动保留；执行提交由临时索引独立冻结。执行code为`152bea77d43284bb41f1887a8f43124be7e6db98`、engine为`a28faf6a24c667aa4f49f208fb93d294ad5aab4f`，不是当前主工作区HEAD。DiffGRM原Git为浅克隆，便携快照保留相同完整源码树，不声称补回缺失父历史。训练已结束，不重复提交旧Job。
- 异常已处理：首个baseline在共同`initial.pt.tmp`并发发布时失败，尚无已提交训练状态；实际确认common与H100完整Tensor哈希`d1a8d37a...cc09`及identity相同后，仅baseline02同源重试一次。失败占用计入最终账本。新campaign应先独立发布并核验共同状态后再并行，不能用同机Linux锁检查保证跨节点互斥；已有common不得重建。

## 07当前入口：用户改为交替联合训练，固定SID方案已停止

- 追加诊断（2026-10-21）已完成：固定两组best25及各自SID，在同一22,363用户验证集完成beam32/256四格；没有重训、改SID或再次评估测试集。B相对A的NDCG@10提升由3.27%缩至1.59%，HR@10由3.56%缩至1.07%，绝对差也缩小；Top10命中净增55→25，Top5净增41→40。搜索宽度影响收益幅度，但不能解释全部val/test差距，也未证明稳定泛化。两组beam32精确复现原指标，四格逐用户/权重/SID与57件本地证据校验通过，新增1张H100×1067秒=0.296389 GPU小时且已释放。此前7.428056训练账本保持原样，含本诊断07累计7.724444 GPU小时。见[本轮报告](../../experiments/07_item_local_attention/beam_validation_20261021/REPORT.md)、[验收汇总](../../experiments/07_item_local_attention/beam_validation_20261021/SUMMARY.json)。远端独立目录`experiment1_stage07/beam_validation_20261021_r1`；本地`outputs/stage07_item_local_joint/beam_validation_20261021_r1/verified/`。本次是已完成的单独验证请求，不要重复启动或改动旧Goal交付。
- 最终交付（2026-10-20）：修复后的r2两组均完成45轮及唯一正式测试，均选中第25轮；全部8个GPU作业已释放，累计`7.428056 GPU小时`（含旧失败/中止/短测，预算上限28小时）。A测试HR@10/NDCG@10为`0.08013236/0.04636612`，B为`0.08035595/0.04656316`；B相对A仅+0.279%/+0.425%，未达3%初筛。真实r3成对审计、335件本地备份字节校验、14份实际CPU权重/状态回读和曲线检查均通过，见[结果报告](../../experiments/07_item_local_attention/joint_alternating/REPORT.md)、[总验收](../../experiments/07_item_local_attention/joint_alternating/FINAL_ACCEPTANCE_20261020_r3.json)。本地备份为`outputs/stage07_item_local_joint/delivery_20261020_final_r3/`；13件原始输入已保存身份并在远端核对，但数据/cache字节不在包内。正式执行根`842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1`、引擎`1c879252f6395d6e64462511823c3929b9c42c00`保持冻结。CPU精度复演与四份权重全部SID码位精确匹配，原FP32/FP64差异保留，不声称追溯历史CUDA标志。没有修改训练权重、SID或测试以通过验收。接续以[本轮入口](../../experiments/07_item_local_attention/joint_alternating/EXECUTION_R2_STATE.md)为准，不重训、不重复测试；下方未完成/r1/固定SID描述均为旧历史。
- 最新直接指令是“我要的是联合训练”“按照之前码本交互跑联合训练的方式修改代码，交替优化”。它取代本页下方的07固定SID方案；见[范围变更](../../experiments/07_item_local_attention/joint_alternating/SCOPE_OVERRIDE_20261020.json)。本轮新增局部注意力仍在历史hard/ST SID embedding之后、item_mlp之前，属于推荐器侧。Tokenizer阶段冻结推荐器/local参数，通过DIGER-ST将推荐损失传到可训练码本；推荐器阶段冻结Tokenizer，训练推荐器/local。文本特征和OPQ预处理仍固定。
- 新方案复用05 `refined_joint`完整交替配方及锁定离线Refined-PSE v2/配套DiffGRM暖启动，关闭全局Mixer；对照组/商品内注意力组共用该起点。5轮推荐器暖身后循环1个Tokenizer阶段+3轮推荐器，最多45逻辑轮，每Tokenizer阶段96batch。它不是后来的共享起点四组短对照（24次Tokenizer更新+3轮恢复），也不接旧07的84轮或05联合训练最终权重。
- 当前固定SID短验收`nancy-stage07-ocn-repair-profile-20261020-r1`已按本次范围修改主动停止，为CANCELLED，10:58:04 Pod删除；此次不是平台低利用率回收。[终态](../../experiments/07_item_local_attention/execution_acceptance/ocn_repair_r1/PROFILE_TERMINAL_FOR_JOINT_SCOPE.json)、[事件](../../experiments/07_item_local_attention/execution_acceptance/ocn_repair_r1/PROFILE_EVENTS_STOP_FOR_JOINT_SCOPE.json)。新消耗364秒，连同旧11273秒累计3.2325 GPU小时，28小时上限还剩24.7675小时，[总账](../../experiments/07_item_local_attention/execution_acceptance/ocn_repair_r1/CUMULATIVE_AFTER_SCOPE_CHANGE.json)。不把中断profile称为GPU验收或研究结果通过。
- 联合r1实现历史：交替联合训练入口及真实joint边界恢复已实现，当时执行根`a6236b7ad7bdead84869e83221049268d3a123ab`、引擎`1c879252f6395d6e64462511823c3929b9c42c00`。本地及远程20项核心CPU检查通过，runner/wrapper另有CPU验收，见[本轮入口](../../experiments/07_item_local_attention/joint_alternating/README.md)。已验证阶段更新/冻结、推荐损失到码本的真实梯度、SID刷新及恢复。新配置[joint_alternating_seed2024.json](../../configs/07_item_local_attention/joint_alternating_seed2024.json)，13件输入取自05 `refined_joint`资产锁；新入口的两组真实CPU prepare均已[通过](../../experiments/07_item_local_attention/joint_alternating/REMOTE_CPU_PREPARATION.json)。最终正式运行使用上方r2版本842ff2a。
- 新联合短验收r1已真实FAILED，11:25:14 Pod删除，不能再按首次QUEUING快照等待。失败发生在构建模型之前：runner的`import genrec.models.DIFF_GRM.trainer as trainer_module`受到父包同名DIFF_GRM类遮蔽，需改为`importlib.import_module`并先在CPU真实生产导入链核验；[原始日志](../../experiments/07_item_local_attention/joint_alternating/SMOKE_R1_FAILURE_LOGS.json)。本次Scheduled11:23:32到Pod删除11:25:14共102秒，累计11739秒、3.260833 GPU小时，[更新账本](../../experiments/07_item_local_attention/joint_alternating/ALLOCATION_AFTER_SMOKE_R1.json)。初始0.5小时验收预算已用466秒，28小时总上限保留。旧r1执行副本与产物冻结保留；修复后独立r2准备尚未完成，未提交重试或正式两组。

## 07固定SID修复部署历史（被上方联合训练指令取代）

- 网络已恢复，CLI当前配置和认证检查通过，既有CPU桥接连接已刷新；无需重新登录或修改代理。新root部署CPU门已通过：[远程CPU验收](../../experiments/07_item_local_attention/execution_acceptance/ocn_repair_r1/REMOTE_CPU_PREPARATION.json)。单H100短验收`nancy-stage07-ocn-repair-profile-20261020-r1`已提交，最近状态为QUEUING，[提交](../../experiments/07_item_local_attention/execution_acceptance/ocn_repair_r1/PROFILE_SUBMISSION.json)、[初始状态](../../experiments/07_item_local_attention/execution_acceptance/ocn_repair_r1/PROFILE_STATUS_INITIAL.json)；正式两组未启动。继续时先查同一Job与持久`delivery/profile_attempt1/`，不重复提交。下方网络中断是已恢复的历史记录。
- 用户先选择修复，随后已明确同意“累计28 GPU小时，按此方案推进”。[预算及两组训练授权](../../experiments/07_item_local_attention/execution_acceptance/REPAIR_BUDGET_AUTHORIZATION_20261020.json)：旧11,273秒计入，先最多0.5小时H100验收/测速，通过梯度、恢复、共同起点、实测预算及合法平台策略检查后，自主运行修复后A/B各一次从头训练；不重复询问这项已获授权的范围。
- 本地DiffGRM引擎已提交`8a48a300daa6621d6c705966ff73fa837ef8f12e`：只在OCN teacher打分中禁用autocast缓存并继承原精度。原8项模型检查与新增4项真实CPU BF16回归通过，覆盖全部梯度/参数更新、数值兼容和实际完整保存恢复。见[本地验收](../../experiments/07_item_local_attention/execution_acceptance/repair_planning_20261020/LOCAL_REPAIR_ACCEPTANCE.json)。H100修复尚未验收；远程冻结1dc2a0c、旧84轮与历史证据均未改动，未提交新GPU作业。
- [修复后最小两组与预算方案](../../experiments/07_item_local_attention/REPAIR_PLAN.md)：A=修复后关闭注意力，B=同修复开启注意力，两组同seed同共有初始参数/随机状态从头训练；不接旧84轮，03仅背景。新campaign为`stage07_ocn_repair_seed2024_20261020_r1`，独立远程根`experiment1_stage07/goal_20261020_ocn_repair_r1/`，代码、配置及账本与旧轮次分开；当前仍非H100或完整训练验收通过。
- 新执行入口已冻结为根`2fc97b62693628f6507ad2ca60472f6063a84250`、引擎`8a48a30`；两份增量bundle及哈希已在本地准备，新远程根已独占创建。随后两次上传失败，刷新既有CPU桥接及平台账号检查均返回`ECONNRESET`，无凭据HTTPS对照同样连接重置；目前不能确认登录已失效或Notebook终态。尚未通过远程CPU准备、未申请新GPU。[冻结部署清单](../../experiments/07_item_local_attention/execution_acceptance/ocn_repair_r1/FROZEN_DEPLOYMENT.json)、[网络中断记录](../../experiments/07_item_local_attention/execution_acceptance/ocn_repair_r1/NETWORK_INTERRUPTION.json)。网络恢复后继续同一campaign同步与验收，不重问预算、不重置旧11,273秒，也不覆盖旧84轮。

## 07诊断历史：恢复核验发现OCN/BF16缺梯度

- CPU实读完整断点为84轮/8,736次更新，最佳验证75轮；85轮验证中断、测试未开始。原始49件状态/配置/曲线已在远程`delivery/recovery_attempt2/pre_resume/`逐件保全。恢复核验v2确认正式执行8b7b9af/1dc2a0c及输入/曲线，但Adam参数组90项仅50份状态，四层Decoder共40个注意力/MLP线性参数没有更新状态，检查没有通过。
- CPU合成对照在锁定原版39e94e6和交互版都复现了BF16＋OCN缺梯度；float32或禁用autocast缓存后梯度恢复。仅在teacher打分上下文禁用缓存的候选补丁已做四层CPU复验：两版缺梯度均从40项变为0、首次loss不变；实际冻结源码/配置未修改，GPU修复未验收。见[诊断、原始结果和未应用补丁](../../experiments/07_item_local_attention/execution_acceptance/ocn_gradient_diagnosis_20261020/README.md)。
- 不能断言未知历史runtime的03正式训练一定有同一问题。修复后直接从84轮继续会混合训练行为；原Goal又限制不自动追加基线、总计8 GPU小时。因此已异步询问用户选择修复后另定必要对照/预算，或明确接受旧行为下受限探索。答复前不申请GPU，Goal未完成；不要放宽优化器检查以通过交付。

## 07平台中断与原恢复准备（未提交r2）

- 第一正式Job `nancy-stage07-item-local-formal-20261020-0045` 已为 `CANCELLED`，04:07:54 Pod删除；不是等待命令超时。平台事件明确因利用率未满足自动回收条件而停止，当前Live规则是 `gpu < 40% for 3h`。见[终态](../../experiments/07_item_local_attention/execution_acceptance/FORMAL_CANCELLED_STATUS_20261020.json)、[事件](../../experiments/07_item_local_attention/execution_acceptance/FORMAL_CANCELLED_EVENTS_20261020.json)、[回收规则](../../experiments/07_item_local_attention/execution_acceptance/RECOVERY_RESOURCE_POLICY_20261020.json)。
- 历史GPU利用率36个五分钟采样平均约16.1%，见[平台原始指标](../../experiments/07_item_local_attention/execution_acceptance/FORMAL_CANCELLED_METRICS_20261020_V2.json)。现有CPU观察显示第85轮训练已写CSV、其验证中断；最后可恢复的完整轮次必须以CPU读取 `training_state/last.pt` 为准，不能仅凭CSV推断。
- 两次短测与第一正式Job按Scheduled到Pod删除累计11,273秒，即3.131388889 GPU小时；8小时总预算还剩17,527秒。见[接续前资源账本](../../experiments/07_item_local_attention/execution_acceptance/RECOVERY_PRIOR_ALLOCATION_AUDIT_20261020.json)。所有恢复尝试必须加入同一总账，不重新获得8小时额度。
- 当前同族作业清单没有活动Job，见[实时清单](../../experiments/07_item_local_attention/execution_acceptance/RECOVERY_JOB_INVENTORY_20261020.json)。先在既有CPU桥接核对并保全完整断点，再决定接续；不重训03基线、不重复已通过短测、不改冻结模型或配方。下方RUNNING均是第一正式Job较早的历史快照。

## 07执行历史：首次验证与第一正式Job（已被上方终态取代）

- 范围修正前的旧Goal是[07商品内局部注意力探索](../../experiments/07_item_local_attention/README.md)：当时只新增一组seed2024、复用03 PSE基线，SID与Tokenizer固定。此范围已被用户后续要求的两组交替联合训练及累计28小时授权取代；不作为当前执行指令。
- 执行代码：初版根仓库`64abf0f06eb4ba4d9759091f74d397c91061726b`；当前启动修复版`8b7b9afe17ef425efe9279524ecf7a1e97148853`，仅规范平台单进程rank环境，模型和训练配置未变。独立DiffGRM引擎始终为`1dc2a0c5ad01e2a2878b9de400b91f3a7e809f25`。历史runtime未知、配置为封存证据加锁定默认值重建，不称严格配对。
- 本地及远程16项CPU检查均通过；12件输入按03资产锁核验，数据计数105,668训练、22,363验证/测试、12,101商品。[本地CPU收据](../../experiments/07_item_local_attention/execution_acceptance/LOCAL_CPU_GATE_20261020.json)、[远程CPU收据](../../experiments/07_item_local_attention/execution_acceptance/REMOTE_CPU_PREPARATION_20261020.json)。
- 第一短测`nancy-stage07-item-local-smoke-20261020-0045`已FAILED，训练0步：平台注入WORLD_SIZE=1，旧init_device仅按变量存在误判DDP，被恢复检查拒绝。[终态](../../experiments/07_item_local_attention/execution_acceptance/SMOKE_TERMINAL_20261020.json)。子进程启动环境修复后，r2短测`nancy-stage07-item-local-smoke-20261020-0045-r2`已SUCCEEDED，32batch/2update、loss有限、新模块参数更新、checkpoint重载和验证样本beam测速通过；[原始结果](../../experiments/07_item_local_attention/execution_acceptance/GPU_SMOKE_RESULTS_20261020.json)。短测全程未评估测试数据，零命中不作为质量结论。
- 当前唯一正式Job为`nancy-stage07-item-local-formal-20261020-0045`，分布式训练空间、开发区-H100-cuda13.2版本-183核、1张H100，[已提交](../../experiments/07_item_local_attention/execution_acceptance/FORMAL_SUBMISSION_20261020.json)且首次验证期间[确认RUNNING](../../experiments/07_item_local_attention/execution_acceptance/FORMAL_STATUS_FIRST_VAL_20261020.json)。第20轮首次验证已完成：HR@10=0.059607387202074855、NDCG@10=0.040013885470594035、weighted_score=0.04393258621668883，完整验证耗时212.7286秒；[轻量快照核对](../../experiments/07_item_local_attention/execution_acceptance/first_validation_20261020/SNAPSHOT_CHECK.json)通过，实际配置无batch截断、验证beam32。它是早期val，不与03最终test直接判胜负，尚非最终选模/测试结果。继续时先查询此Job及持久产物，禁止重复提交；不要重新跑已通过短测。
- 独立远程根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261020_0045/`；`code/`、`engine/`为固定版本；正式目录`runs/stage07_item_local_seed2024_20261020_0045/`，短测在其`smoke/`，原始控制台及进程收据在`delivery/`。个人新路径已验证可写，历史主项目只读复用。
- 总预算为单H100合计8 GPU小时，包含smoke、初始化、训练和最终评估。[预算门](../../experiments/07_item_local_attention/execution_acceptance/FORMAL_BUDGET_GATE_20261020.json)：短测首batch约2.41秒冷启动，后段约16batch/s；按0.08秒/batch保守估原120轮完整任务约5.65小时，200轮上限约9.36小时，不能保证跑满。正式Job硬上限7.5小时，预留1200秒最终评估；两次短测按创建至终态保守合计485秒，加正式上限为7.6347 GPU小时。固定原选模/最多200轮/早停协议，不通过缩短协议偷换完成；到预算保留断点并明确未完成。完整last状态及恢复入口见本轮runner与`training_state/last.pt`。
- 本轮尚未完成有效正式训练和最终评估；Goal保持active。CPU桥接`env-zzl`为既有资源，不停止；CLI exec通道曾返回127，现已验证通过现有CLI的SSH通道执行，认证有效，不据此重建账号或Notebook。
- 两次短测的完整单Job事件已保留；Scheduled到Pod删除实际合计388秒、0.107778 GPU小时，见[短测账本](../../experiments/07_item_local_attention/execution_acceptance/SMOKE_RESOURCE_ACCOUNTING_20261020.json)，不含正式Job且不是费用结算。后续CPU打包/字节恢复/模型结果/资源核对工具已准备并提交`1cf2467`，真实最终包仍未产生，不将工具就绪称为交付通过；不更新当前GPU执行副本。本地900秒原生Job等待曾以观察超时结束，随后已重新确认同一Job RUNNING；观察超时不代表训练失败，不能据此重提训练。

## 当前入口：本轮全部验收通过，交付已提交aec2307

- **训练**：64次搜索＋12次Top-3复验＋10次新seed正式训练＋3次旧协议配对，89/89成功、队列worker为0。全部89项/740个元数据文件及178份best/last完整状态已分别[核对](../../experiments/06_continuous_baseline/execution_acceptance/completed_metadata_precision_v2_batch20/AUDIT.json)、[交叉审计](../../experiments/06_continuous_baseline/execution_acceptance/checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_22.json)。训练仍为`bbaa45b`/源码SHA256 `b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907`，外部supervisor为`37ef406`/578f，输入、配置和预算未变。
- **选模与最终结果**：SASRec锁定配置13，eSASRec锁定配置19；按5个新seed的验证均值、在测试前封存选择`sasrec_fullce`。全部89条曲线、验证选模、训练结束→封存→13项测试→最终测速的原始时序已[独立核对](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_SELECTION_AND_ORDER_AUDIT.json)。5seed测试HR@10/NDCG@10为SASRec `0.072280±0.001294 / 0.041515±0.000845`，eSASRec `0.064920±0.001759 / 0.039181±0.000795`，均为样本标准差。范围是12,101件真实商品全库、每组22,363条测试、不屏蔽历史商品的item-level评估，不能与03–05的SID-level直接排榜。[六项指标和原始seed](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/summary.json)。
- **最终产物核验**：13项逐样本指标重算、10份正式导出、CPU交付检查均通过；最终固定seed3031的12组/2,400次测速重算及10份权重/全部商品向量重载核验通过，见[CPU交付](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/CPU_DELIVERY_PASS.json)、[测速](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_BENCHMARK_AUDIT.json)、[导出](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_EXPORT_AUDIT.json)。时序及冻结源码证明自有训练/评估排空后逐个测速；没有连续进程追踪或kernel profile，不扩张为外部负载不存在或逐kernel精度已证明。
- **GPU已释放**：`nancy-stage06-20261016-8h100-supervisor-v3`于15:14:19 SUCCEEDED，15:14:40 Pod删除，平台剩余节点0；本Goal的8张H100已全部释放。[终态](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/SUPERVISOR_V3_TERMINAL_PLATFORM_20261016.json)、[事件](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/SUPERVISOR_V3_TERMINAL_EVENTS_20261016.json)。现存非Goal GPU Notebook及CPU桥接不受影响，不为交付重提GPU。
- **完整备份与本地恢复已核验**：[本地全量备份审计](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_BACKUP_LOCAL_AUDIT.json)PASS：9分卷/1,300文件、14,409,004,551解压字节，178份best/last摘要全部与独立payload ledger交叉核对；412份可用文件已安全提取到`outputs/stage06_continuous/final_delivery_20261016/verified/`，含10份导出、13份逐样本结果、锁定输入/27份源码及代表SASRec13/3031完整状态。独立副本`recoverable_primary/`的[原恢复消费者CPU检查](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/LOCAL_RECOVERY_TRANSACTION_AUDIT.json)也PASS：model、Adam、scheduler和CPU RNG均加载，CUDA RNG仅保留/校验字节、未执行，保留COMPLETE且没有启动训练。其余seed完整状态保留在已验证分卷中，并非全部展开到磁盘。
- **本地复用入口与剩余边界**：[本地产物索引](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/LOCAL_ARTIFACT_INDEX.json)已核对路径/大小，列出10份导出、13份逐样本结果、178份checkpoint分卷成员及SHA、首选SASRec13/3031与`recoverable_primary/`。本轮全部验收通过，交付已提交`aec2307`，未push；Goal工具状态以当前任务状态为准。五次Goal作业实际分配33.2203 GPU小时、平台runtime32.9347 GPU小时，按核实费率折算分别33.2203/32.9347点券。用户已直接回复“接受，以此完成Goal”，[费用确认收据](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FEE_ACCEPTANCE_20261016.json)接受以上计算口径作为费用交付，费用不再是阻塞。两者仍是计算金额，个人日账单含非Goal Notebook，实际Goal结算未核实，`actual_goal_settlement_verified=false`；见[资源/费用边界](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/FINAL_RESOURCE_AUDIT.json)。

最新[10.16周报](<feishu-link>)已生成并回读。给定白盒Decoder学习适配Encoder、列表生成仍是后续研究设想，不属于本Goal的已完成实验。

本轮[最终交付总验收](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_DELIVERY_STATUS.json)为`ALL_REQUIRED_EVIDENCE_PASS_WITH_USER_ACCEPTED_FEE_BOUNDARY`；[Git交付核查](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_GIT_DELIVERY_AUDIT.json)记录明确文件提交范围及交付提交`aec2307`。当前无需再执行本Goal；下一研究为尚未实施的固定推荐器编码对照，需另行确定范围。Goal工具状态以当前任务状态为准。

## 06运行历史快照：以下14:57及以前状态均已被上方当前入口取代

14:57最新核实：唯一新Job `nancy-stage06-20261016-8h100-supervisor-v3` RUNNING；64次搜索全部完成，SASRec Top-3为13、16、01，6次复验中2次完成、4次运行；eSASRec的6次复验全部完成，按三个seed的验证均值锁定配置19，5个正式seed中2次完成、3次运行。合计74项完成、7项运行，后续8项尚未展开。这74项/617个元数据文件及148份best/last完整状态已核对，见[元数据审计](../../experiments/06_continuous_baseline/execution_acceptance/completed_metadata_precision_v2_batch17/AUDIT.json)、[完整状态](../../experiments/06_continuous_baseline/execution_acceptance/checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_19.json)。两模型全32项搜索的Top-3排序、eSASRec三seed均值选择及正式3031–3035的配置/seed已[独立核对](../../experiments/06_continuous_baseline/execution_acceptance/completed_metadata_precision_v2_batch16/DEPENDENCY_AUDIT.json)。无失败，最终模型族选择和测试尚未进行，Goal未完成。训练worker仍为`bbaa45b`/b240、外部supervisor为`37ef406`/578f，数据缓存和89项有效预算不变；以下带时间段落保留为历史快照。

eSASRec正式配置为512维、1层、4头、dropout0.3、LR0.001、batch128、负样本1024，10,945,536个参数。搜索/复验同配置3个seed实际耗时15.19、14.40、25.69分钟，分别118、112、200轮；这是含逐轮完整验证与保存的训练耗时，不能当作最终测速或SASRec正式配置耗时。14:57实际7个worker运行，3个eSASRec正式seed与4个SASRec复验各占一卡。

达到300轮上限的搜索有eSASRec配置29（最佳255轮、154,200次更新）与SASRec配置28（最佳294轮、308,100次更新），均未进入Top-3。按原预算结束，不擅自追加轮数，也不声称所有配置都已充分收敛。费用侧已成功读取当天公共项目总账与个人项目聚合读数，但均不能直接归为本Goal逐Job费用；[读取权限](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/BILLING_ACCESS_20261016_01.json)、[项目返回结构](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/BILLING_DETAIL_SHAPE_20261016_01.json)、[个人聚合读数](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/PERSONAL_BILLING_AGGREGATE_20261016_01.json)分别保留范围，实际Goal结算仍未确认。

本轮恢复的[GPU/实际续训核验](../../experiments/06_continuous_baseline/execution_acceptance/scheduler_failure_20261016_01/gpu_gate_supervisor_v3/GPU_GATE_AUDIT.json)保留。新节点训练阶段34条资源采样中，CPU平均约7.99核、限流增量0，容器内存最高约8.29GB/配额1800GiB；8个当前worker最近最多10轮的保存耗时占阶段中位数合计约0.9%–5.8%。见[原始窗口](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/RESOURCE_SNAPSHOT_SUPERVISOR_V3_01.json)、[独立重算](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/RESOURCE_SNAPSHOT_SUPERVISOR_V3_01_AUDIT.json)。这不证明全部共享盘I/O或kernel效率已审计。

13:21恢复进展：新的唯一8H100 Job `nancy-stage06-20261016-8h100-supervisor-v3` 已RUNNING，节点qb-prod-gpu406，正在增强GPU gate，尚未把“已启动”记作“队列恢复通过”。38项/309个元数据文件及76份完整状态已核对，8份中断任务的16份best/last状态CPU通过，旧产物不变。外部supervisor提交`37ef406`、哈希`578fdcae...`；训练worker仍运行冻结`bbaa45b`/b240，同一数据、配置、seed与89项有效预算，测试锁定。先核实新Job与独立恢复pipeline，不再查旧FAILED作业判定当前状态。[CPU恢复与本地回读](../../experiments/06_continuous_baseline/execution_acceptance/scheduler_failure_20261016_01/RECOVERY_READBACK_AUDIT.json)、[新作业提交](../../experiments/06_continuous_baseline/execution_acceptance/scheduler_failure_20261016_01/PLATFORM_RECOVERY_SUBMISSION.json)。

13:10最新状态：`nancy-stage06-20261016-8h100-precision-v2`已FAILED，13:02:43平台删除Pod。38项完成、8项中断、18项未启动；本次是调度器把空command误作进程身份的一部分，继而误判活进程失败，清理超时覆盖原错误；不是已观察到再次梯度溢出。38项完整状态已CPU通过，拟保持`bbaa45b`/b240训练代码冻结，用单独版本化的外部supervisor恢复8项和剩余清单，先做Linux CPU验收，尚未重提GPU。见[故障与恢复边界](../../experiments/06_continuous_baseline/execution_acceptance/scheduler_failure_20261016_01/README.md)。下文RUNNING均为更早快照。

进度快照（12:57）：新Job `nancy-stage06-20261016-8h100-precision-v2` 在线核实仍RUNNING；34项搜索技术完成、8项运行、22项待启动（当前展开64项，后续25项尚未满足依赖）。其中32项、四批260个元数据文件已[独立核对](../../experiments/06_continuous_baseline/execution_acceptance/completed_metadata_precision_v2_batch04/AUDIT.json)；34项共68份best/last完整payload已在CPU严格加载，核对模型形状/PAD、优化器形状/步数/超参、scheduler与曲线、Python/NumPy/CPU RNG状态接受及CUDA字节态，见[完整审计收据](../../experiments/06_continuous_baseline/execution_acceptance/checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_05.json)。CUDA RNG未逐项在GPU复演，不混同此前GPU smoke的20→21步恢复。没有失败任务、Top-3或正式配置，test仍锁定。原28项不计入这34项，训练源码和缓存哈希未变。

原故障配置17的正式搜索已完整完成：相同配置哈希`be7fcdf0...`、seed2024，53轮/27,242次更新，最佳第3轮、随后patience50早停；逐轮末尾记录的最大梯度范数6.6774，完整best/last也已通过CPU审计。[完成收据](../../experiments/06_continuous_baseline/execution_acceptance/checkpoint_payload_precision_v2/CRITICAL_CONFIG17_COMPLETE.json)。这解决了原配置无法跑完的数值问题，不是全部89项已完成或推荐效果已胜出的证据。

资源核对：旧失败8卡Job与两次单卡诊断已按平台Scheduled→Pod删除核对，共约7.0314 GPU小时、按单价折算约7.0314点券，非已确认结算；见[旧尝试账本](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/OLD_ATTEMPTS_ALLOCATION_AUDIT.json)。新Job当前采样无CPU限流，内存/显存未逼近配额；已保存[资源窗口](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/RESOURCE_SNAPSHOT_V2_01.json)、[平台CPU/磁盘/网络原始采样](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/PLATFORM_METRICS_V2_WINDOW01.json)和[阶段耗时](../../experiments/06_continuous_baseline/execution_acceptance/resource_accounting_20261016/PHASE_TIMINGS_V2_01.json)。后者7项可用worker最近最多10轮中，保存阶段中位耗时占各阶段中位数合计约0.8%–3.5%；不据此声称完整共享盘I/O或孤立kernel效率已审计。当前CLI无按Job结算入口，网页独立会话未登录；费用结算待最终核实，不用公共项目余额差替代。

最新作业更新（11:54）：修复提交 `bbaa45b515a15912a04665721e0a5c43d64dc9e9` 的新Job `nancy-stage06-20261016-8h100-precision-v2` 已RUNNING，节点qb-prod-gpu001、一个8H100实例。服务器36项CPU gate、增强GPU gate和初始独占测速通过；本地独立核对4项训练梯度检查、35轮/17,990次更新回归、10任务8卡补位和2,400次原始测速均通过。原清单89项已进入动态队列，首个完成任务SASRec配置01的完整best/last、优化器、调度器及CPU/CUDA RNG独立审计通过（64轮、最佳14轮）；这不代表89项完成，实时数量仍须在线核查。继续先核实这个新Job，不重提、不误查旧FAILED作业。源码仍为b240哈希；旧结果不采用，test仍锁定。见[提交收据](../../experiments/06_continuous_baseline/execution_acceptance/PLATFORM_SUBMISSION_20261016_PRECISION_V2.json)、[GPU独立核对](../../experiments/06_continuous_baseline/execution_acceptance/gpu_precision_v2_20261016_01/GPU_GATE_AUDIT.json)、[测速重算](../../experiments/06_continuous_baseline/execution_acceptance/gpu_precision_v2_20261016_01/INITIAL_BENCHMARK_AUDIT.json)。下文11:38及更早记录均为历史快照。

最新更新（11:38）：CUDA FP32 Math注意力候选修复已通过3批梯度对照、故障轮次514步重放、35轮从头压力检查及本地73项CPU gate。两个单卡诊断均已SUCCEEDED并释放。新版本源码SHA256 `b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907`；接着在独立 `goal_20261016_precision_v2/code/` 同步验收，GPU gate将再检查实际提交源码。原28项保留为旧版本记录，不进入新版本选模；按原清单重新完成89项，旧作业及诊断消耗另列。尚未重新提交8卡，test仍锁定。见[修复及边界](../../experiments/06_continuous_baseline/NUMERICAL_FAILURE.md)。

运行更新（2026-10-16 11:25）：06 Job `nancy-stage06-20261016-8h100-01`因eSASRec搜索配置17的数值异常FAILED，8卡已释放。队列已停止：28项技术完成、1项失败、7项中断、28项待启动；28项有效性待审，尚无完整89项、Top-3或最终测试结果。独立1卡诊断在另一张H100复现原BF16路径异常，Math路径未在同段重放中失败；仍需定位及验证统一修复。诊断Job也已结束并释放。禁止原样盲目重提或将其算作基础设施故障。见[故障与证据](../../experiments/06_continuous_baseline/NUMERICAL_FAILURE.md)。下文“排队”“运行中”均为此前快照。

## 1. 先知道现在做到哪里

- **已经完成**：01–04历史实验、05原有联合/冻结实验及后续诊断和共享起点Mixer短对照；06双协议数据准备、模型CPU/GPU验收、89项训练、冻结选模、最终评估、逐样本重算、导出和最终测速。详细证据见顶部及第9节。
- **目前最主要的观察**：推荐梯度能回传并改变正式 SID；但尚未得到“码本交互带来明确、稳定推荐收益”的证据。最新 Mixer 对照的 HR 和 NDCG 变化方向不一致。
- **06当前状态**：89/89训练成功，验证预选SASRec＋FullCE；13项测试、10份正式导出、最终测速、完整本地备份与代表状态CPU恢复检查均通过，8张H100已释放。本地产物索引和全部验收已通过，用户接受保留未核实结算说明的费用计算交付，交付提交`aec2307`。
- **当前下一步**：当前无需再执行本Goal，不重复下载、训练或重提GPU。下一研究为尚未实施的固定推荐器编码对照，需要另行确定实验范围；给定白盒Decoder学习适配Encoder仍属研究设想。

证据优先级：实际配置/代码、结果 JSON/CSV 和完成收据 > 有日期的分析文档 > 聊天中的推测。旧 README 的“下一步”“尚未运行”只描述当时状态，不覆盖后续完成收据。

本摘要保留历史结果和06各时点记录。当前训练和最终GPU工作已结束，终态与删除Pod均有收据；以下带日期的“排队”“运行中”“尚未测试”只描述当时状态，不能当作恢复或重提指令。

## 2. 研究主线与已完成阶段

核心问题：商品 Semantic ID 的构造、位置容量和与推荐器的耦合方式，如何影响生成难度与推荐效果？目前研究对象是**单个 item 生成**，不是列表生成。

| 阶段 | 为什么做 / 实际做了什么 | 已有观察及边界 | 详细入口 |
|---|---|---|---|
| 01：AR 基线 | RQ-KMeans、RQ-VAE，3×256 语义码 + collision suffix，各 3 seeds | 当前三 seeds 的 RQ-KMeans 推荐指标均更高；更好的重构不必然带来更好的推荐。逐位置 NLL 不等同于语义信息量 | [01 报告](../../experiments/01_ar_baseline_3x256/README.md) |
| 02：AR 容量对照 | 新增 PSE、RQK 128×256×512 和 512×256×128；复用 01 的两条 seed2024 基线 | 容量布局改变位置预测难度，但两个非均匀 RQK 未超过均匀布局；组合容量相同不代表参数量相同。新增运行只有单 seed | [02 报告](../../experiments/02_ar_capacity_ablation/README.md) |
| 03：DiffGRM 基线 | PSE、RQ-KMeans、RQ-VAE，4×256，seed2024 | PSE 的 SID-level 指标最高；不能据此宣布严格商品级胜出，也不能直接与 AR 排性能榜 | [03 报告](../../experiments/03_diffgrm_tokenizer_comparison/README.md) |
| 04：离线码本优化 | Original PSE → Refined（额外梯度微调、无交互）/ Cross（相同训练加 self-attention）；再冻结 SID、从头训练各自 DiffGRM | Refined 接近 Original，Cross 未超过 Refined。目的是隔离“额外训练码本”与“交互”的作用；Original PSE 本来就经过 OPQ/PQ 拟合，并非没训练 | [04 报告](../../experiments/04_crosspse_offline/README.md) |
| 05：联合训练与归因 | 交替训练 Tokenizer/推荐器，加入直接推荐反馈；随后补冻结对照、诊断和更严格的共享起点实验 | 训练链路已跑通；参数/SID 的改变不自动等于推荐收益，历史组间比较仍需考虑不同初始化、恢复预算和选模 | [05 报告](../../experiments/05_crosspse_joint/README.md)及下文 |
| 06：连续向量强基线 | 完整/旧前缀两套商品级协议；SASRec＋FullCE/eSASRec的64搜索、12复验、10正式和3配对，共89次有效训练 | 验证预选SASRec，5seed商品级测试HR@10/NDCG@10为0.072280/0.041515；13项指标重算、10份导出与最终测速通过。GPU已释放，完整本地备份与代表状态CPU恢复检查通过；用户已接受费用计算交付，全部验收通过，交付提交aec2307 | [06报告](../../experiments/06_continuous_baseline/README.md)、[最终结果](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/summary.json) |

历史 01–04 共 14 次独立正式运行、17 条阶段结果；复用基线不算新 seed。01–02 为严格 item-level；03–05 主要是 SID-level。除 01 外，这里的主要实验都是 seed2024，不宣称稳定提升或显著性。

04 的测试 HR@10 / NDCG@10：Original `0.074901 / 0.043919`，Refined `0.075705 / 0.044091`，Cross `0.073604 / 0.043196`。来源：[04 results.csv](../../experiments/04_crosspse_offline/results.csv)。其动机是检查“离线目标与推荐目标可能不一致”，不是已经证明该不一致就是失败原因。

## 3. 05 阶段：不要混淆两套“四组”

### 3.1 早期长期交替训练：已完成，但不是共享起点的 Mixer 消融

Cross 与 Refined 分别从各自 04 阶段的 Tokenizer 和推荐器起点开始。Joint 先训练 5 个 Decoder epoch，再循环 1 个 Tokenizer phase + 3 个 Decoder epoch；Frozen 只继续训练推荐器。每 phase 6 次 Tokenizer optimizer update，Cross/Refined 的初始化不同。

以下是**验证集选模后的测试集 SID-level**结果，不与后文固定第 3 轮的验证分数混比：

| 组 | Test HR@10 | Test NDCG@10 | 最佳总 epoch | 最佳点 Decoder epoch / Tokenizer phase |
|---|---:|---:|---:|---:|
| Frozen Cross | 0.074230 | 0.044670 | 15 | 15 / 0 |
| Joint Cross v2 | 0.077315 | 0.045987 | 5 | 5 / 0 |
| Frozen Refined | 0.075661 | 0.044648 | 15 | 15 / 0 |
| Joint Refined | 0.077270 | 0.044354 | 19 | 15 / 4 |

关键解释：Cross v2 的最佳点**尚未更新 Tokenizer**，其测试优势不能归因于联合更新；Refined 的 HR@10 略升而 NDCG@10 等下降。Cross 与 Refined 起点不同，不能直接把差异解释为 Mixer 的贡献。

更早的 SIA、DIGER ST v1 也已运行并保留，不是未完成任务：v1 最佳点经历过 2 个 Tokenizer phase，但“最佳点在更新后”本身不是因果证明。参数和结果见 [results.csv](../../experiments/05_crosspse_joint/results.csv)、[HISTORY.md](../../experiments/05_crosspse_joint/HISTORY.md)。代码中的 DIGER 命名不代表已完成某篇论文全部设置的严格复现。

### 3.2 后续机制检查：已完成

这些是独立的新短实验/诊断，不是把历史权重精确续训，也不是完整多轮交替训练。

| 顺序 | 实验意图与设置 | 核对后的结果 | 证据入口 |
|---|---|---|---|
| SID 一致性 | 固定快照，对比显式 Gumbel 采样与确定性最近邻 SID，并核对最佳点累计改码 | 显式带噪编码差异很大，最佳点正式改码很少；但旧 guided 打分可能改变 train/eval 模式，不能证明历史每批实际加噪 | [诊断](../../experiments/05_crosspse_joint/diagnostics/sid_consistency_20261011/README.md) |
| 噪声 A/B/C | 共同 Cross v2 epoch5 起点；A 冻结，B 无噪声更新 6 次，C 按原调度实际施加噪声更新 6 次；各恢复 3 轮 | 固定 decoder_3 的 HR@10：A/B `0.065957`，C `0.066136`；NDCG@10：`0.047498 / 0.047521 / 0.047666`。没有支持“去噪已解决问题”；C 也不是旧执行的精确重放 | [协议](../../experiments/05_crosspse_joint/pilots/noise_update/README.md)、[完成汇总](../../outputs/pilots/stage05_noise/noise_pilot_20261012_1720/summary.json) |
| 推荐反馈三组 | 同起点、同旧 Mixer、无噪声；Frozen / 仅辅助目标 Cross-Recon / 额外推荐反馈 Cross-Rec；更新组各 24 次，再恢复 3 轮 | decoder_3 的 HR@10：`0.065957 / 0.066136 / 0.065823`；NDCG@10：`0.047498 / 0.047460 / 0.047432`。反馈改变了编码，但没有额外净收益 | [协议](../../experiments/05_crosspse_joint/pilots/feedback_update/README.md)、[原汇总](../../outputs/pilots_delivery/feedback_analysis_20261013/summary.json) |
| 逐样本重放 | 复用上面三组固定 decoder_3，完整验证集每组 22,363 样本，不新增训练 | 原总体指标精确复现；Rec 相对 Recon 新增 24、丢失 31 次 HR@10 SID 命中，净 −7。变化目标子集的局部 +2 不能解释成主动重编码成功 | [结果及案例](../../experiments/05_crosspse_joint/pilots/feedback_update/paired_analysis/RESULTS.md) |
| 硬 SID 更新方向 | 固定共同 initial 的完整推荐器；比较三组 post_tokenizer，四目标全 mask，train/val 各一次前向 | Rec−Recon 的 Val NLL 约 −0.000627，很小且主要涉及标签/上下文变化；不是 CPD Top-K 推荐改善，也不独立证明表征更好 | [结果](../../experiments/05_crosspse_joint/pilots/feedback_update/hard_sid_compatibility/RESULTS.md) |

上述新短实验仅使用验证集、SID-level；不能拿来对照早期 beam256 的正式测试成绩。固定快照诊断、非零梯度、SID 改变数、Attention 权重均不能独立构成因果收益证据。

### 3.3 最新四组共享起点 Mixer 对照：已完整结束

**目的**：排除旧 Cross/Refined 起点不同的影响，检查同样推荐反馈和恢复预算下，Mixer 是否增加收益。

- 运行：`factorial_20261013_1816_formal`；执行代码 `78d446d365a589cfb476424e732e673795e1e79b`；单 H100，四组合计业务用时约 90 分钟。
- 起点：从反馈实验的共同 initial 提取有效码本 E₀作为新基础码本；Mixer attention 输出投影置零、残差系数固定 1，保证启用/禁用时初始有效码本和正式 SID 相同。这是暖启动重参数化，不是随机从头训练，不是旧 Mixer 权重续训。
- Joint 各更新 24 次 Tokenizer（384 batch，累积16）；推荐反馈权重1、SIA为0、无 Gumbel；基础码本 LR `1e-5`、Mixer LR `1e-4`；辅助均衡/锚定权重各 `0.01`。
- 四组随后冻结 Tokenizer，各恢复 3 个完整 Decoder epoch（每轮1652 batch、104次更新，共312次）；LR `1e-4`，batch64，label smoothing `0.2`，Tokenizer FP32、Decoder BF16。
- **预先固定第3轮验证集 SID-level**，不选各组最好轮数、不用 test；两个 Frozen 的完整最终模型哈希、全部验证分数和数据顺序一致。

| 组 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | 正式 SID 改变商品数 / 12,101 |
|---|---:|---:|---:|---:|---:|
| no_mixer_frozen | 0.061977 | 0.065957 | 0.046157 | 0.047498 | 0 |
| no_mixer_joint | 0.062380 | 0.066136 | 0.046345 | 0.047611 | 16 |
| mixer_frozen | 0.061977 | 0.065957 | 0.046157 | 0.047498 | 0 |
| mixer_joint | 0.062112 | 0.065868 | 0.046361 | 0.047628 | 284 |

Mixer 的增量按 `(mixer_joint−mixer_frozen)−(no_mixer_joint−no_mixer_frozen)` 计算：HR@10 `−0.00026830`，NDCG@10 `+0.00001670`；weighted score 增量 `−0.00004030`。HR 的总体命中数差为 −6，**不是本轮已做逐样本配对检验**。

初步结论：无 Mixer 更新有小幅正向信号；加 Mixer 改码更多，但未形成一致额外收益。不能把“284 对16件商品改码”写成更有效，也不能据此确认“全局交互不稳定”是唯一原因。当前只有一个 seed、24 次更新和3轮恢复；Mixer 参数量、学习率和优化路径也不同，`full` attention 同时包含层内/层间交互。

本轮已完成代码实施、服务器测试、四组 GPU smoke 和四组正式短实验；**不是待启动，也不是完整长期四组重训已完成**。

证据：[实验协议](../../experiments/05_crosspse_joint/pilots/mixer_factorial/README.md)、[原始 summary](../../archive/server_sync/mixer_factorial_20261014_results/summary.json)、[COMPLETE](../../archive/server_sync/mixer_factorial_20261014_results/COMPLETE.json)、[FORMAL_PASS](../../archive/server_sync/mixer_factorial_20261014_results/FORMAL_PASS.json)、[共同初始化](../../archive/server_sync/mixer_factorial_20261014_results/common_initialization.json)。这些结果文件位于被忽略的 `archive/`，不是因为代码已提交就自动进入 Git。

<!-- stage05-long-factorial-handoff-begin -->
<!-- stage05-long-delivery-verified -->

最终验收：[FINAL_DELIVERY_AUDIT.json](../../experiments/05_crosspse_joint/long_mixer_factorial/FINAL_DELIVERY_AUDIT.json)。
### 3.4 共享起点带噪声长期四组（2026-10-20）

`goal_20261020_long_mixer_v1`四组35R训练与四次正式测试已完成，控制检查通过。Joint为5R＋10×(96 batch/6T更新＋3R)，累计60T与1920次真实加噪调用；Frozen35R/0T。四组从Aug-28完整推荐器与同一学习后有效E0开始，新Full Mixer输出投影0、残差1。两Frozen被固定在恒等函数，35轮轨迹、模型Tensor、SID和测试指标完全一致；这符合预期控制。

预定固定R35验证差中之差：HR10−0.0002235836、NDCG10+0.0001411057、weighted+0.0000681678。验证选模后的独立测试中，MJ−NJ为HR10−0.0004471672、NDCG10+0.0005716081、weighted+0.0003678530，选择的R/T预算不同，不能取代主比较。Full使正式SID移动更多，尚无一致稳定推荐收益证据；只约束本seed/E0及当前Full参数化，未分离码本内部与码本之间作用。

全部7次H100分配均核对Pod释放，实际35549秒=9.874722222 GPU小时，验收636秒；最后评估Pod15:34:02删除，不再申请GPU。远程完整备份的字节和八份实际CPU模型恢复已通过；本地同一归档的字节、Git恢复和跨副本绑定也已通过。

证据：[长期入口](../../experiments/05_crosspse_joint/long_mixer_factorial/README.md)、[报告](../../experiments/05_crosspse_joint/long_mixer_factorial/REPORT.md)、[验收矩阵](../../experiments/05_crosspse_joint/long_mixer_factorial/ACCEPTANCE_MATRIX.md)、[实际资源](../../experiments/05_crosspse_joint/long_mixer_factorial/RESOURCE_RELEASE_FINAL.json)。训练根提交6968269、引擎1c87925，分析4b40c90；不能把当前07工作分支当作本实验执行版本。保持CPU桥接env-zzl供其他工作复用，不停止非Goal资源。
<!-- stage05-long-factorial-handoff-end -->

## 4. 05代码机制与06边界

| 模块 | 当前职责 |
|---|---|
| [cross_pse.py](../../external/DiffGRM/genrec/models/DIFF_GRM/cross_pse.py) | `CrossPSECodebookMixer`：基础码本、身份标签、self-attention、残差；Refined 关闭 Mixer；最近邻量化和重构等辅助目标 |
| [joint_cross_pse.py](../../external/DiffGRM/genrec/models/DIFF_GRM/joint_cross_pse.py) | `JointCrossPSEAlignment`：基于距离的软分配、可选 Gumbel、严格 one-hot 前向/软反向的 ST、自身正式硬码刷新 |
| [model.py](../../external/DiffGRM/genrec/models/DIFF_GRM/model.py) | `set_joint_phase` 冻结一侧参数；ST 分配映射到推荐器独立 SID embedding，让推荐梯度到达量化模块 |
| [最新四组入口](../../scripts/train/run_diger_mixer_factorial.py) / [协议检查](../../scripts/checks/diger_mixer_factorial_protocol.py) | E₀重基、共同完整起点、固定预算、梯度与参数隔离、Frozen 负对照、输出和输入哈希保护 |
| [噪声入口](../../scripts/train/run_diger_noise_pilot.py) / [反馈入口](../../scripts/train/run_diger_feedback_pilot.py) | 一次 Tokenizer 干预后冻结恢复；恢复每批模式并记录真实采样/加噪调用；保存可追溯快照 |

特别注意：**05量化码本的64维码字不等于其推荐器的256维SID embedding。** 上表的05 Decoder仍预测离散SID，Tokenizer改变经由ST分配、正式SID和推荐器适配生效。06另行实现连续查询向量→共享商品表点积→全库检索基线，已完成正式实验；它没有实现“给定任意Decoder自动学习适配Encoder”。

## 5. 连续基线与尚待研究的内容

| 内容 | 当前状态 / 缺什么 |
|---|---|
| 直接生成连续 embedding | 06连续查询向量→全库商品检索基线已完成89项训练和最终评估；SASRec＋FullCE按验证结果获选。完整本地备份与代表状态CPU恢复检查均通过；用户已接受费用计算交付，全部验收通过，交付提交aec2307，见顶部 |
| 给定白盒 Decoder 学适配 Encoder | 研究设想，不是已经验证可适配任意 Decoder 的算法；需要明确冻结哪些参数、编码约束及梯度路径 |
| 换成 RankMixer 或其他交互模块 | 讨论过；现有实现是 self-attention，不能称为已完成 RankMixer 实验 |
| 现有三组统一恢复3→6轮 | 硬 SID 诊断曾建议；本次未找到该方案的完成结果，不当作已执行 |
| 最新四组多 seed / 长期交替训练 / 严格 item-level 评估 | 未完成；当前是单 seed 的验证集短对照，不能宣称稳定优劣或真实商品级收益 |
| 列表生成 | 早期动机，尚未开展；当前单 item 结果不构成列表建模证据 |
| 迁移后01–05全部正式重训/独立重评 | 工程迁移和有限 smoke 不等于全部历史训练已复现；不为补齐表述而重跑所有实验 |

学长关于“全局更新可能不稳定”的话是对现象的解释假设，不是现有实验已证明的机制。最近提出的两个新方向来自聊天讨论，不能引用为论文或运行证据。

路线取舍为先建立06连续向量强基线，再讨论Decoder适配编码。前者已完成固定89项训练、最终评估及完整本地备份/代表状态CPU恢复检查，费用计算交付已获用户接受，全部验收通过并提交aec2307；后者仍是独立研究设想，不在本Goal执行，不把“必须涨点”设为完成条件。

## 6. 环境、版本与资产：接手时必须核对

### 代码版本（06后续Goal实施前的历史快照）

本小节的分支、HEAD和“尚未提交/同步”保留为首次交接时记录；06当前分支及实际执行提交以顶部和第9节的新收据为准。

- 主仓库分支：`codex/project-organization`；HEAD：`78d446d365a589cfb476424e732e673795e1e79b`。
- 独立引擎 `external/DiffGRM`：`39e94e6153c97e59877ba02d4ef1d9a202ed5384`。
- 独立引擎 `external/genrec`：`b6c8b41d1d20a812590dae11408b69b1ed2a8313`。
- 核对开始时三仓库无已跟踪文件改动；根仓库有两个无关未跟踪目录 `draft_bbd47eb9_folder/`、`draft_e98ce1f0_folder/`，未读取、移动或提交。
- 这份摘要及索引是本次新文档改动，**尚未提交/同步**。之后 HEAD 改变时须更新版本，不能把此处基线当成未来实际运行版本。

根仓库不自动记录 `external/` 两个独立 Git 的修改，也没有自动双向同步。05最新四组执行副本在 `78d446d`；其 provenance 中另有 HDD 主项目 `ae52c08`，分别指执行代码和只读资产/引擎所在项目，**不是整个服务器主项目都已升级到78d446d的证明**。

### 资产位置（按现有合同/收据，非本次在线检查）

| 位置 | 用途 / 注意事项 |
|---|---|
| `C:/Users/nancy/Documents/experiment1` | 本地审查代码；`data/`仍无正式文件，但06两份锁定输入及生成数据已保存在`outputs/stage06_continuous/data_20261015_a`和`data_20261015_b`。历史共同初始化权重仍不在本地，不据此在本机正式训练 |
| `/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized` | 服务器主项目、历史数据/Tokenizer/初始化。旧项目已退役清理的记录见 `INSPIRE.md`，不要重建旧路径 |
| `/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp` | 共享项目外独立环境；最近正式收据为 Python3.10.20、Torch2.6.0+cu118、NumPy2.2.6，不改动学长其他环境 |
| `/inspire/ssd/project/project-public/<inspire-account>/nancy/experiment1_analysis` | 配对分析/硬 SID 诊断的稀疏代码和轻量输出；不是全项目备份 |
| `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training` | 最新四组的独立 `code/`、`delivery/`、`outputs/`；正式权重在 `outputs/pilots/stage05_mixer_factorial/factorial_20261013_1816_formal` |

完整权重/优化器不包含在本地最新轻量收据里；恢复训练前须到实际存储核验。独立训练副本还借用历史 Git 对象，不能据此删除其源对象库。存储可用量与配额要在下一次运行前重新实测，不能用 Memory 利用率或旧截图代替磁盘配额。

操作规范见 [INSPIRE.md](../../INSPIRE.md)、[工作流](EXPERIMENT_WORKFLOW.md)、[目录地图](PROJECT_MAP.md)。旧报告/复现配方/输入锁不重写；新实验独立 run_id 和输出；本地审查→测试→确认并提交各仓版本→同步核验→smoke→约定预算正式运行→结果核对归档。

## 7. 首次交接核对范围与已知缺口（01–05历史记录）

- 阅读了01–05报告、05主要结果CSV、最新协议与训练/梯度路径，并直接读取噪声/反馈/最新四组 JSON；重算了第3轮指标与上述四组增量。
- 三个训练汇总均与各自 `COMPLETE.json` 的 SHA256 一致；配对重放和硬 SID 诊断的汇总也与完成收据一致。最新 summary 为 `1491f38402aca6d73f98728f86b34afcc1b264f4e20f17df934003f907dacedc`。
- 最新运行 provenance 的37项源码/配置哈希：9项本地字节完全相同，28项在 CRLF→LF 后相同，无其他内容差异。**不把换行规范化后的相同写成全部逐字节相同。**
- 本地 `tests/test_diger_mixer_factorial.py`：39项中30项通过、9项跳过（8项缺Torch/NumPy，1项Windows符号链接权限）；不冒称本次通过真实张量/GPU验收。历史服务器39项无跳过及GPU通过记录另见 [执行收据](../../archive/server_sync/mixer_factorial_20261013_1816/EXECUTION.md)。
- `tests/run_repro_checks.py` 本次**未完整通过**：默认临时目录有权限问题，项目内临时目录又被测试的项目外隔离要求拒绝；改用另一个可写工作区后，临时重算5个实验包、14项CSV数值/字节比较通过，但只读检查子进程120秒超时，runner未继续完成其余测试。这不是已定位的训练算法错误；本次未修改测试、放宽守卫或把历史通过记录当成本次全绿。相关环境/耗时限制留待需要完整回归时单独检查。
- 06更新前的交接摘要31个本地链接均已核验；最新四组表格的四项推荐指标（6位小数）和正式改码数量已逐项对照原JSON，Git差异无空白错误。旧阶段结果表分别有6/5/3/3/6行，包含已注明的复用基线和历史版本，不把行数当作独立seed数。06新增链接另在本轮交付时核查。
- 当前主README、05报告和部分旧指南仍包含较早时点的“未运行”“旧副本未删”等措辞；最新短对照状态以本摘要引用的完成收据为准，不据此改写历史说明或认为必须重新迁移。
- `outputs/`、`archive/`、`data/`、`external/` 默认被根Git忽略；本摘要中的本地相对链接在只克隆主仓库后可能缺文件，需要对应证据/引擎/资产副本。两处旧迁移验证目录遇到Windows读取权限限制，本次不宣称完成整个文件系统或迁移资产的全量审计。

## 8. 06第一阶段：本地数据与CPU验收完成（2026-10-15历史快照）

- 输入：原Beauty的22,363用户、12,101件商品、198,502交互；从本地迁移包仅提取`all_item_seqs.json`、`id_mapping.json`，哈希与既有资产锁及最新05收据一致，未重新编号或下载数据。
- `full_train_prefix_v1`：至少1件历史，覆盖最后两个留出位置之前的全部合法目标，每次输入最近50件历史；131,413个训练目标。
- `legacy_prefix_v1`：保留原代码至少2件历史、目标位置最多50的规则；105,668个训练目标；通过直接AST执行现有`_sliding_train`作为独立参照核查。
- 两套验证/测试分别为每用户一条，各22,363条；两套协议的留出文件逐字节相同。重复商品合法，泄漏按交互位置判断。
- 37项标准库CPU测试通过、零跳过。两次从备份独立生成，各全量核验326,533条JSONL记录；11项输入/配置/候选/统计/划分内容文件哈希全部一致。共同manifest哈希为`6ecbe2768c31a6c212146ac8d3f01a45bc72cfacd0794166fa4730d121a0fdfa`。
- 实际目录：`outputs/stage06_continuous/data_20261015_a/`、`data_20261015_b/`，每份带原始输入、六个划分、manifest及运行来源。轻量证据在[06验收入口](../../experiments/06_continuous_baseline/ACCEPTANCE.md)，数据本体和新代码均尚未提交/同步，不能只靠主仓库HEAD描述本次执行代码。
- 本地Python3.14.6，仅标准库；没有Torch/GPU依赖、模型训练、测试集推荐评估或调参。远端数据和环境未在线核验。
- 下一步：实施模型及训练/评估接口，先做CPU验证；GPU smoke、调参与正式训练尚未开始。旧05两套“四组”状态不变。

## 9. 06后续Goal：训练、最终评估与资源释放已完成（2026-10-16）

- 实际训练为`bbaa45b515a15912a04665721e0a5c43d64dc9e9`，源码SHA256 `b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907`；恢复supervisor为单独提交`37ef4065546c3fb13397a4d9cdc0d3f29d1462df`，未修改冻结训练代码。原数值故障版本的28项不进入最终89项；修复后38项保留、8项从完整状态恢复，其余按原清单完成。
- 固定89项全部成功：64搜索、12确认、10正式、3旧协议配对。SASRec配置13为256维/1层/1头、dropout0.3、LR1e-4、batch256、3,900,672参数；eSASRec配置19为512维/1层/4头、dropout0.3、LR1e-3、batch128、负样本1024、10,945,536参数。正式seed均为3031–3035；旧协议只用SASRec的3031–3033。
- 5个正式seed验证NDCG@10均值为SASRec 0.05552036、eSASRec 0.05151127，因此在最终测试前选定SASRec。两模型6项测试原始分数、均值/样本标准差，旧协议配对差值、训练目标数/更新量/暴露量和耗时均在[summary](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/summary.json)；不按测试最好seed重新选择。两项达到300轮上限的搜索不在Top-3，不追加预算，也不宣称全部候选充分收敛。
- [CPU交付](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/CPU_DELIVERY_PASS.json)、[验证选择与时序](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_SELECTION_AND_ORDER_AUDIT.json)、[13项逐样本审计索引](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/evaluation_index.json)、[10份导出重载](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_EXPORT_AUDIT.json)及[最终测速](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_BENCHMARK_AUDIT.json)均通过。最终测速固定seed3031和H100 GPU0，batch1/128，各预热50、重复200次；记录query、全库Top20与含H2D/D2H的端到端，未含CPU历史预处理、模型加载或网络。容量与训练损失同时不同，不能把差异全部归因于架构。
- 最后8卡Job于15:14:19成功、15:14:40删除Pod/节点归零；全部Goal GPU已释放。既有CPU桥接`env-zzl`和先前存在的非Goal Notebook保持原状。远端实际根目录为`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/goal_20261016_precision_v2/`，有效pipeline为`code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3/`；不要按历史排队/运行快照重提。
- **完整本地备份及最终总验收均通过**：[9分卷/1,300文件逐文件核验](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FINAL_BACKUP_LOCAL_AUDIT.json)PASS，178份best/last摘要与独立ledger一致，412份可用文件安全提取至`outputs/stage06_continuous/final_delivery_20261016/verified/`；代表SASRec13/3031的独立副本通过[原消费者CPU恢复检查](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/LOCAL_RECOVERY_TRANSACTION_AUDIT.json)，CUDA RNG仅保留字节、未执行，未追加训练。本地复用路径见[已验证产物索引](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/LOCAL_ARTIFACT_INDEX.json)。用户已[接受费用计算口径](../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/FEE_ACCEPTANCE_20261016.json)：实际分配33.2203/平台runtime32.9347 GPU小时按费率折算同值点券作为费用交付，仍保留实际结算未核实说明；不再等待费用决定。本轮全部验收通过，交付提交`aec2307`（未push），当前无需再执行本Goal；Goal工具状态以当前任务状态为准。

## 10. 新对话开场词

```text
继续 experiment1 的06连续向量强基线项目。先读 docs/guides/PROJECT_HANDOFF.md、INSPIRE.md、experiments/06_continuous_baseline/README.md 和 ACCEPTANCE.md，再核对当前Git及实际数据身份。

06已经完成CPU/GPU验收、固定89项有效训练、验证封存选择、13项测试逐样本重算、10份导出重载和最终测速；最后8卡Job已SUCCEEDED且Pod删除、节点归零。9个备份分卷/1,300文件全量核验及代表SASRec13/3031独立副本CPU恢复检查均已PASS，412份可用文件在outputs/stage06_continuous/final_delivery_20261016/verified/；CUDA RNG仅字节保全。本地复用以execution_acceptance/final_delivery_20261016/LOCAL_ARTIFACT_INDEX.json中的路径和摘要为准；用户已接受33.2203分配/32.9347平台runtime点券计算口径作为费用交付，并保留actual_goal_settlement_verified=false；本轮全部验收通过，交付提交aec2307且未push，当前无需再执行本Goal，不重复下载。下一研究为尚未实施的固定推荐器编码对照，另行确定实验范围。不要重生成数据、重跑训练或重提GPU，不停止非Goal Notebook/CPU桥接。

区分05早期长期四组、最新共享起点短四组与06连续模型工作；旧SID-level短对照不构成06商品级推荐结果。06按验证结果选择SASRec＋FullCE，5seed商品级测试HR@10/NDCG@10均值为0.072280/0.041515；Decoder适配编码及列表生成仍是后续设想。费用确认见execution_acceptance/final_delivery_20261016/FEE_ACCEPTANCE_20261016.json：费用交付已获用户接受，计算金额仍不冒称平台实际结算；本轮全部验收通过，交付提交aec2307，Goal工具状态以当前任务状态为准。
```

每次完成一项明确任务，更新本摘要的日期、已完成状态、证据入口和下一步；稳定操作规则留在已有指南中，不复制多份“最新版”。

## 附录：06初始模型CPU验收及首个Job提交（历史快照）

- 当前分支`codex/continuous-baseline-8h100`，执行commit `4d7c4662c077ca3aaa9d78363e1ee09d8bb34432`已提交并同步；原HEAD `78d446d`不包含06实现。既有README改动、无关草稿及两个历史引擎保持原样。
- 真实CPU验收72项零跳过（模型18、队列9、交付8、数据37）；源码SHA256 `7bff9f19bfef7ef8ec4ee9b058a6ff6cf7baeea3f469c5dc3f87c1cf25031ccc`。缓存全部326,533条独立核验通过。[证据](../../experiments/06_continuous_baseline/execution_acceptance/local_cpu_20261016_01/CPU_GATE.json)。
- 两模型各真实数据3次CPU更新、22,363条验证、独立指标重算和导出重载通过；不计正式89次训练，没有测试集推荐指标。[集成收据](../../experiments/06_continuous_baseline/execution_acceptance/local_cpu_20261016_01/cpu_integration.json)。
- 远端35项模型/队列/交付CPU测试、326,533条缓存核验通过，源码与本地gate一致。[远端gate](../../experiments/06_continuous_baseline/execution_acceptance/remote_cpu_20261016_01/CPU_GATE.json)。
- 8H100 Job `nancy-stage06-20261016-8h100-01`已提交，最近平台状态QUEUING，正式训练0/89。预算仍为89次，smoke不计入；先核实此Job再恢复，禁止重复提交。[提交收据](../../experiments/06_continuous_baseline/execution_acceptance/PLATFORM_SUBMISSION_20261016.json)。
- 本轮独立根目录`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/goal_20261016/`，详见INSPIRE.md。原主项目个人目录写入报quota exceeded，未删除历史文件；全部新输入和执行代码在新目录重新核验。
- 前述第8节是第一阶段时点记录。当前CPU桥接已实测，既有环境Python3.10.20、Torch2.6.0+cu118、NumPy2.2.6；未改共享环境。
- 已授权的后续工作：提交并同步固定版本/原数据，远端CPU gate后一个8H100批处理Job，固定89次训练、冻结选择后测试测速；Job退出释放GPU，CPU完成交付、账单核对及交接。
