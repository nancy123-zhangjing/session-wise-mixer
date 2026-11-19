# Five-arm execution tracking

Status: formal runs submitted; **not accepted and not complete**. No freeze branches.

## Active locations

Remote prefix: `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/`.

| Arm | Job | Root |
|---|---|---|
| CF-M0 | nancy-five-cf-m0-1113-v1 | session_wise_ar_five_20261113_v1 |
| AR-T0 | nancy-five-ar-t0-1113-r2 | session_wise_ar_five_20261113_r2 |
| AR-T1 | nancy-five-ar-t1-1113-r2 | session_wise_ar_five_20261113_r2 |
| AR-D0 | nancy-five-ar-d0-1113-r2 | session_wise_ar_five_20261113_r2 |
| AR-D1 | nancy-five-ar-d1-1113-r2 | session_wise_ar_five_20261113_r2 |

Each formal job requests one H100. GPU workspace is 分布式训练空间. CPU file channel is `nancy-s09-e-prep-1101-r1` in CPU资源空间; it was originally stopped and must be stopped after final delivery. `env-zzl` was stopped after an unsuccessful resource request. Completed/failed short gates do not retain resources.

`control.py` constants and its original `submit()` describe attempt v1, not current AR attempt r2. Do not blindly resubmit. Current small local receipts: `LATEST_PROGRESS.json`, `DEPLOYMENT_PROGRESS.json`, `raw/*R2_SUBMITTED.json`. Remote phase status is `<arm>_STATE.json`, checkpoints under `runs/<arm>/`, log `<arm>.log`. These are live progress, not final acceptance receipts.

## Source and repair lineage

- Scientific implementation: b66a50f and its prerequisite commits.
- Original uploaded package SHA256: 33b88a2ddee3d63e22855c31398b6e7fa17e765d8332bfb8923c68f78ae8b537.
- d87e2b2 repairs the upstream cached device attribute after moving DIGER to CUDA. The original native sources are unchanged. See DEVICE_FIX.md.
- 2bd8c68 removes final-catalog suffix-capacity enforcement from intermediate pretraining diagnostics only. Final catalog enforcement remains. See DIAGNOSTIC_FIX.md.
- All AR r2 parents/checkpoints are copied with preserved training state and updated source/parent metadata into a separate root. Original v1 remains intact. `DIAGNOSTIC_MIGRATION.json` records old/new SHA and paths.
- Resume points: T0 step400, T1 step19800, D0 step4200, D1 step400. These are pretraining updates; saved RNG, optimizer, stream and model are preserved. Only incomplete work after those snapshots may be repeated.
- CF original engine, parent, model settings and running job were unaffected.

## Verified boundary

Combined GPU gate passed TIGER and DIGER full architectures on real parents, batch256 FP32. CPU/GPU loss matched, restored next-step parameters matched exactly, and both produced complete unique five-item lists. `GPU_ACCEPTANCE.json` is a numerical/restore gate, not recommendation-quality evidence. The later diagnostic-only correction passed two targeted CPU tests including deliberately overflowing intermediate codes; it does not change training updates.

Initial GPU step measurements: T1 0.08706 seconds, D1 0.12103 seconds. These are one-step estimates, not steady-state end-to-end throughput. Native budgets imply roughly 4.8h TIGER recommendation updates or 55.9h DIGER maximum updates before validation/pretraining; DIGER can early-stop. Do not change budgets to meet an invented runtime target.

Important observation: early TIGER pretraining losses and Mixer correction norms are very large. Preserve curves and final catalog health; do not claim quality acceptance or automatically lower learning rate. The intermediate overflow diagnostic failure alone does not establish that the selected final tokenizer is unusable.

Remaining: follow all runs, check first full validations and throughput, full best/final tests, source/identity/recovery and artifact audit, comparisons with existing CF-M1, release temporary resources, final report and Goal completion.

## 2026-11-14 02:31 新完整节点（非最终验收）

T0第4轮/55428更新全验证NDCG@5=0.005145779132106038，刷新best；CE=1.9412778982145857，验证457.12秒。D1第2轮/27714次R及T更新全验证NDCG@5=0.0008843805151886596，best仍第1轮0.0010131827140747124；碰撞商品比例由49.03%增至66.28%，语义SID变化97.70%，修正相对RMS=4.82/10.21/14.75。D1第二轮各列表位置命中率为0/0/0/0/0.00674044，仅第五位置有命中，属于需要继续观察的退化信号，不据此擅自改模型或提前停止。D0第二轮尚未完成，因此不作错轮对照。

CF R24/T190全验证NDCG@5=0.005400685569597435，比已有M1同轮低3.90%，best仍R9。全部新增节点验证样本数71657，checkpoint及预测SHA见对应JSON。D1本轮验证662.64秒，恢复至约11分钟量级；三组新节点证据已纳入阶段曲线。未重新推理、未修改训练配置。

## 2026-11-14 02:51 DIGER第二轮同预算对照

D0/D1均27714次R更新和27714次T更新，各71657条全验证。

| 指标 | D0无Mixer | D1层内Mixer |
|---|---:|---:|
| 第1轮NDCG@5（当前best） | 0.001642068 | 0.001013183 |
| 第2轮NDCG@5 | 0.000297857 | 0.000884381 |
| 第2轮碰撞商品比例 | 21.0850% | 66.2752% |
| 第2轮SID变化比例 | 92.7666% | 97.6976% |
| 第2轮语义CE | 2.141390 | 2.058952 |

D0相对第1轮下降约81.9%，D1约12.7%；D1本轮分数约为D0的2.97倍，但两者绝对分数均低且没有超过各自第1轮。D0编码碰撞仍远低于D1，推荐分数却更低，因此不能用碰撞率单独解释推荐效果，也不能将所有退化归因于Mixer。两组SID均大量变化，联合训练移动目标/适应不足是待验证假说，不是已证明机制。继续按原patience15与120轮上限，不擅自调整训练。D0本轮验证678.81秒，未重新推理；SHA和完整指标见AR_D0_SECOND_VALIDATION.json。

## 2026-11-14 02:59 T0第五轮完整验证

T0第5轮/69285更新：全验证71657条，NDCG@5=0.0031079254936429634，低于第4轮best=0.005145779132106038；训练CE从1.941278降至1.923879，验证454.11秒。Tokenizer冻结，SID和有效码本变化均0，因此本次下降不能归因于Tokenizer持续更新；尚不能凭单轮波动认定过拟合或确定具体机制。节点权重和预测SHA见AR_T0_FIFTH_VALIDATION.json。T1没有推荐验证，无法做同系列Mixer同轮比较；继续执行T0原200000更新预算。

## 2026-11-14 03:05 D1第三轮完整验证

D1第3轮/41571次R与T更新，全验证71657条，NDCG@5=0.00026031534439909513；best仍第1轮0.0010131827140747124。碰撞商品比例77.9083%，碰撞冗余62.7983%，语义SID变化99.6085%；第三层利用率76.9531%，perplexity=34.7091（第2轮153.1122），说明码字使用进一步集中。语义CE由2.058952降至1.912836，但推荐指标继续下跌；这支持训练目标改善未转化为生成效果的观察，不证明唯一原因。D0第三轮尚未完成，不作错轮比较。验证661.34秒，节点及预测SHA见AR_D1_THIRD_VALIDATION.json。继续原早停和预算，未新增实验。

## 2026-11-14 03:18–03:25 临时CPU通道恢复与CF R32

临时CPU nancy-s09-e-prep-1101-r1达到原4小时自动停止时间，依次观察RUNNING（倒计时已过）、COMMITTING、STOPPED。确认STOPPED后按原规格start，使用--post-start none；恢复RUNNING后旧SSH通道报TunnelError。通过notebook connection refresh刷新连接，随后AR/CF远端指标读取均成功。原4个GPU作业持续RUNNING，未停止/重启、未改变实验设置。平台状态与恢复收据保存在raw/CPU_BEFORE_RESTART.json、CPU_RESTART_20261114.json、CPU_AFTER_RESTART.json、CPU_CONNECTION_REFRESH_20261114.json；连接收据仅本地留存，不提交潜在敏感连接内容。

CF R32/T270全验证71657条，NDCG@5=0.00567279386040767，相对有Mixer同轮+1.15%，best仍R9=0.006129979838986507。当前已进入R33验证；R32权重与预测身份见CF_INTERIM_COMPARISON.json及原验证summary。该进展不是最终测试/恢复验收。

## 2026-11-14 03:26 D0第三轮与T0第六轮

D0第3轮/41571次R和T更新，全验证NDCG@5=0.0010259078476177017，较第2轮回升但best仍第1轮0.0016420679327425387；同轮D1=0.00026031534439909513。D0碰撞商品比例31.8233%，D1同轮77.9083%；D0 SID变化87.3415%，D1同轮99.6085%。D0碰撞率继续上升而推荐分数回升，故碰撞率不是逐轮分数变化的充分解释。D0本轮验证718.47秒。

T0第6轮/83142推荐更新，全验证NDCG@5=0.0040311509333954875，从第5轮回升，但best仍第4轮0.005145779132106038。CE继续降至1.912516，Tokenizer/SID保持冻结。T1无推荐指标，无法做TIGER组内完整消融。两新增节点均71657条，权重和预测SHA见AR_D0_THIRD_VALIDATION.json、AR_T0_SIXTH_VALIDATION.json；本次只归档已有评估，未重跑或改配置。


## 2026-11-14 CF R34完整验证

CF-M0 R34/T290全验证71657条，NDCG@5=0.005711814204310005；同轮CF-M1=0.005461494742127955，相对差异4.58%。best仍为R9的0.006129979838986507。逐轮表、曲线及来源清单已更新到R34。本次只读取已有结果，未改训练设置或重复推理；R35和best/final测试尚待完成。


## 2026-11-14 D1第四轮完整验证与CF节点盘点

D1第4轮/55428次R与T更新，全验证71657条，NDCG@5=0.0015383142545570365，超过原第1轮best=0.0010131827140747124约51.83%，成为新best。第三轮0.0002603之后显著回升；碰撞商品比例反而继续升至81.1055%，第三层perplexity降至24.5398，SID变化91.4523%。因此此前三轮下跌不能外推为持续单调崩塌，碰撞增加也不能单独解释指标走势。总loss=2.155982、语义CE=1.833721、后缀loss=0.259031。D0第四轮未完成，不作错轮Mixer比较。节点和预测身份见AR_D1_FOURTH_VALIDATION.json。

CF R5–R34共30个完整验证节点的checkpoint均存在且非空，盘点见CF_CHECKPOINT_INVENTORY.json。这仅是路径/大小核对，尚未完成逐文件SHA、结构和恢复验收，不代表最终验收通过。


## 2026-11-14 CF R35最终验证

CF-M0 R35/T300全验证71657条，NDCG@5=0.005713081585116052；CF-M1同轮=0.005517515338743016，相对差异3.54%。best仍R9=0.006129979838986507，低于M1 best R9=0.006534307434012639。31个验证节点已齐，曲线已更新；best/final测试与最终验收仍待完成。未重复训练或评估。


## 2026-11-14 CF best全测试完成（final待完成）

CF-M0验证选best R9/T40，全测试53194条NDCG@5=0.008245519147706479，低于已有CF-M1 best测试0.008854949313989101约6.88%。summary checkpoint SHA与BEST.json一致，原自动测试执行了JointTrainer.restore；该事实仅覆盖best实际GPU恢复，不替代31节点完整验收。final测试及TEST_COMPLETE总收据尚未完成，未重复评估。详情见CF_BEST_TEST.json。


## 2026-11-14 T0第七轮完整验证

T0第7轮/96999次推荐更新，全验证71657条NDCG@5=0.003259437265772693；best仍第4轮0.005145779132106038，第6轮0.0040311509333954875之后再次下降。训练CE=1.904129759923704，Tokenizer/SID仍冻结。第5到7轮表现为下降、回升、再下降，暂不能从单一loss或编码变化解释。T1未进入推荐训练，无法提供同轮Mixer对照。节点权重与预测SHA见AR_T0_SEVENTH_VALIDATION.json，继续原200000更新预算，未改变早停或训练配置。


## 2026-11-14 DIGER第四轮同轮对照

D0第4轮/55428次R/T更新，全验证71657条NDCG@5=0.0014743626765266425，比第3轮回升但未超过第1轮best=0.0016420679327425387；D1同轮0.0015383142545570365，比D0高4.34%，但D1历史best仍低于D0历史best。D0训练总loss=2.1424400937221275，碰撞商品比例=38.7957%，SID变化=76.4448%。两组第4轮都回升，仍处早期波动阶段，不能只凭此前下降判定单调崩塌。继续原patience15和120轮上限，未改配置。节点与预测SHA见AR_D0_FOURTH_VALIDATION.json。


## 2026-11-14 DIGER有Mixer第五轮完整验证

D1第5轮/69285次R/T更新，全验证71657条NDCG@5=0.0008459906783719412；best仍第4轮0.0015383142545570365，本轮相对best下降约45.00%。训练总loss=2.100586215522902，语义CE=1.7806734689299686，碰撞商品比例=81.9351%，SID变化=93.2047%。D0第五轮尚未完成，不作错轮对比。第4轮回升之后第5轮再次下降，继续按原patience15与120轮上限观察，不自行增加冻结或修订训练。节点与预测身份见AR_D1_FIFTH_VALIDATION.json。


## 2026-11-14 TIGER第八轮完整验证

T0第8轮/110856次推荐更新，全验证71657条NDCG@5=0.0029848826527204574，低于第7轮0.003259437265772693；best仍第4轮0.005145779132106038。训练CE=1.8963873533612225，Tokenizer/SID仍冻结。第6至8轮验证连续下降，训练CE继续改善，但尚不据此改变原200000更新预算。T1没有推荐节点，无法作同系列Mixer对照。权重/预测SHA与全指标见AR_T0_EIGHTH_VALIDATION.json。


## 2026-11-14 DIGER无Mixer第五轮刷新best

D0第5轮/69285次R/T更新，全验证71657条NDCG@5=0.0033764510883917583，超过原第1轮best=0.0016420679327425387约105.62%，成为新best；D1同轮0.0008459906783719412，D0约为其3.99倍。D1历史best仍第4轮0.0015383142545570365。D0训练总loss=2.083001904955862，碰撞商品比例=42.3471%，SID变化=66.7972%。D0持续联合训练后明显改善，说明不能把早期低分视为最终结论；同样不据单节点宣称整体方法优劣。节点和预测身份见AR_D0_FIFTH_VALIDATION.json。继续原预算与早停，不新增实验。


## 2026-11-14 DIGER有Mixer第六轮完整验证

D1第6轮/83142次R/T更新，全验证71657条NDCG@5=0.0014326253525003703，从第5轮0.0008459906783719412回升，但未超过第4轮best=0.0015383142545570365。训练总loss=2.070932441706979，碰撞商品比例=80.5183%，SID变化=92.8225%。D0第6轮尚未完成，不作错轮Mixer对照。保持原训练预算和早停规则；权重和预测身份见AR_D1_SIXTH_VALIDATION.json。


## 2026-11-14 TIGER第九轮完整验证

T0第9轮/124713次推荐更新，全验证71657条NDCG@5=0.004419633244735153，从第8轮0.0029848826527204574回升，仍未超过第4轮best=0.005145779132106038。训练CE=1.889943394051322，Tokenizer/SID保持冻结。此前连续下降后再次回升，当前曲线仍有明显波动；继续原200000更新预算，不据局部下降改早停。T1没有推荐节点，无法同系列Mixer对照。权重与预测身份见AR_T0_NINTH_VALIDATION.json。


## 2026-11-14 DIGER第六轮同轮对照及D1第七轮

D0第6轮/83142次R/T更新，全验证71657条NDCG@5=0.0031154421513666557，略低于第5轮best=0.0033764510883917583；D1第6轮为0.0014326253525003703，D0同轮约为其2.17倍。D1第7轮/96999次R/T更新全验证NDCG@5=0.0009199033585684325，低于第6轮，best仍第4轮0.0015383142545570365。D0第7轮未完成，不作错轮对照。所有节点保持71657条全验证，权重与预测身份见AR_D0_SIXTH_VALIDATION.json和AR_D1_SEVENTH_VALIDATION.json。继续原预算和早停，不更改训练。


## 2026-11-14 TIGER第十轮完整验证

T0第10轮/138570次推荐更新，全验证71657条NDCG@5=0.0046710387719276925，从第9轮0.004419633244735153继续回升，仍未超过第4轮best=0.005145779132106038。训练CE=1.884830638143137，Tokenizer/SID保持冻结。T1未进入推荐训练，无同系列Mixer节点可比。原200000更新预算不变，继续后续训练；权重与预测身份见AR_T0_TENTH_VALIDATION.json。


## 2026-11-14 TIGER第十一轮新最佳

T0第11轮/152427次推荐更新，71657条全验证NDCG@5=0.007043733550694812，超过第4轮旧best=0.005145779132106038，提升约36.89%。该节点成为当前best；T1未进入推荐训练，无同系列Mixer结果。此前回落后恢复并创新高，不能将中期下降判断为持续退化。尚未完成训练或测试，继续原200000次更新预算。权重和预测身份见AR_T0_ELEVENTH_VALIDATION.json。


## 2026-11-14 DIGER有Mixer第八轮完整验证

D1第8轮/110856次R/T更新，71657条全验证NDCG@5=0.0011392697463794921，较第7轮0.0009199033585684325回升，仍低于第4轮best=0.0015383142545570365。D0第8轮尚未完成，不作错轮对照。节点身份见AR_D1_EIGHTH_VALIDATION.json。继续原预算和早停，不改训练配置。


## 2026-11-14 DIGER第七轮同轮对照

D0第7轮/96999次R/T更新，71657条全验证NDCG@5=0.0030827766809761355，略低于第6轮0.0031154421513666557，best仍第5轮0.0033764510883917583。D1同为第7轮0.0009199033585684325，D0约为其3.35倍。D0训练总loss=2.039570；碰撞商品49.9441%，相邻节点SID变化50.7457%。这是单seed中途结果，继续原预算及早停规则。节点权重和预测身份见AR_D0_SEVENTH_VALIDATION.json。


## 2026-11-14 TIGER第十二轮新最佳

T0第12轮/166284次推荐更新，71657条全验证NDCG@5=0.007940415149855854，再次刷新best，较第11轮0.007043733550694812提升约12.73%。T1没有推荐节点，同系列Mixer对照缺失。当前仅为验证结果，尚无best/final测试，不据此宣布方法整体优胜。继续原200000更新预算。节点身份见AR_T0_TWELFTH_VALIDATION.json。


## 2026-11-14 DIGER有Mixer第九轮新最佳

D1第9轮/124713次R/T更新，71657条全验证NDCG@5=0.001857717393845603，超过第4轮旧best=0.0015383142545570365，提升约20.76%，原早停等待计数由此重置。D0第9轮未完成，暂无同轮对照。不能沿用之前假定不再改善的最早结束时间。继续原预算和早停规则。节点身份见AR_D1_NINTH_VALIDATION.json。


## 2026-11-14 DIGER无Mixer第八轮新最佳

D0第8轮/110856次R/T更新，71657条全验证NDCG@5=0.004791170058168311，超过第5轮旧best=0.0033764510883917583，提升约41.90%。D1同轮为0.0011392697463794921，D0约为其4.21倍。原早停等待计数随新best重置，不能继续使用之前假设不改善时的结束估计。继续原预算与早停。节点身份见AR_D0_EIGHTH_VALIDATION.json。


## 2026-11-14 TIGER第十三轮完整验证

T0第13轮/180141次推荐更新，71657条全验证NDCG@5=0.007728235526563155，较第12轮best=0.007940415149855854略低约2.67%，仍高于第11轮。T1无推荐节点，无法同系列Mixer对照。继续原200000更新预算；尚未完成最终测试。节点身份见AR_T0_THIRTEENTH_VALIDATION.json。


## 2026-11-14 DIGER有Mixer第十轮完整验证

D1第10轮/138570次R/T更新，71657条全验证NDCG@5=0.0005719469183080919，较第9轮best=0.001857717393845603下降约69.21%。D0第10轮尚未完成，不作错轮对照。该节点为真实全验证波动，尚不能单凭一次下降判断后续不可恢复；继续原早停和预算，不改配置。节点身份见AR_D1_TENTH_VALIDATION.json。


## 2026-11-14 DIGER无Mixer第九轮新最佳

D0第9轮/124713次R/T更新，71657条全验证NDCG@5=0.0063277115815283756，较第8轮best=0.004791170058168311提升约32.07%，再次刷新best。D1同为第9轮0.001857717393845603，D0约为其3.41倍。原早停等待计数随新best重置，继续原预算。该比较仍是单seed中途验证，不替代最终测试。节点身份见AR_D0_NINTH_VALIDATION.json。


## 2026-11-14 TIGER第十四轮新最佳

T0第14轮/193998次推荐更新，71657条全验证NDCG@5=0.00807443092331738，超过第12轮旧best=0.007940415149855854，提升约1.69%。剩余6002次更新至原200000预算，仍需实际最终点验证以及best/final全测试。T1无推荐节点，无同系列Mixer对照。节点身份见AR_T0_FOURTEENTH_VALIDATION.json。

## 2026-11-14 07:29 CST：T0训练完成，D1第11轮

- 只读远端核验：T0 train/complete.json已生成，200000次推荐更新完成；best为193998步（第14轮），验证NDCG@5=0.00807443092331738；实际final为200000步（14.43316轮），验证=0.005795711094421075。test/complete.json尚未生成，平台作业仍RUNNING，不能视为测试或最终验收完成。
- D1第11轮152427步完整验证71657条，NDCG@5=0.0010855502659019777，best仍为第9轮0.001857717393845603。本轮SID变化99.3475%，碰撞商品83.4732%，总loss=2.0672706385399016。D0、D1平台均RUNNING，原配置继续。
- 新节点验证身份与checkpoint SHA相符，归档两份小收据并刷新已有指标曲线，不涉及重新训练或推理。大权重保持远端。T1失败处置仍未解决，整体Goal保持进行中。

## 2026-11-14 07:31 CST：T0 best全测试完成

- 第14轮（193998步）best全测试53194条，NDCG@5=0.008107861893687628；相应全验证=0.00807443092331738。测试与验证checkpoint/config身份一致，保存AR_T0_BEST_TEST.json。
- 实际final测试尚未完成，平台T0/D0/D1仍RUNNING。此收据不代表完整续训恢复或最终验收通过。

## 2026-11-14 07:34 CST：T0全部验证节点CPU检查

- audit_ar_nodes.py在远端只读检查T0全部15个验证checkpoint，通过SHA、验证身份/样本数、配置/源码/父状态/环境、模型状态字段、优化器计数、数据流排列与计数、CPU随机状态可载入、CUDA随机状态存在、SID范围与完整路径唯一性检查。结果见AR_T0_NODE_AUDIT_RESULT.json。
- 审计直接读取已完成训练的实际文件；不执行forward、优化更新或额外评测。边界：未逐节点执行CUDA数值恢复，未声称所有张量有限性或所有调度器状态数值等价。本脚本在delivery目录，不改变已冻结训练源码。

## 2026-11-14 07:37 CST：T0全测试与GPU收尾

- best测试0.008107861893687628；final测试0.00477805561597721，各53194条，test/complete.json已生成。
- 已实际校验远端源码清单和两个预测文件SHA，验证/测试身份一致。作业SUCCEEDED、worker instance_success；无需停掉仍运行的DIGER或CPU读通道。
- 报告AR_T0_COMPLETED_REPORT.md保留恢复证据边界。整体Goal未完成。

## 2026-11-14 07:40 CST：D0第10轮回落

- D0第10轮138570步，全验证71657条，NDCG@5=0.0034863334879290993；历史最佳仍为第9轮0.0063277115815283756。本轮较best下降约44.9%，同轮D1为0.0005719469183080919。
- 总loss从2.02133779略降至2.01730953；SID变化比例34.9459%，碰撞商品比例54.2040%。不能据此前第8/9轮连续提高就断言已稳定收敛，也不能仅凭这一轮认定持续崩塌。
- checkpoint与验证SHA一致，已归档AR_D0_TENTH_VALIDATION.json并更新曲线；D0/D1实时平台状态均RUNNING，配置、预算与早停不变。

## 2026-11-14：D1第12轮完整验证

- 166284步，全验证71657条，NDCG@5=0.0014942316033381457；较第11轮0.0010855502659019777回升，仍低于第9轮最佳0.001857717393845603。D0尚无同第12轮结果，不作错位轮次的严格消融比较。
- 本轮总loss=2.04740237，SID变化95.3300%，碰撞商品83.4638%；三层修正/基础码本RMS比4.3729/13.6506/20.2409。编码仍有大幅变化；不由这一个节点推断因果机制。
- 验证身份与checkpoint SHA一致，归档AR_D1_TWELFTH_VALIDATION.json并刷新曲线。D0/D1平台均RUNNING，保持原预算与早停。

## 2026-11-14：D0第11轮验证回升

- 152427步，全验证71657条，NDCG@5=0.006073020835002444；从第10轮0.0034863334879290993回升，best仍为第9轮0.0063277115815283756。同第11轮D1为0.0010855502659019777。
- 总loss=2.01291504，SID变化29.2878%，碰撞商品54.8005%。SID变化继续减小，但碰撞商品比例仍缓慢增加；第10轮下跌未延续为连续下降，不能单点判为训练崩塌。
- 验证与checkpoint身份相符，已归档并更新曲线。D0/D1均RUNNING，继续原训练和早停规则。

## 2026-11-14：D1第13轮刷新best

- 180141步，全验证71657条，NDCG@5=0.0019651954843594795，超过第9轮0.001857717393845603；best_step已更新，bad_epochs=0。第11→12→13轮连续回升，但绝对水平仍较低。D0尚无同第13轮指标。
- 总loss=2.03388451，SID变化96.4392%，碰撞商品86.8848%，第三层perplexity=12.7078。推荐提升与碰撞恶化同时出现，不能简单认为碰撞比例决定推荐分数。
- 原patience=15自动重置；若后面不再提升，最早第28轮触发早停，未来提升还会继续延后，不能沿用第24轮预计终点。配置预算不改，两组均RUNNING。
- checkpoint/验证SHA核对一致，已归档AR_D1_THIRTEENTH_VALIDATION.json并更新曲线。

## 2026-11-14：D0第12轮刷新best

- 166284步，全验证71657条，NDCG@5=0.006487202148675596，超过第9轮0.0063277115815283756；best_step更新，bad_epochs=0。同第12轮D1为0.0014942316033381457。
- 总loss=2.00522795，SID变化27.3956%，碰撞商品55.3039%。编码变化逐步减小，碰撞商品比例继续增加；推荐指标创best，不能把碰撞比例单独用作推荐质量判断。
- 原patience=15重置，若以后不再提高，最早第27轮触发早停；D1对应最早第28轮，后续新best仍会延后。未改配置或预算。
- checkpoint与验证SHA一致，归档AR_D0_TWELFTH_VALIDATION.json并刷新曲线。两组平台均RUNNING，整体Goal未完成。

## 2026-11-14：D1第14轮验证回落

- 193998步，全验证71657条，NDCG@5=0.001255996521024628；best仍为第13轮0.0019651954843594795。同第14轮D0结果尚无，不作错位轮次严格比较。
- 总loss=2.03602854；SID变化99.6737%，碰撞商品78.1693%，第三层perplexity升至142.6883。碰撞比例下降、码字使用更分散，但推荐指标下降，再次说明这些编码指标不能单独替代推荐效果。
- checkpoint与验证SHA一致，归档AR_D1_FOURTEENTH_VALIDATION.json并更新曲线。两组均RUNNING，按原配置与早停继续。

## 2026-11-14：D0第13轮验证回落

- 180141步，全验证71657条，NDCG@5=0.005024501700209792；best仍为第12轮0.006487202148675596。同第13轮D1为0.0019651954843594795。
- 总loss=2.00906195，SID变化25.6339%，碰撞商品56.2919%。较第12轮推荐回落约22.55%，仍高于同轮有Mixer组；不据单次下降认定崩塌。
- checkpoint与验证SHA一致，已归档并更新曲线。两组平台均RUNNING，继续原配置和早停规则。

## 2026-11-14：D1第15轮验证回升但未刷新best

- 207855步，全验证71657条，NDCG@5=0.0015950522966358405，较第14轮回升约27.0%；best仍为第13轮0.0019651954843594795。D0尚无同第15轮结果，不作错位轮次严格比较。
- 总loss=2.05080233，SID变化96.7655%，碰撞商品80.6488%，第三层perplexity=80.1772。编码仍剧烈变化，本次推荐回升不代表已稳定。
- checkpoint与验证SHA一致，归档并更新曲线。两组平台均RUNNING，原配置与早停规则保持不变。

## 2026-11-14：D0第14轮小幅回升

- 193998步，全验证71657条，NDCG@5=0.005560058547187225，较第13轮回升约10.66%；best仍为第12轮0.006487202148675596。同第14轮D1为0.001255996521024628。
- 总loss=2.00860678，SID变化25.0932%，碰撞商品56.6928%。推荐表现仍有波动，尚未刷新最佳；不据单个节点推断后续趋势。
- checkpoint与验证SHA一致，归档并刷新曲线。两组平台均RUNNING，原配置和早停不变。

## 2026-11-14：D1第16轮验证明显回落

- 221712步，全验证71657条，NDCG@5=0.0006317013464024306，较第15轮下降约60.4%；best仍为第13轮0.0019651954843594795。D0尚无同第16轮结果，不作错位轮次严格比较。
- 总loss=2.04823633，SID变化97.8747%，碰撞商品79.8937%，第三层perplexity=144.4017。码字使用较第15轮更分散而推荐下降，编码状态指标不能单独代替推荐效果。
- checkpoint与验证SHA一致，归档并更新曲线。两组平台均RUNNING，继续原配置和早停规则。

## 2026-11-14：D0第15轮验证回落

- 207855步，全验证71657条，NDCG@5=0.00420499725869128，较第14轮下降约24.4%；best仍为第12轮0.006487202148675596。同第15轮D1为0.0015950522966358405。
- 总loss=2.01118062，SID变化23.2383%，碰撞商品57.2334%。SID变化减小未带来本轮推荐提升；不据单点确认具体退化机制。
- checkpoint与验证SHA一致，归档并更新曲线。两组平台均RUNNING，继续原配置和早停规则。

## 2026-11-14：D1第17轮验证回升但仍低于best

- 235569步，全验证71657条，NDCG@5=0.0010776222440485315，较第16轮回升约70.6%；best仍为第13轮0.0019651954843594795。D0尚无同第17轮结果，不作错位轮次严格比较。
- 总loss=2.06506305，SID变化97.4739%，碰撞商品81.6089%。本次回升未恢复到第15轮水平，编码仍剧烈变动，不据单点认定训练已稳定。
- checkpoint与验证SHA一致，归档并更新曲线。两组平台均RUNNING，继续原配置和早停规则。

## 2026-11-14：D0第16轮继续回落

- 221712步，全验证71657条，NDCG@5=0.0034310467682520667，较第15轮下降约18.4%；best仍为第12轮0.006487202148675596。同第16轮D1为0.0006317013464024306。
- 总loss=2.00773887，SID变化23.1544%，碰撞商品57.4944%。第14至16轮连续下降，但总loss本轮略降；不能只凭训练loss判断推荐收益，也尚不能确定具体下降原因。
- checkpoint与验证SHA一致，归档并更新曲线。两组平台均RUNNING，继续原配置和早停规则。

## 2026-11-14：D1第18轮验证基本持平

- 249426步，全验证71657条，NDCG@5=0.001074363935683643，较第17轮下降约0.30%；best仍为第13轮0.0019651954843594795。D0尚无同第18轮结果，不作错位轮次严格比较。
- 总loss=2.05829596，SID变化96.2435%，碰撞商品82.4758%。连续两轮推荐分数接近，但编码仍剧烈变化，不能据此认定训练稳定。
- checkpoint与验证SHA一致，归档并更新曲线。两组平台均RUNNING，继续原配置和早停规则。

## 2026-11-14：D0第17轮刷新最佳

- 235569步，全验证71657条，NDCG@5=0.007283309203289069，超过第12轮旧best 0.006487202148675596约12.27%。同第17轮D1为0.0010776222440485315。
- 总loss=2.00183869，SID变化22.2129%，碰撞商品58.0723%。此前连续下降后本轮明显回升，不能将第14至16轮的下降直接解释为不可逆崩塌。
- checkpoint与验证SHA一致，归档并更新曲线。CPU读取通道恢复，两组平台重新确认RUNNING；D1最新完整节点仍为第18轮。继续既定配置及早停规则，未启动新实验。

## 2026-11-14：D1第19轮验证完成

- 263283步，全验证71657条，NDCG@5=0.00105047218091592；best仍为第13轮0.0019651954843594795。D0最新完整验证为第17轮，尚无同第19轮结果。
- 总loss=2.05592313，SID变化94.6309%，碰撞商品79.5395%。近期推荐分数接近，但编码仍剧烈变化；仅为相关观察，未证明因果。
- checkpoint与验证SHA一致，归档并更新曲线；两组尚未完成训练或最终测试，不作最终验收结论。

## 2026-11-14：D0第18轮略微刷新最佳

- 249426步，全验证71657条，NDCG@5=0.00728592003034135，相比第17轮0.007283309203289069提升约0.036%，实际基本持平；best更新为第18轮。同第18轮D1为0.001074363935683643。
- 总loss=2.00455392，SID变化20.2088%，碰撞商品58.4638%。连续两轮较高分数，不据此提前认定后续稳定或训练收敛。
- checkpoint与验证SHA一致；完整节点归档并更新曲线，继续原预算和早停规则。

## 2026-11-14：D1第20轮回升

- 277140步，全验证71657条，NDCG@5=0.0014425280426219886；较第19轮回升，但best仍为第13轮0.0019651954843594795。D0尚无同第20轮结果。
- 总loss=2.06430509，SID变化96.0291%，碰撞商品79.0735%。编码仍大幅变动，本轮推荐回升不证明训练已稳定。
- 验证checkpoint SHA一致，归档并更新曲线；继续原训练预算与早停规则。

## 2026-11-14：D0第19轮明显回落

- 263283步，全验证71657条，NDCG@5=0.0029417414588912387；best仍为第18轮0.00728592003034135。同第19轮D1=0.00105047218091592。
- 总loss=2.00467818，与第18轮接近；SID变化19.3606%，碰撞商品58.5477%。SID变化比例下降并未保证推荐分数稳定，不能只凭loss或总体编码变动率解释本轮回落。
- 验证checkpoint SHA一致，归档并更新曲线；未改早停或训练设置，继续观察后续完整节点。


## 2026-11-14 DIGER D1 epoch 21

Full validation (71657 sessions): NDCG@5=0.0007712745939918183. Best remains epoch 13, 0.0019651954843594795; 8 validations without improvement. Semantic SID change=96.5790%, collision item fraction=80.1174%, train loss=2.057862666510284. D0 epoch 21 is pending; no mismatched-epoch comparison. This completed checkpoint predates the cap35 restart.


## 2026-11-14 DIGER D0 epoch 20

Full validation 71657 sessions: NDCG@5=0.006318296496071631, up from epoch19 0.0029417414588912387; best remains epoch18=0.00728592003034135. Same epoch D1=0.0014425280426219886. SID changed18.4937%, collision items59.0604%. Validation took651.54s. First complete validation after cap35 continuation; no final acceptance yet.


## 2026-11-14 DIGER D1 epoch22

Full validation71657: NDCG@5=0.0011644053061065649; best remains epoch13=0.0019651954843594795, 9 validations without improvement. SID change=0.982475757598877, collision item fraction=0.8208426237106323. D0 epoch22 pending, do not compare different epochs. Continuing authorized cap35 with original patience15.


## 2026-11-14 DIGER D0 epoch21 new best

Full validation71657: NDCG@5=0.007530203861786788, improves epoch18 best0.00728592003034135 by3.35%. Same-epoch D1=0.0007712745939918183. SID changed19.9758%, collision item fraction59.6663%, train loss2.002269122684548. Best now step290997/epoch21; patience counter reset. Original patience15 cannot trigger before authorized35-epoch cap from this new best. Test not yet run for this node; no final acceptance.


## 2026-11-14 DIGER D1 epoch23

Full validation71657 NDCG@5=0.0011306011484391854, slightly below epoch22=0.0011644053061065649. Best remains epoch13=0.0019651954843594795, bad_epochs10. SID changed97.0544%, collision item fraction80.6208%, total loss2.0597485586204187. D0 epoch23 pending; no cross-epoch strict comparison. If no new best, original patience15 triggers at epoch28; no rule changes.


## 2026-11-14 DIGER D0 epoch22

Full validation71657 NDCG@5=0.0038333026156822814, down48.10% from epoch21 best0.007530203861786788. Same epoch D1=0.0011644053061065649. SID changed20.2368%, collision item fraction60.9340%, total loss1.9997389358582018. Best remains epoch21; completed checkpoint and validation SHA match. No training settings changed; not final acceptance.


## 2026-11-14 DIGER D1 epoch24

Full validation71657 NDCG@5=0.0010094285433016597. Best remains epoch13=0.0019651954843594795, 11 validations without improvement. Encoding: {"utilization": [1.0, 0.99609375, 0.984375], "perplexity": [243.83840942382812, 201.1265106201172, 123.72364807128906], "collision_item_fraction": 0.7870059609413147, "collision_redundancy": 0.6341349739000746, "relative_correction_rms": [4.941566467285156, 11.073689460754395, 16.3076229095459], "semantic_sid_changed": 0.933445155620575, "suffix_changed": 0.7105704545974731, "effective_relative_change": [0.22040347754955292, 0.40638402104377747, 0.2975901663303375]}. D0 epoch24 pending; no cross-epoch strict comparison. Original patience15 remains unchanged under authorized35-epoch execution cap.


## 2026-11-14 DIGER D0 epoch23

Full validation71657 NDCG@5=0.0065173098682377376, recovering from epoch22=0.0038333026156822814; best remains epoch21=0.007530203861786788. Same epoch D1=0.0011306011484391854. Encoding={"utilization": [1.0, 1.0, 1.0], "perplexity": [248.71409606933594, 248.93370056152344, 253.69996643066406], "collision_item_fraction": 0.6123228669166565, "collision_redundancy": 0.4596383296047726, "relative_correction_rms": [0.0, 0.0, 0.0], "semantic_sid_changed": 0.16899701952934265, "suffix_changed": 0.20385906100273132, "effective_relative_change": [0.015597175806760788, 0.03410008177161217, 0.03748083859682083]}. Checkpoint and validation SHA match. No config changes; final acceptance pending.


## 2026-11-14 DIGER D1 epoch25

Full validation71657 NDCG@5=0.0011105620649109575. Best remains epoch13=0.0019651954843594795; 12 validations without improvement. Same-epoch D0 pending. Encoding={"utilization": [1.0, 1.0, 0.98046875], "perplexity": [238.36328125, 199.56533813476562, 89.86436462402344], "collision_item_fraction": 0.8323079347610474, "collision_redundancy": 0.6793437733035048, "relative_correction_rms": [4.865671634674072, 11.922876358032227, 16.993703842163086], "semantic_sid_changed": 0.9636465311050415, "suffix_changed": 0.7320096492767334, "effective_relative_change": [0.3035486936569214, 0.4624626934528351, 0.30972185730934143]}. If no new best, patience15 stops at epoch28; otherwise authorized35-epoch execution cap remains. No configuration changes, test and final acceptance pending.


## 2026-11-14 DIGER D0 epoch24

Full validation71657 NDCG@5=0.005428861578832373, below epoch23=0.0065173098682377376. Best remains epoch21=0.007530203861786788. Same epoch D1=0.0010094285433016597. Encoding={"utilization": [1.0, 1.0, 1.0], "perplexity": [248.90309143066406, 249.00160217285156, 253.64939880371094], "collision_item_fraction": 0.6225764155387878, "collision_redundancy": 0.4683072334079046, "relative_correction_rms": [0.0, 0.0, 0.0], "semantic_sid_changed": 0.1686241626739502, "suffix_changed": 0.20581655204296112, "effective_relative_change": [0.020495323464274406, 0.03436759486794472, 0.038990505039691925]}. Checkpoint and validation SHA match. Continue authorized35-epoch cap with original120-epoch learning-rate schedule; final tests and acceptance pending.


## 2026-11-14 DIGER D1 epoch26 and read-channel recovery

Full validation71657 NDCG@5=0.00112151091233072. Best remains epoch13=0.0019651954843594795, bad_epochs13. Same-epoch D0 pending. Encoding={"utilization": [1.0, 1.0, 0.97265625], "perplexity": [242.09225463867188, 203.35531616210938, 65.64630889892578], "collision_item_fraction": 0.810589075088501, "collision_redundancy": 0.659675615212528, "relative_correction_rms": [4.671833038330078, 11.240747451782227, 17.582202911376953], "semantic_sid_changed": 0.9237509369850159, "suffix_changed": 0.7141125798225403, "effective_relative_change": [0.2755785882472992, 0.36905437707901, 0.31135624647140503]}. Existing CPU stopped and was restarted; SSH refresh still timed out. Read-only job shell successfully recovered progress.json via base64 chunks. GPU training untouched. Original patience15/cap35 unchanged; final tests pending.


## 2026-11-14 DIGER D0 epoch25

Full validation71657 NDCG@5=0.007209331039666567, up from epoch24=0.005428861578832373. Best remains epoch21=0.007530203861786788. Same epoch D1=0.0011105620649109575. Encoding={"utilization": [1.0, 1.0, 1.0], "perplexity": [248.865234375, 248.63607788085938, 253.2788543701172], "collision_item_fraction": 0.6358128190040588, "collision_redundancy": 0.4777218493661447, "relative_correction_rms": [0.0, 0.0, 0.0], "semantic_sid_changed": 0.1535234898328781, "suffix_changed": 0.18223340809345245, "effective_relative_change": [0.01770806312561035, 0.03256967291235924, 0.04068000614643097]}. Read from unchanged remote progress via job shell, checkpoint SHA matches validation. No training configuration changes; final tests and acceptance pending.
