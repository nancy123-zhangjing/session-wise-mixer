# Stage08 验收矩阵

核对日期：2026-10-24。本轮为 Beauty、seed2024 的 baseline / no_context / axial 三臂，同起点、35R / 3640次R更新 / 60次T更新，完整验证集22363人、beam256；没有新增seed、测试集评估或GPU训练回放。

**训练、控制审计、完整字节与双Git恢复、六份真实网络CPU恢复、五次GPU释放均已有真实通过收据。** 最后的文档/supplement绑定由外置 `FINAL_DELIVERY_BINDING.json` 及其真实verify结果判定；本表与该绑定配套使用，不能仅凭本文把Goal标为完成。

路径约定：

- `D`＝`C:/Users/nancy/Documents/experiment1/outputs/stage08_session_codeword/delivery_20261024_final_r1/`，本地交付根。
- `G`＝`D/restored/goal_20261024_session_codeword_v1/`，全新恢复的Goal根。
- `C`＝本地 `outputs/stage08_session_codeword/control/`；`E`＝本目录；`R`＝`G/reports/final_r1/`。
- 远端来源＝`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_session_codeword/goal_20261024_session_codeword_v1/`，完整包来自其 `delivery/final_package_r1/`。

基础包内的 `G/delivery/ACCEPTANCE_MATRIX_AT_PACKAGE.md` 保留打包时状态，SHA为 `0d37f06f0588fa09012fcbb3a0e051fe4d8acdc8e48a5d850ab122ffee932c30`；**该历史快照不改写**。本版矩阵和最新Windows恢复说明通过最终supplement另行绑定。

## 前置9项

| 要求 | 权威证据 | 实际核对状态 | 范围与剩余项 |
|---|---|---|---|
| P1 双仓库冻结 | `C/SOURCE_FREEZE.json`；`G/source_bundle_restore/{code,engine}`；BACKUP_VERIFICATION / FULL_MODEL_RESTORE | **PASS**：code `152bea77d43284bb41f1887a8f43124be7e6db98`、engine `a28faf6a24c667aa4f49f208fb93d294ad5aab4f`；实际独立恢复两个Git，源码树与归档字节核对通过。 | 便携engine保留相同完整源码树，不声称补回原浅仓库缺失的父历史。 |
| P2 原工作区及其他任务保全 | SOURCE_FREEZE；`C/preexisting_root_changes/`；[HANDOFF](../HANDOFF.md) | **PASS（冻结保全范围）**：使用临时index制作执行提交，保留两个原HEAD、主index及六份原文快照；本轮未用执行副本覆盖原工作区。 | 原HEAD与执行commit不同是有意保留；交接中的本轮文档更新不允许清理其他任务未提交内容，不声称对所有并发修改做了终态审计。 |
| P3 真实输入与初始化来源 | `G/delivery/INPUTS.json`；BACKUP_VERIFICATION；FULL_MODEL_RESTORE.inputs | **PASS**：原封正式manifest SHA为 `6386be010907cf563753e448f24b3c465c830aa543e50a08f246a420c6f94abd`，14条记录及14份本地输入字节全部一致，含13项原资产和E0。 | 使用pre-05推荐器/E0，没有重新拟合或接用05/07最终联合权重。BOOTSTRAP_INPUTS是准备记录，不能替代正式manifest。 |
| P4 Linux CPU及数据构造 | `C/LINUX_ACCEPTANCE.json`、FORMAL_READINESS | **PASS**：真实Linux107项、0 skip；train105668、val22363；原收据SHA已核。 | 本次未重跑这些测试；test条数构造不是测试指标验收。训练早期前缀扩样、验证最近50件历史的原构造保留。 |
| P5 H100与精度/梯度/短恢复 | `G/smoke/H100_GATE.json`、`SESSION_SMOKE_PASS.json`、`PROFILE.json` | **PASS**：真实H100，R BF16 autocast / T FP32；history-only、目标独立、零起点输出等价、F回C梯度、T冻结及短状态下一阶段恢复一致。 | 此证据只覆盖短验收；最终完整轨迹和六模型恢复分别由F3/F14验收，不能互相冒充。 |
| P6 H100测速及启动规则 | PROFILE / FORMAL_READINESS；`E/platform_pre_smoke/POLICY.json` | **PASS**：短采样按完整预算外推，含20%余量预计9.22 GPU小时；启动policy与绑定SHA相符。 | 预计值没有当作实际费用；正式完整val成本见F7，真实占用见F9/F10。 |
| P7 三臂共同起点 | `E/FORMAL_START_CONTROL.json`；COMMON_PUBLICATION_AUDIT；FINAL_CONTROLS.common_initial | **PASS**：三臂及H100共用完整初态；实际共同文件SHA `1a1277e4e29940c1ac43f174be4937b24cdc2eba1b6fbc26e952434721caa96b`，runtime tensor摘要 `d1a8d37ab940c3246b86b19a723ece8cc1322efca5426400f8d154735da8cc09`；全局Full Mixer统一开启。 | common不删除、不重新初始化。不同工具的tensor/SID摘要算法不能直接横比。 |
| P8 baseline同源一次重试 | COMMON_PUBLICATION_AUDIT；baseline01失败/释放收据；baseline02起点与正式完成记录 | **PASS**：第一次在共同initial临时发布处失败，尚无committed训练状态；保留已验共同起点，仅同源重试baseline一次。 | 失败占用计入账本，失败证据保留；后续campaign应先单发布者完成共同起点，再并行启动。 |
| P9 有界启动与短验收预算 | 原/最终allocation ledger；PLATFORM_RELEASE；GPU_ACCOUNTING | **PASS**：原总上限16 GPU小时、验收上限0.5小时；短验收保守占用308.158秒；最终全部占用也在上限内。 | 原预留15.1599小时是启动天花板，不能写成实际使用；最终reservation为0。 |

## 最终15项

下表“PASS”只对应列明的真实证据；F15的结论以外置最终绑定为准。BACKUP_VERIFICATION指 `G/BACKUP_VERIFICATION.json`；FULL_MODEL_RESTORE指 `D/verification/FULL_MODEL_RESTORE.json`；FINAL_CONTROLS指 `R/FINAL_CONTROLS.json`。

| 要求 | 权威证据 | 实际核对状态 | 范围与剩余项 |
|---|---|---|---|
| F1 三臂完整35R/3640R更新/60T更新 | `G/formal/<arm>/TRAIN_COMPLETE.json`、progress与last；FINAL_CONTROLS、FULL_MODEL_RESTORE | **PASS**：三臂均trained_not_tested，45阶段完整；last实际进度均35R/3640R更新/60T。 | 没有以partial、smoke、仅文件名或进程退出替代完整训练验收。 |
| F2 同序、同随机流与更新日程 | FINAL_CONTROLS.paired_controls；`R/summary.json` | **PASS**：all_three_arms=true、common_stages=45；阶段数据顺序、seed、batch/update、LR及T噪声日程配对一致。 | 不要求三组loss相同；不是冻结SID单独训练推荐器。 |
| F3 F真实学习与R/T冻结边界 | FINAL_CONTROLS.F_trajectories；原parameter_audit | **PASS**：baseline F完整45阶段保持不变；三臂各10个T阶段冻结为真；新增臂R1首个优化器更新读出层变化，R2检查出现具名core参数变化。 | 证据是阶段边界哈希及每阶段首个更新的记录，不声称每个参数每步均非零；全F哈希变化不替代core变化证据。 |
| F4 全部24个完整验证节点 | FINAL_CONTROLS.all_24_full_validation_nodes | **PASS**：每臂R0/5/10/15/20/25/30/35，各22363人，val、beam256、batch8、top10。 | smoke32用户和部分节点没有计入最终24节点。 |
| F5 固定R35主比较及含R0选模 | `R/fixed_R35.csv`、`best_validation_weighted_including_R0.csv`；六状态实际progress | **PASS**：主比较固定R35；按封存weighted_score选最大，精确同分取最早，包含R0；best为baseline R15/T24、no_context R30/T54、axial R35/T60。 | 公式合理性容差1e-7；不以重算浮点微差改变选模。固定预算和best模型表现分别解释。 |
| F6 实际SID和统计一致 | FINAL_CONTROLS.checkpoint_saved_buffer_checks；FULL_MODEL_RESTORE | **PASS**：CPU读取六份实际best/last，对齐各自进度、保存SID、变化/碰撞快照及F哈希；12101商品×4位；未refresh或重新分配。 | SID碰撞敏感，不能直接宣称item级收益；不同哈希算法的输出不混比。 |
| F7 正式推理成本与候选覆盖 | `R/summary.json`；[人读报告](../REPORT.md) | **PASS（本轮观测）**：R35全val同步模型计时，baseline/NC/AX合计310.33/286.14/290.66秒；AX对NC+1.58%、对baseline−6.34%；候选有效/重复/不足K齐全。 | p50/p95按batch8，含首batch；不同H100节点及SID搜索变化不能归因为F加速。候选unique是合法SID数，不是不同商品数；fallback未暴露仍为null；残差是phase末次前向，不是验证均值。 |
| F8 test锁定与单seed | protocol/identity、五次任务记录、完整validation；FULL_MODEL_RESTORE限制字段 | **PASS（本Goal记录范围）**：只seed2024、val；未追加seed/test，CPU恢复也未tokenize/evaluate测试集、未算推荐指标。 | 结论限单seed Beauty SID级探索，不声称稳定增益；不与06 item级指标直接横排。 |
| F9 五次分配释放及上限 | `G/delivery/PLATFORM_RELEASE.json`与原证据；组件审计第7项 | **PASS**：smoke01、baseline01、baseline02、no_context01、axial01共5次均释放；完整清单、exact task空usage及终态原证据绑定；无预留。 | 累计保守占用28617.2137秒≤57600，短验收308.1582秒≤1800；未停止其他任务资源。 |
| F10 耗时、费用口径 | `R/accounting/GPU_ACCOUNTING.json/.md`；原ledger、规范化ledger与单价证据 | **PASS（占用统计和估算）**：平台生命周期27919秒＝7.755278 GPU小时；至首次确认释放的保守观测28617.2137秒＝7.949226小时。以保守观测核预算，按展示1点/GPU小时线性估算约7.95点，含失败与短验收。 | **这是两个耗时统计口径，不能当作实际结算费用的上下界；实际账单未核实。** process耗时、reservation与分配占用分开保留。 |
| F11 结果报告、曲线和汇报草稿 | `R/FINAL_REPORT.md`、summary、CSV/PNG、微信草稿、`R/FINAL_REPORT_BINDING.json` | **PASS（已打包结果版本）**：报告和数据绑定通过，固定R35 NDCG10约+1.1%、HR5反向、best baseline更强均保留；符合观测筛线，仅建议跨seed复验小信号。 | 不把筛线视为显著性。微信未发送；最新交接/矩阵属于F15补充包，不改已封原报告绑定。 |
| F12 包内容完整且绑定正确 | `D/package/PACKAGE.json`、`G/MANIFEST.json`；组件审计 | **PASS（基础包）**：11分卷，归档714991805字节；1995个清单成员含源码、输入、共同initial/六状态、原始证据、报告、账本、恢复工具与打包时文档。 | 最新本表和Windows wrapper/说明由supplement补充；基础包AT_PACKAGE矩阵保留历史状态，不回写。 |
| F13 本地完整字节、全新解包和双Git恢复 | `G/BACKUP_VERIFICATION.json` | **PASS**：`STAGE08_FULL_BYTES_AND_MINIMAL_CPU_STATE_RESTORE_PASS`；所有member/input字节及两个Git完整树通过，六份最小状态恢复通过。 | 首次Windows长路径失败目录保留；经子进程longpaths wrapper在全新目录重做成功。这个最小状态收据本身不证明真实网络，另看F14。 |
| F14 六份真实网络/优化器恢复 | `D/verification/FULL_MODEL_RESTORE.json` | **PASS**：`STAGE08_FULL_ACTUAL_MODEL_CPU_RESTORE_PASS`；6个真实模型strict加载、优化器名称/形状映射、scheduler、CPU/Python/NumPy RNG及真实历史有限CPU前向通过；输入/状态未改。 | CUDA身份保留，CUDA RNG字节保存但不GPU回放；没有SID刷新、训练步或测试评估。CPU恢复不等于重算原GPU指标。 |
| F15 最终文档/supplement与交付绑定 | 本表、最新HANDOFF、Windows wrapper；`D/final_delivery_supplement.zip`、`D/FINAL_DELIVERY_BINDING.json`及实际verify结果 | **外置配套验收**：最终binding必须为 `STAGE08_FINAL_LOCAL_DELIVERY_PASS`、final_delivery_complete=true，且真实verify成功，才闭合本项。组件审计中的root closeout保留封装前PENDING快照，不回写旧收据。 | 最终绑定核包/manifest/报告、release/账本、两恢复收据、组件审计、本表、HANDOFF与wrapper实际SHA。本表不包含自身或随后绑定的SHA；以外置收据闭合，避免循环哈希。 |

## 本次真实核对及版本绑定

本次读了真实恢复与组件审计收据，并重算组件审计引用的7个本地文件SHA，全部匹配；没有重新训练、恢复模型或调用平台。组件审计为 `COMPONENTS_PASS_AWAITING_FINAL_BINDING_AND_DOCS`，`goal_complete=false`，本表沿用这一范围。

| 本地证据 | 实际SHA256 |
|---|---|
| `D/verification/FINAL_COMPONENT_AUDIT.json` | `ee1a2322b1974df80548cb16ece9ab8ed660a778ed3b150bcfc881b148818347` |
| `G/BACKUP_VERIFICATION.json` | `1e6178b59a410528554a871fb9ba8f615a9870ef8490081d114a8702dbc71bfa` |
| `D/verification/FULL_MODEL_RESTORE.json` | `75c1297940f15f0f07c8e16dcfd7626551de9338db78a41401a224d2c1d6a233` |
| `D/package/PACKAGE.json` | `f5a01e874247daa7e7c6621472d08c9c920d374209352b9615754498913e736c` |
| 11分卷拼接的归档流 | `188f930eb264eba20d15008fb33a2634214674399f9e9c6adcccef0bec39422a` |
| `G/MANIFEST.json` | `e8bb7378fecf899ad0ddc91e948b5feaf502fe708d8ef5ce7a942085f67506a6` |
| `R/FINAL_REPORT_BINDING.json` | `d64738fa9dad1d6cb1dc932ef3ee0f5025c457d8e3143d202cb1b395a056e6de` |

**最终判定方式：本表说明要求和已核范围；随后生成的外置FINAL_DELIVERY_BINDING及其真实verify结果负责把最终版本闭合。** 外置收据是包后产生的材料，不冒充已存在于基础包；任何新增补件都必须进入supplement并绑定SHA。
