# H1 R20→R26 最终交付逐项审计

审计时间：2026-10-28T19:32:27.921984+00:00。

**结论：PASS（本轮实验与交付要求均有现存证据支持，无实质缺口）。**

审计依据为本目录 `GOAL_OBJECTIVE.md` 全文（包括后续累计8.5 GPU小时批准）及 `docs/research/onerec_listwise/H1_LATE_TRAINING_RECOVERY_PLAN_20261028.md`。结果提交为 `39624bd361b712574e8a332b0cf386dd8a4677e0`。本次只读交叉复核，未训练、未申请GPU、未查询远端、未重跑测试/指标/模型恢复或解包；只新增本文件。

执行了 8763 项元数据、实际日志、来源字节与收据交叉检查。对完整归档/冷CPU模型/独立复算/图表视觉验收分别引用其实际已通过收据，并检查当前关键来源SHA；不把收据范围扩大为GPU重演或结算审计。

## 逐项要求

| 要求 | 结论 | 实际证据和限定 |
|---|---|---|
| 任务范围与依据 | PASS | GOAL_OBJECTIVE.md及锁定H1方案；仅H1、seed2024、两恢复策略和固定R26。旧固定LR曲线仅为历史参照；不以结果是否正向作为完成条件。 |
| 源码锁定、历史一致性与Git | PASS | PACKAGE_R2、code/RECOVERY_EXECUTION_MANIFEST.json、ACTUAL_CONFIG：8b5acc8恢复扩展，40个原绑定文件与历史源字节一致。实际执行由混合历史底座及明确扩展的manifest定义，不由当前HEAD单独定义。结果已提交39624bd；原draft/generalization目录及逐样本大文件保持未跟踪，不被强制清理。 |
| 共同R20完整分叉 | PASS | 两份fork.json与r2 H100 initial：父SHA a672ea4d…890bf1；model/alignment/双Adam动量/R scheduler共同指纹相同。Decay只按方案改变T参数组LR，不能要求整个T optimizer（含LR字段）与Freeze完全相同。初始8192商品预测匹配原R20，见INITIAL_EVALUATION_AUDIT。 |
| Decay真实更新与LR | PASS | 实际training.jsonl共4238行＝80T＋4158R。逐项检查T201…280，实际使用u200…279及1:10码本/Mixer LR；首次5e-6/5e-5，末次2.0929340784e-6/2.0929340784e-5，下一步T280计划值2.0610737385e-6/2.0610737385e-5。 |
| Freeze真实冻结 | PASS | 实际仅4158条R，无T optimizer更新；T计数始终200。最终CPU恢复确认alignment和T optimizer完整指纹仍与R20相同；推荐器model/R optimizer已更新。固定SID不变是定义，不作为推荐改善证明。 |
| 阶段、R scheduler、有效batch | PASS | 两臂R21…R26均693更新、709443条列表/轮，原phase26/27/28/30/31/32保留；累计R18018。每条实际R LR沿原35R/500 warmup公式核对，未重新warmup。microbatch256、有效1024，末批835按原归一化。 |
| 继承配置与损失/噪声 | PASS | ACTUAL_CONFIG与实际fork/H100源码绑定相同：H、5件、历史50、E1/D4、hidden256/4头/FFN1024、4×256×64码本、Full Mixer残差1、BF16训练、FP32推理；其他损失/噪声沿原设置。R只读hard_codes，T时钟不直接改变R Gumbel或掩码深度，见INTERPRETATION_AUDIT。 |
| CPU验收与修复过程 | PASS | 原CPU_TESTS.log保留32项；其后修复optimizer快照别名的13项另存日志和PROVENANCE。已实际核原journal第59276行record SHA及日志SHA、exit0、13项、特定alias回归名。旧CPU_REAL_PARENT两文件只作为修复前历史CPU恢复，不作r2证据。 |
| 真实H100短验收与随机恢复 | PASS | preflight_r2/preflight_r1/SUMMARY及两报告SHA一致；Decay T/R、Freeze R真实BF16梯度及冻结边界、disk exact_loaded_state/exact_next_update、FP32输出重复性通过。验收更新丢弃，正式从原R20重新分叉。首轮失败如实保存并计费；不重跑已过测试。 |
| 数据次序、种子、无副作用测量 | PASS | 锁定recovery_run按原phase_id设置seed_all并调用原phase_indices，跳T不压缩R阶段。既有CPU诊断异常/嵌套/真实下一随机更新测试及H100 replay支持RNG、模式、缓存、梯度保全；未声称再次逐batch独立复演GPU。 |
| 完整验证协议 | PASS | 实际两臂R23/R26各71657条、4733用户；beam32、最多32候选、batch32、FP32、相同商品级定义。五件主列表与聚合@10/20/50明确分列；保持原消歧/去重/PAD及不滤历史。无新增beam256/test推理/seed/H0/J/R35分支。 |
| 17份预测独立复算 | PASS | analysis_report＋17个audit及当前raw/recomputed/summary SHA逐项一致：本轮16份＋原R20全量，共456589条、0 mismatches，最大绝对误差6.7209896204e-8。固定R26主终点；R23只是过程，8192为固定SHA子集；完整预测的子集只提取、不重复解码。 |
| 三组用户聚类bootstrap | PASS | 三份comparisons实际存在并与SUMMARY同值：2000次、seed2024、95%分位区间、按用户有放回携带全部样本；primary为重抽样user sums/user sample counts得到样本均值，非用户等权。另列user-average仅辅助，未重新运行bootstrap。 |
| 编码诊断、实际LR与图表 | PASS | Decay initial/T210…T280九状态，Freeze initial一次；全目录10728商品、4×256码位的使用数/熵/SID/碰撞/变化/位移/重构/修正范数保留；临时副本诊断及正式40T刷新边界不改。4PNG＋4PDF当前SHA/文件类型与FIGURE_MANIFEST一致，plot源文件SHA亦一致；既有视觉检查见RESULT_TABLES。 |
| 边界状态和日志 | PASS | LOCAL_DELIVERY清单和当前文件/manifest匹配：两臂各initial、last和8个T/R阶段边界；Freeze的两个跳T边界亦保存，不代表T被执行。formal_jobs、fork、complete、training/evaluations/catalog均在3155文件清单内。 |
| 本地备份实际验收 | PASS | 复用真实verify_local_delivery的全26块、整体SHA、3155文件/3156成员、安全新目录解包及落盘哈希PASS；本次核其verifier/receipt/内外manifest当前SHA和归档大小，不重新解包或再次跑完整验收。1722453940字节；archive607f9124…f6c83cd；manifestf6dee18b…9efbaec。 |
| 依赖可恢复性 | PASS | 新包明确非数据/PSE自包含；包内references/R20_best.pt匹配父SHA。dataset=outputs/kuairec_listwise_v1/run1，assets=outputs/stage09_restore/goal_20261027_r1/assets/pse_v1；旧R20预测另行依赖。DELIVERY_INDEX明确data逐文件绑定/asset manifest，冷CPU实际加载证明与原执行一致。 |
| 最终真实CPU恢复 | PASS | CPU_FINAL_DECAY/FREEZE均由归档冷导入实际模型、双Adam、scheduler、cursor；当前父/last/complete文件SHA与收据匹配，状态往返除显式CUDA RNG不执行外相同，原身份未重写。终点R18018、T280/200、phase33，R LR0.00016066336338464165。此项不是CUDA精确续训或指标重演。 |
| 预算、费用与四Job释放 | PASS | FINAL_RESOURCE_ACCOUNTING及全部原JSON SHA保持一致：两个预验收含失败410 GPU秒；正式23194秒；总23604秒=6.556666667 GPUh，8.5h预算余6996秒=1.943333333h。并行墙钟3:18:56。四Job终态与SuccessfulDeletePod/JobTerminated齐；1点/GPUh估6.56点，非结算单。CPU入口与排队不计入GPU。 |
| 报告、正负结果与结论边界 | PASS | REPORT/SUMMARY/METRICS当前字节与DELIVERY_INDEX匹配；Freeze R26 NDCG5=.0103645345，Decay=.0061593003，R20=.0093720557。三组CI和辅助覆盖/缺项/历史曲线均列出；明确当前Decay设置未阻退化、Freeze保住并提高本次验证表现，不能推及长期/测试/H0胜负，也不单独归因SID漂移。 |
| 交接和最后结项操作 | PASS | 已存在RESULT commit39624bd及项目/09交接入口，报告含下一步建议与恢复依赖。HANDOFF/EXECUTION“最后审计”与SUMMARY.goal_closure是等待本审计的状态/静态要求，非科学工作缺失；coordinator在本PASS后仅追加审计和交接状态提交，并在Goal工具中结项。 |

## 固定终点的主要结论

| 比较（NDCG@5） | 绝对差 | 95%用户聚类区间 |
|---|---:|---|
| decay_R26_minus_freeze_R26 | -0.004205234191 | [-0.004552471814, -0.003850607136] |
| decay_R26_minus_H1_R20 | -0.003212755422 | [-0.003597892410, -0.002851117854] |
| freeze_R26_minus_H1_R20 | +0.000992478768 | [+0.000641782252, +0.001319805170] |

区间仅反映这些固定模型在验证用户上的抽样不确定性，不包含训练seed、历史R20选模和本次事后恢复策略提案的不确定性。不以冻结SID不变证明效果；新两组是整体策略对照，尚未单独分离码本几何、SID切换、T正则/随机优化与后续R适应的贡献。

## 关键来源SHA256（审计时字节）

| 文件 | SHA256 |
|---|---|
| GOAL_OBJECTIVE.md | `711e85205b2982bcb3eea80d80ede5c1c0c47e4e27db524ca62f934276c3d423` |
| REPORT.md | `41337553aceca47563d97b29c659391544b1d4cb23b8f0c26c3cca41c1f61c66` |
| SUMMARY.json | `baf0dd5859c8f1ad23c0a17d9cf8c5e941869714d7fc6dc56a1abcceb9159a7e` |
| ACTUAL_CONFIG.json | `917cb32ae293a0f17f342543412563355ff860e40e04ba134b38923e5e719d3b` |
| DELIVERY_INDEX.json | `b0e5e41ef91be715ecb8436e3a082624b31ad82c7fb1a1eec2248108366e00b8` |
| LOCAL_DELIVERY_AUDIT.json | `6f030e1c4dbebcb0017806f6e91367d7e6a933dcf875e3e703182a3151e7ff30` |
| CPU_FINAL_DECAY.json | `daa4dd7d78e3f4bca5fa8b1fe4aaa1e7718601c39017090eaf2e213196044a8b` |
| CPU_FINAL_FREEZE.json | `e2978f5d00b11d2d8da5b28b02958d5187a75ed3b834590ba9eb58acbcbfe39e` |
| CPU_TESTS.log | `c0df092aa8e10fb3630859fcd27c2c5083ff86e31e3604cfbe0ba029ab1f89bc` |
| CPU_RECOVERY_R2_TESTS.log | `86a82abdb60172224fbd3087ed3c4c59769e7e8b446d4c77b71acacf2a19f7b4` |
| CPU_RECOVERY_R2_TESTS_PROVENANCE.json | `1dbb45314ecfb8a3b435146219702c72ded8a5da4664831d873d558c66d8d127` |
| FINAL_RESOURCE_ACCOUNTING.json | `8c6a19a60f33a15a80724607806174b15ee16ff558efb37954a90f1ed41cac20` |
| final_analysis/analysis_report.json | `61cae3b9b4fa000f0a0ce7903fcbdbb584ff7ac1aac9d565e026b10295469d60` |
| final_analysis/RESULT_TABLES.json | `fc3dcf3c8eaaf69bc28c04db9b7f0b08ec2f6441d6bcf74690bb31f73632c688` |
| figures/FIGURE_MANIFEST.json | `b6f0150f6ffb57926ec5a49654c6a95f524eb6f5895e606f07301d711fa624cd` |

源码扩展commit：`8b5acc8853101975a0ba09fe904f990549643dcc`；执行manifest：`5dec44adb958cc3f9c0d65e811986c2f40097f45ad9ef066daf24e4c654d797f`；父checkpoint：`a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1`。

## 协调者结项边界

本文件PASS不等于已经调用Codex Goal complete。协调者将本审计及最终HANDOFF/EXECUTION状态文字提交后，再记录Goal工具完成状态。现有SUMMARY/REPORT/DELIVERY_INDEX保持原字节；SUMMARY.goal_closure只是静态交付要求。此处不构造包含自身SHA的循环绑定。

所有旧报告、模型、数据和失败验收记录保留。用户草稿及generalization目录未因本审计清理；大模型/数据/逐样本压缩产物留在本地备份而不强行入Git。
