# 09阶段交接

最新补充测试（2026-10-29）：固定R26的Decay、Freeze与原H1 R20已统一完成测试比较（53194条/4153用户、beam32/FP32）。测试NDCG@5依次R20=0.00794367、Decay=0.00456691、Freeze=0.00915702；Freeze较R20+15.27%，差值95%区间在零以上。原两个r1任务成功结束，无重提/重训；独立复算/本地归档通过，GPU释放，新增0.974167 GPU小时、估0.97点（非结算）。见[测试报告](h1_recovery_test_20261029/REPORT.md)和[独立分析](h1_recovery_test_20261029/analysis_r1/analysis_report.json)。下面保留旧续训时间语义；不再把旧“未测试”视为当前状态。

追加状态（2026-10-29）：H1两臂固定R26实验、17份复算、三份bootstrap、曲线、归档SHA与两臂CPU恢复已通过，最终逐项审计已通过，本轮实验交付完成，见[最终审计](h1_recovery_20261028/FINAL_DELIVERY_AUDIT.md)。Decay验证NDCG@5=0.00615930，Freeze=0.01036453，起点R20=0.00937206；Freeze较起点+10.59%，其差值95%区间在零以上。Freeze保留原H1交互，不是H0；本轮未评估测试集或其他seed。所有本轮GPU已释放，累计6.556667 GPU小时，估6.56点、未核实结算（本轮上限8.5小时，独立于原75小时预算）。接手读[续训报告](h1_recovery_20261028/REPORT.md)、[交付索引](h1_recovery_20261028/DELIVERY_INDEX.json)、[执行记录](h1_recovery_20261028/EXECUTION.md)；禁止重复提交已有任务。新备份恢复仍依赖原09数据/PSE资产，路径和SHA绑定在CPU恢复收据。下文原四组及旧三支建议保留历史含义。

2026-10-28。本轮四组seed2024实验及最终交付已验收。先读[报告](REPORT.md)、[结构化结果](SUMMARY.json)，不要再启动既有提交脚本。

- H0测试NDCG@5=0.008607，H1=0.007944；H1相对-7.71%，单seed用户配对区间在零以下。J1的best是共享R5/0T，不代表训练后的Mixer收益。
- H1验证best为R20，晚期存在真实码字使用塌缩；J1 R35全目录仅1种SID。当前优先问题是编码更新稳定性；没有追加调参。
- 四组全部35R/400T。两份5R暖身各计算一次，总130R/90090R更新/1600T更新；42次完整验证，4次完整测试，8次固定验证子集评估。
- 冻结科学源码main=1c423662d9a25a926328ee6466c30c0f2857b4f6，DiffGRM=98b834cc5ce4428d1a95b3e4f9651833b5fbd5cd，GenRec=b6c8b41d1d20a812590dae11408b69b1ed2a8313。报告/分析收尾源码与训练源码分别保留。
- 实际配置为preparation_20261027/runtime_efficiency_r2.json：微批256×4、eval batch32；base.json的16/4是早期配置，不是正式实测。
- 远端：/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1
- 本地归档：C:/Users/nancy/Documents/experiment1/outputs/stage09_delivery/goal_20261027_r1
- 独立恢复：C:/Users/nancy/Documents/experiment1/outputs/stage09_restore/goal_20261027_r1
- 模型在runs/warmup_J、warmup_H、J0、J1、H0、H1；每份manifest绑定initial/best/last、状态、验证和日志。共享数据沿用outputs/kuairec_listwise_v1/run1，生成资产在恢复根assets/。
- 最终预测在sidecars/evaluations/evaluations/。H1正式采用H1_test_beam32_r2、H1_validation_subset_beam32_r2、H1_validation_subset_beam256_r2；原H1 r1为节点失败证据，不能用于分析。
- 操作侧包在sidecars/operations；四run包不能单独替代源码/共同资产/侧包。完整索引见formal_execution_20261027/FINAL_LOCAL_ASSET_VERIFICATION_20261028.json。
- 已下载归档完整SHA和成员SHA均复核；代表J1/H1已实际CPU加载模型、alignment、优化器/调度/RNG并前向。只用推理权重不能称为完整续训恢复。
- 原训练resume强校验源码、配置和阶段时钟。后续R20诊断须建立新分叉记录，保留原优化器矩和R调度，不伪改旧binding，不覆盖原目录。
- 所有GPU已释放，累计41.198889 GPU小时。按1点/GPU小时估41.20点，未核实结算。CPU Notebook是临时文件入口，启停状态不影响共享盘资产。

下一步仅为建议：R20原设置 / 降Mixer LR / 固定编码三支到R26。以验证NDCG@5、SID变化、碰撞率和残差范数判断；没有授权在本Goal追加此实验。详见[训练设置调研](../../docs/research/onerec_listwise/H1_TRAINING_STABILITY_REVIEW_20261028.md)。
