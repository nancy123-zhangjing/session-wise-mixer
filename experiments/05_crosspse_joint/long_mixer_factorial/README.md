# 共享起点、带噪Full Mixer长期四组：交付入口

**训练、四次正式测试、图表、完整计算审计和七个GPU Job释放均已完成。完整归档已在本地落盘并通过全量字节、源码Git恢复及与远端八个真实CPU模型恢复的同包绑定。** 当前交付见[FINAL_DELIVERY_AUDIT.json](FINAL_DELIVERY_AUDIT.json)；旧执行时点保留在[EXECUTION_NOTES.md](EXECUTION_NOTES.md)。

本实验研究从统一起点训练新增Full Mixer的整体作用，包含码本内部和码本之间交互。四组共用05训练前Aug28完整离线推荐器，以及已完成短四组的有效码本E₀；新Mixer输出投影为0、残差系数固定1，保证起始有效编码/SID一致。两个Frozen全程不更新Tokenizer，Mixer残差一直为0，因此是功能等价负对照；实际35轮轨迹、最佳/最终模型与test均相同。

两个Joint均先5R，再10×(96个Tokenizer batch/6次更新＋3R)，合计35R/3640次推荐器更新、60T/1920次实际加噪调用；Frozen合计35R/0T。Gumbel跨累计60次T更新余弦1.0→0.2，推理和SID刷新确定性。每臂独立单H100，并行最多四臂，没有改变批量或优化算法。完整参数、起点、控制和证据边界见研究报告。

## 主要观察

| 比较 | HR@10差值 | NDCG@10差值 | weighted差值 |
|---|---:|---:|---:|
| 固定R35主DID：`(MJ−MF)−(NJ−NF)` | −0.000223583598 | +0.000141105655 | +0.000068167822 |
| 各自验证选定best的test：`MJ−NJ`，描述性比较 | −0.000447167196 | +0.000571608096 | +0.000367852998 |

第二行两个模型分别是R15/T24与R10/T12，不是固定预算DID。Full终态改变303件商品SID，无Mixer Joint改变18件，但部分排序指标增益与HR@10负向差值并存；更多改码不等于整体推荐更好。结论限于seed2024、该共同预训练起点和当前协议；不分离within/between，不排除额外参数化和优化路径解释。

## 结果与证据

- [完整研究报告：参数、结果、控制、图表、解释与交付边界](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/REPORT.md)
- [独立完整计算AUDIT：四臂训练、八个实际权重、全部正式测试](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_audit_r1/AUDIT.json)
- [固定R35原始CSV](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_audit_r1/fixed_epoch35_validation.csv)、[各自best正式test原始CSV](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_audit_r1/best_selected_test.csv)、[图文件与来源SHA](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_figures_r1/FIGURES.json)
- [独立测试覆盖审计：四段2796/2796完整批次、收据与日志对应](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/TEST_COVERAGE_AUDIT.json)
- [最终资源释放与预算审计](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/RESOURCE_RELEASE_FINAL.json)
- [恢复指南：完整归档、源码Git、模型状态与证据边界](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/RESTORE_GUIDE.md)
- [已核验的远程恢复收据](backup_evidence_remote_r1/REMOTE_RESTORE_EVIDENCE_AUDIT.json)、[历史短四组与噪声A/B/C背景](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/PRIOR_EVIDENCE_CONTEXT.md)

七个GPU Job从Scheduled到SuccessfulDeletePod累计35549 GPU秒，即9.874722222小时，包含验收、失败重试、正式训练/验证/测试和保留时间；低于12 GPU小时上限。按quota单价1点/GPU小时估算9.874722222点，实际发票未核对。

## 版本与完整备份状态

训练根仓库为`69682698c15ba796b98e6329c950061a2f68a05b`，引擎为`1c879252f6395d6e64462511823c3929b9c42c00`；独立分析/交付为V3R2、`4b40c90b56490868e5972e75bafbf8e50ca8a3de`。分析工具没有覆盖训练执行版本。

远程`delivery/final_package_r1`归档大小为1,536,215,400字节，SHA256为`c2ad35b0c87b219e6435a970dfd62bf5677cf7e7c2924b34c53e48c6fec47af7`。PACKAGE/MANIFEST记载1871个数据文件，另含MANIFEST.json自身。远程已核对全部成员字节/14件输入，并从该备份实际重建Pipeline、恢复四臂best/last的模型、Adam、scheduler、计数、CPU RNG、CUDA RNG字节、SID与E₀ anchor；CPU恢复没有执行CUDA。

<!-- stage05-long-delivery-verified -->

完整备份目录为`outputs/stage05_long/package_r1/`，已校验解压在`outputs/stage05_long/extract_r1/goal_20261020_long_mixer_v1/`，实际Git恢复在`outputs/stage05_long/git_restore_r1/`。本地完整备份和同包模型恢复绑定见[FINAL_BACKUP_BINDING_AUDIT.json](FINAL_BACKUP_BINDING_AUDIT.json)，逐项总验收见[FINAL_DELIVERY_AUDIT.json](FINAL_DELIVERY_AUDIT.json)。
