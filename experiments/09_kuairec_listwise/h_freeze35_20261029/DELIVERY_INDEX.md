# 第09阶段冻结续训交付索引

- [主实验完整报告](REPORT.md)：固定R35、共同网格best、自身起点、真实训练/验证曲线。
- [冷启动与热门偏置完整报告](cold_popularity_audit_20261030/FINAL_REPORT.md)：五模型前后段与逐目标贡献、频次和集中度，包含公平C R32与C best R23。
- [最终科学验收](FINAL_ACCEPTANCE.json)、[验收说明](FINAL_DELIVERY_AUDIT.md)、[机器可读索引](DELIVERY_INDEX.json)。
- [66状态覆盖表](final_cpu_restore_r1/CHECKPOINT_COVERAGE.csv)、[9文件实际恢复](final_cpu_restore_r1/FULL_MODEL_RESTORE.json)、[恢复独立复审](final_cpu_restore_r1/REVIEW.md)。
- [C与A−C第二实现统计复核](final_independent_review_r1/REPORT.md)、[五模型诊断复核](cold_popularity_audit_20261030/VERIFY_C_SUPPLEMENT.json)、[诊断报告审核](cold_popularity_audit_20261030/final_report_review_r1/REVIEW.md)。

运行备份在 `outputs/stage09_restore/h_freeze35_20261029_final_r1.tar`，6,846,443,520字节，SHA256 `9006d8d835ed735bd79bfe382ae362ebc6266ba9c1522bf9a20e6b0dcb399f16`。新鲜解压目录为同级 `h_freeze35_20261029_final_r1_restored/`，3,405份payload逐SHA与关键恢复通过。它需要已有数据/PSE及历史预测，不是自包含包；具体外部依赖及绑定见JSON索引和原归档内DELIVERY_MANIFEST。

恢复用法保留在 [BACKUP_USAGE.md](BACKUP_USAGE.md)，使用原CPU环境及冻结版本；不要从当前开发HEAD替换原训练源码。科学扩展提交是 `e4ca4b32920c95ef7190ad475d18e6f31fa3d276`，最终文档/统计Git提交是另外的交付版本。

最终分析、图表、恢复收据和中文报告生成于运行包之后，另存后处理归档及Git bundle。实际路径、文件SHA、内容提交、独立解压/克隆校验与收尾提交在 `FINAL_DELIVERY_CLOSURE.json` 中记录；该闭环收据本身另行提交，不能自称包含在它所哈希的归档中。

本轮四项已授权GPU分配共21.810556 GPU小时，全部释放；费用仅按历史单价估算约21.81点，结算未核实。原0 GPU CPU文件入口的关闭状态单列于生命周期证据，不混入GPU耗时。
