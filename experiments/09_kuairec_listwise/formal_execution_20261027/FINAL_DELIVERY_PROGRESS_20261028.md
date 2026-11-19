# 09收尾状态与交接入口

2026-10-28：实验、评估、备份恢复及文档验收完成，所有GPU已释放。后续方案未执行。

- [最终报告](../REPORT.md)、[结构化结果](../SUMMARY.json)、[交接](../HANDOFF.md)。
- 四组35R/400T、42份完整验证、4次完整测试、8次固定验证子集评估完成；H1采用r2最终输出，r1节点失败证据保留。
- 12份最终预测和8个best/R35角色的完整验证已独立复算，6份用户配对bootstrap完成；曲线已绘制并检查。
- 6份run包（2暖身、4分支）、共同资产包、operations/evaluations两侧包、3份源码bundle已下载并独立恢复。代表J1/H1实际CPU恢复及前向通过。
- [最终本地资产核验](FINAL_LOCAL_ASSET_VERIFICATION_20261028.json)核对373个文件记录，包括整包、恢复成员、现有数据、checkpoint、源码和预测绑定。
- [资源账单](ALL_GPU_RELEASE_ACCOUNTING_20261028.json)：13个Job、148316 GPU秒、41.198889小时；全部释放。1点/GPU小时估算41.20点，未核实结算。
- [侧包r2](backup_final_sidecars_r2.py)已处理H1 r2路径，原helper保留历史；不能重新提交既有GPU任务或覆盖已生成的独占输出。
- 最新训练设置诊断见[H1检查与文献](../../../docs/research/onerec_listwise/H1_TRAINING_STABILITY_REVIEW_20261028.md)。验证best的正向点估计不代表测试提升；后续仅建议R20短分叉诊断。

完整路径、源码版本、模型SHA和恢复范围以HANDOFF及资产核验为准，不依据旧观察器状态重启作业。
