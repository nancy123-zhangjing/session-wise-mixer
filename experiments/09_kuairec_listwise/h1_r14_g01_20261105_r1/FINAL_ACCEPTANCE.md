# 最终验收记录

- 父R14摘要与71657条逐样本预测完全一致：R14_EVIDENCE.json。
- 模型、双优化器、时钟、CUDA RNG恢复及初始有效编码/SID恒等：GPU_GATE.json。
- 原45相位/35R/400T调度不缩短，实际新增4158R/80T，结束13860R/200T：acceptance/RESULT.json，accept_results.py逐步核对。
- 两块T和后续R诊断、有效码本位移、SID变动：RESULT.json中的catalog及acceptance/DIAGNOSTICS.json。R14前指标来自原预测同子集抽取，不重跑推理。
- R17/R20完整验证、固定R20唯一完整测试，指标逐样本独立重算并核对源标签：acceptance/*_audit.json。
- R17与R20完整checkpoint均已CUDA往返恢复一致：RESULT.json中的checkpoint_restore。
- 冻结源码SHA不变、结果文件大小和SHA清单：SOURCE_MANIFEST.json及ARTIFACT_MANIFEST.json。
- 平台作业nancy-s09-r14-g01-1105-r1状态SUCCEEDED，finished_at=1793881859000；实例instance_success。未配置keep-after-success，任务进程全部正常结束，无本任务新增CPU资源；共享env-zzl与其他作业保持不动。
- 大结果留远端共享盘；本地只取回摘要与清单。未删除结果文件或唯一副本。
- 结论及边界：REPORT.md。结果负面不影响实验执行验收。

远端根：/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_r14_g01_20261105_r1。
报告同步与本地下载文件SHA核对由最终同步收据记录。
