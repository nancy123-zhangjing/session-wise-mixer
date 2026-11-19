# D分散200T：科学结果已验收，平台收尾待完成

> 2026-11-01最终更新：本文件标题及下方“平台收尾待完成/当前阻塞”内容为历史快照。远端补充包和两收据已同步且三份SHA一致；两CPU均Live确认STOPPED，全部GPU已释放。最终8项验收实际通过，268科学输入及冻结工具证据重新核对通过。权威完成收据为[FINAL_DELIVERY_CLOSURE.json](FINAL_DELIVERY_CLOSURE.json)，逐项表为[FINAL_GOAL_ACCEPTANCE.md](FINAL_GOAL_ACCEPTANCE.md)。本轮未重跑实验、推理或统计，负结果与原报告保持一致。

本文件不能作为Goal完成收据。唯一未解决的执行条件是平台连通性；仍需远端同步最终科学结果包、确认临时CPU停止，然后执行最终逐项验收。不得为负结果追加实验。

## 已验证研究结论

| 比较 | A冻结对照 | D分散更新 | D−A及用户配对95%区间 |
|---|---:|---:|---|
| 固定R35/T200测试NDCG@5 | 0.00895634 | 0.00552839 | -0.00342796 [-0.00388395,-0.00297584] |
| 各自best测试NDCG@5 | R32/T200：0.00943079 | R23/T120：0.00807444 | -0.00135636 [-0.00185479,-0.00089362] |

主比较下降38.27%。best的R/T预算不同，仅作补充。一个seed、事后设计，2000次用户bootstrap只描述固定模型的样本不确定性。

D在固定时间切片前后段都低于A。后段减前段的D−A绝对差为+0.00142026，但两个时段的差都为负；这不能解释成时间泛化改善。没有新增按时间到达的训练数据。

最后20T使60.84%商品的SID变化；最后三R结束的固定8192诊断仍低于该块之前。不同SID数9513→9643、碰撞14.03%→12.41%，但推荐指标下降。后期R学习率较低、适应编码变化不足是与过程一致的猜想，并未被单独检验；不能据此证明Tokenizer过拟合或MEDA/one-epoch机制。

权威中文报告：[final_report_r2/REPORT.md](final_report_r2/REPORT.md)。其中已给出主指标、聚合@10/20/50、候选填充率、best、编码稳定性和时间分析；四张PNG实际查看通过，SVG/PDF为同图导出，检查范围见VISUAL_REVIEW_ACCEPTED.json。初版final_report保留。

## 已完成证据

- 10个完整验证、2份各53194条测试；独立指标、样本身份、共同网格R23选模及2000用户配对统计均通过。
- 原R5完整父状态、全部20990新增更新、200个配对T batch/204800次出现、原400T噪声和35R日程通过；没有追加seed/H0/T预算/MEDA重置。
- 31固定诊断状态、11个catalog快照、四槽使用、位移和Mixer校验通过；时间和五频次分组保持原定义，H0仅历史参照。
- 远端模型归档3464文件，4.12GB源内容；本地完整3.64GB包、55块SHA及全新目录逐文件SHA通过。归档SHA与路径见delivery/ARCHIVE_RECEIPT.json，外部数据/PSE/旧对照依赖见RECOVERY.md。
- COLD_RESTORE：7个关键真实CPU模型/双优化器/scheduler恢复通过，全41状态和跨R的T RNG隔离通过。CPU不应用CUDA RNG；原CUDA字节保留，CUDA下一步精确恢复已在短测通过。
- 全部5个GPU尝试终态且实例释放；包含启动及每次60秒释放余量的累计上界14.3753 GPUh，短测834秒。标价1点券/实例h为估算依据，实际结算未核实。
- 最终离线工具r3：TOOLS_MANIFEST_R3.json和acceptance_tools_r3.tar.gz。科学补充包delivery/science_delivery_r1.tar.gz已在本地逐成员核验，尚未远端同步。

## 当前阻塞与继续操作

平台API连续ECONNRESET，缓存CPU通道也不可达；本地代理经公共站点与平台探测未找到可用SII路径。只进行了只读诊断，未修改全局网络设置。已向用户异步请求恢复网络或提供可用接入方式。

GPU无需恢复。CPU r2最后一次成功查询为RUNNING，当前状态无法核验，未确认关闭。旧CPU r1此前已STOPPED，但最终仍应重新取得两者状态。不要把查询失败或过去的RUNNING当成当前平台事实。

网络恢复后：

1. 先查现有CPU nancy-s09-d-prep-1031-r2，不能重复创建/启动训练。
2. 用该入口上传delivery/science_delivery_r1.tar.gz、SCIENCE_DELIVERY_MANIFEST_R1.json、SCIENCE_DELIVERY_RECEIPT_R1.json到既有远端ROOT/delivery；remote sha256sum与本地核对后保存delivery/SCIENCE_DELIVERY_REMOTE_VERIFIED.json，status=passed_remote_sha_matches_local，字段archive_sha256与本地一致。
3. 关闭r2并查询两CPU的STOPPED，分别保存closure_cpu_r1/nancy-s09-d-prep-1031-r1_status.json和r2_status.json（原CLI success/data结构）；保存停止事件/生命周期。不要删除远端科学产物。
4. 核对本轮本地工作进程/临时设施均退出，保存LOCAL_WORK_PROCESSES_CLOSED.json，其中matching_owned_process_count=0；关闭范围限本轮，不动其他工作。
5. 运行goal_completion_audit.py，要求八项、所有科学输入SHA、最新工具、备份与资源均通过，才生成FINAL_DELIVERY_CLOSURE.json和FINAL_GOAL_ACCEPTANCE.md，然后update_goal complete。

远端ROOT：/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1。模型主归档已封存，不要重复seal或覆盖。解压目录outputs/stage09_spread200_20261031_fresh_r1已实际通过，禁止删掉重跑。所有统计/图表已完成，不重复训练、推理或统计。
