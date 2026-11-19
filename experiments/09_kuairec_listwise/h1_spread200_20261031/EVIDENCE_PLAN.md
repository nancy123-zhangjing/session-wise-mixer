# 最终验收证据清单

本表已按实际产物更新。科学、本地交付恢复和GPU收尾均通过；最后的远端科学补充包同步及CPU停止仍受平台连通性阻塞。本表不是最终Goal通过收据。

| 目标项 | 权威证据及应核验内容 | 当前完成边界 |
|---|---|---|
| 1 协议与保护 | PROTOCOL、PARENT_BINDING、EXECUTION_PACKAGE、实际源码manifest；旧报告与HANDOFF只作背景；原目录无覆盖、旧监控未启动 | 通过：父状态/源码/配置/数据绑定与实际起点已验，CURRENT_SCIENTIFIC_BINDINGS_CHECK重新核268个科学输入SHA一致；旧监控未恢复 |
| 2 唯一D与共享起点 | 父SHA与真实CPU/GPU起点验收；最终24255R/200T、40阶段、20990新增步；R5暖身不重训；H1/Full Mixer/seed/数据绑定 | 通过：final_analysis/ANALYSIS、COLD_RESTORE；原13014行前缀保留、全20990步及终点一致，唯一D |
| 3 精确配对 | CPU_INPUT_ACCEPTANCE、前20T GPU数值重放、ALL_T200_ACCEPTED及全部真实T日志索引SHA；每个原40T边界seed、拆半后的独立T RNG；400T时钟及T LR、35R余弦LR | 通过：全200T/204800次有序出现、全R及双时钟、全部41源状态的跨R paired T RNG/Tokenizer/T优化器不变均已实际验收 |
| 4 参数隔离与恢复 | PREFLIGHT_REPORT、next_T/next_R CUDA精确恢复；preflight_only拒绝；所有边界字段和时钟、冷恢复记录 | 通过：GPU下一T/R精确恢复短测；新解压源的7关键真实CPU模型/双优化器/scheduler恢复及全41边界通过。CPU不应用CUDA RNG，范围明确 |
| 4b 同断点并行调度 | ASYNC_DEPLOYMENT SHA及独立runtime/launcher；真实30R状态/256预测行并行等价；首30次正式续跑前缀验收；最终ASYNC_DELIVERY_AUDIT的11任务checkpoint/完整状态/验证提取绑定与无状态变化 | 通过：独立运维包、全部11只读任务正确cp/完整状态前后不变、四完整验证同源提取、首30真实续跑及最终成功指针均实际验收 |
| 5 预算与唯一资源 | 最终status/instances/events与FINAL_GPU_RESOURCE_AUDIT；累计24GPUh、短测0.5GPUh，失败/启动/释放计入 | 通过：5作业全部终态释放；含每次60秒释放余量上界14.3753GPUh，短测834秒，分配区间无重叠。1点券/实例h为报价依据，实际结算未核 |
| 6 指标与选模 | 10完整验证、六节点selection、固定35及best最多2测试；全部新指标/身份独立复算；2000用户配对bootstrap；复用A/C | 通过：10验证、2测试各53194条独立通过，锁best=R23/T120；固定R35、best补充及固定验证的新比较各2000用户bootstrap完成，原统计按SHA复用 |
| 7a 过程 | 31状态8192固定诊断；20个T前后码本观察引用11快照；碰撞/四槽使用/SID变化/位移/Mixer | 通过：final_process/PROCESS；31状态、11快照和所有独立快照检查实际通过，T前复用经全R隔离证明；未重复推理 |
| 7b 时间泛化 | 固定切点1598745441188500、38644/33013、五训练商品组；联合用户抽样；描述性分解闭合；旧H0历史参照 | 通过：final_temporal/TEMPORAL；两段D均低于A，差值变化未写成泛化改善，分解闭合通过；没有因果或在线适应断言 |
| 8a 中文报告与图表 | 权威final_report_r2/REPORT.md及四组PNG/SVG/PDF，实际查看并核表格图轴 | 通过：四PNG实际查看，最终PNG与所看字节相同，数值表/13输出SHA核验通过；VISUAL_REVIEW_ACCEPTED单独记录，SVG/PDF检查范围明确 |
| 8b 备份与恢复 | 3464源文件主归档、本地SHA/新解压/真实冷恢复、外部依赖；最终工具与科学结果包 | 模型主归档通过：远端封存、本地55块+整包SHA、新目录逐文件SHA和真实恢复全部完成。工具r3与64文件科学补充包本地全SHA通过；补充包远端同步/SHA受网络阻塞待做 |
| 8c 资源收尾 | 全部GPU终态实例释放；临时CPU r1/r2 STOPPED；本轮设施退出；保留远端科学产物 | GPU全部通过。CPU r1此前STOPPED；r2最后成功查询RUNNING，当前平台不可达、stop未确认。还需最终CPU状态/本地设施记录及goal_completion_audit；Goal未完成 |

最终检查还必须逐条核对禁项：没有追加seed/H0训练/其他T预算/MEDA重置；没有测试调参；没有飞书写入/对外消息；没有恢复原监控/覆盖旧证据。结果为负不触发加跑。

脚本的status、绿色检查、目录存在和摘要本身不能替代上述内容核验。所有待做项清零前保持Goal active。
