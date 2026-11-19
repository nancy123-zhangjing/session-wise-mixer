# 四组实验运行状态（2026-11-06）

当前Goal active，四组均已提交并进入真实GPU训练，尚未完成、尚未最终验收。
本文件与DEPLOYMENT.json覆盖封存README/VERIFICATION中提交前的状态。

| 分支 | 平台Job | 配置 |
|---|---|---|
| r26_hold | nancy-s09-stab-r26hold-1106-v1 | IncrementLinear R26起，固定0.0642713567839196到R35/T400 |
| r26_low | nancy-s09-stab-r26low-1106-v1 | 同一R26起，固定0.01到R35/T400 |
| release20_freeze | nancy-s09-stab-rel20freeze-1106-v1 | Release1 R20起，冻结Tokenizer到R35/T200 |
| release20_linear | nancy-s09-stab-rel20linear-1106-v1 | 同一R20起，新增修正0.1→0.01到R35/T400 |

每组1×H100/20CPU/200GiB，shm32GiB，最长12小时，独立Job。
四组都已产生真实优化器更新，证明已通过运行入口的父SHA/状态和GPU起点SID连续性检查。
完整恢复验收仍要等各节点保存后，由job.py的verify阶段执行。

代码9项本地测试通过、两个父文件完整元数据已核验；源码包SHA见DEPLOYMENT.json。
平台状态见PLATFORM_LATEST.json，实际更新/验证结果见PROGRESS_LATEST.json；这些是有时间戳的快照，不代表未来实时状态。

## 后续继续工作

2026-11-06 07:23更新：release20_linear完成R32/T360全量验证，NDCG@5=0.007376757700086224，继续低于R29的0.008591245041209374与R26的0.010221711080860566，当前best仍R26；已进入最后40次T更新，快照为T397。对应freeze的R32/T200验证为0.009130940342432907。freeze继续R34训练（406/693）；两组均RUNNING，最终结果与验收未完成。来源：2026-11-05T23:23:25.792855+00:00进度快照。

2026-11-06 06:58更新：release20_freeze完成R32/T200全量验证，NDCG@5=0.009130940342432907，低于R29的0.009580364673450125与R26的0.010141875317915307，当前best仍R26；已进入R33训练（52/693）。release20_linear的R32全量验证完成27648/71657条，尚无完整R32分数。两组Job仍RUNNING，前两组已通过独立结果验收。来源：2026-11-05T22:58:38.502478+00:00进度快照。

2026-11-06 06:50更新：r26_hold与r26_low均完成独立CPU验收，RESULT.status=passed；两组各80个结果文件均存在且大小匹配，ARTIFACT_MANIFEST的SHA与验收收据一致。独立重算测试指标与原summary在约1e-11范围内一致。平台Job均SUCCEEDED，worker均instance_success，验收CPU进程已退出。本地保留R26_HOLD_ACCEPTED_RESULT.json、R26_LOW_ACCEPTED_RESULT.json、R26_ACCEPTANCE_ARTIFACTS.json、R26_LOW_ACCEPTANCE_ARTIFACTS.json及两份TERMINAL_INSTANCES收据；大文件未下载。剩余release20两组仍运行，整体Goal未完成。

2026-11-06 06:45更新：r26_hold/r26_low平台均SUCCEEDED，train/test/verify三个阶段exit_code均为0，最佳点与R35测试全部完成。R35测试NDCG@5分别为0.0069714102814371175与0.008471508559958534；最佳点测试分别为0.008901687583150365（R26）与0.009136392992277963（R32）。06:45已在共享CPU启动r26_hold独立验收，PID=1906738，日志acceptance/r26_hold.log；r26_low独立验收尚未启动。其余两组仍RUNNING；四组整体Goal未完成。依据PROGRESS_LATEST/PLATFORM_LATEST的06:45快照及ACCEPT_R26_HOLD_LAUNCH.json。

2026-11-06 06:17更新：两组R26分支的最佳点测试均完成53194条全量样本。r26_hold选择起点R26，测试NDCG@5=0.008901687583150365；r26_low选择R32，测试NDCG@5=0.009136392992277963（相对高2.6366%）。原始远端summary及路径已记录于本地小文件BEST_TEST_INTERIM.json。两组已进入R35测试；该结果尚待独立逐样本验收，不代表整个实验已验收。release20_freeze正在R32训练，release20_linear正在R31训练；四个Job仍RUNNING。

2026-11-06 05:48:47更新：release20_linear完成R29/T320全量验证，NDCG@5=0.008591245041209374，低于同轮freeze的0.009580364673450125，当前best仍R26（0.010221711080860566）。已继续T更新至T350；freeze进入R31（91/693）。r26_hold/r26_low测试日志均显示已处理3008/53194条，测试尚未完成。四组均RUNNING，最终验收待完成。来源：2026-11-05T21:48:47.104260+00:00进度快照。

2026-11-06 05:46:48更新：r26_hold的R35验证NDCG@5=0.008194274174023684，验证best为起点R26（0.009973123477893297）。r26_hold与r26_low均已生成training_validation_complete收据，累计R更新24255/T400，测试日志已有模型初始化输出，但暂无完成的测试指标。r26_low的R35比hold高20.1648%。两组last SHA分别为9095f26205ee4f2ae6e306c7e270310f7f1d98fe1caab6b4b8bd97150398800f和7c65fd375012d333ad2e6b3524f8fe2adb027c5d24693f3fc27eefb02ff7afc6。四组仍RUNNING，尚未最终验收。

2026-11-06 05:46更新：r26_low完成R35/T400全量验证，NDCG@5=0.00984663314934258；该组验证best确定为R32（0.010166548781938825），起点R26为0.009973123477893297。训练累计R更新24255、T更新400；测试及恢复/最终验收尚未完成，Job仍RUNNING。其余分支尚无新完整节点。来源：2026-11-05T21:46:14.216343+00:00进度快照。

2026-11-06 05:30更新：release20_freeze完成R29/T200全量验证，NDCG@5=0.009580364673450125，低于R26的0.010141875317915307，当前best仍为R26；已进入R30训练。r26_hold/r26_low均完成R35/T400训练更新，正在R35全量验证；release20_linear正在R29/T320全量验证。四组均RUNNING，尚无测试结果或最终验收。来源：2026-11-05T21:29:59.968459+00:00进度快照。

最近完整验证快照：2026-11-06 04:13:34（北京时间），四组平台状态均为RUNNING，测试和最终验收尚未完成。

| 分支 | 起点验证 NDCG@5 | 首个验证节点 | 第二个验证节点 |
|---|---:|---:|---:|
| r26_hold | R26: 0.009973123478 | R29: 0.008998192639 | R32: 0.008343258284 |
| r26_low | R26: 0.009973123478 | R29: 0.009567240088 | R32: 0.010166548782 |
| release20_freeze | R20: 0.008905992734 | R23: 0.009821451265 | R26: 0.010141875318 |
| release20_linear | R20: 0.008905992734 | R23: 0.009690363321 | R26: 0.010221711081 |

同起点配对比较：r26_low在R32比r26_hold高21.85%，比共同R26起点高1.94%；release20_linear在R26比freeze高0.79%。这是单seed中途验证结果，尚不能代替最终测试结论。两对分支起点和当前R轮次不同，不用这四个当前分数直接评选最佳方案。来源：同时间戳PROGRESS_LATEST.json及MONITOR_EVENTS.jsonl。

1. 使用现有四个作业名查询，禁止因观测超时或网络中断直接重提。
2. 只读collect_status.py获取日志、验证、测试和恢复收据；新完整验证点或异常时向用户汇报。
3. 关注平台GPU<40%持续3小时回收规则；初始3-4个采样含启动空闲，不能据此判断长期利用率。
4. 每组训练/测试/恢复完成后，CPU串行运行accept_arm.py --arm <arm>，独立复算预测和标签、核对更新/样本顺序/系数/源码/checkpoint，生成acceptance/<arm>/RESULT.json及ARTIFACT_MANIFEST.json。
5. 核对每个Job终态及GPU释放，整理四组best/R35结果和历史参照；所有要求通过才完成Goal。
6. checkpoint、完整预测、大日志均留远端；本地只取代码、指标摘要和小型收据。不下载大包，不删除唯一副本。

共享根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_stability_followup_20261106_v1`。
已有CPU文件入口env-zzl不属于本任务，不停止它。


## 2026-11-06 08:26 live update

release20_freeze R35 full validation completed: NDCG@5=0.009257274032968324. Validation-selected best remains R26 (0.010141875317915307). Test stage has started; acceptance pending. release20_linear R35 validation: 16384/71657. Both jobs RUNNING. Large files remain remote.


## 2026-11-06 08:58 live update

release20_linear R35 validation completed: NDCG@5=0.007966989138592617. Best remains R26 (0.010221711080860566); test stage started. release20_freeze best R26 test full 53194 examples completed: NDCG@5=0.008810520432918875, independent audit pending; R35 test in progress. Both jobs RUNNING.


## 2026-11-06 09:28 acceptance update

release20_freeze independent acceptance passed; 95 artifact entries, manifest SHA matches receipt and fresh remote existence/size check passed. CPU audit process exited; GPU job SUCCEEDED, worker instance_success. Best R26 test=0.008810520432918875; R35 test=0.008123407834742972. release20_linear best R26 test=0.009138335972989632 (full 53194 samples, independent acceptance pending), R35 test now running. Three arms accepted; last arm still pending.


## Final closure 2026-11-06
All four arms accepted; jobs and workers terminal success. See FINAL_REPORT.md and FINAL_CLOSURE.json. Large artifacts retained remote; shared env-zzl preserved.
