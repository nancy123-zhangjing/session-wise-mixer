# 2026-10-16：搜索阶段数值故障

本记录属于06连续向量基线，与历史04、05实验无关。Goal尚未完成。

最终实验状态（2026-10-16）：统一修复后的89项有效训练、13项测试独立重算、10份正式导出与最终测速已完成；最后GPU Job已成功并释放。以下12:57及更早文字均为故障处理时的历史快照，不能据此判断当前队列。旧数值版本36项已启动运行和旧调度队列46项已启动运行的目录、best/last、receipt及CURRENT目标仍存在，见[只读保留检查](execution_acceptance/resource_accounting_20261016/OLD_ATTEMPTS_RETENTION_20261016_02.json)；该检查没有重新加载或完整备份所有旧权重。最终结论与未完成交付项见[报告](REPORT.md)。

12:57更新：原故障任务`search__esasrec_17__seed2024`已在统一修复版本下按原配置完整完成53轮/27,242次更新；最佳验证点为第3轮，随后按patience50早停。逐轮末尾记录的最大梯度范数6.6774，最终训练损失0.5373；这些范数不是所有更新步的完整采样。best/last完整模型、优化器、scheduler和CPU随机状态加载接受检查通过，CUDA随机状态字节态存在。[正式搜索完成收据](execution_acceptance/checkpoint_payload_precision_v2/CRITICAL_CONFIG17_COMPLETE.json)、[完整状态审计](execution_acceptance/checkpoint_payload_precision_v2/LOCAL_RECEIPT_AUDIT_05.json)。新队列目前34项技术完成，无失败；测试仍锁定。原配置数值问题的验证通过不代替89项全部完成或推荐质量结论。

11:35更新：单卡精度校准、514步重放及35轮从头压力检查已通过；CUDA注意力FP32 Math候选修复已在本地实现，本地73项CPU gate零跳过。尚需同一提交版本的服务器CPU验收及增强GPU gate，未重启8卡。原28项技术完成保留为旧版本证据，不进入新版本选模；新版本须从头完成原清单89项，旧作业和诊断消耗另列，不追加搜索配置。不能由此断言28项全部存在同一数值错误。

## 当前状态

8H100 Job `nancy-stage06-20261016-8h100-01` 于11:08:39业务退出，平台FAILED；11:09:23已删除Pod并释放8卡。队列为 `STOPPED_FOR_AUDIT`：28项技术完成、1项失败、7项被调度器中断、28项待启动；尚未展开后续25项。28项完成结果的数值有效性待审，不能计为已验收的89项成果。Top-3、正式配置及最终模型均未选定，测试仍锁定。

失败任务 `search__esasrec_17__seed2024`：512维、1层、2头、dropout=0.1、LR=0.001、weight_decay=1e-5、batch=256、负样本256；配置不作修改。执行commit `4d7c4662c077ca3aaa9d78363e1ee09d8bb34432`，源码SHA256 `7bff9f19bfef7ef8ec4ee9b058a6ff6cf7baeea3f469c5dc3f87c1cf25031ccc`。

第27轮最后梯度范数已约4.07e16，第28轮在 `clip_grad_norm_(error_if_nonfinite=True)` 停止。保护检查保留，不跳过失败，不将此数值故障当作基础设施重试。

## 已确认的诊断

保留第27轮完整checkpoint、优化器及随机状态，SHA256 `31f9ebaf43457e9fd44eae8c09a5a5eceafc582aa8a702066cd0d84dbfa33f9d`。模型与优化器张量均有限；CPU FP32和BF16短诊断均未复现，不据此排除CUDA路径问题。

独立1H100 Job `nancy-stage06-grad-diag-20261016-01`，使用原源码、原配置和同一checkpoint重放第28轮，不写训练checkpoint，不计正式实验：

| CUDA路径 | 完成更新 | 观察 |
|---|---:|---|
| 原BF16、默认注意力 | 147/514 | 第148步复现；loss=5.6673，单个梯度元素全部有限，FP32总范数inf，FP64总范数2.84e19 |
| BF16、Math注意力 | 514/514 | 未触发异常，但首步范数约2549，仍明显偏离FP32参考 |
| FP32、Math注意力 | 514/514 | 未触发异常，首步范数约1.76，最大约80.31 |

该Job平台SUCCEEDED，11:19:05已删除Pod。诊断结果指向精度/注意力计算路径，尚不足以认定具体PyTorch bug；不同后端的训练dropout随机数消耗也可能不同，不能把逐步梯度差全归因于同一因素。

## 修复门槛

下一步在同数据、固定负例、关闭dropout的条件下比较梯度与FP32参考，再用原失败配置进行超过原故障轮次的有界稳定性检查。只有验证后才修改正式实现；不降低LR/容量，不关闭非有限检查，不盲目重提8卡。

如统一精度策略改变源码，不能把旧版本28项结果或7项中断checkpoint混入新版本89项；须记录失效范围、重跑范围及额外GPU消耗，使用新目录、新版本，重新通过CPU和GPU gate。

原主作业业务时长3055.983秒，8卡业务用量6.7911 GPU小时，按当次8点/节点小时折算约6.7911点；这是进程内估算，不冒充平台账单。诊断及重跑费用另列。

轻量证据：[故障包清单](execution_acceptance/failure_20261016_01/manifest.json)、[CPU诊断](execution_acceptance/failure_20261016_01/CPU_DIAGNOSTIC.json)、[GPU诊断](execution_acceptance/failure_20261016_01/gpu_gradient_diagnostic/DIAGNOSTIC_COMPLETE.json)。全部原始checkpoint保留在独立远端执行目录；本地完整故障元数据在 `archive/server_sync/stage06_continuous_20261016/failure_evidence_01/`，故障包SHA256为 `e5d7ecc08ee1742357cd82796a8cd07e6f2f1dc4b46069211df5fbfa24e62096`。

第二轮独立1H100 Job `nancy-stage06-precision-diag-20261016-02`：固定权重、数据和负例，关闭dropout比较3批梯度。原BF16默认路径相对梯度误差约1e12；FP32 Math注意力、外层BF16的最大相对L2误差约0.00119，最小cosine约0.99999945。算子追踪确认原训练选用efficient attention，但不据此认定具体库bug。

12:04资源核对：原8卡Job从平台Scheduled到Pod删除共3,122秒，约6.9378 GPU小时；两次单卡诊断分别89秒、248秒。三项合计约7.0314 GPU小时，按各自单价折算约7.0314点券。这与前述旧主进程内6.7911 GPU小时的口径不同，包含启动/释放窗口与诊断。平台结算点券仍为未核实，不将公共项目余额差归因于本Goal。[原始事件及独立核对](execution_acceptance/resource_accounting_20261016/OLD_ATTEMPTS_ALLOCATION_AUDIT.json)。

同一策略完成第28轮514步重放和seed2024从头35轮（17,990次更新），每轮131,413个目标、原dropout0.1，最大梯度范数6.6774。没有验证选模、测试、训练checkpoint写入；不是新增正式训练或调参机会。业务诊断耗时186.86秒，平台11:34:59删除Pod并释放。原始数据和脚本见[精度诊断](execution_acceptance/failure_20261016_01/gpu_precision_diagnostic/DIAGNOSTIC_COMPLETE.json)、[独立复核](execution_acceptance/failure_20261016_01/gpu_precision_diagnostic/AUDIT.json)。
