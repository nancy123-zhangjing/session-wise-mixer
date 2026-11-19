# D实验备份与恢复操作说明

当前为待执行说明，不能作为备份或恢复通过证据。正式产物只有在训练、全部验证及限定测试成功结束后才封存；最终实际路径和SHA以交付收据为准。下列本地目录为预定的新目录，已经存在时必须换新的后缀，不能覆盖或删除旧目录。

## 保留的外部依赖

| 依赖 | 当前本地路径 | 核验依据 |
|---|---|---|
| 原完整R5/T0父状态 | `outputs/stage09_restore/goal_20261027_r1/runs/H1/initial.pt` | SHA256 `991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1` |
| KuaiRec切分、目录、标签与训练seen集合 | `outputs/kuairec_listwise_v1/run1` | 封存INPUTS、执行manifest及各预测summary.dataset_binding的逐文件SHA |
| PSE资产 | `outputs/stage09_restore/goal_20261027_r1/assets/pse_v1` | 封存INPUTS及执行绑定的资产SHA |
| 旧R5固定诊断预测 | `outputs/stage09_restore/goal_20261027_r1/runs/warmup_H/validation_R05/per_sample.jsonl.gz` | 新run请求中的warmup_predictions_sha256；只复用 |
| A/C旧验收与预测、原H1验证、时间分析缓存 | 原保存位置，见analyze_d.py/temporal_d.py/report_d.py中的受SHA约束引用 | 用于比较和报告，不是运行D模型所需的新增训练数据；不得重新推理旧模型 |

归档包含本轮独立源码、各阶段完整状态、日志和新预测；不会重复打包上述既存大型依赖。远端依赖实际路径在归档INPUTS.json和DELIVERY_MANIFEST.json中。跨机器迁移必须逐SHA核验依赖，不能仅替换路径后默认内容相同。

GPU短测实际torch为2.6.0+cu118，硬件H100 80GB。作业镜像和共享Python路径见DEPLOYMENT.json。当前本地验收环境为`.runtime/stage08_restore_env/Scripts/python.exe`（torch2.6.0+cpu）。已实际采集ENVIRONMENT_LOCAL.json和ENVIRONMENT_SHARED_INTERPRETER.json：本地Python3.12.14、共享Python3.10.20，numpy2.2.6/pyarrow25.0.0一致；共享包元数据从CPU入口读取，不能当作GPU设备检查。两份清单均绑定collect_environment.py的SHA。包的版本与容器标签不能替代已通过的数值和恢复证据。

## 成功终态后封存与独立解压

远端仅本轮root：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_spread200_20261031_r1`。当前同断点恢复由独立`jobs/formal_async_r2`执行，成功后`FORMAL_COMPLETION.json`绑定该尝试的result和resume_gate。已SHA绑定的归档入口为root外层`archive_delivery_async_r2.py`，不能写入已绑定code目录，也不能误用只支持旧尝试的远端旧工具。远端使用共享Python运行`archive_delivery_async_r2.py seal --root <上述root>`。工具拒绝失败或未完成的科学终态，且拒绝覆盖已有交付包。资源释放另查平台，封存成功不等于GPU已释放。

下载delivery中的stage09_h1_spread200_r1.tar.gz和ARCHIVE_RECEIPT.json到新建本地delivery目录。下载之前核对实际压缩包与解压体积、C盘余量；不得清理旧工作区来腾空间。工具会核整包SHA、安全路径、每文件大小与SHA，并要求解压空间余量。

以下PowerShell命令在`C:/Users/nancy/Documents/experiment1`运行。它们是最终科学完成后的操作入口；当前不会提前运行：

```powershell
$d = 'experiments/09_kuairec_listwise/h1_spread200_20261031'
$p = '.runtime/stage08_restore_env/Scripts/python.exe'
& $p -X utf8 -B "$d/archive_delivery.py" extract --archive "$d/delivery/stage09_h1_spread200_r1.tar.gz" --receipt "$d/delivery/ARCHIVE_RECEIPT.json" --destination 'outputs/stage09_spread200_20261031_fresh_r1'
& $p -X utf8 -B "$d/cold_restore.py" --extracted 'outputs/stage09_spread200_20261031_fresh_r1' --parent 'outputs/stage09_restore/goal_20261027_r1/runs/H1/initial.pt' --dataset 'outputs/kuairec_listwise_v1/run1' --assets 'outputs/stage09_restore/goal_20261027_r1/assets/pse_v1' --output "$d/COLD_RESTORE.json"
```

冷恢复从新解压目录实际导入封存源码和engine，真实构建CPU模型并恢复关键checkpoint及双优化器、scheduler、进度；另检查所有41边界。它不训练、不推理；保留并核验原CUDA随机状态字节，但CPU检查不应用CUDA RNG。不能据此声称CPU与CUDA训练轨迹等价；CUDA下一T/R精确恢复的证据来自已完成的GPU短测。

## 最终CPU验收顺序

只有完整科学产物及解压、冷恢复证据就绪后，依次运行下面命令。完整D预测需独立指标复算；旧A/C已接受统计按SHA复用。输出目录已存在时先调查，不能删除重跑以掩盖失败。

```powershell
$run = 'outputs/stage09_spread200_20261031_fresh_r1/runs/H1-Spread200-R35'
& $p -X utf8 -B "$d/audit_async_delivery.py" --extracted 'outputs/stage09_spread200_20261031_fresh_r1' --output "$d/ASYNC_DELIVERY_AUDIT.json"
& $p -X utf8 -B "$d/analyze_d.py" --run $run --output "$d/final_analysis"
& $p -X utf8 -B "$d/temporal_d.py" --analysis "$d/final_analysis" --output "$d/final_temporal"
& $p -X utf8 -B "$d/process_d.py" --run $run --cold-restore "$d/COLD_RESTORE.json" --output "$d/final_process"
& $p -X utf8 -B "$d/report_d.py" --analysis "$d/final_analysis" --temporal "$d/final_temporal" --process "$d/final_process" --operational-audit "$d/ASYNC_DELIVERY_AUDIT.json" --output "$d/final_report_r2"
```

报告自动生成后仍需查看四组图并核对定量解释、表格与坐标；最后按EVIDENCE_PLAN逐项审查科学、归档恢复及资源收尾。须另保存实际执行的最新版验收工具和全部最终分析/报告的SHA清单；早期acceptance_tools_r1.tar.gz不能冒充最终工具包。平台GPU终态与实例释放、临时CPU r2停止、本轮实际时长和费用证据必须单独记录。任何一步缺证都保持Goal未完成。

新调度额外验收必须证明：独立运维包SHA及冻结科学源码各自真实绑定；剩余11个只读任务全部对应指定阶段checkpoint及其完整状态指纹，前后状态完全一致；四份完整验证与诊断提取的预测源相同；真实恢复首30步固定前缀验收保留；最终train/test退出成功及完成指针一致。`audit_async_delivery.py`已准备并通过语法检查，尚未在最终产物上运行；不能将脚本存在当成通过收据。科学指标、完整20990条新增更新、41个边界和冷恢复仍由各自独立工具完整验收。

## 本次实际验收与传输恢复补充

以上命令是可复查入口。2026-10-31已实际完成新目录3464文件核验、COLD_RESTORE、ASYNC_DELIVERY_AUDIT、ANALYSIS、TEMPORAL、PROCESS。权威中文报告是final_report_r2/REPORT.md；final_report为保留的初版。不要在已有输出目录上重复执行以覆盖证据。四图实际查看和最终PNG字节比对由final_report_r2/VISUAL_REVIEW_ACCEPTED.json单独记录；REPORT_BINDING中的pending只描述生成时阶段。

首次整包下载受临时CPU入口运行周期中断影响。原完整远端归档没有改动，本地旧前缀保留.partial_r1。transfer_archive_chunks.py按64MiB分55块，源分块时重验整包SHA；前7块复用及其余块下载均逐SHA通过，最后组装整包SHA与封存值一致。小型传输证据见delivery/chunks_r1中的MANIFEST、PREFIX_REUSE_ACCEPTED、DOWNLOAD_SHA_RECORDS及ASSEMBLED_ACCEPTED。CPU内存曾接近100%，但平台事件未直接给出OOM原因，不将其写成已确认因果。

FINAL_GPU_RESOURCE_AUDIT.json覆盖全部5个GPU尝试和启动/释放余量，实际通过14.3753 GPU小时保守上界、短测834秒及无重叠。费用按live 1点券/实例h估算，实际结算未核。CPU最终停止和完整Goal验收在最后单独生成，不以科学结果负面或报告生成代替资源收尾。
