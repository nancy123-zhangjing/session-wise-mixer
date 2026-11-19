# E单卡实验：备份与恢复入口

本次唯一正式模型为5×(40T＋6R)、R35/T200。用户已用单卡方案取代八卡；不要根据旧设计文档重启八卡，不重复训练或解码。实际验收结果以各JSON为准，本说明不是通过收据。

## 完整模型包

- 本地：本目录delivery/stage09_h1_block40_single_r1.tar.gz。
- 远端：/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_block40_r6_single_20261101_r1/delivery/下同名文件。
- 封包收据：delivery/ARCHIVE_RECEIPT.json；内嵌逐文件清单：DELIVERY_MANIFEST.json。
- 包应为3144064546字节，SHA256为c0f8e459843fc8856a41d63d486dc38e78be2de7f57d371d99d314056a111bc7。
- 内容包括冻结源码、配置、全部R/T边界、优化器/scheduler/RNG、预测、日志、诊断和运维绑定；数据与原父状态为下列已有外部依赖。
- 断点续传的最终证明为delivery/RESUMED_DOWNLOAD_ACCEPTED.json；仅文件存在或部分分块通过不代表全包通过。

## 外部依赖（保留原资产，不重复打包）

所有本地相对路径均相对C:/Users/nancy/Documents/experiment1：

| 内容 | 路径 |
|---|---|
| 原H1 R5/T0父状态 | outputs/stage09_restore/goal_20261027_r1/runs/H1/initial.pt |
| 既定数据 | outputs/kuairec_listwise_v1/run1 |
| 固定PSE资产 | outputs/stage09_restore/goal_20261027_r1/assets/pse_v1 |
| 已有D归档解压 | outputs/stage09_spread200_20261031_fresh_r1 |
| A/C及H0已验结果 | experiments/09_kuairec_listwise/h_freeze35_20261029/final_analysis |
| 真实CPU恢复环境 | .runtime/stage08_restore_env/Scripts/python.exe |

父状态SHA256为991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1。其余数据、资产和代码SHA由冻结E_EXECUTION_MANIFEST及各checkpoint执行绑定核验，不能只靠目录名匹配。

## 已安排的独立恢复范围

archive_delivery.py extract在新的outputs/stage09_block40_single_20261101_fresh_r1目录独立解压，核整包、内嵌清单及所有文件SHA；输出LOCAL_EXTRACTION_VERIFIED.json。

cold_restore.py在新CPU进程中仅从这份独立解压源码导入，实际加载关键checkpoint的模型、Tokenizer、双优化器、scheduler和进度，检查36个边界及R阶段Tokenizer/T优化器/T随机流不变。输出COLD_RESTORE.json。CPU检查不执行训练/推理，不应用CUDA RNG；原CUDA状态字节仍保存在认证checkpoint中，不能把CPU恢复称为新增GPU续步测试。

正常分析使用Python310，冷恢复使用上面的torch2.6 CPU环境。详细调用参数见ANALYSIS_README.md。已有通过的统计与收据应复用；除非确有损坏需要恢复，不再次运行全部分析。

## 分析、飞书与资源交付

science_delivery_r1.tar.gz为轻量补充包，包含新E分析/图表/报告、恢复与资源证据、分析代码、飞书可编辑XML及回读证据；不替代完整模型包。SCIENCE_DELIVERY_LOCAL_VERIFIED.json核包内全部成员，SCIENCE_DELIVERY_REMOTE_VERIFIED.json核上传后的远端SHA。

最后的FINAL_DELIVERY_CLOSURE.json应同时绑定科学验收、完整包/独立恢复、补充包远端校验、GPU释放、临时CPU停止、heartbeat关闭及飞书回读。没有这份最终收据时不宣称Goal已完成。
