# 离线验收精度修正

本目录只修正交付审计，不修改训练代码、权重、SID或测试结果。原r2工具和失败收据保留。

原检查把CPU FP64最近邻及`1e-6×(1+距离)`作为必要条件，但生产距离计算在GPU上完成。四份真实checkpoint各有1、4、3、4个码位与严格CPU FP32/FP64最近邻不同，共193,616个码位；不能用普通FP32误差直接解释，也没有扩大旧容差。

独立CPU诊断将点积两侧输入舍入至TF32的10位小数精度，范数仍使用原始FP32。最近偶数和最近远离零两种舍入变体都逐码位复现四份完整SID，零不匹配。新审计要求四份checkpoint存在同一种全量精确匹配方式，禁止逐位置或逐checkpoint拼接不同精度；原FP64差异统计仍写入结果。损坏SID及混配方式的拒绝测试通过，本地和实际远端CPU环境各运行6项检查。

这证明保存的SID与明确的CPU数值复演一致，**不等于已追溯历史CUDA精度开关，也不声称CPU逐比特复现CUDA内核**。两组依然按照同一冻结训练代码及保存SID评估。

真实成对审计已通过：[审计收据](../JOINT_PAIR_AUDIT_20261020_r3.json)。原失败及诊断分别为[原失败](../JOINT_PAIR_AUDIT_20261020_2024.json)、[FP32/FP64差异](../SID_PRECISION_DIAGNOSIS_20261020_2030.json)、[全量TF32复演](../SID_TF32_DIAGNOSIS_20261020_2034.json)。本目录[版本清单](MANIFEST.json)绑定脚本，[远端测试收据](REMOTE_CPU_TESTS.json)仅代表工具测试。

远端审计入口为`<goal>/delivery/postprocess_tools_r3/scripts/analysis/audit_item_local_joint_pair.py`，替代[r2运行说明](../postprocess_tools_r2/POSTPROCESS_RUNBOOK.md)第3步的旧入口。其他打包、字节恢复、14份权重回读和曲线工具不变。正式审计不能替代这些尚需独立完成的交付步骤。
