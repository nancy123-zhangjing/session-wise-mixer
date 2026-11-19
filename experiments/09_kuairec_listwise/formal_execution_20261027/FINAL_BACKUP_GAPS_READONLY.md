# Stage09 最终备份缺口清单（2026-10-28，只读核对）

本清单不是备份完成证明。仅查阅本地源码、实际部署/启动记录、已有备份清单与恢复收据；没有访问远端、读取测试数据或重新计算大型归档 SHA。

固定远端根：`G=/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1`。以下 `G/` 均指此精确路径，不能扩展至其他实验目录。

## 已有交付，可引用而不重复搬迁

- `outputs/stage09_delivery/goal_20261027_r1/stage09_run_warmup_J_r1.tar`、`stage09_run_warmup_H_r1.tar`，对应 `STAGE09_RUN_warmup_J_R1.json`、`STAGE09_RUN_warmup_H_R1.json`。每包 10 文件，已有独立本地解包复验收据 `experiments/09_kuairec_listwise/formal_execution_20261027/{J,H}_WARMUP_LOCAL_RESTORE.json`，恢复位置为 `outputs/stage09_restore/goal_20261027_r1/runs/warmup_{J,H}`。
- `stage09_generated_assets_r1.tar` 与 `experiments/09_kuairec_listwise/assets_preparation_20261027/GENERATED_ASSETS_BACKUP_MANIFEST.json`、`GENERATED_ASSETS_LOCAL_RESTORE.json`：18 个共同资产文件已恢复到同一 restore 根下 `assets/`，包含 BGE dense 向量与完整 PSE/OPQ/PCA 资产。
- `main.bundle`、`diffgrm.bundle`、`genrec.bundle` 已恢复到同一 restore 根下 `code/`；`LOCAL_SOURCE_RESTORE_ACCEPTANCE.json` 核实三个冻结 HEAD 和 40 个绑定源文件。冻结训练源码主版本为 `1c423662d9a25a926328ee6466c30c0f2857b4f6`。新分析/包装脚本须另记版本，不能替换这个执行源码身份。
- 原数据继续引用本地 `outputs/kuairec_listwise_v1/run1/` 及原 manifest；不再复制 train/validation/test parquet 或原始 KuaiRec。BGE 模型也不重复打包。

## 尚需完成：四个正式 run 包

每个 `G/runs/{J0,J1,H0,H1}` 只有在 `complete.json` 存在且 `.running` 不存在、平台终态/资源释放另行核实后，才运行既有 `backup_completed_run.py pack --run <ARM>`。该 helper 只递归打包指定 run 的普通文件，既有包不覆盖；下载后用其 `restore` 做新目录独立验收。

预期输出固定为 `G/delivery/stage09_run_<ARM>_r1.tar` 和外置 `G/delivery/STAGE09_RUN_<ARM>_R1.json`。必须核对：`initial.pt/best.pt/last.pt`、`binding.json/resolved_config.json/complete.json`、`training.jsonl/attempts.jsonl`，以及实际生成的每个 `validation_Rxx/{summary.json,per_sample.jsonl.gz}`；正常四分支各有 R08、11、14、17、20、23、26、29、32、35 的全验证。重试产物若存在也保存，但不能把未完成验证当作 accepted 结果。

SID 并非缺失的外部文件：`scripts/listwise_diffgrm/training.py:276` 的 checkpoint 保存完整 `alignment.state_dict()`；`external/DiffGRM/genrec/models/DIFF_GRM/joint_cross_pse.py:146` 把 `hard_codes` 注册为持久 buffer。initial/best/last 各自保存对应 SID、模型、两优化器、scheduler、游标与 RNG。最终 CPU 恢复要实读并验 SID 形状/范围/指纹；不必为了备份重新拟合或额外导出一套未绑定 SID。

## 最小补充包 A：运行外围文件和来源索引

这些在 run 目录以外，现有单 run helper 不会包含。建议待写入停止后按明确 allowlist 收集普通文件，逐文件 SHA、外置 manifest、封包前后复核、下载后新目录解包复验。不要直接递归整个 G/delivery（那里已有大型归档）。

实际执行记录确认的远端路径：

- `G/config/efficiency_r2.json`，SHA `d8e3a40f9c7ac446dbca6f4e8b01811ee4598c0d4d07e18310f524afda9bd2bb`。
- `G/delivery/FORMAL_WARMUP_BINDING.json`、`FORMAL_BRANCH_BINDING.json`，以及实际执行版本的 `formal_warmup_precheck.py`、`formal_branch_acceptance.py`、`formal_four_arm_job.py`、`backup_completed_run.py`。最终评估 helper 与实际填好占位符的 job 配置同样收录其真实路径和 SHA；模板本身不能替代实际提交记录。
- `G/delivery/four_arm_supervisor_r1/` 下的普通文件：`started.json`、最终 `supervisor.json`、四个 `ready_<ARM>.json`、`gpu_inventory.json`、`pair_J.json/pair_H.json`、四个 `train_<ARM>.log` 及实际生成的 stderr 文件。最终 supervisor 尚须等作业终态，不可用启动记录冒充。
- `G/logs/warmup_j_20261027_r1.log`、`warmup_h_20261027_r1.log`、`four_arm_training_20261027_r1.log`。发生新 attempt 时只加入真实提交记录指向的新日志，不猜目录。

本地已有文件直接纳入最终小型 evidence 包/索引，无须从远端重复下载：

- `experiments/09_kuairec_listwise/preparation_20261027/DEPLOYMENT.json`、`SOURCE_LOCK.json`、`runtime_efficiency_r2.json`、`validation_hash1024.json`；`outputs/stage09_delivery/goal_20261027_r1/INPUT_TRANSFER_MANIFEST.json`、`BGE_TRANSFER_MANIFEST.json`；`experiments/09_kuairec_listwise/assets_preparation_20261027/BGE_M3_DOWNLOAD_VERIFIED.json`。
- 固定 1024 子集已在冻结源码 `code/experiments/09_kuairec_listwise/preparation_20261027/validation_hash1024.json` 中，SHA `c74343799024ea0ca0bdb806ab482f362ff227880a2676cf1541a3511a7942af`。保存索引引用即可，不再抽样。
- 本地数据目录中的 `manifest.json/config.json/construction.json/reproducibility.json/split_statistics.json/validation.json/trace_validation.json` 及 item mapping、train_seen 的现有 SHA 引用；完整输入 payload 的大小/SHA 已由 INPUT_TRANSFER_MANIFEST 绑定。恢复时仍须验现存数据 hash，不能只保留无验证的路径。
- 来源注意：INPUT_TRANSFER_MANIFEST 的 root 是旧 project-public 地址；DEPLOYMENT.json 已明确 `historical_root_role` 并绑定当前 global_user 地址。两者共同保留，不修改历史清单；最终总索引以当前 G 为实际执行根。
- `formal_execution_20261027/` 内实际的 warmup/formal submit/dryrun/status/events/metrics、start gates、initial-pair、GPU inventory、wrapper SHA 和全部恢复收据；资源诊断 R1 的 FAILED 原状态保留，R2 passed 不覆盖它。

## 最小补充包 B：最终评估原始结果

以下是现有 `FINAL_EVAL_JOB_TEMPLATES.json` 明确计划的路径，目前不能视作已存在或已完成。所有四 best 先同时冻结后，各 ARM 执行：

- `G/evaluations/<ARM>_test_beam32/`
- `G/evaluations/<ARM>_validation_subset_beam32/`
- `G/evaluations/<ARM>_validation_subset_beam256/`

四臂合计 12 个目录，每个必须保存 `summary.json` 和 `per_sample.jsonl.gz`；仅下载汇总指标不足以独立复算。保存实际 all-four-best 冻结 manifest（模板中的路径仍是占位符，最终按实际生成路径登记）、每次评估前后 checkpoint SHA、实际评估提交参数/平台终态和 `G/logs/final_eval_{j0,j1,h0,h1}_20261027_r1.log`。这些文件不在 run 包内。

## 最终本地总索引仍需补齐

- 引用既有 2 warmup、共同资产、3 bundle 的 archive/manifest SHA 和恢复收据；加入 4 run 包、上述 A/B 包各自的独立恢复证明。总索引避免再把已打包 tar 嵌入新 tar。
- 对恢复的 J1/H1 做冻结源码下 CPU 实际前向与状态加载验收；字节还原 PASS 不等于模型已恢复运行。
- 12 份逐样本结果独立复算、两种架构内 J1-J0/H1-H0 比较、训练/验证曲线、解释/局限、下一步、交接与最终分析 commit。原始曲线由 run 内 training.jsonl 和 validation summaries 覆盖；派生图表/分析脚本另绑定来源。
- 更新 `preparation_20261027/GPU_USAGE_LEDGER.json`，连同 `BUDGET_APPROVAL.json`、`h100_preflight_r2/APPROVED_RESOURCE_PLAN.json` 及所有实际 Job 的 scheduled/释放证据保存。最终费用区分实际结算与按实例时长×单价估算，不把 75 GPUh 预算当实耗；所有 GPU 释放后才可闭合账本。

排除：重复 KuaiRec parquet/原始事件、BGE 大模型/cache、.runtime/共享 Conda、已归档生成资产/源码的重复副本、05/06/07/08 实验产物、任何其他用户/实验目录。不能打包活动中 `.running` run，不能删除失败证据来制造完成状态。本次仅提出以上最小缺口，不宣称 A/B 包、四分支备份或最终验收已经完成。
