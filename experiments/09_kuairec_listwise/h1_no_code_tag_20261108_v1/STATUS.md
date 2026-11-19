# H1 去中心编号标记执行状态

最终状态：R20/T200、R14 最佳点测试、六节点 checkpoint 恢复和独立指标验收均已完成。GPU 作业成功终止，共享 CPU 入口保留。最终对照见 `FINAL_REPORT.md`，文件清单与回读见 `ARTIFACT_MANIFEST.json`、`FINAL_CLOSURE.json` 和 `DELIVERY_READBACK.json`。以下为执行过程记录。

作业 `nancy-s09-nocodetag-1108-v1` 已在启智单张 H100 上运行。最新已核对日志为 R6 第 48 次推荐器更新，累计 R 更新 3513、T 更新 40。平台节点为 `qb-prod-gpu055`。本地 `PLATFORM_LATEST.json` 和 `PROGRESS_LATEST.json` 是查询快照。

2026-11-09 00:34 的最后一次有效查询显示作业 RUNNING。R8 已完成全部 693 次更新，累计 R 更新 5544、T 更新 40；R8 完整验证已处理 26016/71657 条。随后直连返回 ENOTFOUND，现有本地代理返回 ECONNRESET；本地观察中断，作业终态与 R8 分数待重查。保持同一个作业名。

父状态来自原 H1 R5/T0。训练日志已记录 `parent_restored`，表示提交的程序已通过父哈希、科学源码差异、配置绑定、模型/优化器/scheduler/RNG 恢复及初始码本与 SID 一致性检查。

远端根为 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_no_code_tag_20261108_v1`。每个节点保存于 `runs/H1-NoCodeTag/checkpoints/`，监控入口 `python control.py status`。验收源码已部署，`AUDIT_SOURCE_CHECK.json` 记录了 SHA 回读。

后续依次核对 R8/R11/R14/R17/R20、锁定 best 与单次完整测试、六节点严格恢复和 `GPU_COMPLETE.json`。平台确认 SUCCEEDED 后执行 `python control.py start_audit`，通过状态入口观察 `audit.log`，`ACCEPTED_RESULT.json` 通过后运行 `python finalize.py`。最终报告和清单回读一致、单 worker 成功终止后才完成 Goal。

唯一提交意图已保存。观测失败时读取同一作业；如作业明确失败，先检查最近完整 checkpoint 和中断位置，再决定是否从该状态恢复。共享 CPU 入口 env-zzl 保留。

启智访问连续三个 Goal 回合复查失败。直连 DNS 无法解析 `<institution-host>`，本地代理返回 `ECONNRESET`。作业实际状态仍待确认。`BASELINE_REFERENCE.json` 已保存原 H1 的同协议对照摘要和 SHA256。等待恢复启智访问后从同一作业继续。
