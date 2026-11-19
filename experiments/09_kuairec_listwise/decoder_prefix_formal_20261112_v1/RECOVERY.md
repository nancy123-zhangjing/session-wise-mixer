# 2026-11-13 评估恢复

四个原作业均因 12 小时上限终止。四个 last.pt 已在 CPU 成功加载并记录 SHA，均为 r_updates=13860、t_updates=200。A 组游标 phase=25/update=0；B 组 phase=24/update=693，原训练循环将跳过全部更新，仅重新完成 R20 验证。

recover_evaluation.py inspect 核对平台终态、超时事件、冻结源码及断点；submit 在重新校验 SHA 后归档旧锁和日志，复用未改动的 experiment.py 提交独立恢复作业。未完成的评估使用原入口的 retry 目录，不把部分预测当作完整指标。恢复提交入口不是可自动重复执行的后台重试器；存在同名作业时停止供人工核对。

RECOVERY_JOBS.json 为本地监控的恢复作业映射；原提交和终止收据保留。远端训练源码、模型、随机行为和训练日程均未更改。恢复成功提交不代表评估完成，最终验收仍需完整指标、checkpoint 核验和资源释放。
