# 07 同一验证集的 beam32 / beam256 诊断

用户授权：固定已经训练完成的两组模型，只评估搜索宽度的影响，不重新训练。

- A：交替联合训练 control；B：同配置加商品内注意力。各自使用按原验证规则选出的第 25 轮 best checkpoint 及其保存的 SID，不重新量化。
- 四格：A/32、A/256、B/32、B/256；同一 Beauty 验证集 22,363 用户、同一顺序、历史长度 50、batch 8、confidence 解码、输出 Top-10。
- 原冻结源码：root `842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1`、engine `1c879252f6395d6e64462511823c3929b9c42c00`。新评估脚本单独保存并绑定 SHA256；不改旧目录。
- A best SHA256：`55960f52906b0c4972a90b0ad06c078ef12fceac10fa4400b113e41d9a5c1f5f`；B：`f093c22e4e0cb354870b467e41f4ebab96d90047daf9d0e3d33991e3108b348b`。
- A SID SHA256：`a5483ad7d06a0076e035e5746f5c5d85ecc92eac511fa05e87ab91c257c1d0c1`；B：`075bf705f54deaba794324193de95e6d7f0ae3f985166b39652f90d15859b6e2`。
- 报告 HR/NDCG@5/10、合法率、重复率、不重复合法候选数、逐用户命中/排名、同步 GPU 后计时。沿用原 SID 级指标，不与 item-level 检索指标混算。
- 验收：四格完整、验证样本一致、权重/SID 不变、无训练步骤和测试集评估；检查 beam32 是否复现已有验证结果，若不一致先解释差异再归因。
- 判断：比较同一验证集上的 B−A 在 beam32 和 beam256 下是否明显缩小，以及候选覆盖变化是否对应；这只诊断搜索宽度影响，不能独自证明泛化收益或原 val/test 差距完全由 beam 造成。
- 本轮资源：单节点 1 张 H100；平台最长 1 小时，进程最长 3,400 秒；无平台自动重试。原 07 已完成任务累计 7.428055556 GPU 小时，本轮增量单独记账。
- 远程旧根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261020_item_local_joint_r2`。
- 新根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/beam_validation_20261021_r1`。

本诊断没有重新选择 checkpoint、调参或再次查看测试集。
