# TIGER 无 Mixer：训练及全测试结果

训练及 best/final 全测试完成，GPU 作业 SUCCEEDED，唯一 worker 为 instance_success。整体五组 Goal 尚未完成。

| 节点 | 更新数 | 验证 NDCG@5 | 测试 NDCG@5 |
|---|---:|---:|---:|
| 验证选 best，第14轮 | 193998 | 0.008074431 | 0.008107862 |
| 实际 final，约14.433轮 | 200000 | 0.005795711 | 0.004778056 |

完成原定200000次更新、15个完整验证节点，每次验证71657条；best/final各测试53194条。最佳点以验证集选择，不依据测试选点。最终点相对best测试下降约41.1%。Tokenizer及SID在推荐训练期间固定，因此这次下降不能归因于持续更新码本；单个末端回落尚不能证明某种具体机制。

15个节点已核验checkpoint SHA、配置/来源/父状态/环境、续训状态字段和计数、SID合法范围及完整SID唯一性，见AR_T0_NODE_AUDIT_RESULT.json。实际远端源文件与训练收据中的来源清单一致，best/final测试summary、验证身份及预测文件SHA一致，见AR_T0_TEST_AUDIT.json。

恢复边界：以上为实际文件CPU结构检查与自动GPU推理加载证据；AR测试未调用完整_restore，不应表述为15个节点均已通过CUDA数值恢复。启动前GPU恢复短测可复用，具体边界见AR_ACCEPTANCE_CHECKLIST.md。

权重及预测保留远端：/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/session_wise_ar_five_20261113_r2/runs/AR-T0/。本地只保存小收据、路径、SHA和报告。资源回执：raw/CURRENT_JOB_LIST.json、raw/AR_T0_FINAL_INSTANCES.json。
