# CF R9 冻结对照（2026-11-14）

用户已授权实现并启动两组新增实验。原五组继续，不修改 DIGER 作业。

| 分支 | 原始父组 | 父节点 | 后续 |
|---|---|---|---|
| CF-M0-F9 | 当前 CF 无 Mixer | R9/T40 | 冻结 Tokenizer、SID、目录量化表示，只训练推荐器到 R35 |
| CF-M1-F9 | 当前 CF 层内 Mixer | R9/T40 | 同上，包括冻结 Mixer |

共同 seed=2024。沿用 v2_cf.json、已验收 runtime.locked.json、原教师、prepared、各自预训练资产、数据划分和评测噪声。有效 batch=1024、micro=256、BF16、H100 单卡。各自恢复完整模型、优化器、数据流和随机状态；推荐器学习率按原 35 轮总预算继续，不重置 optimizer/warmup。

父状态分别 SHA256：
- M0：78718413b7b4588d49c1d8c185ec51000ea33a1ef8c628bf001068b9543ae20e
- M1：c9ba73da3bb3b56e73b98d13b00d797fca8eb3107ff842476c3707a43f3dc147

唯一训练干预：从 R9 验证边界开始跳过所有后续 T 更新及 catalog refresh。T 保持40，R 从6237次更新继续至24255次（新增18018次、26轮）。Tokenizer、其优化器、SID、量化表示、T数据流及专用随机生成器不变；R训练和推理流程保持原实现。

复用原 R9 验证分数作为共同起点及 best 候选，不重复 R9 验证。R10至R35逐轮全验证并保存不可覆盖完整checkpoint；按验证 NDCG@5 选best，best和实际R35全测试。训练结束不提前按测试挑点。

实现为独立 scripts/session_wise_ar/cf_freeze.py 子类入口，不修改被冻结的 scripts/rqcd_mixer 引擎或配置。checkpoint增加freeze_policy，独立BRANCH.json记录父路径、SHA和入口SHA；拒绝混入其它日程的中途checkpoint。沿用原恢复严格身份核对和真实GPU全状态加载，新增合成测试核对T完全不动及续训一致性。不重复预训练或已有数值短测。

比较：每个冻结分支对各自持续联合训练；两冻结分支互比Mixer净效果；同轮和验证选best均报告。R9来自既有验证结果，属于单seed探索性冻结时机，不宣称独立多seed确认。正式接收还需所有节点身份、冻结不变量、来源、测试和资源释放核验。
