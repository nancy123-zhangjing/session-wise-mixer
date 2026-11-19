# 阶段 05：码本交互 × 推荐反馈更新的严格对照

## 要回答的问题

之前 Cross 与 Refined 的历史实验起点不同，不能直接把指标差异归因于交互模块。本轮把起点统一，验证：**同样获得推荐反馈、同样恢复 Decoder 时，加入交互模块是否有额外收益？**

| 组别 | 交互模块 | Tokenizer 更新 | Decoder 恢复 |
|---|---|---|---|
| `no_mixer_frozen` | 禁用 | 不更新 | 3 个完整 epoch |
| `no_mixer_joint` | 禁用 | 基础码本更新 24 次 | 3 个完整 epoch |
| `mixer_frozen` | 启用但冻结 | 不更新 | 3 个完整 epoch |
| `mixer_joint` | 启用 | 基础码本与 Mixer 更新 24 次 | 3 个完整 epoch |

## 如何保证起点一致

从已完成的推荐反馈短对照取更新前的完整模型快照，将其**有效码本 E₀**作为四组共同基础码本；推荐器完整权重、item slots、正式 SID、OPQ 和重构 anchor 保持相同。

新的 Mixer 使用固定残差系数 1，attention 输出投影零初始化。因此启用或禁用时初始有效码本都恰好为 E₀，同时启用分支仍有首步梯度。不能简单禁用旧 Mixer 后使用旧基础码本，那样起点会改变。

这是从旧 Cross 编码功能出发的**暖启动重参数化**，不是 Original PSE 从头训练，也不是旧 attention 权重的继续训练。

## 固定训练与评估条件

- Beauty，4×256，seed 2024；同一完整推荐器起点，四组优化器均重新创建。
- 两个 Joint：384 个 batch、梯度累积 16，即 24 次 Tokenizer 更新。推荐反馈权重均为 1；无 Gumbel 噪声；SIA 为 0。
- 码本学习率 `1e-5`，Mixer 学习率 `1e-4`；其余辅助损失和优化器设置沿用已锁定短对照。
- 四组随后冻结 Tokenizer，Decoder 学习率 `1e-4`，各恢复 3 个完整 epoch；每轮 1652 个 batch、104 次更新。训练顺序、随机种子、数据顺序相同。
- 保持历史实际设置：label smoothing `0.2`，训练 batch 64、验证 batch 8；Tokenizer FP32，Decoder BF16。
- 仅验证集、SID-level。统一比较第 3 轮，不取各组最高分，不评估或用测试集选参数。

入口：`scripts/train/run_diger_mixer_factorial.py`。协议：`configs/05_crosspse_joint/pilots/diger_mixer_factorial_seed2024.json`。测试：`tests/test_diger_mixer_factorial.py --require-torch`。

## 结果怎么看

主要记录第 3 轮 HR@10 / NDCG@10，并同时保留更新后的确定性 SID 变化、碰撞率、推荐梯度和实际参数更新证据。

交互模块的增量：

`(mixer_joint − mixer_frozen) − (no_mixer_joint − no_mixer_frozen)`

两个 Frozen 的有效编码完全相同，它们的恢复轨迹与验证结果也应相同；不一致先视为控制失效。本轮单 seed、小预算结果只是方向性证据。启用 Mixer 包含参数容量和优化路径变化，不能单独证明“信息交换机制”；`full` attention 还包含层内交互。

**本轮是一次 Tokenizer 更新后冻结恢复，不是完整多轮交替联合训练。** 若短对照显示增量，再考虑更长恢复和多 seed；不在这一轮同时改训练策略。

## 版本与资产保存

- 历史数据、初始化与快照从 HDD 主项目只读加载，不重新拟合 SID，不修改历史证据。
- 新代码、交付收据和权重使用已授权的全局个人目录：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training/`。
- `code/` 为锁定 commit 的稀疏执行副本；不是数据备份。`delivery/<唯一名称>/` 保存版本校验、测试和启动日志；`outputs/pilots/stage05_mixer_factorial/<唯一名称>/` 保存训练产物。
- 一份共同初始快照，每组保留 `post_tokenizer` 和 `decoder_3` 的完整模型、优化器及 RNG 状态；中间轮只存指标。恢复时必须用 E₀重建非持久 anchor，再加载完整状态。
- GPU 短测试与正式运行分目录；只有四组完整完成且守卫通过才写 `COMPLETE.json`。
