# 05 阶段版本演进

这里保留探索过程，主结论以 [README](README.md) 的四组对照为准。全部为 Beauty、4×256、seed 2024、SID-level；不同方法不是多个独立 seed。

| 顺序 | 版本 / 短名称 | 为什么做 | 主要变化 | 当前观察 |
|---:|---|---|---|---|
| 1 | SIA / `cross_sia` | 将原来分开训练的 Tokenizer 与 Decoder 联系起来 | 用辅助对齐损失建立间接联系，交替更新 | 最佳总 epoch 5，尚未更新 Tokenizer；不能用于证明直接推荐梯度有效 |
| 2 | DIGER ST v1 / `cross_diger_v1` | 让推荐损失直接影响 SID 量化 | hard-forward / soft-backward；推荐梯度回传码本；温度 1→0.2、噪声 1→0 | 最佳总 epoch 12，即 10 个 Decoder epoch + 2 个 Tokenizer phase；值得跟进，但还混有继续训练 Decoder 的影响 |
| 3 | DIGER ST v2 / `cross_joint_v2` | 检查 v1 退火是否过激 | 温度固定 1；噪声 1→0.2；其余沿用 v1 方案 | 最佳点回到总 epoch 5，未能证明慢退火让 Tokenizer 更新更有效 |
| 4 | Frozen Cross / `cross_frozen` | 排除“Decoder 多训练了一段” | 使用 Cross 对应初始化，只训练 Decoder | 最佳验证分数高于 Joint v2；其最终测试分数较低，提示不能只按测试高低归因 |
| 5 | Joint Refined / `refined_joint` | 看不带交互的基础码本联合更新 | mixer 关闭；从 Refined 的离线 Tokenizer 和 Decoder 起点继续 | HR@10 略升，NDCG@10 等不升，需要自身 Frozen 对照 |
| 6 | Frozen Refined / `refined_frozen` | 补齐 Refined 内部对照 | 同一 Refined 初始化，只训练 Decoder | 已完成，不再是“待跑”；目前 Joint Refined 没有全面优于它 |

两版历史尝试的测试结果如下；四组主要对照不再重复放在这里。

| 版本 | 最佳总 epoch | 最佳点 Decoder epoch / Tokenizer phase | HR@10 | NDCG@10 | 最佳验证 weighted |
|---|---:|---:|---:|---:|---:|
| SIA | 5 | 5 / 0 | 0.077807 | 0.045916 | 0.050886 |
| DIGER v1 | 12 | 10 / 2 | 0.075303 | 0.044644 | 0.051084 |

“最佳点在 Tokenizer 更新之后”只证明该 checkpoint 经历过联合训练，不单独证明提升由 Tokenizer 更新造成。历史 SIA、v1、v2 应保留，不需要称为“无效结果”或删除。

## 旧说明中已纠正的内容

- Frozen Refined 已有完整结果，不再列为下一步待完成实验。
- 本地已补齐六组证据，不再写“只有服务器汇总、缺少本地 metadata”。
- 不再把 Joint Refined 与 Joint Cross 称作严格“唯一变化为 mixer”的对照：两者初始化模型不同。
- 不再根据 SID hash 相同或碰撞率相同，直接推断整个训练中 Tokenizer 没变；冻结状态要看对应参数/阶段记录。
- 不再把匹配的计划预算写成所有运行实际训练量完全一致。

旧版原文保留于 [previous_README.md](reproducibility/previous_README.md)，但它不应再作为当前结论的依据。
