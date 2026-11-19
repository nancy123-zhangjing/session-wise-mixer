# Cross-PSE × DiffGRM 交替联合训练

## 1. 这一阶段要回答什么

离线 Cross-PSE 先生成 SID、再单独训练 Decoder 时，Tokenizer 学到的交互信息不一定真正服务于推荐目标。因此本阶段用 DIGER-style 的可微离散桥接和交替训练，让推荐损失能够更新 Cross-PSE，同时每个阶段只更新 Tokenizer 或 Decoder 一侧。

## 2. 当前实现

- 初始化：使用已经训练好的 Cross-PSE v3 tokenizer 与对应 DiffGRM checkpoint；
- 前向：仍使用 hard SID，保证实际训练和推理面对离散编码；
- 反向：用 straight-through soft assignment 把推荐梯度传回 Tokenizer；
- 优化：Tokenizer phase 与 Decoder phase 交替进行，不同时更新两侧参数；
- 稳定项：保留码本 balance 与 anchor 约束，降低码本塌缩和 SID 剧烈漂移风险；
- 当前正式结果：`DIGER ST v2 slow-anneal`，Beauty、`4×256`、seed 2024、SID-level 评估。

## 3. 当前主要结果

| 方法 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | 说明 |
|---|---:|---:|---:|---:|---|
| Original PSE + DiffGRM | 0.05205 | 0.07490 | 0.03657 | 0.04392 | 静态 PSE 基线 |
| Offline Cross-PSE v3 + DiffGRM | 0.05151 | 0.07360 | 0.03611 | 0.04320 | 码本交互离线完成，之后固定 SID |
| Frozen Cross-PSE + Decoder continuation | 0.05362 | 0.07423 | 0.03807 | 0.04467 | 固定 Cross-PSE，仅继续训练 Decoder |
| Joint Refined-PSE + DiffGRM | 0.05259 | 0.07727 | 0.03643 | 0.04435 | 只联合更新基础码本，mixer 关闭 |
| Joint Cross-PSE v2 + DiffGRM | **0.05563** | **0.07732** | **0.03902** | **0.04599** | DIGER-style 梯度桥接与交替训练 |

Joint v2 的最佳 checkpoint 出现在 epoch 5。Tokenizer phase 中，item SID change rate 从约 `0.19%` 增长到 `1.73%`，整体碰撞率维持在约 `17.9%`，说明 Tokenizer 确实被更新，同时没有观察到明显碰撞恶化。

Frozen 对照的最佳验证集 weighted score 为 `0.05110`（epoch 15），高于 Joint v2 的 `0.05067`（epoch 5）；其测试集 weighted score 为 `0.05058`，低于 Joint v2 的 `0.05225`。

Joint Refined-PSE 的最佳验证集 weighted score 为 `0.04903`（wall epoch 19，对应 15 个 Decoder epoch 和 4 个 Tokenizer phase），测试集 weighted score 为 `0.05094`。其每次 Tokenizer phase 仅改变约 `0.0083%～0.0496%` 的 item SID，说明当前基础码本更新较保守。

## 4. 当前结论边界

Frozen 对照表明，Decoder 单独继续训练本身就能带来收益。Joint Cross-PSE v2 的最佳 checkpoint 又出现在第一次 Tokenizer 更新之前，因此目前的测试集提升不能归因于 Tokenizer 更新。Joint Refined-PSE 虽然在四次 Tokenizer phase 后达到最佳点，但 hard SID 变化很小，也还不能排除 Decoder continuation 的贡献。

原因是当前联合版本同时包含两件事：

1. Tokenizer 被推荐损失更新；
2. Decoder 也获得了额外训练。

因此结果中仍混有“Decoder 继续训练”的影响；同时还缺少一个关闭 mixer、但保持同样联合训练方式的对照。

## 5. 下一步最小实验

| 对照 | 保持一致 | 唯一改变 | 回答的问题 |
|---|---|---|---|
| Frozen Cross-PSE + Decoder continuation（已完成） | 初始化、训练轮数、Decoder 更新预算、评估协议 | Tokenizer 完全冻结 | 当前提升是否只是 Decoder 多训练了一段 |
| Joint Refined-PSE（已完成） | 梯度桥接、交替节奏、优化预算、正则与评估协议 | 关闭跨码本 mixer | 跨码本交互本身是否有额外价值 |
| Frozen Refined-PSE | Refined-PSE artifact、初始 Decoder、Decoder 更新预算与评估协议 | 冻结基础码本 | Refined-PSE 的小幅提升来自码本更新还是 Decoder continuation |

下一步先完成 Frozen Refined-PSE 的 seed 2024 对照。之后分别计算 `Joint Refined − Frozen Refined` 和 `Joint Cross − Frozen Cross`；只有前者能判断普通码本更新是否有效，两组增益之差才能进一步判断跨码本交互是否有额外价值。

Frozen Refined-PSE 已建立独立运行入口，使用与 Joint Refined-PSE 相同的 tokenizer artifact、初始 Decoder 和35个 Decoder epoch，唯一变化是基础码本与 hard SID 全程冻结。

Joint Refined-PSE 已建立独立配置与运行入口。它沿用 Joint Cross-PSE v2 的 DIGER 梯度桥和交替训练协议，但只允许基础码本更新，跨码本 attention mixer 始终关闭。

Frozen Cross-PSE 正式结果的 best epoch 为 15，训练耗时约 1.38 小时，SID hash 与 Joint v2 一致，说明冻结控制有效。运行说明见 `docs/diffgrm_crosspse_frozen_control.md`。

## 6. 本地材料状态

当前目录记录的是已在服务器得到的汇总结果。完整 joint run 的 `per_seed.csv`、`evaluation_history.csv`、联合训练 CSV、run metadata 和配置还没有作为证据包复制进本地报告目录；后续正式汇报前应补一次归档。
