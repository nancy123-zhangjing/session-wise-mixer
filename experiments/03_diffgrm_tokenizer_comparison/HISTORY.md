# 03 阶段版本演进

主报告只使用三组 **v6、seed 2024** 正式运行。早期 pilot、smoke/profile 和未完成配置独立保留，不混入方法均值或重复 seed。

| 顺序 | 版本 / 证据 | 为什么调整 | 在本阶段的角色 |
|---:|---|---|---|
| 1 | diff_formal_pse_4x256；Aug-10-2026_14-50-32 | 100 epochs 约 10,400 updates，warmup 10,000；大部分训练仍在 warmup，best epoch 100 | 完成的早期 pilot，不作正式 v6 基线 |
| 2 | diff_formal_pse_4x256_schedfix_v2；Aug-11-2026_05-52-00 | 累积步数改为 4，检查 optimizer-step 调度；best epoch 27 | 历史探索，训练预算与正式版不同 |
| 3 | schedfix v3 / officialbatch v4 配置与入口 | 数值稳定和批量设置调整 | 当前本地材料未确认完整对应正式运行，不因存在配置而写成“已跑完” |
| 4 | repo_repro_v5_bf16；Aug-11-2026_10-58-33 | 仓库 Beauty 参数、BF16、有效 batch 1024、200 epochs；但 warmup 前可触发早停 | 历史版本，best epoch 45；不可与 v6 metadata 混用 |
| 5 | repo_repro_v6_bf16；Aug-11-2026_14-09-45 | 延后至 warmup 完成后累计 patience，仍从 epoch 20 保存验证最佳模型 | 正式 PSE 基线，best epoch 45 |
| 6 | rqkmeans_4x256_v6_bf16；Aug-12-2026_06-37-17 | 保留 v6 下游设置，替换为残差 KMeans 的 SID | 正式 RQ-KMeans 对照，best epoch 55 |
| 7 | rqvae_4x256_v6_bf16；Aug-12-2026_12-04-25 | 保留 v6 下游设置，替换为冻结 RQ-VAE 的 SID | 正式 RQ-VAE 对照，best epoch 55 |

表中缩写均可在 [runs.csv](runs.csv) 或保存的 metadata 中回溯。早期 `diff_smoke_*` / `diff_profile_*` 只用于检查流程和速度，不是完整推荐实验。

## 必须分开的三件事

1. **Tokenizer 训练版本**：生成哪套固定 SID，包括输入 embedding、码本和 item 顺序。
2. **Decoder 训练版本**：采用哪组配置、seed、训练日程，以及验证集选了哪个 checkpoint。
3. **后处理版本**：同一模型的 loss / 解码顺序 / attention 分析与画图；重新画图不是重新训练。

PSE v5 与 v6 的最终测试数值相同，但两次训练的身份和日程不同。本阶段只绑定 v6 的 Aug-11_14-09-45，避免“指标一样就共用任意 checkpoint”。

## 本次整理纠正的解释

- 不把 SID-level 的 PSE 领先直接解释为严格 item-level 最优。
- 不把 fully-masked NLL 当作推荐效果的充分代理，也不直接等同于 AR 的逐位置 SFT loss。
- 不根据平均 Attention 权重作因果结论。
- 原版 PSE 也经过 OPQ/PQ 拟合，“冻结”指下游训练中不再更新。
- 与 AR 跨模型统一 Tokenizer / 评估口径是另一项研究设计，不是当前内部对照必须补跑的前提。

历史原文保留于 [previous_README.md](reproducibility/previous_README.md)。当前整理未找回历史运行 commit，不补写不存在的代码溯源。
