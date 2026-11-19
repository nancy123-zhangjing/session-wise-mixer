# 阶段报告索引

`output/` 只保留当前值得阅读和汇报的结果，已按研究推进顺序编号。原始日志、metadata 和 tokenizer 产物仍在 `outputs/`，历史压缩包在 `archive/`。

| 编号 | 报告 | 里面放了什么 |
|---:|---|---|
| 01 | [`01_ar_baseline_3x256/`](01_ar_baseline_3x256/) | RQ-KMeans / RQ-VAE 的 3 seeds AR 基线；逐位置 SFT 指标、推荐指标、SID 统计，以及补充 Attention 分析和 PDF 版本 |
| 02 | [`02_ar_capacity_ablation/`](02_ar_capacity_ablation/) | PSE、RQ-VAE 与三种 RQ-KMeans 容量布局的 seed 2024 内部对照；逐位置 NLL/accuracy、推荐结果、碰撞率、码本统计与 AR Attention |
| 03 | [`03_diffgrm_tokenizer_comparison/`](03_diffgrm_tokenizer_comparison/) | PSE、RQ-KMeans、RQ-VAE + DiffGRM；逐位置 NLL、CPD 解码顺序、推荐指标与 Attention 图 |
| 04 | [`04_crosspse_offline/`](04_crosspse_offline/) | Original PSE、Refined-PSE、Cross-PSE 的离线 tokenizer 对照；说明只离线加入码本交互为何没有带来稳定增益 |
| 05 | [`05_crosspse_joint/`](05_crosspse_joint/) | Cross-PSE 与 DiffGRM 的交替联合训练；当前结果、能下的结论、不能下的结论和后续 matched controls |

建议汇报时按 `01 → 02 → 03 → 04 → 05` 讲：先说明现象，再说明由现象引出的 Cross-PSE 方法，最后报告联合训练的初步结果。
