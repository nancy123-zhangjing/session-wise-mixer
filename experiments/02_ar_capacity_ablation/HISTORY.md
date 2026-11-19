# 02 阶段版本演进

主问题与统一结果见 [README.md](README.md)。这里记录“为什么补这三组”，并明确它们与第一阶段基线的关系。

| 顺序 | 实验 | 目的 | 当前观察 |
|---:|---|---|---|
| 1 | 复用 RQ-KMeans / RQ-VAE 3×256，seed 2024 | 保留 01 的 AR 内部参照 | 不是新增训练，不重复计数 |
| 2 | PSE 3×256 + AR | 补齐同一 AR 框架下的 Tokenizer 类型对照 | HR@10 较高，NDCG@10 未超过均匀 RQ-KMeans；语义 SID 碰撞较多 |
| 3 | RQ-KMeans 128×256×512 + AR | 在名义组合容量不变时，将容量更多分配到后部位置 | 前部难度降低、后部增加；没有获得推荐提升 |
| 4 | RQ-KMeans 512×256×128 + AR | 反向分配，检查布局变化的方向性 | SID0 更难、SID2 更容易；碰撞更少，但推荐指标下降更明显 |
| 5 | 对三组补充 attention | 看位置依赖的层间模式是否改变 | 总体模式相近；不足以解释推荐高低 |

## 三次新增正式运行

| 短名称 | run_id | 原运行时间目录 | 最佳 epoch |
|---|---|---|---:|
| PSE | ar_formal_pse_3x256 | AmazonReviews2014_Aug-14-2026_05-17-06 | 40 |
| RQK 128×256×512 | ar_formal_rqkmeans_128x256x512 | AmazonReviews2014_Aug-14-2026_07-34-19 | 55 |
| RQK 512×256×128 | ar_formal_rqkmeans_512x256x128 | AmazonReviews2014_Aug-14-2026_10-19-23 | 70 |

全部 seed 2024。正式结果绑定上述运行，不从文件修改时间推断“最新有效实验”。证据包中还含早期 profile / smoke 的 metadata，不能混入这三次正式结果。

复用基线分别是 01 的 RQ-KMeans `Aug-08-2026_03-18-07` 和 RQ-VAE `Aug-07-2026_08-29-16`。01 的其他 seed 仍保留，但五组主表只取 seed 2024。

## 本次整理修正

- 旧 README 写“验证 weighted score 选模”；归档配置为 `val_metric: ndcg@10`，trainer 按该字段选模。主文已修正，并明确历史完整 resolved config 缺失。
- “相同组合容量”不等于相同词表总大小或相同参数量：768 与 896 个语义码字的区别单独列出。
- PSE / RQ-KMeans / RQ-VAE 共享原始 embedding，但预处理不同；不再笼统称所有设置完全一致。
- normalized NLL 是按 ln(Kᵢ) 标准化的诊断，不等于 label-smoothed 优化目标，也不直接等于语义信息量。
- 基线 attention 图是三个 seed 聚合，新增三组图是单 seed；保留原图但标清统计层级，避免错当五组同口径比较。
- 所有方向性结论均限定当前 seed 2024；没有把未跑的 RQ-VAE 非均匀容量实验写为已完成。

旧版完整保存在 [previous_README.md](reproducibility/previous_README.md)。后续研究应使用新 run_id、独立结果目录和固定代码/输入绑定，不覆盖这批基线结果。
