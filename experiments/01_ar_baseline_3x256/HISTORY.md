# 01 阶段版本演进

本文件区分研究过程与正式结论。主结果见 [README.md](README.md)，整理前的原文见 [previous_README.md](reproducibility/previous_README.md)。

| 顺序 | 工作 | 目的 / 已得到的结论 |
|---:|---|---|
| 1 | AR 流程与评估口径准备 | 建立 3 位语义 SID + 1 位 collision suffix；以严格 item-level 命中评估，不把共享 SID 当作同一商品 |
| 2 | RQ-VAE / RQ-KMeans 各三个 seed | 固定主要下游设置，比较逐位置生成难度与推荐效果；形成六次正式基线 |
| 3 | Tokenizer 几何与码本统计 | 补充重构、邻域保持、利用率和碰撞；发现重构更好不等于推荐更好 |
| 4 | Attention / 前缀碰撞补充 | 观察生成 SID2 时 BOS、SID0、SID1 的权重；作为描述性证据保留 |
| 5 | 接入 02、03 后续阶段 | 02 补 PSE 与容量布局；03 在 DiffGRM 内部比较 Tokenizer，不把不同模型的评估口径直接混在一起 |

## 正式运行身份

以下时间是已绑定运行目录标识，不是代码版本号；完整路径以 [runs.csv](runs.csv) 为准。

| Tokenizer | seed | 原运行时间目录 | 最佳 epoch |
|---|---:|---|---:|
| RQ-KMeans | 2024 | AmazonReviews2014_Aug-08-2026_03-18-07 | 55 |
| RQ-KMeans | 2025 | AmazonReviews2014_Aug-08-2026_08-48-38 | 55 |
| RQ-KMeans | 2026 | AmazonReviews2014_Aug-08-2026_14-30-32 | 50 |
| RQ-VAE | 2024 | AmazonReviews2014_Aug-07-2026_08-29-16 | 70 |
| RQ-VAE | 2025 | AmazonReviews2014_Aug-08-2026_06-04-42 | 55 |
| RQ-VAE | 2026 | AmazonReviews2014_Aug-08-2026_11-37-10 | 60 |

RQ-VAE 上游 Tokenizer 的健康 checkpoint 分别选择 step 4,000、5,000、4,000；这是量化器训练步数，不是 AR 的最佳 epoch。原训练图仍在 [rqvae_training.png](report_assets/rqvae_training.png)，不应当作逐位置 SFT Loss 图。

## 本次整理修正

- “还未进行扩散实验”“下一步开始 DiffGRM”改为当时的阶段背景，不再作为当前状态。
- 共享原始 item embedding，不代表两种 Tokenizer 的预处理完全一样。
- 逐位置图使用 NLL；优化 loss 含 label smoothing，源 CSV 的两列均保留。
- teacher-forcing 前缀正确率与自由生成推荐指标分开，不用前者直接证明后者的错误传播。
- 保留旧 PDF、补充报告和所有原图；其中如有旧状态表述，以当前主 README 和锁定证据为准。
- 不把当前源码快照 commit 冒充六次历史训练的 commit；原始运行缺失的记录明确标为缺失。

本阶段的正式基线不包含早期 profile / smoke。它们即使出现在迁移包中，也不能用“最新文件”规则替换上述六次运行。
