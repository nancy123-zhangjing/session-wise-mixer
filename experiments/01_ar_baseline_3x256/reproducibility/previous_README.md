# AR 生成式推荐第一阶段实验汇总

> **当前研究问题：**固定 `3 × 256` Semantic ID 结构和下游 AR 模型，只改变 Tokenizer，观察不同语义空间是否会改变 Decoder 的逐位置生成难度和最终推荐效果。

## 1. 当前完成情况

- 在 Amazon Reviews 2014 Beauty 数据集上完成 AR_GRM 的 SFT 训练；
- 完成 `RQ-KMeans 3 × 256` 和 `RQ-VAE 3 × 256` 的公平对照；
- 两种 Tokenizer 分别运行 2024、2025、2026 三个随机种子，共 6 次正式实验；
- 记录 train、validation、test 的逐位置 NLL、normalized NLL、Token accuracy 和 Prefix accuracy；
- 使用严格 item-level 口径评估 HR@5/10、NDCG@5/10、legal@10 和 duplicate@10；
- 当前还没有进行 RL 和扩散模型实验。

## 2. 实验设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | Amazon Reviews 2014 — Beauty |
| 用户数 / 物品数 / 交互数 | 22,363 / 12,101 / 198,502 |
| Train / Validation / Test | 105,668 / 22,363 / 22,363 |
| 数据划分 | Leave-one-out，训练集使用 sliding windows |
| Item embedding | sentence-t5-xl，768 维 |
| Semantic ID | 3 个语义位置，每个位置 256 个 code |
| Item 唯一性 | 增加第 4 位 collision suffix |
| 对照方法 | RQ-KMeans 与 RQ-VAE |
| AR 模型 | 2 层 Encoder、4 层 Decoder、hidden size 256 |
| 最大训练轮数 | 100 epochs，Early stopping |
| 有效 batch size | 512 |
| Beam search | Beam 64，输出 Top-10 |
| 随机种子 | 2024、2025、2026 |
| 模型选择 | Validation NDCG@10 最优 checkpoint |
| 测试口径 | 严格 item-level：预测 item ID 必须与目标 item ID 完全一致 |

两组实验除 Tokenizer 及其生成的 Semantic ID 外，数据、输入 embedding、SID 深宽、AR 模型和训练评估设置保持一致。

## 3. 学长要求的逐位置 SFT Loss 图

这一部分是当前汇报的核心。图中直接使用训练日志记录的 NLL，没有进行平滑处理：

- 横轴：AR 模型训练 epoch；
- 纵轴：对应位置真实 token 的 NLL，越低表示该位置越容易预测；
- 蓝色：第 1 个语义位置；
- 橙色：第 2 个语义位置；
- 绿色：第 3 个语义位置；
- 红色：第 4 位 collision suffix，它不是语义码，只用于区分发生 SID 碰撞的 item；
- 实线：训练集 NLL；
- 虚线和圆点：验证集 NLL，验证每 5 个 epoch 进行一次。

### 3.1 Seed 2024

#### RQ-KMeans

![RQ-KMeans seed 2024 逐位置 SFT NLL](report_assets/position_nll_rq_kmeans_seed2024.png)

#### RQ-VAE

![RQ-VAE seed 2024 逐位置 SFT NLL](report_assets/position_nll_rqvae_seed2024.png)

### 3.2 Seed 2025

#### RQ-KMeans

![RQ-KMeans seed 2025 逐位置 SFT NLL](report_assets/position_nll_rq_kmeans_seed2025.png)

#### RQ-VAE

![RQ-VAE seed 2025 逐位置 SFT NLL](report_assets/position_nll_rqvae_seed2025.png)

### 3.3 Seed 2026

#### RQ-KMeans

![RQ-KMeans seed 2026 逐位置 SFT NLL](report_assets/position_nll_rq_kmeans_seed2026.png)

#### RQ-VAE

![RQ-VAE seed 2026 逐位置 SFT NLL](report_assets/position_nll_rqvae_seed2026.png)

### 3.4 从曲线中直接看到的现象

- 两种 Tokenizer 的训练 NLL 都随训练推进而下降；
- 验证集第 1、2 个位置的 NLL 在训练中后期出现不同程度回升；
- 第 3 个位置和 collision suffix 的 NLL 明显低于前两个位置；
- 三个 seed 的整体曲线形态基本一致；
- RQ-KMeans 和 RQ-VAE 的具体差异需要结合最终测试集逐位置结果一起查看。

这里只描述曲线本身，不根据曲线单独判断 Tokenizer 优劣，也不将验证 NLL 后期上升直接归因于某一种机制。

## 4. 测试集逐位置结果

下表为三个随机种子的 `均值 ± 标准差`。

| Tokenizer | 位置 | 位置作用 | NLL | Normalized NLL | Token accuracy | Prefix accuracy |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| RQ-KMeans | 1 | Semantic | 4.888 ± 0.087 | 0.881 ± 0.016 | 7.99% ± 0.69% | 7.99% ± 0.69% |
| RQ-KMeans | 2 | Semantic | 3.838 ± 0.064 | 0.692 ± 0.012 | 15.40% ± 0.55% | 1.30% ± 0.09% |
| RQ-KMeans | 3 | Semantic | 0.586 ± 0.021 | 0.106 ± 0.004 | 85.03% ± 0.86% | 1.04% ± 0.07% |
| RQ-KMeans | 4 | Collision suffix | 0.133 ± 0.003 | 0.024 ± 0.001 | 98.19% ± 0.10% | 0.99% ± 0.05% |
| RQ-VAE | 1 | Semantic | 3.960 ± 0.143 | 0.714 ± 0.026 | 10.53% ± 0.11% | 10.53% ± 0.11% |
| RQ-VAE | 2 | Semantic | 4.314 ± 0.359 | 0.778 ± 0.065 | 10.90% ± 0.08% | 1.52% ± 0.11% |
| RQ-VAE | 3 | Semantic | 1.433 ± 0.295 | 0.258 ± 0.053 | 61.33% ± 7.65% | 0.83% ± 0.05% |
| RQ-VAE | 4 | Collision suffix | 0.135 ± 0.022 | 0.024 ± 0.004 | 98.31% ± 0.73% | 0.77% ± 0.02% |

![测试集逐位置 NLL 均值](report_assets/position_nll.png)

![测试集逐位置 Token accuracy 与 Prefix accuracy](report_assets/position_accuracy.png)

需要注意：Token accuracy 只判断当前位置是否正确；Prefix accuracy 要求从第 1 位到当前位置全部正确，因此能够体现前部错误在自回归生成中的累积。

## 5. 最终推荐指标

以下结果均采用严格 item-level 测试，表中为三个随机种子的 `均值 ± 标准差`。

| 指标 | RQ-KMeans | RQ-VAE | RQ-KMeans 相对变化 |
| --- | ---: | ---: | ---: |
| HR@5 | **0.03618 ± 0.00194** | 0.03261 ± 0.00017 | +10.9% |
| HR@10 | **0.05689 ± 0.00230** | 0.05199 ± 0.00058 | +9.4% |
| NDCG@5 | **0.02352 ± 0.00111** | 0.02121 ± 0.00004 | +10.9% |
| NDCG@10 | **0.03020 ± 0.00111** | 0.02742 ± 0.00020 | +10.1% |
| legal@10 | 0.99999 ± 0.00001 | 0.99980 ± 0.00018 | 两者均接近 1 |
| duplicate@10 | 0 | 0 | 无重复推荐 |

![三个 seed 的下游推荐指标均值](report_assets/downstream_metrics.png)

### 5.1 每个 seed 的结果

| Tokenizer | Seed | Best epoch | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RQ-KMeans | 2024 | 55 | 0.03765 | 0.05777 | 0.02463 | 0.03111 |
| RQ-KMeans | 2025 | 55 | 0.03398 | 0.05429 | 0.02242 | 0.02896 |
| RQ-KMeans | 2026 | 50 | 0.03689 | 0.05862 | 0.02352 | 0.03052 |
| RQ-VAE | 2024 | 70 | 0.03242 | 0.05259 | 0.02118 | 0.02763 |
| RQ-VAE | 2025 | 55 | 0.03269 | 0.05142 | 0.02119 | 0.02724 |
| RQ-VAE | 2026 | 60 | 0.03273 | 0.05196 | 0.02125 | 0.02739 |

![三个 seed 的 HR@10 与 NDCG@10](report_assets/seed_consistency.png)

三个 seed 上的差值方向一致，但目前只有三个 seed，因此这里只汇报均值、标准差和结果方向，不强调统计显著性。

## 6. Tokenizer 本身的指标

| 指标 | RQ-KMeans | RQ-VAE |
| --- | ---: | ---: |
| Reconstruction cosine | 0.91167 ± 0.00002 | **0.92735 ± 0.00171** |
| Reconstruction MSE | 0.17667 ± 0.00004 | **0.14529 ± 0.00342** |
| Neighbor Recall@10 | **0.23880 ± 0.00123** | 0.13577 ± 0.01666 |
| Unique semantic IDs | 11,858.0 ± 11.3 | 11,848.7 ± 93.3 |
| SID collision rate | 3.75% ± 0.14% | 4.02% ± 1.48% |

![Tokenizer 重构、邻域保持和 SID 碰撞](report_assets/tokenizer_geometry.png)

### 6.1 各语义位置的码本利用率

| Tokenizer | 位置 1 | 位置 2 | 位置 3 |
| --- | ---: | ---: | ---: |
| RQ-KMeans utilization | 100.0% ± 0.0% | 100.0% ± 0.0% | 100.0% ± 0.0% |
| RQ-VAE utilization | 32.8% ± 3.8% | 47.4% ± 14.7% | 94.4% ± 1.2% |
| RQ-KMeans normalized entropy | 98.69% ± 0.08% | 97.98% ± 0.47% | 97.59% ± 0.45% |
| RQ-VAE normalized entropy | 78.63% ± 1.93% | 83.99% ± 6.17% | 97.71% ± 1.01% |

![码本利用率和 normalized entropy](report_assets/codebook_usage.png)

这些指标用于描述两种 Tokenizer 生成的离散空间，不单独作为最终推荐性能的判断依据。

## 7. 补充：RQ-VAE Tokenizer 训练曲线（不是 AR 逐位置 Loss）

这张图记录的是上游 **RQ-VAE Tokenizer** 的 validation loss，只用于说明 Tokenizer 训练和 checkpoint 选择；学长要求的 AR 逐位置 SFT Loss 是第 3 节的六张图。

图中直接画出日志里的三个 loss：`total = reconstruction + quantization`。三幅子图使用相同的纵轴范围，红色虚线和圆点表示最终选用的健康 checkpoint，没有进行平滑处理。

| Seed | 选用 step | Best validation total loss | Tokenizer 训练时间 |
| ---: | ---: | ---: | ---: |
| 2024 | 4k | 0.1564 | 8.8 min |
| 2025 | 5k | 0.1542 | 9.1 min |
| 2026 | 4k | 0.1548 | 9.1 min |

![RQ-VAE validation total loss](report_assets/rqvae_training.png)

## 8. 现阶段可以汇报的结论

在当前 `Amazon Beauty + 3 × 256 SID + AR_GRM` 设置下：

1. RQ-KMeans 的四项最终推荐指标均高于 RQ-VAE；
2. 两种方法的 legal@10 均接近 100%，duplicate@10 均为 0；
3. RQ-VAE 的点对点重构指标更好，但 Neighbor Recall@10 更低；
4. 两种方法的逐位置难度分布不同，测试集差异主要出现在第 2、3 个语义位置；
5. 当前结果只覆盖一个数据集、一种 SID 深宽和一个 AR 模型，不能据此判断某一种 Tokenizer 在所有生成推荐任务中都更好。

## 9. 下一步

1. 复现 DiffGRM 扩散模型，记录逐位置生成 Loss 和每个位置实际被解码的先后顺序；
2. 在扩散模型中分别使用当前 RQ-KMeans 和 RQ-VAE Semantic ID，保持其他设置一致；
3. 比较 AR 的逐位置生成难度与扩散模型实际解码顺序是否对应；
4. 再比较不同 SID 深宽结构，例如 `2/3/4/5 × 256`；
5. RL 阶段作为后续可选扩展，在基础对照完成后再决定是否加入。

