# DiffGRM × 三种 Tokenizer 阶段性实验汇报

> 数据集：Amazon Reviews 2014 — Beauty  
> 对照组：PSE、RQ-KMeans、RQ-VAE  
> Semantic ID：4 × 256  
> Decoder seed：2024  

## 1. 研究目标

本阶段是希望沿着下面这条链路分析 Semantic ID 对扩散式生成推荐的影响：

```text
Tokenizer
  → Semantic ID 空间的碰撞率、码本利用率和分布
  → 各 SID 位置的生成难度
  → DiffGRM 的动态解码顺序与位置间信息依赖
  → 最终推荐效果
```


## 2. 实验设计

### 2.1 控制变量

| 项目 | 设置 |
|---|---|
| 数据集 | Amazon Reviews 2014 Beauty |
| 物品数 | 12,101 |
| 完整测试样本数 | 22,363 |
| Item Embedding | sentence-transformers/sentence-t5-base |
| Embedding 预处理 | PCA 256 + 归一化 |
| Semantic ID | 4 个位置，每个位置 256 个编码 |
| Encoder | 1 Layer |
| Decoder | 4 Layers，4 Heads |
| Hidden / FFN | 256 / 1024 |
| OCN | Guided Masking，Least-confident First |
| CPD | Confidence-based Parallel Decoding |
| 训练 Batch | Physical Batch 64，Gradient Accumulation 16，Effective Batch 1024 |
| 优化 | LR 0.01，Warmup 10,000 Updates，Weight Decay 0 |
| 数值设置 | BF16，Gradient Clip 1.0，Finite-value Guard |
| 最大训练轮数 | 200 Epochs |
| 评估 | Epoch 20 起，每 5 Epochs 评估一次 |
| 早停 | Warmup 完成后启用，Patience 5 |
| 正式测试 | Beam 256，Top-10 |
| 随机种子 | Decoder seed 2024 |

三组实验保持数据、Embedding、SID 容量、DiffGRM 架构、优化器、训练日程和评估方式一致，只替换 Tokenizer 产生的 Semantic ID。

### 2.2 三种 Tokenizer

| Tokenizer | 本实验中的实现 |
|---|---|
| PSE | DiffGRM 原始 OPQ/PQ 路径：`OPQ4, IVF1, PQ4×8` |
| RQ-KMeans | 在共享 PCA256 Embedding 上进行 4 阶 Residual KMeans，每阶 256 Codes，Tokenizer seed 2024 |
| RQ-VAE | 在共享 PCA256 Embedding 上训练 4×256 RQ-VAE，Latent Dim 32，冻结最佳 Tokenizer 后训练 DiffGRM |

RQ-VAE Tokenizer 的最佳 checkpoint 出现在 Step 3,000；完整训练配置上限为 60,000 Steps。正式 DiffGRM 使用冻结的最佳 Semantic ID，不会在下游训练过程中更新 Tokenizer。

### 2.3 复现参数说明

当前 v6 采用作者仓库 README 中 Beauty 的实际参数：Effective Batch 1024、LR 0.01、Label Smoothing 0.2、最多 200 Epochs。它与论文正文中部分描述的 100 Epochs、Label Smoothing 0.1 存在差异。

## 3. Semantic ID 空间分析

### 3.1 完整 SID 与前缀碰撞率

碰撞率定义为：属于重复 SID（或重复前缀）分组的物品数，占全部 12,101 个物品的比例。

| Tokenizer | Prefix-1 | Prefix-2 | Prefix-3 | Full 4-digit SID | 唯一完整 SID | 最大碰撞组 |
|---|---:|---:|---:|---:|---:|---:|
| PSE | 99.98% | 60.22% | 27.11% | **17.95%** | 10,656 | 40 |
| RQ-KMeans | 99.85% | 71.23% | 8.04% | **1.78%** | 11,989 | 5 |
| RQ-VAE | 100.00% | 40.62% | 6.44% | **2.10%** | 11,967 | 4 |

阶段性观察：

- Prefix-1 几乎必然碰撞，这是 12,101 个物品映射到 256 个首位编码的自然结果。
- RQ-VAE 在 Prefix-2 和 Prefix-3 上的区分能力最好；RQ-KMeans 在完整四位 SID 上的碰撞率最低。
- PSE 的完整碰撞率达到 17.95%，明显高于另外两组。这意味着 PSE 中较多物品共享相同的四位 SID。
- Prefix-2 的结果并不是单调决定最终碰撞率：RQ-KMeans 的 Prefix-2 碰撞较高，但加入第 3、4 位后碰撞快速下降。

### 3.2 逐位置码本覆盖率（至少出现一次）

这里的覆盖率仅定义为 `该位置在全部 12,101 个物品中至少出现过一次的编码数 ÷ 256`。它只表示有没有“完全未使用”的 Code，不等同于编码分布均衡，也不单独代表 Tokenizer 质量；分布是否均衡需要结合下一节的归一化熵判断。

| Tokenizer | SID0 | SID1 | SID2 | SID3 |
|---|---:|---:|---:|---:|
| PSE | 256/256（100.0%） | 256/256（100.0%） | 256/256（100.0%） | 256/256（100.0%） |
| RQ-KMeans | 197/256（77.0%） | 219/256（85.5%） | 232/256（90.6%） | 236/256（92.2%） |
| RQ-VAE | 256/256（100.0%） | 256/256（100.0%） | 256/256（100.0%） | 256/256（100.0%） |

### 3.3 逐位置归一化熵

归一化熵越接近 1，说明该位置的 256 个编码使用越均衡。

| Tokenizer | SID0 | SID1 | SID2 | SID3 |
|---|---:|---:|---:|---:|
| PSE | 0.966 | 0.959 | 0.959 | 0.961 |
| RQ-KMeans | 0.829 | 0.853 | 0.872 | 0.886 |
| RQ-VAE | 0.985 | 0.994 | 0.997 | 0.998 |

阶段性观察：

- RQ-VAE 四个位置均覆盖全部 256 个编码，并且分布最接近均匀；本次 DiffGRM 4×256 RQ-VAE 没有出现明显的边际码本坍缩。该结果与此前 AR 3×256 实验中的 RQ-VAE 低覆盖率不矛盾，因为两阶段使用了不同的输入维度、SID 层数和独立训练得到的 Tokenizer，不能把前一阶段的覆盖率直接套用到本实验。
- PSE 的单位置利用率和熵也较高，但完整 SID 碰撞率仍然很高。这说明“每个位置单独分布均衡”不等于“多个位置的组合足够唯一”，还需要观察位置间组合结构。
- RQ-KMeans 前层码本利用率较低，但完整 SID 碰撞率最低，说明残差层提供了较强的组合区分能力。

## 4. 下游推荐结果

当前评估为论文/仓库对齐的 **SID-level**：预测 SID 与目标 SID 一致即记为命中，不要求解码到唯一的具体商品。

| Tokenizer | Best Epoch | HR@5 | HR@10 | NDCG@5 | NDCG@10 | Legal@10 | Duplicate@10 | Fit Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PSE | 45 | **0.05205** | **0.07490** | **0.03657** | **0.04392** | 1.0000 | **0.00077** | 4.31 h |
| RQ-KMeans | 55 | 0.02741 | 0.03855 | 0.01894 | 0.02255 | 1.0000 | 0.00706 | 4.28 h |
| RQ-VAE | 55 | 0.03430 | 0.05169 | 0.02295 | 0.02854 | 1.0000 | 0.00720 | 4.34 h |

![三种 Tokenizer 的推荐指标](report_assets/recommendation_metrics.png)

结果事实：

- 在当前 SID-level、seed 2024 条件下，PSE 的 HR 和 NDCG 最高，RQ-VAE 居中，RQ-KMeans 最低。
- 三组 Legal@10 都为 1，说明 Top-10 SID 均可映射到合法编码。
- PSE 的 Top-10 重复率最低；RQ-KMeans 和 RQ-VAE 的 Duplicate@10 均约为 0.7%。
- 三组 Fit Time 和 Peak VRAM 基本一致，因此下游结果差异不是由明显不同的训练预算造成的。

必须保留的解释限制：

> PSE 的完整 SID 碰撞率为 17.95%，显著高于 RQ-KMeans 和 RQ-VAE。在 SID-level 评估中，共享目标 SID 的物品无法被区分，因此 PSE 更高的 HR/NDCG 可能同时包含“推荐能力”和“标签粒度更粗”两部分影响。当前结果不能直接推出 PSE 的严格 item-level 推荐效果最好。

## 5. 逐位置生成难度

### 5.1 最佳 Epoch 的 Train OCN 诊断

Train NLL 只统计当前 OCN View 中被 Mask 的位置，不同位置的 Mask Rate 不完全相同。

| Tokenizer | SID0 NLL | SID1 NLL | SID2 NLL | SID3 NLL |
|---|---:|---:|---:|---:|
| PSE | 3.108 | 3.238 | 3.275 | 3.377 |
| RQ-KMeans | 3.682 | 3.664 | 3.622 | 3.603 |
| RQ-VAE | 3.365 | 3.692 | 3.739 | 3.789 |

训练阶段的位置结构不同：PSE 和 RQ-VAE 的后部位置更难；RQ-KMeans 的四个位置更接近，并呈轻微的后部位置更容易趋势。

### 5.2 Test Fully Masked 诊断

测试时把四个 SID 位置放在同一个 Fully Masked 上下文中，统一测量“没有任何目标 SID 已揭示时”的位置难度。

| Tokenizer | 指标 | SID0 | SID1 | SID2 | SID3 |
|---|---|---:|---:|---:|---:|
| PSE | NLL | 6.448 | 6.472 | 6.381 | 6.469 |
| PSE | Accuracy | 0.53% | 0.53% | 0.76% | 0.54% |
| RQ-KMeans | NLL | **5.606** | **5.862** | **5.830** | **5.881** |
| RQ-KMeans | Accuracy | **1.35%** | **1.04%** | **1.27%** | **1.20%** |
| RQ-VAE | NLL | 6.901 | 6.972 | 6.842 | 6.915 |
| RQ-VAE | Accuracy | 0.71% | 0.46% | 0.53% | 0.45% |

| PSE | RQ-KMeans | RQ-VAE |
|---|---|---|
| ![](report_assets/position_nll_pse.png) | ![](report_assets/position_nll_rq_kmeans.png) | ![](report_assets/position_nll_rqvae.png) |

阶段性观察：

- RQ-KMeans 在 Fully Masked 条件下的四个位置 NLL 最低、Accuracy 最高，但其最终 HR/NDCG 最低。
- 因此 Fully Masked 单步预测能力不能直接等价为 CPD Beam Search 推荐能力。
- 一个可能原因是，正式 CPD 会逐步揭示部分 SID 并进行组合搜索；另一个重要混杂因素是三组 SID 碰撞率不同。
- Fully Masked NLL 适合比较“无目标 SID 上下文时的位置难度”，不宜作为下游推荐效果的单一代理指标。

训练 OCN 与 Test Fully Masked 的上下文不同，因此也不能把两者简单相减后解释为普通的训练—测试泛化差距。

## 6. CPD 动态解码顺序

DiffGRM 不固定按照 SID0→SID1→SID2→SID3 解码。每一轮同时预测尚未生成的位置，并优先选择置信度最高的位置。

![三种 Tokenizer 的 CPD 解码顺序](report_assets/cpd_decode_order.png)

### 6.1 各位置平均解码 Rank

平均 Rank 越小，表示该位置通常越早被生成。

| Tokenizer | SID0 | SID1 | SID2 | SID3 |
|---|---:|---:|---:|---:|
| PSE | 2.532 | 2.504 | **2.442** | 2.522 |
| RQ-KMeans | 2.339 | **2.325** | 2.584 | 2.751 |
| RQ-VAE | **2.370** | 2.466 | 2.623 | 2.541 |

阶段性观察：

- PSE 的位置顺序比较均衡，SID2 略早；第一轮更常选择 SID2/SID3。
- RQ-KMeans 明显更倾向先生成 SID0/SID1，再生成 SID2/SID3。
- RQ-VAE 更倾向先生成 SID0，SID2 相对更晚。
- 最大单格频率约为 0.30～0.33，说明三组都只是形成位置偏好，而不是所有样本采用同一个固定顺序。

这说明 Tokenizer 改变 Semantic ID 空间后，DiffGRM 对“哪个位置更容易先确认”的判断也会发生变化。

## 7. Decoder Self-Attention

### 7.1 统计口径

Attention 使用各组最佳 checkpoint 和完整测试集 22,363 个样本，沿 Beam Width 1 的 Greedy CPD 轨迹统计。捕获的是 Decoder Self-Attention 在 Masked Softmax 之后、Attention Dropout 之前的概率。

该统计用于机制诊断，不会改变正式 Beam 256 推荐指标；Attention 权重也不应直接解释为因果贡献。

### 7.2 对已生成与未生成位置的注意力

![对 Revealed 与 Masked 位置的注意力](report_assets/masked_revealed_attention.png)

| Tokenizer | Rank 2 Revealed | Rank 3 Revealed | Rank 4 Revealed |
|---|---:|---:|---:|
| 均匀基线 | 0.250 | 0.500 | 0.750 |
| PSE | **0.306** | **0.560** | **0.787** |
| RQ-KMeans | 0.276 | 0.532 | 0.772 |
| RQ-VAE | 0.268 | 0.522 | 0.765 |

三组都对已经生成的位置给予高于均匀基线的注意力，说明后续解码会利用前面已经生成的 SID。偏好强度在当前 seed 下表现为：

```text
PSE > RQ-KMeans > RQ-VAE
```

### 7.3 逐 Decoder Layer 分析

![逐层 Revealed Attention 偏差](report_assets/layer_attention_bias.png)

对已生成位置的额外偏好主要来自 Layer 1：PSE 最强，RQ-KMeans 次之，RQ-VAE 较弱；Layer 2～4 的平均 Self-Attention 更接近均匀分布。

这不能说明后面三层没有作用。它们仍然包含 Cross-Attention、FFN、残差连接，并会处理从第一层传递的隐藏表示。此外，不同 Head 的偏好可能在平均后相互抵消，Head-level 结果需要作为后续补充检查。

### 7.4 具体 Query-Key 位置偏差

下图展示 `Attention - 0.25`。红色表示高于四位置均匀基线，蓝色表示低于均匀基线。

![逐 Query-Key Attention 偏差](report_assets/position_attention_deviation.png)

随着解码推进，当前仍为 MASK 的 Query 位置对自身的注意力逐渐低于 0.25，而对其他已生成位置的注意力上升。该模式在 PSE 中最明显，在 RQ-KMeans 和 RQ-VAE 中较弱，与 Revealed/Masked 汇总结果一致。

此图信息较密，适合作为附图；正文汇报优先使用前两张 Attention 图。

## 8. 结果之间的联合关系

| Tokenizer | SID空间特征 | Fully Masked难度 | 优先解码位置 | 对Revealed依赖 | SID-level推荐结果 |
|---|---|---|---|---|---|
| PSE | 边际使用均衡，但完整碰撞率高 | 中等 | SID2/SID3 略早 | 最强 | 最高 |
| RQ-KMeans | 前层利用率较低，完整碰撞率最低 | 最容易 | SID0/SID1 较早 | 中等 | 最低 |
| RQ-VAE | 边际最均衡，完整碰撞率较低 | 最难 | SID0 较早 | 最弱 | 居中 |

当前结果支持以下阶段性判断：

1. **Tokenizer 确实改变了 DiffGRM 的生成行为。** 三组在逐位置难度、解码顺序和 Attention 依赖上均表现出不同模式。
2. **边际码本均衡不保证组合唯一。** PSE 四个位置利用率均为 100%，但完整 SID 碰撞率仍达到 17.95%。
3. **Fully Masked NLL 不能单独预测推荐效果。** RQ-KMeans 的 Fully Masked NLL 最低，但当前 SID-level HR/NDCG 最低。
4. **DiffGRM 会利用已生成 SID。** 三组在 Rank 2～4 都对 Revealed 位置表现出高于均匀基线的注意力，且主要由第一层建立。
5. **当前不能把 PSE 的最高指标直接解释成最好的 item-level 推荐。** PSE 的 SID 标签粒度更粗，碰撞可能提高 SID-level 命中概率。

因此，现阶段更合理的结论不是“某个 Tokenizer 已经胜出”，而是：

> Semantic ID 的碰撞结构、逐位置生成难度和 Decoder 的动态信息利用方式存在明显耦合。评价 Tokenizer 时必须同时观察空间结构、生成机制和严格推荐指标，不能只看重构质量、逐位置 Loss 或 SID-level HR/NDCG 中的任意一项。

## 9. 当前限制

1. 当前只有 Decoder seed 2024，无法判断差异是否具有随机种子稳定性。
2. 当前采用 SID-level 评估；不同 Tokenizer 的 SID 碰撞率差异较大，可能影响 HR/NDCG 的可比性。
3. Attention 统计使用 Greedy CPD 轨迹，而正式推荐使用 Beam 256；Attention 是机制诊断，不是正式搜索路径的完全复刻。
4. 当前只统计 Decoder Self-Attention，尚未系统比较 Cross-Attention 对用户历史的依赖。
5. AR 阶段使用 3 位语义 SID + 碰撞后缀和严格 item-level 评估；本阶段为 4 位 SID-level，二者不能直接比较绝对指标。
6. 当前未对三种 Tokenizer 完成完全同口径的重构误差与近邻保持分析，因此本报告不比较它们的 Encoder-side 重构能力。

## 10. 下一步实验建议

按优先级建议：

### 第一优先级：补充严格 item-level 评估

在相同 Top-K 解码结果上要求预测商品 ID 与目标商品 ID 完全一致，避免多个商品共享 SID 时被视为同一个正确答案。这一步用于判断 PSE 的高 SID-level 指标中有多少来自较高碰撞率。

### 第二优先级：补充 Decoder seeds 2025、2026

对三种 Tokenizer 使用相同的 Decoder seeds，报告 Mean ± Std，验证以下现象是否稳定：

- 推荐指标排序；
- 逐位置 NLL；
- CPD 平均解码 Rank；
- Revealed Attention 偏好。

### 第三优先级：补充 Head-level 与 Cross-Attention

- 检查 Layer 2～4 是否所有 Head 都接近均匀，还是不同 Head 的偏好相互抵消；
- 分析生成某个 SID 时，Cross-Attention 主要关注用户历史中的哪些 Item/位置；
- 为后续 session-level 生成分析建立统计口径。

### 第四优先级：统一 AR 与 DiffGRM 的对照口径

让 AR 和 DiffGRM 使用同一套 4×256 Semantic ID、相同数据划分和严格 item-level 评估，再比较同一个 Tokenizer 在固定顺序生成和动态顺序生成中的表现。

## 附录：训练与验证曲线

### PSE

| 训练 Loss / LR | Validation HR / NDCG | Fully Masked NLL |
|---|---|---|
| ![](report_assets/training_pse.png) | ![](report_assets/validation_pse.png) | ![](report_assets/full_mask_nll_pse.png) |

### RQ-KMeans

| 训练 Loss / LR | Validation HR / NDCG | Fully Masked NLL |
|---|---|---|
| ![](report_assets/training_rq_kmeans.png) | ![](report_assets/validation_rq_kmeans.png) | ![](report_assets/full_mask_nll_rq_kmeans.png) |

### RQ-VAE

| 训练 Loss / LR | Validation HR / NDCG | Fully Masked NLL |
|---|---|---|
| ![](report_assets/training_rqvae.png) | ![](report_assets/validation_rqvae.png) | ![](report_assets/full_mask_nll_rqvae.png) |

---

本报告中的结果均来自三组 seed 2024 正式最佳 checkpoint。所有 Attention 统计均通过 Softmax 行和校验，最大误差不超过 `1.35×10⁻⁷`。
