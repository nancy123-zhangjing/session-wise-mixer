# PSE 码本交互实验：阶段性结果

## 1. 研究问题

本阶段希望验证：**在 PSE 中显式加入跨码本信息交互，能否得到更适合生成模型解码的 Semantic ID，并改善 DiffGRM 的逐位置生成表现和推荐效果。**

为了把“重新训练码本”和“加入码本交互”两个因素分开，设置了三组对照：

| 方法 | PSE 码本是否训练 | 是否加入跨码本 Self-Attention | 对照目的 |
|---|---:|---:|---|
| Original PSE | 否 | 否 | 原始基线 |
| Refined-PSE | 是 | 否 | 观察重新训练码本本身的影响 |
| Cross-PSE | 是 | 是 | 观察码本交互相对 Refined-PSE 的额外作用 |

因此，判断“码本交互是否有效”时，最关键的是 **Refined-PSE 与 Cross-PSE 的比较**，而不只是 Original PSE 与 Cross-PSE 的比较。

## 2. 实验设置

三组实验保持以下条件一致：

- Amazon Reviews 2014 Beauty 数据；
- Sentence-T5-base 物品表征、PCA/OPQ 处理；
- 4 层、每层 256 个码字的 SID；
- Tokenizer seed 与 DiffGRM seed 均为 2024；
- 相同的 DiffGRM 网络、OCN/CPD 训练流程、batch、学习率调度和 BF16 设置；
- 测试指标采用当前 DiffGRM 复现协议下的 SID-level HR、NDCG、legal 和 duplicate。

Refined-PSE 与 Cross-PSE 使用完全相同的原始 PSE 初始化、训练商品和优化目标，两者唯一的核心差异是是否启用跨码本 Self-Attention；

## 3. 正式实验结果

### 3.1 Tokenizer 确实发生了变化，但优化幅度较小

| 指标 | Original PSE | Refined-PSE | Cross-PSE |
|---|---:|---:|---:|
| 最佳 tokenizer step | — | 2900 | 2950 |
| 重构误差改善量 | — | 0.000193 | 0.000218 |
| 相对码本变化 RMS | — | 1.72% | 1.80% |
| 商品完整 SID 变化率 | — | 5.06% | 5.74% |
| 完整 SID 碰撞率 | 17.949% | 17.965% | 17.965% |
| 各位置码本利用率 | 100% | 100% | 100% |

两种可训练方法都改变了约 5% 的商品 SID，且没有出现码本坍缩，说明训练实际生效。Cross-PSE 的重构误差略优于 Refined-PSE，但两者改善量都很小。

前缀碰撞率也基本没有变化：

| 前缀长度 | Original PSE | Refined-PSE | Cross-PSE |
|---:|---:|---:|---:|
| SID0 | 99.983% | 99.983% | 99.983% |
| SID0–1 | 60.218% | 60.119% | 60.177% |
| SID0–2 | 27.105% | 27.254% | 27.328% |
| SID0–3 | 17.949% | 17.965% | 17.965% |

因此，Cross-PSE 后续效果下降不能简单归因于碰撞率上升或码本利用率下降。

Cross-PSE 的 Mixer 参数发生了非零更新，但平均跨码本 Attention 为 74.96%，几乎等于四个码本均匀注意时的 75%。各码本之间的平均权重大致分布在 23.1%–28.3%，说明当前交互更接近弱而均匀的全局混合，没有学出很明确的跨码本依赖结构。

### 3.2 最终推荐效果没有改善

| 方法 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | Legal@10 | Duplicate@10 |
|---|---:|---:|---:|---:|---:|---:|
| Original PSE | **0.05205** | 0.07490 | **0.03657** | 0.04392 | 1.0000 | **0.00077** |
| Refined-PSE | 0.05196 | **0.07571** | 0.03646 | **0.04409** | 1.0000 | 0.00085 |
| Cross-PSE | 0.05151 | 0.07360 | 0.03611 | 0.04320 | 1.0000 | 0.00135 |

![三组方法的推荐指标](report_assets/01_recommendation_metrics.png)

Refined-PSE 与 Original PSE 基本持平：HR@10 和 NDCG@10 分别提高约 0.00080 和 0.00017，但 HR@5、NDCG@5 略低，整体差异很小。

Cross-PSE 没有超过 Refined-PSE：

- HR@10 从 0.07571 降至 0.07360；
- NDCG@10 从 0.04409 降至 0.04320；
- Duplicate@10 也由 0.00085 增至 0.00135。

因此，当前 seed 下没有证据表明简单的全局 Self-Attention 码本交互能改善下游推荐。

### 3.3 Cross-PSE 的测试集逐位置 NLL 明显更高

| 方法 | SID0 | SID1 | SID2 | SID3 | 四位置平均 |
|---|---:|---:|---:|---:|---:|
| Original PSE | 6.448 | 6.472 | 6.381 | 6.469 | 6.442 |
| Refined-PSE | **6.217** | **6.262** | **6.216** | **6.193** | **6.222** |
| Cross-PSE | 6.794 | 6.805 | 6.718 | 6.773 | 6.772 |

![测试集逐位置 NLL](report_assets/02_position_nll.png)

Refined-PSE 在四个位置上的 fully-masked 测试 NLL 都低于原始 PSE，但 Cross-PSE 在四个位置上都明显更高。与此同时，Cross-PSE 的训练阶段 masked-view NLL 反而最低。由于训练和测试的 masking context 不完全相同，不能把它直接等同于标准训练—测试 gap；但在三组使用同一协议的条件下，这至少说明 Cross-PSE 学到的 SID 结构没有更好地迁移到 fully-masked 解码场景。

### 3.4 解码顺序和 Decoder Attention 没有显示更强的结构依赖

![CPD 解码顺序](report_assets/03_decode_order.png)

Original PSE 在第一轮更常选择 SID2/SID3，Refined-PSE 更明显地优先选择 SID2；Cross-PSE 第一轮相对偏向 SID3，但其余位置更接近均匀选择。当前结果没有显示 Cross-PSE 形成了更清晰、稳定的逐位置生成结构。

进一步统计 Decoder 对“已经解码位置”的 Attention，并减去均匀分布基线：

| 方法 | Decode rank 2 | Decode rank 3 | Decode rank 4 |
|---|---:|---:|---:|
| Original PSE | +5.64 pp | +5.98 pp | +3.69 pp |
| Refined-PSE | +8.05 pp | +7.88 pp | +4.67 pp |
| Cross-PSE | +1.24 pp | +1.69 pp | +1.24 pp |

![Decoder 对已解码位置的 Attention 偏好](report_assets/04_revealed_attention_bias.png)

Original PSE 和 Refined-PSE 的 Decoder 会明显增加对已解码 SID 的注意，而 Cross-PSE 更接近均匀注意。这一现象不能单独证明因果关系，但与 Cross-PSE 没有提升推荐效果、测试 NLL 更高的结果是一致的：当前交互没有让下游 Decoder 更充分地利用 SID 之间的依赖。

## 4. 阶段性结论

本轮实验得到的是一个较明确的负向结果：

1. Refined-PSE 说明重新训练码本本身没有明显破坏推荐效果，但提升非常有限。
2. Cross-PSE 的码本和 SID 确实发生了变化，实验也没有出现坍缩或碰撞率显著恶化，因此不能把结果不佳归因于实现未生效。
3. Cross-PSE 相比 Refined-PSE 的推荐指标下降，四个位置的测试 NLL 同时升高；当前简单的“将全部码字排列后做一层全局 Self-Attention”没有验证出额外价值。
4. 一个可能原因是当前 Mixer 学到的交互接近均匀混合，缺少针对不同码本角色的结构；另一个可能原因是 Tokenizer 只优化重构、均衡和锚定目标，与下游生成推荐目标并不完全一致。

由于目前只有 seed 2024，数值上的小差异还不能作为稳定统计结论；但逐位置 NLL、推荐指标与 Attention 诊断的方向一致，已经足以说明当前 Cross-PSE 设计不值得直接扩展到多 seed。


本阶段最直接的结论可以概括为：**码本交互的研究方向仍值得继续，但当前的一层全局 Self-Attention 更像弱而均匀的码本扰动，尚未形成对 DiffGRM 有帮助的结构化 SID 依赖。**
