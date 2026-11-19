# AR-GRM 第一阶段五组 Tokenizer 实验汇总（seed 2024）

## 1. 实验设置

### 1.1 研究问题

在同一套 AR 生成推荐框架下，比较不同 Tokenizer 及不同码本容量分配对逐位置生成难度、SID 空间结构和最终推荐效果的影响。

### 1.2 五组对照

| 编号 | Tokenizer | 三位 Semantic ID 码本 | 对照目的 |
|---|---|---|---|
| A | PSE | 256×256×256 | 与 RQ 类 Tokenizer 比较 |
| B | RQ-KMeans | 128×256×512 | 将容量更多分配到后部位置 |
| C | RQ-KMeans | 256×256×256 | 均匀容量基线 |
| D | RQ-KMeans | 512×256×128 | 将容量更多分配到前部位置 |
| E | RQ-VAE | 256×256×256 | 与 RQ-KMeans、PSE 比较 |

三种 RQ-KMeans 布局的总组合容量均为 \(2^{24}\)，因此 B、C、D 只改变容量在三个位置之间的分配。

### 1.3 统一实验口径

| 项目 | 设置 |
|---|---|
| 数据集 | Amazon Reviews 2014 - Beauty |
| 物品数 | 12,101 |
| 输入表征 | 共享的 sentence-t5-xl 物品向量 |
| 推荐模型 | 相同 AR-GRM 训练及解码流程 |
| SID 长度 | 3 位 Semantic ID + 1 位碰撞 suffix |
| 随机种子 | 2024 |
| 推荐评估 | 严格 item-level，解码 item 与目标 item 完全一致才算命中 |
| 推荐指标 | HR@5/10、NDCG@5/10、legal@10、duplicate@10 |
| 逐位置指标 | NLL、normalized NLL、accuracy |
| checkpoint | 按验证集 weighted score 选择最佳 epoch，再进行测试 |

不同位置的码本大小可能不同，因此使用以下指标进行公平比较：

\[
\text{normalized NLL}_i=\frac{\text{NLL}_i}{\log K_i},
\]

其中 \(K_i\) 是位置 \(i\) 的码本大小。normalized NLL 越小，表示相对于该位置的类别数越容易预测。



## 2. SFT 阶段逐位置 Loss

以下曲线展示各次验证时 SID0、SID1、SID2 的 normalized NLL。碰撞 suffix 不属于 Semantic ID，未画入曲线，其测试结果在第 3 节单独列出。

### 2.1 PSE 256×256×256

![PSE 256×256×256 逐位置 SFT Loss](report_assets/val_position_curve_pse_256x256x256.png)

### 2.2 RQ-KMeans 128×256×512

![RQ-KMeans 128×256×512 逐位置 SFT Loss](report_assets/val_position_curve_rqk_128x256x512.png)

### 2.3 RQ-KMeans 256×256×256

![RQ-KMeans 256×256×256 逐位置 SFT Loss](report_assets/val_position_curve_rqk_256x256x256.png)

### 2.4 RQ-KMeans 512×256×128

![RQ-KMeans 512×256×128 逐位置 SFT Loss](report_assets/val_position_curve_rqk_512x256x128.png)

### 2.5 RQ-VAE 256×256×256

![RQ-VAE 256×256×256 逐位置 SFT Loss](report_assets/val_position_curve_rqvae_256x256x256.png)

### 2.6 Loss 曲线的主要现象

- 五组均存在位置间难度差异，但差异大小和顺序受 Tokenizer 影响。
- 三种 RQ-KMeans 中，512×256×128 的 SID0 最难、SID2 最容易；128×256×512 则降低了 SID0 难度，但提高了 SID1、SID2 难度。
- PSE 的 SID1、SID2 较容易生成，但不能据此直接判断 item 更容易命中，因为其 Semantic ID 碰撞较多，更多区分任务被交给了 suffix。
- 验证 Loss 在最佳 epoch 后回升，说明后期出现不同程度的过拟合；最终测试均使用最佳 checkpoint，而不是最后一个 epoch。

## 3. 测试集逐位置结果

### 3.1 原始 NLL

| Tokenizer / 码本 | SID0 | SID1 | SID2 | Suffix |
|---|---:|---:|---:|---:|
| PSE 256×256×256 | 4.654 | 2.074 | 1.848 | 0.612 |
| RQ-KMeans 128×256×512 | 4.204 | 4.244 | 0.817 | 0.148 |
| RQ-KMeans 256×256×256 | 4.982 | 3.908 | 0.568 | 0.133 |
| RQ-KMeans 512×256×128 | 5.750 | 3.663 | 0.427 | 0.121 |
| RQ-VAE 256×256×256 | 4.122 | 4.716 | 1.161 | 0.121 |

### 3.2 Normalized NLL

| Tokenizer / 码本 | SID0 | SID1 | SID2 | Suffix |
|---|---:|---:|---:|---:|
| PSE 256×256×256 | 0.839 | 0.374 | 0.333 | 0.110 |
| RQ-KMeans 128×256×512 | 0.866 | 0.765 | 0.131 | 0.027 |
| RQ-KMeans 256×256×256 | 0.898 | 0.705 | 0.102 | 0.024 |
| RQ-KMeans 512×256×128 | 0.922 | 0.661 | 0.088 | 0.022 |
| RQ-VAE 256×256×256 | 0.743 | 0.850 | 0.209 | 0.022 |

![五组测试集逐位置 normalized NLL](report_assets/test_normalized_nll_seed2024.png)

### 3.3 逐位置 Accuracy

| Tokenizer / 码本 | SID0 | SID1 | SID2 | Suffix |
|---|---:|---:|---:|---:|
| PSE 256×256×256 | 9.43% | 41.25% | 48.95% | 81.85% |
| RQ-KMeans 128×256×512 | 10.57% | 11.85% | 78.18% | 97.52% |
| RQ-KMeans 256×256×256 | 7.25% | 15.89% | 85.71% | 98.09% |
| RQ-KMeans 512×256×128 | 5.28% | 19.80% | 90.08% | 98.67% |
| RQ-VAE 256×256×256 | 10.44% | 10.86% | 68.86% | 98.87% |

测试集结果说明：

- 码本容量前移会使 SID0 变难、后部位置变容易，位置间的信息差可以被人为控制。
- PSE 的前三位 Loss 较低，但 suffix NLL 为 0.612，明显高于其他四组；这与其更高的前三位碰撞率相对应。
- 因此判断“是否容易生成”时，必须同时看前三位 Semantic ID 和 suffix，不能只看某一位 Loss。

## 4. 最终推荐结果

| Tokenizer / 码本 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | legal@10 | duplicate@10 |
|---|---:|---:|---:|---:|---:|---:|
| PSE 256×256×256 | 3.734% | **5.956%** | 2.347% | 3.053% | 100.000% | 0.000% |
| RQ-KMeans 128×256×512 | 3.510% | 5.621% | 2.327% | 3.003% | 100.000% | 0.000% |
| **RQ-KMeans 256×256×256** | **3.765%** | 5.777% | **2.463%** | **3.111%** | 100.000% | 0.000% |
| RQ-KMeans 512×256×128 | 3.403% | 5.491% | 2.216% | 2.883% | 99.983% | 0.000% |
| RQ-VAE 256×256×256 | 3.242% | 5.259% | 2.118% | 2.763% | 99.976% | 0.000% |

![五组严格 item-level 推荐指标](report_assets/recommendation_metrics_seed2024.png)

### 4.1 固定 3×256 的 Tokenizer 对照

- PSE 的 HR@10 最高。
- RQ-KMeans 的 HR@5、NDCG@5、NDCG@10 最高。
- PSE 与 RQ-KMeans 的结果接近，且 seed 2024 下均高于 RQ-VAE。

### 4.2 固定 RQ-KMeans 的容量分配对照

以 256×256×256 为基线：

| 布局 | HR@5 变化 | HR@10 变化 | NDCG@5 变化 | NDCG@10 变化 |
|---|---:|---:|---:|---:|
| 128×256×512 | -6.77% | -2.71% | -5.53% | -3.44% |
| 512×256×128 | -9.62% | -4.95% | -10.06% | -7.31% |

当前 seed 2024 中，均匀布局 256×256×256 的综合推荐效果最好。两种非均匀布局确实改变了逐位置 Loss，但没有转化为推荐收益，其中 512×256×128 的下降更明显。

## 5. 必要的 Tokenizer 指标

### 5.1 SID 前缀碰撞率

碰撞率表示有多少 item 与其他 item 共享相同的前 \(k\) 位 SID。数值越高，说明生成到该位置时仍有更多 item 无法区分。

| Tokenizer / 码本 | 前 1 位 | 前 2 位 | 前 3 位（完整 Semantic ID） | 最大碰撞组 |
|---|---:|---:|---:|---:|
| PSE 256×256×256 | 99.97% | 86.03% | 31.32% | 51 |
| RQ-KMeans 128×256×512 | 100.00% | 49.23% | 5.56% | 9 |
| RQ-KMeans 256×256×256 | 100.00% | 32.00% | 3.66% | 5 |
| RQ-KMeans 512×256×128 | 100.00% | 20.48% | **2.74%** | 4 |
| RQ-VAE 256×256×256 | 100.00% | 66.16% | 2.85% | 5 |

- PSE 的三位 SID 碰撞率最高，因此依赖 suffix 区分更多 item。
- 512×256×128 的碰撞率最低，却没有获得最佳推荐效果。
- 这说明降低碰撞有帮助，但碰撞率不能单独决定下游推荐性能。

### 5.2 逐位置码本利用率

| Tokenizer / 码本 | SID0 | SID1 | SID2 |
|---|---:|---:|---:|
| PSE 256×256×256 | 100.0% | 100.0% | 100.0% |
| RQ-KMeans 128×256×512 | 100.0% | 100.0% | 100.0% |
| RQ-KMeans 256×256×256 | 100.0% | 100.0% | 100.0% |
| RQ-KMeans 512×256×128 | 100.0% | 100.0% | 100.0% |
| RQ-VAE 256×256×256 | 37.1% | 64.1% | 95.7% |

利用率 100% 只表示该位置的每个码字至少被一个 item 使用，并不表示各码字使用均匀。RQ-VAE 的前两层存在较多未使用码字。

### 5.3 逐位置归一化熵

| Tokenizer / 码本 | SID0 | SID1 | SID2 |
|---|---:|---:|---:|
| PSE 256×256×256 | 0.954 | 0.947 | 0.957 |
| RQ-KMeans 128×256×512 | 0.987 | 0.985 | 0.962 |
| RQ-KMeans 256×256×256 | 0.987 | 0.984 | 0.976 |
| RQ-KMeans 512×256×128 | 0.985 | 0.980 | 0.985 |
| RQ-VAE 256×256×256 | 0.808 | 0.904 | 0.984 |

归一化熵越接近 1，表示码字使用越均匀。RQ-KMeans 三种布局整体较均匀；PSE 虽然利用率为 100%，均匀程度略低；RQ-VAE 的 SID0 分布最不均匀。

## 6. 汇总结论

1. 不同 Tokenizer 会形成不同的逐位置信息结构，AR decoder 对这些空间的适配结果也不同。
2. RQ-KMeans 的位置容量分配能够直接调节逐位置 Loss，但当前两种非均匀布局均未超过均匀的 256×256×256。
3. PSE 的前三位较容易生成，但碰撞率与 suffix Loss 较高，说明它把一部分 item 区分压力后移到了 suffix。
4. 低碰撞率、高码本利用率或较低的单位置 Loss，都不能单独保证更好的推荐效果，需要结合严格 item-level HR/NDCG 判断。
5. 以上新增对照目前只有 seed 2024，只能作为阶段性趋势；若要形成稳定结论，需要补跑 seed 2025、2026 后报告 mean ± std。

## 附录：Decoder Attention

Attention 不属于 Tokenizer 本身的质量指标，因此不放入第 5 节主表。它用于补充观察 decoder 生成 SID2 时如何利用 BOS、SID0 和 SID1。

### A.1 原 3×256 基线（RQ-KMeans、RQ-VAE，3 seeds 均值 ± 标准差）

![原 3×256 基线生成 SID2 时的逐层 Attention](report_assets/sid2_attention_by_layer_baselines.png)

### A.2 新增三组（seed 2024）

![新增三组生成 SID2 时的逐层 Attention](report_assets/sid2_attention_by_layer.png)

五组的共同现象是：Layer 1 主要关注最近的 SID1 和 SID0，Layer 2～4 则明显转向 BOS 与 SID0。不同 Tokenizer 和容量分配改变了注意力强度，但没有改变这一总体层间模式。Attention 只能作为 decoder 适配方式的描述性证据，不能单独解释推荐指标变化。
