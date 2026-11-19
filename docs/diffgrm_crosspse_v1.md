# DiffGRM + Cross-PSE v1

## 1. 实验问题

在保持 DiffGRM、数据、4×256 SID 布局和原始 PSE 初始化不变的条件下，只加入一次跨码本 self-attention，检验“码本在量化前发生信息交互”是否能改善 SID 表征与下游生成推荐。

## 2. Cross-PSE v1

1. 复用原始 PSE 流程：Sentence-T5-base 768 维表示，经训练集拟合的 PCA（whiten）降到 256 维并归一化，再拟合 `OPQ4,IVF1,PQ4x8`。
2. 从 FAISS 倒排表读取原始 PSE 的 4 个编码，并在 IVF 粗中心对应的残差空间中恢复四本 `[256, 64]` 码本。
3. 将四本码本的 1024 个词向量排成一列，加入码本身份和码字身份向量，经过一层 4-head self-attention。
4. 用残差形式得到新码本：`C_cross = C_base + alpha * Attention(C_base)`。
5. 原始 PSE 码本固定，只训练 Mixer；每个商品仍在四本新码本中分别寻找最近码字，得到 4×256 SID。

## 3. 控制变量

保持不变：Beauty 数据与划分、item embedding/PCA、PSE 拟合集、4×256 布局、DiffGRM 网络、OCN/CPD、训练批量、学习率调度、评估协议和 decoder seed。

唯一主要变化：原 PSE 码本之后增加跨码本 Mixer，并据此重新分配商品 SID。

## 4. Tokenizer 参数

| 参数 | v1 设置 | 含义 |
|---|---:|---|
| `crosspse_attention_scope` | `full` | 1024 个码字均可互相注意，包含层内与跨层交互 |
| `crosspse_heads` | 4 | self-attention 头数；每个码字向量 64 维，每头 16 维 |
| `crosspse_dropout` | 0 | 离线 tokenizer 小模型不使用 dropout，减少额外随机性 |
| `crosspse_residual_scale_init` | 0 | 初始时严格等于原 PSE，训练后由可学习的 `alpha` 控制改变量 |
| `crosspse_train_base_codebooks` | `false` | 冻结原始 OPQ/PQ 码本，避免把“码本重训”和“跨码本交互”混在一起 |
| `crosspse_lr` | 1e-3 | Mixer 的 AdamW 学习率 |
| `crosspse_weight_decay` | 1e-4 | Mixer 权重衰减 |
| `crosspse_max_steps` | 5000 | tokenizer 最大更新步数 |
| `crosspse_warmup_ratio` | 0.05 | 前 250 步 warmup，之后 cosine 衰减 |
| `crosspse_batch_size` | 1024 | 每步用于重构训练的商品数 |
| `crosspse_eval_interval` | 100 | 每 100 步在 PSE 拟合集的补集上验证 |
| `crosspse_patience` | 10 | 验证重构连续 10 次不改善则早停 |
| `crosspse_balance_weight` | 0.01 | 防止码字使用过度集中 |
| `crosspse_balance_temperature` | 0.1 | 计算软码字分配时的温度 |
| `crosspse_anchor_weight` | 0.1 | 约束新码本不要偏离原 PSE 过远 |
| `crosspse_mixed_precision` | `bf16` | RTX 3080 上的 tokenizer 混合精度 |

Tokenizer 的总目标为：

`重构损失 + 0.01 × 码本均衡损失 + 0.1 × 原码本锚定损失`。

这些参数是第一版可控起点，不应在看到下游测试集结果后反复调参。若 v1 能跑通，后续应使用验证集统一调整，再固定参数比较。

## 5. DiffGRM 参数

沿用 PSE v6 基线：encoder 1 层、decoder 4 层、4 heads、hidden 256、FFN 1024、BF16；物理 batch 64、梯度累积 16、有效 batch 1024；AdamW 学习率 0.01，warmup 10000 updates 后 cosine；最多 200 epochs；从 epoch 20 开始每 5 epochs 验证；测试 beam 256；SID-level HR/NDCG。

## 6. 运行顺序

```bash
cd ~/experiment1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate grm-exp

bash scripts/run_diff_crosspse_v1.sh preflight 2024 python
bash scripts/run_diff_crosspse_v1.sh tokenizer-profile 2024 python
bash scripts/run_diff_crosspse_v1.sh tokenizer 2024 python
bash scripts/run_diff_crosspse_v1.sh smoke 2024 python
bash scripts/run_diff_crosspse_v1.sh profile 2024 python
```

上述检查均通过后，再在 tmux 中执行正式实验：

```bash
bash scripts/run_diff_crosspse_v1.sh formal 2024 python
```

正式训练结束后：

```bash
bash scripts/run_diff_crosspse_v1.sh attention 2024 python
bash scripts/run_diff_crosspse_v1.sh summarize 2024 python
bash scripts/run_diff_crosspse_v1.sh compare 2024 python
```

## 7. 必看结果

Tokenizer：重构误差/余弦相似度、相对原 PSE 的 SID 改变率、逐位置利用率/熵、前缀与完整 SID 碰撞率、跨码本 attention mass、几何邻居保持率。

DiffGRM：逐位置 NLL/accuracy、逐位置 CPD 解码先后、HR@5/10、NDCG@5/10、legal/duplicate，以及 decoder attention。所有下游结论均与同 seed 的原始 PSE 4×256 v6 基线比较。
