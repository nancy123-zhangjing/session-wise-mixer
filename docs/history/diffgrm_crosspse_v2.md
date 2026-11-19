# DiffGRM + Cross-PSE v2

## 目标

在原始 PSE 与 DiffGRM v6 均不改变的条件下，为四本 256 码字的 PSE 码本加入一次全局 self-attention，避免 v1 因零初始化可学习门控而退化成原 PSE。

## 相对 v1 的修改

| 项目 | v1 | v2 |
|---|---:|---:|
| 原 PSE 码本 | 冻结 | 冻结 |
| 初始 Cross-PSE 输出 | 等于 PSE | 等于 PSE |
| residual scale | 可学习，从0开始 | 固定为1 |
| Attention 输出投影 | 常规初始化 | 权重和偏置初始化为0 |
| Mixer 验证集 | PSE 拟合集之外的40件商品 | 从12061件拟合商品中固定抽取10%，约1206件 |
| 外部40件商品 | 用于选 checkpoint | 只用于额外诊断 |
| anchor weight | 0.1 | 0.01 |

固定 `alpha=1` 并将 Attention 输出层置零，使初始码本仍严格等于原 PSE，同时第一步就能向 Attention 输出层传递有效梯度。原始 PSE 码本仍冻结，因此主要干预仍是跨码本 Mixer。

## 保持不变

- Amazon Reviews 2014 Beauty、leave-one-out 与训练样本；
- Sentence-T5-base、PCA whiten 256维、归一化；
- `OPQ4,IVF1,PQ4x8` 原始 PSE 底座与 4×256 SID；
- Cross-PSE seed 2024、DiffGRM decoder seed 2024；
- DiffGRM v6 的网络、OCN/CPD、BF16、有效 batch 1024、学习率调度、早停、beam 和 SID-level 评估。

## Tokenizer 参数

```yaml
crosspse_attention_scope: full
crosspse_heads: 4
crosspse_residual_scale_init: 1.0
crosspse_learn_residual_scale: false
crosspse_zero_init_attention_output: true
crosspse_train_base_codebooks: false
crosspse_validation_fraction: 0.1
crosspse_split_seed: 2024
crosspse_lr: 0.001
crosspse_max_steps: 5000
crosspse_eval_interval: 100
crosspse_patience: 10
crosspse_balance_weight: 0.01
crosspse_anchor_weight: 0.01
```

## 运行顺序

```bash
bash scripts/run_diff_crosspse_v2.sh preflight 2024 python
bash scripts/run_diff_crosspse_v2.sh tokenizer-profile 2024 python
bash scripts/run_diff_crosspse_v2.sh tokenizer 2024 python
bash scripts/run_diff_crosspse_v2.sh inspect 2024 python
bash scripts/run_diff_crosspse_v2.sh smoke 2024 python
bash scripts/run_diff_crosspse_v2.sh profile 2024 python
bash scripts/run_diff_crosspse_v2.sh formal 2024 python
bash scripts/run_diff_crosspse_v2.sh attention 2024 python
bash scripts/run_diff_crosspse_v2.sh summarize 2024 python
bash scripts/run_diff_crosspse_v2.sh compare 2024 python
```

正式 DiffGRM 前必须确认：PSE 基线匹配率为100%，验证集约1206件商品，码本 delta 和 SID 改变率不再接近0，四个位置没有码本坍缩，且 tokenizer artifact 检查通过。
