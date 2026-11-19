# Cross-PSE + DiffGRM：DIGER 式交替联合训练

## 研究目的

检验“带跨码本交互的 SID 是否只有在推荐损失能够直接更新 Tokenizer
时才真正有用”。本实验与上一轮 SIA pilot 使用相同的 Cross-PSE v3、
DiffGRM 初始化、4×256 结构、数据集和评估协议，只替换梯度耦合方式。

## 梯度路径

Tokenizer phase 中，Cross-PSE 为每个商品、每个 SID 位置计算码字分配：

1. 距离得到 assignment logits；
2. 加入随训练衰减的 Gumbel 扰动；
3. softmax 得到软分配；
4. 前向取 argmax one-hot，保证模型看到的是离散 SID；
5. 反向使用 straight-through soft distribution。

软分配用于三条推荐梯度路径：

- 历史商品的 SID embedding；
- DiffGRM 中已经揭示的目标 SID embedding；
- 目标 SID 的 soft/hard straight-through 交叉熵标签。

因此 Decoder 即使被冻结，推荐 Loss 仍能经过它回传到 Cross-PSE。

## 交替训练

- Recommender phase：冻结 Cross-PSE，只更新 DiffGRM；使用确定性硬 SID。
- Tokenizer phase：冻结 DiffGRM，只更新 Cross-PSE；目标为推荐 Loss、重构
  Loss、码本均衡和锚定约束之和。
- 每个 Tokenizer phase 结束后，确定性刷新全目录硬 SID，再进行评估和后续
  Recommender phase。

SIA 权重在本实验中设为 0，避免把“间接对齐收益”混入“直接推荐梯度收益”。

## 必须通过的 smoke 判据

1. hard-forward 输出严格为 one-hot；
2. `diger_direct_gradient_norm > 0`；
3. Tokenizer phase 中 Decoder 梯度为 0、Cross-PSE 梯度大于 0；
4. Recommender phase 中结论相反；
5. Loss、梯度、assignment entropy 均有限；
6. SID refresh 后无明显码本塌缩或碰撞率突增。

## 运行

```bash
bash scripts/run_diff_crosspse_diger.sh preflight 2024 python
bash scripts/run_diff_crosspse_diger.sh smoke 2024 python
bash scripts/run_diff_crosspse_diger.sh formal 2024 python
bash scripts/run_diff_crosspse_diger.sh summarize 2024 python
```

该版本的结果目录和 run ID 均包含 `diger_st`，不会覆盖上一轮 SIA pilot。
