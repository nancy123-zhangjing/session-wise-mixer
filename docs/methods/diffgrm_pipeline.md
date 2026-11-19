# DiffGRM 扩散生成推荐实验 Pipeline

论文：[DiffGRM: Diffusion-based Generative Recommendation Model](https://arxiv.org/abs/2510.21805)

## 1. 这一阶段要回答什么

先复现论文的 `PSE 4×256 + OCN + CPD` Beauty 基线，确认扩散模型可以正常训练和评估；随后再固定扩散模型，把 Tokenizer 换成 RQ-KMeans 和 RQ-VAE。

当前先做论文基线，不马上混入 RQ Tokenizer，避免“模型没复现好”和“Tokenizer 不合适”两个因素无法区分。

## 2. 模型与 AR 的区别

- AR：按固定的 `SID0 → SID1 → SID2 ...` 顺序预测；
- DiffGRM：把若干 SID 位置替换成 MASK，使用非因果 Self-Attention 并行预测；
- OCN 训练：根据模型置信度，把较难的位置优先加入 mask，形成嵌套的训练 views；
- CPD 推理：从全 MASK 开始，优先填入当前置信度最高的位置，因此不同样本的解码顺序可以不同。

## 3. 当前正式设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | Amazon Reviews 2014 - Beauty |
| Tokenizer | PSE（OPQ + 分块 PQ） |
| SID | 4 个位置 × 每位 256 个 code |
| 语义编码器 | sentence-t5-base |
| 模型 | Encoder 1 层，MD-Decoder 4 层，hidden 256，4 heads |
| OCN | least-confident first，static order，4 views |
| Loss reduction | 每个 view 内平均后，再对 views 等权平均 |
| CPD | Validation beam 32，Test beam 256 |
| 训练 | physical batch 64，gradient accumulation 4，effective batch 256 |
| 调度 | warmup 10,000 optimizer steps，最多 100 epochs，每 epoch 验证，patience 15 |
| 正式 run ID | `diff_formal_pse_4x256_schedfix_v2` |
| 最优模型 | `0.8 × NDCG@10 + 0.2 × Recall@10` |
| Seeds | 2024、2025、2026 |

论文 Table 7 给出的 Beauty label smoothing 是 0.1；代码仓库 README 示例写的是 0.2。当前正式配置以论文表格为准，并在结果中保留该说明。

## 4. 记录哪些指标

### 4.1 推荐指标

- HitRate/Recall@5、HitRate/Recall@10；
- NDCG@5、NDCG@10；
- legal@10；
- duplicate@10。
- PSE Semantic ID 的唯一数、碰撞物品率、碰撞组数和最大碰撞组；
- SID 前 1/2/3/4 位 prefix 的 item 碰撞率、碰撞组数和最大碰撞组。

论文复现基线采用 SID-level 指标。Tokenizer 替换实验会另外建立严格 item-level 对照，不能把两种口径直接放进同一张结果表。

### 4.2 逐位置 Loss

记录两种上下文：

1. `train / ocn_masked_views`
   - 只统计某位置实际被 OCN mask 时产生的监督；
   - 输出 smoothed CE、原始 NLL、normalized NLL、accuracy、mask rate；
   - 不同位置的上下文和监督次数可能不同。

2. `val/test / fully_masked`
   - 四个位置同时 MASK；
   - 所有位置使用完全一致的可见上下文；
   - 更适合直接比较 SID0～SID3 的固有预测难度。

`normalized NLL = NLL / log(256)`。

### 4.3 顺序统计

必须分成两个方向：

- `ocn_noising_hard_to_easy`：训练时 OCN 的加噪顺序，rank 1 表示最难、最先被 mask；
- `cpd_greedy_decode_easy_to_hard`：测试时 beam-width-one CPD 诊断路径，rank 1 表示置信度最高、最先被解码。

每种顺序输出：

- 每个位置出现在 rank 1/2/3/4 的频率；
- 每个位置的平均 rank；
- CPD 被选中时的平均 confidence、token accuracy 和 NLL。

正式 Top-K 指标仍使用配置的 CPD beam search。Beam-width-one 路径只用于得到容易解释的样本级顺序，不替代正式推荐评估。

### 4.4 Decoder Self-Attention

最佳 checkpoint 训练结束后，沿 beam-width-one CPD 路径统计 Decoder Self-Attention：

- decode rank 1/2/3/4；
- 当前被选中解码的 query SID 位置；
- 被关注的 key SID 位置；
- key 在该步是 `masked` 还是已经 `revealed`；
- 每层、每个 head 及跨层/跨 head 的平均 attention。

DiffGRM 的目标序列没有 BOS，也不是固定左到右解码，所以这里不能照搬 AR 的“生成 SID2 时关注 BOS/SID0/SID1”。对应问题改写为：“CPD 在第 r 步选择 SIDq 时，对四个 SID 位置分别分配多少注意力，其中多少给已解码位置、多少给仍 masked 的位置”。Attention 是 softmax 后、dropout 前的权重。

## 5. 服务器运行步骤

### 第零步：归档旧调度 pilot（只需运行一次）

旧 seed 2024 使用 accumulation 16，使 10,000-step warmup 占据约 96% 的训练。先进行非破坏性归档：

```bash
cd ~/experiment1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate grm-exp

bash scripts/run_diff_formal.sh archive-pilot 2024 python
bash scripts/run_diff_formal.sh attention-pilot-smoke 2024 python
```

归档目录：

```text
outputs/archive/diff_pilot_seed2024_warmup_misaligned_20260810/
```

第二条命令使用旧 checkpoint 的128个测试样本验证 attention 捕获代码；输出只作 smoke，不进入正式汇总。

### 第一步：同步后检查文件

```bash
cd ~/experiment1

test -f configs/diff_formal_common.yaml && echo "common config: OK"
test -f configs/diff_formal_pse.yaml && echo "PSE config: OK"
test -f configs/diff_smoke.yaml && echo "smoke config: OK"
test -f configs/diff_profile.yaml && echo "profile config: OK"
test -f scripts/run_diff_formal.sh && echo "runner: OK"
test -f scripts/preflight_diff_formal.py && echo "preflight: OK"
test -f scripts/summarize_diff_formal.py && echo "summary: OK"

grep -n "full_mask_diagnostics" \
  external/DiffGRM/genrec/models/DIFF_GRM/model.py
```

### 第二步：激活环境并做 preflight

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate grm-exp
cd ~/experiment1

bash scripts/run_diff_formal.sh preflight 2024 python
```

最后应出现：

```text
DiffGRM formal preflight: OK
```

### 第三步：运行两批 smoke test

```bash
bash scripts/run_diff_formal.sh smoke 2024 python
```

第一次运行可能需要下载 sentence-t5-base、生成 embedding 和训练 OPQ/PQ，这一段 GPU 利用率不一定持续很高。缓存生成后，后续 seed 会直接复用。

smoke 成功的标志：

- 出现 `Train Loss`；
- 出现 `fully_masked position diagnostics`；
- 出现 `ocn_noising_hard_to_easy mean rank`；
- 出现 `cpd_greedy_decode_easy_to_hard mean rank`；
- 最后出现 `Test Results:`；
- 没有 Traceback 或 CUDA OOM。

### 第四步：做短 profile

```bash
bash scripts/run_diff_formal.sh profile 2024 python
```

该阶段训练100个 batch、评估20个 batch；测试侧直接使用与正式实验相同的 beam 256，用来确认正式 batch 和 CPD beam 在10 GB RTX 3080上是否安全。Profile 结果不会进入正式汇总。

运行时另开终端查看：

```bash
watch -n 2 nvidia-smi
```

如果 profile 峰值显存超过约 9.3 GB，正式实验前优先把 `eval_batch_size` 从8降到4；不要先改论文的模型宽度和层数。

### 第五步：先跑 seed 2024 正式实验

```bash
tmux new -s diff2024
```

进入 tmux 后：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate grm-exp
cd ~/experiment1

bash scripts/run_diff_formal.sh formal 2024 python
```

按 `Ctrl+B`，再按 `D`，可以退出 tmux 而不中止训练。

重新连接：

```bash
tmux attach -t diff2024
```

### 第六步：检查 seed 2024

```bash
cd ~/experiment1
latest_log="$(ls -t outputs/console/diff_formal_v2_pse_seed2024_*.log | head -1)"

grep -E \
  "Train Loss|position diagnostics|mean rank|Val Results|Test Results|Run metadata" \
  "$latest_log" | tail -n 100
```

确认训练曲线、最佳 epoch、NDCG/Recall 和顺序统计都正常后，再运行另外两个 seed。

随后先小样本验证 attention，再跑完整 test attention：

```bash
bash scripts/run_diff_formal.sh attention-smoke 2024 python
bash scripts/run_diff_formal.sh attention 2024 python
```

### 第七步：运行剩余 seeds

可以在一个 tmux 中顺序运行：

```bash
tmux new -s diff-seeds
```

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate grm-exp
cd ~/experiment1

bash scripts/run_diff_formal.sh formal 2025 python
bash scripts/run_diff_formal.sh formal 2026 python
bash scripts/run_diff_formal.sh summarize 2024 python
```

不要在同一张 RTX 3080 上同时启动多个 seed，否则速度和显存都会互相影响。

## 6. 输出文件

单次运行：

```text
outputs/console/diff_formal_v2_pse_seed<seed>_*.log
outputs/diffusion_diagnostics/AmazonReviews2014/DIFF_GRM/*_position_metrics.csv
outputs/diffusion_diagnostics/AmazonReviews2014/DIFF_GRM/*_order_metrics.csv
outputs/diffusion_diagnostics/AmazonReviews2014/DIFF_GRM/*_training_metrics.csv
outputs/diffusion_diagnostics/AmazonReviews2014/DIFF_GRM/*_evaluation_metrics.csv
outputs/run_metadata/AmazonReviews2014/DIFF_GRM/*_run_metadata.json
external/DiffGRM/saved/*/pytorch_model.bin
outputs/attention/diff_formal_pse_4x256_schedfix_v2/seed<seed>/*
```

三 seed 汇总：

```text
outputs/reports/diff_formal_pse_4x256_schedfix_v2/per_seed.csv
outputs/reports/diff_formal_pse_4x256_schedfix_v2/summary.json
outputs/reports/diff_formal_pse_4x256_schedfix_v2/position_per_seed.csv
outputs/reports/diff_formal_pse_4x256_schedfix_v2/position_summary.csv
outputs/reports/diff_formal_pse_4x256_schedfix_v2/order_per_seed.csv
outputs/reports/diff_formal_pse_4x256_schedfix_v2/order_summary.csv
outputs/reports/diff_formal_pse_4x256_schedfix_v2/training_history.csv
outputs/reports/diff_formal_pse_4x256_schedfix_v2/evaluation_history.csv
outputs/reports/diff_formal_pse_4x256_schedfix_v2/full_mask_nll_history.csv
outputs/reports/diff_formal_pse_4x256_schedfix_v2/position_nll_test.png
outputs/reports/diff_formal_pse_4x256_schedfix_v2/decode_order_heatmaps.png
outputs/reports/diff_formal_pse_4x256_schedfix_v2/training_loss_and_lr.png
outputs/reports/diff_formal_pse_4x256_schedfix_v2/validation_metrics.png
outputs/reports/diff_formal_pse_4x256_schedfix_v2/full_mask_nll_by_epoch.png
```

## 7. 完成基线后的下一阶段

在 PSE 基线稳定以后，再做受控 Tokenizer 替换：

1. 固定数据、embedding、DiffGRM 模型、OCN、CPD 和训练预算；
2. 将 PSE 换成 RQ-KMeans；
3. 将 PSE 换成 RQ-VAE；
4. 对比逐位置 NLL、mask 分配、解码顺序、碰撞率和最终推荐指标；
5. 再把 DiffGRM 与已经完成的 AR 结果放到同一个研究矩阵中。

这一顺序可以区分：性能变化来自扩散生成范式本身，还是来自 Tokenizer 与生成方式是否匹配。
