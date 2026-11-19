# 正式实验：RQ-VAE 3×256 vs RQ-KMeans 3×256

这轮实验只改变 SID tokenizer。两边共享 Beauty 数据、Sentence-T5-XL item embedding、
3 个语义位置、256 码本、collision suffix、AR 模型和评估协议。

## 0. 同步代码后更新依赖

```bash
cd ~/experiment1
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate grm-exp
python -m pip install -r requirements-linux.txt
python scripts/preflight_formal.py
```
## 1. 先做时间和显存 profile

profile 会提取并缓存正式 embedding、训练 50 步 RQ-VAE，并用正式 AR 模型跑 1 个 epoch。
它不是正式结果。

```bash
cd ~/experiment1
set -o pipefail
bash scripts/profile_formal.sh 2024 python 2>&1 | \
  tee "outputs/console/formal_profile_$(date +%Y%m%d_%H%M%S).log"
```

首次 Sentence-T5-XL 下载和 embedding 提取无法通过 50 步 profile 缩短，但之后所有实验复用缓存。

## 2. 单种子正式实验

推荐分阶段运行，任何阶段中断都可以重跑同一条命令。

```bash
# 共享 embedding（已经 profile 成功时会直接复用）
bash scripts/run_formal_pair.sh 2024 prepare python

# RQ-VAE tokenizer：最多 60k steps，可从 last.pt 恢复
bash scripts/run_formal_pair.sh 2024 rqvae-tokenizer python

# 两次下游 AR
bash scripts/run_formal_pair.sh 2024 rqvae-ar python
bash scripts/run_formal_pair.sh 2024 rqkmeans-ar python
```

也可以在 tmux 中一次完成：

```bash
bash scripts/run_formal_pair.sh 2024 all python 2>&1 | \
  tee "outputs/console/formal_pair_seed2024_$(date +%Y%m%d_%H%M%S).log"
```

## 3. 三种子

2024 完整通过后，再依次运行：

```bash
bash scripts/run_formal_pair.sh 2025 all python
bash scripts/run_formal_pair.sh 2026 all python
```

三组完成后汇总均值和标准差：

```bash
python scripts/summarize_formal_results.py
```

## 4. 主要输出

- `data/formal_embeddings/`：两边共享且只生成一次的 item embedding；
- `outputs/tokenizers/rqvae/Beauty/seed*/`：RQ-VAE checkpoint、曲线、SID 和空间指标；
- `outputs/tokenizers/rqkmeans/Beauty/seed*/`：RQ-KMeans SID、重建 embedding 和空间指标；
- `outputs/position_loss_logs/`：逐位置 loss、NLL、normalized NLL、accuracy；
- `outputs/run_metadata/`：最佳 epoch、测试指标、训练时长和峰值显存；
- `outputs/console/`：完整控制台日志。

第四个生成位置是 collision suffix，分析时必须和前三个语义位置分开。
