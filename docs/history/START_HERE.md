# 从零开始：AR 生成式推荐复现实验

> **历史文档提示：**这是项目最初搭建阶段的操作手册，里面的本地 GPU、环境状态和部分输出路径已经过时。当前项目结构与实验进度请从 [`../README.md`](../README.md) 开始查看。

## 结论先行

第一轮不要直接复刻工业版 OneRec，也不要先做 RL。推荐按下面顺序执行：

1. 用 DiffGRM 作者仓库中的 `AR_GRM` 跑通 Amazon Beauty。
2. 记录 4 个 SID 生成位置的训练/验证交叉熵，输出 CSV、TensorBoard 和两张图。
3. 固定 leave-one-out 和 full-catalog constrained generation，报告 NDCG@5/10、HR@5/10。
4. 用 3 个随机种子完成正式 AR 基线。
5. 再在同一仓库切换 `DIFF_GRM`，保证数据、SID 和指标尽量一致。
6. 最后才做 tokenizer 对比和可选的 DPO/GRPO。

选择这个入口的原因是：它同时包含 AR 和 diffusion 两个 decoder，后续比较时不用重新拼数据与评估。`phonism/genrec` 保留为更成熟的 TIGER/OneRec-SFT 参考实现；其中 OneRec-SFT 是 1.7B 全参数训练，不适合当前 6 GB GPU 做第一轮。

## 你现在拥有什么

```text
experiment1/
├── README.md                         # 原研究交接文档
├── docs/START_HERE.md                # 本操作手册
├── environment.yml                   # Python 3.10 + PyTorch 2.6/CUDA 12.4
├── upstreams.lock                    # 两个上游仓库及固定 commit
├── configs/ar_smoke_6gb.yaml         # RTX 3060 6 GB 冒烟配置
├── patches/diffgrm-position-loss.patch
├── scripts/
│   ├── bootstrap_upstreams.ps1       # 克隆并打监控补丁
│   ├── run_ar_smoke.ps1              # 跑 AR 冒烟实验
│   ├── plot_position_loss.py         # 画逐位置 loss 图
│   └── analyze_sid_collisions.py     # 检查 SID 冲突
├── external/
│   ├── DiffGRM/                      # 作者仓库，AR 与 diffusion 主实验入口
│   └── genrec/                       # TIGER / OneRec-SFT 参考 pipeline
├── data/                             # 下载数据与缓存（不进 Git）
└── outputs/                          # 日志、checkpoint、图（不进 Git）
```

DiffGRM 的 AR 模型已补充以下监控：

- `reduction='none'` 的逐 token 交叉熵；
- 每个 SID 位置分别累计 `loss_sum` 与 `valid_token_count`；
- 多卡时先跨进程求和，再做除法；
- train、validation、test 分开记录；
- TensorBoard 标量与 CSV 同时保留。

## 一、环境准备

当前机器检测结果：RTX 3060 Laptop 6 GB，NVIDIA 驱动 581.95；系统只有 Python 3.14，没有 Conda。不要在 Python 3.14 的全局环境直接安装研究依赖。

先安装 Miniconda 或 Miniforge，然后在项目根目录执行：

```powershell
conda env create -f environment.yml
conda activate grm-exp
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-windows.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期最后一条显示 PyTorch 2.6、`True` 和 RTX 3060。PyTorch 2.6 官方不再发布 Conda 包，所以 `environment.yml` 先建 Python/科学计算环境，再用官方 CUDA 12.4 wheel 安装 torch。若 Conda 解依赖失败，可先创建 Python 3.10 环境，再按 `environment.yml` 分组安装；不要直接使用上游 `requirements.txt`，其中包含 Linux 专用的 Triton/NVIDIA pip 包，并有 `panda` 拼写项，不适合作为 Windows 锁文件。

两个上游仓库已经克隆。以后换机器时执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_upstreams.ps1
```

脚本会 checkout 到 `upstreams.lock` 固定的 commit，并应用逐位置 loss 补丁。

### 在实验室 Linux 机器上同步

推荐把本项目的轻量代码推送到私有 GitHub/Gitee/实验室 GitLab。`external/`、`data/`、`outputs/` 都已忽略，不会把上游仓库、数据、checkpoint 上传。

本地首次推送：

```powershell
git add .
git commit -m "Initialize AR-GRM experiment pipeline"
git remote add origin <你的私有仓库URL>
git push -u origin main
```

实验室 Linux 机器：

```bash
git clone <你的私有仓库URL> experiment1
cd experiment1
bash scripts/bootstrap_upstreams.sh
```

在 Linux 上先按 GPU 驱动安装对应的 PyTorch wheel，再执行：

```bash
python -m pip install -r requirements-linux.txt
```

运行环境建议在实验室机器上重建，不要复制 Windows 的 `.venv` 或 Conda 目录。先用 `nvidia-smi` 确认驱动/GPU，再选择 PyTorch CUDA wheel。

## 二、先跑最小闭环

激活环境后，在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_ar_smoke.ps1 -PythonExe python
```

Linux 机器对应命令：

```bash
bash scripts/run_ar_smoke.sh python
```

首次运行会执行：

```text
Amazon Beauty 原始数据下载
        ↓
leave-one-out 划分（train / validation / test）
        ↓
商品文本 → MiniLM embedding
        ↓
RQ-KMeans → 4 位 Semantic ID
        ↓
AR encoder-decoder 的 teacher-forcing SFT
        ↓
受约束 beam search
        ↓
NDCG@5/10、Recall@5/10 + 每位置 loss
```

冒烟配置只跑 2 个 epoch，并缩小模型、历史长度和 beam。它的目的仅是证明 pipeline 完整，不能把结果写成论文复现值。

如果 6 GB 显存仍 OOM，依次把 `configs/ar_smoke_6gb.yaml` 中的：

1. `train_batch_size` 从 64 改成 32 或 16；
2. `eval_batch_size` 从 16 改成 8；
3. `n_embd` 从 128 改成 96，同时保证可被 `n_head` 整除。

不要通过缩小 `codebook_size` 来解决 OOM，因为这会改变研究对象本身。

## 三、检查输出

主要输出位置：

```text
outputs/logs/                  训练文本日志
outputs/tensorboard/           TensorBoard event 文件
outputs/checkpoints/           checkpoint
outputs/position_loss_logs/    逐位置 loss CSV
```

CSV 格式：

```text
epoch,split,position,loss,valid_tokens
1,train,1,...,...
1,train,2,...,...
1,val,1,...,...
```

画图：

```powershell
$csv = Get-ChildItem outputs\position_loss_logs\AmazonReviews2014\AR_GRM\*.csv |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

python scripts/plot_position_loss.py --input $csv.FullName --output-dir outputs/figures/ar_smoke
```

输出两张图：

- `position_loss_by_epoch.png`：横轴 epoch，每条线是一个 SID 位置；
- `position_loss_profile.png`：横轴 SID 生成位置，展示最后一个 epoch 的位置难度轮廓。

查看 TensorBoard：

```powershell
tensorboard --logdir outputs/tensorboard
```

## 四、逐位置 Loss 到底怎么算

对第 `d` 个 SID 位置，记录：

```text
L_d = 所有有效样本在位置 d 的交叉熵之和 / 位置 d 的有效 token 数
```

这比“先求每个 batch 的均值，再平均 batch”更可靠，因为最后一个 batch 较短，且未知商品会用 `-100` 忽略。当前 AR 模型只预测 4 个 SID digit，不预测 EOS，所以图中只出现位置 1–4；不要把 padding 或 EOS 混入 SID 位置曲线。

模型优化目标仍是四个位置均值：

```text
L_SFT = (L_1 + L_2 + L_3 + L_4) / 4
```

建议同时保留两种图。`loss vs epoch` 能说明训练动态；`loss vs position` 才直接回答“哪个位置更难生成”。

## 五、NDCG 与 Hit Rate 的口径

本 pipeline 对每个用户只有一个 test target，采用 leave-one-out：最后一个交互用于测试，倒数第二个用于验证，其余用于训练。在这个单正例协议下：

```text
HR@K = Recall@K = 目标 item 是否出现在 Top-K
NDCG@K = 命中时 1 / log2(rank + 1)，否则 0
```

因此日志里的 `recall@5`、`recall@10` 可以分别作为 `HR@5`、`HR@10` 报告，但论文表格脚注必须写明“single-target leave-one-out，因此 HR 与 Recall 等价”。

必须固定并记录：

- Amazon 版本：2014；
- 类别：Beauty；
- 过滤协议：上游的 5-core 数据；
- 划分：按时间 leave-one-out；
- 候选：通过合法 SID 图约束生成，而不是 sampled negatives；
- K：5、10；
- beam 设置；
- tokenizer、SID 位数和码本大小；
- 随机种子和上游 commit。

不同仓库的数字只有在这些口径一致时才能横向比较。

## 六、从冒烟实验升级到正式 AR 基线

冒烟通过后建立一份正式配置，不直接改冒烟文件。至少完成：

| 项目 | 第一轮正式设置 |
|---|---|
| Dataset | Amazon 2014 Beauty |
| Split | leave-one-out |
| Decoder | AR_GRM |
| SID | 4 × 256，先用论文对应的 OPQ/PQ，再做 RQ-KMeans |
| Seeds | 2024、2025、2026 |
| Metrics | NDCG@5/10、HR@5/10 |
| Loss | train/val 每个 SID 位置 + total |
| Checkpoint selection | validation NDCG@10 |
| Reporting | 3 seeds 的 mean ± std |

正式复现时逐步恢复作者 README 的模型规模和训练参数。6 GB GPU 很可能无法保持作者的 batch 1024；如果只能降低 batch，必须在实验记录中声明。若要严格等效大 batch，应补梯度累积，不能把小 batch 结果直接称为“完全复现”。

一次正式 run 至少保存：

```text
config.yaml
command.txt
git_commits.txt
position_loss.csv
metrics.json 或完整日志
best checkpoint
两张 loss 图
SID collision report
```

检查 SID 冲突：

```powershell
python scripts/analyze_sid_collisions.py `
  --input <item_id2tokens_*.npy> `
  --output outputs/reports/ar_beauty_seed2024_sid_collision.json
```

冲突不能忽略：多个 item 共享同一 SID 时，按 SID 命中的指标可能虚高。至少报告 collision rate，并确认 evaluator 最终按 item 还是按 SID 判定。

## 七、RL 阶段怎么处理

学长原话里的“RL 阶段 loss 也可以画一条”应理解为可选项，不是第一轮 AR 基线的硬要求。当前选定的 `AR_GRM` 只有 SFT/NTP，没有 DPO/GRPO；不要为了有一条“RL 曲线”而把普通 CE 冒充 RL loss。

若后续切到 OneRec 的 IPA/DPO 或其他偏好训练，实现时必须分别记录：

```text
loss/ntp
loss/dpo
loss/total = alpha * loss_ntp + beta * loss_dpo
reward/chosen_mean
reward/rejected_mean
reward/margin
preference/pair_accuracy
```

还要固定 reference model、DPO beta、候选采样数、chosen/rejected 生成规则。OneRec 原论文依赖工业曝光和多反馈 reward model，公开数据上只能做“结构复现/替代实验”，不能宣称完整复现其线上 IPA。

## 八、与后续 DiffGRM/tokenizer 研究如何衔接

第一轮 AR 完成后，按以下实验矩阵推进：

```text
decoder {AR_GRM, DIFF_GRM}
    × tokenizer {OPQ/PQ, RQ-KMeans, 后续新方法}
    × seed {2024, 2025, 2026}
```

每次只换一个因素。保持数据划分、SID 位数、码本大小、embedding 来源、训练预算和评估协议不变。每个实验都比较：

- NDCG/HR；
- 每个 SID 位置的 loss；
- SID 冲突率、码本利用率；
- 训练时间、峰值显存、推理时间；
- diffusion 的 timestep × SID position loss；
- diffusion 推理时各位置首次确定的顺序。

只有当某个现象跨 seed 稳定出现，例如“AR 的后两位始终困难，而 diffusion 对位置更均衡”，才进入 decoder-aware tokenizer 的方法设计。

## 九、每周汇报模板

每周只回答四件事：

1. 问题：本周验证哪个假设？
2. 证据：固定协议下的指标、曲线和统计是什么？
3. 结论：现象是否跨 seed 稳定，是否存在混杂变量？
4. 下一步：只改变哪个变量，预期观察到什么？

第一周的合格交付是：Beauty AR 冒烟闭环、逐位置 train/val loss 两张图、NDCG/HR 日志、SID 冲突报告，以及一页失败/环境记录。不是追求 SOTA。
