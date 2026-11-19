# DIGER 推荐反馈短对照

## 1. 这次想回答什么

当前已经验证推荐梯度能回传到 Tokenizer，但这不等于推荐反馈带来了实际收益。这次比较**只按重构等目标更新码本**与**额外加入推荐反馈更新码本**，判断推荐反馈是否有额外贡献；不是重复检查梯度是否非零，也不是直接验证 Mixer 的独立效果。

上一轮噪声短对照在相同的 3 轮 Decoder 恢复后，A / B / C 的验证 HR@10 分别为 0.065957 / 0.065957 / 0.066136，NDCG@10 为 0.047498 / 0.047521 / 0.047666。B / C 的正式 SID 分别只改变 22 / 23 件商品。这是单 seed、SID-level 验证观察：去噪没有明显优势，正式编码干预也较弱，不足以确认或者否定整个联合优化方向。

因此，新实验适度增加 Tokenizer 更新量，并拆开推荐反馈与重构微调。**所有组重新从共同起点开始；旧 A/B/C 输出保留，不拿旧 A 代替本轮主要冻结对照。**

本页描述固定实验协议。2026-10-12 已将代码提交为 `299e5fa` 并快进同步到服务器，服务器真实张量测试和三组单 H100 smoke 均通过，随后提交正式后台 Job。运行状态以 [准备与启动记录](preparation/gpu_smoke_20261012_2150/README.md)、Job 日志及独立输出收据为准；smoke 通过不代表推荐效果得到提升。

2026-10-12 本地检查：新协议 16 项、新运行器 14 项、旧噪声协议 40 项及历史复现安全回归 78 项通过。新运行器的 5 项真实张量测试因本机没有 Torch/NumPy 跳过，另有 1 项符号链接测试因 Windows 权限跳过；不把跳过计为通过。旧目录迁移审计固定在 75 脚本时点，其已存在的后续开发差异没有重新封锁或者修改收据。

## 2. 三组如何控制变量

三组都加载同一份 Cross v2 第 5 轮最佳权重。该起点已训练 Decoder，但 Tokenizer 尚未更新；Decoder、基础码本、Cross Mixer 和正式 SID 的起点完全一致。

| 组别 | Tokenizer 阶段 | 后续处理 | 对照目的 |
|---|---|---|---|
| `frozen` | 不更新 Tokenizer | 固定 Tokenizer，训练 3 轮 Decoder | 只继续训练 Decoder 的参考轨迹 |
| `cross_recon` | 更新 24 次；推荐反馈权重为 0 | 固定更新后的 Tokenizer，训练相同 3 轮 Decoder | 仅重构等目标微调的作用 |
| `cross_rec` | 更新相同 24 次；推荐反馈权重为 1 | 固定更新后的 Tokenizer，训练相同 3 轮 Decoder | 推荐反馈是否有额外收益 |

**三组全部保留已有 Cross Mixer。** `cross_recon` 只是关闭 Tokenizer 阶段的推荐反馈，并没有去掉交互模块，不是第四阶段或第五阶段旧的 Refined-PSE。恢复阶段三组都正常训练推荐器。

Tokenizer 的目标为：

\[
L_{\mathrm{tok}}=L_{\mathrm{recon}}+0.01L_{\mathrm{balance}}+0.01L_{\mathrm{anchor}}+\lambda_{\mathrm{rec}}L_{\mathrm{rec}}.
\]

`cross_recon` 使用 \(\lambda_{\mathrm{rec}}=0\)，`cross_rec` 使用 \(\lambda_{\mathrm{rec}}=1\)。Tokenizer 阶段冻结 Decoder 参数，但 `cross_rec` 保留经 Decoder 回传到 Tokenizer 的计算图；恢复阶段冻结 Tokenizer，只更新 Decoder。

这一次统一不加 Gumbel 噪声，以减少随机分配对机制比较的干扰。**这是一套机制对照条件，不宣称无噪声已经调优或优于加噪声。**

### 共同参数

| 项目 | 设置 |
|---|---|
| 数据与编码 | Amazon Reviews 2014 Beauty；4×256 SID；seed 2024 |
| 权重起点 | Cross joint v2 第 5 轮；Tokenizer optimizer updates = 0 |
| Tokenizer 预算 | 384 个 batch；梯度累积 16；24 次 optimizer update |
| 正式 train / eval batch size | 64 / 8，与之前短对照一致 |
| 基础码本 / Mixer 学习率 | `1e-5` / `1e-4`，沿用此前短对照 |
| Decoder 预算 / 学习率 | 相同 3 个完整 epoch / 固定 `1e-4` |
| 分配距离温度 / 反向 softmax 温度 | 0.1 / 1.0 |
| 精度与裁剪 | Tokenizer float32；Decoder BF16；gradient clipping 1.0；TF32 关闭 |
| 评估 | SID-level 验证集，沿用验证 beam；不运行 test |
| 主要比较点 | 完整轨迹，以及相同恢复第 3 轮；不早停、不按不同最佳轮数比较 |

384 个 batch / 24 次更新是上一轮的四倍，目的是检验弱干预这一可能解释，不是追求更多 SID 变化，也不宣称 24 次已最优。

历史起点没有完整优化器和随机状态。因此三组统一新建优化器，并配对训练数据顺序和随机流：这是**共同权重起点的新短实验**，不是精确恢复历史训练。恢复阶段学习率不继承长期训练的 warmup。

固定共同初始推荐器的硬 SID 检查也已完成：Rec 相对 Recon 的验证 NLL 仅下降约 0.000627，目标换码子集主要是标签换位评分变化，不等于推荐收益。详见 [检查结果](hard_sid_compatibility/RESULTS.md)。

## 3. 第 6 / 24 次更新的诊断不改变训练轨迹

更新组在第 6 次、第 24 次 optimizer update 后增加轻量 probe，只根据当前 effective codebooks 计算全商品目录的无噪声正式 SID，记录相对起点的变化、利用率和碰撞等统计。

- 不刷新训练数据中的 SID mapping，不更改后续 batch 的数据构造。
- 不运行验证集，不根据分数选择更新步数，也不按 SID 变化率提前停止。
- 不消耗或者改变后续训练需要的随机状态。
- probe 是诊断记录，不包含完整训练恢复状态，不能用它独立恢复训练。

384 个 batch 完成后，才按统一流程刷新正式 mapping，运行更新后验证，并开始固定 Tokenizer 的 Decoder 恢复。

这样可以观察 6 → 24 次更新是否使正式编码变化更充分，而不会把额外验证或中途 mapping 刷新混入更新量的变化。

## 4. 文件与结果保存

| 内容 | 位置 |
|---|---|
| 固定协议与输入绑定 | `configs/05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json` |
| 输入及预算检查 | `scripts/checks/diger_feedback_pilot_protocol.py` |
| 独立运行入口 | `scripts/train/run_diger_feedback_pilot.py` |
| 协议 / 运行器测试 | `tests/test_diger_feedback_protocol.py`、`tests/test_diger_feedback_runner.py` |
| 新输出 | `outputs/pilots/stage05_feedback/<唯一名称>/` |

只接受项目根目录下这一专用输出前缀，目录必须不存在。不要写入历史 `evidence/`、旧 noise 输出、旧权重目录或初始化文件，不用“最新权重”代替绑定的起点。

每组保存初始、干预后、Decoder 恢复第 1 / 2 / 3 轮的完整快照，包含模型、正式 SID、优化器和随机状态；完整快照保存在服务器新输出目录中。汇总保存源码版本、输入哈希、梯度路由、实际 optimizer 更新数、正式 SID 变化、验证样本覆盖及验证指标。

全三组运行使用 `--arm all`。如果单独使用 `--arm frozen`、`--arm cross_recon` 或 `--arm cross_rec`，其完成收据只证明所选 arm 完成；**存在 `COMPLETE.json` 不等于三组全部完成**，需要核对收据和 summary 中的实际 arm 清单。

## 5. 运行顺序

本地项目根目录先检查协议和标准库测试：

```powershell
python -B tests/test_diger_feedback_protocol.py
python -B tests/test_diger_feedback_runner.py
python -B scripts/checks/diger_feedback_pilot_protocol.py --plan-only
```

具有完整数据和权重的 Linux 服务器主目录检查绑定输入：

```bash
source runtime_env.sh
python -B tests/test_diger_feedback_runner.py --require-torch
python -B scripts/checks/diger_feedback_pilot_protocol.py
```

再使用单张 GPU 做小流程测试。下列命令在服务器项目根目录运行；每次都使用新的输出名称：

```bash
feedback_stamp="$(date +%Y%m%d_%H%M%S)"
python -u -B scripts/train/run_diger_feedback_pilot.py \
  --arm all --smoke \
  --output "outputs/pilots/stage05_feedback/feedback_smoke_${feedback_stamp}"
```

smoke 只验收三组共享起点、反馈开关、梯度路由、冻结状态、probe 不干扰训练及保存流程，不解释它的小样本推荐分数。缩小预算的 smoke 不会满足正式的第 6 / 24 步诊断预算。

`--require-torch` 是小规模合成张量检查，可在 CPU 开发机执行，不启动训练；它禁止因缺少 Torch/NumPy 而跳过张量测试。GPU smoke 则必须在实际 GPU 环境验证真实模型和资产。两者不能互相替代。

GPU smoke 通过、固定输入和存储写入条件确认后，再提交独立 GPU Job 正式运行：

```bash
feedback_stamp="$(date +%Y%m%d_%H%M%S)"
python -u -B scripts/train/run_diger_feedback_pilot.py \
  --arm all \
  --output "outputs/pilots/stage05_feedback/feedback_pilot_${feedback_stamp}"
```

不要把网页终端前台命令当作已经可靠提交的长期 Job。提交、同步和 GPU 正式启动应另行确认；本页不记录未发生的启动结果。

## 6. 怎么解读结果

完整训练后的逐样本推理诊断另见 [配对分析方案](paired_analysis/README.md)。这一步复用固定第 3 轮快照，不增加训练或修改本页的实验协议。

2026-10-13 已完成三组完整验证集重放，原总体指标全部精确复现；推荐反馈组相对仅重构组新增 24、丢失 31 次 HR@10 SID 命中，净 −7。目标变化子集净 +2 仍低于 Frozen，且其中两次新增来自保留原目标编码。详见 [结果与下一步](paired_analysis/RESULTS.md)，不把局部差异写成编码优化成功。

随后新增 [固定初始推荐器的硬 SID 兼容性诊断](hard_sid_compatibility/README.md)：只使用 `initial` / `post_tokenizer` 快照，冻结完整推荐器、四个目标位置统一全部 masked，比较训练集和验证集的逐位置 NLL、平滑 CE 和原标签锚定评分。它不重新训练、不生成 Top-K、不运行 test，不把兼容性损失的正负当作 Tokenizer 内在质量或继续研究的必过门。运行及完成状态另以新独立输出收据为准。

后续开发入口统一归入 `scripts/analysis/`：[推理与捕获](../../../../scripts/analysis/evaluate_feedback_paired.py)、[标准库配对统计](../../../../scripts/analysis/feedback_paired_analysis.py)。本轮实际启动的 `45ffc26` 仍使用当时的 `scripts/analyze/` 路径；目录规范化不改动其运行源码与收据。

- **`cross_rec` 持续优于 `cross_recon` 和 `frozen`：** 支持本设置下推荐反馈有额外价值；仍需重复实验，不宣称稳定提升。
- **两个更新组接近，且都优于 `frozen`：** 收益可能主要来自重构等目标微调，尚未看到推荐反馈的额外贡献。
- **正式 SID 变化增多，但恢复后的推荐效果下降：** 更新量不足不再是唯一解释，要检查更新方向和 Decoder 适应过程。
- **正式 SID 仍几乎不变：** 干预仍弱，进一步检查梯度尺度和硬分配边界；不据此否定 Mixer 或所有联合训练。

主要看同恢复轮数的 HR / NDCG 完整轨迹，也查看变化商品在验证历史和目标中的覆盖。不能只挑最高分，也不能把推荐训练 loss 降低、非零梯度或更多 SID 变化单独解释成推荐收益。

这次保持 Mixer 不变，只研究推荐反馈。未来若要判断跨码本交互的独立作用，仍需要共同初始化、共同训练预算的有 / 无 Mixer 对照；推荐反馈有效并不是开展该对照的必要前提。
