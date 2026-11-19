# H1 历史 SID Grid Attention：B / C，seed 2024

状态：B/C 已上传并各用一张 H100 启动正式训练；执行记录见 [EXECUTION.md](EXECUTION.md)。以下检查收据保留准备阶段原始状态，不作为当前运行状态。

| 组别 | 历史商品进入原 MLP 前的处理 | 起点与训练日程 |
|---|---|---|
| A，已有对照 | 原来的 4 码位拼接 → MLP → Encoder | 原 H1 R5 → R20，冻结整个 Tokenizer+SID，推荐器继续到 R35 |
| B，新增 | 横向注意力：每件商品内部的 4 个码位交互 | 与 A 相同，seed 2024 |
| C，新增 | 横向注意力后接纵向注意力：同一码位跨历史商品交互 | 与 A 相同，seed 2024 |

这里的 A 指原 H1 Freeze；使用原 H1 的中心编号标记设置，不叠加去编号标记、Mixer 门控、学习率衰减或 loop。

## 固定的结构与参数

```text
历史商品及已完成列表前缀的 SID
  → 原推荐器 embedding：[B,L,4,256]
  → B：1 层横向注意力
    C：1 层横向注意力 → 1 层纵向注意力
  → 恢复 [B,L,4,256]
  → 原来的四码位拼接、MLP：[B,L,256]
  → 原来的商品位置向量、Encoder、Decoder
```

横向、纵向均为独立的 Pre-LN + 4 头注意力 + 残差；每头 64 维，LayerNorm eps=1e-5，新增 attention dropout=0，残差系数 1，无新增 FFN。横向参数跨商品共享，纵向参数跨码位共享；两方向不共享参数。不另外加入位置或中心编号向量，保留原模型已有的码位 embedding 区间和 Encoder 商品位置向量。

输出投影权重和 bias 初始化为 0，Q/K/V 正常初始化。因此 R5 接入时是恒等变换，但输出投影首步有梯度。B/C 横向初始参数相同；新层初始化不消耗主训练 RNG。新增参数量 B=263,680，C=527,360。C 对 B 的比较同时增加了结构、参数量和计算量，不能单独归因于纵向拓扑。

PAD 不参与 key/value，PAD query 的修正归零；全 PAD 的内部行使用安全 dummy key，避免全遮罩 softmax。H 的第 j 件商品只看到真实历史和前 j−1 件商品，训练/生成沿用同一入口，未来目标不进入网格。

## 训练、恢复与冻结

- 直接复用 `goal_20261027_r1/code` 冻结源码，不改当前共享训练代码或多 seed 作业。
- 父 checkpoint：`goal_20261027_r1/runs/H1/initial.pt`，R5/T0；SHA256：`991352d4b48998427d44397411eb4178678487de21f1e08e5cd9d39240a06fc1`。
- 先严格加载原模型、Tokenizer、R/T Adam、scheduler、RNG，再接新层。新参数追加到同一个 R optimizer 参数组；旧参数矩和学习率不重置，新参数从空 Adam 状态开始。
- R5→R20 维持每轮交替块 40 次 T 更新、3 个 R epoch；R20/T200 后不再执行 T 更新或 SID 刷新，R 继续到 R35。**Grid 属于推荐器，冻结后继续随 R 学习。**
- 原始 35R/400T 计划不缩短：R cosine 学习率和 DIGER 退火分母不重设。实际最终预算是 R35/T200（包含继承的 R5），R optimizer 总计 24,255 步，新增 Grid 累计 20,790 步。
- 保持 global batch=1024、microbatch=256、eval batch=32，单 H100，BF16 训练、FP32 验证和测试，beam=32；不自动改变 microbatch 或降精度。
- 新 checkpoint 显式保存 Grid 配置、代码 SHA、R 参数名顺序、两类 Adam 时钟和冻结状态 SHA。恢复时严格匹配，不能用原来的通用 `AlternatingTrainer.restore` 直接加载 Grid checkpoint。
- R20 冻结摘要覆盖整个 alignment（含 SID、DIGER 时钟）及 T optimizer；后续 checkpoint/阶段边界验证其不变。

## 保存及结果口径

每 100 次更新保存可恢复 `last.pt`。保留每个验证节点的独立完整 checkpoint：R8/11/14/17/20/23/26/29/32/35；`initial.pt` 保留共同 R5 起点。严格按全验证集 NDCG@5 选 best，平局取早；测试 validation-best 与固定 R35，若两者相同则复用。测试中断可用 `--resume` 接续，已验收测试文件 SHA 不变时不重跑该点。

结果包含 `training.jsonl`、逐节点 `summary.json`/逐样本预测、`FREEZE_R20.json`、`TRAINING_COMPLETE.json`、`TEST_RESULTS.json`、`GPU_COMPLETE.json`。末项表示训练与测试的程序内检查完成，平台资源释放与最终实验报告仍需后续任务完成。

## 文件与使用

| 文件 | 用途 |
|---|---|
| `grid.py` | 横向/纵向注意力与安全 padding；接入旧 backbone 的预留入口 |
| `protocol.py` | R5 继承、Grid 参数登记、恢复时钟、R20 冻结 |
| `experiment.py` | 单卡训练/验证/测试循环，支持 `--resume` |
| `config_B.json` / `config_C.json` | 各组结构配置；`base_config` 与原 H1 完全一致 |
| `test_grid.py` | 真实冻结引擎上的小样本 CPU 测试 |
| `verify_parent_cpu.py` | 真实 R5 checkpoint 的只读加载和预测一致性检查 |
| `LOCAL_CHECKS.json` / `PARENT_CPU_CHECK.json` | 本地检查范围、结果和限制 |

以下是启动入口说明；当前 B/C 已提交，不要再次提交或覆盖现有输出目录：

```bash
python experiment.py --arm B --output /path/to/new_run_B
python experiment.py --arm C --output /path/to/new_run_C
# 中断后，用同一 arm、同一输出目录加 --resume。
```

默认历史根目录为 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1`，数据与资产路径在各组配置中。新实验目录里的 Python 和 JSON 一并上传即可；不覆盖历史源码。普通 `--help` 不启动训练。

本地检查使用 `.runtime/stage08_restore_env/Scripts/python.exe`，PyTorch 2.6.0+cpu。准备阶段真实父节点只做 CPU 加载、前向和状态检查，没有更新其权重，也没有重做数据集推理/统计。正式运行现已确认 CUDA 上完成 T/R 更新，新增 Grid 的 Adam 状态和单卡 CUDA RNG 已保存；完整训练、测试、资源释放和最终验收仍待完成。
