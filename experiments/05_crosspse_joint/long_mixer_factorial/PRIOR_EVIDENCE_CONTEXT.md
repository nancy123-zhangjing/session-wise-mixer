# 本轮长期四组的历史证据背景

核对日期：2026-10-20。本文件只梳理已经完成的历史短实验，解释本轮长期实验为什么这样设计；**不包含本轮最终结果，也不构成本轮完成验收**。以下数值来自本地原始 JSON，主要比较点均为预先固定的 `decoder_3` 验证结果，不能混入历史 beam256 测试表。

## 1. 总目标与每组历史实验回答的问题

总目标是检验：在相同有效编码和推荐器起点、相同推荐反馈与训练预算下，加入可训练 Full Mixer 是否带来额外推荐收益。这里的交互包含码本内部及码本之间的整体交互。

| 实验 | 真正回答的问题 | 没有回答的问题 |
|---|---|---|
| 噪声 A/B/C | 同一 Cross v2 第5轮权重下，冻结、无噪声更新、实际施加原噪声配置的单次更新，经过相同3轮推荐器恢复后有何差别？ | 没有关闭 Mixer，不能归因 Mixer；仅6次 Tokenizer 更新，不能判断长期累计作用。 |
| 共享起点短四组 | 同一有效码本 E₀重基、同一完整推荐器下，禁用/启用 Mixer 与冻结/更新 Tokenizer 的差分是否不同？ | 只做一次24次更新后冻结恢复，没有反复交替；无 Gumbel，不能直接预测本轮带噪长期结果。 |
| 本轮长期四组的预设问题 | 共同 Aug28 推荐器与 E₀起点下，完整35轮推荐器和60次 Tokenizer 更新后，Full Mixer 的增量是否存在、是否持续？ | 不单独区分 within/between，不排除新增参数化、学习率分组与优化路径解释，也不验证其他 seed 或预训练起点。 |

早期 Cross/Refined 长期实验的初始化不同，不能直接作为“只改 Mixer 开关”的严格对照；这正是后续共享起点短四组的动机。旧长期 Cross v2 的最佳点处于推荐器第5轮、尚未更新 Tokenizer，不能把该最佳测试结果解释为 Tokenizer 收益。来源：[阶段05原报告](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/README.md)、[cross_joint_v2 复现配方](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/reproducibility/recipes/cross_joint_v2.json)。

## 2. 共享起点短四组：已经完成了什么

运行标识为 `factorial_20261013_1816_formal`，正式执行 commit 为 `78d446d365a589cfb476424e732e673795e1e79b`。本地 [COMPLETE.json](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/COMPLETE.json) 的 `summary_sha256`、[FORMAL_PASS.json](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/FORMAL_PASS.json) 的同字段均与实际 [summary.json](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/summary.json) 字节 SHA256 一致：

`1491f38402aca6d73f98728f86b34afcc1b264f4e20f17df934003f907dacedc`

summary 明确记录 `all_protocol_arms_complete=true`、`no_test_evaluation=true`、`single_tokenizer_block_then_frozen_recovery=true` 和 `not_full_multicycle_joint_training=true`。这些是已完成正式短实验的证据，不能把准备目录中的 smoke 当作正式结果，也不能据此称长期四组已完成。此次核对针对本地原始结果及收据，不等于重新加载历史远端权重执行评估。

起点先取反馈短实验的共同 initial（对应 Cross v2 第5轮、Tokenizer 更新次数0），再把已学习的有效 E₀放入四组相同基础码本。新 Mixer 为 full attention、4 heads、dropout0、输出投影零初始化、残差系数固定1。四组共同有效码本、正式 SID 和 anchor 一致，优化器重新建立；不是继续训练旧 attention，也不是 Original PSE 从头训练。[共同初始化原记录](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/common_initialization.json) 中精确位置为 `/initial_controls`、`/E0_array_sha256`、`/module_constructor`；[归档协议](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/protocol.json) 的 `/shared_checkpoint` 绑定 Sep-01，而非 Aug28。

两个 Joint 各384个 Tokenizer batch、累积16，即24次更新，**实际加噪次数0**；两个 Frozen 不更新 Tokenizer。随后四组均冻结 Tokenizer，推荐器固定 LR `1e-4`，训练3个完整 epoch，每轮1652 batch/104次更新，合计312次。推荐反馈权重1，SIA0；码本/Mixer LR 分别 `1e-5/1e-4`，均衡及 anchor 权重均0.01；batch64、eval8、label smoothing0.2，Tokenizer FP32、推荐器 BF16。统一比较固定第3轮验证，不选各组最高分、不跑 test。[短四组说明](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/pilots/mixer_factorial/README.md) 与原始 `/arms/*/settings`、`/arms/*/training` 相符。

下表保留9位小数，精确原值见每行来源。SID 改变数相对共同初始正式编码，目录共12,101件商品。

| 组 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | weighted score | SID改变商品数 | 原始位置 |
|---|---:|---:|---:|---:|---:|---:|---|
| no_mixer_frozen | 0.061977373 | 0.065957161 | 0.046156925 | 0.047497932 | 0.051189778 | 0 | [指标](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/summary.json:436)；`/arms/0/validation/4/metrics` |
| no_mixer_joint | 0.062379824 | 0.066136028 | 0.046344611 | 0.047611024 | 0.051316025 | 16 | [指标](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/summary.json:935)；`/arms/1/validation/4/metrics` |
| mixer_frozen | 0.061977373 | 0.065957161 | 0.046156925 | 0.047497932 | 0.051189778 | 0 | [指标](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/summary.json:1598)；`/arms/2/validation/4/metrics` |
| mixer_joint | 0.062111523 | 0.065867728 | 0.046361113 | 0.047627721 | 0.051275723 | 284 | [指标](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/summary.json:2097)；`/arms/3/validation/4/metrics` |

四组 SID 数值来自同一 summary 的 `/arms/{0,1,2,3}/snapshots/1/deterministic_sid_change/changed_items`；各 `snapshots/1/label` 均为 `decoder_3`。两个 Frozen 的完整最终模型哈希相同（`ba01c04756f2a6a5040d9f27303ace81b7e6f9d574d7797ad9f3883fa1754116`），全部验证指标、恢复数据顺序与预算相同，见 [负对照原记录](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/summary.json:2545) `/frozen_negative_control`。

从未四舍五入的原值重算：

`ΔMixer = (mixer_joint − mixer_frozen) − (no_mixer_joint − no_mixer_frozen)`

| 指标 | 短四组 ΔMixer |
|---|---:|
| HR@5 | -0.000268300317489 |
| HR@10 | -0.000268300317489 |
| NDCG@5 | +0.000016502405340 |
| NDCG@10 | +0.000016697284516 |
| weighted score | -0.000040302233753 |

证据支持的表述是：Full Mixer 改变了更多正式 SID，但没有形成各指标一致的额外收益。HR@10 差值对应总体净少6次 SID 命中，**这不是逐样本配对显著性检验**。284对16只能说明重编码范围不同；**更多 SID 变化不等于推荐增益**，也不能由此确认“全局交互不稳定”是唯一原因。

参数化差异是实测存在的：无 Mixer Joint 训练65,536个基础码本元素；Mixer Joint 另训练33,408个交互参数，使用高10倍的分组 LR。见 [optimizer_groups](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/summary.json:2314) `/arms/3/optimizer_groups`。实际8个交互参数张量发生变化、基础码本变化、Tokenizer 阶段完整推荐器不变，见 [更新完整性](C:/Users/nancy/Documents/experiment1/archive/server_sync/mixer_factorial_20261014_results/summary.json:2328) `/arms/3/tokenizer_update_integrity`。这些证明干预执行了，不能单独证明机制带来收益。

## 3. 噪声 A/B/C：选择带噪设置的依据与限制

正式运行 `noise_pilot_20261012_1720`，协议为 `stage05_diger_noise_once_freeze_recover_v1`。本地 [COMPLETE.json](C:/Users/nancy/Documents/experiment1/outputs/pilots/stage05_noise/noise_pilot_20261012_1720/COMPLETE.json) 绑定实际 [summary.json](C:/Users/nancy/Documents/experiment1/outputs/pilots/stage05_noise/noise_pilot_20261012_1720/summary.json)，SHA256 核对一致：

`e52a09fec3dc3650dfabc8fad67b72d3b4f002be97ce943ea9e42659fbe7d079`

三组共用 Sep-01 的 Cross v2 第5轮完整权重，Tokenizer 尚未更新；保留旧 Cross Mixer。A 冻结，B 无噪声、C 带噪声，B/C 各96 batch、6次更新，然后三组同样恢复3轮推荐器。C 的退火总跨度仍是60次更新，只执行最初6步；不把1.0→0.2压缩到单次 phase。C 在第6步完成后记录强度约0.980423，说明它检验的是原始强噪声阶段，不是已经走完整条退火曲线。[原协议](C:/Users/nancy/Documents/experiment1/outputs/pilots/stage05_noise/noise_pilot_20261012_1720/protocol.json) 与 [噪声说明](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/pilots/noise_update/README.md) 给出设置与模式修复背景。

| 组 | Tokenizer更新 | 实际加噪调用 | 固定R3 HR@10 | 固定R3 NDCG@10 | weighted score | SID改变商品数 | 原始指标位置 |
|---|---:|---:|---:|---:|---:|---:|---|
| A 冻结 | 0 | 0 | 0.065957161 | 0.047497932 | 0.051189778 | 0 | [A](C:/Users/nancy/Documents/experiment1/outputs/pilots/stage05_noise/noise_pilot_20261012_1720/summary.json:813)；`/arms/0/validation/4/metrics` |
| B 无噪声更新 | 6 | 0 | 0.065957161 | 0.047521426 | 0.051208573 | 22 | [B](C:/Users/nancy/Documents/experiment1/outputs/pilots/stage05_noise/noise_pilot_20261012_1720/summary.json:1670)；`/arms/1/validation/4/metrics` |
| C 带噪声更新 | 6 | 192 | 0.066136028 | 0.047666024 | 0.051360025 | 23 | [C](C:/Users/nancy/Documents/experiment1/outputs/pilots/stage05_noise/noise_pilot_20261012_1720/summary.json:2527)；`/arms/2/validation/4/metrics` |

B/C 的调用证据分别为 `/arms/1/training/0` 和 `/arms/2/training/0`，见 [B训练](C:/Users/nancy/Documents/experiment1/outputs/pilots/stage05_noise/noise_pilot_20261012_1720/summary.json:847)、[C训练](C:/Users/nancy/Documents/experiment1/outputs/pilots/stage05_noise/noise_pilot_20261012_1720/summary.json:1704)。两者 `gumbel_calls=192` 只表示分配入口调用；B 的 `noise_applied_calls=0`，C 才是192次实际加噪。SID 数值来自 `/arms/{0,1,2}/snapshots/4/deterministic_sid_change/changed_items`，不是临时随机 SID 的差异比例。

这个短结果没有支持“去掉噪声已解决问题”；在固定R3点，C 的上述指标反而略高。但单 seed、6次更新、差值很小，不能证明噪声普遍有益或当前噪声强度最优。本轮保留带噪协议的依据是：没有现成证据要求去掉原协议的噪声，而且两个 Joint 可以使用相同累计退火、配对数据和独立噪声随机流，在同一噪声条件下比较 Mixer 的额外作用。**本轮四组不是新的噪声消融**；噪声同时出现在两个 Joint，属于固定的训练条件。

旧 guided 打分会递归切换 train/eval，曾可能破坏“推荐器eval、Tokenizer train”的混合模式；调度日志中的 `noise_scale` 不能证明历史每批实际加噪。A/B/C 新入口每批恢复模式并统计真实调用。因此C是把原噪声配置完整施加的新干预，不是旧长期执行的精确重放。也不能把旧长期结果与C的差异全部归因于噪声。

## 4. 与本轮35R/60T协议的关键差异

本轮定义以 [长期协议](C:/Users/nancy/Documents/experiment1/configs/05_crosspse_joint/long/mixer_long_seed2024.json) 为准；这里仅转述预设配置，不报告其执行是否完成。

| 项目 | 共享起点短四组 | 噪声A/B/C | 本轮长期四组 |
|---|---|---|---|
| 推荐器权重起点 | Cross v2 已训练5轮后的 Sep-01；再从共同 initial 重基 | 同一 Sep-01 第5轮 | 05训练开始前的 Aug-28 完整离线权重，四组统一重新开始 |
| 编码起点 | 共同已学习有效E₀，新零输出 Full Mixer | 旧 Cross 完整 Tokenizer/Mixer | 沿用短四组相同有效E₀重基；新零输出 Full Mixer，不继续训练旧 Mixer |
| 初始优化器 | 全部新建 | 全部新建 | 全部新建；跨交替阶段保留完整状态 |
| 推荐器训练 | 直接恢复3轮/312次更新 | 直接恢复3轮/312次更新 | 从Aug28先5轮，再10×3轮；合计35轮/3640次更新 |
| Joint Tokenizer | 一次384 batch/24次更新 | B/C一次96 batch/6次更新 | 10个phase，每个96 batch/6次；合计60次更新 |
| 冻结组训练量 | 3轮推荐器、0T | A为3轮、0T | 35轮推荐器、0T；不额外训练到45轮 |
| 噪声 | 两Joint均无 | B无；C只走累计退火最初6步 | 两Joint均1.0→0.2余弦，跨累计60次更新，不按周期重置；各1920次真实加噪调用 |
| 推荐器学习率 | 固定1e-4 | 固定1e-4 | 峰值1e-3、前500次R更新warmup，按3640次R更新余弦衰减；T不推进调度 |
| 主比较 | 固定恢复R3验证 | 固定恢复R3验证 | 固定逻辑R35验证的四组差分；初始及R5/10/15/20/25/30/35统一验证 |
| 最佳点/测试 | 不取各组最高分、不跑test | 同左 | 验证weighted选最佳，允许初始epoch0；四组封存选择后各一次beam256正式test，与R35结果分开报告 |

本轮先训练的5轮是**从Aug28重新执行的5轮**，不能把旧Sep-01权重当起点再追加30轮。沿用“5R＋10×(1T＋3R)”的节奏不等于恢复旧历史轨迹：新建优化器、随机数对齐、调度及模式/缓存修复都会影响轨迹。共同E₀本身来自已经学习过的 Cross 编码，因而本轮检验的是该暖启动功能之上的额外可训练交互；没有抹去预训练阶段已经发生的交互作用。

## 5. 可以与不可以作出的结论

短四组已经收紧了“初始有效编码不同”的混杂；本轮长期版本进一步收紧累计推荐器更新、反复交替、优化器/调度连续性和实际噪声配对。保留两个 Frozen 是对编码等价与确定性训练轨迹的负对照，即使预期它们相等也不能省掉其验收。主要增量仍是四组差分，不能只看 `mixer_joint` 自己是否随训练涨分。

Full attention 允许同一码本内部与不同码本之间的交互。启用Full与禁用之间的比较只能识别这一整套模块在当前条件下的整体效果；**不能分离within/between，也不能排除参数化或优化路径**。本轮没有容量匹配的额外网络、within-only、between-only或额外学习率搜索，报告时须保留这一限制。

两项历史短实验均为 seed2024、固定共同权重起点、小预算、验证集 SID-level 观察；没有重复seed、置信区间或显著性证据。SID改变、非零梯度、参数确实更新、Attention形状或碰撞率接近，只能用于核对干预及机制现象，不能替代推荐指标收益。SID-level 命中还受编码碰撞影响，不宜与阶段06的item-level连续检索数值直接排名。

本轮长期结果无论正负，都应分别报告：固定R35指标及ΔMixer、验证选定最佳点的正式测试、最佳点是否已经历T更新、SID与碰撞轨迹，以及控制/恢复/资源/备份验收。不得由本文件的历史收据推断本轮已训练完成，也不应预先要求Mixer必须涨点。

历史原始summary、协议和完成收据的逐字节副本另存于本轮prior_evidence/，来源与SHA映射见PRIOR_EVIDENCE_SOURCES.json；完整交付包会收集为delivery/prior_evidence/。这些是历史指标元数据副本，不包含历史完整权重，不代表重新执行了历史评估，也不构成本轮长期完成证据。
