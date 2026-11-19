> 2026-10-26退役说明：本文属于旧的单目标历史侧交互研究，不能作为OneRec请求级列表生成的当前实现或执行方案。相关代码、模型和恢复包已按用户要求删除；历史结果保留。当前方案见 docs/research/onerec_listwise/PLAN_20261026.md。

# Session-wise码本交互：本轮结论与依据

日期：2026-10-23。完成范围：文献及源码研究、具体设计、合成CPU验证、后续实验计划。**没有申请GPU、启动训练或改生产代码。**

## 先说结论

目前最值得先验证的是：**在相同05联合训练框架里，让历史中的真实码字先结合session交互，是否比只在单商品内交互更有效？** 不先把问题扩大成动态SID、整段session生成或连续生成。

首选“一轮时间×码位的分轴交互”：把历史展开成`[历史商品数L,4个码位,64维码字]`，先跨码位、再跨时间，得到只作用于历史端的路由修正。商品输出仍使用全局规范SID和原DiffGRM目录。这样既接上05的物理码本与交替优化，也暂时避开“每个session的目标编码不同，生成后如何查商品”的索引问题。

备选为相同接口下的一轮完整交互。**loop先保留，不与grid一起默认叠上去**：grid决定谁与谁交流，loop决定同一规则反复更新几轮；后者增加计算，本机CPU也没有显示它能省时间。先证实单轮session信息有价值，再判断是否需要多轮。

原型9类检查全部通过，证明接口、掩码和梯度可实现；它没有给出推荐效果。当前仍不能说这个方案创新成立、一定涨点，或已经解决生成时延问题。

## 1. 已有实验给了什么证据

| 实验族 | 已完成的内容 | 本次如何使用 |
|---|---|---|
| 05早期长期四组 | 项目既有结果；起点及旧协议见原报告 | 仅作历史探索，不混作后面共享起点长期的复验 |
| 05共享起点短四组 | 同一初态短Tokenizer干预及3R恢复 | 说明控制起点的重要性，不拿短期结果替代长期效果 |
| **05共享起点长期四组** | 同E0和推荐器，35R、Joint60T，Full/无Mixer×Joint/Frozen，单seed | 当前最可靠的05控制参考，但只得到微弱混合信号 |
| 07商品内SID embedding交互 | 独立256维推荐器embedding后加attention，非05物理码字交互；新增seed不稳定 | 提醒不能把“加attention”当成稳定改善；不作为新session主线 |
| 本文session-wise | 原始文献/作者代码阅读、合成原型、设计 | **尚无真实数据训练或推荐指标** |

05共享起点长期固定R35验证、beam32的主要DID：HR@10 `−0.000223584`，NDCG@10 `+0.000141106`；方向混合，不能宣称Full Mixer稳定有效。[05完整报告](../../../experiments/05_crosspse_joint/long_mixer_factorial/REPORT.md)中的各自best测试是另一种比较，不能替代固定预算DID。

07的[交替联合两组](../../../experiments/07_item_local_attention/joint_alternating/REPORT.md)及[多seed验证](../../../experiments/07_item_local_attention/multiseed_20261021/report/REPORT.md)不能与上述三种05实验合并。它们也不能推出session-wise会无效；它们只约束我们需要怎样的对照和证据。

所有已完成数字本轮均核对了当前项目文件；历史记忆只用来定位，不将旧摘要当实时实验状态。更多实现边界见 [CODE_AUDIT.md](evidence/CODE_AUDIT.md)。

## 2. “session-wise”有三个不同层次

1. **条件化表示/路由**：同一全局SID，历史中的码字表示受当前历史影响。本轮首选这个。
2. **条件化量化/身份**：同一商品在不同历史里成为不同SID；需要多身份归并、有限候选或动态索引，不能只画一个context encoder就算闭环。
3. **改成整段session编码或多未来商品生成**：连预测任务也改变，不是本轮的最小扩展。

只把05全局`C[4,256,64]`上的Full Mixer改为grid/loop，仍然没有历史H输入，就仍是全局Mixer优化。反过来，仅在推荐器256维商品向量上再堆普通历史attention，又会弱化与05编码主线的联系。

本设计把H明确放进**历史码字到物理码本的条件路由**。但要坦诚：它仍是历史侧表示扩展，尚未让目标SID本身条件化；也没有凭空证明它比已有历史Encoder更有用。

## 3. 最接近的离散推荐文献

以下是方法和相关源码阅读，均未复现作者结果。标题、原文、代码版本/哈希、阅读位置和评估边界完整记录在 [文献笔记](evidence/LITERATURE_SESSION.md) 与 [来源清单](evidence/sources_session.json)。

| 文献 | 已核实的相关机制 | 本项目应借鉴或避免的误读 |
|---|---|---|
| **ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation**，2025，[原文](https://arxiv.org/html/2502.13581v3)、[作者代码](https://github.com/google-deepmind/action_piece) | 商品特征集合内及相邻行为间合并；历史和下一商品目标分开编码 | 跨商品tokenization已有前例；它不是session注意力重写VQ中心；控制推理分词集成次数 |
| **Pctx: Tokenizing Personalized Context for Generative Recommendation**，2025，[原文](https://arxiv.org/html/2510.21276v1)、[作者代码](https://github.com/YoungZ365/Pctx) | 按商品聚合上下文侧面，允许有限多SID并归并到item | 若以后做动态身份，这是更直接的参考；其item评估不能和本项目SID分数直接比较 |
| **PIT: A Dynamic Personalized Item Tokenizer for End-to-End Generative Recommendation**，2026，[原文](https://arxiv.org/html/2602.08530v1) | 商品提出多个SID、历史选择伪标签，维护动态beam index | 联合学习与索引必须一起考虑；本轮未找到可确认作者实现，不伪称代码验证 |
| **TIGER / Recommender Systems with Generative Retrieval**，2023，[原文](https://arxiv.org/html/2305.05065v2) | 固定RQ语义码、历史预测、碰撞区分token | 历史条件预测本来就有；新增研究应指出条件信息进入哪个环节。本轮仅论文锚点 |
| **DiffGRM**，2025，[原文](https://arxiv.org/html/2510.21805v1)、[作者代码](https://github.com/liuzhao09/DiffGRM) | 固定PSE/OPQ、历史Encoder、离散目标解码；SID-based评估 | 当前基础模型锚点。项目05联合更新是本地扩展，不能当原论文已经验证的机制 |

较接近本项目的可检验价值是：**在统一离散输出和搜索协议下，保留码位结构、用物理量化几何条件路由，是否比相同计算下的无跨商品路由更有效？** 若以后要写方法贡献，还需与普通历史加深、冻结Tokenizer等对照区分机制。单纯“session-aware＋axial＋loop”不足以作为新颖性证明。

## 4. Grid与loop该如何选择

文献的grid并非同一种算法，学长也没有指定论文版本。原文及作者实现位置见 [GRID_LOOP笔记](evidence/LITERATURE_GRID_LOOP.md) 和 [版本清单](evidence/sources_grid_loop.json)。

| 原始工作 | 核实后的含义 | 对本项目的选择 |
|---|---|---|
| **Axial Attention in Multidimensional Transformers**，2019，[原文](https://arxiv.org/html/1912.12180v1)、[作者代码](https://github.com/google-research/google-research/tree/master/axial) | 分别沿各轴做注意力，通过中间状态传播信息 | `[L,M]`有时间与码位身份，适合作首选；不是全注意力的数学等价替换 |
| **MaxViT: Multi-Axis Vision Transformer**，2022，[原文](https://arxiv.org/html/2204.01697)、[作者代码](https://github.com/google-research/maxvit) | 局部窗口加稀疏全局grid，不是简单行列attention | M只有4，图像邻域假设不自然；先不照搬窗口分组和MBConv |
| **Grid-Attention: Enhancing Computational Efficiency of Large Vision Models without Fine-Tuning**，2024，[原文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06640.pdf)、[作者代码](https://github.com/pengyuLPY/GridAttn) | 可逆重排到子网格，各组MHA后恢复；不平均合并token | 可借鉴保留token、减少连接的思想；图像冗余不保证推荐码字也能安全稀疏化 |
| **Universal Transformers**，ICLR2019（预印本2018），[原文](https://arxiv.org/html/1807.03819)、[作者代码](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py) | 沿计算深度共享更新参数，状态逐轮传递；ACT另属可选机制 | loop可作深度/共享消融，不能因省参数宣称省延迟；首版不引入ACT |

特别不建议按`[码本M,码字编号K]`机械做网格：不同码本的第17号中心没有天然对应，17/18也没有空间邻接。`[历史位置L,码位M]`的两个轴有任务意义，但不同OPQ子空间坐标能否被共享交互块有效利用，仍是学习假设。

| 统一历史接口的候选 | 优点 | 代价/风险 | 决策 |
|---|---|---|---|
| one-full session | 所有码位可直接交互；简单参照 | LM平方注意力；额外历史层可能重复原Encoder | **唯一备选**，第一轮同跑 |
| one-axial session | 先商品内组合、再历史关联；配对项较小 | 两次投影、两跳路径，可能限制表达 | **首选**，r=1 |
| full＋shared loop | 不增加核心参数，多轮状态修正 | 增加计算；更深/共享/过平滑需拆分 | 单轮有信号后再试 |
| axial＋shared loop | 可同时分轴和反复修正 | 一次改变两个因素，不宜直接归因 | 单轮有信号后、与r1及untied2层对照 |

在L50/M4/d64、batch4的CPU增量微基准里，full1约8.320ms、axial1约6.952ms、full2约14.257ms、axial2约10.805ms；新增参数均41,152。注意力配对40,000→10,800不是3.70倍端到端提速。该结果只帮助工程取舍，H100及完整模型尚未测量。

## 5. 首选的闭环与05血缘

核心路径是：

```text
05 C → 全局Full Mixer → Cbar
历史hard/ST分配 A → Z0=A Cbar（64维）
→ session分轴交互 → 对Cbar距离logits的有界偏置
→ delta=0.1*(PH−P0)W → 加到原AW（256维）
→ 原item_mlp/历史Encoder/4步Decoder → canonical SID目录
```

区别于07之处不是单看维度：新交互实际读取05的物理码字与距离几何，且T推荐反馈经路由回到C/全局Mixer；07直接处理独立W的256维向量。本方案仍允许原历史Encoder继续工作，作用位置在码位合并之前。

R训练推荐器/W/新F，冻结物理码本；T冻结新F和推荐器参数，但保留输入梯度，训练C/全局Mixer。目标SID同一版本不随H变化，每T后全库统一刷新。新F归R是明确的配方调整，不把它说成原05已经存在的功能。

`PH−P0`是有符号残差，不是新SID；R/T/推理都调用相同确定性分支。checkpoint绑定码本、F、推荐器与目录。没有按H重编码全库，也没有target进入历史输入。详见 [DESIGN.md](DESIGN.md) 的形状、公式、掩码、真实接入位置及限制。

## 6. 已运行的验证、尚未回答的问题

[CPU_VERIFICATION.json](evidence/CPU_VERIFICATION.json)：9/9类检查通过。实际导入了当前05 Mixer及自定义ST运算，验证零初始化恒等、context作用、无跨样本/未来位置泄漏、padding/全PAD梯度有限、R/T冻结与回C、loop状态传递，以及合成SID目录碰撞和恢复。

合成toy头没有复现完整DiffGRM target-ST、Gumbel、辅助损失、trainer和生产beam。它不证明真实数据没有预处理泄漏，也不证明真实@50支持。首次失败及修复保留证据；没有通过放松生产审计来交付。

另查到当前joint封装无条件要求target字段，后续需拆分history-only生成与label准备，再验证去掉/置换target不影响预测；本轮不能把合成原型的无target接口写成生产服务已经可用。新增概率几何明确FP32，CPU BF16反向检查也通过，真实CUDA/OCN仍须验收。

主要未知有四个：原Encoder之上是否仍有可学的额外信息；真实码本距离分布是否令残差过弱；共同物理码字坐标是否适合跨book共享更新；质量收益是否覆盖额外编码成本。任何一个都不能从论文结构或非零梯度直接推出答案。

## 7. 下一轮最小方案和判定

第一轮4臂：B0原05 Full Joint；B1与首选同结构但时间轴只自看；B2 full1；B3 axial1。首选B3−B1保持相同参数、两次投影及矩阵尺寸，仅改变时间轴可见性。B2/B3连接模式与投影次数同时不同，只比较整体质量/成本，不宣称纯grid因果效应。

采用共同Aug28推荐器＋canonical E0，35R/60T，首seed2024；统一完整验证集、beam256与SID-level@10。固定R35 NDCG@10为主，HR@10为护栏；不拿旧05 beam32数值代替新B0。@50需要先补真实返回长度与合法唯一候选审计，不能只把指标列表加50。

预设继续标准：首选相对B0及B1的NDCG@10各至少相对+1%，HR@10相对降幅≤0.5%，端到端延迟额外≤20%，才进入复验。这是拟定探索门槛，不是显著性保证。仅full有信号则补同为1次MHA的item-only控制，再判断上下文作用；若单轮没有信号，不直接追加loop追涨。

随后才做r1/r2共享/两层不共享，及seed2025/2026稳定性验证。测试留到方案封存之后。首轮可4臂各1张H100并行，粗预留12–16 GPU小时，先≤0.5小时真实验收测速再核算；**本轮未申请，旧授权不沿用**。完整配置、资源边界和给学长的3个定位问题见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

## 8. 交付与保护边界

- 生产代码、历史报告和已有未提交改动保持原状；新材料仅在本目录。
- 根仓库初态`037876c`，独立DiffGRM `1c87925`，genrec `b6c8b41`。起始与最终状态/保护文件字节比较见 `evidence/INITIAL_STATE.json`、`FINAL_STATE.json`。
- 使用已有CPU环境；优化器step=0，GPU调用=0。作者实现只读，没有安装或执行作者训练环境。
- 原型及证据入口：[DESIGN.md](DESIGN.md)、[CPU检查](evidence/CPU_VERIFICATION.json)、[微基准](evidence/CPU_BENCHMARK.json)、[完成核对](evidence/COMPLETION_AUDIT.json)。

当前结论是“选出了一个可验证且检索闭合的起步设计”，不是“已经证明session-wise有效”。
