# 连续检索、查询感知量化与项目接口核对

核对日期：2026-10-16。这是全文阅读笔记，不是实验复现。本文不复制论文成绩作为本项目的可比分数。

## R01 SASRec：成熟参照，不是新连续生成算法

- 题名：Self-Attentive Sequential Recommendation；Kang、McAuley；ICDM 2018。此处阅读全文版本[1808.09781v1，2018-08-20](https://arxiv.org/html/1808.09781v1)。
- §III-A：历史商品查表得到输入；§III-D Eq.(6)：历史网络输出与共享商品表点积，按分数排序。§III-E用正负商品的二分类目标共同学习网络和商品表，未冻结一个任意预先给定的推荐器。
- 在一个历史上做一次查询网络前向，即可对多个候选打分；不等于逐个商品连续采样或列表联合生成。
- §IV-A/Table II：原论文Beauty为52,024用户、57,289商品，非本项目22,363/12,101；§IV-D用1个正例加100个随机负例排名，非全库评估。§IV-G训练效率数字不能当在线端到端推理时延。
- 与本项目：06采用另一PreLN/GELU/FullCE实现，且全前缀末位监督；历史向量→query→商品表打分的基本机制已有。不能把换损失和充分调参描述成新生成范式。

## S01 工业双塔：NDR的规模、采用及索引代价

- Yi等，Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations，RecSys 2019。[官方记录](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)，[已阅读全文的官方PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/6417b9a68bd77033d65e431bdba855563066dc8c.pdf)。短地址`pubtools/5716.pdf`曾超时，改用同机构长地址取得全文。
- §5/§5.1：YouTube的额外召回通道，query和candidate两塔共同学习；候选塔使用视频ID及内容等特征；不是无条件冻结的通用Decoder反演。
- §6.3（PDF第8页）：约10M视频索引每几小时重建，动态新视频是索引更新的实际原因；10%新日数据留出，跨7天汇总。Table 3报告加入召回通道的真实A/B，不能解释为替代整个推荐系统。
- 作用：反驳“大库不能做连续向量检索”和“连续路线没有工业采用”的笼统前提；不证明概率连续生成普及，更不说明任意模型都应使用向量检索。

## S02 ScaNN/AVQ：固定查询下改编码，并不是空白问题

- Guo等，Accelerating Large-Scale Inference with Anisotropic Vector Quantization，ICML 2020；[1908.10396v3，2020-05-12全文](https://arxiv.org/html/1908.10396v3)。
- §2.1/§2.2：预先编码候选、在线编码query、做MIPS；可用分区减少比较对象、量化降低带宽和存储。精确全库打分只是实现选择之一。
- §3与§4：重构误差不必与点积排序误差一致；对高分query-item对加权，改变码本和分配目标。已有“让编码适应下游打分”的具体先例，但不是为任意非线性Decoder设计编码。
- §5.1固定其他条件比较重构与score-aware目标；§5.2在相同100/200 bits下比较；§5.3将预处理算入speed-recall曲线。Glove1.2M是120万条100维词向量，CPU Xeon W-2135单线程，不是商品级推荐结果或本项目H100测速。
- 作用：若把下一研究改成压缩/大库效率，AVQ、PQ/OPQ与ANN质量—速度曲线应成为强对照。不能仅与普通MSE重构相比就声称提出下游感知编码，也不能将普通未压缩向量当作唯一大库基线。

## P03 本地模型接口：固定Decoder到底固定什么

2026-10-16只读检查：`scripts/continuous/model.py`。

- L110定义一张商品表；L135–150用这张表编码历史并返回末位query。
- L152–154打分再次读取同一张表；L156–171以FullCE或sampled softmax训练。
- 因此直接解冻`item_embedding`会同时改变输入与输出，不能回答“固定查询，只改输出编码”。必须冻结原输入表及网络，另建输出副本。
- 在固定query的设置，`q(h)^T E_phi(x_i)`是受参数化约束的线性分类头。更少可训练参数不等于更少总存储；如果仍保留原商品表，不能声称消除了商品表。
- 全部正式模型/输入输出表/映射在`experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/LOCAL_ARTIFACT_INDEX.json`中有本地定位；本轮只查看索引，不加载运行模型。

## 本调研的逻辑推论（不冒充论文结论）

1. 固定query矩阵Q后，新得分为S=QV^T，秩仍不超过query维度；两个历史若产生相同query，任何候选编码都无法让它们获得不同排名。这限制了“任意Decoder均可由新Encoder补救”的说法。
2. 固定生成器G的图像反演学习E，使G(E(x))接近x，与固定推荐query后调整候选V，函数方向和目标不同，不能直接类比为同一实验。
3. 对多解目标使用单点平方误差会倾向条件均值；这只针对该目标/模型，不说明所有连续生成都会取平均或失去多样性。
4. 无约束“学商品向量”已有成熟先例。可检验价值必须另加适配参数、数据量、候选更新或泛化等明确条件；在新颖性审查前，涨点只代表局部机制信号。
