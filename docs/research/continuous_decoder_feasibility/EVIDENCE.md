# 关键论断与证据

截止：2026-10-16。R01—R12是12篇核心，S01—S06是6篇必要补证，字段详见[RELATED_WORK.csv](RELATED_WORK.csv)。所有推荐迁移判断另标“推论”，没有复现论文训练或独立审计生产系统。

证据分为：**项目记录**（本项目既有实测/源码）、**机制**（全文与静态代码）、**作者实证**（作者报告的离线/线上结果）、**推论**（本调研分析）、**未知**（本次来源不足）。全文读过不等于结果已复现，作者报告上线不等于独立获得了线上日志。

## P01 项目现状和评价边界

- **项目记录**：[交接§3](../../guides/PROJECT_HANDOFF.md)和[05最新协议](../../../experiments/05_crosspse_joint/pilots/mixer_factorial/README.md)；原始[四组summary](../../../archive/server_sync/mixer_factorial_20261014_results/summary.json)的`arms`/`no_test_evaluation`/`single_tokenizer_block_then_frozen_recovery`。
- 最新四组为seed2024、Joint各24次Tokenizer更新、统一恢复3轮，固定第3轮验证集SID-level；Mixer增量HR@10负、NDCG@10极小正，不能宣布稳定有益或普遍无效。
- 早期长期四组是另一实验族、验证选模后的测试结果；Cross v2最佳点尚未更新Tokenizer，不能归因为联合编码收益。
- [06报告§1—5、§9](../../../experiments/06_continuous_baseline/REPORT.md)与[summary](../../../experiments/06_continuous_baseline/execution_acceptance/final_delivery_20261016/summary.json)：已完成89有效训练；当前是12101商品的完整item-level连续检索基线，不是新连续生成器。现有H100单条SASRec推理p50约1.25ms，没有同条件SID时延/商品质量对照。

## P02 已有离散模型并非逐商品独立生成K次

- **机制**：[model.py L1601–1676](../../../external/DiffGRM/genrec/models/DIFF_GRM/model.py)的`generate`调用`fast_beam_search_for_eval`；[beam.py](../../../external/DiffGRM/genrec/models/DIFF_GRM/beam.py)将活跃beam展成`B*BEAM_ACT`一起前向，按SID深度循环。
- 现有4位置CPD/beam分支不能简单换算成“Top10需要40次串行完整模型调用”；beam宽度增加每步运算量、不同decode配置也会改变深度。这里只静态核对，没有重新测速或声称减少到某个实测延迟。

## E01 连续query与商品表联合学习已是SASRec机制

- **机制**：R01，[1808.09781v1全文](https://arxiv.org/html/1808.09781v1)，§III-D Eq.(6)、§III-E、§IV-C/D。
- 历史输出与共享商品表点积，网络与商品表训练；原评估是正例加100随机负例，且原Beauty规模不同。本项目FullCE、PreLN、全库口径不能直接与原表成绩对比。
- **边界**：没有证明连续生成分布或固定任意Decoder反演；§IV-G的训练速度不是在线请求时延。

## E02 DreamRec生成的是连续oracle向量，但多步生成

- **机制**：R02，[2310.20453v1全文](https://arxiv.org/html/2310.20453v1)，§4.2、Eq.(14)/(15)、Algorithm 2、§4.2.3；[作者代码](https://github.com/YangZhengyi98/DreamRec)的README与`DreamRec.py`。
- 目标恢复、CFG、反向T步；作者命令T=500/2000/500，代码每步条件与无条件MLP；最后仍是商品表内积检索。
- **边界**：代码是本次访问main的静态观察，未锁定截止日期commit，不是历史发表实现的执行证明；本调研没运行。

## E03 DreamRec作者自己指出速度限制

- **作者实证/限制**：同一[全文§6](https://arxiv.org/html/2310.20453v1)：多次迭代造成采样慢、训练更耗时。
- **可支持**：连续表示不天然意味着单次前向或低延迟。**不可支持**：因此整个连续方向推广失败，或所有后续少步方法都慢。

## E04 用户2502.10833v2是SETRec

- **机制**：R03，[版本页](https://arxiv.org/abs/2502.10833v2)、[全文§3.1–3.3 Eq.(5)/(7)、§4](https://arxiv.org/html/2502.10833v2)。真实题名为Order-agnostic Identifier for Large Language Model-based Generative Recommendation。
- CF加多分支语义连续token；多query并行处理同商品内部维度；AE/推荐目标学习与LLM微调。它直接覆盖减少商品内部串行解码，并非任意Decoder的结构自动发现。
- **作者实证边界**：A5000全部测试用户总时间不是单请求p50；8:1:1/all-warm-cold不等于本项目LOO；CF表冻结、完整候选计数、seed与测速合同部分未知。

## E05 ContRec冻结设置与连续空间控制

- **机制**：R04，[2504.12007v5全文](https://arxiv.org/html/2504.12007v5)，§3.2、§3.5.1、§4.1.4。
- σ-VAE预训后，论文骨干阶段冻结Tokenizer；生成先有LLM文本，再有100步扩散。多通道/方差等用于防坍塌；不能归纳为推荐Decoder反馈联合更新Tokenizer的一步方法。
- **实现未知**：[作者early-access代码](https://github.com/Quhaoh233/ContRec)中optimizer包含item/user encoder、`.eval()`不等于冻结；当前静态路径不足以核实论文设置。main未锁截止日期commit，不能升级为已复现错误或推翻论文。

## E06 ContRec的速度和重复次数不可误读

- **作者实证**：同一[全文附录C.2 Table 5及主表说明](https://arxiv.org/html/2504.12007v5)：Beauty每batch总1.7643s，LLM和扩散分别1.5385/0.2258s；作者更倾向对话场景。
- 主表五次是推理重复，不是五次训练seed。该时间缺完整batch/同步/传输绑定，不能除以猜测的batch换算成本项目p50。

## E07 FAVE已研究单步生成和表示空间学习

- **机制**：R05，[2604.04427v1全文§4.2–4.4 Eq.(11)–(25)](https://arxiv.org/html/2604.04427v1)；[作者model.py](https://github.com/Blue130/Fave/blob/main/src/model.py)的`switch_to_finetune`、[fave.py](https://github.com/Blue130/Fave/blob/main/src/fave.py)的`euler_sampler`。
- 第一阶段学空间，第二阶段冻结item/position表，历史mask先验加JVP一致性，finetune采样N=1。不是为任意冻结Decoder训练输入Encoder。
- **作者实证边界**：§5.2屏蔽已交互商品；§6.3 L40/Beauty4.49ms不是当前H100端到端p50合同；具体训练seed/候选统计与时间范围部分未知。代码只静态查阅，main未锁commit。

## E08 MIRAGE是更近的机制重合，证据仍属预印本

- **机制/作者实证**：R06，[2607.23762v1全文](https://arxiv.org/html/2607.23762v1)，§4.3–4.5 Eq.(10)–(18)、§5.1、§5.6、附录B.4。
- 训练期共现图约束商品几何，目标恢复+CE/历史重建，q=1直接恢复；图只用于训练。Beauty统计与本项目相同，但历史20、早停及实际训练候选等仍不同；不能直接搬表分数比较。
- **未知**：作者代码尚未找到；相对速度图无完整计时合同；页眉KDD2027不证明截止时已接收。此处依据2026-07-26已公开v1机制判断重合，不背书其所有性能主张。

## E09 离散SID也不能预设比ANN更快

- **机制/作者限制**：R07，[TIGER 2305.05065v3全文](https://arxiv.org/html/2305.05065v3)，§3、附录E。
- SID在推荐训练期间固定；4位置含冲突后缀，beam有成本；作者附录明确讨论推理可能比ANN贵。
- **边界**：这是该方法的结构/限制，不是全体SID效率定理；本项目CPD也不等于原TIGER AR。

## E10 混合方法的端到端含义要精确

- **机制**：R08，[COBRA 2503.02453v1全文§3.1–3.4](https://arxiv.org/html/2503.02453v1)，Eq.(1)、(7)–(13)。先SID后条件dense、分组ANN、BeamFusion；dense Encoder与推荐器联合。
- **边界**：没有据此证明离散RQ分配也接受推荐梯度；也没有消除全部SID步骤。它说明混合路线可行，不是当前Mixer有效的证据。

## E11 推荐目标学习SID已有直接先例

- **论文机制主张**：R09，[DIGER 2601.19711v3全文§3–4](https://arxiv.org/html/2601.19711v3)，软分配/ST、推荐与辅助目标，仍输出SID。
- **边界**：作者代码更正见E12。项目05的本地ST检查是另一份独立证据，不能将“同名DIGER”当论文全配置复现。

## E12 DIGER作者对历史实现有明确更正

- **固定版本静态证据**：[main README](https://github.com/junchen-fu/DIGER/blob/fafed45ba1234328f332d42c2541e95ee0bf1e0c/README.md)、[旧trainer L376–377](https://github.com/junchen-fu/DIGER/blob/fafed45ba1234328f332d42c2541e95ee0bf1e0c/trainer.py#L376-L377)；[修复trainer](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/trainer.py)、[修复launcher](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/scripts/run_gradient_fix_frqud.sh)。commit日期分别2026-08-30和08-20。
- 作者承认旧路径推荐梯度detach；修复代码显示soft反向/硬前向路径，默认固定原ID Encoder、更新RQ和推荐器。
- **可支持**：不能用旧论文结果单独确认梯度联合机制有效。**不可支持**：本项目05梯度必然也断了，或新分支效果已被我们复现。

## E13 连续路线存在生产采用报告，但不是普及率调查

- **作者一手实证**：R10，[PinRec v7 §4.2–4.5、§6](https://arxiv.org/html/2504.10507v7)连续向量+ANN与多surface上线；S01的NDR见E19（成熟检索）；R08 [COBRA §4.4](https://arxiv.org/html/2503.02453v1)是混合召回的广告A/B。
- [NVIDIA GTC26官方Pinterest工程师会话](https://www.nvidia.com/zh-tw/gtc/session-catalog/sessions/gtc26-p81273/)为PinRec部署补证，不能与论文合计为两套独立线上实验。
- **发表元数据**：[Crossref正式DOI登记](https://api.crossref.org/works/10.1145/3770855.3818462)经独立元数据复核成功读取，确认KDD2026、ACM proceedings-article、7037–7048、online 2026-08-08。正式题名在Retrieval后多Model；[arXiv版本页](https://arxiv.org/abs/2504.10507v7)列相同related DOI。直接ACM正文403，本轮再次访问Crossref也受限制；全文机制仍依据可读arXiv v7，不宣称读过ACM正文。登记细节见[混合/工业笔记](notes/discrete_hybrid_sources.md)。
- **未知**：没有全行业采用比例、失败部署样本或标准化推广调查。不能断言已经普及，也不能因少见报道断言未推广/已失败。

## E14 PinRec生产时延是质量与系统工程权衡

- **作者实证**：[PinRec v7 §4.5.3–4.5.4](https://arxiv.org/html/2504.10507v7)。L40S+CPU Faiss IVF-HNSW，80QPS时OC p50/p90=40/65ms、UC=23/46ms，单模块p50约两塔3–4倍；服务并行后总体延迟增量<1%。多商品生成仍AR。
- **边界**：不是固定Decoder方法，也不是少步生成必然更快；离线百万随机候选未来集合召回，不是本项目12101单目标全库。平台MAU不等于候选库或训练集大小。

## E15 pSp是固定可微生成器上的编码参数学习

- **机制**：R11，[pSp 2008.00951v2全文§3、Fig.2、Appendix A](https://arxiv.org/html/2008.00951v2)，人工FPN/map2style输出W+，固定G只训练E，以图像任务损失回传。
- [固定代码coach.py](https://github.com/eladrich/pixel2style2pixel/blob/5cfff385beb7b95fbce775662b48fcc80081928d/training/coach.py#L173)、[train_options.py](https://github.com/eladrich/pixel2style2pixel/blob/5cfff385beb7b95fbce775662b48fcc80081928d/options/train_options.py#L24)核对可选train_decoder及默认False；2022代码默认参数不等于2021论文配置。
- **边界**：不是自动网络结构搜索，也不是固定推荐query函数的逆。

## E16 pSp摊销加速有误差与可达域限制

- **作者实证/限制**：同一[全文§4.1 Table 1、§5](https://arxiv.org/html/2008.00951v2)：预测潜码可快于逐图迭代，但重构未必更优，生成器未覆盖的内容会失败。
- **可迁移启发**：比较直接优化latent与E预测的近似差距，并计入E训练成本。**不可迁移的结论**：其图像秒数直接证明推荐提速。

## E17 用户PMLR论文的“通用”限定同架构模型族

- **机制/身份**：R12，[PMLR正式记录](https://proceedings.mlr.press/v118/zhang20a.html)、[全文§1–2及Appendix B.2](https://proceedings.mlr.press/v118/zhang20a/zhang20a.pdf)。Rapid Model Comparison by Amortizing Across Models；AABI2019/PMLR2020，非推荐论文。
- 输入(x,theta)，固定theta，跨训练模型参数分布优化E的ELBO；topic-aware交互与两层MLP仍人工设计。[固定workshop代码](https://github.com/tufts-ml/any_parameter_encoder/tree/17069bc84dd8e56031c8cca2612dcb5da93b0c2b)对应论文路径。
- **边界**：同架构、同参数域，不是所有网络；product reviews是文本topic任务，不是商品推荐实验。

## E18 APE的摊销存在近似差距

- **作者实证**：同一[PDF Table 1、Fig.1、Appendix A/B](https://proceedings.mlr.press/v118/zhang20a/zhang20a.pdf)。300组合中APE0.05s、VI216.30s，但预测likelihood与模型排序一致性未等价于VI（真实文本排序一致率75%）。
- **边界**：时间不含全部元训练；E对未覆盖theta/新架构的泛化不能预设。PDF用文本核对；图像截图失败，未宣称做过公式页视觉复核。

## E19 NDR是大库连续检索的正式工业先例

- **作者实证**：S01，[官方PDF§5、§6.3、Tables 2/3](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/6417b9a68bd77033d65e431bdba855563066dc8c.pdf)。两塔共同学习，实验约10M视频index每几小时更新，离线跨日留出与线上A/B；新增召回通道，而非替换全部排序系统。
- **边界**：2019年的实例不提供2026年全行业普及率，更不是连续随机生成已部署的替代证据；后者另见PinRec。

## E20 下游打分感知编码也有成熟先例

- **机制/作者实证**：S02，[AVQ/ScaNN 1908.10396v3全文§2–5](https://arxiv.org/html/1908.10396v3)。对固定MIPS改量化损失、相同bitrate比较、包含预处理的speed-recall曲线；Glove1.2M/单线程CPU。
- **边界**：若命题转向“给固定打分器学编码以压缩/提速”，应纳入该强近邻；它未处理任意非线性Decoder，也不是商品级HR/NDCG。

## E21 连续生成并不要求取消分类监督

- **机制/作者实证**：S03，[DiffuRec 2304.00686v4 PDF§4、Eq.(18)–(20)、§5.3/5.9](https://arxiv.org/pdf/2304.00686v4)。连续加噪/恢复后用全商品CE/rounding，默认32反演步，扫描2–1024。
- **边界**：少步时间图不等于默认32步优于SASRec，也没有本项目计时合同。

## E22 下游可用性不能用重构Loss单独替代

- **作者实证（跨领域）**：S04，[e4e 2102.02766v1全文§3–8、Tables 2–4](https://arxiv.org/html/2102.02766v1)。固定StyleGAN2，控制offset/latent分布，重构失真、感知和编辑存在权衡；不同指标也非处处一致。
- **边界**：仅提供评价设计的启发，不直接证明推荐中某个正则更好；不是CVPR论文，作者仓库标SIGGRAPH2021。

## E23 固定推荐骨干加商品编码旁路已有直接近邻

- **机制/作者实证**：S05，[EPIC 2609.03522v1 §4.2–4.3、§5.3、Table 3](https://arxiv.org/html/2609.03522v1)。SID/Tokenizer/主干冻结，学习完整SID item encoder与历史推理，后验残差干预token；四步不变。
- T4上训练参数0.391M vs全调6.847M，推理93→124.8ms/user，增加34.2%。不等于纯输入Encoder替换、原函数不变或速度提高。
- **未知**：匿名作者仓库入口访问失败；只有正文机制和作者结果，未审计/复现代码。

## E24 低参数商品表适配已有更直接先例

- **机制**：S06，[FedPEFT 2512.13734v1 Methodology/Eq.(1)–(2)](https://arxiv.org/html/2512.13734v1)：预训/warmup后冻结完整E，学习E+B(A)；A是逐商品低维表，B零初始化。另有hash/RQ残差，不等于本方案共享V m_i，也未固定整个query主干。
- **发表与实证范围**：[AAAI2026正式记录](https://ojs.aaai.org/index.php/AAAI/article/view/38660)，40(19):16235–16243，DOI 10.1609/aaai.v40i19.38660；正文Table1/Table3为小库联邦训练与通信分析，不是线上ANN延迟。候选评估口径、源码冻结契约未审计。
- **推论**：参数高效商品适配不是空白；NEXT_STEP只作受限机制诊断，不证明超过FedPEFT等全部已有适配。

## I01 固定query输出适配为何属于分类头微调

**推论，结合项目源码**：[scripts/continuous/model.py L110、L135–171](../../../scripts/continuous/model.py)。固定query q后，优化候选V的CE恰是多分类线性头训练；V=E_phi(x)为受约束的头参数化。输入表与输出表必须解耦才能固定query。

S=QV^T的秩不超过query维度；相同query给所有候选产生同一组打分，无法仅靠新V恢复用户区分。向所有V加相同向量，softmax概率不变。因此编码不是唯一可识别的“真实embedding”，白盒访问也不保证可逆/可达。

这些是接口与代数分析，不是某论文已证明本项目会涨点或不会涨点。

## U01 仍未证实、不能据此下结论的内容

1. 连续生成的行业采用比例及具体失败部署原因；不存在可据本轮资料量化的“推广失败率”。
2. 所有方法在本项目完整前缀、相同容量/调参、同设备、同候选过滤下的排序和质量—延迟优势。
3. 一个训练好的E对任意新架构Decoder零训练迁移，或自动产生Encoder结构。
4. 在现有Beauty上，共享输出Encoder是否优于自由表和同参数低秩线性适配；本轮没有训练证据。
5. 所有作者公开实现与发表实验是否一致；浮动main、代码缺失和作者更正已逐条注明。

细节与引用追踪见[连续生成笔记](notes/continuous_sources.md)、[离散/混合笔记](notes/discrete_hybrid_sources.md)、[反演笔记](notes/inversion_sources.md)、[连续检索笔记](notes/retrieval_sources.md)与[检索记录](SEARCH_LOG.md)。
