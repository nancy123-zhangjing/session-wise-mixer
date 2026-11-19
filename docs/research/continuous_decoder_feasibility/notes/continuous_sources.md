# 连续生成路线：一手来源精读证据

检索与阅读截止：2026-10-16。范围为5篇核心（DreamRec、SETRec、ContRec、FAVE、MIRAGE）和1篇补证（DiffuRec）。没有运行作者代码、训练或修改既有实验。以下“论文报告”“公开代码观察”“本项目推断”分别标明；不把跨论文分数拼成排行榜。

本文不使用英文逐字引文，全部为短述、事实字段和必要公式定位；后续总报告不必再复制论文原句。

## 先给结论

1. **连续商品表示生成已经有充分先例；一步生成也已经有直接先例。** DreamRec支持生成向量再落地商品，但其作者复现命令使用500–2000步；FAVE、MIRAGE明确实现一步目标恢复。仅把SID改为连续向量，或仅把生成压成一步，不足以构成新的研究贡献。[DreamRec方法与算法](https://arxiv.org/html/2310.20453v1)、[FAVE §4.4](https://arxiv.org/html/2604.04427v1)、[MIRAGE §4.5](https://arxiv.org/html/2607.23762v1)。
2. **“根据生成任务学习表示”也已有高度相关工作，但不等于为任意固定Decoder自动构造Encoder。** SETRec训练语义Tokenizer并联合适配LLM；FAVE先学习商品空间、再冻结embedding做一步整合；MIRAGE在流模型训练时重排可学习商品表几何。它们均预设网络与损失，没有自动搜索任意Encoder结构或证明任意Decoder可适配。[SETRec §3](https://arxiv.org/html/2502.10833v2)、[FAVE §4.2–4.3](https://arxiv.org/html/2604.04427v1)、[MIRAGE §4.3–4.4](https://arxiv.org/html/2607.23762v1)。
3. **ContRec不宜当作低延迟或Decoder反馈学Tokenizer的直接模板。** v5论文设置在推荐骨干训练时冻结预训练Tokenizer，推理100次扩散，且先生成LLM推理文本；公开early-access代码与论文冻结描述还存在待核验差异。[ContRec §3.5、§4.1.4、附录C.2](https://arxiv.org/html/2504.12007v5)、[作者代码](https://github.com/Quhaoh233/ContRec)。
4. **最值得保留的研究空间是更严格的条件，而不是更大的口号。** 例如“固定一个已有Decoder，在不改它的参数和在线代价时，用可复用方法学习输入/输出编码，并证明改善超出自由商品表重训练”。这只是据下述差异提出的可检验问题，不是已确认的创新空白。

## 版本与来源登记

| 条目 | 本次精读版本及日期 | 一手全文 | 作者代码及本次检查深度 |
|---|---|---|---|
| DreamRec | 2310.20453v1，2023-10-31 | [HTML](https://arxiv.org/html/2310.20453v1)、[PDF](https://arxiv.org/pdf/2310.20453v1) | [YangZhengyi98/DreamRec](https://github.com/YangZhengyi98/DreamRec)；读README、DreamRec.py关键训练/采样/评估路径 |
| SETRec | 2502.10833v2，2025-05-25；SIGIR 2025 | [版本页](https://arxiv.org/abs/2502.10833v2)、[HTML](https://arxiv.org/html/2502.10833v2)、[PDF](https://arxiv.org/pdf/2502.10833v2) | [Linxyhaha/SETRec](https://github.com/Linxyhaha/SETRec)；README、配置入口，尚未完整定位主模型文件 |
| ContRec | 2504.12007v5，2026-02-24 | [版本页](https://arxiv.org/abs/2504.12007v5)、[HTML](https://arxiv.org/html/2504.12007v5) | [Quhaoh233/ContRec](https://github.com/Quhaoh233/ContRec)；读README、code/model_interface.py |
| FAVE | 2604.04427v1，2026-04-06；版本页注明SIGIR 2026接收 | [版本页](https://arxiv.org/abs/2604.04427v1)、[HTML](https://arxiv.org/html/2604.04427v1)、[PDF](https://arxiv.org/pdf/2604.04427v1) | [Blue130/Fave](https://github.com/Blue130/Fave)；读src/model.py、src/fave.py、README |
| MIRAGE | 2607.23762v1，2026-07-26；按预印本处理 | [版本页](https://arxiv.org/abs/2607.23762v1)、[HTML](https://arxiv.org/html/2607.23762v1) | 论文称代码待发表公开，本次未找到作者仓库，无法核对实际入口 |
| DiffuRec（补证） | 2304.00686v4，2023-10-30；TOIS文章版本 | [版本页](https://arxiv.org/abs/2304.00686v4)、[28页PDF](https://arxiv.org/pdf/2304.00686v4) | [WHUIR/DiffuRec](https://github.com/WHUIR/DiffuRec)；确认作者仓库，未逐模块审计 |

版本注意：ContRec v1题为“Generative Recommendation with Continuous-Token Diffusion”，v5题为“Diffusion Generative Recommendation with Continuous Tokens”，本记录以截止日期内的v5为准，不混用版本设置。MIRAGE v1正文页眉写KDD 2027，但其arXiv公开日期为2026-07-26；**不能把未来会议模板当作截至2026-10-16已获接收的证据**。

代码链接为本次可读的公开main快照。尝试GitHub提交API未成功，未取得带截止日期的commit SHA；因此只据可见源码报告控制流，不将其升级为已锁定、已复现的论文执行版本。没有下载或执行仓库。

## 1. DreamRec：向量生成成立，但原方案不是一步

**身份与重点阅读。** Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion。读§4.1、§4.2.1–4.2.3、算法1/2、§5.1、表1、附录B；不是只读摘要。[全文](https://arxiv.org/html/2310.20453v1)。

**机制与训练对象。** 历史Transformer生成条件c，去噪MLP恢复目标embedding；Eq14为目标恢复MSE，Eq15为classifier-free guidance，算法2从高斯噪声迭代T步。最终向量与商品表做内积Top-K（§4.2.3），生成向量本身可不等于真实商品。作者代码用同一个可学习item_embedding作为历史输入和目标，optimizer接收model.parameters；没有冻结给定Decoder再单独训练编码器。[DreamRec.py](https://raw.githubusercontent.com/YangZhengyi98/DreamRec/main/DreamRec.py)，重点Tenc、cacu_x/cacu_h、训练主循环。

**真实调用成本。** [作者README](https://github.com/YangZhengyi98/DreamRec)三个复现命令分别设YooChoose T=500、KuaiRec T=2000、Zhihu T=500。DreamRec.py的sample（约197–204行）确有反向for循环，p_sample（约182行）每步调用条件和无条件MLP；历史编码可先做一次，但“产生一个商品向量”并非一次完整模型前向。论文搜索T∈{50,100,200,500,1000,2000}。不能用其标题支持一步加速。

**证据条件。** §5.1：历史10、维度64，8:1:1顺序划分；附录B商品数9,514/7,261/4,838；表1五次实验均值±SD。论文候选排名用内积，公开代码对整表打分；未见ANN或百万级商品库证据。主文没有符合当前项目p50/p95、同步、H2D范围的速度表。代码evaluate固定batch100并以整除循环取批，predict打分包括item_num+1表而未显式剔除PAD；若复用要独立审计尾批/PAD及每10轮测试访问，不能照搬为本项目无泄漏验收。该只读观察不反推发表结果真假。

**支持与反证。** 支持“直接生成偏好向量再检索”的语义；反证“连续必然一步/必然比SID快”。局限是目标表与生成器共同学习、无固定任意Decoder适配结论。对本项目可借鉴的是条件与生成器解耦的接口、检索计入完整链路，以及把空间坍塌作为单独诊断；不能预设去负例MSE一定可靠。

## 2. SETRec：用户2502.10833v2的真实对应

**身份纠正。** 该链接确实是推荐论文，真实标题为 **Order-agnostic Identifier for Large Language Model-based Generative Recommendation**，方法SETRec；不是通用逆问题或编码器结构自动发现论文。[版本元数据](https://arxiv.org/abs/2502.10833v2)。

**机制与训练对象。** §3.1明确采用连续token：一个预训练CF向量加N个语义向量；语义由AE多分支编码、重建。§3.2/Eq5用不同可学习query并行生成多个信息维度，sparse attention去掉同商品内部依赖，保留历史商品时序；Eq7对每维全商品token表打分，再融合CF/语义分数，不用逐索引beam。§3.3联合生成损失与AE重建；T5全量微调，Qwen LoRA，不是给定固定Decoder仅学Encoder。[方法与实验](https://arxiv.org/html/2502.10833v2)。

**步骤与检索。** 一个item的N+1维token并行生成，不是迭代扩散，也不是并行生成10个推荐位置的列表模型。仍有多维商品表打分/grounding成本。论文§4：Toys/Beauty/Sports/Steam，8:1:1划分，按all/warm/cold候选分别Recall/NDCG@5/10。不是本项目逐用户leave-one-out；未在所读正文找到可直接核验的四库完整候选数或seed列表，不能默认12,101或五训练seed。

**效率证据范围。** 表1/2给单张RTX A5000上**全部测试用户**的推理总秒数，不是单请求p50；训练用4张A5000。测试计时的batch、预热、同步、数据传输范围尚未明确，故不与当前H100 1.25ms比较。表格HTML若把时间拆成“1 26”一类文本，不据此手工猜小数。

**代码证据与未知。** [作者README](https://github.com/Linxyhaha/SETRec)允许使用SASRec_item_embed.pkl预训练CF token，语义Tokenizer在SETRec训练中创建并训练；提供T5训练/推理入口，Qwen独立推理脚本标为待完善。本次未定位到完整主模型源码，CF底表是否冻结、投影层梯度与一次调用的实际batch形状尚未完成代码验证，不用README代替完整检查。

**对本项目的判断。** 这是“减少同商品内部串行生成、并行产生连续表示”的强直接先例；把编码方案服务于下游推荐目标也不是空白。可借鉴多query和grounding接口、AE约束与CF/语义分工；若要声称通用方法，必须再证明跨固定Decoder协议，而非仅把同一框架适配到可微调T5/Qwen。

## 3. ContRec：连续Tokenizer与LLM-扩散接口，有明确速度代价

**重点位置。** v5 §3.2 Eq9–12、§3.4、§3.5、§4.1.4、表1/2、附录C.2表5。[全文](https://arxiv.org/html/2504.12007v5)。

**论文方法。** σ-VAE把预训练协同/语义表示变成多通道连续token；固定噪声方差、mask与多通道防坍塌。LLM先生成推理内容，再给扩散条件；扩散生成偏好向量后，以cosine及类别/品牌文本加权做商品grounding。§3.5.1明确先训练Tokenizer，骨干阶段保持Tokenizer固定；端到端指LLM与扩散损失联通，**不是Tokenizer被推荐Decoder反馈联合优化**。§4.1.4为1000训练噪声步、100推理步，且另有LLM文本生成，不能把§3.5“single inference process”解释为一次神经网络前向。

**评估证据。** 全局时间点90/95百分位划分、历史最多20；Beauty表1列12,101商品但交互278,641，且与22,363用户、平均长度8.9存在待核对统计关系，不能据同名Beauty判定数据字节相同。表2是**五次推理**均值±SD，不是五次独立训练seed；主文未给seed列表。全商品库/历史屏蔽以实际代码路径另核对。

**速度反证。** 附录C.2表5为每batch平均秒数：Beauty总1.7643、LLM 1.5385、扩散0.2258；作者认为更适合对话推荐而非海量在线后端。附录实现提到batch24/A800，但未把该batch与测速表逐项绑定，亦无预热/同步/H2D合同，不能换算成单请求毫秒或与本项目直接对比。

**公开代码与论文的待核对差异。** [model_interface.py](https://raw.githubusercontent.com/Quhaoh233/ContRec/main/code/model_interface.py)：generate约106–143行确实先model.generate再diffusion_net.predict；训练conditioning约65–84行取answer的token embedding，经output_decoder作为条件，并非该文件直接取LLM hidden输出。configure_optimizers约275–305行把item/user_encoder列入AdamW；insert_diffusion_tokens只eval()，eval不等于冻结梯度。故当前early-access快照尚不能证明论文“冻结Tokenizer”的实现一致性。test的inner分支还引用val_content，不能不审计直接复用。没有实际运行这些分支，不宣称已经复现bug或推翻论文。[仓库状态](https://github.com/Quhaoh233/ContRec)。

**研究启示。** 可借鉴连续空间坍塌控制与条件接口，不能据它宣称低延迟或已解决任意Decoder适配。若拿此为先例/基线，要先明确采用论文版本还是当前源码，并独立验收训练梯度、冻结范围与评估。

## 4. FAVE：一步与两阶段空间训练的直接先例

**身份。** 2604.04427v1，2026-04-06；[arXiv版本页](https://arxiv.org/abs/2604.04427v1)注明Accepted by SIGIR 2026。[全文](https://arxiv.org/html/2604.04427v1)。

**机制/冻结。** §4.2 Eq11把流匹配写成目标embedding恢复；Eq12全库目标CE与源端历史重建避免可学习商品表坍塌。§4.3.1明确第二阶段冻结embedding，用历史中随机商品的Bernoulli mask表示作先验；平均速度与JVP一致性整合轨迹。§4.4 Eq25一次前向输出目标向量，再与全部候选点积。网络/三层MLP decoder结构是预设的；这里的decoder主要承担历史重建，不等同用户说的任意推荐Decoder。

**作者代码核对。** [src/model.py](https://raw.githubusercontent.com/Blue130/Fave/main/src/model.py)49–57行switch_to_finetune冻结item和position embedding；246–273行推理构造随机历史先验并传finetune。 [src/fave.py](https://raw.githubusercontent.com/Blue130/Fave/main/src/fave.py)375–402行euler_sampler在finetune设N=1，循环内一次xstart_model，返回nfe=N，支持真实one-step；pretrain仍用sample_N。它还保留随机历史选择/mask与forward内噪声，故一步不等于完全确定的点估计。仅静态核对，未执行。

**评估边界。** §5.2对全部**未交互商品**排序，H/N@10/20，维度128、4头、batch512、历史50；历史屏蔽与当前项目不屏蔽历史不一致。三数据集ML-100k/Beauty/Steam，正文未完整给可核验商品规模、划分细节与五训练seed，保留未知。表1虽有显著性说明，本次未见明确训练重复数/seed列表，不能替它补成五seed。

**效率范围。** §6.3表3，单张L40；Beauty per-sample latency：DiffuRec103.19ms、FMRec93.74ms、FAVE4.49ms；对应Infer.Time为42.21/40.44/2.25s。论文未充分绑定Infer.Time的样本/批总量、warmup、同步、H2D、TopK计时边界；因此只可说其同文测量中快于这些迭代基线，不可说快于本项目SASRec，也不能把4.49ms解释为生产请求p50。表2包含stage/consistency/matching/prior消融，支持至少需把一步机制与表示约束拆开看。[表2/3与§6.3](https://arxiv.org/html/2604.04427v1)。

**对研究的强反证。** “连续生成+学习item embedding+防坍塌+一步推理”已高度覆盖；新的切入点必须更具体，例如固定既有Decoder时编码学习的约束/可迁移性，或在相同质量下真正降低完整检索开销，不能只改损失名字。

## 5. MIRAGE：重整商品几何、保留一步推理的更近先例

**版本/可验证性。** [2607.23762v1](https://arxiv.org/abs/2607.23762v1)，2026-07-26，截止前公开；按预印本登记，KDD 2027页眉不作已接收证据。正文明确代码待发表公开，本次无源码可验。[全文](https://arxiv.org/html/2607.23762v1)。

**机制。** §3.3只从训练交互构建固定共现邻居；§4.3同时用目标恢复、catalog CE、历史重建；§4.4 Eq10–14用路径内部状态到目标邻居的距离，权重4t(1−t)，更新共享商品embedding几何；不添加独立通用Encoder，不改直线概率路径。§4.5 Eq16为q步Euler，q=1近似为Eq17直接目标恢复，Eq18仍对全商品打分。图与邻居检索仅训练用。

**评估与支持。** §5.1/表2：Beauty 22,363用户/12,101商品/198,502交互，另Sports18,357、Toys11,924、CDs64,443商品；历史只保留20，LOO/full-catalog，N@20早停，RTX4090。主表五seed2022–2026；后续分析只固定2026。表1Beauty MIRAGE H@10=8.53±0.13%、N@10=4.94±0.07%；这只是同文证据锚点，**不能与本项目len50、数据字节/训练覆盖/历史屏蔽合同未对齐的结果并榜**。

**效率与不确定性。** §5.6/Fig6展示相对延迟与1–30步稳定性；附录B.4称13s/epoch对FMRec/FAVE11s。没有给足独立可复算的batch、预热、同步、传输与检索计时协议；源码未公开，不能独立确认图确实完全移出推理或one-step实际入口，也不把“相同容量”推广为所有比较均严格公平。MIRAGE的q=1仍从高斯源采样，并非确定query。§4.3训练ranking允许全库或采样候选，实际主表所用负例配置尚待源码确定。

**对本项目的强反证/启示。** 通过生成器目标重塑商品空间且不增加在线图成本，已是明确工作；普通联合训练或换连续item表不足以宣称创新。更有价值的问题是：若Decoder保持固定，适配损失怎样改变编码且保留可区分性？这与MIRAGE的联合流模型+商品表几何训练有可测差异，但目前不能仅据“未写任意Decoder”就断言没有相关先例。

## 6. DiffuRec补证：连续生成可以继续使用分类监督

精读[2304.00686v4 PDF](https://arxiv.org/pdf/2304.00686v4)§4、算法1/2、§5.1/5.3/5.9，版本日期[2023-10-30](https://arxiv.org/abs/2304.00686v4)。目标embedding逐步加噪，Transformer重建；Eq18–20用全商品CE训练与内积rounding，不是纯MSE。§5.3默认reverse steps=32，探索2–1024；§5.9/Fig11明确速度随反演步数变化，2步接近SASRec类模型，但没有证明默认32步更快。

表1商品数Beauty12,101、Toys11,924、ML1M3,416、Steam13,044；LOO、全候选H/N@5/10/20，Beauty历史50，ML1M200；每方法五次重复平均，具体seed未列。RTX3090+Xeon的单样本时间图没有本项目端到端测量合同。[作者仓库](https://github.com/WHUIR/DiffuRec)确认存在，未完整审计源码，不声明冻结范围已代码复核。

它足以反驳“输出连续向量就不能用CE/原分类机制全推翻”；也说明MSE、CE和多步生成是可分开的设计选择。默认32步不能被误写为一次生成。

## 横向可比性与未知项清单

| 文献 | 生成步数证据 | 商品落地 | 编码/商品表与骨干关系 | 不能填补的未知 |
|---|---|---|---|---|
| DreamRec | 作者命令500/2000/500，CFG每步条件/无条件MLP | 整表内积TopK | 可学习共享item表与生成器联合训练 | 无严格端到端速度合同；main未锁commit |
| SETRec | 多query并行连续token，无扩散循环 | 多维全商品表打分后融合 | 语义AE与推荐训练；T5全调/Qwen LoRA | 主模型源码定位、CF底表冻结、候选规模/seed及计时batch |
| ContRec | 论文1000训练噪声步、100推理步，另有文本生成 | cosine+文本语义加权 | 论文先训Tokenizer后冻结；公开代码待核验一致性 | early-access论文/代码差异，五推理≠五训练，统计/速度范围 |
| FAVE | Eq25与源码finetune N=1 | 全部未交互商品内积 | 第一阶段学空间、第二阶段冻item/position表 | candidate精确规模、seed/重复数、完整计时合同 |
| MIRAGE | Eq17、默认q=1；无代码验证 | 全商品内积 | 共享表几何与流骨干训练，无新增通用Encoder | 未公开源码、历史屏蔽、实际训练候选、相对图计时合同 |
| DiffuRec | 默认32，扫描2–1024 | 全商品CE/内积rounding | 可学习分布表征+Transformer，未逐模块核验 | 具体seed、速度同步/传输边界、源码版本 |

后续最小方案应同时固定：输入数据字节/训练目标覆盖、候选库与历史屏蔽、模型容量和监督预算、模型/编码器冻结范围、checkpoint选择、训练seed与采样seed、NFE及是否包含conditioning/ANN/TopK/H2D。缺任一项，不把“更快/更好”解释成编码学习本身的收益。

本次流匹配追踪还查到[FlowRec 2508.17618v1](https://arxiv.org/html/2508.17618v1)（历史行为先验与single-step alignment）；为遵守核心条目预算，没有继续做其作者代码/表格精读，也未纳入上表的已审计核心。不是因为发现其无关。未来若主方案采用同类历史先验，必须先补读它，不能忽略该近邻。

## 追加定向反证检查（不扩核心条目）

仅做两组检索：第一组围绕 frozen/item embeddings/low-rank、frozen backbone/item adapter、parameter-efficient item encoder；第二组核对 FedPEFT、REPREC 以及 frozen recommender/item-side adapter。没有追加第三组；二手摘要命中的 PFA 等未取得一手全文，不用于下述结论。有限检索不能证明“固定 query、商品侧共享映射”是研究空白。

**S06补证 FedPEFT。** *Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation*，[2512.13734v1，2025-12-14](https://arxiv.org/abs/2512.13734v1)，[AAAI官方条目](https://ojs.aaai.org/index.php/AAAI/article/view/38660)。[全文Methodology/Compressed Embedding，Eq.1–2](https://arxiv.org/html/2512.13734v1)冻结完整商品底表，学习残差 $B a_i$，$a_i$逐商品可学、$B$零初始化，是比AVQ更贴近的商品侧PEFT先例。它不等同NEXT_STEP的共享 $V m_i$，也没有固定用户/query主干的证据；不能用其通信收益推断在线检索时延。[作者仓库](https://github.com/young1010/FedPEFT)未做源码冻结契约审计。完整元数据和评估边界由主交付S06/E24登记。

**REPREC仅留待核候选。** [2607.24845v1](https://arxiv.org/abs/2607.24845v1)一手提交日期为2026-07-24。[全文§2.2、Appendix B](https://arxiv.org/html/2607.24845v1)冻结序列推荐器与LLM，学习的是用户到soft-token的MLP injector，使用200个采样负例；不等同固定内积评分器下的商品输出编码。本轮不将其纳入核心或证明本方案新颖性。

因此NEXT_STEP只能定位为参数受限诊断：检查在冻结query的条件下，共享非线性商品映射是否具有超出等参数线性映射的作用。四臂比较不构成“优于所有商品侧PEFT”证据；未来若升级方法贡献，需要补齐FedPEFT式逐商品低秩残差等更近对照，并独立验证跨Decoder/跨数据迁移。
