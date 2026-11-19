# KuaiRec session-wise：TIGER、DIGER与CF的五组实验设计

版本：v1，2026-11-13。状态：**设计文档，尚未实现或启动本轮实验。**

原则：CF无Mixer严格沿用已有CF；TIGER、DIGER尽量保留各自方法和训练方式。用组内严格消融回答Mixer问题，用完整结果与训练过程比较方法竞争力。不为对齐CF而改变DIGER的逐batch联合更新。

本文中的“原设”指论文或核实的作者代码；“适配”指KuaiRec五商品任务必须明确的改动；“补全”指原文未给出或尚未核实、为本轮提出的具体实现选择。补全不冒充原作者配置。本文给出了可实施的候选设置，但不是GPU运行授权或数值验收记录。

## 1. 实验矩阵与研究问题

| 实验ID | 本次新增 | 生成方式 | 推荐阶段Tokenizer | Mixer |
|---|---|---|---|---|
| AR-T0 | 是 | TIGER式逐SID token自回归 | 完全冻结 | 无 |
| AR-T1 | 是 | 同AR-T0 | 完全冻结，包含Mixer | 层内，RQ预训练时学习 |
| AR-D0 | 是 | DIGER式逐SID token自回归 | 每个batch联合更新码本 | 无 |
| AR-D1 | 是 | 同AR-D0 | 每个batch联合更新码本与Mixer | 层内，预训练和联合训练时学习 |
| CF-M0 | 是 | 连续扩散生成z，再RQ落地 | 原5R暖身、30次10T＋1R | 无 |
| CF-M1 | 否，复用已完成结果 | 同CF-M0 | 同CF-M0 | 现有B1层内Mixer |

这里是五个新实验、六个比较对象。新ID避免与此前B0/B1/B2和A/D/E混淆。

- AR-T1−AR-T0：交互是否改善固定Tokenizer两阶段路线。
- AR-D1−AR-D0：交互在可微联合训练AR路线中是否有收益。
- CF-M1−CF-M0：当前CF路线的收益是否依赖Mixer。
- 不同路线的best和训练曲线：当前连续生成方法相对于传统AR参照的竞争力。

跨路线允许商品表示、网络、优化器和训练预算不同，不能把差值归因于单独的“AR/扩散”或“联合训练”。TIGER与DIGER也不是只差一个联合训练开关。

## 2. 来源和版本边界

### 2.1 TIGER

依据为论文《Recommender Systems with Generative Retrieval》v3的§3、§4.1。保留RQ语义ID、固定Tokenizer、T5式encoder-decoder、逐token预测及其公布的结构与训练参数。[论文](https://arxiv.org/html/2305.05065v3)

目前没有核实可直接使用的作者完整训练入口。早期项目Stage01的AR_GRM是已有适配，不当成TIGER官方实现；也不把第三方仓库称作官方。本文TIGER应报告为“依据论文配置的session-wise适配”。推荐器优化器等补全见第5节。

### 2.2 DIGER

采用作者`gradient-fix`，本轮只读核实提交：`ec0a9b51d02272be1935e1a6438cdc30bf9fadc1`。使用Beauty发布配置作为KuaiRec首轮参照，不宣称其在KuaiRec最优。[固定配置](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/config/beauty_gradient_fix.yaml)

保留True-E2E FrQUD、cached-hard前向/soft反向、频率门控探索、逐batch联合更新及原约束。历史main存在推荐梯度截断，不能用它替代修正版。[作者说明](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/README.md)

读取公开源码仅用于方案核查，没有执行下载到的脚本。逐文件来源、字节数与SHA见同目录`SOURCE_REVIEW_20261113.json`。

### 2.3 当前CF

以`configs/rqcd_mixer/v2_cf.json`和`experiments/rqcd_mixer/targets_formal_20261113_v1/`的冻结源码清单、运行配置、最终审计为准。冻结清单记录的Git提交为`72e74be3c07ca790f7983004f199d96175ae6472`；逐文件SHA优先于口头版本名。

CF-M0唯一科学变量是Mixer不存在，从RQ预训练起移除；不删除已经训练好的CF-M1中的Mixer后直接评估。复用CF-M1已保存的无Mixer初始RQ状态、推荐器初始状态、教师和随机流规则，重新预训练及训练无Mixer支路。

## 3. 共同数据和评测任务

| 项目 | 本轮方案 | 来源/性质 |
|---|---|---|
| 数据 | 原KuaiRec listwise固定划分 | 沿用现有协议 |
| train / validation / test | 709,443 / 71,657 / 53,194个session样本 | 沿用 |
| 目录 | 10,728件，原商品ID与目录行顺序不变 | 沿用 |
| 历史/目标 | 最多50件真实历史；5件不同目标，原顺序 | 沿用 |
| seed | 首轮均2024 | 项目共同控制；DIGER推荐训练发布值2020改为2024，显式记录 |
| 训练信息 | 仅训练划分更新推荐器；验证选点；测试不调参 | 沿用 |
| 主选点指标 | 全验证集商品级NDCG@5 | 任务适配；DIGER原配置为NDCG@10 |
| 其他指标 | Recall@5、HR@5、五件合法且不重复的完成率 | 沿用现有实现 |
| 候选集合 | 相同全目录；排除当前列表已选商品；不额外过滤所有历史商品 | 沿用CF的实际落地规则 |

沿用`item_metrics`：预测排名j命中五件真实目标集合任一商品记相关，DCG按预测位置折损，以五件相关商品的理想DCG归一化；Recall为命中数/5；HR为是否至少命中一件。不是把五个位置的“下一商品NDCG”平均，也不是SID准确率。训练保留目标时间顺序，主评测遵循现有列表相关性口径。

所有方法可以用全目录静态商品特征拟合Tokenizer，但不能把验证/测试交互拿来更新推荐器或协同教师。保持已有目录可见性协议，不新增未来交互特征。

### 3.1 商品特征：明确数据适配，不强制全部改成CF

- **CF-M0**：复用已完成CF-M1的SASRec教师商品表及同一全目录RMS缩放，不重训教师。教师`items.pt`的SHA256为`a1d072ba5fbe37258691a1f47f1aac6a342da9cefb28fb207bf73db8705d64e2`。具体路径绑定原收据，1914件无训练支持商品的标记和处理不变。
- **AR四组本轮候选输入**：复用项目已有、归一化的KuaiRec内容PCA256资产，SHA256为`3eaee85a60aba7e232e600585f7c5f428c318fb8475db0ebb24ac74e94b570c7`。不加协同对齐、SASRec蒸馏或新特征融合。
- 这是KuaiRec的显式数据适配：TIGER论文使用Sentence-T5内容向量，DIGER发布配置使用4096维内容向量；这里使用已有256维内容，调整RQ输入/输出端尺寸，保留各自内部层宽与潜维。不能写成原始特征提取器的逐项复现。换成新提取的原式内容向量将是另一版方案，不静默替换。
- AR两系列共用内容资产，但不共用不同结构的预训练RQ checkpoint；每系列有/无Mixer共用基础初始化和预训练数据顺序。CF不能借用AR的内容RQ，也不能借用旧内容B0充当CF-M0。

## 4. 四个AR组的session-wise适配

### 4.1 商品间追加Encoder，商品内自回归

设真实历史为H，目标列表为(i1,…,i5)。训练第j件时，Encoder读`H + i1…i(j−1)`；推理读`H + 已生成前缀`。当前商品和未来目标绝不能进入Encoder。

商品ID→当前版本SID→码位embedding→商品优先、码位次序展开→T5 Encoder。每生成完一件，就把其SID追加并重新编码。Decoder每件重新从开始标记出发，只自回归生成当前商品SID。列表前缀不另放Decoder，也不加H1的商品压缩MLP，不引入新位置标记或新前缀attention模块。

AR保留原本**3个RQ语义码＋1个消歧码**，不是CF的4层真实RQ。50件历史对应200个SID token；生成第五件时，54件上下文对应216个SID token，另计原模型特殊token。DIGER原`max_length=210`不足以涵盖此上界：适配为至少覆盖真实所需长度，本文取224作为输入上限；TIGER同取224。不要把其错误套到当前商品的输出长度。TIGER保留论文的用户hash token（2000槽位），使用稳定hash规则；DIGER不额外新增用户token。

短历史按真实长度追加前缀，PAD正确屏蔽；不在PAD后制造虚假的商品间距。列表前缀另加，不挤掉原50件历史。

### 4.2 batch和epoch的定义

为保留AR原配置的下一商品任务batch含义，将每个session逻辑展开成五个`(历史＋真实前缀, 当前目标)`条件任务，四组使用相同排列规则。只展开索引，不复制大文件。**AR的batch=256个条件任务，不是256个session再额外乘5。**

一轮session等价遍历=五个位置各见一次，共`709443×5=3,547,215`个条件任务；保留尾批，每轮`ceil(3547215/256)=13,857`次优化器更新。DIGER每次更新同时更新推荐器与Tokenizer；TIGER只更新推荐器。

CF保持每batch1024个session、每个session含五个位置：每R轮693次更新，35轮24,255次R更新，另300次T更新。不得为简化曲线，把AR一个条件任务和CF一个完整session记成相同“样本”。

AR的推荐损失保留原来每个有效码位、每个条件任务的平均规则；不因五位置展开再乘5，不引入额外列表级正则。DIGER的量化约束继续只按当前目标出现次数平均，不能因历史去重把目标重复权重丢掉，也不能把所有历史都加到约束均值里。

### 4.3 推理规则

四个AR组统一采用：**每件商品内部beam=20，选得分最高的完整合法商品后提交，再追加Encoder生成下一件**，共五次。DIGER的20来自Beauty配置；TIGER的20是本轮推理补全，不冒充论文公布的唯一beam宽度。

这是商品间逐件提交、商品内beam；不是对整条五商品列表维护20条beam，也不是拿第一次预测的top5候选当列表。当前商品分数使用其各SID token的累计log概率；固定长度下不额外加长度奖励。原模型EOS若保留，必须一致计入，两支路相同。

推理用合法目录前缀约束及列表去重，按固定商品ID规则处理完全同分；对全局token词表，过滤保留原模型概率的排序语义，不在候选过滤后另加可学习重排。这是五件有效商品输出协议的适配。训练交叉熵仍按原模型输出词表计算，不额外加CF的RQ路径损失。

TIGER固定SID映射；DIGER使用与checkpoint绑定的刷新映射。3层SID碰撞沿用消歧后缀，后缀256个槽位，按锁定实现确定性编号。若任一桶超过256且原消歧过程仍无法解决，记为接口不满足，不截断、取模或默默挑第一件；需要显式修订映射/词表协议。消歧码不是第4层可微RQ。

## 5. TIGER式两组的具体配置

### 5.1 RQ预训练与推荐器

| 项目 | AR-T0 / AR-T1共同设置 | 来源 |
|---|---|---|
| RQ输入 | 当前内容256维 | 数据适配，第3节 |
| RQ Encoder | 256→512→256→128→32，ReLU隐藏层 | 内部结构沿TIGER；首层输入适配 |
| RQ Decoder | 镜像重建到256维 | 同上 |
| RQ码本 | 3×256，每中心32维，L2量化 | TIGER |
| 初始化 | k-means；100次迭代上限 | 方法沿TIGER；迭代上限为本轮补全 |
| 预训练损失 | 特征重建平方L2＋RQ损失，commitment β=0.25 | TIGER公式；归约明确定义见下文，不套CF四层平均 |
| 预训练优化器 | Adagrad，lr=0.4，batch=1024商品 | TIGER |
| 预训练预算 | 20,000目录epoch；无效果驱动提前停止 | TIGER |
| 下游预训练点 | 固定最终20,000epoch；每50epoch记录损失/碰撞/利用率 | 最终点及诊断频率为补全；不按推荐测试选RQ |
| 推荐阶段Tokenizer | Encoder、Decoder、码本、Mixer全部冻结 | TIGER两阶段；Mixer为扩展 |
| 推荐器 | T5式Encoder 4层、Decoder 4层，d_model128，FFN1024 | TIGER论文 |
| 注意力 | 6头，d_kv64；总投影维384，不要求128能被6整除 | TIGER论文，不用通用MHA的错误整除假设 |
| FFN/dropout | ReLU / 0.1 | TIGER论文 |
| 输入 | SID token流＋2000槽位用户hash token | TIGER，hash算法锁定为SHA256原始用户ID字符串取模2000（补全） |
| SID表示 | 3个语义位＋消歧位的token身份独立；T5式共享词表实现 | TIGER式语义词表，不压缩成商品MLP向量 |
| 推荐目标 | 当前商品SID自回归交叉熵，PAD不计损失 | TIGER式；session展开适配 |
| 推荐训练 | batch256条件任务；最多200,000步 | TIGER公布的Beauty/Sports设置作为本轮上限 |
| 学习率 | 前10,000步0.01；之后0.01×sqrt(10000/step) | TIGER论文；不是从0线性暖身 |

200,000步在本轮展开口径下约14.43个session等价epoch，**不是200,000轮，也不会自然到R35**。完成14个完整epoch后再训练剩余步数；尾段结束再做一次验证，不为凑R35延长或重压缩调度。两支路均跑到相同200,000步，选各自验证best。

TIGER预训练损失归约固定为：每个商品先计算`||x−x_hat||² + Σ_l (||stopgrad(r_l)−c_l||² + 0.25||r_l−stopgrad(c_l)||²)`，最后对商品batch取平均；平方范数对维度求和，三层求和。前向逐层硬残差量化，采用标准RQ-VAE直通路径；码本及Mixer直接由码本拟合项学习，不声称重建梯度直接穿过argmin。这个明确定义与CF的逐维MSE、逐层平均归约不同，不能混用损失数值比较收敛高低。

### 5.2 论文未完整公开的细节：明确补全，不编造原设

论文给出了推荐学习率及T5X框架，但目前未核实其完整训练入口。本文候选采用T5X式Adafactor：参数尺度缩放开启，β1=None，二阶衰减指数0.8，update clipping=1，eps=(1e−30,1e−3)，无weight decay；外部控制上述学习率，不叠加另一套自动relative-step日程。依据为[T5X优化器配置](https://github.com/google-research/t5x/blob/main/t5x/examples/t5/t5_1_1/base.gin)及[Adafactor实现](https://github.com/google-research/t5x/blob/main/t5x/adafactor.py)，**不是已验证的TIGER原优化器证据**。

其他补全：label smoothing=0；不额外加z-loss；T5相对位置32桶/max_distance128、LayerNorm epsilon1e−6；输入/输出词表共享按所选T5式实现锁定。Adagrad未公布细节采用无weight decay、无lr_decay，initial accumulator=0、eps1e−10。这些需要在实现说明中保留“补全”标签；若后来找到作者完整入口，应先给出差异，再更新版本，不能静默混用。

AR-T1的Mixer从RQ预训练开始加入，使用同一Adagrad与学习率；推荐阶段冻结。它接收RQ量化目标，不接收当前列表推荐反馈。因此这一组是编码结构扩展，不宣称学到了当前session的列表条件。

## 6. DIGER式两组的具体配置

### 6.1 RQ预训练

采用固定提交的`run_rqvae_pretrain.sh`实际覆盖值，不把`main.py`未被采用的默认4层/2048维误写成执行配置。

| 项目 | AR-D0 / AR-D1共同设置 | 来源/改动 |
|---|---|---|
| 输入 | 当前内容256维 | 原Beauty输入4096维，KuaiRec资产适配 |
| RQ Encoder | 256→2048→1024→512→256 | 内部层宽/潜维沿作者脚本 |
| RQ Decoder | 镜像重建到256维 | 预训练沿原方法 |
| RQ码本 | 3×256，每中心256维 | 原设，不改成CF的32维 |
| 量化 | L2，β0.25，VQ权重1；dropout0、BN关闭 | 原设 |
| 初始化 | k-means=True，最多100次 | 原预训练launcher覆盖值；推荐配置中的False不是要求重置预训练码本 |
| Sinkhorn | 各层epsilon0.003，50次迭代 | 原预训练脚本 |
| 优化器 | AdamW，lr1e−3，wd1e−4 | 原预训练脚本 |
| batch/预算 | 1024商品，10,000目录epoch | Beauty预训练原设 |
| 调度 | 前50epoch暖身，随后线性衰减 | 原设 |
| 检查/选点 | 每50epoch，最低碰撞率best_collision；相同保留较早点 | 原预训练实现 |

AR-D1从预训练起使用Mixer；因此两支路分别训练并按相同规则选RQ，不能直接共享有Mixer的已训练RQ权重。不使用CF的best_objective选点替代DIGER的best_collision。

### 6.2 推荐器及逐batch联合训练

| 项目 | AR-D0 / AR-D1共同设置 | 来源/改动 |
|---|---|---|
| 网络 | T5 Encoder6层＋Decoder6层，width128，FFN512 | Beauty修正版 |
| attention/FFN | 4头、d_kv64、ReLU、dropout0.1 | 原设 |
| 码位embedding | 四张独立256×128表；每码位输出与其表绑定 | 作者model.py；第4张是消歧码 |
| 推荐优化器 | AdamW lr1e−3、wd0.05 | 原设 |
| Tokenizer优化器 | AdamW lr1e−5、wd0.05；Mixer加入同一学习率组 | 原设＋Mixer接入 |
| AdamW细节 | betas(0.9,0.999)、eps1e−8；参数分组依原实现 | PyTorch原调用默认，不新增免衰减规则 |
| batch | 256个条件任务；不进行跨batch梯度累积 | 原True-E2E设定；五位置展开见第4节 |
| 更新节奏 | 每个batch一次backward，两个optimizer各step一次 | 保留原核心方法 |
| 额外5R暖身 | 无；freeze_id_epochs=0 | 原设，不套CF |
| 调度/裁剪 | 两optimizer沿原cosine调度；warmup500更新；各模块norm clip1 | 原设 |
| 上限/早停 | 最多120个session等价epoch；每epoch验证，连续15次未改善停止 | Beauty上限/规则；epoch内容适配 |
| 选点 | 全验证NDCG@5 | 原NDCG@10改为本任务共同主指标 |
| 冻结 | 内容表和整个RQ Encoder | 原设 |
| 可训练 | 推荐器、码本；AR-D1另有Mixer | 原设＋Mixer |
| RQ Decoder | 保留预训练资产，True-E2E损失不经过它，不新增重建分支 | 原修正版实际路径 |
| 分配梯度 | cached hard SID前向、当前soft assignment反向 | 原设 |
| 探索 | FrQUD，temperature2.0，Gumbel尺度1，不做温度退火 | Beauty修正版 |
| 频率控制 | 热门阈值比例1.5、usage EMA momentum0.99 | 原设；不是EMA平滑码本 |
| 损失权重 | code1、latent reconstruction1、VQ1；QS/balance/gate等为0 | 原设 |
| SID刷新 | 初始建图，每epoch训练后整体刷新，再验证 | 作者训练循环，不等于每batch硬SID都刷新 |

120epoch最坏为1,662,840次联合更新，通常可能因早停更早结束；这是预算上限计算，不是完成时长预测或申请资源指令。它显著大于CF的300次T更新，必须公开报告。不为节省时间偷偷改成按session每轮10T；实际执行费用和可用时限需在运行阶段据吞吐评估。

两个支路使用相同上限/早停规则，不强迫best或停止轮次相同；早停后不假造35/120轮数据。cosine调度按完整120epoch预定义，不能因早停发生而回溯压缩。

### 6.3 梯度和损失的保留方式

对语义位使用`a_ST = one_hot(c_cached) + p_current − stopgrad(p_current)`；历史以及追加的列表前缀用a_ST乘各码位embedding；当前商品teacher forcing和语义目标分布沿修正版可微实现。消歧后缀使用硬标签，不对离散后缀编号求梯度。

每个条件任务：`L = L_semantic + L_suffix + L_latent + L_VQ`，系数均1。前三个语义位和一个后缀共同按四码位归约，不能给后缀另乘四倍。`L_latent = mean((q−z)^2)`，z来自冻结Encoder；VQ沿原逐层定义。两项约束按当前目标出现次数平均，history仅通过推荐路径提供反馈。Mixer不增加额外正则损失。

Mixer必须接入所有使用码本中心的位置：预训练、当前soft分配、量化q、VQ/latent loss、Sinkhorn/硬目录刷新及导出。不能只改变训练forward却让缓存SID仍用未修正基础码本。码本逐batch更新、硬目录逐epoch刷新、当前soft反向梯度，这三件事分别记录。

## 7. 层内Mixer的共同定义与组内控制

保留现有B1的结构逻辑；根据各模型原RQ层数/维度实例化，不强行改原RQ尺度。

| 参数 | TIGER扩展 | DIGER扩展 | 已有CF-M1 |
|---|---:|---:|---:|
| 层数L/中心数K/维度d | 3/256/32 | 3/256/256 | 4/256/32 |
| Mixer层数/heads | 1 / 4 | 1 / 4 | 1 / 4 |
| attention范围 | 同层256中心 | 同层256中心 | 同层256中心 |
| Q/K/V/输出投影 | 层间共享参数 | 同左 | 同左 |
| 输入标记 | 仅层身份；无中心编号 | 同左 | 同左 |
| 额外FFN/dropout | 不加 / 0 | 同左 | 同左 |

计算沿当前`CodebookMixer`：`x_l = LayerNorm(C_l) + e_l`，4头attention得到raw修正D_l。令`s_l = sqrt(mean(C_l^2))`，取第1个RQ层作参考：

`Δ_l = D_l × stopgrad(s_l / max(s_0, 1e−8))`

`C_eff,l = C_l + Δ_l`

LayerNorm eps1e−5；层身份初始化std0.02；Q/K/V Xavier、bias0；输出投影权重和bias都为0。初始有效码本与无Mixer一致。此比例不保证各层最终相对修正完全相同，需实测记录。

组内先创建共同RQ Encoder/Decoder/基础中心，再用独立随机流初始化Mixer；推荐器初始参数另存并逐字段一致。数据排列、验证规则相同。各自预训练后出现SID或潜空间差异是Mixer处理的下游结果，不应强行覆盖成同一SID。

所以组内消融估计的是“从预训练起加入Mixer的整体效果”，不是只在推荐阶段加入Mixer的纯效果，也没有额外参数量匹配对照。

## 8. CF-M0：只去掉Mixer的完整设置

直接以既有`v2_cf.json`为基线。原CLI的arm映射已支持B0=none；未来执行选择B0，其他科学配置不变。当前配置中的`target.comparison_arm=B1`是旧比较标识，未来新版本若更新元信息必须与科学差异分开，不为绕过来源检查改旧冻结配置。

| 部分 | 保留设置 |
|---|---|
| 输入/教师 | 同一冻结SASRec表；全目录RMS范数缩放；256维CF输入 |
| RQ | 256→512→256→128→32；4层×256；镜像Decoder；β0.25 |
| RQ预训练 | 10,000epoch；batch1024；AdamW1e−3、wd1e−4；50epoch暖身＋线性衰减 |
| RQ选点 | 每50epoch全目录固定目标评估，最低best_objective；不是最低碰撞 |
| 推荐器 | width256、4头、d_kv64、FFN1024、Encoder2层、Decoder4层、dropout0.1 |
| 历史/前缀 | 固定真实历史Encoder；生成商品进入原去噪Decoder前缀；不随AR改变 |
| 扩散 | 1000噪声等级、cosine offset0.008、beta_max0.999；推理20步DDIM、eta0 |
| 推荐训练 | 5R暖身，再30次10T＋1R，共R35/T300 |
| R优化 | lr3e−4、wd0，warmup500＋cosine；每有效batch1024session |
| T优化 | 基础码本lr1e−5、wd1e−4；不再存在Mixer参数 |
| 共用优化细节 | AdamW，betas(0.9,0.999)、eps1e−8、clip norm1 |
| 软量化 | temperature0.1、softmax_temperature1；Gumbel按累计T由1余弦降到0.2 |
| R损失 | denoise MSE＋0.1×(RQ path CE＋bucket CE) |
| T损失 | R损失＋1×latent＋1×VQ |
| 桶温度 | 0.1，保留原固定距离尺度和停止梯度规则 |
| 冻结范围 | RQ Encoder固定；RQ Decoder不参与推荐阶段；R/T各自冻结另一侧参数 |
| 目录 | T块末刷新；当前R和评估绑定同版本SID/中心/树/桶 |
| 落地 | 四层合法路径贪心＋选中桶内z距离最近邻；非全目录NN |
| 评估 | R5…R35共31次全验证；best及R35测试；不提前停止 |
| 随机性 | seed2024，原分离随机流；评估seed20241111、noise_v1，复用噪声SHA |
| 执行 | 单卡；BF16；R/T micro256，有效1024；eval256；fused AdamW；确定性设置不变 |

原CF prepared资产已包含B0/B1/B2初始状态，应优先复用其中B0与同一推荐器初始状态。新支路使用自己的输出路径和来源收据，不改旧PREPARED文件，不修改旧报告里的B1结果。

已有CF-M1参考结果：验证选出的best为R9/T40，验证NDCG@5=0.0065343074，测试=0.0088549493；R35验证=0.0055175153、测试=0.0066755924。最终表应回读原最终审计而非手抄对话数值。以上不是本轮新测。

## 9. 选点、比较和必要统计

### 9.1 验证和测试

- TIGER：每完整session等价epoch验证，加200,000步最终部分epoch；固定步数训练，不因验证停止。第0步仅可作数值诊断，不纳入best。
- DIGER：每session等价epoch结束刷新SID后全验证；NDCG@5连续15次不改善早停，最多120轮。
- CF：沿用R5到R35每轮验证，不能给无Mixer额外增加选点机会。
- 同指标相同分数保留较早点；测试仅对验证best及实际训练末端。best和末端是相同checkpoint时复用同一次完整测试。
- 原始next-item NDCG@10不能代替本轮五商品列表NDCG@5；不因模型原配置不同采用不同测试指标或候选库。

### 9.2 主结果与过程结果分开

主表六个对象：best位置、best验证/测试NDCG@5/Recall@5/HR@5、实际末端结果、参数量、推荐训练更新数、Tokenizer更新数、预训练/推荐训练/验证/测试分别耗时。跨路线称完整方法比较；不能称同计算预算的纯架构对照。

过程图：横轴分别画session等价遍历量和累计GPU时间；标注同一曲线的实际optimizer更新数。共同进度点5、10、20、35轮只在实际到达且有验证节点时列出，TIGER200k步只到约14.43轮，不补造20/35轮。若要列共同预算下的测试，只在运行前固定一个预算点，再测试对应验证选点；本轮不默认增加这种补测。

不同路线总验证次数不同，best有不同选点机会，和预算一并披露。前35轮比较也仍含batch、日程和输入差异，只能称相同数据遍历量下的观察。

### 9.3 必要诊断

- 每个验证点：训练损失各分项、推荐验证曲线、各层中心利用率/perplexity、原始语义SID碰撞、语义SID变化、消歧后缀变化单独记录。
- Mixer组：修正RMS/基础中心RMS、有效中心变化量；Tokenizer冻结后这些值应保持固定。
- AR：SID合法率、完整五件率、每位置命中贡献；CF仍保留其原落地诊断，不临时改全目录NN作主结果。
- 稳定性：非有限值、实际梯度范数；DIGER检查推荐损失单独到码本/Mixer的梯度，不能用VQ梯度冒充推荐梯度。
- 单seed只报告描述性差异，不宣称跨seed稳定性或显著性。没有完成这五组之前不自动扩大实验矩阵。

## 10. 实现、效率和版本管理边界

本轮只交付文档。未来实现分任务提交：AR任务适配、层内Mixer接入、CF无Mixer配置/来源、共同评测与报告；原模型源码固定提交，适配文件隔离，旧实验快照不覆盖。

效率优化只改变执行：预先打包条件索引、只读商品资产、批量token查表、缓存固定SID、推理KV缓存、向量化beam、减少逐tokenCPU同步。训练不同前缀可以批处理，但不改变batch统计、loss权重、dropout或FrQUD使用频率的累计语义。

AR首轮数值参考用FP32；BF16若作为执行优化引入，需先核验loss/梯度/排序与恢复，再在同系列两支路共用，不因为H100而改变有效batch。DIGER每batch去重/usage更新有批依赖，不能把普通梯度累积直接当作原batch256等价替代。CF继续使用已验收BF16执行配置。

未来必要检查只覆盖新增部分：1）五位置前缀无泄漏和AR因果mask；2）无Mixer原路径不变，零初始化Mixer前向等效；3）DIGER推荐梯度到码本/Mixer及冻结Encoder；4）SID/后缀/索引与checkpoint一致；5）loss归约、更新数和恢复；6）一条五件列表按同一指标计算。复用已通过的CF检查，不重跑历史实验做证明。

大checkpoint和逐样本预测留远端，文档、配置、代码版本、SHA清单和指标摘要留本地。没有吞吐实测前不承诺耗时，也不在本设计提交时申请卡或启动监控。

## 11. 本版明确撤回/保留的早期建议

| 早期建议 | 本版决定 |
|---|---|
| 所有路线都改用CF输入 | 撤回；CF保持CF，AR采用内容输入，数据适配单列 |
| 四个AR统一2层Encoder＋4层Decoder | 撤回；各自尊重TIGER论文与DIGER配置 |
| DIGER改5R＋10T/1R | 撤回；保留逐batch联合更新 |
| 所有路线统一R35 | 撤回；各自原式预算/停止规则，并报告过程 |
| AR也用4层真实RQ＋第五个消歧token | 撤回；原AR为3层RQ＋第4个消歧token |
| AR固定历史、列表前缀进入Decoder | 撤回；采用前缀追加Encoder、当前商品内AR |
| CF随AR改前缀方式 | 不做；CF无Mixer唯一科学改动是去Mixer |
| TIGER＋Mixer也接受推荐反馈 | 不做；它只在RQ预训练学，推荐阶段冻结 |
| 四个AR共用一份RQ checkpoint | 不做；模型结构不同；只在同系列控制初始条件 |

## 12. 文档审查与证据边界

已完成：本地CF配置、执行配置、源码清单、最终审计、现有Mixer/指标实现核对；DIGER固定版本配置、预训练脚本、模型及联合训练循环只读核对；TIGER论文结构、预算和学习率核对。未执行：新代码实现、真实数据预处理、训练、推理、GPU短测或资源申请。

仍需明确保留的边界：TIGER完整作者训练入口未核实，Adafactor等属于有来源的候选补全；AR使用现有内容PCA256、稳定用户hash、逐件beam提交、共同列表评测均为本文适配，不能写成原论文完整复现。正式实现配置和本文有差异时，应先更新设计版本并说明差异，不能以“原模型默认”为由覆盖用户已确定的session-wise/Mixer安排。

### 参考与本地证据

- [TIGER论文v3](https://arxiv.org/html/2305.05065v3)
- [DIGER修正版固定提交](https://github.com/junchen-fu/DIGER/tree/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1)
- [DIGER配置](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/config/beauty_gradient_fix.yaml)、[训练循环](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/trainer.py)、[模型/梯度接口](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/model.py)
- [DIGER预训练覆盖参数](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/scripts/run_rqvae_pretrain.sh)、[预训练默认补充](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/scripts/rqvae/main.py)、[预训练选点](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/scripts/rqvae/trainer.py)
- `configs/rqcd_mixer/v2_cf.json`；`docs/research/rqcd_mixer/RQCD_TARGETS_DESIGN_20261113.md`
- `experiments/rqcd_mixer/targets_formal_20261113_v1/{SOURCE_MANIFEST.json,runtime.locked.json,FINAL_AUDIT.json}`
- `scripts/rqcd_mixer/{tokenizer.py,catalog.py,evaluation.py,run.py,training.py}`
- `experiments/09_kuairec_listwise/decoder_prefix_formal_20261112_v1/REPORT.md`：前缀结构的项目内证据，不直接外推TIGER的结构优劣。
