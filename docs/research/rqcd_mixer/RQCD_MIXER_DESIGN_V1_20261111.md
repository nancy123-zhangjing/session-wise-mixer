# RQ 码本交互与连续扩散列表推荐：模型设计与实验参数（V1）

**版本：2026-11-11｜状态：设计已确认，尚未按本方案完成实现与实验验证。**本文供模型评审和后续实现使用，集中说明信息流、训练、参数及对照。来源分为“文献／官方实现”“原实验沿用”“本次设计选择”；借鉴某项设置不表示复现整篇论文，也不表示该设置已经最优。

## 研究目标与整体结构

研究问题是：在根据历史逐件生成五商品列表的任务中，让 RQ 码本中心进行层内或层间交互，能否学到更适合推荐的编码，并改善最终推荐质量。三组均采用 RQ Tokenizer、连续扩散推荐器和交替联合训练，主要区别为是否加入 Mixer、Mixer 的注意力可见范围。

**商品侧：**固定内容特征 → RQ Encoder → 连续向量 z → 使用有效码本逐层残差量化 → 四位 SID。

**推荐侧：**历史商品 ID → SID → 独立 SID embedding → T5 Encoder → 带历史和已生成商品前缀条件的扩散 Decoder → 连续向量 → 目录约束量化 → 实际商品；重复五次。

| 对象／模块 | 输入 → 输出 | 职责 |
| --- | --- | --- |
| 固定内容特征 | 已有 BGE-M3 特征 → PCA256＋归一化 → uᵢ∈R²⁵⁶ | 复用现有资产与拟合边界；新 RQ 分支直接读取这份特征，不经过旧 PSE 的 OPQ 分块。 |
| RQ Encoder | 256 → 512 → 256 → 128 → 32 | 得到量化前商品表示 zᵢ；联合训练期间冻结。 |
| 基础码本＋Mixer | [4,256,32] → 有效码本 [4,256,32] | 改变中心之间的信息交互；不直接输入某个用户或 session。 |
| 残差量化器 | zᵢ [32] → SID [4] 与 qᵢ [32] | 四层依次选中心，qᵢ 为四个中心之和。 |
| RQ Decoder | 32 → 128 → 256 → 512 → 256 | 只在 RQ 预训练时重建内容特征；不负责生成推荐商品。 |
| SID embedding＋历史 Encoder | 历史 ID [B,50] → SID [B,50,4] → [B,200,256] | 每件商品保留四个 token，输出包含历史上下文的表示。 |
| 扩散 Decoder | 前缀 SID token＋一个带噪查询 → 当前商品干净向量预测 [B,32] | 同一网络用于不同噪声等级和五个生成位置。 |
| 目录约束落地 | 生成向量 [B,32] → 合法 SID → 商品 ID [B] | 最终输出 [B,5] 的实际商品列表。 |

B 为 session 批量。沿用现有 KuaiRec 划分：目录 10,728 件商品，训练／验证／测试分别为 709,443／71,657／53,194 条样本，历史最多 50 件，目标为按时间组织的五件互异商品。目录内容可用于 Tokenizer 预训练，验证与测试行为标签不参与训练或统计量拟合。

核心区别：z 是 RQ Encoder 输出，q 是选中中心之和，SID 是四个整数编号；推荐器的四张 256 维 embedding 表与 32 维 RQ 中心是不同参数。结构参考 [TIGER 的 RQ Semantic ID](https://arxiv.org/html/2305.05065v3)、[ETEGRec 的独立 SID embedding](https://github.com/BishopLiu/ETEGRec/blob/main/model.py)和[DreamRec 的条件连续去噪](https://github.com/YangZhengyi98/DreamRec/blob/main/DreamRec.py)，组合及五商品生成流程属于本次设计。

## Tokenizer 与两种 Mixer

### 残差量化与商品编码

令 Cˡ 为第 l 层有效码本。真实商品按确定性最近邻生成目录 SID：

$$
r_{i,1}=z_i,\qquad s_{i,l}=\arg\min_k\|r_{i,l}-C^l_k\|_2^2.
$$

$$
v_{i,l}=C^l_{s_{i,l}},\qquad r_{i,l+1}=r_{i,l}-v_{i,l},\qquad q_i=\sum_{l=1}^{4}v_{i,l}.
$$

四层逐步表达剩余误差。四位 SID 全部来自真实 RQ 层，没有额外消歧码位；碰撞通过 SID→商品多值映射处理。使用四层是本次规格，不能称为 TIGER 原论文的“四层 RQ”：TIGER 的第四位用于碰撞消歧。

### Mixer 的结构与掩码

| 参数 | 共同设置 | 选择理由 |
| --- | --- | --- |
| 注意力结构 | 1 层，宽度 32，4 头，每头 8 维；不附加 FFN | 贴合中心维度，保留原 H1“归一化＋注意力残差修正”的轻量组织；两种交互容量一致。 |
| 参数共享 | 所有 RQ 层共用 Q/K/V 和输出投影 | 层内与层间组仅切换掩码，不同时改变参数量。 |
| 输入标记 | 每层一个可学习身份向量 [4,32]；不加中心编号标记 | 区分残差层级；避免人为绑定不同层的同号中心。 |
| 输入归一化 | Xˡₖ＝LayerNorm(Bˡₖ)＋eˡ；LayerNorm eps=1e-5，逐中心处理 | 先对中心内容归一化，再加入共享尺度的层身份，减轻跨层数值尺度差异。 |
| 初始化／dropout | Q/K/V 正常初始化；输出投影及其 bias 为零；dropout=0 | 初始有效码本等于基础码本，避免一开始引入随机编码改动。 |

把四层中心展开成 1,024 个 token，得到原始修正 U＝Wₒ·Attention(X)。层内组只允许同层中心互相读取；因果组中，第 l 层允许读取第 1…l 层全部中心。相同层内部不按中心编号施加因果顺序。两组采用同一套结构、层标记和幅度校准。Mixer 是全局码本变换，列表信息通过推荐损失的梯度间接进入码本。

### 以第一层为参照校准输出尺度

用当前基础码本 B 的均方根衡量每层尺度，第一层作为参照；比例计算停止梯度：

$$
s_l=\sqrt{\frac{1}{256\cdot32}\sum_{k,d}(B^l_{k,d})^2},\quad \Delta C^l=\operatorname{sg}\!\left(\frac{s_l}{s_1}\right)U^l,\quad C^l=B^l+\Delta C^l.
$$

sg 表示前向保留数值、反向停止梯度。该尺度规则在预训练和联合训练中一致使用，不额外叠加旧实验的 0.1 门控。分母实现中取 max(s₁,1e-8)；若尺度接近零或非有限，应记录异常，不能靠截断掩盖塌缩。

这是本次研究设计，不是已有论文验证的结论。按比例缩放只能校正尺度：实际相对修正 RMS(ΔCˡ)/sˡ＝RMS(Uˡ)/s¹；只有原始输出幅度接近时，各层相对修正才接近。必须记录实际比例，它不是硬性幅度上限。

## 推荐器如何生成五件商品

### 历史与推荐前缀

历史商品 ID 先查当前 SID 表，再按四个码位分别查询 E₁…E₄。每张 E 的形状为 [256,256]，历史与推荐前缀共用这四张表。按“商品时间顺序→商品内码位顺序”展开为 200 个 token；历史不足 50 件时在前面补 PAD，PAD 的四个 token 均屏蔽，不把真实编号 0 当成 PAD。

| 模块 | 具体设置 | 来源／理由 |
| --- | --- | --- |
| 历史 Encoder | T5 Encoder，2 层；d_model=256，4 头，d_kv=64，d_ff=1024 | 输入组织参考 ETEGRec；深度 2 为本次容量选择，不照搬其 6 层。 |
| 扩散 Decoder | T5 风格 Decoder，4 层；宽度、头数、FFN 同上 | TIGER 使用 4 层 Decoder，作为深度参考；连续去噪任务及宽度为本次适配。 |
| 每层结构 | Encoder：双向自注意力→FFN；Decoder：因果自注意力→历史交叉注意力→FFN | 采用 T5 的子层前归一化、残差和最终归一化。 |
| FFN／正则 | ReLU；dropout=0.1；T5 RMS 风格归一化 eps=1e-6 | 沿用 T5／ETEGRec 的结构口径；宽度 1024 为 256 的四倍扩展。 |
| 位置关系 | 自注意力相对位置偏置：32 桶，max_distance=128；Encoder 双向、Decoder 因果 | T5 默认机制；交叉注意力不额外添加相对位置偏置。不叠加绝对商品位置或额外码位标记。 |
| 连续输入／输出 | 输入线性层 32→256；当前查询输出经线性层 256→32 | 线性层负责空间接口，非线性建模由 Decoder 承担。 |
| 噪声时间条件 | t 的 32 维正弦编码→MLP 32→128→32（GELU），加到带噪向量后投影至 256 维 | 保留扩散时间条件；具体宽度和相加位置是本次实现选择。 |

参数参考：[ETEGRec 官方配置](https://github.com/BishopLiu/ETEGRec/blob/main/config/scientific.yaml)、[T5 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)、[TIGER §4](https://arxiv.org/html/2305.05065v3)。推荐器从随机初始化开始，使用 T5 结构，不加载语言预训练 T5 权重。

历史编码输出 M 的形状为 [B,200,256]，在同一份列表生成中复用。生成第 j 件时，Decoder 输入为前 j−1 件的 4(j−1) 个 SID token 加一个带噪查询；最大长度 17。当前查询可以读取全部已生成前缀，并通过交叉注意力读取历史 M。四层 Decoder 各有参数，但不同噪声步和列表位置复用同一套网络。

训练采用真实前缀（teacher forcing）：预测第 j 件时只给真实的前 j−1 件 SID，不给当前或未来目标 SID。推理使用前面实际落地的商品 SID。第一版不增加显式列表位置 embedding，位置由前缀长度及相对位置机制表达。

### 固定潜空间标准化与扩散

选择 RQ 预训练模型后冻结 Encoder。按训练集目标出现次数 nᵢ 计算一次均值 μ 和全局标量 s；验证／测试目标和 PAD 不参与：

$$
\mu=\frac{\sum_i n_i z_i}{\sum_i n_i},\qquad s=\sqrt{\frac{\sum_i n_i\|z_i-\mu\|_2^2}{32\sum_i n_i}}.
$$

$$
x_i=(z_i-\mu)/s,\qquad \hat z=s\hat x+\mu.
$$

单一标量保留潜空间相对欧氏几何，不额外放大低方差维度；也不把每个向量压到单位球面。μ、s 在联合训练全程固定。历史与前缀使用 SID embedding，标准化仅服务连续生成目标。该处理沿用本地 scripts/rqcd/tokenizer.py 的 latent_scaling 思路。

训练时，对每条列表的五个位置分别均匀抽 t∈{1,…,1000} 和 ε∼N(0,I₃₂)，一次加噪、一次去噪预测，不展开多步采样反向传播：

$$
x_{j,t}=\sqrt{\bar\alpha_t}\,x_{i_j}+\sqrt{1-\bar\alpha_t}\,\epsilon,\quad \hat x_{j,0}=f_\theta(x_{j,t},t,M,\operatorname{SID}(i_{<j})).
$$

| 参数 | 设置 | 理由 |
| --- | --- | --- |
| 训练目标 | 预测干净 x₀；MSE 对 batch、五位置、32 维取平均 | 参考 DreamRec 的干净目标恢复；不引入额外 SNR 权重。 |
| 噪声日程 | 1000 等级；cosine，offset=0.008，β 上限 0.999 | 沿用已有扩散基础日程，固定为共同设置；未声称最优。 |
| 推理 | 20 步 DDIM，η=0；每件独立初始高斯噪声 | DDIM 支持稀疏采样；20 步参考 PreferDiff 的推荐实验。 |
| 采样边界 | 从 1000 降至 1，再到干净等级 0；不把潜向量裁到图像的 [-1,1] | 潜空间没有图像像素取值约束。 |

具体日程：f(t)=cos²(((t/1000+0.008)/1.008)·π/2)，βₜ=min(1−f(t)/f(t−1),0.999)，ᾱₜ=∏ᵥ₌₁ᵗ(1−βᵥ)，ᾱ₀=1。训练和采样使用同一份截断后累计系数。20 个采样等级按 tₖ=1+floor(999k/19+0.5)，k=0…19，倒序调用。

每次预测 x̂₀ 后，利用当前 xₜ 反推出噪声，再推进到下一等级 u：

$$
\hat\epsilon=\frac{x_t-\sqrt{\bar\alpha_t}\hat x_0}{\sqrt{1-\bar\alpha_t}},\qquad x_u=\sqrt{\bar\alpha_u}\hat x_0+\sqrt{1-\bar\alpha_u}\hat\epsilon.
$$

到 u=0 后反标准化，仅此时进行量化和商品落地，再追加该商品 SID 作为下一件的前缀。五件共 100 次 Decoder 调用。依据：[DreamRec 去噪目标](https://github.com/YangZhengyi98/DreamRec/blob/main/DreamRec.py)、[DDIM](https://arxiv.org/html/2010.02502v4)、[PreferDiff](https://arxiv.org/html/2410.13117v2)；本方案未引入它们的全部损失或 CFG 机制。

## 训练阶段、损失与梯度

### RQ-VAE 预训练

以 ETEGRec 官方 RQ-VAE 为实现底座，保留硬最近邻＋标准 STE 预训练路径；三组分别训练，B1／B2 的 Mixer 从此阶段参与。编码网络和解码网络采用前述对称 MLP，中间层 ReLU，末层线性，无 BatchNorm、无 dropout；RQ Decoder 输出不额外 L2 归一化。

目标为原始 256 维特征重建 MSE 加逐层 VQ 损失。VQ 每层包含“中心接近停止梯度的残差”与 0.25 倍“残差接近停止梯度的中心”；维度、样本和层数均按平均归约。预训练 STE 使重建梯度回到 Encoder／Decoder；基础码本与 Mixer 主要由 VQ 码本项更新。此阶段不采用下一节的推荐软分配桥接。

| 设置 | 唯一值／规则 | 来源／选择思路 |
| --- | --- | --- |
| 训练长度／batch | 10,000 个商品 epoch；1024 件／batch；保留尾批 | ETEGRec 公开预训练脚本；本目录每 epoch 11 次更新，合计 110,000 次。 |
| 初始化 | 首批商品的逐层残差做 K-means，100 次迭代；Mixer 零输出初始化 | K-means 参考官方实现；零输出是本次受控起点设计。 |
| 优化器 | AdamW；lr=1e-3，weight_decay=1e-4；betas=(0.9,0.999)，eps=1e-8 | ETEGRec 预训练配置与优化器默认值。 |
| 日程／裁剪 | 50 个商品 epoch warmup，之后线性衰减；梯度范数裁剪 1 | ETEGRec main.py 与 trainer.py。 |
| 损失／选码 | 重建 MSE＋1×VQ；承诺系数 0.25；四层均关闭 Sinkhorn | 跟随其最近邻预训练脚本；四层、32 维为本模型适配。 |
| 检查与选点 | 每 50 epoch 检查全目录；选 best_collision；相同碰撞率保留较早节点；另存 best_loss | 沿用官方双 best 保存机制，本次明确使用碰撞最低者接下游。 |

碰撞率定义为 1−不同 SID 数／商品数，不是“处于碰撞桶的商品占比”。预训练遍历既定目录内容，不另切 90/10 商品集合；低碰撞不代表生成空间已经足够好，仍记录特征重建、量化误差、逐层利用率和 z 的范数。来源：[预训练脚本](https://github.com/BishopLiu/ETEGRec/blob/main/RQVAE/run_pretrain.sh)、[参数](https://github.com/BishopLiu/ETEGRec/blob/main/RQVAE/main.py)、[选点逻辑](https://github.com/BishopLiu/ETEGRec/blob/main/RQVAE/trainer.py)、[重建损失](https://github.com/BishopLiu/ETEGRec/blob/main/RQVAE/models/rqvae.py)。

### 冻结与更新分工

| 阶段 | 更新 | 冻结／不用 |
| --- | --- | --- |
| RQ 预训练 | RQ Encoder、基础码本、Mixer、RQ Decoder | 推荐器尚未训练 |
| 推荐器预热 5R | SID embedding、历史 Encoder、扩散 Decoder、投影与时间 MLP | 全部 Tokenizer；RQ Decoder 不参与 |
| T：10 个 optimizer step | 基础码本、Mixer | RQ Encoder／Decoder；推荐器参数冻结，但保留输入梯度 |
| R：完整训练集 1 epoch | 推荐器全部参数 | 全部 Tokenizer；SID 与目录索引固定 |

预训练完成后固定 z 表和标准化统计。每段 T 结束重新生成全目录确定性 SID、合法路径和碰撞桶，再进入 R。T 数据迭代器跨周期继续，不反复从前十个 batch 开始；R 每轮完整打乱遍历一次。T 时推荐器设为 eval 以关闭 dropout，但不能用 no_grad 包住其计算；R 时 Tokenizer 设为 eval 并缓存当前有效码本。

冻结 RQ Encoder 和可微码本连接参考 [DIGER gradient-fix 配置](https://github.com/junchen-fu/DIGER/blob/gradient-fix/config/beauty_gradient_fix.yaml)。本方案的少量 T／完整 R 交替是本次选择，不是 DIGER 的逐 batch 同步更新，也不是 ETEGRec 的完整 epoch 对完整 epoch。

### 离散 SID 怎样向码本传梯度

联合训练使用 DIGER 修正版启发的软分配桥接。距离 Dᵢˡₖ＝‖rᵢˡ−Cˡₖ‖²。为使 0.1 的温度适应不同 RQ 层，在选定预训练模型后，用训练目标频次计算一次各层典型距离 aˡ，并固定：

$$
a_l=\max\left(\frac{\sum_i n_i\left(\frac{1}{256}\sum_k\|r^{\mathrm{pre}}_{i,l}-C^{l,\mathrm{pre}}_k\|_2^2\right)}{\sum_i n_i},10^{-8}\right).
$$

$$
p_{i,l}=\operatorname{softmax}\left(-\frac{D_{i,l}}{0.1a_l}+\sigma(u)g_{i,l}\right).
$$

g 是标准 Gumbel 噪声。额外 softmax 温度固定为 1；第 u 次 T 更新（u=1…300）使用 σ(u)=0.2+0.4[1+cos(π(u−1)/299)]，从 1.0 降至 0.2。温度和噪声数值参考原 H1；固定距离归一化是本次适配，不是 H1 原公式，也不宣称四层熵因此相同。

区分两条路径：

$$
a^{Q}_{i,l}=\operatorname{onehot}(\arg\max p_{i,l})-\operatorname{sg}(p_{i,l})+p_{i,l}.
$$

$$
v_{i,l}=a^{Q}_{i,l}C^l,\qquad r_{i,l+1}=r_{i,l}-v_{i,l}.
$$

$$
a^{E}_{i,l}=\operatorname{onehot}(\bar s_{i,l})-\operatorname{sg}(p_{i,l})+p_{i,l},\qquad e_{i,l}=a^{E}_{i,l}E_l.
$$

第一式用于当前 RQ 内部和量化约束，层间残差不额外 detach；第二式用于历史／真实前缀输入，s̄ 是本段 T 开始前缓存的硬 SID。它们都硬前向、软反向，但前向硬编号来源不同。于是推荐损失可经冻结推荐器→SID embedding 输入→软分配→有效码本→基础码本／Mixer 回传。冻结参数不等于禁止输入梯度。

缓存 SID 在 T 段内固定，同一样本和噪声下的硬前向推荐损失可能不随 T 更新立即变化；非零软反向梯度不等于已经提高真实推荐质量，要在刷新后验证。实现参考 [DIGER 修正版量化器](https://github.com/junchen-fu/DIGER/blob/gradient-fix/vq.py)，缓存、温度缩放和小块交替为本次组合。

### 四项损失及权重

下面 N=5B，按目标商品的出现次数平均，重复出现的商品保留其权重。MSE 表示对样本与向量维度取平均；逐层 VQ 再对四层取平均。

| 损失 | 计算与作用 | 阶段／权重 |
| --- | --- | --- |
| 去噪 Ldenoise | MSE(x̂₀,xᵢ)，在标准化空间预测量化前目标；五位置等权 | R、T 均使用；1 |
| 软量化对齐 Lalign | 真实 z 与预测 ẑ 在同一真实 SID 前缀下的选码分布对齐；单向 KL，真实分布停止梯度 | R、T 均使用；3e-4 |
| 潜空间重建 Llatent | MSE(q,z)，约束四层中心之和接近固定连续目标 | 仅 T；1 |
| 逐层 LVQ | 每层 MSE(vˡ,sg(rˡ))＋0.25·MSE(sg(vˡ),rˡ)，再层间平均 | 仅 T；外部权重 1，内部 β=0.25 |

$$
\mathcal L_R=\mathcal L_{\mathrm{denoise}}+3\times10^{-4}\mathcal L_{\mathrm{align}},\qquad \mathcal L_T=\mathcal L_R+\mathcal L_{\mathrm{latent}}+\mathcal L_{\mathrm{VQ}}.
$$

推荐器预热同样使用 L_R，对齐项从预热开始启用；第一版不另设损失开关日程。Llatent 与 LVQ 的目标只取本批五位置商品，历史商品通过推荐反馈路径参与 T 更新。

对齐精确定义：第 l 层两侧都减去同一真实商品缓存 SID 的前 l−1 层有效中心之和 bᵢˡ；分别对 −‖(zᵢ−bᵢˡ)−Cˡₖ‖²/(0.1aˡ) 和 −‖(ẑᵢ−bᵢˡ)−Cˡₖ‖²/(0.1aˡ) 做 softmax，得到 P、Q。对齐不加 Gumbel，在全部 256 个中心上计算：

$$
\mathcal L_{\mathrm{align}}=\frac{1}{4N}\sum_{i,l}\sum_k\operatorname{sg}(P_{i,l,k})\log\frac{\operatorname{sg}(P_{i,l,k})}{Q_{i,l,k}}.
$$

KL 对 256 个中心求和，对目标和层取平均；采用 log_softmax 稳定计算。R 阶段可通过冻结的码本计算把梯度传给预测向量；T 阶段 P 的目标支路停止梯度，Q 支路仍可更新预测路径及有效码本。目录合法候选约束用于最终落地，不在这里增加依赖可变目录的 KL mask。

Llatent 关心最终合成误差，LVQ 还约束每层残差分工；二者有重叠，不能当作完全独立的信息。最后一层残差误差与最终误差数值相关，但 stop-gradient 使反向作用不同。权重 1／1／0.25 参考 DIGER 修正版；软对齐借鉴 [ETEGRec SIA](https://arxiv.org/html/2409.05546v3#S3.SS3.SSS1)，3e-4 参考其报告的参数量级。这里改成“预测商品向量对真实商品向量”、单向 KL 与目标停止梯度，属于适配，不是其对称 KL 原式，也未证明更稳定。

## 联合训练预算与运行参数

预训练完成 → 固定 Tokenizer 预热推荐器 5R → 重复 30 次“10T→刷新 SID→1R”，最终 R35/T300。R 表示完整遍历推荐训练集，T 表示优化器实际更新次数；它们都不是扩散噪声等级 t。全程按该预算定义调度，不因中途查看 R20 就提前将学习率衰减到零。

| 参数 | 设置 | 来源／合理性 |
| --- | --- | --- |
| 预热／总预算 | 5R 预热＋30 个交替周期；R35/T300 | 5R、35R 延续原实验观察窗口；10T＋1R 是双方已确定的小块交替，既控制总量又及时适应。 |
| R 峰值学习率 | 3e-4 | 介于原 H1 的 1e-3 与先前连续路线的 1e-4 之间，作为本次工程初值。 |
| 基础码本／Mixer 学习率 | 均为 1e-5，T 阶段恒定 | 码本沿用原 H1／DIGER 量级；Mixer 已预训练，不再默认给十倍学习率。 |
| 优化器 | 两个独立 AdamW；betas=(0.9,0.999)，eps=1e-8 | R 与 T 的状态及步数独立，切换时不重建。 |
| R 学习率日程 | 前 500 次 R 更新线性 warmup；之后余弦降至 0 | 原实验设置；时钟仅随 R 更新，预热 5R 与学习率 warmup 不同。 |
| Weight decay／梯度裁剪 | R：0；T：1e-4；可训练参数梯度范数上限 1 | 沿用原 H1 的优化器基础配置。 |
| T／R 有效 batch | 均为 1024 条 session，包含各自的五个目标 | 对齐样本预算；不能把五个目标算作五次优化器更新。 |
| 默认微批／精度 | 单卡 microbatch=32，累积 32 次；推荐器 BF16，距离、softmax、损失归约 FP32 | 实现默认值；显存调整仅改变微批与累积，有效 batch 不变。 |
| 尾批 | 保留全部样本，按有效组实际条数加权累积 | 防止尾批及微批数量改变损失权重。 |

709,443 条训练样本对应每 R 693 次更新，R35 共 24,255 次；T 共 300 次。为消除端点歧义：R 第 u 次更新，u≤500 时 lr=3e-4·u/500；之后 lr=3e-4·[1+cos(π(u−500)/(24255−500))]/2。T 的 Gumbel 日程只随 T 更新推进。

初值约定：共享参数直接复制；四张 SID embedding、Mixer 层标记采用 Normal(0,0.02)；自定义线性投影采用 Xavier uniform、bias=0；T5 主体沿用对应实现的初始化；Mixer 输出投影保持零初始化。Tokenizer 距离计算保持 FP32；标准化统计量用 FP64 累计。各随机流分开保存，验证不消耗训练随机流。

原实验依据：项目 configs/09_kuairec_listwise/base.json、scripts/listwise_diffgrm/alignment.py。它们只是数据协议和参数量级来源，不能把新的学习率、日程和距离归一化描述为旧 H1 的原样设置。

## 商品落地、验证与三组对照

### 从生成向量映射回实际商品

先从目录中排除本次列表已输出的商品。按四层残差量化逐层选择最近中心，但每次只允许选择“与已选 SID 前缀一致、且仍包含可用商品”的码字。得到完整 SID 后，在对应碰撞桶中选择距离生成向量最近的实际商品：

$$
i^*=\arg\min_{i\in\mathcal B(\mathrm{SID}),\ i\notin\mathcal L_{\mathrm{selected}}}\|\hat z-z_i\|_2^2.
$$

距离相同按原始商品 ID 升序；不使用目标标签或测试流行度。历史已看商品允许再次推荐，当前五件输出按商品 ID 去重；相同 SID 若包含不同商品可以重复出现。桶内比较量化前 zᵢ，因为同一 SID 的 qᵢ 相同，无法用于区分桶内商品。首轮不增加最近邻全目录替代落地、额外重排或多列表择优。

### 验证和选模

| 项目 | 设置 |
| --- | --- |
| 完整验证节点 | R5、R8、R11、R14、R17、R20、R23、R26、R29、R32、R35 |
| 主选模指标 | 完整验证集、单份五件列表、真实商品级 NDCG@5；平局选较早 checkpoint |
| 推理预算 | 每件 20 步 DDIM；每样本一份列表；训练与推理的五件顺序一致 |
| 评测随机性 | 固定 seed=20241111；按 seed、split、sample_id、列表位置派生初始噪声，不加入组名、轮次或 checkpoint |
| 测试集 | 只评验证选出的 best 与最终 R35；不依据测试结果改参数或选点 |
| 保存 | 每个正式验证节点保存完整 checkpoint：权重、两侧优化器及计数、随机流、数据位置、固定统计与目录版本 |

NDCG@5 以真实五件目标集合判相关，按推荐位置折扣，并用五个相关商品的理想 DCG 归一化；不要求预测第 j 件恰好等于真实第 j 件。报告 Recall@5、HR@5 和有效五件输出率。实现依据为 scripts/listwise_diffgrm/metrics.py。固定噪声使用稳定哈希派生并按样本生成，不能依赖进程内 hash 或 batch 顺序。

每 R 记录各损失；每 T 块记录 SID 改变比例、逐层码字利用率与困惑度、碰撞率、有效码本相对变化及 RMS(ΔCˡ)/RMS(Bˡ)。生成诊断记录自由 RQ 的目录合法率、目录约束改码率和桶大小，帮助区分连续生成、编码和落地接口的问题。需要观察训练曲线，但 loss 下降与推荐指标提高分别报告。

### 对照范围与结论边界

| 组别 | 结构差异 | 共同训练规则 |
| --- | --- | --- |
| B0：无 Mixer | 直接使用基础码本；码本仍接收联合训练反馈 | 独立 RQ 预训练→5R 预热→30 次 10T＋1R |
| B1：层内交互 | 每层内部四头注意力，共享参数，按参照层尺度校准 | 与 B0 相同；Mixer 从预训练阶段参与 |
| B2：层间因果交互 | 同样的四头注意力；每层读取自身和前层 | 与 B1 相同，主要改变注意力掩码 |

首轮 seed=2024；后续确认方向时按同一协议补 2025、2026，首轮不能据单 seed 声称稳定优势。三组使用相同数据、初始可共享参数、样本顺序、损失权重、训练预算和评测噪声；B1／B2 的初始注意力权重相同。参数初始随机数、数据顺序、扩散噪声、选码噪声和 dropout 独立管理，避免多一个模块消耗随机数而打乱其他条件。

三组分别预训练，后续 z 空间可能不同，因此主比较衡量完整方法的效果，不单独识别“仅联合阶段 Mixer”的因果贡献。相同 R/T 与 epoch 预算不等于相同 FLOPs，需同时报告训练时间、显存和推理延迟。旧 H0／H1 使用不同生成器和 beam 推理，作为外部参照，不当作只改变 Mixer 的严格消融。

实现验收关注：尺寸与掩码正确、T／R 更新对象正确、推荐梯度实际到达码本／Mixer、目录缓存随版本刷新、固定噪声下推理可复现、checkpoint 恢复包含完整状态。效果验收按同协议比较 best 与最终点，并结合曲线和编码状态解释；本设计不预设哪组胜出。

## 主要参考与本地依据

| 依据 | 采用范围 | 保留的差异 |
| --- | --- | --- |
| [TIGER](https://arxiv.org/html/2305.05065v3) | RQ Semantic ID、32 维潜空间和 MLP 结构、四层 Decoder 的容量参考 | 本模型四层真实 RQ；输出连续向量而非自回归 SID。 |
| [ETEGRec 官方实现](https://github.com/BishopLiu/ETEGRec) | 独立 SID embedding＋T5；RQ 预训练配方；软分布对齐思想 | Encoder 2 层、扩散 Decoder 4 层；单向停止目标梯度 KL；10T＋1R 非其原日程。 |
| [DIGER gradient-fix](https://github.com/junchen-fu/DIGER/tree/gradient-fix) | 冻结 RQ Encoder、软分配梯度桥接、潜空间及逐层量化约束 | 采用交替更新；距离归一化及温度继承／适配 H1，不照搬其全部参数。 |
| [DreamRec](https://github.com/YangZhengyi98/DreamRec)；[DDIM](https://arxiv.org/html/2010.02502v4)；[PreferDiff](https://arxiv.org/html/2410.13117v2) | 条件连续生成、干净向量预测、多步稀疏采样与 20 步参考 | 本模型用 SID 历史、RQ 落地和五件前缀；未照搬其 CFG／排序目标。 |
| experiment1 原实验 | KuaiRec 数据协议、五件列表评测、基础优化器参数、验证节点 | 训练总量改为 R35/T300；新 RQ 与连续生成架构独立实现。 |

本地核对位置：configs/09_kuairec_listwise/base.json；scripts/listwise_diffgrm/alignment.py；scripts/listwise_diffgrm/metrics.py；scripts/rqcd/tokenizer.py；docs/research/onerec_listwise/MODEL_EXPERIMENT_PLAN_20261027.md。公开源码核对日期为 2026-11-11，正式实现时记录实际使用的版本／提交号。
