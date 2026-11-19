# Grid 与 loop：原始文献、实现和本项目适用边界

核对日期：2026-10-23。本文只做结构研究；没有运行作者模型、安装环境或验证推荐收益。原文与作者代码的版本、方法位置和文件哈希见 [sources_grid_loop.json](sources_grid_loop.json)。不假定学长说的 grid / loop 特指其中任何一篇论文。

## 先明确网格的两个轴

- **历史码位状态**：`X ∈ R[B,L,M,D]`，L 是历史商品数，M 是每件商品的码位数，D 是推荐向量宽度。`X[b,l,m]` 是这段历史第 l 件商品第 m 个码位的当前表示。沿 M 交互讨论同一商品的组合，沿 L 交互讨论同一码位在历史中的关系。它是每次请求的激活，不是整张量化码本。
- **全局码本中心**：`C ∈ R[M,K,d]`，每个码本有 K 个中心。当前量化侧是 `[4,256,64]`，推荐侧 token 表则可视为 `[4,256,256]`。`C[0,17]` 与 `C[1,17]` 没有天然语义对应；17 与18也没有“邻近中心”保证。不同 OPQ 子空间的坐标含义也不同。

因此，同样画成矩阵不表示可以共用图像的邻域假设。`[L,M]` 的行列有任务意义；在 `[M,K]` 上只让同一整数索引跨码本交流，则引入了任意配对。重编号码本中心并同步修改 SID 后，若模型行为因此改变，说明结构依赖编号，而非真实商品关系。可考虑共现关系、可学习匹配或共同隐空间，但这是另一个待验证设计。

## 四篇论文到底分别做什么

### 1. Axial Attention in Multidimensional Transformers（2019）

[原文 §3.1–3.2](https://arxiv.org/html/1912.12180v1#S3.SS1) 将注意力一次限制在一个轴：其他轴并入 batch，沿当前轴做标准注意力，再恢复形状。顺序叠加不同轴，允许信息经过中间位置到达全局；“感受野覆盖全局”不表示与一层全注意力等价。原文完整模型是带因果遮罩、shift 和内外 decoder 的图像自回归模型，不能把全部结构直接称为推荐模型。

作者实现已核对 [axial/transformers.py](https://github.com/google-research/google-research/blob/7951440944924ac61b6e2f9a2a3c715834005c40/axial/transformers.py#L53)：`AttnBlock(axis)`、`nn.attn_nd(time_axis=...)`，以及行/列独立 block；残差前有 LayerNorm。其行列位置编码是图像位置，不应原样赋予无序中心编号。

对历史 `[L,M]` 可自然改写为：

\[
U=X+\alpha\,\mathrm{Attn}_M(\mathrm{LN}(X)),\qquad
Y=U+\beta\,\mathrm{Attn}_L(\mathrm{LN}(U)).
\]

这是本项目的**候选迁移设计**，不是论文已验证的推荐结论。若用最终历史预测下一件商品，所有合法历史可相互看；若同时对多个历史位置作 next-item 监督，L 轴须按各监督目标施加因果遮罩，不能看到未来。Padding 也必须在两轴正确处理。

### 2. MaxViT（ECCV 2022）

[原文 §3 与附录 A.1](https://arxiv.org/html/2204.01697) 的 block attention 是局部 `P×P` 窗口；grid attention 将相隔较远的位置组成固定 `G×G` 的稀疏全局组。一个 MaxViT block 还含 MBConv、两次注意力/FFN及残差。**它不是简单的先沿行、再沿列的 axial attention。**

作者实现已核对 [maxvit/models/maxvit.py](https://github.com/google-research/maxvit/blob/70a470723b810f9eb05b2aa935cf75db8290c69e/maxvit/models/maxvit.py#L805)：`window_partition` 按局部窗口重排，`grid_partition` 按固定网格大小重排，分别调用注意力并恢复；`call` 串联 MBConv、block 和 grid。

可借鉴的是“任务相关的局部组合＋稀疏跨组信息传播”。但 M=4 时复制图像默认 `P=G=7` 无意义；历史时间与码位也没有图像的平移等变性。应重新定义分组，或使用轴向注意力，并如实称为借鉴多轴思想，而不是 MaxViT 复现。原文顺序/并行分支消融仅说明其图像实验结果，不能据此宣布推荐上哪一种最好。

### 3. Grid-Attention: Enhancing Computational Efficiency of Large Vision Models without Fine-Tuning（ECCV 2024）

[原文 §3.2–3.3、Algorithm 1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06640.pdf) 把局部 `S×S` 中不同偏移的位置分到 `S²` 个子图：`[B,H,W,D] → [B S²,H/S,W/S,D]`，各子图内部做原 MHA，随后逆置换恢复。**文中的 Grid Clustering 不是 k-means，也不是平均合并 token；token 总数保留，改变的是注意力连接。**

作者代码 [image_encoder.py](https://github.com/pengyuLPY/GridAttn/blob/dd63f6f53f30600e844571c4f5b1968ac67fd35b/GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM/segment_anything/modeling/image_encoder.py#L174) 和 [utils_py.py](https://github.com/pengyuLPY/GridAttn/blob/dd63f6f53f30600e844571c4f5b1968ac67fd35b/GridAttn_on_SAM_and_ExpeditSAM/Expedit-SAM/segment_anything/modeling/utils_py.py#L48) 已逐项核对：reshape/permute → attention → inverse；SAM 版本训练时 shuffle，推理时固定分组。原文还讨论固定棋盘分组训练时的信息隔离，以及 shuffle/deformable 方案。

启示是“改变连接图而保留已有权重”，并提醒固定稀疏分组会阻断某些信息。但其依据是图像空间冗余，不能推定码本 token 也有相同冗余，更不能承诺直接替换后无需训练即可保持推荐质量。该论文与 MaxViT 的 grid 也不相同：这里固定分组数 `S²`，每组随输入增长；MaxViT 固定每组 `G²` 个 token。

### 4. Universal Transformers（ICLR 2019）

[原文 §2.1–2.2](https://arxiv.org/html/1807.03819#S2.SS1) 沿**计算深度**重复应用共享参数的注意力与状态变换：`H^(t+1)=F_theta(H^t, position, t)`。同一步的全部位置并行，但后一步依赖前一步。原文有位置/步数信号，以及可选的逐位置 Adaptive Computation Time（ACT）；固定步数版本也是有效研究对象。

作者实现已核对 [universal_transformer_util.py](https://github.com/tensorflow/tensor2tensor/blob/bafdc1b67730430d38d6ab802cbd51f9d053ba2e/tensor2tensor/models/research/universal_transformer_util.py#L195)：固定步数通过 `tf.foldl` 反复调用同一更新函数，basic 更新在共享 recurrent layer scope 中做 attention/FFN；`step_preprocess` 加位置/步数信号；ACT 路径另有 halting 状态和循环。**它不等于循环播放同一个不变输入，也不等于训练时反复修改全局码本。**

对项目可让共享 `F_theta` 反复更新 `[L,M,D]` 状态，或者让 `F_theta` 内部采用两个轴向块。收益假设是参数共享、多轮条件修正；不是“循环必然更快”。重复 T 次通常约付出 T 次计算。ACT 即使理论上减少部分位置的更新，实际 GPU batch 仍可能受最慢样本和掩码开销限制，需测量才能谈加速。

## 计算量、稳定性和比较口径

以下为标准注意力 `QK^T` 与 `AV` 的主项，D 为隐宽，省略 batch、head 常数。每次注意力还包含 `O(ND²)` 投影，FFN 通常也是同阶；故 attention 配对数不能直接当作端到端速度比。

| 候选 | 历史 `[L,M,D]` 的注意力主项 | 参数及信息路径 | 主要限制 |
|---|---|---|---|
| 展平后全注意力 | `O(L²M²D)` | LM 个 token 可直接交流 | 成本较高，可能没有必要的结构约束 |
| 两轴顺序交互 | `O((LM²+ML²)D)` | 同商品跨码位、同码位跨历史；两个轴块通常不共享参数 | 两跳传播不等于全注意力；两套投影/FFN有常数成本 |
| 共享全注意力 loop T 次 | `O(T L²M²D)` | T 次迭代共享核心参数，状态逐轮变化 | 省参数不省轮次；可能过度平滑或优化困难 |
| 两轴交互＋共享 loop T 次 | `O(T(LM²+ML²)D)` | 每轮两个轴，在轮间共享参数 | 同时改变连接和深度，须分别做消融 |

例如 `L=50,M=4`：全注意力每头40,000对，双轴一轮10,800对，只是配对主项约3.70倍差异。以相同 T 比较 shared/untied 才能看共享效果；以实测 FLOPs/延迟匹配比较才能看预算收益。需要同时报告参数量、训练更新数、显存和推理时间。

MaxViT 在图像 N=HW 上的注意力主项为 `O(N(P²+G²)D)`，P/G固定才相对N线性。ECCV2024 GridAttn 的 S² 子组各 N/S² 个 token，标准注意力推导为 `O(N²D/S²)`；固定 S 仍为二次，只改善系数。本文按代码中的矩阵维度推导，不沿用该论文印刷 Eq.(1)–(2) 中将第二个 matmul 写成 `(HW)^3` 的非标准复杂度表达。

若把公式换成全局 `[M,K,d]` 码本，展平注意力为 `O(M²K²d)`，机械轴向为 `O((MK²+KM²)d)`。M=4、K=256 时配对数 `1,048,576 → 266,240`。**这个省算式不解决同号中心跨码本配对缺乏语义的问题，也不保证量化质量。** 当前05 `CrossPSECodebookMixer` 确实展平 `[M,K,d]` 后做 MHA（`cross_pse.py:124`），不能与每个 session 的 `[L,M,D]` 全注意力混为同一种实验。

稳定性的可检验设计包括 Pre-LN、残差、受控更新幅度、少量迭代及初始近恒等。它们是工程假设，不是四篇论文对本任务的保证。尤其避免“门控=0且输出投影=0”同时使用造成整个新增分支初始无梯度；跟踪每轮状态变化、码位余弦相似度和实际梯度。若只更新请求内激活，不会直接重分配全库 SID，但仍可能影响推荐器训练；若更新量化中心，则还要检查跨阶段码本漂移和 SID 稳定性。

## 研究定位与最小区分

可以先区分两个问题：**需要哪些位置互相看（grid/axial）**，以及**同一更新规则是否值得反复用（loop）**。比较 full、axial、full+shared-loop、axial+shared-loop 时，须固定输入对象、输出接口、因果/padding口径和训练数据；增加相同步数但不共享参数的参照，避免将更深误判为共享收益。最终仍需在同样推荐质量下比较完整推理成本。

这些机制已有明确先例。仅将“axial＋共享循环”命名为新模块，不能自动构成研究创新；较有意义的潜在问题是：如何利用真实商品组合/历史关系定义交互，如何稳定适配推荐目标，以及这种结构是否在受限搜索预算下改善质量。上述方向均未在本文得到实验验证，没有一篇文献能替我们选出普遍最优结构。
