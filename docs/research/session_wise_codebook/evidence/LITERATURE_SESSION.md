# Session-wise 码本交互：近邻文献与实现边界

核查日期：2026-10-23。范围是离散编码、上下文相关 tokenization 与索引，不以连续生成作为主线。下文区分论文方法、作者源码静态阅读和本项目建议；没有复现这些论文，没有执行作者代码、下载权重或使用 GPU。精确版本、源码 commit、文件 SHA256 和阅读位置见 [sources_session.json](sources_session.json)。

## 先确定“session-wise”究竟改什么

以下是本轮设计判断，并非论文已经替本项目证明的结论。

| 层次 | 改变的对象 | SID 与目录如何处理 | 本轮关联 |
|---|---|---|---|
| 条件化表示 | 给定历史 h，让已选中的码字表示在多个历史商品之间交互 | 商品 SID 与 SID→商品映射保持固定 | 可先检验跨商品码字交互是否有用，范围最小 |
| 条件化编码 | 同一商品 i 在不同 h 下得到不同的 SID(i,h) | 必须规定有限多 SID、动态索引，或其他落到真实商品的方法 | Pctx 最直接；PIT 提供训练与索引共同演化的近邻 |
| 改预测任务 | 一次预测多个未来商品或整个 session | 需要额外定义顺序、集合、去重和多目标指标 | 不能仅凭“session-wise”自动推导出来 |

“每个 session 调整码字向量、仍沿用原索引”属于第一层；“调整后重新最近邻量化、改变索引”进入第二层。两者不能使用同一个实验名称。原推荐器本来就会建模历史，所以增加一次普通历史 attention 还不足以说明实现了新的码本交互机制。

这里“固定 SID”指同一 tokenizer 版本内使用全局规范身份，并不要求联合训练中永久冻结 tokenizer。若沿用 05 的交替优化，可以在 tokenizer 更新后统一刷新全库规范 SID；关键是同一版本内，目标 SID 不因请求的历史而变化。

## 优先阅读的五篇

### 1. ActionPiece：最接近“跨商品 token 发生交互”的前例

[论文 v3](https://arxiv.org/html/2502.13581v3)；重点读 §3 的集合表示、跨集合合并、SPR 与训练/推理。

它把一次行为表示为商品特征集合，学习集合内部及相邻行为之间的合并规则；历史的分词因此受上下文影响。词表规则来自语料，不是每个 session 用 attention 重写整张 VQ 码本。训练仍预测下一个商品，推理还可对历史做多次分词集成。实验是 Amazon 2014 的 Sports、Beauty、CDs，按时间留末项评估。它支持“上下文可以进入 tokenization”这个研究动机，不能证明本项目的码字 attention 或联合更新必然有效。

**源码已读：**[作者 core.py](https://github.com/google-deepmind/action_piece/blob/ae8e61a89ade8d545a16119bcb5b3a43d9da852f/genrec/models/ActionPiece/core.py#L233) 的内外集合计数、跨商品合并与解码；[tokenizer.py](https://github.com/google-deepmind/action_piece/blob/ae8e61a89ade8d545a16119bcb5b3a43d9da852f/genrec/models/ActionPiece/tokenizer.py#L434) 明确将历史 `[:-1]` 与目标 `[-1:]` 分开编码。复用思想时应控制推理集成次数，否则额外计算量会混入提升。

### 2. Pctx：最接近“同一商品随上下文改变 SID”的前例

[论文 v1](https://arxiv.org/html/2510.21276v1)；重点读 §2.2、§2.3 及附录 C。

辅助序列模型对包含当前商品的行为前缀编码；按商品聚合这些上下文表示并压成有限个语义侧面，再结合内容表示量化。一个商品可以有多个 SID；生成后将同商品路径归并。其 Amazon 2023 三类数据采用留末项、全候选评估，AR beam 为 50。这里的目标是有限多身份，不是请求到来后对全库任意重编码。不能把“历史条件下重新量化商品”直接作为尚无前人的新点，也不能从这篇推出本项目有效。

**源码已读：**[cluster.py](https://github.com/YoungZ365/Pctx/blob/1e9f101d97a153734c7286d693e822d20fa16a47/genrec/models/Pctx/cluster.py#L283) 按商品聚类；[tokenizer.py](https://github.com/YoungZ365/Pctx/blob/1e9f101d97a153734c7286d693e822d20fa16a47/genrec/models/Pctx/tokenizer.py#L1847) 融合上下文/文本、构建行为到 SID 的映射。实现有 FAISS residual quantizer 与 RQ-VAE 分支，不能笼统称为只用 RQ-VAE。[model.py](https://github.com/YoungZ365/Pctx/blob/1e9f101d97a153734c7286d693e822d20fa16a47/genrec/models/Pctx/model.py#L109) 按 item 合并 beam；[evaluator.py](https://github.com/YoungZ365/Pctx/blob/1e9f101d97a153734c7286d693e822d20fa16a47/genrec/evaluator.py#L70) 将标签和预测转为 item ID 后判命中。尚未完成缓存构建/数据划分的全链路审计。

### 3. PIT：最接近“推荐损失反过来选择编码，并维护可用索引”的前例

[论文 v1](https://arxiv.org/html/2602.08530v1)；重点读 §3.2–3.5，不只看“端到端”的表述。

Item-to-Token 从商品表示提出多个 SID，User-to-Token 根据历史选择损失较小者，作为动态伪标签更新；另有协同信号辅助任务。动态 beam index 保留每商品多个路径并同步服务。公开实验为 Amazon 2014 Beauty/Sports/Toys，5-core、历史长度 50、留末项、R/N@5,10；另有作者报告的工业结果。商品 tokenizer 本身并非直接接收每个用户历史，个性化体现在选择环节。它不能证明“任意动态 SID 都无需维护目录”，也不是简单的码本 attention。

**实现边界：**本轮定向检索未定位到可确认的作者开源实现，故只据原文解析；不以第三方复现补成“作者代码已验证”。公开评估的候选过滤细节未做独立代码核验。

### 4. TIGER：固定 SID 与上下文预测的概念锚点

[论文 v2](https://arxiv.org/html/2305.05065v2)；重点读 §3.1–3.2、碰撞处理与实验设置。

商品内容经 RQ-VAE 得到固定残差码，碰撞附加区分 token；历史 SID 输入序列模型预测下一商品 SID。历史条件预测早已存在，新增点必须说明条件信息究竟进入表示、量化还是索引。实验使用 Amazon 2014 Beauty/Sports/Toys、R/N@5,10。它不是 session 条件化 tokenizer 的证据。本轮未核实原作者完整实现，标记为论文锚点，不把第三方 TIGER 仓库当作原实现。

### 5. DiffGRM：本项目应保持清楚的模型与指标锚点

[论文 v1](https://arxiv.org/html/2510.21805v1)；重点读 §3.1–3.4、§4.1–4.2。

PSE 对商品内容做 OPQ/分块量化，生成固定 SID；历史编码器提供条件，离散 masked decoder 在目标码位间交互。原方法不是每个 session 重估商品码。论文用 Amazon Beauty/Sports/Toys 留末项，报告 SID-based 的 R/N@5,10。因此不能直接与 Pctx 的 item 命中或本项目连续检索指标横排，也不能用论文结论替代项目联合训练的验证。

**源码已读：**[tokenizer.py](https://github.com/liuzhao09/DiffGRM/blob/238911435b6ca4f576d0b4d943359b98c865c172/genrec/models/DIFF_GRM/tokenizer.py#L167) 固定 OPQ 编码与反向映射；[model.py](https://github.com/liuzhao09/DiffGRM/blob/238911435b6ca4f576d0b4d943359b98c865c172/genrec/models/DIFF_GRM/model.py#L544) 先拼接各 SID embedding，经 MLP 得到每商品表示后再进入历史编码器；输出使用共享 embedding 切片。这是本次访问的上游 HEAD，并非项目当前冻结源码，也不是论文实验 commit 的证明。

## 对本项目设计的约束与可验证问题

**本轮更容易解释的起步问题：同一 tokenizer 版本保持全局规范 SID 和输出目录时，保留码位结构的跨历史商品交互，是否比相同预算的普通历史建模更有用？** 这是本轮建议，尚无实验答案。可继承 05 的“编码要服务于推荐任务”的动机，但不能把 05 的全局码本更新、07 的商品内交互和新的跨商品条件表示视为同一种处理。

若后续确实要做 `SID(i,h)`，应先解决以下接口，才能称为完整可推理方案：

1. **码的身份与目录。** 全局 `SID→item` 是否仍有效？同一 SID 在两个 context 下对应不同商品时如何消歧？采用有限多 SID、同步动态索引或其他映射，要显式选择。
2. **训练标签与部署可用信息。** 历史上下文不得含未来行为；训练可使用目标商品构造监督，但预测网络不能读取目标商品。不能靠训练时真实目标选出的 session 编码，在推理时假装已知。
3. **排序单位。** 多 SID 命中同一商品时如何聚合、去重；碰撞/无效路径如何处理。最终应补充真实 item-level 指标，避免仅提高 SID 命中。
4. **代价。** 上下文编码、额外 attention、候选 SID 数和集成都计入时延/内存。动态编码不是天然更快，也不是天然更稳定。

ActionPiece、Pctx、PIT 分别提供跨商品分词、有限多上下文身份、可更新索引的参考。这些是研究邻居，不是可直接拼接后宣称新方法的模块清单。本轮未证明该方向具备新颖性，也未证明它优于现有 05/07；这里先建立能区分机制、能落到真实商品的实验边界。
