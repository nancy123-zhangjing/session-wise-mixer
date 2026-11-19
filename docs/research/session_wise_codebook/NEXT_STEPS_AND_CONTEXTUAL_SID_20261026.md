> 2026-10-26退役说明：本文属于旧的单目标历史侧交互研究，不能作为OneRec请求级列表生成的当前实现或执行方案。相关代码、模型和恢复包已按用户要求删除；历史结果保留。当前方案见 docs/research/onerec_listwise/PLAN_20261026.md。

# Session-wise 后续判断与上下文 SID 可行性

日期：2026-10-26。状态：只读核查与研究设计；本次仅新增此研究记录，没有修改模型代码、提交远端任务或使用 GPU。文中的新实验均未执行。旧 DESIGN/EXPERIMENT_PLAN 保留历史，不以其中的概率路由原型代替 10 月 24 日真正执行的 64→256 残差分支。

## 1. 先把 baseline 和联合训练说清楚

本轮三臂均完成 35 个推荐器 epoch、3640 次推荐器更新、60 次 Tokenizer 更新，实际为 5R＋10×(1T 阶段＋3R)。R 阶段训练推荐器和新 F，冻结物理码本与全局 Mixer；T 阶段冻结推荐器/F 的参数，保留经过 F 输入的梯度路径，更新码本与 Mixer。它是交替联合训练。

baseline 关闭新增 F，但保留原 Full Mixer。三组共用 pre-05 Aug-28 推荐器和有效码本 E0，不接 05/07 联合训练的最终模型。训练完成到 R35，不等于最后测试一定使用 R35；原验证规则选中 baseline R15/T24、no_context R30/T54、axial R35/T60。

以下均是测试集 HR/NDCG，采用 0–1 数值；旧实验与本轮均使用 beam256，但初始化、验证选模设置等未全部相同。

| 实验 | HR@10 | NDCG@10 | 说明 |
|---|---:|---:|---|
| 05 早期 Frozen Cross | 0.074230 | 0.044670 | 历史预训练 Cross 路线 |
| 05 早期 Joint Cross v2 | 0.077315 | 0.045987 | best R5/T0，尚未执行 Tokenizer 更新 |
| 05 早期 Frozen Refined | 0.075661 | 0.044648 | 历史预训练 Refined 路线 |
| 05 早期 Joint Refined | 0.077270 | 0.044354 | best R15/T24 |
| 05 共享起点长期四组：Mixer＋Joint | 0.074230 | 0.044581 | 共同 pre-05/E0 起点 |
| 08 baseline | 0.074230 | 0.044581 | 与上一行 @5/@10 四指标逐项精确一致 |
| 08 axial | 0.073648 | 0.043653 | 本轮 session-wise 分支 |

因此，“当前 baseline 相比对应的 05 基线下降”不成立；“它比某些历史结果低”成立。相对早期 Joint Cross v2，当前 HR@10 低约 3.99%，NDCG@10 低约 3.06%。这不是差一个数量级，也不能归因于本轮没有联合训练。早期 Cross 的较高结果又恰好来自 T0，不能作为 Tokenizer 联合更新有效的证明。初始化/优化轨迹及选模不同是可确认的实验差别，具体哪一项造成分数差距仍未被单变量实验识别。

更精确地说，早期 Cross 与本轮使用同一个 Aug-28_19-01-02 离线推荐器，但前者加载 Cross-PSE v3 完整已学 Tokenizer；共享起点长对照及本轮把有效码本 E0 作为新 base，重建输出为零的 Full Mixer，保留初始有效几何与 SID，却不继承旧 Mixer 内部参数。不能简单说两个推荐器起点都不同。05 早期 Refined 和 07 则使用另一份 Aug-28_14-15-28 离线推荐器、Refined-PSE v2、Mixer OFF；07 的 recipe_id=refined_joint 指初始资产和配方，不是加载 05 最终联合权重。

05 早期与共享起点长期实验的选模验证采用 beam32，本轮采用 beam256，测试均为256。当前 baseline 恰好也选中 R15/T24，并复现旧 long 对应组的四项测试指标；这不代表所有阶段的验证过程完全相同。

05 共享起点短四组只有短恢复及验证结果，不混入此表。07 的 0.104011/0.105129 是 beam256 验证 HR@10；07 测试约 0.0801，且走另一条 Refined 暖启动路线。不能与本轮测试 0.0742 直接解释成同一个模型退化。

证据：

- [05 早期原始结果](../../../experiments/05_crosspse_joint/results.csv)
- [05 长期四组测试](../../../experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_audit_r1/best_selected_test.csv)
- [08 实际训练报告](../../../experiments/08_session_codeword/REPORT.md)
- [08 测试机器汇总](../../../experiments/08_session_codeword/test_best_20261025/SUMMARY.json)
- 实际完整训练状态：`outputs/stage08_session_codeword/delivery_20261024_final_r1/restored/goal_20261024_session_codeword_v1/formal/{baseline,no_context,axial}/progress.json`。

## 2. 当前结果支持什么

按原规则选 best 后，axial 在验证与测试上都未超过 baseline，不能描述成“验证明显提升但测试失效”。固定 R35 的验证 NDCG@10 约 +1.10%，回答的是等训练预算问题；它不等于实际选模后的收益。测试 axial 的 NDCG@10 相对 baseline −2.08%，HR@50 −2.79%，仅 HR@5 略高。

这足以暂缓给现有模块加 loop、加深或延长训练，但不足以否定整个 session-wise 方向。现有模型本来就有历史 Encoder，新 F 可能与其重复，也可能混入无效上下文；这些是待检验解释，不是已确认原因。残差非零、梯度存在、SID 改变都不能证明上下文有效。现有结果只有 seed2024，且属于受碰撞影响的 SID 级评价。

## 3. 四种不同的“session-wise”

| 方案 | 同一 checkpoint 下随历史变化什么 | 输出身份/目录 | 当前建议 |
|---|---|---|---|
| 已做的连续历史残差 | 历史商品输入向量 e(i,H) | 全局目标 SID | 先诊断，暂不扩展规模 |
| 仅历史侧上下文离散码 | 输入 SID_in(i,H) | 标准目标 SID_out(i) | 改 SID 的优先候选 |
| 有限多 SID / 有限上下文目录 | 商品使用若干预定义别名之一 | 静态 alias→item 表或有限路由目录 | 第二步研究目标编码时考虑 |
| 任意请求动态目标 SID | 全库每件商品的 SID(i,H) | 通常需请求专属目录 | 暂不优先 |

当前 F 已经是 session-conditioned 表示；不改变目标 SID，并不意味着“不是真正 session-wise”。全局 SID 随 T 更新而变化，与同一模型下同一商品因不同历史而改变码，是两种不同的变化。

若目标码依赖当前请求，必须回答：生成完这个码，怎样快速、唯一地找到真实商品？如果每次先给全库 N 件商品重新编码再建反向表，就引入在线 O(N) 计算，与避免扫描大目录的效率动机冲突。小数据集上能运行，不等于解决了大目录服务问题。

## 4. 最直接的参考论文及边界

**ActionPiece** 更适合帮助理解“只改变历史 token 化”。它把历史看成有序的商品特征集合，学习商品内及相邻商品的合并，token 可以跨商品；目标单独编码，不根据 session 选择目标身份。重点读 §3.2–3.3、Appendix H。它不是把当前四位 OPQ SID 接上一个注意力层，变量长度、特征集合、AR 解码与集成都不同。[原文](https://arxiv.org/html/2502.13581v3) · [官方代码](https://github.com/google-deepmind/action_piece)

源码核查边界：论文附录说目标不做合并，公开 encode_labels 仍调用允许商品内确定性合并的核心实现；稳妥结论是目标编码不依赖 session。多历史随机编码的推理集成也有额外前向成本。[tokenizer](https://github.com/google-deepmind/action_piece/blob/main/genrec/models/ActionPiece/tokenizer.py) · [core](https://github.com/google-deepmind/action_piece/blob/main/genrec/models/ActionPiece/core.py)

**Pctx** 是“同一商品有限多个 SID”的直接参考：用训练中的上下文表示按商品聚类，中心与内容特征结合后量化，形成有限 alias 集合；生成多个 alias 后归并到 item。它不是每个请求任意重编码整个目录。重点读 §2.2–2.4 与实验/附录的合并、低频 alias 处理。其 AR 共享前缀动机不能直接照搬到 DiffGRM 的任意顺序解码。[原文](https://arxiv.org/html/2510.21276v1) · [官方仓库](https://github.com/YoungZ365/Pctx)

公开配置默认走 FAISS Residual Quantizer，不能简单称为端到端神经 RQ-VAE。公开模型对搜索到的 alias 路径概率做 logsumexp，但有限 beam 会漏路径，多 alias 也会占用候选名额。移植前应独立核对数据边界：所读 tokenizer 构建增强列表时可能纳入验证/测试 occurrence 的频率，不能直接声称原样复用已证明无泄漏；本项目应使聚类、合并和增强分布全部 train-only。这是静态代码风险发现，不是对论文实验的泄漏定论。[配置](https://github.com/YoungZ365/Pctx/blob/main/genrec/models/Pctx/config.yaml) · [tokenizer](https://github.com/YoungZ365/Pctx/blob/main/genrec/models/Pctx/tokenizer.py) · [model](https://github.com/YoungZ365/Pctx/blob/main/genrec/models/Pctx/model.py)

另外，ETEGRec 的交替训练/编码与推荐对齐，LETTER 的协同信息参与编码，以及 TIGER 的商品消歧都值得按需补读。它们提醒我们：“根据推荐目标学习编码”和“在请求中动态改变目标身份”不是同一个贡献。[ETEGRec §3.3–3.4](https://arxiv.org/html/2409.05546v2) · [LETTER §3.1–3.2](https://arxiv.org/html/2405.07314v2) · [TIGER §3.1–3.2](https://arxiv.org/html/2305.05065v3)

以上是原文及代码阅读，没有复现这些论文，也不据此认定新方案有新颖性或必然提升。

## 5. 下一步最小实验：先判断上下文是否有用

最值得先问：**在当前训练好的 axial 中，跨历史信息究竟帮助了预测，还是新增变换本身起作用？**

固定同一 axial best、其 SID、完整验证集、beam256、batch8 和评估随机性，仅做三个推理设置：原 F；时间轴 attention 改为 self-only（其余权重不变）；F 输出残差置零。比较 HR/NDCG@5/10/20/50，以原 NDCG@10/HR@10 作为主判断，同时看逐用户配对差、历史长度分组、原 F 与 self-only 的输出差异。这组实验未运行。

- 原 F≈self-only：优先检查上下文利用和与已有 Encoder 的重复，不直接加 loop。
- self-only 更好：再考虑短窗口/选择性门控，针对干扰提出新设计。
- 原 F 更好但仍不如 baseline：上下文可能有用，问题可能在融合或训练代价。
- F off 更好：提示这条附加分支可能产生负作用。

这是推理干预，有分布变化，不能代替重训消融。F off 也不等于原 baseline，因为推荐器及码本已经走过不同的优化路径。用户级配对区间不能代替跨训练 seed 的稳定性证据。

## 6. 如果进入“改 SID”研究：优先只改历史侧

一个可验证的实现候选如下。商品原 OPQ 分块为 x[t,m]∈R^64，物理码本为 C，有效码本为 C̄=FullMixer(C)，原历史分配为 s(i,m)。原 SID 按有效码本分配，因此下列取码与重新量化都使用 C̄，不能绕开已有 Mixer。

1. 取当前已知历史的码字 C̄[m,s(i_t,m)]，沿用现有书轴/时间轴 F，时间轴只看该位置及之前的有效历史，最长仍为50。
2. 将 F 的读出从 64→256 改成零初始化 64→64，得到 δ[t,m]。
3. 重新选择输入码 z[t,m]=argmin_k ||x[t,m]+δ[t,m]−C̄[m,k]||²。
4. 用原 256 维输入 embedding 表 W[m,z[t,m]]，再走原 item_mlp、history Encoder 和 DiffGRM。
5. 目标仍预测 canonical SID(i*)；取消旧 256 维加法残差，避免同时改两种机制。

δ=0 时必须逐位复现原历史 SID。训练采用 hard-forward/soft-backward 或明确的 ST 估计，推理使用确定性 argmin；需要检查梯度与实际改码率，不能把 δ 非零当成改码成功。软反向是梯度近似，不保证硬码变化有利。已知历史商品允许其四位组合不是合法输出商品；它只是输入表示，不能套输出目录过滤。

联合训练保留：R 更新 F/W/推荐器并冻结 C/Mixer，使用当前阶段缓存的有效码本 C̄；T 冻结 F/W/推荐器参数并更新 C/Mixer，使用实时计算的 C̄，推荐梯度穿过输入量化回传物理 C 与 Mixer。T 后刷新 canonical 目标目录。同一规则不等于各臂训练后的目录字节相同；交替训练可能使它们沿不同路径变化，应照实报告。

关键实现约束（尚未修改）：当前 unique-item 去重不能用于上下文码；同一商品在不同位置可以有不同输入码。R 阶段也需要建 ST 图；目前历史 ST 主要在 T 阶段创建。无 target 的推理入口也必须计算上下文输入码。不能把这些历史码写回全局 item2tokens/hard_codes。

最小可信重训对照是三臂：canonical 输入；同参数量且时间轴 self-only 的离散重编码；因果跨历史离散重编码。共用已锁定 pre-05/E0 初态、同 R/T 预算与验证选模规则；首轮不同时加入 loop、目标 alias 或新损失。旧连续残差只作背景，如需证明离散优于连续需另做预算/容量对齐。以验证收益作为继续投入的信号，随后固定方案补 seed；最终测试只验证已经封存的决定。持续不优于 self-only 就停止这个具体重编码方案，而不是扩大搜索追偶然最好值。

## 7. 如果一定要改变目标 SID：有限 alias，而非任意重编码

可考虑 Pctx 式每商品少量离线 alias，或 K 个有限上下文目录。两者不要混称为同一个算法：前者生成别名后按 item 汇总，后者先用历史选择 route，再在该固定目录解码。route 必须仅依赖可见历史，并进入模型条件和目录选择；不能用未知目标决定路由。K=2 只是未来试验起点，不是有依据的最优值，低频商品应保留单一标准码。

离线目录约 O(KN)，在线不必重编码全库。但当前 tokenizer 一商品一 SID、反向单值映射、缓存与 evaluator 都不直接支持多 alias。当前 confidence beam 在完整四位搜索结束后判合法、去重和回填，没有现成 trie 可直接复用，也不返回最终可聚合分数。DiffGRM 的 confidence 路径分数不自动等于 AR 归一化路径概率，不能直接照搬 Pctx 并称为真实 item 概率。

训练至少比较单 SID、同 alias 数但不按上下文选择、按上下文选择；否则无法区分上下文作用和容量/增强作用。先把映射与评价打通，再谈与交替联合训练结合。只读可行性核查不构成正式训练方案验收。

## 8. 所有目标改码方案必须先解决商品级评价

当前默认 evaluator 比较完整 SID；约18%的商品位于 SID 碰撞组。可选 xitem 路径也需核查修正：目标若由 SID 逆映射会丢失真实商品身份，多 alias 展开还需全局 item 去重。应使用 batch 中原始 target_item_id，按真实商品汇总、去重，确定不借助测试答案的碰撞处理策略。旧基线须用相同商品级规则重评，不能把新 item 指标直接与旧 SID 指标排榜。

有限 alias 还应报告每商品 alias 数、覆盖率、同 item 占用 beam 数、不同商品候选数及端到端耗时。码长/beam/生成步骤未减少时，不能宣称解决了解码时延。

冻结代码核查位置：`.../restored/goal_20261024_session_codeword_v1/engine/genrec/models/DIFF_GRM/` 中 session_codeword.py、model.py、joint_cross_pse.py、tokenizer.py、beam.py、evaluator.py。实际执行源码与当前工作区 HEAD 不同，本记录依据冻结执行副本。

## 9. 与学长沟通时最需要确定的三件事

1. 当前先验证“历史编码更适合推荐器”，是否可以保留标准目标 SID；还是必须研究同一商品在不同请求下的目标身份？
2. 近期目标是固定生成预算提高质量，还是相同质量下降低服务耗时？两者实验主指标不同。
3. 如果目标是 decoder-adaptive encoder，是否先以 DiffGRM 为固定结构、继续交替训练来验证一种编码机制，再讨论跨 decoder 通用性？一次在一个 decoder 有效不能证明通用。

建议顺序：当前 F 的最小上下文贡献诊断 → 必要时做历史侧离散重编码 → 有证据后才考虑有限目标 alias。任意请求重编码全库不作为当前首选。现阶段没有新结果可支持加 loop 或直接宣布动态 SID 更优。
