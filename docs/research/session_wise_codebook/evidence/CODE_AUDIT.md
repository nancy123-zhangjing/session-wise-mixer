# 05码本主线的session-wise扩展：当前代码审计

审计日期：2026-10-23。这里只审查现有代码与候选集成契约，没有修改生产代码、训练模型、申请GPU或安装环境。源码字节与版本见同目录`code_source_snapshot.json`：根仓库`037876cbb3de339ad011e5abcf0e66169967ce3b`，引擎`1c879252f6395d6e64462511823c3929b9c42c00`，审计时引擎工作区干净。下列行号均指这份当前源码，不将历史不同执行版本混成一个实验。

## 1. 核心结论

05的物理码本是`C[M,K,64]`；推荐器的SID embedding是另一张独立的256维表`W`。现有推荐器没有直接接收64维码字。T阶段经ST分配把推荐梯度传回码本，R和推理默认仅查当前hard SID；因此“64维session条件路由”是可以设计的新扩展，不是当前已经实现的功能。

建议首先保留05的全局码本、全目录canonical SID与原生4步离散输出，只在历史编码端增加**依赖当前历史H、读取05真实64维码字及距离几何的路由残差**。这有实质05血缘，但应准确称为“共享静态码本上的session条件路由/历史表示”，不能称动态session码本或动态商品SID。仅把全局Mixer改为grid或loop，若输入仍没有H，就不是session-wise。

## 2. 当前真实计算路径

```text
固定文本/OPQ槽位 X_i[M,64]
                    ↓ 距离计算
C[M,K,64] → 全局Full Mixer → Cbar[M,K,64]
                    ↓
T：a_ST[...,M,K]（硬前向、软反向）
R/推理：hard_codes[i,M]（全目录共享快照）
                    ↓
独立SID表 W_m[K,256]，按分配加权/按hard ID查表
                    ↓
历史[B,L,M,256] → concat[M×256] → item_mlp → [B,L,256]
                    ↓
历史位置编码＋history Transformer → encoder_hidden
                    ↓
原生4步confidence masked filling/beam → 商品SID → 目录合法性过滤
```

| 当前证据 | 精确位置 | 解释 |
|---|---|---|
| 全局码本交互 | `external/DiffGRM/genrec/models/DIFF_GRM/cross_pse.py:113-134` | `[M,K,d]`展成`[1,MK,d]`做全注意力；没有batch历史或session输入，返回`base+scale×mixed`。 |
| 固定商品槽位与SID | `joint_cross_pse.py:127-145` | 槽位重排到目录ID并注册buffer；`item_slots`不是随session变化的编码。 |
| ST分配 | `joint_cross_pse.py:261-316` | 距离经温度和可选Gumbel得到soft，再argmax；自定义ST前向严格one-hot，反向走soft。PAD分配全零。 |
| 独立256维表 | `model.py:361-369`、`:670-683` | `nn.Embedding(vocab_size,n_embd)`与64维码本不同；ST按digit乘对应的`W_m`。 |
| 历史压缩及序列建模 | `model.py:949-1006` | 码位先合成item表示，再经历史Transformer；模型已经有商品间历史交互，不能将任意新attention叫作“首次session建模”。 |
| 输出表共享 | `model.py:404-413`、`:855-887` | 输出logits使用推荐器SID表；不是与物理64维码字直接点积。 |

`joint_sequence_adapter`虽然可将256维历史pool映射为`[M,64]`（`model.py:685-693`），它属于SIA辅助路径。当前05长期协议的两侧SIA权重为0（`scripts/train/run_diger_mixer_long.py:330-335`）；不能将该适配器冒充当前前向实际使用的05码字表示。

## 3. R、T与推理究竟暴露了什么

| 阶段 | 当前输入桥 | 是否有完整soft分布 | 新分支必须补的接口 |
|---|---|---|---|
| 推荐器R | `forward`调用`prepare_joint_batch(...,differentiable=False)`，读取hard快照 | 否；没有`history_sid_st` | 取hard历史码字`onehot(hard)@Cbar`，以及确定性距离logits。需要显式取得Cbar/cache。 |
| Tokenizer T | `forward_joint_tokenizer`调用可微prepare，分别对history/target计算ST | 有ST张量；**原soft张量没有返回**，只有entropy等摘要 | 可用现有ST构造64维代理；若需要P0，新增确定性距离概率接口，不额外抽Gumbel。 |
| 推理/验证 | `trainer.evaluate`先prepare普通hard batch，再求encoder_hidden | 否；与R一样没有ST/soft | 必须调用与R相同的历史路由函数，不能只有训练有新分支。 |

证据：`model.py:602-668`、`:896-913`、`:695-709`；`trainer.py:1155-1166`。`diger_assignments`返回的是`expanded_ST,diagnostics`，不是可直接拿到原soft的API。PAD通过clamp先参与unique查找，但最终分配乘valid mask，新增概率或路由也必须处理PAD，而不能把PAD的均匀softmax当商品。

`JointCrossPSEAlignment.set_recommender_cache`确实缓存detached Cbar（`joint_cross_pse.py:243-251`），但当前仅` sia_loss`消费这个cache（`:328-341`）；原R推荐路径并不使用它。若新增64维历史分支，应显式增加“R/eval用有效冻结cache、T用live Cbar”的访问契约，不能一直复用detached cache而声称T梯度回到码本。

## 4. 交替更新、梯度与全目录刷新

- `model.py:522-540`：R参数是所有非`joint_alignment.`参数；T优化器分组当前只遍历`joint_alignment.mixer.named_parameters()`。新增模块放在哪里会实际决定它能否训练，不能只在图上画箭头。
- `model.py:542-566`：R冻结整个alignment，推荐器train、alignment eval并建cache；T先冻结所有参数，再只打开允许的码本/Mixer分组，推荐器eval、alignment train并清cache。
- `model.py:695-763`：T并非`no_grad`穿过推荐器；推荐器参数冻结，但梯度必须沿输入与目标ST回码本。首批用`autograd.grad(recommendation_loss, alignment_params)`验证有限非零梯度。
- `model.py:1248-1253`和`:1297-1309`：当前T除了历史分支，还通过可见目标码位的ST输入、ST标签NLL把梯度传向码本。不能把“固定canonical目录”误写成“没有目标侧梯度”。
- `scripts/train/run_diger_noise_pilot.py:322-356`：实际训练helper在阶段入口设相位；T FP32，R显式BF16；每批混合train/eval模式由既有helper修复。新F在T冻结时仍需对输入保持autograd，不允许整个F包`no_grad`。
- `scripts/train/run_diger_mixer_long.py:202-234`：每阶段检查另一侧完整状态不变，T后调用`refresh_joint_codes`。
- `joint_cross_pse.py:397-419`：全目录商品槽位在同一Cbar下确定性最近邻，重建hard_codes；`model.py:568-594`重建正反映射并失效`_cb2items`缓存。

本设计中“target SID固定”建议严格定义为：**同一个目录版本内不随H变化，继续按05在每个T阶段结束后全局刷新**。这不等于从训练开始到结束永不变化。若另设永久不变target表，应单独立项，它会改变05目标侧ST/刷新协议，不能悄悄与原05结果混比。

## 5. 可行首选：64维session条件路由残差

以下为拟议接口，尚未接入或验证完整生产模型。

1. 保留`Cbar=Mixer(C)`与当前hard目录。对历史得到`A_base`：R/eval为canonical hard one-hot，T为原history ST。构造`Z0=A_base@Cbar`，形状`[B,L,M,64]`，且PAD严格mask。
2. 给Z0加入历史位置和book身份。`F`在同一条历史的`L×M`网格上产生`q_delta[B,L,M,64]`：one-full块作为朴素session对照；grid沿L和M交替attention；loop递归更新同一状态并共享F参数。
3. 从固定商品槽位X和当前Cbar补算确定性`l0=-distance(X,Cbar)/tau`。构造`b=gamma*tanh(dot(q_delta,normalize(Cbar))/sqrt(64))`，`PH=softmax(l0+b)`，`P0=softmax(l0)`。
4. 继续使用独立推荐器表W：`E_history=A_base@W + eta*(PH-P0)@W`，随后走原`item_mlp`、历史encoder与离散decoder。canonical target、label、输出目录不使用PH，因此不需要每个session重编全目录。
5. F的最后输出投影zero-init，gamma与eta固定非零，使`q_delta=0 → PH=P0 → E_history=原路径`，实现共同功能起点。R阶段C冻结、F和W训练；T阶段F和W冻结，保留对输入的梯度，推荐损失通过原ST和新路由几何回到C。

这继承05的关键是：前向真实读取Cbar64及其量化几何，梯度回到共享码本，H改变相同码字空间中的路由。它**同时增加了推荐器侧的session条件前端**，不等于只替换全局Mixer，也不是“05已经有的模块改了个名字”。07的256维单商品SID表attention不能直接拿来当作这里的物理码字交互。

### 必须保留的解释及失败条件

- `PH-P0`是有正有负、各槽和为0的连续残差；最终E不是新的概率分配或合法SID，不能把它送进输出目录。动态的只是历史表示，商品输出编码仍全局共享。
- PH/P0必须共用完全相同l0、温度与掩码，zero-init差分才能精确抵消。原T基桥继续用既有Gumbel；新bias支路可用确定性距离，不能增加随机抽样却声称原随机流和加噪调用量不变。
- 不可把eta/gamma也初始化为0：双零门会阻断F起步。F输出为0的首步主要训练out_proj，不能要求所有上游参数首步都非零。
- `/tau`距离logits可能远大于有界b。需检查数值范围、PH−P0大小、tanh饱和和反向梯度；不能仅凭公式认定路由有实质作用。
- F必须仅依赖已知历史及共享码本。当前loader有target_item_id供监督/审计，但修改target字段不得改变`encoder_hidden`或history routing。已知历史内部双向attention符合当前prefix→下一商品协议；若以后改为同一序列同时监督多个位置，需重新加因果掩码。
- 历史只有SID的输入不应被迫要求推理时知道未来target_item_id才能构造路由；应给路由单独的`history_ids/history_mask`签名。
- 这条线不是连续商品生成；输出仍为4位离散SID。loop增加历史编码计算，也没有减少原生4步生成轮数。

## 6. 建议集成位置与最小对照

| 文件/函数 | 候选改动契约（本次未执行） |
|---|---|
| 新`session_codeword_routing.py` | 仅接收历史64维码字、位置/book标签、mask和共享Cbar/logits；实现one-full/grid/loop组合。不得读取target或目录评估结果。 |
| `joint_cross_pse.py` | 新增只读Cbar访问和确定性history距离logits接口；T返回live张量，R/eval可读冻结cache；避免重复Mixer与Gumbel抽样。 |
| `model.py:949`附近 | 取得原A_base@W后，加上述路由残差，再送原item_mlp。三阶段调用相同接口；`generate`已复用一次encoder_hidden，不能每个beam重算历史路由。 |
| `model.py:522-566` | 若按推荐方案F归R，将其放到`model.session_codeword_router`等非alignment前缀；R优化器会收集它，T冻结它但保留输入梯度。若把F放alignment下，需要显式扩展T分组，形成另一方案，不可两种归属混写。 |
| runner的`PhaseAudit`与checkpoint | 扩展参数身份/冻结审计、严格初态验证和新前缀允许清单；保存F及其优化器状态/loop轮数/概率温度/输入目录版本。新构造不得改变共有参数或原随机流。 |
| 评估、tokenizer mapping | 若维持canonical输出，可沿用原全目录mapping刷新；必须验证branch开关/数据顺序不改变目录，不能把session内路由索引写回hard_codes。 |

至少保留：05原版；同64维距离/路由头但不跨商品的item-only控制；one-full session；grid；loop；grid＋loop。首轮GPU可先只比较原版与首选，但原型/设计阶段应列明消融意图。item-only控制用于区分“增加05语义旁路/参数”与“session上下文”的收益；one-full控制用于区分轴分解本身与任何session交互。预算有限时由预设判据选结构，不能因grid/loop听起来新就预定它胜出。

grid的L轴和M轴有明确身份；它与`C[M,K,64]`上的“码本×码字”grid不同。后者相同k跨book没有天然语义对齐。one-full只有一个MHA，grid常含两个MHA，参数未必相等，应报告实际参数/投影次数。以L=50、M=4为例，每head attention pair为full40000、grid10800，但不含投影、路由K维点积和框架开销，不能据此报实测加速。loop共享参数不共享计算，R轮需递归读`Z^(r)`，不能每轮重新对同一Z0计算再声称发生迭代。

## 7. 目录合法性、所谓trie与@50的真实支持

当前native confidence路径没有检索到active trie/prefix constraint：`generate_w_decoding_graph=True`虽由runner设置，但模型/beam当前路径并不读取此标志。不能在方案里写“已有trie保证合法”。

- `model.py:1636-1673` → `beam.py:606-637` → `iterative_mask_decode`；按split取beam（`beam.py:155-175`），`TOP_K_FINAL=min(config.top_k_final,n_return_sequences)`。
- 默认simple分支在完整SID生成后，通过`tokenizer.codebooks_to_item_id`过滤和去重（`beam.py:476-519`）。不足K会复制最后一条合法序列；完全没有合法结果还有回退首条的路径。因此不是硬保证K个合法且唯一结果。
- 反向`tokens2item`是单值覆盖映射（`tokenizer.py:722-728`），碰撞SID只留下其中一个ID；`cb2items`另保留所有成员（`:950-981`）。当前主评估比较SID，不是唯一真实商品ID；不能把合法目录查询当作消除了碰撞。
- evaluator允许配置遍历任意k（`evaluator.py:272-278`），trainer按`max(config.topk)`请求生成（`trainer.py:1199-1206`）。但当前正式默认`topk=[5,10]`，`top_k_final=10`，val beam32/test256（`configs/shared/diff_formal_common.yaml:77-92`）。**只把topk加入50仍可能只返回10条，@50将失去其应有含义。**
- 增加@50必须同时设`top_k_final≥50`、生成请求≥50、有效beam≥50，并记录每用户前50的合法/唯一覆盖及padding重复；val32不能直接提供50个distinct候选。推荐所有模型采用同一明确beam（例如256）评估@50，仍保留@5/@10；如另改搜索宽度，须单独列因子。
- weighted固定为`0.8*NDCG@10+0.2*HR@10`（`evaluator.py:210-211`），不因加@50自动改变。选模规则和新增指标应提前冻结，不按结果改k或回看test选方案。

## 8. 历史结果的角色与边界

05长期报告`experiments/05_crosspse_joint/long_mixer_factorial/REPORT.md:67-90`：固定R35主DID的HR@10为−0.000223583598、NDCG@10为+0.000141105655、weighted为+0.000068167822，混合且很小；本方案不能假设全局Full已获稳定收益。

07联合报告`experiments/07_item_local_attention/joint_alternating/REPORT.md:7-15`明确局部层位于独立256维表后、item_mlp前，且全局Mixer关闭；它不是本方案的基底。新增两seed的`multiseed_20261021/report/FINDINGS.md`和`REPORT.md`表明2025增益、2026基本持平/略降，仅验证集，没有新test。它支持“不要把加attention等同稳定收益”的审计要求，不支持把07模块直接更名为05 session扩展。

## 9. 现有CPU环境及本次验证边界

- 已有本地解释器：`C:/Users/nancy/Documents/experiment1/.runtime/continuous_env/Scripts/python.exe`，本次只读检查确认文件存在；`pyvenv.cfg`为Python3.12.14，非本次新建环境。
- `experiments/07_item_local_attention/joint_alternating/cpu_acceptance_20261020/ACCEPTANCE.json`记录该解释器实际通过20项CPU生产模型/冻结/恢复相关检查。`experiments/05_crosspse_joint/long_mixer_factorial/EXECUTION_NOTES.md:36`记录其已有Torch2.6/Numpy2.2.6；版本是记录证据，本次审计没有import Torch或调用CUDA。
- 该本地环境缺完整Pipeline依赖，见`long_mixer_factorial/RESTORE_GUIDE.md:107-110`。可做小张量原型和独立模型模块验收，不能据此承诺完整真实Pipeline CPU恢复。
- 记录中的既有远端完整环境为`/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python`。本次未登录、调用或修改远端环境，不把历史路径当作当前可用性保证。
- 既有相关测试可作为未来集成门：`tests/test_cross_pse_module.py`、`test_joint_cross_pse.py`、`test_diff_crosspse_diger.py`、`test_diger_sid_consistency.py`、`test_diger_mixer_long.py`。这里只读列出，没有再次执行。

本审计支持“接口与训练协议可闭合”的判断；不支持推荐效果、线上延迟、GPU预算或正式实现已经通过的结论。后续CPU合成原型应至少验证共同起点、两条历史引起不同routing、无target泄漏、PAD隔离、R/T冻结与回C梯度、迭代状态传递以及canonical目录恒定性；其通过也不等于真实数据收益已证实。
