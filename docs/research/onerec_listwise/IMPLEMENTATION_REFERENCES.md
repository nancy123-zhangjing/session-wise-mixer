# KuaiRec列表实验：具体设置和来源

更新：2026-10-28。对应 `configs/09_kuairec_listwise/base.json` 和 `scripts/listwise_diffgrm/`。以下把论文/作者代码、项目已有设置、本轮新设计分开。**有来源不等于该数值已经在KuaiRec上验证最优。** 真实资产、两轮H100验收和两份共享暖身已完成，四组正式训练正在运行，尚无最终对照结果；即时状态见[执行记录](../../../experiments/09_kuairec_listwise/EXECUTION.md)。

## 来源索引

- **D1：DiffGRM论文**：[§3.2 PSE、§3.3 OCN、§3.4 CPD、附录B Table 7](https://arxiv.org/html/2510.21805v1)。骨干、并行码位表示及噪声/解码思想的来源。
- **D2：作者代码**：[README Beauty复现命令](https://github.com/liuzhao09/DiffGRM#beauty)、[模型默认文件](https://github.com/liuzhao09/DiffGRM/blob/main/genrec/models/DIFF_GRM/config.yaml)。本地对应 `external/DiffGRM/README.md:72`、`genrec/models/DIFF_GRM/config.yaml:10`。复现命令覆盖默认值，优先级见 `genrec/utils.py:325`。
- **P1：05共享起点长期对照**：[冻结配方](../../../configs/05_crosspse_joint/long/mixer_long_seed2024.json)、[实际运行配置](../../../experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/formal/resolved_config_train_mixer_joint.json)、`scripts/train/run_diger_mixer_long.py`。这里指已完成的共享起点长期四组，不是05早期四组或后来短对照。
- **P2：项目真实实现**：`external/DiffGRM/genrec/models/DIFF_GRM/model.py`、`joint_cross_pse.py`、`cross_pse.py`。列表入口直接加载这些类，不重新实现另一套Transformer/PSE再声称复用。
- **O1：OneRec**：[§3.2 Session-wise List Generation](https://arxiv.org/html/2502.18965v1)。参考其按请求生成多件及前缀条件思想；本项目的人工会话、DiffGRM内层及四码编码不是OneRec原配方。
- **B1：BGE-M3**：[官方模型卡](https://huggingface.co/BAAI/bge-m3)。用其多语言dense 1024维表示处理KuaiRec文本。模型版本由40位revision显式锁定，不能只写随时间变化的`main`。
- **K1：本次已验收数据协议**：[数据报告](kuairec_preparation/REPORT.md)、[固定数据配置](../../../configs/kuairec_listwise_v1.json)。数据量、切分、五件目标及历史范围均来自这份协议，不从Beauty迁移。
- **N1：本轮研究设计**：[实施方案](MODEL_EXPERIMENT_PLAN_20261027.md)。明确标注的工程选择和待验证假设，不冒充论文既有结论。

## 逐项配置依据

| 设置 | 实际值/实现 | 来源及理由 |
|---|---|---|
| Encoder / Decoder层数 | 1 / 4 | D1 Table 7、D2 Beauty CLI和P1实际配置一致。作者回退默认为2/2，没有找到1/4与2/2直接层数消融，不宣称1/4最优。|
| hidden、head、FFN | 256、4、1024 | D1 Beauty、D2和P1。不是模型默认FFN512。四Decoder层与四码位无一一对应关系。|
| dropout | embedding、attention、residual均0.1 | P1/D2；保留现有正则，不单独调Mixer组。|
| norm与激活 | LayerNorm，epsilon=1e-5；骨干FFN GELU，item_mlp ReLU | P2真实实现。不是论文额外提出的列表组件。|
| SID embedding及输出 | 共享embedding点积输出；每码位256类，embedding宽256 | P2；PSE中心的64维与推荐器256维SID embedding属于两种参数。|
| PSE | 4套×256中心×64维，OPQ4,IVF1,PQ4x8；保留IVF1残差 | D1 PSE和P2现有tokenizer。旋转、粗中心冻结；T阶段更新中心。|
| 中文文本向量 | BGE-M3 dense1024；CLS、L2归一化 | B1官方dense用法；替换英文Beauty内容编码器是新数据适配。固定本次使用的revision和文本hash。|
| 文本长度 | max_length=512 | N1资源/短内容工程选择，**不是BGE-M3能力上限**。编码入口记录实际截断件数，后续若改长度四组必须共用新资产。|
| 文本模板 | 既有合并文本逐字使用，不添加行为统计或人工描述 | K1。缺文本保留空串及原标记，不剔除商品或编造内容。|
| PCA | 256维，whiten=True，然后L2；全静态目录拟合 | P2历史tokenizer流程；solver=full是本轮确定性实现选择。完整静态目录可用属于K1离线假设，不使用留出行为。|
| 历史 / 目标 | 最近最多50件；目标5件 | K1。五件是已确认的人工会话代理任务；不是还原真实线上请求。|
| 位置容量 | 两路线共用54个历史位置、5个商品位置向量 | N1：H在原50历史后增加最多4个已生成目标前缀；J/H共同初始化。新增商品位置打破20个全MASK码位中重复码位的对称性。|
| J路线 | 20位共同去噪；仍四套共享码本/输出表 | N1，将D1扩成列表，不能将n_digit设20。|
| H路线 | 外层逐商品、内层原四码DiffGRM；训练真前缀、推理生成前缀 | O1提供前缀组织思想；具体混合结构是N1，五件共享模型参数。|
| OCN排序 | guided least、MSP、静态探测、不逐步刷新 | P1/P2。teacher探测保留既有raw-H cross-KV分支，学生/正式推理使用投影KV；这是原代码语义，不称两者完全同一前向。|
| H噪声视图 | 每商品遮1、2、3、4位，4视图 | P2。列表损失平均五件而非相加。|
| J噪声视图 | 从1–19抽3个不同遮盖深度，再加20位全MASK | N1。平均监督50码位/列表，与H相同；不代表FLOPs相等，也不是作者原版已验证的噪声方案。|
| 损失归一化 | 每视图先平均masked CE，再平均视图、商品、列表 | P1的per_view原则，N1适配列表；尾batch按实际列表数归一化。|
| label smoothing | 0.2 | P1和D2 README。D1 Table 7 Beauty为0.1，不能写成完全照搬论文超参数。|
| Full Mixer | 全局full attention，4 heads、dropout0，残差固定1，输出投影零初始化 | P1/P2。初始有效码本严格为E0；不新增07局部注意力或已退役08模块。|
| R优化器 | AdamW，lr1e-3、wd0、betas(.9,.999)、eps1e-8 | P1；不是D1 Beauty的lr0.01。关闭Mixer组也交替更新码本。|
| R学习率 | 500更新warmup，随后cosine，时钟只随R更新 | P1运行器。论文Beauty warmup10000是另一个配方。|
| T优化器 | 码本lr1e-5、Mixer lr1e-4、wd1e-4；同Adam参数 | P1。R/T优化器持久保留动量，不每阶段重建。|
| T目标 | 推荐1、重构1、balance.01、anchor.01，SIA0 | 权重来自P1。本轮在整个有效批的唯一目标商品上统一算辅助项一次，H不重复加五份；这是N1的归一化适配，避免改变微批大小时balance/重构目标也变化。|
| ST分配 | assignment温度.1、softmax温度1；Gumbel1→.2 cosine | P1/P2。历史/五件目标各分配一次；H前缀复用同一target ST；masked soft-label路径保留梯度。|
| 梯度裁剪 | norm1，非有限梯度报错 | P1。真实BF16梯度和参数更新已通过两轮H100验收，验收不是正式质量结论。|
| 确定性执行 | deterministic_algorithms=True；CUDA启动前CUBLAS_WORKSPACE_CONFIG=:4096:8；TF32关闭 | N1执行验收设置。四组正式训练、恢复验收及测速一致，避免只在验收时启用确定性却用另一模式报告正式吞吐；H100实际算子已通过验收，不静默忽略非确定性错误。|
| 有效batch / microbatch | 1024条列表 / 正式256，累积4；base初值16 | N1。沿用P1有效batch数量尺度，但单位变为列表，不称训练信号量不变。H100 R2效率验收后将微批从16调至256、推理batch从4调至32，四组共用冻结runtime_efficiency_r2.json；base.json保留初值。|
| R/T预算 | 5R共享暖身，再10×(40T更新+3R epoch) | P1交替比例+N1数据量重算。709443条训练列表→每R epoch693更新，合计24255R/400T。旧60T不能机械照搬；35R未证明从零收敛。|
| 共起点 | 各架构5R暖身后复制模型、PSE、Adam、scheduler、RNG | P1控制原则；本次无Beauty权重。两架构内±Mixer成对一致，不要求J/H暖身后权重一致。|
| 推理 | confidence逐码、全列表beam32；同val/test | P2 CPD原则+N1列表搜索预算，**不是论文Beauty beam256的复现**。J/H最多20个逻辑填码阶段，不称一步生成。|
| 碰撞/合法性 | 任意已填码坐标匹配全目录；重构误差、原ID顺序消歧；列表内互异 | N1必要item级适配。重复SID若bucket容量足够可对应不同商品；不使用标签倒推商品，不过滤历史已看商品。|
| 选模 | 完整validation的单份列表item-level NDCG@5；平局取较早 | N1，符合5个相关目标。R5及每轮3R结束验证，测试不参与选择。|
| @10/20/50 | 最多32份完整五件列表按分数排序，按列表名次→位置展开，首次item去重 | N1，明确是多列表候选聚合指标，不当作单份五件列表质量。|
| seeds | 首轮2024，后续同协议2025/2026 | P1/P项目习惯+N1统计设计；代码允许seed显式覆盖，不自动启动三seed。|

## 几处实现边界

1. 原单商品入口与已完成模型不修改。新增入口通过源文件加载真实骨干和PSE类；每个checkpoint绑定新代码、引擎源码、配置、数据及资产hash。
2. 推理缓存仅在本次生成调用内存在。J复用同一历史编码；H换商品前缀后重算，不能把前一个前缀的Encoder结果继续用在新前缀上。
3. 以上有三类依据：论文支持设计思想、历史代码支持沿用设置、任务约束支持新适配。后两类尤其不能改写成“论文证明这样最好”。
4. GPU时间与显存已有H100验收、完整暖身及资源短测记录；正式四组仍在运行，收敛情况和推荐收益尚待完整验证与统一测试。CPU微型更新、资源短测和暖身均不能替代正式对照。

## 5R暖身的依据边界（2026-10-27补查）

这里的5R指固定PSE、关闭Mixer，先训练推荐器5个epoch，再复制完整状态给同架构的两个分支；与学习率前500次更新逐渐增大的warmup是两件事。

- 项目来源：P1冻结配置的`decoder_warmup_epochs=5`、`cycles=10`、`decoder_epochs_per_cycle=3`构成35R安排。本轮保留这个阶段划分，但KuaiRec推荐器从随机初始化开始；旧05使用已有Beauty推荐器权重。因此沿用5这个数字不等于它已在当前数据上验证合适。
- 相近论文：[ETEGRec §3.4、§4.1.5](https://arxiv.org/html/2409.05546v3#S3.SS4)支持为稳定性而交替优化tokenizer与推荐器。其循环从1个tokenizer epoch开始，再训练推荐器C−1个epoch，C在{2,4}中选择；不是“推荐器先单独训练5轮”。其交替训练消融也不能证明本项目5R暖身有效。
- [DIGER §4.3](https://arxiv.org/html/2601.19711v3#S4.SS3)支持预训练量化器初始化后联合优化；作者修复版[Beauty配置](https://github.com/junchen-fu/DIGER/blob/ec0a9b51d02272be1935e1a6438cdc30bf9fadc1/config/beauty_gradient_fix.yaml)设`freeze_id_epochs=0`。它不是本轮5R后分块交替安排的直接来源。
- [DiffGRM附录B](https://arxiv.org/html/2510.21805v1#A2)的10000-step warmup是学习率安排，不是固定码本、推荐器单独训练若干轮。

本轮5R的设计动机是先建立推荐反馈，再开放码本更新，并为±Mixer保留共同起点。这是待验证的稳定性假设；公平对照本身只要求配对初始化与协议一致，并不强制5R。当前没有0R/5R对照，不能声称暖身必要、5轮最优或R5已经收敛。此次四组仅回答既定训练协议下Full Mixer是否带来收益；不因本次文献补查中途改配方或自动新增消融。
