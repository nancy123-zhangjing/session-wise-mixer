# 08：已退役的单目标历史侧码字交互

> 2026-10-26：用户确认session-wise指OneRec请求级列表生成，本实验不属于该任务。专属代码、配置、模型及备份已删除；下文仅为历史记录，不能再执行或据此恢复。见[退役说明](RETIREMENT_20261026.md)及[新方案](../../docs/research/onerec_listwise/PLAN_20261026.md)。

2026-10-25追加：按原验证规则选出的三组best（R15/R30/R35）已在完整测试集统一完成HR/NDCG@5/10/20/50，beam256，无重训或SID刷新。axial仅HR@5略高于baseline，其余七项低于baseline。三组测试、逐用户候选、本地独立复算和GPU释放已验收，见[正式测试报告](test_best_20261025/REPORT.md)。下方“测试锁定”为10月24日训练交付时的历史边界，本次有用户单独授权。

日期：2026-10-24。三组正式35R/60T、完整验证集beam256已结束，45阶段/24验证节点及六份保存状态的CPU控制审计通过；测试集保持锁定。全部5次GPU分配已核实释放，保守观测占用7.949226 GPU小时，实际结算未核实。本地11分卷、1995成员和14份输入字节、两个独立Git仓库、六份实际网络/优化器/RNG及CPU历史前向恢复均已通过。最终文档和包后收据与基础包通过外置`FINAL_DELIVERY_BINDING.json`配套验收，位置见[交接](HANDOFF.md)及[执行记录](execution_20261024/EXECUTION_STATE.json)。

| 固定R35 | HR@10 | NDCG@10 |
|---|---:|---:|
| baseline | 0.099048 | 0.059208 |
| no_context | 0.098511 | 0.059202 |
| axial | 0.099674 | 0.059860 |

axial的NDCG@10相对两组约+1.10%/+1.11%，但HR@5相对baseline约−1.30%；各自按验证选best时，baseline R15仍优于axial R35。暂时只把它当跨seed复验的小信号，不称稳定收益。见[完整报告](REPORT.md)、[交接与恢复](HANDOFF.md)、[微信草稿](wechat_draft.md)。

下面保留启动前的设计与复审记录；其中“尚未上传/验收”等是历史边界，不是当前进度。本轮冻结范围见[执行方案](EXECUTION_PLAN_20261024.md)，启动验收包含Linux107项检查和真实Beauty三臂H100 gate。

## 本轮做什么

复用当前DiffGRM、05共享起点和交替联合训练。在历史商品的4个码位合成为一个item向量之前，让**选中的64维物理码字**先在同商品内、再沿历史交互。用零初始化的64→256线性层生成逐码位残差，加到原推荐器SID embedding，再走原item_mlp、历史Encoder和离散Decoder。

```text
原历史分配 A ─────────────→ A@W [B,L,4,256] ───────────────┐
       └→ A@Cbar [B,L,4,64]                                │
          → 位置/码位标记 → 码位轴MHA → 因果时间轴MHA → FFN │
          → LN → Linear_zero(64→256) → 残差 ──────────────┤
                                                         ↓
                   相加 → 原item_mlp → 历史Encoder → DiffGRM
```

这次采用直接残差，替代[旧设计](../../docs/research/session_wise_codebook/DESIGN.md)的`0.1*(PH-P0)@W`概率差分读出；两者不等价，旧CPU原型与计时不能当作新实现的验证。新模块没有改变同一目录版本内的目标SID身份，不是上下文相关重新量化，也不减少原生解码步数。

选型依据、论文阅读位置、参数来源和风险见[本次调研](../../docs/research/session_wise_codebook/IMPLEMENTATION_REVIEW_20261024.md)。DiffGRM提供现有骨干；Axial Attention提供分轴连接机制；ActionPiece/Pctx提供上下文编码的动机及边界。均不证明本方法会涨点。

## 首版固定设置

| 设置 | 值与理由 |
|---|---|
| 码本 / 推荐embedding | 继承`Cbar[4,256,64]`、`W[4,256,256]` |
| 新模块 | 一轮、两个独立MHA、4头、FFN128、dropout0、Pre-LN |
| 新增参数 | 默认设置70,464个；no_context与axial完全相同 |
| 残差读出 | LayerNorm后Linear64→256，权重与bias初始为0，固定scale1 |
| 历史 | 最多50；训练保留早期前缀扩样`t=2..min(Luser−3,50)`；val/test取各自目标前最近50，不改划分 |
| R优化 | 推荐器、W、新模块；AdamW，LR1e-3，500步warmup后cosine，weight decay0 |
| T优化 | 物理码本LR1e-5、全局Mixer LR1e-4；新模块/W冻结，但保留对输入的梯度 |
| 节奏 | 先5R，再10次(T+3R)：35R/3640次R更新、60次T更新 |
| 验证 | 同一完整验证集、beam256、SID-level HR/NDCG@5/10；测试入口锁定 |

头数、FFN128、dropout0、scale1是预先固定的起步设置，没有证明最优。scale1依靠零读出保持初态一致，不保证训练后的修正幅度小；开启残差/原embedding范数比等诊断。只在读出开始学习后检查核心的非零梯度；no_context时间轴的Q/K梯度也可合法为0。

## 三组新对照

| 臂 | 新分支 | 主要问题 |
|---|---|---|
| baseline | 禁用 | 整体是否优于原模型 |
| no_context | 同样两套MHA，时间轴仅能看自身 | 控制额外参数、码字旁路和计算深度 |
| axial | 仅开放对此前历史的可见性 | 跨历史信息是否带来额外收益 |

全局Full Mixer在三组统一开启，这只是保持05血缘，不是已有证据证明它更好。三组从同一E0和pre-05推荐器开始，完整共同state也包含新增模块；baseline仅在R优化器中剔除该模块。没有接05/07最终权重，也不把05早期长期、共享起点短对照或07商品内交互当成本轮对照结果。

优先比较固定R35的NDCG@10，同时报告HR@10、碰撞与候选覆盖、实测额外时延。best权重沿用旧`0.8*NDCG@10+0.2*HR@10`封存，候选包含初始R0，单独报告，不替代固定预算比较；若R0被选中，不能声称训练提高了该选模指标。旧05验证beam32的数字不能直接作为这里的baseline。

同一快照内目标SID不随session变化，但每T后会统一刷新，三臂的SID可以逐渐不同。axial与no_context比较的是上下文分支及其经T反馈的总效应，不能当作固定Tokenizer上的纯表示效果。

若axial同时超过baseline和no_context且代价可接受，再补2025/2026 seed；只有no_context改善不能证明session交互有效。loop实现可配置、共享core，但首轮协议固定1轮。暂定的探索阈值及更强的后续对照见调研第7节，训练前再冻结判断口径。

## 代码与本地验证

- 引擎：[session_codeword.py](../../external/DiffGRM/genrec/models/DIFF_GRM/session_codeword.py)、[model.py](../../external/DiffGRM/genrec/models/DIFF_GRM/model.py)、[joint_cross_pse.py](../../external/DiffGRM/genrec/models/DIFF_GRM/joint_cross_pse.py)。默认关闭，不改变旧调用的架构。
- 独立入口：[run_session_codeword_joint.py](../../scripts/train/run_session_codeword_joint.py)。复用05实际阶段循环、优化器时钟、冻结审计与完整恢复，不修改历史runner。
- 配置：[session_codeword_seed2024.json](../../configs/08_session_codeword/session_codeword_seed2024.json)。GPU预算为未设置占位，不继承旧Goal的GPU授权。
- 验证入口：[verify_session_codeword_cpu.py](../../scripts/checks/verify_session_codeword_cpu.py)，[最新复审收据](review_20261024/CPU_ACCEPTANCE.json)绑定实际源文件SHA256及逐套原始日志；[初版收据](code_acceptance/CPU_ACCEPTANCE_20261024.json)保留，不冒充最新源码。Windows下旧Linux共享文件锁检查会跳过，不能称跨平台已验收。

只查看计划、不导入Torch、不启动训练：

```powershell
& ./.runtime/continuous_env/Scripts/python.exe -S -B scripts/train/run_session_codeword_joint.py --mode plan
```

重新检查本地CPU链路：

```powershell
& ./.runtime/continuous_env/Scripts/python.exe -X utf8 -B scripts/checks/verify_session_codeword_cpu.py --output experiments/08_session_codeword/review_20261024/CPU_ACCEPTANCE.json
```

已覆盖原型实际维度`[B,50,4,64]→[B,50,4,256]`、PAD/全PAD/因果及跨样本隔离、零起点/RNG、真实DiffGRM损失和BF16、R/T冻结与新分支单独回码本的梯度、无target生成、每请求只计算一次F、完整新增参数/优化器/调度/RNG恢复与运行属性篡改拒绝。码本cache在加载状态后失效；live获取也在FP32下执行。

复查还增加了真实context构造函数的CPU探针：真实F和CrossPSE重基、构造前配置注入、完整common发布、YAML/非Tensor配置漂移拒绝；Pipeline数据与大依赖仍为合成替代。无target且只提供历史ID时自动推导PAD mask，提供mask时校验形状并屏蔽0号PAD。构建及读取物理码本cache均隔离外层autocast。

恢复身份同时绑定实际读取的三份根YAML及稳定的解析后模型设置。验证复用trainer每batch的截止检查，触及进程deadline前120秒退出并保留已提交阶段；这是协作式进程限制，不能替代平台作业硬截止或实际分配时长总账。

当前检查使用合成目录和已有CPU环境；尚未验证真实Beauty整条Pipeline、新模块H100显存/吞吐、Linux并发锁、真实资产可用性或推荐质量。下一步是冻结根仓库与独立引擎版本、在独立服务器目录准备并校验资产，再进行H100短验收。CPU通过不等于可以跳过这些步骤。

根仓库不跟踪external引擎内容，后续同步必须同时包含两处变更；当前六份既有文档改动属于本轮开始前的工作区状态，本轮没有覆盖它们。
