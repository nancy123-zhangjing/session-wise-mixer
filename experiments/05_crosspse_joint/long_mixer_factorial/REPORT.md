# 阶段05共享起点带噪Full Mixer长期四组：结果报告

**四组完整训练、四次正式测试、训练控制、八份实际权重审计、图表及GPU释放均通过。完整备份已下载至本地，全量字节与源码Git恢复通过，并与远程同一归档的八状态真实CPU恢复完成绑定。** 更新于2026-10-20；逐项交付验收见[FINAL_DELIVERY_AUDIT.json](FINAL_DELIVERY_AUDIT.json)。

## 1. 研究问题与当前结论

完整训练审计已通过：四组均35R/3640次推荐器更新，两Joint各60T/1920次实际加噪；两Frozen的完整35轮轨迹、验证与最终模型一致。固定R35的ΔMixer在HR@5、NDCG@5、NDCG@10和weighted为正，HR@10为负；weighted增量仅`+0.000068167822309`，HR@10对应总体净少5次SID命中。这是本协议预算末期部分指标的微弱正向信号，尚不能称为一致或稳定推荐收益。

四组各自按验证weighted选定best后的一次正式测试，Full Mixer Joint相对无Mixer Joint的weighted高`0.000367852998421`、NDCG@10高`0.000571608095622`、HR@5高`0.000670750793722`，HR@10低`0.000447167195815`。同样呈部分指标受益、Top10命中略少的混合结果；两个best分别是R15/T24与R10/T12，不能将这组测试差当作固定R35差中之差或同训练预算的因果量。

各自最佳验证weighted分别为Frozen两组`0.052176661839059`、无Mixer Joint `0.051967036679846`、Mixer Joint `0.051890160699695`；两个Joint各自最高验证weighted都没有超过Frozen。该描述不替代固定R35主比较或独立test。最佳点分别R15/T0、R10/T12、R15/T0、R15/T24，两个Joint的选中点确实经历过T更新，未出现本轮最佳点仍为起点的情况。

直接来源为[完整计算AUDIT.json](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_audit_r1/AUDIT.json)，生成时间`2026-10-20T07:41:10.806283+00:00`（北京时间15:41:10），本地字节SHA256：`f7aa255e9716c4be60641c689504e059b63fbd38c410fd12ce550b541a9c788e`，其`tests_checked=true`、`checkpoints_checked=true`。四组固定R35数值与较早保留的[训练审计](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/TRAINING_AUDIT_FINAL_R1.json)完全一致。本文从收据和四份实际TEST_RESULT重算，未再次本地加载八个大权重。

计算审计的状态仍明确为`FORMAL_ARTIFACT_AUDIT_PASS_NOT_DELIVERY`；它不负责资源/完整备份恢复，故其相关布尔值保持false。资源释放由另一个[最终资源审计](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/RESOURCE_RELEASE_FINAL.json)独立证明。远程同一归档的字节与模型实际恢复已另行通过，本地完整备份仍在传输/验收，不能把任一单项PASS改写为Goal完成。

综合五项指标，本轮观察到Full的部分排序指标增益和HR@10负向差值并存；四组可信对照是验收标准，实验没有预设Mixer必须涨点。

## 2. 固定研究定义与起点

| 项目 | 实际固定定义 | 已取得证据 |
|---|---|---|
| campaign | `goal_20261020_long_mixer_v1` | 实际identity、独立AUDIT；最终本地manifest与全部成员字节核验通过 |
| 训练根仓库 | `69682698c15ba796b98e6329c950061a2f68a05b` | 实际HEAD/干净状态、执行源码字节哈希 |
| 训练引擎 | `1c879252f6395d6e64462511823c3929b9c42c00` | 实际HEAD/源码哈希；浅历史边界另记，不称完整Git历史 |
| 分析与交付版本 | `4b40c90b56490868e5972e75bafbf8e50ca8a3de`，V3R2，`delivery/analysis_tools_v3r2` | `ANALYSIS_SOURCE_LOCK_R3.json`与V3R2文件锁；训练审计的auditor SHA与文件锁相同 |
| 推荐器起点 | `AmazonReviews2014_Aug-28-2026_19-01-02/pytorch_model.bin` | 14件输入锁、全部离线tensor载入检查及共同初态 |
| E₀重基 | 四组同一已学习有效E₀、相同item顺序/OPQ/slots/anchor；新Full Mixer输出投影0、固定残差1 | canonical E₀文件及数组SHA、共同初态实际tensor、四臂environment与初始指标 |
| 数据与结构 | 原Beauty LOO滑窗，users22363/items12101/interactions198502，train105668/val22363/test22363；4×256、码字64、推荐器embedding256 | 输入锁和完整训练审计；实际入口运行时split长度断言；四组训练与evaluate resolved config一致，关闭07 |

这是新的共享起点暖启动实验：从Aug28重新执行前5轮，而不是从旧Sep-01第5轮权重追加30轮；E₀包含既有Cross预训练所学功能，不检验“预训练阶段完全没有交互”的反事实。[长期固定配置](C:/Users/nancy/Documents/experiment1/configs/05_crosspse_joint/long/mixer_long_seed2024.json)与[历史证据背景](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/PRIOR_EVIDENCE_CONTEXT.md)说明设计来源，完整计算审计证明本轮实际执行与之相符。

| 组 | Mixer | 允许的Tokenizer更新 | 推荐器预算 | Tokenizer预算 |
|---|---|---|---|---|
| no_mixer_frozen | 禁用 | 无 | 35个完整R epoch/3640次更新 | 0 |
| no_mixer_joint | 禁用 | 基础码本 | 同上 | 10×96 batch/60次更新 |
| mixer_frozen | Full启用但冻结 | 无 | 同上 | 0 |
| mixer_joint | Full启用 | 基础码本及Mixer | 同上 | 10×96 batch/60次更新 |

Joint节奏为5R，再10×(96个T batch、累积16、6次T更新＋3R)，共45个阶段；Frozen共35个R阶段。每个R epoch实际1652 batch、104次更新，最后4个batch按真实累积窗口归一化。四组已完整跑满；跨阶段保留优化器和调度状态，不按性能早停。共同前5轮的完整模型分别执行同一训练路径，四臂loss相同；后续按相同逻辑epoch/周期配对数据顺序与随机数。

完整参数已逐项读取[固定协议](C:/Users/nancy/Documents/experiment1/configs/05_crosspse_joint/long/mixer_long_seed2024.json)、四组实际resolved配置及[实际evaluate配置](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/formal/resolved_config_evaluate_all.json)，并结合完整checkpoint审计核对；不是按历史默认值推定。

| 类别 | 本轮实际设置 |
|---|---|
| 样本构造 | Beauty原始编号、leave-one-out、train_sliding=true，min_hist_len=2、max_hist_len=50；不采用06新前缀协议 |
| 编码与推荐器 | SID4×256、码字64；推荐器embedding256，encoder1层、decoder4层、4heads、inner1024，dropout/embedding/attention/residual dropout均0.1；共享embedding及decoder输出embedding；LayerNorm eps1e-5 |
| OCN与解码路径 | guided、least、guided_refresh_each_step=false，diffusion loss按per_view；confidence模式，原生4步；不启用ablate_decode和item_local_attention |
| 批次与累积 | train64、eval8、梯度累积16；名义基础有效batch1024；每R1652 batch/104次更新，尾部4个microbatch按真实窗口归一化 |
| 推荐器优化器 | AdamW，峰值LR1e-3、weight_decay0；前500次R更新warmup，之后余弦衰减到累计3640次R更新；T不推进R调度 |
| Tokenizer优化器 | AdamW；基础码本LR1e-5，启用Mixer时其分组LR1e-4，weight_decay1e-4；累计60次更新跨10周期保留状态 |
| Adam公共参数 | betas=(0.9,0.999)，eps=1e-8；协议显式绑定，完整best/last实际param_groups、step与动量已审计 |
| 损失与裁剪 | label_smoothing0.2、max_grad_norm1；推荐反馈权重1、重构权重1、balance0.01、anchor0.01；Tokenizer和推荐器两侧SIA均0 |
| 温度与噪声 | 分配距离温度0.1、balance温度0.1，反向softmax恒1；Joint Gumbel跨累计60次更新余弦1.0→0.2，每phase96 batch/6更新；history/target两次独立分配；推理和SID刷新确定性 |
| Full Mixer | 4heads、full attention、dropout0，初始attention输出投影0；残差系数固定1、不可学习；基础码本可训练；每个T阶段后刷新全目录SID与映射 |
| 精度与模式 | Tokenizer FP32、推荐器训练/评估显式CUDA BF16 autocast；TF32关闭。Accelerator的mixed_precision="no"，实际BF16由入口显式控制；保留OCN teacher autocast缓存及混合train/eval修复 |
| 随机数 | seed2024；R逻辑epoch基数2024101300，数据基数2024101400，T阶段基数2024101201，独立Gumbel基数2024101200；Gumbel采样不消耗外部推荐器随机流 |
| 验证与测试 | SID-level，不展开到item；val beam_act/max=32/32，test=256/256，最终top10；初始及R5/10/15/20/25/30/35验证；max_eval_batches=None |
| 单次执行环境 | 单节点单H100 80GB HBM3、单进程、use_ddp=false；Python3.10.20、Torch2.6.0+cu118、CUDA runtime11.8、NumPy2.2.6；复用既有grm-exp环境 |

版本证据：[分析源锁](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/ANALYSIS_SOURCE_LOCK_R3.json)、[V3R2文件锁](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/ANALYSIS_TOOLING_V3R2.json)、[实际evaluate环境](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/formal/environment_evaluate_all.json)。训练审计使用的auditor SHA为`51d2b6f31e380f451a5e08c810eb23c48556fb180e1b55ad3a4a0a006bb71e6f`，与V3R2锁一致；训练源码和独立分析版本始终分开。

实际phase循环由本轮入口驱动。通用resolved配置中的`epochs=45`、`early_stop_after_warmup=true`属于沿用的trainer字段；本轮不调用其fit早停循环，真实完成量以四份progress和完整last的35R/3640次更新为准。Mixer开关也在恢复每臂时由`configure(model, arm)`设置，不能把初始化resolved中的单个`crosspse_mixer_enabled=false`误读成Full组训练时被禁用；两个Joint实际Tokenizer Adam状态分别为1个和9个参数张量，Full更新干预另有直接证据。

## 3. 主要结果：固定R35验证，统一训练预算

**本表是主分析。** 四臂完整R35验证，SID-level、val beam32；由完整训练审计的`/summaries/<arm>/final_validation`与`/fixed_epoch35_validation/<arm>`相互核对。下表展示9位小数，差分使用未四舍五入原值计算；不是各臂最高点、测试或旧短实验结果。

| 组 | R epoch/更新 | T更新/真实噪声调用 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | weighted |
|---|---|---|---|---|---|---|---|
| no_mixer_frozen | 35/3640 | 0/0 | 0.059696821 | 0.064660377 | 0.045212378 | 0.046889020 | 0.050443292 |
| no_mixer_joint | 35/3640 | 60/1920 | 0.060143988 | 0.064928677 | 0.045255460 | 0.046869679 | 0.050481479 |
| mixer_frozen | 35/3640 | 0/0 | 0.059696821 | 0.064660377 | 0.045212378 | 0.046889020 | 0.050443292 |
| mixer_joint | 35/3640 | 60/1920 | 0.060725305 | 0.064705093 | 0.045664268 | 0.047010785 | 0.050549647 |

每项指标从未四舍五入的原值独立计算：

`ΔMixer = (mixer_joint − mixer_frozen) − (no_mixer_joint − no_mixer_frozen)`

| 固定R35对比 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | weighted |
|---|---|---|---|---|---|
| no_mixer_joint−no_mixer_frozen | +0.000447167195815 | +0.000268300317489 | +0.000043081513875 | -0.000019341194155 | +0.000038187093781 |
| mixer_joint−mixer_frozen | +0.001028484550373 | +0.000044716719581 | +0.000451889859752 | +0.000121764461219 | +0.000106354916090 |
| 差中之差ΔMixer | +0.000581317354559 | -0.000223583597907 | +0.000408808345877 | +0.000141105655374 | +0.000068167822309 |

三个差值已从收据原始浮点值独立重算，ΔMixer与收据`/Mixer_increment`一致。因为两个Frozen完全相同，本轮差中之差数值恰好等于`mixer_joint−no_mixer_joint`；保留两个Frozen仍是必要控制。按22363个验证目标折算，HR@5/HR@10的总体命中净差分别为无Mixer Joint对Frozen `+10/+6`、Mixer Joint对Frozen `+23/+1`、Full增量 `+13/−5`，仅为总体计数差，未进行逐样本配对检验或显著性判断。

训练内解释：无Mixer Joint在HR上小幅改善但NDCG@10略低于Frozen；Mixer Joint在这五项固定R35指标都略高于自身Frozen，但相对无Mixer Joint的Top10覆盖更少、Top5及NDCG较好。因此不能只依据weighted为正就写“Full全面优于无Mixer”。所有组的R35 weighted均低于各自最佳验证值，说明预算末期点与最佳模型评价回答不同问题；这里没有证据把后期下降唯一归因为过拟合、噪声或编码漂移。

结果来源是`formal/<arm>/TRAIN_COMPLETE.json`内R35 validation、与其相同的`progress.json`及实际`last.pt`。已下载完整训练审计将这些进度、checkpoint字节与实际模型tensor绑定，并独立重算上述五指标和差分；最终CSV/图表另与原始收据SHA绑定。

## 4. 补充结果：各自验证选定最佳点的一次正式测试

选模规则固定为验证weighted=`0.8*NDCG@10+0.2*HR@10`。候选仅初始0及R5/10/15/20/25/30/35；相等时保留最早达到的点。四组全部训练和控制通过后先封存选择，再按各自best做一次正式test，SID-level、beam256。测试不参与调参或重新择点。

| 组 | 选中R epoch | 该best已完成T更新 | 该点val weighted | test HR@5 | test HR@10 | test NDCG@5 | test NDCG@10 | test weighted |
|---|---|---|---|---|---|---|---|---|
| no_mixer_frozen | 15 | 0 | 0.052176661839059 | 0.052989313 | 0.073335420 | 0.037344353 | 0.043896811 | 0.049784534 |
| no_mixer_joint | 10 | 12 | 0.051967036679846 | 0.052273845 | 0.074676922 | 0.036814325 | 0.044009527 | 0.050143007 |
| mixer_frozen | 15 | 0 | 0.052176661839059 | 0.052989313 | 0.073335420 | 0.037344353 | 0.043896811 | 0.049784534 |
| mixer_joint | 15 | 24 | 0.051890160699695 | 0.052944596 | 0.074229755 | 0.037724711 | 0.044581136 | 0.050510860 |

选择值来自`/summaries/<arm>/{best_epoch,best_tokenizer_updates,best_validation}`，且实际best checkpoint审计的R/T计数与之相同。各自最佳验证weighted上，两个Joint相对Frozen分别低`0.000209625159213`与`0.000286501139364`，Mixer Joint比无Mixer Joint低`0.000076875980151`。这是不同最佳R/T预算下的描述，不能把这个差值冒充固定预算DID或测试效果。

四份实际test及启动/选择/权重哈希链已逐项核对，均来自实际test split、不是验证样本测速；封存选择SHA为`a9f7a5a738fcbe70c3027733dd08315345a96529f466f65f678cb237a87d04cf`，四臂选择与训练完成收据一致。NF/NJ/MF/MJ耗时分别421.319/425.989/467.383/468.355秒。完整原值见[best_selected_test.csv](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_audit_r1/best_selected_test.csv)及各臂TEST_RESULT；两个Frozen的全部test指标也完全相同。

下表是**各自best选模程序的描述性测试差值**，使用未四舍五入原值计算；不是固定R35的主DID。

| 各自best的test对比 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | weighted |
|---|---|---|---|---|---|
| no_mixer_joint−no_mixer_frozen | -0.000715467513303 | +0.001341501587444 | -0.000530027948200 | +0.000112716293301 | +0.000358473357460 |
| mixer_joint−mixer_frozen | -0.000044716719581 | +0.000894334391629 | +0.000380358782127 | +0.000684324388922 | +0.000726326355881 |
| mixer_joint−no_mixer_joint | +0.000670750793722 | -0.000447167195815 | +0.000910386730327 | +0.000571608095622 | +0.000367852998421 |

HR@5/HR@10的总体命中净差分别为无Mixer Joint对Frozen `−16/+30`、Mixer Joint对Frozen `−1/+20`、Full Joint对无Mixer Joint `+15/−10`；这仍不是逐样本命中得失或显著性检验。两个Joint的test weighted均高于Frozen，而各自最高val weighted低于Frozen，说明验证选择与独立测试的排序没有完全一致；保持既定选择，不按测试结果反选checkpoint或改参数。

本轮两个Joint的best分别已有12/24次T更新；Frozen始终为0T。全部模型实际训练到35R后才使用此前保存的best测试，选择较早checkpoint不等于没有跑满预算。各臂best处于不同R/T预算，本表评价固定选模程序，不能代替同预算R35主分析。

四份正式test已与[COMPUTE_COMPLETE.json](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/formal/COMPUTE_COMPLETE.json)和完整计算审计相互核对。每臂在唯一评估Job中有一个完成结果和一个对应的LONG_EVAL记录；没有根据结果重新选择或新增测试。

### 测试覆盖范围的执行证据

测试覆盖没有只凭配置标注为“完整”。独立[TEST_COVERAGE_AUDIT.json](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/TEST_COVERAGE_AUDIT.json)核对了实际metadata、成功收据、冻结执行路径和原始完整日志：

| 证据环节 | 读到的实际行为及其含义 |
|---|---|
| 数据长度的运行时检查 | 冻结入口`context_for`先构建实际Pipeline，再逐split断言train105668、val22363、test22363；断言失败不会继续写environment或进入test。现有evaluate environment和成功test证明这条检查已越过，22363并非仅从计划抄入。 |
| 实际单进程检查 | 同一路径在Pipeline创建后断言`accelerator.num_processes==1`，并要求CUDA可见设备恰为1张H100。实际resolved另记`use_ddp=false`、`num_proc=1`；其中num_proc本身是数据处理配置，不独立证明GPU执行进程数。 |
| 完整DataLoader | `evaluate`按实际`pipeline.tokenized_datasets['test']`构造DataLoader，batch8、shuffle=false、num_workers=0，不传入subset/自定义sampler，默认drop_last=false，再交给单进程Accelerator准备；2796批进度已由实际日志确认，尾批3个样本由实际长度和batch推导。 |
| 截断分支实际关闭 | 实际evaluate resolved的`max_eval_batches=null`；冻结trainer将其转为`eval_limit=0`，只有`if eval_limit and batch_index>=eval_limit`才break。因此当前正常返回路径没有batch数截断。deadline异常或进程终止不会写出本轮成功TEST_RESULT。 |
| test beam确实选中256分支 | trainer每批设置`current_split='test'`，confidence生成路径调用冻结beam实现；该实现优先读取`vectorized_beam_search['test']`，实际act/max均256。全局默认32不会覆盖显式test分支；random/ablation分支未启用。 |
| 指标覆盖与正常完成 | trainer对DataLoader逐batch枚举，把每个batch的逐样本指标extend后整体求均值；TEST_RESULT只在evaluate正常返回后写入。四份结果均为实际test、`beam_profile_not_test=false`，与日志LONG_EVAL精确一致。 |

直接源码位置：[冻结入口的evaluate](C:/Users/nancy/Documents/experiment1/scripts/train/run_diger_mixer_long.py:238)、[运行时单进程与split长度断言](C:/Users/nancy/Documents/experiment1/scripts/train/run_diger_mixer_long.py:349)、[冻结引擎trainer的完整循环及截断条件](C:/Users/nancy/Documents/experiment1/.runtime/stage05_long/git_restore_recipe_probe_20261020T053859459026Z/engine/genrec/models/DIFF_GRM/trainer.py:1120)、[按实际split选beam](C:/Users/nancy/Documents/experiment1/.runtime/stage05_long/git_restore_recipe_probe_20261020T053859459026Z/engine/genrec/models/DIFF_GRM/beam.py:156)。本次对19份根仓库/引擎执行源字节逐项核对，全部等于本轮identity；evaluate resolved与四组训练resolved的核心架构、批次、模式参数亦一致。

原始[evaluate.log](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/evaluate_all_r1/evaluate.log)为876,623字节，SHA256=`3c6a3184571c6def97341b5dfb9075aab70cb9f33fa1749bcef3ca6945f83520`。四个顺序执行段均有2,798条tqdm记录：0..2796每个整数进度完整出现、单调不减，2796/2796终点重复一次是tqdm关闭输出；每段只有一个匹配该臂的完成LONG_EVAL。覆盖审计保存每段原始byte范围/分段SHA及完成JSON对应关系，最终LONG_DONE还与COMPUTE_COMPLETE精确相同。

由“实际Dataset长度/单进程断言＋无截断执行路径＋四段完整2796批日志＋正常完成收据”可以证明四组执行了完整测试批次，并支持22363样本的覆盖。**TEST_RESULT本身没有独立samples/batches字段，日志也没有逐样本ID或预测；尾批大小3来自长度与batch推导。本项仍不是逐样本ID重放或独立预测复算。** evaluator内部虽累计total_seqs，SID_STATS日志只打印比率，不用其代替数量证明。最终结果metadata/图表包也不等于完整权重备份。

## 5. 控制与机制现象：先证实干预执行，再解释指标

| 核对项 | 预设要求 | 最终观察与直接证据 |
|---|---|---|
| 同一初始功能 | 四臂完整共同权重、有效E₀、正式SID、商品顺序一致 | 训练审计通过；共同初态Tensor SHA为`e6d3379d57a65a4bd0c5b3e2f4656e69b74596920f349c18671b5a2b296645c4` |
| 配对R轨迹 | 逻辑epoch1..35的phase/data seed、data顺序SHA、更新次数及LR一致 | `/controls/common_recommender_epochs`完整为1..35 |
| 配对T干预 | 两Joint周期0..9的phase/data/Gumbel seed、实际调用和累计schedule一致 | `/controls/common_tokenizer_cycles`完整为0..9 |
| 暖启动负对照 | 四臂共同前5R loss一致 | 完整训练审计的paired-controls已检查并通过 |
| Frozen负对照 | 两Frozen全35轮loss、全部val、最终模型/SID一致；T始终0 | `/controls/frozen_loss_epochs`完整1..35，最终及best模型/SID SHA均相同；正式test全部指标也相同 |
| 真实噪声 | 两Joint各1920次history/target实际加噪；R阶段0 | 两个Joint `/summaries/<arm>/noisy_calls=1920`，全部阶段检查通过；Frozen均0 |
| 60步累计退火 | `n(s)=0.2+0.4*(1+cos(pi*s/60))`，softmax恒1，不逐周期重置 | 完整progress逐周期start/end及实际forward均值检查已通过 |
| 梯度、更新与隔离 | 有限预期梯度、实际参数变化、完整Adam状态；T不改R，R不改Tokenizer | 实际best/last中R Adam状态84项；T状态无Mixer Joint为1项、Mixer Joint为9项；步数/动量/冻结侧检查通过 |
| SID现象 | 阶段SID/变化与碰撞统计；最终及best从实际hard_codes独立重算 | `/summaries/<arm>/checkpoint_audit/{last,best}`实际重算并与阶段记录一致，见下表 |

两个Frozen最终模型Tensor SHA均为`14cc553e57f1efc679e8cc397fc526ea0cb993dd467c176c6590571be579ff26`，best均为`7227982d25726011d134df898cd1b41b03ea1839d2df6be7d49e5c5991472cbb`；正式SID SHA均为共同初始`ee955d3b79e69e4c9c96fce6b7f4f8f46aad17eeedd0f55d04298078b24967ea`。两份已完成test相同，负对照一致性已延伸至正式测试。

### 两个Frozen为什么完全相同

两Frozen是刻意设置的**功能等价负对照**。`no_mixer_frozen`直接使用共同基础码本E₀；`mixer_frozen`虽然启用Full attention，但本轮新Mixer的attention输出投影权重与bias均初始化为0，残差系数固定1。因此attention内部即使计算了权重，输出到有效码本的残差仍为0：

`E_no_mixer = E₀；E_mixer = E₀ + 1 × 0 = E₀。`

Frozen全程不更新基础码本或Mixer，所以这条恒等关系一直保持。相同item slots与确定性最近邻量化给出相同正式SID；两组推荐器的完整权重、优化器初态、数据顺序、逻辑epoch随机流、学习率与更新次数也相同。本轮在相同确定性后端实际核验出了相同35轮loss、验证轨迹、最终/最佳模型tensor和正式test；这里既有机制依据，也有完整运行证据，不是只凭“seed相同”作推定。

这两组的作用是验证：启用这个零输出但冻结的模块，没有偷偷改变初始编码、映射、推荐器训练或随机路径；它们不构成两个独立seed的统计重复。其相同结果也不等于“训练后Full交互没有作用”。两个Joint进入T后，`no_mixer_joint`只更新基础码本，`mixer_joint`同时允许Full分支的输出投影等参数被更新，后者才可以从恒等起点偏离，形成同码本内部及码本之间的新增交互。两Joint之差、相对于各自Frozen的差及预设DID，回答的是**从这个统一起点训练新增Full模块的整体作用**，包含其额外参数化和优化路径；不回答另一个非恒等预训练Mixer被冻结时的效果。

| 实际checkpoint | SID改变商品数 | 改变数字数 | 改变商品比例 | 唯一完整SID数 | 碰撞商品数 | 碰撞率 |
|---|---:|---:|---:|---:|---:|---:|
| 两Frozen last（R35/T0） | 0 | 0 | 0% | 10652 | 2174 | 17.965457400% |
| no_mixer_joint last（R35/T60） | 18 | 19 | 0.148748037% | 10650 | 2177 | 17.990248740% |
| mixer_joint last（R35/T60） | 303 | 306 | 2.503925295% | 10650 | 2176 | 17.981984960% |
| 两Frozen best（R15/T0） | 0 | 0 | 0% | 10652 | 2174 | 17.965457400% |
| no_mixer_joint best（R10/T12） | 4 | 4 | 0.033055119% | 10651 | 2176 | 17.981984960% |
| mixer_joint best（R15/T24） | 114 | 117 | 0.942070903% | 10651 | 2174 | 17.965457400% |

精确位置是各`checkpoint_audit`的`actual_SID_change_since_initial`与`actual_SID_statistics/full_sid`，均从实际saved hard_codes重算。所有这些checkpoint每位均使用256个码字，最大碰撞组均40；两Joint终态碰撞商品相对初始分别仅增加3和2件。这些统计相近，不解释码是否语义更优。Full终态改码303件对18件，仍远多于无Mixer；结合HR@10增量为负，再次说明“改码更多”不能等同“推荐整体更好”。

退火端点表述要准确：实际前向使用每次更新前的累计步0..59，完成第60次更新后的调度状态才是0.2；不能把最终buffer值当作最后一次前向一定已经使用0.2。类似地，最终R scheduler归零是调度状态，不替代实际3640次更新证据。

原始CSV及三组PNG/PDF图已生成并下载，覆盖0..35实际验证节点、R/T损失、正式SID变化/碰撞及R累计LR/T累计噪声；[FIGURES.json](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_figures_r1/FIGURES.json)绑定完整计算AUDIT和六个图文件的SHA，本文已逐文件核对字节，并目视检查三份PNG的坐标、标签、图例和完整显示，未见裁切或漏点。两Frozen曲线、共同LR及两Joint噪声轨迹重合是对应控制一致的表现，不是漏画。图只连接实际观察点，单seed不绘制置信区间。

![固定预算验证曲线](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_figures_r1/validation_curves.png)

![训练损失与正式SID变化及碰撞](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_figures_r1/loss_and_sid_curves.png)

![累计更新对应的学习率与噪声调度](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/final_results_r1/delivery/final_figures_r1/update_schedules.png)

## 6. 历史对照与保留噪声的依据

这些是已读取原始JSON并核对完成收据的历史事实，完整来源、精确JSON指针及行号已写在 [PRIOR_EVIDENCE_CONTEXT.md](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/PRIOR_EVIDENCE_CONTEXT.md)，与上文本轮完整训练验证结果分别陈述。

- 共享E₀短四组从Sep-01第5轮权重出发，无Gumbel，Joint一次24次T更新后统一恢复3R；固定R3的HR@10 ΔMixer为−0.000268300317489，NDCG@10为+0.000016697284516，weighted为−0.000040302233753。无Mixer/Mixer Joint分别改变16/284件商品SID，未呈一致额外收益。
- 噪声A/B/C从相同Sep-01第5轮权重出发，B/C各6次T更新、随后同样3R。固定R3的HR@10为0.065957161/0.065957161/0.066136028，NDCG@10为0.047497932/0.047521426/0.047666024；实际加噪调用0/0/192。该短结果不支持“去噪已解决问题”，也不证明噪声强度最优。
- 本轮改为共同Aug28起点、重新执行5R，再累计60T/35R、跨阶段保留状态及R warmup/cosine调度。不能把它称为单纯把短实验拉长或历史精确续训，也不能把历史短对照与本轮数值差全部归因某一个因素。

短四组与本轮在HR@10的Full增量方向均负，但weighted从短对照的微负变成本轮R35微正、NDCG增量也不同。由于起点、噪声、交替节奏、累计更新量与推荐器LR同时改变，这不构成同配方的独立重复，不能据此单独证明“长期训练使Mixer有效”或“加入噪声逆转了效果”。本轮没有新增噪声消融，噪声是两个Joint共同固定的训练条件；仍需以本轮固定预算和测试的各自证据说明观察。

本轮保留带噪声设置有两个明确依据：已有A/B/C没有显示去噪优于原带噪配置，C在固定R3仅有很小正向差值，不能支持优先删掉噪声；同时研究目标是Full Mixer在完整原交替协议中的额外作用，两个Joint使用相同的独立Gumbel流、相同累计退火和数据顺序，可以把噪声作为共同训练条件控制。Frozen不执行T，自然没有T加噪调用。这个设计能够比较“在该带噪联合更新条件下，加Full的整体效果”，但不识别噪声本身的因果收益或最优强度。

## 7. 结论的证据边界

最终解释限于一个seed、一个已学习E₀/推荐器共同起点、当前Beauty/SID-level协议与固定优化预算。Full同时包含码本内部及码本之间交互，本轮未单独比较within-only/between-only；加入Mixer同时增加参数、改变函数参数化及优化路径，没有容量匹配对照。因此即使ΔMixer为正，也应表述为“Full Mixer整体在该条件下的额外效果”，不能证明收益专门来自某一类信息交换；若为负或混合，也不能推广为码本交互普遍无效。

**更多SID变化不等于推荐增益。** 非零推荐梯度、参数确实更新、注意力权重或碰撞率接近均是干预/现象证据，不等于更好的推荐。SID碰撞会影响SID-level解释，不能与阶段06 item-level连续检索直接排序。两个Joint选中best已分别经过12/24次T更新，但不同最佳R/T预算下的test差不等于固定预算DID；没有重复seed或逐样本配对检验，不作统计显著性结论。

## 8. 交付证据状态

| 范围 | 本次取得的证据 | 当前结论 |
|---|---|---|
| 版本、输入与共同起点 | 实际训练identity、四臂environment/resolved、14件锁定输入及共同initial的完整审计；独立V3R2分析锁 | 训练审计通过；正式打包后的本地完整资产仍需另验 |
| 完整训练与固定预算研究结果 | 四臂35R/3640、Joint60T/1920、所有控制、八个真实best/last、固定R35指标与DID | 已通过包含真实权重与全部测试的完整计算审计，本文重算一致 |
| 选模及正式测试 | 四臂封存选择、四份完成test及哈希链；实际2796批/臂日志、test/eval8/beam256/无截断路径 | 4/4正式test与COMPUTE_COMPLETE/完整计算审计一致；独立覆盖审计通过 |
| 图表与原始数据导出 | 四份原始CSV和六个PNG/PDF文件；FIGURES绑定完整AUDIT，六图SHA逐个通过；三份PNG已目视检查 | 六图及来源SHA进入最终验收，三份PNG已经实际查看 |
| 资源与费用 | 最终资源审计覆盖全部7个Job、平台终态及Pod删除事件，含失败和重试 | 全部释放；实际35549 GPU秒/9.874722222小时，按1点/小时估算9.874722222点，实际发票未核对 |
| 远程完整归档与实际恢复 | final_package_r1归档已生成；同一归档全成员字节、14件输入及八个best/last的CPU真实消费者恢复通过，原包/原状态未改写 | 远程归档可恢复已证明；CPU恢复保留CUDA RNG字节，未在此执行CUDA |
| 完整本地备份 | 三个分卷与完整归档SHA、1871清单文件、14输入和9正式状态全部核验；从最终下载包重建两仓库 | 字节、Git与远程八状态CPU恢复的同包绑定全部通过，见FINAL_BACKUP_BINDING_AUDIT.json |
| 05入口及项目交接 | 历史证据保留，当前长期实验入口与Handoff指向本报告及最终审计 | 全部交付条件由最终验收逐项绑定 |

实际资源来源为[RESOURCE_RELEASE_FINAL.json](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/RESOURCE_RELEASE_FINAL.json)，本地SHA256=`ec238b545ac0903f2e649a387db5397fe0aa6650de2f01e983211f6067f810c5`，生成于北京时间15:43:31。它独立核对最终Job清单和每个Job的终态/Pod删除事件，`all_goal_jobs_released=true`；不以训练进程退出代替释放证明。

| Job（均以nancy-stage05-long开头） | 平台终态 | Scheduled→Pod释放（北京时间） | 实际GPU秒 |
|---|---|---|---:|
| 20261020-smoke-r1 | SUCCEEDED | 12:09:31→12:20:07 | 636 |
| no-mixer-frozen-20261020-r1 | FAILED | 12:24:23→12:25:09 | 46 |
| mixer-frozen-20261020-r1 | SUCCEEDED | 12:24:23→14:34:44 | 7821 |
| no-mixer-joint-20261020-r1 | SUCCEEDED | 12:24:23→14:36:08 | 7905 |
| no-mixer-frozen-20261020-r2 | SUCCEEDED | 12:40:53→14:54:55 | 8042 |
| mixer-joint-20261020-r1 | SUCCEEDED | 12:24:23→14:57:02 | 9159 |
| evaluate-20261020-r1 | SUCCEEDED | 15:01:42→15:34:02 | 1940 |
| **合计** | **7个Job全部释放** | 各单卡分配时间求和 | **35549** |

GPU总量包含保留时间、验收、失败、重试和正式评估，按每张卡Scheduled→SuccessfulDeletePod求和；四卡并行墙钟时间不能替代GPU总量。实际35549秒低于43200秒上限，预算余量7651秒；首次H100验收636秒低于1800秒上限。quota单价为1点/分配GPU小时，对应估算9.874722222点；`actual_goal_invoice_settlement_verified=false`，不称已核实账单。

失败的Frozen无Mixer r1来自跨节点初始化同名临时文件竞争，0模型更新、占卡46秒，释放后在相同冻结代码及已核验共有初态下单独启动r2。该失败、重试及其费用保留，不抹去执行历史；单机flock测试不作为跨节点排他性证明。

### 同一完整归档的远程恢复已经通过

远程归档目录为`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training/goal_20261020_long_mixer_v1/delivery/final_package_r1`。远程恢复原始收据已保存在[backup_evidence_remote_r1](backup_evidence_remote_r1/REMOTE_RESTORE_EVIDENCE_AUDIT.json)，其SUMMARY、PACKAGE、MANIFEST、BYTE_VERIFICATION和RESTORE_PASS的SHA链一致。本地完整归档与同包绑定另由[FINAL_BACKUP_BINDING_AUDIT.json](FINAL_BACKUP_BINDING_AUDIT.json)核验。

| 归档/恢复字段 | 已核对值 |
|---|---|
| 完整压缩归档大小 | 1,536,215,400字节 |
| 完整归档SHA256 | `c2ad35b0c87b219e6435a970dfd62bf5677cf7e7c2924b34c53e48c6fec47af7` |
| 文件清单 | PACKAGE.files与MANIFEST记录均为1871个数据文件；另含MANIFEST.json自身，因此不把1871误写为含manifest在内的tar总成员数 |
| MANIFEST SHA256 | `14bc3502ee5a207dd64ccf29657099080e982650418d1f9016e9ca633c4b1ce8` |
| 远程字节校验收据SHA256 | `cdc76fa608e3dcbdf85a4389301b3679857d8bf5db6192cb9466272065125044` |
| 远程CPU模型恢复收据SHA256 | `c882878f5f2f4f92ac73cde8a03e03bfd5fd11061daa75567073b9cd5290dfc1` |
| 远程汇总状态 | `REMOTE_FINAL_ARCHIVE_BYTE_AND_MODEL_RESTORE_PASS`，8个正式状态已恢复，原归档和状态字节未改写，goal_complete=false |

收据包中的精确位置为`delivery/restore_check_execution_r1/REMOTE_RESTORE_SUMMARY.json`、`delivery/package_extract_r1/goal_20261020_long_mixer_v1/BYTE_VERIFICATION.json`和`delivery/restore_check_cpu_r1/goal_20261020_long_mixer_v1/RESTORE_PASS.json`。字节检查覆盖1871个清单文件及全部14件锁定输入；恢复器只使用该已验证备份里的code、engine和输入创建新Pipeline，再恢复四臂各自的best与last，八份checkpoint文件SHA均与本轮正式权重绑定。

八状态恢复实际检查了strict模型载入、完整Adam/调度器/累计更新计数、Python/NumPy/CPU RNG、CUDA RNG字节、正式SID和非persistent E₀ anchor，并确认原始checkpoint没有改写。该检查在远程CPU执行，`cuda_execution_verified_here=false`；同后端CUDA下一阶段重放仍由本轮H100验收证明。**本地已完成相同归档的接收、全量字节核验和最终包Git恢复，1871份清单文件与全部14件输入均通过；两仓库1493个跟踪文件与19份执行源码逐字节相同。** 最终绑定同时核对两地BYTE_VERIFICATION的完整JSON语义，并分别保留Windows/Linux原始收据SHA；CPU恢复绑定远端BYTE，Git恢复绑定本地BYTE。引擎浅历史边界已实际恢复。不同审计器Tensor摘要算法不直接混比，以checkpoint原始文件SHA连接各项检查。恢复配方见 [RESTORE_GUIDE.md](C:/Users/nancy/Documents/experiment1/experiments/05_crosspse_joint/long_mixer_factorial/RESTORE_GUIDE.md)。

## 9. 本地复用与完整交付验收

<!-- stage05-long-delivery-verified -->

| 产物 | 本地位置与用途 |
|---|---|
| 完整原始分卷 | `outputs/stage05_long/package_r1/`：PACKAGE.json与三个分卷，完整归档SHA见上表 |
| 全量已校验解压 | `outputs/stage05_long/extract_r1/goal_20261020_long_mixer_v1/`：共同initial、四臂best/last、14输入、原始记录、代码/引擎与bundles |
| 实际源码Git恢复 | `outputs/stage05_long/git_restore_r1/`：code与engine两个独立仓库，HEAD/tree、fsck、浅边界及1493份跟踪文件核验通过 |
| 本地验收原始收据副本 | [backup_evidence_local_r1/INDEX.json](backup_evidence_local_r1/INDEX.json)：原始SHA、实际字节/Git恢复日志与分块下载历史 |
| 同一归档模型恢复绑定 | [FINAL_BACKUP_BINDING_AUDIT.json](FINAL_BACKUP_BINDING_AUDIT.json)：本地与远端同包、八个真实CPU模型恢复与完整源码复核 |
| 独立科学复核 | [INDEPENDENT_SCIENTIFIC_DELIVERY_REVIEW.json](INDEPENDENT_SCIENTIFIC_DELIVERY_REVIEW.json)：独立重算对照、检查日志覆盖和六图SHA |
| 完整需求验收 | [FINAL_DELIVERY_AUDIT.json](FINAL_DELIVERY_AUDIT.json)、[验收矩阵](ACCEPTANCE_MATRIX.md)：科学、资源、备份恢复、入口和交接的逐项证据 |

`last.pt`同时就是R35最终完整状态，不另外复制相同权重为final.pt。GPU已经全部释放；完整模型恢复在既有CPU环境完成，CUDA RNG在CPU恢复中保留字节，真实CUDA边界重放使用本轮H100验收证据。所有历史短实验和本轮失败/重试均保留。完整权重保存在被Git忽略的outputs目录，源码与证据单独提交；Git提交不能替代大文件备份。
