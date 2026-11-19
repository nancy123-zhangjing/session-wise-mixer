# 创智平台项目资产合同

## TIGER AdamW 预训练修复版正式对照资产（2026-11-15）

适用 sii-shared / 公共科研项目；共享根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/session_wise_ar_tiger_adamw_20261114_v1`。使用版本化配置 `configs/session_wise_ar/tiger_adamw.json`（保留原 `tiger.json` 不变），仅将 Stage-1 RQ 预训练优化器由 Adagrad 0.4 改为 AdamW `lr=1e-3, wd=1e-4, warmup=50, linear decay, clip=1.0`。`parents/tiger_adamw/initial.pt` 保存共同父初始化，`runs/AR-T0A|AR-T1A/{pretrain,train,test}` 保存 10,000 epoch 预训练、200,000 step 推荐训练（15 个验证节点逐样本预测与 checkpoint）和 best/final 测试集预测。两张单卡 H100 作业 `nancy-tiger-adamw-t0a-1114-v1` 与 `nancy-tiger-adamw-t1a-1114-v1` 均已 `SUCCEEDED` 释放。本地入口 `experiments/session_wise_ar/tiger_adamw_20261114/`（含 `RESULTS.json` 与 `REPORT.md`）。

## TIGER RQ 预训练优化器诊断资产（2026-11-14）

适用 sii-shared / 公共科研项目；共享根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/tiger_pretrain_probe_20261114_v1`。CPU笔记本运行的500 epoch RQ-only诊断（Adagrad 0.4/0.05/0.01、AdamW 1e-3 × T0/T1），只读复用`session_wise_ar_five_20261113_r2`源码与共同父状态；`results/`含STEP1/STEP2/COMPLETE.json，无checkpoint和推荐器产物。本地入口`experiments/session_wise_ar/tiger_pretrain_probe_20261114/`。

## CF 教师直推与 DDIM 步数诊断资产（2026-11-14）

适用 sii-shared / 公共科研项目；共享根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_teacher_onestep_20261114_v1`。仅推理：`diagnose.py`只读复用`rqcd_targets_formal_20261113_v1`源码、教师best.pt和CF/冻结组checkpoint；`results/`保存教师两种直推与四个CF节点1/2/5步（及一次20步复现）的逐样本预测npz和`COMPLETE.json`。本地入口`experiments/rqcd_mixer/teacher_and_onestep_20261114/`。无训练产物。

## RQCD 协同目标两组实验资产（2026-11-13）

适用 sii-shared / 公共科研项目；共享根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_targets_formal_20261113_v1`。
两组使用 `v2_cf.json`（纯协同）与 `v2_hybrid.json`（内容＋协同对齐），均为B1层内Mixer、seed2024、R35/T300，统一去噪MSE＋RQ落地监督。`teacher/` 为仅训练一次的共同SASRec教师；`cf_prepared/hybrid_prepared`、`cf_rq/hybrid_rq`、`cf/hybrid` 分别保存两组准备资产、RQ预训练与联合训练结果。
源码基点与逐文件SHA见 `SOURCE_MANIFEST.json`。两组各独立一张H100；CF作业先完成教师再继续训练，Hybrid在教师完成并核对SHA后才提交，不空占第二张卡。本地入口 `experiments/rqcd_mixer/targets_formal_20261113_v1/`，其 `control.py watch` 是本机教师依赖衔接进程。实时运行情况看本轮收据与平台，不根据本条判断完成。大文件只留远端，不变更原RQCD三组和Decoder前缀四组资产。

## Decoder 前缀四组 R20 正式实验资产（2026-11-12）

适用 sii-shared / 公共科研项目；共享根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/decoder_prefix_formal_20261112_v1`。
H0-A/H1-A 为非对称 Decoder 前缀，H0-B/H1-B 为独立前缀交叉注意力；seed2024，每组独立单 H100。
各自新结构暖身R5，同方案H0/H1检查R5共同状态，然后交替至R20/T200停止；学习率/DIGER仍保留原R35/T400日程，不在本轮自动延长或冻结续训。
`code/`为两批H100一致性验收后的源码快照，`training_cache.pt`仅含原collator验证后的静态商品ID和mask；数据/初始PSE只读复用原goal_20261027_r1。
每组`warmup/`、`train/checkpoints/`、`test_best/`、`test_R20/`保留完整模型和预测；大型结果只留远端。本地入口`experiments/09_kuairec_listwise/decoder_prefix_formal_20261112_v1/`。
此条记录资产与协议，实时进度及完成状态应读平台和该目录的启动/最终验收收据，不能据此判断实验完成。


## RQCD 三组best全量最近邻对照资产（2026-11-12）

补充诊断资产：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_history_probe_20261112_v1`，同账号/项目，只读复用B0/R25、B1/R28和原8192验证子集。`results/SHUFFLE.json`绑定不同用户、同历史长度置换，`results/B*/per_sample.npz`保留第一件自由生成及t100/500/900真实/打乱历史逐样本结果；无训练。本地`experiments/rqcd_mixer/history_probe_20261112_v1/REPORT.md`和`FINAL_ACCEPTANCE.json`保存解释与关闭收据。首件命中率与教师加噪恢复率不可混作整列表NDCG。

适用 sii-shared / 公共科研项目。共享根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_full_nn_20261112_v1`。
`results/B0|B1|B2/validation|test/`保存全量同向量RQ/NN读出的向量、预测与摘要；沿用原RQ前缀，非NN自主滚动生成。只读复用正式best，无新训练。
`results/COMPLETE.json`、`CROSS_ARM.json`和`REPORT.md`保存全量对照及配对统计；本地入口`experiments/rqcd_mixer/full_nn_20261112_v1/`，资源终态见`FINAL_ACCEPTANCE.json`。此次全量证据优先于下方8192子集的效果估计。

## RQCD 固定子集落地诊断资产（2026-11-12）

适用 sii-shared / 公共科研项目；共享根 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_landing_diagnostic_20261112_v1`。
只读复用下方正式三组各自best；`results/SUBSET.json`锁定8192条验证样本，`results/B*/vectors_and_predictions.npz`保存同向量RQ/NN配对诊断，`results/ORACLE_AUDIT.json`解释真实向量回查中的重复输入与近重复边界。此处无训练产物。
本地摘要入口 `experiments/rqcd_mixer/landing_diagnostic_20261112_v1/`；结论见`REPORT.md`，完成和资源关闭见`FINAL_ACCEPTANCE.json`。原正式训练目录不变，不以诊断NN读出冒充独立NN滚动生成评测。

## RQCD Mixer 三组正式实验资产（2026-11-11）

适用 sii-shared / 公共科研项目；GPU 作业使用分布式训练空间，文件维护使用 CPU资源空间。
共享根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/rqcd_mixer_formal_20261111_v1`。
`B0_rq/B1_rq/B2_rq` 保存预训练、RQ Encoder/Decoder 和选点恢复资产；`B0/B1/B2` 保存联合训练、
R5至R35全部不可覆盖 checkpoint 与逐样本预测；`prepared` 保存共同初始化及固定推理噪声。
`ASSET_MANIFEST.json` 为文件路径/大小/SHA 清单，包含外部复用数据；`report/REPORT.md` 为报告。
本地入口 `experiments/rqcd_mixer/formal_20261111_v1/`，最终完成与资源关闭以其中 `FINAL_ACCEPTANCE.json` 为准。
大文件保留远端，本地不复制权重；不得把该根视为待清理临时目录。

## H0 两个新 seed 的 R20 冻结对照资产（2026-11-10）

适用 sii-shared / 公共科研项目。共享根为
`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h0_multiseed_r20_freeze_20261110_v1`。
从 `multiseed_freeze_20261109_v1/seed2025_H0` 与 `seed2026_H0` 的不可覆盖 R20/T200 checkpoint 分叉，
冻结整个 Tokenizer/SID、保留原推荐器优化器及学习率日程，继续至 R35/T200；Mixer 始终关闭。
原父实验不修改。源码/数据同父实验；`MANIFEST.json` 保存父 checkpoint SHA 及实际训练源码绑定。
本地入口 `experiments/09_kuairec_listwise/h0_multiseed_r20_freeze_20261110_v1/control.py`，
每组独立单卡，方案、启动记录、验收收据均在该目录。大型文件仅留共享盘；本段不代表运行已完成。

## 两个新 seed 的完整方案对照资产（2026-11-09）

适用 sii-shared / 公共科研项目。独立共享根为
`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/multiseed_freeze_20261109_v1`。
seed2025/2026 各有 H0 持续交替至 R35/T400、原 H1 在 R20/T200 冻结 Tokenizer 后至 R35 的独立目录。
只读复用 `h_freeze35_20261029_r1/code` 及原数据/PSE资产；`MANIFEST.json` 绑定实际源码和种子配置。
每个验证节点的 checkpoint、逐样本预测留共享盘，本地只留小型收据。方案、四个独立提交记录与验收入口位于
`experiments/09_kuairec_listwise/multiseed_freeze_20261109_v1/`。以 live 状态和最终验收记录判断完成，不以本段证明作业完成；不得将本组 H0 标成 H0 Freeze。

## 实验结果存储与重放节点保留约定（2026-11-05）

后续实验的 checkpoint、完整逐样本预测、大型日志与交付归档默认留在远端，
不自动下载或重复打包到本地。本地只保留必要源码/配置、指标摘要、远端路径、
SHA 清单和小型验收收据。现有本地结果没有获得整批删除授权。

原 H1 R5→R20 重放的独立资产根为
`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_replay_r5_r20_20261105_r1`，
适用 sii-shared / 公共科研项目。父状态只读引用原
`goal_20261027_r1/runs/H1/initial.pt`；源码/数据/资产亦原位只读复用。
各节点完成后在 `runs/H1-Replay-R20/checkpoints/` 独立保存
R8/T40、R11/T80、R14/T120、R17/T160、R20/T200 的完整状态，不以滚动 best/last
代替历史节点。实际完成与一致性以该根的 `REPLAY_VERIFICATION.json` 为准。
仅在精确一致性验收通过后，允许删除本次新 run 内与已保存节点 SHA 相同的
best/last 重复文件；保留所有节点状态和验证证据。当前状态与提交记录见
`experiments/09_kuairec_listwise/h1_replay_r5_r20_20261105_r1/`，不以此资产约定证明完成。

## 09冻结续训资源关闭确认（2026-10-30 12:40）

本轮4项已授权GPU分配已全部释放，累计21.810555556 GPU小时（含短测，实际结算未核实）。原0GPU文件Notebook `nancy-stage09-prep-20261027-r1`（sii-shared / CPU资源空间）已停止并由独立平台查询确认STOPPED，资源未删除。09-h1跟踪已删除。运行及后处理归档、外部依赖、科学源码和结果均保留；新对话无需自动重启此入口。证据见[最终CPU释放收据](experiments/09_kuairec_listwise/h_freeze35_20261029/final_lifecycle_r1/CPU_RELEASE.json)与[最终闭环](experiments/09_kuairec_listwise/h_freeze35_20261029/FINAL_DELIVERY_CLOSURE.json)。

## 09冻结续训最终归档与恢复资产

适用Project=公共科研项目；原运行根仍为 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h_freeze35_20261029_r1`。完整运行包为同级 `h_freeze35_20261029_final_r1.tar`，6,846,443,520字节，SHA256 `9006d8d835ed735bd79bfe382ae362ebc6266ba9c1522bf9a20e6b0dcb399f16`；本地同名包在 `outputs/stage09_restore/`，独立恢复树为 `h_freeze35_20261029_final_r1_restored/`。代码/66完整状态/预测/日志由包内DELIVERY_MANIFEST绑定；不要覆盖原code或以当前开发HEAD替换科学源码。

此包非自包含：保留 `outputs/kuairec_listwise_v1/run1`、原 `goal_20261027_r1/assets/pse_v1`、已绑定历史结果及早期A/B产物，具体文件SHA与用途见[最终交付索引](experiments/09_kuairec_listwise/h_freeze35_20261029/DELIVERY_INDEX.md)和[恢复用法](experiments/09_kuairec_listwise/h_freeze35_20261029/BACKUP_USAGE.md)。打包后的统计、图表、恢复收据与报告另有后处理归档/Git绑定，不能认为已在先生成的运行tar中。实时资源与收尾状态仍只由本轮EXECUTION、GPU_LEDGER和生命周期收据管理。

## 09三组冻结续训的独立资产

Project=公共科研项目；CPU资源空间用于文件与验收，分布式训练空间用于独立单H100作业。本轮根为`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h_freeze35_20261029_r1`；`code/`保留历史恢复r2全部3029文件，仅新增绑定e4ca4b3的5份recovery35模块，`delivery/job.py`为有界单卡训练后测试入口，运行时不得覆盖源码。`runs/H1-Freeze20-R35`、`runs/H0-Freeze32-R35`、`runs/H0-Freeze20-R35`分别是A/B/C，名称不能混用；每轮/阶段完整checkpoint独立保存，滚动last不是历史替代。

数据/PSE只读复用原`goal_20261027_r1/inputs/dataset`与`assets/pse_v1`；A从`h1_recovery_20261028_r2/runs/H1-freeze/last.pt`恢复，并认证原H1 R20祖先；B/C分别使用原H0的best.pt/initial.pt。新源和父状态绑定见`code/FREEZE35_EXECUTION_MANIFEST.json`及`FREEZE35_PARENTS.json`。短测产物位于`preflight_r1/`，不得用于正式续训。作业名、预算、即时进度与交付状态仅以[本轮执行记录](experiments/09_kuairec_listwise/h_freeze35_20261029/EXECUTION.md)及其GPU台账为准，不把下方旧作业当当前任务。

## 09正式四分支的调度与恢复入口

执行根保持`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1`，科学源码冻结`1c423662d9a25a926328ee6466c30c0f2857b4f6`。`delivery/formal_four_arm_job.py`管理两个GPU上的四个独立模型进程，GPU0=J0/H0、GPU1=J1/H1，无DDP；`runs/J0`至`runs/H1`分别保存checkpoint，`delivery/four_arm_supervisor_r1`保存ready、initial配对验收和进程日志。它不自动恢复或清理训练目录；观察失败不能直接重提。当前正式作业已启动，具体Job、调度时间、进度及费用记录以[09执行记录](experiments/09_kuairec_listwise/EXECUTION.md)和[台账](experiments/09_kuairec_listwise/preparation_20261027/GPU_USAGE_LEDGER.json)为准。下方为历史记录。

## 2026-10-27：09改用独立global_user目录，06/07远端副本已清理

- 正式共同暖身已于北京时间20:35左右提交并RUNNING：J为`nancy-stage09-warmup-j-20261027-r1`（`qb-prod-gpu002`，平台上限2.15小时），H为`nancy-stage09-warmup-h-20261027-r1`（`qb-prod-gpu146`，2.4小时），各1张H100、无DDP。见`experiments/09_kuairec_listwise/formal_execution_20261027/`中的`J_CREATE.json`、`H_CREATE.json`、`STATUS_START.json`及`j_LOG_START.json`、`h_LOG_START.json`。TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果；四分支尚未启动。

- 最新授权见[75 GPU小时预算批准](experiments/09_kuairec_listwise/preparation_20261027/BUDGET_APPROVAL.json)和[已批准资源计划](experiments/09_kuairec_listwise/h100_preflight_r2/APPROVED_RESOURCE_PLAN.json)；J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果。

- 用户最新同意09放在`global_user`，本轮新执行根为`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1/`。该许可覆盖此前09只写原主项目的限制；只准备本轮代码副本、输入、依赖、缓存和产物，不迁移原项目，不恢复已清理的06/07目录。1MiB探针、661708800字节代码/数据包部署及BGE四分片/整包/11文件远端回读均已通过；BGE收据为`experiments/09_kuairec_listwise/assets_preparation_20261027/BGE_REMOTE_TRANSFER_VERIFIED.json`。单H100内容编码已实际读写新根，完整Goal容量仍未验收。
- 远端32条真实文本CPU probe通过：编码5.4594659秒、5.86138件/秒，带30%余量的CPU全量估算40.4383分钟。随后内容编码Job `nancy-stage09-bge-encode-20261027-r1`已SUCCEEDED，日志确认10728件、exit0；节点`qb-prod-gpu050`、1张H100 80GB，仅冻结BGE推理，不是正式推荐训练。编码占用135秒，后续H100验收R1占266秒、R2占255秒，三个准备Job均已释放；已终态的准备作业合计656 GPU秒（0.182222 GPU小时），准备阶段原7200秒上限下剩6544秒（历史）。最新已批准累计75 GPU小时（270000秒），包含已终态作业656秒，269344秒为两暖身分配前余额；当前新增占用和实际余额待平台终态事件核算。
- CPU共同PCA/PSE及独立审计已完成：`toy=false`、10728件、码本`[4,256,64]`，最近邻SID与误差重算一致。见`experiments/09_kuairec_listwise/assets_preparation_20261027/PSE_MANIFEST.json`和`PSE_ASSET_AUDIT.json`。9513个唯一SID，1505件商品位于多商品碰撞桶（14.0287%），最大桶84件；这是商品可区分性的限制，不是工程验收失败。正式阶段按用户最新要求以2卡分别跑J/H共同暖身、4卡分别跑四个独立分支，每模型单卡、无DDP，累计75 GPU小时预算已批准。详见09执行记录。
- R1/R2四臂及supervisor均passed，真实BF16梯度、R/T隔离、恢复、共同初始化、SID刷新和无目标列表生成通过，收据在`experiments/09_kuairec_listwise/h100_preflight_r1/`及`h100_preflight_r2/`。R1的`METRIC_RECOMPUTE.json`仅验证beam2小批商品指标计算，不是正式效果。R2使用微批256、有效batch1024、推理batch32；科学模型/beam/评价定义不变，正式训练不接短验收状态。
- 远端训练及代表checkpoint恢复保持已验收版本`1c423662d9a25a926328ee6466c30c0f2857b4f6`冻结；本地`83a1b9c`新增analysis仅用于离线分析，不同步改变远端训练源码binding。
- 共同BGE/PSE资产18文件、94,105,600字节tar已在`outputs/stage09_restore/goal_20261027_r1/`独立恢复PASS，见`experiments/09_kuairec_listwise/assets_preparation_20261027/GENERATED_ASSETS_LOCAL_RESTORE.json`；不含正式训练模型。R2预算估后续基础58.1855 GPU小时，含25%余量和已用量累计72.9141 GPU小时；2卡暖身、最多4卡分支预计墙钟15.868/19.835小时（基础/含余量，不含排队）。按1点/GPU小时仅作未结算估算。用户已批准累计75 GPU小时上限，J/H暖身已提交且RUNNING；TRAINING_START.log快照已确认J完成113次R更新、H完成59次R更新，均在epoch1、T=0；尚无完整暖身结果；详见09执行记录及`h100_preflight_r2/BUDGET_ESTIMATE.json`。

- 用户明确授权清理`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/`和`experiment1_stage07/`下全部已盘点运行/备份子目录，现两根为空。下文相关远端路径只作历史定位，不再是现存可加载入口；不得自动重建。
- 本地06/07正式备份和结果报告保留，重新SHA校验通过。清单、恢复范围和09存储阻塞见`experiments/09_kuairec_listwise/preparation_20261027/REMOTE_CLEANUP_20261027.md`。05共享输入、项目外模型环境及当前代码依赖不在清理范围。
- 清理完成后，原主项目09目录的136305字节写入仍报`Disk quota exceeded`，原路径配额尚未恢复。06在2026-10-16也曾在主项目遇到`errno122`，随后改用`global_user`继续，见`experiments/06_continuous_baseline/execution_acceptance/PLATFORM_SUBMISSION_20261016.json`的`storage_note`。主项目盘点8.2G是已用量，不是配额上限；两挂载入口的统计和写入表现不同，但具体配额归属、数额及是否完全独立尚未核实。

## 阶段 08 已退役：原单目标历史码字交互

- 用户确认研究目标为OneRec的请求级列表生成，原08代码及模型已按要求清理。`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_session_codeword/`不再是可执行或可恢复资产入口；不得根据旧收据重建旧实验。
- 共享原始数据、01–07模型及项目外`grm-exp`环境保留。历史结果及删除清单见`experiments/08_session_codeword/RETIREMENT_20261026.md`。
- 新09列表生成研究已有本地实现`scripts/listwise_diffgrm/`和固定配置，不复用原08的语义。最新执行根与权限以上方global_user记录为准；此前主项目`experiment1_reorganized/outputs/stage09/goal_20261027_r1/`仅保留早期写入失败的收据和残留，不是当前部署目标。Notebook/Job只是执行入口，不改变文件归属；已有项目外模型环境只读复用，不安装或升级其中依赖。
- 09实际准备、GPU预算和执行状态记录在`experiments/09_kuairec_listwise/EXECUTION.md`，不能根据本地runner或CPU检查自动启动正式训练。具体入口见`docs/research/onerec_listwise/IMPLEMENTATION.md`。

## 阶段 07 商品内局部注意力的独立执行副本

- Project仍为公共科研项目；CPU资源空间准备与同步，分布式训练空间运行单H100持久Job。
- 本阶段个人共享根目录为 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/`；每轮使用独立Goal子目录，`code/`与`engine/`分别接收根仓库和DiffGRM引擎的锁定Git bundle，`delivery/`保存传输校验和作业收据。
- 原始Beauty数据、PSE量化资产和正式SID从HDD主项目只读校验后复制到该轮`inputs/`；不修改历史主项目，不重新拟合量化器。训练、smoke和缓存分别写独立输出目录。
- 继续使用主项目外已有`grm-exp`环境，不升级共享环境。交替联合训练入口为 `scripts/train/run_item_local_joint.py`，GPU包装器为 `scripts/train/item_local_joint_gpu_job.py`；复用05锁定离线Refined-PSE和配套DiffGRM暖启动，码本通过交替优化更新。旧固定SID入口 `scripts/train/run_item_local_attention.py` 仅用于历史追溯。配置、版本、预算和即时状态保存在本轮实验收据，不把旧作业状态当成当前事实。

## 阶段 06 连续向量基线的独立执行副本

- Project仍为公共科研项目；CPU资源空间同步/验收，分布式训练空间运行GPU Job。独立根目录`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/`，不与05训练副本混用，不代表主项目迁移。
- 每个Goal使用独立子目录：`delivery/`保存Git bundle、传输哈希与平台收据，`inputs/prepared/`和`inputs/cache/`原样保存已验收06输入，`code/`为从bundle克隆的固定版本。
- 06数值修复副本位于`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/goal_20261016_precision_v2/code/`；只读复用`goal_20261016/inputs/`的锁定输入，原`goal_20261016/code/`及输出保留作版本追溯，禁止混入修复版本选模。
- 进程管理修复作为独立部署保存在`goal_20261016_precision_v2/orchestration_v3/`，入口`scripts/train/continuous_queue_supervisor.py`，以`DEPLOYMENT_MANIFEST.json`绑定其独立提交与哈希；训练worker仍从原`code/`冻结版本启动。恢复输出使用`code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3/`；原pipeline保留，完成项和中断状态的继承须有独立CPU核对收据，不能把调度版本当作新模型版本。
- 入口`scripts/train/run_continuous_campaign.py`；输出位于该`code/outputs/stage06_continuous/<run_id>/`。运行期间不更新执行代码；CPU gate与正式作业绑定同一源码和数据哈希。
- 使用主项目外的`grm-exp`环境，不升级共享环境。Job名、实际版本、费用和完成状态写本轮收据，不作为永久资产状态。

## 适用范围与主目录

- Project：公共科研项目。CPU资源空间用于开发与文件处理；分布式训练空间用于 GPU 运行。
- HDD 主项目：`/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized`。
- 本地工作副本：`C:/Users/nancy/Documents/experiment1`。两个名称不同，不代表本地副本已经废弃。
- 旧服务器 `nancy/experiment1` 已退出运行路径，不重建同名目录或指回旧副本的软链接。`experiment1_retired_20261009` 在完整核验后于 2026-10-12 清理；新项目保留独立迁移资产，本地工作副本不随服务器退役而改名。
- 新主目录内的 `data/`、`outputs/`、`external/DiffGRM/saved/` 是独立迁移后的运行资产；不得软链回旧 `experiment1`。
- `archive/legacy_server_20261008/` 保留旧脚本、文档、证据包及旧引擎源码原始字节，仅用于追溯，不作为启动入口。
- 历史原始日志中的机器路径保持原样。开发层通过 `runtime_paths.py` 显式重定位已知旧前缀，不改写历史指标和哈希锁。

## 环境与代码版本

- 模型环境：`/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp`，Python 3.10；它在新旧项目之外，删除旧项目不得删除此环境。
- 严格历史 CSV 重算使用 `/inspire/hdd/global_user/<inspire-account>/miniconda3/bin/python`，已验收 Python 3.14.7。Python 3.10 的 statistics 标准差末位可能不同，不据此改历史结果。
- 根仓库移植基线：`5cc5ef3d2a318e84a58d7906c01bfc900639946d`；迁移路径兼容在此后单独提交。运行时记录实际 `git rev-parse HEAD`；工作区若有改动，另存补丁，不把未提交代码当作该 commit。
- DiffGRM 开发引擎：`external/DiffGRM`，独立 Git，基线 `39e94e6153c97e59877ba02d4ef1d9a202ed5384`。
- GenRec 开发引擎：`external/genrec`，独立 Git，基线 `b6c8b41d1d20a812590dae11408b69b1ed2a8313`。
- 引擎基线是已锁定的可重放源码，不冒充未记录的历史运行 commit。根 Git 不自动跟踪引擎修改，后续必须分别记录版本。
- 离线 Git bundle 不是 GitHub。服务器上引擎 origin 指向新主目录内的离线包；根 origin 是共享盘交付包。未建立自动双向同步。
- 代码真源与同步：本地审查并提交主仓库；服务器主目录接收同一 commit 后运行。引擎修改在独立仓库另行提交并记录，不同时维护两套未同步的修改。迁移验收与收据索引见 `docs/guides/EXPERIMENT_WORKFLOW.md`，不以此文件替代运行证据。

## 使用与保全

- Linux 上从主项目根目录 `source runtime_env.sh` 后使用模型环境，输出仍放新主目录；正式任务使用唯一 run_id。
- 当前路径兼容测试：`python -B tests/test_runtime_paths.py`。历史目录整理审计对应原封存版本，不用于否定随后明确提交的功能改动；历史证据和哈希锁不因迁移而修改。
- 复现历史实验优先使用 `scripts/repro/` 的封存配方，以新主目录作为 `--project-root`，不重新拟合历史 SID。
- 每份历史实验的初始化数据、模型与 Tokenizer 以 `experiments/01*` 至 `05*` 中的 recipe 和 assets.lock 为准。
- 迁移清单、源/目标哈希、退役收据和验收报告存于 `archive/server_migration_20261008/`；2026-10-12 的逐文件清理记录与删除后验收存于其 `cleanup_20261012/` 子目录。本地对应记录在 `archive/server_sync/server_cleanup_20261012/`。
- 原始迁移包保留于本地 `C:/Users/nancy/Downloads/experiment1_migration_20260923_retry/`，完整 SHA256 和 gzip/tar 校验通过后清理了远端重复包。服务器直接使用已迁入的锁定输入；如需从压缩包恢复损坏文件，先从本地备份重新上传。该原包不包含后续新短测试，后增产物不得以“原包已有备份”为由删除。
- `.git`、可重建字节码、编辑器缓存不从旧服务器副本迁入；新仓库及引擎使用已恢复的真实 Git 历史。私人凭据不作为实验资产复制。
- 不删除历史证据、锁文件、配对 SID/权重或环境来释放空间；是否清理需先核对消费者和替代副本。

## 阶段 05 新短对照资产

- 固定一次更新、冻结、恢复的训练入口是 `scripts/train/run_diger_noise_pilot.py`；固定参数和共同起点哈希在 `configs/05_crosspse_joint/pilots/diger_noise_pilot_seed2024.json`。
- 新输出使用主目录下 `outputs/pilots/stage05_noise/<唯一名称>/`，不覆盖历史 `outputs/reports/` 或 `experiments/05_crosspse_joint/evidence/`。
- 同步包、后台 Job 日志与启动收据使用 `outputs/pilots_delivery/<唯一名称>/`。代码通过增量 Git bundle 快进同步；不以逐文件覆盖代替版本同步。
- GPU smoke 的原始 JSON/日志在 `experiments/05_crosspse_joint/pilots/noise_update/preparation/`；完整 GPU 权重保存在对应共享盘输出。轻量证明包不含模型权重，不能单独恢复训练。
- 运行状态与即时资源信息以 Job、启动收据和输出文件为准，不把它们写成此资产合同的永久状态。未保存完整历史优化器状态的共同起点实验不得称为历史精确续训。

## 阶段 05 推荐反馈对照资产

- 入口为 `scripts/train/run_diger_feedback_pilot.py`，固定协议为 `configs/05_crosspse_joint/pilots/diger_feedback_pilot_seed2024.json`。三组从同一锁定权重开始，分别不更新 Tokenizer、仅重构等目标更新、额外推荐反馈更新；三组均保留 Cross Mixer。
- 新输出仅使用 `outputs/pilots/stage05_feedback/<唯一名称>/`。GPU smoke 与正式 pilot 分开保存，正式实验只使用验证集，不覆盖既有噪声对照及历史证据。
- 同步包、GPU 启动入口和收据继续存放 `outputs/pilots_delivery/<唯一名称>/`。启动前须通过服务器 `--require-torch` 张量检查、18 项输入绑定检查及三组 GPU smoke；后台 Job 记录实际 commit 和输入哈希，不依赖本地电脑持续登录。

## 阶段 05 只读分析的 SSD 工作副本

- 限定分析根目录：`/inspire/ssd/project/project-public/<inspire-account>/nancy/experiment1_analysis`，适用于公共科研项目的 CPU/GPU 分析作业。
- `code/` 是按本地已提交版本同步的稀疏分析代码副本，不是 HDD 主项目的完整备份。每次锁定实际 commit、必要配置与测试文件；不要把 SSD 同步称为 HDD 主项目同步。
- `delivery/<唯一名称>/` 保存增量 bundle、校验、启动与完成收据；`outputs/analysis/stage05_feedback/<唯一名称>/` 保存逐样本轻量统计。短测试和正式统计分目录，不覆盖旧产物。
- 数据、初始化及历史快照仍从 HDD 主项目只读加载，使用主项目之外的 `grm-exp` 环境；临时缓存写独立临时目录，新结果只写个人 SSD。
- 固定推荐器诊断入口为 `scripts/analysis/evaluate_feedback_hard_compatibility.py`，独立重算入口为 `scripts/analysis/feedback_hard_compat_analysis.py`。必须保留实际解析的原配置、正式 SID、输入哈希及模型未改变收据。
- 作业运行时不更新 SSD 代码；所有作业结束后再快进同步。原始逐样本压缩记录与轻量归档保留在输出区，研究总结和版本绑定收据按阶段目录归档，不把模型权重加入 Git。

## 阶段 05 交互模块四组对照的独立训练副本

- 已授权本轮代码与新权重使用 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training`。这不是项目主目录迁移；原项目数据、历史模型与初始化仍从 HDD 主项目只读加载。
- `code/` 是提交版本锁定的稀疏执行副本，不作为完整备份；`delivery/<唯一名称>/` 保存同步、短测试和启动收据，`outputs/pilots/stage05_mixer_factorial/<唯一名称>/` 保存四组产物。运行期间不更新执行代码。
- 入口 `scripts/train/run_diger_mixer_factorial.py`；实验定义见 `experiments/05_crosspse_joint/pilots/mixer_factorial/README.md`。共同有效码本 E₀重基、SID、完整推荐器状态、随机种子和恢复预算必须一致。
- 正式启动前执行 `tests/test_diger_mixer_factorial.py --require-torch`、输入哈希核验与四组 GPU smoke，并检查 GPU 内可以读取全局个人目录与实际写入。`DIFFGRM_ENGINE_ROOT` 可显式指向只读历史引擎，禁止因稀疏副本缺引擎而跳过真实张量测试。

## KuaiRec 留五数据与静态 PSE 资产

- 适用 Project：公共科研项目；CPU资源空间准备，分布式训练空间读取。独立根目录：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_stage0_20261117_v1`。
- `source_dataset/` 仅保存公开 KuaiRec 的已清洗事件分片和重建所需目录文件，不是旧数据目录的完整备份。源发布审计在 `docs/research/onerec_listwise/kuairec_preparation/SOURCE_AUDIT.json`，本次逐文件传输 SHA 在阶段零目录清单。
- `dataset/` 为 `kuairec_l5o_records_v1`：按用户最后五条记录测试、此前五条验证，history128，保留目标重复记录；EV 主阈值1.0。训练/验证/测试规范化行哈希与本地重建一致，见 `experiments/09_kuairec_listwise/l5o_stage0_20261117/REMOTE_PREPARED.json`。
- `assets/` 复用 `goal_20261027_r1/assets/pse_v1` 的纯静态文本拟合资产；原始字节不变，仅另建数据绑定。未继承旧协议训练后的推荐器、码本或 Mixer。旧全局时间切分数据和资产保持原样。
- `code/` 与 `SOURCE_MANIFEST.json` 为阶段零执行快照；短测和正式训练分开记账，不能从资产存在推断正式实验已授权或已验收。
