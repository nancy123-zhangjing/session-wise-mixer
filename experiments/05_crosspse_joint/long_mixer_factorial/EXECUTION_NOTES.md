# 运行交接（非完成报告）

用户在本Goal后续消息明确允许“多申请几张h100来跑”，据此四组各用1张H100、最多同时4张；旧Goal正文的同时1卡已被该人工授权更新。累计上限仍是12 GPU小时，各卡时间求和。不得新增seed/消融或缩减35轮/60次更新。

当前运行根：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_training/goal_20261020_long_mixer_v1`。CPU桥接`env-zzl`已复用，其他阶段可能也使用它，不停止或删除该桥接。模型Python为`/inspire/hdd/project/project-public/<inspire-account>/nancy/conda-envs/grm-exp/bin/python`；不改共享环境。

当前四个正式Job：

| 实验臂 | 当前Job后缀 | 已观察Scheduled（北京时间） | 平台cap秒 | 父进程cap秒 |
|---|---|---|---:|---:|
| no_mixer_frozen | `no-mixer-frozen-20261020-r2` | 12:40:53 | 9420 | 9240 |
| no_mixer_joint | `no-mixer-joint-20261020-r1` | 12:24:23 | 9480 | 9300 |
| mixer_frozen | `mixer-frozen-20261020-r1` | 12:24:23 | 9420 | 9240 |
| mixer_joint | `mixer-joint-20261020-r1` | 12:24:23 | 9480 | 9300 |

完整名称均以`nancy-stage05-long-`开头。旧no_mixer_frozen R1已FAILED并在12:25:09删除Pod（从12:24:23起计46 GPU秒），构造Pipeline前退出，0训练更新。验收Job已SUCCEEDED并删除Pod，636 GPU秒。当前已释放实际分配合计682秒；`ALLOCATION_LEDGER_2.json`保守保留另外三组R1的完整cap及NMF R2完整cap，并另预留2640秒最终测试，总计41122秒（约11.4228 GPU小时）。运行中的已分配时间包含在这些cap内，不能又当作完成费用重复相加。

原始`FORMAL_READINESS.json`（R1）在远端已归档为`delivery/FORMAL_READINESS_R1.json`，当前远端canonical指针是R2版本，绑定ledger2；本地两个版本见`FORMAL_READINESS.json`（原始R1）与`FORMAL_READINESS_R2.json`。后三组STARTED中的旧SHA必须用R1归档核对，不能拿新的指针内容冒充原始收据。后续换ledger须保留这些历史。

训练代码仍为根仓库`69682698c15ba796b98e6329c950061a2f68a05b`、引擎`1c879252f6395d6e64462511823c3929b9c42c00`，不在运行期间修改远端code/engine。引擎为浅历史，恢复bundle前需要保留边界`ad7b971c7e525e9fea6fb8e362a5c49dccb2473c`；本地`.runtime/stage05_long/engine.shallow`已保留。

跨节点冷启动曾竞争同名`formal/identity.json.tmp`。不能用同主机flock测试当跨节点排他性证明；本轮已只读重载并核验`formal/common/initial.pt`与验收初态逐张量相同，再单独重启NMF R2。四组实际初态均为`e6d3379d57a65a4bd0c5b3e2f4656e69b74596920f349c18671b5a2b296645c4`。现在共享初态/身份均已落盘，不删除、不覆盖它们。后续恢复先查唯一Job和旧Pod释放，再使用已有共有初态；不同时冷启动写同名共有文件。

只读检查器：本地`.runtime/stage05_long/monitor_progress.py`，远端`prep/monitor_progress.py`，可用模型Python传`--goal-root <上面目录> --save`运行。它只核对提交完整阶段的JSON，输出“monitoring_not_completion”；平台活性仍须用实时CLI单独确认。快照写`delivery/progress_snapshots/`，最近精简快照指针`delivery/PROGRESS_CURRENT.json`。`progress.json`写于完整阶段结束；正在整轮训练或整轮验证不代表卡住。恢复应以完整`last.pt`中的progress为准，不能仅凭旧JSON重做已提交更新。

12:49:39快照：no_mixer_joint R5/T6、192次真实噪声调用；mixer_frozen R5/T0；mixer_joint R4/T0；NMF R2初始验证后正在R1。已有前三组共同前4轮loss、数据/RNG、更新和LR完全一致，四组初态相同，正式TEST_STARTED数为0。NMF完整R1提交后才能增加四组共同轨迹证据。

`mixer_joint`所在`qb-prod-gpu018`的R epoch平均约207秒，其他两组约136/149秒，可能无法在当前slot内完成全部35轮。继续监控实际进度和父进程deadline（约14:59:29）；如需迁移/续跑，必须保持完整阶段checkpoint及所有模型/优化器/scheduler/RNG/SID/数据身份，先确认旧Pod释放并更新累计账本。不得当作性能早停，不得改训练轮数。已有快速组释放后的实际开销可替换cap，重新核对全局12 GPU小时及最终测试预留后再提交有限剩余阶段；不能盲目重提整个完整预算。

只有全部四组`TRAIN_COMPLETE.json`核实35R、3640R更新、两个Joint60T和1920噪声调用，且控制核对通过后，才运行同版本的`--mode evaluate`（包装器`--mode evaluate --arm all`）。先封存所有best选择，再逐臂一次正式test。该任务尚未创建；预算已预留。测试beam测速用的是val样本，不能误称提前访问了正式测试集。

训练结束仍需：固定R35验证对照、best测试及差中之差、原始表图、实际分配与标价费用记录、完整权重和输入/源码bundle的本地备份、逐文件哈希和代表恢复检查、全部本Goal GPU释放证明、05入口与项目交接更新。轻量gate包不含可恢复完整模型备份。最终按`ACCEPTANCE_MATRIX.md`逐项审计，全部实证达成后才允许Goal complete。

2026-10-20 13:26:06更新：实际完整边界为 no_mixer_frozen R9/T0、no_mixer_joint R15/T24、mixer_frozen R15/T0、mixer_joint R13/T18；正式完成仍0/4，测试未开始。共同前9轮数据/RNG/LR/预算匹配，前5轮loss完全相同，两Frozen共同前9轮loss完全相同，两Joint共同前三个Tokenizer周期的实际噪声与顺序配对。此为部分前缀核对，不能当作长期验收。独立审计8项及备份传输5项测试通过；分析工具独立放在delivery/analysis_tools_v2，训练code/engine不变。完整正式审计、图表、完整备份和恢复仍待完成。分析/打包命令须使用OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4，避免CPU桥接线程过量。

独立分析/备份工具提交e124868a5fa92c5431772b8bcae876c6aced5421，ref为codex/stage05-mixer-long-delivery，父提交是训练6968269。没有切换当前工作区分支或修改其他聊天改动。远端工具V2归档SHA为364394ec6981d368999713063e397572223447cf7f04baf5e6116cdb6a8a0a6e；13项工具测试及实际共有初态独立重载核验通过，见ANALYSIS_DEPLOY_V2_PASS.json。正式完整审计和完整备份仍未执行。恢复入口restore_mixer_long_backup.py先要求完整包逐字节通过，随后在全新scratch中用备份自身code/engine/14件输入重建模型并恢复各臂best/last的模型、两个优化器、scheduler和CPU/Python/NumPy RNG；CUDA RNG保留字节，CPU检查不宣称执行CUDA。本地continuous_env有Torch2.6/Numpy2.2.6，可进行独立Tensor审计及字节检查，缺少完整Pipeline依赖；完整模型消费者恢复使用现有远端grm-exp CPU环境和隔离备份副本，不安装/升级共享环境。

13:39独立工具V3R2通过25项测试，并对四臂实际完整阶段checkpoint正向核对Adam/scheduler/RNG存在及真实SID统计。进度为R14/19/19/15，仍非正式长期完成。当前工具目录delivery/analysis_tools_v3r2，源码提交4b40c90b56490868e5972e75bafbf8e50ca8a3de，归档SHA2a95d2d53ac53a5adcdb7f1b0f43eff5beb42414f76a6e19f169b63066d3f2f5。审计新增SID范围/连续性/真实重算、weighted反馈及best/last progress精确绑定；绘图输入绑定TRAIN_COMPLETE/progress字节哈希；完整包强制包含三份Git bundles。V3首次工具打包漏了测试依赖的协议配置，未进入模型检查，失败日志保留；V3R2在新目录补齐配置后通过。当前训练code/engine仍是696/1c。

源码Git恢复探针已实际通过（不是最终下载包恢复），见RESTORE_GUIDE.md和GIT_RESTORE_RECIPE_PROBE.json；Windows须开core.longpaths=true并禁用自动换行，否则Git checkout即使退出0也可能漏深层历史CSV。引擎须先写ASCII无BOM浅边界再fetch。最终备份下载后还必须从该包重做Git及完整Pipeline消费者恢复。资源独立核对PARTIAL_RESOURCE_AUDIT_2.json得到已释放实际682秒，不包括仍运行四卡，不能称最终资源释放。

14:51实时更新：mixer_frozen与no_mixer_joint均完成35R/3640R更新，独立best/last审计通过，并分别在14:34:44与14:36:08确认SuccessfulDeletePod。两个Joint累计60T/1920次实际加噪；剩余NMF与MJ均提交R34，正在最后一轮及其验证。当前控制检查全通过，正式测试仍0；最新快照PROGRESS_20261020T065101Z.json。含smoke与失败R1，四个已释放Job实际共16408GPU秒（PARTIAL_RESOURCE_AUDIT_4.json），另两个仍运行。当前账本仍保守保留旧完整cap，不把已运行时间重复相加。MJ最近R34用159.285秒，较早期加快，预计现有deadline内完成；尚未提交续跑。全部训练结束并核实释放后，重新按实际分配锁单卡evaluate专用ledger，拟给3960秒平台cap/3780秒processcap；保守16408+9420+9480+3960=39268GPU秒，仍小于43200。测试配置、选择规则、训练协议均不改变。

15:03更新：四组35R/3640R更新全部完成；两Joint60T/1920次真实加噪。独立四组TRAINING_AUDIT_FINAL_R1.json重载8个best/last并核验初态、完整轨迹和实际SID，全部通过。NMF R2在14:54:55删除Pod计8042秒，MJ R1在14:57:02删除Pod计9159秒；含smoke及失败R1的六次已释放分配共33609GPU秒。新ALLOCATION_LEDGER_3.json只预约evaluate_all_r1单卡3960秒，process3780秒，保守累计37569秒<43200；R2历史readiness保留，远端canonical已发布R3（c0ae89f3f04afb5f33849855adcff4c99c1b7cf381cd439af22e6acde8430171）。fresh quota/policy/image/availability、dry-run与唯一性检查通过后，15:01提交nancy-stage05-long-evaluate-20261020-r1；15:02:42实时平台确认为RUNNING、qb-prod-gpu088、1H100，入口STARTED存在，尚无测试结果或失败收据。选择封存与4次正式test仍待核实，不能称完整交付。


2026-10-20最终科学计算与远程恢复核验：四组35R训练于14:56:44全部结束，验证选模封存后四组各一次beam256正式测试于15:33:54全部结束，独立AUDIT、完整测试日志覆盖和三张实际图形检查均通过。最后评估Pod15:34:02删除，所有7次分配共35549 GPU秒（9.874722222小时），含验收636秒和失败NMF R1的46秒；不再申请GPU。完整归档1,536,215,400字节/1871条manifest文件记录，另含MANIFEST.json；远程字节及八份实际CPU模型恢复通过。完整本地下载曾遇SCP和并发连接超时，已逐块验证有效前缀并改用单连接哈希分块读取，不丢弃已校验数据、不改归档。当前本地完整包字节与Git恢复仍待下载完成，Goal保持未完成。历史进度快照与失败证据保留。


最终本地完整备份验收：三个分卷与拼接归档、1871份清单文件、14件输入及9正式状态全部校验通过；最终包源码Git恢复与同一归档八份CPU实际恢复完成绑定。实际下载/恢复路径、原始收据SHA和失败传输历史见backup_evidence_local_r1及FINAL_BACKUP_BINDING_AUDIT。完整科学报告转为REPORT.md，REPORT_DRAFT.md保留历史交付前快照。
