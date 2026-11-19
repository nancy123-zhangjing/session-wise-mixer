# 多随机种子交替联合训练：执行记录

## 2026-10-21 23:12（北京时间）

- 用户已授权新增 seed2025、seed2026，各运行 control / item_local，共四组；维持原交替联合训练，仅验证集评估 beam32/256。
- 远端独立根目录：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/goal_20261021_multiseed_r1`。
- 根执行代码提交：`037876cbb3de339ad011e5abcf0e66169967ce3b`；引擎：`1c879252f6395d6e64462511823c3929b9c42c00`。已成功部署；不重复 bootstrap，不修改冻结副本。
- 本地及远端 CPU 验收均为 40 项通过、0 跳过；远端四组输入已准备。CPU_GATE SHA256：`038e6c82332dfbc681a48153426850f318abb5f26587a07965ae8529125e55ad`。远端记录下载在 `REMOTE_CPU_DELIVERY/`，20 份测试/准备证据哈希已独立核对。
- 网络一度重置；用户确认正常浏览器可访问后，同一本机代理恢复，官方 `notebook connection refresh` 成功。未更改代理或登录配置。
- 两个单 H100 短验收已于 23:10:58 提交，任务名见 `SMOKE_SUBMISSION.json`；23:12 尚在 QUEUING，事件显示等待资源释放。不得重复提交。
- 四组正式任务仅完成 live dry-run，**尚未提交，尚未开始正式训练**。
- 先等待两个同 seed A/B 短验收成功，核对初始化、梯度、恢复状态并确认平台释放；再提交四个独立单 H100 Job。
- 每正式组固定 45 logical epochs（35 推荐器 epochs / 3640 推荐器更新，60 Tokenizer 更新），验证选模规则不变。平台单组上限 2.95h；首次六作业总预留 12.3 GPUh，执行保护上限 14 GPUh，不自动重试。
- `FORMAL_DRY_RUN_LIVE.json`、`SMOKE_DRY_RUN_LIVE.json` 是最新带 bash 包装的计划；旧无 LIVE 的 dry-run 仅历史记录。
- 尚无新模型指标。旧 seed2024 只作历史参考；不得用旧指标填充新增 seed。

## 后处理提示

- `summarize_results.py` 已通过 8 项离线测试；旧四个验证 cell 的 89,452 行重算一致。缺新结果会拒绝生成报告。
- 若确实发生恢复，须注意旧 runner 重新序列化 last.pt 后可能与原 FIT_COMPLETE 的 training_state_sha256 不同；当前评估器严格拦截。需要基于真实恢复证据处理，不能直接放宽或重新训练。

## 2026-10-21 23:17 更新

- 两个 seed 的 A/B 短验收均通过。每对初始 base / RNG / SID 哈希相同，推荐器与 Tokenizer 交替更新、冻结侧不变、梯度与恢复检查均通过。
- 两个短验收容器均有 SuccessfulDeletePod 释放事件；验收记录见 `SMOKE_ACCEPTANCE.json`。
- 四组正式训练于约 23:17 成功提交；任务名见 `FORMAL_SUBMISSION.json`。提交前清单确认没有同名正式任务。
- **禁止再次提交首次正式矩阵。** 后续先查询四任务状态及共享盘 `delivery/read_progress.py --mode formal`。
- 只读进度脚本位于远端 Goal 的 delivery 目录，未写入冻结 code/engine。

## 2026-10-21 23:19 更新

- 四个正式 Job 均为 RUNNING，每个独占 1 H100，节点分别为 gpu476 / gpu532 / gpu170 / gpu168（以实时平台状态为准）。
- 正式初始化凭据已读取并保存 `FORMAL_INITIAL_PROGRESS.json`：seed2025 内 A/B 的 base / RNG / SID 完全一致，seed2026 内也一致；均匹配各自先前 smoke。
- 四组正在完整初始验证，尚未完成第一个训练 epoch；不得将 RUNNING 解释为已经产出新推荐指标。
- 后续重点：首批 epoch 实测速度、每次 Tokenizer 阶段是否正常、各组完整 45 logical epochs、最终两 beam 验证及容器释放。

## 2026-10-21 23:26 更新

- 四组持续运行。最近共享盘快照 `FORMAL_PROGRESS_2328.json`：2025 A 已完成2轮、正跑第3轮；2025 B、2026 A/B 已完成1轮、正跑第2轮。初始验证均完成，未见错误。
- `postprocess/` 工具已通过作者及独立 reviewer 的10项 CPU 检查（其中3项真实 Torch），并上传到远端 Goal/postprocess；8份文件逐一 SHA 一致，见 `POSTPROCESS_UPLOAD.json`。**实际备份还未开始**，必须四组训练及两beam验证全部完成后打包。
- CPU打包建议输出到 Goal 外的 `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/backup_20261021_multiseed_r1`；入口及verify/restore-git/checkpoint读回命令见 `postprocess/README.md`。完整成功标志为 PACK_COMPLETE.json。
- 本地 CPU Torch 环境 `.runtime/continuous_env/Scripts/python.exe`，可用于最终所有 checkpoint 的 CPU 读回；禁止用系统 Python 缺 Torch 而跳过。
- 本地C盘此时约37.3GiB空余，远端数据树当前约4.2GiB（仍增长）；最终下载前以实际 PACKAGE/原始体积复核本地空间。不要删除历史资产以腾空间。
- 最新一次平台四任务查询 `FORMAL_STATUS_2320.json` 均为 RUNNING；随后的真实训练日志和阶段记录持续推进。不要因下一次观察超时重启任务。
- 早期 snapshot 文件名末尾的时间是手动标签，精确时刻以平台记录、文件修改时间和收据UTC为准；`FORMAL_PROGRESS_2328.json` 实际保存于23:25:52，随后 `FORMAL_STATUS_2328.json` 保存于23:26:35。

## 正式首个 Tokenizer 阶段检查

- 记录时间：2026-10-21 23:35:21（北京时间）。
- seed2025 control 已完成6个logical epochs，推荐器520次更新，Tokenizer首次6次更新完成；推荐损失至码本的直接梯度norm=0.3565230667591095；冻结参数不变，梯度有限，catalogue mapping与最近中心分配检查均通过。已回到epoch7推荐器训练。
- 其它三组仍在训练或第5轮后的验证，不提前宣称它们的正式Tokenizer阶段已验收。
- 新增只读诊断 summarize_sid_collisions.py 已以CPU真实旧checkpoint复核旧2024 A/B结果；新结果缺失会拒绝生成报告。旧两组均12101商品、10651唯一SID、2171碰撞商品（17.9407%），重复超额1450（11.9825%），最大组40。最终对新六模型一并核对，仍不冒称item-level推荐指标。
- 已保存四个正式Job的单任务完整启动事件：FORMAL_START_EVENTS_2025_control / 2025_item-local / 2026_control / 2026_item-local.json。含各自Scheduled/Started时刻，用于终态后核对实际资源时长。


## 2026-10-21T23:40:56 首次正式交替更新均通过

- 四组首个Tokenizer阶段（logical epoch6）都实际完成6次更新，推荐直接梯度非零、冻结参数不变、梯度有限、SID刷新检查通过。证据 FIRST_TOKENIZER_ACCEPTANCE.json 绑定原始进度快照。
- 此刻已完成逻辑轮数：2025 control=10, 2025 item_local=6, 2026 control=6, 2026 item_local=7。全部继续训练，尚未达到完整45轮或最终beam32/256评估。
- 两个短验收完整事件已保存SMOKE_FULL_EVENTS_2025/2026.json；Scheduled至SuccessfulDeletePod分别220秒和249秒，合计469 GPU秒（0.1302778 GPU小时），这是资源占用口径，不是平台结算。

## 独立结果解释审阅

- 核对时间：2026-10-21 23:49:57（北京时间）。四个正式平台任务仍为 RUNNING，各1张H100；最新训练快照完成10–14/45阶段，已发生的Tokenizer更新检查通过。
- 新增2025/2026是本轮主要复验证据。2024参与提出问题，只作历史探索参考；三seed合并只称描述统计，不能称三次独立确认。
- 关注每个新增seed的配对B−A方向和幅度；跨seed样本标准差不是置信区间，也不能用均值大于标准差代替显著性检验。
- best由同一验证集beam32选择，再在同一验证集比较beam32/256；即使方向一致，也只支持当前暖启动和选模协议下的初步复现，不声称测试泛化或独立端到端预训练复现。
- SID碰撞总量相同也不能排除碰撞成员变化；SID级增益尚不等于真实商品级增益。

## 2026-10-22 00:06:16 观察连接恢复

- CPU Notebook进度读取一次以255退出，提示cached target unavailable；同一时刻平台四个正式Job均为RUNNING。
- 官方 notebook connection refresh 成功，随后原四组共享盘进度恢复读取。未重启、重提或修改GPU作业。
- 最新完成阶段依次为2025 control=23、2025 item_local=17、2026 control=18、2026 item_local=19；已完成Tokenizer检查全部通过。尚无最终双beam验证结果。
- 23:55核查seeds产物约4.48GiB，本地C盘空余36.87GiB；最终仍须以PACKAGE实际大小复核完整下载和解包所需空间。
- 独立只读审阅summarize_sid_collisions.py未发现实质问题；此处是脚本审阅，不代表尚未产生的新模型统计已经验收。


## 2026-10-22 00:38:38 后半程进度

- 最新共享盘快照：PROGRESS_20261022_003820.json。
- seed2025 control：39/45阶段，推荐器更新3120，Tokenizer阶段9，最终验证完成=False。
- seed2025 item_local：31/45阶段，推荐器更新2496，Tokenizer阶段7，最终验证完成=False。
- seed2026 control：32/45阶段，推荐器更新2600，Tokenizer阶段7，最终验证完成=False。
- seed2026 item_local：32/45阶段，推荐器更新2600，Tokenizer阶段7，最终验证完成=False。
- 四个原正式Job持续运行，未重启、未恢复续训、未改执行代码；不能在完整训练和双beam结果未齐时提前比较或完成Goal。
- 进度查询现在也展示实际子进程children与best_epoch，训练完成后可观察评估自动启动；原始快照一直保留完整字段。


## 2026-10-22 01:02:18 首组正式完成

- seed2025 control 已完整完成45逻辑阶段、3640推荐器更新、60 Tokenizer更新和10次SID刷新；best epoch25。两beam验证各22363用户完成，beam32原选模指标精确重现。
- 轻量收据已下载至 outputs/stage07_item_local_joint/multiseed_20261021/light/seed2025_control；收据formal_completed，两个子进程退出0，清理后无本任务进程组存活。完整逐文件、逐用户与checkpoint验收待全量备份。
- 平台确认00:58:42 SuccessfulDeletePod，首组GPU释放；另外三组仍RUNNING。终态事件 FORMAL_FINAL_EVENTS_2025_control.json。
- 后续仅跟进原任务，不重复提交；全部8个验证单元完成及GPU释放后开始CPU全量备份。



## 2026-10-22 01:23:00 全部训练和评估完成，正在全量备份

- 四组均完成45逻辑阶段、3640 R更新、60 T更新、10次SID刷新；8个验证单元各22363用户完成，无新测试集评估。
- 四个正式Job均已获得退出0和SuccessfulDeletePod证据；最终删除时间：2025 A 00:58:42、2025 B 01:21:20、2026 A 01:21:29、2026 B 01:18:41。所有本次GPU已释放（两个smoke此前释放）。
- 轻量文件已全部下载并核对wrapper引用SHA/大小，收据formal_completed、children退出0、无存活本任务进程组、双beam绑定同checkpoint和SID。LIGHT_METRICS.json保存原始精度指标。尚未完成逐用户与实际checkpoint本地全量验收。
- 初步结果：2025两beam B均提升；2026 beam32 HR/NDCG略降，beam256 HR完全相同、NDCG只+0.0000011653。因此暂不支持稳定增益。best分别25/19/25/25，均按既定规则选模。
- CPU全量打包已启动：remote backup_20261021_multiseed_r1；本地exec session42649，输出PACK_REMOTE.log。初报840源文件、5497790466原始字节（未含Git归档）。不可重启重复pack；等待PACK_COMPLETE后下载、验证、Git新目录恢复和CPU checkpoint读回。
- 本地目标 package / verified / restored_git 位于 outputs/stage07_item_local_joint/multiseed_20261021，下层目录当前未创建，C盘空余约36.12GiB。Goal仍active。



## 2026-10-22 01:31:29 全量分卷下载完成

- 远端pack成功，PACK_COMPLETE已核对。855文件，压缩4872019271字节，10分卷；archive SHA 4583a41ff271e7f08c5b9984d4904fb708da544b04010897e860d3073fad99de。
- 10分卷已全部下载，SCP均退出0；1至6卷已单独校验SHA/大小，其余随整包verify统一核验。没有失败重传。
- 本地verify已运行，exec session87796，目标 outputs/stage07_item_local_joint/multiseed_20261021/verified。不要重复启动或覆盖。
- 四个正式任务最新平台状态全部SUCCEEDED；六次分配含smoke均已释放。RESOURCE_USAGE.json/md保存28696 GPU秒=7.971111小时及15份证据SHA，非结算账单。
- 下一步：等待verify成功，再独立并行运行restore-git、CPU checkpoint审计、summarize_results和SID碰撞统计；最后研究解读、独立验收、保存离线分析脚本/记录并完成Goal。



## 2026-10-22 01:40:15 数据验收通过，源码恢复补充处理中

- 855文件、整包和逐文件SHA、重新解包全部PASS。CHECKPOINT_READBACK.json实际CPU状态读回PASS；report/SUMMARY.json逐用户重算PASS；report/collisions/SID_COLLISIONS统计PASS。report/FINDINGS.md给出单seed差值和边界，REPORT开头已明确2026大beam近乎持平，不能用NDCG 2/2严格正值称稳定。
- 原restore-git的长路径问题已定位：Windows路径最长315，原helper禁用了global/system配置。后续恢复必须显式启用core.longpaths。
- 进一步发现engine源仓库本来是浅克隆；原bundle头没有浅边界，bundle verify成功不能证明父历史完整，fresh fetch报缺13978a...。原package保持不动，原source/engine_git_audit.json的完整历史解读将由补充收据更正。
- 远端真实engine/.git/shallow已只读下载到source_supplement/engine.shallow，41字节，SHA b19ce47d24ae7eea13296c35490344a804223b919bddf5f4f7be2d55377e5916，边界ad7b971c7e525e9fea6fb8e362a5c49dccb2473c。远端HEAD1c87925...、shallow=true、status干净已再次确认。SOURCE_SHALLOW.json绑定原package/manifest/bundle/audit的SHA。
- /root/beam_trace_impl正仅新增postprocess/restore_git_with_shallow.py及小回归；小fixture已验证unbundle->写回真实shallow->checkout可clean/fsck。等工具完成后运行全新目录恢复，确认原有浅历史完整恢复（不是完整上游历史）。
- 已保留3个失败诊断目录：本交付/restored_git、outputs/multiseed_git_restore_r1、本交付/restored_git_longpaths。不得把它们当最终恢复。没有修改冻结代码/数据、没有新GPU。Goal仍未完成。


## Final acceptance completed

- FINAL_ACCEPTANCE_PASSED: 4 new alternating-joint runs and 8 validation cells completed, paired starting state and full budgets verified. No new test evaluation.
- All 6 GPU allocations released successfully; observed total 28,696 GPU seconds (7.971111 hours), not a billing statement.
- Original 10-part archive and 855 extracted files verified; 56 checkpoints read on CPU; exact frozen source and original shallow history independently restored in restored_git_verified. No relocated training resume was claimed.
- Final source review supersedes the earlier pending restoration status. 39 posthoc files preserved in analysis/. No restoration, transfer, or GPU work remains.
- Delivery: outputs/stage07_item_local_joint/multiseed_20261021/DELIVERY.md and FINAL_ACCEPTANCE.json. Goal is ready to mark complete.
- Goal tool confirmed status=complete after final acceptance; elapsed 13,333 seconds (about 3 hours 42 minutes).
