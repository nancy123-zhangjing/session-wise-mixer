# H1 R20续训执行记录

2026-10-28。Goal已启动，目标是Decay与Freeze两组到R26，随后完成验证、分析、备份与资源释放。原09四组已完成，不重跑历史固定学习率组。

> 当前有效状态（2026-10-29）：两组R26、17份独立复算、三份bootstrap、四组曲线、完整本地字节交付和两臂最终CPU恢复均已通过；最终逐项审计已通过，本轮实验交付完成，见[最终审计](FINAL_DELIVERY_AUDIT.md)。本轮GPU已释放，累计6.556667 GPU小时（已批上限8.5小时）。有效远端目录为`h1_recovery_20261028_r2`，运行扩展commit为`8b5acc8`，包SHA为`057d01605fdc6d1183ad40bfb053c5d577bdb2f9c6909d8f7210ab388ff65d3b`。先读`REPORT.md`与`DELIVERY_INDEX.json`。下文按时间保留历史过程；早期0.5小时授权、r1包及“正式未启动”等均非当前指令。

## 授权与锁定

- 方案：`docs/research/onerec_listwise/H1_LATE_TRAINING_RECOVERY_PLAN_20261028.md`。
- 父checkpoint：`outputs/stage09_restore/goal_20261027_r1/runs/H1/best.pt`，SHA256 `a672ea4d31be5281c2c24c1c7e774bc27eb7e679ec732181a197b703da890bf1`。
- R20/T200、phase25/update0、R更新13860；新终点phase33/update0、R更新18018。Decay T280，Freeze T200。
- 目前只授权累计0.5 GPU小时验收/测速；正式训练总上限尚待测速后确认。原75 GPU小时授权不沿用。

## 已完成的本地工作

- 核对当前Git HEAD为658ae7b，起始无已跟踪文件修改；保留所有既有draft目录和generalization_audit目录。
- 新增recovery控制器：显式父/执行绑定、R20分叉、T余弦后半程、冻结T跳过、原阶段编号保留、更新与边界恢复。
- 新增无副作用诊断、固定SHA样本选择、验证入口及短GPU验收入口。
- CPU汇总32项测试通过，日志`CPU_TESTS.log`；覆盖真实tiny-H恢复、冻结、R/T梯度、随机轨迹、指标等价与诊断状态保全。独立执行副本已实际恢复真实R20两组状态，见`CPU_REAL_PARENT_DECAY.json`与`CPU_REAL_PARENT_FREEZE.json`；这是CPU恢复，不是CUDA精确续训验收。
- 发现当前工作树15个原绑定文件与历史checkpoint字节不一致。已恢复归档中的40个原绑定源码全部匹配；本次以归档原字节为执行底座，仅添加新recovery模块。不得把当前源码的绑定删改后伪装成历史源码。
- 不修改原报告、原权重或恢复归档；新的执行副本、包和收据写本轮独立路径。

## 平台准备

- 账号sii-shared已实际认证成功。
- Live H100 quota显示开发区-H100-cuda12.8版本-119核提供1,10,200与2,20,400规格，单价每卡每小时1点；创建前仍需复核可用性与镜像。
- 既有零GPU CPU Notebook `nancy-stage09-prep-20261027-r1`已启动，后续status确认RUNNING、0GPU/4CPU/16GiB。该CPU入口用于文件同步。
- CLI部分查询出现间歇ECONNRESET；没有据此重复创建Notebook。文件连接已恢复。
- 训练扩展源码及32项CPU测试已提交：`e5e3198c610752eb71eaa218b21dcdffdd1b7593`。运行底座为已验证历史归档，而非当前HEAD中有字节差异的旧模块。
- 最终包SHA256：`743b0074dd61333db30778eefc05c54f6ba551fc8e136038e523610a74ab20b7`；执行manifest SHA256：`fc757724cfdaca408c104609c306088c8444c36219a7a38910e71093be96f461`。详见`PACKAGE_FINAL.json`。
- 远端独立目录：`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_recovery_20261028_r1`。REMOTE_READY校验通过：3027源码文件、40个原绑定文件、8个数据文件、内容/PSE资产manifest及R20权重均一致。
- 23:00提交短验收Job `nancy-h1-recovery-preflight-20261028-r1`：两张H100、20CPU、400GiB，单节点、每臂独占一卡、无DDP；镜像`ngc-pytorch:25.02-cuda12.8.0-py3`。平台限制0.2小时、外层脚本600秒、单臂480秒，自动容错和状态通知均关闭，成功/失败不保留容器。提交记录见`PREFLIGHT_CREATE.json`。
- 23:01首次状态QUEUING，尚未确认GPU进程启动。正式训练仍未授权或启动。实际分配时间待events核算。

## 下一项工作

用户已明确批准累计8.5 GPU小时（含本轮已耗、当前单价估算8.5点、未核实结算）。两组各单独一个1-H100 Job并行，完成后分别释放；后续须完成训练、评估复算、报告与备份验收。

## GPU验收结果与修复

- 首轮在衰减臂第二次重用R20内存快照时失败：Adam的CPU step张量在`load_state_dict`后与输入快照共享，后续更新改变了快照计数。磁盘原R20文件未变、未开始正式训练。
- 新增CPU复现测试先验证失败；恢复时复制R/T优化器与scheduler状态后，13项恢复测试全通过。修复commit：`8b5acc8853101975a0ba09fe904f990549643dcc`。首轮验收现场保留。
- 修复后独立执行目录为`/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/h1_recovery_20261028_r2`。包与源码再次远端核验通过，详见`PACKAGE_R2.json`。
- 补验Job `nancy-h1-recovery-preflight-20261028-r2`，两臂报告均passed、退出码0，共同模型/编码/优化器状态核对通过，BF16梯度、真实LR、冻结、磁盘恢复及下一随机更新、FP32输出重复性通过。验收更新全部丢弃。报告SHA已对照supervisor SUMMARY核验。
- 两次Job均有删除Pod及终止事件。保守计费口径用Scheduled到最后JobTerminated：首轮127秒×2、补验78秒×2，共410 GPU秒 = 0.113889 GPU小时；非平台结算。
- 固定8192验证子集SHA：`7359700f6483f315287d502b200375471d6bb50203dbcc43b567af8b9327f130`，两臂共用，不用于替代完整验证终点。
- 实测预算：Decay原始3.354 GPU小时，Freeze3.117；剩余加25%为8.089 GPU小时，加已耗累计8.203。两卡并行原始约3.354小时，含余量4.192小时，不含排队。单价live确认为1点/GPU小时。细目见`BUDGET_ESTIMATE.json`，已包含R23/R26完整验证与所有子集诊断。
- 等待预算确认期间无本轮GPU占用；未沿用原75 GPU小时授权。

## 正式续训已启动

- 正式launcher及预算/独立复算工具已提交`1307252`，运行模块仍为通过H100验收的`8b5acc8`及同一冻结包。launcher在CPU共享目录上完成真实3027文件、manifest和父权重校验。
- Decay：`nancy-h1-recovery-formal-decay-20261028-r1`，1 H100 / 10 CPU / 200GiB，23:32:00 Scheduled，23:32:09容器启动，平台max_time=4.3h。
- Freeze：`nancy-h1-recovery-formal-freeze-20261028-r1`，1 H100 / 10 CPU / 200GiB，23:33:51 Scheduled，23:33:59容器启动，平台max_time=4.0h。
- 均为2026-10-28北京时间，节点qb-prod-gpu181；无DDP、自动容错和容器保留。两任务状态RUNNING，应用日志确认正在运行固定8192样本的共同R20起点评估。
- 日志路径：新root下`formal_jobs/H1-{decay,freeze}/training.log`；训练输出：`runs/H1-{decay,freeze}`。两个Job名、资源和起始事件已保存为`FORMAL_*` JSON。
- 预算计数从两次Scheduled起累计，加短验收410 GPU秒。平台时限合计8.3 GPU小时，留少量余量给启动/释放；仍需按实际分配累计监控，不能把运行耗时当结算单。
- 正式固定子集：`delivery/validation_subset_8192.json`；R23/R26完整验证，不使用测试集。

## 运行中独立核查（尚非最终结果）

- 正式两臂fork记录的parent_hashes与execution_binding完全一致。
- 已下载两份initial_R20及Decay after_T240既有预测，逐样本CPU复算通过，指标误差最大4.77e-8；没有新增GPU推理。8192条样本的两臂R20商品输出逐条一致，且均与旧R20对应输出一致。
- 仅在固定8192诊断子集：R20 NDCG@5=0.008266785869486858；第一块40T结束但尚未训练R时=0.007424406535886746。它们不是完整71657验证的主终点结果。
- 编码诊断T200→T240：使用数181/137/190/218→161/122/159/184；独立SID9600→9630；碰撞商品比例12.9101%→12.5%；交互修正/基础码本范数0.823687→1.162784。码字使用下降尚未伴随大面积SID碰撞，不能据此单独判定改善或塌缩。
- 已准备独立复算/三组bootstrap、PNG/PDF曲线、带逐文件和64MiB块SHA的备份、冷导入归档源码的CPU恢复验收工具。备份测试6通过/1因Windows symlink权限跳过，恢复测试5通过；正式模型备份及R26恢复仍待训练完成。
- 本轮末次日志：Freeze已进入R22；Decay正在R21结束后的8192样本诊断。训练继续按原协议，未改变设置或扩大实验。

## 2026-10-29 00:25运行观察

- 连续6次平台查询（00:15—00:20）两原Job均RUNNING；最后应用日志均已进入R23训练。不存在已确认的终态或重启。
- R21固定8192子集摘要：Decay NDCG@5=0.00665940096405393，Freeze=0.008220432933740085；本次读的是已完成摘要，最终仍要独立复算并以R26完整验证为主。
- 首小时平台worker-0聚合GPU利用率：Decay 48个1分钟采样平均22.4583%、最大39.5%；Freeze 46点平均25.2337%、最大40%。显存约6%—7%。日志更新正常，利用率低不等于停止训练。
- Live规则再次确认为Job `gpu < 40% for 3h`。目前不足3小时且没有回收事件；摘要平均值不足以证明平台一定何时触发。实际配置`checkpoint_every_updates=100`，另有每T/R边界保存，保留现有设置。
- 若后续出现终态，先检查原Job events/logs并保存实际用量，不能因观察超时重启。仅在确认终态、故障原因和剩余预算可覆盖后，才考虑从同源last.pt显式resume；存在`.running`锁时先保留其owner证据，再归档已核实失效的锁，不能对活跃任务移锁。

## 2026-10-29 R23完整验证观察

- 两组均完成71657条完整验证；逐样本CPU独立复算通过，各4733用户、0不一致项，最大数值误差小于5.51e-8。归档SHA、两组预测与摘要receipt SHA、安全解压和8192子集提取均已核验，见`R23_EVALUATION_AUDIT.json`。
- 完整验证NDCG@5：Decay `0.00674910534`，Freeze `0.00962905195`；共同R20历史完整预测为`0.00937205572248708`。相对起点分别约-27.99%、+2.74%。
- 完整验证HR@5：Decay `0.03091114615`，Freeze `0.04520144578`；按既有多列表聚合规则的NDCG@50：`0.01205758305`、`0.01442404358`。
- 仅8192诊断子集NDCG@5：Decay `0.00589538349`，Freeze `0.00899271429`；不可与上面的全量值混用。
- R23是预先规定的过程观察，不据此改参数、提前选模或替代R26主终点；尚未完成终点bootstrap、最终模型备份和资源释放验收。当前结果提示冻结编码可能更利于保住R20效果，不能据此宣布长期稳定或测试泛化改善。
- 01:27左右应用日志：Decay已完成T280、进入R24；Freeze为T200、进入R25。原两Job继续运行，未重启或扩大实验。

## R26完成与交付进行中

- 两臂全部训练及规定验证完成。Decay为R18018/T280/phase33，Freeze为R18018/T200/phase33，无正式训练重启。
- Freeze 02:41:22正常退出0，02:41:26删除Pod；Decay 02:50:49正常退出0，02:50:52删除Pod。所有时间为2026-10-29北京时间。四Job最终状态和删除/终止事件见`FINAL_PLATFORM_STATUS.json`、`FORMAL_*_FINAL_EVENTS.json`及既有预验收events。
- 以Scheduled到最后JobTerminated计，包含两次预验收共23604 GPU秒=6.556667 GPU小时。正式并行墙钟3小时18分56秒；预算余量1.943333小时。按1点/GPU小时估6.556667点，平台账单未核实，见`FINAL_RESOURCE_ACCOUNTING.json`。
- 02:04确认原零GPU CPU Notebook为STOPPED，启动后文件入口重新连通；本轮未单独核验该次停止原因。GPU Job独立运行，未受此入口重启影响。
- 最终远端备份目录为新root下`backup/final_package_20261029_r1`。归档1722453940字节，SHA `607f9124b9cb5f6c8e906e92a9eaf200b40cfc5f3cba8ed7e8ea0d1feb6c83cd`；manifest SHA `f6dee18b9931c9efe9798211034e8c7205c5a405829f877c781828f299efbaec`。本地下载至`outputs/stage09_delivery/h1_recovery_20261028_r2`，独立恢复目录预定`outputs/stage09_restore/h1_recovery_20261028_r2`。远端打包成功不等于本地交付完成。
- 预测摘要及其receipt SHA已核，见`REMOTE_ENDPOINT_SUMMARIES.json`；R26逐样本独立复算、置信区间、最终恢复检查尚待本地解压完成。`INTERPRETATION_AUDIT.md`记录两策略的因果解释边界。

## 最终本地交付与分析

- 03:12:08归档下载完成，03:18:06独立整包/26块/内外manifest/3155文件SHA与安全解压验收通过。两臂initial、last和各8个阶段边界checkpoint全部齐全。见`LOCAL_DELIVERY_AUDIT.json`。
- 两臂从认证归档源码各独立冷启动CPU恢复通过。模型、双优化器、scheduler、计数和CPU状态roundtrip匹配，数据/PSE内容与科学配置一致，仅显式重定位数据/资产路径；原checkpoint/binding不改写。Freeze alignment及T优化器与R20完全一致。CPU最终恢复不等于CUDA精确轨迹复验。
- 修复后13项CPU回归的原始执行输出已从当前线程journal封存至`CPU_RECOVERY_R2_TESTS.log`及溯源JSON。时间23:07:45，早于r2 H100验收，包含快照alias回归；与旧32项`CPU_TESTS.log`、修复前`CPU_REAL_PARENT_*`分别标明，不冒充同一版本。
- 最终独立复算17份、456589条记录全部通过，最大误差6.72e-8；3组2000次用户聚类配对bootstrap完成。R26全量NDCG@5：Decay0.0061593003，Freeze0.0103645345；共同R20为0.0093720557。Freeze−R20=+0.0009924788，95%区间[+0.0006417823,+0.0013198052]；Decay−R20=-0.0032127554。
- 四张正式PNG均经可读性检查，4张PDF及全部文件SHA复核完成。完整验证和诊断子集分面，历史原LR轨迹标为参照，不当作重跑控制组。
- `REPORT.md`、`SUMMARY.json`、`METRICS.csv`、`ACTUAL_CONFIG.json`及`DELIVERY_INDEX.json`汇总结果和依赖。新归档不自包含数据/PSE及旧R20预测，须保留索引中的原09本地资产；没有删除历史文件、重训基线或追加试验。

## 最终验收与交接（2026-10-29）

最终独立逐项审计 PASS，无实质缺口，见 [FINAL_DELIVERY_AUDIT.md](FINAL_DELIVERY_AUDIT.md)。实验结果与交付证据已提交 39624bd；本次仅补充审计及三处结项文字，REPORT、SUMMARY、DELIVERY_INDEX 保持验收时字节。两组固定 R26 结果、全部备份与实际 CPU 恢复已完成，本轮 GPU 全部释放，累计 6.556667 GPU 小时。提交本审计后由协调者调用 Goal 完成工具；没有新增训练或扩大实验。
