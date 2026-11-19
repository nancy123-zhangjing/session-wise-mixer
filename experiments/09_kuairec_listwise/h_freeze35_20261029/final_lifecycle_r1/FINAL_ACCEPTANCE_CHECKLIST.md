# 最终验收清单：待新证据逐项签收

编制时间：2026-10-30 10:44，北京时间。此文件是收尾检查表，不是 PASS 收据；后续状态必须由相应真实文件及 SHA 证明。它不改原训练、选模、测试或统计协议，也不授权新增 GPU 工作。

依据：`../GOAL_OBJECTIVE.md`、`../BACKUP_USAGE.md`、`../PLOT_USAGE.md`、`../cold_popularity_audit_20261030/PROTOCOL.md`，以及已验收 A/B 分支证据与本目录资源释放收据。路径未另注明时，相对 `experiments/09_kuairec_listwise/h_freeze35_20261029/`。

## 当前证据边界

| 项目 | 本次编制时已知状态 | 不可据此代替的验收 |
|---|---|---|
| 三原作业与短测释放 | `final_lifecycle_r1/RELEASE.json` 已核，四次单卡分配均成功且 Pod 删除，21.810555556 GPU 小时；根任务已完成证据上传 | 模型、指标、完整恢复及真实结算 |
| 全包 | `FINAL_PACK_RECEIPT.json` 为 `packed_not_model_restored`，远端打包成功；本地下载由根任务执行中 | 已下载字节、成员解压或模型恢复 |
| A/B | 先行分支审计及 A−B 配对已验收，可按原路径与 SHA 复用 | C、A−C 主对照及最终全包恢复 |
| C | R32/R35 完整验证来源证据与 R35 训练完成通知已保存；源产物已在远端完成 | 新 C 逐样本指标、用户配对区间及独立复核 |
| 冷启动/热门诊断 | 三模型初版已独立复核并交付，`STATUS.json` 仍标 C 固定 R32 与最终 best 待补 | C 补充统计、补充报告与最终交付 |
| CPU 恢复入口 | 已只读确认 Python/Torch CPU 环境、数据/PSE manifest 和依赖文件存在/字节数；没有开始最终模型恢复 | `FULL_MODEL_RESTORE.json` 及逐状态/逐角色审计 |
| 独立统计复核 | `final_independent_review_r1/PREPARED.md`、`review.py` 已准备；未发现正式复核收据 | 任何数值复算通过结论 |

旧 `DELIVERY_READINESS_20261030.json` 是凌晨连接受阻时的历史快照，其中 C 当前状态未知、缺少远端全包等条目已经过时；保留作为历史，不能再当成最终事实源。此前 CPU 短测、源文件相同、检查点端点比较、早期 A/B 备份或目录存在均不得替代下列最终条件。

## 逐项必须满足的条件

### 1. 固定科学协议、输入与执行版本

- [ ] 最终交付索引绑定原 `GOAL_OBJECTIVE.md` SHA `74f3a37ae3a57e9602d63b62501efb46782bfc72da42c02c9e4dd3143048cab7`、冻结执行包清单、历史底座及扩展差异；不以当前 Git HEAD 替换冻结源。
- [ ] 最终归档 `goal/code/` 的实际源码、引擎、配置与执行 manifest SHA 全部通过；数据/PSE 外部文件完整 SHA 通过。
- [ ] 明确 A：R26/T200→R35/T200；B：R32/T360→R35/T360；C：原 R5/T0→R20/T200 重放→R35/T200。三组终点 R 更新均为 24,255。推荐器内部 SID embedding 继续训练，编码端及商品 SID 冻结。
- [ ] 原学习率日程、阶段编号、数据顺序、seed2024、BF16 训练、FP32/TF32-off/beam32/batch32 评估及候选聚合规则保持原值；不新增 seed、Decay、beam 或 R35 以后训练。

证据：`EXECUTION_PACKAGE.json`、`DETERMINISTIC_PACKAGE_AUDIT.json`、`INPUT_AUDIT*.json`、`H100_ACCEPTANCE_AUDIT.json`，最终归档的 `DELIVERY_MANIFEST.json`、`goal/code/`，以及最终恢复状态中的 binding/计数。

### 2. C 重放、训练完成、完整验证与锁定选模

- [ ] C 原 R8/11/14/17/20 重放关口及 R20 冻结边界全部通过原检查；不放宽容差，不把重放描述成找回已缺失旧 R20 权重的 bitwise 证明。
- [ ] C R35 的真实训练边界、结束标记、checkpoint 与后续验证完成相互一致；最终结论不只依据 Job SUCCEEDED。
- [ ] 完整验证节点齐全：A R20/23/26/29/32/35；B 父 R32 与 R35；C 原 R5 起点、重放 R8/11/14/17/20、冻结后 R23/26/29/32/35。
- [ ] A/C 仅在共同节点 R20/23/26/29/32/35 按完整单列表 NDCG@5 取最大、平局选早。当前来源预期 A R32、C R23，必须由实际 selection 与完整节点重新核对；不按测试或诊断分组选模。B R35 标识固定终点，不解释成共同网格 best。

证据：`C_R*_REPLAY_EVIDENCE.json`、`C_R20_REPLAY_FREEZE_EVIDENCE.json`、`C_R32_FROZEN_VALIDATION_EVIDENCE.json`、`C_R35_FROZEN_VALIDATION_EVIDENCE.json`、`OBSERVATION_20261030_RECOVERED_FINAL.json`；最终解压 `goal/runs/H0-Freeze20-R35/` 中实际训练日志、milestones、selection、验证 summary/预测及绑定收据。

### 3. C 测试与逐样本独立指标

- [ ] C 固定 R35 与原规则选出 best 的必要测试齐全；原有效同模型同协议结果复用。不能新增 C R5 测试或按轮次扩大测试范围。
- [ ] C 正式分支分析报告 `final_analysis/C/analysis_report.json` 为 passed，完整预测逐样本 ID、用户、五件标签、候选列表与 item 消歧/去重/PAD/历史过滤按原规则复算；验证 71,657 条、测试 53,194 条与锁定 split 一致。
- [ ] 报告单列表 HR/Recall/NDCG@5、原聚合 @10/@20/@50、候选覆盖；分清列表 HR 与逐商品命中、聚合 @50 与生成单份 50 件列表。
- [ ] 每份 summary、预测、checkpoint 来源和源码/数据绑定 SHA 齐备；输入运行前后不变；所有差异满足原容差，不能修改原产物以通过。
- [ ] 新 C 测试指标核验完成后及时独立通知，并登记 `NOTIFICATIONS.json`，不延迟到所有备份收尾。

文件：`final_analysis/C/analysis_report.json`、`REPORT.md`、`METRICS.csv`、`audits/`；实际选中的 C 测试 summary/预测。预期 C 完整数值复算覆盖 13 份模型输出：11 份验证（含历史 R5）及 R35/R23 两份测试；若原锁定选择有差异先报告诊断，不硬套预期。

### 4. 三组最终比较与用户配对 bootstrap

- [ ] 使用冻结兼容入口 `analyze_results_compat.py --all-ready`，复用 A/B 既有审计，不重做模型推理，不覆盖先行报告。
- [ ] 三组固定 R35 为主结果，A−C 为主要交互对照；A/C 共同网格各自 best 为补充；A−B 仅作不同冻结时点补充；B R32 与 R35 并列。
- [ ] A−C 固定 R35、共同节点 best、A−B 及各组授权的自身起点比较均有原协议的 2,000 次、seed2024、95% 用户配对聚类区间。C 相对 R5/R20 的自身起点只比较验证，不补起点测试。
- [ ] 用户重抽样携带其全部样本，样本平均主估计使用重抽样后的样本总数作为分母；用户等权估计单列，不能混用。
- [ ] 最终 all-ready 报告的输入/结果 SHA、A/B 缓存引用及新 C 来源全闭合；A 的旧 R23/R26 summary 缺 groups 由已锁兼容分支处理，不能改旧 summary 或绕过审核。

文件：新建的最终比较目录 `analysis_report.json`、`REPORT.md`、`METRICS.csv`、配对结果及来源清单；具体目录由根任务记录。已有可复用证据：`final_analysis/A/analysis_report.json`（SHA `66aec576ae2fe110ea593e4f83c6e24b8cca8bcac46a8595eb476f7593479937`）、`final_analysis/B/analysis_report.json`（SHA `7596546bb014f56b69d0a574d9cfd3a991e267bf667af6b80603cc6f48361a7e`）、`early_ab_comparison_20261029/COMPLETE.json`。这些不包含新的 A−C 验收。

### 5. 最终统计第二实现独立复核

- [ ] `final_independent_review_r1/review.py` 针对真实 C 分支报告和最终比较报告生成全新正式收据，status=passed、inputs_unchanged=true、torch_imported=false。
- [ ] 独立从原始预测及 source split 重算 C 全部 13 份指标；为新 A−C 配对读取 A R32/R35 的 validation/test 四份逐样本贡献。
- [ ] 独立复算 C35−R5/C35−R20 验证，以及 A−C 固定 R35和各自 best 的 test/validation 六组配对；使用原容差与原 bootstrap 设置，不调宽标准。
- [ ] 收据写清复核范围及没有重跑的 A/B 旧审计、A−B 旧配对；既有独立证据单独绑定，不声称第二实现检查了未覆盖项目。

文件：`final_independent_review_r1/` 下明确命名的新 JSON 收据；`PREPARED.md` 和脚本存在本身不算通过。

### 6. 冷启动/热门偏置 C 补充

- [ ] 保留 `cold_popularity_audit_20261030/initial_r1/` 不变；新目录接入 C 固定 R32/T200 与按原共同网格选出的 C best，相同模型去重。
- [ ] 固定旧切点 1598745441188500 微秒，验证前/后段 38,644/33,013 条，95 个跨切点目标块仍归前段；不重切数据。
- [ ] 三类训练覆盖与 head/mid/tail 训练目标频次分层、逐目标命中/DCG贡献、热门推荐份额、不同推荐商品数、推荐 Top10 集中度全部复用原定义；贡献闭合至整体指标。
- [ ] 完成 C 与冻结 H1 R32 的配对、组成/组内描述性分解、原用户 bootstrap 与独立复核；输入 SHA 前后不变。
- [ ] 中文补充报告明确等 R/T 比较与选模 best 比较的结果、限制及下一步判断；不据此改训练、选模、测试或直接启动 session 留出实验。
- [ ] `STATUS.json` 两项 C pending 均被真实完成证据解除，报告已交付并在 NOTIFICATIONS 去重登记。

文件：该目录 `PROTOCOL.md`、新输出 REQUEST/RESULTS/COMPLETE/PAIRED_BOOTSTRAP/可复算数组/CSV、独立 VERIFY、更新报告及 STATUS。初版 VERIFY/REPORT_REVIEW 不代替 C 的独立审核。

### 7. 全量训练与验证曲线

- [ ] 在 A/B/C 统计审核后，按 `PLOT_USAGE.md` 生成三组真实 R-loss、实际 `learning_rates_used.R`、完整验证 NDCG@5 曲线，不插值补造缺失 LR 或轮次。
- [ ] loss 仅用各组实际新增 R 更新：A R27–35、B R33–35、C R6–35；完整验证图保留 A 历史点、B 父 R32、C R5 与重放节点。
- [ ] 明确 R20/R32 冻结点、不同起点、A/C best；图中不混用测试指标与验证指标，不能以训练 loss 直接排序推荐质量。
- [ ] CSV、PNG/PDF 与 `PLOT_PROVENANCE.json` 来源 SHA 闭合；视觉检查图例、坐标和分支名称。

H1 单独阶段图、historical-smoke 图不能代替最终三组曲线。

### 8. 本地完整字节下载与新鲜安全解压

- [ ] 本地 tar 字节数 **6,846,443,520**、SHA **9006d8d835ed735bd79bfe382ae362ebc6266ba9c1522bf9a20e6b0dcb399f16** 与 `FINAL_PACK_RECEIPT.json` 一致。
- [ ] 嵌入 manifest SHA **b4650e450f9d367e9ae5aa8780b393c2c7a79d8f9730c1aeafb2618a559cac11** 与打包收据一致；正式下载证据只能声明实际执行的校验。若使用 SCP 后整包 SHA，不能伪称执行了 downloader 的逐块协议。
- [ ] 使用 `backup_delivery.py verify` 到全新目录 `outputs/stage09_restore/h_freeze35_20261029_final_r1_restored`；路径穿越、链接、Windows 大小写/保留名冲突、未列成员等全部拒绝；逐成员/落盘 SHA 通过。
- [ ] `BACKUP_VERIFICATION.json` 为 `bytes_and_structural_coverage_passed`，所有 **3,405 个 payload** 与 manifest 一致；实际模型恢复仍明确 pending。
- [ ] Dataset/PSE/历史预测全部外部文件重新核 SHA；保留早期 A/B 包及历史原路径，不为了省空间擅自删除。
- [ ] 将归档内 `references/lifecycle/release.json` 的 evidence 按数组序号映射到解压后的 `references/lifecycle/evidenceNNN.json` 核 SHA，不因其中仍保存原远端路径而跳过。

### 9. 全部检查点覆盖与真实 CPU 恢复

- [ ] **63 份不可覆盖完整状态＋3 份 rolling last**：A14＋last、B6＋last、C43＋last。须包含三个 R35 training boundary、C R20_pre_freeze、全部要求阶段、三起点、最终与选中 best。
- [ ] 输出“应保存节点—实际文件—SHA—阶段/下一位置—R/T计数—字段检查—真实恢复”覆盖清单。best/last 不替代历史节点，不能以文件数量代替覆盖。
- [ ] 最终 `restore` 只能在全包 verify 通过后运行，绑定真实 `BACKUP_VERIFICATION.json` SHA；使用既有 CPU Python，输出新建且在解压树外的 `final_cpu_restore_r1`。不直接调用私有 worker，不造假 receipt，不在失败后自动重跑。
- [ ] A/B/C 每组 `checked_states` 精确覆盖全部不可覆盖节点＋last，核模型/码本/Mixer（若启用）/SID、双优化器、scheduler、随机状态、形状/类型/有限性、阶段与 R/T 更新、冻结指纹及绑定。
- [ ] `restored` 精确覆盖 manifest 的全部关键 roles：三组 initial/final、C R20、原选中 best；同路径仅去重一次。要求真实模型、两优化器与 scheduler 装载后的指纹相等，CPU RNG 恢复，真实 B2×L50 历史的无目标前向有限，SID 不刷新、状态不变、updates_performed=0。
- [ ] `FULL_MODEL_RESTORE.json` 为 `real_cpu_restores_passed`，逐组明细和日志齐全；关键源 checkpoint 前后 SHA 不变，外部依赖前后校验通过。内存路径重定位有 old/new 指纹且不改变科学状态。
- [ ] CPU 明确 cuda_exact_restore=false；CUDA RNG 仅保留来源指纹，不冒充实际 CUDA RNG 恢复。本轮 H100 短测真实下一步更新、冻结隔离、连续性与诊断无扰动由 `H100_ACCEPTANCE_AUDIT.json` 独立支持。

已有 `cpu_restore_r2/`、早期 A CPU 尝试、端点冻结比较及本次环境只读检查不能勾选这些最终项。

### 10. 释放、最终台账与费用边界

- [x] 资源子项已有独立审计：`final_lifecycle_r1/RELEASE.json`、STATUS/EVENTS、ACCOUNTING_REVIEW 与 `GPU_LEDGER.json` 绑定原短测/A/B/C 四次分配；总计 78,518 GPU 秒，包括短测 219 秒。
- [x] B 后续控制器重复 Terminated 无新 Pod，不盲按任意最晚事件追加分配；C 原次收尾边界 05:00:50，Pod 删除 05:00:43，保守多计 7 秒。
- [ ] 最终报告/交付索引绑定实际上传和归档证据 SHA；核其与本地最终台账一致。已释放不等于已经删除历史产物或必须删除平台作业记录。
- [ ] 费用仅按既有记录单价估算，结算核实状态保持 false，除非取得可归属账单；预算余量不是新增实验授权。原 CPU 文件入口若仍承担下载消费，不提前关闭。

### 11. 中文报告、通知、交接与 Git

- [ ] 最终 REPORT 直接回答 H1 是否持续改善、统一 R20 冻结下交互差异、H0 从自身 best 晚期冻结后是否追平；结果无论正负都按原协议交付。
- [ ] 展示固定 R35 主比较、共同网格 best 补充、B32/B35、重要设置/起点/冻结点/更新计数；解释 R35 与 best 不一致及退化，不以验证结果冒充测试。
- [ ] 保留单 seed、用户区间不覆盖训练 seed/选模不确定性、R20 冻结点源于之前 H1 观察、C 重放不是旧权重 bitwise 找回、A−B 不能识别纯 Mixer 效果等限制。
- [ ] H1 已完成的 R29/R32/R35 验证、训练完成和测试核验通知均沿 `NOTIFICATIONS.json` 复用并去重；C 新完整验证/训练完成已通知，测试核验与 C 诊断补充按真实完成后登记。
- [ ] 更新 `REPORT.md`、`FINAL_DELIVERY_AUDIT.md`、`docs/guides/PROJECT_HANDOFF.md` 及必要资产索引，绑定完整包、SHA、恢复、科学统计、C 诊断、曲线与 GPU 释放证据。
- [ ] 说明哪些报告/收据在远端 tar 打包后新生成，只作为另行有 SHA 绑定的交付件；本检查表本身即打包后文件，不能声称已包含在既有 tar 中。
- [ ] 仅提交本任务文件的 Git commit，保留无关改动和未跟踪文件；记录实际 commit，不把“准备提交”当已提交。

### 12. 整体完成关口

- [ ] 上述原 Goal 全部要求及新增 C 冷热门补充都真实通过，无待处理缺失/失败项。
- [ ] 最后独立验收明确区分字节、结构、恢复、指标、资源和结算边界；任何单一 COMPLETE/status/SUCCEEDED 均不能替代整体核验。
- [ ] 完成用户可读最终交付后，停止 `09-h1` 跟踪并回读确认；最后才将 Goal 标为 complete。不能因 H1 阶段报告、GPU 已释放或 pack 成功提前结束。

## 供独立复核定位的固定根路径

- 仓库：`C:/Users/nancy/Documents/experiment1`
- 本轮记录：`experiments/09_kuairec_listwise/h_freeze35_20261029`
- 全包：`outputs/stage09_restore/h_freeze35_20261029_final_r1.tar`
- 新鲜解压：`outputs/stage09_restore/h_freeze35_20261029_final_r1_restored`
- CPU Python：`.runtime/stage08_restore_env/Scripts/python.exe`
- Dataset：`outputs/kuairec_listwise_v1/run1`，manifest SHA `8fe1506fbbe3361048ad9b2862415a793ecefe538a8478cc3ab0d7e2a3b9b649`
- PSE：`outputs/stage09_restore/goal_20261027_r1/assets/pse_v1`，manifest SHA `7cc6dcdfcdeb4087e38fd2080f322a01375b189b284a59d5c9bfd279862525d4`

本表不改上述冻结执行文件、不启动网络/模型、不更新状态或报告、不提交 Git。实际勾选及最终通过必须以后续新增收据为准。
