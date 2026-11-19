# 步骤 1：当前代码在旧划分上重跑 H 冻结对照（2026-11-19，事先登记）

- **批准：** [2026-11-18_3 第 56 轮](../../../docs/discussions/2026-11-18_3.md#turn-56)，用户原话：“好，做步骤1”。
- **背景：** [步骤 0](../mixer_reversal_check_20261119/STEP0_CODE_DIFF.md) 发现，训练代码和 Mixer 代码在两轮之间没有实质改动。剩下的代码侧差别只有执行优化（fused AdamW、张量缓存、微批 1024 等）和运行中冻结的实现方式。

## 要回答的问题

用当前代码（含 v2 的执行设置和运行中冻结）在旧划分上训练，能不能重现首波的 Mixer 增益？

- **能：** 代码不是“趋势相反”的原因，原因在数据或任务定义，包括划分方式、列表定义、历史长度和训练规模。
- **不能：** 数值层面的执行差别也能翻转结论，需要逐项排查。

## 设计

除下表所列，其余与首波旧划分的冻结对照完全相同。旧配置逐项核对过：`h_old_reference(seed)` 与 `multiseed_freeze_20261109_v1/config_seed{2025,2026}.json` 完全一致，与 `h0_multiseed_r20_freeze_20261110_v1/config_seed2026.json` 也一致。

| 项目 | 首波旧划分 | 本对照 |
|---|---|---|
| 代码 | 旧代码树 `h_freeze35_20261029_r1/code`（`e4ca4b3` 一系） | 提交 `709ba67`，快照 `code_709ba67`（SHA-256 前缀 `e00ea8529e37beb8`）；H 训练代码与 v2 的 `7311328` 相同，DiffGRM 引擎与 `ar_deps` 复制自 `code_7311328` |
| 数据、资产 | `goal_20261027_r1/inputs/dataset`、`assets/pse_v1` | 相同 |
| 历史长度、列表长度 | 50、5 | 相同 |
| 每单位更新数、单位数、暖身、码本日程 | 693（一遍）、35、5、每 3 个单位 40 次 T | 相同 |
| 冻结 | R20 后冻结（seed 2024 为续跑，2025、2026 为单次运行中跳过 T） | R20 后运行中冻结（与 v2 相同的实现） |
| 验证与选点 | 每 3 个单位一次，曝光 NDCG@5，单点最佳 | 相同 |
| 评估 | 不屏蔽历史，beam 32 | 相同 |
| 执行设置 | 无，微批 256，评估 batch 32 | v2 模板：fused AdamW、张量缓存、批量 beam 前沿，微批 1024，评估 batch 128；有效 batch 都是 1,024 |
| 精度 | bf16 | bf16 |
| seed | 2024、2025、2026 | 相同 |

- **配置差异（相对首波）：** 只有 `execution.*`、`training.microbatch_size`、`training.freeze_after_epoch`、`training.checkpoint_every_updates`、`evaluation.batch_size`，以及标记字段 `h_split_control`。单元测试 `OldSplitControlTest` 锁定了这一点。
- **默认值不变：** 改动前后，v2 各组（H、TIGER、CF、SASRec）、节奏对照和冒烟的 54 份配置逐字节相同。
- **作业：** 与 v2 相同的布局，h-a 为 4 卡（seed 2024、2025），h-b 为 2 卡（seed 2026）。
- **远端根目录：** `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/oldsplit_v2code_20261119/`。
- **预计时长：** 首波每次验证（71,657 个样本）约 39 分钟，每次测试（53,194 个样本）约 30 分钟，单臂全程约 16 小时。本对照用了批量 beam 前沿和更大的评估 batch，预计 10–14 小时。作业上限 24 小时。

## 事先登记的分析与判读

- **主比较：** 每个 seed，测试集曝光 NDCG@5（不屏蔽，首波的主指标）H1F − H0F，在验证最佳点和 R35 两个选点上计算。测试集每个用户有多条样本，按用户聚类 bootstrap 2,000 次。
  - 首波的对应值：best 为 +0.00096 / +0.00119 / +0.00049，R35 为 +0.00070 / +0.00212 / +0.00143，6 个区间都不含零。
  - 辅助：EV-NDCG@5（阈值 1），首波 6 个配对中 5 个显著为正；冻结后的验证轨迹。
- **工具：** `scripts.mixer_reversal.analyze old --rerun-root <根目录>`，与首波的重算用同一套函数。
- **判读：**
  1. **重现：** 两个选点下三 seed 均值都为正，至少两个 seed 在两个选点下都是 H1F > H0F，且 6 个配对中至少 4 个显著为正。
     - 结论：当前代码能重现旧增益，“代码”不是原因，原因在数据或任务定义。
     - 下一步：设计数据变体，逐项区分划分方式、列表定义、历史长度和训练规模（另行批准）。
  2. **不重现：** 两个选点下三 seed 均值都不为正，或多数 seed H1F ≤ H0F。
     - 结论：执行层面的差别足以改变结论。
     - 下一步：在旧划分上逐项关闭执行优化排查（另行批准）。
  3. **介于两者之间：** 照实报告，不下结论。
- **边界：**
  - 区间只覆盖测试用户抽样，不覆盖训练随机性。
  - 与首波逐 seed 比较时，同一 seed 也不会逐位相同，因为 fused AdamW 和微批大小会改变数值舍入。所以只比较方向和量级。

## 执行顺序

1. CPU 核对：在 env-zzl 上用快照运行 `validate_config` 和单元测试，并对作业做 dry-run。
2. GPU 冒烟：`h-b` 布局，`--smoke`（每单位 60 次更新、R2 后冻结、beam 4），写到 `smoke/`，不作为结果。
3. 冒烟通过后，正式提交 h-a、h-b，再检查日志和 GPU 利用率。若平台回收作业，由守护脚本从 `last.pt` 续跑。
4. 测试全部完成后，自动运行分析，再写报告。

## 执行记录

- **CPU 核对（11:35 前后，env-zzl，快照 `code_709ba67`）：**
  - 三个 seed 的正式配置和冒烟配置都通过 `validate_config`；旧数据和资产目录存在。
  - 单元测试 `OldSplitControlTest`、`HFreezeTest`（含 CPU 上运行中冻结的端到端测试）、`LockedConfigTest` 和作业布局测试通过。
  - `BudgetTest` 的 TIGER 部分在快照里报错，原因是快照没有带 TIGER 的设计文档；它与本作业无关，本地仓库中通过。
- **dry-run：** 通过。提交命令仍带 L5O 的 `--dataset/--assets` 参数（作业脚本要求），但旧划分模式下配置固定使用旧数据路径，不读取这两个参数。
- **GPU 冒烟：** `nancy-oldv2-smoke-h-b`，11:41 提交，记录见 [SUBMIT_smoke.json](SUBMIT_smoke.json)。
- **守护与分析脚本：**
  - [guard_oldsplit.sh](../../../scripts/mixer_reversal/guard_oldsplit.sh) 由节奏对照的守护脚本改出，只换了作业名、根目录、代码和测试节点（best、final）。
  - [run_oldsplit.sh](../../../scripts/mixer_reversal/run_oldsplit.sh) 等 12 个测试摘要齐全后运行 `analyze old --rerun-root`。

## 冒烟结果与暂停（2026-11-19 12:30 补记）

- **冒烟：** `nancy-oldv2-smoke-h-b` SUCCEEDED，12:24 前后结束。
  - 两个槽位返回码都是 0，日志里没有 traceback。
  - 锁定配置带 `h_split_control`，不屏蔽历史，按曝光 NDCG@5 选点，历史 50。
  - 暖身、训练、运行中冻结，以及 best 和 final 两个测试都已产出，测试覆盖完整的 53,194 个旧测试样本（beam 4，仅冒烟）。
- **正式作业没有提交：** 用户在第 57 轮暂停了步骤 1。第 58 轮的数据因素设计改为从 L5O 一侧出发，不需要本对照，见 [DESIGN.md](../data_factor_design_20261119/DESIGN.md)。代码和快照保留；以后如有需要，可按本计划直接提交。
