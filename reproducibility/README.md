# 01–04 阶段：复现与资产入口

此前归档整理不只是整理文件名，而是把 **结果身份、原始证据、代码快照、历史 SID、数据映射和模型权重** 关联起来；当时原目录和压缩包没有删除、移动或覆盖。05 保持其已封存的 [独立入口](../experiments/05_crosspse_joint/reproducibility/README.md)。

归档基线已保存于 `32f3b4e`。当前实验包已整体迁至 `experiments/`；原始证据、历史 README 中的旧路径保留，当前入口和包定位已更新。`outputs/`、原迁移压缩包和历史源码快照不因目录调整改变。

日常复现与首次历史导入现已分开，操作总览见 [B1工作流](WORKFLOW_B1.md)。两个日常 `build` 命令已退役，不会再通过当前 HEAD 重写封存记录。

## 1. 哪些已经完成，哪些没有

本页表格只说明 01–04 复现入口；2026-10-09 已完成服务器运行资产独立迁移及 **05 的一组 Cross Joint v2 GPU 短测**，详见 [迁移收尾与验收范围](../docs/guides/EXPERIMENT_WORKFLOW.md#六2026-10-09-迁移收尾与验收范围)。这不把 01–04 的 GPU 短测、正式重训或历史权重独立评估自动改为“已通过”。

| 层次 | 当前状态 |
|---|---|
| 查原始结果 / 重算报告表 | 已建立；metadata 与原报告逐项核对 |
| 源码与输入固定 | 已建立源码快照、SHA256、配置覆盖顺序、精确运行身份；不自动取“最新模型” |
| 恢复历史数据 / SID / 最佳权重 | 原迁移包内均有绑定资产，可按实验恢复，不必解压全部 |
| 固定历史 SID，随机初始化 Decoder 重训 | 提供隔离执行入口；GPU 正式重训尚未验收 |
| GPU 短测 | DiffGRM 提供有限 batch 的 smoke；尚未执行。旧 AR trainer 无 batch 截断支持，此入口明确拒绝 AR smoke，不伪装成短测 |
| 重新拟合 Tokenizer | 保存原方案与可用产物/代码，但本入口不自动重训；这会生成新 SID，应当作为另一次实验 |
| 用历史最佳权重重新评估 | 权重可恢复，尚未独立执行；表中结果仍来自历史运行记录 |

**重要边界：**历史运行没有记录完整 runtime commit 和完整 resolved config。现有快照是现在绑定的可检查复现源，不是补造的历史版本；也不保证换硬件后逐 bit 相同。AR 早期记录的参数少于 DiffGRM，因此更需要实际 GPU 验收。

## 2. 从哪里找到一组实验

| 阶段 | 配方目录 | 实验 ID 示例 |
|---|---|---|
| 01：两种 AR Tokenizer × 3 seeds | [01 recipes](../experiments/01_ar_baseline_3x256/reproducibility/recipes/) | `rqk_seed2024`、`rqvae_seed2025` |
| 02：五组 AR 内部对照 | [02 recipes](../experiments/02_ar_capacity_ablation/reproducibility/recipes/) | `pse_seed2024`、`rqk_128_256_512_seed2024` |
| 03：三种 Tokenizer + DiffGRM | [03 recipes](../experiments/03_diffgrm_tokenizer_comparison/reproducibility/recipes/) | `pse_seed2024`、`rqvae_seed2024` |
| 04：离线 Refined / Cross-PSE | [04 recipes](../experiments/04_crosspse_offline/reproducibility/recipes/) | `refined_seed2024`、`cross_seed2024` |

所有命令在项目根目录执行。只查证据和重算表格不需要 GPU，也不需要安装 PyTorch。

```powershell
python scripts/repro/project_repro.py list
python scripts/repro/project_repro.py verify
python scripts/repro/project_repro.py analyze
python tests/test_project_repro.py
python scripts/repro/stage05_repro.py verify
python tests/test_stage05_repro.py
```

`verify` 检查证据、配方、辅助代码和结果表的 hash，并重新比对原始结果。`analyze` 从保存的证据生成相同结果表，不读取“最近一次运行”。不要手工改 CSV 后再绕过校验。

## 3. 数据和权重怎样恢复

大资产不重复放进 Git；本机原迁移包已核验，服务器新主目录的独立运行副本也已通过迁移清单校验。新旧目录处于同一共享存储，不能把它们称为独立灾备；其他独立备份仍需单独确认。迁移包锁定为：

- 文件名：`experiment1.tar.gz`，4,623,792,331 字节。
- SHA256：`a74836899900bccb9d44a18f6403b4220b50615f53b26503c82825dcb2f9bc09`。
- 本机位置记录在被Git忽略的 `asset_locations.local.json`；新机器参考 [模板](asset_locations.example.json)，或显式传入 `--archive`。不依赖特定Windows路径，也不自动猜测最新压缩包。
- [sources.lock.json](sources.lock.json)：迁移包和小证据包来源；[migration_assets.lock.json](migration_assets.lock.json)：归档资产索引。

先检查目标服务器/目录是否已有同一份输入：

```bash
python scripts/repro/project_repro.py check-inputs --stage 03 --experiment pse_seed2024 --project-root /你的绝对路径/experiment1
```

缺少时，只恢复该组必需文件。已配置该机器的本机映射后可省略 `--archive`；显式参数仍优先。下面在项目根目录执行：

```powershell
python scripts/repro/asset_locations.py check
python scripts/repro/project_repro.py restore-inputs --stage 03 --experiment pse_seed2024 --project-root .
```

附加选项：

- `--include-final`：同时恢复该次历史最佳推荐模型；它是评估资产，**不是这次从头训练的初始化模型**。
- `--include-tokenizer`：恢复该方法可用的量化产物、训练 checkpoint、embedding 等补充资产；不代表已经重新拟合 Tokenizer。

已存在且 hash 相同的文件直接复用；不同则停止，不覆盖原件。数据可能较大，请确认目标磁盘空间。完整资产依赖以各 recipe 的 `required_artifacts`、`tokenizer_artifacts`、`evaluation_artifacts` 为准，而不是只看报告目录。

## 4. 怎样隔离重跑

这批 01–04 实验都是：**使用历史固定 SID + 随机初始化 Decoder**。05 才涉及加载既有模型继续训练，不能混用。

先看计划，不启动训练：

```bash
python scripts/repro/project_repro.py run --stage 03 --experiment pse_seed2024 --mode smoke
```

在 GPU 服务器激活 `grm-exp`、完成输入校验后，显式执行：

```bash
python scripts/repro/project_repro.py run --stage 03 --experiment pse_seed2024 --mode smoke --execute
```

查看新 run 的 `completion.json`、`replay_check.json` 和 `console.log`：必须是 `status=completed_and_checked` 且诊断校验通过，不能只凭退出码0就认定验收。再决定正式重训：

```bash
python scripts/repro/project_repro.py run --stage 03 --experiment pse_seed2024 --mode formal --execute
```

**不要为了整理项目现在就重跑全部实验。** 当前先保留可复现条件，后续需要研究对照或验收时再分组运行。01/02 的 AR 没有可靠 batch 截断，`--mode smoke --execute` 会拒绝；不要用它的失败推断环境故障。可审核后运行 formal，或另开代码版本增加真正短测支持。

执行入口会：

1. 校验固定输入，使用锁定源码和证据中的配置覆盖顺序。
2. 在 `outputs/project_replays/唯一run_id/` 新建独立工作区；复制所需输入，不写历史缓存或权重。
3. 写入 `resolved_config.json`、环境、启动参数及完成状态。随机种子与历史对应，但 run_id 独立。
4. 拒绝联合训练/续训配置混入，校验已有 SID 没有被悄悄重建。

01 的均匀 RQ-KMeans 历史缓存叫 `RQKMEANS3x8`，当前锁定引擎使用 `RQKMEANS_256x256x256`。执行器只在新工作区建立字节完全相同的缓存别名，记录来源并校验 SHA256；不改原件，也不重新量化。身份以内容 hash 为准，不以名字相似为准。

这里复制的是输入文件，会额外占用空间。命令只负责固定 SID 的下游重放，不宣称是完整 Tokenizer 端到端复现。新运行结果不能覆盖这次归档的 `evidence/` 或 `results.csv`。

## 5. 源码与环境怎样锁定

| 快照 | 固定 commit | 作用 |
|---|---|---|
| project | `d614a1e3af42e94a8a09ac28801c4b58b14ae8bd` | 原配置、脚本、patch、依赖清单 |
| DiffGRM | `39e94e6153c97e59877ba02d4ef1d9a202ed5384` | AR 与 DiffGRM 的实际推荐训练引擎 |
| genrec | `b6c8b41d1d20a812590dae11408b69b1ed2a8313` | RQ-VAE 等历史代码依赖，不是 AR 推荐训练入口 |

快照位于 [code/](code/)，hash 位于 [code_versions.json](code_versions.json)。新增整理脚本另由各阶段 `package_integrity.json` 锁定，不能误称包含在旧 commit 内。

[environment/](environment/) 保存原服务器导出的 conda/pip 记录。导出里的旧绝对路径只作证据，不要原样覆盖共享机器环境。GPU/驱动、Python 和依赖版本应在新的独立环境验证；已有 H100 迁移环境不因本次整理被修改。

## 6. 原文件与衍生文件的区别

- `evidence/`：按字节复制的 metadata、诊断、报告、配置、Attention 和 Tokenizer JSON/CSV；源路径和 hash 在 `evidence_manifest.json`。
- `derived/`（01/02）：从九份正式 AR 控制台日志解析的 train loss / val 曲线，有日志 hash 和来源记录。它们不是历史原生 CSV。
- `results.csv / summary.csv`：重新计算的汇总。三个 seed 用样本标准差；单 seed 留空。
- `analysis_checks.json`：历史最佳轮数是否确由验证指标选出，逐位置数据是否齐全、有限。AR 核对 NDCG@10；DiffGRM 核对 weighted score。
- `previous_README.md` 与原 PDF：保留旧叙述，不作为更新后结论的优先入口。
- 03 的 PSE 利用率补充来自 04 summary 的 `base_pse`，另存辅助证据；不假装原 PSE metadata 已记录该字段。

原 `build` 会重新生成锁和衍生表，现已在日常入口中退役。首次历史导入只通过 [独立入口](../scripts/legacy/README.md)，默认计划，禁止覆盖已有或半成品包。发现 hash 不一致先查原因，不可删除控制文件绕过保护。归档包应整体保存（根目录代码/环境 + 阶段包 + 对应辅助脚本），不能只复制一份 README。

`tests/test_daily_repro.py` 在禁止读取旧归档的环境里验证日常分析，并在临时副本重算；不证明旧导入来源和大资产可以删除。完整标准库回归见 [`tests/README.md`](../tests/README.md)。B2a只调整测试位置及封存辅助代码绑定，原实验配方、历史源码快照和资产身份不变，见 [本批记录](layout_migrations/20261008_b2a_tests_docs/README.md)。

B2b将开发层的34份配置按阶段归类；本复现入口继续读取封存证据和锁定源码，五份完整性锁与历史导入的source/configs路径均不变。当前目录校验见 [B2b记录](layout_migrations/20261008_b2b_configs/README.md)，不要为了配置迁移改历史配方。

B2c将4个复现工具整组迁至 `scripts/repro/`，5份 `package_integrity.json` 仅重绑辅助代码路径/哈希，科学 `files` 清单、历史初始化和代码锁不变。当前验收见 [B2c记录](layout_migrations/20261008_b2c_scripts/README.md)。
