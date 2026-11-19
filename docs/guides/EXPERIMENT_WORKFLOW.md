# 后续研究的项目组织与版本管理

## 一、以后只认三层，不再到处找“最新版”

| 层 | 放什么 | 平时怎么用 |
|---|---|---|
| 开发层：`configs/ scripts/ external/` | 将要跑的代码与配置 | 用 Git 分支修改和审查 |
| 证据层：`outputs/ data/ archive/` | 原始记录、数据、权重、历史压缩包 | 不手工改数，不按目录名字判断可删 |
| 实验包：`experiments/01...05/` + `reproducibility/` | 汇报、run 索引、证据副本、资产与代码锁 | 查已有结论、复现、与新实验对照 |

01–05 已按照这个关系整理。B2a 将 12 个既有测试迁入 `tests/`，文档分为现用指南、方法和历史记录；B2b 把配置分为 `configs/shared/` 和 01–05 阶段目录，现用脚本同步地址，参数和加载顺序未变。B2c进一步将脚本按职责分类，详见 [项目地图](PROJECT_MAP.md)；缓存、原始运行目录和历史源码不变。

B1已经将历史导入和日常复现分开；先看 [具体操作](../../reproducibility/WORKFLOW_B1.md)。旧 `build` 已退役，不要靠重写锁“修好”校验。迁移包位置只写在被忽略的本机配置中，不修改历史身份。

## 二、开始一轮实验前

1. 写一句研究问题：这轮要排除哪个解释？例如“同一初始化和训练量下，收益来自额外训练还是 Tokenizer 更新？”
2. 确认基线的 `run_id + seed + checkpoint + SID hash`，从阶段 recipe 找，不用 `ls -t | head` 自动取最新。
3. 在干净 Git 状态创建分支，例如 `codex/exp-matched-control`；先保存现有改动，再做新实验。
4. 在 [对应阶段的配置目录](../../configs/README.md) 新建配置，不覆盖旧版本；共享层只引用、不复制。注明继承配置及顺序、唯一 run_id、seed、变动变量、不变变量、选模指标与评估口径。
5. 让 AI 给出改动说明和测试；自己看 `git diff --stat`、`git diff`，或 IDE 的 Source Control 对比。确认后提交，再上服务器训练。

实际训练引擎在 `external/DiffGRM`，是独立 Git 仓库：只提交主仓库不会保存它的修改。若改引擎，必须同时记录其 commit 或可应用的 patch 与基准 commit，再更新本次实验清单。不要把根目录 commit 当作全部代码版本。

## 三、运行时至少存这些

每次 run 至少保存：

- 根仓库和上游仓库 commit，是否有未提交改动；有改动时保存 diff，不只写一个 commit。
- 完整合并后的 `resolved_config`、命令、seed、环境版本、GPU。
- 数据 / item mapping / embedding / SID / 初始化权重的 SHA256。
- train、validation 的完整曲线，逐位置诊断，最佳轮数和**选模依据**。
- 最佳 checkpoint、最终 test 指标、Tokenizer 更新记录与必要 Attention 数据。

Windows 本地用于读报告和改代码；CPU 开发机用于同步、检查和准备；GPU 作业用于正式训练。模型和数据放共享持久目录，作业临时空间不能是唯一副本。AI 可以协助检查、生成配置和汇总，但启动正式长训练、改变实验问题仍应由你明确确认。

## 四、跑完以后

1. 核对 run 身份、有限值、是否加载正确最佳模型、评估口径；先确认没跑错，再讨论涨跌。
2. 保存原始产物，不修改旧 `outputs/`。按同样列名生成新结果表。
3. 在所属阶段 README 写“问题—设置—关键结果—能说明什么—不能说明什么”。失败/被替代版本放 HISTORY，不混进正式汇总。
4. 把轻量 metadata、CSV、配置、代码版本及 hash 纳入 Git；大模型、数据、完整控制台日志留在校验过的归档/共享存储，至少有独立备份。
5. 只有实际运行通过，才把 `gpu_smoke` / `full_retraining` 从 `not_run` 改成验收记录；报告整理不是训练复现验收。

单 seed 是观察，不写稳定提升；AR item-level 与 Diff SID-level 分别讨论。02/04 的复用基线引用原 run，不能算成新重复实验。

## 五、已保存基线与后续变更审查

01–05 的归档整理已保存为基线 commit `32f3b4e`（`chore: preserve stages 01-05 before project restructuring`），不是待再次提交的工作。该 commit 是整理版本，不是过去实验的运行版本；历史源码快照中的 commit 仍按原记录解释。

当前整理分支为 `codex/project-organization`。A、B1、B2a、B2b、B2c 分别为 `737bebf`、`5f4f433`、`07d6a83`、`3d41254`、`fd5faee`；跨平台字节保护与运行路径兼容分别为 `5cc5ef3`、`594b6ad`。原始证据、历史 README 和方法/历史文档中的旧路径保留，`outputs/` 原运行内容不变。当前服务器状态见第六节。

```powershell
git status --short
git diff --stat
git diff -- README.md experiments/README.md configs/README.md scripts/README.md docs/ tests/
python scripts/repro/project_repro.py verify
python scripts/repro/stage05_repro.py verify
python tests/run_repro_checks.py
python tests/test_runtime_paths.py
```

检查范围与依赖要求见 [tests 说明](../../tests/README.md)；统一检查入口为 [run_repro_checks.py](../../tests/run_repro_checks.py)。本地检查不替代 GPU smoke 或完整训练验收。

布局审计和对应 layout 测试用于各批次当时的路径限定验收，不能要求后续明确提交的功能/说明改动仍逐字节等同于旧收据。它们留在原版本供追溯，不再作为当前 HEAD 的日常总验收，也不重写历史锁来掩盖差异。

`git diff` 默认不显示 untracked 新文件，务必也看 `git status` 和 IDE 未跟踪文件列表。只加入已审查的项目文件，**不要 `git add .`**：当前还有与研究归档无关的草稿目录和授权截图。

A 批的文件哈希、路径映射和验收范围保留在 [A 批迁移记录](../../reproducibility/layout_migrations/20261007_output_to_experiments/README.md)，B1 记录保留在 [B1 验收](../../reproducibility/layout_migrations/20261007_b1_workflow/README.md)；不为后续整理改写旧收据。B2a 单独记录 tests/docs 迁移，没有重训或重新生成历史图表。

以后确需提交本轮变更时，先审查目录迁移和每个文件差异，再在 IDE 中逐项暂存已确认的变更，不重复执行旧归档整理的批量暂存命令。目录迁移应同时核对旧位置移出与新位置加入属于同一批文件；不要只暂存新目录而漏掉迁移的另一侧。暂存后检查：

```powershell
git diff --cached --stat
git diff --cached --check
```

确认只包含预期文件，并明确决定提交后，再为本轮目录调整创建独立 commit。以后每个研究改动单独提交；先短测，再正式运行，最后另提交结果摘要与清单。

当前不清理旧文件。将来真要清理时，逐项确认：是否已有独立副本、是否被 recipe/metadata 引用、hash 是否一致、能否恢复；先列候选清单，再决定删除。

## 六、2026-10-09 迁移收尾与验收范围

工程迁移已完成有限范围验收，可以准备下一轮研究实验；不需要再次迁移数据、重建模型环境或重跑全部历史结果。模型算法、历史指标、原始证据、配方、输入锁和源码快照没有因本次收尾而改变。

### 当前只使用哪个目录

- 本地工作副本仍为 `C:/Users/nancy/Documents/experiment1`。
- 服务器主项目为 `/inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized`。
- 旧服务器项目已改名为同一父目录下的 `experiment1_retired_20261009`；未删除文件，原 `experiment1` 路径已不存在。不要重建旧路径或建立指回退役备份的软链接。
- 模型环境仍为同一父目录下的 `conda-envs/grm-exp`，不在退役项目内部。

已有 Linux 终端也要切换目录，不沿用旧终端的工作位置：

```bash
cd /inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized
source runtime_env.sh
```

### 实际通过的检查

下表对应运行版本 `594b6add85c0a72731890fa45fa67d414fa358a2`；这次后续文档收尾不改变已验收的训练代码。

| 检查 | 已完成范围 | 不能据此宣称什么 |
|---|---|---|
| 运行资产迁移 | 2,571 项条目校验；数据、Tokenizer、历史输出与权重为独立副本，运行软链接及 Git origin 不依赖旧项目 | 不能据此删除其他 archive 材料或模型环境 |
| 旧目录退出路径后的检查 | 20 项路径测试 + 78 项标准库复现测试通过；在旧路径不存在时读取新目录资产正常 | 不等于所有历史模型已正式重训 |
| 05 四组主要实验输入 | `cross_joint_v2`、`cross_frozen`、`refined_joint`、`refined_frozen` 各 13 项固定输入通过；111 个证据副本、源码快照和六组结果/metadata 校验通过 | 输入齐全不等于四组均通过 GPU 训练 |
| 一组 H100 GPU smoke | `cross_joint_v2`，3 epoch、每 epoch 最多 2 个训练 batch、评估最多 2 个 batch；交替更新、梯度隔离、直接推荐梯度、验证/测试及新短测 checkpoint 保存与重新加载通过 | 小样本指标不能与历史正式结果比较，也不证明推荐效果提升 |
| 正式训练及独立评估 | 迁移后的历史正式重训、历史最佳模型独立重评均未验收；旧目录改名后没有重新占 GPU 做第二次短测 | 不写成“01–05 全部正式复现成功” |

GPU 短测期间，包装器和训练子进程禁止 Python 访问旧项目的 open/listdir/scandir/chdir；随后又实际改名旧目录并通过 CPU 路径、资产和输入检查。旧实验的精确运行 commit、完整配置等历史缺口仍保留，不反向补造。

### 收据放在哪里

原始记录不混入历史科学证据或旧配方：

- 服务器：新主项目下的 `archive/server_migration_20261008/`。
- 本地镜像：`archive/server_sync/server_runtime_20261009/`，已下载 GPU、退役和改名后检查收据。
- 本页是纳入 Git 的验收摘要和原始收据身份索引，不替代原始记录。

以下 SHA256 已核对本地与服务器一致，路径相对于上述各自收据根目录：

```text
4294b32cf5030e7a92538fe11b08f5fd0091988233b54c20727c3a24959507db  gpu_acceptance_20261009/receipt.json
7fa33ba54b3b3d45b49ed7b7ef9d00f38398aec6c8fef45ecb7fa912be1d97cf  retirement_20261009/completion.json
d4544fc834d23397d199cd1b944d40d12bf40a417a533783a102771ce40f2665  retirement_20261009/post_rename_checks.json
d1e9157f4fd648cdfcff8eadad28abf10aadada92bd99e3c4d968216174b4f44  retirement_20261009/files.json
```

GPU 原始轻量证据包已下载为本地收据根目录下的 `gpu_smoke_evidence_20261009.tar.gz`，SHA256 为 `94ea9f6273de1fa4643bc45fc03d374a2256a6e20eba4b61b34576d0ace7063b`；完整短测输出及权重仍保留在服务器 `outputs/stage05_replays/stage05_replay_cross_joint_v2_smoke_20261009T063218_698856Z/`。本次没有用它改写任何正式结果表。

### 从这里接回研究流程

整理分支的提交与服务器同步完成后，在新的 `codex/exp-...` 分支开展下一轮实验。先明确研究问题、基线和控制变量；再修改、短测、提交代码与配置、核对服务器版本，然后正式运行。不要直接复用旧训练脚本内固定的历史 run_id。

`scripts/repro/` 使用封存源码，适合历史重放；下一轮新研究需要审查 `scripts/train/` 的当前开发入口、唯一 run_id、初始化绑定与完整运行记录。迁移已通过的 smoke 不替代新模型代码改动后的 smoke。

退役备份保留且不再作为运行输入。删除、压缩包清理和个人材料迁出仍是独立事项，不是继续实验的前置条件，也没有获得本次收尾的删除授权。
