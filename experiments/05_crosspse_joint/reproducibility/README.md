# 05 阶段复现操作说明

## 先分清三个层次

| 目标 | 当前状态 | 入口 |
|---|---|---|
| 重算报告中的表格和验证曲线 | 已完成，本地无需 GPU | `analyze` |
| 根据固定源码、输入和配置重新训练 | 四组主实验输入已校验；Cross Joint v2 GPU 短测通过；完整正式重训仍未验收 | `check-inputs` → `run --mode smoke` → `run --mode formal` |
| 直接加载历史最终模型重新评估 | 已绑定最终权重、SID、码本；独立评估入口尚未验收 | 暂不作为已完成能力 |

首次归档整理没有启动训练、修改模型算法、删除原始文件或提交 Git commit；随后归档基线已保存于 `32f3b4e`，见第 6 节。CPU 测试通过不等于 GPU 训练已通过，也不保证换 GPU 后浮点结果逐位相同。

2026-10-07 的归档检查保留在 [原验收记录](CHECKS.md)，不改写当时结论；2026-10-09 迁移后的新检查见 [迁移收尾与收据索引](../../../docs/guides/EXPERIMENT_WORKFLOW.md#六2026-10-09-迁移收尾与验收范围)。阶段主报告及旧 CHECKS 中“尚未 GPU 验收”是注明日期时的状态，不再作为当前迁移进度。通过的是运行链路，不是推荐指标复现。

## 1. 哪些文件必须保留

### 随 Git 管理的小文件

- 整个 `experiments/05_crosspse_joint/`：证据副本、配方、锁文件、源码快照、报告。
- [stage05_repro.py](../../../scripts/repro/stage05_repro.py)：日常校验、重算、输入恢复和重跑入口；首次历史导入已分离到 [legacy](../../../scripts/legacy/README.md)。
- [test_stage05_repro.py](../../../tests/test_stage05_repro.py)：CPU 安全与一致性测试。
- 根目录 [.gitattributes](../../../.gitattributes)：避免 Windows/Linux 换行符转换破坏字节 hash。

虽然根仓库忽略了 `external/`，本实验包仍保存了两个模型仓库的源码 tar。因此不能只记录一个根仓库 commit，就认为它已经包含了模型代码。新入口直接使用锁定的 DiffGRM 源码快照，不读取当前工作区可能被改过的模型代码。

### 不放进 Git、但必须单独备份的大文件

原迁移包：

~~~text
C:\Users\nancy\Downloads\experiment1_migration_20260923_retry\experiment1.tar.gz
SHA256: a74836899900bccb9d44a18f6403b4220b50615f53b26503c82825dcb2f9bc09
大小: 4,623,792,331 bytes
~~~

其中已定位本阶段需要的 38 个数据、初始化和最终产物文件，合计 449,712,966 bytes；逐文件位置、大小、hash 见 [assets.lock.json](assets.lock.json)。不要因为报告已整理，就删除这个迁移包。最好保留另一份独立备份。

如果服务器已完整解压这份迁移包，可以直接校验已有文件，不必再次上传或解压。这个项目包并不是“单独下载 README 就能重跑”。

## 2. 先在本地校验并重算报告

以下命令在 **Windows PowerShell、本地项目根目录**执行。只需要 Python 标准库，不需要安装 PyTorch 或占 GPU。

~~~powershell
Set-Location "C:\Users\nancy\Documents\experiment1"
python scripts/repro/stage05_repro.py verify
python tests/test_stage05_repro.py
python scripts/repro/stage05_repro.py analyze
~~~

- `verify`：检查 111 个原始证据副本、配方和锁文件、脚本、源码快照的 hash，核对六组结果与 metadata。
- `test`：验证路径安全、不覆盖已有文件、输入 hash、配对初始化、最佳点阶段计数、smoke 梯度隔离及配置检查。
- `analyze`：仅重新生成 `results.csv`、`runs.csv` 和验证曲线 SVG，不改原始 CSV，不训练模型。

若校验失败，先检查差异；不要为了让检查变绿，直接改 hash 或重新生成锁文件。

## 3. 初始化是如何绑定的

| 实验 | Tokenizer artifact | 初始 Decoder |
|---|---|---|
| Cross Frozen / Joint v2 / SIA / DIGER v1 | `outputs/tokenizers/diff_crosspse/Beauty/v3_catalog_seed2024/` | `external/DiffGRM/saved/AmazonReviews2014_Aug-28-2026_19-01-02/pytorch_model.bin` |
| Refined Frozen / Joint | `outputs/tokenizers/diff_refinedpse/Beauty/v2_catalog_seed2024/` | `external/DiffGRM/saved/AmazonReviews2014_Aug-28-2026_14-15-28/pytorch_model.bin` |

每组配方绑定的输入包括：

- 对应的初始 Decoder checkpoint；
- Tokenizer 的 `best.pt`、基础码本、OPQ 切分后的商品向量、SID 的 NPY / JSON、summary；
- 固定 item keys 与 Beauty 原始/预处理数据；
- 有序的配置覆盖列表和独立 smoke 配置。

已有历史 hash 的四类输入（Tokenizer checkpoint、基础码本、SID JSON、item keys）已与历史 metadata 核对。其余文件的 hash 取自迁移快照，不能倒推成历史训练时记录过的 hash。

**初始 SID JSON 的文件 hash，与训练后 SID 的规范化内容 hash / NPY 文件 hash，不是一种口径。** 不要仅因为这几个字符串不同，就认定恢复错文件。配方分别列出初始化输入和最终评估产物，不将两者混用。

旧脚本仍保留，有些会自动寻找最新匹配 checkpoint；本阶段的固定复现请使用下面的新入口，它不按修改时间挑权重。

## 4. 到创智机器上怎么做

先同步本说明第 1 节列出的 Git 小文件。以下命令在 **创智 Linux 终端**运行，不在 Windows PowerShell 里运行。使用此前建立的 grm-exp 环境，训练前确认是在 GPU 实例上。

~~~bash
cd /inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_reorganized
source runtime_env.sh
python scripts/repro/stage05_repro.py verify
python scripts/repro/stage05_repro.py check-inputs --experiment cross_joint_v2 --project-root .
~~~

只有最后一步全部通过，才说明这组训练所需文件已在当前机器上就位。2026-10-09 已在新服务器目录完成 `cross_joint_v2`、`cross_frozen`、`refined_joint`、`refined_frozen` 各 13 项固定输入检查；这不代表四组均已重训。后续启动仍先校验选定实验，避免运行期间输入变化。

### 如果提示缺少输入

不要重新运行 Tokenizer 来“补文件”，否则初始化就变了。用原迁移包恢复固定输入。此前上传路径若仍存在，可执行：

~~~bash
python scripts/repro/stage05_repro.py restore-inputs --experiment cross_joint_v2 --project-root . --archive /inspire/hdd/project/project-public/<inspire-account>/nancy/experiment1_migration_20260923_retry/experiment1.tar.gz
~~~

这个命令先校验整个压缩包，再只提取配方列出的文件；已有相同文件跳过，已有不同文件报错并保留。不会批量覆盖整个项目。压缩包如已换位置，只修改 `--archive`，不要改锁文件。

也可以配置被Git忽略的 `reproducibility/asset_locations.local.json`，之后省略 `--archive`；身份仍按原锁校验，详见 [B1工作流](../../../reproducibility/WORKFLOW_B1.md)。Windows 本地副本不要求存放大数据和权重；正式训练使用已独立迁移并校验的新服务器目录，不因本地输入缺少就重新拟合 SID。

可额外加 `--include-final` 恢复这次历史运行的最终 checkpoint、对应 SID 和码本；它只是恢复文件，不会启动评估。

### 先看计划，不启动训练

~~~bash
python scripts/repro/stage05_repro.py run --experiment cross_joint_v2 --project-root . --mode smoke
~~~

不加 `--execute` 时只校验并显示选择的配方/初始化，不创建训练任务。

### 再做 GPU 短测试

确认要占用 GPU 时才执行：

迁移验收已执行过下面的 `cross_joint_v2` 短测并通过。这里保留命令供以后复验，不要求再次运行；其他配方或新模型代码的短测不因此自动通过。

~~~bash
python scripts/repro/stage05_repro.py run --experiment cross_joint_v2 --project-root . --mode smoke --execute
~~~

联合版 smoke 使用独立的小 batch/短训练配置，检查有限 loss、Decoder/Tokenizer 分阶段梯度隔离；DIGER 版还检查直接推荐梯度。Frozen 两组使用各自专门的冻结 smoke 配置，不会被误改成交替更新。

完成后，新输出目录应有 `completion.json`（进程退出码）和 `smoke_check.json`（短测检查通过）。有退出码 0 但没有后者，也不能当作 smoke 验收通过。smoke 的指标不能用于正式结果表。

### smoke 通过后再正式重跑

~~~bash
python scripts/repro/stage05_repro.py run --experiment cross_joint_v2 --project-root . --mode formal --execute
~~~

其他组只替换 `--experiment`：`cross_frozen`、`refined_frozen`、`refined_joint`、`cross_sia`、`cross_diger_v1`。先验收一组，不要求现在把六组全部再跑一遍。

每次执行都会建立全新目录：

~~~text
outputs/stage05_replays/stage05_replay_<短名称>_<smoke或formal>_<UTC时间>/
├── engine/                    # 从源码 tar 展开的固定代码；模型保存于 engine/saved/
├── resolved_config.json       # 完整合并配置与本次绝对路径，JSON 可被 YAML loader 读取
├── configuration_audit.json   # 正式模式：核对历史 metadata 已记录的关键配置
├── environment.json           # 本次 Python / packages / torch / CUDA / GPU
├── launch.json                # argv、cwd、配方、源码版本、配置 hash
├── console.log                # 实时写入的日志
├── diagnostics/               # 训练、验证、逐位置、解码顺序和联合阶段记录
├── run_metadata/              # 本次结果身份及最终导出
├── cache/                     # 独立可写缓存，不污染历史 tokenizer 映射
├── completion.json            # 进程结束状态
└── smoke_check.json           # 仅 smoke 验收通过时生成
~~~

训练数据会复制到这次独立 cache；旧源码、输入权重、原结果目录均不会被训练入口覆盖。resolved config 不包含 loader 在启动时生成的当前时间等动态字段，这些另外由日志和 metadata 记录。

## 5. 已知历史缺口

| 项目 | 已有证据 | 不能宣称的内容 |
|---|---|---|
| 根仓库 | 整理前源码基线 d614a1e；归档整理已保存于 32f3b4e | d614a1e 或 32f3b4e 就是六次历史训练的运行 commit |
| DiffGRM / GenRec | 当前固定 commit + 源码 tar；169 个受控源码文件与迁移包逐字节一致 | 迁移包与每个更早训练时刻的源码必定完全相同 |
| 配置 | 迁移 YAML、metadata 关键参数、当前锁定源码默认值 | 找回了当时所有 CLI 参数及完整 resolved config |
| 环境 | 迁移时 conda / pip 记录 | 能证明六组历史运行的所有库版本、硬件都完全一样 |
| 结果 | 原始 CSV、summary、metadata 相互核对 | 本次已经重新训练或独立运行过测试集评估 |

现阶段属于“证据已整理、四组主要输入已校验、Cross Joint v2 GPU 短测通过、正式重训与历史模型独立评估未验收”。新短测 checkpoint 的保存/重新加载已通过，但不能替代历史最终模型独立评估。历史配方内的原始 verification 字段和旧验收记录保持原样；最新验收单独由迁移收据记录，不修改封存锁使其变绿。

## 6. 已保存基线、以后怎么改

01–05 的归档整理已保存为 commit `32f3b4e`（`chore: preserve stages 01-05 before project restructuring`），无需再执行此前的归档暂存/提交命令。该 commit 不替代源码快照中的 `d614a1e`，也不是历史训练运行版本。

A批已提交为 `737bebf`，B1=`5f4f433`，B2a=`07d6a83`，B2b=`3d41254`，B2c=`fd5faee`；服务器迁移的跨平台字节保护与运行路径兼容分别为 `5cc5ef3`、`594b6ad`。本复现入口的历史证据、配方、输入锁和源码快照不变。旧服务器项目仅改名为退役备份，原始 `outputs/` 内容独立迁入新目录。后续研究改动另行审查、提交和同步，根 Git 仍未配置 GitHub 推送。先检查当前分支和 diff：

~~~powershell
git status --short
git diff -- README.md experiments/README.md experiments/05_crosspse_joint/README.md experiments/05_crosspse_joint/reproducibility/README.md
~~~

以后准备提交后续变更时，只逐项暂存已审查的文件，避免误提交根目录的 draft 文件夹及授权图片；迁移审查和暂存范围见 [项目工作流](../../../docs/guides/EXPERIMENT_WORKFLOW.md)。暂存后检查：

~~~powershell
git diff --cached --stat
git diff --cached --check
~~~

之后每轮新实验：先开独立实验分支 → 修改并短测 → 提交代码和配置 → 记录 commit、独立 run_id、输入 hash → 正式运行 → 收集证据并另一次提交报告。不要用新的结果覆盖这六组原证据；重大协议变化单独建新的配方版本/阶段，不反复改写已封存的历史定义。

旧 `build` 命令已退役。首次导入旧资料只通过 [独立历史导入入口](../../../scripts/legacy/README.md)，需要未封存的隔离工作区；当前包存在时会拒绝。修订报告/辅助工具应保留科学证据与历史绑定，通过单独审计更新当前工具的完整性锁，不能以重建锁的方式掩盖不明来源的文件变动。
