# 项目地图：以后文件放哪里

先记住三处：**看结论去 `experiments/`，改实验去 `configs/` 和 `scripts/`，找原始产物去 `outputs/`。**

本轮把75个根层脚本全部归类，不删除旧材料、不改训练参数；配置已在上一批按01–05阶段归类。根目录的私人草稿和授权图保留原位，仅加入Git忽略规则。

## 1. 整个项目的分工

| 位置 | 内容 | 管理方式 |
|---|---|---|
| [experiments/](../../experiments/README.md) | 01–05的正式报告、结果表、证据副本、版本演进、复现配方 | 纳入Git；原始证据不手改，失败版本也保留身份 |
| [configs/](../../configs/README.md) | shared共用配置 + 01–05阶段配置 | 纳入Git；新实验新配置，写明唯一run_id |
| [scripts/](../../scripts/README.md) | 按职责分类的运行工具，详见下表 | 纳入Git；变更与配置一起审查 |
| [tests/](../../tests/README.md) | 标准库安全回归、布局检查、模型功能测试 | 纳入Git；模型测试依赖另行准备 |
| [docs/](../README.md) | guides现用指南；methods方法原文；history旧版本记录 | 纳入Git；旧文中的路径是历史记录，不直接照抄运行 |
| [reproducibility/](../../reproducibility/README.md) | 源码快照、环境记录、资产身份锁、迁移记录 | 纳入Git；本机资产位置文件例外，不能提交 |
| `external/` | DiffGRM / GenRec独立代码仓库 | 根仓库忽略；改引擎必须单独保存commit/patch，根commit不会保存引擎改动 |
| `data/` | 数据、embedding、缓存 | 不入根Git；持久存储、hash和独立备份；本地可为空 |
| `outputs/` | 每次运行的metadata、reports、attention、tokenizers、日志与权重等 | 不入根Git；不批量改名，新运行独立run_id |
| `archive/` | 原证据包、旧导出、历史导入材料 | 不入根Git；仍可能被旧工具依赖，未执行删除 |
| `patches/`、`upstreams.lock` | 上游基准与补丁 | 纳入Git；不等同于完整的历史运行版本 |
| `environment.yml`、`requirements-*.txt` | 项目环境入口 | 纳入Git；与历史导出及新机器实际环境分开记录 |
| `.vscode/`、`.runtime/`、`tmp/`、`__pycache__/` | 编辑器设置、工具缓存、预览文件 | 忽略；不把这些目录作为正式结果或唯一副本 |
| 根目录4个`draft_*_folder`、2张授权图 | 沟通草稿、个人实习材料、授权相关图片 | 原位保留并忽略；不得顺手提交，不属于实验包 |

`archive/`、`data/`和`outputs/`被忽略，不代表无用，也不代表Git能恢复它们。大迁移包尚未核验第二份独立备份，所以这次不清理。

## 2. 脚本的九个入口

| 目录 | 什么时候用 |
|---|---|
| [repro/](../../scripts/repro/README.md) | 查已有实验、校验、恢复输入、从锁定源码隔离重跑 |
| [train/](../../scripts/train/README.md) | 开发新实验时启动训练/profile/smoke |
| [checks/](../../scripts/checks/README.md) | 开跑前核对条件，短跑后检查是否符合对照要求 |
| [prepare/](../../scripts/prepare/README.md) | 准备embedding、fit mask、输入和上游代码 |
| [tokenizers/](../../scripts/tokenizers/README.md) | 训练或检查Tokenizer本身 |
| [analysis/](../../scripts/analysis/README.md) | 汇总指标、计算碰撞、分析Attention、画图 |
| [reports/](../../scripts/reports/README.md) | 生成阶段报告、PDF、报告图片；部分旧工具会写既有报告，不能随意执行 |
| [audit/](../../scripts/audit/README.md) | 核验这次目录整理，不负责模型科学评估 |
| [legacy/](../../scripts/legacy/README.md) | 特定历史材料导入/归档，平时不用 |

例如旧命令 `bash scripts/run_diff_crosspse_diger_v2.sh ...` 现在是 `bash scripts/train/run_diff_crosspse_diger_v2.sh ...`。文件名和参数不变。**复现已归档的05实验时，优先用repro配方，不直接重启旧shell。**

## 3. 已有实验怎样串起来

`01 AR基线 → 02 AR容量/Tokenizer对照 → 03 DiffGRM三种Tokenizer → 04 离线码本交互 → 05 联合训练与Frozen对照`

每阶段看README得到结论，看HISTORY了解演进，看runs/recipe确认是哪次运行。原始证据和派生结果分开；同一个run被后续阶段引用，不算新实验。

当前四组Joint/Frozen主对照已经完成；下一步仍应围绕初始化、更新量、选模时点控制变量。目录整理没有带来新的模型结果，也不能把单seed现象写成稳定提升。

## 4. 下一次实验的固定流程

1. **确定问题**：写清楚基线、只改哪个变量、看什么指标；放现用研究说明，不先复制一堆配置。
2. **保存代码**：创建`codex/exp-简短主题`分支，改对应阶段配置/脚本；让AI解释diff，检查后单独commit。引擎改动在external仓库另存。
3. **短测**：先标准库检查，再在训练环境运行相应preflight/smoke；记录实际通过范围，不把本机语法检查当GPU验收。
4. **正式跑**：唯一run_id，保存完整合并配置、根/引擎commit、输入与初始化hash、日志、验证曲线和最佳checkpoint。
5. **归档结果**：原始产物留outputs；审核后的轻量证据、汇总与结论进入所属experiments阶段。科学结论变更另一个commit。

不要从目录名或修改时间猜“最新版”，不要用重新生成完整性锁修复不明hash差异。新阶段只用于新研究问题；同阶段新seed不另开阶段。

B2c布局审计核验的是本次整理时点，不是允许随时刷新的“最新锁”。以后确实修改算法或增加配置时，需要为新变更制定验收，不能改写旧迁移收据让它通过。

## 5. 这次保留不动的边界

- 方法/历史文档原文、历史README、已有源码tar和旧迁移收据原样保留。
- 运行数据和大资产保持原路径，避免让metadata、checkpoint引用失效。
- 两份研究沟通草稿、两份个人实习草稿及授权图不进入Git；个人材料最终存放位置由你决定。这次没有查看授权图内容。
- 不删除任何压缩包或过程文件。清理是下一项独立操作，需要先确认独有内容、依赖和备份。

当前目录验收与完整路径映射见 [B2c记录](../../reproducibility/layout_migrations/20261008_b2c_scripts/README.md)。
