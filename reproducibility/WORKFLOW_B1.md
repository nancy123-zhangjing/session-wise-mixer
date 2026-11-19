# B1：日常复现和历史导入分开

现在查看、校验、重算01–05的结果，不需要重新解压旧报告包，也不应再次运行 `build`。正式训练的数据和权重仍是另一层资产，需要按配方恢复和校验。

## 平时怎么用

| 目的 | 入口 | 是否需要旧 archive/ |
|---|---|---|
| 01–04 查实验、校验、重算 | `project_repro.py list / verify / analyze` | 不需要 |
| 05 校验、重算 | `stage05_repro.py verify / analyze` | 不需要 |
| 查看重跑计划 | 两个入口的 `run`，不加 `--execute` | 不需要；05仍先检查本机输入 |
| 恢复训练输入 | 两个入口的 `restore-inputs` | 需要迁移大包，可以放在项目外 |
| 实际重跑 | 两个入口的 `run --execute` | 需要已校验输入和GPU；本批没有执行 |
| 首次导入旧材料 | [独立历史导入入口](../scripts/legacy/README.md) | 需要，且禁止覆盖既有封存包 |

在项目根目录快速检查：

```powershell
python scripts/repro/project_repro.py verify
python scripts/repro/stage05_repro.py verify
python tests/test_daily_repro.py
```

`analyze` 会重算派生汇总表，不改原始证据；`verify` 是只读检查。已有结果若校验失败，先查原因，不要重写锁文件使它“通过”。

## 大文件位置怎么管理

资产“是什么”由原 `assets.lock.json` 中的 SHA256 和成员路径决定；“这台机器上放在哪儿”另存于被Git忽略的 `reproducibility/asset_locations.local.json`。

本机映射已配置为已核验的 Downloads 迁移包。新机器参考 [配置模板](asset_locations.example.json)，只填写该机器实际可访问的路径，不要修改原资产锁。显式传 `--archive` 时优先使用该参数，不读取本机映射。

相对路径的基准：显式 `--archive` 相对于启动命令的当前目录（保持旧行为）；本机映射里的相对路径相对于项目根目录。

```powershell
python scripts/repro/asset_locations.py check
```

上述命令只读，会检查完整包的大小和SHA256。换位置不等于换资产；大小或hash不符会拒绝。

确实需要恢复时，才运行例如：

```powershell
python scripts/repro/project_repro.py restore-inputs --stage 03 --experiment pse_seed2024 --project-root .
python scripts/repro/stage05_repro.py restore-inputs --experiment cross_joint_v2 --project-root .
```

这两条会写入配方指定的输入文件，请在准备训练的机器上运行，并确认空间。已有相同文件复用，已有不同文件报错且不覆盖。它们不会启动训练。恢复测试本批仅使用合成小包，没有把大数据自动解压到本机。

## 这次确认了什么

- 原有51项测试及新增27项测试，共78项CPU测试通过。
- 在审计守卫禁止访问旧 `archive/`、迁移大包和本机映射的情况下，五阶段日常校验、分析与01–04计划正常工作；真实实验包零写入。
- 临时副本重算14份CSV，与原件字段值和字节一致，重复生成稳定；05的SVG内容一致。没有覆盖原报告或图片。
- 05的六个dry-run准确报告每组缺少13项本机输入；这不是archive依赖问题，也不能写成“已经具备GPU复现条件”。
- 本机迁移包大小为4,623,792,331字节，SHA256与原锁一致。尚未核验第二份独立备份或服务器现存副本。
- 历史证据、实验配方、结果、源码快照及初始化锁不变；包装完整性锁只更新本批辅助代码绑定。

详见 [B1验收记录](layout_migrations/20261007_b1_workflow/README.md)。A批迁移记录保留为历史快照，不为B1改写旧收据。

## 为什么现在还不删 archive

旧导入器和3个旧Stage1报告生成工具仍会读取旧资料，独有pilot、沟通稿及备份也未全部处理。已验证的是**日常复现不需要旧导入目录**，不是“全部旧材料都可删除”。

B1记录的是当时的隔离检查；之后B2a/B2b/B2c已完成测试、文档、配置和脚本归类。本页命令已指向当前 `scripts/repro/`；旧收据不改写。未改训练参数或算法，也未启动训练。
