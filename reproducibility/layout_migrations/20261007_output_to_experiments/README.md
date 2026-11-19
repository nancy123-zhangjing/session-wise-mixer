# A 批：实验包目录迁移记录

日期：2026-10-07。重构前基线：`32f3b4e2039f40129f709646b535742eadd2de53`。

## 做了什么

- 将整个 `output/` 移到 `experiments/`，545 个文件一一对应，未删减包内内容。
- 更新当前文档、注册表、复现脚本与 `.gitattributes`；训练产生原始结果的 `outputs/` 不变。
- 5 份 `package_integrity.json` 仅更新 `project_repro.py` / `stage05_repro.py` 中适用的辅助脚本哈希。源码快照、历史代码版本、初始化资产与配方不变；没有通过重新 build 改写历史来源。
- 修正3个旧 Stage1 报告工具的路径，但没有运行它们、覆盖历史 PDF 或图片。这些生成器仍依赖 archive 中的旧输入，解除依赖留到 B 批。

## 核对结果

| 检查 | 结果 |
|---|---|
| 迁移前记录的已有文件 | 699 个，迁移后全部能定位 |
| 字节完全不变 | 679 个；其中实验包内 537 个 |
| 已审查的变化 | 20 个文档、路径辅助脚本、注册表或完整性锁 |
| 原始证据 | 01–05 共 373 份，全部通过原 hash 校验 |
| 原实验身份 | 23 条注册记录，仅 package 路径变化，其他字段与顺序不变 |
| 保护对象 | 结果 CSV、历史 README、图表、配方、初始化锁和源码快照均未改变 |
| 本地测试 | 原 44 项 + 新增布局 7 项，共 51 项通过 |
| 当前文档链接 | 独立检查22份文档的173条本地相对链接，无缺失 |
| GPU 重跑 | 未执行；本次验收不代表已完成训练复现 |

审计范围是原有受跟踪文件和整个实验包，不是远程机器或所有被忽略的大资产。未删除 archive、草稿、授权截图、数据或权重；未暂存、commit 或 push。

## 怎么复查

在项目根目录执行：

```powershell
python scripts/test_project_repro.py
python scripts/test_stage05_repro.py
python scripts/test_project_layout.py
python scripts/project_repro.py verify
python scripts/stage05_repro.py verify
```

迁移的原始检查记录：

- [before.json](before.json)：迁移前699份文件的实际字节哈希、旧→新路径，以及原注册表和完整性锁。
- [receipt.json](receipt.json)：迁移后结果，列明20份变化的迁移前/后哈希；其余文件与 before 相同。
- [审计脚本](../../../scripts/audit_layout_migration.py)：只读验证，拒绝任意结果/配方/源码快照改动；只允许已审查的路径、文档及指定辅助脚本哈希更新。

```powershell
python scripts/audit_layout_migration.py verify
```

这是**本次整理工作区的历史快照审计**，不是之后允许覆盖旧记录的工具，也不是跨操作系统重写换行符后的通用仓库检查。之后科研文件有合法改动时，不应修改 before 使之通过；日常实验包校验使用上面的两个 verify 命令。

## 接下来

先审查并独立提交本次迁移，再开展 B 批：解除旧导入目录依赖，整理脚本、配置和文档。C 批删除另行批准。原路径在历史证据中保留，不把历史记录替换成新的运行地点。

Git 在暂存之前可能显示 `output/` 的删除和 `experiments/` 的未跟踪文件，这是尚未暂存的目录迁移；不是数据丢失。暂存时须同时包含旧位置和新位置，不要把根目录草稿或授权图一起加入。
