# B2c：完成剩余脚本职责归类

日期：2026-10-08。迁移前提交为 `3d41254`（B2b配置归类）；分支 `codex/project-organization`。本记录是结构调整验收，不是新训练结果。

## 改了什么

75个原 `scripts/` 根层脚本全部移入职责目录，旧根层不放重复副本或兼容壳；原有legacy导入器留在原位。另新增本批审计脚本。

| 目录 | 从根层迁入数量 | 用途 |
|---|---:|---|
| repro | 4 | 日常校验、重算、资产恢复和隔离重跑 |
| audit | 4 | A/B1/B2a/B2b历史审计，另新增B2c当前审计 |
| prepare | 5 | 上游准备、embedding、fit mask和RQ-VAE输入 |
| tokenizers | 3 | Tokenizer训练与检查 |
| reports | 4 | 阶段报告、PDF及报告图生成 |
| analysis | 16 | 汇总、Loss、Attention、碰撞及画图 |
| checks | 15 | preflight、smoke产物检查、查checkpoint |
| train | 23 | AR/DiffGRM训练、profile和smoke启动器 |
| legacy | 1 | 旧pilot归档；已有importer没有移动 |

完整旧→新映射及迁移前732个受控文件的hash见 [before.json](before.json)。该文件在移动前生成，并保存5份完整性锁的原始JSON；不是事后生成的基线。

配套变化仅包括：脚本根目录深度、跨目录调用/导入、现用指南和索引、5份辅助代码绑定。`.gitattributes`明确分类后的脚本使用LF。4个根目录draft和2张授权图仅加入`.gitignore`，原位保留；没有读取凭据图片或提交私人内容。

## 什么没有变

- 34份YAML字节、模型参数、覆盖顺序、训练工作目录和原run_id不变。
- 01–05的科学报告、evidence、衍生结果、配方、数据/SID/初始化hash、源码tar不变。
- 5份`package_integrity.json`只允许6个既有辅助代码/测试的路径或hash重绑，科学`files`数组及其他字段不变；并未重建实验包。
- 历史方法文档、旧README、A/B1/B2a/B2b收据和原审计常量不改写。旧audit文件只调整物理ROOT。
- `outputs/`、`data/`、`external/`、`archive/`和个人草稿未迁移、删除或清空。

原始数据层保留目录不是“漏归类”：它们的用途已在 [项目地图](../../../docs/guides/PROJECT_MAP.md) 列明，原路径被历史运行记录引用，不能为了整齐随便更名。

## 验收方式

在项目根目录运行：

```powershell
python -B tests/run_repro_checks.py
python -B tests/test_entrypoint_layout.py
python -B tests/test_config_layout.py
python -B tests/test_script_layout.py
python -B scripts/audit/audit_script_layout.py verify --check-receipt
```

检查范围：

- 78项基础回归 + 9项B2a约束 + 12项B2b约束 + 15项B2c检查，共114项标准库测试。
- 75个迁移脚本与1个legacy loader严格撤销批准的路径/ROOT/import变化后匹配原hash；6个基础测试也只改导入/入口路径。参数、函数主体或顺序的额外修改会被拒绝。
- 旧布局测试对当前caller先撤销本批受控变换，再验证旧约束；历史seal断言使用迁移前快照，当前5份seal另做完整对象检查，不伪造旧收据。
- 24个Bash脚本语法检查，3个PowerShell解析，Python脚本AST解析。没有执行训练、bootstrap或旧报告生成器。
- 日常测试禁止读取旧archive和本机资产映射；5阶段校验、分析和计划可运行，真实实验包零写入。14份CSV在临时副本重算后字段值/字节一致，05 SVG重复生成稳定。
- 25份当前导航的268个本地链接全部可解析；跨目录调用、6个动态runner、4个同目录包装器目标存在。
- 验收通过：732个既有受控文件中639个字节不变、93项受控修改；545个实验包文件中535个科学文件不变，另外10个仅是5份当前操作说明和5份辅助绑定。373份原始证据校验通过。具体文件结果见 [receipt.json](receipt.json)。

## 仍未验收的内容

本机没有完整训练依赖，未运行6个模型功能测试、GPU smoke或正式训练。标准库检查使用本机Python，不等于原Python 3.10训练环境验收。

05的6组计划仍各缺13项本机训练输入；没有自动恢复大资产。本次未核验大迁移包第二份独立备份，所以仍不能删除唯一来源。个人材料最终项目外位置也尚未指定。

这批只完成研究文件的职责分类和入口整合。清理重复压缩包、迁出个人材料、GPU复现属于后续独立操作，不与目录搬迁混做。
