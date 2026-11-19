# 一次性历史导入，不是日常复现

`import_experiments.py` 将旧封装入口与日常 `verify/analyze/restore-inputs/run`
分开。原两个主脚本的 `build` 命令已退役；这里也不是修复 integrity seal、
重绑当前 HEAD 或覆盖现有阶段包的命令。

默认只输出计划，不读封存锁、不查询 Git HEAD、不创建目录、不训练：

```powershell
python -B scripts/legacy/import_experiments.py --stages 05 --archive "D:/historical/experiment1.tar.gz" --source "D:/historical/05_joint_source"
python -B scripts/legacy/import_experiments.py --stages 01-04 --archive "D:/historical/experiment1.tar.gz"
```

当前项目已有封存包和共享锁，实际执行必定拒绝。计划中的
`execution_blockers` 列出所发现的已有目标；不要通过删除它们绕过保护。

## 人工审阅后的首次导入

只有在隔离的首次导入工作区、人工备齐旧来源，并审阅了要封存的代码版本后，
才能额外传入 `--execute --expected-root-commit <完整Git提交ID>`。
执行前先拒绝已有或半成品目标，再检查来源和预期根提交；没有 `--force`。
目标阶段目录只允许预存普通 `README.md` 模板，空 `reproducibility/`、
`evidence/`、`derived/`、`figures/` 及已有结果表均会阻止导入。
共享来源锁、迁移索引、代码版本锁、源码对比、源码快照目录及环境目录也会阻止导入，
包括05导入。仅根仓库提交由该参数核对；旧导入器中对外部仓库未提交源码的检查保留。
01–04还拒绝覆盖已有 `experiments/experiment_registry.json`；包及共享控制目录
不得通过链接逃出该隔离工作区。

先导入05，再导入01–04：旧01–04算法最后构建总索引时读取05的 `runs.csv`。
入口在导入前检查它存在，不会写完01–04才发现缺失。01–04还需要四份阶段README模板。
导入途中发生其他错误可能留下半成品；下次执行会拒绝，应人工检查隔离工作区，
而不是在现有封存包上继续或重新封装。

## 本批仍保留的旧输入

- 原迁移 `experiment1.tar.gz` 及同目录的 `grm-exp.environment.yml`、
  `grm-exp.pip-freeze.txt`、`SHA256SUMS`。大资产恢复仍需要迁移包。
- `archive/evidence_bundles/` 的六个小 `tar.gz`，以及
  `repro_sources_01_04/` 中的恢复证据、AR控制台和旧报告工具输入。
- 05的解压来源树：`outputs/run_metadata/`、`outputs/reports/`、原始指标CSV、`configs/`。
  `--source` 可以指向名为 `experiment1` 的目录，或其父目录。
- `archive/deprecated_and_temp/tmp_browser_cache/crosspse_report_20260829/crosspse_report_20260829_094937/`
  下的04 `reports/` 和 `tokenizers/pilots/`。
- 根 `outputs/run_metadata/AmazonReviews2014/DIFF_GRM/` 中的阶段03历史记录。
- `reproducibility/code/` 与05封装中的源码tar：日常校验/复现仍实际使用。

本次仅隔离入口，旧 `_legacy_build` 导入算法未改。未证明旧来源可删除，
也未执行真实历史导入、重封装或训练。

## 早期 pilot 归档工具

[`archive_diff_pilot.py`](archive_diff_pilot.py) 用于特定早期 DiffGRM pilot 的文件归档，不是正式训练或常规汇总入口。它会写归档，先查看参数和目标，不在结构整理时自动执行。

旧 Stage1 报告生成器现在位于 [reports/](../reports/README.md)，仍可能依赖 archive 中的历史来源。目录归类不代表这些来源可以删除。
