# 脚本入口索引

所有命令从项目根目录执行。75 个原根层脚本已按职责归类；旧 `scripts/文件名` 不再作为运行入口，**脚本名和命令参数不变，只变子目录**。历史文档和封存源码保留原路径，复现旧实验优先使用 `repro/`，不要直接复制历史命令。

## 按你要做的事查入口

| 目录 | 放什么 | 典型入口 |
|---|---|---|
| [repro/](repro/README.md) | 查封存结果、校验、恢复输入、隔离重跑 | `project_repro.py`（01–04）、`stage05_repro.py`（05） |
| [train/](train/README.md) | 正式训练、profile、smoke 的启动脚本 | `run_ar_internal_study.sh`、`run_diff_crosspse_diger_v2.sh` |
| [checks/](checks/README.md) | 开跑前检查、smoke 结果核验、产物校验、查 checkpoint | `preflight_*.py`、`check_*_smoke.py` |
| [prepare/](prepare/README.md) | 准备 embedding、fit mask、RQ-VAE 输入和上游仓库 | `prepare_formal_embeddings.py` |
| [tokenizers/](tokenizers/README.md) | 拟合和检查 Tokenizer | `train_cross_pse_tokenizer.py`、`train_rqvae_tokenizer.py` |
| [analysis/](analysis/README.md) | 指标汇总、Loss、碰撞率、Attention、画图 | `summarize_diff_formal.py`、`analyze_diff_cpd_attention.py` |
| [reports/](reports/README.md) | 根据材料生成报告/PDF/报告图 | `build_crosspse_stage_report.py` |
| [audit/](audit/README.md) | 项目目录重构的验收工具 | 当前 `audit_script_layout.py` |
| [legacy/](legacy/README.md) | 首次导入旧资料、归档早期 pilot | 不是日常复现入口 |

## 最常用的操作

查看已经完成的实验，不启动训练：

```powershell
python -B scripts/repro/project_repro.py list
python -B scripts/repro/project_repro.py verify
python -B scripts/repro/stage05_repro.py verify
python -B tests/run_repro_checks.py
```

检查归类后的路径和完整性：

```powershell
python -B tests/test_script_layout.py
python -B scripts/audit/audit_script_layout.py verify --check-receipt
```

**复现已归档实验**：使用 [01–04 复现手册](../reproducibility/README.md) 或 [05 复现手册](../experiments/05_crosspse_joint/reproducibility/README.md)，按配方锁定输入和源码，默认先看计划。

**开发新实验**：先在 [对应阶段配置目录](../configs/README.md) 新建配置和唯一 run_id，再审查训练脚本。部分旧训练脚本会复用 run_id、缓存或自动找产物；不要未经检查就重新运行、覆盖已有结果。

`project_repro.py build` 和 `stage05_repro.py build` 已退役。发现 hash 不符先找原因，不能重建锁来掩盖变化。方法/历史原文、原始产物和科学参数不因本轮归类改变。
