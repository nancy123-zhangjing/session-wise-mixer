# 脚本入口索引

脚本不按文件夹移动，因为 shell 入口之间存在相对路径调用。日常主要使用下面几组。

## 训练入口

| 阶段 | 入口 |
|---|---|
| AR 正式 3×256 基线 | `profile_formal.sh`、`run_formal_pair.sh` |
| AR PSE / 容量消融 | `run_ar_internal_study.sh` |
| DiffGRM PSE 基线 | `run_diff_formal.sh` |
| DiffGRM RQ-KMeans / RQ-VAE | `run_diff_rqkmeans_v6.sh`、`run_diff_rqvae_v6.sh` |
| 离线 Refined/Cross-PSE | `run_diff_refinedpse_v2_catalog.sh`、`run_diff_crosspse_v3_catalog.sh` |
| Cross-PSE 联合 SIA pilot | `run_diff_crosspse_joint.sh` |
| DIGER ST v1 / v2 | `run_diff_crosspse_diger.sh`、`run_diff_crosspse_diger_v2.sh` |
| Frozen Cross-PSE 对照 | `run_diff_crosspse_frozen.sh` |
| Joint Refined-PSE 对照 | `run_diff_refinedpse_joint.sh` |
| Frozen Refined-PSE 对照 | `run_diff_refinedpse_frozen.sh` |

## 结果统计与画图

| 内容 | 脚本 |
|---|---|
| AR 三 seed 汇总 | `summarize_formal_results.py` |
| AR 容量消融汇总/画图 | `summarize_ar_internal_study.py`、`plot_ar_internal_study.py` |
| DiffGRM 汇总 | `summarize_diff_formal.py`、`compare_diff_tokenizers.py` |
| SID 碰撞与 tokenizer 几何 | `analyze_sid_collisions.py`、`analyze_tokenizer_geometry.py` |
| AR Attention | `analyze_ar_self_attention.py`、`summarize_ar_self_attention.py` |
| DiffGRM CPD Attention | `analyze_diff_cpd_attention.py`、`summarize_diff_attention.py` |
| Cross-PSE 阶段报告 | `build_crosspse_stage_report.py` |

## 检查脚本

`preflight_*.py`、`test_*.py` 和 `verify_*.py` 用于短跑前检查、单元测试与 artifact 校验。正式训练前先运行对应 shell 脚本的 `preflight` 或 `smoke` 模式，不要把 smoke 结果当成正式实验结果。

Frozen Cross-PSE 对照由 `preflight_diff_crosspse_frozen.py` 检查控制变量，并由 `check_diff_crosspse_frozen_smoke.py` 验证训练期间只有 Decoder 梯度、Cross-PSE 参数和 SID 均未变化。

Frozen Refined-PSE 对照由 `preflight_diff_refinedpse_frozen.py` 检查与 Joint Refined-PSE 的匹配关系，并由 `check_diff_refinedpse_frozen_smoke.py` 验证基础码本、SID 和 hash 全程冻结。

`bootstrap_upstreams.ps1/.sh` 只用于重新拉取固定版本上游仓库；普通训练不需要重复执行。
