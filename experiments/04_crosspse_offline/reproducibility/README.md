# 04｜离线 Refined / Cross-PSE 复现

公共操作见 [项目复现手册](../../../reproducibility/README.md)。三组比较中，PSE复用03正式v6；本阶段只有两次新增下游训练。

| 实验 ID | 固定Tokenizer来源 | 最佳推荐模型时间戳 |
|---|---|---|
| `refined_seed2024` | `diff_refinedpse/Beauty/v2_catalog_seed2024` | `Aug-28-2026_14-15-28` |
| `cross_seed2024` | `diff_crosspse/Beauty/v3_catalog_seed2024` | `Aug-28-2026_19-01-02` |
| `pse_seed2024` | 复用03 PSE v6 | `Aug-11-2026_14-09-45` |

离线Tokenizer训练结束后，固定SID，再随机初始化并训练DiffGRM。它不是05阶段交替联合训练。Tokenizer选模在 `fit_catalog` 上进行，**不是独立验证集**；不要把小样本pilot与最终catalog版混用。

## 证据和依赖

- [recipes/](recipes/) 固定配置覆盖顺序、SID/summary、随机Decoder初始化及基线引用。
- [assets.lock.json](assets.lock.json) 同时绑定下游必需输入、可用的Tokenizer `best.pt` / codebooks / OPQ输入等补充资产、各组最终推荐模型。
- [evidence_manifest.json](evidence_manifest.json) 保留原结果、逐位置与解码顺序、训练曲线、attention、Tokenizer summary和pilot副本。
- [analysis_checks.json](analysis_checks.json) 核对DiffGRM验证集weighted score选模和四位置数据；它不把Tokenizer的catalog选择包装成独立验证。
- [code_versions.json](code_versions.json) 关联根目录共享源码快照。
- [previous_README.md](previous_README.md) 保留原报告。03/04共用基线不重复算新实验。

一部分原材料在 `archive/deprecated_and_temp/tmp_browser_cache/crosspse_report_20260829/`；目录名虽含tmp，它却是有效证据来源。本次只复制和建立索引，没有删除它。

```bash
python scripts/repro/project_repro.py verify --stage 04
python scripts/repro/project_repro.py analyze --stage 04
python scripts/repro/project_repro.py run --stage 04 --experiment cross_seed2024 --mode smoke
```

计划不会启动GPU。输入恢复选项 `--include-tokenizer` 只取回补充产物，不自动重训量化器。当前已完成本地数据校验；GPU smoke、正式重训和checkpoint重新评估尚未执行。要重训Tokenizer，另开run并与固定历史SID路径区分。
