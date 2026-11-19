# analysis：脚本清单

用于推荐结果汇总、逐位置Loss、SID碰撞/几何、AR/DiffGRM Attention与画图。部分工具依赖GPU/模型权重，部分仅需CSV；先查看--help。输出仍写原outputs等运行目录，不自动替换已封存证据。

所有命令从项目根目录执行，例如脚本路径写成 `scripts/analysis/文件名`。历史配方内部的旧源码路径保留原样，不手工改历史锁。

- [analyze_ar_self_attention.py](analyze_ar_self_attention.py)
- [analyze_ar_smoke_v2.sh](analyze_ar_smoke_v2.sh)
- [analyze_diff_cpd_attention.py](analyze_diff_cpd_attention.py)
- [analyze_diger_sid_consistency.py](analyze_diger_sid_consistency.py)：CPU 诊断 Gumbel 编码稳定性、确定性 SID 变化和验证集覆盖；不训练模型，不覆盖历史结果。
- [analyze_sid_collisions.py](analyze_sid_collisions.py)
- [analyze_tokenizer_geometry.py](analyze_tokenizer_geometry.py)
- [compare_diff_tokenizers.py](compare_diff_tokenizers.py)
- [compare_pse_refinement_tokenizers.py](compare_pse_refinement_tokenizers.py)
- [plot_ar_internal_study.py](plot_ar_internal_study.py)
- [plot_position_loss.py](plot_position_loss.py)
- [run_formal_attention.sh](run_formal_attention.sh)
- [summarize_ar_internal_attention.py](summarize_ar_internal_attention.py)
- [summarize_ar_internal_study.py](summarize_ar_internal_study.py)
- [summarize_ar_self_attention.py](summarize_ar_self_attention.py)
- [summarize_diff_attention.py](summarize_diff_attention.py)
- [summarize_diff_formal.py](summarize_diff_formal.py)
- [summarize_formal_results.py](summarize_formal_results.py)

返回 [脚本总入口](../README.md) · [项目地图](../../docs/guides/PROJECT_MAP.md)
