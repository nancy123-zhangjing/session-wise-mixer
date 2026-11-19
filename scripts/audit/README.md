# audit：脚本清单

用于检验目录重构，不是科学实验评估。当前布局使用audit_script_layout.py verify --check-receipt。其余四个audit对应A/B1/B2a/B2b历史时点，保留原常量和收据；不要要求它们直接验证新布局。

所有命令从项目根目录执行，例如脚本路径写成 `scripts/audit/文件名`。历史配方内部的旧源码路径保留原样，不手工改历史锁。

- [audit_config_layout.py](audit_config_layout.py)
- [audit_entrypoint_layout.py](audit_entrypoint_layout.py)
- [audit_layout_migration.py](audit_layout_migration.py)
- [audit_repro_workflow.py](audit_repro_workflow.py)
- [audit_script_layout.py](audit_script_layout.py)

返回 [脚本总入口](../README.md) · [项目地图](../../docs/guides/PROJECT_MAP.md)
