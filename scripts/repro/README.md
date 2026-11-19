# repro：脚本清单

用于日常校验、重算汇总、按hash恢复输入和隔离重跑。01–04用project_repro，05用stage05_repro；run默认只看计划，实际训练需要明确的--execute。两个build入口已退役，禁止用重建锁掩盖异常。

所有命令从项目根目录执行，例如脚本路径写成 `scripts/repro/文件名`。历史配方内部的旧源码路径保留原样，不手工改历史锁。

- [asset_locations.py](asset_locations.py)
- [project_replay.py](project_replay.py)
- [project_repro.py](project_repro.py)
- [stage05_repro.py](stage05_repro.py)

返回 [脚本总入口](../README.md) · [项目地图](../../docs/guides/PROJECT_MAP.md)
