# 阶段 05 补充诊断

这里保存正式训练之后新增的检查，不修改 `../evidence/`、历史结果表或复现锁。

| 诊断 | 研究问题 | 结果入口 |
|---|---|---|
| DIGER SID 一致性 | 带噪声的临时编码与正式编码差多少；参数更新实际留下多少 SID 变化 | [2026 年 10 月 11 日结果](sid_consistency_20261011/README.md) |

每次诊断使用独立日期目录：`README.md` 解释结果，`raw/` 保留程序输出，`execution.json` 记录代码、命令、环境和测试。复用脚本在 `scripts/analysis/`，参数在 `configs/diagnostics/`。不要把诊断当成新增训练 seed 或新的推荐性能结果。

基于诊断开展的新训练另放 [pilots 中的噪声更新短对照](../pilots/noise_update/README.md)，不混入这里的只读统计。
