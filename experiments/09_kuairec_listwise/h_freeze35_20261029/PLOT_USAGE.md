# 训练与验证曲线

三组完成且 `analyze_results.py` 已各自复算后运行：

```powershell
.runtime/stage08_restore_env/Scripts/python.exe experiments/09_kuairec_listwise/h_freeze35_20261029/plot_results.py --runs-root <本地runs目录> --branch-reports <包含A/B/C分析目录的根目录> --output <全新图表目录>
```

输入结构为 `runs/H1-Freeze20-R35`、`runs/H0-Freeze32-R35`、`runs/H0-Freeze20-R35`，以及 `分析根/A/analysis_report.json`、`METRICS.csv`、`audits/`，B/C同理。脚本核对分析报告绑定、CSV和完整验证审计。缺节点、未完成结果、冲突的重复更新、实际LR缺失都会拒绝，不插值或补造。

输出三组 PNG/PDF：R训练loss、实际使用的R学习率、同一坐标下的完整验证单列表NDCG@5，另附数据CSV和SHA来源记录。loss按每次更新的实际列表数加权，只统计R目标；A仅绘本轮R27–35，B仅R33–35，C绘R6–35。LR直接读 `learning_rates_used.R`，使用累计R更新计数映射回原轮次，不重建日程或添加终点。

验证图包含A的R20/23/26历史点、B的R32父模型、C的R5起点及重放节点。空心点表示历史来源，星号仅标A/C共同验证网格best；B不能混作同冻结时点对照。R20、R32冻结位置及不同训练起点会明确标注。图中没有测试指标或聚合@50；这是seed2024的单次探索，曲线不证明跨seed稳定性。

三组的编码表示不同，训练loss不能直接代替推荐质量排序；质量比较以真实商品级验证/测试为准。

仅检查历史日志解析和绘图时可显式运行：

```powershell
.runtime/stage08_restore_env/Scripts/python.exe experiments/09_kuairec_listwise/h_freeze35_20261029/plot_results.py --history-smoke outputs/stage09_restore/goal_20261027_r1/runs/H0/training.jsonl --output <全新历史检查目录>
```

历史检查严格锁定原H0日志SHA，图上有醒目的 `NOT FREEZE35 RESULTS`。原日志没有实际LR字段，因此此模式只画历史R-loss，不能充当本轮结果交付。
