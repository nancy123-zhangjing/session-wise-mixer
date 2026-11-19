# H0/H1 三 seed 冻结对照最终报告

两组新增 H0 Freeze 已通过验收，两张 GPU 已释放。大型 checkpoint 和逐样本预测保留远端，本地保存指标与校验收据。

H0 continued：交替训练到 R35/T400。H0/H1 freeze：各自交替训练到 R20/T200，然后冻结整个 Tokenizer 和 SID，只训练推荐器到 R35。H1 使用 Mixer，H0 不使用。主比较固定 R35，辅助比较按验证 NDCG@5 选择的 best，不按测试集挑选。

## R35

| Seed | 组别 | 轮次 | 验证 NDCG@5 | 测试 NDCG@5 |
|---|---|---|---:|---:|
| 2024 | H0_continued | R35 | 0.008645 | 0.008483 |
| 2024 | H0_freeze | R35 | 0.008268 | 0.008256 |
| 2024 | H1_freeze | R35 | 0.010446 | 0.008956 |
| 2025 | H0_continued | R35 | 0.005950 | 0.004922 |
| 2025 | H0_freeze | R35 | 0.006288 | 0.004996 |
| 2025 | H1_freeze | R35 | 0.007284 | 0.007113 |
| 2026 | H0_continued | R35 | 0.007447 | 0.006921 |
| 2026 | H0_freeze | R35 | 0.007280 | 0.006905 |
| 2026 | H1_freeze | R35 | 0.008299 | 0.008336 |

测试均值 ± 样本标准差（3 seeds）：

- H0_continued: 0.006775 ± 0.001785
- H0_freeze: 0.006719 ± 0.001638
- H1_freeze: 0.008135 ± 0.000938

H1 Freeze 相对 H0 Freeze 的均值提升：21.08%。

## best

| Seed | 组别 | 轮次 | 验证 NDCG@5 | 测试 NDCG@5 |
|---|---|---|---:|---:|
| 2024 | H0_continued | R32 | 0.008920 | 0.008607 |
| 2024 | H0_freeze | R23 | 0.008898 | 0.008471 |
| 2024 | H1_freeze | R32 | 0.010690 | 0.009431 |
| 2025 | H0_continued | R29 | 0.007078 | 0.006237 |
| 2025 | H0_freeze | R26 | 0.007090 | 0.005890 |
| 2025 | H1_freeze | R32 | 0.007327 | 0.007079 |
| 2026 | H0_continued | R20 | 0.008008 | 0.008112 |
| 2026 | H0_freeze | R20 | 0.008008 | 0.008112 |
| 2026 | H1_freeze | R29 | 0.008520 | 0.008604 |

测试均值 ± 样本标准差（3 seeds）：

- H0_continued: 0.007652 ± 0.001251
- H0_freeze: 0.007491 ± 0.001398
- H1_freeze: 0.008371 ± 0.001193

H1 Freeze 相对 H0 Freeze 的均值提升：11.75%。

## 观察与边界

1. 相同冻结日程下，H1 在三个 seed 的 best 和 R35 测试均领先 H0，支持继续研究 Mixer 路线，优势不能仅归因于冻结。
2. H0 冻结并非普遍有效：best 测试在 2024/2025 下降，2026 保持同一 R20；R35 仅 2025 略升，其余略降。
3. 冻结不能保证后续推荐器训练不回落，应同时报告 best 与固定终点。
4. 三 seed 是有限重复证据，不宣称统计显著。2024 属于探索性历史实验，运行环境与新 seed 有差异；比较的是整个训练方案，不能定位 Mixer 内部具体机制。

## 验收

FINAL_ACCEPTANCE.json 为 passed，4 份测试审计通过，最大复算误差 4.876876169568334e-08。父状态、10395 次新增 R 更新、R35/T200 终点、冻结指纹、5 个完整验证节点、best 选择与测试文件哈希通过；两作业 SUCCEEDED，worker instance_success。未重跑训练或推理，未删除原结果。

来源：FINAL_ACCEPTANCE.json、RESULTS_LATEST.json、REFERENCE_SEED2024.json 和上一批 FINAL_ANALYSIS.json。逐轮验证见 VALIDATION_COMPARISON.csv，最佳点与终点见 FINAL_COMPARISON.csv。
