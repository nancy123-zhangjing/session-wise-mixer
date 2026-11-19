# 08阶段验证集 Top20/Top50 补充统计

复用三组固定R35/60T的last权重；同一验证集22,363人，beam256、batch8，最终返回50个SID。没有新增训练、重选best或评估测试集。

| 模型 | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | HR@50 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.071994 | 0.050485 | 0.099048 | 0.059208 | 0.131825 | 0.067487 | 0.154809 | 0.072231 |
| no_context | 0.070831 | 0.050301 | 0.098511 | 0.059202 | 0.131870 | 0.067609 | 0.150919 | 0.071549 |
| axial | 0.071055 | 0.050642 | 0.099674 | 0.059860 | 0.130305 | 0.067615 | 0.152082 | 0.072144 |

| axial相对对照 | 指标 | 绝对差 | 相对差 |
|---|---|---:|---:|
| baseline | HR@5 | -0.000939 | -1.30% |
| baseline | NDCG@5 | +0.000158 | +0.31% |
| baseline | HR@10 | +0.000626 | +0.63% |
| baseline | NDCG@10 | +0.000652 | +1.10% |
| baseline | HR@20 | -0.001520 | -1.15% |
| baseline | NDCG@20 | +0.000128 | +0.19% |
| baseline | HR@50 | -0.002728 | -1.76% |
| baseline | NDCG@50 | -0.000087 | -0.12% |
| no_context | HR@5 | +0.000224 | +0.32% |
| no_context | NDCG@5 | +0.000341 | +0.68% |
| no_context | HR@10 | +0.001163 | +1.18% |
| no_context | NDCG@10 | +0.000657 | +1.11% |
| no_context | HR@20 | -0.001565 | -1.19% |
| no_context | NDCG@20 | +0.000006 | +0.01% |
| no_context | HR@50 | +0.001163 | +0.77% |
| no_context | NDCG@50 | +0.000595 | +0.83% |

三组原有HR/NDCG@5、@10均精确复现；Top50逐用户候选已保存，并在本地CPU独立重算通过。

本轮仅将最终SID去重等整数后处理移到CPU，训练与搜索逻辑保持原样；每组首批原版Top50及原版Top10前缀均已逐位核对一致。

指标按SID匹配、首次命中计分；单目标下HR与Recall等价，没有展开成商品级排名。这里只补充同一轮次不同K的表现，不能据此证明跨seed稳定提升。

beam256是内部搜索宽度，返回50是最终候选数。本轮累计GPU时长、费用及释放核验另见资源release记录，不能沿用原训练的费用数字。

证据：[机器汇总](SUMMARY.json)；原始下载位于`C:/Users/nancy/Documents/experiment1/outputs/stage08_session_codeword/topk_validation_20261024_r2/remote_results`。

## 本轮结果怎样理解

axial在固定R35的NDCG@10有小幅优势，但这种优势没有一致延续到Top20/50：相对baseline，HR@20约−1.15%、HR@50约−1.76%，NDCG@20约+0.19%、NDCG@50约−0.12%。这些仍是单seed、验证集、SID级结果，不能解释为稳定泛化收益或全方向无效。

正式比较最终推荐质量，应先固定选模规则和评估设置，再统一评估测试集。本轮延续测试锁定，只补验证诊断。若比较各组最终可选模型，应使用原weighted规则选出的best（baseline R15、no_context R30、axial R35）；固定R35则回答相同训练预算下的比较问题，两种口径分别报告。

## GPU与留档

本轮含主动中止的首轮评估，保守累计占用 0.662329 GPU小时，低于1小时上限。按已查询展示单价1点/GPU小时估算约0.66点，实际账单未核实。六个评估Job均已释放并从本轮清单清理，旧训练资源与产物未改。

首轮因Top50逐条GPU同步过慢主动停止，部分结果没有用于报告。第二版仅把整数去重和指标输入一次移到CPU；GPU搜索、原argsort顺序和模型保持不变，各组首批完整Top50及Top10前缀逐项对照原实现通过，完整验证的原HR/NDCG@5、@10精确复现。这里的速度变化属于评估实现优化，不作为模型时延收益。

[用卡与释放账本](../../../outputs/stage08_session_codeword/topk_validation_20261024_r2/GPU_RELEASE.json) · [首轮中止记录](../../../outputs/stage08_session_codeword/topk_validation_20261024_r1/GPU_RELEASE.json) · [本地下载核验](../../../outputs/stage08_session_codeword/topk_validation_20261024_r2/DOWNLOAD_VERIFICATION.json)
