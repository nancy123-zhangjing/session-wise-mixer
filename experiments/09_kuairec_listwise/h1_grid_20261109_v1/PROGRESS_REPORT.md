# B/C 阶段进展：首个完整节点 R8/T40

实验仍在运行，尚未完成 R35 或最终验收。

| 组别 | 验证 NDCG@5 | 相对 A | 验证 HR@5 | 验证 Recall@5 |
|---|---:|---:|---:|---:|
| A：原 H1（R20 前也是原 H1 Freeze 的共同前段） | 0.006513359 | — | 0.029920315 | 0.006076169 |
| B：横向注意力 | 0.006224108 | −4.44% | 0.029348145 | 0.005984063 |
| C：横向＋纵向注意力 | 0.006051157 | −7.10% | 0.028273581 | 0.005766359 |

事实：此节点 B/C 均落后于 A；B 高于 C。不能据首个节点判断最终最佳点或冻结后的 R35 收益。尚未进行本轮测试集评估。

B/C 均为 71,657 条完整验证，FP32、beam32。R8/T40 完整 checkpoint 的 SHA256 与只读加载通过，新增 Grid Adam 已更新 2,079 步，原 R 累计 5,544 步。证据见 `R08_VERIFIED.json`，A 来源见 `BASELINE_A.json`。

本次验证耗时 B=2,433.59 秒（40.56 分钟），C=2,464.06 秒（41.07 分钟）。后续每个三轮 R 区间的纯 R 训练约 44–46 分钟，再加约 41 分钟验证和小段 T 更新，所以下一 R11 完整结果粗估约 1.5 小时后。若后续速度保持，全部剩余训练、验证、best/R35 测试粗估 14–16 小时；不包含异常中断、排队或平台回收造成的延迟。

平台近段 GPU 利用率偏低，回收风险仍需观察；科学配置没有改变。后续继续至 R20 冻结 Tokenizer/SID，再训练推荐器及 Grid 到 R35。最终报告另由完整验收结果生成。


## R11/T80

Full validation: 71,657 examples; FP32; beam 32. Both complete; node checkpoints retained remotely.

| Arm | NDCG@5 | vs A |
|---|---:|---:|
| A | 0.006016181 | reference |
| B | 0.005762434 | -4.22% |
| C | 0.005757006 | -4.31% |

Both Grid arms remain below A at R11. This is an early single-seed observation, not a conclusion about best or R35 performance. Training continues unchanged.

## R14/T120

| Arm | Validation NDCG@5 | vs A |
|---|---:|---:|
| A | 0.008571204 | reference |
| B | 0.008363017 | -2.43% |
| C | 0.008836601 | +3.10% |

Both summaries read back: full validation 71,657 samples, FP32, beam 32, R updates 9702 / T updates 120. C exceeds A at this node; B remains below A with a narrower gap than R11. This is a single-seed intermediate result, not evidence of final or statistically reliable improvement.


## R17/T160

| Arm | Validation NDCG@5 | vs A |
|---|---:|---:|
| A | 0.007947733 | reference |
| B | 0.008084854 | +1.73% |
| C | 0.007723583 | -2.82% |

Both full validation summaries read back: 71,657 samples, FP32, beam 32, R updates 11781 / T updates 160. B exceeds A at this node; C falls below A after exceeding it at R14. Both arms decrease from their own R14 results. Intermediate single-seed observations only; continue the locked schedule.


## R20/T200

| Arm | Validation NDCG@5 | vs A |
|---|---:|---:|
| A | 0.009372056 | reference |
| B | 0.008551500 | -8.76% |
| C | 0.008045600 | -14.15% |

B full validation summary read back: 71,657 samples, FP32, beam 32, R updates 13860 / T updates 200. FREEZE_R20.json read back; B has entered R21 with T still 200. Frozen-state invariance across later checkpoints and continued Grid parameter changes remain for final audit. B improves over its R17 but remains below A at R20; no final-performance conclusion.

C full summary and FREEZE_R20.json also read back: 71,657 validation samples, FP32, beam 32, R13860/T200. Both arms are below A at R20. C remains below its own R14 score; its R14 lead did not persist to the fixed freeze point. This does not determine post-freeze best or R35 performance. Later checkpoint freeze invariance and Grid changes remain pending final audit.


## R23/T200

B 完整验证已读取远端 summary 并核对：71657 条、beam32、FP32、R15939/T200。

| 组别 | 验证 NDCG@5 | 相对 A 同轮 |
|---|---:|---:|
| A 原 H1 Freeze | 0.009629052 | — |
| B 横向注意力 | 0.007158417 | -25.66% |
| C 横向+纵向 | 0.006846527 | -28.90% |

B 相对自身 R20 下降 16.29%，目前最佳仍为 R20。这是冻结后的第一个验证点，不能据此断定最终趋势或归因于 Grid；继续按原协议训练。C 完整验证也已核对同一评估口径及 R15939/T200：相对自身 R20 下降 14.90%，当前 best 仍为 R14。两组冻结后首个节点均下降，尚不能据此判断最终趋势或确定原因。证据：B_R23_SUMMARY.json、B_R23_SUMMARY_READBACK.json、R23_COMPARISON.json。

C R23 证据：C_R23_SUMMARY.json、C_R23_SUMMARY_READBACK.json。


## R26：B/C 完整验证均已核对

B 的完整验证已确认 71,657 样本、beam=32、FP32，累计 R 更新 18,018、T=200。

| 组别 | R26 验证 NDCG@5 | 相对 A |
|---|---:|---:|
| A：原 H1 Freeze | 0.010364534 | — |
| B：横向注意力 | 0.005844802 | -43.61% |
| C：横向＋纵向注意力 | 0.007315993 | -29.41% |

B 相对自身 R23 再下降 18.35%，当前最佳仍为 R20（0.008551500）。这是冻结后继续下降的观察，不能仅据此确定原因；测试和最终节点仍待完成。B 已进入 R27，C 正在 R26 验证。按当前速度，剩余训练、验证和两次测试约需 5–6 小时，最终审计与同步另计。

证据：B_R26_SUMMARY_READBACK.json、B_R26_SUMMARY.json、R26_COMPARISON.json。

C 的 R26 完整评估口径亦核对通过（71,657 样本、beam=32、FP32、R18,018/T200）。C 相对自身 R23 回升 6.86%，但仍低于 A，当前最佳仍为 R14（0.008836601）。B/C 后期走势已出现分化，不能概括为两组都持续下降。目前均进入 R27，剩余训练、验证与测试约 5 小时，最终审计和同步另计。证据：C_R26_SUMMARY_READBACK.json、C_R26_SUMMARY.json、R26_COMPARISON.json。


## R29 完整验证

两组均核对 71657 样本、beam32、FP32、R20097/T200。

| 组别 | NDCG@5 | 相对 A | 相对自身 R26 |
|---|---:|---:|---:|
| A | 0.010552616 | — | — |
| B | 0.007192233 | -31.84% | 23.05% |
| C | 0.007193384 | -31.83% | -1.68% |

B 回升，C 小幅下降，两者仍低于 A。当前最佳仍为 B R20、C R14。均进入 R30，预计剩余训练、验证和测试约 3.5–4 小时，最终审计另计。证据：B/C_R29_SUMMARY.json 和 R29_COMPARISON.json。


## R32 partial readback: B complete

B full validation (71,657 examples, beam 32, FP32): NDCG@5 = 0.007488702916753966; A R32 = 0.010689730123073586, B vs A = -29.9449%; B vs R29 = +4.1221%. B best remains R20 (0.008551499872693947). C R32 is still validating; no C R32 metric yet. B has entered R33. Estimated remaining training/validation/tests: about 2.5-3 hours, final artifact and resource audit additional.


## R32 complete readback: B and C

Both full validations passed the 71,657-example / beam32 / FP32 / R22176-T200 checks. A = 0.010689730123073586, B = 0.007488702916753966, C = 0.008102479994755352. B vs A = -29.94488326146103%; C vs A = -24.20313795138478%. C vs R29 = 12.637938791934866%. Best remains B R20 / C R14. Both original jobs have entered R33; remaining training, validation and tests estimated 2.5-3 hours, final audit additional.


## R35 full validation complete

All 71,657 validation examples; beam=32, FP32. Both arms finished R35/T200.

| Arm | R35 NDCG@5 | vs A | Validation-selected best |
|---|---:|---:|---|
| A: original H1 Freeze | 0.0104455471 | — | R32: 0.0106897301 |
| B: row attention | 0.0071662878 | -31.39% | R20: 0.0085514999 |
| C: row + column | 0.0079305923 | -24.08% | R14: 0.0088366006 |

B and C best tests are running; fixed R35 tests and final artifact/resource audits remain pending. These validation results do not establish the cause of degradation or a general conclusion about Grid Attention.
