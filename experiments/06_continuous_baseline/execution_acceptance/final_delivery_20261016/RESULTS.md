# Stage 06 continuous retrieval baselines

Validation-only selected baseline: `sasrec_fullce`.

89 verified training runs; full-catalog item-level evaluation, no history filtering.
Sample standard deviations use ddof=1. No test metric influenced selection.

| Model | Test HR@10 (mean ± SD) | Test NDCG@10 (mean ± SD) |
|---|---:|---:|
| sasrec_fullce | 0.072280 ± 0.001294 | 0.041515 ± 0.000845 |
| esasrec | 0.064920 ± 0.001759 | 0.039181 ± 0.000795 |

Legacy protocol paired differences, all seed-level metrics and configurations: `summary.json`.
All 89 training curves remain in their run directories; five-seed figures are in `curves/`.

- Leave-one-out is not a global temporal split or online evaluation.
- Legacy/full protocols change target exposure and optimizer update counts.
- SASRec/eSASRec differ in architecture and training loss, not a causal one-factor ablation.
- This is a continuous retrieval baseline, not learned Decoder-adapted encoding.
