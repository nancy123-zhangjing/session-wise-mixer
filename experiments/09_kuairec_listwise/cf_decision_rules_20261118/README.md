# CF decision rules, inference only (2026-11-18)

**Status:** finished 2026-11-18 15:50 CST; results in [REPORT.md](REPORT.md).

Approved by the user in [2026-11-18 turn 15](../../../docs/discussions/2026-11-18.md#turn-15). The plan is in [turn 13](../../../docs/discussions/2026-11-18.md#turn-13), with the reference row kept. No training. Validation split only; the test split is not used.

## Question

Is CF's "one sample per slot" decision a main bottleneck? Every candidate rule keeps the session-wise data flow: each slot is conditioned on the items chosen for earlier slots, and the landed SID becomes the next prefix.

| Rule | Decision inside a slot | Session-wise |
|---|---|---|
| A | Original: one noise per slot, greedy layer-wise landing | Yes |
| B | Same single vector; the item with the highest landing probability (C with K = 1) | Yes |
| C4, C16 | K vectors under the same prefix; the item with the highest mean landing probability | Yes |
| D16 | 16 complete lists from rule A; the one that agrees most with the others (minimum Bayes risk) | Yes; the output is one of the generated lists |
| REF16 | Top five of the slot-1 mean landing distribution over 16 vectors | No; diagnostic only |

## Inputs

- **Checkpoints:** first-wave L5O CF, `cf_repeat_v2`, both arms (CF-M0 without Mixer, CF-M1 with Mixer), R10, R19, R30, R35, one training seed.
- **Source:** `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_wave_20261117_v1`.
- **History filter:** on, as in the recorded evaluation.
- **Noise:**
  - Set 0 is the recorded validation noise table.
  - Set k ≥ 1 uses the same generator, with the noise version suffixed by `+decision_rules_kXX`.

## Gates

- **Rule A reproduction:** must reproduce the recorded validation predictions row for row on the GPU; otherwise the run stops.
- **Scorer consistency:** the landing scorer must agree with `losses.landing_loss` on the slot-1 vectors.

## Run

- **Code:** [scripts/cf_decision](../../../scripts/cf_decision), commit `e2c469e` (first attempt) and `bfbbdc8` (rerun). Tests: [tests/test_cf_decision.py](../../../tests/test_cf_decision.py), 8 passed on env-zzl.
- **CPU self-test:** CF-M0 R30, 64 users, FP32. Every rule ran. The scorer NLL was 14.88929 against 14.88930 from `landing_loss`. Rule A matched the recorded BF16 GPU predictions on 82.8% of rows, which is expected on CPU in FP32.
- **Job, first attempt:** `nancy-cf-decision-rules-1118`, 1 × H100, submitted 2026-11-18 14:40 CST ([SUBMIT.json](SUBMIT.json)). It stopped about a minute after starting at 14:42, before any inference, while loading the first catalog.
  - **Cause:** the NGC image enables TF32 by default. The script built the catalog before applying the recorded runtime, which turns TF32 off, so the catalog's own SID self-check failed. A CPU self-test cannot show this, because CPUs do not use TF32.
  - **Fix:** commit `bfbbdc8` applies the recorded runtime before any GPU load and asserts that TF32 is off. Unit tests and a CPU self-test on CF-M1 R19 passed again.
- **Job, rerun:** `nancy-cf-decision-rules-1118-r2` on `bfbbdc8`, submitted 14:47 CST ([SUBMIT_r2.json](SUBMIT_r2.json)).
- **Remote output:** `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/cf_decision_rules_20261118/runs`.
- **Analysis:** [scripts/cf_decision/analyze.py](../../../scripts/cf_decision/analyze.py) runs on CPU. It reuses the 2026-11-17 deep-profile definitions and a paired user bootstrap with 2,000 draws.

## Pre-registered reading

- **C or D clearly above A (CI excludes zero) in both arms, at most checkpoints:** the sampling decision is a main bottleneck. The new rule is then applied to the improved CF as a supplementary analysis.
- **No gain:** direction 1 is not the main bottleneck. Move to the training side (CFG, then a discriminative loss).
- **REF far above every session-wise rule:** the current slot-to-slot conditioning hurts. This is a separate discussion and does not change the paradigm.

## Follow-up: K = 32 and 64 (user request, 2026-11-18_2 turn 22)

- **Code:** `fec2976` (rule D can be skipped; analysis reads several run directories). Unit tests and a CPU self-test (K = 2, 4, rule D skipped) passed.
- **First attempt:** `nancy-cf-decision-k64-1118`, submitted 16:29 ([SUBMIT_k32_64.json](SUBMIT_k32_64.json)), output `runs_k32_64`.
  - CF-M0 R10 passed the reproduction gate, and C32 finished (0.0328).
  - C64 then ran out of GPU memory: 64 users × 64 vectors = 4,096 history copies per batch in the encoder attention.
  - The output folder is kept as a record.
- **Rerun:** `nancy-cf-decision-k64-1118-r2`, submitted 16:36 ([SUBMIT_k32_64_r2.json](SUBMIT_k32_64_r2.json)), output `runs_k32_64_r2`. Same code with `--users-per-batch 16`, so K = 64 uses 1,024 rows per batch, the size C16 already ran at.

