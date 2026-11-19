# CF round-1 screening (2026-11-18)

The plan was agreed in [2026-11-18_2 turns 22–26](../../../docs/discussions/2026-11-18_2.md#turn-22). In [turn 27](../../../docs/discussions/2026-11-18_2.md#turn-27) the user asked for the code changes and checks first, with no runs yet. **Status:** formal screening complete; see [Formal screening results](#formal-screening-results-2026-11-18). Earlier status: code and CPU checks done. The GPU smoke test was approved by the user in 2026-11-18_2 turn 33 ("好，检查完了跟我汇报"). At the time of the smoke test, the formal runs were not yet approved.

## Design

- **Setup:** no-Mixer arm (arm `B0` of the CF code), seed 2024, the first-wave CF configuration (`cf_repeat_v2`), validation only.
- **Budget:** training stops after R20 of the unchanged 35-R schedule. The learning-rate and T clocks are those of the full run.
- **Evaluation:** in-training validation uses the original rule A. Afterwards, `scripts.cf_decision.run` re-scores R10, R15 and R20:
  - rule A, which must reproduce the run's own record exactly;
  - rule C16.
- **Decision:** a group is kept if its R20 C16 validation EV-NDCG@5 beats B0 by at least 0.003, with a paired user-bootstrap 95% interval that excludes zero.

| Job | Group | Change (everything else as B0) |
|---|---|---|
| a | B0 | None. Must reproduce the first-wave CF-M0 validation R1–R20 |
| a | L1 | Landing cross-entropy weight 0.1 → 1 |
| a | L2 | Landing weight 1; denoising MSE weight 1 → 0.1 |
| a | E128 | RQ Encoder output 32 → 128 (256→512→256→128); own preparation and RQ pretraining |
| b | H1 | History head on the encoder (attention pooling, cosine to the CF target vectors, temperature 0.1), cross-entropy on the 5 true targets, weight 0.5; not used at inference |
| b | H2 | Classifier-free guidance: 10% of R samples use a learnable null history. Evaluated at w ∈ {1, 1.5, 2, 3} |
| b | D1 | Generate in the 256-d teacher-vector space; land through a frozen copy of the RQ Encoder; RQ and SIDs unchanged |
| b | E64 | RQ Encoder output 32 → 64 (256→512→256→128→64); own preparation and RQ pretraining |

## Code

- **`scripts/rqcd_mixer/screen.py`:** all switches live under the config key `screen`. Prepared parents and RQ assets are compared without that key. The first-wave parent may be reused only through one explicit source digest.
- **`model.py`, `losses.py`, `training.py`, `catalog.py`, `evaluation.py`, `run.py`:** the switches; guided generation; teacher-space generation; `train --stop-after`; and `test` refuses screening configs.
- **`scripts/cf_decision`:** evaluates one screening run (`--run-dir`, `--prepared`, `--guidance`, `--history-head`); analysis against a reference label.
- **`scripts/cf_screen`:** the groups, a per-GPU worker (own parent → train → evaluate), a 4-slot job, dry-run-first submission with a `--smoke` mode, and the two CPU checks below.

## CPU checks (env-zzl)

1. **Unit tests:** `tests.test_cf_screen`, `test_cf_decision`, `test_rqcd_mixer`, `test_cf_freeze` and `test_l5o_v2` gave 55 tests OK. `test_rqcd_targets` gave 12 OK; it is run through discovery, as before.
2. **B0 equivalence** (`scripts/cf_screen/equivalence.py`): first-wave parent, FP32 on CPU, dropout seeded as in `train()`. The check ran three R updates, one T update, then generation for 32 validation users.
   - Old code against new code: model, tokenizer, every loss value and every generated item are bit-identical.
   - Each version run twice: identical.
   - A first attempt without the dropout seed differed. The cause was the check script, not the training code.
3. **Decision-rule regression:** the first-wave self-test (CF-M0 R30, 64 users) reproduces the earlier numbers exactly (A, B, C4, REF4, D4 and the scorer consistency).
4. **Per-group check** (`scripts/cf_screen/cpu_check.py`, one process per group because the notebook is capped at 16 GiB; peak about 14 GiB):
   - **What it runs:** the real run.py parent checks and `_joint`, a state round trip (save, then restore into a fresh trainer), two R updates and one T update (FP32, small batches), and generation with every rule the group is evaluated with.
   - **E groups:** prepared and pretrained 50 RQ epochs on CPU, through the same code as the formal run.
   - **Result:** all 8 groups exit 0, and every list is valid (5 distinct items, none from the history).

| Group | Generation / latent dim | Parameters updated | Check |
|---|---|---|---|
| B0 | 32 / 32 | 81 | total = MSE + 0.1 × landing |
| L1 | 32 / 32 | 81 | total = MSE + 1.0 × landing |
| L2 | 32 / 32 | 81 | total = 0.1 × MSE + 1.0 × landing |
| H1 | 32 / 32 | 84 (+ query, head) | + 0.5 × history CE (≈ ln 10,728 at start); HEAD lists valid |
| H2 | 32 / 32 | 82 (+ null memory) | guided A and C2 at w = 2 valid |
| D1 | 256 / 32 | 81; 8 frozen RQ Encoder tensors unchanged through R and T | teacher-space targets rebuilt on restore |
| E128 | 128 / 128 | 81 | own parent; after the 50-epoch CPU pretraining, 10,665 distinct SIDs |
| E64 | 64 / 64 | 81 | own parent; after the 50-epoch CPU pretraining, 10,674 distinct SIDs |

Outputs are in [checks/](checks/).

- **Note for E128 and E64:** after only 50 epochs, the first codebook level uses 79% and 59% of its codes. The first-wave 32-d RQ used 100% after 10,000 epochs. The formal runs use the full 10,000 epochs, and their code use will be reported with the results.
- **Not yet checked (GPU only):**
  - BF16;
  - exact reproduction of in-training validation by the post-hoc evaluator;
  - the 4-slot job layout.

  `scripts/cf_screen/submit.py --smoke` runs both jobs with R1, a 50-epoch RQ and 256-user evaluation. It needs the user's approval.

## GPU smoke test

- **Jobs:** `nancy-cf-screen-smoke-a-1118` and `nancy-cf-screen-smoke-b-1118`, 4 × H100 each, the same layout as the formal runs.
- **Code:** commit `e9e81de` (snapshot `code_e9e81de`). Its scripts, CF configs and design documents are byte-identical to the CPU-checked tree.
- **Submitted:** 2026-11-18 18:43 CST ([SUBMIT_smoke.json](SUBMIT_smoke.json)).
- **Budgets:** R1, a 50-epoch RQ for the E groups, and evaluation of 256 users with A (gate), C2, guidance 2 for H2 and HEAD for H1.
- **Output:** `cf_screen_20261118/smoke/`. Smoke outputs are never results.

### Smoke results (2026-11-18 18:43–18:58)

- **All 8 groups pass.** Each one trained R1 (about 265 s, the same as the first wave), validated in training, and ran the post-hoc evaluation with every rule: A, C2, REF2; A_w2 and C2_w2 for H2; HEAD for H1.
- **Rule A reproduction:** in all 8 groups, the post-hoc rule A reproduced the run's own validation exactly (256/256 rows), and the landing scorer matched `landing_loss`.
- **B0 on the GPU:** full-split R1 validation EV-NDCG@5 = 0.010498538175014102, exactly the first-wave CF-M0 R1 record.
- **Bug found:** D1's post-hoc evaluation built the extra noise at the 32-d latent width instead of the 256-d generation width.
  - Fixed in `a0fc1d0`, with a regression test (18 screening and decision-rule tests OK). `scripts/rqcd_mixer` did not change.
  - The D1 evaluation was rerun on its own (`nancy-cf-screen-smoke-d1-1118`, [SUBMIT_smoke_d1.json](SUBMIT_smoke_d1.json)) and passed. Job b had recorded FAILED because of this one slot.
- **E groups after the 50-epoch smoke RQ:** E128 has 10,673 distinct SIDs and E64 10,681. First-level code use is 77% and 60%.
- **GPU utilization:** average 65% (job a) and 57% (job b), including startup.
- **These smoke numbers are not results:** R1, 256 users.


## Formal screening results (2026-11-18)

- **Jobs:** `nancy-cf-screen-a-1118` and `nancy-cf-screen-b-1118`, created 20:15 CST ([SUBMIT.json](SUBMIT.json)). Code `a0fc1d0`, recorded in each evaluation's `RUN.json`.
  - The submit record was written into this directory at 20:15. The submit command does not appear in either session's history (session-wise1, session-wise2), so who ran it is unknown.
  - Evaluation used the pre-registered C16. C32 was not added.
- **Status:** both SUCCEEDED, all 8 slots exit 0. Job a took 7,059 s, job b 8,986 s.
- **Read:** after the OAuth re-authorization at about 00:03 on 11-19.
- **Analysis:** `scripts.cf_decision.analyze --reference-label B0` on env-zzl, run from `code_a0fc1d0`. Output is [RESULTS.json](RESULTS.json), SHA-256 prefix `e82249f31cb023e7`. Paired user bootstrap, 2,000 draws, seed 2024; 7,176 validation users, 5,680 with effective views.

### Gates

- **B0 against the first wave:** B0's in-training validation metrics are identical to the first-wave CF-M0 record at all of R1–R20.
- **Rule A reproduction:** in all 8 groups, post-hoc rule A reproduced the run's own validation at R10, R15 and R20 (7,176/7,176 rows; identical EV-NDCG@5).
- **Scorer consistency:** the landing scorer agrees with `landing_loss` to within 1×10⁻⁶ everywhere.

### Validation EV-NDCG@5, R20

Paired differences are against B0, with 95% intervals.

| Group | Change | C16 | C16 − B0 | A | A − B0 |
|---|---|---:|---|---:|---|
| B0 | none | 0.0349 | | 0.0157 | |
| L1 | landing CE 0.1 → 1 | 0.0330 | −0.0020 [−0.0044, +0.0006] | 0.0169 | +0.0012 [−0.0012, +0.0035] |
| L2 | landing 1, MSE 1 → 0.1 | 0.0302 | −0.0047 [−0.0081, −0.0012] | 0.0155 | −0.0002 [−0.0028, +0.0022] |
| E128 | Encoder output 32 → 128 | 0.0356 | +0.0007 [−0.0024, +0.0037] | 0.0224 | +0.0066 [+0.0035, +0.0098] |
| E64 | Encoder output 32 → 64 | 0.0321 | −0.0028 [−0.0059, +0.0004] | 0.0165 | +0.0007 [−0.0022, +0.0036] |
| H1 | history head, weight 0.5 | 0.0376 | +0.0027 [+0.0003, +0.0052] | 0.0195 | +0.0038 [+0.0015, +0.0061] |
| H2 | classifier-free guidance (w = 1) | 0.0345 | −0.0005 [−0.0027, +0.0020] | 0.0172 | +0.0014 [−0.0007, +0.0034] |
| D1 | teacher-space generation | 0.0271 | −0.0079 [−0.0105, −0.0051] | 0.0139 | −0.0018 [−0.0046, +0.0008] |

**C16 − B0 at R10 / R15 / R20:**

| Group | R10 | R15 | R20 |
|---|---|---|---|
| L1 | +0.0025 | −0.0019 | −0.0020 |
| L2 | −0.0028 | −0.0054 * | −0.0047 * |
| E128 | +0.0021 | +0.0046 * | +0.0007 |
| E64 | +0.0049 * | +0.0073 * | −0.0028 |
| H1 | +0.0068 * | +0.0027 * | +0.0027 * |
| H2 | +0.0002 | −0.0007 | −0.0005 |
| D1 | −0.0038 * | −0.0051 * | −0.0079 * |

\* the interval excludes zero.

### Reading against the pre-registered rule

**No group passes.** The rule asked for R20 C16 at least 0.003 above B0, with an interval that excludes zero.

- **H1 is closest:** +0.0027, with an interval that excludes zero. It is the only group positive at all three checkpoints under both C16 and A, with every interval excluding zero.
- **E128:** gains under rule A at all three checkpoints (+0.0092 / +0.0064 / +0.0066, intervals exclude zero). Under C16 the gain mostly disappears.
- **E64:** unstable; positive at R10 and R15, negative at R20.
- **L1, H2:** about zero.
- **L2, D1:** worse. D1's interval excludes zero at all three checkpoints.
- **H2 guidance:** helps single-sample A, but not C16.
  - Rule A: A_w2 at R20 is 0.0207, +0.0036 [+0.0010, +0.0062] over H2's own A.
  - Rule C16: C16_w1.5 and C16_w2 are 0.0342 and 0.0341, against 0.0345 at w = 1. w = 3 is lower again.

### History-head readout (H1, rule HEAD; diagnostic only)

Recommending directly from H1's history head gives 0.0842 / 0.0912 / 0.0906 at R10 / R15 / R20. That is about 2.4 times H1's own C16 (0.0376). At R20 it recommends 940 distinct items, with 0.70 of recommendations in the day-window top 100.

- **What HEAD is:** a discriminative full-catalog scorer. It applies cosine similarity against the CF target vectors, which are the teacher's 256-d item vectors. It is not a generative output, not session-wise, and not a CF result.
- **What it shows:** the history encoder already carries the information to rank the targets well. The generation-and-landing path is not using it.
- **Comparison:** first-wave validation for H and TIGER is about 0.07. First-wave SASRec full **test** EV-NDCG@5 is 0.1089, a different split.

### RQ codebooks of E128 and E64

- **Selection:** pretraining ran 10,000 epochs. The selected checkpoint (`best_objective`) is epoch 50, the same rule and epoch as the first-wave 32-d RQ.
- **At the selected checkpoint, level-1 code use is:**
  - E128: 77%;
  - E64: 60%;
  - first-wave 32-d RQ: 84%.
- **Other levels and SIDs:** levels 2–4 use 100% in all three. Distinct SIDs: 10,673, 10,681 and 10,684.
- **Correction to the CPU-check note above:** it said the first-wave RQ used 100% after 10,000 epochs. That is true of later epochs, but the checkpoint actually used is epoch 50, where level 1 uses 84%.

### Boundaries

- Validation only, one training seed, a 20-R budget.
- With the C16 rule, single-checkpoint differences of about 0.003 are within R-to-R variation. B0's own rule A moves from 0.0192 at R19 to 0.0157 at R20.
