# TIGER Mixer final tokenizer failure — 2026-11-14

AR-T1 finished all 20,000 RQ pretraining epochs (220,000 updates), selected the prescribed final checkpoint, then failed at recommendation initialization. Platform status is FAILED. The unchanged final-catalog guard correctly rejected a semantic bucket of 461 items because the suffix vocabulary has 256 entries. No items were dropped or assigned duplicate full IDs. This is not the earlier intermediate-diagnostic bug. Other four jobs continue.

| Final RQ state | AR-T0 | AR-T1 |
|---|---:|---:|
| Used centers per layer | 10 / 176 / 121 | 15 / 12 / 81 |
| Collision item fraction | 0.4183445 | 0.9908650 |
| Collision redundancy | 0.2819724 | 0.9665362 |
| Maximum semantic bucket | 87 | 461 |
| Reconstruction loss (dimension sum) | 0.9998108 | 19.4437992 |
| VQ loss (dimension and layer sum) | 400.9613713 | 3122.6679736 |

The T0 directory is valid but the pretraining quality is concerning. This pair cannot presently support an adequate-implementation TIGER performance claim. It also cannot establish that Mixer is generally harmful: both branches had large early losses, and the optimizer/initialization adaptation needs investigation.

## Verified source distinction and hypothesis

The [TIGER paper, section 4](https://arxiv.org/html/2305.05065v3) specifies Adagrad lr0.4, batch1024, 20,000 epochs and reports high codebook usage. It does not disclose the initial accumulator in that paragraph. The approved local plan section 5.2 explicitly filled this in as zero; runtime implements zero. Thus this is not a discrepancy between approved configuration and code.

[TensorFlow Adagrad documentation](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad) uses initial_accumulator_value=0.1. This is evidence of an alternative implementation default, **not evidence that TIGER used it**. With the current zero accumulator, a parameter's first update is approximately -0.4 sign(g) when |g| is much larger than epsilon. This can be large relative to small initial weights/codebook vectors. It is a plausible contributor, not an established causal diagnosis. Input scale, initialization and residual-gradient details remain other possibilities.

## Proposed scoped next step — not authorized, not executed

Keep the four active jobs and all failed artifacts unchanged. As a separate diagnostic, use the same TIGER common parent and same data order for T0/T1; change only Adagrad initial_accumulator_value from 0 to 0.1, preserve lr0.4/epsilon/batch/loss/Mixer, and run each for 200 pretraining epochs (2,200 updates). Record the first-step/early loss and weight scale plus existing 50-epoch code-health diagnostics. Use at most one H100 at a time and a total diagnostic cap of 0.25 GPU-hours. No recommender training or full five-arm rerun is included. Its result would inform a subsequent explicit paired-revision decision, not silently replace the existing branches.

This modifies a value explicitly locked in the approved design and adds training; it requires user direction. An alternative is to retain the strict current configuration and report T1 as a failed arm, which would leave the original all-five-completed Goal unmet. Do not enlarge suffix capacity, choose an earlier RQ checkpoint, change learning rate, or resubmit identical failing initialization automatically.

Evidence: AR_T0_STAGE_TRANSITION.json, AR_T1_STAGE_TRANSITION.json, raw/CURRENT_JOB_LIST.json and raw/CURRENT_PROGRESS_SNAPSHOT.json. Large checkpoints remain in the r2 remote root; no artifacts were removed.
