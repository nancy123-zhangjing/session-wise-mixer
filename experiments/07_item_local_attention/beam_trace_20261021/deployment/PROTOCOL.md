# Stage07 validation search trace, 2026-10-21

Purpose: locate where small/wide-beam target predictions are lost, and compare paired A/B outcomes. This is diagnostic evaluation of existing models, not a new training experiment.

## Frozen controls

- A control and B item-local attention, selected epoch25, seed2024, each with its own saved SID mapping; original alternating-joint training.
- Original root commit `842ff2aa3b0ba95d0ba2a66dea74c485abaa11a1`; engine `1c879252f6395d6e64462511823c3929b9c42c00`.
- A checkpoint SHA `55960f52906b0c4972a90b0ad06c078ef12fceac10fa4400b113e41d9a5c1f5f`; B `f093c22e4e0cb354870b467e41f4ebab96d90047daf9d0e3d33991e3108b348b`.
- Four cells: A/B x beam32/256. Same full 22363 validation users, history50, batch8, Top10, confidence decoding, simple legal filtering/deduplication, original BF16 encoder/custom decoder precision scope.
- No training, optimizer steps, SID refresh, test examples/evaluation, candidate-rule changes or target-guided decisions.
- New isolated remote root `/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage07/beam_trace_20261021_r1`.
- Prior reference is read-only `beam_validation_20261021_r1/results`; every new per-user baseline field must match its corresponding prior reference exactly.

## Observations

1. Rounds1/2/3: target-compatible paths before and after global pruning; compatibility follows arbitrary filled positions, not a left-to-right prefix. Candidate counts before pruning refer to logically valid masked-position expansions.
2. Each round: retained paths versus distinct partial states; paths/states with at least one catalog SID completion.
3. Final round: target rank and ties in each surviving compatible branch, before the original per-branch argmax; do not expand the beam differently.
4. Full final candidates before filtering: target presence, legal/distinct counts, actual score-sorted target rank after legal filtering/deduplication.
5. Mutually exclusive fate: lost at round1/2/3, lost at final argmax, generated but below Top10, hit. Pair users across all four cells.

## Acceptance and resources

- CPU tests verify pinned AST preserves original statements, target compatibility with arbitrary masks, duplicate/legality counts, final argmax/ties/rank cutoff, and unchanged decoder outputs/RNG.
- CPU preflight binds original files, 13 input artifacts, saved SID and model tensors, plus evaluator/observer hashes.
- Each GPU worker first compares original and observed decode exactly on the same batch, then compares the entire new prediction stream against the archived reference. Any mismatch or incomplete trace fails closed.
- One node, four H100s, one independent process per cell, no DDP. Platform limit30 minutes; process limit1700 seconds; aggregate allocation cap2 GPU hours for this submission. No automatic retry. Read actual speed after100 batches; stop if projected completion exceeds the bound.
- 8 GPUs would not help this four-cell plan without further data sharding and extra setup. If four-card placement cannot start promptly, two-card queuing is the fallback, with a new explicit execution receipt and no duplicate job.
- Completion requires four full cells, exact original prediction reproduction, source/weight/SID immutability, verified local archive, and confirmed GPU release.

## Interpretation limits

SID-level diagnostics, including catalog collisions. One selected training seed, exploratory validation evidence. Duplicate/invalid-path occupancy describes search inefficiency but alone does not establish causality of B's gain. Instrumented runtime includes observer overhead and is not serving latency. No stable-gain claim or test retuning from these results.
