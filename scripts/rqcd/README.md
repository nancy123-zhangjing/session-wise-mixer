# RQCD-H implementation

Implements the fixed-RQ, sequential five-item continuous-diffusion baseline in
[`RQ_CONTINUOUS_SESSION_BASELINE_PROTOCOL_20261104.md`](../../docs/research/onerec_listwise/RQ_CONTINUOUS_SESSION_BASELINE_PROTOCOL_20261104.md).
This is a new module; the historical DiffGRM, RQ-VAE trainer and their outputs are
not modified. Importing this package never starts a workload.

## Entry points

Use the project root as working directory, Python 3.10+ and PyTorch 2.6+,
NumPy and PyArrow. No text encoder, Transformers, FAISS or external GenRec
installation is required for this path: it reads existing PCA content vectors.
The CUDA build of PyTorch must support H100. Do not install/upgrade dependencies
inside a shared historical environment as part of launching this experiment.

```text
python -m scripts.rqcd.run inspect
python -B tests/test_rqcd.py -v
```

These following commands are **future execution examples**, not evidence that
formal training or H100 testing has run. Each output directory must be new;
only an explicit `--resume` may reopen it. No command allocates remote resources.

### Bounded H100 acceptance and throughput measurement

Run on an already available, authorized H100:

```text
python -m scripts.rqcd.preflight --device cuda:0 --require-h100 --updates 3 --output outputs/rqcd_h100_preflight_NEW
```

This uses synthetic data and the full network dimensions. It checks the optimized
views against a sequential FP32 reference, a real BF16/fused AdamW update, exact
next-update restoration, frozen tables, RQ/Sinkhorn gradients and complete
five-item generation. It records peak memory, update time and the **actual
attention/optimizer operators** selected by the profiler. Timed updates exclude
warmup and profiling. The synthetic catalog has 512 items; this is not a formal
full-catalog throughput estimate.

`--require-h100` fails on CPU or a different GPU. For a local full-architecture
CPU diagnostic, use `--device cpu` without that flag; it explicitly uses FP32,
8 synthetic lists per update, and cannot qualify CUDA/BF16 kernels.

### Stage A: fit and export fixed RQ

```text
python -m scripts.rqcd.run fit-tokenizer --dataset outputs/kuairec_listwise_v1/run1 --content outputs/kuairec_listwise_assets_v1/content_features.npy --device cuda:0 --output outputs/rqcd_h_v1_NEW/tokenizer
```

Outputs include `best.pt`, `last.pt`, `initial.pt`, the exact content split and
`catalog.pt`. The catalog contains raw latent vectors, deterministic SIDs, books,
reconstruction errors and training-target-fitted scaling. `COMPLETE.json` is
written only after export succeeds. The default config verifies the approved
content SHA and existing dataset manifest; it never refits BGE/PCA.

The standalone RQ implementation follows the existing first-three-STE /
last-Sinkhorn residual gradient semantics. K-means is bounded at 100 iterations,
as specified in the new protocol. It uses deterministic vectorized membership
reductions rather than nondeterministic GPU scatter atomics. Initialization RNG
is isolated; this is not a claim of bitwise identity with historical Beauty RQ.

### Stage B: train, validate and seal

```text
python -m scripts.rqcd.run train --dataset outputs/kuairec_listwise_v1/run1 --catalog outputs/rqcd_h_v1_NEW/tokenizer/catalog.pt --device cuda:0 --output outputs/rqcd_h_v1_NEW/recommender
```

The default 8192-row probe manifest is read from the protocol's existing path.
An alternate physical copy can be passed using `--probe-manifest`; population,
split SHA, indices and IDs must match. No old diagnostics are rerun.

`train` includes per-epoch teacher-forced diagnostic MSE, the 11 locked full
validation points and best/R35 selection. It does **not** automatically test.
Each epoch is saved independently; optimizer-boundary `last.pt` is written
every 100 updates and before expensive validation. A failed validation can
resume from its pending epoch boundary. An incomplete prediction `.partial`
file does not count as a completed evaluation.

```text
python -m scripts.rqcd.run train --dataset outputs/kuairec_listwise_v1/run1 --catalog outputs/rqcd_h_v1_NEW/tokenizer/catalog.pt --device cuda:0 --output outputs/rqcd_h_v1_NEW/recommender --resume
```

The source, config, inputs, environment, asset, optimizer clock and RNG must
match. Checkpoints are trusted local Torch files; do not load untrusted pickles.
A `.running` lock prevents concurrent writers. Following an abrupt process
death, inspect process ownership before manually removing a stale lock.

### Test only a sealed completed run

```text
python -m scripts.rqcd.run test --dataset outputs/kuairec_listwise_v1/run1 --catalog outputs/rqcd_h_v1_NEW/tokenizer/catalog.pt --run outputs/rqcd_h_v1_NEW/recommender --device cuda:0 --output outputs/rqcd_h_v1_NEW/test
```

Only sealed best and final checkpoints can be evaluated. If best is R35, one
prediction file is reused. Output includes item predictions, targets, sample/user
IDs, timestamps, metrics and raw-versus-constrained quantization diagnostics.
There is one generated list, no beam scoring or hidden reranking.

```text
python -m scripts.rqcd.analysis --predictions outputs/rqcd_h_v1_NEW/test/best/predictions.jsonl.gz --split test --output outputs/rqcd_h_v1_NEW/test_analysis
```

Analysis computes sample-weighted user-cluster bootstrap intervals, output
concentration and (for `--split validation`) the locked temporal split.
`--reference` accepts **already saved** tables with the same explicit
`sample_id/user_id/target/metrics` schema. It refuses unmatched IDs or targets;
historical formats require a reviewed adapter, never a new inference run.
No pairing is silently inferred from row order. This utility is not an automatic
reproduction of every historical cold/head/mid/tail audit.

## H100 implementation decisions

| Path | Optimization | Scientific boundary |
|---|---|---|
| Training | All five target views in one decoder batch | Each retains its own causal prefix; no future-target visibility |
| History | Encode once per list batch, project cross-attention K/V once per layer | Shared gradients accumulate from all five views |
| Attention | `scaled_dot_product_attention` auto-dispatch | Determinism remains enabled; actual kernel availability is measured |
| Precision | BF16 training, FP32 parameters/MSE/Adam state | No FP8, no TF32; RQ and final generation retain protocol precision |
| Optimizer | Fused AdamW on CUDA | Same LR/betas/epsilon/decay; CPU uses reference implementation |
| Data | Pack Arrow columns once, optionally retain compact ID tensors on GPU | No row-by-row Python collation in the training loop |
| Random labels | Generate timestep/noise for the full effective batch before splitting | Same labels for different microbatch partitioning; dropout RNG still shape-dependent |
| Accumulation | Weight each microbatch by actual group size | Final 835-list group is not divided by 8 blindly |
| Generation | Cache history K/V across 250 decoder calls | FP32/no-TF32 generation, same fixed DDIM trajectory |
| Catalog | Batched GPU posting masks and distance operations | Greedy constrained RQ, exact tie policy, no ANN/popularity fallback |

`torch.compile`, FP8, CFG, EMA and gradient checkpointing remain off.
Do not disable deterministic algorithms just to force a faster attention kernel.
With deterministic algorithms and padding masks, a PyTorch build may choose math
or another fused backend instead of FlashAttention; the preflight records this.

The 128-list microbatch is a protocol choice, not an H100 memory-tuned optimum.
Changing it requires a documented configuration revision and validation because
dropout random-number consumption can change even when the expected objective
does not. No automatic batch/precision/step-count fallback is implemented.

## File map

- `model.py`: pre-LN SDPA network, independent five-view training, DDIM.
- `tokenizer.py`: bounded K-means, RQ-VAE, Sinkhorn, health and scaling.
- `catalog.py`: immutable tables, constrained quantization and generation.
- `data.py`: validated Parquet-to-tensor packing.
- `training.py`: weighted accumulation, RNG and complete state restoration.
- `run.py`: input binding, RQ/train/test lifecycle and sealed selection.
- `evaluation.py`: streaming predictions, true-item metrics and MSE probe.
- `analysis.py`: saved-prediction statistics; no model invocation.
- `preflight.py`: bounded synthetic device acceptance/measurement.

Local tests validate semantics and restoration, not H100 performance or
recommendation quality. A real H100 preflight and real-asset startup gate are
still required before a formal run.
