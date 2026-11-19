# DiffGRM Beauty repository reproduction v6

V6 preserves the V5 repository-aligned Beauty setup: PSE 4x256, effective base
batch 1024, learning rate 0.01, 10,000 optimizer-update warmup, 200 epochs,
label smoothing 0.2, BF16, and finite-value guards.

## Why V6 exists

V5 evaluated every five epochs with patience five. Its seed-2024 run selected
epoch 45 but stopped at epoch 70, before the 10,000-update warmup ended near
epoch 96. That checkpoint remains a valid completed result, but the run did not
observe the full learning-rate schedule.

V6 keeps validation at epochs 20, 25, ..., 200, so early checkpoints and all
diagnostic curves are retained. Best-checkpoint selection is active throughout.
Only the patience counter is deferred until cumulative optimizer updates reach
`warmup_steps`.

## Expected schedule on the Beauty split

- Micro-batches per epoch: 1652
- Gradient accumulation: 16
- Optimizer updates per epoch: `ceil(1652 / 16) = 104`
- Warmup completion: between epochs 96 and 97
- First validation eligible for patience counting: epoch 100
- Patience: five unsuccessful validations
- Earliest possible early stop: epoch 120
- Maximum epoch: 200

The formal log must begin with:

```text
[RUN] kind=formal_v6
[TRAINING] physical_batch=64, accumulation=16, effective_batch=1024
[TRAINING] Evaluation config: start from epoch 20, interval: 5
[TRAINING] Early stopping: patience=5 evaluations, activation=after warmup update 10000
```

Before warmup completion, validation logs should state that early stopping is
deferred. Those validations may still replace the best checkpoint.
