# DiffGRM Beauty repository reproduction v5

This run intentionally follows the authors' executable repository protocol.
It does not silently mix the paper table with repository defaults.

## Source reconciliation

- Paper Table 7: `lr=0.01`, `warmup_steps=10000`, `max_epochs=100`,
  `label_smoothing=0.1`.
- Official Beauty README command: `train_batch_size=1024`, `lr=0.01`,
  `label_smoothing=0.2`.
- Official `DIFF_GRM/config.yaml`: `epochs=200`; the README command does not
  override this value.

Therefore v5 uses the executable repository combination: base batch 1024,
learning rate 0.01, 10,000 optimizer-update warmup, 200 epochs, and label
smoothing 0.2. The paper discrepancies remain recorded in config and metadata.

Full CPD validation is much slower than training on the single RTX 3080. The
repository default evaluates every epoch; v5 evaluates every five epochs and
uses patience 5, corresponding to about 25 epochs without improvement. This is
a compute adaptation that changes only checkpoint/early-stop granularity, not
optimizer updates or the learning-rate schedule.

## 10 GB GPU adaptation

The server cannot fit a physical batch of 1024. It uses physical batch 64 and
16-way gradient accumulation, giving an effective base batch of 1024. BF16 and
fail-fast finite checks are numerical engineering adaptations.

## Expected schedule on the current Beauty split

- Micro-batches per epoch: 1652
- Optimizer updates per epoch: `ceil(1652 / 16) = 104`
- Total optimizer updates: `200 * 104 = 20800`
- Warmup ends near epoch: `10000 / 104 = 96.15`
- Cosine-decay phase: about 104 epochs
- Full validation epochs: 20, 25, ..., 200 (at most 37 evaluations)
- Early stopping: 5 unsuccessful evaluations (about 25 epochs)

The formal run must log `accumulation=16`, `effective_batch=1024`, and
`updates/epoch=104`. Any other values mean that the configuration overlay did
not load and the run must be stopped.
