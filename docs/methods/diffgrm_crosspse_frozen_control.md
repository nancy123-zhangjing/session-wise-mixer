# Frozen Cross-PSE + DiffGRM Decoder continuation

## Purpose

This is the matched control for the DIGER-ST V2 joint run. It tests whether
the apparent downstream gain can be explained by additional Decoder training
without any recommendation-driven Tokenizer update.

## Controlled setup

- Same Cross-PSE v3 artifact and hard SID initialization.
- Same pretrained DiffGRM checkpoint.
- Same data, seed 2024, model, batch, BF16, LR and 500-update warmup.
- Same total Decoder budget as DIGER-ST V2: 35 Decoder epochs.
- Same evaluation interval, early stopping and SID-level metrics.
- Cross-PSE parameters are excluded from the optimizer and hard SIDs are never
  refreshed.

The trainer checks after every epoch that all Cross-PSE parameters and hard
codes are bitwise unchanged. The completed run metadata stores the initial and
final SID hashes and the invariant result.

## Commands

```bash
bash scripts/run_diff_crosspse_frozen.sh preflight 2024 python
bash scripts/run_diff_crosspse_frozen.sh smoke 2024 python
bash scripts/run_diff_crosspse_frozen.sh formal 2024 python
bash scripts/run_diff_crosspse_frozen.sh summarize 2024 python
```

The smoke run must end with `Frozen Cross-PSE smoke evidence: PASS` before the
formal run is started.
