# Frozen Refined-PSE + DiffGRM Decoder continuation

## Purpose

This is the matched Decoder-only control for Joint Refined-PSE. It determines
whether the small improvement observed by Joint Refined-PSE comes from updating
the base PSE codebooks or simply from continuing to train the Decoder.

## Controlled setup

- Same offline Refined-PSE v2 catalog artifact and hard SID initialization.
- Same matching pretrained Refined-PSE DiffGRM checkpoint.
- Same Beauty data, seed 2024, model, BF16, LR, batch and evaluation protocol.
- Same 35-epoch Decoder update budget as Joint Refined-PSE.
- Base codebooks and all other tokenizer parameters are excluded from the
  optimizer; hard SIDs are never refreshed.

The trainer verifies after every epoch that all tokenizer parameters and hard
codes remain bitwise unchanged. The smoke checker additionally requires a zero
tokenizer gradient, recommender-only phases and identical initial/final SID
hashes.

## Commands

```bash
bash scripts/run_diff_refinedpse_frozen.sh preflight 2024 python
bash scripts/run_diff_refinedpse_frozen.sh smoke 2024 python
bash scripts/run_diff_refinedpse_frozen.sh formal 2024 python
bash scripts/run_diff_refinedpse_frozen.sh summarize 2024 python
```

Start the formal run only after the smoke run prints
`Frozen Refined-PSE smoke evidence: PASS`.

## Final comparison

Use validation results aligned at 5, 10, 15, 20, 25, 30 and 35 Decoder epochs.
The causal quantities are:

- `Joint Refined-PSE - Frozen Refined-PSE`: generic codebook-update effect;
- `Joint Cross-PSE - Frozen Cross-PSE`: Cross-PSE joint-update effect;
- difference between the two effects: additional value of cross-codebook
  interaction under joint optimization.
