# Alternating Cross-PSE + DiffGRM (SIA-only)

This experiment initializes both sides from completed separate training: the
4x256 Cross-PSE v3 catalog tokenizer and its trained DiffGRM decoder. It then
adds the differentiable Sequence-Item Alignment path from ETEGRec. It is a new
run and does not overwrite either baseline.

## Optimization protocol

- Recommender phase: freeze Cross-PSE; optimize OCN loss + 0.1 SIA.
- Tokenizer phase: freeze DiffGRM; optimize reconstruction, balance, anchor,
  and 0.1 SIA.
- SIA: symmetric KL between per-position soft code distributions for the
  pooled sequence representation and target item OPQ slots.
- First evaluate and preserve the pretrained epoch-0 state. Start with 5
  recommender epochs to train the new adapter. Then alternate one short tokenizer
  phase and three recommender epochs for at most 10 tokenizer phases.
- Use at most 45 fine-tuning epochs: 35 recommender and 10 tokenizer-only.
- Refresh the full catalog's hard SIDs after every tokenizer phase.
- Each tokenizer phase uses 96 physical batches (6 optimizer updates under
  accumulation 16); the phase boundary never cuts through an accumulation.

## Commands

```bash
bash scripts/run_diff_crosspse_joint.sh preflight 2024 python
bash scripts/run_diff_crosspse_joint.sh smoke 2024 python
bash scripts/run_diff_crosspse_joint.sh formal 2024 python
bash scripts/run_diff_crosspse_joint.sh summarize 2024 python
```

The smoke run must show both `tokenizer` and `recommender` phases, finite SIA,
non-zero phase-appropriate gradient norms, a SID refresh record, and final test
metrics. It is not a reportable experiment.
