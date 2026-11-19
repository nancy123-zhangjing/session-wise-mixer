# Joint Refined-PSE + DiffGRM control

## Question

This experiment separates generic recommendation-driven PSE codebook updates
from cross-codebook interaction. It is the no-mixer counterpart of Joint
Cross-PSE DIGER-ST V2.

## Controlled comparison

Joint Refined-PSE retains the DIGER hard-forward/soft-backward bridge, one-sided
alternating optimization, schedule, learning rates, regularizers, Decoder
budget and SID-level evaluation used by Joint Cross-PSE V2. It initializes from
the completed offline Refined-PSE v2 catalog tokenizer and its matching DiffGRM
checkpoint.

The defining constraint is:

- base PSE codebooks: trainable in tokenizer phases;
- cross-codebook attention mixer: disabled and excluded from the optimizer;
- DiffGRM parameters: frozen in tokenizer phases;
- Refined-PSE parameters: frozen in recommender phases.

## Commands

```bash
bash scripts/run_diff_refinedpse_joint.sh preflight 2024 python
bash scripts/run_diff_refinedpse_joint.sh smoke 2024 python
bash scripts/run_diff_refinedpse_joint.sh formal 2024 python
bash scripts/run_diff_refinedpse_joint.sh summarize 2024 python
```

Start the formal run only after the smoke run prints
`Joint Refined-PSE smoke evidence: PASS`.

## Interpretation

- Frozen Cross-PSE vs Joint Refined-PSE: whether recommendation-driven
  codebook updates help beyond Decoder continuation.
- Joint Refined-PSE vs Joint Cross-PSE: whether the cross-codebook mixer adds
  value beyond trainable codebooks alone.
