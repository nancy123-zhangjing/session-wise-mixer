# Refined-PSE / Cross-PSE v3 controlled experiment

This experiment separates two effects while keeping the DiffGRM decoder and
the original 4x256 PSE initialization fixed.

- **Original PSE**: frozen OPQ/PQ codebooks (existing baseline).
- **Refined-PSE v1**: train the PSE codebooks with reconstruction, balance and
  anchor losses, but disable all cross-codebook mixing.
- **Cross-PSE v3**: use the same trainable codebooks and objective, and add one
  full self-attention block over all 4x256 codewords.

Both new variants use the same catalog split, seed, optimization schedule and
step-0 validation baseline. The only intended difference between Refined-PSE
and Cross-PSE v3 is whether the mixer is enabled. Tokenizer artifacts are
frozen before training DiffGRM, so the downstream comparison does not jointly
optimize the tokenizer and recommendation decoder.

The two variants also load the exact same cached base PSE artifact from
`outputs/tokenizers/diff_pse_base/Beauty/seed2024`. The paired audit checks the
base SID and codebook hashes before accepting the comparison.

The tokenizer must be inspected before downstream training. A useful learned
variant should select a checkpoint after step 0, change a non-trivial fraction
of item SIDs, preserve codebook utilization, and avoid a large collision-rate
regression. If step 0 remains best, the result means the tokenizer objective
did not improve over the original PSE initialization under this protocol.

## Corrected catalog-selection protocol

The first Refined-PSE v1 / Cross-PSE v3 run exposed a protocol mismatch: the
base PSE had already fitted all 12061 catalog-fit items, but refinement used
10855 of them and selected checkpoints on the remaining 1206. This gave step 0
an unfair initialization advantage and both learned variants reverted to the
original PSE.

The corrected variants are Refined-PSE v2-catalog and Cross-PSE v3-catalog.
They both train and select checkpoints on exactly the original 12061-item PSE
fit catalog. The 40-item mask complement is reported only as an external
diagnostic. New output directories preserve the failed pilot. Downstream
stages run a strict readiness check and stop if step 0 remains best, the SID
mapping is unchanged, or codebook utilization collapses.
