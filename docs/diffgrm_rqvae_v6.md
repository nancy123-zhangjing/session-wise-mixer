# DiffGRM v6 with a frozen RQ-VAE 4x256 tokenizer

This is a controlled tokenizer replacement for the PSE and RQ-KMeans v6
experiments. It uses the exact shared sentence-T5-base PCA256 item matrix,
trains one RQ-VAE tokenizer with seed 2024, and freezes its Semantic IDs for
all downstream DiffGRM decoder seeds.

The first-pass sequence on the server is:

```bash
bash scripts/run_diff_rqvae_v6.sh preflight 2024 python
bash scripts/run_diff_rqvae_v6.sh tokenizer-profile 2024 python
bash scripts/run_diff_rqvae_v6.sh tokenizer 2024 python
bash scripts/run_diff_rqvae_v6.sh smoke 2024 python
bash scripts/run_diff_rqvae_v6.sh profile 2024 python
bash scripts/run_diff_rqvae_v6.sh formal 2024 python
```

Do not start downstream training unless the formal tokenizer stage produced
`best.pt`, `semantic_ids.json`, `semantic_ids.npy`, `summary.json`, and
`geometry_summary.json` under:

```text
outputs/tokenizers/diff_rqvae/Beauty/seed2024/
```

After accepting decoder seed 2024:

```bash
bash scripts/run_diff_rqvae_v6.sh attention 2024 python
bash scripts/run_diff_rqvae_v6.sh remaining-seeds 2024 python
bash scripts/run_diff_rqvae_v6.sh compare 2024 python
```

The `semantic_ids_sha256` recorded by all three downstream runs must be
identical. The tokenizer is never retrained for decoder seeds 2025 or 2026.
