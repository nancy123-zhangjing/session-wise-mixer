# DiffGRM v6: RQ-KMeans tokenizer swap

This experiment changes only the Semantic ID tokenizer of the PSE v6
baseline. Both sides use Beauty, sentence-T5-base embeddings, PCA256 with
post-PCA normalization, a 4x256 SID layout, the same DiffGRM architecture,
training schedule, decoding configuration, and evaluation protocol.

The RQ-KMeans tokenizer seed is fixed at 2024. Decoder seeds 2024, 2025, and
2026 therefore share exactly one tokenizer. Each completed run records the
Semantic ID file path and SHA256 hash so that this can be verified.

Run on the server from `~/experiment1`:

```bash
bash scripts/run_diff_rqkmeans_v6.sh preflight 2024 python
bash scripts/run_diff_rqkmeans_v6.sh smoke 2024 python
bash scripts/run_diff_rqkmeans_v6.sh profile 2024 python
bash scripts/run_diff_rqkmeans_v6.sh formal 2024 python
```

To run all three decoder seeds from scratch in tmux:

```bash
tmux new -s diff-rqk
bash scripts/run_diff_rqkmeans_v6.sh all-seeds 2024 python
```

If the formal seed-2024 run has already completed, avoid rerunning it:

```bash
bash scripts/run_diff_rqkmeans_v6.sh remaining-seeds 2024 python
```

Detach with `Ctrl-b`, then `d`. Reconnect with:

```bash
tmux attach -t diff-rqk
```

Summarize completed runs or calculate CPD attention separately:

```bash
bash scripts/run_diff_rqkmeans_v6.sh summarize 2024 python
bash scripts/run_diff_rqkmeans_v6.sh compare 2024 python
bash scripts/run_diff_rqkmeans_v6.sh attention 2024 python
```

Do not set `force_regenerate_opq: true` between decoder seeds. The three
decoder seeds should report the same `Semantic ID SHA256` value.
