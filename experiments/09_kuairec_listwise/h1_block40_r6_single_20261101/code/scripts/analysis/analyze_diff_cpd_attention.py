"""Analyze DiffGRM decoder self-attention along greedy CPD trajectories.

DiffGRM has no BOS token and no fixed left-to-right target order.  At each CPD
rank this script records attention from the position selected for decoding to
all SID positions, while marking every key as still MASKed or already revealed.
The official beam-256 recommendation metrics are unchanged; this is a
beam-width-one, sample-level diagnostic matching the decode-order statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFF_ROOT = PROJECT_ROOT / "external" / "DiffGRM"
RUN_ID = "diff_formal_pse_4x256_repo_repro_v6_bf16"
RQK_V6_RUN_ID = "diff_formal_rqkmeans_4x256_v6_bf16"
RQVAE_V6_RUN_ID = "diff_formal_rqvae_4x256_v6_bf16"
CROSSPSE_V1_RUN_ID = "diff_formal_crosspse_4x256_v1_bf16"
CROSSPSE_V2_RUN_ID = "diff_formal_crosspse_4x256_v2_bf16"
REFINEDPSE_V1_RUN_ID = "diff_formal_refinedpse_4x256_v1_bf16"
CROSSPSE_V3_RUN_ID = "diff_formal_crosspse_4x256_v3_bf16"
REFINEDPSE_V2_CATALOG_RUN_ID = "diff_formal_refinedpse_4x256_v2_catalog_bf16"
CROSSPSE_V3_CATALOG_RUN_ID = "diff_formal_crosspse_4x256_v3_catalog_bf16"
V5_RUN_ID = "diff_formal_pse_4x256_repo_repro_v5_bf16"
V4_RUN_ID = "diff_formal_pse_4x256_officialbatch_v4_bf16"
V3_RUN_ID = "diff_formal_pse_4x256_schedfix_v3_bf16"
V2_RUN_ID = "diff_formal_pse_4x256_schedfix_v2"
PILOT_RUN_ID = "diff_formal_pse_4x256"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DIFF_ROOT))

from genrec.pipeline import Pipeline  # noqa: E402
from runtime_paths import resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--run-id",
        choices=(
            RUN_ID, RQK_V6_RUN_ID, RQVAE_V6_RUN_ID,
            CROSSPSE_V1_RUN_ID, CROSSPSE_V2_RUN_ID,
            REFINEDPSE_V1_RUN_ID, CROSSPSE_V3_RUN_ID,
            REFINEDPSE_V2_CATALOG_RUN_ID, CROSSPSE_V3_CATALOG_RUN_ID,
            V5_RUN_ID, V4_RUN_ID, V3_RUN_ID, V2_RUN_ID, PILOT_RUN_ID,
        ),
        default=RUN_ID,
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def latest_metadata(seed: int, run_id: str) -> Path:
    matches = []
    root = PROJECT_ROOT / "outputs" / "run_metadata"
    for path in root.rglob("*_run_metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("run_id") == run_id and int(data.get("seed", -1)) == seed:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No completed {run_id} seed {seed} metadata found")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def resolve_diff_path(value: str | Path) -> Path:
    return resolve_project_path(value, project_root=PROJECT_ROOT, relative_to=DIFF_ROOT)


def safe_std(total: float, square_total: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    mean = total / count
    return math.sqrt(max(square_total / count - mean * mean, 0.0))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_attention(rows: list[dict], output: Path, n_digit: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, n_digit, figsize=(4.0 * n_digit, 3.8), dpi=180)
    if n_digit == 1:
        axes = [axes]
    image = None
    for rank, ax in enumerate(axes, start=1):
        matrix = np.full((n_digit, n_digit), np.nan, dtype=float)
        for query in range(n_digit):
            for key in range(n_digit):
                values = [
                    float(row["mean_attention"])
                    for row in rows
                    if int(row["decode_rank"]) == rank
                    and int(row["query_position"]) == query
                    and int(row["key_position"]) == key
                ]
                if values:
                    weights = [
                        int(row["n_layer_head_examples"])
                        for row in rows
                        if int(row["decode_rank"]) == rank
                        and int(row["query_position"]) == query
                        and int(row["key_position"]) == key
                    ]
                    matrix[query, key] = float(np.average(values, weights=weights))
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="equal")
        for query in range(n_digit):
            for key in range(n_digit):
                if not np.isnan(matrix[query, key]):
                    ax.text(
                        key, query, f"{matrix[query, key]:.2f}",
                        ha="center", va="center",
                        color="white" if matrix[query, key] > 0.5 else "black",
                    )
        ax.set_title(f"CPD decode rank {rank}")
        ax.set_xlabel("Key SID position")
        ax.set_ylabel("Selected query position")
        ax.set_xticks(range(n_digit), [f"SID{i}" for i in range(n_digit)])
        ax.set_yticks(range(n_digit), [f"SID{i}" for i in range(n_digit)])
    if image is not None:
        fig.colorbar(image, ax=list(axes), label="Mean self-attention", shrink=0.78)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_samples < 0:
        raise ValueError("batch-size must be positive and max-samples non-negative")

    metadata_path = (
        resolve_project_path(args.metadata, project_root=PROJECT_ROOT, relative_to=Path.cwd())
        if args.metadata
        else latest_metadata(args.seed, args.run_id)
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != args.run_id or int(metadata.get("seed", -1)) != args.seed:
        raise ValueError("Metadata does not match the requested DiffGRM run")
    checkpoint = (
        resolve_project_path(args.checkpoint, project_root=PROJECT_ROOT, relative_to=Path.cwd())
        if args.checkpoint
        else resolve_diff_path(metadata["checkpoint"])
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    os.chdir(DIFF_ROOT)
    config_files = [
        str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_common.yaml"),
    ]
    if args.run_id == RQK_V6_RUN_ID:
        config_files.extend([
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml"),
            str(PROJECT_ROOT / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_rqkmeans_v6.yaml"),
        ])
    elif args.run_id == RQVAE_V6_RUN_ID:
        config_files.extend([
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml"),
            str(PROJECT_ROOT / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_rqvae_v6.yaml"),
        ])
    elif args.run_id == CROSSPSE_V1_RUN_ID:
        config_files.extend([
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml"),
            str(PROJECT_ROOT / "configs" / "04_crosspse_offline" / "diff_formal_crosspse_v1.yaml"),
        ])
    elif args.run_id == CROSSPSE_V2_RUN_ID:
        config_files.extend([
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml"),
            str(PROJECT_ROOT / "configs" / "04_crosspse_offline" / "diff_formal_crosspse_v2.yaml"),
        ])
    elif args.run_id == REFINEDPSE_V1_RUN_ID:
        config_files.extend([
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml"),
            str(PROJECT_ROOT / "configs" / "04_crosspse_offline" / "diff_formal_refinedpse_v1.yaml"),
        ])
    elif args.run_id == CROSSPSE_V3_RUN_ID:
        config_files.extend([
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml"),
            str(PROJECT_ROOT / "configs" / "04_crosspse_offline" / "diff_formal_crosspse_v3.yaml"),
        ])
    elif args.run_id == REFINEDPSE_V2_CATALOG_RUN_ID:
        config_files.extend([
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml"),
            str(
                PROJECT_ROOT
                / "configs" / "04_crosspse_offline" / "diff_formal_refinedpse_v2_catalog.yaml"
            ),
        ])
    elif args.run_id == CROSSPSE_V3_CATALOG_RUN_ID:
        config_files.extend([
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml"),
            str(
                PROJECT_ROOT
                / "configs" / "04_crosspse_offline" / "diff_formal_crosspse_v3_catalog.yaml"
            ),
        ])
    else:
        config_files.append(
            str(PROJECT_ROOT / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_pse.yaml")
        )
    if args.run_id == RUN_ID:
        config_files.append(
            str(PROJECT_ROOT / "configs" / "shared" / "diff_formal_repo_v6.yaml")
        )
    elif args.run_id == V5_RUN_ID:
        config_files.append(
            str(PROJECT_ROOT / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_repo_v5.yaml")
        )
    elif args.run_id == V4_RUN_ID:
        config_files.append(
            str(PROJECT_ROOT / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_bf16_v4.yaml")
        )
    elif args.run_id == V3_RUN_ID:
        config_files.append(
            str(PROJECT_ROOT / "configs" / "03_diffgrm_tokenizer_comparison" / "diff_formal_bf16_v3.yaml")
        )
    pipeline = Pipeline(
        model_name="DIFF_GRM",
        dataset_name="AmazonReviews2014",
        checkpoint_path=str(checkpoint),
        trainer=object(),
        config_dict={
            "rand_seed": args.seed,
            "eval_batch_size": args.batch_size,
            "force_regenerate_opq": False,
        },
        config_file=config_files,
    )
    expected_sid_hash = metadata.get("semantic_ids_sha256")
    actual_sid_hash = getattr(pipeline.tokenizer, "semantic_ids_sha256", None)
    if expected_sid_hash and actual_sid_hash != expected_sid_hash:
        raise RuntimeError(
            "Tokenizer mismatch: checkpoint metadata and reconstructed "
            f"Semantic IDs have different SHA256 values "
            f"({expected_sid_hash} != {actual_sid_hash})"
        )
    model = pipeline.model
    device = torch.device(pipeline.config["device"])
    model.to(device)
    model.eval()
    for block in model.decoder_blocks:
        block.self_attn.capture_attention = True
        block.cross_attn.capture_attention = False

    dataset = pipeline.tokenized_datasets[args.split]
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn[args.split],
    )

    n_digit = int(pipeline.config["n_digit"])
    n_layers = len(model.decoder_blocks)
    n_heads = model.decoder_blocks[0].self_attn.n_head
    # Dimensions: layer, head, decode rank, query position, key position,
    # key state (0=revealed, 1=masked).
    shape = (n_layers, n_heads, n_digit, n_digit, n_digit, 2)
    sums = torch.zeros(shape, dtype=torch.float64)
    square_sums = torch.zeros_like(sums)
    counts = torch.zeros(shape, dtype=torch.long)
    selected_position_counts = torch.zeros(n_digit, n_digit, dtype=torch.long)
    max_row_sum_error = 0.0
    processed = 0

    progress = tqdm(dataloader, desc=f"Diff CPD attention seed{args.seed}")
    with torch.inference_mode():
        for batch in progress:
            labels = batch["labels"].to(device).clamp(0, pipeline.config["codebook_size"] - 1)
            encoder_hidden = model(batch, return_loss=False).hidden_states
            batch_size = labels.size(0)
            batch_index = torch.arange(batch_size, device=device)
            decoder_input = torch.zeros_like(labels)
            unresolved = torch.ones(batch_size, n_digit, dtype=torch.bool, device=device)

            for rank in range(n_digit):
                outputs = model.forward_decoder_only(
                    {
                        "decoder_input_ids": decoder_input,
                        "encoder_hidden": encoder_hidden,
                        "mask_positions": unresolved.float(),
                    },
                    return_loss=False,
                    digit=None,
                    use_cache=False,
                )
                probabilities = F.softmax(outputs.logits, dim=-1)
                confidence, predicted = probabilities.max(dim=-1)
                confidence = confidence.masked_fill(~unresolved, -1.0)
                next_position = confidence.argmax(dim=1)
                selected_position_counts[rank].scatter_add_(
                    0,
                    next_position.cpu(),
                    torch.ones(batch_size, dtype=torch.long),
                )

                for layer_index, block in enumerate(model.decoder_blocks):
                    attention = block.self_attn.last_attention
                    if attention is None:
                        raise RuntimeError(f"Layer {layer_index + 1} did not capture attention")
                    attention_cpu = attention.to(dtype=torch.float64, device="cpu")
                    unresolved_cpu = unresolved.cpu()
                    next_position_cpu = next_position.cpu()
                    selected_rows = attention_cpu[
                        torch.arange(batch_size), :, next_position_cpu, :
                    ]
                    max_row_sum_error = max(
                        max_row_sum_error,
                        float((selected_rows.sum(-1) - 1.0).abs().max().item()),
                    )
                    for query_position in range(n_digit):
                        query_mask = next_position_cpu.eq(query_position)
                        if not bool(query_mask.any()):
                            continue
                        for key_position in range(n_digit):
                            key_masked = unresolved_cpu[query_mask, key_position]
                            values = selected_rows[query_mask, :, key_position]
                            for key_state in (0, 1):
                                state_mask = key_masked.eq(bool(key_state))
                                if not bool(state_mask.any()):
                                    continue
                                selected = values[state_mask]
                                sums[layer_index, :, rank, query_position, key_position, key_state] += selected.sum(0)
                                square_sums[layer_index, :, rank, query_position, key_position, key_state] += selected.square().sum(0)
                                counts[layer_index, :, rank, query_position, key_position, key_state] += selected.size(0)
                    block.self_attn.last_attention = None

                next_token = predicted[batch_index, next_position]
                decoder_input[batch_index, next_position] = next_token
                unresolved[batch_index, next_position] = False

            processed += batch_size
            progress.set_postfix(examples=processed)

    if max_row_sum_error > 1e-5:
        raise RuntimeError(f"Attention row-sum validation failed: {max_row_sum_error:.3e}")

    head_rows = []
    layer_rows = []
    overall_rows = []
    state_names = {0: "revealed", 1: "masked"}
    for layer in range(n_layers):
        for rank in range(n_digit):
            for query in range(n_digit):
                for key in range(n_digit):
                    for state in (0, 1):
                        for head in range(n_heads):
                            count = int(counts[layer, head, rank, query, key, state])
                            if not count:
                                continue
                            total = float(sums[layer, head, rank, query, key, state])
                            square = float(square_sums[layer, head, rank, query, key, state])
                            head_rows.append({
                                "seed": args.seed,
                                "split": args.split,
                                "decoder_layer": layer + 1,
                                "head": head + 1,
                                "decode_rank": rank + 1,
                                "query_position": query,
                                "query_token": f"SID{query}",
                                "key_position": key,
                                "key_token": f"SID{key}",
                                "key_state": state_names[state],
                                "mean_attention": total / count,
                                "std_attention": safe_std(total, square, count),
                                "n_examples": count,
                            })
                        count = int(counts[layer, :, rank, query, key, state].sum())
                        if count:
                            total = float(sums[layer, :, rank, query, key, state].sum())
                            square = float(square_sums[layer, :, rank, query, key, state].sum())
                            layer_rows.append({
                                "seed": args.seed,
                                "split": args.split,
                                "decoder_layer": layer + 1,
                                "decode_rank": rank + 1,
                                "query_position": query,
                                "query_token": f"SID{query}",
                                "key_position": key,
                                "key_token": f"SID{key}",
                                "key_state": state_names[state],
                                "mean_attention": total / count,
                                "std_attention_over_heads_examples": safe_std(total, square, count),
                                "n_head_examples": count,
                            })

    for rank in range(n_digit):
        for query in range(n_digit):
            for key in range(n_digit):
                for state in (0, 1):
                    count = int(counts[:, :, rank, query, key, state].sum())
                    if not count:
                        continue
                    total = float(sums[:, :, rank, query, key, state].sum())
                    square = float(square_sums[:, :, rank, query, key, state].sum())
                    overall_rows.append({
                        "seed": args.seed,
                        "split": args.split,
                        "decode_rank": rank + 1,
                        "query_position": query,
                        "query_token": f"SID{query}",
                        "key_position": key,
                        "key_token": f"SID{key}",
                        "key_state": state_names[state],
                        "mean_attention": total / count,
                        "std_attention_over_layers_heads_examples": safe_std(total, square, count),
                        "n_layer_head_examples": count,
                    })

    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else PROJECT_ROOT / "outputs" / "attention" / args.run_id
    )
    output_dir = output_root / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "attention_by_head.csv", head_rows)
    write_csv(output_dir / "attention_by_layer.csv", layer_rows)
    write_csv(output_dir / "attention_overall.csv", overall_rows)
    write_csv(
        output_dir / "selected_position_frequency.csv",
        [
            {
                "seed": args.seed,
                "split": args.split,
                "decode_rank": rank + 1,
                "position": position,
                "count": int(selected_position_counts[rank, position]),
                "frequency": (
                    int(selected_position_counts[rank, position])
                    / max(int(selected_position_counts[rank].sum()), 1)
                ),
            }
            for rank in range(n_digit)
            for position in range(n_digit)
        ],
    )
    run_info = {
        "run_id": args.run_id,
        "seed": args.seed,
        "split": args.split,
        "checkpoint": str(checkpoint),
        "metadata": str(metadata_path),
        "examples": processed,
        "diagnostic_path": "beam-width-one greedy CPD",
        "attention_stage": "masked softmax before dropout",
        "decoder_layers": n_layers,
        "attention_heads": n_heads,
        "sid_positions": n_digit,
        "has_bos": False,
        "key_state_definition": {
            "masked": "position unresolved before the current CPD update",
            "revealed": "position predicted in an earlier CPD rank",
        },
        "max_row_sum_error": max_row_sum_error,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        plot_attention(overall_rows, output_dir / "cpd_self_attention_heatmaps.png", n_digit)
    except ImportError as error:
        print(f"[WARN] Skipped attention plot: {error}")
    print(f"Saved DiffGRM CPD attention analysis to {output_dir}")
    print(f"examples={processed}, max_row_sum_error={max_row_sum_error:.3e}")


if __name__ == "__main__":
    main()
