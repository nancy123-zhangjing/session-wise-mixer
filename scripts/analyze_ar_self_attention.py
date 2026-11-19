"""Collect decoder self-attention from a trained AR_GRM checkpoint.

The analysis uses ground-truth prefixes (teacher forcing), never calls beam
search, and records the masked softmax probabilities before attention dropout.
Normal training/inference is unchanged because attention capture is opt-in.
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFF_ROOT = PROJECT_ROOT / "external" / "DiffGRM"
sys.path.insert(0, str(DIFF_ROOT))

from genrec.pipeline import Pipeline  # noqa: E402


VARIANTS = {
    # Historical stage-1 variants.  Keep these keys so the old launcher and
    # supplement remain reproducible.
    "rq_kmeans": {
        "config": PROJECT_ROOT / "configs" / "ar_formal_rqkmeans.yaml",
        "run_id": "ar_formal_rqkmeans_3x256",
        "sid_quantizer": "rq_kmeans",
        "report": PROJECT_ROOT / "outputs" / "reports" / "formal_3x256" / "per_seed.csv",
        "report_column": "quantizer",
        "report_value": "rq_kmeans",
        "artifact_dir": "rqkmeans",
    },
    "rqvae": {
        "config": PROJECT_ROOT / "configs" / "ar_formal_rqvae.yaml",
        "run_id": "ar_formal_rqvae_3x256",
        "sid_quantizer": "rqvae",
        "report": PROJECT_ROOT / "outputs" / "reports" / "formal_3x256" / "per_seed.csv",
        "report_column": "quantizer",
        "report_value": "rqvae",
        "artifact_dir": "rqvae",
    },
    # New single-seed internal controls.
    "pse_256x256x256": {
        "config": PROJECT_ROOT / "configs" / "ar_formal_pse.yaml",
        "run_id": "ar_formal_pse_3x256",
        "sid_quantizer": "opq_pq",
        "report": PROJECT_ROOT / "outputs" / "reports" / "ar_internal_study" / "per_seed.csv",
        "report_column": "variant",
        "report_value": "pse_256x256x256",
        "artifact_dir": None,
    },
    "rqk_128x256x512": {
        "config": PROJECT_ROOT / "configs" / "ar_formal_rqkmeans_128_256_512.yaml",
        "run_id": "ar_formal_rqkmeans_128x256x512",
        "sid_quantizer": "rq_kmeans",
        "report": PROJECT_ROOT / "outputs" / "reports" / "ar_internal_study" / "per_seed.csv",
        "report_column": "variant",
        "report_value": "rqk_128x256x512",
        "artifact_dir": "rqkmeans_128x256x512",
    },
    "rqk_512x256x128": {
        "config": PROJECT_ROOT / "configs" / "ar_formal_rqkmeans_512_256_128.yaml",
        "run_id": "ar_formal_rqkmeans_512x256x128",
        "sid_quantizer": "rq_kmeans",
        "report": PROJECT_ROOT / "outputs" / "reports" / "ar_internal_study" / "per_seed.csv",
        "report_column": "variant",
        "report_value": "rqk_512x256x128",
        "artifact_dir": "rqkmeans_512x256x128",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        "--quantizer",
        dest="variant",
        choices=sorted(VARIANTS),
        required=True,
        help="Experiment identity. --quantizer is retained as a backward-compatible alias.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 uses the complete split; a small value is useful for smoke tests.",
    )
    parser.add_argument(
        "--per-seed-report",
        type=Path,
        help="Optional report override; by default each variant uses its matching report.",
    )
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "attention" / "formal_3x256",
    )
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def metadata_from_report(report: Path, variant: str, seed: int) -> Path:
    if not report.exists():
        raise FileNotFoundError(
            f"Formal per-seed report not found: {report}\n"
            "Run scripts/summarize_formal_results.py first, or pass --metadata explicitly."
        )
    spec = VARIANTS[variant]
    with report.open(encoding="utf-8", newline="") as handle:
        matches = [
            row for row in csv.DictReader(handle)
            if row[spec["report_column"]] == spec["report_value"]
            and int(row["seed"]) == seed
        ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {variant} seed {seed} row in {report}, found {len(matches)}"
        )
    return project_path(matches[0]["metadata"])


def load_run(args: argparse.Namespace) -> tuple[dict, Path, Path]:
    spec = VARIANTS[args.variant]
    report = project_path(args.per_seed_report or spec["report"])
    metadata_path = (
        project_path(args.metadata)
        if args.metadata is not None
        else metadata_from_report(report, args.variant, args.seed)
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_id") != spec["run_id"]:
        raise ValueError(
            f"Metadata run_id {metadata.get('run_id')} != {spec['run_id']}"
        )
    if metadata.get("sid_quantizer") != spec["sid_quantizer"]:
        raise ValueError(
            f"Metadata quantizer {metadata.get('sid_quantizer')} "
            f"!= {spec['sid_quantizer']}"
        )
    if int(metadata.get("seed", -1)) != args.seed:
        raise ValueError(f"Metadata seed {metadata.get('seed')} != {args.seed}")

    if args.checkpoint is not None:
        checkpoint = project_path(args.checkpoint)
    else:
        recorded = Path(metadata["checkpoint"])
        checkpoint = recorded if recorded.is_absolute() else DIFF_ROOT / recorded
        checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint}\n"
            "The checkpoint remains on the GPU server and is normally under "
            "external/DiffGRM/saved/<run>/pytorch_model.bin."
        )
    return metadata, metadata_path, checkpoint


def pipeline_overrides(variant: str, seed: int, batch_size: int) -> dict:
    spec = VARIANTS[variant]
    values = {
        "rand_seed": seed,
        "sid_fit_seed": seed,
        "eval_batch_size": batch_size,
    }
    if spec["sid_quantizer"] == "rqvae":
        values.update({
            "rqvae_seed": seed,
            "precomputed_semantic_ids_path": str(
                PROJECT_ROOT / "outputs" / "tokenizers" / "rqvae" /
                "Beauty" / f"seed{seed}" / "semantic_ids.json"
            ),
        })
    elif spec["sid_quantizer"] == "rq_kmeans":
        values.update({
            "rq_kmeans_seed": seed,
            "tokenizer_artifact_dir": str(
                PROJECT_ROOT / "outputs" / "tokenizers" / spec["artifact_dir"] /
                "Beauty" / f"seed{seed}"
            ),
            "force_regenerate_pca": False,
            "force_regenerate_codes": False,
            "force_regenerate_opq": False,
        })
    else:  # PSE / OPQ-PQ
        values.update({
            "force_regenerate_codes": False,
            "force_regenerate_opq": False,
        })
    return values


def safe_std(total: float, square_total: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    mean = total / count
    return math.sqrt(max(square_total / count - mean * mean, 0.0))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_samples < 0:
        raise ValueError("batch-size must be positive and max-samples must be non-negative")

    metadata, metadata_path, checkpoint = load_run(args)
    common_config = PROJECT_ROOT / "configs" / "ar_formal_common.yaml"
    variant_spec = VARIANTS[args.variant]
    quantizer_config = variant_spec["config"]

    # Config paths in the formal YAML files were designed to be resolved from
    # external/DiffGRM, matching the original training launcher.
    os.chdir(DIFF_ROOT)
    pipeline = Pipeline(
        model_name="AR_GRM",
        dataset_name="AmazonReviews2014",
        checkpoint_path=str(checkpoint),
        trainer=object(),  # Avoid constructing a trainer or touching training logs.
        config_dict=pipeline_overrides(args.variant, args.seed, args.batch_size),
        config_file=[str(common_config), str(quantizer_config)],
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

    n_layers = len(model.decoder_blocks)
    n_heads = model.decoder_blocks[0].self_attn.n_head
    n_targets = int(pipeline.config["n_digit"])
    sequence_length = n_targets + 1  # [BOS] + target SID tokens
    sums = torch.zeros(n_layers, n_heads, n_targets, sequence_length, dtype=torch.float64)
    square_sums = torch.zeros_like(sums)
    counts = torch.zeros_like(sums, dtype=torch.long)
    max_row_sum_error = 0.0
    max_future_attention = 0.0
    processed = 0
    observed_shape = None

    progress = tqdm(dataloader, desc=f"Attention {args.variant} seed{args.seed}")
    with torch.inference_mode():
        for batch in progress:
            labels = batch["labels"].to(device)
            loss_batch = {
                "history_sid": batch["history_sid"].to(device),
                "decoder_input_ids": labels.clamp_min(0),
                "decoder_labels": labels,
            }
            model(loss_batch, return_loss=True)
            valid_queries = labels.ne(-100).cpu()
            batch_size = labels.size(0)

            for layer_index, block in enumerate(model.decoder_blocks):
                attention = block.self_attn.last_attention
                if attention is None:
                    raise RuntimeError(f"Layer {layer_index + 1} did not capture attention")
                if observed_shape is None:
                    observed_shape = list(attention.shape)
                expected = (batch_size, n_heads, sequence_length, sequence_length)
                if tuple(attention.shape) != expected:
                    raise RuntimeError(
                        f"Unexpected attention shape {tuple(attention.shape)}, expected {expected}"
                    )

                attention = attention.to(dtype=torch.float64, device="cpu")
                for query in range(n_targets):
                    valid = valid_queries[:, query]
                    if not bool(valid.any()):
                        continue
                    selected = attention[valid, :, query, :query + 1]
                    sums[layer_index, :, query, :query + 1] += selected.sum(dim=0)
                    square_sums[layer_index, :, query, :query + 1] += selected.square().sum(dim=0)
                    counts[layer_index, :, query, :query + 1] += selected.size(0)

                    row_error = (selected.sum(dim=-1) - 1.0).abs().max().item()
                    max_row_sum_error = max(max_row_sum_error, row_error)
                    if query + 1 < sequence_length:
                        future = attention[valid, :, query, query + 1:]
                        if future.numel():
                            max_future_attention = max(
                                max_future_attention,
                                future.abs().max().item(),
                            )

                # Release the retained GPU view before processing the next batch.
                block.self_attn.last_attention = None

            processed += batch_size
            progress.set_postfix(examples=processed)

    if max_row_sum_error > 1e-5 or max_future_attention > 1e-7:
        raise RuntimeError(
            "Attention validation failed: "
            f"row_sum_error={max_row_sum_error:.3e}, "
            f"future_attention={max_future_attention:.3e}"
        )

    query_names = [f"SID{index}" for index in range(n_targets)]
    if pipeline.config.get("use_collision_suffix", False):
        query_names[-1] = "suffix"
    key_names = ["BOS"] + [f"SID{index}" for index in range(n_targets)]

    head_rows: list[dict] = []
    layer_rows: list[dict] = []
    for layer in range(n_layers):
        for query in range(n_targets):
            for key in range(query + 1):
                for head in range(n_heads):
                    count = int(counts[layer, head, query, key].item())
                    total = float(sums[layer, head, query, key].item())
                    square_total = float(square_sums[layer, head, query, key].item())
                    head_rows.append({
                        "variant": args.variant,
                        "quantizer": variant_spec["sid_quantizer"],
                        "seed": args.seed,
                        "split": args.split,
                        "decoder_layer": layer + 1,
                        "head": head + 1,
                        "query_position": query,
                        "query_target": query_names[query],
                        "key_position": key,
                        "key_token": key_names[key],
                        "mean_attention": total / count,
                        "std_attention": safe_std(total, square_total, count),
                        "n_examples": count,
                    })

                total = float(sums[layer, :, query, key].sum().item())
                square_total = float(square_sums[layer, :, query, key].sum().item())
                count = int(counts[layer, :, query, key].sum().item())
                layer_rows.append({
                    "variant": args.variant,
                    "quantizer": variant_spec["sid_quantizer"],
                    "seed": args.seed,
                    "split": args.split,
                    "decoder_layer": layer + 1,
                    "query_position": query,
                    "query_target": query_names[query],
                    "key_position": key,
                    "key_token": key_names[key],
                    "mean_attention": total / count,
                    "std_attention_over_examples_and_heads": safe_std(
                        total, square_total, count
                    ),
                    "n_head_examples": count,
                })

    output_dir = project_path(args.output_root) / args.variant / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "attention_by_head.csv", list(head_rows[0]), head_rows)
    write_csv(output_dir / "attention_by_layer.csv", list(layer_rows[0]), layer_rows)

    run_info = {
        "variant": args.variant,
        "quantizer": variant_spec["sid_quantizer"],
        "run_id": metadata.get("run_id"),
        "seed": args.seed,
        "split": args.split,
        "teacher_forcing": True,
        "attention_stage": "masked_softmax_before_dropout",
        "checkpoint": str(checkpoint),
        "checkpoint_best_epoch": metadata.get("best_epoch"),
        "metadata": str(metadata_path),
        "examples": processed,
        "decoder_layers": n_layers,
        "attention_heads": n_heads,
        "total_sid_positions": n_targets,
        "semantic_sid_positions": int(
            pipeline.config.get("semantic_n_digit", n_targets)
        ),
        "semantic_codebook_sizes": list(pipeline.tokenizer.semantic_codebook_sizes),
        "codebook_sizes": list(pipeline.tokenizer.codebook_sizes),
        "observed_attention_shape": observed_shape,
        "max_row_sum_error": max_row_sum_error,
        "max_future_attention": max_future_attention,
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved attention analysis to {output_dir}")
    print(
        f"examples={processed}, shape={observed_shape}, "
        f"row_sum_error={max_row_sum_error:.3e}, "
        f"future_attention={max_future_attention:.3e}"
    )


if __name__ == "__main__":
    main()
