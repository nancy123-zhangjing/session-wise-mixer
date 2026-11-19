"""Aggregate formal AR self-attention runs and draw SID2 diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_QUANTIZERS = ("rq_kmeans", "rqvae")
EXPECTED_SEEDS = (2024, 2025, 2026)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "attention" / "formal_3x256",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "formal_3x256" / "attention",
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def discover_runs(root: Path, allow_partial: bool) -> list[tuple[str, int, Path]]:
    runs = []
    missing = []
    for quantizer in EXPECTED_QUANTIZERS:
        for seed in EXPECTED_SEEDS:
            path = root / quantizer / f"seed{seed}"
            required = (
                path / "attention_by_layer.csv",
                path / "attention_by_head.csv",
                path / "run_info.json",
            )
            if all(item.exists() for item in required):
                runs.append((quantizer, seed, path))
            else:
                missing.append(f"{quantizer}/seed{seed}")
    if missing and not allow_partial:
        raise FileNotFoundError(
            "Missing formal attention runs: " + ", ".join(missing) +
            ". Run bash scripts/analysis/run_formal_attention.sh all python first."
        )
    if not runs:
        raise FileNotFoundError(f"No attention runs found under {root}")
    return runs


def summarize_layer_rows(per_seed_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in per_seed_rows:
        key = (
            row["quantizer"],
            int(row["decoder_layer"]),
            int(row["query_position"]),
            row["query_target"],
            int(row["key_position"]),
            row["key_token"],
        )
        grouped[key].append(float(row["mean_attention"]))

    output = []
    for key, values in sorted(grouped.items()):
        quantizer, layer, query_position, query_target, key_position, key_token = key
        output.append({
            "quantizer": quantizer,
            "decoder_layer": layer,
            "query_position": query_position,
            "query_target": query_target,
            "key_position": key_position,
            "key_token": key_token,
            "n_seeds": len(values),
            "mean_attention": statistics.mean(values),
            "seed_std_attention": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return output


def summarize_head_rows(per_head_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in per_head_rows:
        key = (
            row["quantizer"],
            int(row["decoder_layer"]),
            int(row["head"]),
            int(row["query_position"]),
            row["query_target"],
            int(row["key_position"]),
            row["key_token"],
        )
        grouped[key].append(float(row["mean_attention"]))

    output = []
    for key, values in sorted(grouped.items()):
        quantizer, layer, head, query_position, query_target, key_position, key_token = key
        output.append({
            "quantizer": quantizer,
            "decoder_layer": layer,
            "head": head,
            "query_position": query_position,
            "query_target": query_target,
            "key_position": key_position,
            "key_token": key_token,
            "n_seeds": len(values),
            "mean_attention": statistics.mean(values),
            "seed_std_attention": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return output


def plot_sid2_layer_heatmap(summary: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ("BOS", "SID0", "SID1")
    display_names = {"rq_kmeans": "RQ-KMeans", "rqvae": "RQ-VAE"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    image = None
    for axis, quantizer in zip(axes, EXPECTED_QUANTIZERS):
        matrix = np.full((4, 3), np.nan, dtype=float)
        deviations = np.full_like(matrix, np.nan)
        for row in summary:
            if row["quantizer"] != quantizer or row["query_target"] != "SID2":
                continue
            layer = int(row["decoder_layer"]) - 1
            key = keys.index(row["key_token"])
            matrix[layer, key] = float(row["mean_attention"])
            deviations[layer, key] = float(row["seed_std_attention"])
        if np.isnan(matrix).any():
            raise RuntimeError(f"Incomplete SID2 layer matrix for {quantizer}: {matrix}")
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        axis.set_title(display_names[quantizer])
        axis.set_xticks(range(3), keys)
        axis.set_yticks(range(4), [f"Layer {index}" for index in range(1, 5)])
        axis.set_xlabel("Key token")
        for layer in range(4):
            for key in range(3):
                value = matrix[layer, key]
                color = "white" if value >= 0.55 else "black"
                axis.text(
                    key,
                    layer,
                    f"{value:.3f}\n±{deviations[layer, key]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=color,
                )
    fig.suptitle(
        "Generating SID2: decoder self-attention by layer\n"
        "mean ± seed std; averaged over test examples and heads",
        fontsize=13,
    )
    fig.colorbar(image, ax=axes, label="Attention weight", shrink=0.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sid2_head_heatmap(summary: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ("BOS", "SID0", "SID1")
    display_names = {"rq_kmeans": "RQ-KMeans", "rqvae": "RQ-VAE"}
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    image = None
    for row_index, quantizer in enumerate(EXPECTED_QUANTIZERS):
        for layer in range(1, 5):
            axis = axes[row_index, layer - 1]
            matrix = np.full((8, 3), np.nan, dtype=float)
            for row in summary:
                if (
                    row["quantizer"] == quantizer
                    and row["query_target"] == "SID2"
                    and int(row["decoder_layer"]) == layer
                ):
                    matrix[int(row["head"]) - 1, keys.index(row["key_token"])] = float(
                        row["mean_attention"]
                    )
            if np.isnan(matrix).any():
                raise RuntimeError(
                    f"Incomplete SID2 head matrix for {quantizer} layer {layer}"
                )
            image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
            axis.set_title(f"{display_names[quantizer]} · Layer {layer}", fontsize=10)
            axis.set_xticks(range(3), keys)
            axis.set_yticks(range(8), [f"H{head}" for head in range(1, 9)])
            if layer == 1:
                axis.set_ylabel("Attention head")
    fig.suptitle(
        "Generating SID2: decoder self-attention by layer and head\n"
        "averaged over test examples and three seeds",
        fontsize=13,
    )
    fig.colorbar(image, ax=axes, label="Attention weight", shrink=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    runs = discover_runs(input_root, args.allow_partial)

    per_seed_layer = []
    per_seed_head = []
    run_info = []
    for quantizer, seed, path in runs:
        layer_rows = read_csv(path / "attention_by_layer.csv")
        head_rows = read_csv(path / "attention_by_head.csv")
        info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
        if quantizer != info["quantizer"] or seed != int(info["seed"]):
            raise ValueError(f"Run identity mismatch in {path}")
        per_seed_layer.extend(layer_rows)
        per_seed_head.extend(head_rows)
        run_info.append(info)

    layer_summary = summarize_layer_rows(per_seed_layer)
    head_summary = summarize_head_rows(per_seed_head)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "attention_per_seed_by_layer.csv", per_seed_layer)
    write_csv(output_dir / "attention_summary_by_layer.csv", layer_summary)
    write_csv(output_dir / "attention_summary_by_head.csv", head_summary)
    (output_dir / "attention_runs.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if all(
        sum(info["quantizer"] == quantizer for info in run_info) == 3
        for quantizer in EXPECTED_QUANTIZERS
    ):
        plot_sid2_layer_heatmap(
            layer_summary,
            output_dir / "sid2_self_attention_by_layer.png",
        )
        plot_sid2_head_heatmap(
            head_summary,
            output_dir / "sid2_self_attention_by_head.png",
        )

    print("\nGenerating SID2: mean attention ± seed std")
    for quantizer in EXPECTED_QUANTIZERS:
        print(f"\n{quantizer}")
        for layer in range(1, 5):
            selected = [
                row for row in layer_summary
                if row["quantizer"] == quantizer
                and row["query_target"] == "SID2"
                and int(row["decoder_layer"]) == layer
            ]
            rendered = ", ".join(
                f"{row['key_token']}={float(row['mean_attention']):.4f}"
                f"±{float(row['seed_std_attention']):.4f}"
                for row in selected
            )
            print(f"Layer {layer}: {rendered}")
    print(f"\nSaved attention summary to {output_dir}")


if __name__ == "__main__":
    main()
