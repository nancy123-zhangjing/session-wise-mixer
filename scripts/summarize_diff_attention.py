"""Create report-ready DiffGRM attention comparisons from saved CSV files.

This script is deliberately post-processing only: it does not load a model or
touch the GPU.  It compares PSE, RQ-KMeans and RQ-VAE under the same decoder
seed and writes four figures that keep decode order, key state, decoder layer
and positional deviation separate instead of averaging all of them together.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    ("pse", "PSE", "diff_formal_pse_4x256_repo_repro_v6_bf16"),
    ("rq_kmeans", "RQ-KMeans", "diff_formal_rqkmeans_4x256_v6_bf16"),
    ("rqvae", "RQ-VAE", "diff_formal_rqvae_4x256_v6_bf16"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--attention-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "attention",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "diff_attention_comparison",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing attention result: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_attention(rows: list[dict], count_key: str) -> float:
    count = sum(int(row[count_key]) for row in rows)
    if not count:
        return float("nan")
    return sum(
        float(row["mean_attention"]) * int(row[count_key])
        for row in rows
    ) / count


def load_runs(args: argparse.Namespace) -> list[dict]:
    runs = []
    for key, label, run_id in DEFAULT_RUNS:
        root = args.attention_root / run_id / f"seed{args.seed}"
        info_path = root / "run_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing attention result: {info_path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if int(info.get("seed", -1)) != args.seed or info.get("split") != "test":
            raise ValueError(f"Unexpected run_info contents: {info_path}")
        if int(info.get("examples", 0)) <= 128:
            raise ValueError(
                f"{label} has only {info.get('examples')} examples; "
                "run the full attention analysis instead of attention-smoke"
            )
        if float(info.get("max_row_sum_error", 1.0)) > 1e-5:
            raise ValueError(f"Attention validation failed: {info_path}")
        runs.append({
            "key": key,
            "label": label,
            "run_id": run_id,
            "root": root,
            "info": info,
            "position": read_csv(root / "selected_position_frequency.csv"),
            "overall": read_csv(root / "attention_overall.csv"),
            "layer": read_csv(root / "attention_by_layer.csv"),
        })

    example_counts = {int(run["info"]["examples"]) for run in runs}
    architectures = {
        (
            int(run["info"]["decoder_layers"]),
            int(run["info"]["attention_heads"]),
            int(run["info"]["sid_positions"]),
        )
        for run in runs
    }
    if len(example_counts) != 1:
        raise ValueError(f"Attention example counts differ across runs: {example_counts}")
    if len(architectures) != 1:
        raise ValueError(f"Decoder architectures differ across runs: {architectures}")
    return runs


def decode_order_rows(runs: list[dict], n_digit: int) -> list[dict]:
    output = []
    for run in runs:
        lookup = {
            (int(row["decode_rank"]), int(row["position"])): float(row["frequency"])
            for row in run["position"]
        }
        for rank in range(1, n_digit + 1):
            row_sum = sum(lookup.get((rank, position), 0.0) for position in range(n_digit))
            if abs(row_sum - 1.0) > 1e-5:
                raise ValueError(
                    f"{run['label']} decode-rank {rank} frequencies sum to {row_sum}"
                )
            for position in range(n_digit):
                output.append({
                    "tokenizer": run["key"],
                    "tokenizer_label": run["label"],
                    "decode_rank": rank,
                    "position": position,
                    "frequency": lookup[(rank, position)],
                })
    return output


def state_mass_rows(runs: list[dict], n_digit: int) -> list[dict]:
    output = []
    for run in runs:
        grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
        rank_rows: dict[int, list[dict]] = defaultdict(list)
        for row in run["overall"]:
            rank = int(row["decode_rank"])
            grouped[(rank, row["key_state"])].append(row)
            rank_rows[rank].append(row)

        for rank in range(1, n_digit + 1):
            rows = rank_rows[rank]
            # Each sample/layer/head contributes one value for every key.  The
            # total count therefore equals base_count * n_digit.
            base_count = sum(int(row["n_layer_head_examples"]) for row in rows) / n_digit
            if base_count <= 0 or not math.isclose(base_count, round(base_count), abs_tol=1e-6):
                raise ValueError(f"Invalid attention counts for {run['label']} rank {rank}")
            masses = {}
            for state in ("revealed", "masked"):
                state_rows = grouped[(rank, state)]
                masses[state] = sum(
                    float(row["mean_attention"]) * int(row["n_layer_head_examples"])
                    for row in state_rows
                ) / base_count
            if abs(masses["revealed"] + masses["masked"] - 1.0) > 1e-5:
                raise ValueError(
                    f"{run['label']} rank {rank} state masses do not sum to one"
                )
            uniform_revealed = (rank - 1) / n_digit
            for state in ("revealed", "masked"):
                uniform_mass = (
                    uniform_revealed if state == "revealed" else 1.0 - uniform_revealed
                )
                output.append({
                    "tokenizer": run["key"],
                    "tokenizer_label": run["label"],
                    "decode_rank": rank,
                    "key_state": state,
                    "attention_mass": masses[state],
                    "uniform_mass": uniform_mass,
                    "deviation_from_uniform": masses[state] - uniform_mass,
                })
    return output


def layer_state_rows(runs: list[dict], n_digit: int, n_layers: int) -> list[dict]:
    output = []
    for run in runs:
        grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in run["layer"]:
            grouped[(int(row["decoder_layer"]), int(row["decode_rank"]))].append(row)

        for layer in range(1, n_layers + 1):
            for rank in range(1, n_digit + 1):
                rows = grouped[(layer, rank)]
                base_count = sum(int(row["n_head_examples"]) for row in rows) / n_digit
                revealed_rows = [row for row in rows if row["key_state"] == "revealed"]
                revealed_mass = sum(
                    float(row["mean_attention"]) * int(row["n_head_examples"])
                    for row in revealed_rows
                ) / base_count
                baseline = (rank - 1) / n_digit
                output.append({
                    "tokenizer": run["key"],
                    "tokenizer_label": run["label"],
                    "decoder_layer": layer,
                    "decode_rank": rank,
                    "revealed_attention_mass": revealed_mass,
                    "uniform_revealed_mass": baseline,
                    "deviation_from_uniform": revealed_mass - baseline,
                })
    return output


def position_deviation_rows(runs: list[dict], n_digit: int) -> list[dict]:
    output = []
    for run in runs:
        grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
        for row in run["overall"]:
            grouped[
                (
                    int(row["decode_rank"]),
                    int(row["query_position"]),
                    int(row["key_position"]),
                )
            ].append(row)
        for (rank, query, key), rows in sorted(grouped.items()):
            value = weighted_attention(rows, "n_layer_head_examples")
            output.append({
                "tokenizer": run["key"],
                "tokenizer_label": run["label"],
                "decode_rank": rank,
                "query_position": query,
                "key_position": key,
                "mean_attention": value,
                "uniform_attention": 1.0 / n_digit,
                "deviation_from_uniform": value - 1.0 / n_digit,
                "n_layer_head_examples": sum(
                    int(row["n_layer_head_examples"]) for row in rows
                ),
            })
    return output


def setup_plotting() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 220,
    })


def annotate_matrix(ax, matrix: np.ndarray, fmt: str, threshold: float | None = None) -> None:
    if threshold is None:
        threshold = float(np.nanmax(np.abs(matrix))) * 0.55
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isnan(value):
                continue
            ax.text(
                col,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if abs(value) > threshold else "black",
                fontsize=8,
            )


def plot_decode_order(rows: list[dict], output: Path, n_digit: int) -> None:
    import matplotlib.pyplot as plt

    labels = [item[1] for item in DEFAULT_RUNS]
    matrices = []
    for label in labels:
        matrix = np.zeros((n_digit, n_digit))
        for row in rows:
            if row["tokenizer_label"] == label:
                matrix[int(row["decode_rank"]) - 1, int(row["position"])] = float(row["frequency"])
        matrices.append(matrix)
    vmax = max(float(np.max(matrix)) for matrix in matrices)
    vmax = max(vmax, 1.0 / n_digit)
    fig, axes = plt.subplots(1, len(labels), figsize=(11.2, 3.5), constrained_layout=True)
    image = None
    for index, (ax, label, matrix) in enumerate(zip(axes, labels, matrices)):
        image = ax.imshow(matrix, vmin=0.0, vmax=vmax, cmap="Blues", aspect="equal")
        annotate_matrix(ax, matrix, ".3f", threshold=vmax * 0.55)
        ax.set_title(label)
        ax.set_xlabel("Selected SID position")
        if index == 0:
            ax.set_ylabel("CPD decode rank")
        ax.set_xticks(range(n_digit), [f"SID{i}" for i in range(n_digit)])
        ax.set_yticks(range(n_digit), [f"Rank {i}" for i in range(1, n_digit + 1)])
    fig.suptitle("CPD position selection order")
    fig.colorbar(image, ax=axes, label="Selection frequency", shrink=0.84)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_state_mass(rows: list[dict], output: Path, n_digit: int) -> None:
    import matplotlib.pyplot as plt

    labels = [item[1] for item in DEFAULT_RUNS]
    fig, axes = plt.subplots(1, len(labels), figsize=(11.2, 3.8), sharey=True, constrained_layout=True)
    colors = {"revealed": "#3977b7", "masked": "#d4a72c"}
    for ax, label in zip(axes, labels):
        selected = [row for row in rows if row["tokenizer_label"] == label]
        revealed = np.asarray([
            next(float(row["attention_mass"]) for row in selected
                 if int(row["decode_rank"]) == rank and row["key_state"] == "revealed")
            for rank in range(1, n_digit + 1)
        ])
        masked = 1.0 - revealed
        x = np.arange(1, n_digit + 1)
        ax.bar(x, revealed, color=colors["revealed"], label="Revealed keys")
        ax.bar(x, masked, bottom=revealed, color=colors["masked"], label="Masked keys")
        ax.plot(
            x,
            np.arange(n_digit) / n_digit,
            color="black",
            marker="D",
            markersize=4,
            linewidth=1.2,
            label="Uniform revealed baseline",
        )
        for rank, value in zip(x, revealed):
            ax.text(rank, max(value / 2, 0.035), f"{value:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.12 else "black")
        ax.set_title(label)
        ax.set_xlabel("CPD decode rank")
        ax.set_xticks(x)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Self-attention mass")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Attention to revealed versus masked SID positions", y=1.14)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_layer_bias(rows: list[dict], output: Path, n_digit: int, n_layers: int) -> None:
    import matplotlib.pyplot as plt

    labels = [item[1] for item in DEFAULT_RUNS]
    fig, axes = plt.subplots(1, len(labels), figsize=(11.2, 3.7), sharex=True, sharey=True, constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    all_values = [abs(float(row["deviation_from_uniform"])) for row in rows]
    limit = max(max(all_values, default=0.0) * 1.15, 0.01)
    for ax, label in zip(axes, labels):
        selected = [row for row in rows if row["tokenizer_label"] == label]
        for layer in range(1, n_layers + 1):
            values = [
                next(float(row["deviation_from_uniform"]) for row in selected
                     if int(row["decoder_layer"]) == layer and int(row["decode_rank"]) == rank)
                for rank in range(1, n_digit + 1)
            ]
            ax.plot(
                range(1, n_digit + 1),
                values,
                marker="o",
                linewidth=1.8,
                color=colors[layer - 1],
                label=f"Layer {layer}",
            )
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_title(label)
        ax.set_xlabel("CPD decode rank")
        ax.set_xticks(range(1, n_digit + 1))
        ax.set_ylim(-limit, limit)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Revealed-attention mass\nminus uniform baseline")
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=n_layers,
        frameon=False,
    )
    fig.suptitle("Layer-wise preference for already revealed SID positions", y=1.14)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_position_deviation(rows: list[dict], output: Path, n_digit: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    labels = [item[1] for item in DEFAULT_RUNS]
    limit = max(
        max(abs(float(row["deviation_from_uniform"])) for row in rows) * 1.02,
        0.005,
    )
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(
        len(labels), n_digit, figsize=(13.0, 9.0), constrained_layout=True
    )
    image = None
    for tokenizer_index, label in enumerate(labels):
        selected = [row for row in rows if row["tokenizer_label"] == label]
        for rank in range(1, n_digit + 1):
            ax = axes[tokenizer_index, rank - 1]
            matrix = np.full((n_digit, n_digit), np.nan)
            for row in selected:
                if int(row["decode_rank"]) == rank:
                    matrix[int(row["query_position"]), int(row["key_position"])] = float(
                        row["deviation_from_uniform"]
                    )
            image = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="equal")
            annotate_matrix(ax, matrix, "+.3f", threshold=limit * 0.56)
            if tokenizer_index == 0:
                ax.set_title(f"Decode rank {rank}")
            if rank == 1:
                ax.set_ylabel(f"{label}\nSelected query")
            ax.set_xticks(range(n_digit), [f"SID{i}" for i in range(n_digit)])
            ax.set_yticks(range(n_digit), [f"SID{i}" for i in range(n_digit)])
            if tokenizer_index == len(labels) - 1:
                ax.set_xlabel("Key position")
    fig.suptitle("Position-wise self-attention deviation from the uniform 0.25 baseline")
    fig.colorbar(
        image,
        ax=axes,
        label="Mean attention minus 0.25",
        shrink=0.80,
        format="%+.3f",
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = load_runs(args)
    n_layers = int(runs[0]["info"]["decoder_layers"])
    n_digit = int(runs[0]["info"]["sid_positions"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    order = decode_order_rows(runs, n_digit)
    state = state_mass_rows(runs, n_digit)
    layer = layer_state_rows(runs, n_digit, n_layers)
    position = position_deviation_rows(runs, n_digit)

    write_csv(args.output_dir / "decode_order_comparison.csv", order)
    write_csv(args.output_dir / "attention_state_summary.csv", state)
    write_csv(args.output_dir / "attention_state_by_layer.csv", layer)
    write_csv(args.output_dir / "position_attention_deviation.csv", position)

    setup_plotting()
    plot_decode_order(order, args.output_dir / "01_cpd_decode_order.png", n_digit)
    plot_state_mass(state, args.output_dir / "02_masked_revealed_attention.png", n_digit)
    plot_layer_bias(
        layer,
        args.output_dir / "03_layer_revealed_attention_bias.png",
        n_digit,
        n_layers,
    )
    plot_position_deviation(
        position,
        args.output_dir / "04_position_attention_deviation.png",
        n_digit,
    )

    manifest = {
        "seed": args.seed,
        "split": "test",
        "examples_per_tokenizer": int(runs[0]["info"]["examples"]),
        "decoder_layers": n_layers,
        "attention_heads": int(runs[0]["info"]["attention_heads"]),
        "sid_positions": n_digit,
        "uniform_attention_per_key": 1.0 / n_digit,
        "attention_stage": runs[0]["info"]["attention_stage"],
        "diagnostic_path": runs[0]["info"]["diagnostic_path"],
        "runs": [
            {
                "tokenizer": run["key"],
                "run_id": run["run_id"],
                "checkpoint": run["info"]["checkpoint"],
                "max_row_sum_error": run["info"]["max_row_sum_error"],
            }
            for run in runs
        ],
        "interpretation_note": (
            "Attention is a beam-width-one greedy CPD diagnostic. "
            "Official beam-256 recommendation metrics are unchanged."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved DiffGRM attention comparison to {args.output_dir}")
    for path in sorted(args.output_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
