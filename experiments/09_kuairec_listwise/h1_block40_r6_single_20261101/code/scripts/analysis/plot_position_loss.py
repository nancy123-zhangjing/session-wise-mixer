"""Plot SID-position loss profiles and per-position learning curves from CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {
                "epoch": int(row["epoch"]),
                "split": row["split"],
                "position": int(row["position"]),
                "role": row.get("role", "semantic"),
                "valid_tokens": int(row["valid_tokens"]),
            }
            for metric in ("loss", "nll", "normalized_nll", "accuracy", "prefix_accuracy"):
                if row.get(metric) not in (None, ""):
                    parsed[metric] = float(row[metric])
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def plot_learning_curves(
    rows: list[dict[str, object]], output: Path, metric: str
) -> None:
    grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if int(row["epoch"]) > 0:
            grouped[(str(row["split"]), int(row["position"]))].append(
                (int(row["epoch"]), float(row[metric]))
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    for (split, position), values in sorted(grouped.items()):
        values.sort()
        linestyle = "-" if split == "train" else "--"
        ax.plot(
            [epoch for epoch, _ in values],
            [loss for _, loss in values],
            marker="o",
            linestyle=linestyle,
            label=f"{split} / SID-{position}",
        )
    ax.set(xlabel="Epoch", ylabel=metric.replace("_", " "), title=f"{metric} by SID position")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_final_profile(rows: list[dict[str, object]], output: Path, metric: str) -> None:
    latest_epoch: dict[str, int] = {}
    for row in rows:
        epoch = int(row["epoch"])
        if epoch > 0:
            split = str(row["split"])
            latest_epoch[split] = max(epoch, latest_epoch.get(split, epoch))

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        split = str(row["split"])
        if int(row["epoch"]) == latest_epoch.get(split):
            grouped[split].append((int(row["position"]), float(row[metric])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split, values in sorted(grouped.items()):
        values.sort()
        ax.plot(
            [position for position, _ in values],
            [loss for _, loss in values],
            marker="o",
            label=f"{split} (epoch {latest_epoch[split]})",
        )
    ax.set(
        xlabel="SID position (generation order)",
        ylabel=metric.replace("_", " "),
        title=f"Final SID-position {metric} profile",
    )
    ax.set_xticks(sorted({int(row["position"]) for row in rows}))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/figures"), type=Path)
    parser.add_argument(
        "--metric",
        default="loss",
        choices=["loss", "nll", "normalized_nll", "accuracy", "prefix_accuracy"],
    )
    args = parser.parse_args()

    rows = read_rows(args.input)
    if args.metric not in rows[0]:
        raise ValueError(
            f"Metric {args.metric!r} is not present in {args.input}; "
            "older smoke CSVs only contain 'loss'."
        )
    plot_learning_curves(rows, args.output_dir / f"position_{args.metric}_by_epoch.png", args.metric)
    plot_final_profile(rows, args.output_dir / f"position_{args.metric}_profile.png", args.metric)
    print(f"Wrote figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
