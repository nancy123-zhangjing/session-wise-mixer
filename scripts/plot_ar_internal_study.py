"""Create the compact figures needed for the AR internal-control report."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "rqk_256x256x256": "RQ-KMeans 256x256x256",
    "rqvae_256x256x256": "RQ-VAE 256x256x256",
    "pse_256x256x256": "PSE 256x256x256",
    "rqk_128x256x512": "RQ-KMeans 128x256x512",
    "rqk_512x256x128": "RQ-KMeans 512x256x128",
}
ORDER = list(DISPLAY)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/reports/ar_internal_study"),
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    output_dir = (args.output_dir or report_dir / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendation = [
        row for row in read_csv(report_dir / "per_seed.csv")
        if int(row["seed"]) == args.seed
    ]
    positions = [
        row for row in read_csv(report_dir / "test_position_metrics.csv")
        if int(row["seed"]) == args.seed and row["role"] == "semantic"
    ]
    if not recommendation or not positions:
        raise SystemExit(f"No seed {args.seed} rows under {report_dir}")

    available = [variant for variant in ORDER if any(
        row["variant"] == variant for row in recommendation
    )]
    colors = plt.cm.tab10(np.arange(len(available)))

    metrics = ("hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10")
    x = np.arange(len(metrics))
    width = 0.8 / len(available)
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, variant in enumerate(available):
        row = next(item for item in recommendation if item["variant"] == variant)
        values = [float(row[metric]) for metric in metrics]
        ax.bar(
            x + (index - (len(available) - 1) / 2) * width,
            values,
            width,
            label=DISPLAY[variant],
            color=colors[index],
        )
    ax.set_xticks(x, ["HR@5", "HR@10", "NDCG@5", "NDCG@10"])
    ax.set_ylabel("Strict item-level score")
    ax.set_title(f"AR recommendation metrics (seed {args.seed})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "recommendation_metrics.png", dpi=200)
    plt.close(fig)

    grouped = defaultdict(dict)
    for row in positions:
        grouped[row["variant"]][int(row["position"])] = float(row["normalized_nll"])
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, variant in enumerate(available):
        if variant not in grouped:
            continue
        ax.plot(
            x,
            [grouped[variant][position] for position in (1, 2, 3)],
            marker="o",
            linewidth=2,
            label=DISPLAY[variant],
            color=colors[index],
        )
    ax.set_xticks(x, ["SID0", "SID1", "SID2"])
    ax.set_ylabel("NLL / log(codebook size)")
    ax.set_title(f"Test normalized NLL by semantic position (seed {args.seed})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "test_normalized_nll_by_position.png", dpi=200)
    plt.close(fig)

    # One uncluttered learning-curve figure per run.  Semantic positions only;
    # the fourth collision suffix is intentionally excluded from this study.
    for row in recommendation:
        path = Path(row["position_metrics_csv"])
        if not path.is_file():
            continue
        all_rows = read_csv(path)
        series = defaultdict(list)
        for item in all_rows:
            if item["split"] != "val" or item["role"] != "semantic":
                continue
            epoch = int(item["epoch"])
            if epoch <= 0:
                continue
            series[int(item["position"])].append(
                (epoch, float(item["normalized_nll"]))
            )
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for position in (1, 2, 3):
            values = sorted(series[position])
            ax.plot(
                [epoch for epoch, _ in values],
                [value for _, value in values],
                marker="o",
                label=f"SID{position - 1}",
            )
        variant = row["variant"]
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation NLL / log(codebook size)")
        ax.set_title(DISPLAY.get(variant, variant))
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"val_position_curve_{variant}.png", dpi=200)
        plt.close(fig)

    print(f"Wrote AR internal-study figures to {output_dir}")


if __name__ == "__main__":
    main()
