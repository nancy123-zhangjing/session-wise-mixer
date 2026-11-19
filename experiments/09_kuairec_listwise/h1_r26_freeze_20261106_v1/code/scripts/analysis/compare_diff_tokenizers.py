"""Create paired DiffGRM tokenizer comparisons by decoder seed."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PSE_RUN_ID = "diff_formal_pse_4x256_repo_repro_v6_bf16"
RQK_RUN_ID = "diff_formal_rqkmeans_4x256_v6_bf16"
RQVAE_RUN_ID = "diff_formal_rqvae_4x256_v6_bf16"
CROSSPSE_V1_RUN_ID = "diff_formal_crosspse_4x256_v1_bf16"
CROSSPSE_V2_RUN_ID = "diff_formal_crosspse_4x256_v2_bf16"
REFINEDPSE_V1_RUN_ID = "diff_formal_refinedpse_4x256_v1_bf16"
CROSSPSE_V3_RUN_ID = "diff_formal_crosspse_4x256_v3_bf16"
REFINEDPSE_V2_CATALOG_RUN_ID = "diff_formal_refinedpse_4x256_v2_catalog_bf16"
CROSSPSE_V3_CATALOG_RUN_ID = "diff_formal_crosspse_4x256_v3_catalog_bf16"
METRICS = (
    "hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10",
    "legal@10", "duplicate@10",
)


def read_by_seed(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pse-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / PSE_RUN_ID,
    )
    parser.add_argument(
        "--rqk-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / RQK_RUN_ID,
    )
    parser.add_argument(
        "--rqvae-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / RQVAE_RUN_ID,
    )
    parser.add_argument("--include-rqvae", action="store_true")
    parser.add_argument(
        "--crosspse-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / CROSSPSE_V2_RUN_ID,
    )
    parser.add_argument(
        "--crosspse-name",
        choices=("cross_pse_v1", "cross_pse_v2"),
        default="cross_pse_v2",
    )
    parser.add_argument("--include-cross-pse", action="store_true")
    parser.add_argument(
        "--refinedpse-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / REFINEDPSE_V1_RUN_ID,
    )
    parser.add_argument("--include-refined-pse", action="store_true")
    parser.add_argument(
        "--refinedpse-name",
        choices=("refined_pse_v1", "refined_pse_v2_catalog"),
        default="refined_pse_v1",
    )
    parser.add_argument(
        "--crosspse-v3-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / CROSSPSE_V3_RUN_ID,
    )
    parser.add_argument("--include-cross-pse-v3", action="store_true")
    parser.add_argument(
        "--crosspse-v3-name",
        choices=("cross_pse_v3", "cross_pse_v3_catalog"),
        default="cross_pse_v3",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "diff_tokenizer_comparison",
    )
    args = parser.parse_args()

    sources = {
        "pse": read_by_seed(args.pse_dir / "per_seed.csv"),
        "rq_kmeans": read_by_seed(args.rqk_dir / "per_seed.csv"),
    }
    if args.include_rqvae:
        sources["rqvae"] = read_by_seed(args.rqvae_dir / "per_seed.csv")
    if args.include_cross_pse:
        sources[args.crosspse_name] = read_by_seed(
            args.crosspse_dir / "per_seed.csv"
        )
    if args.include_refined_pse:
        sources[args.refinedpse_name] = read_by_seed(
            args.refinedpse_dir / "per_seed.csv"
        )
    if args.include_cross_pse_v3:
        sources[args.crosspse_v3_name] = read_by_seed(
            args.crosspse_v3_dir / "per_seed.csv"
        )
    seeds = sorted(set.intersection(*(set(source) for source in sources.values())))
    if not seeds:
        raise SystemExit("No decoder seeds shared by all requested tokenizers")

    rows = []
    for seed in seeds:
        row: dict[str, object] = {"seed": seed}
        for metric in METRICS:
            pse_value = float(sources["pse"][seed][metric])
            for name, source in sources.items():
                value = float(source[seed][metric])
                row[f"{name}_{metric}"] = value
                if name != "pse":
                    row[f"delta_{name}_minus_pse_{metric}"] = value - pse_value
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "tokenizers": list(sources),
        "evaluation_level": "paper_aligned_sid",
        "seeds": seeds,
        "n_paired_seeds": len(seeds),
    }
    for metric in METRICS:
        summary[metric] = {}
        for name, source in sources.items():
            values = np.asarray([float(source[seed][metric]) for seed in seeds])
            summary[metric][name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        for name, source in sources.items():
            if name == "pse":
                continue
            deltas = np.asarray([
                float(source[seed][metric])
                - float(sources["pse"][seed][metric])
                for seed in seeds
            ])
            summary[metric][f"paired_delta_{name}_minus_pse"] = {
                "mean": float(deltas.mean()),
                "std": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
        main_metrics = METRICS[:4]
        x = np.arange(len(main_metrics))
        width = 0.8 / len(sources)
        fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
        display_names = {
            "pse": "PSE", "rq_kmeans": "RQ-KMeans", "rqvae": "RQ-VAE",
            "cross_pse_v1": "Cross-PSE v1", "cross_pse_v2": "Cross-PSE v2",
            "refined_pse_v1": "Refined-PSE", "cross_pse_v3": "Cross-PSE v3",
            "refined_pse_v2_catalog": "Refined-PSE catalog",
            "cross_pse_v3_catalog": "Cross-PSE v3 catalog",
        }
        center = (len(sources) - 1) / 2
        for index, (name, source) in enumerate(sources.items()):
            offset = (index - center) * width
            means = []
            stds = []
            for metric in main_metrics:
                values = np.asarray([float(source[seed][metric]) for seed in seeds])
                means.append(values.mean())
                stds.append(values.std(ddof=1) if len(values) > 1 else 0.0)
            ax.bar(
                x + offset, means, width, yerr=stds, capsize=3,
                label=display_names[name],
            )
        ax.set_xticks(x, [metric.upper() for metric in main_metrics])
        ax.set_ylabel("Paper-aligned SID-level metric")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.output_dir / "recommendation_metrics.png", bbox_inches="tight")
        plt.close(fig)
    except ImportError as error:
        print(f"[WARN] Skipped comparison plot: {error}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved paired tokenizer comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
