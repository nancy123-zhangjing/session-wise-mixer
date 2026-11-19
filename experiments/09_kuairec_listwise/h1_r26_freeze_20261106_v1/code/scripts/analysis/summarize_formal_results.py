"""Aggregate completed formal AR runs as per-seed rows and mean ± std."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "hit_rate@5",
    "hit_rate@10",
    "ndcg@5",
    "ndcg@10",
    "legal@10",
    "duplicate@10",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, default=Path("outputs/run_metadata"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/formal_3x256"))
    parser.add_argument(
        "--min-best-epoch",
        type=int,
        default=5,
        help="Exclude short profile runs; formal evaluation starts at epoch 5.",
    )
    args = parser.parse_args()

    rows = []
    excluded_profiles = []
    # Old profile runs used the same run_id as formal runs. Sort by mtime so a
    # later formal rerun deterministically supersedes an older run for the same
    # quantizer/seed instead of depending on filesystem traversal order.
    metadata_paths = sorted(
        args.metadata_root.rglob("*_run_metadata.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    for path in metadata_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("run_id") not in {"ar_formal_rqvae_3x256", "ar_formal_rqkmeans_3x256"}:
            continue
        best_epoch = data.get("best_epoch")
        if best_epoch is None or int(best_epoch) < args.min_best_epoch:
            excluded_profiles.append(path)
            continue
        row = {
            "quantizer": data["sid_quantizer"],
            "seed": int(data["seed"]),
            "best_epoch": int(best_epoch),
            "fit_hours": (data.get("fit_elapsed_seconds") or 0.0) / 3600,
            "peak_vram_mb": data.get("peak_vram_mb"),
            "metadata": str(path),
        }
        for metric in METRICS:
            row[metric] = data.get("test_results", {}).get(metric)
        rows.append(row)
    if not rows:
        raise SystemExit(f"No completed formal runs under {args.metadata_root}")

    for path in excluded_profiles:
        print(f"Excluded short profile run: {path}", file=sys.stderr)

    # A rerun with the same quantizer/seed supersedes the older timestamped run.
    deduplicated = {}
    for row in rows:
        deduplicated[(row["quantizer"], row["seed"])] = row
    rows = sorted(deduplicated.values(), key=lambda row: (row["quantizer"], row["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["quantizer"]].append(row)
    summary = {}
    for quantizer, group in grouped.items():
        summary[quantizer] = {"n_seeds": len(group), "seeds": [row["seed"] for row in group]}
        for metric in METRICS:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[quantizer][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
