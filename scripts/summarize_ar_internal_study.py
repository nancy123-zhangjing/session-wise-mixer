"""Summarize the AR tokenizer/position-capacity internal controls."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_VARIANTS = {
    "ar_formal_rqkmeans_3x256": "rqk_256x256x256",
    "ar_formal_rqvae_3x256": "rqvae_256x256x256",
    "ar_formal_pse_3x256": "pse_256x256x256",
    "ar_formal_rqkmeans_128x256x512": "rqk_128x256x512",
    "ar_formal_rqkmeans_512x256x128": "rqk_512x256x128",
}
RECOMMENDATION_METRICS = (
    "hit_rate@5",
    "hit_rate@10",
    "ndcg@5",
    "ndcg@10",
    "legal@10",
    "duplicate@10",
)


def mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def resolve_position_csv(raw_path: str | None, root: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_file():
        return path
    matches = list((root / "outputs" / "position_loss_logs").rglob(path.name))
    return max(matches, key=lambda item: item.stat().st_mtime_ns) if matches else None


def read_test_positions(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "test":
                continue
            rows.append({
                "position": int(row["position"]),
                "role": row["role"],
                "nll": float(row["nll"]),
                "normalized_nll": float(row["normalized_nll"]),
                "accuracy": float(row["accuracy"]),
                "prefix_accuracy": float(row["prefix_accuracy"]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-best-epoch", type=int, default=5)
    args = parser.parse_args()

    root = args.project_root.resolve()
    metadata_root = (args.metadata_root or root / "outputs" / "run_metadata").resolve()
    output_dir = (args.output_dir or root / "outputs" / "reports" / "ar_internal_study").resolve()

    selected: dict[tuple[str, int], tuple[int, Path, dict]] = {}
    for path in metadata_root.rglob("*_run_metadata.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        variant = RUN_VARIANTS.get(data.get("run_id"))
        if variant is None or int(data.get("best_epoch") or 0) < args.min_best_epoch:
            continue
        seed = int(data.get("seed", -1))
        key = (variant, seed)
        candidate = (path.stat().st_mtime_ns, path, data)
        if key not in selected or candidate[0] > selected[key][0]:
            selected[key] = candidate

    if not selected:
        raise SystemExit(f"No completed internal-study runs under {metadata_root}")

    recommendation_rows = []
    position_rows = []
    tokenizer_rows = []
    for (variant, seed), (_, path, data) in sorted(selected.items()):
        semantic_sizes = data.get("semantic_codebook_sizes") or [256, 256, 256]
        row = {
            "variant": variant,
            "seed": seed,
            "semantic_codebook_sizes": "x".join(map(str, semantic_sizes)),
            "best_epoch": int(data["best_epoch"]),
            "fit_hours": float(data.get("fit_elapsed_seconds") or 0.0) / 3600,
            "metadata": str(path),
            "position_metrics_csv": str(
                resolve_position_csv(data.get("position_metrics_csv"), root) or ""
            ),
        }
        for metric in RECOMMENDATION_METRICS:
            row[metric] = (data.get("test_results") or {}).get(metric)
        recommendation_rows.append(row)

        collision = data.get("sid_collision_stats") or {}
        tokenizer_row = {
            "variant": variant,
            "seed": seed,
            "semantic_codebook_sizes": "x".join(map(str, semantic_sizes)),
            "semantic_collision_rate": collision.get("collision_rate"),
            "largest_collision_group": collision.get("largest_collision_group"),
        }
        for prefix in collision.get("per_prefix", []):
            tokenizer_row[f"prefix_{prefix['prefix_length']}_collision_rate"] = prefix[
                "collision_rate"
            ]
        for position in collision.get("per_position", []):
            label = int(position["position"]) + 1
            tokenizer_row[f"position_{label}_utilization"] = position[
                "utilization_rate"
            ]
            tokenizer_row[f"position_{label}_normalized_entropy"] = position[
                "normalized_entropy"
            ]
        tokenizer_rows.append(tokenizer_row)

        position_csv = resolve_position_csv(data.get("position_metrics_csv"), root)
        for position in read_test_positions(position_csv):
            position_rows.append({
                "variant": variant,
                "seed": seed,
                "codebook_size": (
                    semantic_sizes[position["position"] - 1]
                    if position["position"] <= len(semantic_sizes)
                    else (data.get("codebook_sizes") or [256] * 4)[position["position"] - 1]
                ),
                **position,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(recommendation_rows[0]))
        writer.writeheader()
        writer.writerows(recommendation_rows)
    if position_rows:
        with (output_dir / "test_position_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(position_rows[0]))
            writer.writeheader()
            writer.writerows(position_rows)
    if tokenizer_rows:
        tokenizer_fields = []
        for row in tokenizer_rows:
            for field in row:
                if field not in tokenizer_fields:
                    tokenizer_fields.append(field)
        with (output_dir / "tokenizer_stats.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tokenizer_fields)
            writer.writeheader()
            writer.writerows(tokenizer_rows)

    grouped = defaultdict(list)
    for row in recommendation_rows:
        grouped[row["variant"]].append(row)
    summary = {}
    for variant, rows in grouped.items():
        summary[variant] = {
            "n_seeds": len(rows),
            "seeds": [row["seed"] for row in rows],
            "semantic_codebook_sizes": rows[0]["semantic_codebook_sizes"],
        }
        for metric in RECOMMENDATION_METRICS:
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            summary[variant][metric] = mean_std(values)

    position_grouped = defaultdict(list)
    for row in position_rows:
        position_grouped[(row["variant"], row["position"], row["role"])].append(row)
    for (variant, position, role), rows in position_grouped.items():
        summary.setdefault(variant, {}).setdefault("test_positions", {})[
            f"position_{position}_{role}"
        ] = {
            "codebook_size": rows[0]["codebook_size"],
            "nll": mean_std([row["nll"] for row in rows]),
            "normalized_nll": mean_std([row["normalized_nll"] for row in rows]),
            "accuracy": mean_std([row["accuracy"] for row in rows]),
        }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
