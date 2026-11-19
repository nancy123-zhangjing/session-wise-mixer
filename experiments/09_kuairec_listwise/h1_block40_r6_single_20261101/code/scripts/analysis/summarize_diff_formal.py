"""Aggregate completed formal DiffGRM runs and draw core diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIFF_ROOT = PROJECT_ROOT / "external" / "DiffGRM"
sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import resolve_project_path  # noqa: E402

RUN_ID = "diff_formal_pse_4x256_repo_repro_v6_bf16"
METRICS = (
    "hit_rate@5",
    "hit_rate@10",
    "recall@5",
    "recall@10",
    "ndcg@5",
    "ndcg@10",
    "legal@10",
    "duplicate@10",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resolve_run_path(value: str) -> Path:
    return resolve_project_path(value, project_root=PROJECT_ROOT, relative_to=DIFF_ROOT)


def summarize_numeric(rows: list[dict], keys: tuple[str, ...], values: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, group in sorted(grouped.items()):
        result = dict(zip(keys, group_key))
        result["n_seeds"] = len({int(row["seed"]) for row in group})
        for value in values:
            numbers = np.asarray([float(row[value]) for row in group], dtype=float)
            result[f"{value}_mean"] = float(numbers.mean())
            result[f"{value}_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
        output.append(result)
    return output


def plot_position_nll(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["split"] == "test"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    for seed in sorted({int(row["seed"]) for row in selected}):
        group = sorted(
            (row for row in selected if int(row["seed"]) == seed),
            key=lambda row: int(row["position"]),
        )
        ax.plot(
            [int(row["position"]) for row in group],
            [float(row["nll"]) for row in group],
            marker="o",
            linewidth=1.8,
            label=str(seed),
        )
    ax.set_title("DiffGRM fully-masked test NLL by SID position")
    ax.set_xlabel("SID position")
    ax.set_ylabel("NLL")
    ax.set_xticks(range(4), ["SID0", "SID1", "SID2", "SID3"])
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Seed", frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_order_heatmaps(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    order_types = (
        ("ocn_noising_hard_to_easy", "OCN training noising: hard to easy"),
        ("cpd_greedy_decode_easy_to_hard", "CPD greedy decode: confident to hard"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=180)
    for ax, (order_type, title) in zip(axes, order_types):
        selected = [row for row in rows if row["order_type"] == order_type]
        matrix = np.zeros((4, 4), dtype=float)
        for rank in range(1, 5):
            for position in range(4):
                values = [
                    float(row["frequency"])
                    for row in selected
                    if int(row["rank"]) == rank and int(row["position"]) == position
                ]
                matrix[rank - 1, position] = float(np.mean(values)) if values else np.nan
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        for rank in range(4):
            for position in range(4):
                ax.text(
                    position,
                    rank,
                    f"{matrix[rank, position]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[rank, position] > 0.5 else "black",
                )
        ax.set_title(title)
        ax.set_xlabel("SID position")
        ax.set_ylabel("Rank")
        ax.set_xticks(range(4), ["SID0", "SID1", "SID2", "SID3"])
        ax.set_yticks(range(4), ["1", "2", "3", "4"])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Frequency", shrink=0.84)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=180)
    for seed in sorted({int(row["seed"]) for row in rows}):
        group = sorted(
            (row for row in rows if int(row["seed"]) == seed),
            key=lambda row: int(row["epoch"]),
        )
        epochs = [int(row["epoch"]) for row in group]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in group], label=str(seed))
        axes[1].plot(epochs, [float(row["lr_end"]) for row in group], label=str(seed))
    axes[0].set_title("OCN training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Learning-rate schedule")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LR at epoch end")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(title="Seed", frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_validation_curves(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = (("ndcg@10", "NDCG@10"), ("hit_rate@10", "HitRate@10"))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=180)
    for ax, (metric, title) in zip(axes, metrics):
        selected = [
            row for row in rows
            if row["split"] == "val" and row["metric"] == metric
        ]
        for seed in sorted({int(row["seed"]) for row in selected}):
            group = sorted(
                (row for row in selected if int(row["seed"]) == seed),
                key=lambda row: int(row["epoch"]),
            )
            ax.plot(
                [int(row["epoch"]) for row in group],
                [float(row["value"]) for row in group],
                marker="o", markersize=2.8, label=str(seed),
            )
        ax.set_title(f"Validation {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
        ax.legend(title="Seed", frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_full_mask_history(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(
        len(seeds), 1, figsize=(8.2, 3.5 * len(seeds)), dpi=180, squeeze=False
    )
    for ax, seed in zip(axes[:, 0], seeds):
        selected = [row for row in rows if int(row["seed"]) == seed]
        for position in range(4):
            group = sorted(
                (row for row in selected if int(row["position"]) == position),
                key=lambda row: int(row["epoch"]),
            )
            ax.plot(
                [int(row["epoch"]) for row in group],
                [float(row["nll"]) for row in group],
                marker="o", markersize=2.5, label=f"SID{position}",
            )
        ax.set_title(f"Seed {seed}: validation fully-masked NLL")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--metadata-root", type=Path, default=PROJECT_ROOT / "outputs" / "run_metadata"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--require-all-seeds", action="store_true")
    args = parser.parse_args()
    args.metadata_root = resolve_project_path(
        args.metadata_root, project_root=PROJECT_ROOT, relative_to=Path.cwd()
    )
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "outputs" / "reports" / args.run_id

    selected: dict[int, tuple[Path, dict]] = {}
    for path in sorted(
        args.metadata_root.rglob("*_run_metadata.json"),
        key=lambda item: (item.stat().st_mtime_ns, str(item)),
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("run_id") != args.run_id:
            continue
        seed = int(data["seed"])
        selected[seed] = (path, data)

    if not selected:
        raise SystemExit(f"No completed {args.run_id} runs found")
    missing = sorted({2024, 2025, 2026} - set(selected))
    if args.require_all_seeds and missing:
        raise SystemExit(f"Missing formal DiffGRM seeds: {missing}")

    per_seed = []
    position_rows = []
    order_rows = []
    training_rows = []
    evaluation_rows = []
    full_mask_history_rows = []
    joint_rows = []
    for seed, (metadata_path, data) in sorted(selected.items()):
        row = {
            "seed": seed,
            "sid_quantizer": data.get("sid_quantizer"),
            "semantic_ids_sha256": (
                data.get("joint_final_semantic_ids_sha256")
                or data.get("semantic_ids_sha256")
            ),
            "best_epoch": data.get("best_epoch"),
            "fit_hours": float(data.get("fit_elapsed_seconds") or 0.0) / 3600,
            "peak_vram_mb": data.get("peak_vram_mb"),
            "metadata": str(metadata_path),
            "sid_collision_rate": (
                (data.get("sid_collision_stats") or {}).get("collision_rate")
            ),
        }
        for prefix in (data.get("sid_collision_stats") or {}).get("per_prefix", []):
            row[f"prefix{prefix['prefix_length']}_collision_rate"] = prefix[
                "collision_rate"
            ]
        for metric in METRICS:
            row[metric] = data.get("test_results", {}).get(metric)
        per_seed.append(row)

        best_epoch = int(data.get("best_epoch") or 0)
        raw_positions = read_csv(resolve_run_path(data["position_metrics_csv"]))
        for raw in raw_positions:
            keep = (
                raw["split"] == "test" and raw["context"] == "fully_masked"
            ) or (
                raw["split"] == "train"
                and raw["context"] == "ocn_masked_views"
                and int(raw["epoch"]) == best_epoch
            )
            if keep:
                position_rows.append({"seed": seed, **raw})
            if raw["split"] == "val" and raw["context"] == "fully_masked":
                full_mask_history_rows.append({"seed": seed, **raw})
        for raw in read_csv(resolve_run_path(data["order_metrics_csv"])):
            keep = (
                raw["split"] == "test"
                and raw["order_type"] == "cpd_greedy_decode_easy_to_hard"
            ) or (
                raw["split"] == "train"
                and raw["order_type"] == "ocn_noising_hard_to_easy"
                and int(raw["epoch"]) == best_epoch
            )
            if keep:
                order_rows.append({"seed": seed, **raw})
        for raw in read_csv(resolve_run_path(data["training_metrics_csv"])):
            training_rows.append({"seed": seed, **raw})
        for raw in read_csv(resolve_run_path(data["evaluation_metrics_csv"])):
            evaluation_rows.append({"seed": seed, **raw})
        joint_path = data.get("joint_metrics_csv") or (
            (data.get("joint_alignment") or {}).get("joint_metrics_csv")
        )
        if joint_path:
            for raw in read_csv(resolve_run_path(joint_path)):
                joint_rows.append({"seed": seed, **raw})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", per_seed)
    write_csv(args.output_dir / "position_per_seed.csv", position_rows)
    write_csv(args.output_dir / "order_per_seed.csv", order_rows)
    write_csv(args.output_dir / "training_history.csv", training_rows)
    write_csv(args.output_dir / "evaluation_history.csv", evaluation_rows)
    write_csv(args.output_dir / "full_mask_nll_history.csv", full_mask_history_rows)
    write_csv(args.output_dir / "joint_training_history.csv", joint_rows)

    position_summary = summarize_numeric(
        position_rows,
        ("split", "context", "position"),
        ("loss", "nll", "normalized_nll", "accuracy", "mask_rate"),
    )
    order_summary = summarize_numeric(
        order_rows,
        ("split", "order_type", "rank", "position"),
        ("frequency",),
    )
    write_csv(args.output_dir / "position_summary.csv", position_summary)
    write_csv(args.output_dir / "order_summary.csv", order_summary)

    summary = {
        "run_id": args.run_id,
        "sid_quantizer": per_seed[0].get("sid_quantizer"),
        "semantic_ids_sha256": sorted({
            row["semantic_ids_sha256"]
            for row in per_seed if row.get("semantic_ids_sha256")
        }),
        "n_seeds": len(per_seed),
        "seeds": sorted(selected),
    }
    collision_values = [
        float(row["sid_collision_rate"])
        for row in per_seed
        if row["sid_collision_rate"] is not None
    ]
    summary["sid_collision_rate"] = {
        "mean": float(np.mean(collision_values)) if collision_values else None,
        "std": (
            float(np.std(collision_values, ddof=1))
            if len(collision_values) > 1 else None
        ),
    }
    summary["prefix_collision_rate"] = {}
    for prefix_length in range(1, 5):
        key = f"prefix{prefix_length}_collision_rate"
        values = [float(row[key]) for row in per_seed if row.get(key) is not None]
        summary["prefix_collision_rate"][str(prefix_length)] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    for metric in METRICS:
        values = [float(row[metric]) for row in per_seed if row[metric] is not None]
        summary[metric] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    try:
        plot_position_nll(position_rows, args.output_dir / "position_nll_test.png")
        plot_order_heatmaps(order_rows, args.output_dir / "decode_order_heatmaps.png")
        plot_training_curves(training_rows, args.output_dir / "training_loss_and_lr.png")
        plot_validation_curves(evaluation_rows, args.output_dir / "validation_metrics.png")
        plot_full_mask_history(
            full_mask_history_rows, args.output_dir / "full_mask_nll_by_epoch.png"
        )
    except ImportError as error:
        print(f"[WARN] Skipped plots because matplotlib is unavailable: {error}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved DiffGRM summary to {args.output_dir}")


if __name__ == "__main__":
    main()
