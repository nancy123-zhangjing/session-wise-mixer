"""Summarize seed-2024 AR self-attention for the three new controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    ("pse_256x256x256", "PSE 256x256x256"),
    ("rqk_128x256x512", "RQ-KMeans 128x256x512"),
    ("rqk_512x256x128", "RQ-KMeans 512x256x128"),
)
SEED = 2024


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_run(path: Path, variant: str) -> tuple[list[dict], list[dict], dict]:
    required = (
        path / "attention_by_layer.csv",
        path / "attention_by_head.csv",
        path / "run_info.json",
    )
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError(f"Missing attention files for {variant}: {missing}")
    layers = read_csv(required[0])
    heads = read_csv(required[1])
    info = json.loads(required[2].read_text(encoding="utf-8"))
    if info.get("variant") != variant or int(info.get("seed", -1)) != SEED:
        raise ValueError(f"Run identity mismatch in {path}: {info}")
    if int(info.get("examples", 0)) != 22363:
        raise ValueError(
            f"Expected the complete 22,363-example test split for {variant}; "
            f"got {info.get('examples')}"
        )
    if float(info.get("max_row_sum_error", 1.0)) > 1e-5:
        raise ValueError(f"Attention row-sum validation failed for {variant}")
    if float(info.get("max_future_attention", 1.0)) > 1e-7:
        raise ValueError(f"Causal-mask validation failed for {variant}")
    return layers, heads, info


def plot_layer_attention(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ("BOS", "SID0", "SID1")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    image = None
    for axis, (variant, display) in zip(axes, VARIANTS):
        matrix = np.full((4, 3), np.nan, dtype=float)
        for row in rows:
            if row["variant"] != variant or row["query_target"] != "SID2":
                continue
            matrix[int(row["decoder_layer"]) - 1, keys.index(row["key_token"])] = float(
                row["mean_attention"]
            )
        if np.isnan(matrix).any():
            raise RuntimeError(f"Incomplete SID2 layer matrix for {variant}: {matrix}")
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
        axis.set_title(display)
        axis.set_xticks(range(3), keys)
        axis.set_yticks(range(4), [f"Layer {index}" for index in range(1, 5)])
        axis.set_xlabel("Visible key token")
        for layer in range(4):
            for key in range(3):
                value = matrix[layer, key]
                axis.text(
                    key,
                    layer,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value >= 0.55 else "black",
                    fontsize=9,
                )
    fig.suptitle(
        "Generating SID2: decoder self-attention by layer (seed 2024)\n"
        "mean over the complete test split and 8 heads"
    )
    fig.colorbar(image, ax=axes, label="Attention weight", shrink=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_head_attention(rows: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = ("BOS", "SID0", "SID1")
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), constrained_layout=True)
    image = None
    for row_index, (variant, display) in enumerate(VARIANTS):
        for layer in range(1, 5):
            axis = axes[row_index, layer - 1]
            matrix = np.full((8, 3), np.nan, dtype=float)
            for row in rows:
                if (
                    row["variant"] == variant
                    and row["query_target"] == "SID2"
                    and int(row["decoder_layer"]) == layer
                ):
                    matrix[int(row["head"]) - 1, keys.index(row["key_token"])] = float(
                        row["mean_attention"]
                    )
            if np.isnan(matrix).any():
                raise RuntimeError(
                    f"Incomplete SID2 head matrix for {variant}, layer {layer}: {matrix}"
                )
            image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
            axis.set_title(f"{display}\nLayer {layer}", fontsize=9)
            axis.set_xticks(range(3), keys)
            axis.set_yticks(range(8), [f"H{head}" for head in range(1, 9)])
    fig.suptitle(
        "Generating SID2: decoder self-attention by head (seed 2024)\n"
        "mean over the complete test split"
    )
    fig.colorbar(image, ax=axes, label="Attention weight", shrink=0.9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "attention" / "ar_internal_study",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "ar_internal_study" / "attention",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    layer_rows = []
    head_rows = []
    run_info = []
    for variant, _ in VARIANTS:
        layers, heads, info = validate_run(
            input_root / variant / f"seed{SEED}", variant
        )
        layer_rows.extend(layers)
        head_rows.extend(heads)
        run_info.append(info)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "attention_by_layer_all.csv", layer_rows)
    write_csv(output_dir / "attention_by_head_all.csv", head_rows)
    (output_dir / "attention_runs.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_layer_attention(layer_rows, output_dir / "sid2_attention_by_layer.png")
    plot_head_attention(head_rows, output_dir / "sid2_attention_by_head.png")
    print(f"Saved internal-study attention summary to {output_dir}")


if __name__ == "__main__":
    main()
