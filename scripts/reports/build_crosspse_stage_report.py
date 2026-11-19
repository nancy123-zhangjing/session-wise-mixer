"""Build concise, dependency-free SVG figures for the Cross-PSE report."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path


RUNS = {
    "Original PSE": "diff_formal_pse_4x256_repo_repro_v6_bf16",
    "Refined-PSE": "diff_formal_refinedpse_4x256_v2_catalog_bf16",
    "Cross-PSE": "diff_formal_crosspse_4x256_v3_catalog_bf16",
}
COLORS = {
    "Original PSE": "#59636f",
    "Refined-PSE": "#16867a",
    "Cross-PSE": "#d95f45",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    common = """
<style>
text { font-family: Arial, Helvetica, sans-serif; fill: #20242a; }
.title { font-size: 22px; font-weight: 600; }
.axis { font-size: 14px; }
.tick { font-size: 12px; fill: #59636f; }
.value { font-size: 12px; font-weight: 600; }
.grid { stroke: #d8dde3; stroke-width: 1; }
.frame { stroke: #7d8792; stroke-width: 1.2; fill: none; }
</style>
"""
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n{common}\n'
        + "\n".join(body)
        + "\n</svg>\n",
        encoding="utf-8",
    )


def legend(x: float, y: float) -> list[str]:
    output = []
    for index, label in enumerate(RUNS):
        item_x = x + index * 185
        output.append(
            f'<line x1="{item_x}" y1="{y}" x2="{item_x + 26}" y2="{y}" '
            f'stroke="{COLORS[label]}" stroke-width="4"/>'
        )
        output.append(
            f'<text x="{item_x + 34}" y="{y + 5}" class="axis">{escape(label)}</text>'
        )
    return output


def recommendation_plot(evidence: Path, output: Path) -> None:
    comparison = load_json(
        evidence / "reports" / "diff_tokenizer_comparison" / "summary.json"
    )
    keys = {
        "Original PSE": "pse",
        "Refined-PSE": "refined_pse_v2_catalog",
        "Cross-PSE": "cross_pse_v3_catalog",
    }
    metrics = [
        ("hit_rate@5", "HR@5"),
        ("hit_rate@10", "HR@10"),
        ("ndcg@5", "NDCG@5"),
        ("ndcg@10", "NDCG@10"),
    ]
    width, height = 1000, 540
    left, right, top, bottom = 82, 36, 105, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = 8.5
    body = [
        '<title>DiffGRM recommendation results for Original PSE, Refined-PSE and Cross-PSE</title>',
        '<text x="500" y="34" text-anchor="middle" class="title">DiffGRM recommendation results (seed 2024)</text>',
    ]
    body += legend(245, 69)
    for tick in range(0, 9, 2):
        y = top + plot_h - tick / ymax * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')
    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>')
    group_w = plot_w / len(metrics)
    bar_w = 48
    for metric_index, (metric, label) in enumerate(metrics):
        center = left + group_w * (metric_index + 0.5)
        body.append(f'<text x="{center:.1f}" y="{height - 42}" text-anchor="middle" class="axis">{label}</text>')
        for run_index, run_label in enumerate(RUNS):
            value = 100.0 * comparison[metric][keys[run_label]]["mean"]
            x = center + (run_index - 1) * 58 - bar_w / 2
            y = top + plot_h - value / ymax * plot_h
            h = top + plot_h - y
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" fill="{COLORS[run_label]}"/>')
            body.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 7:.1f}" text-anchor="middle" class="value">{value:.2f}</text>')
    body.append(f'<text transform="translate(24 {top + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">Test metric (%)</text>')
    write_svg(output, width, height, body)


def position_nll_plot(evidence: Path, output: Path) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 40, 105, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 6.1, 6.9
    body = [
        '<title>Fully-masked test NLL by SID position</title>',
        '<text x="450" y="34" text-anchor="middle" class="title">Position-wise decoding difficulty</text>',
    ]
    body += legend(195, 69)
    for tick in [6.1, 6.3, 6.5, 6.7, 6.9]:
        y = top + plot_h - (tick - ymin) / (ymax - ymin) * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>')
    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>')
    x_values = [left + 70 + index * (plot_w - 140) / 3 for index in range(4)]
    for index, x in enumerate(x_values):
        body.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" class="axis">SID{index}</text>')
    for label, run_id in RUNS.items():
        rows = read_csv(evidence / "reports" / run_id / "position_summary.csv")
        selected = sorted(
            (row for row in rows if row["split"] == "test"),
            key=lambda row: int(row["position"]),
        )
        values = [float(row["nll_mean"]) for row in selected]
        points = []
        for x, value in zip(x_values, values):
            y = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
            points.append((x, y, value))
        body.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points) + f'" fill="none" stroke="{COLORS[label]}" stroke-width="3"/>')
        for x, y, value in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[label]}"/>')
            offset = -10 if label != "Cross-PSE" else 18
            body.append(f'<text x="{x:.1f}" y="{y + offset:.1f}" text-anchor="middle" class="value">{value:.2f}</text>')
    body.append(f'<text transform="translate(24 {top + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">Fully-masked test NLL</text>')
    write_svg(output, width, height, body)


def attention_dir(evidence: Path, run_id: str) -> Path:
    return evidence / "attention" / f"{run_id}_seed2024"


def blue(value: float, low: float = 0.15, high: float = 0.37) -> str:
    ratio = max(0.0, min(1.0, (value - low) / (high - low)))
    start, end = (232, 241, 249), (26, 93, 156)
    rgb = tuple(round(a + ratio * (b - a)) for a, b in zip(start, end))
    return "#%02x%02x%02x" % rgb


def decode_order_plot(evidence: Path, output: Path) -> None:
    width, height = 1180, 500
    body = [
        '<title>CPD position selection order</title>',
        '<text x="590" y="34" text-anchor="middle" class="title">CPD position selection order</text>',
    ]
    panel_x = [70, 430, 790]
    cell_w, cell_h = 68, 70
    for start_x, (label, run_id) in zip(panel_x, RUNS.items()):
        rows = read_csv(attention_dir(evidence, run_id) / "selected_position_frequency.csv")
        matrix = [[0.0] * 4 for _ in range(4)]
        for row in rows:
            matrix[int(row["decode_rank"]) - 1][int(row["position"])] = float(row["frequency"])
        body.append(f'<text x="{start_x + 2*cell_w}" y="78" text-anchor="middle" class="axis">{escape(label)}</text>')
        for rank in range(4):
            body.append(f'<text x="{start_x - 12}" y="{110 + rank*cell_h + cell_h/2 + 4:.1f}" text-anchor="end" class="tick">R{rank+1}</text>')
            for position in range(4):
                value = matrix[rank][position]
                x, y = start_x + position * cell_w, 110 + rank * cell_h
                body.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{blue(value)}" stroke="#ffffff"/>')
                color = "#ffffff" if value > 0.28 else "#20242a"
                body.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 5}" text-anchor="middle" class="value" style="fill:{color}">{value:.2f}</text>')
        for position in range(4):
            body.append(f'<text x="{start_x + position*cell_w + cell_w/2}" y="420" text-anchor="middle" class="tick">SID{position}</text>')
    body.append('<text x="590" y="465" text-anchor="middle" class="axis">Selected SID position; R1 is decoded first</text>')
    write_svg(output, width, height, body)


def revealed_mass(rows: list[dict[str, str]], rank: int) -> float:
    selected = [row for row in rows if int(row["decode_rank"]) == rank]
    base_count = sum(float(row["n_layer_head_examples"]) for row in selected) / 4.0
    return sum(
        float(row["mean_attention"]) * float(row["n_layer_head_examples"])
        for row in selected
        if row["key_state"] == "revealed"
    ) / base_count


def attention_bias_plot(evidence: Path, output: Path) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 40, 105, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin, ymax = 0.0, 9.0
    body = [
        '<title>Decoder preference for already revealed SID positions</title>',
        '<text x="450" y="34" text-anchor="middle" class="title">Decoder preference for already revealed SID positions</text>',
    ]
    body += legend(195, 69)
    for tick in [0, 2, 4, 6, 8]:
        y = top + plot_h - (tick - ymin) / (ymax - ymin) * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{tick}</text>')
    body.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="frame"/>')
    ranks = [2, 3, 4]
    x_values = [left + 95 + index * (plot_w - 190) / 2 for index in range(3)]
    for rank, x in zip(ranks, x_values):
        body.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" class="axis">Rank {rank}</text>')
    for label, run_id in RUNS.items():
        rows = read_csv(attention_dir(evidence, run_id) / "attention_overall.csv")
        values = [100.0 * (revealed_mass(rows, rank) - (rank - 1) / 4.0) for rank in ranks]
        points = []
        for x, value in zip(x_values, values):
            y = top + plot_h - (value - ymin) / (ymax - ymin) * plot_h
            points.append((x, y, value))
        body.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points) + f'" fill="none" stroke="{COLORS[label]}" stroke-width="3"/>')
        for x, y, value in points:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[label]}"/>')
            body.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="value">{value:.2f}</text>')
    body.append(f'<text transform="translate(24 {top + plot_h/2:.1f}) rotate(-90)" text-anchor="middle" class="axis">Excess revealed attention (percentage points)</text>')
    write_svg(output, width, height, body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recommendation_plot(args.evidence, args.output / "01_recommendation_metrics.svg")
    position_nll_plot(args.evidence, args.output / "02_position_nll.svg")
    decode_order_plot(args.evidence, args.output / "03_decode_order.svg")
    attention_bias_plot(args.evidence, args.output / "04_revealed_attention_bias.svg")


if __name__ == "__main__":
    main()
