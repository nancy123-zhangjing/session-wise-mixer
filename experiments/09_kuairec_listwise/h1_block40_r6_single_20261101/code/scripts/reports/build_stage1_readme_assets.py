from pathlib import Path
import csv

import fitz
from PIL import Image
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
# Legacy regeneration depends on the original import bundle, not run evidence.
EVIDENCE = ROOT / "archive/evidence_bundles/repro_sources_01_04/01_ar_baseline_full_evidence/formal_3x256_evidence_v2"
PDF = ROOT / "experiments/01_ar_baseline_3x256/pdf/AR_GRM_3x256_stage1_report.pdf"
OUT = ROOT / "experiments/01_ar_baseline_3x256/report_assets"


# Zero-based PDF page index -> (output name, crop box in 1440x810 render pixels).
CROPS = {
    4: ("downstream_metrics.png", (65, 135, 1375, 680)),
    5: ("seed_consistency.png", (65, 125, 1375, 670)),
    6: ("tokenizer_geometry.png", (65, 130, 1375, 665)),
    7: ("codebook_usage.png", (65, 130, 1375, 670)),
    8: ("position_nll.png", (65, 125, 1375, 680)),
    9: ("position_accuracy.png", (65, 130, 1375, 680)),
}

POSITION_COLORS = {
    1: HexColor("#2A64E8"),
    2: HexColor("#F59A00"),
    3: HexColor("#009B72"),
    4: HexColor("#D94B4B"),
}


def draw_position_nll(csv_path: Path, out_path: Path, title: str) -> None:
    """Plot the recorded NLL points directly; no smoothing or interpolation."""
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    values = {}
    for row in rows:
        key = (row["split"], int(row["position"]))
        values.setdefault(key, []).append((int(row["epoch"]), float(row["nll"])))

    all_epochs = [epoch for points in values.values() for epoch, _ in points]
    max_epoch = max(all_epochs)
    y_max = 5.6

    temp_pdf = out_path.with_suffix(".pdf")
    c = canvas.Canvas(str(temp_pdf), pagesize=(960, 540))
    c.setFillColor(white)
    c.rect(0, 0, 960, 540, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(58, 500, title)
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#667085"))
    c.drawRightString(900, 504, "Recorded NLL values; no smoothing")

    x0, y0, width, height = 65, 65, 835, 385
    c.setFont("Helvetica", 9)
    for tick in range(0, 7):
        value = y_max * tick / 6
        yy = y0 + height * tick / 6
        c.setStrokeColor(HexColor("#DDE3EA"))
        c.setLineWidth(0.7)
        c.line(x0, yy, x0 + width, yy)
        c.setFillColor(HexColor("#5B6573"))
        c.drawRightString(x0 - 8, yy - 3, f"{value:.1f}")

    x_ticks = 5
    for tick in range(x_ticks + 1):
        epoch = round(max_epoch * tick / x_ticks)
        xx = x0 + width * tick / x_ticks
        c.setFillColor(HexColor("#5B6573"))
        c.drawCentredString(xx, y0 - 18, str(epoch))

    c.setStrokeColor(HexColor("#9AA4B2"))
    c.setLineWidth(1)
    c.line(x0, y0, x0, y0 + height)
    c.line(x0, y0, x0 + width, y0)
    c.setFillColor(black)
    c.setFont("Helvetica", 10)
    c.drawCentredString(x0 + width / 2, 28, "Epoch")
    c.saveState()
    c.translate(20, y0 + height / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "NLL")
    c.restoreState()

    for split in ("train", "val"):
        for position in range(1, 5):
            points = sorted(values.get((split, position), []))
            if not points:
                continue
            mapped = [
                (
                    x0 + width * epoch / max_epoch,
                    y0 + height * max(0.0, min(nll, y_max)) / y_max,
                )
                for epoch, nll in points
            ]
            c.setStrokeColor(POSITION_COLORS[position])
            c.setLineWidth(2.1 if split == "train" else 1.8)
            c.setDash([] if split == "train" else [7, 4])
            for first, second in zip(mapped, mapped[1:]):
                c.line(first[0], first[1], second[0], second[1])
            if split == "val":
                c.setDash([])
                c.setFillColor(POSITION_COLORS[position])
                for xx, yy in mapped:
                    c.circle(xx, yy, 2.5, fill=1, stroke=0)

    c.setDash([])
    legend_y = 470
    legend_x = 65
    for position, label in ((1, "Pos 1"), (2, "Pos 2"), (3, "Pos 3"), (4, "Pos 4 / suffix")):
        c.setStrokeColor(POSITION_COLORS[position])
        c.setLineWidth(3)
        c.line(legend_x, legend_y, legend_x + 20, legend_y)
        c.setFillColor(black)
        c.setFont("Helvetica", 9)
        c.drawString(legend_x + 27, legend_y - 3, label)
        legend_x += 115 if position < 4 else 145
    c.setStrokeColor(black)
    c.setLineWidth(2)
    c.setDash([])
    c.line(650, legend_y, 672, legend_y)
    c.setFillColor(black)
    c.drawString(678, legend_y - 3, "train")
    c.setDash([7, 4])
    c.line(750, legend_y, 772, legend_y)
    c.setDash([])
    c.drawString(778, legend_y - 3, "validation")

    c.showPage()
    c.save()

    doc = fitz.open(temp_pdf)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(out_path)
    doc.close()
    temp_pdf.unlink()


def draw_rqvae_training_components(out_path: Path) -> None:
    """Draw raw validation loss components in separate seed panels."""
    root = EVIDENCE / "tokenizer_aux"
    seeds = (2024, 2025, 2026)
    series_colors = {
        "val_total": HexColor("#2A64E8"),
        "val_reconstruction": HexColor("#009B72"),
        "val_quantization": HexColor("#F59A00"),
    }
    labels = {
        "val_total": "total",
        "val_reconstruction": "reconstruction",
        "val_quantization": "quantization",
    }

    temp_pdf = out_path.with_suffix(".pdf")
    c = canvas.Canvas(str(temp_pdf), pagesize=(960, 500))
    c.setFillColor(white)
    c.rect(0, 0, 960, 500, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(45, 462, "RQ-VAE tokenizer validation loss components")
    c.setFillColor(HexColor("#667085"))
    c.setFont("Helvetica", 9)
    c.drawRightString(915, 465, "Raw logged values; total = reconstruction + quantization; no smoothing")

    legend_x = 45
    for key in ("val_total", "val_reconstruction", "val_quantization"):
        c.setStrokeColor(series_colors[key])
        c.setLineWidth(3)
        c.line(legend_x, 432, legend_x + 22, 432)
        c.setFillColor(black)
        c.setFont("Helvetica", 9)
        c.drawString(legend_x + 28, 429, labels[key])
        legend_x += 135
    c.setStrokeColor(HexColor("#D94B4B"))
    c.setDash([5, 3])
    c.line(480, 432, 502, 432)
    c.setDash([])
    c.setFillColor(black)
    c.drawString(508, 429, "selected checkpoint")

    y_min, y_max = 0.0, 0.40
    panel_y, panel_h, panel_w = 78, 300, 260
    for index, seed in enumerate(seeds):
        panel_x = 55 + index * 300
        training_path = root / f"rqvae_seed{seed}" / "training.csv"
        summary_path = root / f"rqvae_seed{seed}" / "summary.json"
        with training_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        import json
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        max_step = max(int(row["step"]) for row in rows)
        best_step = int(summary["best_step"])
        best_loss = float(summary["best_validation_total_loss"])

        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(panel_x, 398, f"Seed {seed}")
        c.setFillColor(HexColor("#667085"))
        c.setFont("Helvetica", 8.5)
        c.drawRightString(panel_x + panel_w, 398, f"selected: {best_step // 1000}k, total={best_loss:.4f}")

        for tick in range(5):
            value = y_min + (y_max - y_min) * tick / 4
            yy = panel_y + panel_h * tick / 4
            c.setStrokeColor(HexColor("#DDE3EA"))
            c.setLineWidth(0.7)
            c.line(panel_x, yy, panel_x + panel_w, yy)
            c.setFillColor(HexColor("#5B6573"))
            c.setFont("Helvetica", 7.5)
            c.drawRightString(panel_x - 5, yy - 3, f"{value:.2f}")
        for step_k in (0, 20, 40, 60):
            xx = panel_x + panel_w * (step_k * 1000) / max_step
            c.setFillColor(HexColor("#5B6573"))
            c.drawCentredString(xx, panel_y - 16, f"{step_k}k")

        c.setStrokeColor(HexColor("#9AA4B2"))
        c.setLineWidth(1)
        c.line(panel_x, panel_y, panel_x, panel_y + panel_h)
        c.line(panel_x, panel_y, panel_x + panel_w, panel_y)

        for key in ("val_total", "val_reconstruction", "val_quantization"):
            points = []
            for row in rows:
                step = int(row["step"])
                value = float(row[key])
                xx = panel_x + panel_w * step / max_step
                yy = panel_y + panel_h * (value - y_min) / (y_max - y_min)
                points.append((xx, yy))
            c.setStrokeColor(series_colors[key])
            c.setLineWidth(2)
            for first, second in zip(points, points[1:]):
                c.line(first[0], first[1], second[0], second[1])

        best_x = panel_x + panel_w * best_step / max_step
        best_y = panel_y + panel_h * (best_loss - y_min) / (y_max - y_min)
        c.setStrokeColor(HexColor("#D94B4B"))
        c.setLineWidth(1.3)
        c.setDash([5, 3])
        c.line(best_x, panel_y, best_x, panel_y + panel_h)
        c.setDash([])
        c.setFillColor(HexColor("#D94B4B"))
        c.circle(best_x, best_y, 4, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica", 10)
    c.drawCentredString(480, 30, "Training step")
    c.saveState()
    c.translate(18, panel_y + panel_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Validation loss")
    c.restoreState()
    c.showPage()
    c.save()

    doc = fitz.open(temp_pdf)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(out_path)
    doc.close()
    temp_pdf.unlink()


def draw_all_position_nll(out_path: Path) -> None:
    """Six-panel comparison of raw per-position AR NLL curves."""
    metrics_dir = EVIDENCE / "position_metrics"
    temp_pdf = out_path.with_suffix(".pdf")
    c = canvas.Canvas(str(temp_pdf), pagesize=(960, 690))
    c.setFillColor(white)
    c.rect(0, 0, 960, 690, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(48, 653, "AR decoder per-position SFT NLL")
    c.setFillColor(HexColor("#667085"))
    c.setFont("Helvetica", 9)
    c.drawRightString(912, 656, "Recorded train/validation NLL; no smoothing")

    legend_x = 48
    for position, label in ((1, "Pos 1"), (2, "Pos 2"), (3, "Pos 3"), (4, "Pos 4 / suffix")):
        c.setStrokeColor(POSITION_COLORS[position])
        c.setLineWidth(3)
        c.line(legend_x, 625, legend_x + 20, 625)
        c.setFillColor(black)
        c.setFont("Helvetica", 8.5)
        c.drawString(legend_x + 26, 622, label)
        legend_x += 105 if position < 4 else 135
    c.setStrokeColor(black)
    c.setLineWidth(2)
    c.setDash([])
    c.line(610, 625, 632, 625)
    c.setFillColor(black)
    c.drawString(638, 622, "train")
    c.setDash([6, 4])
    c.line(710, 625, 732, 625)
    c.setDash([])
    c.drawString(738, 622, "validation")

    layouts = (("rq_kmeans", "RQ-KMeans", 2024), ("rq_kmeans", "RQ-KMeans", 2025),
               ("rq_kmeans", "RQ-KMeans", 2026), ("rqvae", "RQ-VAE", 2024),
               ("rqvae", "RQ-VAE", 2025), ("rqvae", "RQ-VAE", 2026))
    y_max = 5.6
    panel_w, panel_h = 260, 220
    for index, (quantizer, display, seed) in enumerate(layouts):
        col, row = index % 3, index // 3
        panel_x = 60 + col * 300
        panel_y = 345 if row == 0 else 65
        path = metrics_dir / f"{quantizer}_seed{seed}_position_metrics.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        values = {}
        for item in rows:
            key = (item["split"], int(item["position"]))
            values.setdefault(key, []).append((int(item["epoch"]), float(item["nll"])))
        max_epoch = max(epoch for points in values.values() for epoch, _ in points)

        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(panel_x, panel_y + panel_h + 18, f"{display} | seed {seed}")
        for tick in range(5):
            value = y_max * tick / 4
            yy = panel_y + panel_h * tick / 4
            c.setStrokeColor(HexColor("#DDE3EA"))
            c.setLineWidth(0.6)
            c.line(panel_x, yy, panel_x + panel_w, yy)
            if col == 0:
                c.setFillColor(HexColor("#5B6573"))
                c.setFont("Helvetica", 7)
                c.drawRightString(panel_x - 5, yy - 2.5, f"{value:.1f}")
        for epoch in (0, max_epoch // 2, max_epoch):
            xx = panel_x + panel_w * epoch / max_epoch
            c.setFillColor(HexColor("#5B6573"))
            c.setFont("Helvetica", 7)
            c.drawCentredString(xx, panel_y - 13, str(epoch))
        c.setStrokeColor(HexColor("#9AA4B2"))
        c.setLineWidth(0.8)
        c.line(panel_x, panel_y, panel_x, panel_y + panel_h)
        c.line(panel_x, panel_y, panel_x + panel_w, panel_y)

        for split in ("train", "val"):
            for position in range(1, 5):
                points = sorted(values.get((split, position), []))
                mapped = [
                    (panel_x + panel_w * epoch / max_epoch,
                     panel_y + panel_h * max(0.0, min(nll, y_max)) / y_max)
                    for epoch, nll in points
                ]
                c.setStrokeColor(POSITION_COLORS[position])
                c.setLineWidth(1.5 if split == "train" else 1.25)
                c.setDash([] if split == "train" else [5, 3])
                for first, second in zip(mapped, mapped[1:]):
                    c.line(first[0], first[1], second[0], second[1])
                if split == "val":
                    c.setDash([])
                    c.setFillColor(POSITION_COLORS[position])
                    for xx, yy in mapped:
                        c.circle(xx, yy, 1.7, fill=1, stroke=0)

    c.setFillColor(black)
    c.setFont("Helvetica", 9)
    c.drawCentredString(480, 24, "Epoch")
    c.saveState()
    c.translate(18, 330)
    c.rotate(90)
    c.drawCentredString(0, 0, "NLL")
    c.restoreState()
    c.showPage()
    c.save()

    doc = fitz.open(temp_pdf)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(out_path)
    doc.close()
    temp_pdf.unlink()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    for page_index, (name, crop_box) in CROPS.items():
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        full = OUT / f"_{page_index + 1:02d}.png"
        pix.save(full)
        with Image.open(full) as image:
            image.crop(crop_box).save(OUT / name, optimize=True)
        full.unlink()

    metrics_dir = EVIDENCE / "position_metrics"
    for quantizer, display in (("rq_kmeans", "RQ-KMeans"), ("rqvae", "RQ-VAE")):
        for seed in (2024, 2025, 2026):
            draw_position_nll(
                metrics_dir / f"{quantizer}_seed{seed}_position_metrics.csv",
                OUT / f"position_nll_{quantizer}_seed{seed}.png",
                f"{display} | seed {seed} | Per-position SFT NLL",
            )

    draw_rqvae_training_components(OUT / "rqvae_training.png")
    draw_all_position_nll(OUT / "position_nll_all_seeds.png")

    print(f"generated {len(CROPS) + 8} README assets in {OUT}")


if __name__ == "__main__":
    main()
