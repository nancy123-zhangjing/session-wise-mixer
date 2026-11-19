from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[2]
# Legacy report regeneration still requires the historical import bundle.
# Do not run this generator as part of a layout-only migration.
EVIDENCE = ROOT / "archive/evidence_bundles/repro_sources_01_04/01_ar_baseline_full_evidence/formal_3x256_evidence_v2"
OUTPUT = ROOT / "experiments/01_ar_baseline_3x256/pdf/AR_GRM_3x256_stage1_report.pdf"

PAGE_W, PAGE_H = 960, 540
MARGIN = 42

INK = HexColor("#111111")
MUTED = HexColor("#5D6470")
RULE = HexColor("#B8BCC4")
GRID = HexColor("#E5E7EB")
PANEL = HexColor("#F2F2F2")
PALE_BLUE = HexColor("#EAF3FF")
RQK = HexColor("#2563EB")
RQVAE = HexColor("#F59E0B")
GREEN = HexColor("#059669")
RED = HexColor("#DC2626")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return statistics.mean(values)


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt_pct(value, digits=2):
    return f"{value * 100:.{digits}f}%"


def fmt_mean_sd(values, digits=3, percent=False):
    m, s = mean(values), sd(values)
    if percent:
        return f"{m * 100:.{digits}f}% +/- {s * 100:.{digits}f}%"
    return f"{m:.{digits}f} +/- {s:.{digits}f}"


def register_fonts():
    pdfmetrics.registerFont(TTFont("CN", r"C:\Windows\Fonts\simhei.ttf"))
    pdfmetrics.registerFont(TTFont("EN", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("ENB", r"C:\Windows\Fonts\arialbd.ttf"))


def paragraph(c, text, x, y_top, width, height, size=18, color=INK,
              leading=None, font="CN", bold=False, align=TA_LEFT):
    style = ParagraphStyle(
        name="box",
        fontName=font,
        fontSize=size,
        leading=leading or size * 1.35,
        textColor=color,
        alignment=align,
        spaceAfter=0,
        spaceBefore=0,
    )
    if bold:
        text = f"<b>{text}</b>"
    p = Paragraph(text, style)
    p.wrapOn(c, width, height)
    p.drawOn(c, x, y_top - height)
    return p


def header(c, title, page, eyebrow="第一阶段 · AR生成推荐"):
    c.setFillColor(MUTED)
    c.setFont("CN", 11)
    c.drawString(MARGIN, PAGE_H - 34, eyebrow)
    c.setFillColor(INK)
    c.setFont("CN", 27)
    c.drawString(MARGIN, PAGE_H - 75, title)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.8)
    c.line(MARGIN, PAGE_H - 91, PAGE_W - MARGIN, PAGE_H - 91)
    c.setFillColor(MUTED)
    c.setFont("EN", 9)
    c.drawRightString(PAGE_W - MARGIN, 20, str(page))


def footer(c, source):
    c.setFillColor(MUTED)
    c.setFont("CN", 7.8)
    c.drawString(MARGIN, 19, source[:150])


def panel(c, x, y, w, h, fill=PANEL, stroke=None, radius=6):
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1 if stroke else 0)


def metric_box(c, x, y, w, h, value, label, color=INK, note=None):
    panel(c, x, y, w, h, PANEL)
    c.setFillColor(color)
    c.setFont("ENB", 26)
    c.drawString(x + 18, y + h - 42, value)
    c.setFillColor(INK)
    c.setFont("CN", 12)
    c.drawString(x + 18, y + h - 67, label)
    if note:
        c.setFillColor(MUTED)
        c.setFont("CN", 9)
        c.drawString(x + 18, y + 14, note)


def draw_axes(c, x, y, w, h, y_min, y_max, ticks=5, percent=False):
    c.setStrokeColor(RULE)
    c.setLineWidth(0.8)
    c.line(x, y, x, y + h)
    c.line(x, y, x + w, y)
    c.setFont("EN", 8)
    c.setFillColor(MUTED)
    for i in range(ticks + 1):
        value = y_min + (y_max - y_min) * i / ticks
        yy = y + h * i / ticks
        c.setStrokeColor(GRID)
        c.line(x, yy, x + w, yy)
        label = f"{value * 100:.1f}%" if percent else f"{value:.2f}"
        c.drawRightString(x - 7, yy - 3, label)


def grouped_bar(c, x, y, w, h, categories, series, y_max,
                percent=False, show_values=True):
    draw_axes(c, x, y, w, h, 0, y_max, percent=percent)
    n_cat = len(categories)
    n_series = len(series)
    group_w = w / n_cat
    bar_w = min(34, group_w * 0.28)
    for i, cat in enumerate(categories):
        center = x + group_w * (i + 0.5)
        for j, item in enumerate(series):
            value = item["values"][i]
            bh = h * value / y_max
            bx = center + (j - (n_series - 1) / 2) * (bar_w + 5) - bar_w / 2
            c.setFillColor(item["color"])
            c.rect(bx, y, bar_w, bh, fill=1, stroke=0)
            if show_values:
                c.setFillColor(INK)
                c.setFont("ENB", 8)
                label = f"{value * 100:.2f}%" if percent else f"{value:.3f}"
                c.drawCentredString(bx + bar_w / 2, y + bh + 6, label)
        c.setFillColor(MUTED)
        # Category labels may contain Chinese; use the CJK font to avoid tofu boxes.
        c.setFont("CN", 8.5)
        c.drawCentredString(center, y - 17, cat)
    lx = x + w - 190
    ly = y + h + 17
    for item in series:
        c.setFillColor(item["color"])
        c.rect(lx, ly - 7, 10, 10, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("CN", 9)
        c.drawString(lx + 15, ly - 6, item["name"])
        lx += 95


def line_chart(c, x, y, w, h, categories, series, y_min, y_max,
               percent=False, error_values=None, value_labels=False):
    draw_axes(c, x, y, w, h, y_min, y_max, percent=percent)
    n = len(categories)
    xs = [x + w * i / max(n - 1, 1) for i in range(n)]
    for idx, cat in enumerate(categories):
        c.setFillColor(MUTED)
        c.setFont("CN", 8.5)
        c.drawCentredString(xs[idx], y - 17, cat)
    for s_idx, item in enumerate(series):
        points = []
        for i, value in enumerate(item["values"]):
            yy = y + h * (value - y_min) / (y_max - y_min)
            points.append((xs[i], yy))
        c.setStrokeColor(item["color"])
        c.setLineWidth(2.4)
        for i in range(len(points) - 1):
            c.line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i, (px, py) in enumerate(points):
            if error_values:
                err = error_values[s_idx][i]
                e = h * err / (y_max - y_min)
                c.setStrokeColor(item["color"])
                c.setLineWidth(1)
                c.line(px, py - e, px, py + e)
                c.line(px - 4, py - e, px + 4, py - e)
                c.line(px - 4, py + e, px + 4, py + e)
            c.setFillColor(item["color"])
            c.circle(px, py, 4, fill=1, stroke=0)
            if value_labels:
                c.setFillColor(INK)
                c.setFont("EN", 7.5)
                label = f"{item['values'][i] * 100:.1f}%" if percent else f"{item['values'][i]:.2f}"
                c.drawCentredString(px, py + 8 + 9 * s_idx, label)
    lx = x + w - 180
    ly = y + h + 16
    for item in series:
        c.setStrokeColor(item["color"])
        c.setLineWidth(2.5)
        c.line(lx, ly, lx + 15, ly)
        c.setFillColor(INK)
        c.setFont("CN", 9)
        c.drawString(lx + 20, ly - 4, item["name"])
        lx += 90


def simple_table(c, data, x, y_top, widths, row_height=31,
                 header_fill=INK, font_size=10):
    table = Table(data, colWidths=widths, rowHeights=[row_height] * len(data))
    style = TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "CN"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("BACKGROUND", (0, 1), (-1, -1), white),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ])
    for row in range(1, len(data)):
        if row % 2 == 0:
            style.add("BACKGROUND", (0, row), (-1, row), HexColor("#FAFAFA"))
    table.setStyle(style)
    total_h = row_height * len(data)
    table.wrapOn(c, sum(widths), total_h)
    table.drawOn(c, x, y_top - total_h)


def load_data():
    runs = read_csv(EVIDENCE / "reports" / "per_seed.csv")
    position_rows = read_csv(EVIDENCE / "reports" / "position_summary.csv")
    stats = json.loads((EVIDENCE / "reports" / "dataset_statistics.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "data" / "beauty_sentence_t5_xl_manifest.json").read_text(encoding="utf-8"))

    by_q = {"rq_kmeans": [], "rqvae": []}
    for row in runs:
        by_q[row["quantizer"]].append(row)

    geometry = {"rq_kmeans": [], "rqvae": []}
    for q in geometry:
        for seed in (2024, 2025, 2026):
            name = f"{q}_seed{seed}_summary.json" if q == "rq_kmeans" else f"{q}_seed{seed}_geometry_summary.json"
            geometry[q].append(json.loads((EVIDENCE / "tokenizer_geometry" / name).read_text(encoding="utf-8")))

    positions = {}
    for row in position_rows:
        positions[(row["quantizer"], row["split"], int(row["position"]))] = {
            key: float(value) if key not in {"quantizer", "split", "role"} else value
            for key, value in row.items()
        }

    rqvae_training = {}
    for seed in (2024, 2025, 2026):
        rqvae_training[seed] = read_csv(EVIDENCE / "tokenizer_aux" / f"rqvae_seed{seed}" / "training.csv")
    rqvae_summaries = {
        seed: json.loads((EVIDENCE / "tokenizer_aux" / f"rqvae_seed{seed}" / "summary.json").read_text(encoding="utf-8"))
        for seed in (2024, 2025, 2026)
    }
    return runs, by_q, geometry, positions, rqvae_training, rqvae_summaries, stats, manifest


def main():
    register_fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    runs, by_q, geometry, positions, training, rq_summaries, stats, manifest = load_data()

    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("AR生成推荐第一阶段实验汇报")
    c.setAuthor("张婧")

    # Slide 1
    c.setFillColor(white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont("CN", 13)
    c.drawString(MARGIN, PAGE_H - 48, "生成式推荐 · Semantic ID 空间设计")
    c.setFillColor(INK)
    paragraph(c, "固定 3×256 结构下，<br/>Tokenizer 如何影响 AR Decoder？", MARGIN, 335, 720, 150, size=42, leading=54, bold=True)
    c.setFillColor(RQK)
    c.rect(MARGIN, 124, 94, 6, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont("CN", 14)
    c.drawString(MARGIN, 88, "RQ-KMeans vs. RQ-VAE · Amazon Beauty · 3 seeds")
    c.setFont("CN", 11)
    c.drawString(MARGIN, 56, "阶段性实验汇报｜张婧")
    c.showPage()

    # Slide 2
    header(c, "研究问题：好的语义表征，是否也更容易被 Decoder 生成？", 2)
    paragraph(c, "现有工作通常固定 SID 层数和码本大小，但这种设计同时影响两件事：", MARGIN, 416, 860, 46, size=17)
    panel(c, MARGIN, 155, 405, 215, PALE_BLUE)
    paragraph(c, "Encoder / Tokenizer 侧", MARGIN + 24, 340, 350, 38, size=20, bold=True)
    paragraph(c, "能否重构物品 embedding？<br/>能否保留局部邻域与空间结构？<br/>Semantic ID 是否均衡、低碰撞？", MARGIN + 24, 294, 345, 115, size=16, leading=29)
    panel(c, 513, 155, 405, 215, HexColor("#FFF4E5"))
    paragraph(c, "Decoder 侧", 537, 340, 350, 38, size=20, bold=True)
    paragraph(c, "每个位置是否容易预测？<br/>误差是否随位置累积？<br/>最终 HR / NDCG 是否更高？", 537, 294, 345, 115, size=16, leading=29)
    paragraph(c, "第一阶段策略：固定数据、SID 结构和 AR 模型，仅替换 Tokenizer。", MARGIN, 115, 860, 46, size=18, bold=True, color=RQK)
    footer(c, "研究设计来源：configs/ar_formal_common.yaml")
    c.showPage()

    # Slide 3
    header(c, "公平对照只改变 Tokenizer，其余训练协议保持一致", 3)
    labels = [
        ("文本信息", "Sentence-T5-XL\n12101 × 768"),
        ("Tokenizer", "RQ-KMeans\n或 RQ-VAE"),
        ("Semantic ID", "3 × 256\n+ 消歧后缀"),
        ("AR Decoder", "2层 Encoder\n4层 Decoder"),
        ("评估", "严格 item-level\nHR / NDCG"),
    ]
    start_x, box_w, gap, y, box_h = 45, 158, 28, 228, 150
    for i in range(len(labels) - 1):
        x1 = start_x + i * (box_w + gap) + box_w
        x2 = start_x + (i + 1) * (box_w + gap)
        c.setStrokeColor(RULE)
        c.setLineWidth(2)
        c.line(x1 + 4, y + box_h / 2, x2 - 5, y + box_h / 2)
        c.setFillColor(RULE)
        c.line(x2 - 12, y + box_h / 2 + 5, x2 - 5, y + box_h / 2)
        c.line(x2 - 12, y + box_h / 2 - 5, x2 - 5, y + box_h / 2)
    for i, (title, body) in enumerate(labels):
        x = start_x + i * (box_w + gap)
        panel(c, x, y, box_w, box_h, PANEL if i != 1 else PALE_BLUE, RULE)
        c.setFillColor(INK)
        c.setFont("CN", 17)
        c.drawString(x + 17, y + box_h - 34, title)
        paragraph(c, body.replace("\n", "<br/>"), x + 17, y + 93, box_w - 34, 68, size=13, color=MUTED, leading=22)
    metric_box(c, 94, 116, 170, 72, "3", "随机种子", RQK, "2024 / 2025 / 2026")
    metric_box(c, 294, 116, 170, 72, "105,668", "训练样本", INK, "滑动窗口")
    metric_box(c, 494, 116, 170, 72, "22,363", "验证 / 测试样本", INK, "leave-one-out")
    metric_box(c, 694, 116, 170, 72, "Beam 64", "自回归推理", RQVAE, "Top-10")
    footer(c, "数据来源：dataset_statistics.json；ar_formal_common.yaml")
    c.showPage()

    # Slide 4
    header(c, "正式实验规模与评估口径", 4)
    table_data = [
        ["模块", "设置", "控制目的"],
        ["数据", "Amazon Beauty；22,363 用户；198,502 交互", "固定数据与 leave-one-out 切分"],
        ["物品表征", "Sentence-T5-XL；768维；归一化", "两种 Tokenizer 使用同一份 embedding"],
        ["SID", "3个语义位置 × 256；另加消歧后缀", "语义容量和输出长度保持一致"],
        ["AR 模型", "2层 Encoder + 4层 Decoder；d=256", "下游生成器完全一致"],
        ["训练", "有效 batch=512；最多100 epoch；早停", "按验证集 NDCG@10 选 checkpoint"],
        ["评估", "严格 item-level；Beam=64", "完整 Item ID 一致才算命中"],
        ["指标", "HR@5/10、NDCG@5/10、legal、duplicate", "同时检查效果与生成合法性"],
    ]
    simple_table(c, table_data, MARGIN, 425, [115, 390, 365], row_height=39, font_size=10.5)
    paragraph(c, "注：第4位是 collision suffix，不应解释为第4层语义码。", MARGIN, 82, 860, 34, size=13, color=RED)
    footer(c, "数据来源：configs/*.yaml；dataset_statistics.json；embedding manifest")
    c.showPage()

    # Slide 5
    metrics = ["hit_rate@5", "hit_rate@10", "ndcg@5", "ndcg@10"]
    labels_m = ["HR@5", "HR@10", "NDCG@5", "NDCG@10"]
    rqk_vals = [mean([float(r[m]) for r in by_q["rq_kmeans"]]) for m in metrics]
    rqv_vals = [mean([float(r[m]) for r in by_q["rqvae"]]) for m in metrics]
    rel = [(a - b) / b for a, b in zip(rqk_vals, rqv_vals)]
    header(c, "RQ-KMeans 在四项推荐指标上均优于 RQ-VAE", 5)
    grouped_bar(c, 70, 135, 575, 280, labels_m, [
        {"name": "RQ-KMeans", "values": rqk_vals, "color": RQK},
        {"name": "RQ-VAE", "values": rqv_vals, "color": RQVAE},
    ], y_max=0.065, percent=True)
    metric_box(c, 690, 330, 210, 88, f"+{rel[1]*100:.1f}%", "HR@10 相对提升", RQK, "0.0569 vs 0.0520")
    metric_box(c, 690, 217, 210, 88, f"+{rel[3]*100:.1f}%", "NDCG@10 相对提升", RQK, "0.0302 vs 0.0274")
    metric_box(c, 690, 104, 210, 88, "≈100%", "Top-10 合法率", GREEN, "duplicate@10 = 0")
    paragraph(c, "结论：差异并非由非法 SID 或重复推荐造成。", 70, 95, 570, 34, size=14, bold=True)
    footer(c, "数据来源：reports/per_seed.csv；均值基于3个随机种子")
    c.showPage()

    # Slide 6
    header(c, "三个 seed 的提升方向一致，但样本数仍不足以强调显著性", 6)
    seeds = ["2024", "2025", "2026"]
    for idx, metric in enumerate(["hit_rate@10", "ndcg@10"]):
        x = 65 + idx * 445
        title = "HR@10" if idx == 0 else "NDCG@10"
        c.setFillColor(INK)
        c.setFont("ENB", 16)
        c.drawString(x, 425, title)
        a = [float(next(r for r in by_q["rq_kmeans"] if r["seed"] == s)[metric]) for s in seeds]
        b = [float(next(r for r in by_q["rqvae"] if r["seed"] == s)[metric]) for s in seeds]
        ymax = max(a + b) * 1.2
        line_chart(c, x, 145, 370, 240, seeds, [
            {"name": "RQ-KMeans", "values": a, "color": RQK},
            {"name": "RQ-VAE", "values": b, "color": RQVAE},
        ], 0, ymax, percent=True, value_labels=True)
    deltas = read_csv(EVIDENCE / "reports" / "paired_delta.csv")
    avg_hr = mean([float(r["hit_rate@10"]) for r in deltas])
    avg_nd = mean([float(r["ndcg@10"]) for r in deltas])
    paragraph(c, f"配对绝对提升：HR@10 +{avg_hr:.4f}；NDCG@10 +{avg_nd:.4f}。三个 seed 均为正。", 65, 102, 820, 38, size=15, bold=True, color=RQK)
    footer(c, "数据来源：reports/per_seed.csv；reports/paired_delta.csv")
    c.showPage()

    # Slide 7
    header(c, "RQ-VAE 重构更准，但局部邻域保持明显更弱", 7)
    geom_metrics = [
        ("reconstruction_cosine", "重构余弦", 0.0, 1.0, False),
        ("neighbor_recall_at_10", "Neighbor Recall@10", 0.0, 0.28, True),
        ("collision_rate", "SID 碰撞率", 0.0, 0.07, True),
    ]
    for idx, (key, label, ymin, ymax, pct) in enumerate(geom_metrics):
        x = 55 + idx * 300
        values_a = [float(g[key]) for g in geometry["rq_kmeans"]]
        values_b = [float(g[key]) for g in geometry["rqvae"]]
        c.setFillColor(INK)
        c.setFont("CN", 14)
        c.drawString(x, 430, label)
        grouped_bar(c, x + 5, 175, 245, 210, ["三种子均值"], [
            {"name": "RQ-KMeans", "values": [mean(values_a)], "color": RQK},
            {"name": "RQ-VAE", "values": [mean(values_b)], "color": RQVAE},
        ], ymax, percent=pct)
    paragraph(c, "RQ-VAE：点对点重构余弦更高；但局部邻域 Recall@10 低约 43%，且碰撞率跨 seed 更不稳定。", 55, 127, 850, 50, size=15, bold=True)
    paragraph(c, "这说明“重构误差更低”并不等价于“保留推荐所需的局部语义结构”。", 55, 83, 850, 35, size=15, color=RQK)
    footer(c, "数据来源：tokenizer_geometry/*.json；每项为3个随机种子均值")
    c.showPage()

    # Slide 8
    header(c, "RQ-VAE 的前两层码本利用不足，语义容量向后层集中", 8)
    cats = ["位置1", "位置2", "位置3"]
    util_a = [mean([g["per_position"][i]["utilization_rate"] for g in geometry["rq_kmeans"]]) for i in range(3)]
    util_b = [mean([g["per_position"][i]["utilization_rate"] for g in geometry["rqvae"]]) for i in range(3)]
    ent_a = [mean([g["per_position"][i]["normalized_entropy"] for g in geometry["rq_kmeans"]]) for i in range(3)]
    ent_b = [mean([g["per_position"][i]["normalized_entropy"] for g in geometry["rqvae"]]) for i in range(3)]
    c.setFillColor(INK); c.setFont("CN", 15); c.drawString(65, 428, "Code utilization")
    line_chart(c, 65, 155, 360, 230, cats, [
        {"name": "RQ-KMeans", "values": util_a, "color": RQK},
        {"name": "RQ-VAE", "values": util_b, "color": RQVAE},
    ], 0, 1.05, percent=True, value_labels=True)
    c.setFillColor(INK); c.setFont("CN", 15); c.drawString(535, 428, "Normalized entropy")
    line_chart(c, 535, 155, 360, 230, cats, [
        {"name": "RQ-KMeans", "values": ent_a, "color": RQK},
        {"name": "RQ-VAE", "values": ent_b, "color": RQVAE},
    ], 0, 1.05, percent=True, value_labels=True)
    paragraph(c, "RQ-VAE 第1层平均只使用约三成码字，第3层才接近满利用；层间难度分配与 RQ-KMeans 明显不同。", 65, 108, 830, 42, size=14.5, bold=True)
    footer(c, "数据来源：tokenizer_geometry/*.json 中 per_position；第1-3位均为语义码")
    c.showPage()

    # Slide 9
    header(c, "下游差异主要出现在第 2 / 3 个语义位置", 9)
    cats4 = ["语义1", "语义2", "语义3", "消歧后缀"]
    nll_a = [positions[("rq_kmeans", "test", i)]["nll_mean"] for i in range(1, 5)]
    nll_b = [positions[("rqvae", "test", i)]["nll_mean"] for i in range(1, 5)]
    nll_sa = [positions[("rq_kmeans", "test", i)]["nll_std"] for i in range(1, 5)]
    nll_sb = [positions[("rqvae", "test", i)]["nll_std"] for i in range(1, 5)]
    line_chart(c, 65, 137, 575, 280, cats4, [
        {"name": "RQ-KMeans", "values": nll_a, "color": RQK},
        {"name": "RQ-VAE", "values": nll_b, "color": RQVAE},
    ], 0, 5.5, error_values=[nll_sa, nll_sb], value_labels=True)
    metric_box(c, 685, 323, 220, 91, "Pos 1", "RQ-VAE 更容易", RQVAE, "NLL 3.96 vs 4.89")
    metric_box(c, 685, 211, 220, 91, "Pos 2", "RQ-KMeans 更容易", RQK, "NLL 3.84 vs 4.31")
    metric_box(c, 685, 99, 220, 91, "Pos 3", "RQ-KMeans 优势最大", RQK, "NLL 0.59 vs 1.43")
    paragraph(c, "消歧后缀几乎无差异，最终推荐差距来自语义位置，而非后缀机制。", 65, 92, 575, 35, size=14, bold=True)
    footer(c, "数据来源：reports/position_summary.csv；测试集均值 +/- 标准差")
    c.showPage()

    # Slide 10
    header(c, "Token accuracy 进一步确认：RQ-KMeans 在后两层更容易解码", 10)
    acc_a = [positions[("rq_kmeans", "test", i)]["accuracy_mean"] for i in range(1, 5)]
    acc_b = [positions[("rqvae", "test", i)]["accuracy_mean"] for i in range(1, 5)]
    pre_a = [positions[("rq_kmeans", "test", i)]["prefix_accuracy_mean"] for i in range(1, 5)]
    pre_b = [positions[("rqvae", "test", i)]["prefix_accuracy_mean"] for i in range(1, 5)]
    c.setFillColor(INK); c.setFont("CN", 15); c.drawString(55, 425, "逐位置 Token Accuracy")
    line_chart(c, 55, 155, 380, 225, cats4, [
        {"name": "RQ-KMeans", "values": acc_a, "color": RQK},
        {"name": "RQ-VAE", "values": acc_b, "color": RQVAE},
    ], 0, 1.05, percent=True, value_labels=True)
    c.setFillColor(INK); c.setFont("CN", 15); c.drawString(515, 425, "Prefix Accuracy（累计正确）")
    line_chart(c, 515, 155, 380, 225, cats4, [
        {"name": "RQ-KMeans", "values": pre_a, "color": RQK},
        {"name": "RQ-VAE", "values": pre_b, "color": RQVAE},
    ], 0, 0.12, percent=True, value_labels=True)
    paragraph(c, "Teacher forcing 下后部 token 很容易，但 prefix accuracy 很低：前部错误会沿自回归链条累积。", 55, 105, 840, 38, size=14.5, bold=True)
    footer(c, "数据来源：reports/position_summary.csv；测试集三种子均值")
    c.showPage()

    # Slide 11
    header(c, "RQ-VAE 最佳健康 checkpoint 集中在 4k-5k step", 11)
    colors = [RQK, RQVAE, GREEN]
    series = []
    y_values = []
    for color, seed in zip(colors, (2024, 2025, 2026)):
        rows = training[seed]
        vals = [float(r["val_total"]) for r in rows]
        y_values.extend(vals)
        series.append({"name": str(seed), "values": vals, "color": color})
    steps = [str(int(float(r["step"])) // 1000) + "k" for r in training[2024]]
    # Reduce x labels to 10 points while retaining full curves.
    cats_train = [label if i % 6 == 0 or i == len(steps) - 1 else "" for i, label in enumerate(steps)]
    ymin = min(y_values) * 0.95
    ymax = max(y_values) * 1.03
    line_chart(c, 60, 145, 590, 275, cats_train, series, ymin, ymax)
    summary_data = [["Seed", "Best step", "Best val loss", "Tokenizer耗时"]]
    for seed in (2024, 2025, 2026):
        s = rq_summaries[seed]
        summary_data.append([str(seed), f"{s['best_step']//1000}k", f"{s['best_validation_total_loss']:.4f}", f"{s['elapsed_seconds']/60:.1f} min"])
    simple_table(c, summary_data, 670, 410, [44, 62, 76, 78], row_height=38, font_size=8.7)
    paragraph(c, "总 Loss 后期抬升主要来自量化项；正式 Semantic ID 使用的是满足利用率与唯一性约束的最佳健康 checkpoint，而非 60k 最终状态。", 670, 220, 250, 120, size=12.2, leading=21, bold=True)
    footer(c, "数据来源：tokenizer_aux/rqvae_seed*/training.csv 与 summary.json")
    c.showPage()

    # Slide 12
    header(c, "阶段结论：Decoder 友好性不能由重构误差单独判断", 12)
    conclusions = [
        ("已确认", "RQ-KMeans 在当前 3×256 AR 设置下稳定优于 RQ-VAE，四项推荐指标相对提升约 9%-11%。", RQK),
        ("关键机制线索", "RQ-VAE 点对点重构更好，但局部邻域保持更弱，前两层码本利用不足，第2/3位更难解码。", RQVAE),
        ("不能过度推断", "结果只覆盖一个数据集、一种 SID 结构和一个 AR Decoder；尚不能声称 RQ-KMeans 普遍优于 RQ-VAE。", MUTED),
        ("研究含义", "Semantic Space 需要同时考虑 Encoder 表征质量和 Decoder 可学习性，而不是只优化重构。", GREEN),
    ]
    positions_xy = [(45, 302), (495, 302), (45, 120), (495, 120)]
    for (title, body, color), (x, y) in zip(conclusions, positions_xy):
        c.setStrokeColor(color)
        c.setLineWidth(5)
        c.line(x, y, x, y + 135)
        c.setFillColor(INK)
        c.setFont("CN", 18)
        c.drawString(x + 20, y + 108, title)
        paragraph(c, body, x + 20, y + 88, 375, 82, size=13, color=MUTED, leading=22)
    c.setFillColor(MUTED)
    c.setFont("CN", 9)
    c.drawString(MARGIN, 76, "外部参照：GRID 在 Beauty 上也报告了 RQ-KMeans 优于 RQ-VAE 的定性趋势。")
    c.setFont("EN", 8)
    c.drawString(MARGIN, 58, "https://arxiv.org/abs/2507.22224")
    footer(c, "本页结论基于当前六次正式实验；外部参照为 GRID (arXiv:2507.22224)")
    c.showPage()

    # Slide 13
    header(c, "下一阶段：从“Tokenizer 对照”推进到“模型如何适应 Semantic Space”", 13)
    steps = [
        ("01", "完成 AR 阶段汇总", "固定 3×256\nRQ-KMeans vs RQ-VAE", True),
        ("02", "复现 DiffGRM", "记录逐位置 Loss\n与实际解码顺序", False),
        ("03", "扩散模型换 Tokenizer", "沿用同一 embedding\n和严格 item-level 评估", False),
        ("04", "SID 深宽消融", "2/3/4/5 × 256\n3 × 128/256/512", False),
        ("05", "决定设计偏向", "Encoder 表征保持\nvs Decoder 解码适配", False),
    ]
    start_x, y, bw, gap = 45, 215, 155, 28
    for i in range(len(steps) - 1):
        x1 = start_x + i * (bw + gap) + bw
        x2 = start_x + (i + 1) * (bw + gap)
        c.setStrokeColor(RULE); c.setLineWidth(2)
        c.line(x1 + 4, y + 75, x2 - 5, y + 75)
    for i, (num, title, body, done) in enumerate(steps):
        x = start_x + i * (bw + gap)
        fill = PALE_BLUE if done else PANEL
        panel(c, x, y, bw, 180, fill, RQK if done else None)
        c.setFillColor(RQK if done else MUTED)
        c.setFont("ENB", 24)
        c.drawString(x + 15, y + 142, num)
        c.setFillColor(INK); c.setFont("CN", 15)
        c.drawString(x + 15, y + 108, title)
        paragraph(c, body.replace("\n", "<br/>"), x + 15, y + 91, bw - 30, 68, size=11.5, color=MUTED, leading=20)
    paragraph(c, "近期优先级：先复现 DiffGRM，再在 AR 与扩散模型中使用同一套 Tokenizer 做交叉比较。", MARGIN, 145, 860, 42, size=16, bold=True, color=RQK)
    paragraph(c, "RL 阶段作为可选扩展，不应抢在扩散基线和 Semantic Space 诊断之前。", MARGIN, 103, 860, 34, size=14, color=MUTED)
    footer(c, "下一步设计依据：学长任务清单；GRID 深宽消融设置")
    c.showPage()

    c.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
