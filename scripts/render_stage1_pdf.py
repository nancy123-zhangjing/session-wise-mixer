from pathlib import Path

import fitz
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output" / "pdf" / "AR_GRM_3x256_stage1_report.pdf"
OUT = ROOT / "tmp" / "pdfs" / "stage1_report" / "rendered"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    rendered = []
    for index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        path = OUT / f"page-{index + 1:02d}.png"
        pix.save(path)
        rendered.append(path)

    thumbs = []
    for index, path in enumerate(rendered, start=1):
        image = Image.open(path).convert("RGB")
        image.thumbnail((480, 270))
        canvas = Image.new("RGB", (500, 305), "white")
        canvas.paste(image, ((500 - image.width) // 2, 10))
        ImageDraw.Draw(canvas).text((12, 282), f"{index:02d}", fill="#182330")
        thumbs.append(canvas)

    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    montage = Image.new("RGB", (columns * 500, rows * 305), "#E8EEF3")
    for index, thumb in enumerate(thumbs):
        montage.paste(thumb, ((index % columns) * 500, (index // columns) * 305))
    montage_path = OUT.parent / "montage.png"
    montage.save(montage_path)
    print(f"pages={len(rendered)}")
    print(montage_path)


if __name__ == "__main__":
    main()
