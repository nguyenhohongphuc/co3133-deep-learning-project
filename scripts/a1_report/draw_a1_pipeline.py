"""Draw the A1 report pipeline as editable SVG and a high-resolution PNG.

Run from any directory: python scripts/a1_report/draw_a1_pipeline.py
Requires Pillow. This script does not train or evaluate a model.
"""
from __future__ import annotations

from html import escape
from math import hypot
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/a1/assets/pipeline"
WIDTH, HEIGHT, SCALE = 1200, 770, 3
INK = "#182D45"
MUTED = "#526578"
BLUE = "#254F80"
TEAL = "#08776F"
LINE = "#8092A4"


class Diagram:
    """Keep the vector and raster exports on the same coordinate system."""

    def __init__(self):
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            '<title>Shared Fashion-MNIST training and evaluation pipeline</title>',
            '<desc>Shared preprocessing feeds training and validation on the left '
            'and held-out test evaluation on the right. The best validation '
            'checkpoint is transferred to test inference.</desc>',
            '<rect width="1200" height="770" fill="white"/>',
        ]
        self.fonts = {}

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            candidates = [
                Path("/System/Library/Fonts/Supplemental")
                / ("Arial Bold.ttf" if bold else "Arial.ttf"),
                Path("/usr/share/fonts/truetype/dejavu")
                / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
            ]
            for path in candidates:
                if path.exists():
                    self.fonts[key] = ImageFont.truetype(str(path), round(size * SCALE))
                    break
            else:
                self.fonts[key] = ImageFont.load_default(size=round(size * SCALE))
        return self.fonts[key]

    def text(self, x, y, value, size=22, color=INK, bold=False, anchor="left", rotation=0):
        font = self.font(size, bold)
        svg_anchor = {"left": "start", "center": "middle", "right": "end"}[anchor]
        transform = f' transform="rotate({rotation} {x} {y})"' if rotation else ""
        self.svg.append(
            f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" '
            f'fill="{color}" text-anchor="{svg_anchor}" '
            f'dominant-baseline="central"{transform}>{escape(value)}</text>'
        )
        if rotation:
            bounds = font.getbbox(value)
            layer = Image.new("RGBA", (bounds[2] + 16 * SCALE, (size + 16) * SCALE))
            ImageDraw.Draw(layer).text(
                (8 * SCALE, 8 * SCALE - bounds[1]), value, font=font, fill=color
            )
            layer = layer.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
            self.image.paste(
                layer,
                (round(x * SCALE - layer.width / 2), round(y * SCALE - layer.height / 2)),
                layer,
            )
        else:
            self.draw.text(
                (x * SCALE, y * SCALE), value, font=font, fill=color,
                anchor={"left": "lm", "center": "mm", "right": "rm"}[anchor],
            )

    def box(self, x, y, width, height, fill="white", stroke="#CED9E3", radius=13):
        self.draw.rounded_rectangle(
            (x * SCALE, y * SCALE, (x + width) * SCALE, (y + height) * SCALE),
            radius=radius * SCALE, fill=fill, outline=stroke, width=round(1.4 * SCALE),
        )
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        )

    def arrow(self, points, color=LINE, dashed=False, width=2.6):
        coordinates = " ".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="8 6"' if dashed else ""
        self.svg.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dash}/>'
        )
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            length = hypot(x2 - x1, y2 - y1)
            if dashed:
                for start in range(0, int(length), 14):
                    end = min(start + 8, length)
                    segment = [
                        ((x1 + (x2 - x1) * t / length) * SCALE,
                         (y1 + (y2 - y1) * t / length) * SCALE)
                        for t in (start, end)
                    ]
                    self.draw.line(segment, fill=color, width=round(width * SCALE))
            else:
                self.draw.line(
                    [(x1 * SCALE, y1 * SCALE), (x2 * SCALE, y2 * SCALE)],
                    fill=color, width=round(width * SCALE),
                )
        (x1, y1), (x2, y2) = points[-2:]
        length = hypot(x2 - x1, y2 - y1)
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        triangle = [(x2, y2), (x2 - 10 * dx + 4.8 * dy, y2 - 10 * dy - 4.8 * dx),
                    (x2 - 10 * dx - 4.8 * dy, y2 - 10 * dy + 4.8 * dx)]
        self.draw.polygon([(x * SCALE, y * SCALE) for x, y in triangle], fill=color)
        self.svg.append(
            '<polygon points="' + " ".join(f"{x},{y}" for x, y in triangle)
            + f'" fill="{color}"/>'
        )

    def card(self, x, y, width, title, lines, color, height=80):
        self.box(x, y, width, height, stroke="#C4D3E2" if color == BLUE else "#BCDCD7")
        self.text(x + 18, y + 23, title, size=24, bold=True, color=color)
        for i, line in enumerate(lines):
            self.text(x + 18, y + 52 + i * 24, line, size=21, color=MUTED)

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        self.image.save(OUT / "a1_overall_pipeline.png", dpi=(300, 300))
        (OUT / "a1_overall_pipeline.svg").write_text(
            "\n".join(self.svg + ["</svg>"]) + "\n", encoding="utf-8"
        )


def main():
    d = Diagram()
    d.text(25, 29, "Shared data preparation", size=27, bold=True)
    d.text(1175, 29, "FIVE ARCHITECTURES  /  ONE PROTOCOL", size=17, color=MUTED, anchor="right")

    for x in (25, 425, 825):
        d.box(x, 70, 350, 110, fill="#F7F9FC")
    top_cards = [
        (25, "Fashion-MNIST", ["70,000 grayscale images", "10 classes · 28 × 28 pixels"]),
        (425, "Stratified Split", ["54,000 train · 6,000 validation", "10,000 official test · seed 42"]),
        (825, "Shared preprocessing", ["ToTensor + Normalize", "Train / validation / test loaders"]),
    ]
    for x, title, details in top_cards:
        d.text(x + 18, 98, title, size=25, bold=True)
        d.text(x + 18, 132, details[0], size=21, color=MUTED)
        d.text(x + 18, 158, details[1], size=21, color=MUTED)
    d.arrow([(380, 125), (415, 125)])
    d.arrow([(780, 125), (815, 125)])

    d.box(25, 255, 550, 455, fill="#F0F5FB", stroke="#D4E0ED", radius=20)
    d.box(625, 255, 550, 455, fill="#EFF8F6", stroke="#CEE5E0", radius=20)
    d.arrow([(1000, 182), (1000, 213), (300, 213), (300, 250)], color=BLUE)
    d.arrow([(1000, 213), (900, 213), (900, 250)], color=TEAL)
    d.text(55, 285, "TRAIN & VALIDATE", size=25, bold=True, color=BLUE)
    d.text(55, 312, "Training and validation loaders", size=20, color=MUTED)
    d.text(655, 285, "FINAL TEST EVALUATION", size=25, bold=True, color=TEAL)
    d.text(655, 312, "Held-out test loader", size=20, color=MUTED)

    d.card(115, 340, 430, "Train forward + loss", ["Model logits → cross-entropy"], BLUE)
    d.card(115, 450, 430, "Backpropagation + Adam", ["Update model parameters"], BLUE)
    d.card(115, 560, 430, "Validation + model selection",
           ["No gradients; monitor macro-F1", "Save best.pt on improvement"], BLUE, height=100)
    d.arrow([(330, 423), (330, 446)], color=BLUE)
    d.arrow([(330, 533), (330, 556)], color=BLUE)
    d.arrow([(112, 610), (72, 610), (72, 380), (111, 380)], color=BLUE, dashed=True, width=2.1)
    d.text(50, 493, "next epoch", size=18, color=BLUE, anchor="center", rotation=-90)
    d.text(325, 687, "Repeat until patience or epoch limit", size=19, color=MUTED, anchor="center")

    d.card(655, 340, 490, "Held-out test batches", ["Same normalization and tensor shape"], TEAL)
    d.card(655, 450, 490, "Restore checkpoint + infer", ["Saved config + best.pt → logits"], TEAL)
    d.card(655, 560, 490, "Predictions + evaluation",
           ["Argmax → accuracy / macro-F1", "Timing, confusion matrix, examples"], TEAL, height=100)
    d.arrow([(900, 423), (900, 446)], color=TEAL)
    d.arrow([(900, 533), (900, 556)], color=TEAL)
    d.arrow([(548, 610), (599, 610), (599, 490), (651, 490)], color=TEAL, width=2.8)
    d.text(585, 551, "best.pt", size=18, color=TEAL, anchor="center", rotation=-90)
    d.text(900, 687, "Evaluate after checkpoint selection", size=19, color=MUTED, anchor="center")
    d.text(600, 744, "Shared model interface: images [B, 1, 28, 28] → logits [B, 10]",
           size=22, color=INK, anchor="center")
    d.save()
    print(f"Created {OUT / 'a1_overall_pipeline.svg'}")
    print(f"Created {OUT / 'a1_overall_pipeline.png'}")


if __name__ == "__main__":
    main()
