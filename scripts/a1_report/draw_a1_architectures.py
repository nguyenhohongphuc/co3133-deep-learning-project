"""Draw implementation-specific A1 architectures as SVG and high-resolution PNG.

Run: python scripts/a1_report/draw_a1_architectures.py
Requires Pillow; reuses the report pipeline's drawing primitives.
The thumbnail is a real training image; nodes and feature maps are schematic.
"""
from __future__ import annotations

import base64
import gzip
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from draw_a1_pipeline import Diagram, SCALE, INK, MUTED, BLUE, TEAL, LINE


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/a1/assets/architectures"
PURPLE = "#7356A4"
SOFT_BLUE = "#F0F5FB"
SOFT_TEAL = "#EFF8F6"


class Architecture(Diagram):
    def __init__(self, title, parameters, height):
        super().__init__()
        self.height = height
        self.image = Image.new("RGB", (1200 * SCALE, height * SCALE), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
            f'viewBox="0 0 1200 {height}">',
            f'<title>{title}</title>',
            f'<rect width="1200" height="{height}" fill="white"/>',
        ]
        self.text(25, 28, title, size=29, bold=True)
        self.text(1175, 28, parameters, size=21, color=MUTED, anchor="right")

    def line(self, start, end, color="#C7D4E1", width=1.2):
        x1, y1 = start
        x2, y2 = end
        self.draw.line([(x1 * SCALE, y1 * SCALE), (x2 * SCALE, y2 * SCALE)],
                       fill=color, width=max(1, round(width * SCALE)))
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                        f'stroke="{color}" stroke-width="{width}"/>')

    def circle(self, x, y, radius=10, color=BLUE, fill="white"):
        self.draw.ellipse(((x-radius)*SCALE, (y-radius)*SCALE,
                           (x+radius)*SCALE, (y+radius)*SCALE),
                          fill=fill, outline=color, width=2*SCALE)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" '
                        f'fill="{fill}" stroke="{color}" stroke-width="2"/>')

    def thumbnail(self, x, y, side, patches=False):
        raw = ROOT / "data/FashionMNIST/raw"
        labels = gzip.decompress((raw / "train-labels-idx1-ubyte.gz").read_bytes())[8:]
        index = labels.index(0)
        pixels = gzip.decompress((raw / "train-images-idx3-ubyte.gz").read_bytes())
        tile = Image.frombytes("L", (28, 28), pixels[16+784*index:16+784*(index+1)])
        enlarged = tile.resize((side*SCALE, side*SCALE), Image.Resampling.NEAREST)
        self.image.paste(enlarged.convert("RGB"), (x*SCALE, y*SCALE))
        buf = BytesIO()
        tile.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        self.svg.append(f'<image x="{x}" y="{y}" width="{side}" height="{side}" '
                        f'style="image-rendering:pixelated" href="data:image/png;base64,{encoded}"/>')
        if patches:
            for k in range(1, 4):
                p = side * k / 4
                self.line((x+p, y), (x+p, y+side), color="#66D7CC", width=1.6)
                self.line((x, y+p), (x+side, y+p), color="#66D7CC", width=1.6)

    def feature_stack(self, x, y, side, color=BLUE):
        for offset in (18, 9, 0):
            self.box(x+offset, y-offset, side, side, fill=SOFT_BLUE if color==BLUE else SOFT_TEAL,
                     stroke=color, radius=4)
        for k in range(1, 5):
            p = side*k/5
            self.line((x+p,y), (x+p,y+side))
            self.line((x,y+p), (x+side,y+p))

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        self.image.save(OUT / f"{name}.png", dpi=(300,300))
        (OUT / f"{name}.svg").write_text("\n".join(self.svg+["</svg>"])+"\n", encoding="utf-8")


def linear():
    d = Architecture("Linear / softmax classifier", "7,850 parameters", 330)
    d.thumbnail(45, 105, 130)
    d.text(110, 263, "1 × 28 × 28", size=22, anchor="center")
    d.arrow([(190,170),(245,170)])
    d.box(260,100,170,140,fill=SOFT_BLUE)
    for i in range(8):
        d.box(281,112+i*14,27,9,fill="#D0DEEE",radius=2)
    d.text(365,145,"Flatten",size=24,bold=True,anchor="center",color=BLUE)
    d.text(365,184,"784",size=25,anchor="center")
    d.arrow([(442,170),(500,170)])
    d.box(515,86,310,170,fill=SOFT_BLUE)
    d.text(670,113,"Linear 784 → 10",size=25,bold=True,anchor="center",color=BLUE)
    for a in (145,177,209):
        for b in (145,177,209):d.line((570,a),(770,b))
    for y in (145,177,209):
        d.circle(570,y,color=BLUE)
        d.circle(770,y,color=TEAL)
    d.arrow([(840,170),(905,170)])
    d.box(920,100,245,140,fill=SOFT_TEAL,stroke="#BCDCD7")
    d.text(1042,139,"10 class logits",size=25,bold=True,anchor="center",color=TEAL)
    d.text(1042,183,"[B, 10]",size=24,anchor="center")
    d.text(600,309,"One affine layer · no hidden activation · logits passed directly to cross-entropy",
           size=22,color=MUTED,anchor="center")
    d.save("a1_linear")


def mlp():
    d = Architecture("Multilayer perceptron", "235,146 parameters", 395)
    columns=[(165,784,"Input"),(450,256,"Hidden 1"),(735,128,"Hidden 2"),(1065,10,"Logits")]
    ys=[119,158,197,236,275]
    for (x1,_,_),(x2,_,_) in zip(columns,columns[1:]):
        for a in ys:
            for b in ys:d.line((x1+12,a),(x2-12,b),width=.85)
    for x,n,label in columns:
        d.text(x,74,label,size=25,bold=True,anchor="center",color=TEAL if n==10 else BLUE)
        for y in ys:d.circle(x,y,12,color=TEAL if n==10 else BLUE,fill=SOFT_TEAL if n==10 else SOFT_BLUE)
        d.text(x,312,str(n)+(" units" if n!=10 else " outputs"),size=24,anchor="center")
    d.text(450,343,"ReLU + dropout 0.2",size=22,anchor="center",color=PURPLE)
    d.text(735,343,"ReLU + dropout 0.2",size=22,anchor="center",color=PURPLE)
    d.text(600,379,"Flattened pixels → 256 → 128 → 10  |  Selected neurons shown; dropout is active only during training",
           size=19,color=MUTED,anchor="center")
    d.save("a1_mlp")


def cnn():
    d = Architecture("Convolutional neural network", "50,378 parameters", 410)
    d.thumbnail(35,123,135)
    d.text(102,284,"[B, 1, 28, 28]",size=22,anchor="center")
    d.arrow([(185,190),(247,190)])
    d.text(325,85,"Block 1",size=26,bold=True,anchor="center",color=BLUE)
    d.feature_stack(275,139,105,BLUE)
    d.text(327,278,"32 × 14 × 14",size=23,anchor="center",color=BLUE)
    d.text(327,310,"Conv 1→32 · BN · ReLU",size=20,anchor="center")
    d.text(327,337,"MaxPool 2×2",size=20,anchor="center",color=MUTED)
    d.arrow([(413,190),(485,190)])
    d.text(570,85,"Block 2",size=26,bold=True,anchor="center",color=TEAL)
    d.feature_stack(525,159,85,TEAL)
    d.text(569,278,"64 × 7 × 7",size=23,anchor="center",color=TEAL)
    d.text(569,310,"Conv 32→64 · BN · ReLU",size=20,anchor="center")
    d.text(569,337,"MaxPool 2×2",size=20,anchor="center",color=MUTED)
    d.arrow([(646,190),(727,190)])
    d.box(745,108,420,173,fill=SOFT_TEAL,stroke="#BCDCD7")
    d.text(955,137,"Classification head",size=26,bold=True,anchor="center",color=TEAL)
    for y,line in [(175,"Flatten: 64 × 7 × 7 → 3,136"),(210,"Dropout 0.25"),(245,"Linear 3,136 → 10")]:
        d.text(955,y,line,size=23,anchor="center")
    d.text(955,316,"[B, 10] logits",size=24,anchor="center",color=TEAL)
    d.text(600,389,"3×3 convolutions, stride 1, padding 1 · pooling halves each spatial dimension · feature maps are schematic",
           size=19,color=MUTED,anchor="center")
    d.save("a1_cnn")


def lstm():
    d = Architecture("LSTM: image rows as a sequence", "82,186 parameters · main row variant", 465)
    d.text(25,69,"28 timesteps · 28 pixel values per row · hidden size 128 · one unidirectional layer",size=22,color=MUTED)
    d.thumbnail(35,132,130)
    for x,title in [(290,"Row 1"),(590,"Row 2"),(920,"Row 28")]:
        d.box(x,113,180,64,fill=SOFT_BLUE)
        d.text(x+90,134,title,size=24,bold=True,anchor="center",color=BLUE)
        d.text(x+90,159,"[B, 28]",size=20,anchor="center")
        d.arrow([(x+90,181),(x+90,227)])
        d.box(x-8,232,196,80,fill=SOFT_TEAL,stroke="#BCDCD7")
        d.text(x+90,257,"LSTM cell",size=25,bold=True,anchor="center",color=TEAL)
        d.text(x+90,287,"hidden: 128",size=22,anchor="center")
    d.arrow([(177,146),(278,146)])
    d.text(101,291,"Input image",size=22,anchor="center")
    d.arrow([(486,272),(570,272)],color=TEAL)
    d.text(526,251,"h_1, c_1",size=19,color=TEAL,anchor="center")
    d.arrow([(786,272),(900,272)],color=TEAL,dashed=True)
    d.text(841,250,"…",size=30,color=TEAL,anchor="center")
    d.text(200,328,"h_0 = c_0 = 0",size=20,color=MUTED,anchor="center")
    d.arrow([(207,272),(277,272)],color=TEAL)
    d.arrow([(1010,317),(1010,359)],color=TEAL)
    d.box(742,365,430,75,fill=SOFT_TEAL,stroke="#BCDCD7")
    d.text(957,387,"Final h_28 → linear 128 → 10",size=23,bold=True,anchor="center",color=TEAL)
    d.text(957,419,"[B, 10] logits",size=22,anchor="center")
    d.text(310,371,"The same LSTM weights are reused",size=22,color=MUTED,anchor="center")
    d.text(310,400,"at every timestep; only the final h is classified.",size=21,color=MUTED,anchor="center")
    d.save("a1_lstm")


def transformer():
    d = Architecture("Patch Transformer encoder", "540,170 parameters", 555)
    d.thumbnail(25,97,120,patches=True)
    d.text(85,244,"7×7 patches",size=20,color=MUTED,anchor="center")
    d.arrow([(158,156),(205,156)])
    d.box(220,100,225,110,fill=SOFT_BLUE)
    d.text(332,128,"16 patch tokens",size=24,bold=True,anchor="center",color=BLUE)
    d.text(332,168,"[B, 16, 49]",size=23,anchor="center")
    d.arrow([(457,156),(497,156)])
    d.box(510,100,250,110,fill=SOFT_BLUE)
    d.text(635,128,"Linear 49 → 128",size=24,bold=True,anchor="center",color=BLUE)
    d.text(635,168,"[B, 16, 128]",size=23,anchor="center")
    d.arrow([(774,156),(814,156)])
    d.box(827,100,348,110,fill="#F5F1FA",stroke="#D8CEE7")
    d.text(1001,126,"Add CLS + positions",size=24,bold=True,anchor="center",color=PURPLE)
    d.text(1001,166,"Dropout 0.1 · [B, 17, 128]",size=22,anchor="center")
    d.arrow([(1001,214),(1001,255),(530,255),(530,293)],color=PURPLE)
    d.box(160,300,630,213,fill=SOFT_BLUE,stroke="#C4D3E2",radius=17)
    d.text(190,328,"Encoder block × 4",size=27,bold=True,color=BLUE)
    d.box(183,354,583,57,fill="white")
    d.text(474,383,"4-head self-attention → residual + LayerNorm",size=22,anchor="center")
    d.arrow([(474,414),(474,432)],color=BLUE)
    d.box(183,436,583,57,fill="white")
    d.text(474,465,"FFN 128→256→128 (GELU) → residual + LayerNorm",size=21,anchor="center")
    d.arrow([(798,405),(852,405)],color=TEAL)
    d.box(864,309,311,198,fill=SOFT_TEAL,stroke="#BCDCD7")
    d.text(1019,340,"CLS output",size=26,bold=True,anchor="center",color=TEAL)
    for y,line in [(380,"LayerNorm (128)"),(420,"Linear 128 → 10"),(466,"[B, 10] logits")]:
        d.text(1019,y,line,size=23,anchor="center")
    d.text(600,540,"4 heads · d_model = 128 · d_head = 32 · learned positional embeddings · dropout = 0.1",
           size=21,color=MUTED,anchor="center")
    d.save("a1_transformer")


def main():
    for draw in (linear,mlp,cnn,lstm,transformer):draw()
    print(f"Created five architecture illustrations (PNG + SVG) in {OUT}")


if __name__ == "__main__":main()
