"""Export the A3 proposal Markdown to a paginated PDF using ReportLab.

Run from the repository root: python scripts/a3_export_proposal.py
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re

from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]


def inline(value: str) -> str:
    value = escape(value)
    value = re.sub(r"\[([^]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#244D70">\1</link>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r'<font name="ProposalMono" size="8.8">\1</font>', value)
    return value


def register_fonts() -> None:
    directory = Path("/System/Library/Fonts/Supplemental")
    if (directory / "Arial.ttf").exists():
        for name, filename in (("Proposal", "Arial.ttf"), ("Proposal-Bold", "Arial Bold.ttf"),
                               ("Proposal-Italic", "Arial Italic.ttf"), ("Proposal-BoldItalic", "Arial Bold Italic.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
        pdfmetrics.registerFontFamily("Proposal", normal="Proposal", bold="Proposal-Bold",
                                      italic="Proposal-Italic", boldItalic="Proposal-BoldItalic")
        pdfmetrics.registerFont(TTFont("ProposalMono", str(directory / "Courier New.ttf")))
    else:
        for name, filename in (("Proposal", "DejaVuSans.ttf"), ("Proposal-Bold", "DejaVuSans-Bold.ttf"),
                               ("Proposal-Italic", "DejaVuSans-Oblique.ttf"), ("Proposal-BoldItalic", "DejaVuSans-BoldOblique.ttf")):
            path = Path("/usr/share/fonts/truetype/dejavu") / filename
            if not path.exists():
                raise RuntimeError("Install Arial or DejaVu Sans to reproduce the proposal's Unicode text")
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily("Proposal", normal="Proposal", bold="Proposal-Bold",
                                      italic="Proposal-Italic", boldItalic="Proposal-BoldItalic")
        pdfmetrics.registerFont(TTFont("ProposalMono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))


def footer(canvas, document) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B8BFC5"))
    canvas.line(48, height - 35, width - 48, height - 35)
    canvas.setFont("Proposal", 8)
    canvas.setFillColor(colors.HexColor("#3A4650"))
    canvas.drawString(48, height - 26, "HCMUT | CO3133 | Group NNs | A3 M1 Dataset Proposal")
    canvas.drawString(48, 25, "18 November 2026 | Awaiting instructor approval")
    canvas.drawRightString(width - 48, 25, f"Page {document.page}")
    canvas.restoreState()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/a3/a3_proposal.md")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/a3/a3_proposal.pdf")
    args = parser.parse_args()
    register_fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ProposalBody", fontName="Proposal", fontSize=10.2, leading=13.5,
                              spaceAfter=6, textColor=colors.black))
    styles.add(ParagraphStyle("ProposalTitle", fontName="Proposal-Bold", fontSize=22, leading=27,
                              spaceBefore=4, spaceAfter=13))
    styles.add(ParagraphStyle("ProposalHeading", fontName="Proposal-Bold", fontSize=13, leading=17,
                              spaceBefore=9, spaceAfter=8, keepWithNext=True))
    styles.add(ParagraphStyle("ProposalCell", fontName="Proposal", fontSize=8.8, leading=11.6))
    styles.add(ParagraphStyle("ProposalCaption", fontName="Proposal-Italic", fontSize=8.4, leading=11,
                              spaceBefore=4, spaceAfter=10, alignment=TA_CENTER))
    styles.add(ParagraphStyle("ProposalCode", fontName="ProposalMono", fontSize=8.8, leading=11.4,
                              spaceAfter=8, borderPadding=7, backColor=colors.HexColor("#F2F4F5")))
    width = A4[0] - 96
    story = []
    blocks = re.split(r"\n\s*\n", args.source.read_text(encoding="utf-8").strip())
    for block in blocks:
        if block == "<!-- pagebreak -->":
            story.append(PageBreak())
        elif block.startswith("# "):
            story.append(Paragraph(inline(block[2:]), styles["ProposalTitle"]))
        elif block.startswith("## "):
            story.append(Paragraph(inline(block[3:]), styles["ProposalHeading"]))
        elif block.startswith("| "):
            lines = [line for line in block.splitlines() if not re.match(r"^\|[\s:|\-]+\|$", line)]
            rows = [[Paragraph(inline(cell.strip()), styles["ProposalCell"])
                     for cell in line.strip().strip("|").split("|")] for line in lines]
            column_count = len(rows[0])
            widths = {2: [width * .30, width * .70],
                      3: [width * .19, width * .22, width * .59],
                      4: [width * .31, width * .23, width * .23, width * .23],
                      5: [width * .36, width * .16, width * .16, width * .16, width * .16]}[column_count]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EDF1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#8A98A4")),
                ("LINEBELOW", (0, 1), (-1, -1), .25, colors.HexColor("#D6DCE1")),
            ]))
            story.extend([table, Spacer(1, 9)])
        elif block.startswith("!["):
            match = re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", block)
            if not match:
                raise RuntimeError(f"Invalid image block: {block}")
            caption, relative = match.groups()
            image_path = (args.source.parent / relative).resolve()
            with PillowImage.open(image_path) as image:
                image_width, image_height = image.size
            height = width * image_height / image_width
            picture = Image(str(image_path), width=width, height=height)
            story.append(KeepTogether([picture, Paragraph(inline(caption), styles["ProposalCaption"])]))
        elif block.startswith("```"):
            lines = block.splitlines()[1:-1]
            story.append(Paragraph("<br/>".join(escape(line) for line in lines), styles["ProposalCode"]))
        else:
            for line in block.splitlines():
                story.append(Paragraph(inline(line), styles["ProposalBody"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(args.output), pagesize=A4, leftMargin=48, rightMargin=48,
                                 topMargin=49, bottomMargin=45,
                                 title="A3 Dataset Proposal - Group NNs", author="Group NNs")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
