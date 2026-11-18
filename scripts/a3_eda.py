"""Audit ABO paired data and export the EDA evidence for the A3 M1 proposal.

Run from the repository root: python scripts/a3_eda.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import tarfile
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from a3_prepare_abo import ROOT, clean_text, verify_splits, write_csv, write_json


NAMES = {"FINENECKLACEBRACELETANKLET": "Necklace / bracelet", "FINEEARRING": "Earring",
         "FINERING": "Ring", "WALL_ART": "Wall art"}
COLORS = {"train": "#285F83", "validation": "#55A6A0", "test": "#DDA24E"}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def raw_audit(archive_path: Path, rows: list[dict], classes: set[str]) -> dict:
    selected_sources = {(row["source_file"], int(row["source_line"])): row for row in rows}
    counts = {stage: Counter() for stage in ("raw_all", "raw_selected_classes")}
    checked, mismatches = 0, []
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if not member.name.endswith(".json.gz"):
                continue
            with gzip.open(archive.extractfile(member), "rt", encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    record = json.loads(line)
                    titles = record.get("item_name", [])
                    any_text = any(clean_text(entry.get("value", "")) for entry in titles)
                    english = any(entry.get("language_tag", "").startswith("en_") and clean_text(entry.get("value", "")) for entry in titles)
                    types = {entry.get("value") for entry in record.get("product_type", []) if entry.get("value")}
                    stages = ["raw_all"]
                    if types & classes:
                        stages.append("raw_selected_classes")
                    for stage in stages:
                        count = counts[stage]
                        count["records"] += 1
                        count["missing_text_any_language"] += not any_text
                        count["no_english_title"] += not english
                        count["missing_main_image_id"] += not bool(record.get("main_image_id"))
                        count["missing_either_modality"] += not any_text or not record.get("main_image_id")
                        count["missing_or_ambiguous_label"] += len(types) != 1
                    row = selected_sources.get((member.name, number))
                    if row:
                        checked += 1
                        valid = (record.get("item_id") == row["item_id"]
                                 and record.get("main_image_id") == row["image_id"]
                                 and row["product_type"] in types
                                 and any(entry.get("language_tag") == row["language"]
                                         and clean_text(entry.get("value", "")) == row["title"] for entry in titles))
                        if not valid:
                            mismatches.append(row["item_id"])
    if checked != len(rows) or mismatches:
        raise RuntimeError(f"Native-pair source validation failed: checked={checked}, mismatches={mismatches[:10]}")
    result = {}
    for stage, count in counts.items():
        result[stage] = dict(count)
        result[stage]["rates_percent"] = {key: round(value / count["records"] * 100, 5)
                                           for key, value in count.items() if key != "records"}
    result["selected_source_pairs_checked"] = checked
    result["selected_source_pair_mismatches"] = 0
    return result


def label_flags(row: dict) -> list[str]:
    text = row["title"].casefold()
    flags = []
    if row["product_type"] == "SHOES" and re.search(r"\bboots?\b", text):
        flags.append("shoes_title_mentions_boot")
    if row["product_type"] == "SHOES" and re.search(r"\bsandals?\b", text):
        flags.append("shoes_title_mentions_sandal")
    if row["product_type"] in {"CHAIR", "SOFA"} and re.search(r"\b(cushions?|covers?|slipcovers?|replacement)\b", text):
        flags.append("furniture_title_mentions_accessory_terms")
    return flags


def quantiles(values: list[int | float]) -> dict:
    levels = [0, 25, 50, 75, 90, 95, 99, 100]
    return {f"p{level}": float(np.percentile(values, level)) for level in levels}


def figures(rows: list[dict], output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    labels = sorted({row["product_type"] for row in rows})
    fig, axis = plt.subplots(figsize=(10, 5))
    left = np.zeros(len(labels))
    for split, color in COLORS.items():
        values = [sum(row["product_type"] == label and row["split"] == split for row in rows) for label in labels]
        axis.barh([NAMES.get(label, label.title()) for label in labels], values, left=left, label=split, color=color)
        left += values
    axis.set_xlabel("Paired product records")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "a3_class_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for split, color in COLORS.items():
        titles = [row["title"] for row in rows if row["split"] == split]
        axes[0].hist([len(title.split()) for title in titles], bins=np.arange(0, 50, 2), density=True,
                     histtype="step", linewidth=1.8, label=split, color=color)
        axes[1].hist([len(title) for title in titles], bins=np.arange(0, 360, 15), density=True,
                     histtype="step", linewidth=1.8, color=color)
    axes[0].set_xlabel("Whitespace-delimited words per title")
    axes[1].set_xlabel("Characters per title")
    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("Density")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "a3_text_length_distribution.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist([min(int(row["width"]), int(row["height"])) for row in rows], bins=24, color=COLORS["train"])
    axes[0].set_xlabel("Image short side (pixels)")
    axes[0].set_ylabel("Paired records")
    axes[1].hist([int(row["width"]) / int(row["height"]) for row in rows], bins=36, color=COLORS["validation"])
    axes[1].set_xlabel("Width / height")
    axes[1].set_ylabel("Paired records")
    fig.tight_layout()
    fig.savefig(output / "a3_image_geometry.png", dpi=180)
    plt.close(fig)


def review_sample(rows: list[dict], output: Path, seed: int) -> list[dict]:
    generator = random.Random(seed)
    sampled = []
    for label in sorted({row["product_type"] for row in rows}):
        members = [row for row in rows if row["product_type"] == label]
        sampled.extend(generator.sample(members, 2))
    flagged = [row for row in rows if label_flags(row) and row["item_id"] not in {item["item_id"] for item in sampled}]
    sampled.extend(generator.sample(flagged, min(12, len(flagged))))
    font = ImageFont.load_default(size=14)
    for page in range((len(sampled) + 11) // 12):
        sheet = Image.new("RGB", (1200, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(sampled[page * 12:(page + 1) * 12]):
            left, top = (index % 4) * 300, (index // 4) * 400
            with Image.open(ROOT / row["image_path"]) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (280, 235))
                sheet.paste(thumb, (left + (300 - thumb.width) // 2, top + 8))
            caption = f"{page * 12 + index + 1}. {row['item_id']}\n{row['product_type']}\n" + textwrap.fill(row["title"], 35, max_lines=6)
            draw.multiline_text((left + 8, top + 250), caption, font=font, fill="black", spacing=3)
        sheet.save(output / f"a3_pair_review_{page + 1}.jpg", quality=93)
    review_path = output / "a3_pair_review.csv"
    previous = {row["item_id"]: row for row in read_csv(review_path)} if review_path.exists() else {}
    review = [{"item_id": row["item_id"], "product_type": row["product_type"], "title": row["title"],
               "image_path": row["image_path"], "selection": "label_flag" if index >= 24 else "stratified_random",
               "flags": " | ".join(label_flags(row)),
               "pair_review": previous.get(row["item_id"], {}).get("pair_review", "pending"),
               "label_review": previous.get(row["item_id"], {}).get("label_review", "pending"),
               "notes": previous.get(row["item_id"], {}).get("notes", "")}
              for index, row in enumerate(sampled)]
    write_csv(review_path, review, list(review[0]))
    return review


def proposal_examples(rows: list[dict], output: Path) -> None:
    examples = ["B079PVL9VD", "B07K29PBG7", "B000HC0CEK"]
    indexed = {row["item_id"]: row for row in rows}
    canvas = Image.new("RGB", (1200, 450), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    for index, item_id in enumerate(examples):
        row = indexed[item_id]
        left = index * 400
        with Image.open(ROOT / row["image_path"]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (370, 250))
            canvas.paste(thumb, (left + (400 - thumb.width) // 2, 8))
        caption = f"{row['product_type']} | {item_id}\n" + textwrap.fill(row["title"], 32, max_lines=5)
        draw.multiline_text((left + 10, 270), caption, font=font, fill="black", spacing=4)
    canvas.save(output / "a3_proposal_pair_examples.jpg", quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/a3/eda")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "data/processed/abo"
    rows = read_csv(processed / "selected_pairs.csv")
    catalog = read_csv(processed / "clean_catalog.csv")
    classes = {row["product_type"] for row in rows}
    native = raw_audit(ROOT / "data/raw/abo/abo-listings.tar", rows, classes)
    validation = verify_splits(rows, 4)
    word_lengths = [len(row["title"].split()) for row in rows]
    char_lengths = [len(row["title"]) for row in rows]
    group_sizes = Counter(row["group_id"] for row in rows)
    flagged = [{**row, "flags": " | ".join(label_flags(row))} for row in rows if label_flags(row)]
    write_csv(args.out / "a3_label_review_candidates.csv", flagged,
              ["item_id", "product_type", "split", "title", "image_path", "flags"])
    write_csv(args.out / "a3_text_length_distribution.csv",
              [{"words": value, "pairs": count} for value, count in sorted(Counter(word_lengths).items())], ["words", "pairs"])
    figures(rows, args.out)
    reviews = review_sample(rows, args.out, 261)
    proposal_examples(rows, args.out)
    summary = {
        "raw_missingness": native, "selected_pairs": len(rows), "classes": len(classes),
        "train_validation_test": dict(Counter(row["split"] for row in rows)),
        "selected_missing_text": sum(not row["title"].strip() for row in rows),
        "selected_missing_image_files": sum(not (ROOT / row["image_path"]).is_file() for row in rows),
        "word_length_quantiles": quantiles(word_lengths), "character_length_quantiles": quantiles(char_lengths),
        "image_short_side_quantiles": quantiles([min(int(row["width"]), int(row["height"])) for row in rows]),
        "image_aspect_ratio_quantiles": quantiles([int(row["width"]) / int(row["height"]) for row in rows]),
        "selected_class_counts": dict(sorted(Counter(row["product_type"] for row in rows).items())),
        "class_imbalance_ratio": max(Counter(row["product_type"] for row in rows).values()) / min(Counter(row["product_type"] for row in rows).values()),
        "label_flag_counts": dict(Counter(flag for row in rows for flag in label_flags(row))),
        "label_flag_unique_pairs": len(flagged), "review_sample_size": len(reviews),
        "pair_review_counts": dict(Counter(row["pair_review"] for row in reviews)),
        "label_review_counts": dict(Counter(row["label_review"] for row in reviews)),
        "groups": {"count": len(group_sizes), "largest": max(group_sizes.values()),
                   "largest_fraction": max(group_sizes.values()) / len(rows),
                   "size_quantiles": quantiles(list(group_sizes.values()))},
        "clean_catalog_pairs": len(catalog), "split_validation": validation,
        "manifest_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(processed.glob("*.csv"))},
        "review_method": "Two fixed-seed random pairs per class plus 12 fixed-seed label-flag candidates. Automated flags request inspection and do not establish label errors.",
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__, "pillow": Image.__version__},
    }
    write_json(args.out / "a3_eda_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
