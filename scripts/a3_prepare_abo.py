"""Clean native ABO main-image/title pairs and create grouped A3 splits.

Dependencies: Pillow, numpy. Run scripts/a3_download_abo.py first.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import random
import re
import shutil
import tarfile
import textwrap
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = ["SHOES", "CHAIR", "BOOT", "SOFA", "SANDAL", "FINENECKLACEBRACELETANKLET",
                   "FINERING", "RUG", "TABLE", "FINEEARRING", "HANDBAG", "WALL_ART"]
FIELDS = ["item_id", "title", "language", "product_type", "image_id", "image_path",
          "brand", "model_number", "source_file", "source_line", "width", "height",
          "image_sha256", "pixel_sha256", "phash", "group_id", "label", "split"]


def clean_text(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value)
    value = "".join(character if unicodedata.category(character) != "Cc" else " " for character in value)
    return " ".join(value.split())


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_listings(archive_path: Path) -> tuple[list[dict], list[dict], dict]:
    candidates = defaultdict(list)
    exclusions = []
    raw_types = Counter()
    total = 0
    with tarfile.open(archive_path) as archive:
        for member in sorted(archive.getmembers(), key=lambda entry: entry.name):
            if not member.name.endswith(".json.gz"):
                continue
            with gzip.open(archive.extractfile(member), "rt", encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    record = json.loads(line)
                    total += 1
                    item_id = record.get("item_id", "")
                    source = {"item_id": item_id, "source_file": member.name, "source_line": number}
                    types = sorted({entry["value"] for entry in record.get("product_type", []) if entry.get("value")})
                    raw_types.update(types)
                    titles = [entry for entry in record.get("item_name", [])
                              if entry.get("language_tag", "").startswith("en_") and entry.get("value")]
                    reason = ""
                    if not item_id:
                        reason = "missing_item_id"
                    elif len(types) != 1:
                        reason = "missing_or_ambiguous_product_type"
                    elif not titles:
                        reason = "missing_english_title"
                    elif not record.get("main_image_id"):
                        reason = "missing_main_image_id"
                    if reason:
                        exclusions.append({**source, "reason": reason})
                        continue
                    title = min(titles, key=lambda entry: (entry["language_tag"] != "en_US", entry["language_tag"], entry["value"]))
                    text = clean_text(title["value"])
                    if len(text) < 3 or not any(character.isalpha() for character in text):
                        exclusions.append({**source, "reason": "empty_or_invalid_title"})
                        continue
                    brand = " | ".join(sorted({clean_text(entry["value"]) for entry in record.get("brand", []) if entry.get("value")}))
                    model = " | ".join(sorted({clean_text(entry["value"]) for entry in record.get("model_number", []) if entry.get("value")}))
                    candidates[item_id].append({**source, "title": text, "language": title["language_tag"],
                                               "product_type": types[0], "image_id": record["main_image_id"],
                                               "brand": brand, "model_number": model,
                                               "preferred_market": record.get("domain_name") == "amazon.com"})
    pairs = []
    for item_id, records in sorted(candidates.items()):
        if len({row["product_type"] for row in records}) > 1:
            exclusions.extend({**row, "reason": "conflicting_item_labels"} for row in records)
            continue
        records.sort(key=lambda row: (row["language"] != "en_US", not row["preferred_market"], row["source_file"], row["source_line"]))
        pairs.append(records[0])
        exclusions.extend({**row, "reason": "duplicate_item_record"} for row in records[1:])
    return pairs, exclusions, {"raw_records": total, "raw_product_types": dict(raw_types.most_common()),
                                "english_unique_items_before_image_validation": len(pairs)}


def prepare_images(archive_path: Path, raw_dir: Path, pairs: list[dict], download: bool, workers: int) -> tuple[list[dict], list[dict]]:
    image_paths = {}
    index = raw_dir / "images.csv.gz"
    if not index.exists() and archive_path.exists():
        with tarfile.open(archive_path) as archive:
            with archive.extractfile("images/metadata/images.csv.gz") as source, index.open("wb") as output:
                shutil.copyfileobj(source, output)
    with gzip.open(index, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image_paths[row["image_id"]] = row["path"]
    if not image_paths:
        raise RuntimeError("Image metadata CSV not found in ABO image archive")
    wanted = set()
    retained, exclusions = [], []
    for row in pairs:
        relative = image_paths.get(row["image_id"])
        if not relative:
            exclusions.append({**row, "reason": "image_id_missing_from_archive_metadata"})
            continue
        member_name = "images/small/" + relative
        destination = raw_dir / member_name
        if not destination.resolve().is_relative_to(raw_dir.resolve()):
            raise RuntimeError(f"Invalid image path in source metadata: {relative}")
        row["image_path"] = destination.relative_to(ROOT).as_posix()
        wanted.add(member_name)
        retained.append(row)
    if download:
        from a3_download_abo import download_images
        download_images(sorted(wanted), raw_dir, workers)
    if not archive_path.exists():
        missing = [row["image_path"] for row in retained if not (ROOT / row["image_path"]).is_file()]
        if missing:
            raise RuntimeError(f"{len(missing)} main images are missing. Run with --download-images.")
        return retained, exclusions
    extracted = set()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if member.name not in wanted or not member.isfile():
                continue
            destination = raw_dir / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Extract from the verified archive on every run, repairing modified local images.
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with archive.extractfile(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(destination)
            extracted.add(member.name)
    absent = wanted - extracted
    if absent:
        raise RuntimeError(f"{len(absent)} indexed images are absent from the archive")
    print(f"Extracted {len(extracted):,} unique main images", flush=True)
    return retained, exclusions


def inspect_image(path: str) -> dict:
    try:
        filename = ROOT / path
        with Image.open(filename) as original:
            original.verify()
        with Image.open(filename) as original:
            rgb = ImageOps.exif_transpose(original).convert("RGB")
            rgb.load()
        width, height = rgb.size
        if min(width, height) < 32:
            return {"reason": "image_short_side_below_32px"}
        grayscale = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        # A 2-D cosine transform followed by the top 8x8 coefficient median gives a 64-bit pHash.
        frequencies = np.arange(8)[:, None]
        coordinates = np.arange(32)[None, :]
        cosine = np.cos(np.pi * frequencies * (2 * coordinates + 1) / 64)
        coefficients = cosine @ grayscale @ cosine.T
        bits = coefficients > np.median(coefficients)
        phash = sum(int(bit) << (63 - index) for index, bit in enumerate(bits.ravel()))
        digest = hashlib.sha256(f"{width}x{height}:".encode() + rgb.tobytes()).hexdigest()
        return {"width": width, "height": height, "image_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                "pixel_sha256": digest, "phash": f"{phash:016x}"}
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return {"reason": "corrupt_or_unreadable_image", "detail": str(error)}


class Groups:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        self.parent[max(first, second)] = min(first, second)


def group_pairs(pairs: list[dict], distance: int) -> tuple[list[dict], list[dict]]:
    groups = Groups(len(pairs))
    seen = {}
    for index, row in enumerate(pairs):
        keys = [("image", row["image_id"]), ("pixels", row["pixel_sha256"]), ("title", row["title"].casefold())]
        if row["brand"] and len(row["model_number"]) >= 5 and row["model_number"].casefold() not in {"default", "unknown", "generic"}:
            keys.append(("model", row["brand"].casefold(), row["model_number"].casefold()))
        for key in keys:
            if key in seen:
                groups.join(index, seen[key])
            else:
                seen[key] = index
    # BK-tree searches every hash within the chosen Hamming radius without an all-pairs scan.
    tree = None
    edges = []
    for index, row in enumerate(pairs):
        value = int(row["phash"], 16)
        if tree is None:
            tree = (value, index, {})
            continue
        stack = [tree]
        while stack:
            node_value, node_index, children = stack.pop()
            difference = (value ^ node_value).bit_count()
            if difference <= distance:
                groups.join(index, node_index)
                edges.append({"item_id_a": pairs[node_index]["item_id"], "item_id_b": row["item_id"], "distance": difference})
            stack.extend(child for metric, child in children.items() if difference - distance <= metric <= difference + distance)
        node = tree
        while True:
            difference = (value ^ node[0]).bit_count()
            if difference == 0:
                break
            if difference not in node[2]:
                node[2][difference] = (value, index, {})
                break
            node = node[2][difference]
    for index, row in enumerate(pairs):
        row["group_id"] = pairs[groups.find(index)]["item_id"]
    return pairs, edges


def assign_splits(rows: list[dict], seed: int) -> dict:
    labels = sorted({row["product_type"] for row in rows})
    label_ids = {label: index for index, label in enumerate(labels)}
    groups = defaultdict(list)
    for row in rows:
        row["label"] = label_ids[row["product_type"]]
        groups[row["group_id"]].append(row)
    totals = Counter(row["product_type"] for row in rows)
    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    targets = {split: {label: count * fraction for label, count in totals.items()} for split, fraction in fractions.items()}
    counts = {split: Counter() for split in fractions}
    ordered = list(sorted(groups.items()))
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=lambda entry: -len(entry[1]))
    for _, members in ordered:
        composition = Counter(row["product_type"] for row in members)
        # Largest groups first, then minimize the increase in normalized squared target error.
        def cost(split: str) -> float:
            return sum(((counts[split][label] + number - targets[split][label]) ** 2
                        - (counts[split][label] - targets[split][label]) ** 2) / max(targets[split][label], 1)
                       for label, number in composition.items())
        split = min(fractions, key=cost)
        counts[split].update(composition)
        for row in members:
            row["split"] = split
    for label in labels:
        if any(counts[split][label] == 0 for split in fractions):
            raise RuntimeError(f"Class {label} has no samples in at least one split. Select classes with more independent groups.")
    return {"label_to_id": label_ids, "target_ratios": fractions,
            "actual_sizes": {split: sum(counter.values()) for split, counter in counts.items()}}


def verify_splits(rows: list[dict], distance: int) -> dict:
    # All near-neighbor edges were joined before splitting. Group isolation covers their transitive closure.
    for field in ("item_id", "group_id", "image_id", "pixel_sha256", "phash"):
        seen = {}
        for row in rows:
            key = row[field]
            if key in seen and seen[key] != row["split"]:
                raise RuntimeError(f"Cross-split overlap for {field}: {key}")
            seen[key] = row["split"]
    title_splits = {}
    for row in rows:
        title = row["title"].casefold()
        if title in title_splits and title_splits[title] != row["split"]:
            raise RuntimeError("Title overlap across splits")
        title_splits[title] = row["split"]
    return {"cross_split_item_overlap": 0, "cross_split_group_overlap": 0,
            "cross_split_image_id_overlap": 0, "cross_split_pixel_overlap": 0,
            "cross_split_normalized_title_overlap": 0, "phash_grouping_hamming_radius": distance,
            "all_manifest_images_verified": True}


def preview_pairs(rows: list[dict], path: Path) -> None:
    examples = []
    for label in sorted({row["product_type"] for row in rows}):
        examples.extend([row for row in rows if row["product_type"] == label][:2])
    width, height, columns = 300, 350, 4
    canvas = Image.new("RGB", (width * columns, height * ((len(examples) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    for index, row in enumerate(examples):
        left, top = (index % columns) * width, (index // columns) * height
        with Image.open(ROOT / row["image_path"]) as image:
            thumbnail = ImageOps.contain(image.convert("RGB"), (280, 230))
            canvas.paste(thumbnail, (left + (width - thumbnail.width) // 2, top + 8))
        label = row["product_type"].replace("_", " ")
        caption = textwrap.fill(row["title"], width=37, max_lines=4)
        draw.text((left + 10, top + 240), label, fill="black", font=font)
        draw.multiline_text((left + 10, top + 262), caption, fill="black", font=font, spacing=3)
    canvas.save(path)


def write_report(path: Path, summary: dict, distribution: list[dict]) -> None:
    radius = summary["validation"]["phash_grouping_hamming_radius"]
    command = ("python scripts/a3_prepare_abo.py --classes " + " ".join(summary["selected_classes"])
               + f" --max-per-class {summary['max_per_class']} --seed {summary['seed']} --phash-distance {radius}")
    lines = ["# ABO cleaning report", "",
             "Data and images are credited to Amazon.com under CC BY 4.0. "
             "The official listing metadata and image index were downloaded in full. "
             "Native images with a maximum axis of 256 pixels were downloaded for the 12 selected product types.", "",
             "| Stage | Pairs or records |", "|---|---:|",
             f"| Raw listing records | {summary['raw_records']:,} |",
             f"| Unique items with English titles and pairing metadata | {summary['english_unique_items_before_image_validation']:,} |",
             f"| Items in selected class scope before image validation | {summary['in_scope_pairs_before_image_validation']:,} |",
             f"| Clean pairs in the selected class scope | {summary['clean_catalog_pairs']:,} |",
             f"| Selected experiment pairs | {summary['selected_pairs']:,} |", "",
             f"Seed {summary['seed']}. At most {summary['max_per_class']:,} pairs per class. "
             "The split targets 80% train, 10% validation and 10% test, with connected groups kept together.", "",
             "| Product type | Clean pairs | Train | Validation | Test |", "|---|---:|---:|---:|---:|"]
    for row in distribution:
        if row["selected"]:
            lines.append(f"| {row['product_type']} | {row['clean_catalog']:,} | {row['train']:,} | {row['validation']:,} | {row['test']:,} |")
    lines.extend(["", "## Cleaning and grouping", "",
                  "Titles are normalized with HTML decoding, Unicode NFKC and whitespace cleanup. "
                  "Natural product names remain in the text. Images must decode successfully and have a short side of at least 32 pixels. "
                  "Exact duplicate image/title/label pairs are removed. Identical decoded images with conflicting product labels are excluded. "
                  "Exclusion reasons and source file/line locations are recorded in data/processed/abo/excluded.csv.", "",
                  "Group links use main-image ID, decoded-pixel hash, identical normalized title, "
                  f"brand/model identifiers of at least five characters, and pHash Hamming distance at most {radius}. "
                  "Links are computed before subset selection, including paths through unselected records. "
                  "The checks found zero cross-split overlap in item IDs, group IDs, image IDs, decoded pixels and normalized titles.", "",
                  f"The experiment contains {summary['selected_group_count']:,} groups. "
                  f"The largest group has {summary['largest_selected_group']:,} selected pairs.", "",
                  "## Limits", "",
                  "These are automated quality checks. The image/title preview provides a small sample for visual inspection. "
                  "Shared models and similar hashes can conservatively group distinct products. "
                  "The native product labels can overlap semantically, and product titles can be strong label cues. "
                  "Visual inspection found chair cushions under CHAIR and boots under SHOES. The source labels are retained. "
                  "The selected class scope and cap define a custom subset, so report its exact split when presenting results.", "",
                  "## Reproduction", "", "```bash", "python scripts/a3_download_abo.py",
                  command + " --download-images --workers 16", "```", "",
                  "Run the same preparation command without --download-images to reproduce cleaning offline. "
                  "See summary.json for source URLs, archive/index SHA-256 values, package versions and actual counts. "
                  "The individual-image source ledger is data/raw/abo/image_sources.jsonl.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/abo")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/abo")
    parser.add_argument("--audit", type=Path, default=ROOT / "results/a3/data")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Product types to clean (default: 12 named object categories).")
    parser.add_argument("--min-class-size", type=int, default=200)
    parser.add_argument("--max-per-class", type=int, default=1250,
                        help="Reproducible A3 subset cap. The complete clean catalog is also retained.")
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-images", action="store_true", help="Download native small images for the selected product types")
    args = parser.parse_args()
    if (not 0 <= args.phash_distance <= 8 or args.workers < 1 or len(set(args.classes)) < 5
            or args.max_per_class < 1 or args.min_class_size < 1):
        parser.error("Use 0..8 for pHash distance, positive workers, and at least 5 classes")
    args.raw, args.out, args.audit = args.raw.resolve(), args.out.resolve(), args.audit.resolve()
    pairs, excluded, summary = load_listings(args.raw / "abo-listings.tar")
    print(f"Read {summary['raw_records']:,} records; "
          f"{len(pairs):,} unique items have English titles and pairing metadata", flush=True)
    write_json(args.out / "metadata_summary.json", summary)
    if args.metadata_only:
        args.out.mkdir(parents=True, exist_ok=True)
        write_csv(args.out / "metadata_candidates.csv", pairs, FIELDS)
        return
    write_csv(args.out / "metadata_candidates.csv", pairs, FIELDS)
    scoped = []
    for row in pairs:
        if row["product_type"] in args.classes:
            scoped.append(row)
        else:
            excluded.append({**row, "reason": "outside_selected_class_scope"})
    summary["requested_classes"] = sorted(args.classes)
    summary["in_scope_pairs_before_image_validation"] = len(scoped)
    print(f"Selected class scope: {len(scoped):,} products", flush=True)
    pairs, missing = prepare_images(args.raw / "abo-images-small.tar", args.raw, scoped, args.download_images, args.workers)
    excluded.extend(missing)
    paths = sorted({row["image_path"] for row in pairs})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        inspected = dict(zip(paths, executor.map(inspect_image, paths)))
    clean, seen_pairs = [], set()
    for row in pairs:
        quality = inspected[row["image_path"]]
        if "reason" in quality:
            excluded.append({**row, **quality})
            continue
        row.update(quality)
        identity = (row["title"].casefold(), row["pixel_sha256"], row["product_type"])
        if identity in seen_pairs:
            excluded.append({**row, "reason": "duplicate_image_title_label_pair"})
            continue
        seen_pairs.add(identity)
        clean.append(row)
    image_labels = defaultdict(set)
    for row in clean:
        image_labels[row["pixel_sha256"]].add(row["product_type"])
    unambiguous = []
    for row in clean:
        if len(image_labels[row["pixel_sha256"]]) > 1:
            excluded.append({**row, "reason": "identical_image_conflicting_labels"})
        else:
            unambiguous.append(row)
    clean, edges = group_pairs(unambiguous, args.phash_distance)
    counts = Counter(row["product_type"] for row in clean)
    eligible = [label for label, count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0])) if count >= args.min_class_size]
    chosen = args.classes
    if len(set(chosen)) != len(chosen) or len(chosen) < 5 or any(label not in eligible for label in chosen):
        raise RuntimeError(f"Choose at least 5 distinct eligible classes. Eligible classes: {eligible}")
    selected = []
    generator = random.Random(args.seed)
    for label in sorted(chosen):
        members = [dict(row) for row in clean if row["product_type"] == label]
        generator.shuffle(members)
        selected.extend(members[:args.max_per_class])
    selected.sort(key=lambda row: row["item_id"])
    if len(selected) < 5000:
        raise RuntimeError(f"Only {len(selected)} paired samples selected. Need at least 5,000 for A3.")
    split_info = assign_splits(selected, args.seed)
    validation = verify_splits(selected, args.phash_distance)
    for split in ("train", "validation", "test"):
        write_csv(args.out / f"{split}.csv", [row for row in selected if row["split"] == split], FIELDS)
    write_csv(args.out / "clean_catalog.csv", clean, FIELDS)
    write_csv(args.out / "selected_pairs.csv", selected, FIELDS)
    write_csv(args.out / "excluded.csv", excluded, ["item_id", "source_file", "source_line", "reason", "detail"])
    write_csv(args.audit / "near_duplicate_pairs.csv", edges, ["item_id_a", "item_id_b", "distance"])
    distribution = [{"product_type": label, "clean_catalog": counts[label], "selected": label in chosen,
                     **{split: sum(row["product_type"] == label and row["split"] == split for row in selected)
                        for split in ("train", "validation", "test")}} for label in sorted(counts)]
    write_csv(args.audit / "class_distribution.csv", distribution,
              ["product_type", "clean_catalog", "selected", "train", "validation", "test"])
    summary.update({"clean_catalog_pairs": len(clean), "clean_catalog_classes": len(counts),
                    "selected_pairs": len(selected), "selected_classes": sorted(chosen),
                    "unique_selected_images": len({row["image_id"] for row in selected}),
                    "seed": args.seed, "max_per_class": args.max_per_class,
                    "split": split_info, "validation": validation,
                    "exclusion_counts": dict(Counter(row["reason"] for row in excluded)),
                    "near_duplicate_edges": len(edges),
                    "selected_group_count": len({row["group_id"] for row in selected}),
                    "largest_selected_group": max(Counter(row["group_id"] for row in selected).values()),
                    "language_counts": dict(Counter(row["language"] for row in selected)),
                    "title_word_count": {"min": min(len(row["title"].split()) for row in selected),
                                         "median": float(np.median([len(row["title"].split()) for row in selected])),
                                         "max": max(len(row["title"].split()) for row in selected)},
                    "image_short_side_min": min(min(row["width"], row["height"]) for row in selected),
                    "sources": [json.loads(path.read_text()) for path in sorted(args.raw.glob("*.json")) if not path.name.endswith("summary.json")],
                    "versions": {"numpy": np.__version__, "pillow": Image.__version__}})
    write_json(args.audit / "summary.json", summary)
    write_json(args.out / "labels.json", split_info["label_to_id"])
    preview_pairs(selected, args.audit / "pair_preview.jpg")
    write_report(args.audit / "cleaning_report.md", summary, distribution)
    print(json.dumps({key: summary[key] for key in ("clean_catalog_pairs", "selected_pairs", "selected_classes", "split", "exclusion_counts", "selected_group_count", "largest_selected_group")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
