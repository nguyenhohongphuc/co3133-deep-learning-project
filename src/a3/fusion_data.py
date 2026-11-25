"""Shared text-only manifests/collator, with sonphan's image preprocessing."""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms as T

from .data import ABODataset, TextCollator, load_split

REPO = Path(__file__).resolve().parents[2]


def repo_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def split_fingerprint(folder):
    folder = repo_path(folder)
    return {n: hashlib.sha256((folder / n).read_bytes()).hexdigest()
            for n in ("train.csv", "validation.csv", "test.csv", "labels.json")}


class Letterbox:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = ImageOps.exif_transpose(image)
        image = ImageOps.contain(image, (self.size, self.size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), "white")
        canvas.paste(image, ((self.size - image.width) // 2, (self.size - image.height) // 2))
        return canvas


def image_transform(train=False):
    # Mirrors origin/sonphan:68e9f7e, src/a3_image_only/dataset.py.
    steps = [Letterbox()]
    if train:
        steps += [T.RandomHorizontalFlip(p=0.5),
                  T.RandomAffine(5, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255)]
    return T.Compose(steps + [T.ToTensor(), T.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def load_fusion_splits(cfg, *, check_images=True):
    folder = repo_path(cfg["data"]["dir"])
    splits = load_split(folder)
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    if len(labels) != 12 or sorted(labels.values()) != list(range(12)):
        raise ValueError("Expected the official labels.json with 12 contiguous labels")
    root = repo_path(cfg["data"]["image_root"]).resolve()
    for name, rows in splits.items():
        if {r["label"] for r in rows} != set(range(12)):
            raise ValueError(f"{name}: not all 12 labels are present")
        if len({r["item_id"] for r in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate item_id")
        for row in rows:
            if labels.get(row["product_type"]) != row["label"] or not row["title"].strip():
                raise ValueError(f"{name}: invalid label or title for {row['item_id']}")
            path = (root / row["image_path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Image path escapes image_root: {row['image_path']}")
            if check_images and not path.is_file():
                raise FileNotFoundError(f"Missing {path}; download ABO images or set data.image_root")
    # Additional checks beyond the shared item_id/group_id checks.
    for a, b in combinations(splits, 2):
        for key in ("image_id", "image_sha256", "pixel_sha256", "title"):
            def values(rows):
                return {r[key].strip().casefold() for r in rows if r.get(key)}
            overlap = values(splits[a]) & values(splits[b])
            if overlap:
                raise ValueError(f"Leakage between {a}/{b}: {key} ({len(overlap)} values)")
    return splits, labels


def make_dataset(cfg, rows, *, train=False):
    return ABODataset(rows, image_root=repo_path(cfg["data"]["image_root"]),
                      load_images=True, transform=image_transform(train))


def make_collator(cfg, tokenizer):
    return TextCollator(tokenizer, max_length=cfg["data"]["max_length"])
