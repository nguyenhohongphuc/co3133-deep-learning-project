"""ABO image/title dataset dùng chung cho A3 (image-only, text-only, fusion).

Cả ba model đọc cùng một split (train.csv / validation.csv / test.csv do scripts/a3_prepare_abo.py tạo),
cùng một transform ảnh và cùng một collator, nên khác biệt kết quả chỉ đến từ model.

Hợp đồng:
    load_split(split_dir) -> {"train": rows, "validation": rows, "test": rows}, rows là list[dict]
    ABODataset[i] -> {"image": Tensor[3,224,224] | None, "title": str, "label": int, "item_id": str}
    TextCollator(batch) -> {"input_ids": [B,L], "attention_mask": [B,L], "labels": [B],
                            "item_id": list[str], "images": [B,3,224,224] (chỉ khi có ảnh)}

Chạy trực tiếp để kiểm tra nhanh:
    python -m src.a3.data --split-dir data/processed/abo --image-root .
"""
from __future__ import annotations

import csv
import pathlib

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

SPLITS = ("train", "validation", "test")
# Số dòng của split cố định (results/a3/data/summary.json, seed 42). Khác số này nghĩa là file split đã bị đổi.
EXPECTED_SIZES = {"train": 8501, "validation": 1063, "test": 1063}
REQUIRED_COLUMNS = ("item_id", "title", "product_type", "image_path", "group_id", "label", "split")
NUM_CLASSES = 12

# Tiền xử lý ảnh theo proposal M1 (reports/a3/a3_proposal.md, mục preprocessing).
# TODO(người làm image-only): xác nhận các thông số này trước khi chạy thật, đặc biệt mức color jitter.
IMAGE_SIZE = 224
PAD_COLOR = (255, 255, 255)                 # nền trắng, giống ảnh catalog ABO
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_COLOR_JITTER = 0.1                    # "mild color jitter"; giá trị cụ thể chưa chốt


def _read_csv(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_split(split_dir) -> dict[str, list[dict]]:
    """Đọc 3 file split và kiểm tra trước khi dùng. Sai ở đâu thì raise ở đó, không bỏ qua.

    Kiểm tra: đủ cột, đúng số dòng, cột split khớp tên file, nhãn nằm trong [0, 12),
    mỗi product_type chỉ ứng với một label, và không item_id / group_id nào nằm ở 2 split.
    """
    split_dir = pathlib.Path(split_dir)
    splits = {}
    for name in SPLITS:
        path = split_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Không thấy {path}. Chạy scripts/a3_prepare_abo.py hoặc lấy file split chung của nhóm.")
        rows = _read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if rows and c not in rows[0]]
        if not rows or missing:
            raise ValueError(f"{path}: thiếu cột {missing or list(REQUIRED_COLUMNS)}")
        if len(rows) != EXPECTED_SIZES[name]:
            raise ValueError(f"{path}: {len(rows)} dòng, split cố định phải có {EXPECTED_SIZES[name]}")
        wrong = {r["split"] for r in rows} - {name}
        if wrong:
            raise ValueError(f"{path}: cột split chứa {sorted(wrong)}, phải toàn là '{name}'")
        for r in rows:
            r["label"] = int(r["label"])
        splits[name] = rows

    # Nhãn: dùng nguyên cột label, chỉ kiểm tra là nó nhất quán chứ không map lại.
    label_of_type = {}
    for name, rows in splits.items():
        for r in rows:
            if not 0 <= r["label"] < NUM_CLASSES:
                raise ValueError(f"{name}.csv: item {r['item_id']} có label {r['label']} ngoài [0, {NUM_CLASSES})")
            previous = label_of_type.setdefault(r["product_type"], r["label"])
            if previous != r["label"]:
                raise ValueError(f"product_type {r['product_type']} ứng với 2 label: {previous} và {r['label']}")
    if len(label_of_type) != NUM_CLASSES:
        raise ValueError(f"Có {len(label_of_type)} class, cần {NUM_CLASSES}")

    # Chống leakage: một item hay một group (các bản gần trùng) chỉ được nằm ở một split.
    for column in ("item_id", "group_id"):
        owner = {}
        for name, rows in splits.items():
            for r in rows:
                if owner.setdefault(r[column], name) != name:
                    raise ValueError(f"{column} {r[column]} nằm ở cả {owner[r[column]]} và {name}")
    return splits


def class_names(splits: dict[str, list[dict]]) -> list[str]:
    """Tên class theo thứ tự label (dùng cho confusion matrix, per-class F1)."""
    names = {r["label"]: r["product_type"] for rows in splits.values() for r in rows}
    return [names[i] for i in range(NUM_CLASSES)]


class PadToSquare:
    """Thu ảnh cho vừa size x size (giữ tỉ lệ) rồi pad trắng phần còn thiếu.

    Viết thành class thay vì lambda vì DataLoader trên Windows (num_workers > 0) phải pickle transform.
    """

    def __init__(self, size: int = IMAGE_SIZE, color=PAD_COLOR):
        self.size = size
        self.color = color

    def __call__(self, img: Image.Image) -> Image.Image:
        return ImageOps.pad(img, (self.size, self.size), color=self.color)


def build_image_transform(train: bool):
    """Giữ tỉ lệ ảnh, pad trắng thành 224x224, rồi normalize theo ImageNet.

    Không crop như A2: đồ vật trong ảnh catalog thường nằm sát mép, crop dễ cắt mất chi tiết
    (mũi giày, chân ghế). Train thêm color jitter nhẹ; validation/test thì không augment.
    """
    steps = [PadToSquare()]
    if train and TRAIN_COLOR_JITTER:
        j = TRAIN_COLOR_JITTER
        steps.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class ABODataset(Dataset):
    """Một split của ABO. Mỗi mẫu là dict để model nào cần gì tự lấy.

    rows: list dòng của một split (load_split(...)["train"]).
    image_root: thư mục gốc để ghép với image_path trong CSV (đường dẫn tương đối dạng
        data/raw/abo/images/...). Mặc định là thư mục repo; trên Colab đổi thành nơi chứa thư mục data/.
    load_images=False: không mở file ảnh (text-only chạy nhanh hơn, không cần tải ảnh về).
    transform: mặc định là transform đánh giá (không augment). Train thì truyền build_image_transform(True).
    """

    def __init__(self, rows: list[dict], image_root=".", load_images: bool = True, transform=None):
        self.rows = rows
        self.image_root = pathlib.Path(image_root)
        self.load_images = load_images
        self.transform = transform if transform is not None else build_image_transform(train=False)

    def __len__(self) -> int:
        return len(self.rows)

    def image_path(self, idx: int) -> pathlib.Path:
        return self.image_root / self.rows[idx]["image_path"]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = None
        if self.load_images:
            with Image.open(self.image_path(idx)) as img:
                image = self.transform(img.convert("RGB"))
        return {"image": image, "title": row["title"], "label": row["label"], "item_id": row["item_id"]}


class TextCollator:
    """Gộp list mẫu thành batch. Tokenize ở đây (theo batch) thay vì trong __getitem__:

    - padding theo câu dài nhất trong batch: title trung vị chỉ 12 từ, nên batch thường ngắn
      hơn nhiều so với max_length, tính nhanh hơn so với pad cố định lên 64;
    - tokenizer "fast" xử lý cả batch một lần nhanh hơn gọi từng câu.
    tokenizer=None: bỏ qua phần text (dùng cho image-only).
    """

    def __init__(self, tokenizer=None, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        out = {"labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
               "item_id": [b["item_id"] for b in batch]}
        if self.tokenizer is not None:
            encoded = self.tokenizer([b["title"] for b in batch], padding="longest", truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            out["input_ids"] = encoded["input_ids"]
            out["attention_mask"] = encoded["attention_mask"]
        if batch[0]["image"] is not None:
            out["images"] = torch.stack([b["image"] for b in batch])
        return out


if __name__ == "__main__":
    import argparse

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/processed/abo")
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--tokenizer", default="distilbert/distilbert-base-uncased")
    args = parser.parse_args()

    splits = load_split(args.split_dir)
    for name, rows in splits.items():
        print(f"{name}: {len(rows)} mẫu")
    print("classes:", class_names(splits))

    collate = TextCollator(AutoTokenizer.from_pretrained(args.tokenizer), max_length=64)
    text_only = ABODataset(splits["train"], args.image_root, load_images=False)
    batch = next(iter(DataLoader(text_only, batch_size=8, collate_fn=collate)))
    print("text-only:", {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})

    with_images = ABODataset(splits["train"], args.image_root, transform=build_image_transform(train=True))
    batch = next(iter(DataLoader(with_images, batch_size=8, collate_fn=collate)))
    print("có ảnh:", {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})
