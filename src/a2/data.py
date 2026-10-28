"""30VNFoods dataset và DataLoader cho A2.

Đọc danh sách ảnh từ split_v2.json (notebooks/a2_eda.ipynb) nên mọi model dùng đúng một split.
Cùng hợp đồng với src/data/loaders.build_loaders của A1:
    build_loaders(config, normalize) -> (train_loader, val_loader, test_loader)
    mỗi batch là (images, labels): images [B, 3, 224, 224] float, labels [B] long

Chạy trực tiếp để kiểm tra một batch và lưu ảnh minh hoạ (handbook 3.1):
    python -m src.a2.data
"""
from __future__ import annotations

import json
import pathlib

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.seed import worker_init_fn


def load_split(path) -> dict:
    """Đọc split JSON và kiểm tra số lượng khớp với 'counts'."""
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    for name in ["train", "val", "test"]:
        if len(split[name]) != split["counts"][name]:
            raise ValueError(f"{name}: {len(split[name])} images but counts says {split['counts'][name]}")
    return split


class VNFoods(Dataset):
    def __init__(self, root, items: list[dict], transform=None, draft_size: int = 320):
        self.root = pathlib.Path(root)
        self.items = items
        self.transform = transform
        self.draft_size = draft_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        img = Image.open(self.root / item["path"])
        # JPEG: giải mã thẳng ở độ phân giải nhỏ hơn (>= draft_size), nhanh hơn nhiều với ảnh ~1000 px
        img.draft("RGB", (self.draft_size, self.draft_size))
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, item["label"]


def build_transforms(config: dict, normalize: str, train: bool):
    data_config = config["data"]
    stats = data_config["stats"][normalize]
    size = data_config["image_size"]
    if train:
        steps = [transforms.RandomResizedCrop(size, scale=tuple(data_config["crop_scale"]))]
        if data_config.get("hflip", False):
            steps.append(transforms.RandomHorizontalFlip())
        jitter = data_config.get("color_jitter", 0.0)
        if jitter:
            steps.append(transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter))
    else:
        steps = [transforms.Resize(data_config["resize"]), transforms.CenterCrop(size)]
    steps += [transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"])]
    return transforms.Compose(steps)


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_loaders(config: dict, normalize: str) -> tuple[DataLoader, DataLoader, DataLoader]:
    split = load_split(config["split_file"])
    training_config = config["training"]
    loaders = []
    for i, name in enumerate(["train", "val", "test"]):
        train = name == "train"
        dataset = VNFoods(config["data_dir"], split[name], build_transforms(config, normalize, train))
        loaders.append(DataLoader(
            dataset,
            batch_size=training_config["batch_size"],
            shuffle=train,
            num_workers=training_config["num_workers"],
            pin_memory=True,
            persistent_workers=training_config["num_workers"] > 0,
            worker_init_fn=worker_init_fn,
            generator=_make_generator(config["seed"] + i),
        ))
    return tuple(loaders)


def save_example_batch(config: dict, normalize: str, out_png, n: int = 16):
    """Lưu một batch train sau tiền xử lý (đã bỏ Normalize để hiển thị)."""
    import matplotlib.pyplot as plt

    split = load_split(config["split_file"])
    dataset = VNFoods(config["data_dir"], split["train"], build_transforms(config, normalize, train=True))
    loader = DataLoader(dataset, batch_size=n, shuffle=True, generator=_make_generator(config["seed"]))
    images, labels = next(iter(loader))
    stats = config["data"]["stats"][normalize]
    mean = torch.tensor(stats["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(stats["std"]).view(1, 3, 1, 1)
    shown = (images * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(10, 10.5))
    for ax, img, label in zip(axes.flat, shown, labels):
        ax.imshow(img)
        ax.set_title(split["classes"][label], fontsize=9)
        ax.axis("off")
    fig.suptitle(f"One training batch after preprocessing: images {list(images.shape)}, labels {list(labels.shape)}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return images, labels


if __name__ == "__main__":
    import yaml

    with open("configs/a2.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = pathlib.Path("results/a2/figures/a2_example_batch.png")
    images, labels = save_example_batch(config, "train", out)
    print("images:", tuple(images.shape), images.dtype, "| labels:", tuple(labels.shape), labels.dtype,
          "| min/max label:", int(labels.min()), int(labels.max()))
    print("saved", out)
