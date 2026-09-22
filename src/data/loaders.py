"""DataLoader factory - the interface every other module depends on.

Owner: TV 1. Task A1-04.

CONTRACT (agreed 20/09, do not change without telling TV 2 and TV 3):
    build_loaders(config) -> (train_loader, val_loader, test_loader)
    each batch is (images, labels) with images [B, 1, 28, 28] float and labels [B] long
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.common.seed import worker_init_fn
from src.data.fashion_mnist import FashionMNISTSubset
from src.data.split import load_split
from src.data.transforms import build_transforms


def _make_generator(seed: int) -> torch.Generator:
    """Create a deterministic generator for one DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

def build_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the three loaders from the saved split file.

    TODO (TV 1, A1-04):
      - read config['split_file'] (created by src/data/split.py)
      - apply build_transforms(config, train=...)
      - seed the loaders via src.common.seed.worker_init_fn
      - save one example batch for the report (handbook 3.1)
    """
    split = load_split(config["split_file"])

    data_config = config["data"]
    training_config = config["training"]

    train_dataset = FashionMNISTSubset(
        root=config["data_dir"],
        indices=split["train_idx"],
        transform=build_transforms(config, train=True),
        train=True,
    )

    val_dataset = FashionMNISTSubset(
        root=config["data_dir"],
        indices=split["val_idx"],
        transform=build_transforms(config, train=False),
        train=True,
    )

    test_dataset = FashionMNISTSubset(
        root=config["data_dir"],
        indices=list(range(split["counts"]["test"])),
        transform=build_transforms(config, train=False),
        train=False,
    )

    batch_size = training_config["batch_size"]
    num_workers = training_config["num_workers"]
    common_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": worker_init_fn,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=_make_generator(config["seed"]),
        **common_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        generator=_make_generator(config["seed"] + 1),
        **common_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        generator=_make_generator(config["seed"] + 2),
        **common_args,
    )

    return train_loader, val_loader, test_loader
