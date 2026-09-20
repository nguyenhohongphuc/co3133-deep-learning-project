"""DataLoader factory - the interface every other module depends on.

Owner: TV 1. Task A1-04.

CONTRACT (agreed 20/09, do not change without telling TV 2 and TV 3):
    build_loaders(config) -> (train_loader, val_loader, test_loader)
    each batch is (images, labels) with images [B, 1, 28, 28] float and labels [B] long
"""
from __future__ import annotations

from torch.utils.data import DataLoader


def build_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the three loaders from the saved split file.

    TODO (TV 1, A1-04):
      - read config['split_file'] (created by src/data/split.py)
      - apply build_transforms(config, train=...)
      - seed the loaders via src.common.seed.worker_init_fn
      - save one example batch for the report (handbook 3.1)
    """
    raise NotImplementedError
