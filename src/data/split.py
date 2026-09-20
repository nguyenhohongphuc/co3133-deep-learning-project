"""Fixed train/val/test split. Owner: TV 1. Task A1-01.

Handbook: every model in the main comparison uses the SAME partitions and the SAME
seed (10, 12.2); the report must state the split, seed, stratification and the exact
number of samples per split (10, 3.1).

Contract: writes results/a1/splits/split_seed<seed>.json as
    {"seed": 42, "val_size": 6000, "stratified": true,
     "train_idx": [...], "val_idx": [...], "test": "official 10k",
     "counts": {"train": 54000, "val": 6000, "test": 10000}}
Every loader reads this file; nobody re-splits on the fly.
"""
from __future__ import annotations

import pathlib


def build_split(labels, val_size: int, seed: int, stratified: bool = True) -> dict:
    """Return the split dict described above. TODO (TV 1, A1-01)."""
    raise NotImplementedError


def save_split(split: dict, path: pathlib.Path) -> None:
    """TODO (TV 1, A1-01): write JSON + print per-class counts for the report."""
    raise NotImplementedError


def load_split(path: pathlib.Path) -> dict:
    """TODO (TV 1, A1-01): read the JSON and sanity-check the counts."""
    raise NotImplementedError
