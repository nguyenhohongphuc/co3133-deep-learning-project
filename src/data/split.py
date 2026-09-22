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
import argparse
import json
import pathlib
import numpy as np
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import FashionMNIST

def _class_counts(labels: np.ndarray, indices: np.ndarray) -> list[int]:
    """Return the number of samples of each class in one split."""
    return np.bincount(labels[indices], minlength=10).tolist()


def build_split(labels, val_size: int, seed: int, stratified: bool = True) -> dict:
    """Return the split dict described above. TODO (TV 1, A1-01)."""
    labels = np.asarray(labels)
    all_indices = np.arange(len(labels))
    
    if val_size <= 0 or val_size >= len(labels):
        raise ValueError("val_size must be between 1 and the dataset size - 1")
    
    if stratified:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_size,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(all_indices, labels))
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_indices)
        val_idx = shuffled[:val_size]
        train_idx = shuffled[val_size:]
        
    # Sanity checks: no overlap and no omitted sample.
    if len(set(train_idx).intersection(set(val_idx))) != 0:
        raise ValueError("Train and validation splits overlap.")
    
    if len(train_idx) + len(val_idx) != len(labels):
        raise ValueError("Split does not cover all official training samples.")
    
    return {
        "seed": seed,
        "val_size": val_size,
        "stratified": stratified,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test": "official 10k",
        "counts": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": 10000,
        },
        "class_counts": {
            "train": _class_counts(labels, train_idx),
            "val": _class_counts(labels, val_idx),
        },
    }


def save_split(split: dict, path: pathlib.Path) -> None:
    """TODO (TV 1, A1-01): write JSON + print per-class counts for the report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as file:
        json.dump(split, file, indent=2)
        
    print(f"Saved split to: {path}")
    print(f"Seed: {split['seed']}")
    print(f"Stratified: {split['stratified']}")
    print(f"Counts: {split['counts']}")
    print(f"Train samples per class: {split['class_counts']['train']}")
    print(f"Validation samples per class: {split['class_counts']['val']}")


def load_split(path: pathlib.Path) -> dict:
    """TODO (TV 1, A1-01): read the JSON and sanity-check the counts."""
    path = pathlib.Path(path)

    with path.open(encoding="utf-8") as file:
        split = json.load(file)
        
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    
    if len(train_idx) != split["counts"]["train"]:
        raise ValueError("Saved train count does not match train indices.")
    
    if len(val_idx) != split["counts"]["val"]:
        raise ValueError("Saved validation count does not match validation indices.")
    
    if set(train_idx).intersection(val_idx):
        raise ValueError("Saved train and validation splits overlap.")
    
    return split

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
        
    dataset = FashionMNIST(
        root=config["data_dir"],
        train=True,
        download=False,
    )
    
    split = build_split(
        labels=dataset.targets,
        val_size=config["data"]["val_size"],
        seed=config["seed"],
        stratified=config["data"]["stratified"],
    )
    
    save_split(split, pathlib.Path(config["split_file"]))


if __name__ == "__main__":
    main()