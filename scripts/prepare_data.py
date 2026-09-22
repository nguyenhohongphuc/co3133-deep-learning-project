"""Download the datasets. Owner: TV 1. Task A1-01.

    python scripts/prepare_data.py --dataset fashion-mnist --out data/

Handbook 10: Fashion-MNIST carries the primary results; MNIST is for debugging only.
"""
from __future__ import annotations

import argparse


def main() -> None:
    from pathlib import Path
    from torchvision.datasets import FashionMNIST, MNIST
    
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashion-mnist", choices=["fashion-mnist", "mnist"])
    p.add_argument("--out", default="data/")
    args = p.parse_args()
    
    dataset_cls = FashionMNIST if args.dataset == "fashion-mnist" else MNIST
    root = Path(args.out)

    train_dataset = dataset_cls(root=root, train=True, download=True)
    test_dataset = dataset_cls(root=root, train=False, download=True)

    print(f"Dataset: {args.dataset}")
    print(f"Storage path: {root.resolve()}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Official test samples: {len(test_dataset)}")
    print(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"Raw image shape: {train_dataset.data.shape}")


if __name__ == "__main__":
    main()
