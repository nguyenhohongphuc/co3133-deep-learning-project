"""Download the datasets. Owner: TV 1. Task A1-01.

    python scripts/prepare_data.py --dataset fashion-mnist --out data/

Handbook 10: Fashion-MNIST carries the primary results; MNIST is for debugging only.
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fashion-mnist", choices=["fashion-mnist", "mnist"])
    p.add_argument("--out", default="data/")
    args = p.parse_args()
    raise NotImplementedError  # TODO (TV 1, A1-01): torchvision download + print counts


if __name__ == "__main__":
    main()
