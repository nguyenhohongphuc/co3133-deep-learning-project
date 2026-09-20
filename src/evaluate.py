"""Evaluate a checkpoint and write every required output. Owner: TV 2.

    python -m src.evaluate --config configs/a1/cnn.yaml --checkpoint checkpoints/a1_cnn_best.pt

Produces the handbook 12.1 outputs: accuracy, macro-F1, parameter count, training and
inference time, confusion matrix, correct / incorrect examples.
"""
from __future__ import annotations

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    return p.parse_args()


def main() -> None:
    """TODO (TV 2, A1-11)."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
