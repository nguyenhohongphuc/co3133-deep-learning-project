"""Preprocessing and augmentation. Owner: TV 1. Task A1-01.

Handbook 3.1: the report needs cleaning, resize/normalization, augmentation,
input tensor shape and one example batch after preprocessing.
"""
from __future__ import annotations


def build_transforms(config: dict, train: bool):
    """Return the torchvision transform pipeline for train / eval.

    TODO (TV 1, A1-01):
      - normalization constants measured on the TRAIN split only (never the test set)
      - augmentation only when train=True; record what was used for the report
    """
    raise NotImplementedError
