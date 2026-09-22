"""Preprocessing and augmentation. Owner: TV 1. Task A1-01.

Handbook 3.1: the report needs cleaning, resize/normalization, augmentation,
input tensor shape and one example batch after preprocessing.
"""
from __future__ import annotations
from torchvision import transforms

def build_transforms(config: dict, train: bool):
    """Return the torchvision transform pipeline for train / eval.

    TODO (TV 1, A1-01):
      - normalization constants measured on the TRAIN split only (never the test set)
      - augmentation only when train=True; record what was used for the report
    """
    data_config = config["data"]
    steps = [transforms.ToTensor()]

    augmentation = data_config.get("augmentation", "none")

    if train and augmentation != "none":
        raise ValueError(
            f"Unsupported augmentation setting: {augmentation}. "
            "Add it explicitly and document it before using it."
        )

    if data_config.get("normalize", False):
        mean = data_config["mean"]
        std = data_config["std"]
        steps.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(steps)
