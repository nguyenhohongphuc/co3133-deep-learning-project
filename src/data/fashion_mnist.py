"""Fashion-MNIST Dataset. Owner: TV 1. Task A1-04.

Handbook 10: primary results on Fashion-MNIST; MNIST only for debugging.
"""
from __future__ import annotations

from torch.utils.data import Dataset
from pathlib import Path
from typing import Sequence

from torchvision.datasets import FashionMNIST

class FashionMNISTSubset(Dataset):
    """Wraps torchvision Fashion-MNIST and keeps only the indices of one split.

    TODO (TV 1, A1-04): __init__(root, indices, transform), __len__, __getitem__.
    __getitem__ returns (image tensor [1, 28, 28], label int).
    """

    def __init__(
        self,
        root: str | Path,
        indices: Sequence[int],
        transform=None,
        train: bool = True,
    ):
        self.dataset = FashionMNIST(
            root=str(root),
            train=train,
            download=False,
        )
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        original_index = self.indices[index]

        # image is a PIL image here; label is an integer class ID.
        image, label = self.dataset[original_index]

        if self.transform is not None:
            image = self.transform(image)

        return image, int(label)