"""Fashion-MNIST Dataset. Owner: TV 1. Task A1-04.

Handbook 10: primary results on Fashion-MNIST; MNIST only for debugging.
"""
from __future__ import annotations

from torch.utils.data import Dataset


class FashionMNISTSubset(Dataset):
    """Wraps torchvision Fashion-MNIST and keeps only the indices of one split.

    TODO (TV 1, A1-04): __init__(root, indices, transform), __len__, __getitem__.
    __getitem__ returns (image tensor [1, 28, 28], label int).
    """

    def __init__(self, root, indices, transform=None):
        raise NotImplementedError
