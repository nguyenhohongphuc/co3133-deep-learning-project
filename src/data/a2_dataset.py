"""A2 dataset + loaders. Owner: TV 1. Task A2-04.

Preparation only: main A2 experiments may not start before the proposal is approved
(handbook 7.5, 15.1). Split by the leakage-prevention unit named in the proposal (18).
"""
from __future__ import annotations

from torch.utils.data import DataLoader


def build_a2_loaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Same contract as src/data/loaders.build_loaders. TODO (TV 1, A2-04)."""
    raise NotImplementedError
