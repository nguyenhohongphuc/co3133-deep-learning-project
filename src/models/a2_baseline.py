"""A2 simple baseline. Owner: TV 3. Task A2-06. Handbook 20, item 1."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.registry import register


@register("a2_baseline")
class A2Baseline(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        raise NotImplementedError  # TODO (TV 3, A2-06)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
