"""CNN - your own design. Owner: TV 2. Task A1-09.

Handbook 11.1: design it yourselves (a pretrained model alone does not count) and
explain convolution, pooling and feature maps in the report.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.registry import register


@register("cnn")
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, channels: list[int] | None = None,
                 kernel_size: int = 3, pool: str = "max", dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError  # TODO (TV 3, A1-09)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
