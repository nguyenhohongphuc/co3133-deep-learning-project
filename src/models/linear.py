"""Linear / softmax classifier. Owner: TV 1. Task A1-06.

Handbook 11.1: flatten the image, one Linear layer producing LOGITS, cross-entropy loss,
and NO softmax before CrossEntropyLoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.registry import register


@register("linear")
class LinearClassifier(nn.Module):
    """[B, 1, 28, 28] -> [B, num_classes] logits."""

    def __init__(self, num_classes: int = 10, in_features: int = 28 * 28):
        super().__init__()
        raise NotImplementedError  # TODO (TV 1, A1-06)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
