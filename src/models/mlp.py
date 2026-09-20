"""MLP. Owner: TV 1. Task A1-07.

Handbook 11.1: at least one hidden layer; explain the activation and any regularization.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.registry import register


@register("mlp")
class MLP(nn.Module):
    def __init__(self, num_classes: int = 10, in_features: int = 28 * 28,
                 hidden_sizes: list[int] | None = None, activation: str = "relu",
                 dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError  # TODO (TV 1, A1-07)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
