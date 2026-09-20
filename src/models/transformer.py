"""Transformer encoder classifier. Owner: TV 3. Task A1-12.

Handbook 11.1: token embedding / projection + positional encoding are required, and
the report must explain what goes into and comes out of attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.registry import register


@register("transformer")
class PatchTransformer(nn.Module):
    def __init__(self, num_classes: int = 10, sequence: str = "patches", patch_size: int = 7,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 positional_encoding: str = "learned"):
        super().__init__()
        raise NotImplementedError  # TODO (TV 3, A1-12): patch embed + CLS token + pos. encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
