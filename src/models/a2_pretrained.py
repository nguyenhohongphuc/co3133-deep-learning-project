"""A2 pretrained / modern model + fine-tuning protocol. Owner: TV 2. Task A2-07.

Handbook 20: at least one pretrained or modern model AND a clear fine-tuning procedure
(which layers are frozen, learning rates, schedule). freeze_backbone is the factor of
the controlled experiment in A2-11.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.registry import register


@register("a2_pretrained")
class A2Pretrained(nn.Module):
    def __init__(self, num_classes: int = 10, backbone: str = "resnet18",
                 pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        raise NotImplementedError  # TODO (TV 2, A2-07)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
