"""LSTM / GRU over an image sequence. Owner: TV 3. Tasks A1-10 and A1-15 (ablation).

Handbook 11.1: represent each image as a sequence of rows, columns or patches, and
document the timestep definition, the input size and the hidden representation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.registry import register


def to_sequence(x: torch.Tensor, mode: str, patch_size: int = 7) -> torch.Tensor:
    """[B, 1, 28, 28] -> [B, T, F] for mode in {rows, cols, patches}.

    rows:    T=28, F=28      cols: T=28, F=28
    patches: T=(28/p)^2, F=p*p
    Shared by the RNN and the Transformer. TODO (TV 3, A1-10).
    """
    raise NotImplementedError


@register("rnn")
class SequenceRNN(nn.Module):
    def __init__(self, num_classes: int = 10, cell: str = "lstm", sequence: str = "rows",
                 patch_size: int = 7, hidden_size: int = 128, num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        raise NotImplementedError  # TODO (TV 3, A1-10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
