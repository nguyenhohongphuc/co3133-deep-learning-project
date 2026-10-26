"""LSTM / GRU over an image sequence. Owner: TV 3. Tasks A1-10 and A1-15 (ablation).

Handbook 11.1: represent each image as a sequence of rows, columns or patches, and
document the timestep definition, the input size and the hidden representation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.common.registry import register


def to_sequence(x: torch.Tensor, mode: str, patch_size: int = 7) -> torch.Tensor:
    """[B, 1, 28, 28] -> [B, T, F] for rows, cols, patches."""

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected x to be a torch.Tensor, got {type(x)}")

    if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
        raise ValueError(f"Expected x to have shape [B, 1, 28, 28], got {x.shape}")

    x = x.squeeze(1)  # [B, 28, 28]

    if mode == "rows":
        return x.contiguous()

    elif mode == "cols":
        return x.transpose(1, 2).contiguous()

    elif mode == "patches":
        if patch_size <= 0 or 28 % patch_size != 0:
            raise ValueError("patch_size must divide 28")

        patches = x.unfold(1, patch_size, patch_size) \
                   .unfold(2, patch_size, patch_size)

        return patches.contiguous().view(
            x.size(0), -1, patch_size * patch_size
        )

    else:
        raise ValueError(f"Unsupported sequence mode: {mode}")


@register("rnn")
class SequenceRNN(nn.Module):
    def __init__(self, 
                 num_classes: int = 10, 
                 cell: str = "lstm", 
                 sequence: str = "rows",
                 patch_size: int = 7, 
                 hidden_size: int = 128, 
                 num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()

        if cell.lower() == "lstm":
            rnn_class = nn.LSTM
        elif cell.lower() == "gru":
            rnn_class = nn.GRU
        else:
            raise ValueError(f"Unknown cell type: {cell}. Expected 'lstm' or 'gru'.")

        if sequence == "patches":
            if patch_size <= 0 or 28 % patch_size != 0:
                raise ValueError("patch_size must divide 28")

            input_size = patch_size * patch_size
        else:
            input_size = 28

        # Lưu cấu hình vào object
        self.sequence = sequence
        self.patch_size = patch_size
        self.bidirectional = bidirectional

        # khởi tạo RNN-T
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(output_size, num_classes)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## [B, 1, 28, 28] -> [B, T, F]
        x = to_sequence(x, self.sequence, self.patch_size)

        # Chạy rnnt
        output, hidden = self.rnn(x)

        # Lấy hidden representation của layer cuối
        # LSTM trả về (h_n, c_n); GRU trả về h_n
        h_n = hidden[0] if isinstance(hidden, tuple) else hidden

        if self.bidirectional:
            features = torch.cat(
                [h_n[-2], h_n[-1]],
                dim=1
            )
        else:
            features = h_n[-1]

        # [B, hidden_size * directions] -> [B, num_classes]
        logits = self.classifier(features)

        return logits
        raise NotImplementedError
