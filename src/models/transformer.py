"""Transformer encoder classifier. Owner: TV 3. Task A1-12.

Handbook 11.1: token embedding / projection + positional encoding are required, and
the report must explain what goes into and comes out of attention.
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
@register("transformer")
class PatchTransformer(nn.Module):
    def __init__(self, num_classes: int = 10, sequence: str = "patches", patch_size: int = 7,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 positional_encoding: str = "learned"):
        super().__init__()
        # 1. Validate configuration
        if sequence not in {"rows", "cols", "patches"}:
            raise ValueError(f"Unsupported sequence: {sequence}")

        if sequence == "patches":
            if patch_size <= 0 or 28 % patch_size != 0:
                raise ValueError("patch_size must divide 28")

            input_size = patch_size * patch_size
            num_tokens = (28 // patch_size) ** 2
        else:
            input_size = 28
            num_tokens = 28

        if d_model <= 0 or nhead <= 0 or d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.sequence = sequence
        self.patch_size = patch_size

        # 2. Token projection: F -> d_model (49 -> 128)
        self.patch_embed = nn.Linear(input_size, d_model)

        # 3. Learnable CLS token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        # 4. Positional encoding, including CLS position
        self.num_tokens = num_tokens + 1

        if positional_encoding == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_tokens, d_model)
            )

        elif positional_encoding == "sinusoidal":
            position = torch.arange(
                self.num_tokens, dtype=torch.float32
            ).unsqueeze(1)

            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )

            pe = torch.zeros(self.num_tokens, d_model)

            pe[:, 0::2] = torch.sin(position * div_term)

            pe[:, 1::2] = torch.cos(
                position * div_term[:pe[:, 1::2].shape[1]]
            )

            self.register_buffer(
                "pos_embed", pe.unsqueeze(0)
            )

        else:
            raise ValueError(
                f"Unsupported positional encoding: {positional_encoding}"
            )

        self.dropout = nn.Dropout(dropout)

        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 6. Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        # 7. Initialize learned embeddings
        nn.init.normal_(self.cls_token, std=0.02)

        if positional_encoding == "learned":
            nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Convert image to sequence
        # [B, 1, 28, 28] -> [B, T, F]
        x = to_sequence(x, self.sequence, self.patch_size)

        # 2. Project each token to d_model dimensions
        # [B, T, F] -> [B, T, d_model]
        x = self.patch_embed(x)

        # 3. Get batch size
        B = x.size(0)

        # 4. Expand the CLS token for each image
        # [1, 1, d_model] -> [B, 1, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # 5. Prepend CLS token to the sequence
        # [B, T, d_model] -> [B, T+1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)

        # 6. Add positional encoding and dropout
        x = self.dropout(x + self.pos_embed)

        # 7. Pass tokens through Transformer encoder
        # [B, T+1, d_model] -> [B, T+1, d_model]
        x = self.encoder(x)

        # 8. Extract the CLS representation
        # [B, T+1, d_model] -> [B, d_model]
        features = self.norm(x[:, 0, :])

        # 9. Predict class logits
        # [B, d_model] -> [B, num_classes]
        logits = self.classifier(features)

        return logits

