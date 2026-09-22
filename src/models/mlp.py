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
        
        if hidden_sizes is None:
            hidden_sizes = [256]
            
        if not hidden_sizes:
            raise ValueError("MLP must contain at least one hidden layer.")
        
        activation_layers = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        
        activation_name = activation.lower()
        if activation_name not in activation_layers:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Choose one of {list(activation_layers)}."
            )
            
        activation_layer = activation_layers[activation_name]
        
        layers = []
        previous_features = in_features
        
        for hidden_features in hidden_sizes:
            layers.append(nn.Linear(previous_features, hidden_features))
            layers.append(activation_layer())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            previous_features = hidden_features
            
        layers.append(nn.Linear(previous_features, num_classes))
        
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.network(x)
        return logits