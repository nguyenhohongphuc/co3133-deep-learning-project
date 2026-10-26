"""CNN - your own design. Owner: TV 2. Task A1-09.

Handbook 11.1: design it yourselves (a pretrained model alone does not count) and
explain convolution, pooling and feature maps in the report.
"""
from __future__ import annotations

import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, channels: list[int]|None = None, kernel_size:int  = 3, num_classes = 10, dropout: float = 0.0, image_size: int = 28):
        super().__init__()
        self.in_channels = in_channels
        if channels is None:
            channels = [16, 32, 64]
        pool_layer = nn.MaxPool2d
        
        layers = []
        prev_channel = in_channels
        size = image_size
        for out_channels in channels:
            layers.append(nn.Conv2d(prev_channel, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(pool_layer(kernel_size=2, stride=2)) # này đang suy nghĩ nên để số mấy 
            prev_channel = out_channels
            size //= 2  
        self.feature = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_channel * size * size ,num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        logits = self.classifier(x)
        return logits


        