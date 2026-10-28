"""Model cho A2: CNN baseline và ResNet-50 pretrained.

CNN phát triển từ src/models/cnn.py của A1 (giữ nguyên file A1). Thay đổi cho ảnh RGB 224 px:
  - 3 kênh vào, 30 lớp ra, mặc định 4 block thay vì 2;
  - Global Average Pooling trước lớp Linear. Với 2 block + Flatten như A1, ở 224 px lớp Linear
    sẽ có ~6 M tham số (56*56*64*30) và mỗi đầu ra chỉ nhìn được vùng ~10 px.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, channels: list[int] | None = None, kernel_size: int = 3,
                 num_classes: int = 30, dropout: float = 0.0):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        layers = []
        prev_channel = in_channels
        for out_channels in channels:
            layers.append(nn.Conv2d(prev_channel, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channel = out_channels
        self.feature = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev_channel, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature(x))


class ResNet50(nn.Module):
    """ResNet-50 ImageNet (torchvision IMAGENET1K_V2), thay lớp fc 1000 lớp bằng fc num_classes.

    freeze_backbone=True: chỉ lớp fc mới được train. Backbone để ở chế độ eval để BatchNorm
    giữ nguyên thống kê ImageNet (nếu không, running mean/var vẫn bị cập nhật dù không có gradient).
    """

    def __init__(self, num_classes: int = 30, freeze_backbone: bool = True, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.net = tv_models.resnet50(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for name, p in self.net.named_parameters():
                p.requires_grad = name.startswith("fc.")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone and mode:
            for name, module in self.net.named_children():
                if name != "fc":
                    module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


MODELS = {
    "cnn": SmallCNN,
    "resnet50_frozen": ResNet50,
    "resnet50_full": ResNet50,
}


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
