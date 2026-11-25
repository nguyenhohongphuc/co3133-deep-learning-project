"""ResNet50 (2048) + the team's mean-pooled DistilBERT (768)."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from transformers import DistilBertConfig, DistilBertModel

from .text_only import TextEncoder


class FusionClassifier(nn.Module):
    def __init__(self, model_name="distilbert/distilbert-base-uncased", *,
                 revision=None, image_weights="IMAGENET1K_V2", num_classes=12,
                 hidden_dim=256, dropout=0.2, text_config=None, text_encoder=None):
        super().__init__()
        weights = None if image_weights is None else models.ResNet50_Weights[image_weights]
        self.image_encoder = models.resnet50(weights=weights)
        self.image_dim = self.image_encoder.fc.in_features
        self.image_encoder.fc = nn.Identity()
        if text_encoder is not None:
            self.text_encoder = text_encoder
        elif text_config is not None:
            # Saved configuration + checkpoint weights: no Hub access at evaluation.
            backbone = DistilBertModel(DistilBertConfig.from_dict(text_config))
            self.text_encoder = TextEncoder(model_name, backbone=backbone)
        else:
            self.text_encoder = TextEncoder(model_name, revision=revision)
        self.text_dim = self.text_encoder.out_dim
        if self.text_dim != 768:
            raise ValueError(f"Expected the team's 768-dimensional DistilBERT, got {self.text_dim}")
        self.head = nn.Sequential(
            nn.Linear(self.image_dim + self.text_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )
        self._image_trainable = False
        self._text_trainable = 0
        self.configure_finetuning(False, 0)

    def encode_modalities(self, batch):
        image = self.image_encoder(batch["images"])
        text = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        return image, text

    def forward(self, batch):
        image, text = self.encode_modalities(batch)
        return self.head(torch.cat((image, text), dim=-1))

    def configure_finetuning(self, image_layer4: bool, text_last: int):
        layers = self.text_encoder.bert.transformer.layer
        if not 0 <= text_last <= len(layers):
            raise ValueError(f"text_last must be between 0 and {len(layers)}")
        self.image_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if image_layer4:
            self.image_encoder.layer4.requires_grad_(True)
        self.text_encoder.unfreeze_last(text_last)
        self._image_trainable = image_layer4
        self._text_trainable = text_last
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BN statistics and dropout stay fixed, including during head warm-up.
        self.image_encoder.eval()
        self.text_encoder.eval()
        if mode and self._image_trainable:
            self.image_encoder.layer4.train()
        if mode and self._text_trainable:
            for layer in self.text_encoder.bert.transformer.layer[-self._text_trainable:]:
                layer.train()
        return self


def build_fusion(cfg, *, pretrained=True, text_config=None):
    if not pretrained and text_config is None:
        raise ValueError("Offline construction requires an explicit saved text_config")
    m = cfg["model"]
    return FusionClassifier(
        m["name"], revision=m.get("revision"),
        image_weights=m["image_weights"] if pretrained else None,
        hidden_dim=m["hidden_dim"], dropout=m["dropout"], text_config=text_config,
    )
