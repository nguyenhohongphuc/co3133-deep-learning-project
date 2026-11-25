from torch import nn
from torchvision import models

def build_model(cfg, n_classes, pretrained=True):
    options = {'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT), 'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT)}
    factory, weights = options[cfg.backbone]
    model = factory(weights=weights if pretrained else None)
    dim = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(cfg.dropout),nn.Linear(dim,n_classes))
    return model
