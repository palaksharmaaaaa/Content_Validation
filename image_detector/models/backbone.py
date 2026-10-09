"""
image_detector.models.backbone: PyTorch neural architectures for image authenticity classification.
Self-contained neural module with zero outside dependencies.
"""
from __future__ import annotations

from torch import nn
from torchvision import models


def build_image_classifier(
    architecture: str = "resnet18",
    pretrained: bool = True,
    num_classes: int = 2,
    hidden_dim: int = 64,
) -> nn.Module:
    """
    Constructs a PyTorch classification backbone with transfer learning weights
    and a sequential dropout-linear-relu-linear head matching ai_detector.pt.
    """
    arch = architecture.lower()
    if arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    elif arch == "mobilenet_v3":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    else:
        raise ValueError(f"unknown architecture {architecture!r}; expected resnet18, resnet50 or mobilenet_v3")

    return model
