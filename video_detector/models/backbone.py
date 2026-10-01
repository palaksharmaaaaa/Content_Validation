"""
video_detector.models.backbone: PyTorch neural architecture for video temporal continuity classification.
Self-contained neural module with zero outside dependencies.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class VideoTemporalTransitionModel(nn.Module):
    """
    Temporal transition discriminator that classifies inter-frame optical discrepancies
    as natural physical motion vs synthetic diffusion warping.
    """

    def __init__(self, architecture: str = "resnet18", pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_video_temporal_model(pretrained: bool = True) -> nn.Module:
    return VideoTemporalTransitionModel(pretrained=pretrained)
