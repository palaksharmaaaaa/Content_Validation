"""
audio_detector.models.backbone: PyTorch neural architecture for acoustic AI synthesis classification.
Self-contained neural module with zero outside dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn


class AudioClassifierNet(nn.Module):
    """Multi-layer perceptron for acoustic AI synthesis classification."""

    def __init__(self, in_features: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_audio_classifier(
    checkpoint_path: Optional[str | Path] = None,
    in_features: int = 5,
    hidden: int = 32,
    device: Optional[torch.device] = None,
) -> AudioClassifierNet:
    """Instantiates and optionally loads pre-trained weights for AudioClassifierNet."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifierNet(in_features=in_features, hidden=hidden)
    
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.is_file():
            state = torch.load(ckpt, map_location=dev, weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            elif isinstance(state, nn.Module):
                model = state

    model.to(dev)
    model.eval()
    return model
