from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "ai_detector.pt"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AIImageDetector:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        self.device = DEVICE
        self.checkpoint_path = Path(checkpoint_path)
        self.model: nn.Module | None = None
        self.class_to_idx: dict[str, int] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Trained detector not found: {self.checkpoint_path}. "
                "Run: python models\\train_ai_detector.py"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model
        self.class_to_idx = checkpoint["class_to_idx"]

    def predict(self, image_path: str | Path) -> dict:
        self.load()
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Could not open image: {exc}") from exc

        inputs = self.transform(image).unsqueeze(0).to(self.device)
        assert self.model is not None
        with torch.no_grad():
            probabilities = torch.softmax(self.model(inputs), dim=1)[0]

        ai_index = self.class_to_idx["ai_generated"]
        real_index = self.class_to_idx["real"]
        ai_probability = float(probabilities[ai_index].item())
        real_probability = float(probabilities[real_index].item())
        confidence = max(ai_probability, real_probability)

        if ai_probability >= 0.50:
            prediction = "LIKELY AI-GENERATED"
        elif real_probability >= 0.70:
            prediction = "LIKELY REAL"
        else:
            prediction = "UNCERTAIN"

        return {
            "prediction": prediction,
            "ai_probability": round(ai_probability, 4),
            "real_probability": round(real_probability, 4),
            "confidence": round(confidence, 4),
            "top_label": "ai_generated" if ai_probability >= real_probability else "real",
            "device": str(self.device),
            "model": str(self.checkpoint_path),
        }
