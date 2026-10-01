"""
image_detector.trainer: Dedicated, self-sufficient trainer for Image AI Detection models.
Supports:
1. Building PyTorch neural backbones (ResNet-18, ResNet-50, MobileNetV3).
2. Training and fine-tuning on custom image datasets (real vs ai_generated).
3. Automatic dataset validation and metric evaluation.
4. CLI and programmatic execution.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from image_detector.config import DEFAULT_CHECKPOINT, IMAGE_SIZE, SUPPORTED_EXTENSIONS
from image_detector.models.backbone import build_image_classifier

logger = logging.getLogger("image_detector.trainer")


class ImageDataset(Dataset):
    """PyTorch Dataset for image authenticity classification."""

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label


class ImageDetectorTrainer:
    """
    Self-sufficient trainer for Image AI Detection.
    Handles data ingestion, transfer learning, fine-tuning, and model serialization.
    """

    def __init__(
        self,
        architecture: str = "resnet18",
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.architecture = architecture.lower()
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_image_classifier(architecture=self.architecture, pretrained=True)
        self.model.to(self.device)

    def prepare_data(
        self, dataset_dir: Path | str, batch_size: int = 16, val_split: float = 0.2
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Scans dataset directory formatted with 'ai_generated' and 'real' subdirectories, or pre-split 'train' and 'val' subdirectories."""
        dataset_path = Path(dataset_dir)
        
        train_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Check for pre-split structure (train/ and val/)
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"
        if train_dir.exists() and (train_dir / "ai_generated").exists():
            train_ai = [p for p in (train_dir / "ai_generated").rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
            train_real = [p for p in (train_dir / "real").rglob("*") if (train_dir / "real").exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
            train_samples = [(p, 0) for p in train_ai] + [(p, 1) for p in train_real]

            val_samples = []
            if val_dir.exists():
                val_ai = [p for p in (val_dir / "ai_generated").rglob("*") if (val_dir / "ai_generated").exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
                val_real = [p for p in (val_dir / "real").rglob("*") if (val_dir / "real").exists() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
                val_samples = [(p, 0) for p in val_ai] + [(p, 1) for p in val_real]

            random.seed(42)
            random.shuffle(train_samples)
            if val_samples:
                random.shuffle(val_samples)

            train_loader = DataLoader(ImageDataset(train_samples, train_transform), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(ImageDataset(val_samples, val_transform), batch_size=batch_size, shuffle=False) if val_samples else None
            logger.info("Prepared Pre-Split Image Dataset: %d train, %d val samples.", len(train_samples), len(val_samples))
            return train_loader, val_loader

        ai_dir = dataset_path / "ai_generated"
        real_dir = dataset_path / "real"

        ai_files = [p for p in ai_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if ai_dir.exists() else []
        real_files = [p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if real_dir.exists() else []

        all_samples = [(p, 0) for p in ai_files] + [(p, 1) for p in real_files]
        if not all_samples:
            raise ValueError(f"No valid image files found in {dataset_path}/ai_generated or {dataset_path}/real")

        random.seed(42)
        random.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1.0 - val_split))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        train_loader = DataLoader(ImageDataset(train_samples, train_transform), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ImageDataset(val_samples, val_transform), batch_size=batch_size, shuffle=False) if val_samples else None

        logger.info("Prepared Image Dataset: %d train, %d val samples.", len(train_samples), len(val_samples))
        return train_loader, val_loader

    def train(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        lr: float = 1e-4,
        dataset_root: Optional[Path | str] = None,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """Runs the training and evaluation loop. Supports both direct DataLoaders or dataset_root path."""
        if train_loader is None:
            if dataset_root is None:
                raise ValueError("Either train_loader or dataset_root must be provided to train().")
            train_loader, val_loader = self.prepare_data(dataset_root, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        history = {"train_loss": [], "val_acc": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

            avg_train_loss = total_loss / max(1, total_samples)
            history["train_loss"].append(avg_train_loss)

            val_acc = 0.0
            if val_loader:
                self.model.eval()
                correct = 0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        val_total += len(labels)
                val_acc = correct / max(1, val_total)
                history["val_acc"].append(val_acc)

            logger.info("Epoch [%d/%d] - Train Loss: %.4f | Val Acc: %.2f%%", epoch + 1, epochs, avg_train_loss, val_acc * 100.0)

        self.save_checkpoint()
        return history

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """Serializes trained weights and metadata."""
        save_path = path or self.checkpoint_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "architecture": self.architecture,
            "class_to_idx": {"ai_generated": 0, "real": 1},
        }
        torch.save(checkpoint, save_path)
        logger.info("Saved trained Image AI detector checkpoint to %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="Train Image AI Detection Model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory with real/ and ai_generated/")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default=str(DEFAULT_CHECKPOINT), help="Output checkpoint file")
    args = parser.parse_args()

    trainer = ImageDetectorTrainer(checkpoint_path=Path(args.output))
    train_loader, val_loader = trainer.prepare_data(args.dataset, batch_size=args.batch_size)
    trainer.train(train_loader, val_loader, epochs=args.epochs, lr=args.lr)


if __name__ == "__main__":
    main()
