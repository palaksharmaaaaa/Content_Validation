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
import os
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


_NORMALISE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _transforms():
    """(training, validation) preprocessing; training adds a flip and a small brightness/contrast jitter."""
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    train = transforms.Compose([resize, transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.1, contrast=0.1), transforms.ToTensor(), _NORMALISE])
    return train, transforms.Compose([resize, transforms.ToTensor(), _NORMALISE])


def _scan(folder: Path) -> List[Path]:
    """Supported image files under ``folder`` (none when it does not exist), in a stable order."""
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS) if folder.is_dir() else []


def _labelled(root: Path) -> List[Tuple[Path, int]]:
    """``root/ai_generated`` as label 0 and ``root/real`` as label 1."""
    return [(p, 0) for p in _scan(root / "ai_generated")] + [(p, 1) for p in _scan(root / "real")]


class ImageDataset(Dataset):
    """PyTorch Dataset for image authenticity classification."""

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # A truncated/corrupt file in a large in-place library must not abort a whole training run:
        # fall through to the next sample (bounded so an all-bad dataset still fails loudly).
        for offset in range(len(self.samples)):
            path, label = self.samples[(idx + offset) % len(self.samples)]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, label
            except Exception as exc:
                logger.warning("Skipping unreadable training image %s: %s", path.name, exc)
        raise RuntimeError("No readable images in dataset.")


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
        """Reads a dataset folder with ``ai_generated/`` and ``real/`` (split at random, fixed seed), or pre-split ``train/`` and ``val/``."""
        root = Path(dataset_dir)
        train_tf, val_tf = _transforms()
        rng = random.Random(42)                                   # a private generator: training never reseeds the process-wide one

        if (root / "train" / "ai_generated").is_dir():
            train_samples = _labelled(root / "train")
            val_samples = _labelled(root / "val") if (root / "val").is_dir() else []
            rng.shuffle(train_samples)
            rng.shuffle(val_samples)
        else:
            all_samples = _labelled(root)
            if not all_samples:
                raise ValueError(f"No valid image files found in {root}/ai_generated or {root}/real")
            rng.shuffle(all_samples)
            split_idx = int(len(all_samples) * (1.0 - val_split))
            train_samples, val_samples = all_samples[:split_idx], all_samples[split_idx:]

        train_loader = DataLoader(ImageDataset(train_samples, train_tf), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ImageDataset(val_samples, val_tf), batch_size=batch_size, shuffle=False) if val_samples else None
        logger.info("Prepared image dataset: %d train, %d val samples.", len(train_samples), len(val_samples))
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
        self.training_meta = {                                  # saved with the checkpoint: what the model was trained on
            "train_samples": len(train_loader.dataset) if train_loader is not None else 0,
            "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
            "unit": "images", "epochs_run": epochs, "initial_weights": "torchvision ImageNet-1K",
        }

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

        self.training_meta["final_val_accuracy"] = history["val_acc"][-1] if history["val_acc"] else None
        self.save_checkpoint()
        return history

    def loaders_from_samples(
        self,
        train_samples: List[Tuple[Path, int]],
        val_samples: List[Tuple[Path, int]],
        batch_size: int = 16,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Builds loaders that read images IN PLACE from their original locations (label 0=ai_generated,
        1=real) -- no dataset copy, no files written. Used with core.media_library's content-hash
        train/val partition so validation images can never be trained on.
        """
        train_tf, val_tf = _transforms()
        train_loader = DataLoader(ImageDataset(train_samples, train_tf), batch_size=batch_size, shuffle=True) if train_samples else None
        val_loader = DataLoader(ImageDataset(val_samples, val_tf), batch_size=batch_size, shuffle=False) if val_samples else None
        return train_loader, val_loader

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """Serializes trained weights and metadata."""
        save_path = path or self.checkpoint_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "architecture": self.architecture,
            "class_to_idx": {"ai_generated": 0, "real": 1},
            "meta": dict(getattr(self, "training_meta", {})),
        }
        partial = save_path.with_name(save_path.name + ".partial")
        torch.save(checkpoint, partial)
        os.replace(partial, save_path)                            # a crash mid-write never leaves a half-written checkpoint
        logger.info("Saved trained Image AI detector checkpoint to %s", save_path)

    def prepare_feature_bank(
        self,
        dataset_dir: Path | str,
        output_npz_path: Path | str,
        max_samples_per_class: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extracts compact 512-dim embeddings + 12-dim forensic vectors from an image dataset folder
        into a rebuildable .npz cache keyed by content hash. Source images are only read, never
        modified or deleted.
        """
        from image_detector.feature_store import FeatureStore

        dataset_path = Path(dataset_dir)
        ai_files = _scan(dataset_path / "ai_generated")
        real_files = _scan(dataset_path / "real")
        if max_samples_per_class:
            ai_files = ai_files[:max_samples_per_class]
            real_files = real_files[:max_samples_per_class]

        samples: List[Tuple[Any, int]] = [(p, 0) for p in ai_files] + [(p, 1) for p in real_files]
        if not samples:
            raise ValueError(f"No valid image files found in {dataset_path}")

        store = FeatureStore(checkpoint_path=self.checkpoint_path, device=str(self.device))
        return store.build_feature_bank(
            samples=samples,
            output_npz_path=output_npz_path,
        )

    def train_from_feature_bank(
        self,
        feature_npz_path: Path | str,
        epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.2,
        combine_forensics: bool = False,
    ) -> Dict[str, Any]:
        """
        Ultra-fast zero-retention training directly on precomputed .npz feature archives.
        Trains without reading any image files from disk.
        """
        from image_detector.feature_store import FeatureBankDataset, FeatureClassifierHead

        if combine_forensics:
            # The 524-feature head (512 embedding + 12 forensic measures) has no place in the checkpoint format the detector loads, so
            # training it would produce a model that is thrown away. Refuse instead of reporting a result nobody can use.
            raise ValueError("combine_forensics=True trains a head the detector cannot load; use combine_forensics=False")
        dataset = FeatureBankDataset(feature_npz_path, combine_forensics=combine_forensics)
        total_samples = len(dataset)
        if total_samples < 2:
            raise ValueError(f"Feature bank in {feature_npz_path} must contain at least 2 samples.")

        indices = list(range(total_samples))
        random.Random(42).shuffle(indices)

        split = int(total_samples * (1.0 - val_split))
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices) if val_indices else None

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False) if val_set else None

        in_dim = 524 if combine_forensics else 512
        hidden = self.model.fc[1].out_features if hasattr(self.model, "fc") else 64
        head = FeatureClassifierHead(in_features=in_dim, hidden_dim=hidden, num_classes=2).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)

        history: Dict[str, List[float]] = {"train_loss": [], "val_acc": []}

        for epoch in range(epochs):
            head.train()
            total_loss = 0.0
            total_count = 0

            for feats, labels in train_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = head(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                total_count += len(labels)

            avg_loss = total_loss / max(1, total_count)
            history["train_loss"].append(avg_loss)

            val_acc = 0.0
            if val_loader:
                head.eval()
                correct = 0
                val_total = 0
                with torch.no_grad():
                    for feats, labels in val_loader:
                        feats, labels = feats.to(self.device), labels.to(self.device)
                        preds = torch.argmax(head(feats), dim=1)
                        correct += (preds == labels).sum().item()
                        val_total += len(labels)
                val_acc = correct / max(1, val_total)
                history["val_acc"].append(val_acc)

            logger.info(
                "FeatureBank Epoch [%d/%d] - Loss: %.4f | Val Acc: %.2f%%",
                epoch + 1, epochs, avg_loss, val_acc * 100.0
            )

        # Splice the trained head onto the SAME backbone that produced the cached embeddings
        # (the existing checkpoint, if any) -- otherwise the head would sit on mismatched features.
        if not combine_forensics and hasattr(self.model, "fc"):
            if self.checkpoint_path.is_file():
                state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state["model_state_dict"])
            self.model.fc.load_state_dict(head.head.state_dict())
            self.training_meta = {"train_samples": len(train_indices), "val_samples": len(val_indices), "unit": "images (cached embeddings)",
                                  "epochs_run": epochs, "initial_weights": "torchvision ImageNet-1K",
                                  "final_val_accuracy": history["val_acc"][-1] if history["val_acc"] else None}
            self.save_checkpoint()

        return history


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
