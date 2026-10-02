"""
audio_detector.trainer: Dedicated, self-sufficient trainer for Audio AI Detection models.
Supports:
1. Feature extraction and dataset preparation from audio recordings.
2. Neural classification head training on STFT acoustic feature embeddings.
3. Optimization of vocoder cutoff and Wiener flatness discriminants.
4. CLI and programmatic execution.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from audio_detector.config import DEFAULT_AUDIO_CHECKPOINT, MODELS_DIR, SUPPORTED_EXTENSIONS
from audio_detector.features import compute_spectral_features
from audio_detector.models.backbone import AudioClassifierNet
from audio_detector.validator import AudioValidator

logger = logging.getLogger("audio_detector.trainer")


class AcousticFeatureDataset(Dataset):
    """Dataset of acoustic feature vectors (vocoder cutoff, flatness, silence ratio, HF ratio)."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class AudioDetectorTrainer:
    """
    Independent trainer for Audio AI Detection.
    Extracts acoustic spectral features from authentic speech and synthetic voice clones,
    and trains discriminative neural classifiers.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_AUDIO_CHECKPOINT
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validator = AudioValidator()
        self.model = AudioClassifierNet().to(self.device)

    def extract_features_from_file(self, audio_path: Path | str) -> Optional[np.ndarray]:
        """Extracts 5-dimensional acoustic feature vector."""
        samples, sr, _ = self.validator.extract_pcm_samples(audio_path)
        if samples is None or len(samples) < 1000:
            return None
        f = compute_spectral_features(samples, sr)
        return np.array([
            float(f["has_vocoder_cutoff"]),
            f["cutoff_freq_hz"] / 10000.0,
            f["spectral_flatness"] * 100.0,
            f["digital_silence_ratio"],
            f["high_freq_ratio"],
        ], dtype=np.float32)

    def prepare_data_from_directory(
        self, dataset_dir: Path | str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scans dataset with 'ai_generated' and 'real' subdirectories."""
        dataset_path = Path(dataset_dir)
        ai_dir = dataset_path / "ai_generated"
        real_dir = dataset_path / "real"

        ai_files = [p for p in ai_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if ai_dir.exists() else []
        real_files = [p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if real_dir.exists() else []

        X: List[np.ndarray] = []
        y: List[int] = []

        for p in ai_files:
            vec = self.extract_features_from_file(p)
            if vec is not None:
                X.append(vec)
                y.append(0)  # AI

        for p in real_files:
            vec = self.extract_features_from_file(p)
            if vec is not None:
                X.append(vec)
                y.append(1)  # Real

        if not X:
            raise ValueError(f"No valid audio samples found in {dataset_path}")

        logger.info(
            "Extracted acoustic features from %d audio recordings (%d AI, %d Real).",
            len(X),
            y.count(0),
            y.count(1),
        )
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 15,
        batch_size: int = 16,
        lr: float = 1e-3,
        val_split: float = 0.2,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Trains acoustic classifier model. By default holds out a random `val_split` of X; if an
        explicit (X_val, y_val) set is given (e.g. the content-hash validation partition from
        core.media_library), all of X is used for training and that set is used for validation.
        """
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_train, y_train = X, y
            X_hold, y_hold = X_val, y_val
        else:
            n_total = len(X)
            indices = np.random.permutation(n_total)
            val_size = int(n_total * val_split) if n_total > 5 else 0
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_hold, y_hold = X[val_idx], y[val_idx]
        val_size = len(X_hold)

        train_dataset = AcousticFeatureDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        history: Dict[str, List[float]] = {"loss": [], "val_accuracy": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for feats, labels in train_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)

            avg_loss = total_loss / max(1, len(X_train))
            history["loss"].append(avg_loss)

            val_acc = 0.0
            if val_size > 0:
                self.model.eval()
                with torch.no_grad():
                    val_feats = torch.tensor(X_hold, dtype=torch.float32, device=self.device)
                    val_labels = torch.tensor(y_hold, dtype=torch.long, device=self.device)
                    preds = self.model(val_feats).argmax(dim=-1)
                    val_acc = float((preds == val_labels).float().mean().item())
                history["val_accuracy"].append(val_acc)
                logger.info(
                    "Audio Trainer Epoch [%d/%d] - Loss: %.4f - Val Acc: %.2f%%",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    val_acc * 100.0,
                )
            else:
                logger.info("Audio Trainer Epoch [%d/%d] - Loss: %.4f", epoch + 1, epochs, avg_loss)

        self.save_checkpoint()
        return history

    def arrays_from_samples(
        self, samples: List[Tuple[Path, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts acoustic feature vectors from audio files IN PLACE (label 0=ai_generated, 1=real).
        Files that can't be decoded are skipped. Returns empty arrays if nothing was usable.
        """
        X: List[np.ndarray] = []
        y: List[int] = []
        for path, label in samples:
            vec = self.extract_features_from_file(path)
            if vec is not None:
                X.append(vec)
                y.append(label)
        if not X:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    def save_checkpoint(self, path: Optional[Path | str] = None) -> None:
        """Serializes audio classifier checkpoint."""
        save_path = Path(path) if path else self.checkpoint_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "in_features": 5,
            "class_to_idx": {"ai_generated": 0, "real": 1},
        }
        torch.save(checkpoint, save_path)
        logger.info("Saved trained Audio AI detector checkpoint to %s", save_path)

    def export_feature_dataset(
        self,
        dataset_dir: Path | str,
        output_npz_path: Path | str,
    ) -> Dict[str, Any]:
        """
        Extracts acoustic features from audio recordings and saves to compressed .npz archive.
        """
        output_path = Path(output_npz_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(dataset_dir)
        ai_dir = dataset_path / "ai_generated"
        real_dir = dataset_path / "real"

        ai_files = [p for p in ai_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if ai_dir.exists() else []
        real_files = [p for p in real_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS] if real_dir.exists() else []

        X: List[np.ndarray] = []
        y: List[int] = []
        total_source_bytes = 0

        for p in ai_files + real_files:
            label = 0 if p in ai_files else 1
            if p.is_file():
                total_source_bytes += p.stat().st_size
            vec = self.extract_features_from_file(p)
            if vec is not None:
                X.append(vec)
                y.append(label)

        if not X:
            return {"success": False, "error": "No valid audio samples found."}

        arr_X = np.array(X, dtype=np.float32)
        arr_y = np.array(y, dtype=np.int64)

        np.savez_compressed(str(output_path), features=arr_X, labels=arr_y)
        npz_bytes = output_path.stat().st_size
        savings_pct = (((total_source_bytes - npz_bytes) / max(1, total_source_bytes)) * 100.0) if total_source_bytes > 0 else 0.0

        return {
            "success": True,
            "output_path": str(output_path),
            "samples_processed": len(y),
            "source_size_mb": round(total_source_bytes / (1024 * 1024), 2),
            "compressed_size_mb": round(npz_bytes / (1024 * 1024), 3),
            "space_savings_pct": round(savings_pct, 1),
        }

    def train_from_feature_bank(
        self,
        npz_path: Path | str,
        epochs: int = 15,
        batch_size: int = 16,
        lr: float = 1e-3,
        val_split: float = 0.2,
    ) -> Dict[str, Any]:
        """Trains audio model directly from a .npz feature bank with zero raw audio on disk."""
        data = np.load(str(npz_path))
        X = data["features"]
        y = data["labels"]
        return self.train(X, y, epochs=epochs, batch_size=batch_size, lr=lr, val_split=val_split)


def main():
    parser = argparse.ArgumentParser(description="Train Audio AI Detection Model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to audio dataset with real/ and ai_generated/")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=str, default=str(DEFAULT_AUDIO_CHECKPOINT), help="Output checkpoint file")
    args = parser.parse_args()

    trainer = AudioDetectorTrainer(checkpoint_path=Path(args.output))
    X, y = trainer.prepare_data_from_directory(args.dataset)
    trainer.train(X, y, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
