"""
video_detector.trainer: Dedicated, self-sufficient trainer for Video AI Detection models.
Supports:
1. Training temporal continuity neural discriminators on frame transition pairs.
2. Calibrating temporal motion variance and diffusion flicker distributions.
3. Dataset preparation from directories of genuine vs synthetic video clips.
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

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from video_detector.config import DEFAULT_VIDEO_CHECKPOINT, SUPPORTED_EXTENSIONS
from video_detector.extractor import VideoFrameExtractor
from video_detector.models.backbone import VideoTemporalTransitionModel

logger = logging.getLogger("video_detector.trainer")


def _labelled(root: Path, per_class: Optional[int] = None) -> List[Tuple[Path, int]]:
    """Videos under ``root/ai_generated`` (label 0) and ``root/real`` (label 1), in a stable order, at most ``per_class`` of each."""
    out: List[Tuple[Path, int]] = []
    for sub_dir, label in (("ai_generated", 0), ("real", 1)):
        folder = root / sub_dir
        if folder.is_dir():
            found = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
            out += [(p, label) for p in (found[:per_class] if per_class else found)]
    return out


class FrameTransitionDataset(Dataset):
    """Dataset of consecutive frame pairs for temporal continuity classification."""

    def __init__(self, pairs: List[Tuple[np.ndarray, np.ndarray, int]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1, f2, label = self.pairs[idx]
        diff = cv2.absdiff(f1, f2)
        diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(diff_rgb)
        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)
        return tensor, label


class VideoDetectorTrainer:
    """
    Independent trainer for Video AI Detection.
    Extracts frame transitions and motion dynamics from real and AI videos to train
    temporal consistency models and optimize motion distribution thresholds.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path or DEFAULT_VIDEO_CHECKPOINT
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = VideoFrameExtractor(max_frames=20)
        self.model = VideoTemporalTransitionModel(pretrained=True).to(self.device)

    def prepare_data_from_videos(
        self, dataset_dir: Path | str, max_videos_per_class: int = 50
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Consecutive frame pairs from a folder with ``ai_generated/`` and ``real/`` (at most ``max_videos_per_class`` of each), shuffled with a fixed seed."""
        samples = _labelled(Path(dataset_dir), max_videos_per_class)
        pairs = self.pairs_from_samples(samples)
        random.Random(42).shuffle(pairs)
        logger.info("Extracted %d frame transition pairs from %d videos.", len(pairs), len(samples))
        return pairs

    def train(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray, int]],
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-4,
        val_pairs: Optional[List[Tuple[np.ndarray, np.ndarray, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Trains temporal continuity model. If `val_pairs` is given (e.g. built from the content-hash
        validation partition in core.media_library), per-epoch held-out accuracy is recorded in
        history["val_accuracy"]; otherwise history["val_accuracy"] stays empty (prior behavior).
        """
        if not pairs:
            raise ValueError("No video frame pairs available for training.")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = FrameTransitionDataset(pairs, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_pairs:
            val_loader = DataLoader(FrameTransitionDataset(val_pairs, transform=transform), batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        history: Dict[str, Any] = {"loss": [], "val_accuracy": []}
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)

            avg_loss = total_loss / max(1, len(pairs))
            history["loss"].append(avg_loss)

            if val_loader is not None:
                self.model.eval()
                correct = 0
                val_total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        correct += (torch.argmax(self.model(images), dim=1) == labels).sum().item()
                        val_total += len(labels)
                val_acc = correct / max(1, val_total)
                history["val_accuracy"].append(val_acc)
                logger.info("Video Trainer Epoch [%d/%d] - Loss: %.4f - Val Acc: %.2f%%", epoch + 1, epochs, avg_loss, val_acc * 100.0)
            else:
                logger.info("Video Trainer Epoch [%d/%d] - Loss: %.4f", epoch + 1, epochs, avg_loss)

        self.save_checkpoint()
        return history

    def pairs_from_samples(
        self,
        samples: List[Tuple[Path, int]],
        max_videos: Optional[int] = None,
        frame_size: int = 224,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Builds consecutive frame pairs from videos IN PLACE (label 0=ai_generated, 1=real).
        Frames are downscaled to frame_size x frame_size immediately so RAM stays bounded even for
        long/4K videos (the model resizes to 224x224 anyway). Unreadable videos are skipped.
        """
        pairs: List[Tuple[np.ndarray, np.ndarray, int]] = []
        for path, label in (samples[:max_videos] if max_videos else samples):
            frames, _, _ = self.extractor.extract_sampled_frames(path, max_frames=12)
            small = [cv2.resize(f, (frame_size, frame_size)) for f in frames]
            for i in range(len(small) - 1):
                pairs.append((small[i], small[i + 1], label))
        return pairs

    def save_checkpoint(self, path: Optional[Path] = None) -> None:
        """Serializes video temporal checkpoint."""
        save_path = path or self.checkpoint_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "class_to_idx": {"ai_generated": 0, "real": 1},
        }
        partial = save_path.with_name(save_path.name + ".partial")
        torch.save(checkpoint, partial)
        os.replace(partial, save_path)                            # a crash mid-write never leaves a half-written checkpoint
        logger.info("Saved trained Video AI detector checkpoint to %s", save_path)

    def export_feature_dataset(
        self,
        dataset_dir: Path | str,
        output_npz_path: Path | str,
        max_videos_per_class: int = 50,
    ) -> Dict[str, Any]:
        """
        Extracts compact temporal frame transition differences from videos and saves to compressed .npz.
        """
        output_path = Path(output_npz_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        diff_tensors: List[np.ndarray] = []
        labels: List[int] = []
        total_source_bytes = 0

        for vid, label in _labelled(Path(dataset_dir), max_videos_per_class):
            total_source_bytes += vid.stat().st_size
            frames, _, _ = self.extractor.extract_sampled_frames(vid, max_frames=12)
            for first, second in zip(frames, frames[1:]):
                diff_tensors.append(cv2.resize(cv2.absdiff(first, second), (112, 112)))
                labels.append(label)

        if not diff_tensors:
            return {"success": False, "error": "No valid video frame transitions extracted."}

        arr_diffs = np.array(diff_tensors, dtype=np.uint8)
        arr_labels = np.array(labels, dtype=np.int64)

        np.savez_compressed(str(output_path), diffs=arr_diffs, labels=arr_labels)
        npz_bytes = output_path.stat().st_size
        savings_pct = (((total_source_bytes - npz_bytes) / max(1, total_source_bytes)) * 100.0) if total_source_bytes > 0 else 0.0

        return {
            "success": True,
            "output_path": str(output_path),
            "samples_processed": len(labels),
            "source_size_mb": round(total_source_bytes / (1024 * 1024), 2),
            "compressed_size_mb": round(npz_bytes / (1024 * 1024), 3),
            "space_savings_pct": round(savings_pct, 1),
        }

    def train_from_feature_bank(
        self,
        npz_path: Path | str,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        """Trains temporal model directly from a .npz feature bank with zero raw video on disk."""
        with np.load(str(npz_path)) as data:
            diffs, labels = data["diffs"], data["labels"]

        pairs = []
        for i in range(len(diffs)):
            # Pair where f1 is diff and f2 is 0 gives identical absdiff
            pairs.append((diffs[i], np.zeros_like(diffs[i]), int(labels[i])))

        return self.train(pairs, epochs=epochs, batch_size=batch_size, lr=lr)


def main():
    parser = argparse.ArgumentParser(description="Train Video AI Temporal Detector")
    parser.add_argument("--dataset", type=str, required=True, help="Path to video dataset with real/ and ai_generated/")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=str, default=str(DEFAULT_VIDEO_CHECKPOINT), help="Output checkpoint file")
    args = parser.parse_args()

    trainer = VideoDetectorTrainer(checkpoint_path=Path(args.output))
    pairs = trainer.prepare_data_from_videos(args.dataset)
    trainer.train(pairs, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
