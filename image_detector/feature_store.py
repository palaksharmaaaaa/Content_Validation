"""
image_detector.feature_store: Zero-retention, cross-platform Feature Store & Embedding Cache.
Optional, rebuildable cache of per-image features -- it never deletes or copies source media.

Capabilities:
1. Extract 512-dim CNN latent embeddings + 12-dim physical/spectral forensic features.
2. Build compressed .npz feature archives keyed by each file's SHA-256 CONTENT hash (never its
   name) plus a fingerprint of the backbone that produced the embeddings. Rows whose hash and
   fingerprint still match are reused; anything else is recomputed, so the cache is always safe
   to delete and rebuild.
3. PyTorch FeatureBankDataset and FeatureClassifierHead for fast zero-I/O training.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from core.imageio import imread
from core.media_library import compute_file_sha256
from image_detector.config import DEFAULT_CHECKPOINT, IMAGE_SIZE
from image_detector.features import analyze_fft_radial_power_spectrum, calculate_sensor_noise_profile, calculate_surface_smoothness, compute_ela, detect_inpainting_and_manipulation, detect_screenshot
from image_detector.models.backbone import build_image_classifier

logger = logging.getLogger("image_detector.feature_store")


def _decode_image_input(image_input: Union[str, Path, bytes, bytearray, io.BytesIO, np.ndarray, Image.Image]) -> Tuple[Optional[np.ndarray], Optional[Path]]:
    """
    Decodes any image input type (path, bytes, io.BytesIO, numpy array, PIL Image)
    into a BGR OpenCV numpy array in RAM without writing to disk.
    """
    source_path = None
    if isinstance(image_input, (str, Path)):
        source_path = Path(image_input)
        if not source_path.is_file():
            return None, source_path
        img_bgr = imread(str(source_path))
        return img_bgr, source_path

    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR), None
        elif image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR), None
        return image_input, None

    if isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None

    if isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img_bgr, None

    if hasattr(image_input, "read"):
        raw = image_input.read()
        nparr = np.frombuffer(raw, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img_bgr, None

    return None, None


class FeatureStore:
    """
    Extracts, caches, and serializes compact mathematical forensic representations.
    Caches lightweight .npz archives next to (never instead of) the original media.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.backbone: Optional[nn.Module] = None
        self.fingerprint = "uninitialized"
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._load_backbone()

    def _load_backbone(self) -> None:
        """Loads feature extraction backbone (stripping the final classification layer)."""
        try:
            has_checkpoint = self.checkpoint_path.exists()
            # Without a trained checkpoint, use ImageNet weights: an untrained random backbone would
            # produce meaningless embeddings.
            full_model = build_image_classifier(architecture="resnet18", pretrained=not has_checkpoint)
            self.fingerprint = (
                "ckpt:" + compute_file_sha256(self.checkpoint_path)[:16] if has_checkpoint else "imagenet-resnet18"
            )
            if has_checkpoint:
                state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
                architecture = state_dict.get("architecture", "resnet18") if isinstance(state_dict, dict) else "resnet18"
                if architecture != "resnet18":
                    raise ValueError(f"the feature store embeds with ResNet-18 (512 values); the checkpoint is {architecture}")
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                full_model.load_state_dict(state_dict)
            full_model.eval()
            full_model.to(self.device)
            # Remove the linear classifier head to get 512-dim embedding
            self.backbone = full_model
        except Exception as exc:
            logger.warning("Could not initialize neural backbone for embeddings: %s. Using statistical vectors only.", exc)
            self.backbone = None
            self.fingerprint = "no-backbone"

    def extract_features(
        self,
        image_input: Union[str, Path, bytes, bytearray, io.BytesIO, np.ndarray, Image.Image],
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extracts both the 512-dim neural embedding and the 12-dim physical forensic feature vector.
        Operates entirely in-memory with zero disk footprint.
        """
        img_bgr, path = _decode_image_input(image_input)
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Physical & Spectral Forensic Metrics (12-dim vector)
        noise_mean, noise_std = calculate_sensor_noise_profile(gray)
        smoothness = calculate_surface_smoothness(gray)
        fft_res = analyze_fft_radial_power_spectrum(gray)
        if fft_res.get("spectral_decay_alpha") is None:
            return None                                           # no measurable spectrum: not a feature row
        fft_alpha = float(fft_res["spectral_decay_alpha"])
        fft_anomalous = 1.0 if fft_res.get("is_anomalous_decay") else 0.0
        azimuthal_var = float(fft_res.get("azimuthal_directional_variance", 0.0))
        peak_energy_ratio = float(fft_res.get("peak_energy_ratio", 1.0))

        ela_mean, ela_map = compute_ela(img_bgr)

        # Quick inpainting & screenshot indicators
        inpaint_res = detect_inpainting_and_manipulation(path or "mem", img_bgr, ela_map)
        noise_incon = float(inpaint_res.get("noise_inconsistency", 0.0))
        is_manip = 1.0 if inpaint_res.get("is_manipulated") else 0.0

        screenshot_res = detect_screenshot(path or "mem", img_bgr)
        is_screen = 1.0 if screenshot_res.get("is_screenshot") else 0.0

        forensic_vec = np.array([
            noise_mean,
            noise_std,
            smoothness,
            fft_alpha,
            fft_anomalous,
            ela_mean,
            azimuthal_var,
            peak_energy_ratio,
            noise_incon,
            is_manip,
            is_screen,
            float(h) / max(1.0, float(w)),
        ], dtype=np.float32)

        # 2. Neural latent embedding (512-dim). Without one the row is not a feature row at all: a zero vector would train
        #    and fit the out-of-distribution gate on numbers no image produced, so the image is skipped instead.
        if self.backbone is None:
            return None
        try:
            rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            tensor = self.transform(rgb_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Forward through ResNet backbone up to avgpool
                x = self.backbone.conv1(tensor)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)
                x = self.backbone.avgpool(x)
                embedding = torch.flatten(x, 1).cpu().numpy().squeeze(0).astype(np.float32)
        except Exception as exc:
            logger.warning("Embedding failed (%s: %s); skipping this image", type(exc).__name__, exc)
            return None

        return {
            "embedding": embedding,
            "forensics": forensic_vec,
        }

    def build_feature_bank(
        self,
        samples: List[Tuple[Any, int]],
        output_npz_path: Union[str, Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts compact representations for (path_or_bytes, label) samples into one .npz archive.
        Source files are only ever read. Rows are keyed by content hash (file bytes for paths, the
        raw bytes for in-memory inputs); rows already present in an existing archive with the same
        backbone fingerprint are reused instead of recomputed.
        """
        output_path = Path(output_npz_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cached: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if output_path.exists():
            try:
                with np.load(str(output_path), allow_pickle=False) as data:   # closed before the file is replaced below
                    if str(data["fingerprint"]) == self.fingerprint:
                        for h, e, f in zip(data["hashes"], data["embeddings"], data["forensics"]):
                            cached[str(h)] = (e, f)
            except Exception as exc:  # stale/corrupt cache is simply rebuilt
                logger.info("Ignoring unusable feature cache %s: %s", output_path.name, exc)

        embeddings_list: List[np.ndarray] = []
        forensics_list: List[np.ndarray] = []
        labels_list: List[int] = []
        hashes_list: List[str] = []
        reused = 0

        total_samples = len(samples)
        for idx, (item, label) in enumerate(samples):
            if progress_callback:
                progress_callback(idx + 1, total_samples, f"sample {idx + 1}")  # never a file name

            if isinstance(item, (str, Path)):
                try:
                    content_hash = compute_file_sha256(item)
                except OSError:
                    continue
            elif isinstance(item, (bytes, bytearray)):
                content_hash = hashlib.sha256(bytes(item)).hexdigest()
            else:
                content_hash = ""  # non-hashable in-memory input: always recomputed, never reused

            if content_hash and content_hash in cached:
                emb, fore = cached[content_hash]
                reused += 1
            else:
                feats = self.extract_features(item)
                if feats is None:
                    continue
                emb, fore = feats["embedding"], feats["forensics"]
            embeddings_list.append(emb)
            forensics_list.append(fore)
            labels_list.append(int(label))
            hashes_list.append(content_hash)

        if not embeddings_list:
            return {
                "success": False,
                "error": "No valid samples could be extracted.",
                "total_processed": 0,
            }

        partial = output_path.with_name(output_path.name + ".partial")
        with open(partial, "wb") as handle:                       # written aside, then swapped in: a crash never leaves a half-written cache
            np.savez_compressed(
                handle,
                embeddings=np.array(embeddings_list, dtype=np.float32),
                forensics=np.array(forensics_list, dtype=np.float32),
                labels=np.array(labels_list, dtype=np.int64),
                hashes=np.array(hashes_list),
                fingerprint=np.array(self.fingerprint),
            )
        os.replace(partial, output_path)
        logger.info("Feature bank saved: %s (%d samples, %d reused from cache)", output_path.name, len(labels_list), reused)
        return {
            "success": True,
            "output_path": str(output_path),
            "samples_processed": len(labels_list),
            "samples_reused_from_cache": reused,
            "compressed_size_mb": round(output_path.stat().st_size / (1024 * 1024), 3),
            "backbone_fingerprint": self.fingerprint,
        }

    @staticmethod
    def load_feature_bank(npz_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads (embeddings, forensics, labels) from a compressed .npz archive."""
        with np.load(str(npz_path), allow_pickle=False) as data:
            return data["embeddings"], data["forensics"], data["labels"]


class FeatureBankDataset(Dataset):
    """
    Ultra-fast PyTorch Dataset loading directly from precomputed .npz feature archives.
    Eliminates all disk image reading and JPEG decoding during model training epochs.
    """

    def __init__(
        self,
        npz_path_or_arrays: Union[str, Path, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        combine_forensics: bool = True,
    ):
        if isinstance(npz_path_or_arrays, (str, Path)):
            emb, fore, y = FeatureStore.load_feature_bank(npz_path_or_arrays)
        else:
            emb, fore, y = npz_path_or_arrays

        if combine_forensics:
            # Concatenate 512-dim embedding with 12-dim forensic features -> 524-dim input vector
            self.X = np.concatenate([emb, fore], axis=1).astype(np.float32)
        else:
            self.X = emb.astype(np.float32)

        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class FeatureClassifierHead(nn.Module):
    """
    Lightweight classification head trained directly on feature bank representations.
    Matches the layer layout of the backbone's fc head (Dropout, Linear, ReLU, Linear) so it can be spliced back in.
    """

    def __init__(self, in_features: int = 524, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
