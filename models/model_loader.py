"""
Centralized model loader for AI image, video, and audio classification models.
Supports local PyTorch checkpoints and Hugging Face transformer pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from utils.logging_utils import get_logger

logger = get_logger("model_loader")


class ModelRegistry:
    _instance: Optional[ModelRegistry] = None
    _loaded_models: dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def load_hf_image_classifier(self, model_name: str) -> Any | None:
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        try:
            from transformers import pipeline
            logger.info("Attempting to load Hugging Face image classifier: %s", model_name)
            pipe = pipeline("image-classification", model=model_name)
            self._loaded_models[model_name] = pipe
            return pipe
        except Exception as exc:
            logger.warning("Could not load Hugging Face model %s: %s", model_name, exc)
            return None

    def load_local_pytorch_model(self, checkpoint_path: Path, model_builder) -> Any | None:
        path_str = str(checkpoint_path)
        if path_str in self._loaded_models:
            return self._loaded_models[path_str]

        if not checkpoint_path.is_file():
            return None

        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model = model_builder()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            bundle = {"model": model, "checkpoint": checkpoint, "device": device}
            self._loaded_models[path_str] = bundle
            return bundle
        except Exception as exc:
            logger.error("Failed loading local checkpoint %s: %s", checkpoint_path, exc)
            return None
