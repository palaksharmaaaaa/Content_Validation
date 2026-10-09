"""core.perception.recognizer: zero-shot recognition with SigLIP 2 Base (Google, Apache-2.0).

Recognises scenes, animal species, vehicle types, photo genre and time of day by comparing an image (or a crop) with a
text vocabulary (core.perception.vocab). No per-class training. One image encoding serves every vocabulary.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.perception.hub import load_kwargs, silence_noise
from core.perception.vocab import VOCABS

silence_noise()

logger = logging.getLogger("core.perception.recognizer")

MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"   # pinned: identical weights on every machine
TEXT_LEN = 64            # SigLIP 2 text tower is trained with fixed-length 64 token prompts


class ZeroShotRecognizer:
    """Loads on first use; thread-safe. ``classify`` returns ``[]`` if the model is unavailable."""

    def __init__(self, model_id: str = MODEL_ID):
        self._id = model_id
        self._model = None
        self._proc = None
        self._text: Dict[str, "object"] = {}
        self._lock = threading.Lock()
        self._failed = False

    def _ensure(self):
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    from transformers import AutoModel, AutoProcessor

                    self._proc = AutoProcessor.from_pretrained(self._id, **load_kwargs(self._id, MODEL_REVISION if self._id == MODEL_ID else None))
                    self._model = AutoModel.from_pretrained(self._id, **load_kwargs(self._id, MODEL_REVISION if self._id == MODEL_ID else None)).eval()
                except Exception as exc:
                    logger.warning("Zero-shot recognizer %s unavailable: %s", self._id, exc)
                    self._failed = True
            return self._model

    @property
    def available(self) -> bool:
        """True once the model has loaded."""
        return self._ensure() is not None

    def _text_features(self, vocab: str):
        import torch

        if vocab not in self._text:
            prompts = [p for _label, p in VOCABS[vocab]]
            tokens = self._proc(text=prompts, padding="max_length", max_length=TEXT_LEN, return_tensors="pt")
            with torch.no_grad():
                feats = self._model.get_text_features(**tokens).pooler_output
            self._text[vocab] = feats / feats.norm(dim=-1, keepdim=True)
        return self._text[vocab]

    def embed(self, images_rgb: Sequence[np.ndarray]):
        """L2-normalised image embeddings for RGB uint8 arrays."""
        import torch
        from PIL import Image

        pil = [Image.fromarray(a) for a in images_rgb]
        with torch.no_grad():
            feats = self._model.get_image_features(**self._proc(images=pil, return_tensors="pt")).pooler_output
        return feats / feats.norm(dim=-1, keepdim=True)

    def classify_embeddings(self, feats, vocab: str, top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """For each embedding, the ``top_k`` (label, probability) of the vocabulary. Probabilities are within the closed
        vocabulary, so a high value means "best of these options", not "certainly this"."""
        import torch

        labels = [label for label, _ in VOCABS[vocab]]
        with torch.no_grad():
            probs = (self._model.logit_scale.exp() * feats @ self._text_features(vocab).T).softmax(dim=-1)
        out = []
        for row in probs:
            top = torch.topk(row, min(top_k, len(labels)))
            out.append([(labels[int(i)], float(p)) for p, i in zip(top.values, top.indices)])
        return out

    def classify(self, images_rgb: Sequence[np.ndarray], vocab: str, top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """Top matches in ``vocab`` for each RGB image/crop."""
        if self._ensure() is None or not len(images_rgb):
            return [[] for _ in images_rgb]
        return self.classify_embeddings(self.embed(images_rgb), vocab, top_k)

    def classify_multi(self, image_rgb: np.ndarray, vocabs: Sequence[str], top_k: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """Encode one image once and score it against several vocabularies."""
        if self._ensure() is None:
            return {v: [] for v in vocabs}
        feats = self.embed([image_rgb])
        return {v: self.classify_embeddings(feats, v, top_k)[0] for v in vocabs}


_default: Optional[ZeroShotRecognizer] = None
_default_lock = threading.Lock()


def get_recognizer() -> ZeroShotRecognizer:
    """The process-wide recognizer."""
    global _default
    with _default_lock:
        if _default is None:
            _default = ZeroShotRecognizer()
        return _default
