"""core.perception.hub: keep model loading quiet and offline-friendly.

* OpenCV's DNN warnings ("Targets are not supported by the new graph engine") and transformers' progress bars and
  config notices are harmless noise, so they are silenced.
* A model that is already in the local Hugging Face cache is loaded with ``local_files_only=True``: no network request,
  no "unauthenticated requests to the HF Hub" warning, and it works offline. Only a missing model touches the network.
"""
from __future__ import annotations

import os
from typing import Dict

_done = False


def silence_noise() -> None:
    """Idempotent: lower OpenCV, Hugging Face and transformers log levels and disable their progress bars."""
    global _done
    if _done:
        return
    _done = True
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def is_cached(repo_id: str) -> bool:
    """True if the model's config is already in the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo_id, "config.json"), str)
    except Exception:
        return False


def load_kwargs(repo_id: str) -> Dict[str, bool]:
    """``from_pretrained`` keyword arguments: offline when cached, normal (downloading) otherwise."""
    return {"local_files_only": True} if is_cached(repo_id) else {}
