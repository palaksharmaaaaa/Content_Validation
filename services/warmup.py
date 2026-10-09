"""
services.warmup: load the heavy models in the background while the user is still choosing a file.

The first analysis otherwise pays for loading every model (several seconds). ``start_background_warmup`` starts one daemon thread, once
per process, that loads them in turn. Every loader already takes its own lock, so an analysis that arrives mid-warm-up simply waits
for, or shares, the model being loaded. A model that fails to load is logged and skipped; it never stops the app.

Set ``OMNI_WARMUP=0`` to disable it (for example when only audio is analysed and the image models would be loaded for nothing).
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional, Sequence

logger = logging.getLogger("services.warmup")

ENV_WARMUP = "OMNI_WARMUP"

_started = False
_lock = threading.Lock()


def default_loaders() -> Sequence[Callable[[], object]]:
    """The loaders to run, imported lazily so merely importing this module stays cheap."""
    def object_detector():
        from core.perception.detector import get_object_detector
        return get_object_detector()._ensure()

    def recognizer():
        from core.perception.recognizer import get_recognizer
        return get_recognizer()._ensure()

    def age():
        from core.perception.age import get_age_estimator
        return get_age_estimator()._ensure()

    def face_authenticity():
        from image_detector.face_authenticity import get_face_authenticity
        return get_face_authenticity().available

    return (object_detector, age, recognizer, face_authenticity)


def enabled() -> bool:
    """False when ``OMNI_WARMUP`` is set to 0, false, no or off."""
    return os.environ.get(ENV_WARMUP, "1").strip().lower() not in {"0", "false", "no", "off"}


def start_background_warmup(loaders: Optional[Sequence[Callable[[], object]]] = None) -> Optional[threading.Thread]:
    """Start the warm-up thread once per process; later calls return ``None``. Also ``None`` when disabled."""
    global _started
    if not enabled():
        return None
    with _lock:
        if _started:
            return None
        _started = True
    chosen = list(loaders) if loaders is not None else list(default_loaders())

    def run() -> None:
        try:
            from core.atomic_io import purge_stale_sessions

            purge_stale_sessions()                       # scratch folders of sessions that ended without clearing their files
        except Exception as exc:  # noqa: BLE001 - housekeeping must never stop the warm-up
            logger.warning("stale-session sweep failed: %s", exc)
        for load in chosen:
            try:
                load()
            except Exception as exc:  # noqa: BLE001 - a model that cannot load must not stop the others
                logger.warning("warm-up of %s failed: %s", getattr(load, "__name__", load), exc)

    thread = threading.Thread(target=run, name="model-warmup", daemon=True)
    thread.start()
    return thread


def _reset_for_tests() -> None:
    global _started
    with _lock:
        _started = False
