"""
services.forensic_service: Unified Enterprise Forensic Service Orchestration Layer.
Decouples core analytical capabilities from presentation frameworks (Streamlit/FastAPI/CLI).
Provides:
1. Unified lifecycle management for image, video, and audio forensic engines.
2. Headless execution interface for single and batch forensic analyses.
3. Thread-safe feedback registration and scoring-constant recalibration (not model training -- see each package's learner.py).
4. Comprehensive health checks, metrics, and diagnostics.
"""
from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any, Dict, Optional

from audio_detector.attribution import AudioModelAttributionEngine
from audio_detector.content import AudioContentAnalyzer
from audio_detector.detector import AudioAIDetector
from audio_detector.learner import AudioSelfImprover
from audio_detector.pipeline import AudioForensicPipeline
from image_detector.attribution import ImageModelAttributionEngine
from image_detector.content import ImageContentAnalyzer
from image_detector.detector import ImageAIDetector
from image_detector.face import FaceDeepfakeDetector
from image_detector.learner import ImageSelfImprover
from image_detector.pipeline import ImageForensicPipeline
from video_detector.content import VideoContentAnalyzer
from video_detector.detector import VideoAIDetector
from video_detector.learner import VideoSelfImprover
from video_detector.pipeline import VideoForensicPipeline

logger = logging.getLogger("services.forensic_service")


class ForensicService:
    """
    Enterprise-grade facade orchestrating multi-modal deepfake and AI forensics.
    Completely decoupled from Streamlit and presentation mechanics.
    """

    _instance: Optional[ForensicService] = None
    _init_lock = threading.Lock()

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._locks: dict = {}
        self.checkpoint_path = checkpoint_path

        # Lazy components
        self._image_pipeline: Optional[ImageForensicPipeline] = None
        self._video_pipeline: Optional[VideoForensicPipeline] = None
        self._audio_pipeline: Optional[AudioForensicPipeline] = None
        self._image_detector: Optional[ImageAIDetector] = None
        self._video_detector: Optional[VideoAIDetector] = None
        self._audio_detector: Optional[AudioAIDetector] = None
        self._face_detector: Optional[FaceDeepfakeDetector] = None
        self._content_analyzer: Optional[ImageContentAnalyzer] = None
        self._attribution_engine: Optional[ImageModelAttributionEngine] = None
        self._audio_content_analyzer: Optional[AudioContentAnalyzer] = None
        self._audio_attribution_engine: Optional[AudioModelAttributionEngine] = None
        self._image_improver: Optional[ImageSelfImprover] = None
        self._audio_improver: Optional[AudioSelfImprover] = None
        self._video_improver: Optional[VideoSelfImprover] = None

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[Path] = None) -> ForensicService:
        """Returns thread-safe singleton instance of ForensicService."""
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls(checkpoint_path=checkpoint_path)
            elif checkpoint_path is not None and Path(checkpoint_path) != cls._instance.checkpoint_path:
                logger.warning("ForensicService already exists with checkpoint %s; ignoring the requested %s",
                               cls._instance.checkpoint_path, checkpoint_path)
            return cls._instance

    def _lock_for(self, name: str) -> threading.RLock:
        """One re-entrant lock per lazily built component, so building one model never blocks the others."""
        with self._lock:
            return self._locks.setdefault(name, threading.RLock())

    # -------------------------------------------------------------------------
    # Lazy Initializers
    # -------------------------------------------------------------------------
    @property
    def image_detector(self) -> ImageAIDetector:
        """The image detector (built on first use)."""
        with self._lock_for("image_detector"):
            if self._image_detector is None:
                self._image_detector = ImageAIDetector(checkpoint_path=self.checkpoint_path)
                self._image_detector.load()
            return self._image_detector

    @property
    def image_pipeline(self) -> ImageForensicPipeline:
        """The image pipeline (built on first use)."""
        with self._lock_for("image_pipeline"):
            if self._image_pipeline is None:
                self._image_pipeline = ImageForensicPipeline(
                    detector=self.image_detector,
                    content_analyzer=self.content_analyzer,
                    attribution_engine=self.attribution_engine,
                )
            return self._image_pipeline

    @property
    def content_analyzer(self) -> ImageContentAnalyzer:
        """The image content analyser (built on first use)."""
        with self._lock_for("content_analyzer"):
            if self._content_analyzer is None:
                self._content_analyzer = ImageContentAnalyzer()
            return self._content_analyzer

    @property
    def attribution_engine(self) -> ImageModelAttributionEngine:
        """The image attribution engine (built on first use)."""
        with self._lock_for("attribution_engine"):
            if self._attribution_engine is None:
                self._attribution_engine = ImageModelAttributionEngine()
            return self._attribution_engine

    @property
    def audio_content_analyzer(self) -> AudioContentAnalyzer:
        """The audio content analyser (built on first use)."""
        with self._lock_for("audio_content_analyzer"):
            if self._audio_content_analyzer is None:
                self._audio_content_analyzer = AudioContentAnalyzer()
            return self._audio_content_analyzer

    @property
    def audio_attribution_engine(self) -> AudioModelAttributionEngine:
        """The audio attribution engine (built on first use)."""
        with self._lock_for("audio_attribution_engine"):
            if self._audio_attribution_engine is None:
                self._audio_attribution_engine = AudioModelAttributionEngine()
            return self._audio_attribution_engine

    @property
    def video_detector(self) -> VideoAIDetector:
        """The video detector (built on first use)."""
        with self._lock_for("video_detector"):
            if self._video_detector is None:
                self._video_detector = VideoAIDetector(frame_detector=self.image_detector)
                self._video_detector.load()
            return self._video_detector

    @property
    def audio_detector(self) -> AudioAIDetector:
        """The audio detector (built on first use)."""
        with self._lock_for("audio_detector"):
            if self._audio_detector is None:
                self._audio_detector = AudioAIDetector()
                self._audio_detector.load()
            return self._audio_detector

    @property
    def face_detector(self) -> FaceDeepfakeDetector:
        """The face-risk heuristic shared by image and video (built on first use)."""
        with self._lock_for("face_detector"):
            if self._face_detector is None:
                self._face_detector = FaceDeepfakeDetector()
            return self._face_detector

    @property
    def image_improver(self) -> ImageSelfImprover:
        """The image feedback learner (built on first use)."""
        with self._lock_for("image_improver"):
            if self._image_improver is None:
                self._image_improver = ImageSelfImprover()
            return self._image_improver

    @property
    def audio_improver(self) -> AudioSelfImprover:
        """The audio feedback learner (built on first use)."""
        with self._lock_for("audio_improver"):
            if self._audio_improver is None:
                self._audio_improver = AudioSelfImprover()
            return self._audio_improver

    @property
    def video_pipeline(self) -> VideoForensicPipeline:
        """The video pipeline (built on first use)."""
        with self._lock_for("video_pipeline"):
            if self._video_pipeline is None:
                self._video_pipeline = VideoForensicPipeline(
                    detector=self.video_detector,
                    content_analyzer=VideoContentAnalyzer(image_content_analyzer=self.content_analyzer),
                    audio_detector=self.audio_detector,
                )
            return self._video_pipeline

    @property
    def audio_pipeline(self) -> AudioForensicPipeline:
        """The audio pipeline (built on first use)."""
        with self._lock_for("audio_pipeline"):
            if self._audio_pipeline is None:
                self._audio_pipeline = AudioForensicPipeline(
                    detector=self.audio_detector,
                )
            return self._audio_pipeline

    @property
    def video_improver(self) -> VideoSelfImprover:
        """The video feedback learner (built on first use)."""
        with self._lock_for("video_improver"):
            if self._video_improver is None:
                self._video_improver = VideoSelfImprover()
            return self._video_improver

    # -------------------------------------------------------------------------
    # Core Service Methods
    # -------------------------------------------------------------------------
    @staticmethod
    def _guarded(kind: str, path: str | Path, run) -> Dict[str, Any]:
        """Run one analysis; any failure becomes a ``PROCESSING_ERROR`` report instead of an exception, so a batch or a UI
        callback never dies on one bad file."""
        try:
            return run()
        except Exception as exc:
            logger.error("%s analysis failed for %s: %s", kind, path, exc, exc_info=True)
            return {
                "content_valid": False,
                "filename": Path(path).name,
                "path": str(path),
                "error": str(exc),
                "final_status": "PROCESSING_ERROR",
            }

    def analyze_image(self, image_path: str | Path, sensitivity: str = "balanced", source: str = "User Upload") -> Dict[str, Any]:
        """Runs the linear 5-stage forensic evaluation on a single image."""
        return self._guarded("Image", image_path, lambda: self.image_pipeline.analyze(image_path=image_path, sensitivity=sensitivity, source=source))

    def analyze_audio(self, audio_path: str | Path, sensitivity: str = "balanced") -> Dict[str, Any]:
        """Runs the linear acoustic forensic evaluation pipeline on an audio track."""
        return self._guarded("Audio", audio_path, lambda: self.audio_pipeline.analyze(audio_path=audio_path, sensitivity=sensitivity))

    def analyze_video(self, video_path: str | Path, sensitivity: str = "balanced", audio_forensics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Runs the multi-modal temporal forensic evaluation pipeline on a video file."""
        return self._guarded("Video", video_path, lambda: self.video_pipeline.analyze(
            video_path=video_path, sensitivity=sensitivity, audio_forensics=audio_forensics))

    def record_feedback(
        self,
        media_path: str | Path,
        modality: str,
        user_label: str,
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Registers ground-truth calibration feedback atomically."""
        mod = modality.lower()
        if mod == "image":
            return self.image_improver.record_feedback(
                image_path=str(media_path),
                user_label=user_label,
                metrics=metrics or {},
                notes=notes,
            )
        elif mod == "audio":
            return self.audio_improver.record_feedback(
                audio_path=str(media_path),
                user_label=user_label,
                metrics=metrics or {},
                notes=notes,
            )
        elif mod == "video":
            return self.video_improver.record_feedback(
                video_path=str(media_path),
                user_label=user_label,
                metrics=metrics or {},
                notes=notes,
            )
        else:
            raise ValueError(f"Unsupported modality '{modality}'. Expected image, video, or audio.")

    _COMPONENTS = (
        "image_pipeline", "video_pipeline", "audio_pipeline", "image_detector", "video_detector",
        "audio_detector", "face_detector", "content_analyzer", "attribution_engine",
    )

    @staticmethod
    def _security_probes() -> Dict[str, str]:
        """Each guard is exercised, not assumed: a loopback URL must be refused, the pixel ceiling must be set, and an atomic write
        followed by a read must give the data back."""
        import tempfile
        from pathlib import Path

        from PIL import Image

        from core.atomic_io import atomic_read_json, atomic_write_json
        from core.security import validate_secure_url

        try:
            ssrf = "ACTIVE" if validate_secure_url("http://127.0.0.1/x.png")[0] is False else "INACTIVE"
        except Exception:
            ssrf = "INACTIVE"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / "probe.json"
                atomic_write_json(target, {"ok": 1})
                atomic = "ACTIVE" if atomic_read_json(target, default=None) == {"ok": 1} else "INACTIVE"
        except Exception:
            atomic = "INACTIVE"
        return {
            "anti_ssrf": ssrf,
            "decompression_bomb_protection": "ACTIVE" if Image.MAX_IMAGE_PIXELS else "INACTIVE",
            "atomic_storage": atomic,
        }

    def health_check(self) -> Dict[str, Any]:
        """Builds every component (loading models on first use) and reports which ones are usable.

        ``status`` is ``HEALTHY`` only if all components initialise and the security guards are in place.
        """
        components: Dict[str, str] = {}
        for name in self._COMPONENTS:
            try:
                getattr(self, name)
                components[name] = "READY"
            except Exception as exc:  # report, never raise: this is a diagnostic
                logger.warning("health_check: %s failed to initialise: %s", name, exc)
                components[name] = f"ERROR: {type(exc).__name__}"
        security = self._security_probes()
        healthy = all(v == "READY" for v in components.values()) and all(v == "ACTIVE" for v in security.values())
        return {"status": "HEALTHY" if healthy else "DEGRADED", "components": components, "security": security}
