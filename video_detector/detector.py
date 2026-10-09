"""
video_detector.detector: Complete, self-contained Video AI Detection engine.
Combines:
1. Multi-scale temporal frame sampling.
2. Inter-frame motion vector variance normalized by temporal step.
3. Diffusion flickering & high-frequency frame jitter analysis.
4. Per-frame spatial forensic evaluation and aggregation.
5. Contiguous temporal timeline segmentation with exact bounds.
6. Rule-based feedback calibration (see learner.py -- adjusts scoring constants, not model weights).
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from core.shared_results import shift_probability_by_log_odds
from video_detector.config import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_VIDEO_CHECKPOINT,
    NOISE_AI_THRESHOLD,
    NOISE_AI_THRESHOLD_SENSITIVE,
    SMOOTH_AI_THRESHOLD,
    SMOOTH_AI_THRESHOLD_SENSITIVE,
    NOISE_BASELINE,
)
from video_detector.extractor import VideoFrameExtractor
from video_detector.learner import VideoSelfImprover
from video_detector.models.backbone import VideoTemporalTransitionModel
from video_detector.schemas import VideoForensicResult
from video_detector.scoring import (
    calculate_video_epistemic_uncertainty,
    evaluate_video_decision,
    normalize_percentages,
    pool_video_temporal_score,
)
from video_detector.temporal import (
    compute_interframe_motion_variance,
    detect_diffusion_flickering,
    group_temporal_segments,
)

logger = logging.getLogger("video_detector.detector")


class VideoAIDetector:
    """
    Completely independent, self-contained Video AI Detector with rule-based feedback calibration (see learner.py).
    Evaluates temporal continuity, motion variance, boundary warping, diffusion flicker,
    and frame-level spatial synthesis to provide holistic video verification.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path | str] = None,
        frame_detector: Optional[Any] = None,
        self_improver: Optional[VideoSelfImprover] = None,
        max_sampled_frames: int = DEFAULT_MAX_FRAMES,
    ):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_VIDEO_CHECKPOINT
        self.frame_detector = frame_detector
        self.self_improver = self_improver or VideoSelfImprover()
        self.extractor = VideoFrameExtractor(max_frames=max_sampled_frames)
        self.model = None
        self.device = None
        self._is_loaded = False
        self._load_lock = threading.RLock()

    def load(self) -> bool:
        """Loads once; concurrent callers wait for the first load instead of loading the model twice."""
        with self._load_lock:
            return self._load_locked()

    def _load_locked(self) -> bool:
        """Loads neural weights if video checkpoint exists."""
        if self._is_loaded:
            return True
        if self.checkpoint_path.is_file():
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
                model = VideoTemporalTransitionModel(pretrained=False)
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
                model.to(device)
                model.eval()
                self.model = model
                self.device = device
                logger.info("VideoAIDetector loaded video checkpoint: %s", self.checkpoint_path.name)
            except Exception as e:
                logger.warning("Could not load video checkpoint, using the statistical frame analysis: %s", e)

        if self.frame_detector is not None and hasattr(self.frame_detector, "load"):
            self.frame_detector.load()

        self._is_loaded = True
        return True

    def _score_frame_internal(self, frame_bgr: np.ndarray, sensitivity: str) -> Dict[str, Any]:
        """Scores one frame from its median-filter noise residual and bilateral-filter smoothness (used when no external frame detector is attached)."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        noise = float(np.mean(cv2.absdiff(gray, blurred)))

        bilateral = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
        smooth = float(np.mean(cv2.absdiff(gray, bilateral)))

        comp_noise = max(0.2, noise - NOISE_BASELINE)
        noise_thresh = NOISE_AI_THRESHOLD_SENSITIVE if sensitivity in ("high", "aggressive") else NOISE_AI_THRESHOLD
        smooth_thresh = SMOOTH_AI_THRESHOLD_SENSITIVE if sensitivity in ("high", "aggressive") else SMOOTH_AI_THRESHOLD

        p_noise_ai = float(1.0 / (1.0 + np.exp((comp_noise - noise_thresh) * 2.0)))
        p_smooth_ai = float(1.0 / (1.0 + np.exp((smooth - smooth_thresh) * 1.3)))
        score = (p_noise_ai * 0.55) + (p_smooth_ai * 0.45)

        thresh = 0.50 if sensitivity in ("high", "aggressive") else 0.60
        label = "LIKELY AI-GENERATED" if score >= thresh else ("LIKELY REAL" if score <= 0.35 else "UNDECIDED")
        return {"label": label, "prediction": label, "ai_prob": round(score, 3), "real_prob": round(1.0 - score, 3), "frame_noise": round(noise, 3)}

    def _analyze_frames(
        self, frames: List[np.ndarray], timestamps: List[float], sensitivity: str,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Scores every sampled frame (blank frames are recorded as UNDECIDED and excluded from the mean)."""
        analyzed: List[Dict[str, Any]] = []
        scores: List[float] = []
        total = len(frames)
        for idx, (frame, ts) in enumerate(zip(frames, timestamps)):
            if progress_callback:
                progress_callback(idx + 1, total, f"Analyzing frame {idx+1}/{total}")
            if float(np.var(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) < 5.0:
                analyzed.append({"frame_idx": idx, "timestamp": ts, "label": "UNDECIDED", "ai_prob": 0.5})
                continue
            if self.frame_detector is not None and hasattr(self.frame_detector, "predict_frame"):
                frame_res = self.frame_detector.predict_frame(frame, sensitivity=sensitivity)
            else:
                frame_res = self._score_frame_internal(frame, sensitivity=sensitivity)
            analyzed.append({
                "frame_idx": idx, "timestamp": ts, "label": frame_res["label"],
                "ai_prob": frame_res["ai_prob"], "real_prob": frame_res["real_prob"],
                **({"frame_noise": frame_res["frame_noise"]} if "frame_noise" in frame_res else {}),
            })
            scores.append(frame_res["ai_prob"])
        return analyzed, scores

    def _neural_transition_probability(self, frames: List[np.ndarray]) -> Optional[float]:
        """Mean P(AI) of the optional neural temporal discriminator over inter-frame difference images."""
        if self.model is None or len(frames) < 2:
            return None
        try:
            from torchvision import transforms

            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            step = max(1, (len(frames) - 1) // 16)
            tensors = [
                val_transform(Image.fromarray(cv2.cvtColor(cv2.absdiff(frames[i], frames[i + 1]), cv2.COLOR_BGR2RGB)))
                for i in range(0, len(frames) - 1, step)
            ]
            if not tensors:
                return None
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(batch), dim=1)
                return float(torch.mean(probs[:, 0]).item())  # class 0 = AI, class 1 = Real
        except Exception as e:
            logger.debug("Neural transition inference bypassed: %s", e)
            return None

    @staticmethod
    def _ai_duration_pct(segments: List[Dict[str, Any]], duration: float) -> Optional[float]:
        """Percentage of the timeline covered by segments labelled AI-generated; None when the duration is unknown."""
        if not duration or duration <= 0:
            return None
        ai_seconds = sum(float(s.get("duration_seconds", 0.0)) for s in segments if s.get("label") == "LIKELY AI-GENERATED")
        return round(min(100.0, ai_seconds / duration * 100.0), 1)

    @staticmethod
    def _forensic_cues(
        neural_transition_ai: Optional[float], temporal_res: Dict[str, Any], flicker_res: Dict[str, Any], mean_frame_ai: float
    ) -> List[str]:
        cues: List[str] = []
        if neural_transition_ai is not None:
            cues.append(f"Neural temporal discriminator transition AI probability: {neural_transition_ai * 100:.1f}%")
        risk = temporal_res["temporal_warping_risk"]
        if risk in ("HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER"):
            cues.append(f"Temporal warping detected (motion variance: {temporal_res['motion_variance']})")
        elif risk == "UNNATURAL_FREEZE":
            cues.append(f"Unnatural frame-to-frame stillness detected (motion variance: {temporal_res['motion_variance']})")
        if flicker_res["has_diffusion_flicker"]:
            cues.append(f"Diffusion generation flicker detected across frames (flicker score: {flicker_res['flicker_score']})")
        if mean_frame_ai > 0.60:
            cues.append(f"High spatial synthetic artifact ratio across frames ({mean_frame_ai*100:.1f}%)")
        return cues

    def analyze_video(
        self,
        video_path: str | Path,
        sensitivity: str = "balanced",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        extra_log_lrs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive temporal and spatial video AI detection.
        ``extra_log_lrs``: optional capped base-10 log-odds terms from the dimension checks
        (see video_detector.dimension_checks); None/empty leaves behavior unchanged.
        """
        self.load()
        path = Path(video_path)

        metadata = self.extractor.get_video_metadata(path)
        if "error" in metadata:
            return VideoForensicResult(
                valid=False,
                filename=path.name,
                error=metadata["error"],
            ).to_dict()

        duration = metadata.get("duration_seconds", 0.0)

        # 1. Temporal Frame Sampling
        frames, timestamps, temporal_step = self.extractor.extract_sampled_frames(path)
        if not frames:
            return VideoForensicResult(
                valid=False,
                filename=path.name,
                error="No frames could be extracted from video.",
            ).to_dict()

        # 2. Dynamic Calibration from VideoSelfImprover
        calib = self.self_improver.load_calibration()
        offsets = calib.get("sensitivity_offsets", {})
        t_weights = calib.get("temporal_weights", {})
        motion_thresholds = calib.get("motion_thresholds", {})

        # 3. Temporal Consistency & Warping
        temporal_res = compute_interframe_motion_variance(
            frames, temporal_step=temporal_step, motion_thresholds=motion_thresholds
        )
        flicker_res = detect_diffusion_flickering(frames)

        # 4. Frame-level Spatial Analysis
        analyzed_frames, frame_ai_scores = self._analyze_frames(frames, timestamps, sensitivity, progress_callback)

        # 4b. Neural Temporal Discriminator Inference (if model available)
        neural_transition_ai = self._neural_transition_probability(frames)

        # Cache small representative subset of keyframes for downstream scene/face analysis
        keyframe_step = max(1, len(frames) // 12)
        cached_keyframes = [frames[i] for i in range(0, len(frames), keyframe_step)][:12]

        # Deallocate raw frame pixel arrays to prevent memory leaks
        del frames

        frame_noises = [f["frame_noise"] for f in analyzed_frames if "frame_noise" in f]
        mean_frame_noise = round(float(np.mean(frame_noises)), 3) if frame_noises else None

        # 5. Temporal Segments
        temporal_segments = group_temporal_segments(analyzed_frames, duration_seconds=duration)

        # 6. Holistic Score Pooling
        mean_frame_ai = float(np.mean(frame_ai_scores)) if frame_ai_scores else 0.5
        raw_ai_prob, raw_real_prob = pool_video_temporal_score(
            mean_frame_ai=mean_frame_ai,
            warping_risk=temporal_res["temporal_warping_risk"],
            has_flicker=flicker_res["has_diffusion_flicker"],
            temporal_weights=t_weights,
            sensitivity_offset=offsets.get("video_ai_offset", 0.0),
            neural_prob=neural_transition_ai,
        )

        # Capped dimension-check terms (container metadata); empty input leaves the probability untouched.
        raw_ai_prob, raw_real_prob, extra_cues = shift_probability_by_log_odds(raw_ai_prob, extra_log_lrs)

        # Epistemic Uncertainty
        uncertainty = calculate_video_epistemic_uncertainty(raw_ai_prob)
        target_undecided = max(4.0, min(25.0, uncertainty * 20.0))

        ai_pct, real_pct, undecided_pct = normalize_percentages(
            ai_val=raw_ai_prob * 100.0,
            real_val=raw_real_prob * 100.0,
            undecided_val=target_undecided,
            min_undecided=4.0,
            decimals=1,
        )

        final_label = evaluate_video_decision(ai_pct, real_pct, sensitivity)

        cues = self._forensic_cues(neural_transition_ai, temporal_res, flicker_res, mean_frame_ai)
        cues.extend(extra_cues)

        res = VideoForensicResult(
            valid=True,
            filename=path.name,
            duration_seconds=duration,
            total_frames=metadata.get("total_frames", 0),
            sampled_frames_count=len(analyzed_frames),
            ai_percentage=ai_pct,
            real_percentage=real_pct,
            undecided_percentage=undecided_pct,
            confidence=round(max(raw_ai_prob, raw_real_prob), 2),
            label=final_label,
            prediction=final_label,
            temporal_consistency=temporal_res,
            diffusion_flicker=flicker_res,
            mean_frame_noise=mean_frame_noise,
            is_blank=bool(analyzed_frames) and not frame_ai_scores,
            ai_duration_pct=self._ai_duration_pct(temporal_segments, duration),
            temporal_segments=temporal_segments,
            forensic_cues=cues,
            metadata=metadata,
        )
        res_dict = res.to_dict()
        res_dict["keyframes"] = cached_keyframes
        return res_dict

    predict = analyze_video
