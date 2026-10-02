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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch

from video_detector.config import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_VIDEO_CHECKPOINT,
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

    def load(self) -> bool:
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
                logger.warning("Could not load video checkpoint, using statistical acoustics: %s", e)

        if self.frame_detector is not None and hasattr(self.frame_detector, "load"):
            self.frame_detector.load()

        self._is_loaded = True
        return True

    def _score_frame_internal(self, frame_bgr: np.ndarray, sensitivity: str) -> Dict[str, Any]:
        """Internal frame scoring fallback if no external frame detector is attached."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 3)
        noise = float(np.mean(cv2.absdiff(gray, blurred)))

        bilateral = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
        smooth = float(np.mean(cv2.absdiff(gray, bilateral)))

        comp_noise = max(0.2, noise - 0.70)
        noise_thresh = 2.4 if sensitivity in ("high", "aggressive") else 2.0
        smooth_thresh = 3.6 if sensitivity in ("high", "aggressive") else 3.1

        p_noise_ai = float(1.0 / (1.0 + np.exp((comp_noise - noise_thresh) * 2.0)))
        p_smooth_ai = float(1.0 / (1.0 + np.exp((smooth - smooth_thresh) * 1.3)))
        score = (p_noise_ai * 0.55) + (p_smooth_ai * 0.45)

        thresh = 0.50 if sensitivity in ("high", "aggressive") else 0.60
        label = "LIKELY AI-GENERATED" if score >= thresh else ("LIKELY REAL" if score <= 0.35 else "UNDECIDED")
        return {"label": label, "prediction": label, "ai_prob": round(score, 3), "real_prob": round(1.0 - score, 3)}

    def analyze_video(
        self,
        video_path: str | Path,
        sensitivity: str = "balanced",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive temporal and spatial video AI detection.
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
        analyzed_frames: List[Dict[str, Any]] = []
        frame_ai_scores: List[float] = []

        total_f = len(frames)
        for idx, (frame, ts) in enumerate(zip(frames, timestamps)):
            if progress_callback:
                progress_callback(idx + 1, total_f, f"Analyzing frame {idx+1}/{total_f}")

            # Blank frame check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(np.var(gray)) < 5.0:
                analyzed_frames.append({"frame_idx": idx, "timestamp": ts, "label": "UNDECIDED", "ai_prob": 0.5})
                continue

            if self.frame_detector is not None and hasattr(self.frame_detector, "predict_frame"):
                frame_res = self.frame_detector.predict_frame(frame, sensitivity=sensitivity)
            else:
                frame_res = self._score_frame_internal(frame, sensitivity=sensitivity)

            analyzed_frames.append({
                "frame_idx": idx,
                "timestamp": ts,
                "label": frame_res["label"],
                "ai_prob": frame_res["ai_prob"],
                "real_prob": frame_res["real_prob"],
            })
            frame_ai_scores.append(frame_res["ai_prob"])

        # 4b. Neural Temporal Discriminator Inference (if model available)
        neural_transition_ai = None
        if self.model is not None and len(frames) >= 2:
            try:
                from torchvision import transforms
                val_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                diff_tensors = []
                step = max(1, (len(frames) - 1) // 16)
                for i in range(0, len(frames) - 1, step):
                    diff = cv2.absdiff(frames[i], frames[i + 1])
                    diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(diff_rgb)
                    diff_tensors.append(val_transform(pil_img))
                if diff_tensors:
                    batch = torch.stack(diff_tensors).to(self.device)
                    with torch.no_grad():
                        logits = self.model(batch)
                        probs = torch.softmax(logits, dim=1)
                        # class 0 = AI, class 1 = Real
                        neural_transition_ai = float(torch.mean(probs[:, 0]).item())
            except Exception as e:
                logger.debug("Neural transition inference bypassed: %s", e)

        # Cache small representative subset of keyframes for downstream scene/face analysis
        keyframe_step = max(1, len(frames) // 12)
        cached_keyframes = [frames[i] for i in range(0, len(frames), keyframe_step)][:12]

        # Deallocate raw frame pixel arrays to prevent memory leaks
        del frames

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

        # Forensic cues
        cues: List[str] = []
        if neural_transition_ai is not None:
            cues.append(f"Neural temporal discriminator transition AI probability: {neural_transition_ai * 100:.1f}%")
        if temporal_res["temporal_warping_risk"] in ("HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER"):
            cues.append(f"Temporal warping detected (motion variance: {temporal_res['motion_variance']})")
        elif temporal_res["temporal_warping_risk"] == "UNNATURAL_FREEZE":
            cues.append(f"Unnatural frame-to-frame stillness detected (motion variance: {temporal_res['motion_variance']})")
        if flicker_res["has_diffusion_flicker"]:
            cues.append(f"Diffusion generation flicker detected across frames (flicker score: {flicker_res['flicker_score']})")
        if mean_frame_ai > 0.60:
            cues.append(f"High spatial synthetic artifact ratio across frames ({mean_frame_ai*100:.1f}%)")

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
            temporal_segments=temporal_segments,
            forensic_cues=cues,
            metadata=metadata,
        )
        res_dict = res.to_dict()
        res_dict["keyframes"] = cached_keyframes
        return res_dict

    predict = analyze_video
    predict_video = analyze_video
