"""
image_detector.scoring: Self-contained Bayesian evidence pooling and scoring engine for image forensics.
Contains:
1. Multi-evidence log-likelihood ratio summation (additive Bayesian updates).
2. Epistemic uncertainty estimation via Shannon entropy.
3. Invariant percentage normalization: P(AI) + P(Real) + P(Undecided) = 100.0%.
4. Calibrated threshold decision classification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.decision import normalize_percentages  # noqa: F401  (re-exported: tests and callers import it from here)
from image_detector.config import SENSITIVITY_PRIORS


def calculate_image_epistemic_uncertainty(prob_ai: float) -> float:
    """
    Calculates epistemic uncertainty from binary probability using Shannon entropy.
    Entropy is maximal (1.0) when P(AI) = 0.5, and approaches 0 when P is near 0 or 1.
    """
    p_safe = float(np.clip(prob_ai, 1e-6, 1.0 - 1e-6))
    entropy = -p_safe * math.log2(p_safe) - (1.0 - p_safe) * math.log2(1.0 - p_safe)
    return max(0.0, min(1.0, float(entropy)))


def pool_bayesian_log_odds(
    sensitivity: str, log_lrs: Dict[str, float]
) -> Tuple[float, float, float]:
    """
    Pools evidence using Bayesian log-likelihood ratio summation with cue correlation discounting:
    Posterior Log-Odds = Prior Log-Odds + sum(w_i * Log_LR_i)

    NOTE: this posterior is expressed in base-10 log-odds (odds = 10**posterior below),
    not base-e. `audio_detector.scoring.pool_acoustic_evidence`, `video_detector.scoring.
    pool_video_temporal_score`, and `core.decision.generate_final_decision`'s cross-modal
    fusion are each their own independent evidence model operating on different physical
    signals (vocoder/flatness for audio, motion/flicker for video, noise/FFT for image) --
    they are not interchangeable and intentionally are not unified into one shared
    function. `core.decision` additionally uses natural-log (base-e) odds for its own
    fusion step. Do not compare a raw posterior_log_odds value from this function against
    one from core.decision or assume they're on the same scale -- convert through
    probability (prob_ai/prob_real) instead, which is base-independent. Every threshold in
    this module's evaluate_taxonomy_classification (e.g. the 62.0/48.0 ai_pct cutoffs) and
    SENSITIVITY_PRIORS below are calibrated specifically against this function's base-10
    probability mapping; changing the base here requires re-deriving all of them.
    Returns:
        posterior_log_odds: float (base-10; see note above)
        prob_ai: float in [0.0, 1.0]
        prob_real: float in [0.0, 1.0]
    """
    prior_log_odds = SENSITIVITY_PRIORS.get(sensitivity.lower(), 0.0)

    # Correlation discounting: when both bilateral smoothness and noise flatness fire together,
    # discount the second cue by 0.70x to account for physical redundancy
    weighted_sum = 0.0
    has_noise = "sensor_noise" in log_lrs

    for k, v in log_lrs.items():
        weight = 1.0
        if k == "surface_smoothness" and has_noise:
            weight = 0.70
        weighted_sum += v * weight

    total_posterior = prior_log_odds + weighted_sum

    # Limit odds to prevent numerical overflow
    clamped_log_odds = float(np.clip(total_posterior, -6.0, 6.0))
    odds = float(10.0 ** clamped_log_odds)
    p_ai = odds / (1.0 + odds)
    p_real = 1.0 - p_ai

    return round(total_posterior, 3), p_ai, p_real


# Below this AI score a local noise inconsistency is treated as ordinary processing, not as AI editing.
_INPAINT_MIN_AI_PCT = 35.0
# A composite / AI-enhanced verdict claims AI involvement, so the score itself must clearly lean synthetic.
# Between the two thresholds the noise inconsistency is neither dismissed as processing nor called AI.
_COMPOSITE_MIN_AI_PCT = 60.0
# A digital-art finding with no declaration (watermark / label / AI metadata) names the image AI or CGI only if the score
# agrees: saturation and flat colour alone also describe corals, sunsets and cartoons-by-hand.
_ART_ONLY_MIN_AI_PCT = 58.0


def evaluate_taxonomy_classification(
    ai_pct: float = 0.0,
    real_pct: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
    watermark_data: Optional[Dict[str, Any]] = None,
    cutout_data: Optional[Dict[str, Any]] = None,
    scanned_data: Optional[Dict[str, Any]] = None,
    art_data: Optional[Dict[str, Any]] = None,
    noise_mean: float = 2.0,
    smoothness: float = 2.0,
    is_square_gen: bool = False,
    is_canonical_gen: bool = False,
    prob_ai: Optional[float] = None,
    prob_real: Optional[float] = None,
    watermark_detected: Optional[bool] = None,
    cutout_detected: Optional[bool] = None,
    scanned_detected: Optional[bool] = None,
    art_detected: Optional[bool] = None,
    screenshot_data: Optional[Dict[str, Any]] = None,
    screenshot_detected: Optional[bool] = None,
    inpainting_data: Optional[Dict[str, Any]] = None,
    inpainting_detected: Optional[bool] = None,
    screen_recapture_data: Optional[Dict[str, Any]] = None,
    screen_recapture_detected: bool = False,
    metadata_absent: bool = False,
    synthetic_signal_count: int = 99,
    text_regions_count: int = 0,
) -> Tuple[str, str, str, List[str]]:
    """
    Evaluates converging forensic signals, provenance, device profiles, and physical indicators to assign
    each image strictly to one of the taxonomy states:
    1. AUTHENTIC_REAL_PHOTOGRAPH: Authentic real-life camera/mobile capture or physical scan.
    2. AUTHENTIC_EDITED: Real photo with conventional edits (cropping, background removal, Canva, Photoshop).
    3. AI_ENHANCED_COMPOSITE: Real photo augmented via neural models (inpainting, Topaz Photo AI and similar enhancers).
    4. FULLY_AI_GENERATED: Synthesized end-to-end via generative diffusion/transformer models.
    5. SCREENSHOTS: Screen captures from any device (Mobile, Tablet, Laptop, Desktop) in any orientation,
       with sub-classification of inner content (Authentic, AI-Enhanced, AI-Generated).
    """
    from image_detector.schemas import ImageTaxonomyState

    if prob_ai is not None:
        ai_pct = prob_ai * 100.0 if prob_ai <= 1.0 else prob_ai
    if prob_real is not None:
        real_pct = prob_real * 100.0 if prob_real <= 1.0 else prob_real

    c = _TaxonomyInputs(
        ai_pct=ai_pct, real_pct=real_pct, noise_mean=noise_mean, smoothness=smoothness,
        is_square_gen=is_square_gen, is_canonical_gen=is_canonical_gen,
        metadata=dict(metadata or {}),
        watermark=_with_flag(watermark_data, "watermark_detected", watermark_detected),
        cutout=_with_flag(cutout_data, "is_cutout", cutout_detected),
        scanned=_with_flag(scanned_data, "is_scanned", scanned_detected),
        art=_with_flag(art_data, "is_digital_art", art_detected),
        screenshot=_with_flag(screenshot_data, "is_screenshot", screenshot_detected),
        inpainting=_with_flag(inpainting_data, "is_manipulated", inpainting_detected),
        screen_recapture=dict(screen_recapture_data or {}),
        recapture_flag=bool(screen_recapture_detected),
        metadata_absent=bool(metadata_absent),
        synthetic_signal_count=int(synthetic_signal_count),
        text_count=int(text_regions_count),
    )
    for stage in _TAXONOMY_STAGES:
        outcome = stage(c, ImageTaxonomyState)
        if outcome is not None:
            state, reasons = outcome
            return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons
    raise AssertionError("the final taxonomy stage always returns a state")  # pragma: no cover


def _with_flag(data: Optional[Dict[str, Any]], key: str, override: Optional[bool]) -> Dict[str, Any]:
    out = dict(data or {})
    if override is not None:
        out[key] = override
    return out


@dataclass
class _TaxonomyInputs:
    """Normalised evidence handed to each taxonomy stage (detector dicts plus derived booleans)."""

    ai_pct: float
    real_pct: float
    noise_mean: float
    smoothness: float
    is_square_gen: bool
    is_canonical_gen: bool
    metadata: Dict[str, Any]
    watermark: Dict[str, Any]
    cutout: Dict[str, Any]
    scanned: Dict[str, Any]
    art: Dict[str, Any]
    screenshot: Dict[str, Any]
    inpainting: Dict[str, Any]
    screen_recapture: Dict[str, Any]
    recapture_flag: bool
    metadata_absent: bool
    synthetic_signal_count: int
    text_count: int

    @property
    def is_digital_art(self) -> bool:
        return bool(self.art.get("is_digital_art"))

    @property
    def is_screenshot(self) -> bool:
        return bool(self.screenshot.get("is_screenshot"))

    @property
    def is_inpainted(self) -> bool:
        return bool(self.inpainting.get("is_manipulated"))

    @property
    def has_watermark(self) -> bool:
        return bool(self.watermark.get("watermark_detected"))

    @property
    def has_ai_iptc(self) -> bool:
        return (self.metadata.get("iptc_digital_source_type") == "trainedAlgorithmicMedia"
                or self.metadata.get("photoshop_credit") == "Made with Google AI")

    @property
    def has_pure_ai_meta(self) -> bool:
        return bool(self.metadata.get("ai_signature_found") and not self.metadata.get("ai_enhancer_signature_found"))

    @property
    def has_camera(self) -> bool:
        return bool(self.metadata.get("camera_make") and self.metadata.get("camera_model"))

    @property
    def is_scanned(self) -> bool:
        return bool(self.scanned.get("is_scanned"))

    @property
    def is_ai_enhancer(self) -> bool:
        return bool(self.metadata.get("ai_enhancer_signature_found"))

    @property
    def canvas_matches_generator(self) -> bool:
        return self.is_canonical_gen or self.is_square_gen

    @property
    def declares_ai(self) -> bool:
        """Watermark, embedded 'made with AI' label, or a pure-AI metadata signature (all unauthenticated declarations)."""
        return self.has_watermark or self.has_ai_iptc or self.has_pure_ai_meta


_Outcome = Optional[Tuple[Any, List[str]]]


def _stage_screenshot(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """1. Screen captures (any device/orientation), classified by the nature of the displayed content."""
    if not c.is_screenshot:
        return None
    device = c.screenshot.get("device_type", "Device")
    orient = c.screenshot.get("orientation", "Portrait")
    screen_res = c.screenshot.get("screen_resolution", "")
    details = c.screenshot.get("details", f"{device} screen capture in {orient} orientation")
    reasons: List[str] = []

    if c.declares_ai or c.ai_pct >= 62.0 or (c.is_digital_art and c.ai_pct >= _ART_ONLY_MIN_AI_PCT):
        reasons.append(f"Screen capture from {device} ({orient} orientation) displaying fully AI-generated media")
        if c.has_watermark:
            reasons.append(c.watermark.get("details", "AI generator watermark detected inside display"))
        if c.has_ai_iptc:
            reasons.append("Embedded metadata declares: 'Made with Google AI' (unauthenticated label)")
        if c.is_digital_art:
            reasons.append(c.art.get("details", "AI digital artwork / synthetic rendering displayed on screen"))
        reasons.append(details)
        return S.AI_GENERATED_SCREENSHOT, reasons

    if c.is_ai_enhancer or c.is_inpainted:
        reasons.append(f"Screen capture from {device} ({orient} orientation) displaying AI-enhanced / spliced media")
        if c.is_inpainted:
            reasons.append(c.inpainting.get("details", "Localized generative inpainting / composite detected"))
        if c.is_ai_enhancer:
            reasons.append(c.metadata.get("signature_details", "Neural image enhancement / upscaling signature detected"))
        reasons.append(details)
        return S.AI_ENHANCED_SCREENSHOT, reasons

    reasons.append(f"Authentic digital screen capture from {device} ({orient} orientation, {screen_res})")
    reasons.append(details)
    reasons.append("No sign of AI generation or editing was found in the captured content (an estimate, not proof)")
    return S.AUTHENTIC_SCREENSHOT, reasons


def _stage_declared_or_art_synthesis(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """2. Fully AI-generated: explicit declaration (watermark / label / metadata) or non-optical digital art."""
    if not (c.declares_ai or (c.is_digital_art and not c.has_camera and not c.is_scanned and c.ai_pct >= _ART_ONLY_MIN_AI_PCT)):
        return None
    reasons: List[str] = []
    if (c.is_digital_art and c.art.get("visual_medium") == "Digital 3D CGI / AI Neural Painting"
            and not c.has_watermark and not c.has_ai_iptc and not c.has_pure_ai_meta):
        reasons.append("Looks like a 3D or CGI-style rendering (flat colour and no camera grain); the tool cannot confirm how it was made")
        reasons.append(c.art.get("details", "Absence of natural camera-sensor noise"))
        return S.PROCEDURAL_CGI_SYNTHETIC, reasons

    if c.has_watermark:
        reasons.append(c.watermark.get("details", "AI generator watermark detected in corner"))
    if c.has_ai_iptc:
        reasons.append("Embedded metadata declares: 'Made with Google AI' (trainedAlgorithmicMedia; unauthenticated label)")
    if c.has_pure_ai_meta:
        reasons.append(c.metadata.get("signature_details", "AI generator footprint detected in metadata"))
    if c.is_digital_art:
        reasons.append(c.art.get("details", "AI digital artwork / synthetic painting style detected"))
        reasons.append("Non-optical color rendering and absence of camera-like fine grain")
        if c.cutout.get("is_cutout"):
            reasons.append("Synthetic 3D asset with transparent alpha background cutout")
    else:
        reasons.append(f"Synthetic generation metrics (Bilateral Smoothness: {c.smoothness:.2f}, Noise residual: {c.noise_mean:.2f})")
    if c.canvas_matches_generator:
        reasons.append("Canvas dimensions match standard generative model diffusion canvas")
    return S.FULLY_AI_GENERATED, reasons


def _stage_enhanced_composite(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """3. AI-enhanced / composite: explicit enhancer, composite label or inpainting evidence."""
    composite_label = c.metadata.get("iptc_digital_source_type") == "compositeWithTrainedAlgorithmicMedia"
    # Local noise inconsistency alone (HDR, portrait-mode blur, selective retouching) is not evidence of AI: it
    # only counts as a composite when the pixel score also leans synthetic.
    inpainting_counts = c.is_inpainted and c.ai_pct >= _COMPOSITE_MIN_AI_PCT
    if not (c.is_ai_enhancer or composite_label or inpainting_counts):
        return None
    reasons: List[str] = []
    if inpainting_counts:
        reasons.append(c.inpainting.get("details", "Localized generative inpainting / composite detected"))
    if c.is_ai_enhancer:
        reasons.append(c.metadata.get("signature_details", "Neural image enhancement / upscaling software detected"))
        if c.metadata.get("camera_make"):
            reasons.append(f"Original base capture from camera hardware: {c.metadata.get('camera_make')} {c.metadata.get('camera_model') or ''}".strip())
    return S.AI_ENHANCED_COMPOSITE, reasons


def _stage_camera_base_heavily_altered(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """3b. Score-band fallback: genuine camera base with overwhelming synthetic pixel evidence but no explicit enhancer
    signature (e.g. a generic AI upscaler/denoiser that leaves no metadata footprint)."""
    if not (c.has_camera and not c.is_scanned and c.ai_pct >= 62.0):
        return None
    cam = f"{c.metadata.get('camera_make', '') or ''} {c.metadata.get('camera_model', '') or ''}".strip()
    return S.AI_ENHANCED_COMPOSITE, [
        f"Camera hardware tags present ({cam}); unauthenticated metadata",
        f"Overwhelming synthetic-leaning pixel evidence despite camera tags ({c.ai_pct:.1f}% AI) -- "
        "consistent with an AI upscaler/denoiser/generative-fill pass, or with the phone's own beautify/HDR/night-mode processing, that left no metadata footprint",
    ]


def _stage_pixel_evidence_synthesis(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """4. High-confidence generative synthesis from pixel evidence alone (no declaration, no camera)."""
    if not ((c.ai_pct >= 58.0 or (c.ai_pct >= 48.0 and c.canvas_matches_generator)) and not c.has_camera and not c.is_scanned):
        return None
    has_real_noise = c.noise_mean > 1.85 and c.smoothness > 1.60
    # Corroboration gate: with no camera / AI signature / C2PA marker (the common state of downloaded or re-shared images)
    # the blended score alone is not enough; require >= 2 independently synthetic-leaning pixel signals.
    lacks_corroboration = c.metadata_absent and c.synthetic_signal_count < 2
    if has_real_noise or lacks_corroboration:
        return None
    reasons = [
        f"High posterior probability of generative synthesis ({c.ai_pct:.1f}% AI)",
        "Synthetic bilateral surface over-smoothing and absence of Poisson sensor noise",
    ]
    if c.canvas_matches_generator:
        reasons.append("Canvas dimensions match standard generative model diffusion canvas")
    if c.cutout.get("is_cutout"):
        reasons.append("Synthetic character / asset rendered on isolated solid background canvas")
    return S.FULLY_AI_GENERATED, reasons


def _stage_screen_recapture(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """5. Authentic photograph of a physical display."""
    if not (c.screen_recapture.get("is_screen_recapture") or c.recapture_flag):
        return None
    reasons = ["Optical camera recapture of physical display screen (CRT/LCD/OLED)"]
    if c.screen_recapture.get("details"):
        reasons.append(c.screen_recapture["details"])
    if c.metadata.get("camera_make"):
        reasons.append(f"Recaptured with hardware camera: {c.metadata.get('camera_make')} {c.metadata.get('camera_model') or ''}".strip())
    return S.AUTHENTIC_RECAPTURED_SCREEN, reasons


def _stage_graphic_edit(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """6. Conventionally edited / graphic-design photograph, or a synthetic cutout asset."""
    is_cutout = bool(c.cutout.get("is_cutout"))
    is_graphic_edit = bool(c.metadata.get("graphic_editor_signature_found"))
    has_graphic_text = c.text_count >= 3
    if not (is_cutout or is_graphic_edit or (has_graphic_text and not c.is_digital_art)):
        return None
    reasons: List[str] = []
    # Only authentic if verified camera hardware OR genuine sensor noise >= 1.35 with low AI probability
    if (c.has_camera or c.noise_mean >= 1.35 or c.real_pct >= 50.0) and c.ai_pct < 50.0:
        if is_cutout:
            reasons.append(c.cutout.get("details", "Background removal or studio solid background replacement detected"))
        if is_graphic_edit:
            reasons.append(c.metadata.get("signature_details", "Graphic layout composition software detected"))
        if has_graphic_text:
            reasons.append(f"Graphic design typography / text elements overlaid on image ({c.text_count} text blocks detected)")
        reasons.append(f"The noise evidence does not point to AI (noise residual {c.noise_mean:.2f}, {c.ai_pct:.0f}% AI score)")
        return S.AUTHENTIC_EDITED, reasons
    if c.ai_pct >= 50.0 or c.noise_mean < 1.10:
        reasons.append(c.cutout.get("details", "Isolated synthetic character / object on solid background canvas"))
        reasons.append("Absence of camera-like fine grain across subject boundaries")
        return S.FULLY_AI_GENERATED, reasons
    return None


def _stage_local_processing(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """6b. Uneven sensor noise with a real-leaning score: ordinary local processing, not generative editing."""
    if not (c.is_inpainted and not c.is_scanned and c.ai_pct < _INPAINT_MIN_AI_PCT):
        return None
    return S.AUTHENTIC_EDITED, [
        f"Sensor noise differs between regions of the image (inconsistency ratio: {c.inpainting.get('noise_inconsistency', 0.0):.2f})",
        "The score does not indicate AI: uneven noise is typical of HDR, portrait-mode blur, beautify filters and selective retouching",
    ]


def _stage_authentic_photograph(c: _TaxonomyInputs, S: Any) -> _Outcome:
    """7. Default: authentic camera / phone photograph (or a scan of a physical print)."""
    m = c.metadata
    reasons: List[str] = []
    if c.is_scanned:
        reasons.append(c.scanned.get("details", "High-resolution flatbed scan of physical photographic print"))
        reasons.append("Consistent with a scan of a printed photograph (heuristic)")
        return S.AUTHENTIC_REAL_PHOTOGRAPH, reasons
    if m.get("camera_make"):
        reasons.append(f"Camera hardware EXIF tags: {m.get('camera_make')} {m.get('camera_model') or ''}".strip())
    if m.get("has_optical_parameters"):
        optics = []
        if m.get("focal_length"):
            optics.append(f"f={m['focal_length']}mm")
        if m.get("f_number"):
            optics.append(f"f/{m['f_number']}")
        if m.get("iso"):
            optics.append(f"ISO {m['iso']}")
        if optics:
            reasons.append(f"Physical lens optical parameters: {', '.join(optics)}")
    reasons.append(f"Nothing in the analysis points to AI generation or editing (noise residual {c.noise_mean:.2f}, smoothness {c.smoothness:.2f}); an estimate, not proof")
    return S.AUTHENTIC_REAL_PHOTOGRAPH, reasons


_TAXONOMY_STAGES = (
    _stage_screenshot,
    _stage_declared_or_art_synthesis,
    _stage_enhanced_composite,
    _stage_camera_base_heavily_altered,
    _stage_pixel_evidence_synthesis,
    _stage_screen_recapture,
    _stage_graphic_edit,
    _stage_local_processing,
    _stage_authentic_photograph,
)
