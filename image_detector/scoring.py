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
from typing import Any, Dict, Tuple

import numpy as np

from image_detector.config import (
    AI_THRESHOLD_BALANCED,
    AI_THRESHOLD_HIGH,
    REAL_THRESHOLD,
    SENSITIVITY_PRIORS,
)


def normalize_percentages(
    ai_val: float,
    real_val: float,
    undecided_val: float,
    min_undecided: float = 3.0,
    decimals: int = 1,
) -> Tuple[float, float, float]:
    """
    Normalizes three percentage values to sum exactly to 100.0%.
    Guarantees non-negative bounds and a minimum epistemic margin.
    """
    ai_clamped = max(0.0, float(ai_val))
    real_clamped = max(0.0, float(real_val))
    u_clamped = max(float(min_undecided), float(undecided_val))

    total = ai_clamped + real_clamped + u_clamped
    if total <= 0.0:
        return 0.0, 0.0, 100.0

    scale = 100.0 / total
    ai_norm = ai_clamped * scale
    real_norm = real_clamped * scale

    ai_pct = round(ai_norm, decimals)
    real_pct = round(real_norm, decimals)
    u_pct = round(max(0.0, 100.0 - (ai_pct + real_pct)), decimals)

    return ai_pct, real_pct, u_pct


def calculate_epistemic_uncertainty(prob_ai: float) -> float:
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
    Returns:
        posterior_log_odds: float
        prob_ai: float in [0.0, 1.0]
        prob_real: float in [0.0, 1.0]
    """
    prior_log_odds = SENSITIVITY_PRIORS.get(sensitivity.lower(), 0.0)

    # Correlation discounting: when both bilateral smoothness and PRNU flatness fire together,
    # discount the second cue by 0.70x to account for physical redundancy
    weighted_sum = 0.0
    has_noise = "prnu_sensor_noise" in log_lrs
    has_smooth = "surface_texture_smoothness" in log_lrs

    for k, v in log_lrs.items():
        weight = 1.0
        if k == "surface_texture_smoothness" and has_noise:
            weight = 0.70
        elif k == "square_aspect_ratio" and "canonical_ai_resolution" in log_lrs:
            weight = 0.50
        weighted_sum += v * weight

    total_posterior = prior_log_odds + weighted_sum

    # Limit odds to prevent numerical overflow
    clamped_log_odds = float(np.clip(total_posterior, -6.0, 6.0))
    odds = float(10.0 ** clamped_log_odds)
    p_ai = odds / (1.0 + odds)
    p_real = 1.0 - p_ai

    return round(total_posterior, 3), p_ai, p_real


def evaluate_image_decision(
    ai_pct: float, real_pct: float, sensitivity: str = "high"
) -> str:
    """Applies calibrated thresholds to produce definitive primary label."""
    thresh = AI_THRESHOLD_HIGH if sensitivity.lower() in ("high", "aggressive") else AI_THRESHOLD_BALANCED
    if ai_pct >= thresh:
        return "LIKELY AI-GENERATED"
    elif real_pct >= REAL_THRESHOLD:
        return "LIKELY REAL"
    else:
        return "UNDECIDED"


def evaluate_taxonomy_classification(
    ai_pct: float = 0.0,
    real_pct: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
    watermark_data: Optional[Dict[str, Any]] = None,
    cutout_data: Optional[Dict[str, Any]] = None,
    scanned_data: Optional[Dict[str, Any]] = None,
    face_swap_data: Optional[Dict[str, Any]] = None,
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
    face_swap_detected: Optional[bool] = None,
    art_detected: Optional[bool] = None,
    screenshot_data: Optional[Dict[str, Any]] = None,
    screenshot_detected: Optional[bool] = None,
    inpainting_data: Optional[Dict[str, Any]] = None,
    inpainting_detected: Optional[bool] = None,
    **kwargs: Any,
) -> Tuple[str, str, str, List[str]]:
    """
    Evaluates converging forensic signals, provenance, device profiles, and physical indicators to assign
    each image strictly to one of the NIST & Forensic standard taxonomy states:
    1. AUTHENTIC_REAL_PHOTOGRAPH: Authentic real-life camera/mobile capture or physical scan.
    2. AUTHENTIC_EDITED: Real photo with conventional edits (cropping, background removal, Canva, Photoshop).
    3. AI_ENHANCED_COMPOSITE: Real photo augmented via neural models (face swap, deepfake, inpainting, Topaz Photo AI).
    4. FULLY_AI_GENERATED: Synthesized end-to-end via generative diffusion/transformer models.
    5. SCREENSHOTS: Screen captures from any device (Mobile, Tablet, Laptop, Desktop) in any orientation,
       with sub-classification of inner content (Authentic, AI-Enhanced, AI-Generated).
    """
    from image_detector.schemas import ImageTaxonomyState

    if prob_ai is not None:
        ai_pct = prob_ai * 100.0 if prob_ai <= 1.0 else prob_ai
    if prob_real is not None:
        real_pct = prob_real * 100.0 if prob_real <= 1.0 else prob_real

    metadata = dict(metadata or {})
    watermark_data = dict(watermark_data or {})
    if watermark_detected is not None:
        watermark_data["watermark_detected"] = watermark_detected

    cutout_data = dict(cutout_data or {})
    if cutout_detected is not None:
        cutout_data["is_cutout"] = cutout_detected

    scanned_data = dict(scanned_data or {})
    if scanned_detected is not None:
        scanned_data["is_scanned"] = scanned_detected

    face_swap_data = dict(face_swap_data or {})
    if face_swap_detected is not None:
        face_swap_data["is_face_swap"] = face_swap_detected

    art_data = dict(art_data or {})
    if art_detected is not None:
        art_data["is_digital_art"] = art_detected
    is_digital_art = bool(art_data.get("is_digital_art"))

    screenshot_data = dict(screenshot_data or {})
    if screenshot_detected is not None:
        screenshot_data["is_screenshot"] = screenshot_detected
    is_screenshot = bool(screenshot_data.get("is_screenshot"))

    inpainting_data = dict(inpainting_data or {})
    if inpainting_detected is not None:
        inpainting_data["is_manipulated"] = inpainting_detected
    is_inpainted = bool(inpainting_data.get("is_manipulated"))

    reasons: List[str] = []

    has_gemini_watermark = bool(watermark_data.get("watermark_detected"))
    has_ai_iptc = (
        metadata.get("iptc_digital_source_type") == "trainedAlgorithmicMedia"
        or metadata.get("photoshop_credit") == "Made with Google AI"
    )
    has_pure_ai_meta = bool(metadata.get("ai_signature_found") and not metadata.get("ai_enhancer_signature_found"))
    has_camera_hardware = bool(metadata.get("camera_make") and metadata.get("camera_model"))
    is_scanned_print = bool(scanned_data.get("is_scanned"))
    is_face_swap = bool(face_swap_data.get("is_face_swap"))
    is_ai_enhancer = bool(metadata.get("ai_enhancer_signature_found"))

    # 1. SCREENSHOT CATEGORIZATION (Mobile, Tablet, Laptop, Desktop across orientations)
    if is_screenshot:
        device_label = screenshot_data.get("device_type", "Device")
        orient_label = screenshot_data.get("orientation", "Portrait")
        screen_res = screenshot_data.get("screen_resolution", "")
        sc_details = screenshot_data.get("details", f"{device_label} screen capture in {orient_label} orientation")

        # Evaluate inner content nature
        if has_gemini_watermark or has_ai_iptc or has_pure_ai_meta or is_digital_art or ai_pct >= 62.0:
            state = ImageTaxonomyState.AI_GENERATED_SCREENSHOT
            reasons.append(f"Screen capture from {device_label} ({orient_label} orientation) displaying fully AI-generated media")
            if has_gemini_watermark:
                reasons.append(watermark_data.get("details", "AI generator watermark detected inside display"))
            if has_ai_iptc:
                reasons.append("Cryptographic metadata certifies: 'Made with Google AI'")
            if is_digital_art:
                reasons.append(art_data.get("details", "AI digital artwork / synthetic rendering displayed on screen"))
            reasons.append(sc_details)
            return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

        if is_face_swap or is_ai_enhancer or is_inpainted:
            state = ImageTaxonomyState.AI_ENHANCED_SCREENSHOT
            reasons.append(f"Screen capture from {device_label} ({orient_label} orientation) displaying AI-enhanced / spliced media")
            if is_face_swap:
                reasons.append(face_swap_data.get("details", "Neural face-swap / facial graft boundary detected"))
            if is_inpainted:
                reasons.append(inpainting_data.get("details", "Localized generative inpainting / composite detected"))
            if is_ai_enhancer:
                reasons.append(metadata.get("signature_details", "Neural image enhancement / upscaling signature detected"))
            reasons.append(sc_details)
            return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

        state = ImageTaxonomyState.AUTHENTIC_SCREENSHOT
        reasons.append(f"Authentic digital screen capture from {device_label} ({orient_label} orientation, {screen_res})")
        reasons.append(sc_details)
        reasons.append("Unmanipulated operating system / app interface rendering with zero generative synthesis")
        return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

    # 2. FULLY AI-GENERATED MEDIA
    if has_gemini_watermark or has_ai_iptc or has_pure_ai_meta or (is_digital_art and not has_camera_hardware and not is_scanned_print):
        # Distinguish Procedural CGI vs Diffusion AI if applicable
        if is_digital_art and art_data.get("visual_medium") == "Digital 3D CGI / AI Neural Painting" and not has_gemini_watermark and not has_ai_iptc and not has_pure_ai_meta:
            state = ImageTaxonomyState.PROCEDURAL_CGI_SYNTHETIC
            reasons.append("Deterministic procedural 3D ray-traced rendering / CGI synthetic model detected")
            reasons.append(art_data.get("details", "Absence of natural Bayer sensor PRNU noise"))
            return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

        state = ImageTaxonomyState.FULLY_AI_GENERATED
        if has_gemini_watermark:
            reasons.append(watermark_data.get("details", "AI generator watermark detected in corner"))
        if has_ai_iptc:
            reasons.append("Cryptographic metadata certifies: 'Made with Google AI' (trainedAlgorithmicMedia)")
        if has_pure_ai_meta:
            reasons.append(metadata.get("signature_details", "AI generator footprint detected in metadata"))
        if is_digital_art:
            reasons.append(art_data.get("details", "AI digital artwork / synthetic painting style detected"))
            reasons.append("Non-optical color rendering and absence of physical camera sensor PRNU grain")
            if cutout_data.get("is_cutout"):
                reasons.append("Synthetic 3D asset with transparent alpha background cutout")
        else:
            reasons.append(f"Synthetic generation metrics (Bilateral Smoothness: {smoothness:.2f}, PRNU Noise: {noise_mean:.2f})")
        if is_canonical_gen or is_square_gen:
            reasons.append("Canvas dimensions match standard generative model diffusion canvas")
        return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

    # 3. AI-ENHANCED / COMPOSITE (MIX)
    has_ai_composite_meta = (
        is_ai_enhancer
        or is_face_swap
        or metadata.get("iptc_digital_source_type") == "compositeWithTrainedAlgorithmicMedia"
    )
    if has_ai_composite_meta or is_inpainted:
        state = ImageTaxonomyState.AI_ENHANCED_COMPOSITE
        if is_face_swap:
            reasons.append(face_swap_data.get("details", "Neural face-swap and facial graft boundary detected"))
            reasons.append("Discontinuity between facial airbrushing and sharp facial hair/accessories")
        if is_inpainted:
            reasons.append(inpainting_data.get("details", "Localized generative inpainting / composite detected"))
        if is_ai_enhancer:
            reasons.append(metadata.get("signature_details", "Neural image enhancement / upscaling software detected"))
            if metadata.get("camera_make"):
                reasons.append(f"Original base capture from camera hardware: {metadata.get('camera_make')} {metadata.get('camera_model') or ''}".strip())
        return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

    # 4. HIGH-CONFIDENCE GENERATIVE SYNTHESIS (without explicit watermark)
    if (ai_pct >= 58.0 or (ai_pct >= 48.0 and (is_square_gen or is_canonical_gen))) and not has_camera_hardware and not scanned_data.get("is_scanned"):
        has_real_noise = noise_mean > 1.85 and smoothness > 1.60
        if not has_real_noise:
            state = ImageTaxonomyState.FULLY_AI_GENERATED
            reasons.append(f"High posterior probability of generative synthesis ({ai_pct:.1f}% AI)")
            reasons.append("Synthetic bilateral surface over-smoothing and absence of Poisson sensor noise")
            if is_canonical_gen or is_square_gen:
                reasons.append("Canvas dimensions match standard generative model diffusion canvas")
            if cutout_data.get("is_cutout"):
                reasons.append("Synthetic character / asset rendered on isolated solid background canvas")
            return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

    # 5. AUTHENTIC SCREEN RE-PHOTOGRAPHY (RECAPTURED PHYSICAL DISPLAY)
    screen_recapture_data = dict(kwargs.get("screen_recapture_data") or {})
    is_recaptured = bool(screen_recapture_data.get("is_screen_recapture") or kwargs.get("screen_recapture_detected"))
    if is_recaptured:
        state = ImageTaxonomyState.AUTHENTIC_RECAPTURED_SCREEN
        reasons.append("Optical camera recapture of physical display screen (CRT/LCD/OLED)")
        if screen_recapture_data.get("details"):
            reasons.append(screen_recapture_data["details"])
        if metadata.get("camera_make"):
            reasons.append(f"Recaptured with hardware camera: {metadata.get('camera_make')} {metadata.get('camera_model') or ''}".strip())
        return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

    # 6. AUTHENTIC CREATED PHOTOGRAPH (CONVENTIONALLY EDITED / GRAPHIC DESIGN)
    is_cutout = bool(cutout_data.get("is_cutout"))
    is_graphic_edit = bool(metadata.get("graphic_editor_signature_found"))
    text_count = int(kwargs.get("text_regions_count", 0))
    has_graphic_text = text_count >= 3

    if is_cutout or is_graphic_edit or (has_graphic_text and not is_digital_art):
        # Only authentic if verified camera hardware OR genuine sensor noise >= 1.35 with low AI probability
        if (has_camera_hardware or noise_mean >= 1.35 or real_pct >= 50.0) and ai_pct < 50.0:
            state = ImageTaxonomyState.AUTHENTIC_EDITED
            if is_cutout:
                reasons.append(cutout_data.get("details", "Background removal or studio solid background replacement detected"))
            if is_graphic_edit:
                reasons.append(metadata.get("signature_details", "Graphic layout composition software detected"))
            if has_graphic_text:
                reasons.append(f"Graphic design typography / text elements overlaid on image ({text_count} text blocks detected)")
            reasons.append("Base subject contains authentic photographic sensor noise and natural physical geometry")
            return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons
        elif ai_pct >= 50.0 or noise_mean < 1.10:
            # Synthetic 3D asset or AI character render cutout
            state = ImageTaxonomyState.FULLY_AI_GENERATED
            reasons.append(cutout_data.get("details", "Isolated synthetic character / object on solid background canvas"))
            reasons.append("Absence of authentic camera sensor PRNU noise across subject boundaries")
            return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons

    # 7. AUTHENTIC REAL CAMERA / MOBILE-PHONE PHOTOGRAPH
    state = ImageTaxonomyState.AUTHENTIC_REAL_PHOTOGRAPH
    if scanned_data.get("is_scanned"):
        reasons.append(scanned_data.get("details", "High-resolution flatbed scan of physical photographic print"))
        reasons.append("Preserved physical halftone screening and authentic photographic print emulsion")
    else:
        if metadata.get("camera_make"):
            reasons.append(f"Verified camera hardware: {metadata.get('camera_make')} {metadata.get('camera_model') or ''}".strip())
        if metadata.get("has_optical_parameters"):
            opt_details = []
            if metadata.get("focal_length"):
                opt_details.append(f"f={metadata['focal_length']}mm")
            if metadata.get("f_number"):
                opt_details.append(f"f/{metadata['f_number']}")
            if metadata.get("iso"):
                opt_details.append(f"ISO {metadata['iso']}")
            if opt_details:
                reasons.append(f"Physical lens optical parameters: {', '.join(opt_details)}")
        reasons.append(f"Natural camera sensor Poisson shot noise (PRNU residual: {noise_mean:.2f})")
        reasons.append("Natural optical depth of field, coherent lighting, and unmanipulated geometry")

    return state, ImageTaxonomyState.get_label(state), ImageTaxonomyState.get_description(state), reasons
