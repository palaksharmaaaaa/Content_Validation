"""
Unified Decision Engine for media validation, content usability,
and multi-modal AI generation detection (Image, Video, Audio).
"""
from typing import Any, Dict, Optional


def generate_final_decision(
    file_validation: Dict[str, Any],
    quality_result: Dict[str, Any],
    ai_result: Optional[Dict[str, Any]] = None,
    audio_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates an aggregated verdict combining file readability, visual quality,
    coverage metrics, and AI synthetic percentages.
    """
    if not file_validation.get("readable", False):
        return {
            "content_valid": False,
            "final_status": "INVALID_FILE",
            "reason": file_validation.get("error", "File is unreadable or corrupted."),
            "modality_scores": {},
        }

    # Video content status check
    if quality_result.get("content_status") == "INVALID":
        return {
            "content_valid": False,
            "final_status": "INVALID_CONTENT",
            "reason": "Video is largely blank, dark, or lacks usable visual content.",
            "modality_scores": _build_modality_scores(ai_result, audio_result, quality_result),
        }

    # Image blankness check
    if quality_result.get("is_blank") is True:
        return {
            "content_valid": False,
            "final_status": "BLANK_CONTENT",
            "reason": "Uploaded image is blank or has negligible information entropy.",
            "modality_scores": _build_modality_scores(ai_result, audio_result, quality_result),
        }

    # Partial validity
    if quality_result.get("content_status") == "PARTIALLY_VALID":
        final_status = "PARTIALLY_VALID"
        reason = "Content is partially usable, but includes blurred or blank segments."
    elif quality_result.get("quality") == "POOR":
        final_status = "LOW_QUALITY"
        reason = "Content is usable but exhibits low resolution, heavy blur, or extreme exposure."
    else:
        final_status = "VALID"
        reason = "Content is clear, well-formed, and meets quality standards."

    modality_scores = _build_modality_scores(ai_result, audio_result, quality_result)

    # Flag if AI probability is dominant
    ai_detected = False
    if ai_result and ai_result.get("ai_percentage", 0.0) >= 65.0:
        ai_detected = True
    elif quality_result.get("ai_video_rating", {}).get("ai_percentage", 0.0) >= 65.0:
        ai_detected = True

    if ai_detected:
        reason += " [Notice: High likelihood of synthetic/AI-generated media]."

    return {
        "content_valid": True,
        "final_status": final_status,
        "reason": reason,
        "ai_detected": ai_detected,
        "modality_scores": modality_scores,
    }


def _build_modality_scores(
    ai_result: Optional[Dict[str, Any]],
    audio_result: Optional[Dict[str, Any]],
    quality_result: Dict[str, Any],
) -> Dict[str, Any]:
    scores = {}

    # Image Modality
    if ai_result:
        scores["image"] = {
            "ai_percentage": ai_result.get("ai_percentage", 0.0),
            "real_percentage": ai_result.get("real_percentage", 0.0),
            "undecided_percentage": ai_result.get("undecided_percentage", 100.0),
            "prediction": ai_result.get("prediction", "UNDECIDED"),
        }

    # Video Modality (from sampled frames)
    if "ai_video_rating" in quality_result:
        scores["video"] = quality_result["ai_video_rating"]

    # Audio Modality
    if audio_result:
        scores["audio"] = audio_result
    else:
        # Default placeholder when audio track is absent or not analyzed
        scores["audio"] = {
            "ai_percentage": 0.0,
            "real_percentage": 0.0,
            "undecided_percentage": 100.0,
            "status": "No audio track analyzed",
        }

    return scores