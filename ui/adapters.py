"""ui.adapters: thin per-file adapters over the package pipelines (process_single_*) and the batch loop.

The analysis sequence itself lives in each package's pipeline.run(); these functions only handle UI concerns (gate
short-circuit shape, display profile blocks, keyframe file, error packaging)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


from audio_detector import AudioAIDetector
from image_detector import ImageAIDetector, ImageContentAnalyzer as ContentAnalyzer, ImageModelAttributionEngine as ModelAttributionEngine
from video_detector import VideoAIDetector
from audio_detector.dimension_checks import check_audio_gates, gate_short_circuit_result as audio_gate_short_circuit_result, summarize_for_evidence_trail as audio_summarize_for_evidence_trail
from video_detector.dimension_checks import check_video_gates, gate_short_circuit_result as video_gate_short_circuit_result, summarize_for_evidence_trail as video_summarize_for_evidence_trail
from image_detector.dimension_checks import check_image_gates, gate_short_circuit_result, summarize_for_evidence_trail
from audio_detector.pipeline import AudioForensicPipeline
from image_detector.pipeline import ImageForensicPipeline
from video_detector.content import VideoContentAnalyzer
from video_detector.pipeline import VideoForensicPipeline
from ui.profile_view import add_ui_profile_blocks
from ui.validators import validate_file


def process_single_image(
    img_path: str,
    filename: str,
    detector: ImageAIDetector,
    content_analyzer: ContentAnalyzer,
    attribution_engine: ModelAttributionEngine,
    sensitivity: str = "balanced",
    source: str = "User Upload",
) -> Dict[str, Any]:
    """Runs end-to-end NIST-aligned forensic pipeline on a single image."""
    # 0. Pre-analysis gates (before decoding): hard-block hash list + out-of-scope scientific formats.
    gates = check_image_gates(img_path)
    if gates["triggered"]:
        stub = gate_short_circuit_result(img_path, gates)
        return {
            "filename": filename,
            "path": img_path,
            "source": source,
            "success": True,
            "gate_blocked": True,
            "gate": gates,
            "decision": stub,
            "file_res": {"readable": True},
        }

    file_res = validate_file(img_path)
    if not file_res.get("readable"):
        return {
            "filename": filename,
            "path": img_path,
            "source": source,
            "success": False,
            "error": file_res.get("error", "File corrupted or unreadable"),
            "file_res": file_res,
        }

    pipeline = ImageForensicPipeline(detector=detector, content_analyzer=content_analyzer, attribution_engine=attribution_engine)
    run = pipeline.run(img_path, sensitivity=sensitivity, source=source, filename=filename, gates=gates)
    img_profile = add_ui_profile_blocks("image", run.profile)

    return {
        "filename": filename,
        "path": img_path,
        "source": source,
        "success": True,
        "file_res": file_res,
        "image_result": run.quality,
        "ai_result": run.ai_result,
        "content_res": run.content,
        "provenance_res": run.provenance,
        "attribution_res": run.attribution,
        "decision": run.decision,
        "img_profile": img_profile,
        "nine_dimensions_dossier": run.nine_dimensions,
        "newbie_explanation": run.newbie_explanation,
        "dimension_report": run.dimension_report,
        "confidence_band": run.dimension_report.get("confidence_band"),
        "ood": run.dimension_report.get("ood"),
        "attribution_open_set": run.dimension_report.get("attribution_open_set"),
        "gate": gates,
        "dimension_evidence_lines": summarize_for_evidence_trail(run.dimension_report),
    }


def process_single_video(
    vid_path: str,
    filename: str,
    detector: ImageAIDetector,
    content_analyzer: ContentAnalyzer,
    audio_detector: AudioAIDetector,
    attribution_engine: ModelAttributionEngine,
    sensitivity: str = "balanced",
    cache_dir: Optional[Path] = None,
    video_detector: Optional[VideoAIDetector] = None,
) -> Dict[str, Any]:
    """Runs end-to-end NIST-aligned forensic pipeline on a single video."""
    # 0. Pre-analysis gates (before decoding): hard-block hash list + scientific-format recognition.
    gates = check_video_gates(vid_path)
    if gates["triggered"]:
        return {
            "filename": filename,
            "path": vid_path,
            "success": True,
            "gate_blocked": True,
            "gate": gates,
            "decision": video_gate_short_circuit_result(vid_path, gates),
            "file_res": {"readable": True},
        }

    file_res = validate_file(vid_path)
    if not file_res.get("readable"):
        return {
            "filename": filename,
            "path": vid_path,
            "success": False,
            "error": file_res.get("error", "Video corrupted or unreadable"),
            "file_res": file_res,
        }

    pipeline = VideoForensicPipeline(
        detector=video_detector or VideoAIDetector(frame_detector=detector),
        content_analyzer=VideoContentAnalyzer(image_content_analyzer=content_analyzer),
        audio_detector=audio_detector,
    )
    run = pipeline.run(vid_path, sensitivity=sensitivity, filename=filename, gates=gates, cache_dir=cache_dir or Path(vid_path).parent)

    return {
        "filename": filename,
        "path": vid_path,
        "success": True,
        "file_res": file_res,
        "video_result": run.video_result,
        "audio_result": run.audio_result,
        "content_res": run.content,
        "provenance_res": run.provenance,
        "attribution_res": run.attribution,
        "cross_modal_res": run.cross_modal,
        "decision": run.decision,
        "vid_profile": add_ui_profile_blocks("video", run.profile),
        "tmp_kf_path": run.keyframe_path,
        "kf_ai": run.keyframe_ai,
        "nine_dimensions_dossier": run.nine_dimensions,
        "newbie_explanation": run.newbie_explanation,
        "dimension_report": run.dimension_report,
        "confidence_band": run.dimension_report.get("confidence_band"),
        "ood": run.dimension_report.get("ood"),
        "attribution_open_set": run.dimension_report.get("attribution_open_set"),
        "gate": gates,
        "dimension_evidence_lines": video_summarize_for_evidence_trail(run.dimension_report),
    }


def process_single_audio(
    aud_path: str,
    filename: str,
    audio_detector: AudioAIDetector,
    content_analyzer: Any,
    attribution_engine: Any,
    sensitivity: str = "balanced",
) -> Dict[str, Any]:
    """Runs end-to-end NIST-aligned forensic pipeline on a single audio file."""
    # 0. Pre-analysis gates (before decoding): hard-block hash list + symbolic-music recognition.
    gates = check_audio_gates(aud_path)
    if gates["triggered"]:
        return {
            "filename": filename,
            "path": aud_path,
            "success": True,
            "gate_blocked": True,
            "gate": gates,
            "decision": audio_gate_short_circuit_result(aud_path, gates),
            "file_res": {"readable": True},
        }

    file_res = validate_file(aud_path)
    if not file_res.get("readable"):
        return {
            "filename": filename,
            "path": aud_path,
            "success": False,
            "error": file_res.get("error", "Audio corrupted or unreadable"),
            "file_res": file_res,
        }

    pipeline = AudioForensicPipeline(detector=audio_detector, content_analyzer=content_analyzer, attribution_engine=attribution_engine)
    run = pipeline.run(aud_path, sensitivity=sensitivity, filename=filename, gates=gates)
    aud_profile = add_ui_profile_blocks("audio", run.profile)

    return {
        "filename": filename,
        "path": aud_path,
        "success": True,
        "file_res": file_res,
        "sr": run.sample_rate,
        "duration": run.duration,
        "provenance_res": run.provenance,
        "content_res": run.scene,
        "audio_result": run.ai_result,
        "aud_profile": aud_profile,
        "attribution_res": run.attribution,
        "decision": run.decision,
        "spec_img": run.ai_result.get("spectrogram_image"),
        "nine_dimensions_dossier": run.nine_dimensions,
        "newbie_explanation": run.newbie_explanation,
        "dimension_report": run.dimension_report,
        "confidence_band": run.dimension_report.get("confidence_band"),
        "ood": run.dimension_report.get("ood"),
        "attribution_open_set": run.dimension_report.get("attribution_open_set"),
        "gate": gates,
        "dimension_evidence_lines": audio_summarize_for_evidence_trail(run.dimension_report),
    }


def run_batch_pipeline(
    items: List[Dict[str, str]],
    modality: str,
    detectors: Dict[str, Any],
    sensitivity: str = "balanced",
    cache_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """Executes batch forensic analysis across a list of media items."""
    results: List[Dict[str, Any]] = []
    total = len(items)

    for idx, item in enumerate(items):
        path = item["path"]
        filename = item["filename"]

        if progress_callback:
            progress_callback(idx + 1, total, filename)

        try:
            if modality == "image":
                res = process_single_image(
                    img_path=path,
                    filename=filename,
                    detector=detectors["detector"],
                    content_analyzer=detectors["content_analyzer"],
                    attribution_engine=detectors["attribution_engine"],
                    sensitivity=sensitivity,
                    source=item.get("source", "User Upload"),
                )
            elif modality == "video":
                res = process_single_video(
                    vid_path=path,
                    filename=filename,
                    detector=detectors["detector"],
                    content_analyzer=detectors["content_analyzer"],
                    audio_detector=detectors["audio_detector"],
                    attribution_engine=detectors["attribution_engine"],
                    sensitivity=sensitivity,
                    cache_dir=cache_dir,
                    video_detector=detectors.get("video_detector"),
                )
            elif modality == "audio":
                res = process_single_audio(
                    aud_path=path,
                    filename=filename,
                    audio_detector=detectors["audio_detector"],
                    content_analyzer=detectors["content_analyzer"],
                    attribution_engine=detectors["attribution_engine"],
                    sensitivity=sensitivity,
                )
            else:
                res = {"filename": filename, "path": path, "success": False, "error": f"Unknown modality: {modality}"}
        except Exception as exc:
            res = {"filename": filename, "path": path, "success": False, "error": str(exc)}

        results.append(res)

    return results
