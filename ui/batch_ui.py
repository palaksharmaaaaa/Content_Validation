"""
Batch Analysis UI Components for Multi-File Forensic Inspection.
Provides:
1. Executive Batch Summary Metrics & Risk Dashboard.
2. Comparative Batch Overview Data Table.
3. Batch Exporting (CSV and Full NIST-Compliant JSON Dossier).
4. Interactive Media Drill-Down Selector.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

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


def render_batch_summary_dashboard(batch_results: List[Dict[str, Any]], modality: str = "image") -> None:
    """Renders high-level executive metrics for a batch of analyzed media files."""
    if not batch_results:
        return

    valid_results = [r for r in batch_results if r.get("success")]
    total = len(batch_results)
    successful = len(valid_results)
    failed = total - successful

    synthetic_count = sum(
        1 for r in valid_results
        if r.get("decision", {}).get("final_status") in ("LIKELY_SYNTHETIC", "LIKELY AI-GENERATED")
    )
    authentic_count = sum(
        1 for r in valid_results
        if r.get("decision", {}).get("final_status") in ("LIKELY_AUTHENTIC", "LIKELY REAL")
    )
    partial_or_und = successful - (synthetic_count + authentic_count)

    avg_p_ai = (
        sum(r.get("decision", {}).get("authenticity_probabilities", {}).get("p_ai", 0.0) for r in valid_results) / successful
        if successful > 0 else 0.0
    )
    avg_p_real = (
        sum(r.get("decision", {}).get("authenticity_probabilities", {}).get("p_real", 0.0) for r in valid_results) / successful
        if successful > 0 else 0.0
    )

    st.subheader(f"📊 Batch Forensic Summary ({total} Files Evaluated)")

    synth_pct_str = f"{(synthetic_count / successful * 100):.0f}%" if successful > 0 else "0%"
    auth_pct_str = f"{(authentic_count / successful * 100):.0f}%" if successful > 0 else "0%"
    und_pct_str = f"{(partial_or_und / successful * 100):.0f}%" if successful > 0 else "0%"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📁 Total Files", f"{total} items", f"{failed} errors" if failed > 0 else "All parsed")
    col2.metric(
        "🚨 Likely Synthetic",
        f"{synthetic_count} ({synth_pct_str})" if successful > 0 else "0",
        f"Avg P(AI): {avg_p_ai:.1f}%",
        delta_color="inverse",
    )
    col3.metric(
        "✅ Likely Authentic",
        f"{authentic_count} ({auth_pct_str})" if successful > 0 else "0",
        f"Avg P(Real): {avg_p_real:.1f}%",
        delta_color="normal",
    )
    col4.metric(
        "❓ Undetermined / Edited",
        f"{partial_or_und} ({und_pct_str})" if successful > 0 else "0",
    )

    # Risk alert summary
    if successful > 0:
        synth_ratio = synthetic_count / successful
        if synth_ratio >= 0.5:
            st.error(
                f"🚨 **High Synthetic Infiltration Detected:** {synthetic_count} of {successful} "
                f"({synth_ratio:.0%}) files in this batch show definitive generative AI / deepfake artifacts."
            )
        elif authentic_count / successful >= 0.7:
            st.success(
                f"✅ **High Authenticity Rate:** {authentic_count} of {successful} files exhibit authentic "
                "sensor noise (PRNU), natural optical frequencies, and valid hardware metadata."
            )
        else:
            st.info(
                f"ℹ️ **Mixed-Provenance Batch:** Forensic signatures indicate a combination of synthetic, "
                "altered, and genuine camera media."
            )


def render_batch_overview_table(batch_results: List[Dict[str, Any]], modality: str = "image") -> None:
    """Renders a comparative data table and export buttons for batch items."""
    if not batch_results:
        return

    table_data = []
    for r in batch_results:
        fname = r.get("filename", "Unknown")
        if not r.get("success"):
            table_data.append({
                "Filename": fname,
                "Category": "Unreadable / Corrupted",
                "Status": "❌ Error",
                "Verdict": "CORRUPTED_OR_UNREADABLE",
                "P(AI)": "N/A",
                "P(Real)": "N/A",
                "Attributed Model": "N/A",
                "Scene & Setting": "N/A",
                "Content / Entities": r.get("error", "Error"),
                "Suspicious Area / Duration": "N/A",
                "C2PA": "N/A",
            })
            continue

        decision = r.get("decision", {})
        status = decision.get("final_status", "UNDETERMINED")
        probs = decision.get("authenticity_probabilities", {})
        p_ai = probs.get("p_ai", 0.0)
        p_real = probs.get("p_real", 0.0)
        attr = decision.get("model_attribution", {})
        inv = decision.get("content_inventory", {})
        prov = decision.get("provenance", {})
        loc = decision.get("localization", {})
        tax_label = decision.get("taxonomy_label") or status

        # Status badge
        if status in ("LIKELY_SYNTHETIC", "LIKELY AI-GENERATED"):
            status_badge = "🚨 AI Generated"
        elif status in ("LIKELY_AUTHENTIC", "LIKELY REAL"):
            status_badge = "✅ Camera Capture"
        elif status == "PARTIALLY_SYNTHETIC_OR_EDITED":
            status_badge = "⚠️ Partially Synthetic"
        else:
            status_badge = "❓ Undetermined"

        # Content Summary
        humans = f"{inv.get('persons_count', 0)} person(s)" if inv.get("persons_count") else ""
        items = ", ".join(inv.get("identified_items", [])[:2])
        entities_summary = ", ".join(filter(None, [humans, items])) or "General Content"

        # Setting
        setting = f"{inv.get('location_context', '')} {inv.get('setting_type', '')}".strip() or "Standard Scene"

        # Anomaly localization
        if modality == "image":
            anomaly = f"{loc.get('suspicious_image_area_pct', 0.0):.1f}%"
        elif modality == "video":
            anomaly = f"{loc.get('suspicious_video_duration_pct', 0.0):.1f}%"
        else:
            anomaly = f"{loc.get('suspicious_audio_duration_pct', 0.0):.1f}%"

        c2pa_status = "Present" if prov.get("c2pa_present") else "Absent"

        table_data.append({
            "Filename": fname,
            "Category": tax_label,
            "Status": status_badge,
            "Verdict": status,
            "P(AI)": f"{p_ai:.1f}%",
            "P(Real)": f"{p_real:.1f}%",
            "Attributed Model": attr.get("attributed_model", "Unattributable"),
            "Scene & Setting": setting,
            "Content / Entities": entities_summary,
            "Suspicious Area / Duration": anomaly,
            "C2PA": c2pa_status,
        })

    df = pd.DataFrame(table_data)

    st.markdown("##### 📋 Batch Comparison Table")
    st.dataframe(df, width="stretch", hide_index=True)

    # Batch Export Actions
    col_exp1, col_exp2, _ = st.columns([1, 1, 2])
    with col_exp1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Batch CSV Summary",
            data=csv_data,
            file_name=f"batch_{modality}_summary.csv",
            mime="text/csv",
            key=f"csv_dl_{modality}",
        )

    with col_exp2:
        # Full NIST JSON export
        export_dossier = []
        for r in batch_results:
            if r.get("success"):
                export_dossier.append({
                    "filename": r.get("filename"),
                    "final_decision": r.get("decision"),
                    "media_specifications": r.get("img_profile") or r.get("vid_profile") or r.get("aud_profile"),
                })
        json_data = json.dumps(export_dossier, indent=2, ensure_ascii=False, default=str)
        st.download_button(
            "📥 Download Full Batch JSON Dossier",
            data=json_data,
            file_name=f"batch_{modality}_dossier.json",
            mime="application/json",
            key=f"json_dl_{modality}",
        )


def render_batch_file_selector(
    batch_results: List[Dict[str, Any]],
    modality: str = "image",
    key: str = "batch_file_selector",
) -> Optional[Dict[str, Any]]:
    """Renders a selector allowing the user to drill into individual files in the batch."""
    if not batch_results:
        return None

    options = []
    for idx, r in enumerate(batch_results):
        fname = r.get("filename", f"Item #{idx+1}")
        if r.get("success"):
            dec = r.get("decision", {})
            st_text = dec.get("final_status", "")
            p_ai = dec.get("authenticity_probabilities", {}).get("p_ai", 0.0)
            tag = "🚨 AI" if any(w in st_text for w in ("SYNTHETIC", "AI-GENERATED", "AI")) else ("✅ Real" if any(w in st_text for w in ("AUTHENTIC", "REAL")) else "❓ Undet")
            options.append(f"[{tag} {p_ai:.0f}%] {fname}")
        else:
            options.append(f"[❌ Error] {fname}")

    st.markdown("---")
    st.markdown("### 🔬 Inspect Individual Media Forensic Dossier")
    st.caption("Select any file from the batch below to review its spatial heatmap, scene intelligence, and continual learning ratings.")

    selected_label = st.selectbox(
        "Choose media item to inspect:",
        options=options,
        index=0,
        key=key,
    )

    if selected_label:
        selected_idx = options.index(selected_label)
        return batch_results[selected_idx]

    return batch_results[0]
