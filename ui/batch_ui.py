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

import cv2
import pandas as pd
import streamlit as st

from audio_detector import (
    AudioAIDetector,
    AudioValidator,
    build_audio_nine_dimensions_dossier,
    generate_audio_newbie_explanation,
    generate_spectrogram_image,
)
from image_detector import (
    ImageAIDetector,
    ImageContentAnalyzer as ContentAnalyzer,
    ImageModelAttributionEngine as ModelAttributionEngine,
    analyze_image,
    build_nine_dimensions_dossier,
    generate_newbie_explanation,
)
from video_detector import (
    VideoAIDetector,
    analyze_video,
    build_video_nine_dimensions_dossier,
    evaluate_cross_modal_consistency,
    generate_video_newbie_explanation,
)
from ui.feedback_ui import generate_final_decision, profile_media
from ui.validators import analyze_provenance, validate_file


def process_single_image(
    img_path: str,
    filename: str,
    detector: ImageAIDetector,
    content_analyzer: ContentAnalyzer,
    attribution_engine: ModelAttributionEngine,
    sensitivity: str = "high",
    source: str = "User Upload",
) -> Dict[str, Any]:
    """Runs end-to-end NIST-aligned forensic pipeline on a single image."""
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

    # 1. Stage 1: Pre-Analysis Feature & Metadata Extraction FIRST
    img_profile = profile_media(img_path, modality="image", source=source)

    # 2. Provenance & Cryptographic C2PA Verification
    provenance_res = analyze_provenance(img_path)

    # 3. Deep Learning & Statistical Sensor Noise AI Detection
    ai_result = detector.predict(img_path, sensitivity=sensitivity)

    # 4. Scene & Content Intelligence (Living entities, objects, text regions)
    content_res = content_analyzer.analyze_image_content(img_path)

    # 5. Foundation Model Attribution
    attribution_res = attribution_engine.attribute_media(
        img_path,
        modality="image",
        forensic_data=ai_result,
        profile_data=img_profile,
        provenance_data=provenance_res,
    )

    image_result = analyze_image(img_path)

    # 6. Multi-evidence Bayesian Decision & Taxonomy Assignment
    decision = generate_final_decision(
        file_validation=file_res,
        quality_result=image_result,
        ai_result=ai_result,
        content_inventory=content_res,
        provenance_result=provenance_res,
        attribution_result=attribution_res,
    )

    # 7. Stage 2: 9-Dimensional NIST Forensic Dossier
    nine_dims = build_nine_dimensions_dossier(
        profile_data=img_profile,
        ai_result=ai_result,
        content_inventory=content_res,
        provenance_result=provenance_res,
        attribution_result=attribution_res,
    )

    # 8. Stage 5: Plain-English Newbie Narrative Explanation
    newbie_expl = generate_newbie_explanation(
        filename=filename,
        profile_data=img_profile,
        content_inventory=content_res,
        ai_result=ai_result,
        decision=decision,
    )

    return {
        "filename": filename,
        "path": img_path,
        "source": source,
        "success": True,
        "file_res": file_res,
        "image_result": image_result,
        "ai_result": ai_result,
        "content_res": content_res,
        "provenance_res": provenance_res,
        "attribution_res": attribution_res,
        "decision": decision,
        "img_profile": img_profile,
        "nine_dimensions_dossier": nine_dims,
        "newbie_explanation": newbie_expl,
    }


def process_single_video(
    vid_path: str,
    filename: str,
    detector: ImageAIDetector,
    content_analyzer: ContentAnalyzer,
    audio_detector: AudioAIDetector,
    attribution_engine: ModelAttributionEngine,
    sensitivity: str = "high",
    cache_dir: Optional[Path] = None,
    video_detector: Optional[VideoAIDetector] = None,
) -> Dict[str, Any]:
    """Runs end-to-end NIST-aligned forensic pipeline on a single video."""
    file_res = validate_file(vid_path)
    if not file_res.get("readable"):
        return {
            "filename": filename,
            "path": vid_path,
            "success": False,
            "error": file_res.get("error", "Video corrupted or unreadable"),
            "file_res": file_res,
        }

    provenance_res = analyze_provenance(vid_path)
    if video_detector is not None:
        video_result = video_detector.analyze_video(vid_path, sensitivity=sensitivity)
    else:
        video_result = analyze_video(
            vid_path,
            sample_count=24,
            ai_detector=detector,
            sensitivity=sensitivity,
        )
    audio_result = audio_detector.analyze_audio_file(vid_path, sensitivity=sensitivity)

    # Sample representative keyframe (~15% timeline) for scene content inventory & spatial anomaly
    cap = cv2.VideoCapture(vid_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_kf_idx = max(0, int(total_frames * 0.15)) if total_frames > 5 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_kf_idx)
    ret, sample_frame = cap.read()
    if (not ret or sample_frame is None) and target_kf_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, sample_frame = cap.read()
    cap.release()

    content_res = {}
    tmp_kf_path = None
    kf_ai = None

    if ret and sample_frame is not None:
        target_dir = cache_dir if cache_dir else Path(vid_path).parent
        stem = Path(filename).stem
        tmp_kf = target_dir / f"kf_{stem}.jpg"
        cv2.imwrite(str(tmp_kf), sample_frame)
        tmp_kf_path = str(tmp_kf)
        content_res = content_analyzer.analyze_image_content(tmp_kf_path)
        kf_ai = detector.predict(tmp_kf_path, sensitivity=sensitivity)

    cross_modal_res = evaluate_cross_modal_consistency(video_result, audio_result, content_res)
    vid_profile = profile_media(vid_path, modality="video")
    attribution_res = attribution_engine.attribute_media(
        vid_path,
        modality="video",
        forensic_data=video_result.get("ai_video_rating", {}),
        profile_data=vid_profile,
        provenance_data=provenance_res,
    )
    decision = generate_final_decision(
        file_validation=file_res,
        quality_result=video_result,
        ai_result=kf_ai,
        audio_result=audio_result,
        content_inventory=content_res,
        provenance_result=provenance_res,
        cross_modal_result=cross_modal_res,
        attribution_result=attribution_res,
    )

    nine_dims = build_video_nine_dimensions_dossier(
        profile_data=vid_profile,
        video_result=video_result,
        content_inventory=content_res,
        provenance_result=provenance_res,
        attribution_result=attribution_res,
        cross_modal_result=cross_modal_res,
    )
    newbie_expl = generate_video_newbie_explanation(
        filename=filename,
        profile_data=vid_profile,
        content_inventory=content_res,
        video_result=video_result,
        decision=decision,
    )

    return {
        "filename": filename,
        "path": vid_path,
        "success": True,
        "file_res": file_res,
        "video_result": video_result,
        "audio_result": audio_result,
        "content_res": content_res,
        "provenance_res": provenance_res,
        "attribution_res": attribution_res,
        "cross_modal_res": cross_modal_res,
        "decision": decision,
        "vid_profile": vid_profile,
        "tmp_kf_path": tmp_kf_path,
        "kf_ai": kf_ai,
        "nine_dimensions_dossier": nine_dims,
        "newbie_explanation": newbie_expl,
    }


def process_single_audio(
    aud_path: str,
    filename: str,
    audio_detector: AudioAIDetector,
    content_analyzer: Any,
    attribution_engine: Any,
    sensitivity: str = "high",
) -> Dict[str, Any]:
    """Runs end-to-end NIST-aligned forensic pipeline on a single audio file."""
    file_res = validate_file(aud_path)
    if not file_res.get("readable"):
        return {
            "filename": filename,
            "path": aud_path,
            "success": False,
            "error": file_res.get("error", "Audio corrupted or unreadable"),
            "file_res": file_res,
        }

    samples, sr, duration = AudioValidator().extract_pcm_samples(aud_path)
    provenance_res = analyze_provenance(aud_path)
    content_res = content_analyzer.analyze_audio_content(samples, sr, duration)
    audio_result = audio_detector.analyze_audio_file(
        aud_path,
        sensitivity=sensitivity,
        pre_extracted=(samples, sr, duration),
    )
    aud_profile = profile_media(aud_path, modality="audio")
    attribution_res = attribution_engine.attribute_media(
        aud_path,
        modality="audio",
        forensic_data=audio_result,
        profile_data=aud_profile,
        provenance_data=provenance_res,
    )
    decision = generate_final_decision(
        file_validation=file_res,
        quality_result={},
        audio_result=audio_result,
        content_inventory=content_res,
        provenance_result=provenance_res,
        attribution_result=attribution_res,
    )

    spec_img = None
    if samples is not None and len(samples) > 0:
        spec_img = generate_spectrogram_image(samples, sr)

    nine_dims = build_audio_nine_dimensions_dossier(
        profile_data=aud_profile,
        audio_result=audio_result,
        content_inventory=content_res,
        provenance_result=provenance_res,
        attribution_result=attribution_res,
    )
    newbie_expl = generate_audio_newbie_explanation(
        filename=filename,
        profile_data=aud_profile,
        content_inventory=content_res,
        audio_result=audio_result,
        decision=decision,
    )

    return {
        "filename": filename,
        "path": aud_path,
        "success": True,
        "file_res": file_res,
        "sr": sr,
        "duration": duration,
        "provenance_res": provenance_res,
        "content_res": content_res,
        "audio_result": audio_result,
        "aud_profile": aud_profile,
        "attribution_res": attribution_res,
        "decision": decision,
        "spec_img": spec_img,
        "nine_dimensions_dossier": nine_dims,
        "newbie_explanation": newbie_expl,
    }


def run_batch_pipeline(
    items: List[Dict[str, str]],
    modality: str,
    detectors: Dict[str, Any],
    sensitivity: str = "high",
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
        status = decision.get("final_status", "UNDETERMINED_OOD")
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
