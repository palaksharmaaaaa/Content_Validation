"""ui.results.dimensions: The nine-dimension forensic breakdown."""
from __future__ import annotations
from typing import Any, Dict
import streamlit as st
from image_detector.explain import build_nine_dimensions_dossier


_DIMENSION_TABS = (
    ("1. Provenance", "dimension_1"),
    ("2. Pixel Specs", "dimension_2"),
    ("3. PRNU Noise", "dimension_3"),
    ("4. Smoothness", "dimension_4"),
    ("5. Fourier FFT", "dimension_5"),
    ("6. Genre & Scene", "dimension_6"),
    ("7. Visual Medium", "dimension_7"),
    ("8. Sensor Spectrum", "dimension_8"),
    ("9. AI Attribution", "dimension_9"),
)


def _render_dimension_1(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 1')}")
    if d.get("container_atoms"):
        st.write(f"• **Container Atoms:** `{', '.join(str(a) for a in d.get('container_atoms', []))}`")
    if d.get("container_format"):
        st.write(f"• **Audio Stream Format:** `{d.get('container_format')}`")
    if d.get("camera_hardware") or d.get("hardware_origin"):
        cam_name = d.get("camera_hardware") or d.get("hardware_origin")
        lens = f" | Lens: `{d.get('lens_model')}`" if d.get("lens_model") else ""
        st.write(f"• **Hardware Identity:** `{cam_name}`{lens}")
    if d.get("iptc_digital_source_type"):
        st.write(f"• **IPTC Digital Source Type:** `{d.get('iptc_digital_source_type')}`")
    st.write(f"• **C2PA Content Credentials:** `{d.get('c2pa_status')}`")
    if d.get("software_signature"):
        st.write(f"• **Software Fingerprint:** `{d.get('software_signature')}` | Date: `{d.get('date_taken')}`")
    if d.get("gps_coordinates"):
        st.write(f"• **GPS Location:** `{d.get('gps_coordinates')}`")
    st.info(f"Verdict: **{d.get('provenance_verdict')}**")


def _render_dimension_2(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 2')}")
    if d.get("geometry"):
        st.write(f"• **Geometry:** `{d.get('geometry')}` ({d.get('aspect_ratio')}, {d.get('orientation', 'Standard')})")
    if d.get("frame_rate_fps"):
        st.write(f"• **Frame Rate & Duration:** `{d.get('frame_rate_fps')}` • `{d.get('duration')}` ({d.get('total_frames')} frames)")
    if d.get("sample_rate"):
        st.write(f"• **Acoustic Geometry:** `{d.get('sample_rate')}` • `{d.get('duration')}` • `{d.get('channels')}` • `{d.get('bit_depth')}`")
    if d.get("dpi"):
        st.write(f"• **DPI & Bit Depth:** `{d.get('dpi')}` • `{d.get('bit_depth')}` ({d.get('color_space')})")
    if d.get("shannon_entropy"):
        st.write(f"• **Shannon Entropy:** `{d.get('shannon_entropy')}`")
    if d.get("luminance_dynamic_range"):
        st.write(f"• **Luminance Range:** `{d.get('luminance_dynamic_range')}`")
    if d.get("clipping_profile"):
        st.write(f"• **Clipping Profile:** `{d.get('clipping_profile')}`")
    if d.get("unique_colors_quantized"):
        st.write(f"• **Unique Quantized Colors:** `{d.get('unique_colors_quantized', 0):,}`")


def _render_dimension_3(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 3')}")
    if d.get("temporal_warping_variance") is not None:
        st.write(f"• **Inter-Frame Motion Variance:** `{d.get('temporal_warping_variance', 0.0):.2f}`")
    if d.get("vocoder_cutoff_hz") is not None:
        st.write(f"• **Vocoder Brickwall Cutoff:** `{d.get('vocoder_cutoff_hz', 0):,} Hz`")
    if d.get("prnu_residual_mean") is not None:
        st.write(f"• **PRNU Residual Mean:** `{d.get('prnu_residual_mean', 0.0):.3f}` | Flat Region Noise: `{d.get('flat_region_noise', 0.0):.3f}`")
    if d.get("mathematical_physics"):
        st.write(f"• **Mathematical Formulation:** `{d.get('mathematical_physics')}`")
    if d.get("diagnosis"):
        if d.get("is_natural_shot_noise") or d.get("temporal_warping_variance", 0.0) < 140.0:
            st.success(f"Diagnosis: {d.get('diagnosis')}")
        else:
            st.error(f"Diagnosis: {d.get('diagnosis')}")


def _render_dimension_4(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 4')}")
    if d.get("diffusion_flickering_ratio") is not None:
        st.write(f"• **Diffusion Flickering Ratio:** `{d.get('diffusion_flickering_ratio', 0.0):.3f}`")
    if d.get("spectral_flatness_wiener") is not None:
        st.write(f"• **Wiener Spectral Flatness:** `{d.get('spectral_flatness_wiener', 0.0):.4f}`")
    if d.get("smoothness_index") is not None:
        st.write(f"• **Surface Smoothness Index:** `{d.get('smoothness_index', 0.0):.3f}`")
    st.write("• **Physical Principle:** Organic micro-textures maintain high local bilateral variance; neural generative models over-smooth.")
    if d.get("diagnosis"):
        if d.get("is_diffusion_smoothed") or d.get("is_flickering_detected") or d.get("is_synthetic_smoothness"):
            st.warning(f"⚠ {d.get('diagnosis')}")
        else:
            st.success(f"✅ {d.get('diagnosis')}")


def _render_dimension_5(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 5')}")
    if d.get("digital_silence_ratio") is not None:
        st.write(f"• **Digital Zero Silence Ratio:** `{d.get('digital_silence_ratio')}`")
    if d.get("keyframe_fft_alpha") is not None:
        st.write(f"• **Keyframe Spectral Alpha:** `{d.get('keyframe_fft_alpha', 2.05):.3f}`")
    if d.get("spectral_decay_alpha") is not None:
        st.write(f"• **Spectral Decay Alpha:** `{d.get('spectral_decay_alpha', 2.05):.3f}`")
    if d.get("mathematical_physics"):
        st.write(f"• **Field Law:** `{d.get('mathematical_physics')}`")
    if d.get("diagnosis"):
        if d.get("is_anomalous_decay") or d.get("has_digital_zero_silence"):
            st.warning(f"⚠ {d.get('diagnosis')}")
        else:
            st.success(f"✅ {d.get('diagnosis')}")


def _render_dimension_6(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 6')}")
    if d.get("audio_type"):
        st.write(f"• **Dominant Audio Type:** `{d.get('audio_type')}` • `{d.get('estimated_speakers')} speaker(s)`")
        st.write(f"• **Acoustic Environment:** `{d.get('acoustic_environment')}`")
    else:
        st.write(f"• **Primary Visual Genre:** `{d.get('primary_genre')}`")
        st.write(f"• **Living Entities:** `{d.get('persons_count', 0)}` person(s), `{d.get('faces_count', 0)}` face(s) (Stylized Character: `{d.get('is_stylized_character')}`)")
        st.write(f"• **Setting & Atmosphere:** `{d.get('setting_and_environment')}` • `{d.get('atmospheric_mood')}`")
        st.write(f"• **Daytime & Lighting Quality:** `{d.get('lighting_and_daytime')}`")


def _render_dimension_7(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 7')}")
    if d.get("acoustic_delivery_tone"):
        st.write(f"• **Acoustic Delivery Tone:** `{d.get('acoustic_delivery_tone')}`")
    if d.get("visual_medium"):
        st.write(f"• **Visual Medium:** `{d.get('visual_medium')}`")
    if d.get("synthesis_type"):
        st.write(f"• **Temporal Synthesis Type:** `{d.get('synthesis_type')}`")
    if d.get("is_digital_art") is not None:
        st.write(f"• **Digital Art / Painting Detected:** `{d.get('is_digital_art')}`")
    if d.get("canny_edge_density_pct") is not None:
        st.write(f"• **Canny Edge Density:** `{d.get('canny_edge_density_pct', 0.0):.2f}%` | Dark Line Art: `{d.get('dark_line_contours_pct', 0.0):.2f}%`")
    if d.get("laplacian_focus_sharpness") is not None:
        st.write(f"• **Focus Sharpness:** `{d.get('laplacian_focus_sharpness', 0.0):.1f}`")


def _render_dimension_8(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 8')}")
    if d.get("bandwidth_class"):
        st.write(f"• **Acoustic Bandwidth:** `{d.get('bandwidth_class')}` | Nyquist Ceiling: `{d.get('nyquist_ceiling')}`")
    if d.get("sensor_spectrum") or d.get("sensor_modality"):
        st.write(f"• **Acquisition Modality:** `{d.get('sensor_spectrum') or d.get('sensor_modality')}`")
    if d.get("color_channels") is not None:
        st.write(f"• **Color Channels:** `{d.get('color_channels', 3)}` channels")
    if d.get("diagnosis"):
        st.write(f"• **Diagnostic:** {d.get('diagnosis')}")


def _render_dimension_9(d: Dict[str, Any]) -> None:
    st.markdown(f"#### {d.get('title', 'Dimension 9')}")
    reg = f" ({d.get('region_of_origin')})" if d.get("region_of_origin") else ""
    st.write(f"• **Attributed Generator:** `{d.get('attributed_model')}`{reg}")
    st.write(f"• **Attribution Confidence:** `{d.get('confidence', '0%')}`")
    st.write(f"• **Watermark Detected:** `{'YES' if d.get('watermark_detected') else 'None'}`")
    if d.get("suspicious_duration_pct") is not None:
        st.write(f"• **Suspicious Duration Fraction:** `{d.get('suspicious_duration_pct')}`")
    if d.get("spatial_manipulated_area_pct") is not None:
        st.write(f"• **Spatial Manipulation Area:** `{d.get('spatial_manipulated_area_pct', 0.0):.1f}%` of frame")


_DIMENSION_RENDERERS = (_render_dimension_1, _render_dimension_2, _render_dimension_3, _render_dimension_4, _render_dimension_5, _render_dimension_6, _render_dimension_7, _render_dimension_8, _render_dimension_9)


def render_nine_dimensions_breakdown(nine_dims: Dict[str, Any], expanded: bool = True) -> None:
    """
    Renders 9-Dimensions Forensic Taxonomy Analyzer.
    Exhaustively covers all 9 analytical dimensions formalized in GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md.
    """
    if not nine_dims:
        return

    st.markdown("#### Nine-dimension breakdown")
    st.caption("Evaluation across all 9 forensic dimensions (referencing NIST OpenMFC, C2PA and IPTC concepts; informational, not a certified compliance check):")

    with st.expander("Show all nine dimensions", expanded=expanded):
        tabs = st.tabs([title for title, _ in _DIMENSION_TABS])
        for i, (tab, (_title, key)) in enumerate(zip(tabs, _DIMENSION_TABS), start=1):
            with tab:
                _DIMENSION_RENDERERS[i - 1](nine_dims.get(key, {}))


def image_nine_dimensions(item: Dict[str, Any], profile_data: Dict[str, Any], ai_result: Dict[str, Any], content_res: Dict[str, Any]) -> Any:
    """The item's precomputed nine-dimension dossier, or one built on demand from its parts."""
    nine_dims = item.get("nine_dimensions_dossier")
    if not nine_dims and profile_data and ai_result:
        nine_dims = build_nine_dimensions_dossier(
            profile_data=profile_data,
            ai_result=ai_result,
            content_inventory=content_res,
            provenance_result=item.get("provenance_res"),
            attribution_result=item.get("attribution_res"),
        )
    return nine_dims
