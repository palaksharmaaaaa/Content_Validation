"""ui.results.profile: File profile sections (identity, geometry, EXIF, photometrics, palette, physical signals) and stream/signal specs."""
from __future__ import annotations
from typing import Any, Dict, NamedTuple
import streamlit as st


class _ProfileView(NamedTuple):
    """Lookups derived once from the profile dict and shared by the section renderers."""

    profile_data: Any
    geom: Any
    disp: Any
    pcol: Any
    exif: Any
    phys: Any
    file_id: Any
    fname: Any
    fmt: Any
    size_kb: Any
    size_mb: Any
    sha: Any
    src: Any


def _render_profile_section_1(v: _ProfileView) -> None:
    """1. Media & Container Origin"""
    st.markdown("**File**")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("File Name", v.fname[:20] + "..." if len(v.fname) > 23 else v.fname)
    f2.metric("File Size", f"{v.size_kb:.1f} KB ({v.size_mb:.2f} MB)")
    f3.metric("Format / MIME", f"{v.fmt} • {v.profile_data.get('mime_type', 'image/' + str(v.fmt).lower())}")
    f4.metric("Source Origin", v.src)
    f5.metric("SHA-256 (Prefix)", f"{v.sha[:10]}..." if v.sha else "N/A")

    st.markdown("---")


def _render_profile_section_2(v: _ProfileView) -> None:
    """2. Dimensions, Spatial Geometry & DPI"""
    st.markdown("**Size and resolution**")
    w = v.geom.get("width", v.profile_data.get("width", 0))
    h = v.geom.get("height", v.profile_data.get("height", 0))
    tot_pix = v.geom.get("total_pixels", int(w * h))
    mp = v.geom.get("megapixels", round(tot_pix / 1_000_000.0, 2))
    asp_str = v.geom.get("aspect_ratio_str", f"{v.geom.get('aspect_ratio', 0.0)}:1")
    orient = v.geom.get("orientation", "Landscape")

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Width x Height", f"{w} × {h} px")
    g2.metric("Megapixels (MP)", f"{mp:.2f} MP")
    g3.metric("Total Pixel Count", f"{tot_pix:,} px")
    g4.metric("Aspect Ratio", asp_str)
    g5.metric("Orientation", orient)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("DPI Resolution", v.disp.get("dpi_str", "72 x 72 DPI"))
    d2.metric("Bit Depth", v.disp.get("bit_depth", "24-bit (3x8-bit)"))
    d3.metric("Color Space", v.disp.get("color_space", "Standard sRGB"))
    d4.metric("Alpha Channel", "Present (RGBA)" if v.disp.get("has_alpha_channel") else "None (RGB)")

    st.markdown("---")


def _render_profile_section_3(v: _ProfileView) -> None:
    """3. Device & Acquisition EXIF Parameters"""
    st.markdown("**Camera and EXIF**")
    has_cam = bool(v.exif.get("camera_make") or v.exif.get("camera_model"))
    cam_make = v.exif.get("camera_make") or "Unspecified Hardware"
    cam_model = v.exif.get("camera_model") or ""
    lens = v.exif.get("lens_model") or "Standard Lens / Unspecified"
    shutter = v.exif.get("exposure_time") or "N/A"
    aperture = v.exif.get("aperture") or "N/A"
    iso_val = v.exif.get("iso") or "N/A"
    focal = v.exif.get("focal_length") or "N/A"
    flash = v.exif.get("flash", "Not Fired")
    wb = v.exif.get("white_balance", "Auto")
    gps_str = v.exif.get("gps_details", {}).get("coordinates_str", "Not Embedded")
    software = v.exif.get("software") or "None (Clean Exif)"
    date_str = v.exif.get("date_time") or "Unknown / Stripped"

    if has_cam:
        st.success(f"Camera hardware EXIF tags present (unauthenticated): **{cam_make} {cam_model}** | Lens: `{lens}`")
    else:
        st.info("ℹ No embedded hardware camera EXIF tags found (characteristic of stripped web/social uploads or AI synthesis).")

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Shutter Speed", shutter)
    e2.metric("Aperture", aperture)
    e3.metric("ISO Sensitivity", str(iso_val))
    e4.metric("Focal Length", focal)
    e5.metric("Flash / White Balance", f"{flash} • {wb}")

    st.caption(f"• **Capture Date:** `{date_str}` | **Software Tag:** `{software}` | **GPS Coordinates:** `{gps_str}`")

    st.markdown("---")


def _render_profile_section_4(v: _ProfileView) -> None:
    """4. Pixel-by-Pixel Color & Photometric Statistics"""
    st.markdown("**Brightness and colour statistics**")
    entropy = v.pcol.get("shannon_entropy_bpp", v.profile_data.get("pixel_entropy", 0.0))
    lum_mean = v.pcol.get("luminance_mean", 0.0)
    lum_median = v.pcol.get("luminance_median", lum_mean)
    lum_range = v.pcol.get("dynamic_range", 0)
    lum_min = v.pcol.get("luminance_min", 0)
    lum_max = v.pcol.get("luminance_max", 255)
    clip_hi = v.pcol.get("highlight_clipped_pct", 0.0)
    clip_sh = v.pcol.get("shadow_crushed_pct", 0.0)

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Shannon Entropy", f"{entropy:.3f} bits/px")
    p2.metric("Mean / Median Lum", f"{lum_mean:.1f} / {lum_median:.1f}")
    p3.metric("Dynamic Range", f"{lum_range} ({lum_min}..{lum_max})")
    p4.metric("Clipped Highlights", f"{clip_hi:.2f}%")
    p5.metric("Crushed Shadows", f"{clip_sh:.2f}%")

    c_means = v.pcol.get("channel_means", {})
    c_stds = v.pcol.get("channel_stds", {})
    if "red" in c_means:
        st.caption(
            f"• **Channel Distributions (Mean ± Std):** "
            f"Red: `{c_means.get('red', 0.0):.1f} ± {c_stds.get('red', 0.0):.1f}` | "
            f"Green: `{c_means.get('green', 0.0):.1f} ± {c_stds.get('green', 0.0):.1f}` | "
            f"Blue: `{c_means.get('blue', 0.0):.1f} ± {c_stds.get('blue', 0.0):.1f}` | "
            f"Unique Quantized Colors: `{v.pcol.get('unique_quantized_colors', 0):,}`"
        )


def _render_profile_section_5(v: _ProfileView) -> None:
    """5. Dominant Color Palette Swatches"""
    st.markdown("**Dominant colours**")
    palette = v.pcol.get("dominant_palette", [])
    if palette:
        palette_html = '<div style="display: flex; flex-wrap: wrap; gap: 14px; margin-top: 6px; margin-bottom: 12px;">'
        for c in palette:
            palette_html += (
                f'<div style="background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 8px; border: 1px solid #444; text-align: center; min-width: 100px;">'
                f'<div style="width: 100%; height: 32px; background-color: {c["hex"]}; border-radius: 4px; border: 1px solid #222; margin-bottom: 6px;"></div>'
                f'<div style="font-size: 12px; font-weight: bold;">{c.get("color_name", c["hex"])}</div>'
                f'<div style="font-size: 11px; color: #aaa;"><code>{c["hex"]}</code> • <b>{c["coverage_pct"]}%</b></div>'
                f'</div>'
            )
        palette_html += "</div>"
        st.markdown(palette_html, unsafe_allow_html=True)
    else:
        st.write("• Color palette extracted.")

    st.markdown("---")


def _render_profile_section_6(v: _ProfileView) -> None:
    """6. Raw Physical Noise & Forensic Signals"""
    st.markdown("**Noise and frequency properties**")
    prnu_noise = v.phys.get("prnu_noise_mean", 0.0)
    flat_noise = v.phys.get("flat_region_noise_mean", prnu_noise)
    smoothness = v.phys.get("surface_smoothness_index", 0.0)
    fft_alpha = v.phys.get("fft_decay_alpha", 2.05)
    edges_pct = v.phys.get("canny_edge_pct", 0.0)
    dark_lines = v.phys.get("dark_line_art_pct", 0.0)
    sharpness = v.phys.get("laplacian_sharpness_var", 0.0)

    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("PRNU Sensor Noise", f"{prnu_noise:.3f}")
    n2.metric("Flat-Region Noise", f"{flat_noise:.3f}")
    n3.metric("Surface Smoothness", f"{smoothness:.3f}")
    n4.metric("Fourier Decay Alpha", f"{fft_alpha:.2f}")
    n5.metric("Focus / Sharpness Var", f"{sharpness:.1f}")

    st.caption(
        f"• **Edge Density:** `{edges_pct:.2f}%` Canny edges | "
        f"• **Contour Density:** `{dark_lines:.2f}%` dark line-art strokes | "
        f"• **Optical Shot Noise Baseline:** `{'Preserved (>= 1.20)' if flat_noise >= 1.20 else 'Synthetic / Denoiser Absence (< 1.20)'}`"
    )


_PROFILE_SECTIONS = (_render_profile_section_1, _render_profile_section_2, _render_profile_section_3, _render_profile_section_4, _render_profile_section_5, _render_profile_section_6)


def render_pre_analysis_specifications(
    profile_data: Dict[str, Any], expanded: bool = True, source: str = "User Upload"
) -> None:
    """
    Renders Pre-Analysis Feature & Metadata Extraction.
    Shows dimensions, DPI, pixel-by-pixel information, EXIF details, noise, and colors
    before running prediction/detection.
    """
    if not profile_data:
        return

    st.markdown("#### File profile")
    st.caption(
        "Instant extraction of image dimensions, DPI, pixel-by-pixel statistics, EXIF hardware parameters, "
        "color palette, and raw physical noise prior to running AI detection models."
    )

    geom = profile_data.get("spatial_geometry") or profile_data.get("pixel_specifications") or {}
    disp = profile_data.get("display_attributes") or {}
    pcol = profile_data.get("pixel_color_profile") or {}
    exif = profile_data.get("exif_device_details") or profile_data.get("provenance_metadata") or {}
    phys = profile_data.get("raw_physical_signals") or {}
    file_id = profile_data.get("file_identity") or {}

    fname = profile_data.get("filename") or file_id.get("filename", "Uploaded File")
    fmt = profile_data.get("format") or file_id.get("container_format", "N/A")
    size_kb = profile_data.get("file_size_kb") or file_id.get("size_kb", 0.0)
    size_mb = profile_data.get("file_size_mb", size_kb / 1024.0)
    sha = profile_data.get("sha256") or file_id.get("sha256", "")
    src = profile_data.get("source", source)

    v = _ProfileView(profile_data=profile_data, geom=geom, disp=disp, pcol=pcol, exif=exif, phys=phys, file_id=file_id, fname=fname, fmt=fmt, size_kb=size_kb, size_mb=size_mb, sha=sha, src=src)

    with st.expander("Show full image profile", expanded=expanded):
        for render_section in _PROFILE_SECTIONS:
            render_section(v)


def render_video_stream_specs(profile_data: Dict[str, Any]) -> None:
    st.markdown("#### Stream profile")
    st.caption("Low-level container headers, stream geometry, frame rates, codecs, and cryptographic hashes extracted before running detection.")
    geom = profile_data.get("geometry", {})
    codec = profile_data.get("codec_and_container", {})
    hashes = profile_data.get("cryptographic_hashes", {})

    c1, c2, c3, c4 = st.columns(4)
    w_val = geom.get("width", 0)
    h_val = geom.get("height", 0)
    c1.metric("Dimensions", f"{w_val} × {h_val} px" if w_val else "N/A")
    c2.metric("Frame Rate", f"{geom.get('fps', 0.0):.1f} fps")
    c3.metric("Duration", f"{geom.get('duration_seconds', 0.0):.2f} s")
    c4.metric("Total Frames", f"{geom.get('total_frames', 0):,}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Container / Codec", f"{codec.get('container', 'MP4')} • {codec.get('codec', 'AVC')}")
    c6.metric("Aspect Ratio", geom.get("aspect_ratio", "N/A"))
    c7.metric("Bitrate", f"{codec.get('bitrate_kbps', 0.0):.0f} kbps" if codec.get('bitrate_kbps') else "N/A")
    sha_str = hashes.get("sha256", "")
    c8.metric("SHA-256", f"{sha_str[:12]}..." if sha_str else "N/A")

    st.markdown("---")


def render_audio_signal_specs(item: Dict[str, Any], profile_data: Dict[str, Any]) -> None:
    st.markdown("#### Signal profile")
    st.caption("Low-level container headers, sampling rates, bit depths, dynamic ranges, and cryptographic hashes extracted before running detection.")
    sr_val = item.get("sr") or profile_data.get("sample_rate", 44100)
    dur_val = item.get("duration") or profile_data.get("duration", 0.0)
    ch_val = profile_data.get("channels", 1)
    fmt_val = profile_data.get("format", "WAV")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampling Rate", f"{sr_val:,} Hz")
    c2.metric("Duration", f"{dur_val:.2f} s")
    c3.metric("Channels", "Stereo (2 Ch)" if ch_val == 2 else "Mono (1 Ch)")
    c4.metric("Format / Container", str(fmt_val).upper())

    hashes = profile_data.get("cryptographic_hashes", {})
    sha_str = hashes.get("sha256", "")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Bit Depth", profile_data.get("bit_depth", "16-bit PCM"))
    c6.metric("RMS Energy", f"{profile_data.get('rms_energy', 0.0):.4f}" if profile_data.get('rms_energy') else "Normal")
    c7.metric("Dynamic Range", f"{profile_data.get('dynamic_range_db', 0.0):.1f} dB" if profile_data.get('dynamic_range_db') else "Standard")
    c8.metric("SHA-256", f"{sha_str[:12]}..." if sha_str else "N/A")

    st.markdown("---")
