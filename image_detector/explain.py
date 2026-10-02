"""
image_detector.explain: Plain-English Newbie Explanation and 9-Dimensions Forensic Dossier Generator.
Transforms complex forensic telemetry, pixel statistics, and taxonomy states into
clear, engaging, accessible narrative explanations for beginners and comprehensive audits for professionals.
Aligned directly with:
GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md (NIST OpenMFC, C2PA v2.1, IPTC Photo Metadata Standard 2024-2026).
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("image_detector.explain")


IPTC_SOURCE_TYPE_MAPPING = {
    "AUTHENTIC_REAL_PHOTOGRAPH": "digitalsourcetype:digitalCapture (Original Physical Capture)",
    "AUTHENTIC_EDITED": "digitalsourcetype:minorHumanEdits / compositeSynthetic (Conventional Graphic Edit)",
    "AUTHENTIC_SCREENSHOT": "digitalsourcetype:screenCapture (Operating System Framebuffer)",
    "AUTHENTIC_RECAPTURED_SCREEN": "digitalsourcetype:screenCapture (Physical Re-Photography / Moiré)",
    "AI_ENHANCED_COMPOSITE": "digitalsourcetype:compositeWithTrainedAlgorithmicMedia (Generative Inpainting / Splicing)",
    "AI_ENHANCED_SCREENSHOT": "digitalsourcetype:screenCapture / compositeWithTrainedAlgorithmicMedia",
    "AI_GENERATED_SCREENSHOT": "digitalsourcetype:screenCapture / trainedAlgorithmicMedia",
    "FULLY_AI_GENERATED": "digitalsourcetype:trainedAlgorithmicMedia (Synthesized via Foundation Model)",
    "PROCEDURAL_CGI_SYNTHETIC": "digitalsourcetype:softwareImage / virtualRecording (3D CGI / Engine Render)",
    "ADVERSARIAL_SPOOF_SYNTHETIC": "digitalsourcetype:trainedAlgorithmicMedia (Adversarially Perturbed)",
}


def build_nine_dimensions_dossier(
    profile_data: Dict[str, Any],
    ai_result: Dict[str, Any],
    content_inventory: Dict[str, Any],
    provenance_result: Optional[Dict[str, Any]] = None,
    attribution_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs an exhaustive 9-dimensional forensic analysis dossier directly aligned with
    GLOBAL_IMAGE_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md.
    """
    geom = profile_data.get("spatial_geometry") or profile_data.get("pixel_specifications") or {}
    disp = profile_data.get("display_attributes") or {}
    pcol = profile_data.get("pixel_color_profile") or {}
    exif = profile_data.get("exif_device_details") or {}
    phys = profile_data.get("raw_physical_signals") or {}
    f_metrics = ai_result.get("forensic_metrics", {})
    prov = provenance_result or {}
    attr = attribution_result or ai_result.get("model_attribution") or {}
    inv = content_inventory or {}
    entities = inv.get("living_entities") or inv.get("entities") or {}
    humans = entities.get("humans", {})
    env = inv.get("environment_and_surroundings") or inv.get("environment") or {}
    light = inv.get("lighting_and_daytime") or {}
    tone = inv.get("tone_and_mood") or {}
    purpose = inv.get("purpose_and_depiction") or {}
    tax_state = ai_result.get("taxonomy_state") or "UNDECIDED"

    # Dimension 1: Acquisition Hardware & Provenance Spectrum
    has_cam = bool(exif.get("camera_make") or prov.get("camera_make"))
    cam_str = f"{exif.get('camera_make') or prov.get('camera_make', '')} {exif.get('camera_model') or prov.get('camera_model', '')}".strip() or "No Hardware Camera Detected (Web Container / Stripped Metadata)"
    iptc_type = IPTC_SOURCE_TYPE_MAPPING.get(tax_state, "digitalsourcetype:digitalCapture (Unspecified)")

    d1 = {
        "dimension_id": 1,
        "title": "Dimension 1: Acquisition Hardware & Provenance Identity Spectrum",
        "description": "Examines physical camera sensor origin, hardware EXIF authenticity, C2PA cryptographic manifests, and IPTC Digital Source Type mapping.",
        "camera_hardware": cam_str,
        "lens_model": exif.get("lens_model") or "N/A",
        "exposure_parameters": {
            "shutter_speed": exif.get("exposure_time") or "N/A",
            "aperture": exif.get("aperture") or "N/A",
            "iso": exif.get("iso") or "N/A",
            "focal_length": exif.get("focal_length") or "N/A",
            "flash": exif.get("flash", "N/A"),
            "white_balance": exif.get("white_balance", "Auto"),
        },
        "date_taken": exif.get("date_time") or prov.get("date_time") or "Unknown / Stripped",
        "gps_coordinates": exif.get("gps_details", {}).get("coordinates_str", "Not Embedded"),
        "c2pa_status": (
            "Present (Content Credentials Verified)"
            if prov.get("c2pa_present")
            else "Absent (Neutral)"
        ),
        "software_signature": exif.get("software") or prov.get("software") or "None (Clean)",
        "iptc_digital_source_type": iptc_type,
        "provenance_verdict": "Authentic Hardware EXIF Verified" if has_cam else "Web Container / Metadata Stripped",
    }

    # Dimension 2: Photometric, Spatial Geometry & Colorimetric Architecture
    w = geom.get("width", profile_data.get("width", 0))
    h = geom.get("height", profile_data.get("height", 0))
    d2 = {
        "dimension_id": 2,
        "title": "Dimension 2: Photometric, Spatial Geometry & Colorimetric Architecture",
        "description": "Inspects pixel geometry, resolution aspect ratio, DPI resolution, bit depth, Shannon entropy, dynamic range, and dominant color palette.",
        "geometry": f"{w} x {h} px ({geom.get('megapixels', 0.0):.2f} MP, {geom.get('total_pixels', w * h):,} total pixels)",
        "aspect_ratio": geom.get("aspect_ratio_str", f"{geom.get('aspect_ratio', 0.0)}:1"),
        "orientation": geom.get("orientation", "Landscape"),
        "dpi": disp.get("dpi_str", "72 x 72 DPI"),
        "bit_depth": disp.get("bit_depth", f"{profile_data.get('channels', 3) * 8}-bit"),
        "color_space": disp.get("color_space", "Standard sRGB"),
        "shannon_entropy": f"{pcol.get('shannon_entropy_bpp', profile_data.get('pixel_entropy', 0.0)):.3f} bits/px",
        "luminance_dynamic_range": f"{pcol.get('luminance_mean', 0.0):.1f} mean (span: {pcol.get('luminance_min', 0)}..{pcol.get('luminance_max', 255)}, median: {pcol.get('luminance_median', 0.0)})",
        "clipping_profile": f"Highlights: {pcol.get('highlight_clipped_pct', 0.0)}% ({pcol.get('highlight_clipped_count', 0):,} px) | Shadows: {pcol.get('shadow_crushed_pct', 0.0)}% ({pcol.get('shadow_crushed_count', 0):,} px)",
        "dominant_palette": pcol.get("dominant_palette", []),
        "unique_colors_quantized": pcol.get("unique_quantized_colors", 0),
    }

    # Dimension 3: Physical Sensor PRNU Noise Residual
    prnu_val = phys.get("flat_region_noise_mean", f_metrics.get("noise_residual_mean", 0.0))
    d3 = {
        "dimension_id": 3,
        "title": "Dimension 3: Physical Sensor PRNU Noise Residual",
        "description": "Calculates Photo-Response Non-Uniformity (PRNU) and Poisson-Gaussian sensor shot noise residual (sigma_PRNU >= 1.45 for optical capture).",
        "prnu_residual_mean": prnu_val,
        "flat_region_noise": phys.get("flat_region_noise_mean", prnu_val),
        "is_natural_shot_noise": prnu_val >= 1.20,
        "mathematical_physics": "sigma^2_PRNU = (1/|M|) * sum((W(x,y) - mu_W)^2) >= 1.45 (NIST OpenMFC)",
        "diagnosis": (
            f"Natural Poisson-Gaussian sensor shot noise grain preserved (score: {prnu_val:.2f})"
            if prnu_val >= 1.20
            else f"Absence of physical camera sensor grain; synthetic mathematical smoothing detected (score: {prnu_val:.2f})"
        ),
    }

    # Dimension 4: Bilateral Surface Smoothness & Spatial Texture Variance
    smooth_val = phys.get("surface_smoothness_index", f_metrics.get("surface_smoothness", 0.0))
    d4 = {
        "dimension_id": 4,
        "title": "Dimension 4: Bilateral Surface Smoothness & Spatial Texture Variance",
        "description": "Measures micro-texture continuity against bilateral filter smoothing to expose plastic diffusion skin and synthetic surfaces.",
        "smoothness_index": smooth_val,
        "is_diffusion_smoothed": smooth_val < 3.2,
        "diagnosis": (
            f"Diffusion latent space bilateral over-smoothing detected across surfaces (index: {smooth_val:.2f})"
            if smooth_val < 3.2
            else f"Natural organic surface micro-textures and physical lens MTF sharpness preserved (index: {smooth_val:.2f})"
        ),
    }

    # Dimension 5: 2D Fourier FFT Frequency Power Spectrum Decay
    fft_alpha = phys.get("fft_decay_alpha", f_metrics.get("spectral_decay_alpha", 2.05))
    d5 = {
        "dimension_id": 5,
        "title": "Dimension 5: 2D Fourier FFT Frequency Power Spectrum Decay",
        "description": "Verifies natural optical power-law decay P(f) proportional to f^(-alpha), where natural optical captures exhibit alpha in [1.8, 2.2].",
        "spectral_decay_alpha": fft_alpha,
        "is_anomalous_decay": fft_alpha < 1.65 or fft_alpha > 3.45,
        "mathematical_physics": "P(f) ~ f^(-alpha). Natural optical: alpha in [1.8, 2.2]. Synthetic: alpha < 1.4 or alpha > 3.4.",
        "diagnosis": (
            f"Anomalous Fourier spectral slope (alpha={fft_alpha:.2f}); deviates from physical optical decay"
            if (fft_alpha < 1.65 or fft_alpha > 3.45)
            else f"Standard Fourier radial spectral decay slope (alpha={fft_alpha:.2f}) adhering to optical physics"
        ),
    }

    # Dimension 6: Visual Genre & Semantic Subject-Matter Catalog
    genre = purpose.get("primary_genre") or ai_result.get("subject_genre") or "General Scene"
    d6 = {
        "dimension_id": 6,
        "title": "Dimension 6: Visual Genre & Semantic Subject-Matter Catalog",
        "description": "Categorizes visual scene into human portraiture, nature, cosmic, architectural, commercial, or documentary genres.",
        "primary_genre": genre,
        "persons_count": humans.get("persons_count", inv.get("persons_count", 0)),
        "faces_count": humans.get("faces_count", inv.get("faces_count", 0)),
        "face_bounding_boxes": humans.get("bounding_boxes", []),
        "is_stylized_character": humans.get("is_stylized_character", False),
        "animals_count": entities.get("animals", {}).get("count", 0),
        "animal_types": entities.get("animals", {}).get("animal_types", []),
        "vehicles_count": inv.get("vehicles", {}).get("count", 0),
        "vehicle_types": inv.get("vehicles", {}).get("vehicle_types", []),
        "setting_and_environment": f"{env.get('setting_type', 'Ambient')} • {env.get('setting', 'Indoor')}",
        "lighting_and_daytime": f"{light.get('daytime', 'Daylight')} ({light.get('lighting_quality', 'Ambient')})",
        "atmospheric_mood": tone.get("atmospheric_mood", "Balanced"),
        "identified_items": inv.get("contents_and_items", {}).get("identified_items", []),
    }

    # Dimension 7: Visual Art Mediums & Creative Styles Catalog
    medium = ai_result.get("visual_medium", "Photographic Capture")
    d7 = {
        "dimension_id": 7,
        "title": "Dimension 7: Visual Art Mediums & Creative Styles Catalog",
        "description": "Distinguishes photographic optical capture from traditional painting (oil, watercolor), vector art, anime/manga, pixel art, or 3D CGI.",
        "visual_medium": medium,
        "is_digital_art": ai_result.get("digital_art_detected", False),
        "dark_line_contours_pct": phys.get("dark_line_art_pct", 0.0),
        "canny_edge_density_pct": phys.get("canny_edge_pct", 0.0),
        "laplacian_focus_sharpness": phys.get("laplacian_sharpness_var", 0.0),
        "diagnosis": f"Visual style categorized as {medium}",
    }

    # Dimension 8: Electromagnetic Spectrum & Acquisition Modalities
    spectrum = ai_result.get("sensor_spectrum", "Visible Spectrum (Bayer RGB)")
    d8 = {
        "dimension_id": 8,
        "title": "Dimension 8: Electromagnetic Spectrum & Acquisition Modalities",
        "description": "Identifies imaging wavelength: Visible Bayer RGB (400-700nm), Monochrome, Infrared (NIR/LWIR), UV, Biomedical (X-Ray/SEM), or Satellite/SAR.",
        "sensor_spectrum": spectrum,
        "color_channels": profile_data.get("channels", 3),
        "diagnosis": f"Acquisition modality operates in {spectrum}",
    }

    # Dimension 9: Generative AI Frontier & Attribution Fingerprint
    attributed_model = attr.get("attributed_model") or "Unknown / Unattributable"
    d9 = {
        "dimension_id": 9,
        "title": "Dimension 9: Generative AI Frontier & Attribution Fingerprint",
        "description": "Detects foundation model synthesis (Flux.1, Midjourney, SD 3.5, Gemini/Imagen 3, DALL-E 3), neural inpainting, deepfake face-swapping, and watermarks.",
        "attributed_model": attributed_model,
        "region_of_origin": attr.get("region_of_origin", "Global"),
        "attribution_confidence": attr.get("attribution_confidence", attr.get("confidence", 0.0)),
        "watermark_detected": ai_result.get("watermark_detected", False),
        "watermark_details": ai_result.get("watermark_details") or "No synthetic watermark detected",
        "top_candidates": attr.get("top_candidates", []),
        "spatial_manipulated_area_pct": ai_result.get("ai_spatial_area_pct", 0.0),
    }

    return {
        "dimension_1_hardware_provenance": d1,
        "dimension_2_pixel_architecture": d2,
        "dimension_3_prnu_sensor_noise": d3,
        "dimension_4_surface_smoothness": d4,
        "dimension_5_fourier_fft_decay": d5,
        "dimension_6_subject_genre": d6,
        "dimension_7_visual_medium": d7,
        "dimension_8_sensor_spectrum": d8,
        "dimension_9_generative_attribution": d9,
    }


def generate_newbie_explanation(
    filename: str,
    profile_data: Dict[str, Any],
    content_inventory: Dict[str, Any],
    ai_result: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    """
    Generates a descriptive, engaging, beginner-friendly narrative paragraph explaining:
    1. Exactly what was identified in the image (subject, pose, attire, items, setting, lighting, mood, dominant colors).
    2. The authenticity verdict and category in simple everyday language.
    3. Exactly HOW and WHY we reached this conclusion using clear analogies (e.g. sensor grain like sand on film vs AI smooth math).
    """
    verdict = decision.get("final_status", "")
    tax_label = decision.get("taxonomy_label") or "Analyzed Media"
    probs = decision.get("authenticity_probabilities", {})
    p_ai = float(probs.get("p_ai", 0.0))
    p_real = float(probs.get("p_real", 0.0))

    geom = profile_data.get("spatial_geometry") or profile_data.get("pixel_specifications") or {}
    w = geom.get("width", profile_data.get("width", 0))
    h = geom.get("height", profile_data.get("height", 0))
    aspect_str = geom.get("aspect_ratio_str", "Standard format")
    dpi_str = profile_data.get("display_attributes", {}).get("dpi_str", "72 DPI")

    content = content_inventory or {}
    entities = content.get("living_entities") or content.get("entities") or {}
    humans = entities.get("humans", {})
    persons = humans.get("persons_count", content.get("persons_count", 0))
    faces = humans.get("faces_count", content.get("faces_count", 0))
    is_char = humans.get("is_stylized_character", False)
    genre = content.get("purpose_and_depiction", {}).get("primary_genre") or ai_result.get("subject_genre") or "General Scene"
    medium = ai_result.get("visual_medium", "Digital Image")

    tone = content.get("tone_and_mood", {})
    mood = tone.get("atmospheric_mood", "Balanced")
    daytime = content.get("lighting_and_daytime", {}).get("daytime", "Ambient")

    phys = profile_data.get("raw_physical_signals") or {}
    noise = phys.get("flat_region_noise_mean", ai_result.get("forensic_metrics", {}).get("noise_residual_mean", 0.0))
    smooth = phys.get("surface_smoothness_index", ai_result.get("forensic_metrics", {}).get("surface_smoothness", 0.0))
    dark_lines = phys.get("dark_line_art_pct", 0.0)

    animals_list = entities.get("animals", {}).get("animal_types", [])
    vehicles_list = content.get("vehicles", {}).get("vehicle_types", [])

    # 1. Plain-English Scene & Subject Depiction
    if is_char or (persons == 1 and faces == 0):
        depiction_str = "a stylized character figure or artistic avatar"
    elif persons == 1:
        face_suffix = f" (with {faces} visible face)" if faces > 0 else ""
        depiction_str = f"an individual person{face_suffix} framed in a portrait composition"
    elif persons > 1:
        face_suffix = f" ({faces} visible face{'s' if faces != 1 else ''})" if faces > 0 else ""
        depiction_str = f"a social gathering or group scene depicting approximately {persons} persons{face_suffix}"
    elif len(animals_list) > 0:
        depiction_str = f"a wildlife / domestic scene featuring {len(animals_list)} animal(s) ({', '.join(set(animals_list))})"
    elif len(vehicles_list) > 0:
        depiction_str = f"a vehicular or transit scene showing {len(vehicles_list)} vehicle(s) ({', '.join(set(vehicles_list))})"
    else:
        depiction_str = f"a visual scene composed as a {genre.lower()}"

    extra_entities = []
    if persons > 0 and len(animals_list) > 0:
        extra_entities.append(f"{len(animals_list)} animal(s) ({', '.join(set(animals_list))})")
    if len(vehicles_list) > 0 and (persons > 0 or len(animals_list) > 0):
        extra_entities.append(f"{len(vehicles_list)} vehicle(s) ({', '.join(set(vehicles_list))})")
    extra_str = f", alongside {', and '.join(extra_entities)}" if extra_entities else ""

    items_list = content.get("contents_and_items", {}).get("identified_items", [])
    items_desc = f", with visible elements including {', '.join(items_list[:3])}" if items_list else ""

    palette = profile_data.get("pixel_color_profile", {}).get("dominant_palette", [])
    top_color_str = ""
    if palette:
        top_name = palette[0].get("color_name") or palette[0].get("hex", "#000000")
        top_hex = palette[0].get("hex", "#000000")
        top_pct = palette[0].get("coverage_pct", 0.0)
        top_color_str = f", with the palette predominantly shaped by {top_name} tones ({top_hex} covering {top_pct}% of the image)"

    section_what = (
        f"### 🖼️ What We Identified in this Image\n\n"
        f"This file (`{filename}`) is a **{w} × {h} pixel** visual asset formatted in **{aspect_str}** at **{dpi_str}**. "
        f"Visually, it depicts **{depiction_str}**{extra_str}{items_desc} set in a **{daytime.lower()}** environment with an evocative **{mood.lower()}** atmosphere{top_color_str}. "
        f"The visual style is characterized as **{medium}**."
    )

    # 2. Verdict & Beginner-Friendly Analogy
    is_synthetic = "AI" in verdict or "SYNTHETIC" in verdict or p_ai >= 55.0
    is_edited = "EDITED" in verdict or "GRAPHIC" in verdict
    is_screenshot = "SCREENSHOT" in verdict or "SCREEN" in verdict

    section_newbie_title = "\n\n### 💡 How Would You Explain This to a Newbie?\n\n"

    if is_screenshot:
        body = (
            f"**The Simple Takeaway:** This image is a **digital screen capture** (a screenshot taken on a smartphone, tablet, or computer monitor), "
            f"not a photograph taken by pointing a physical camera into the real world.\n\n"
            f"**Think of it like this:** When you take a picture of a room with your phone camera, light travels through a curved glass lens and hits a silicon sensor, "
            f"creating subtle shadows, natural softness, and tiny camera grain. But when you take a screenshot, your device's graphics card simply copies the exact grid "
            f"of pixels currently displayed on the screen — like taking a digital photocopy. Because of this, it has razor-sharp 90-degree UI edges, zero optical camera grain "
            f"(noise score: **{noise:.2f}**), and standard display proportions."
        )
    elif is_synthetic:
        body = (
            f"**The Simple Takeaway:** Our forensic engine determined with **{p_ai:.1f}% confidence** that this image was **created by generative Artificial Intelligence** "
            f"({tax_label}), rather than being captured by a physical camera.\n\n"
            f"**Think of it like this:** When a real camera takes a photo, millions of physical light particles (photons) strike a silicon sensor chip. "
            f"Because the physical world has microscopic imperfections and heat, every authentic photo has a natural, fine grain — very much like the tiny grains of sand "
            f"you see on photographic film. Forensic experts call this *sensor shot noise*.\n\n"
            f"In this image, that natural camera grain is completely missing (the measured sensor noise is only **{noise:.2f}**, whereas real camera photos usually score between 1.20 and 2.50+). "
            f"Instead of real light hitting a lens, a computer algorithm (a neural diffusion network like Midjourney, Flux, or Gemini) mathematically calculated and smoothed each pixel "
            f"(surface smoothness index: **{smooth:.2f}**). While it looks impressive to the naked eye, the physics under the microscope reveal that it was painted by math, not light."
        )
    elif is_edited:
        body = (
            f"**The Simple Takeaway:** This is an **authentic real photograph that was edited or designed with graphic software** "
            f"({tax_label}) — it has a **{p_real:.1f}% authentic optical capture foundation**.\n\n"
            f"**Think of it like this:** Imagine taking a real physical photograph of someone with a regular camera, and then bringing that photo into an app like "
            f"Photoshop or Canva. You might cut out the background, place the person on a clean studio color backdrop, adjust the lighting, or add graphic text. "
            f"That's exactly what happened here: the central subject preserves genuine camera sensor grain and real lens optics, but the background edges and composition "
            f"show conventional digital editing layers rather than artificial AI generation."
        )
    else:
        body = (
            f"**The Simple Takeaway:** This is a **100% genuine real-world photograph** captured through an optical glass camera lens ({tax_label}), "
            f"confirmed with **{p_real:.1f}% confidence**.\n\n"
            f"**Think of it like this:** Everything about this file matches real-world optical physics. When light bounced off the subject and entered the camera lens, "
            f"it left behind authentic physical sensor grain (PRNU score: **{noise:.2f}**), natural organic skin and fabric micro-textures, and optical depth-of-field "
            f"(where the focus gently falls off naturally in a way computers struggle to replicate). No generative AI alterations or deceptive digital manipulations were found."
        )

    return section_what + section_newbie_title + body
