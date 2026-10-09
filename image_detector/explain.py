"""
image_detector.explain: Plain-English Newbie Explanation and 9-Dimensions Forensic Dossier Generator.
Transforms complex forensic telemetry, pixel statistics, and taxonomy states into
clear, engaging, accessible narrative explanations for beginners and comprehensive audits for professionals.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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
}


@dataclass
class _DossierContext:
    """Normalised views of the profile / detector / inventory dicts shared by the nine dimension builders."""

    profile_data: Dict[str, Any]
    ai_result: Dict[str, Any]
    inv: Dict[str, Any]
    prov: Dict[str, Any]
    attr: Dict[str, Any]
    geom: Dict[str, Any]
    disp: Dict[str, Any]
    pcol: Dict[str, Any]
    exif: Dict[str, Any]
    phys: Dict[str, Any]
    f_metrics: Dict[str, Any]
    entities: Dict[str, Any]
    humans: Dict[str, Any]
    env: Dict[str, Any]
    light: Dict[str, Any]
    tone: Dict[str, Any]
    purpose: Dict[str, Any]
    tax_state: str

    @classmethod
    def build(cls, profile_data, ai_result, content_inventory, provenance_result, attribution_result) -> "_DossierContext":
        inv = content_inventory or {}
        entities = inv.get("living_entities") or inv.get("entities") or {}
        return cls(
            profile_data=profile_data,
            ai_result=ai_result,
            inv=inv,
            prov=provenance_result or {},
            attr=attribution_result or ai_result.get("model_attribution") or {},
            geom=profile_data.get("spatial_geometry") or {},
            disp=profile_data.get("display_attributes") or {},
            pcol=profile_data.get("pixel_color_profile") or {},
            exif=profile_data.get("exif_device_details") or {},
            phys=profile_data.get("raw_physical_signals") or {},
            f_metrics=ai_result.get("forensic_metrics", {}),
            entities=entities,
            humans=entities.get("humans", {}),
            env=inv.get("environment_and_surroundings") or inv.get("environment") or {},
            light=inv.get("lighting_and_daytime") or {},
            tone=inv.get("tone_and_mood") or {},
            purpose=inv.get("purpose_and_depiction") or {},
            tax_state=ai_result.get("taxonomy_state") or "UNDECIDED",
        )


def _join_known(*parts: Any) -> str:
    """The non-empty parts joined with a bullet, or "Not determined" when none is known."""
    known = [str(p) for p in parts if p]
    return " • ".join(dict.fromkeys(known)) if known else "Not determined"


def _dimension_1(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 1: Acquisition Hardware & Provenance Spectrum"""
    has_cam = bool(c.exif.get("camera_make") or c.prov.get("camera_make"))
    cam_str = f"{c.exif.get('camera_make') or c.prov.get('camera_make') or ''} {c.exif.get('camera_model') or c.prov.get('camera_model') or ''}".strip() or "No camera make or model recorded"
    iptc_type = IPTC_SOURCE_TYPE_MAPPING.get(c.tax_state, "Not determined")

    d1 = {
        "dimension_id": 1,
        "title": "Dimension 1: Acquisition Hardware & Provenance Identity Spectrum",
        "description": "Examines physical camera sensor origin, hardware EXIF authenticity, C2PA manifest markers (presence only), and IPTC Digital Source Type mapping.",
        "camera_hardware": cam_str,
        "lens_model": c.exif.get("lens_model") or "N/A",
        "exposure_parameters": {
            "shutter_speed": c.exif.get("exposure_time") or "N/A",
            "aperture": c.exif.get("aperture") or "N/A",
            "iso": c.exif.get("iso") or "N/A",
            "focal_length": c.exif.get("focal_length") or "N/A",
            "flash": c.exif.get("flash") or "N/A",
            "white_balance": c.exif.get("white_balance") or "N/A",
        },
        "date_taken": c.exif.get("date_time") or c.prov.get("date_time") or "Not recorded",
        "gps_coordinates": (c.exif.get("gps_details") or {}).get("coordinates_str") or "Not recorded",
        "c2pa_status": (
            "Present (markers only; not cryptographically verified)"
            if c.prov.get("c2pa_present")
            else "Absent (Neutral)"
        ),
        "software_signature": c.exif.get("software") or c.prov.get("software") or "Not recorded",
        "iptc_digital_source_type": iptc_type,
        "provenance_verdict": "Camera hardware EXIF present (unauthenticated)" if has_cam else "No camera make or model recorded",
    }
    return d1


def _dimension_2(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 2: Photometric, Spatial Geometry & Colorimetric Architecture"""
    w = c.geom.get("width") or c.profile_data.get("width") or 0
    h = c.geom.get("height") or c.profile_data.get("height") or 0
    pc = c.pcol
    entropy = pc.get("shannon_entropy_bpp", c.profile_data.get("pixel_entropy"))
    aspect = c.geom.get("aspect_ratio_str") or (f"{c.geom['aspect_ratio']}:1" if c.geom.get("aspect_ratio") else "Not recorded")
    d2 = {
        "dimension_id": 2,
        "title": "Dimension 2: Photometric, Spatial Geometry & Colorimetric Architecture",
        "description": "Inspects pixel geometry, resolution aspect ratio, DPI resolution, bit depth, Shannon entropy, dynamic range, and dominant color palette.",
        "geometry": f"{w} x {h} px ({w * h / 1e6:.2f} MP, {w * h:,} total pixels)" if w and h else "Not recorded",
        "aspect_ratio": aspect,
        "orientation": c.geom.get("orientation", "Not recorded"),
        "dpi": c.disp.get("dpi_str", "Not recorded"),
        "bit_depth": c.disp.get("bit_depth") or "Not recorded",
        "color_space": c.disp.get("color_space", "Not recorded"),
        "shannon_entropy": f"{entropy:.3f} bits/px" if entropy is not None else "Not recorded",
        "luminance_dynamic_range": (
            f"{pc['luminance_mean']:.1f} mean (span: {pc.get('luminance_min')}..{pc.get('luminance_max')}, median: {pc.get('luminance_median')})"
            if pc.get("luminance_mean") is not None else "Not recorded"
        ),
        "clipping_profile": (
            f"Highlights: {pc['highlight_clipped_pct']}% ({pc.get('highlight_clipped_count', 0):,} px) | "
            f"Shadows: {pc.get('shadow_crushed_pct')}% ({pc.get('shadow_crushed_count', 0):,} px)"
            if pc.get("highlight_clipped_pct") is not None else "Not recorded"
        ),
        "dominant_palette": pc.get("dominant_palette", []),
        "unique_colors_quantized": pc.get("unique_quantized_colors"),
    }
    return d2


def _dimension_3(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 3: Sensor-Noise Residual"""
    prnu_val = c.phys.get("flat_region_noise_mean", c.f_metrics.get("noise_residual_mean"))
    d3 = {
        "dimension_id": 3,
        "title": "Dimension 3: Sensor-Noise Residual",
        "description": "Measures the fine grain left after a 3x3 median filter is subtracted from the grey image. Camera photographs usually keep visible grain; heavily denoised, rendered or generated pictures often do not. This is a stand-in for sensor noise, not a PRNU fingerprint.",
        "noise_residual_mean": prnu_val,
        "flat_region_noise": c.phys.get("flat_region_noise_mean", prnu_val),
        "is_natural_shot_noise": None if prnu_val is None else prnu_val >= 1.20,
        "mathematical_physics": "residual = mean(|I - median3x3(I)|) over the image; flat-region residual is the same statistic where the local gradient is small.",
        "diagnosis": (
            "Not measured"
            if prnu_val is None
            else f"Fine camera-like grain is present (score: {prnu_val:.2f})"
            if prnu_val >= 1.20
            else f"Little fine grain; the picture looks smoothed or denoised (score: {prnu_val:.2f})"
        ),
    }
    return d3


def _dimension_4(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 4: Bilateral Surface Smoothness & Spatial Texture Variance"""
    smooth_val = c.phys.get("surface_smoothness_index", c.f_metrics.get("surface_smoothness"))
    d4 = {
        "dimension_id": 4,
        "title": "Dimension 4: Bilateral Surface Smoothness & Spatial Texture Variance",
        "description": "Measures micro-texture continuity against bilateral filter smoothing to expose plastic diffusion skin and synthetic surfaces.",
        "smoothness_index": smooth_val,
        "is_diffusion_smoothed": None if smooth_val is None else smooth_val < 3.2,
        "diagnosis": (
            "Not measured"
            if smooth_val is None
            else f"Surfaces are very smooth (index: {smooth_val:.2f}), as in generated or heavily retouched pictures"
            if smooth_val < 3.2
            else f"Surface texture is not unusually smooth (index: {smooth_val:.2f})"
        ),
    }
    return d4


def _dimension_5(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 5: 2D Fourier FFT Frequency Power Spectrum Decay"""
    fft_alpha = c.phys.get("fft_decay_alpha", c.f_metrics.get("spectral_decay_alpha"))
    unmeasured = fft_alpha is None
    d5 = {
        "dimension_id": 5,
        "title": "Dimension 5: 2D Fourier FFT Frequency Power Spectrum Decay",
        "description": "Verifies natural optical power-law decay P(f) proportional to f^(-alpha), where natural optical captures exhibit alpha in [1.8, 2.2].",
        "spectral_decay_alpha": fft_alpha,
        "is_anomalous_decay": False if unmeasured else (fft_alpha < 1.65 or fft_alpha > 3.45),
        "mathematical_physics": "P(f) ~ f^(-alpha). Natural optical: alpha in [1.8, 2.2]. Synthetic: alpha < 1.4 or alpha > 3.4.",
        "diagnosis": (
            "Not measured (the picture is too small to fit a spectral slope)"
            if unmeasured
            else f"Anomalous Fourier spectral slope (alpha={float(fft_alpha):.2f}); outside the range of ordinary photographs"
            if (fft_alpha < 1.65 or fft_alpha > 3.45)
            else f"Standard Fourier radial spectral decay slope (alpha={float(fft_alpha):.2f}) within the range of ordinary photographs"
        ),
    }
    return d5


def _dimension_6(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 6: Visual Genre & Semantic Subject-Matter Catalog"""
    genre = c.purpose.get("primary_genre") or c.ai_result.get("subject_genre") or "Not determined"
    d6 = {
        "dimension_id": 6,
        "title": "Dimension 6: Visual Genre & Semantic Subject-Matter Catalog",
        "description": "Categorizes visual scene into human portraiture, nature, cosmic, architectural, commercial, or documentary genres.",
        "primary_genre": genre,
        "persons_count": c.humans.get("persons_count", c.inv.get("persons_count", 0)),
        "faces_count": c.humans.get("faces_count", c.inv.get("faces_count", 0)),
        "face_bounding_boxes": c.humans.get("bounding_boxes", []),
        "is_stylized_character": c.humans.get("is_stylized_character", False),
        "animals_count": c.entities.get("animals", {}).get("count", 0),
        "animal_types": c.entities.get("animals", {}).get("animal_types", []),
        "vehicles_count": c.inv.get("vehicles", {}).get("count", 0),
        "vehicle_types": c.inv.get("vehicles", {}).get("vehicle_types", []),
        "setting_and_environment": _join_known(c.env.get("setting_type"), c.env.get("setting")),
        "lighting_and_daytime": _join_known(c.light.get("daytime"), c.light.get("lighting_quality")),
        "atmospheric_mood": c.tone.get("atmospheric_mood") or "Not determined",
        "identified_items": c.inv.get("contents_and_items", {}).get("identified_items", []),
    }
    return d6


def _dimension_7(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 7: Visual Art Mediums & Creative Styles Catalog"""
    medium = c.ai_result.get("visual_medium") or "Not determined"
    d7 = {
        "dimension_id": 7,
        "title": "Dimension 7: Visual Art Mediums & Creative Styles Catalog",
        "description": "Distinguishes photographic optical capture from traditional painting (oil, watercolor), vector art, anime/manga, pixel art, or 3D CGI.",
        "visual_medium": medium,
        "is_digital_art": c.ai_result.get("digital_art_detected", False),
        "dark_line_contours_pct": c.phys.get("dark_line_art_pct"),
        "canny_edge_density_pct": c.phys.get("canny_edge_pct"),
        "laplacian_focus_sharpness": c.phys.get("laplacian_sharpness_var"),
        "diagnosis": f"Visual style categorized as {medium}",
    }
    return d7


def _dimension_8(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 8: Electromagnetic Spectrum & Acquisition Modalities"""
    spectrum = c.ai_result.get("sensor_spectrum") or "Not determined"
    d8 = {
        "dimension_id": 8,
        "title": "Dimension 8: Electromagnetic Spectrum & Acquisition Modalities",
        "description": "Identifies imaging wavelength: visible colour RGB (400-700nm), Monochrome, Infrared (NIR/LWIR), UV, Biomedical (X-Ray/SEM), or Satellite/SAR.",
        "sensor_spectrum": spectrum,
        "color_channels": c.profile_data.get("channels"),
        "diagnosis": f"Acquisition modality operates in {spectrum}",
    }
    return d8


def _dimension_9(c: _DossierContext) -> Dict[str, Any]:
    """Dimension 9: Generative AI Frontier & Attribution Fingerprint"""
    attributed_model = c.attr.get("attributed_model") or "Unknown / Unattributable"
    d9 = {
        "dimension_id": 9,
        "title": "Dimension 9: Generative AI Frontier & Attribution Fingerprint",
        "description": "Detects foundation model synthesis (Flux.1, Midjourney, SD 3.5, Gemini/Imagen 3, DALL-E 3), neural inpainting, enhancement software and watermarks.",
        "attributed_model": attributed_model,
        "region_of_origin": c.attr.get("region_of_origin") or "Not determined",
        "attribution_confidence": c.attr.get("attribution_confidence", c.attr.get("confidence", 0.0)),
        "watermark_detected": c.ai_result.get("watermark_detected", False),
        "watermark_details": c.ai_result.get("watermark_details") or "No Gemini-style sparkle watermark found (the only visible watermark this tool looks for)",
        "top_candidates": c.attr.get("top_candidates", []),
        "spatial_manipulated_area_pct": c.ai_result.get("ai_spatial_area_pct", 0.0),
    }
    return d9


def build_nine_dimensions_dossier(
    profile_data: Dict[str, Any],
    ai_result: Dict[str, Any],
    content_inventory: Dict[str, Any],
    provenance_result: Optional[Dict[str, Any]] = None,
    attribution_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Constructs an exhaustive 9-dimensional forensic analysis dossier.
    """
    c = _DossierContext.build(profile_data, ai_result, content_inventory, provenance_result, attribution_result)
    return {
        "dimension_1": _dimension_1(c),
        "dimension_2": _dimension_2(c),
        "dimension_3": _dimension_3(c),
        "dimension_4": _dimension_4(c),
        "dimension_5": _dimension_5(c),
        "dimension_6": _dimension_6(c),
        "dimension_7": _dimension_7(c),
        "dimension_8": _dimension_8(c),
        "dimension_9": _dimension_9(c),
    }


def _depiction_phrase(persons: int, faces: int, is_char: bool, animals: List[str], vehicles: List[str], genre: str) -> str:
    """Plain-English description of the main subject."""
    if is_char or (persons == 1 and faces == 0):
        return "a stylized character figure or artistic avatar"
    if persons == 1:
        face_suffix = f" (with {faces} visible face)" if faces > 0 else ""
        return f"an individual person{face_suffix} framed in a portrait composition"
    if persons > 1:
        face_suffix = f" ({faces} visible face{'s' if faces != 1 else ''})" if faces > 0 else ""
        return f"a social gathering or group scene depicting approximately {persons} persons{face_suffix}"
    if animals:
        return f"a wildlife / domestic scene featuring {len(animals)} animal(s) ({', '.join(sorted(set(animals)))})"
    if vehicles:
        return f"a vehicular or transit scene showing {len(vehicles)} vehicle(s) ({', '.join(sorted(set(vehicles)))})"
    return f"a visual scene composed as a {genre.lower()}" if genre else "a visual scene"


def _extra_entities_phrase(persons: int, animals: List[str], vehicles: List[str]) -> str:
    extra = []
    if persons > 0 and animals:
        extra.append(f"{len(animals)} animal(s) ({', '.join(sorted(set(animals)))})")
    if vehicles and (persons > 0 or animals):
        extra.append(f"{len(vehicles)} vehicle(s) ({', '.join(sorted(set(vehicles)))})")
    return f", alongside {', and '.join(extra)}" if extra else ""


def _palette_phrase(profile_data: Dict[str, Any]) -> str:
    palette = profile_data.get("pixel_color_profile", {}).get("dominant_palette", [])
    if not palette:
        return ""
    top_name = palette[0].get("color_name") or palette[0].get("hex", "#000000")
    top_hex = palette[0].get("hex", "#000000")
    top_pct = palette[0].get("coverage_pct", 0.0)
    return f", with the palette predominantly shaped by {top_name} tones ({top_hex} covering {top_pct}% of the image)"


def _screenshot_takeaway(noise: float) -> str:
    return (
        f"**The Simple Takeaway:** This image is a **digital screen capture** (a screenshot taken on a smartphone, tablet, or computer monitor), "
        f"not a photograph taken by pointing a physical camera into the real world.\n\n"
        f"**Think of it like this:** When you take a picture of a room with your phone camera, light travels through a curved glass lens and hits a silicon sensor, "
        f"creating subtle shadows, natural softness, and tiny camera grain. But when you take a screenshot, your device's graphics card simply copies the exact grid "
        f"of pixels currently displayed on the screen — like taking a digital photocopy. Because of this, it has razor-sharp 90-degree UI edges, zero optical camera grain "
        f"(noise score: **{noise:.2f}**), and standard display proportions."
    )


def _undetermined_takeaway(p_ai: float, p_real: float) -> str:
    return (
        f"**The Simple Takeaway:** The evidence is **mixed** ({p_ai:.0f}% AI-like, {p_real:.0f}% camera-like), so this tool does not give a verdict.\n\n"
        "Some signals look like a camera and some are weaker than usual. The most common cause is not AI at all: messaging apps, social media and "
        "screenshots recompress photos, which removes the fine sensor grain this tool relies on. Treat this as *unknown* and look at the Evidence tab."
    )


def _synthetic_takeaway(p_ai: float, tax_label: str, noise: float, smooth: float) -> str:
    grain = (
        f"The fine camera-like grain is low here (noise residual **{noise:.2f}**; photographs from cameras are usually above about 1.2) "
        f"and the surface smoothness index is **{smooth:.2f}**, which fits a picture that was generated or heavily smoothed."
        if noise < 1.20 else
        f"The fine grain is not unusually low (noise residual **{noise:.2f}**, surface smoothness index **{smooth:.2f}**), so the lean towards AI "
        f"comes from other signals; see the Evidence tab for which."
    )
    return (
        f"**The Simple Takeaway:** Our heuristic (uncalibrated) score is **{p_ai:.1f}% AI-likelihood**, so this image leans towards "
        f"**generative AI or heavy synthetic processing** ({tax_label}) rather than a plain camera capture. That is a ranking aid, not proof.\n\n"
        f"**What that is based on:** A camera sensor leaves a fine grain in every photo (*sensor shot noise*); many generators produce smoother, grain-free surfaces. {grain} "
        f"Compression, upscaling and beauty filters can also remove grain, so check the Evidence tab before relying on this."
    )


def _edited_takeaway(tax_label: str, p_real: float) -> str:
    return (
        f"**The Simple Takeaway:** This looks like a **photograph that was edited or designed with graphic software** "
        f"({tax_label}); the heuristic (uncalibrated) camera-likelihood is **{p_real:.1f}%**, a ranking aid, not proof.\n\n"
        f"**Think of it like this:** Imagine taking a real physical photograph of someone with a regular camera, and then bringing that photo into an app like "
        f"Photoshop or Canva. You might cut out the background, place the person on a clean studio color backdrop, adjust the lighting, or add graphic text. "
        f"That is what the signals suggest here: the subject keeps camera-like grain, while the background edges and composition "
        f"show conventional editing rather than AI generation. These checks cannot rule out generative edits."
    )


def _authentic_takeaway(tax_label: str, p_real: float, noise: float) -> str:
    return (
        f"**The Simple Takeaway:** This image is **consistent with a plain camera photograph** ({tax_label}); "
        f"the heuristic (uncalibrated) estimate is **{p_real:.1f}%**, which is a ranking aid, not proof.\n\n"
        f"**What that is based on:** A camera sensor leaves a fine grain in every photo, and this picture has it (noise residual: **{noise:.2f}**) "
        f"without the extra smoothness or other signs of generation that this tool looks for. That does not rule out a good fake: "
        f"recent generators, careful retouching and re-photographed screens can pass these checks."
    )


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

    geom = profile_data.get("spatial_geometry") or {}
    w = geom.get("width", profile_data.get("width", 0))
    h = geom.get("height", profile_data.get("height", 0))
    aspect_str = geom.get("aspect_ratio_str")
    dpi_str = profile_data.get("display_attributes", {}).get("dpi_str")

    content = content_inventory or {}
    entities = content.get("living_entities") or content.get("entities") or {}
    humans = entities.get("humans", {})
    persons = humans.get("persons_count", content.get("persons_count", 0))
    faces = humans.get("faces_count", content.get("faces_count", 0))
    genre = content.get("purpose_and_depiction", {}).get("primary_genre") or ai_result.get("subject_genre") or ""
    medium = ai_result.get("visual_medium") or ""
    mood = content.get("tone_and_mood", {}).get("atmospheric_mood") or ""
    daytime = content.get("lighting_and_daytime", {}).get("daytime") or ""

    metrics = ai_result.get("forensic_metrics", {})
    phys = profile_data.get("raw_physical_signals") or {}
    noise = phys.get("flat_region_noise_mean", metrics.get("noise_residual_mean", 0.0))
    smooth = phys.get("surface_smoothness_index", metrics.get("surface_smoothness", 0.0))

    animals = entities.get("animals", {}).get("animal_types", [])
    vehicles = content.get("vehicles", {}).get("vehicle_types", [])
    depiction = _depiction_phrase(persons, faces, humans.get("is_stylized_character", False), animals, vehicles, genre)
    extra_str = _extra_entities_phrase(persons, animals, vehicles)
    items_list = content.get("contents_and_items", {}).get("identified_items", [])
    items_desc = f", with visible elements including {', '.join(items_list[:3])}" if items_list else ""

    format_phrase = ""
    if aspect_str:
        format_phrase = f" formatted in **{aspect_str}**" + (f" at **{dpi_str}**" if dpi_str and dpi_str != "Not recorded" else "")
    setting_phrase = ""
    if daytime or mood:
        setting_phrase = " set in" + (f" **{daytime.lower()}** conditions" if daytime else "") + (" with" if daytime and mood else "") + (f" a **{mood.lower()}** atmosphere" if mood else "")
    section_what = (
        f"### 🖼️ What We Identified in this Image\n\n"
        f"This file (`{filename}`) is a **{w} × {h} pixel** visual asset{format_phrase}. "
        f"Visually, it depicts **{depiction}**{extra_str}{items_desc}{setting_phrase}{_palette_phrase(profile_data)}."
        + (f" The visual style is characterized as **{medium}**." if medium else "")
    )

    is_synthetic = "AI" in verdict or "SYNTHETIC" in verdict or p_ai >= 55.0
    is_edited = "EDITED" in verdict or "GRAPHIC" in verdict
    is_screenshot = "SCREENSHOT" in verdict or "SCREEN" in verdict
    if is_screenshot:
        body = _screenshot_takeaway(noise)
    elif verdict == "UNDETERMINED":
        body = _undetermined_takeaway(p_ai, p_real)
    elif is_synthetic:
        body = _synthetic_takeaway(p_ai, tax_label, noise, smooth)
    elif is_edited:
        body = _edited_takeaway(tax_label, p_real)
    else:
        body = _authentic_takeaway(tax_label, p_real, noise)
    return section_what + "\n\n### 💡 How Would You Explain This to a Newbie?\n\n" + body
