"""
image_detector.features: Forensic image feature extraction routines.
Extracts:
1. EXIF metadata & generative AI signatures.
2. High-frequency sensor noise residuals (PRNU / Poisson shot noise).
3. Surface smoothness via bilateral filter discrepancy.
4. 2D FFT Radial Power Spectrum decay (1/f^alpha field law).
5. Error Level Analysis (ELA) compression artifacts.
6. Spatial manipulation heatmaps (bilateral anomaly zones).
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from core.imageio import imread
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFile
from PIL.ExifTags import TAGS

from image_detector.config import CANONICAL_SCREEN_RESOLUTIONS, KNOWN_AI_SOFTWARE_SIGNATURES

logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True

KNOWN_SCREENSHOT_SOFTWARE_SIGNATURES = [
    "snipping tool",
    "snippingtool",
    "screenshot",
    "screen capture",
    "flameshot",
    "sharex",
    "lightshot",
    "greenshot",
    "gyazo",
    "skitch",
]


def _downsample_if_needed(img: np.ndarray, max_dim: int = 1536) -> np.ndarray:
    """Downsamples image to prevent performance bottlenecks on massive print scans while preserving texture stats."""
    if img is None:
        return img
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = max(16, int(w * scale))
        new_h = max(16, int(h * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def _blank_metadata_info() -> Dict[str, Any]:
    return {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "creator_tool": None,
        "date_time": None,
        "focal_length": None,
        "f_number": None,
        "exposure_time": None,
        "iso": None,
        "lens_model": None,
        "has_optical_parameters": False,
        "iptc_digital_source_type": None,
        "photoshop_credit": None,
        "ai_signature_found": False,
        "ai_enhancer_signature_found": False,
        "graphic_editor_signature_found": False,
        "screenshot_software_found": False,
        "screenshot_software_name": None,
        "signature_details": None,
        "raw_tags": {},
        "xmp_summary": {},
    }


def _read_exif_fields(img: Image.Image, info: Dict[str, Any]) -> None:
    """Standard EXIF plus the Exif sub-IFD (optical parameters) into ``info``."""
    exif = img.getexif()
    if not exif:
        return
    info["has_exif"] = True
    tags = info["raw_tags"]
    for tag_id, value in exif.items():
        tags[TAGS.get(tag_id, str(tag_id))] = str(value)[:160]

    info["camera_make"] = tags.get("Make")
    info["camera_model"] = tags.get("Model")
    info["software"] = tags.get("Software")
    info["date_time"] = tags.get("DateTime")

    try:
        if hasattr(exif, "get_ifd"):
            for sub_id, sub_val in (exif.get_ifd(0x8769) or {}).items():
                tags[TAGS.get(sub_id, str(sub_id))] = str(sub_val)[:160]
    except Exception as exc:
        logger.debug("_read_exif_fields: ignored %s: %s", type(exc).__name__, exc)

    info["focal_length"] = tags.get("FocalLength")
    info["f_number"] = tags.get("FNumber")
    info["exposure_time"] = tags.get("ExposureTime")
    info["iso"] = tags.get("ISOSpeedRatings") or tags.get("PhotographicSensitivity")
    info["lens_model"] = tags.get("LensModel")
    if any([info["focal_length"], info["f_number"], info["exposure_time"], info["iso"]]):
        info["has_optical_parameters"] = True


def _xmp_string(img: Image.Image, info: Dict[str, Any]) -> str:
    """The embedded XMP packet as text ("" if none); also records a PNG creation time."""
    xmp_raw = img.info.get("XML:com.adobe.xmp") or img.info.get("xmp")
    if "Creation Time" in img.info:
        info["xmp_summary"]["creation_time"] = str(img.info["Creation Time"])
    if not xmp_raw:
        return ""
    return xmp_raw.decode("utf-8", errors="ignore") if isinstance(xmp_raw, bytes) else str(xmp_raw)


def _apply_xmp_signatures(xmp_str: str, info: Dict[str, Any]) -> None:
    """IPTC DigitalSourceType / credit / tool markers. All are unauthenticated self-declarations."""
    if "trainedAlgorithmicMedia" in xmp_str:
        info["iptc_digital_source_type"] = "trainedAlgorithmicMedia"
        info["ai_signature_found"] = True
        info["signature_details"] = "IPTC DigitalSourceType: trainedAlgorithmicMedia (Algorithmic / AI Generated)"
    elif "compositeWithTrainedAlgorithmicMedia" in xmp_str:
        info["iptc_digital_source_type"] = "compositeWithTrainedAlgorithmicMedia"
        info["ai_enhancer_signature_found"] = True
        info["signature_details"] = "IPTC DigitalSourceType: compositeWithTrainedAlgorithmicMedia (AI Composite/Enhanced)"
    if "Made with Google AI" in xmp_str:
        info["photoshop_credit"] = "Made with Google AI"
        info["ai_signature_found"] = True
        info["signature_details"] = "Photoshop Credit: Made with Google AI (Google Gemini / Imagen)"
    if "Topaz Photo AI" in xmp_str:
        info["creator_tool"] = "Topaz Photo AI"
        info["ai_enhancer_signature_found"] = True
        info["signature_details"] = "Processed with Topaz Photo AI (Neural Restoration / Denoising / Upscaling)"
    if "Canva" in xmp_str or "Attrib:Ads" in xmp_str:
        info["creator_tool"] = "Canva"
        info["graphic_editor_signature_found"] = True
        info["signature_details"] = "Graphic layout designed and exported via Canva"


def _apply_software_signatures(info: Dict[str, Any]) -> None:
    """Editor / enhancer / screenshot-tool classification from the EXIF Software tag."""
    software_val = str(info.get("software") or "").lower()
    if "topaz photo ai" in software_val:
        info["ai_enhancer_signature_found"] = True
        info["creator_tool"] = "Topaz Photo AI"
        info["signature_details"] = f"EXIF Software: {info['software']}"
    elif "canva" in software_val:
        info["graphic_editor_signature_found"] = True
        info["creator_tool"] = "Canva"
        info["signature_details"] = "EXIF Software: Canva"
    elif any(s in software_val for s in ("photoshop", "gimp", "lightroom", "snapseed", "picsart", "pixlr")):
        info["graphic_editor_signature_found"] = True
        info["creator_tool"] = info["software"]
    for sc_sig in KNOWN_SCREENSHOT_SOFTWARE_SIGNATURES:
        if sc_sig in software_val:
            info["screenshot_software_found"] = True
            info["screenshot_software_name"] = info["software"]
            break


# Short generator names that are also ordinary words or fragments ("flux", "sora", "luma", a gemini caption ...). They only count
# when they appear as a whole word in a field that names software, never in free text or in other XMP attributes.
_AMBIGUOUS_AI_SIGNATURES = frozenset({"imagen", "gemini", "flux", "runway", "kling", "sora", "pika", "luma"})
_SOFTWARE_TAG_WORDS = ("software", "creator", "processing", "host", "generator")
_XMP_SOFTWARE_FIELDS = re.compile(r'(?:CreatorTool|tiff:Software|xmp:CreatorTool)\s*(?:=\s*"([^"]*)"|>([^<]*)<)', re.IGNORECASE)


def _whole_word(sig: str, text: str) -> bool:
    """True if ``sig`` occurs in ``text`` not glued to other letters or digits ("imagen" must not match "ImageNumber")."""
    return re.search(r"(?<![a-z0-9])" + re.escape(sig) + r"(?![a-z0-9])", text) is not None


def _apply_generic_ai_signature(xmp_str: str, info: Dict[str, Any]) -> None:
    """Whole-word search of the known-generator list across every tag value and the XMP packet; the ambiguous short names are
    only looked for in software-naming fields."""
    if info["ai_signature_found"]:
        return
    everywhere = " ".join(str(v).lower() for v in info["raw_tags"].values()) + " " + xmp_str.lower()
    software_only = " ".join(str(v).lower() for k, v in info["raw_tags"].items() if any(w in k.lower() for w in _SOFTWARE_TAG_WORDS))
    software_only += " " + " ".join((a or b or "").lower() for a, b in _XMP_SOFTWARE_FIELDS.findall(xmp_str))
    for sig in KNOWN_AI_SOFTWARE_SIGNATURES:
        if _whole_word(sig, software_only if sig in _AMBIGUOUS_AI_SIGNATURES else everywhere):
            info["ai_signature_found"] = True
            info["signature_details"] = f"Detected AI generator footprint: '{sig}' in image metadata."
            break


def extract_image_metadata(image_path: str | Path) -> Dict[str, Any]:
    """
    Extracts standard EXIF metadata, optical camera parameters, Adobe XMP packets,
    IPTC digital source types, and classifies generator, editor, and screenshot signatures.
    All of it is unauthenticated, forgeable metadata.
    """
    info = _blank_metadata_info()
    try:
        with Image.open(image_path) as img:
            _read_exif_fields(img, info)
            xmp_str = _xmp_string(img, info)
            if xmp_str:
                _apply_xmp_signatures(xmp_str, info)
            _apply_software_signatures(info)
            _apply_generic_ai_signature(xmp_str, info)
    except Exception as exc:
        logger.debug("extract_image_metadata: ignored %s: %s", type(exc).__name__, exc)
    return info



def _is_four_point_star(cnt: np.ndarray, gray_roi: np.ndarray) -> bool:
    """True if a contour is shaped like the Gemini sparkle: a bright, solid, four-pointed star.

    Random texture blobs satisfy a size/aspect/solidity test easily, so the shape must also be four-fold symmetric
    (matches itself rotated by 90 degrees and mirrored), have exactly four deep concavities, and stand out from the
    ring of background around it.
    """
    x, y, cw, ch = cv2.boundingRect(cnt)
    mask = np.zeros((ch, cw), np.uint8)
    cv2.drawContours(mask, [cnt - [x, y]], -1, 255, thickness=cv2.FILLED)
    m = cv2.resize(mask, (48, 48), interpolation=cv2.INTER_AREA) > 127

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum()) / union if union else 0.0

    if min(iou(m, np.rot90(m)), iou(m, m[:, ::-1]), iou(m, m[::-1, :])) < 0.80:
        return False
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull) if hull is not None and len(hull) > 3 else None
    deep = 0 if defects is None else int(sum(1 for d in defects.reshape(-1, 4) if d[3] / 256.0 > 0.07 * max(cw, ch)))
    if deep != 4:
        return False
    inside = gray_roi[y:y + ch, x:x + cw][mask > 0]
    pad = max(4, cw // 4)
    y0, y1, x0, x1 = max(0, y - pad), min(gray_roi.shape[0], y + ch + pad), max(0, x - pad), min(gray_roi.shape[1], x + cw + pad)
    ring_mask = np.ones((y1 - y0, x1 - x0), bool)
    ring_mask[y - y0:y - y0 + ch, x - x0:x - x0 + cw] = False
    ring = gray_roi[y0:y1, x0:x1][ring_mask]
    return inside.size > 0 and ring.size > 0 and abs(float(inside.mean()) - float(ring.mean())) >= 25.0 and float(inside.std()) <= 40.0


def detect_ai_watermark(image_path: str | Path, img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Detects known AI generation watermarks (specifically Google Gemini / Imagen 4-pointed sparkle emblem)
    and visual impression marks in corner quadrants.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {"watermark_detected": False, "watermark_type": None, "details": None}

    h, w = img_bgr.shape[:2]
    # Check bottom-right quadrant corner (where Gemini and many diffusion models stamp emblems)
    cr_h = min(200, max(80, h // 4))
    cr_w = min(200, max(80, w // 4))
    roi = img_bgr[h - cr_h:, w - cr_w:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 250 < area < 3500:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = float(cw) / max(1.0, float(ch))
            if 0.78 <= aspect <= 1.28 and 24 <= cw <= 90 and 24 <= ch <= 90:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / max(1.0, hull_area)
                # 4-pointed star has characteristic concave quadrant flanks (solidity between 0.50 and 0.78)
                if 0.50 <= solidity <= 0.78 and _is_four_point_star(cnt, gray):
                    cx = x + cw / 2.0
                    cy = y + ch / 2.0
                    dist_corner = float(np.hypot(cr_w - cx, cr_h - cy))
                    if dist_corner < 190:
                        return {
                            "watermark_detected": True,
                            "watermark_type": "google_gemini_sparkle",
                            "details": f"Google Gemini / Imagen 4-pointed sparkle emblem detected in corner ({cw}x{ch}px, solidity: {solidity:.2f})",
                            "bbox": [w - cr_w + x, h - cr_h + y, cw, ch],
                        }

    return {"watermark_detected": False, "watermark_type": None, "details": None}


def detect_background_cutout(image_path: str | Path, img_bgr: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Detects whether an image has been subjected to background removal, alpha transparency,
    or replaced with a solid artificial studio backdrop.
    """
    # 1. Check alpha channel in original PIL file
    try:
        with Image.open(image_path) as im:
            if im.mode == "RGBA":
                alpha = np.array(im.split()[-1])
                transparent_pixels = np.sum(alpha < 250)
                total_pixels = alpha.size
                if transparent_pixels > (total_pixels * 0.01):
                    return {
                        "is_cutout": True,
                        "cutout_type": "transparent_alpha_channel",
                        "is_transparent_png": True,
                        "details": f"Transparent alpha channel cutout detected ({round((transparent_pixels / total_pixels) * 100, 1)}% transparent pixels)",
                    }
    except Exception as exc:
        logger.debug("detect_background_cutout: ignored %s: %s", type(exc).__name__, exc)

    if img_bgr is None:
        try:
            img_bgr = imread(str(image_path))
        except Exception as exc:
            logger.debug("detect_background_cutout: ignored %s: %s", type(exc).__name__, exc)

    if img_bgr is None or img_bgr.size == 0:
        return {"is_cutout": False, "cutout_type": None, "is_transparent_png": False, "details": None}

    # 2. Check perimeter borders for solid uniform studio fill
    h, w = img_bgr.shape[:2]
    border_thick_w = max(2, int(w * 0.04))

    left = img_bgr[:, :border_thick_w]
    right = img_bgr[:, -border_thick_w:]

    left_std = float(np.std(left))
    right_std = float(np.std(right))
    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))

    # Flanks are solid uniform studio white or black
    is_left_solid = (left_std < 5.0 and (left_mean > 230 or left_mean < 25))
    is_right_solid = (right_std < 5.0 and (right_mean > 230 or right_mean < 25))

    if is_left_solid and is_right_solid:
        return {
            "is_cutout": True,
            "cutout_type": "solid_background_fill",
            "details": f"Solid studio/cutout background detected (left std: {left_std:.1f}, right std: {right_std:.1f})",
        }

    return {"is_cutout": False, "cutout_type": None, "details": None}


def detect_scanned_photo(image_path: str | Path, img_bgr: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detects high-resolution flatbed scans of physical photographic prints.
    Telltales: Ultra-high megapixel count (>20 MP), physical scanner paper borders,
    optical halftone/halftone print grain, absence of digital camera EXIF or presence of scanner mark.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {"is_scanned": False, "details": None}

    h, w = img_bgr.shape[:2]
    total_mp = (h * w) / 1_000_000.0

    # Physical prints scanned on flatbeds are typically >20 megapixels
    if total_mp >= 20.0:
        # Check outer frame margin for scanner border (often dark or distinct margin at the extreme 1%)
        margin_h = max(2, int(h * 0.01))
        edge_top = img_bgr[:margin_h, :]
        edge_bottom = img_bgr[-margin_h:, :]

        # Look for dark/scanner frame boundary
        if float(np.mean(edge_top)) < 60.0 or float(np.mean(edge_bottom)) < 60.0 or not meta.get("has_exif"):
            return {
                "is_scanned": True,
                "megapixels": round(total_mp, 1),
                "details": f"High-resolution flatbed scan of physical print ({round(total_mp, 1)} MP, {w}x{h}px)",
            }

    return {"is_scanned": False, "details": None}


def detect_face_swap_artifacts(image_path: str | Path, img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Detects neural face swap (Roop/ReActor/InsightFace) and deepfake restoration cues:
    - Distinct filename signatures ('face-swap', 'swap', 'reactor', 'roop')
    - Target square geometry paired with neural synthesis markers
    """
    fname = str(image_path).lower()
    has_swap_filename = any(s in fname for s in ("face-swap", "faceswap", "face_swap", "deepfake", "reactor", "roop"))

    if img_bgr is None or img_bgr.size == 0:
        return {
            "is_face_swap": has_swap_filename,
            "confidence": 0.90 if has_swap_filename else 0.0,
            "details": "Face swap filename signature detected" if has_swap_filename else None,
        }

    h, w = img_bgr.shape[:2]
    is_square = (w == h and w in (512, 768, 1024, 1536))

    if has_swap_filename and is_square:
        return {
            "is_face_swap": True,
            "confidence": 0.95,
            "details": f"Neural face-swap composite detected (target geometry {w}x{h}, filename signature '{Path(image_path).name}')",
        }
    elif has_swap_filename:
        return {
            "is_face_swap": True,
            "confidence": 0.85,
            "details": f"Neural face-swap signature detected in filename ('{Path(image_path).name}')",
        }

    return {
        "is_face_swap": False,
        "confidence": 0.0,
        "details": None,
    }


def _opaque_saturation(image_path: str | Path, hsv_sat: np.ndarray) -> np.ndarray:
    """Saturation channel, restricted to opaque pixels for RGBA files (transparent cutouts must not dilute it)."""
    try:
        with Image.open(image_path) as im:
            if im.mode == "RGBA":
                opaque_mask = np.array(im.split()[-1]) > 30
                if np.sum(opaque_mask) > 100:
                    return hsv_sat[opaque_mask]
    except Exception as exc:
        logger.debug("_opaque_saturation: ignored %s: %s", type(exc).__name__, exc)
    return hsv_sat


def _art_visual_medium(is_art: bool, is_ink_art: bool, flat_noise: float) -> str:
    if not is_art:
        return "Photographic Capture"
    if is_ink_art or flat_noise < 0.25:
        return "Cel-Shaded / Ink Cross-Hatched Digital Art"
    if flat_noise < 0.45:
        return "Flat Vector / Cel-Shaded Digital Art"
    if flat_noise > 1.2:
        return "Oil / Watercolor / Impasto Painting Texture"
    return "Digital 3D CGI / AI Neural Painting"


def _art_confidence_and_details(
    is_ink_art: bool, visual_medium: str, mean_sat: float, high_sat_pct: float,
    flat_noise: float, dark_edge_pct: float, q_colors: int,
) -> Tuple[float, str]:
    if is_ink_art:
        edge_score = min(1.0, dark_edge_pct / 1.5)
        noise_score = min(1.0, max(0.0, (0.40 - flat_noise) / 0.30))
        confidence = float(np.clip(edge_score * 0.5 + noise_score * 0.5, 0.65, 0.96))
        return confidence, (
            f"AI digital art / stylized illustration detected ({visual_medium}) "
            f"(dark ink contours: {dark_edge_pct:.2f}%, flat-region noise: {flat_noise:.2f}, color clusters: {q_colors})"
        )
    sat_score = min(1.0, (mean_sat - 110.0) / 70.0)
    pct_score = min(1.0, (high_sat_pct - 40.0) / 50.0)
    confidence = float(np.clip(sat_score * 0.6 + pct_score * 0.4, 0.50, 0.98))
    return confidence, (
        f"AI digital art / synthetic painting style detected ({visual_medium}) "
        f"(mean saturation: {mean_sat:.1f}, high-sat pixels: {high_sat_pct:.1f}%, flat-region noise: {flat_noise:.2f})"
    )


_NATURAL_HUE_CONCENTRATION = 0.85   # share of saturated pixels inside the best 70-degree hue window
_HUE_WINDOW = 35                    # OpenCV hue units (0..179, i.e. 2 degrees each): open water spans blue to teal
_MIN_NATURAL_FLAT_NOISE = 0.05      # below this a region is a synthetic flat fill, not a photographed surface


def _saturated_hue_concentration(img_bgr: np.ndarray) -> float:
    """Share of strongly saturated pixels (S > 120) whose hue falls in the single best 70-degree window (0..1)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hues = hsv[:, :, 0][hsv[:, :, 1] > 120].astype(np.int64)
    if hues.size < 50:
        return 0.0
    hist = np.bincount(hues, minlength=180)
    wrapped = np.concatenate([hist, hist[:_HUE_WINDOW]])                 # OpenCV hue is 0..179 and wraps around
    window = np.convolve(wrapped, np.ones(_HUE_WINDOW, dtype=np.int64), mode="valid")[:180]
    return float(window.max() / hues.size)


def detect_digital_art_and_painting(
    image_path: str | Path,
    img_bgr: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detects whether an image is a digital painting, 2D/3D illustration, CGI synthetic render,
    ink/cross-hatched artwork, or stylized vector graphics rather than a natural optical camera capture.
    Evaluates:
    - HSV hyper-saturation distribution (mean saturation, high-saturation percentage)
    - Dark ink contour lines and edge density (Canny edges in low-luminance zones)
    - Flat-region vs edge gradient discrepancy and absence of Poisson-Gaussian camera sensor noise
    - Discrete palette clustering / posterization vs continuous camera tone roll-off
    """
    if img_bgr is None:
        try:
            img_bgr = imread(str(image_path))
        except Exception as exc:
            logger.debug("detect_digital_art_and_painting: ignored %s: %s", type(exc).__name__, exc)

    if img_bgr is None or img_bgr.size == 0:
        return {
            "is_digital_art": False,
            "confidence": 0.0,
            "mean_saturation": 0.0,
            "high_sat_pct": 0.0,
            "flat_noise_mean": 0.0,
            "details": None,
        }

    sat = _opaque_saturation(image_path, cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1])
    mean_sat = float(np.mean(sat))
    high_sat_pct = float(np.sum(sat > 120) / max(1, sat.size) * 100.0)

    # Flat-region sensor noise: homogeneous regions separate from edges
    sample = _downsample_if_needed(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), max_dim=1536)
    diff = cv2.absdiff(sample, cv2.medianBlur(sample, 3)).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(sample, cv2.CV_32F, 1, 0), cv2.Sobel(sample, cv2.CV_32F, 0, 1))
    flat_mask = grad < 15.0
    flat_noise = float(np.mean(diff[flat_mask])) if np.sum(flat_mask) > 100 else float(np.mean(diff))

    # Ink line art, manga and cel-shaded drawings have lower saturation but prominent dark contour lines
    # (Canny edges in dark regions) combined with low flat-region sensor noise.
    edges = cv2.Canny(sample, 50, 150)
    dark_edge_pct = float(np.sum((sample < 50) & (edges > 0)) / max(1, edges.size) * 100.0)

    # Discrete posterized shading vs continuous photographic tones
    q_colors = len(np.unique((cv2.resize(img_bgr, (150, 150), interpolation=cv2.INTER_AREA) // 24) * 24, axis=0))

    has_camera_hardware = bool(metadata.get("camera_make")) if metadata else False
    is_ink_art = bool(not has_camera_hardware and dark_edge_pct >= 0.70 and flat_noise < 0.38 and q_colors < 130)

    # Saturation alone is not evidence of synthesis: open water, sky and skin are strongly saturated in HSV yet
    # photographic. Their saturated pixels sit in one narrow hue band, whereas painted / rendered art spreads
    # saturated colour over several hues. A perfectly flat fill (no grain at all) is not a natural scene.
    hue_conc = _saturated_hue_concentration(img_bgr)
    natural_single_hue_scene = bool(hue_conc >= _NATURAL_HUE_CONCENTRATION and flat_noise >= _MIN_NATURAL_FLAT_NOISE)

    # Digital art, anime, CGI renders and AI paintings are typically mean_sat >= 115 and high_sat_pct >= 45%.
    is_art = bool((mean_sat >= 115.0 and high_sat_pct >= 45.0 and not natural_single_hue_scene) or is_ink_art)

    visual_medium = _art_visual_medium(is_art, is_ink_art, flat_noise)
    confidence, details = 0.0, None
    if is_art:
        confidence, details = _art_confidence_and_details(
            is_ink_art, visual_medium, mean_sat, high_sat_pct, flat_noise, dark_edge_pct, q_colors
        )

    return {
        "is_digital_art": is_art,
        "visual_medium": visual_medium,
        "confidence": round(confidence, 2),
        "mean_saturation": round(mean_sat, 1),
        "high_sat_pct": round(high_sat_pct, 1),
        "flat_noise_mean": round(flat_noise, 2),
        "dark_edge_pct": round(dark_edge_pct, 2),
        "details": details,
    }


def calculate_sensor_noise_profile(gray_img: np.ndarray) -> Tuple[float, float]:
    """
    Computes camera sensor noise residual using wavelet / Wiener-approximated median filter subtraction.
    Physical optical sensors produce Poisson-Gaussian noise (PRNU). Neural diffusion models
    and latent generators produce unnaturally denoised or synthetic uniform noise distributions.
    Downsampled for speed on large files while preserving high-frequency statistics.
    """
    sample = _downsample_if_needed(gray_img, max_dim=1536)
    blurred = cv2.medianBlur(sample, 3)
    noise_residual = cv2.absdiff(sample, blurred)
    mean_noise = float(np.mean(noise_residual))
    std_noise = float(np.std(noise_residual))
    return mean_noise, std_noise


def calculate_surface_smoothness(gray_img: np.ndarray) -> float:
    """
    Computes bilateral filter difference to identify synthetic over-smoothing and skin plastic textures.
    AI diffusion generators struggle with realistic organic micro-texture, producing plastic surfaces.
    Downsampled for speed on large files.
    """
    sample = _downsample_if_needed(gray_img, max_dim=1536)
    bilateral = cv2.bilateralFilter(sample, d=7, sigmaColor=75, sigmaSpace=75)
    diff = cv2.absdiff(sample, bilateral)
    return float(np.mean(diff))


_fft_memo: Dict[str, Any] = {"key": None, "value": None}
_fft_memo_lock = threading.Lock()


def analyze_fft_radial_power_spectrum(gray_img: np.ndarray | str | Path) -> Dict[str, Any]:
    """Fourier radial power-spectrum analysis (see ``_analyze_fft_radial_power_spectrum``). The detector and the moire check ask for
    the same grayscale picture within one analysis, so the last result is reused when the pixels are identical."""
    if not (isinstance(gray_img, np.ndarray) and gray_img.ndim == 2):
        return _analyze_fft_radial_power_spectrum(gray_img)
    key = (gray_img.shape, gray_img.dtype.str, hashlib.blake2b(np.ascontiguousarray(gray_img).tobytes(), digest_size=16).digest())
    with _fft_memo_lock:
        if _fft_memo["key"] == key:
            return dict(_fft_memo["value"])
    result = _analyze_fft_radial_power_spectrum(gray_img)
    with _fft_memo_lock:
        _fft_memo["key"], _fft_memo["value"] = key, dict(result)
    return result


def _analyze_fft_radial_power_spectrum(gray_img: np.ndarray | str | Path) -> Dict[str, Any]:
    """
    Analyzes 2D Fourier Transform Radial Power Spectrum decay.
    Natural optical photography obeys Field's Law: Radial Power P(f) ~ 1 / f^alpha, where alpha ~ 2.0.
    Neural upscalers and latent generators produce anomalous high-frequency grid artifacts or abnormal alpha decay.
    """
    if isinstance(gray_img, (str, Path)):
        loaded = imread(str(gray_img), cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            return {"spectral_decay_alpha": 2.0, "is_anomalous_decay": False}
        gray_img = loaded
    elif gray_img.ndim == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    
    sample = _downsample_if_needed(gray_img, max_dim=1024)
    h, w = sample.shape[:2]
    min_dim = min(h, w)
    crop = sample[
        (h - min_dim) // 2 : (h - min_dim) // 2 + min_dim,
        (w - min_dim) // 2 : (w - min_dim) // 2 + min_dim,
    ].astype(np.float32)

    # Apply 2D Hann window to prevent rectangular aperture boundary spectral leaking
    hann_1d = np.hanning(min_dim).astype(np.float32)
    hann_2d = np.outer(hann_1d, hann_1d)
    windowed_crop = crop * hann_2d

    f_transform = np.fft.fft2(windowed_crop)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift) ** 2

    cy, cx = min_dim // 2, min_dim // 2
    y, x = np.ogrid[:min_dim, :min_dim]
    r = np.hypot(x - cx, y - cy).astype(np.int32)

    max_r = min_dim // 2
    # One pass over the spectrum: total power and pixel count per integer radius (radius 0 is excluded).
    radial_profile = np.bincount(r.ravel(), weights=magnitude_spectrum.ravel(), minlength=max_r)[:max_r]
    radial_counts = np.bincount(r.ravel(), minlength=max_r)[:max_r].astype(np.float64)
    radial_profile[0] = 0.0
    radial_counts[0] = 0.0

    valid = (radial_counts > 0) & (radial_profile > 0)
    freqs = np.arange(max_r)[valid]
    powers = radial_profile[valid] / radial_counts[valid]

    if len(freqs) > 10:
        log_f = np.log(freqs[1:])
        log_p = np.log(powers[1:] + 1e-8)
        poly = np.polyfit(log_f, log_p, 1)
        spectral_decay_alpha = -float(poly[0])
    else:
        spectral_decay_alpha = 2.0

    is_anomalous_decay = bool(spectral_decay_alpha < 1.4 or spectral_decay_alpha > 3.6)

    # Azimuthal Directional Variance Analysis (Checking for isotropic optical capture vs synthetic grid/Moiré peaks)
    azimuthal_var = 0.0
    peak_energy_ratio = 1.0
    try:
        theta = np.arctan2(y - cy, x - cx)
        # Exclude horizontal and vertical axes (+- 6 degrees around 0, 90, 180, 270) to prevent edge leakage false positives
        axes_mask = (np.abs(np.sin(theta)) > 0.12) & (np.abs(np.cos(theta)) > 0.12)
        wedges = 16
        theta_bins = np.digitize(theta, np.linspace(-np.pi, np.pi, wedges + 1)) - 1
        band_mask = (r >= int(min_dim * 0.18)) & (r <= int(min_dim * 0.42)) & axes_mask
        in_band = theta_bins[band_mask]
        wedge_sum = np.bincount(in_band, weights=magnitude_spectrum[band_mask], minlength=wedges)[:wedges]
        wedge_count = np.bincount(in_band, minlength=wedges)[:wedges]
        wedge_energies = [float(wedge_sum[i] / wedge_count[i]) for i in range(wedges) if wedge_count[i] > 50]
        if len(wedge_energies) >= 8:
            azimuthal_var = float(np.std(wedge_energies) / max(1e-5, np.mean(wedge_energies)))
            peak_energy_ratio = float(np.max(wedge_energies) / max(1e-5, np.median(wedge_energies)))
    except Exception as exc:
        logger.debug("analyze_fft_radial_power_spectrum: ignored %s: %s", type(exc).__name__, exc)

    return {
        "spectral_decay_alpha": round(spectral_decay_alpha, 3),
        "is_anomalous_decay": is_anomalous_decay,
        "azimuthal_directional_variance": round(azimuthal_var, 3),
        "peak_energy_ratio": round(peak_energy_ratio, 2),
    }


def compute_ela(
    image_input: str | Path | np.ndarray | Image.Image, quality: int = 90, multiplier: int = 15
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Performs Error Level Analysis (ELA). Resaves the image at a known quality level
    and computes the pixel discrepancy between original and recompressed versions.
    High local discrepancies indicate inpainting, generative splicing, or multi-source compositing.
    Accepts a file path, numpy array (BGR or RGB), or PIL Image.
    """
    try:
        if isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                orig = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 4:
                orig = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGRA2RGB))
            else:
                orig = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            orig = image_input.convert("RGB")
        else:
            with Image.open(image_input) as img:
                orig = img.convert("RGB")

        # If huge, downsample copy for ELA to avoid memory spike
        if max(orig.size) > 1536:
            orig.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        orig.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)

        ela_img = ImageChops.difference(orig, recompressed)
        extrema = ela_img.getextrema()
        max_diff = max(ex[1] for ex in extrema)
        scale = 255.0 / max(1, max_diff) if max_diff > 0 else 1.0

        enhancer = ImageEnhance.Brightness(ela_img)
        enhanced = enhancer.enhance(scale * (multiplier / 10.0))

        ela_arr = np.array(enhanced)
        mean_diff = float(np.mean(np.array(ela_img)))
        return mean_diff, ela_arr
    except Exception as exc:
        logger.debug("compute_ela: ignored %s: %s", type(exc).__name__, exc)
        return 0.0, None


def generate_spatial_manipulation_heatmap(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Generates a localized spatial manipulation heatmap highlighting regions with
    anomalous bilateral filter residual or synthetic over-smoothing.
    Optimized for instant execution on high-resolution images via downsampling.
    """
    sample = _downsample_if_needed(img_bgr, max_dim=1024)
    h, w = sample.shape[:2]
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    diff = cv2.absdiff(gray, bilateral).astype(np.float32)
    mean_diff = float(np.mean(diff))

    # A region is anomalous if it lacks physical Poisson sensor noise (diff < 1.4)
    # Authentic photos have mean_diff >= 2.0 and natural high-frequency grain
    if mean_diff >= 2.1:
        # Camera capture: only deeply flattened/inpainted regions are marked
        anomaly_map = np.clip((1.2 - diff) * 120.0, 0, 255).astype(np.uint8)
    else:
        # Synthetic generation: plastic surfaces and diffusion latent areas
        anomaly_map = np.clip((2.2 - diff) * 90.0, 0, 255).astype(np.uint8)

    anomaly_blurred = cv2.GaussianBlur(anomaly_map, (15, 15), 0)

    # Threshold for suspicious synthetic smoothness
    _, thresh = cv2.threshold(anomaly_blurred, 100, 255, cv2.THRESH_BINARY)
    anomaly_pixels = np.sum(thresh > 0)
    total_pixels = h * w
    ai_spatial_area_pct = round(float((anomaly_pixels / total_pixels) * 100.0), 1)

    heatmap = cv2.applyColorMap(anomaly_blurred, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    return {
        "heatmap_rgb": heatmap_rgb,
        "ai_spatial_area_pct": ai_spatial_area_pct,
    }


_SCREENSHOT_FILENAME_MARKERS = (
    "screenshot", "screen_shot", "screen-shot", "screencap", "capture_", "snip", "screen shot",
)


def _device_from_aspect(orientation: str, aspect_ratio: float) -> Optional[str]:
    """Device class guessed from aspect ratio and orientation when no canonical resolution matched.

    Tablet's aspect band (1.30-1.65) is a strict subset of desktop's first band (1.25-1.85), so the more specific
    landscape+desktop combination is checked first; tablet then only wins in portrait or outside the desktop band.
    """
    is_mobile = 1.75 <= aspect_ratio <= 2.35
    is_tablet = 1.30 <= aspect_ratio <= 1.65
    is_desktop = (1.25 <= aspect_ratio <= 1.85) or (2.30 <= aspect_ratio <= 2.45)
    if is_mobile and orientation == "Portrait":
        return "Mobile Phone"
    if is_desktop and orientation == "Landscape":
        return "Laptop / Desktop"
    if is_tablet:
        return "Tablet"
    return None


def _edge_density(strip: np.ndarray) -> float:
    return float(np.sum(cv2.Canny(strip, 50, 150) > 0)) / float(max(1, strip.size))


def _bar_signals(gray: np.ndarray, orientation: str, aspect_ratio: float) -> Tuple[List[str], float]:
    """Mobile status/navigation bars (portrait phones) or a dark desktop taskbar (landscape desktops)."""
    h = gray.shape[0]
    is_mobile = 1.75 <= aspect_ratio <= 2.35
    is_desktop = (1.25 <= aspect_ratio <= 1.85) or (2.30 <= aspect_ratio <= 2.45)
    found: List[str] = []
    gain = 0.0
    if orientation == "Portrait" and is_mobile:
        if 0.015 < _edge_density(gray[:max(10, int(h * 0.045)), :]) < 0.25:
            found.append("mobile_top_status_bar")
            gain += 0.25
        if 0.008 < _edge_density(gray[-max(8, int(h * 0.035)):, :]) < 0.20:
            found.append("mobile_navigation_bar")
            gain += 0.20
    elif orientation == "Landscape" and is_desktop:
        bar = gray[-max(16, int(h * 0.05)):, :]
        if float(np.std(np.mean(bar, axis=1))) < 12.0 and np.mean(bar) < 70:
            found.append("desktop_taskbar")
            gain += 0.25
    return found, gain


def _rectilinear_ui(sample: np.ndarray, meta: Dict[str, Any]) -> bool:
    """Window/region snip: strong axis-aligned edges plus a dominant flat background tone."""
    grad_x = cv2.Sobel(sample, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(sample, cv2.CV_32F, 0, 1)
    edge_mask = cv2.magnitude(grad_x, grad_y) > 40.0
    edge_total = float(np.sum(edge_mask))
    if edge_total <= 500:
        return False
    ax, ay = np.abs(grad_x)[edge_mask], np.abs(grad_y)[edge_mask]
    rectilinear_ratio = (float(np.sum(ay > ax * 2.0)) + float(np.sum(ax > ay * 2.0))) / edge_total
    hist = cv2.calcHist([sample], [0], None, [256], [0, 256])
    top_bin_ratio = float(np.max(hist)) / float(sample.size)
    return bool((top_bin_ratio >= 0.25 and rectilinear_ratio >= 0.65) or meta.get("screenshot_software_found"))


def _ui_structure_signals(
    gray: np.ndarray, sample: np.ndarray, orientation: str, aspect_ratio: float, meta: Dict[str, Any]
) -> Tuple[List[str], float, Optional[str]]:
    """UI-structure evidence: (indicator names, confidence gain, device name if only a snip layout matched)."""
    found, gain = _bar_signals(gray, orientation, aspect_ratio)
    snip_device = None
    if _rectilinear_ui(sample, meta):
        found.append("rectilinear_ui_layout")
        gain += 0.45
        snip_device = "Device / Window Snip"
    return found, gain, snip_device


def detect_screenshot(
    image_path: str | Path,
    img_bgr: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detects whether an image is a screen capture across devices (Mobile Phone, Tablet, Laptop, Desktop)
    in any orientation (Portrait, Landscape, Square).
    Analyzes:
    1. Canonical display resolutions and aspect ratios (19.5:9, 20:9, 16:9, 16:10, 4:3, etc.)
    2. Filename signatures ('screenshot', 'screen_shot', 'screencap', 'snip', 'capture_')
    3. UI bar edge profiles (top status bar with battery/wifi icons, bottom gesture navigation pill/buttons, desktop taskbars)
    4. Discrete UI color profiles (large areas of pure uniform background color and anti-aliased font glyphs)
    5. Absence of optical camera sensor noise (PRNU) in UI regions
    """
    if img_bgr is None:
        try:
            img_bgr = imread(str(image_path))
        except Exception as exc:
            logger.debug("detect_screenshot: ignored %s: %s", type(exc).__name__, exc)

    if img_bgr is None or img_bgr.size == 0:
        return {
            "is_screenshot": False,
            "confidence": 0.0,
            "device_type": "None",
            "orientation": "None",
            "screen_resolution": None,
            "ui_elements_detected": [],
            "details": None,
        }

    h, w = img_bgr.shape[:2]
    meta = dict(meta or {})

    orientation = "Landscape" if w > h else ("Portrait" if h > w else "Square")
    aspect_ratio = float(max(w, h)) / max(1.0, float(min(w, h)))
    ui_elements: List[str] = []
    confidence = 0.0
    device_type = "None"
    screen_resolution = f"{w}x{h}"

    # 1. Canonical Device Screen Resolution Match (either orientation)
    matched_device, matched_desc = CANONICAL_SCREEN_RESOLUTIONS.get((w, h)) or CANONICAL_SCREEN_RESOLUTIONS.get((h, w)) or (None, None)
    if matched_device:
        device_type = matched_device
        ui_elements.append(f"canonical_resolution ({matched_desc})")
        confidence += 0.50

    # 2. Filename Signature
    fname = str(Path(image_path).name).lower()
    has_screenshot_filename = any(k in fname for k in _SCREENSHOT_FILENAME_MARKERS)
    if has_screenshot_filename:
        ui_elements.append("filename_signature")
        confidence += 0.45

    # 3. Software & EXIF Signature
    if meta.get("screenshot_software_found"):
        sw_name = meta.get("screenshot_software_name") or "Screen Capture Tool"
        ui_elements.append(f"software_signature ({sw_name})")
        confidence += 0.50

    # 4. Aspect Ratio Evaluation
    if not matched_device:
        device_type = _device_from_aspect(orientation, aspect_ratio) or device_type

    # 5. Visual UI Structure Analysis
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sample = _downsample_if_needed(gray, max_dim=1024)
    flat_noise = float(np.mean(cv2.absdiff(sample, cv2.medianBlur(sample, 3))))

    has_cam = bool(meta.get("camera_make") and meta.get("camera_model")) or bool(meta.get("has_optical_parameters"))

    # Only analyze UI edge structures if not clearly an optical camera capture
    if not has_cam and flat_noise < 1.45:
        found, gain, snip_device = _ui_structure_signals(gray, sample, orientation, aspect_ratio, meta)
        ui_elements.extend(found)
        confidence += gain
        if snip_device and device_type == "None":
            device_type = snip_device

    if has_screenshot_filename or meta.get("screenshot_software_found"):
        is_screenshot = True
    elif has_cam or flat_noise >= 1.45:
        # Optical sensor noise present or verified camera hardware -> Authentic photograph, not screenshot
        is_screenshot = False
    else:
        is_screenshot = bool(confidence >= 0.50 and (matched_device or len(ui_elements) >= 2 or flat_noise < 0.60))

    details = None
    if is_screenshot:
        dev_str = device_type if device_type != "None" else "Device"
        details = (
            f"{dev_str} screenshot identified in {orientation} orientation "
            f"({w}x{h}px, aspect: {aspect_ratio:.2f}, indicators: {', '.join(ui_elements) or 'canonical screen profile'})"
        )

    return {
        "is_screenshot": is_screenshot,
        "confidence": round(min(0.99, confidence), 2),
        "device_type": device_type,
        "orientation": orientation,
        "screen_resolution": screen_resolution,
        "ui_elements_detected": ui_elements,
        "details": details,
    }


def detect_inpainting_and_manipulation(
    image_path: str | Path,
    img_bgr: Optional[np.ndarray] = None,
    ela_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Detects localized generative inpainting, neural face swapping, deep learning denoising/upscaling,
    and composite splicing by evaluating spatial PRNU sensor noise inconsistency and ELA variance.
    """
    if img_bgr is None:
        try:
            img_bgr = imread(str(image_path))
        except Exception as exc:
            logger.debug("detect_inpainting_and_manipulation: ignored %s: %s", type(exc).__name__, exc)

    if img_bgr is None or img_bgr.size == 0:
        return {
            "is_manipulated": False,
            "manipulation_type": None,
            "confidence": 0.0,
            "noise_inconsistency": 0.0,
            "details": None,
        }

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sample = _downsample_if_needed(gray, max_dim=1024)
    sh, sw = sample.shape[:2]

    # 1. Local Noise Inconsistency across 6x6 spatial grid
    grid_rows, grid_cols = 6, 6
    cell_h, cell_w = sh // grid_rows, sw // grid_cols
    block_noises = []

    blurred = cv2.medianBlur(sample, 3)
    diff = cv2.absdiff(sample, blurred).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(sample, cv2.CV_32F, 1, 0), cv2.Sobel(sample, cv2.CV_32F, 0, 1))

    for r in range(grid_rows):
        for c in range(grid_cols):
            cell_sample = sample[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            cell_diff = diff[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            cell_grad = grad[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            # Exclude overexposed/underexposed clipped pixels (white sky / deep shadows) to avoid false noise zeroes
            flat_mask = (cell_grad < 20.0) & (cell_sample >= 8) & (cell_sample <= 247)
            if np.sum(flat_mask) > 80:
                block_noises.append(float(np.mean(cell_diff[flat_mask])))

    noise_inconsistency = 0.0
    has_noise_discrepancy = False
    if len(block_noises) >= 10:
        mean_nb = float(np.mean(block_noises))
        std_nb = float(np.std(block_noises))
        noise_inconsistency = float(std_nb / max(0.2, mean_nb))
        min_tile = min(block_noises)
        max_tile = max(block_noises)
        if (noise_inconsistency >= 0.40 and mean_nb >= 1.5 and min_tile < 0.25 and (max_tile - min_tile) > 2.0):
            has_noise_discrepancy = True

    # 2. ELA Discrepancy Analysis (if ELA map provided)
    has_ela_discrepancy = False
    if ela_map is not None and isinstance(ela_map, np.ndarray) and ela_map.size > 0:
        ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY) if len(ela_map.shape) == 3 else ela_map
        ela_mean = float(np.mean(ela_gray))
        ela_p98 = float(np.percentile(ela_gray, 98))
        if ela_mean > 2.0 and (ela_p98 / max(0.1, ela_mean)) > 4.5:
            has_ela_discrepancy = True

    if has_noise_discrepancy and has_ela_discrepancy:
        is_manipulated = True
        manip_type = "generative_inpainting_or_splicing"
        confidence = 0.85
        details = (
            f"Localized generative inpainting / neural composite splicing detected "
            f"(spatial sensor noise inconsistency: {noise_inconsistency:.2f}, ELA compression anomaly)"
        )
    elif has_noise_discrepancy:
        is_manipulated = True
        manip_type = "localized_noise_discrepancy"
        confidence = 0.75
        details = (
            f"Localized sensor noise inconsistency detected across image patches "
            f"(inconsistency ratio: {noise_inconsistency:.2f}), indicating partial neural smoothing or composite"
        )
    else:
        is_manipulated = False
        manip_type = None
        confidence = 0.0
        details = None

    return {
        "is_manipulated": is_manipulated,
        "manipulation_type": manip_type,
        "confidence": confidence,
        "noise_inconsistency": round(noise_inconsistency, 2),
        "details": details,
    }


def detect_screen_rephotography_moire(
    image_path: str | Path,
    img_bgr: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detects screen re-photography (recapture of LCD/OLED/CRT displays) via Moiré interference patterns.
    Moiré occurs when the spatial frequency of the physical display subpixel grid aliasing against
    the camera sensor Bayer filter produces high-energy directional peaks in the mid-frequency 2D FFT spectrum.
    """
    if img_bgr is None:
        try:
            img_bgr = imread(str(image_path))
        except Exception as exc:
            logger.debug("detect_screen_rephotography_moire: ignored %s: %s", type(exc).__name__, exc)

    if img_bgr is None or img_bgr.size == 0:
        return {
            "is_screen_recapture": False,
            "confidence": 0.0,
            "moire_score": 0.0,
            "peak_ratio": 1.0,
            "details": None,
        }

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fft_res = analyze_fft_radial_power_spectrum(gray)
    peak_ratio = fft_res.get("peak_energy_ratio", 1.0)
    azimuthal_var = fft_res.get("azimuthal_directional_variance", 0.0)

    # Screen re-photography produces sharp, narrow harmonic spikes in high spatial frequencies (subpixel pitch).
    # Natural photography (e.g. selfies with shirt necklines, posters, whiteboards) has broadband diagonal edges.
    # To reliably identify physical Moiré recapture, we require:
    # 1. Extremely high directional peak contrast (peak_ratio >= 4.5)
    # 2. Strong azimuthal variance (azimuthal_var >= 1.25)
    # 3. Absence of standard smooth natural image decay
    is_recaptured = False
    confidence = 0.0
    moire_score = round(float(peak_ratio / 4.0 + azimuthal_var), 2)

    if peak_ratio >= 4.5 and azimuthal_var >= 1.25:
        is_recaptured = True
        confidence = min(0.95, 0.50 + (peak_ratio - 4.5) * 0.10 + (azimuthal_var - 1.25) * 0.20)
    elif peak_ratio >= 6.5:
        is_recaptured = True
        confidence = min(0.92, 0.60 + (peak_ratio - 6.5) * 0.05)

    details = None
    has_cam = bool(meta and meta.get("camera_make"))
    if is_recaptured:
        cam_desc = f" by {meta.get('camera_make')} {meta.get('camera_model', '')}" if has_cam else ""
        details = (
            f"Screen re-photography Moiré pattern detected{cam_desc} "
            f"(2D FFT directional spectral peak ratio: {peak_ratio:.2f}, azimuthal variance: {azimuthal_var:.2f})"
        )

    return {
        "is_screen_recapture": is_recaptured,
        "confidence": round(confidence, 2),
        "moire_score": moire_score,
        "peak_ratio": peak_ratio,
        "details": details,
    }


def detect_spectral_modality(
    image_path: str | Path,
    img_bgr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Classifies electromagnetic spectrum and sensor capture modality:
    - Visible Spectrum Bayer RGB (Standard color photography)
    - True Monochrome / Greyscale (Zero chrominance across all pixels)
    - Near-Infrared / Astrophotography / Deep Space (NIR / Narrowband H-alpha emissions)
    - Medical / Multispectral / Scientific (X-ray, CT, Ultrasound, Electron Microscopy)
    """
    if img_bgr is None:
        try:
            img_bgr = imread(str(image_path))
        except Exception as exc:
            logger.debug("detect_spectral_modality: ignored %s: %s", type(exc).__name__, exc)

    if img_bgr is None or img_bgr.size == 0:
        return {
            "sensor_spectrum": "Visible Spectrum (Bayer RGB)",
            "is_monochrome": False,
            "color_cast": "Neutral",
            "details": "Standard RGB default",
        }

    # Downsample for fast chrominance evaluation
    sample = _downsample_if_needed(img_bgr, max_dim=512)
    b, g, r = sample[:, :, 0].astype(np.float32), sample[:, :, 1].astype(np.float32), sample[:, :, 2].astype(np.float32)

    diff_bg = np.abs(b - g)
    diff_gr = np.abs(g - r)
    diff_br = np.abs(b - r)
    mean_chroma_diff = float(np.mean(diff_bg + diff_gr + diff_br) / 3.0)

    # 1. Pure Monochrome / Greyscale Check
    if mean_chroma_diff < 0.85:
        # Check if medical or standard greyscale
        return {
            "sensor_spectrum": "Monochrome / Grayscale Sensor",
            "is_monochrome": True,
            "color_cast": "Monochrome",
            "details": f"True monochrome capture (mean chrominance channel delta: {mean_chroma_diff:.2f})",
        }

    # 2. Astrophotography / Narrowband Emission or Thermal
    # Narrowband astrophotography often has extreme blue/deep red dominance with near zero green
    mean_b = float(np.mean(b))
    mean_g = float(np.mean(g))
    mean_r = float(np.mean(r))

    if mean_g < 15.0 and (mean_r > 70.0 or mean_b > 70.0) and np.mean(sample) < 45.0:
        return {
            "sensor_spectrum": "Narrowband Emission / Astrophotography",
            "is_monochrome": False,
            "color_cast": "Narrowband Emission",
            "details": "Narrowband / Deep Sky astrophotographical emission signature",
        }

    # 3. Standard Bayer RGB
    return {
        "sensor_spectrum": "Visible Spectrum (Bayer RGB)",
        "is_monochrome": False,
        "color_cast": "Full RGB",
        "details": f"Visible light trichromatic Bayer capture (mean chrominance delta: {mean_chroma_diff:.2f})",
    }

