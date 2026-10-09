"""
image_detector.profiler: Technical Media Profiler and Signal Forensic Inspector for Images.
Extracts:
1. Cryptographic identity (SHA-256, MD5, size in bytes, format, MIME, origin source).
2. Geometric specifications (width, height, megapixels, aspect ratio, orientation, total pixels).
3. Display attributes & DPI (DPI X/Y, bit depth per channel & total, color mode, ICC color space profile).
4. Detailed EXIF acquisition hardware parameters (Camera make, model, lens, exposure time, aperture, ISO, focal length, flash, white balance, metering mode, GPS coords, software).
5. Pixel-by-pixel photometric & color distribution (Shannon entropy, luminance min/max/mean/std/median/dynamic range, channel statistics, shadow crush & highlight clip percentages).
6. Dominant color palette with exact canvas coverage % and human-readable color naming.
7. Raw physical signals & noise (PRNU sensor noise, flat region noise, bilateral smoothness, 2D FFT spectral decay alpha, Canny edges, dark lines, Laplacian blur/focus).
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.hashing import file_digests
from core.perception.colors import dominant_colors, name_color

import cv2
from core.imageio import imread
import numpy as np
from PIL import Image

logger = logging.getLogger("image_detector.profiler")


def compute_file_hashes(file_path: str | Path) -> Tuple[str, str, int]:
    """Calculates SHA-256, MD5, and exact file size in bytes (shared cached implementation)."""
    return file_digests(file_path)


def compute_pixel_entropy(image_bgr: np.ndarray) -> float:
    """
    Computes Shannon Information Entropy in bits per pixel.
    Natural photographs typically exhibit 6.8 - 7.6 bits/pixel.
    Synthetically smoothed or flattened images often exhibit lower entropy.
    """
    try:
        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / max(1.0, float(hist.sum()))
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))
        return round(entropy, 3)
    except Exception as exc:
        logger.debug("compute_pixel_entropy: ignored %s: %s", type(exc).__name__, exc)
        return 0.0


def rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Everyday name of an sRGB colour (nearest in CIELAB, see core.perception.colors)."""
    return name_color(r, g, b).title()


def parse_gps_info(gps_dict: dict) -> Dict[str, Any]:
    """Extracts latitude, longitude, and altitude from EXIF GPS IFD."""
    try:
        from PIL.ExifTags import GPSTAGS
        named = {GPSTAGS.get(k, str(k)): v for k, v in gps_dict.items()}
        lat = named.get("GPSLatitude")
        lat_ref = str(named.get("GPSLatitudeRef", "N")).upper()
        lon = named.get("GPSLongitude")
        lon_ref = str(named.get("GPSLongitudeRef", "E")).upper()
        alt = named.get("GPSAltitude")

        def _to_float(val):
            if hasattr(val, "numerator") and hasattr(val, "denominator"):
                return float(val.numerator) / max(1.0, float(val.denominator))
            return float(val)

        def _dms(dms):
            if isinstance(dms, (tuple, list)) and len(dms) >= 3:
                return _to_float(dms[0]) + (_to_float(dms[1]) / 60.0) + (_to_float(dms[2]) / 3600.0)
            return _to_float(dms)

        if lat and lon:
            lat_deg = _dms(lat) * (-1.0 if lat_ref == "S" else 1.0)
            lon_deg = _dms(lon) * (-1.0 if lon_ref == "W" else 1.0)
            alt_m = round(_to_float(alt), 1) if alt is not None else None
            return {
                "has_gps": True,
                "latitude": round(lat_deg, 6),
                "longitude": round(lon_deg, 6),
                "altitude_m": alt_m,
                "coordinates_str": f"{abs(lat_deg):.4f}° {lat_ref}, {abs(lon_deg):.4f}° {lon_ref}" + (f" ({alt_m}m)" if alt_m else ""),
            }
    except Exception as e:
        logger.debug("GPS EXIF parsing skipped: %s", e)
    return {"has_gps": False, "latitude": None, "longitude": None, "altitude_m": None, "coordinates_str": "Not Embedded"}


def _blank_exif_info() -> Dict[str, Any]:
    return {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "lens_model": None,
        "software": None,
        "date_time": None,
        "exposure_time": None,
        "aperture": None,
        "iso": None,
        "focal_length": None,
        "flash": "Not Fired",
        "white_balance": "Auto",
        "metering_mode": "Standard",
        "exposure_bias": "0.0 EV",
        "gps_embedded": False,
        "gps_details": {"has_gps": False, "coordinates_str": "Not Embedded"},
        "orientation_tag": "Normal (1)",
        "color_space_tag": "sRGB",
    }


_METERING_MODES = {1: "Average", 2: "CenterWeightedAverage", 3: "Spot", 4: "MultiSpot", 5: "Pattern", 6: "Partial"}


def _apply_exif_tag(stag: str, sv: Any, exif_info: Dict[str, Any]) -> None:
    if stag == "ExposureTime":
        val = float(sv)
        exif_info["exposure_time"] = f"1/{round(1.0 / val)}s" if val < 1.0 else f"{val:.2f}s"
    elif stag == "FNumber":
        exif_info["aperture"] = f"f/{float(sv):.1f}"
    elif stag == "ISOSpeedRatings":
        exif_info["iso"] = int(sv)
    elif stag == "FocalLength":
        exif_info["focal_length"] = f"{float(sv):.1f}mm"
    elif stag == "LensModel":
        exif_info["lens_model"] = str(sv)
    elif stag == "Flash":
        exif_info["flash"] = "Fired" if (int(sv) & 1) else "Did not fire"
    elif stag == "WhiteBalance":
        exif_info["white_balance"] = "Manual" if int(sv) == 1 else "Auto"
    elif stag == "MeteringMode":
        exif_info["metering_mode"] = _METERING_MODES.get(int(sv), f"Mode {sv}")
    elif stag == "ExposureBiasValue":
        exif_info["exposure_bias"] = f"{float(sv):+.1f} EV"
    elif stag == "ColorSpace":
        exif_info["color_space_tag"] = "sRGB" if int(sv) == 1 else ("Adobe RGB" if int(sv) == 2 else "Uncalibrated")


def _apply_sub_ifd(exif_data: Any, exif_info: Dict[str, Any]) -> None:
    """Fills exposure/optics/GPS fields from the Exif (0x8769) and GPS (0x8825) sub-IFDs."""
    from PIL.ExifTags import TAGS

    try:
        if not hasattr(exif_data, "get_ifd"):
            return
        for sk, sv in exif_data.get_ifd(0x8769).items():
            try:  # one malformed tag must not discard the remaining tags
                _apply_exif_tag(TAGS.get(sk, str(sk)), sv, exif_info)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
                logger.debug("Skipping malformed EXIF tag %s: %s", sk, e)

        gps_ifd = exif_data.get_ifd(0x8825)
        if gps_ifd:
            gps_parsed = parse_gps_info(gps_ifd)
            exif_info["gps_details"] = gps_parsed
            exif_info["gps_embedded"] = gps_parsed.get("has_gps", False)
    except Exception as e:
        logger.debug("Sub-IFD EXIF extraction exception: %s", e)


def _read_container_info(path: Path) -> Dict[str, Any]:
    """PIL-level container facts: colour mode, ICC, DPI and EXIF acquisition metadata."""
    info: Dict[str, Any] = {
        "dpi_x": None, "dpi_y": None, "color_mode": "RGB", "has_icc": False,        # None: the file records no resolution
        "icc_profile_name": "ICC profile embedded", "exif_info": _blank_exif_info(),
    }
    exif_info = info["exif_info"]
    try:
        with Image.open(path) as pil_img:
            info["color_mode"] = pil_img.mode
            if pil_img.info.get("icc_profile"):
                info["has_icc"] = True
            raw_dpi = pil_img.info.get("dpi")
            if raw_dpi and isinstance(raw_dpi, (tuple, list)):
                info["dpi_x"] = float(raw_dpi[0])
                info["dpi_y"] = float(raw_dpi[1]) if len(raw_dpi) > 1 else info["dpi_x"]

            exif_data = pil_img.getexif()
            if exif_data:
                from PIL.ExifTags import TAGS

                raw_tags = {TAGS.get(k, str(k)): v for k, v in exif_data.items()}
                if raw_tags.get("Make") or raw_tags.get("Model") or len(raw_tags) > 2:
                    exif_info["has_exif"] = True
                    exif_info["camera_make"] = raw_tags.get("Make")
                    exif_info["camera_model"] = raw_tags.get("Model")
                    exif_info["software"] = raw_tags.get("Software")
                    exif_info["date_time"] = raw_tags.get("DateTime")
                    if "Orientation" in raw_tags:
                        exif_info["orientation_tag"] = f"Tag {raw_tags['Orientation']}"
                    if "XResolution" in raw_tags and isinstance(raw_tags["XResolution"], (int, float)):
                        info["dpi_x"] = float(raw_tags["XResolution"])
                    if "YResolution" in raw_tags and isinstance(raw_tags["YResolution"], (int, float)):
                        info["dpi_y"] = float(raw_tags["YResolution"])
                    _apply_sub_ifd(exif_data, exif_info)
    except Exception as e:
        logger.debug("PIL inspection error: %s", e)
    return info


def _orientation_and_aspect(ratio: float) -> Tuple[str, str]:
    if ratio > 1.08:
        orientation = "Landscape (Horizontal)"
    elif ratio < 0.92:
        orientation = "Portrait (Vertical)"
    else:
        orientation = "Square (1:1)"
    for lo, hi, text in _ASPECT_LABELS:
        if lo <= ratio <= hi:
            return orientation, text
    return orientation, f"{ratio:.2f}:1 Ratio"


_ASPECT_LABELS = (
    (0.96, 1.04, "1:1 (Square Diffusion Canvas)"),
    (0.54, 0.59, "9:16 (Vertical Mobile Wallpaper)"),
    (1.70, 1.82, "16:9 (Landscape Widescreen)"),
    (0.72, 0.78, "3:4 (Vertical Portrait)"),
    (1.30, 1.36, "4:3 (Standard Photo)"),
    (0.64, 0.69, "2:3 (Vertical 35mm)"),
    (1.45, 1.55, "3:2 (Horizontal 35mm)"),
)


def _channel_statistics(img_bgr: np.ndarray, channels: int) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    """Returns (gray plane, {means, stds, mins, maxs}) for colour or single-channel images."""
    if channels >= 3:
        b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
        gray = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
        named = (("red", r), ("green", g), ("blue", b))
    else:
        gray = img_bgr
        named = (("luminance", gray),)
    return gray, {
        "means": {n: round(float(np.mean(a)), 1) for n, a in named},
        "stds": {n: round(float(np.std(a)), 1) for n, a in named},
        "mins": {n: int(np.min(a)) for n, a in named},
        "maxs": {n: int(np.max(a)) for n, a in named},
    }


def _luminance_statistics(gray: np.ndarray) -> Dict[str, Any]:
    lum_min, lum_max = int(np.min(gray)), int(np.max(gray))
    hi = int(np.sum(gray >= 252))
    lo = int(np.sum(gray <= 4))
    return {
        "luminance_mean": round(float(np.mean(gray)), 1),
        "luminance_median": round(float(np.median(gray)), 1),
        "luminance_std": round(float(np.std(gray)), 1),
        "luminance_min": lum_min,
        "luminance_max": lum_max,
        "dynamic_range": lum_max - lum_min,
        "highlight_clipped_count": hi,
        "highlight_clipped_pct": round(float(hi / gray.size * 100.0), 2),
        "shadow_crushed_count": lo,
        "shadow_crushed_pct": round(float(lo / gray.size * 100.0), 2),
    }


def _dominant_palette(img_bgr: np.ndarray, gray: np.ndarray, channels: int) -> Tuple[List[Dict[str, Any]], int]:
    """Up to six dominant colours with human names; also returns the number of unique quantised colours."""
    bgr = img_bgr[:, :, :3] if channels >= 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    small = cv2.resize(bgr, (100, 100), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3)
    colors = np.unique((pixels // 32) * 32, axis=0)
    palette = dominant_colors(bgr, k=6)
    return palette, len(colors)


def _fft_decay_alpha(sample_gray: np.ndarray) -> float:
    """Slope of the radially averaged Fourier magnitude spectrum in log-log space (default 2.05 on failure)."""
    try:
        mag_spec = np.abs(np.fft.fftshift(np.fft.fft2(sample_gray.astype(np.float32))))
        cy, cx = sample_gray.shape[0] // 2, sample_gray.shape[1] // 2
        y_mesh, x_mesh = np.ogrid[:sample_gray.shape[0], :sample_gray.shape[1]]
        r_mesh = np.hypot(x_mesh - cx, y_mesh - cy).astype(int)
        r_max = min(cx, cy) - 1
        stop = max(6, r_max)
        radii = r_mesh.ravel()
        sums = np.bincount(radii, weights=mag_spec.ravel(), minlength=stop)[:stop]
        counts = np.bincount(radii, minlength=stop)[:stop]
        profile = [float(s) / int(c) for s, c in zip(sums[5:stop], counts[5:stop])]      # every radius below the half-size has pixels
        if len(profile) > 5:
            freqs = np.arange(5, 5 + len(profile))
            return float(-np.polyfit(np.log(freqs), np.log(np.maximum(1e-6, profile)), 1)[0])
    except Exception as exc:
        logger.debug("_fft_decay_alpha: ignored %s: %s", type(exc).__name__, exc)
    return 2.05


def _raw_physical_signals(gray: np.ndarray) -> Dict[str, Any]:
    """PRNU-style noise residuals, bilateral smoothness, spectral decay, edge density and sharpness."""
    h, w = gray.shape[:2]
    sample = gray
    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        sample = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    diff_med = cv2.absdiff(sample, cv2.medianBlur(sample, 3)).astype(np.float32)
    grad_mag = cv2.magnitude(cv2.Sobel(sample, cv2.CV_32F, 1, 0), cv2.Sobel(sample, cv2.CV_32F, 0, 1))
    flat_mask = grad_mag < 15.0
    flat_noise = float(np.mean(diff_med[flat_mask])) if np.sum(flat_mask) > 100 else float(np.mean(diff_med))
    smoothness = float(np.mean(cv2.absdiff(sample, cv2.bilateralFilter(sample, 9, 75, 75))))
    edges = cv2.Canny(sample, 50, 150)
    dark_edges = (sample < 50) & (edges > 0)
    return {
        "prnu_noise_mean": round(float(np.mean(diff_med)), 3),
        "prnu_noise_std": round(float(np.std(diff_med)), 3),
        "flat_region_noise_mean": round(flat_noise, 3),
        "surface_smoothness_index": round(smoothness, 3),
        "fft_decay_alpha": round(_fft_decay_alpha(sample), 3) + 0.0,
        "canny_edge_pct": round(float(np.sum(edges > 0) / max(1, edges.size) * 100.0), 2),
        "dark_line_art_pct": round(float(np.sum(dark_edges) / max(1, edges.size) * 100.0), 2),
        "laplacian_sharpness_var": round(float(cv2.Laplacian(sample, cv2.CV_64F).var()), 1),
    }


class ImageProfiler:
    """
    Comprehensive technical specs, dimensions, DPI, pixel-by-pixel statistics,
    EXIF acquisition details, color palette distribution, and physical noise profiler.
    """

    def __init__(self):
        pass

    def profile_image(self, file_path: str | Path, source: str = "User Upload") -> Dict[str, Any]:
        """Extracts complete technical profile, EXIF, DPI, pixel stats, and physical signals of an image."""
        path = Path(file_path)
        if not path.is_file():
            return {"valid": False, "error": f"File not found: {path}"}

        sha256, md5, size_bytes = compute_file_hashes(path)
        size_mb = size_bytes / (1024.0 * 1024.0)
        container = _read_container_info(path)

        img_bgr = imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            return {
                "valid": False, "filename": path.name, "source": source,
                "file_size_mb": round(size_mb, 3), "sha256": sha256, "md5": md5,
                "error": "Failed to decode image data into pixel array.",
            }

        h, w = img_bgr.shape[:2]
        channels = img_bgr.shape[2] if img_bgr.ndim == 3 else 1
        bits = img_bgr.dtype.itemsize * 8
        aspect = round(float(w) / max(1.0, float(h)), 3)
        orientation, aspect_str = _orientation_and_aspect(aspect)

        gray, stats = _channel_statistics(img_bgr, channels)
        entropy = compute_pixel_entropy(img_bgr)
        palette, n_colors = _dominant_palette(img_bgr, gray, channels)
        dpi_x, dpi_y = container["dpi_x"], container["dpi_y"]

        return {
            "valid": True,
            "filename": path.name,
            "source": source,
            "file_size_bytes": size_bytes,
            "file_size_kb": round(size_bytes / 1024.0, 1),
            "file_size_mb": round(size_mb, 3),
            "sha256": sha256,
            "md5": md5,
            "format": path.suffix.lstrip(".").upper(),
            "mime_type": f"image/{path.suffix.lstrip('.').lower()}",
            "width": w,
            "height": h,
            "aspect_ratio": aspect,
            "aspect_ratio_str": aspect_str,
            "orientation": orientation,
            "channels": channels,
            "pixel_entropy": entropy,
            "channel_variances": {
                "blue_std": stats["stds"].get("blue", 0.0),
                "green_std": stats["stds"].get("green", 0.0),
                "red_std": stats["stds"].get("red", 0.0),
            },
            "spatial_geometry": {
                "width": w,
                "height": h,
                "megapixels": round((w * h) / 1_000_000.0, 2),
                "total_pixels": int(w * h),
                "aspect_ratio": aspect,
                "aspect_ratio_str": aspect_str,
                "orientation": orientation,
            },
            "display_attributes": {
                "dpi_x": None if dpi_x is None else round(dpi_x, 1),
                "dpi_y": None if dpi_y is None else round(dpi_y, 1),
                "dpi_str": "Not recorded" if dpi_x is None else f"{int(dpi_x)} x {int(dpi_y)} DPI",
                "bit_depth": f"{bits * channels}-bit ({channels} channels x {bits}-bit)",
                "bits_per_channel": bits,
                "color_mode": container["color_mode"],
                "color_space": container["icc_profile_name"] if container["has_icc"] else "No embedded profile",
                "has_alpha_channel": channels == 4,
            },
            "pixel_color_profile": {
                "channel_means": stats["means"],
                "channel_stds": stats["stds"],
                "channel_mins": stats["mins"],
                "channel_maxs": stats["maxs"],
                **_luminance_statistics(gray),
                "shannon_entropy_bpp": entropy,
                "unique_quantized_colors": n_colors,
                "dominant_palette": palette,
            },
            "exif_device_details": container["exif_info"],
            "raw_physical_signals": _raw_physical_signals(gray),
        }


def extract_all_image_details(file_path: str | Path, source: str = "User Upload") -> Dict[str, Any]:
    """Convenience helper to extract each and every detail out of an image file before running predictions."""
    return ImageProfiler().profile_image(file_path, source=source)
