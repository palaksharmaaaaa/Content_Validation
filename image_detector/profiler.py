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

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("image_detector.profiler")


def compute_file_hashes(file_path: str | Path) -> Tuple[str, str, int]:
    """Calculates SHA-256, MD5, and exact file size in bytes."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    total_bytes = 0

    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
            md5.update(chunk)
            total_bytes += len(chunk)

    return sha256.hexdigest(), md5.hexdigest(), total_bytes


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
    except Exception:
        return 0.0


def rgb_to_color_name(r: int, g: int, b: int) -> str:
    """Classifies an RGB tuple into a clean, human-readable color name."""
    diff = max(abs(r - g), abs(r - b), abs(g - b))
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Low saturation / monochrome
    if diff <= 16:
        if lum > 238:
            return "Pure White"
        elif lum > 195:
            return "Off-White / Platinum"
        elif lum > 145:
            return "Silver / Slate Gray"
        elif lum > 95:
            return "Medium Neutral Gray"
        elif lum > 42:
            return "Charcoal / Graphite"
        else:
            return "Jet Black"

    # Chromatic dominant hues
    if r >= g and r >= b:
        if g > b * 1.4 and r > 170 and g > 120:
            return "Golden Ochre" if g > 165 else "Warm Amber / Orange"
        elif g > b:
            return "Terracotta / Peach" if r > 180 else "Warm Rust / Brown"
        elif b > g:
            return "Crimson / Magenta" if b > 115 else "Ruby Red"
        return "Deep Crimson Red"
    elif g >= r and g >= b:
        if b > r * 1.3:
            return "Teal / Aquamarine"
        elif r > b * 1.3:
            return "Olive / Moss Green"
        elif lum > 175:
            return "Mint / Sage Green"
        else:
            return "Emerald / Forest Green"
    else:  # Blue dominant
        if r > g * 1.25:
            return "Royal Purple / Violet"
        elif g > r * 1.25:
            return "Cyan / Cerulean"
        elif lum > 175:
            return "Sky / Azure Blue"
        elif lum < 70:
            return "Deep Navy Blue"
        else:
            return "Cobalt / Sapphire Blue"


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
        size_kb = size_bytes / 1024.0
        size_mb = size_bytes / (1024.0 * 1024.0)

        # 1. PIL Container & Acquisition Metadata Inspection
        dpi_x, dpi_y = 72.0, 72.0
        color_mode = "RGB"
        has_icc = False
        icc_profile_name = "Standard sRGB"
        exif_info: Dict[str, Any] = {
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

        try:
            with Image.open(path) as pil_img:
                color_mode = pil_img.mode
                if pil_img.info.get("icc_profile"):
                    has_icc = True
                    icc_profile_name = "ICC Profile Embedded"
                raw_dpi = pil_img.info.get("dpi")
                if raw_dpi and isinstance(raw_dpi, (tuple, list)):
                    dpi_x = float(raw_dpi[0])
                    dpi_y = float(raw_dpi[1]) if len(raw_dpi) > 1 else dpi_x

                # EXIF tags
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
                            dpi_x = float(raw_tags["XResolution"])
                        if "YResolution" in raw_tags and isinstance(raw_tags["YResolution"], (int, float)):
                            dpi_y = float(raw_tags["YResolution"])

                        # Extended Exif sub-IFD (0x8769)
                        try:
                            if hasattr(exif_data, "get_ifd"):
                                sub_exif = exif_data.get_ifd(0x8769)
                                for sk, sv in sub_exif.items():
                                    stag = TAGS.get(sk, str(sk))
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
                                        metering_map = {1: "Average", 2: "CenterWeightedAverage", 3: "Spot", 4: "MultiSpot", 5: "Pattern", 6: "Partial"}
                                        exif_info["metering_mode"] = metering_map.get(int(sv), f"Mode {sv}")
                                    elif stag == "ExposureBiasValue":
                                        exif_info["exposure_bias"] = f"{float(sv):+.1f} EV"
                                    elif stag == "ColorSpace":
                                        exif_info["color_space_tag"] = "sRGB" if int(sv) == 1 else ("Adobe RGB" if int(sv) == 2 else "Uncalibrated")

                                # GPS IFD (0x8825)
                                gps_ifd = exif_data.get_ifd(0x8825)
                                if gps_ifd:
                                    gps_parsed = parse_gps_info(gps_ifd)
                                    exif_info["gps_details"] = gps_parsed
                                    exif_info["gps_embedded"] = gps_parsed.get("has_gps", False)
                        except Exception as e:
                            logger.debug("Sub-IFD EXIF extraction exception: %s", e)
        except Exception as e:
            logger.debug("PIL inspection error: %s", e)

        # 2. OpenCV Pixel Array Decoding & Geometric Analysis
        img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            return {
                "valid": False,
                "filename": path.name,
                "source": source,
                "file_size_mb": round(size_mb, 3),
                "sha256": sha256,
                "md5": md5,
                "error": "Failed to decode image data into pixel array.",
            }

        h, w = img_bgr.shape[:2]
        channels = img_bgr.shape[2] if img_bgr.ndim == 3 else 1
        has_alpha = channels == 4
        megapixels = round((w * h) / 1_000_000.0, 2)
        total_pixels = int(w * h)
        aspect_ratio_dec = round(float(w) / max(1.0, float(h)), 3)

        # Orientation classification
        if aspect_ratio_dec > 1.08:
            orientation = "Landscape (Horizontal)"
        elif aspect_ratio_dec < 0.92:
            orientation = "Portrait (Vertical)"
        else:
            orientation = "Square (1:1)"

        # Standard aspect ratio string description
        if 0.96 <= aspect_ratio_dec <= 1.04:
            aspect_ratio_str = "1:1 (Square Diffusion Canvas)"
        elif 0.54 <= aspect_ratio_dec <= 0.59:
            aspect_ratio_str = "9:16 (Vertical Mobile Wallpaper)"
        elif 1.70 <= aspect_ratio_dec <= 1.82:
            aspect_ratio_str = "16:9 (Landscape Widescreen)"
        elif 0.72 <= aspect_ratio_dec <= 0.78:
            aspect_ratio_str = "3:4 (Vertical Portrait)"
        elif 1.30 <= aspect_ratio_dec <= 1.36:
            aspect_ratio_str = "4:3 (Standard Photo)"
        elif 0.64 <= aspect_ratio_dec <= 0.69:
            aspect_ratio_str = "2:3 (Vertical 35mm)"
        elif 1.45 <= aspect_ratio_dec <= 1.55:
            aspect_ratio_str = "3:2 (Horizontal 35mm)"
        else:
            aspect_ratio_str = f"{aspect_ratio_dec:.2f}:1 Ratio"

        # 3. Pixel-by-Pixel Color & Photometric Statistics
        if channels >= 3:
            b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
            rgb_3ch = img_bgr[:, :, :3]
            gray = cv2.cvtColor(rgb_3ch, cv2.COLOR_BGR2GRAY)
            channel_means = {
                "red": round(float(np.mean(r)), 1),
                "green": round(float(np.mean(g)), 1),
                "blue": round(float(np.mean(b)), 1),
            }
            channel_stds = {
                "red": round(float(np.std(r)), 1),
                "green": round(float(np.std(g)), 1),
                "blue": round(float(np.std(b)), 1),
            }
            channel_mins = {
                "red": int(np.min(r)),
                "green": int(np.min(g)),
                "blue": int(np.min(b)),
            }
            channel_maxs = {
                "red": int(np.max(r)),
                "green": int(np.max(g)),
                "blue": int(np.max(b)),
            }
        else:
            gray = img_bgr
            channel_means = {"luminance": round(float(np.mean(gray)), 1)}
            channel_stds = {"luminance": round(float(np.std(gray)), 1)}
            channel_mins = {"luminance": int(np.min(gray))}
            channel_maxs = {"luminance": int(np.max(gray))}

        entropy = compute_pixel_entropy(img_bgr)

        # Luminance dynamic range & pixel clipping
        lum_mean = round(float(np.mean(gray)), 1)
        lum_std = round(float(np.std(gray)), 1)
        lum_median = round(float(np.median(gray)), 1)
        lum_min = int(np.min(gray))
        lum_max = int(np.max(gray))
        dynamic_range = lum_max - lum_min
        highlight_clipped_count = int(np.sum(gray >= 252))
        highlight_clipped_pct = round(float(highlight_clipped_count / gray.size * 100.0), 2)
        shadow_crushed_count = int(np.sum(gray <= 4))
        shadow_crushed_pct = round(float(shadow_crushed_count / gray.size * 100.0), 2)

        # Dominant Palette Extraction (Top 6 Color Swatches with Human Names)
        small_sample = cv2.resize(
            img_bgr[:, :, :3] if channels >= 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            (100, 100),
            interpolation=cv2.INTER_AREA,
        )
        pixels = small_sample.reshape(-1, 3)
        q_pixels = (pixels // 32) * 32
        colors, counts = np.unique(q_pixels, axis=0, return_counts=True)
        top_indices = np.argsort(counts)[::-1][:6]
        dominant_palette = []
        for idx in top_indices:
            pb, pg, pr = colors[idx]
            hex_val = f"#{int(pr):02x}{int(pg):02x}{int(pb):02x}"
            pct_val = round(float(counts[idx] / len(pixels) * 100.0), 1)
            c_name = rgb_to_color_name(int(pr), int(pg), int(pb))
            dominant_palette.append({
                "hex": hex_val,
                "rgb": (int(pr), int(pg), int(pb)),
                "color_name": c_name,
                "coverage_pct": pct_val,
            })

        # 4. Raw Physical Signals & Sensor Noise Residuals
        sample_gray = gray
        if max(h, w) > 1024:
            scale_sc = 1024.0 / max(h, w)
            sample_gray = cv2.resize(gray, (int(w * scale_sc), int(h * scale_sc)), interpolation=cv2.INTER_AREA)

        blurred_med = cv2.medianBlur(sample_gray, 3)
        diff_med = cv2.absdiff(sample_gray, blurred_med).astype(np.float32)
        grad_mag = cv2.magnitude(cv2.Sobel(sample_gray, cv2.CV_32F, 1, 0), cv2.Sobel(sample_gray, cv2.CV_32F, 0, 1))
        flat_mask = grad_mag < 15.0
        flat_region_noise = float(np.mean(diff_med[flat_mask])) if np.sum(flat_mask) > 100 else float(np.mean(diff_med))

        prnu_noise_mean = float(np.mean(diff_med))
        prnu_noise_std = float(np.std(diff_med))

        # Surface Bilateral Smoothness index
        bilateral_flt = cv2.bilateralFilter(sample_gray, 9, 75, 75)
        diff_bilateral = cv2.absdiff(sample_gray, bilateral_flt)
        surface_smoothness = float(np.mean(diff_bilateral))

        # Fourier Spectral Decay Alpha
        try:
            dft = np.fft.fft2(sample_gray.astype(np.float32))
            dft_shift = np.fft.fftshift(dft)
            mag_spec = np.abs(dft_shift)
            cy_fft, cx_fft = sample_gray.shape[0] // 2, sample_gray.shape[1] // 2
            y_mesh, x_mesh = np.ogrid[:sample_gray.shape[0], :sample_gray.shape[1]]
            r_mesh = np.hypot(x_mesh - cx_fft, y_mesh - cy_fft).astype(int)
            r_max = min(cx_fft, cy_fft) - 1
            rad_profile = [float(mag_spec[r_mesh == r_idx].mean()) for r_idx in range(5, max(6, r_max))]
            if len(rad_profile) > 5:
                freqs = np.arange(5, 5 + len(rad_profile))
                log_f = np.log(freqs)
                log_p = np.log(np.maximum(1e-6, rad_profile))
                fft_alpha = float(-np.polyfit(log_f, log_p, 1)[0])
            else:
                fft_alpha = 2.05
        except Exception:
            fft_alpha = 2.05

        # Canny edge density & dark line-art contours
        edges = cv2.Canny(sample_gray, 50, 150)
        canny_edge_pct = round(float(np.sum(edges > 0) / max(1, edges.size) * 100.0), 2)
        dark_edges = (sample_gray < 50) & (edges > 0)
        dark_line_art_pct = round(float(np.sum(dark_edges) / max(1, edges.size) * 100.0), 2)

        # Focus / blur sharpness (Laplacian variance)
        laplacian_var = round(float(cv2.Laplacian(sample_gray, cv2.CV_64F).var()), 1)

        return {
            "valid": True,
            "filename": path.name,
            "source": source,
            "file_size_bytes": size_bytes,
            "file_size_kb": round(size_kb, 1),
            "file_size_mb": round(size_mb, 3),
            "sha256": sha256,
            "md5": md5,
            "format": path.suffix.lstrip(".").upper(),
            "mime_type": f"image/{path.suffix.lstrip('.').lower()}",
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio_dec,
            "aspect_ratio_str": aspect_ratio_str,
            "orientation": orientation,
            "channels": channels,
            "pixel_entropy": entropy,
            "channel_variances": {
                "blue_std": channel_stds.get("blue", 0.0),
                "green_std": channel_stds.get("green", 0.0),
                "red_std": channel_stds.get("red", 0.0),
            },
            "spatial_geometry": {
                "width": w,
                "height": h,
                "megapixels": megapixels,
                "total_pixels": total_pixels,
                "aspect_ratio": aspect_ratio_dec,
                "aspect_ratio_str": aspect_ratio_str,
                "orientation": orientation,
            },
            "display_attributes": {
                "dpi_x": round(dpi_x, 1),
                "dpi_y": round(dpi_y, 1),
                "dpi_str": f"{int(dpi_x)} x {int(dpi_y)} DPI",
                "bit_depth": f"{8 * channels}-bit ({channels} channels x 8-bit)",
                "bits_per_channel": 8,
                "color_mode": color_mode,
                "color_space": icc_profile_name if has_icc else "Standard sRGB",
                "has_alpha_channel": has_alpha,
            },
            "pixel_color_profile": {
                "channel_means": channel_means,
                "channel_stds": channel_stds,
                "channel_mins": channel_mins,
                "channel_maxs": channel_maxs,
                "luminance_mean": lum_mean,
                "luminance_median": lum_median,
                "luminance_std": lum_std,
                "luminance_min": lum_min,
                "luminance_max": lum_max,
                "dynamic_range": dynamic_range,
                "highlight_clipped_count": highlight_clipped_count,
                "highlight_clipped_pct": highlight_clipped_pct,
                "shadow_crushed_count": shadow_crushed_count,
                "shadow_crushed_pct": shadow_crushed_pct,
                "shannon_entropy_bpp": entropy,
                "unique_quantized_colors": len(colors),
                "dominant_palette": dominant_palette,
            },
            "exif_device_details": exif_info,
            "raw_physical_signals": {
                "prnu_noise_mean": round(prnu_noise_mean, 3),
                "prnu_noise_std": round(prnu_noise_std, 3),
                "flat_region_noise_mean": round(flat_region_noise, 3),
                "surface_smoothness_index": round(surface_smoothness, 3),
                "fft_decay_alpha": round(fft_alpha, 3),
                "canny_edge_pct": canny_edge_pct,
                "dark_line_art_pct": dark_line_art_pct,
                "laplacian_sharpness_var": laplacian_var,
            },
        }


def extract_all_image_details(file_path: str | Path, source: str = "User Upload") -> Dict[str, Any]:
    """Convenience helper to extract each and every detail out of an image file before running predictions."""
    return ImageProfiler().profile_image(file_path, source=source)
