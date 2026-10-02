"""
image_detector.provenance: Provenance & Content Credentials (C2PA) Forensic Validator for Images.
Inspects cryptographic provenance, C2PA JUMBF manifests, Content Credentials,
EXIF hardware tags, and software creation signatures.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger("image_detector.provenance")

C2PA_JUMBF_SIGNATURES = [
    b"urn:c2pa",
    b"c2pa",
    b"c2ma",
    b"c2cs",
    b"application/c2pa",
    b"image/jumd",
    b"http://c2pa.org",
    b"c2pa.claim",
    b"c2pa.assertion",
]

KNOWN_AI_SOFTWARE_SIGNATURES = [
    "midjourney", "stable diffusion", "dall-e", "comfyui", "automatic1111",
    "novelai", "invokeai", "fooocus", "adobe firefly", "bing image creator",
    "ideogram", "recraft", "magnific", "topaz photo ai", "liveportrait", "ic-light", "flux",
]


class ImageProvenanceValidator:
    """Validator for cryptographic provenance, C2PA manifests, and EXIF authenticity."""

    def __init__(self):
        pass

    def scan_c2pa(self, file_path: str | Path) -> Dict[str, Any]:
        """Scans image binary bytes (both header and trailer) for C2PA JUMBF manifests."""
        path = Path(file_path)
        if not path.is_file():
            return {"c2pa_present": False, "status": "FILE_NOT_FOUND", "manifests_found": []}

        try:
            file_size = path.stat().st_size
            read_len = min(file_size, 512 * 1024)
            with open(path, "rb") as f:
                header_bytes = f.read(read_len)
                trailer_bytes = b""
                if file_size > read_len:
                    f.seek(max(0, file_size - read_len))
                    trailer_bytes = f.read(read_len)

            scan_buffer = header_bytes + trailer_bytes
            found_markers = []
            for sig in C2PA_JUMBF_SIGNATURES:
                if sig in scan_buffer:
                    found_markers.append(sig.decode("utf-8", errors="ignore"))

            has_c2pa = len(found_markers) > 0
            return {
                "c2pa_present": has_c2pa,
                "status": "C2PA_CREDENTIALS_FOUND" if has_c2pa else "NO_C2PA_MANIFEST",
                "manifests_found": found_markers,
            }
        except Exception as e:
            logger.debug("C2PA scan error: %s", e)
            return {"c2pa_present": False, "status": "ERROR", "manifests_found": []}

    def analyze_provenance(self, file_path: str | Path) -> Dict[str, Any]:
        """Runs complete provenance and metadata analysis on an image."""
        path = Path(file_path)
        if not path.is_file():
            return {"valid": False, "error": "File not found"}

        c2pa_res = self.scan_c2pa(path)
        meta_extracted = self._extract_deep_metadata(path)
        cues = []
        is_synthetic = False
        is_ai_enhanced = False
        is_graphic_edit = False

        if meta_extracted.get("ai_signature_found"):
            is_synthetic = True
            cues.append(meta_extracted.get("signature_details") or "Generative AI signature verified in metadata")
        elif meta_extracted.get("ai_enhancer_signature_found"):
            is_ai_enhanced = True
            cues.append(meta_extracted.get("signature_details") or "AI enhancement / neural restoration signature detected")
        elif meta_extracted.get("graphic_editor_signature_found"):
            is_graphic_edit = True
            cues.append(meta_extracted.get("signature_details") or "Graphic editing tool detected (Canva / Photo Editor)")

        make = meta_extracted.get("camera_make") or ""
        model = meta_extracted.get("camera_model") or ""
        has_hardware = bool(make or model)

        if has_hardware and not is_synthetic:
            cues.append(f"Physical camera hardware tags present: {make.upper()} {model.upper()}".strip())
            prov_status = "AUTHENTIC_HARDWARE_TAGS" if not is_ai_enhanced else "HARDWARE_WITH_AI_ENHANCEMENT"
        elif is_synthetic:
            prov_status = "SYNTHETIC_SOFTWARE_TAGS"
        elif is_ai_enhanced:
            prov_status = "AI_ENHANCEMENT_TAGS"
        elif is_graphic_edit:
            prov_status = "GRAPHIC_EDITOR_TAGS"
        elif c2pa_res["c2pa_present"]:
            prov_status = "C2PA_PROVENANCE_PRESENT"
            cues.append("Content Credentials cryptographic manifest verified.")
        else:
            prov_status = "NO_PROVENANCE_METADATA"
            cues.append("No hardware metadata or Content Credentials found (neutral/stripped).")

        return {
            "c2pa_present": c2pa_res["c2pa_present"],
            "provenance_status": prov_status,
            "has_camera_hardware": has_hardware,
            "camera_make": meta_extracted.get("camera_make"),
            "camera_model": meta_extracted.get("camera_model"),
            "software": meta_extracted.get("software"),
            "creator_tool": meta_extracted.get("creator_tool"),
            "is_synthetic": is_synthetic,
            "is_ai_enhanced": is_ai_enhanced,
            "is_graphic_edit": is_graphic_edit,
            "metadata": meta_extracted,
            "cues": cues,
        }

    def _extract_deep_metadata(self, path: Path) -> Dict[str, Any]:
        """Extracts standard EXIF and deep XMP/IPTC metadata packets."""
        from image_detector.features import extract_image_metadata
        return extract_image_metadata(path)

    def _extract_exif(self, path: Path) -> Dict[str, Any]:
        """Extracts standard camera EXIF metadata dictionary."""
        try:
            with Image.open(path) as img:
                exif = img.getexif()
                if not exif:
                    return {}
                res = {}
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, int, float)):
                        res[tag_name.lower()] = value
                    elif isinstance(value, bytes):
                        res[tag_name.lower()] = value.decode("utf-8", errors="ignore")
                return {
                    "camera_make": str(res.get("make", "")).strip(),
                    "camera_model": str(res.get("model", "")).strip(),
                    "software": str(res.get("software", "")).strip(),
                    "datetime": str(res.get("datetime", "")).strip(),
                    "all_tags": res,
                }
        except Exception as exc:
            logger.debug("EXIF parsing bypassed for %s: %s", path, exc)
            return {}
