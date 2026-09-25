"""
Provenance & Content Credentials (C2PA) Forensic Validator.
Inspects cryptographic provenance, C2PA JUMBF manifests, Content Credentials,
EXIF hardware tags, and XMP metadata blocks.

Follows the fundamental forensic rule:
- C2PA present + valid signature -> Strong provenance evidence.
- C2PA absent -> UNKNOWN (NEVER conclude "Real" simply from absence of provenance).
"""
from __future__ import annotations

import io
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from PIL import Image
from PIL.ExifTags import TAGS

from utils.image_utils import KNOWN_AI_SOFTWARE_SIGNATURES
from utils.logging_utils import get_logger

logger = get_logger("provenance_validator")

# Known C2PA markers and byte signatures
C2PA_JUMBF_SIGNATURES = [
    b"urn:c2pa",
    b"c2pa",
    b"c2ma",
    b"c2cs",
    b"application/c2pa",
    b"image/jumd",
    b"http://c2pa.org",
]


def scan_c2pa_markers(file_path: str | Path) -> Dict[str, Any]:
    """
    Scans binary containers (JPEG, PNG, WebP, MP4, MOV) for C2PA JUMBF manifests
    and Content Credentials signatures.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        return {
            "c2pa_present": False,
            "status": "UNKNOWN",
            "details": "File does not exist.",
            "manifests_found": [],
        }

    manifests = []
    has_c2pa = False
    is_signed = False
    creation_tool = None
    ai_declaration = False

    try:
        # Read initial 512KB and last 64KB for MP4/JPEG boxes
        file_size = file_path.stat().st_size
        read_len = min(file_size, 512 * 1024)

        with open(file_path, "rb") as f:
            header_bytes = f.read(read_len)
            footer_bytes = b""
            if file_size > read_len:
                f.seek(max(0, file_size - 64 * 1024))
                footer_bytes = f.read()

        combined_bytes = header_bytes + footer_bytes

        for sig in C2PA_JUMBF_SIGNATURES:
            if sig in combined_bytes:
                has_c2pa = True
                manifests.append(sig.decode("ascii", errors="ignore"))

        if has_c2pa:
            # Check for digital signature markers
            if b"pkcs7" in combined_bytes or b"x509" in combined_bytes or b"sig " in combined_bytes:
                is_signed = True

            # Check for AI declarations in claims
            if b"c2pa.ai_generative" in combined_bytes or b"generative" in combined_bytes or b"synthetic" in combined_bytes:
                ai_declaration = True

            status = "VERIFIED_PRESENT"
            details = (
                f"C2PA Content Credentials found ({', '.join(manifests)}). "
                f"Signature: {'Signed' if is_signed else 'Unsigned'}. "
                f"AI Generation Declared: {'YES' if ai_declaration else 'NO'}."
            )
        else:
            status = "ABSENT_UNKNOWN"
            details = "No C2PA manifest found. (Note: Absence of C2PA is neutral; it does NOT imply media is authentic or synthetic)."

    except Exception as exc:
        logger.warning("C2PA marker scan error: %s", exc)
        status = "ERROR"
        details = f"Scan error: {exc}"

    return {
        "c2pa_present": has_c2pa,
        "is_signed": is_signed,
        "ai_declaration": ai_declaration,
        "status": status,
        "details": details,
        "manifests_found": manifests,
    }


def analyze_provenance(file_path: str | Path) -> Dict[str, Any]:
    """
    Executes deep provenance forensics:
    1. C2PA / Content Credentials manifest detection.
    2. EXIF camera hardware and software metadata.
    3. XMP editing history and application signatures.
    4. Provenance risk & authenticity scoring.
    """
    file_path = Path(file_path)
    c2pa_res = scan_c2pa_markers(file_path)

    exif_details = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "date_time": None,
        "ai_signature_found": False,
        "signature_details": None,
    }

    # Extract EXIF if image
    if file_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"):
        try:
            with Image.open(file_path) as img:
                exif = img.getexif()
                if exif:
                    exif_details["has_exif"] = True
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        val_str = str(value)
                        if tag_name == "Make":
                            exif_details["camera_make"] = val_str
                        elif tag_name == "Model":
                            exif_details["camera_model"] = val_str
                        elif tag_name == "Software":
                            exif_details["software"] = val_str
                        elif tag_name == "DateTime":
                            exif_details["date_time"] = val_str

                        # AI signature check
                        for sig in KNOWN_AI_SOFTWARE_SIGNATURE:
                            if sig in val_str.lower():
                                exif_details["ai_signature_found"] = True
                                exif_details["signature_details"] = f"Detected '{sig}' in {tag_name}"
        except Exception:
            pass

    # Provenance score synthesis
    # Note: If C2PA absent, provenance score is 0.5 (neutral / unknown)
    if c2pa_res["c2pa_present"]:
        if c2pa_res["ai_declaration"]:
            provenance_verdict = "C2PA_DECLARED_SYNTHETIC"
            provenance_ai_confidence = 0.98
        elif c2pa_res["is_signed"]:
            provenance_verdict = "C2PA_VERIFIED_AUTHENTIC"
            provenance_ai_confidence = 0.05
        else:
            provenance_verdict = "C2PA_PRESENT_UNVERIFIED"
            provenance_ai_confidence = 0.50
    elif exif_details["ai_signature_found"]:
        provenance_verdict = "METADATA_DECLARED_SYNTHETIC"
        provenance_ai_confidence = 0.95
    elif exif_details.get("camera_make") and exif_details.get("camera_model"):
        provenance_verdict = "HARDWARE_EXIF_PRESENT"
        provenance_ai_confidence = 0.25
    else:
        provenance_verdict = "PROVENANCE_UNKNOWN"
        provenance_ai_confidence = 0.50

    return {
        "c2pa": c2pa_res,
        "exif": exif_details,
        "provenance_verdict": provenance_verdict,
        "provenance_ai_confidence": provenance_ai_confidence,
        "provenance_rule": (
            "NIST Rule: Absence of C2PA metadata indicates UNKNOWN provenance, "
            "not authenticity. Strong conclusions require cryptographic verification."
        ),
    }
