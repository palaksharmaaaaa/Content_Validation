"""
core.provenance_view: the single nested provenance shape consumed by the decision layer and the UI.

Each modality package keeps its own scanners (they know their containers); this module only turns their
findings into the shared view: ``c2pa`` / ``exif`` blocks plus a rule-based verdict. C2PA is marker PRESENCE
only: no certificate chain or hash binding is validated anywhere, so nothing here is "verified".
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

PROVENANCE_RULE = (
    "NIST Rule: Absence of C2PA metadata indicates UNKNOWN provenance, "
    "not authenticity. Strong conclusions require cryptographic verification."
)


def build_c2pa_block(present: bool, manifests_found: Iterable[str], ai_declaration: bool = False) -> Dict[str, Any]:
    """The single C2PA block shape shared by all modalities. Reports marker presence only; never 'verified'."""
    manifests = list(manifests_found)
    is_signed = present and any(m in ("c2pa.claim", "c2pa.signature", "c2pa.assertion") for m in manifests)
    if not present:
        status, details = "NO_C2PA", "No C2PA Content Credentials manifest detected."
    elif ai_declaration:
        status, details = "AI_DECLARED_CREDENTIALS", "Metadata declares AI generation/synthesis (unauthenticated claim)."
    elif is_signed:
        status = "SIGNATURE_MARKER_UNVERIFIED"
        details = ("C2PA signature/claim markers found. Presence only: no certificate chain or hash binding "
                   "was validated, so this is NOT cryptographic verification.")
    else:
        status, details = "UNVERIFIED_MANIFEST", "C2PA markers detected but cryptographic signature was not verified."
    return {
        "c2pa_present": present,
        "is_signed": is_signed,
        "creation_tool": None,
        "ai_declaration": bool(ai_declaration and present),
        "status": status,
        "details": details,
        "manifests_found": manifests,
    }


def build_exif_block(
    camera_make: Optional[str] = None,
    camera_model: Optional[str] = None,
    software: Optional[str] = None,
    datetime_original: Optional[str] = None,
    ai_signature_found: bool = False,
    signature_details: Optional[str] = None,
    raw_tags: Optional[Dict[str, Any]] = None,
    has_exif: Optional[bool] = None,
) -> Dict[str, Any]:
    """The single EXIF/camera block shape shared by all modalities (unauthenticated metadata)."""
    return {
        "has_exif": bool(has_exif if has_exif is not None else (camera_make or camera_model or software or raw_tags)),
        "camera_make": camera_make or None,
        "camera_model": camera_model or None,
        "software": software or None,
        "datetime_original": datetime_original or None,
        "ai_signature_found": bool(ai_signature_found),
        "signature_details": signature_details,
        "raw_tags": dict(raw_tags or {}),
    }


def build_provenance_view(c2pa: Dict[str, Any], exif: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based verdict over the c2pa/exif blocks (declarations only; hardware EXIF is unauthenticated)."""
    if c2pa["c2pa_present"]:
        if c2pa["ai_declaration"]:
            verdict, conf = "C2PA_DECLARED_SYNTHETIC", 0.98
        elif c2pa["is_signed"]:
            verdict, conf = "C2PA_SIGNATURE_MARKER_UNVERIFIED", 0.50
        else:
            verdict, conf = "C2PA_PRESENT_UNVERIFIED", 0.50
    elif exif["ai_signature_found"]:
        verdict, conf = "METADATA_DECLARED_SYNTHETIC", 0.95
    elif exif.get("camera_make") and exif.get("camera_model"):
        verdict, conf = "HARDWARE_EXIF_PRESENT", 0.25
    else:
        verdict, conf = "PROVENANCE_UNKNOWN", 0.50
    return {
        "c2pa": c2pa,
        "exif": exif,
        "provenance_verdict": verdict,
        "provenance_ai_confidence": conf,
        "provenance_rule": PROVENANCE_RULE,
    }
