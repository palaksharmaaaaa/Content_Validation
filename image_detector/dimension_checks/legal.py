"""
image_detector.dimension_checks.legal: advisory legal/rights flags (report §22.2). LEGAL_FLAG
findings never alter P(AI) and are not legal advice; they surface facts a reviewer should consider.
"""
from __future__ import annotations

from typing import List

from PIL import Image

from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from image_detector.dimension_checks import _common as C

STAGE = "legal"
DIM = "Section22.2"

AI_LEAN_PERCENT = 60.0


def _f(check_id, title, status, severity, detail, data=None) -> Finding:
    return Finding(check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.LEGAL_FLAG, detail=detail, data=data or {})


@registry.register("image", "rights_and_privacy", phase="post")
def check_rights_and_privacy(ctx: CheckContext) -> List[Finding]:
    out: List[Finding] = []
    copyright_text, artist, has_gps = "", "", False
    try:
        with Image.open(ctx.path) as im:
            ex = im.getexif()
            copyright_text = str(ex.get(0x8298, "") or "").replace("\x00", "").strip()
            artist = str(ex.get(0x013B, "") or "").replace("\x00", "").strip()
            has_gps = bool(ex.get_ifd(0x8825))
    except Exception:
        pass
    head, _tail, _size = C.read_windows(ctx.path)
    xmp = C.extract_xmp(head)
    xmp_low = xmp.lower()
    xmp_rights = [m for m in ("dc:rights", "xmprights:", "cc:license", "plus:licensor", "photoshop:credit") if m in xmp_low]

    if copyright_text or artist or xmp_rights:
        out.append(_f("rights_notice", "Rights / licence notice", FindingStatus.INFO, Severity.LOW,
                      "A copyright, creator or licence notice is embedded; respect the stated terms before reuse.",
                      {"copyright": copyright_text, "artist": artist, "xmp_markers": xmp_rights}))
    else:
        out.append(_f("rights_notice", "Rights / licence notice", FindingStatus.INFO, Severity.NONE,
                      "No embedded rights or licence notice (absence does not mean the image is free to use).",
                      {"copyright": "", "artist": "", "xmp_markers": []}))

    if has_gps:
        out.append(_f("location_privacy", "Embedded location data", FindingStatus.WARN, Severity.LOW,
                      "GPS coordinates are embedded; sharing the file may disclose a precise location (privacy / data-protection concern).",
                      {"gps_present": True}))
    else:
        out.append(_f("location_privacy", "Embedded location data", FindingStatus.PASS, Severity.NONE,
                      "No GPS location embedded.", {"gps_present": False}))

    faces = int(ctx.content.get("faces_count", 0) or 0)
    if faces > 0:
        out.append(_f("biometric_notice", "Identifiable people", FindingStatus.INFO, Severity.MEDIUM,
                      f"{faces} face(s) detected. Biometric-privacy regimes (e.g. GDPR Art. 9, Illinois BIPA, Texas CUBI) and "
                      "likeness/consent rights may apply to processing or publishing this image.", {"faces_count": faces}))
    else:
        out.append(_f("biometric_notice", "Identifiable people", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                      "No faces detected.", {"faces_count": 0}))

    ai_pct = float(ctx.ai_result.get("ai_percentage", 0.0) or 0.0)
    if ai_pct >= AI_LEAN_PERCENT:
        labelled = bool(ctx.provenance.get("c2pa_present")) or "digitalsourcetype" in xmp_low
        if labelled:
            out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.PASS, Severity.NONE,
                          "A machine-readable provenance/AI-disclosure marker is present.", {"labelled": True}))
        else:
            out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.WARN, Severity.LOW,
                          "Image is judged AI-leaning but carries no machine-readable AI disclosure (C2PA / IPTC digitalSourceType); "
                          "EU AI Act Art. 50 expects providers of synthetic media to mark outputs.", {"labelled": False}))
    else:
        out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                      "Not applicable: image is not judged AI-leaning.", {"labelled": None}))
    return out
