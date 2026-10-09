"""
video_detector.dimension_checks.legal: advisory legal/rights flags. LEGAL_FLAG
findings never alter P(AI) and are not legal advice.
"""
from __future__ import annotations

from typing import List

from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from video_detector.dimension_checks import _common as C

STAGE = "legal"
DIM = "Section21.2"
AI_LEAN_PERCENT = 60.0
_GPS_ENTRIES = {"gpmd", "camm", "djmd", "dbgi"}


def _f(check_id, title, status, severity, detail, data=None) -> Finding:
    return Finding(check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.LEGAL_FLAG, detail=detail, data=data or {})


def _faces(content: dict) -> int:
    n = content.get("faces_count")
    if n is None:
        humans = (content.get("living_entities") or {}).get("humans") or {}
        n = humans.get("faces", humans.get("count", 0))
    return int(n or 0)


def _moov_info(ctx: CheckContext):
    head, _t, _s = C.read_windows(ctx.path)
    if C.sniff_video_format(head) not in ("mp4", "mov"):
        return None
    return C.load_moov(ctx.path, C.walk_top_level(ctx.path))


@registry.register("video", "rights_and_privacy", phase="post")
def check_rights_and_privacy(ctx: CheckContext) -> List[Finding]:
    out: List[Finding] = []
    moov = _moov_info(ctx)
    items = (moov or {}).get("text_items", {})
    copyright_text = items.get("\xa9cpy", "")
    author = items.get("\xa9aut", "") or items.get("\xa9prd", "")
    if copyright_text or author:
        out.append(_f("rights_notice", "Rights / licence notice", FindingStatus.INFO, Severity.LOW,
                      "A copyright or creator notice is embedded; respect the stated terms before reuse.",
                      {"copyright": copyright_text, "author": author}))
    else:
        out.append(_f("rights_notice", "Rights / licence notice", FindingStatus.INFO, Severity.NONE,
                      "No embedded rights notice (absence does not mean the footage is free to use).", {"copyright": "", "author": ""}))

    location = items.get("\xa9xyz", "")
    gps_tracks = sorted({t.get("sample_entry") for t in (moov or {}).get("tracks", []) if t.get("sample_entry") in _GPS_ENTRIES})
    if location or gps_tracks:
        out.append(_f("location_privacy", "Embedded location data", FindingStatus.WARN, Severity.LOW,
                      "The file embeds location data" + (f" ({location})" if location else "") +
                      (f" and telemetry track(s) that may carry GPS ({', '.join(gps_tracks)})" if gps_tracks else "") +
                      "; sharing it may disclose a precise location (privacy / data-protection concern).",
                      {"location": location, "telemetry_tracks": gps_tracks}))
    else:
        out.append(_f("location_privacy", "Embedded location data", FindingStatus.PASS, Severity.NONE,
                      "No embedded location data found.", {"location": "", "telemetry_tracks": []}))

    faces = _faces(ctx.content)
    if faces > 0:
        out.append(_f("biometric_notice", "Identifiable people", FindingStatus.INFO, Severity.MEDIUM,
                      f"Up to {faces} face(s) detected. Biometric-privacy regimes (GDPR Art. 9, Illinois BIPA, Texas CUBI) and likeness/"
                      "consent rights may apply; clinical or surgical footage may also carry health-data (HIPAA/GDPR) duties.",
                      {"faces_count": faces}))
    else:
        out.append(_f("biometric_notice", "Identifiable people", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                      "No faces detected.", {"faces_count": 0}))

    ai_pct = float(ctx.ai_result.get("ai_percentage", 0.0) or 0.0)
    if ai_pct >= AI_LEAN_PERCENT:
        labelled = bool(ctx.provenance.get("c2pa_present"))
        if labelled:
            out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.PASS, Severity.NONE,
                          "A machine-readable provenance marker (C2PA) is present.", {"labelled": True}))
        else:
            out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.WARN, Severity.LOW,
                          "Video is judged AI-leaning but carries no machine-readable AI disclosure; EU AI Act Art. 50 expects providers "
                          "of synthetic video to mark outputs and deployers of deepfakes to disclose them.", {"labelled": False}))
    else:
        out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                      "Not applicable: video is not judged AI-leaning.", {"labelled": None}))
    return out
