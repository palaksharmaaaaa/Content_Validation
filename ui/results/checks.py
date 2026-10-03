"""ui.results.checks: the Evidence tab - integrity/metadata/context checks, generator attribution, localisation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import streamlit as st

from ui.stages import collect_findings, render_findings_stage


@dataclass(frozen=True)
class CheckGroup:
    """A titled set of finding stages shown together on the Evidence tab."""
    title: str
    caption: str
    stages: Tuple[str, ...]
    show_weight: bool = False
    note: str = ""


_ADVISORY = "Advisory: these never change the authenticity score."
_WEAK = "Metadata is forgeable and often stripped; absence is never treated as evidence and contributions are small and capped."

IMAGE_CHECKS = (
    CheckGroup("File integrity and security", "True format, trailing data, embedded executables or scripts, instruction-like metadata.",
               ("file_integrity",), note=_ADVISORY),
    CheckGroup("Metadata and container", "JPEG tables, PNG chunks, EXIF coherence, timestamps, embedded thumbnail, ICC profile.",
               ("formats", "metadata"), show_weight=True, note=_WEAK),
    CheckGroup("Faces", "Whether each face looks like a real photograph or a generated one (trained classifier, advisory).",
               ("faces",), note=_ADVISORY),
    CheckGroup("Context, rights and lifecycle", "Re-use fingerprint, rights/privacy flags, AI-disclosure labels, re-encode likelihood.",
               ("context", "legal", "lifecycle"), note=_ADVISORY + " Not legal advice."),
)
VIDEO_CHECKS = (
    CheckGroup("File integrity and security", "True container format, bytes past the declared end, embedded executables, instruction-like text.",
               ("file_integrity",), note=_ADVISORY),
    CheckGroup("Container, metadata and signal", "MP4/MOV boxes, sample tables, creation times, telemetry tracks, interlacing, frame cadence.",
               ("container", "signal"), show_weight=True,
               note="Container fields are rewritten by editors and platforms. Interlace and cadence describe capture or conversion, not AI origin."),
    CheckGroup("Context, rights and lifecycle", "Frame fingerprint for re-use, rights and biometric flags, AI-disclosure, re-encoding.",
               ("context", "legal", "lifecycle"), note=_ADVISORY + " Not legal advice."),
)
AUDIO_CHECKS = (
    CheckGroup("File integrity and security", "True container format, bytes past the declared end, embedded executables, instruction-like text.",
               ("file_integrity",), note=_ADVISORY),
    CheckGroup("Container, metadata and signal", "ID3 / RIFF / FLAC-MD5 / MP3 / Ogg structure, mains-hum (ENF) trace, fake hi-res, band-limiting, loudness.",
               ("container", "signal"), show_weight=True,
               note="Tags are forgeable. A missing mains-hum trace is never evidence of synthesis (battery power and noise reduction remove it)."),
    CheckGroup("Context, rights and lifecycle", "Audio fingerprint for re-use, rights and biometric flags, AI-disclosure, re-encoding.",
               ("context", "legal", "lifecycle"), note=_ADVISORY + " Not legal advice."),
)


def _render_attribution(decision: Dict[str, Any]) -> None:
    attr = decision.get("model_attribution", {})
    state = decision.get("taxonomy_state")
    authentic = state in ("AUTHENTIC_REAL_PHOTOGRAPH", "AUTHENTIC_EDITED") and "Topaz" not in str(attr.get("attributed_model") or "")
    st.markdown("**Generator attribution**")
    if authentic:
        st.write("No generator footprint detected; consistent with a camera capture.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Best match", attr.get("attributed_model", "Unattributable"))
    c2.metric("Origin", attr.get("region_of_origin", "Unknown"))
    c3.metric("Match strength", f"{int(float(attr.get('attribution_confidence', attr.get('confidence', 0.0))) * 100)}%")
    candidates = attr.get("top_candidates", [])
    if candidates:
        st.caption("Other candidates: " + ", ".join(f"{c.get('model')} ({float(c.get('confidence', 0)) * 100:.0f}%)" for c in candidates[:3]))
    st.caption("Attribution only explains a suspicious result. It is not scored as evidence.")


def _render_localization(decision: Dict[str, Any]) -> None:
    loc = decision.get("localization", {})
    rows = [
        ("Image area flagged", float(loc.get("suspicious_image_area_pct", 0))),
        ("Video time flagged", float(loc.get("suspicious_video_duration_pct", 0))),
        ("Audio time flagged", float(loc.get("suspicious_audio_duration_pct", 0))),
    ]
    rows = [(label, value) for label, value in rows if value > 0]
    if rows:
        st.markdown("**Where it looks manipulated**")
        for label, value in rows:
            st.write(f"- {label}: {value:.1f}%")


def render_evidence(item: Dict[str, Any], groups: Sequence[CheckGroup]) -> None:
    """Draw the Evidence tab: each group of findings as a table, then attribution, localisation and the evidence trail."""
    decision = item.get("decision") or {}
    dim_report = item.get("dimension_report") or {}
    for group in groups:
        render_findings_stage(
            group.title, group.caption, collect_findings(dim_report, group.stages),
            show_weight=group.show_weight, advisory_note=group.note,
        )
    _render_attribution(decision)
    _render_localization(decision)
    trail = decision.get("evidence_trail", [])
    if trail:
        with st.expander("Full evidence trail"):
            for line in trail:
                st.markdown(f"- {line}")
