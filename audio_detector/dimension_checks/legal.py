"""
audio_detector.dimension_checks.legal: advisory legal/rights flags. LEGAL_FLAG
findings never alter P(AI) and are not legal advice.
"""
from __future__ import annotations

from typing import List

from audio_detector.dimension_checks import _common as C
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity

STAGE = "legal"
DIM = "Section21.2"
AI_LEAN_PERCENT = 60.0
SPEECH_RATIO = 0.65


def _f(check_id, title, status, severity, detail, data=None) -> Finding:
    return Finding(check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.LEGAL_FLAG, detail=detail, data=data or {})


def _is_speech(content: dict) -> bool:
    return "speech" in str(content.get("dominant_modality", "")).lower() or float(content.get("speech_ratio", 0.0) or 0.0) > SPEECH_RATIO


@registry.register("audio", "rights_and_privacy", phase="post")
def check_rights_and_privacy(ctx: CheckContext) -> List[Finding]:
    out: List[Finding] = []
    fields = C.collect_text_fields(ctx.path)
    copyright_text = next((fields[k] for k in ("id3:TCOP", "id3:TCO", "riff:ICOP", "vorbis:COPYRIGHT") if fields.get(k)), "")
    publisher = next((fields[k] for k in ("id3:TPUB", "riff:IENG", "bext:originator") if fields.get(k)), "")
    if copyright_text or publisher:
        out.append(_f("rights_notice", "Rights / licence notice", FindingStatus.INFO, Severity.LOW,
                      "A copyright, publisher or originator notice is embedded; respect the stated terms before reuse. "
                      "Remember that the sound recording and the underlying composition carry separate rights.",
                      {"copyright": copyright_text, "publisher": publisher}))
    else:
        out.append(_f("rights_notice", "Rights / licence notice", FindingStatus.INFO, Severity.NONE,
                      "No embedded rights notice (absence does not mean the audio is free to use).", {"copyright": "", "publisher": ""}))

    speech = _is_speech(ctx.content)
    if speech:
        out.append(_f("voice_biometric_notice", "Voice recording of a person", FindingStatus.INFO, Severity.MEDIUM,
                      "The recording is speech-dominant. Voiceprints are biometric data under GDPR Art. 9, Illinois BIPA and Texas CUBI; "
                      "consent and likeness rights may apply to processing or publishing it.", {"speech": True}))
    else:
        out.append(_f("voice_biometric_notice", "Voice recording of a person", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                      "Not speech-dominant.", {"speech": False}))

    ai_pct = float(ctx.ai_result.get("ai_percentage", 0.0) or 0.0)
    ai_leaning = ai_pct >= AI_LEAN_PERCENT
    if ai_leaning:
        labelled = bool(ctx.provenance.get("c2pa_present"))
        if labelled:
            out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.PASS, Severity.NONE,
                          "A machine-readable provenance marker (C2PA) is present.", {"labelled": True}))
        else:
            out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.WARN, Severity.LOW,
                          "Audio is judged AI-leaning but carries no machine-readable AI disclosure; EU AI Act Art. 50 expects providers "
                          "of synthetic audio to mark outputs.", {"labelled": False}))
    else:
        out.append(_f("ai_disclosure_label", "AI-content disclosure label", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                      "Not applicable: audio is not judged AI-leaning.", {"labelled": None}))

    if ai_leaning and speech:
        out.append(_f("synthetic_voice_advisory", "Synthetic voice: likeness & telephony rules", FindingStatus.WARN, Severity.LOW,
                      "Synthetic speech may implicate voice-likeness / right-of-publicity laws and disclosure duties; in the US, the FCC "
                      "treats AI-generated voices in robocalls as 'artificial' voices under the TCPA (consent required).", {"speech": True}))
    else:
        out.append(_f("synthetic_voice_advisory", "Synthetic voice: likeness & telephony rules", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                      "Not applicable (not AI-leaning speech).", {}))
    return out
