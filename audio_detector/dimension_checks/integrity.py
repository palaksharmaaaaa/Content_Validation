"""
audio_detector.dimension_checks.integrity: file-integrity & security checks (report Dimension P /
Section 21.6). SECURITY class: never alter P(AI); they surface risks.
"""
from __future__ import annotations

from pathlib import Path

from audio_detector.dimension_checks import _common as C
from core.forensics.bytescan import WINDOW, find_injection, scan_signatures
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity

STAGE = "file_integrity"
DIM = "P"


def _f(check_id, title, status, severity, detail, data=None) -> Finding:
    return Finding(check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.SECURITY, detail=detail, data=data or {})


@registry.register("audio", "format_sniff")
def check_format_sniff(ctx: CheckContext) -> Finding:
    head, _tail, _size = C.read_windows(ctx.path)
    detected = C.sniff_audio_format(head)
    declared = C.EXT_TO_FORMAT.get(ctx.path.suffix.lower(), "unknown")
    data = {"detected": detected, "declared": declared}
    if declared == "unknown" or detected == "unknown":
        return _f("format_sniff", "True format vs. extension", FindingStatus.INFO, Severity.LOW,
                  f"Could not fully classify the container (detected={detected}, extension={ctx.path.suffix or 'none'}).", data)
    same = detected == declared or {detected, declared} <= {"mp3", "aac"}
    if same:
        return _f("format_sniff", "True format vs. extension", FindingStatus.PASS, Severity.NONE,
                  f"File content matches its extension ({detected}).", data)
    return _f("format_sniff", "True format vs. extension", FindingStatus.WARN, Severity.MEDIUM,
              f"Extension claims {declared} but the file content is {detected}; common in re-labelled or disguised uploads.", data)


def _declared_end(path: Path, head: bytes, size: int):
    """(format, declared end offset or None) for formats whose end can be derived structurally."""
    fmt = C.sniff_audio_format(head)
    if fmt == "wav" and len(head) >= 8 and head[:4] == b"RIFF":
        return fmt, 8 + int.from_bytes(head[4:8], "little")
    if fmt == "ogg" and size <= C.MAX_OGG_BYTES:
        pages = C.parse_ogg_pages(path.read_bytes())
        complete = [p for p in pages if not p.get("truncated")]
        if complete:
            last = complete[-1]
            # end of last complete page = start of next (or scan: parse again to find length)
            data = path.read_bytes()
            nseg = data[last["offset"] + 26]
            seg_total = sum(data[last["offset"] + 27 : last["offset"] + 27 + nseg])
            return fmt, last["offset"] + 27 + nseg + seg_total
    return fmt, None


@registry.register("audio", "trailing_data")
def check_trailing_data(ctx: CheckContext) -> Finding:
    head, _tail, size = C.read_windows(ctx.path)
    fmt, end = _declared_end(ctx.path, head, size)
    if end is None:
        return _f("trailing_data", "Bytes beyond declared end of audio", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  f"No structural end marker can be evaluated for format '{fmt}'.", {"format": fmt})
    trailing = size - end
    data = {"format": fmt, "trailing_bytes": trailing, "declared_end": end}
    if trailing < 0:
        return _f("trailing_data", "Bytes beyond declared end of audio", FindingStatus.WARN, Severity.MEDIUM,
                  f"Container declares {-trailing:,} more bytes than the file holds; the file is truncated or was cut.", data)
    if trailing <= 1:  # RIFF odd-size padding
        return _f("trailing_data", "Bytes beyond declared end of audio", FindingStatus.PASS, Severity.NONE,
                  "No data after the declared end of the audio container.", data)
    return _f("trailing_data", "Bytes beyond declared end of audio", FindingStatus.WARN, Severity.MEDIUM,
              f"{trailing:,} unexplained bytes follow the declared end of the container; could be an appended payload.", data)


@registry.register("audio", "polyglot_signatures")
def check_polyglot_signatures(ctx: CheckContext) -> Finding:
    head, tail, size = C.read_windows(ctx.path)
    fmt, end = _declared_end(ctx.path, head, size)
    found = set(scan_signatures(head, False)) | set(scan_signatures(tail, False))
    if end is not None and 1 < size - end <= C.FULL_READ_LIMIT:
        with open(ctx.path, "rb") as f:
            f.seek(end)
            found |= set(scan_signatures(f.read(WINDOW), True))
    sigs = sorted(found)
    if sigs:
        return _f("polyglot_signatures", "Embedded executable / archive / script signatures", FindingStatus.FAIL,
                  Severity.HIGH, f"Audio file carries signatures of other file types: {', '.join(sigs)}. Treat as a potential "
                  "polyglot; do not open it with other tools.", {"signatures": sigs})
    return _f("polyglot_signatures", "Embedded executable / archive / script signatures", FindingStatus.PASS,
              Severity.NONE, "No archive, executable or script signatures found.", {"signatures": []})


@registry.register("audio", "prompt_injection_text")
def check_prompt_injection_text(ctx: CheckContext) -> Finding:
    fields = C.collect_text_fields(ctx.path)
    matches = find_injection(fields)
    if matches:
        return _f("prompt_injection_text", "Instruction-like text in metadata", FindingStatus.WARN, Severity.HIGH,
                  "Metadata contains instruction-style text aimed at automated reviewers or AI models; ignore it and do not "
                  "pass raw tags to a language model.", {"matches": matches})
    return _f("prompt_injection_text", "Instruction-like text in metadata", FindingStatus.PASS, Severity.NONE,
              f"Scanned {len(fields)} free-text metadata field(s); no instruction-like text found.", {"matches": []})
