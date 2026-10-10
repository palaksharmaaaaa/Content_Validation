"""
video_detector.dimension_checks.integrity: file-integrity & security checks. SECURITY class: never alter P(AI); they surface risks.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.forensics.bytescan import WINDOW, find_injection, scan_windows
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from video_detector.dimension_checks import _common as C

STAGE = "file_integrity"
DIM = "P"

_FAMILY = {"mp4": "isobmff", "mov": "isobmff", "mkv": "mkv", "avi": "avi"}


def _f(check_id, title, status, severity, detail, data=None) -> Finding:
    return Finding(check_id=check_id, dimension=DIM, stage=STAGE, title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.SECURITY, detail=detail, data=data or {})


@registry.register("video", "format_sniff")
def check_format_sniff(ctx: CheckContext) -> Finding:
    head, _t, _s = C.read_windows(ctx.path)
    detected = C.sniff_video_format(head)
    declared = C.EXT_TO_FORMAT.get(ctx.path.suffix.lower(), "unknown")
    data = {"detected": detected, "declared": declared}
    title = "True format vs. extension"
    if detected == "unknown" or declared == "unknown":
        return _f("format_sniff", title, FindingStatus.INFO, Severity.LOW,
                  f"Could not fully classify the container (detected={detected}, extension={ctx.path.suffix or 'none'}).", data)
    if _FAMILY.get(detected) == _FAMILY.get(declared):
        return _f("format_sniff", title, FindingStatus.PASS, Severity.NONE, f"File content matches its extension ({detected}).", data)
    return _f("format_sniff", title, FindingStatus.WARN, Severity.MEDIUM,
              f"Extension claims {declared} but the file content is {detected}; common in re-labelled or disguised uploads.", data)


def _printable(t: str) -> bool:
    return len(t) == 4 and all(0x20 <= ord(c) <= 0x7E or ord(c) == 0xA9 for c in t)


def _declared_end(ctx_path, head: bytes, size: int) -> Tuple[str, Optional[int], Dict[str, Any]]:
    """(format, end offset of the structured container or None, extras)."""
    fmt = C.sniff_video_format(head)
    if fmt in ("mp4", "mov"):
        top = C.walk_top_level(ctx_path)
        extras: Dict[str, Any] = {"truncated": False}
        for b in top["boxes"]:
            if not _printable(b["type"]):
                return fmt, b["offset"], extras  # garbage where a box should start: trailing/appended data
        if top["issues"] and any("past the end" in i for i in top["issues"]):
            extras["truncated"] = True
            return fmt, size, extras
        return fmt, top["end_offset"], extras
    if fmt == "avi" and len(head) >= 8:
        riff_size = int.from_bytes(head[4:8], "little")
        if riff_size in (0, 0xFFFFFFFF):
            return fmt, None, {}                  # a streamed AVI leaves the size unset: there is no declared end to compare
        return fmt, 8 + riff_size, {"truncated": False}
    return fmt, None, {}


@registry.register("video", "trailing_data")
def check_trailing_data(ctx: CheckContext) -> Finding:
    head, _t, size = C.read_windows(ctx.path)
    fmt, end, extras = _declared_end(ctx.path, head, size)
    title = "Bytes beyond declared end of video container"
    if end is None:
        return _f("trailing_data", title, FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  f"No structural end marker can be evaluated for format '{fmt}'.", {"format": fmt})
    data = {"format": fmt, "declared_end": end, "trailing_bytes": max(0, size - end)}
    if extras.get("truncated"):
        return _f("trailing_data", title, FindingStatus.WARN, Severity.MEDIUM,
                  "The last box extends past the end of the file: the video is truncated (incomplete download or cut).", data)
    if end > size:
        data["trailing_bytes"] = size - end
        return _f("trailing_data", title, FindingStatus.WARN, Severity.MEDIUM,
                  f"Container declares {end - size:,} more bytes than the file holds; truncated.", data)
    if size - end <= 1:
        return _f("trailing_data", title, FindingStatus.PASS, Severity.NONE, "No data after the declared end of the container.", data)
    return _f("trailing_data", title, FindingStatus.WARN, Severity.MEDIUM,
              f"{size - end:,} unexplained bytes follow the declared end of the container; could be an appended payload.", data)


@registry.register("video", "polyglot_signatures")
def check_polyglot_signatures(ctx: CheckContext) -> Finding:
    head, tail, size = C.read_windows(ctx.path)
    fmt, end, extras = _declared_end(ctx.path, head, size)
    windows = [(head, False), (tail, False)]
    if end is not None and 1 < size - end:
        with open(ctx.path, "rb") as f:
            f.seek(end)
            windows.append((f.read(WINDOW), True))
    sigs = scan_windows(windows)
    title = "Embedded executable / archive / script signatures"
    if sigs:
        return _f("polyglot_signatures", title, FindingStatus.FAIL, Severity.HIGH,
                  f"Video file carries signatures of other file types: {', '.join(sigs)}. Treat as a potential polyglot; "
                  "do not open it with other tools.", {"signatures": sigs})
    return _f("polyglot_signatures", title, FindingStatus.PASS, Severity.NONE, "No archive, executable or script signatures found.",
              {"signatures": []})


@registry.register("video", "prompt_injection_text")
def check_prompt_injection_text(ctx: CheckContext) -> Finding:
    fields = C.collect_text_fields(ctx.path)
    matches = find_injection(fields)
    title = "Instruction-like text in metadata"
    if matches:
        return _f("prompt_injection_text", title, FindingStatus.WARN, Severity.HIGH,
                  "Metadata contains instruction-style text aimed at automated reviewers or AI models; ignore it and do not "
                  "pass raw tags to a language model.", {"matches": matches})
    return _f("prompt_injection_text", title, FindingStatus.PASS, Severity.NONE,
              f"Scanned {len(fields)} free-text metadata field(s); no instruction-like text found.", {"matches": []})
