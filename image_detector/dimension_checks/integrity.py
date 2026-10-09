"""
image_detector.dimension_checks.integrity: file-integrity & security checks. All findings are SECURITY class: they never alter P(AI); they surface risks.
"""
from __future__ import annotations

import re
from pathlib import Path

from core.forensics.bytescan import WINDOW, find_injection, scan_signatures
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from image_detector.dimension_checks import _common as C

STAGE = "file_integrity"
DIM = "Q"


def _f(check_id: str, title: str, status: FindingStatus, severity: Severity, detail: str, data=None) -> Finding:
    return Finding(
        check_id=check_id,
        dimension=DIM,
        stage=STAGE,
        title=title,
        status=status,
        severity=severity,
        evidence_class=EvidenceClass.SECURITY,
        detail=detail,
        data=data or {},
    )


def _declared_format(path: Path) -> str:
    return C.EXT_TO_FORMAT.get(path.suffix.lower(), "unknown")


@registry.register("image", "format_sniff")
def check_format_sniff(ctx: CheckContext) -> Finding:
    head, _tail, _size = C.read_windows(ctx.path)
    detected = C.sniff_format(head)
    declared = _declared_format(ctx.path)
    data = {"detected": detected, "declared": declared}
    if declared == "unknown" or detected == "unknown":
        return _f("format_sniff", "True format vs. extension", FindingStatus.INFO, Severity.LOW,
                  f"Could not fully classify format (detected={detected}, extension={ctx.path.suffix or 'none'}).", data)
    if detected == declared or (detected == "heif" and declared in ("jpeg", "png")):
        if detected == "heif":
            return _f("format_sniff", "True format vs. extension", FindingStatus.WARN, Severity.LOW,
                      "File content is HEIF/AVIF but the extension claims another format.", data)
        return _f("format_sniff", "True format vs. extension", FindingStatus.PASS, Severity.NONE,
                  f"File content matches its extension ({detected}).", data)
    return _f("format_sniff", "True format vs. extension", FindingStatus.WARN, Severity.MEDIUM,
              f"Extension claims {declared} but the file content is {detected}; common in re-labelled or disguised uploads.", data)


def _end_offset(path: Path, head: bytes, size: int):
    fmt = C.sniff_format(head)
    if size > C.FULL_READ_LIMIT:
        return fmt, None
    data = path.read_bytes()
    if fmt == "jpeg":
        return fmt, C.jpeg_end_offset(data)
    if fmt == "png":
        return fmt, C.png_end_offset(data)
    if fmt == "webp" and len(data) >= 12:
        return fmt, 8 + int.from_bytes(data[4:8], "little")
    if fmt == "bmp" and len(data) >= 6:
        return fmt, int.from_bytes(data[2:6], "little")
    return fmt, None


@registry.register("image", "trailing_data")
def check_trailing_data(ctx: CheckContext) -> Finding:
    head, tail, size = C.read_windows(ctx.path)
    fmt, end = _end_offset(ctx.path, head, size)
    if end is None or end > size:
        return _f("trailing_data", "Bytes after image end marker", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  f"No end-of-image structure could be evaluated for format '{fmt}'.", {"format": fmt})
    trailing = size - end
    data = {"format": fmt, "trailing_bytes": trailing, "image_end_offset": end}
    if trailing <= 0:
        return _f("trailing_data", "Bytes after image end marker", FindingStatus.PASS, Severity.NONE,
                  "No data after the image's end marker.", data)
    benign = fmt == "jpeg" and any(m in head for m in (b"MPF\x00", b"MotionPhoto", b"MicroVideo", b"GContainer"))
    if benign:
        return _f("trailing_data", "Bytes after image end marker", FindingStatus.INFO, Severity.LOW,
                  f"{trailing:,} bytes follow the primary image; consistent with a multi-picture / motion-photo container.", data)
    return _f("trailing_data", "Bytes after image end marker", FindingStatus.WARN, Severity.MEDIUM,
              f"{trailing:,} unexplained bytes follow the image end marker; could be an appended payload, archive, or stripped metadata.", data)


@registry.register("image", "polyglot_signatures")
def check_polyglot_signatures(ctx: CheckContext) -> Finding:
    head, tail, size = C.read_windows(ctx.path)
    fmt, end = _end_offset(ctx.path, head, size)
    found = set(scan_signatures(head, False)) | set(scan_signatures(tail, False))
    if end is not None and size - end > 0 and size <= C.FULL_READ_LIMIT:
        trailing = ctx.path.read_bytes()[end : end + WINDOW]
        found |= set(scan_signatures(trailing, True))
    sigs = sorted(found)
    if sigs:
        return _f("polyglot_signatures", "Embedded executable / archive / script signatures", FindingStatus.FAIL,
                  Severity.HIGH, f"Image carries signatures of other file types: {', '.join(sigs)}. Treat as a potential polyglot; "
                  "do not open it with other tools.", {"signatures": sigs})
    return _f("polyglot_signatures", "Embedded executable / archive / script signatures", FindingStatus.PASS,
              Severity.NONE, "No archive, executable or script signatures found.", {"signatures": []})


_SVG_PATTERNS = {
    "script": re.compile(r"<\s*script", re.I),
    "event_handler": re.compile(r"[\s\"'/]on[a-z]{3,}\s*=", re.I),
    "entity": re.compile(r"<!ENTITY", re.I),
    "javascript_uri": re.compile(r"javascript\s*:", re.I),
    "foreign_object": re.compile(r"<\s*foreignObject", re.I),
}


@registry.register("image", "svg_active_content")
def check_svg_active_content(ctx: CheckContext) -> Finding:
    head, _tail, _size = C.read_windows(ctx.path)
    if C.sniff_format(head) != "svg" and ctx.path.suffix.lower() != ".svg":
        return _f("svg_active_content", "SVG active content", FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Not an SVG/XML document.")
    text = head.decode("utf-8", errors="ignore")
    hits = [name for name, rx in _SVG_PATTERNS.items() if rx.search(text)]
    if hits:
        return _f("svg_active_content", "SVG active content", FindingStatus.FAIL, Severity.HIGH,
                  f"SVG contains active or external-entity content ({', '.join(hits)}). Never render untrusted SVG inline.",
                  {"indicators": hits})
    return _f("svg_active_content", "SVG active content", FindingStatus.PASS, Severity.NONE,
              "No scripts, event handlers or external entities found.", {"indicators": []})


@registry.register("image", "prompt_injection_text")
def check_prompt_injection_text(ctx: CheckContext) -> Finding:
    fields = C.collect_text_fields(ctx.path)
    matches = find_injection(fields)
    if matches:
        return _f("prompt_injection_text", "Instruction-like text in metadata", FindingStatus.WARN, Severity.HIGH,
                  "Metadata contains instruction-style text aimed at automated reviewers or AI models; ignore it and "
                  "do not pass raw metadata to a language model.", {"matches": matches})
    return _f("prompt_injection_text", "Instruction-like text in metadata", FindingStatus.PASS, Severity.NONE,
              f"Scanned {len(fields)} free-text metadata field(s); no instruction-like text found.", {"matches": []})
