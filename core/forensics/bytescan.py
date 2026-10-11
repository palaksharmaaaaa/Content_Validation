"""
core.forensics.bytescan: bounded byte-level scans shared by every modality's integrity checks.

* ``read_windows``    -- (head, tail, size) with each window capped (default 4 MB).
* ``scan_signatures`` -- archive / document / executable / script signatures inside a byte buffer.
* ``find_injection``  -- instruction-like text aimed at automated reviewers or language models.
"""
from __future__ import annotations

import re
from pathlib import Path

WINDOW = 4 * 1024 * 1024

_ZIP_LOCAL = b"PK\x03\x04"
_ZIP_END = b"PK\x05\x06"
# A real tag, not the bare word: "<script" or "<html" must be followed by whitespace, "/" or ">" (so prose such as "<html5" or a tag name
# in metadata text does not count), and "<?php" by whitespace.
_ACTIVE_CONTENT = re.compile(rb"<(?:script|html)[\s/>]|<\?php\s")


def read_windows(path: Path, window: int = WINDOW) -> tuple[bytes, bytes, int]:
    """Returns (head, tail, size). head/tail are each at most ``window`` bytes."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(window)
        if size <= window:
            tail = head
        else:
            f.seek(max(0, size - window))
            tail = f.read(window)
    return head, tail, size


def scan_signatures(buf: bytes, trailing_only: bool) -> list[str]:
    """
    Signatures of other file types. ``trailing_only=True`` is for data known to follow the media payload
    (additional low-false-positive signatures are enabled there).
    """
    found: list[str] = []
    low = buf.lower()
    if _ZIP_LOCAL in buf and _ZIP_END in buf:
        found.append("ZIP")
    if b"%PDF-" in buf:
        found.append("PDF")
    if _ACTIVE_CONTENT.search(low):
        found.append("HTML/Script")
    if b"This program cannot be run in DOS mode" in buf:
        found.append("PE executable")
    if trailing_only:
        if b"\x7fELF" in buf:
            found.append("ELF executable")
        if b"Rar!\x1a\x07" in buf:
            found.append("RAR")
        if b"7z\xbc\xaf\x27\x1c" in buf:
            found.append("7z")
    return found


def scan_windows(windows: list[tuple[bytes, bool]]) -> list[str]:
    """Signatures across several byte windows of one file, as ``[(buffer, trailing_only), ...]``. A ZIP is recognised when its local-file
    header is in one window and its end-of-directory record in another, which is what an archive larger than a window looks like."""
    found: set = set()
    for buf, trailing in windows:
        found |= set(scan_signatures(buf, trailing))
    if _ZIP_LOCAL in b"".join(w for w, _ in windows) and any(_ZIP_END in w for w, _ in windows):
        found.add("ZIP")
    return sorted(found)


INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)", re.I),
    re.compile(r"disregard\s+.{0,40}(instructions|rules|guidelines)", re.I),
    re.compile(r"(classify|mark|label|report|treat)\s+(this|the)\s+(image|photo|picture|file|audio|recording|clip|video)\s+as\s+(authentic|real|genuine|safe|human)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\b", re.I),
    re.compile(r"system\s+prompt", re.I),
    re.compile(r"do\s+not\s+(flag|report|detect)\s+this", re.I),
]


def find_injection(fields: dict[str, str]) -> list[dict[str, str]]:
    """Return the metadata fields that contain instruction-like text aimed at AI readers (prompt-injection patterns)."""
    matches: list[dict[str, str]] = []
    for key, text in fields.items():
        for rx in INJECTION_PATTERNS:
            m = rx.search(text)
            if m:
                matches.append({"field": key, "snippet": text[max(0, m.start() - 20) : m.end() + 20][:160]})
    return matches


def extract_xmp(data: bytes) -> str:
    """The first XMP packet found in ``data`` (empty string if none)."""
    start = data.find(b"<x:xmpmeta")
    if start == -1:
        start = data.find(b"<?xpacket")
    if start == -1:
        return ""
    end = data.find(b"</x:xmpmeta>", start)
    end = end + len(b"</x:xmpmeta>") if end != -1 else min(len(data), start + 262144)
    return data[start:end].decode("utf-8", errors="replace")
