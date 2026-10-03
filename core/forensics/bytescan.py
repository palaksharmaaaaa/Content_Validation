"""
core.forensics.bytescan: bounded byte-level scans shared by every modality's integrity checks.

* ``read_windows``    -- (head, tail, size) with each window capped (default 4 MB).
* ``scan_signatures`` -- archive / document / executable / script signatures inside a byte buffer.
* ``find_injection``  -- instruction-like text aimed at automated reviewers or language models.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

WINDOW = 4 * 1024 * 1024

_ZIP_LOCAL = b"PK\x03\x04"
_ZIP_END = b"PK\x05\x06"


def read_windows(path: Path, window: int = WINDOW) -> Tuple[bytes, bytes, int]:
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


def scan_signatures(buf: bytes, trailing_only: bool) -> List[str]:
    """
    Signatures of other file types. ``trailing_only=True`` is for data known to follow the media payload
    (additional low-false-positive signatures are enabled there).
    """
    found: List[str] = []
    low = buf.lower()
    if _ZIP_LOCAL in buf and _ZIP_END in buf:
        found.append("ZIP")
    if b"%PDF-" in buf:
        found.append("PDF")
    if b"<script" in low or b"<html" in low or b"<?php" in low:
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


INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)", re.I),
    re.compile(r"disregard\s+.{0,40}(instructions|rules|guidelines)", re.I),
    re.compile(r"(classify|mark|label|report|treat)\s+(this|the)\s+(image|photo|picture|file|audio|recording|clip|video)\s+as\s+(authentic|real|genuine|safe|human)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\b", re.I),
    re.compile(r"system\s+prompt", re.I),
    re.compile(r"do\s+not\s+(flag|report|detect)\s+this", re.I),
]


def find_injection(fields: Dict[str, str]) -> List[Dict[str, str]]:
    matches: List[Dict[str, str]] = []
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
