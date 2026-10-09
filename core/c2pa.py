"""core.c2pa: presence scan for C2PA Content Credentials markers, shared by image, audio and video.

This only finds the *marker*; it does not parse or cryptographically verify a manifest, so the result is presence evidence and
never proof of origin. A bare four-letter tag (``c2pa``, ``c2ma``, ``c2cs``) turns up by chance in compressed data about once in a
few thousand megabyte-sized files, so it counts only next to the JUMBF box type; a distinctive marker (a URN, a MIME type, a claim
name) counts on its own.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

MARKERS = (
    b"urn:c2pa",
    b"c2pa",
    b"c2ma",
    b"c2cs",
    b"application/c2pa",
    b"image/jumd",
    b"http://c2pa.org",
    b"c2pa.claim",
    b"c2pa.assertion",
)
MIN_DISTINCTIVE_MARKER = 8          # characters; the four-letter tags are too short to trust alone
WINDOW = 512 * 1024                 # bytes read from the start and from the end of the file


def markers_in(buffer: bytes) -> List[str]:
    return [m.decode("ascii") for m in MARKERS if m in buffer]


def has_credentials(buffer: bytes, found: List[str]) -> bool:
    """True for a distinctive marker, or for any marker together with the JUMBF box type."""
    return any(len(m) >= MIN_DISTINCTIVE_MARKER for m in found) or (b"jumb" in buffer and bool(found))


def scan_file(file_path: str | Path) -> Dict[str, Any]:
    """``{c2pa_present, status, manifests_found}`` from the first and last WINDOW bytes of the file."""
    path = Path(file_path)
    if not path.is_file():
        return {"c2pa_present": False, "status": "FILE_NOT_FOUND", "manifests_found": []}
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            buffer = handle.read(min(size, WINDOW))
            if size > WINDOW:
                handle.seek(max(WINDOW, size - WINDOW))
                buffer += handle.read(WINDOW)
    except OSError as exc:
        logger.debug("C2PA scan error on %s: %s", path.name, exc)
        return {"c2pa_present": False, "status": "ERROR", "manifests_found": []}
    found = markers_in(buffer)
    present = has_credentials(buffer, found)
    return {
        "c2pa_present": present,
        "status": "C2PA_CREDENTIALS_FOUND" if present else "NO_C2PA_MANIFEST",
        "manifests_found": found,
    }
