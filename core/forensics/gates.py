"""
core.forensics.gates: pre-analysis gates that short-circuit the pipeline.

* ``HardBlockGate`` -- SHA-256 match against an operator-supplied local blocklist (CSAM/NCII
  hash lists such as those distributed to trust-and-safety teams). No classifier is built or
  bundled; with no blocklist configured the gate is INACTIVE.
* ``recognize_scientific_format`` -- magic-byte sniff for formats whose authenticity cannot be
  scored by this engine (FITS, DICOM, GeoTIFF, OpenEXR, HDF5, NetCDF, AEDAT/AER event files).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

from core.forensics.config import hardblock_file
from core.hashing import file_sha256

_HEX = set("0123456789abcdef")


@dataclass
class GateResult:
    """Outcome of the hard-block gate: ``INACTIVE`` (no list), ``CLEAR`` or ``HARD_BLOCK_ESCALATE``."""
    status: str  # INACTIVE | CLEAR | HARD_BLOCK_ESCALATE
    sha256: str
    triggered: bool

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form for reports."""
        return {"status": self.status, "sha256": self.sha256, "triggered": self.triggered}


def _load_blocklist(path: Path) -> Optional[Set[str]]:
    if not path.is_file():
        return None
    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        token = line.strip().split()[0].lower() if line.strip() and not line.strip().startswith("#") else ""
        if len(token) == 64 and set(token) <= _HEX:
            out.add(token)
    return out


class HardBlockGate:
    """Stops analysis when a file's SHA-256 is on the operator's block list. No classifier is built in."""
    def __init__(self, blocklist_path: Optional[Union[str, Path]] = None):
        self._explicit = Path(blocklist_path) if blocklist_path else None

    def check(self, file_path: Union[str, Path]) -> GateResult:
        """Hash the file and compare it with the block list."""
        path = Path(file_path)
        try:
            digest = file_sha256(path) if path.is_file() else ""
        except OSError:
            digest = ""
        blocklist = _load_blocklist(self._explicit or hardblock_file())
        if blocklist is None or not digest:
            return GateResult("INACTIVE", digest, False)
        if digest in blocklist:
            return GateResult("HARD_BLOCK_ESCALATE", digest, True)
        return GateResult("CLEAR", digest, False)


def _tiff_has_tag(head: bytes, wanted: int) -> bool:
    if len(head) < 8 or head[:4] not in (b"II*\x00", b"MM\x00*"):
        return False
    endian = "<" if head[:2] == b"II" else ">"
    try:
        (ifd,) = struct.unpack(endian + "I", head[4:8])
        if ifd + 2 > len(head):
            return False
        (n,) = struct.unpack(endian + "H", head[ifd : ifd + 2])
        for i in range(min(n, 512)):
            off = ifd + 2 + i * 12
            if off + 12 > len(head):
                break
            (tag,) = struct.unpack(endian + "H", head[off : off + 2])
            if tag == wanted:
                return True
    except struct.error:
        return False
    return False


def recognize_scientific_format(file_path: Union[str, Path]) -> Optional[Dict[str, str]]:
    """Name the scientific/medical format (DICOM, FITS, GeoTIFF, ...) by magic bytes, or None. Such files are recognised but not scored."""
    path = Path(file_path)
    try:
        with open(path, "rb") as f:
            head = f.read(64 * 1024)
    except OSError:
        return None
    if head.startswith(b"SIMPLE  ="):
        return {"type": "FITS", "description": "FITS astronomical image/data file"}
    if len(head) >= 132 and head[128:132] == b"DICM":
        return {"type": "DICOM", "description": "DICOM medical imaging object"}
    if head.startswith(b"\x76\x2f\x31\x01"):
        return {"type": "OpenEXR", "description": "OpenEXR high-dynamic-range image"}
    if head.startswith(b"\x89HDF\r\n\x1a\n"):
        return {"type": "HDF5", "description": "HDF5 scientific data container"}
    if head[:3] == b"CDF" and head[3:4] in (b"\x01", b"\x02", b"\x05"):
        return {"type": "NetCDF", "description": "NetCDF scientific data container"}
    if head.startswith(b"#!AER-DAT"):
        return {"type": "AEDAT", "description": "Neuromorphic address-event (AER) stream"}
    if _tiff_has_tag(head, 34735):
        return {"type": "GeoTIFF", "description": "GeoTIFF georeferenced raster"}
    return None


def recognize_symbolic_music(file_path: Union[str, Path]) -> Optional[Dict[str, str]]:
    """Symbolic (non-waveform) music files: no acoustic authenticity can be scored from them."""
    try:
        with open(Path(file_path), "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if head.startswith(b"MThd"):
        return {"type": "MIDI", "description": "Standard MIDI file (symbolic music, not an audio waveform)"}
    return None
