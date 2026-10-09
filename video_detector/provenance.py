"""
video_detector.provenance: Provenance & Content Credentials (C2PA) Forensic Validator for Video.
Scans video containers for C2PA JUMBF boxes, MP4 atoms, encoder metadata, and software footprints.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from core.c2pa import scan_file as scan_c2pa_file
from core.provenance_view import build_c2pa_block, build_exif_block, build_provenance_view

logger = logging.getLogger("video_detector.provenance")


KNOWN_VIDEO_ATOMS = [b"ftyp", b"moov", b"mdat", b"udta", b"meta", b"mvhd", b"trak"]

# Vendor/generator name fragments that AI video tools commonly embed in container metadata
# (encoder/comment/title atoms, XMP, or similar) -- distinct from KNOWN_VIDEO_ATOMS above,
# which only identifies generic MP4 box *types*, never vendor identity. attribution.py's
# container-signature matching reads this field, not container_atoms.
KNOWN_VIDEO_GENERATOR_SIGNATURES = [
    "bytedance", "jimeng", "kling", "kuaishou", "runway", "gen-2", "gen-3",
    "sora", "openai", "google veo", "veo 2", "veo 3", "deepmind", "luma", "dream machine", "pika labs",
    "hailuo", "minimax",
    "tongyi", "wanxiang", "hunyuan", "vidu", "shengshu", "pixverse",
    "movie gen", "moviegen", "firefly", "nova reel", "novareel",
]


def _whole_word(word: str, text: str) -> bool:
    """True if ``word`` occurs in ``text`` not glued to other letters or digits. A short name such as "veo" or "dji" turns up by
    chance inside compressed video data; only a standalone occurrence, as in a metadata string, counts."""
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text) is not None


class VideoProvenanceValidator:
    """Validator for video container integrity, metadata atoms, and C2PA Content Credentials."""

    def __init__(self):
        pass

    @staticmethod
    def _container_camera_brand(head: bytes) -> Optional[str]:
        """Camera/device brand string found in the first container bytes (unauthenticated; None if absent)."""
        low = head.decode("latin-1").lower()
        for brand in ("Apple", "GoPro", "DJI", "Sony", "Canon", "Nikon", "Samsung", "Panasonic"):
            if _whole_word(brand.lower(), low):
                return brand
        return None

    def scan_c2pa(self, file_path: str | Path) -> Dict[str, Any]:
        """Content Credentials marker scan (presence only; see core.c2pa)."""
        return scan_c2pa_file(file_path)

    def analyze_provenance(self, file_path: str | Path) -> Dict[str, Any]:
        """Analyzes video atoms, encoder signatures, and cryptographic provenance."""
        path = Path(file_path)
        if not path.is_file():
            return {"valid": False, "error": "File not found"}

        c2pa_res = self.scan_c2pa(path)

        # Container inspection for atoms, and separately for vendor/generator name strings
        # that may appear in encoder/comment/title metadata atoms (never in the atom type
        # names themselves, which are the generic box headers in KNOWN_VIDEO_ATOMS).
        atoms_found = []
        vendor_signatures_found = []
        try:
            with open(path, "rb") as f:
                head = f.read(65536)
                file_size = path.stat().st_size
                tail_len = min(file_size, 128 * 1024)
                f.seek(max(0, file_size - tail_len))
                tail = f.read(tail_len)
                # moov/udta sit at the END of non-fast-start MP4s, so atom names are looked up in head and tail
                for atom in KNOWN_VIDEO_ATOMS:
                    if atom in head or atom in tail:
                        atoms_found.append(atom.decode("ascii", errors="ignore"))

            text_blob = (head + tail).decode("latin-1").lower()
            for sig in KNOWN_VIDEO_GENERATOR_SIGNATURES:
                if _whole_word(sig, text_blob):
                    vendor_signatures_found.append(sig)
        except Exception as exc:
            logger.debug("Container atom inspection bypassed for %s: %s", path, exc)

        cues = []
        if c2pa_res["c2pa_present"]:
            status = "C2PA_PROVENANCE_PRESENT"
            cues.append("C2PA Content Credentials markers found in video container (presence only; not cryptographically verified).")
        elif atoms_found:
            status = "STANDARD_CONTAINER_ATOMS"
            cues.append(f"Standard video atoms found: {', '.join(atoms_found)}")
        else:
            status = "UNKNOWN_CONTAINER"
            cues.append("Non-standard or stripped container atoms.")

        camera_make = self._container_camera_brand(head)
        view = build_provenance_view(
            build_c2pa_block(c2pa_res["c2pa_present"], c2pa_res.get("manifests_found", [])),
            build_exif_block(camera_make=camera_make),
        )
        return {
            **view,
            "c2pa_present": c2pa_res["c2pa_present"],
            "camera_make": camera_make,
            "provenance_status": status,
            "container_atoms": atoms_found,
            "vendor_signatures_found": vendor_signatures_found,
            "cues": cues,
        }
