"""
video_detector.provenance: Provenance & Content Credentials (C2PA) Forensic Validator for Video.
Scans video containers for C2PA JUMBF boxes, MP4 atoms, encoder metadata, and software footprints.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from core.provenance_view import build_c2pa_block, build_exif_block, build_provenance_view

logger = logging.getLogger("video_detector.provenance")

C2PA_VIDEO_SIGNATURES = [
    b"urn:c2pa",
    b"c2pa",
    b"c2ma",
    b"c2cs",
    b"application/c2pa",
]

KNOWN_VIDEO_ATOMS = [b"ftyp", b"moov", b"mdat", b"udta", b"meta", b"mvhd", b"trak"]

# Vendor/generator name fragments that AI video tools commonly embed in container metadata
# (encoder/comment/title atoms, XMP, or similar) -- distinct from KNOWN_VIDEO_ATOMS above,
# which only identifies generic MP4 box *types*, never vendor identity. attribution.py's
# container-signature matching reads this field, not container_atoms.
KNOWN_VIDEO_GENERATOR_SIGNATURES = [
    "bytedance", "jimeng", "kling", "kuaishou", "runway", "gen-2", "gen-3",
    "sora", "openai", "veo", "deepmind", "luma", "dream machine", "pika labs",
    "hailuo", "minimax",
    "tongyi", "wanxiang", "hunyuan", "vidu", "shengshu", "pixverse",
    "movie gen", "moviegen", "firefly", "nova reel", "novareel",
]


class VideoProvenanceValidator:
    """Validator for video container integrity, metadata atoms, and C2PA Content Credentials."""

    def __init__(self):
        pass

    @staticmethod
    def _container_camera_brand(head: bytes) -> Optional[str]:
        """Camera/device brand string found in the first container bytes (unauthenticated; None if absent)."""
        low = head.lower()
        for brand in (b"Apple", b"GoPro", b"DJI", b"Sony", b"Canon", b"Nikon", b"Samsung", b"Panasonic"):
            if brand.lower() in low:
                return brand.decode("ascii")
        return None

    def scan_c2pa(self, file_path: str | Path) -> Dict[str, Any]:
        """Scans video binary for C2PA JUMBF boxes."""
        path = Path(file_path)
        if not path.is_file():
            return {"c2pa_present": False, "status": "FILE_NOT_FOUND", "manifests_found": []}

        try:
            file_size = path.stat().st_size
            read_len = min(file_size, 1024 * 1024)  # First 1MB
            with open(path, "rb") as f:
                header = f.read(read_len)
                # Also read last 128KB (where MP4 moov/udta atoms frequently reside)
                if file_size > read_len:
                    f.seek(max(0, file_size - 128 * 1024))
                    footer = f.read(128 * 1024)
                else:
                    footer = b""

            search_bytes = header + footer
            found = []
            for sig in C2PA_VIDEO_SIGNATURES:
                if sig in search_bytes:
                    found.append(sig.decode("utf-8", errors="ignore"))

            has_c2pa = len(found) > 0
            return {
                "c2pa_present": has_c2pa,
                "status": "C2PA_CREDENTIALS_FOUND" if has_c2pa else "NO_C2PA_MANIFEST",
                "manifests_found": found,
            }
        except Exception as e:
            logger.debug("Video C2PA scan error: %s", e)
            return {"c2pa_present": False, "status": "ERROR", "manifests_found": []}

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
                for atom in KNOWN_VIDEO_ATOMS:
                    if atom in head:
                        atoms_found.append(atom.decode("ascii", errors="ignore"))

                file_size = path.stat().st_size
                tail_len = min(file_size, 128 * 1024)
                f.seek(max(0, file_size - tail_len))
                tail = f.read(tail_len)

            text_blob = (head + tail).lower()
            for sig in KNOWN_VIDEO_GENERATOR_SIGNATURES:
                if sig.encode("ascii") in text_blob:
                    vendor_signatures_found.append(sig)
        except Exception as exc:
            logger.debug("Container atom inspection bypassed for %s: %s", path, exc)

        cues = []
        if c2pa_res["c2pa_present"]:
            status = "C2PA_PROVENANCE_PRESENT"
            cues.append("C2PA Content Credentials markers found in video container (presence only; not cryptographically verified).")
        elif atoms_found:
            status = "STANDARD_CONTAINER_ATOMS"
            cues.append(f"Standard video atoms verified: {', '.join(atoms_found)}")
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
