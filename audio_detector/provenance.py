"""
audio_detector.provenance: Provenance & Content Credentials (C2PA) Forensic Validator for Audio.
Scans audio stream chunks (RIFF, ID3, MP4 audio) for encoder provenance and C2PA credentials.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from core.c2pa import scan_file as scan_c2pa_file
from core.provenance_view import build_c2pa_block, build_exif_block, build_provenance_view

logger = logging.getLogger("audio_detector.provenance")


KNOWN_AUDIO_ENCODERS = ["lame", "lavf", "ffmpeg", "coreaudio", "audacity", "pro tools", "elevenlabs"]


class AudioProvenanceValidator:
    """Validator for audio container metadata, ID3/RIFF chunks, and C2PA credentials."""

    def __init__(self):
        pass

    def scan_c2pa(self, file_path: str | Path) -> Dict[str, Any]:
        """Content Credentials marker scan (presence only; see core.c2pa)."""
        return scan_c2pa_file(file_path)

    def analyze_provenance(self, file_path: str | Path) -> Dict[str, Any]:
        """Analyzes audio chunk headers, encoder footprints, and C2PA credentials."""
        path = Path(file_path)
        if not path.is_file():
            return {"valid": False, "error": "File not found"}

        c2pa_res = self.scan_c2pa(path)

        # Basic ID3 / RIFF header inspection
        encoder_found = None
        cues = []
        try:
            with open(path, "rb") as f:
                head = f.read(4096)
                head_str = head.decode("latin1", errors="ignore").lower()
                for enc in KNOWN_AUDIO_ENCODERS:
                    if enc in head_str:
                        encoder_found = enc.upper()
                        cues.append(f"Audio encoder footprint detected: '{encoder_found}'")
                        break
        except Exception as exc:
            logger.debug("analyze_provenance: ignored %s: %s", type(exc).__name__, exc)

        if c2pa_res["c2pa_present"]:
            status = "C2PA_PROVENANCE_PRESENT"
            cues.append("Content Credentials markers found in the audio file (presence only; not cryptographically verified).")
        elif encoder_found:
            status = "STANDARD_ENCODER_METADATA"
        else:
            status = "NO_ENCODER_METADATA"
            cues.append("No encoder software metadata found (neutral/stripped stream).")

        view = build_provenance_view(
            build_c2pa_block(c2pa_res["c2pa_present"], c2pa_res.get("manifests_found", [])),
            build_exif_block(),
        )
        return {
            **view,
            "c2pa_present": c2pa_res["c2pa_present"],
            "provenance_status": status,
            "encoder": encoder_found,
            "metadata": {"encoder": encoder_found},
            "cues": cues,
        }
