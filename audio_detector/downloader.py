"""
audio_detector.downloader: Standalone remote audio downloader and URL validator.
Delegates to core.security for zero-trust anti-SSRF protection, DNS-rebinding immunity,
and safe streaming byte limits.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.security import SecureUrlFetcher, validate_secure_url
from audio_detector.config import MAX_FILE_SIZE_MB

logger = logging.getLogger("audio_detector.downloader")


def is_safe_url(url: str) -> Tuple[bool, str]:
    """Validates URL against SSRF attacks using core zero-trust IP resolution."""
    valid, msg, _ = validate_secure_url(url)
    return valid, msg


class AudioDownloader:
    """Safe downloader for remote audio recordings."""

    def __init__(self, max_size_mb: float = MAX_FILE_SIZE_MB, timeout_seconds: int = 45):
        self.max_size_mb = max_size_mb
        self.timeout_seconds = timeout_seconds
        self._fetcher = SecureUrlFetcher(max_mb=int(max_size_mb), timeout_seconds=timeout_seconds)

    def download_audio(self, url: str, destination_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Downloads an audio track from a URL safely."""
        res = self._fetcher.fetch(url, dest_dir=destination_dir, expected_type="audio")
        if not res.get("success"):
            return {
                "success": False,
                "error": res.get("error", "Download failed"),
                "path": None,
            }

        file_path = res["file_path"]
        size_bytes = Path(file_path).stat().st_size if Path(file_path).is_file() else 0
        return {
            "success": True,
            "path": file_path,
            "url": url,
            "file_size_bytes": size_bytes,
            "file_size_mb": res.get("size_mb", 0.0),
        }
