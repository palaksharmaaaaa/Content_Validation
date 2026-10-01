"""
audio_detector.downloader: Standalone remote audio downloader and URL validator.
Downloads audio files safely with timeout and size enforcement.
Completely self-contained with zero outside dependencies.
"""
from __future__ import annotations

import ipaddress
import logging
from pathlib import Path
import socket
import tempfile
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse
import requests

from audio_detector.config import MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS

logger = logging.getLogger("audio_detector.downloader")


def is_safe_url(url: str) -> Tuple[bool, str]:
    """Validates URL against SSRF attacks by enforcing schemes and blocking private/local IPs."""
    if not url or not isinstance(url, str):
        return False, "Empty or invalid URL."
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in ("http", "https"):
            return False, f"Unsupported URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
        hostname = parsed.hostname
        if not hostname:
            return False, "Missing or invalid hostname in URL."
        hostname_clean = hostname.strip().lower()
        if hostname_clean in ("localhost", "127.0.0.1", "::1"):
            return False, "Access to localhost or loopback address is restricted."
        try:
            addr_info = socket.getaddrinfo(hostname_clean, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip = ipaddress.ip_address(ip_str)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_reserved
                    or ip.is_multicast
                    or ip.is_unspecified
                ):
                    return False, f"Access to restricted network address ({ip_str}) is blocked."
        except socket.gaierror:
            return False, f"Could not resolve hostname '{hostname}'."
        return True, ""
    except Exception as e:
        return False, f"Malformed URL: {e}"


class AudioDownloader:
    """Safe downloader for remote audio recordings."""

    def __init__(self, max_size_mb: float = MAX_FILE_SIZE_MB, timeout_seconds: int = 45):
        self.max_size_mb = max_size_mb
        self.timeout_seconds = timeout_seconds

    def download_audio(self, url: str, destination_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Downloads an audio recording from a URL to a local destination file."""
        safe, err_msg = is_safe_url(url)
        if not safe:
            return {"success": False, "error": f"URL validation failed: {err_msg}", "path": None}

        parsed = urlparse(url)

        dest_dir = destination_dir or Path(tempfile.gettempdir())
        dest_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(parsed.path).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            suffix = ".wav"

        temp_target = dest_dir / f"downloaded_aud_{abs(hash(url)) % 10000000}{suffix}"

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            with requests.get(url, stream=True, timeout=self.timeout_seconds, headers=headers) as resp:
                resp.raise_for_status()

                content_len = resp.headers.get("Content-Length")
                if content_len and int(content_len) > self.max_size_mb * 1024 * 1024:
                    return {"success": False, "error": f"Audio exceeds size limit of {self.max_size_mb} MB", "path": None}

                bytes_downloaded = 0
                max_bytes = int(self.max_size_mb * 1024 * 1024)

                with open(temp_target, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            bytes_downloaded += len(chunk)
                            if bytes_downloaded > max_bytes:
                                f.close()
                                if temp_target.exists():
                                    temp_target.unlink()
                                return {"success": False, "error": f"Download exceeded size limit of {self.max_size_mb} MB", "path": None}
                            f.write(chunk)

            return {
                "success": True,
                "path": str(temp_target),
                "url": url,
                "file_size_bytes": bytes_downloaded,
                "file_size_mb": round(bytes_downloaded / (1024.0 * 1024.0), 3),
            }
        except Exception as e:
            logger.error("Audio download failed for %s: %s", url, e)
            if temp_target.exists():
                try:
                    temp_target.unlink()
                except OSError:
                    pass
            return {"success": False, "error": str(e), "path": None}
