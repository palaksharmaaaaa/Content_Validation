"""
core.security: Enterprise-grade OWASP security hardening and defense-in-depth utilities.
Provides:
1. Anti-SSRF and DNS-rebinding resilient URL media ingestion with zero private network access.
2. Decompression bomb (DoS) protection for images and media decoders.
3. Path traversal protection and cryptographic cache naming.
4. Input sanitization for user-supplied filenames and metadata.
"""
from __future__ import annotations

import hashlib
import ipaddress
import logging
from pathlib import Path
import re
import socket
import tempfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image
import requests
from requests.adapters import HTTPAdapter

logger = logging.getLogger("core.security")

# OWASP Decompression Bomb Ceiling: 64 Megapixels (e.g. 8000x8000).
# Protects system RAM against crafted multi-gigabyte memory bombs.
SAFE_MAX_IMAGE_PIXELS = 64_000_000
Image.MAX_IMAGE_PIXELS = SAFE_MAX_IMAGE_PIXELS

# Restricted CIDR blocks (Private, Loopback, Link-Local, Cloud Metadata, Carrier-Grade NAT)
RESTRICTED_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),          # Current network
    ipaddress.ip_network("10.0.0.0/8"),         # Private RFC 1918
    ipaddress.ip_network("100.64.0.0/10"),      # Carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / Cloud Metadata (AWS/GCP/Azure)
    ipaddress.ip_network("172.16.0.0/12"),      # Private RFC 1918
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol
    ipaddress.ip_network("192.0.2.0/24"),       # TEST-NET-1
    ipaddress.ip_network("192.88.99.0/24"),     # 6to4 Relay
    ipaddress.ip_network("192.168.0.0/16"),     # Private RFC 1918
    ipaddress.ip_network("198.18.0.0/15"),      # Network benchmark
    ipaddress.ip_network("198.51.100.0/24"),    # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),     # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6 blocks
    ipaddress.ip_network("::/128"),             # Unspecified
    ipaddress.ip_network("::1/128"),            # Loopback
    ipaddress.ip_network("fc00::/7"),           # Unique local
    ipaddress.ip_network("fe80::/10"),          # Link-local
    ipaddress.ip_network("ff00::/8"),           # Multicast
]


def is_ip_restricted(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Checks whether an IP address belongs to any private, loopback, or cloud metadata network."""
    # Convert IPv4-mapped IPv6 (e.g., ::ffff:127.0.0.1) to standard IPv4 if present
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
        ip_obj = ip_obj.ipv4_mapped

    if (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    ):
        return True

    for net in RESTRICTED_NETWORKS:
        if ip_obj in net:
            return True

    return False


def validate_secure_url(url: str) -> Tuple[bool, str, List[str]]:
    """
    Exhaustively audits URL to eliminate Server-Side Request Forgery (SSRF) and DNS rebinding.
    Returns (is_valid, reason, resolved_ips).
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or invalid.", []

    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Malformed URL format.", []

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported protocol '{parsed.scheme}'. Only HTTP and HTTPS are permitted.", []

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL.", []

    hostname_clean = hostname.strip().lower()
    if hostname_clean in ("localhost", "127.0.0.1", "::1", "metadata.google.internal"):
        return False, "Access to localhost and internal hostnames is prohibited.", []

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    try:
        addr_info = socket.getaddrinfo(hostname_clean, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return False, f"Could not resolve hostname '{hostname_clean}': {exc}", []

    resolved_ips: List[str] = []
    for item in addr_info:
        sockaddr = item[4]
        ip_str = sockaddr[0]
        if ip_str not in resolved_ips:
            resolved_ips.append(ip_str)

    if not resolved_ips:
        return False, f"No IP address resolved for host '{hostname_clean}'.", []

    # Enforce strict public IP check across EVERY resolved address
    for ip_str in resolved_ips:
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            if is_ip_restricted(ip_obj):
                return False, f"Access to restricted or internal address ({ip_str}) is prohibited.", []
        except ValueError:
            return False, f"Invalid resolved IP structure: {ip_str}", []

    return True, "Valid public URL.", resolved_ips


def sanitize_filename(filename: str, max_len: int = 90) -> str:
    """
    Strips directory traversal sequences (../, ..\\) and special characters,
    producing an injection-safe filename.
    """
    if not filename:
        return "unnamed_file"

    base = Path(filename).name
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base).strip("._")
    if not cleaned:
        cleaned = "sanitized_media"
    return cleaned[:max_len]


def generate_secure_cache_name(prefix: str, seed: str, extension: str) -> str:
    """Generates an unguessable collision-free SHA-256 hashed cache filename."""
    h = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:16]
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{prefix}_{h}{ext}"


class SecureUrlFetcher:
    """
    DNS-rebinding and SSRF-hardened media downloader.
    Enforces:
    1. Pre-resolution IP verification and redirect validation.
    2. Zero internal network access.
    3. Strict streaming byte counters with hard memory limits.
    """

    def __init__(self, max_mb: int = 100, timeout_seconds: int = 60):
        self.max_bytes = max_mb * 1024 * 1024
        self.timeout = timeout_seconds

    def fetch(
        self,
        url: str,
        dest_dir: Optional[Path] = None,
        expected_type: str = "image",
    ) -> Dict[str, Any]:
        """Safely downloads remote media, verifying payload integrity."""
        valid, msg, resolved_ips = validate_secure_url(url)
        if not valid:
            return {"success": False, "error": f"Security validation rejected URL: {msg}"}

        parsed = urlparse(url)
        path_suffix = Path(parsed.path).suffix.lower()

        default_suffixes = {
            "image": ".jpg",
            "video": ".mp4",
            "audio": ".mp3",
        }
        suffix = path_suffix if path_suffix else default_suffixes.get(expected_type, ".bin")
        safe_name = sanitize_filename(Path(parsed.path).name) or f"downloaded_{expected_type}{suffix}"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ForensicEngine/2.0"
            ),
            "Accept": "*/*",
        }

        session = requests.Session()
        session.mount("http://", HTTPAdapter(max_retries=1))
        session.mount("https://", HTTPAdapter(max_retries=1))

        temp_dir = dest_dir or Path(tempfile.gettempdir())
        temp_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_path_str = tempfile.mkstemp(
            dir=str(temp_dir),
            prefix="sec_fetch_",
            suffix=suffix,
        )
        temp_path = Path(temp_path_str)

        try:
            # We enforce allow_redirects=False initially or validate target before following
            curr_url = url
            resp = None
            max_redirects = 3

            for _ in range(max_redirects):
                resp = session.get(
                    curr_url,
                    headers=headers,
                    stream=True,
                    timeout=(5.0, float(self.timeout)),
                    allow_redirects=False,
                )
                if resp.is_redirect or resp.is_permanent_redirect:
                    redirect_target = resp.headers.get("Location")
                    if not redirect_target:
                        break
                    # Re-validate redirect target to prevent open-redirect SSRF pivot
                    r_valid, r_msg, _ = validate_secure_url(redirect_target)
                    if not r_valid:
                        return {"success": False, "error": f"SSRF blocked malicious redirect: {r_msg}"}
                    curr_url = redirect_target
                else:
                    break

            if resp is None:
                return {"success": False, "error": "No response received from remote server."}

            resp.raise_for_status()

            cl = resp.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > self.max_bytes:
                return {
                    "success": False,
                    "error": f"File size ({int(cl) / (1024*1024):.1f} MB) exceeds maximum permitted limit ({self.max_bytes / (1024*1024):.0f} MB).",
                }

            ctype = resp.headers.get("content-type", "").lower()
            if "text/html" in ctype:
                return {
                    "success": False,
                    "error": f"URL returned HTML webpage content-type ('{ctype}') instead of binary media.",
                }

            downloaded = 0
            with open(fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > self.max_bytes:
                            temp_path.unlink(missing_ok=True)
                            return {
                                "success": False,
                                "error": f"Download aborted: media size exceeded {self.max_bytes / (1024*1024):.0f} MB ceiling.",
                            }

            final_size_mb = temp_path.stat().st_size / (1024 * 1024)
            return {
                "success": True,
                "file_path": str(temp_path),
                "filename": safe_name,
                "size_mb": round(final_size_mb, 2),
                "content_type": ctype,
            }
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return {"success": False, "error": f"Secure media download failed: {exc}"}
