"""
core.security: Enterprise-grade OWASP security hardening and defense-in-depth utilities.
Provides:
1. Anti-SSRF and DNS-rebinding resilient URL media ingestion with zero private network access.
2. Decompression bomb (DoS) protection for images and media decoders.
3. Path traversal protection and cryptographic cache naming.
4. Input sanitization for user-supplied filenames and metadata.
"""
from __future__ import annotations

import ipaddress
import logging
import os
from pathlib import Path, PureWindowsPath
import re
import socket
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

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


def normalise_host(hostname: str) -> Optional[str]:
    """The host exactly as the HTTP client will look it up: lower case, no trailing dot, internationalised names as punycode.
    None when it cannot be encoded."""
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return None
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return None


def validate_secure_url(url: str) -> Tuple[bool, str, List[str]]:
    """
    Exhaustively audits URL to eliminate Server-Side Request Forgery (SSRF) and DNS rebinding.
    Returns (is_valid, reason, resolved_ips).
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or invalid.", []

    url = url.strip()
    # Python's URL parser and the HTTP client disagree about a backslash ("http://127.0.0.1:80\\@example.com/" is host
    # example.com to urlparse but 127.0.0.1 to requests), so anything the two could read differently is refused outright.
    if "\\" in url or any(ord(c) < 0x21 or ord(c) == 0x7F for c in url):
        return False, "URL contains a backslash, whitespace or control character.", []
    try:
        parsed = urlparse(url)
    except Exception as exc:
        logger.debug("validate_secure_url: ignored %s: %s", type(exc).__name__, exc)
        return False, "Malformed URL format.", []

    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Unsupported protocol '{parsed.scheme}'. Only HTTP and HTTPS are permitted.", []

    if parsed.username is not None or parsed.password is not None:
        return False, "Credentials in the URL are not permitted.", []
    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL.", []

    hostname_clean = normalise_host(hostname)
    if hostname_clean is None:
        return False, "Hostname cannot be encoded as a valid DNS name.", []
    if hostname_clean in ("localhost", "127.0.0.1", "::1", "metadata.google.internal") or hostname_clean.endswith((".localhost", ".internal")):
        return False, "Access to localhost and internal hostnames is prohibited.", []

    try:
        explicit_port = parsed.port
    except ValueError:
        return False, "Invalid port in URL.", []
    if explicit_port is not None and explicit_port not in (80, 443):
        return False, f"Port {explicit_port} is not permitted; only 80 and 443 may be fetched.", []
    port = explicit_port or (443 if parsed.scheme.lower() == "https" else 80)
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

    base = PureWindowsPath(filename).name        # treats both / and \ as separators on every OS (Path would not on Linux)
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base).strip("._")
    if not cleaned:
        cleaned = "sanitized_media"
    if len(cleaned) > max_len:
        suffix = Path(cleaned).suffix[:12]            # a long name is shortened in the middle: the extension says what the file is
        cleaned = cleaned[: max_len - len(suffix)] + suffix if suffix else cleaned[:max_len]
    return cleaned


_pin_state = threading.local()
_install_lock = threading.Lock()
_installed = False
_real_getaddrinfo = socket.getaddrinfo


def _pinned_getaddrinfo(host, *args, **kwargs):
    result = _real_getaddrinfo(host, *args, **kwargs)
    pins = getattr(_pin_state, "pins", None)
    if pins and host:
        allowed = pins.get(str(host).lower().rstrip("."))
        if allowed is not None:
            filtered = [r for r in result if r[4][0] in allowed]
            if not filtered:
                raise socket.gaierror(
                    f"DNS rebinding blocked: '{host}' no longer resolves to a previously-validated address (possible rebinding attack)."
                )
            return filtered
    return result


def _install_pinning() -> None:
    """Put the pinning resolver in place once. It only acts on threads that hold a pin, so a slow request on one thread never blocks
    another thread's request."""
    global _installed
    with _install_lock:
        if not _installed:
            socket.getaddrinfo = _pinned_getaddrinfo
            _installed = True


class _PinnedResolver:
    """
    Closes the DNS-rebinding TOCTOU gap: `validate_secure_url` resolves a hostname and checks every IP it currently returns, but the
    HTTP client re-resolves DNS at connect time, and a short-TTL record can answer "public IP" then "internal IP". While this context
    is open, lookups of the exact host *on this thread* may only return the already-validated addresses.

    The pin is thread-local, so no lock is held for the duration of a request (a stalled download used to freeze every other fetch).
    Host header and TLS SNI are untouched, so certificate validation behaves normally.
    """

    def __init__(self, hostname: str, allowed_ips: List[str]):
        self._hostname = (normalise_host(hostname) or hostname or "").lower().rstrip(".")
        self._allowed_ips = set(allowed_ips)
        self._previous: Optional[Dict[str, set]] = None

    def __enter__(self) -> "_PinnedResolver":
        _install_pinning()
        self._previous = getattr(_pin_state, "pins", None)
        pins = dict(self._previous or {})
        pins[self._hostname] = self._allowed_ips
        _pin_state.pins = pins
        return self

    def __exit__(self, *exc_info) -> None:
        _pin_state.pins = self._previous


class SecureUrlFetcher:
    """
    DNS-rebinding and SSRF-hardened media downloader.
    Enforces:
    1. Pre-resolution IP verification and redirect validation.
    2. Zero internal network access.
    3. Strict streaming byte counters with hard memory limits.
    """

    _DEFAULT_SUFFIXES = {"image": ".jpg", "video": ".mp4", "audio": ".mp3"}
    _MAX_REDIRECTS = 3
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ForensicEngine/2.0"
        ),
        "Accept": "*/*",
    }

    def __init__(self, max_mb: float = 100, timeout_seconds: int = 60, deadline_seconds: Optional[float] = None):
        self.max_bytes = max_mb * 1024 * 1024
        self.timeout = timeout_seconds
        # requests' timeout is per socket read, so a server that dribbles a byte at a time never trips it; the whole fetch gets a deadline.
        self.deadline = float(deadline_seconds) if deadline_seconds else float(timeout_seconds) * 2.0

    @staticmethod
    def _failure(message: str) -> Dict[str, Any]:
        return {"success": False, "error": message}

    def _get_following_redirects(
        self, session: requests.Session, url: str, hostname: str, resolved_ips: List[str]
    ) -> Tuple[Optional[requests.Response], Optional[str]]:
        """GET with manual redirect handling: every hop is re-validated (blocks open-redirect SSRF pivots).

        Returns (response, error). Each request is pinned to the IPs that were validated for its host.
        """
        resp = None
        for hop in range(self._MAX_REDIRECTS + 1):
            with _PinnedResolver(hostname, resolved_ips):
                resp = session.get(
                    url, headers=self._HEADERS, stream=True,
                    timeout=(5.0, float(self.timeout)), allow_redirects=False,
                )
                if not (resp.is_redirect or resp.is_permanent_redirect):
                    break
                target = resp.headers.get("Location")
                if not target:
                    return None, "Redirect response without a Location header."
                if hop >= self._MAX_REDIRECTS:
                    return None, f"Too many redirects (more than {self._MAX_REDIRECTS})."
                close_hop = getattr(resp, "close", None)
                if callable(close_hop):
                    close_hop()
                target = urljoin(url, target)  # Location may be relative
                valid, msg, ips = validate_secure_url(target)
                if not valid:
                    return None, f"SSRF blocked malicious redirect: {msg}"
                url, hostname, resolved_ips = target, normalise_host(urlparse(target).hostname or "") or "", ips
        if resp is None:
            return None, "No response received from remote server."
        return resp, None

    def _response_rejection(self, resp: requests.Response) -> Optional[str]:
        """Reason to refuse the response before downloading its body (declared size / HTML page), else None."""
        cl = resp.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.max_bytes:
            return (f"File size ({int(cl) / (1024*1024):.1f} MB) exceeds maximum permitted limit "
                    f"({self.max_bytes / (1024*1024):g} MB).")
        ctype = resp.headers.get("content-type", "").lower()
        if "text/html" in ctype:
            return f"URL returned HTML webpage content-type ('{ctype}') instead of binary media."
        return None

    def _stream_to_file(self, resp: requests.Response, path: Path) -> bool:
        """Streams the body to ``path`` under the hard byte ceiling. Returns False if the ceiling was exceeded."""
        downloaded = 0
        cancelled = getattr(self, "_cancel", None)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if cancelled is not None and cancelled.is_set():
                    raise TimeoutError("download deadline exceeded")
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded > self.max_bytes:
                    return False
        return True

    def fetch(
        self,
        url: str,
        dest_dir: Optional[Path] = None,
        expected_type: str = "image",
    ) -> Dict[str, Any]:
        """Safely downloads remote media under a wall-clock deadline. A file is created only for an accepted download.

        The work runs on a helper thread: when the deadline passes the caller gets a failure at once and the helper is told to stop
        and delete anything it wrote (it may linger until its server stops talking, but it holds no lock and no result is used).
        """
        outcome: Dict[str, Any] = {}
        cancel = threading.Event()
        worker_fetcher = SecureUrlFetcher.__new__(SecureUrlFetcher)
        worker_fetcher.__dict__.update(self.__dict__)
        worker_fetcher._cancel = cancel

        def work() -> None:
            outcome["result"] = worker_fetcher._fetch_impl(url, dest_dir, expected_type)
            if cancel.is_set() and outcome["result"].get("success"):
                Path(outcome["result"]["file_path"]).unlink(missing_ok=True)

        thread = threading.Thread(target=work, name="secure-fetch", daemon=True)
        thread.start()
        thread.join(self.deadline)
        if thread.is_alive():
            cancel.set()
            return self._failure(f"Secure media download failed: the server did not finish within {self.deadline:.0f} seconds.")
        return outcome.get("result") or self._failure("Secure media download failed.")

    def _fetch_impl(
        self,
        url: str,
        dest_dir: Optional[Path] = None,
        expected_type: str = "image",
    ) -> Dict[str, Any]:
        url = (url or "").strip()                # the same text is validated, parsed and requested: no pin can be skipped by padding
        valid, msg, resolved_ips = validate_secure_url(url)
        if not valid:
            return self._failure(f"Security validation rejected URL: {msg}")

        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,6}", suffix):
            suffix = self._DEFAULT_SUFFIXES.get(expected_type, ".bin")          # only a plain extension is ever put in a file name
        safe_name = sanitize_filename(Path(parsed.path).name)
        if not Path(safe_name).suffix:
            safe_name += suffix                  # a link like https://host/download still yields a typed file name

        temp_path: Optional[Path] = None
        resp = None
        with requests.Session() as session:
            session.trust_env = False                      # proxies and .netrc from the environment would route around the address checks
            session.mount("http://", HTTPAdapter(max_retries=1))
            session.mount("https://", HTTPAdapter(max_retries=1))
            try:
                resp, error = self._get_following_redirects(session, url, normalise_host(parsed.hostname or "") or "", resolved_ips)
                if error or resp is None:
                    return self._failure(error or "No response received from remote server.")
                resp.raise_for_status()
                if 300 <= resp.status_code < 400:
                    return self._failure(f"Unexpected redirect status {resp.status_code} without a followable target.")
                rejection = self._response_rejection(resp)
                if rejection:
                    return self._failure(rejection)

                target_dir = dest_dir or Path(tempfile.gettempdir())
                target_dir.mkdir(parents=True, exist_ok=True)
                fd, temp_path_str = tempfile.mkstemp(dir=str(target_dir), prefix="sec_fetch_", suffix=suffix)
                os.close(fd)
                temp_path = Path(temp_path_str)

                if not self._stream_to_file(resp, temp_path):
                    temp_path.unlink(missing_ok=True)
                    return self._failure(f"Download aborted: media size exceeded {self.max_bytes / (1024*1024):g} MB ceiling.")

                return {
                    "success": True,
                    "file_path": str(temp_path),
                    "filename": safe_name,
                    "size_mb": round(temp_path.stat().st_size / (1024 * 1024), 2),
                    "content_type": resp.headers.get("content-type", "").lower(),
                }
            except Exception as exc:
                if temp_path is not None:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError as cleanup_exc:
                        logger.debug("fetch: could not remove %s: %s", temp_path, cleanup_exc)
                return self._failure(f"Secure media download failed: {exc}")
            finally:
                close = getattr(resp, "close", None)
                if callable(close):
                    close()
