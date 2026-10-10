"""
core.hashing: the single streaming file-digest implementation.

Gates, validators, profilers and the media library all hash the same upload; digests are cached by
(path, size, mtime_ns) so each file is read once per analysis instead of once per consumer.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

_CHUNK = 1024 * 1024
_CACHE_MAX = 64
_cache: "OrderedDict[tuple, Tuple[str, str, int]]" = OrderedDict()
_lock = threading.Lock()


def _stat_key(path: Path) -> tuple:
    st = path.stat()
    return (str(path.resolve()), st.st_size, st.st_mtime_ns)


def _stream(path: Path) -> Tuple[str, str, int]:
    sha, md5, total = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            sha.update(chunk)
            md5.update(chunk)
            total += len(chunk)
    return sha.hexdigest(), md5.hexdigest(), total


def file_digests(path: str | Path, cached: bool = True) -> Tuple[str, str, int]:
    """Returns (sha256_hex, md5_hex, size_bytes). MD5 is for legacy display only, never for integrity decisions."""
    p = Path(path)
    if not cached:
        return _stream(p)
    key = _stat_key(p)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    value = _stream(p)
    with _lock:
        _cache[key] = value
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return value


def already_recorded(memory: list, path: str | Path, label: str) -> Tuple[bool, str | None]:
    """(True, digest) when ``memory`` already holds this file's content with this label. Calibration steps once per distinct file and
    label, so one reviewer repeating the same submission cannot push the thresholds to their limits. The digest is None if unreadable."""
    try:
        digest = file_sha256(path)
    except OSError:
        return False, None
    label = str(label).upper()
    return any(isinstance(r, dict) and r.get("sha256") == digest and str(r.get("user_label", "")).upper() == label for r in memory), digest


def file_sha256(path: str | Path, cached: bool = True) -> str:
    """Hex SHA-256 of a file. With ``cached=True`` the digest is reused while the file's size and mtime are unchanged."""
    return file_digests(path, cached=cached)[0]
