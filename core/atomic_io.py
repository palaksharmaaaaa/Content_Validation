"""
core.atomic_io: Thread-safe and crash-resilient atomic file persistence utilities.
Guarantees:
1. Transactional writes (write to temp file in same directory -> fsync -> atomic os.replace).
2. Protection against partial writes, torn pages, and crashes.
3. Thread-safe synchronization via per-path re-entrant locks.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import time
import weakref
import functools
import contextlib

logger = logging.getLogger("core.atomic_io")

# Weak values: a path's lock exists only while some caller holds it, so the registry cannot grow without bound,
# yet two callers contending on the same path always share one lock (the holder keeps it alive).
_FILE_LOCKS: "weakref.WeakValueDictionary[str, threading.RLock]" = weakref.WeakValueDictionary()
_REGISTRY_LOCK = threading.Lock()


def _get_path_lock(target_path: Path) -> threading.RLock:
    """Returns a dedicated re-entrant lock for the specified file path."""
    canonical = str(target_path.resolve()).lower()
    with _REGISTRY_LOCK:
        lock = _FILE_LOCKS.get(canonical)
        if lock is None:
            lock = threading.RLock()
            _FILE_LOCKS[canonical] = lock
        return lock


def serialized_on(*path_attrs: str):
    """Method decorator: hold the per-path lock of each named instance attribute for the whole call.

    Makes a read-modify-write across several JSON files (feedback memory + calibration) atomic with respect to other
    threads using the same files. Locks are taken in sorted path order to avoid lock-order deadlocks.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            paths = sorted({Path(getattr(self, a)).resolve() for a in path_attrs}, key=str)
            locks = [_get_path_lock(p) for p in paths]
            with contextlib.ExitStack() as stack:
                for lock in locks:
                    stack.enter_context(lock)
                return fn(self, *args, **kwargs)

        return wrapper

    return deco


def atomic_write_json(file_path: str | Path, data: Any, indent: int = 2) -> None:
    """
    Atomically writes serializable Python data to a JSON file.
    Uses tempfile in the target directory followed by os.replace to guarantee
    that the target file is either fully updated or left untouched.
    """
    target = Path(file_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_path_lock(target)

    with lock:
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.tmp_",
            suffix=".json",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())

            # Atomic swap on same filesystem with Windows transient lock retry
            for attempt in range(12):
                try:
                    os.replace(tmp_path, target)
                    break
                except PermissionError:
                    if attempt == 11:
                        raise
                    time.sleep(0.015 * (attempt + 1))
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            logger.error("Failed atomic JSON write to %s: %s", target, exc)
            raise


def atomic_read_json(file_path: str | Path, default: Any = None) -> Any:
    """
    Thread-safely reads and parses a JSON file, returning `default` if the file
    does not exist or is corrupted.
    """
    target = Path(file_path).resolve()
    if not target.is_file():
        return default

    lock = _get_path_lock(target)
    with lock:
        for attempt in range(6):
            try:
                with open(target, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (PermissionError, json.JSONDecodeError) as exc:
                if attempt == 5:
                    logger.warning("Could not read JSON from %s (%s). Returning default.", target, exc)
                    return default
                time.sleep(0.01 * (attempt + 1))
        return default


def atomic_update_json(
    file_path: str | Path,
    updater: Any,
    default: Any = None,
) -> Any:
    """
    Transactionally reads, modifies, and writes JSON under a unified per-path lock.
    Eliminates read-modify-write lost-update race conditions in multi-threaded workflows.
    """
    target = Path(file_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_path_lock(target)
    with lock:
        current = atomic_read_json(target, default=default)
        updated = updater(current)
        atomic_write_json(target, updated)
        return updated


def get_ephemeral_cache_dir() -> Path:
    """
    Returns cross-platform ephemeral scratch space in OS temporary swap (e.g. %TEMP% or /tmp).
    Guarantees zero persistent media storage inside the project repository directory.
    """
    cache_dir = Path(tempfile.gettempdir()) / "omni_forensics_ephemeral_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_session_cache_dir(session_id: str) -> Path:
    """Per-session scratch directory so concurrent users never share (or wipe) each other's uploads."""
    safe = "".join(c for c in str(session_id) if c.isalnum() or c in "-_")[:64]
    if not safe:
        raise ValueError("session_id must contain at least one alphanumeric character")
    d = get_ephemeral_cache_dir() / f"session_{safe}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def purge_ephemeral_cache(cache_dir: Optional[Path] = None) -> int:
    """Purges all transient media files from OS temp cache with zero disk leak."""
    target_dir = cache_dir or get_ephemeral_cache_dir()
    purged = 0
    try:
        if target_dir.exists():
            for f in target_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink(missing_ok=True)
                        purged += 1
                    except Exception:
                        pass
    except Exception:
        pass
    return purged
