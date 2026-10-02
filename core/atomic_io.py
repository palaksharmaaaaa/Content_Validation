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
from typing import Any, Dict

import time

logger = logging.getLogger("core.atomic_io")

_FILE_LOCKS: Dict[str, threading.RLock] = {}
_REGISTRY_LOCK = threading.Lock()


def _get_path_lock(target_path: Path) -> threading.RLock:
    """Returns a dedicated re-entrant lock for the specified file path."""
    canonical = str(target_path.resolve()).lower()
    with _REGISTRY_LOCK:
        if canonical not in _FILE_LOCKS:
            _FILE_LOCKS[canonical] = threading.RLock()
        return _FILE_LOCKS[canonical]


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
