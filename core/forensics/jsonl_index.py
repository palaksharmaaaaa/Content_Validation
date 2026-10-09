"""core.forensics.jsonl_index: read an operator-supplied JSON-lines reference index once, and again only when the file changes.

Each analysed file is checked against the index; parsing a large index per file is the cost this removes. A line that does not
parse is skipped (and noted at debug level), never fatal.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

_cache: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}
_lock = threading.Lock()
_MAX_CACHED = 8


def load_index(path: Path, prepare: Callable[[Dict[str, Any]], None]) -> List[Dict[str, Any]]:
    """Rows of the index at ``path`` (empty when it does not exist). ``prepare(row)`` adds derived fields and raises for a bad row."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return []
    key = (str(path), getattr(prepare, "__qualname__", repr(prepare)), stat.st_mtime_ns, stat.st_size)
    with _lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            prepare(row)
            rows.append(row)
        except Exception as exc:
            logger.debug("load_index: skipped a line of %s (%s: %s)", path.name, type(exc).__name__, exc)
    with _lock:
        if len(_cache) >= _MAX_CACHED:
            _cache.clear()
        _cache[key] = rows
    return rows
