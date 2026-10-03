"""core.filecache: tiny stat-keyed memoiser for expensive per-file parses (invalidates when the file changes)."""
from __future__ import annotations

import functools
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable


def stat_cached(maxsize: int = 16) -> Callable:
    """Memoise ``fn(path, *args)`` per (path, size, mtime_ns, args). Results must be treated as read-only."""

    def deco(fn: Callable) -> Callable:
        cache: "OrderedDict[tuple, Any]" = OrderedDict()
        lock = threading.Lock()

        @functools.wraps(fn)
        def wrapper(path, *args):
            p = Path(path)
            st = p.stat()
            key = (str(p.resolve()), st.st_size, st.st_mtime_ns) + tuple(a if isinstance(a, (int, str, type(None))) else id(a) for a in args)
            with lock:
                if key in cache:
                    cache.move_to_end(key)
                    return cache[key]
            value = fn(p, *args)
            with lock:
                cache[key] = value
                while len(cache) > maxsize:
                    cache.popitem(last=False)
            return value

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return deco
