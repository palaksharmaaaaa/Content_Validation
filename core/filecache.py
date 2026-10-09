"""core.filecache: tiny stat-keyed memoiser for expensive per-file parses (invalidates when the file changes)."""
from __future__ import annotations

import functools
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable


_KEY_TYPES = (int, float, str, bytes, bool, type(None))


def stat_cached(maxsize: int = 16) -> Callable:
    """Memoise ``fn(path, *args)`` per (path, size, mtime_ns, args). Results must be treated as read-only.

    Only plain arguments (numbers, strings, bytes, None) take part in the key. A call with any other argument is not cached at all:
    keying it by ``id()`` could return a stale result once the object is garbage-collected and its id reused."""

    def deco(fn: Callable) -> Callable:
        cache: "OrderedDict[tuple, Any]" = OrderedDict()
        lock = threading.Lock()

        @functools.wraps(fn)
        def wrapper(path, *args):
            p = Path(path)
            if not all(isinstance(a, _KEY_TYPES) for a in args):
                return fn(p, *args)
            st = p.stat()
            key = (str(p.resolve()), st.st_size, st.st_mtime_ns, *args)
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
