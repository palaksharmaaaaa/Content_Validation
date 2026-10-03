"""core.logging_filters: silence a known-benign Windows asyncio log line without hiding real errors."""
from __future__ import annotations

import logging
import sys

_MARKER = "_call_connection_lost"


class _BenignConnectionReset(logging.Filter):
    """Drops only asyncio's Proactor 'connection lost' callback failure caused by a peer reset (WinError 10054).

    On Windows a browser tab closing or refreshing resets its socket; the Proactor loop then raises
    ConnectionResetError while shutting the already-dead socket down, and asyncio logs a traceback. Nothing is wrong
    with the app. Any other asyncio error passes through unchanged.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if _MARKER not in record.getMessage():
            return True
        exc = record.exc_info[1] if record.exc_info else None
        return not isinstance(exc, ConnectionResetError)


def install_benign_reset_filter() -> None:
    """Idempotent; a no-op off Windows."""
    if sys.platform != "win32":
        return
    logger = logging.getLogger("asyncio")
    if not any(isinstance(f, _BenignConnectionReset) for f in logger.filters):
        logger.addFilter(_BenignConnectionReset())
