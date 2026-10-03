import logging

from core.logging_filters import _BenignConnectionReset


def _record(msg, exc=None):
    return logging.LogRecord("asyncio", logging.ERROR, __file__, 1, msg, None, (type(exc), exc, None) if exc else None)


def test_drops_only_the_benign_reset():
    f = _BenignConnectionReset()
    reset = ConnectionResetError(10054, "forcibly closed")
    assert not f.filter(_record("Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)", reset))
    assert f.filter(_record("Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)", ValueError("real bug")))
    assert f.filter(_record("Task exception was never retrieved", reset))
