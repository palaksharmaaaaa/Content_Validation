"""Regressions for the findings of the independent security audit (Oct 2026)."""
import socket
import threading
import time

import pytest

from core import security
from core.security import SecureUrlFetcher, normalise_host, validate_secure_url


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8089\@example.com/x.png",        # urlparse sees example.com, requests connects to 127.0.0.1
    "http://127.0.0.1\.example.com/x",
    "http://user:pw@example.com/x.png",
    "http://exa mple.com/x",
    "http://example.com/x\r\nHost: evil",
    "http://localhost./x",                              # trailing dot
    "http://LOCALHOST/x",
    "http://app.localhost/x",
    "http://metadata.google.internal./x",
])
def test_ambiguous_or_internal_urls_are_refused(url):
    ok, msg, _ = validate_secure_url(url)
    assert ok is False, (url, msg)


def test_hosts_are_normalised_the_way_the_client_looks_them_up():
    assert normalise_host("Example.COM.") == "example.com"
    assert normalise_host("bücher.example") == "xn--bcher-kva.example"
    assert normalise_host("") is None


def test_the_pin_applies_to_the_punycode_name_of_an_idn_host(monkeypatch):
    calls = []

    def fake(host, *a, **k):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 80))]       # a rebinding answer

    monkeypatch.setattr(security, "_real_getaddrinfo", fake)
    with security._PinnedResolver("bücher.example", ["93.184.216.34"]):
        with pytest.raises(socket.gaierror):
            security._pinned_getaddrinfo("xn--bcher-kva.example", 80)                  # what urllib3 actually asks for
    assert calls == ["xn--bcher-kva.example"]


def test_a_pin_is_per_thread_and_holds_no_lock(monkeypatch):
    monkeypatch.setattr(security, "_real_getaddrinfo", lambda host, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 80))])
    other = {}

    def second():
        other["answer"] = security._pinned_getaddrinfo("example.org", 80)             # not pinned on this thread: untouched

    with security._PinnedResolver("example.org", ["93.184.216.34"]):
        t = threading.Thread(target=second)
        t.start()
        t.join(2)
        assert not t.is_alive()                                                        # a second thread is never blocked by the first
    assert other["answer"][0][4][0] == "10.0.0.5"


def test_a_server_that_trickles_headers_hits_the_deadline_and_blocks_nobody(monkeypatch, tmp_path):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    stop = threading.Event()

    def trickle():
        conn, _ = server.accept()
        conn.recv(4096)
        try:
            for ch in b"HTTP/1.1 200 OK\r\nX-Slow: ":
                if stop.is_set():
                    break
                conn.send(bytes([ch]))
                time.sleep(0.4)
        except OSError:
            pass
        conn.close()

    threading.Thread(target=trickle, daemon=True).start()
    port = server.getsockname()[1]
    monkeypatch.setattr(security, "validate_secure_url", lambda u: (True, "ok", ["127.0.0.1"]))         # reach the local test server
    fetcher = SecureUrlFetcher(max_mb=1, timeout_seconds=5, deadline_seconds=1.5)
    started = time.monotonic()
    result = fetcher.fetch(f"http://127.0.0.1:{port}/a.jpg", dest_dir=tmp_path)
    stop.set()
    server.close()
    assert result["success"] is False and "did not finish" in result["error"]
    assert time.monotonic() - started < 4
    assert not list(tmp_path.iterdir())
