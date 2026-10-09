"""SecureUrlFetcher behaviour with a faked HTTP layer: redirects are re-validated, nothing leaks on failure."""
import os

import pytest
import requests

from core import security
from core.security import SecureUrlFetcher


class FakeResp:
    def __init__(self, status=200, headers=None, chunks=(b"data",), location=None):
        self.status_code = status
        self.headers = dict(headers or {})
        if location:
            self.headers["Location"] = location
        self._chunks = chunks
        self.is_redirect = bool(location)
        self.is_permanent_redirect = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=0):
        yield from self._chunks


@pytest.fixture
def fetcher(monkeypatch):
    monkeypatch.setattr(security, "validate_secure_url", lambda u: ("evil" not in u, "blocked" if "evil" in u else "ok", ["93.184.216.34"]))
    monkeypatch.setattr(security, "_PinnedResolver", lambda host, ips: __import__("contextlib").nullcontext())
    return SecureUrlFetcher(max_mb=1)


def _serve(monkeypatch, responses):
    it = iter(responses)
    monkeypatch.setattr(requests.Session, "get", lambda self, *a, **k: next(it))


def test_success_writes_file_and_reports_metadata(fetcher, monkeypatch, tmp_path):
    _serve(monkeypatch, [FakeResp(headers={"content-type": "image/jpeg"}, chunks=(b"a" * 10, b"b" * 10))])
    r = fetcher.fetch("https://example.com/p/photo.jpg", dest_dir=tmp_path)
    assert r["success"] and r["filename"] == "photo.jpg" and r["content_type"] == "image/jpeg"
    assert os.path.getsize(r["file_path"]) == 20


def test_redirect_is_followed_and_revalidated(fetcher, monkeypatch, tmp_path):
    _serve(monkeypatch, [FakeResp(status=302, location="https://cdn.example.com/x.png"), FakeResp(headers={"content-type": "image/png"})])
    assert fetcher.fetch("https://example.com/a", dest_dir=tmp_path)["success"]
    _serve(monkeypatch, [FakeResp(status=302, location="http://evil.internal/x")])
    bad = fetcher.fetch("https://example.com/a", dest_dir=tmp_path)
    assert not bad["success"] and "SSRF blocked" in bad["error"]


@pytest.mark.parametrize("resp,needle", [
    (FakeResp(headers={"content-type": "text/html"}), "HTML"),
    (FakeResp(headers={"content-length": str(5 * 1024 * 1024)}), "exceeds maximum"),
    (FakeResp(status=404), "download failed"),
    (FakeResp(chunks=(b"x" * (600 * 1024), b"x" * (600 * 1024))), "ceiling"),
])
def test_rejections_leave_no_temp_files(fetcher, monkeypatch, tmp_path, resp, needle):
    _serve(monkeypatch, [resp])
    r = fetcher.fetch("https://example.com/a.jpg", dest_dir=tmp_path)
    assert not r["success"] and needle in r["error"]
    assert list(tmp_path.iterdir()) == []


def test_url_rejected_before_any_network(fetcher, tmp_path):
    r = fetcher.fetch("https://evil.example.com/a.jpg", dest_dir=tmp_path)
    assert not r["success"] and "rejected" in r["error"]


def test_relative_redirect_is_resolved(fetcher, monkeypatch, tmp_path):
    seen = []
    resps = iter([FakeResp(status=302, location="/media/real.jpg"), FakeResp(headers={"content-type": "image/jpeg"})])

    def get(self, url, **k):
        seen.append(url)
        return next(resps)

    monkeypatch.setattr(requests.Session, "get", get)
    assert fetcher.fetch("https://example.com/a", dest_dir=tmp_path)["success"]
    assert seen[1] == "https://example.com/media/real.jpg"


def test_exhausted_redirect_budget_is_an_error_not_a_download(fetcher, monkeypatch, tmp_path):
    _serve(monkeypatch, [FakeResp(status=302, location="https://example.com/next")] * 6)
    r = fetcher.fetch("https://example.com/a", dest_dir=tmp_path)
    assert not r["success"] and "redirect" in r["error"].lower()
    assert list(tmp_path.iterdir()) == []


def test_padded_url_is_pinned_to_its_real_host_and_filename_keeps_an_extension(monkeypatch, tmp_path):
    """Whitespace around a URL must not make the DNS pin apply to an empty host name."""
    pinned = []

    class Recorder:
        def __init__(self, host, ips):
            pinned.append(host)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(security, "validate_secure_url", lambda u: (True, "ok", ["93.184.216.34"]))
    monkeypatch.setattr(security, "_PinnedResolver", Recorder)
    _serve(monkeypatch, [FakeResp(headers={"content-type": "image/jpeg"})])
    r = SecureUrlFetcher(max_mb=1).fetch("  https://Example.com/download  ", dest_dir=tmp_path, expected_type="image")
    assert r["success"] and pinned == ["example.com"]
    assert r["filename"] == "download.jpg"


def test_response_is_closed_on_success_and_on_rejection(fetcher, monkeypatch, tmp_path):
    closed = []

    class Closing(FakeResp):
        def close(self):
            closed.append(1)

    _serve(monkeypatch, [Closing(headers={"content-type": "image/png"})])
    assert fetcher.fetch("https://example.com/a.png", dest_dir=tmp_path)["success"]
    _serve(monkeypatch, [Closing(headers={"content-type": "text/html"})])
    assert not fetcher.fetch("https://example.com/a.png", dest_dir=tmp_path)["success"]
    assert len(closed) == 2


def test_a_failing_cleanup_does_not_replace_the_real_error(fetcher, monkeypatch, tmp_path):
    from pathlib import Path

    _serve(monkeypatch, [FakeResp(headers={"content-type": "image/png"}, chunks=(b"x" * 10,))])
    monkeypatch.setattr(security.SecureUrlFetcher, "_stream_to_file", lambda self, resp, path: (_ for _ in ()).throw(OSError("disk full")))
    real_unlink = Path.unlink

    def broken_unlink(self, *a, **k):
        if self.name.startswith("sec_fetch_"):
            raise PermissionError("locked")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", broken_unlink)
    r = fetcher.fetch("https://example.com/a.png", dest_dir=tmp_path)
    assert r["success"] is False and "disk full" in r["error"]
