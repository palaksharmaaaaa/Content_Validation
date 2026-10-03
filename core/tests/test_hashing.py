import hashlib

from core import hashing
from core.hashing import file_digests, file_sha256


def test_digests_match_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    data = b"x" * 3_000_000 + b"tail"
    p.write_bytes(data)
    sha, md5, size = file_digests(p)
    assert sha == hashlib.sha256(data).hexdigest() and md5 == hashlib.md5(data).hexdigest() and size == len(data)


def test_cache_hits_and_invalidates_on_change(tmp_path, monkeypatch):
    p = tmp_path / "b.bin"
    p.write_bytes(b"one")
    calls = []
    real = hashing._stream
    monkeypatch.setattr(hashing, "_stream", lambda q: calls.append(q) or real(q))
    first = file_sha256(p)
    assert file_sha256(p) == first and len(calls) == 1
    p.write_bytes(b"two!")
    assert file_sha256(p) == hashlib.sha256(b"two!").hexdigest() and len(calls) == 2


def test_uncached_always_streams(tmp_path, monkeypatch):
    p = tmp_path / "c.bin"
    p.write_bytes(b"abc")
    calls = []
    real = hashing._stream
    monkeypatch.setattr(hashing, "_stream", lambda q: calls.append(q) or real(q))
    file_sha256(p, cached=False)
    file_sha256(p, cached=False)
    assert len(calls) == 2
