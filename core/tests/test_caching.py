from core.filecache import stat_cached
from core.forensics.registry import CheckContext, ctx_memo


def test_stat_cached_hits_and_invalidates(tmp_path):
    calls = []

    @stat_cached(4)
    def parse(path):
        calls.append(1)
        return path.read_bytes()

    p = tmp_path / "f.bin"
    p.write_bytes(b"aa")
    assert parse(p) == b"aa" and parse(p) == b"aa" and len(calls) == 1
    p.write_bytes(b"bbb")
    assert parse(p) == b"bbb" and len(calls) == 2


def test_ctx_memo_runs_once_per_context(tmp_path):
    n = []

    @ctx_memo("k")
    def check(ctx):
        n.append(1)
        return len(n)

    ctx = CheckContext(path=tmp_path, modality="audio")
    assert check(ctx) == check(ctx) == 1
    assert check(CheckContext(path=tmp_path, modality="audio")) == 2
