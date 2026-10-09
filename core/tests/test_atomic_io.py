

def test_stale_session_folders_are_removed_and_fresh_or_foreign_ones_are_kept(tmp_path, monkeypatch):
    import os
    import time

    from core import atomic_io

    monkeypatch.setattr(atomic_io, "get_ephemeral_cache_dir", lambda: tmp_path)
    old, fresh, foreign = tmp_path / "session_old", tmp_path / "session_fresh", tmp_path / "not_a_session"
    for d in (old, fresh, foreign):
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "u.jpg").write_bytes(b"x")
    long_ago = time.time() - 3 * 86400
    for p in [old, old / "sub", old / "sub" / "u.jpg", foreign, foreign / "sub", foreign / "sub" / "u.jpg"]:
        os.utime(p, (long_ago, long_ago))
    assert atomic_io.purge_stale_sessions() == 1
    assert not old.exists() and fresh.exists() and foreign.exists()


def test_a_folder_with_one_recent_file_is_not_stale(tmp_path, monkeypatch):
    import os
    import time

    from core import atomic_io

    monkeypatch.setattr(atomic_io, "get_ephemeral_cache_dir", lambda: tmp_path)
    d = tmp_path / "session_active"
    d.mkdir()
    (d / "recent.png").write_bytes(b"x")
    os.utime(d, (time.time() - 5 * 86400,) * 2)               # the folder itself looks old, its newest file is not
    assert atomic_io.purge_stale_sessions() == 0 and d.exists()


def test_failed_write_reports_the_original_error_even_if_cleanup_fails(tmp_path, monkeypatch):
    from pathlib import Path

    from core import atomic_io

    monkeypatch.setattr(atomic_io.os, "replace", lambda *a, **k: (_ for _ in ()).throw(ValueError("replace failed")))
    real_unlink = Path.unlink

    def broken(self, *a, **k):
        if ".tmp_" in self.name:
            raise PermissionError("locked")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", broken)
    try:
        atomic_io.atomic_write_json(tmp_path / "x.json", {"a": 1})
    except ValueError as exc:
        assert "replace failed" in str(exc)
    else:
        raise AssertionError("the write error must propagate")
