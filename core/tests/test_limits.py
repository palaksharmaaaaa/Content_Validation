import pytest

from core.limits import DEFAULTS, limits_for


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for name in list(__import__("os").environ):
        if name.startswith("OMNIFORENSICS_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OMNIFORENSICS_LIMITS_FILE", str(tmp_path / "none.toml"))     # an absent file: the built-in defaults apply


def test_defaults_are_the_requested_limits():
    assert (limits_for("image").file_mb, limits_for("image").batch_mb) == (10, 500)
    assert (limits_for("video").file_mb, limits_for("video").batch_mb) == (100, 500)
    assert (limits_for("audio").file_mb, limits_for("audio").batch_mb) == (10, 500)
    assert limits_for("video").file_bytes == 100 * 1024 * 1024


def test_the_shipped_limits_file_matches_the_defaults(monkeypatch):
    from core.limits import LIMITS_FILE

    monkeypatch.setenv("OMNIFORENSICS_LIMITS_FILE", str(LIMITS_FILE))
    for modality, values in DEFAULTS.items():
        assert (limits_for(modality).file_mb, limits_for(modality).batch_mb) == (values["file_mb"], values["batch_mb"])


def test_file_then_environment_override_the_defaults(monkeypatch, tmp_path):
    f = tmp_path / "limits.toml"
    f.write_text("[image]\nfile_mb = 20\nbatch_mb = 40\n[audio]\nfile_mb = 5\n")
    monkeypatch.setenv("OMNIFORENSICS_LIMITS_FILE", str(f))
    assert (limits_for("image").file_mb, limits_for("image").batch_mb) == (20, 40)
    assert (limits_for("audio").file_mb, limits_for("audio").batch_mb) == (5, 500)          # unset keys keep their default
    monkeypatch.setenv("OMNIFORENSICS_MAX_IMAGE_FILE_MB", "30")
    assert limits_for("image").file_mb == 30 and limits_for("image").batch_mb == 40


@pytest.mark.parametrize("bad", ["0", "-5", "nan", "inf", "ten", ""])
def test_a_bad_value_is_ignored_not_trusted(monkeypatch, bad):
    monkeypatch.setenv("OMNIFORENSICS_MAX_VIDEO_FILE_MB", bad)
    assert limits_for("video").file_mb == 100


def test_a_broken_limits_file_falls_back_to_defaults(monkeypatch, tmp_path):
    f = tmp_path / "limits.toml"
    f.write_text("[image\nfile_mb = ")
    monkeypatch.setenv("OMNIFORENSICS_LIMITS_FILE", str(f))
    assert limits_for("image").file_mb == 10


def test_unknown_modality_is_refused():
    with pytest.raises(ValueError):
        limits_for("text")
