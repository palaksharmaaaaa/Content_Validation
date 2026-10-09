from services.system_status import WARN, collect_status


def test_status_reports_every_component_with_plain_values():
    rows = collect_status()
    names = [r.component for r in rows]
    for label in ("Image", "Video", "Audio"):
        assert f"{label} model" in names and f"{label} calibration" in names
    assert "Hard-block list" in names and "ffmpeg" in names
    assert all(r.state and r.hint for r in rows)
    assert {r.level for r in rows} <= {"ok", "info", "warn"}


def test_blank_project_is_heuristic_not_an_error():
    model_rows = [r for r in collect_status() if r.component.endswith(" model")]
    assert all(r.level != WARN for r in model_rows)


def test_age_row_reports_missing_pointer_and_ready(tmp_path, monkeypatch):
    from core.perception import age
    from services import system_status

    deep = tmp_path / "a" / "b" / "c" / "model.safetensors"
    deep.parent.mkdir(parents=True)
    monkeypatch.setattr(age, "WEIGHTS_PATH", deep)
    assert system_status._age_row().state == "Weights missing"
    deep.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    row = system_status._age_row()
    assert row.state == "Git LFS pointer" and row.level == "warn" and "git lfs pull" in row.hint
    deep.write_bytes(b"x" * 5000)
    assert system_status._age_row().level == "ok"


def test_large_model_rows_check_the_pinned_revision_not_any_cached_copy(monkeypatch):
    from core.perception.detector import MODEL_REVISION as DETECTOR_REV
    from core.perception.recognizer import MODEL_REVISION as RECOGNIZER_REV
    from services import fetch_models, system_status

    asked = {}
    monkeypatch.setattr(fetch_models, "present", lambda repo, revision=None: asked.setdefault(repo, revision) or True)
    rows = {r.component: r for r in system_status.collect_status()}
    assert set(asked.values()) == {DETECTOR_REV, RECOGNIZER_REV}
    assert rows["Object detector"].state == "Ready" and rows["Scene and species recognizer"].state == "Ready"
