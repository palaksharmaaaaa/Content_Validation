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
