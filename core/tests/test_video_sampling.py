"""Adaptive video sampling: content changes and the one-per-second grid are examined, static footage is not re-examined."""
import cv2
import numpy as np
import pytest

from core.perception import age_video as V
from core.perception import video_sampling as S

W, H, FPS = 160, 90, 10


def _video(tmp_path, seconds, frame_at, name="v.mp4"):
    path = str(tmp_path / name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(int(seconds * FPS)):
        vw.write(frame_at(i / FPS))
    vw.release()
    return path


def _grey():
    return np.full((H, W, 3), 110, np.uint8)


class _Marker:
    """An estimator that 'sees a child' in any frame containing a bright red patch, and counts how often it is asked."""

    def __init__(self):
        self.calls = 0

    def assess(self, frame):
        self.calls += 1
        red = bool(np.mean(frame[:, :, 2].astype(int) - frame[:, :, 0].astype(int)) > 12)
        return {"status": "OK", "review_required": red, "youngest_age": 12.0 if red else None, "contains_minor": red,
                "subjects": [{"assessment": "LIKELY_MINOR" if red else "ADULT"}], "model": "stub"}


def test_a_cut_is_a_content_change_and_its_first_frames_are_chosen(tmp_path):
    def frame(t):
        f = _grey()
        if 17 <= t < 19:
            f[:] = (30, 30, 220)                              # a whole different scene for two seconds
        return f
    probes = S.probe_video(_video(tmp_path, 40, frame))
    assert len(S.find_changes(probes)) >= 2                    # into the scene and back out
    times, info = S.choose_times(probes, max_frames=60)
    assert any(17 <= t < 19 for t in times) and info["changes"] >= 2


def test_a_small_local_appearance_in_a_continuous_shot_is_caught_by_the_grid(tmp_path):
    def frame(t):
        f = _grey()
        if 10.0 <= t < 11.0:
            f[30:70, 100:140] = (20, 20, 230)                 # a one-second inset that barely changes the whole picture
        return f
    path = _video(tmp_path, 30, frame)
    r = V.screen_video_file(path, estimator=_Marker(), max_frames=60)
    assert r["review_required"] and r["frames_flagged"] >= 1
    assert r["sampling"]["longest_unexamined_seconds"] <= 1.6


def test_the_budget_is_respected_and_the_gap_is_reported(tmp_path):
    path = _video(tmp_path, 200, lambda t: _grey())
    probes = S.probe_video(path)
    times, _ = S.choose_times(probes, max_frames=20)
    assert 2 <= len(times) <= 20
    assert S.longest_gap(times, probes.duration) > 5.0


def test_static_footage_is_assessed_once_and_reused(tmp_path):
    est = _Marker()
    r = V.screen_video_file(_video(tmp_path, 30, lambda t: _grey()), estimator=est, max_frames=60)
    assert est.calls <= 2 and r["sampling"]["frames_reused"] >= r["sampling"]["frames_examined"] - 2
    assert not r["review_required"] and r["priority"] == "NONE"


def test_a_flagged_scene_that_lasts_is_repeated(tmp_path):
    def frame(t):
        f = _grey()
        if 5 <= t < 15:
            f[:] = (30, 30, 220)
        return f
    r = V.screen_video_file(_video(tmp_path, 30, frame), estimator=_Marker(), max_frames=60)
    assert r["priority"] == "REPEATED" and r["contains_minor"]


def test_an_unreadable_video_is_never_reported_clear(tmp_path):
    r = V.screen_video_file(str(tmp_path / "missing.mp4"), estimator=_Marker())
    assert r["status"] == "NO_FRAMES" and r["review_required"] is True


def test_video_content_analyzer_falls_back_to_given_frames_when_the_file_is_unreadable(monkeypatch):
    from video_detector.content import VideoContentAnalyzer

    monkeypatch.setattr("video_detector.content.screen_video_frames", lambda frames: {"status": "OK", "marker": "from frames"})
    out = VideoContentAnalyzer._screen_minors([np.zeros((8, 8, 3), np.uint8)], "no/such/file.mp4")
    assert out["marker"] == "from frames"
