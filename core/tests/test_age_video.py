"""Video minor screening: frame choice, aggregation, and that a flagged frame clears nothing."""
import numpy as np

from core.perception import age_video as V


class _Stub:
    """Returns a scripted assessment per call."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def assess(self, frame):
        r = self.script[self.calls]
        self.calls += 1
        return r


def _res(flag=False, age=40.0, likely=False, status="OK"):
    subs = [{"assessment": "LIKELY_MINOR" if likely else "POSSIBLE_MINOR" if flag else "ADULT"}] if status == "OK" else []
    return {"status": status, "review_required": flag, "youngest_age": age, "contains_minor": likely, "subjects": subs, "model": "m@x"}


def test_pick_frames_spreads_evenly_and_keeps_ends():
    assert V.pick_frames(5, 16) == [0, 1, 2, 3, 4]
    p = V.pick_frames(100, 5)
    assert p[0] == 0 and p[-1] == 99 and len(p) == 5 and p == sorted(p)
    assert V.pick_frames(0, 5) == []


def test_clear_video_is_priority_none():
    frames = [np.zeros((8, 8, 3), np.uint8)] * 4
    r = V.screen_video_frames(frames, estimator=_Stub([_res()] * 4))
    assert r["priority"] == "NONE" and not r["review_required"] and r["frames_examined"] == 4


def test_single_flagged_frame_still_sends_the_video_to_review():
    frames = [np.zeros((8, 8, 3), np.uint8)] * 10
    script = [_res()] * 10
    script[6] = _res(flag=True, age=19.0)
    r = V.screen_video_frames(frames, timestamps=list(range(10)), estimator=_Stub(script))
    assert r["review_required"] and r["priority"] == "SINGLE_FRAME" and r["flagged_frames"][0]["time"] == 6
    assert r["youngest_age"] == 19.0


def test_repeated_sighting_is_prioritised_and_likely_minor_propagates():
    frames = [np.zeros((8, 8, 3), np.uint8)] * 8
    script = [_res(flag=True, age=12.0, likely=True)] * 4 + [_res()] * 4
    r = V.screen_video_frames(frames, estimator=_Stub(script))
    assert r["priority"] == "REPEATED" and r["contains_minor"] and r["frames_flagged"] == 4


def test_no_frames_is_never_reported_clear():
    r = V.screen_video_frames([], estimator=_Stub([]))
    assert r["status"] == "NO_FRAMES" and r["review_required"] is True


def test_unavailable_model_fails_closed():
    r = V.screen_video_frames([np.zeros((8, 8, 3), np.uint8)], estimator=_Stub([_res(status="UNAVAILABLE", age=None)]))
    assert r["status"] == "UNAVAILABLE" and r["review_required"] is True


def test_an_unreadable_video_is_never_reported_clear_even_if_probing_raises(monkeypatch):
    from core.perception import age_video, video_sampling

    monkeypatch.setattr(video_sampling, "probe_video", lambda p: (_ for _ in ()).throw(RuntimeError("codec exploded")))
    r = age_video.screen_video_file("whatever.mp4")
    assert r["status"] == "NO_FRAMES" and r["review_required"] is True and "RuntimeError" in r["reason"]


def test_a_frame_that_could_not_be_screened_makes_the_video_unavailable_not_clear():
    import numpy as np

    from core.perception import age_video

    class Est:
        def assess(self, frame):
            return {"status": "UNAVAILABLE", "youngest_age": None, "contains_minor": False, "review_required": True, "subjects": [], "model": "m"}

    r = age_video.screen_video_frames([np.zeros((50, 50, 3), np.uint8)] * 3, estimator=Est())
    assert r["status"] == "UNAVAILABLE" and r["review_required"] is True and r["frames_flagged"] == 3
