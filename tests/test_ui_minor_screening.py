"""A minor-screening flag must be visible on the result page, and "nobody flagged" must not read as a guarantee."""
from streamlit.testing.v1 import AppTest


def _page(content):
    from ui.results.content import render_minor_screening
    render_minor_screening(content)


def _run(content):
    return AppTest.from_function(_page, args=(content,), default_timeout=60).run()


def test_flagged_image_shows_a_warning_with_the_apparent_age():
    at = _run({"minors": {"status": "OK", "review_required": True, "contains_minor": True, "youngest_age": 11.4, "n_subjects": 2}})
    assert len(at.warning) == 1
    text = at.warning[0].value
    assert "Likely minor" in text and "human reviewer" in text and "11" in text and "apparent" in text


def test_possible_minor_is_worded_as_possible():
    at = _run({"minors": {"status": "OK", "review_required": True, "contains_minor": False, "youngest_age": 22.0, "n_subjects": 1}})
    assert "Possible minor" in at.warning[0].value


def test_flagged_video_says_which_frames_and_when():
    minors = {"scope": "video_adaptive_frames", "status": "OK", "review_required": True, "contains_minor": True, "youngest_age": 9.0,
              "frames_flagged": 3, "frames_examined": 40, "priority": "SINGLE_FRAME",
              "flagged_frames": [{"time": 12.5}, {"time": 13.0}, {"time": 40.25}]}
    text = _run({"minors": minors}).warning[0].value
    assert "3 of 40" in text and "12.5s" in text and "single frame" in text


def test_clear_result_is_a_caption_that_admits_it_is_not_a_guarantee():
    at = _run({"minors": {"status": "OK", "review_required": False, "n_subjects": 3}})
    assert not at.warning
    assert "not a guarantee" in at.caption[0].value and "3 people checked" in at.caption[0].value


def test_unavailable_screening_is_reported_not_silently_clear():
    at = _run({"minors": {"status": "UNAVAILABLE", "review_required": True}})
    assert not at.warning and "did not run" in at.info[0].value


def test_no_people_and_missing_result():
    assert "no people found" in _run({"minors": {"status": "NO_PEOPLE"}}).caption[0].value
    at = _run({})
    assert not at.warning and not at.info and not at.caption
