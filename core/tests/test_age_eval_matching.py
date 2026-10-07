"""The evaluation must match an annotated child to the subject that is that child, not to a larger overlapping adult."""
from services.age_eval import _best_match


def _s(face, person, assessment):
    return {"face_box": face, "person_box": person, "assessment": assessment}


def test_child_is_matched_by_face_not_by_the_adult_whose_body_box_overlaps():
    adult = _s([383, 101, 65, 98], [298, 74, 248, 790], "ADULT")
    child = _s([379, 367, 66, 70], [337, 320, 177, 570], "LIKELY_MINOR")
    best = _best_match([adult, child], face=[379, 367, 445, 437], person=[323, 317, 508, 896])
    assert best is child


def test_person_box_is_used_only_when_no_face_matches():
    body_only = _s(None, [2, 246, 1084, 455], "POSSIBLE_MINOR")
    assert _best_match([body_only], face=[616, 520, 904, 798], person=[2, 246, 1086, 701]) is body_only
    assert _best_match([body_only], face=[616, 520, 904, 798], person=[521, 450, 1095, 900]) is None


def test_nothing_found_is_none():
    assert _best_match([], face=[0, 0, 10, 10], person=None) is None
