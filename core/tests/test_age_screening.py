"""Age / minor screening: the pairing of faces with bodies, the recall-first rule, and that the model loads and runs."""
import numpy as np
import pytest

from core.perception import age as A


def test_face_joins_smallest_containing_person_and_unmatched_stay_one_sided():
    faces = [(110, 20, 30, 30), (500, 500, 30, 30)]
    persons = [(0, 0, 400, 400), (100, 10, 80, 200), (700, 0, 50, 100)]
    pairs = A.pair_faces_with_persons(faces, persons)
    assert (0, 1) in pairs                  # the face sits in both big and small boxes: the smaller one wins
    assert (1, None) in pairs               # a face inside no person box is still a subject
    assert (None, 0) in pairs and (None, 2) in pairs     # persons without a face are aged from the body
    assert len(pairs) == 4


def test_one_person_takes_only_one_face():
    pairs = A.pair_faces_with_persons([(10, 10, 40, 40), (60, 10, 20, 20)], [(0, 0, 100, 100)])
    assert sorted(pairs, key=str) == sorted([(0, 0), (1, None)], key=str)


@pytest.mark.parametrize("age,share,usable,expected", [
    (9.0, None, True, "LIKELY_MINOR"),
    (17.9, 0.0, True, "LIKELY_MINOR"),
    (30.0, 0.8, True, "POSSIBLE_MINOR"),        # the age-group opinion alone is enough for a review
    (21.0, 0.0, True, "POSSIBLE_MINOR"),        # inside the safety margin above 18
    (26.9, 0.0, True, "POSSIBLE_MINOR"),
    (27.0, 0.0, True, "ADULT"),
    (40.0, 0.3, True, "ADULT"),
    (40.0, 0.05, True, "ADULT"),
    (40.0, None, True, "ADULT"),
    (None, None, False, "UNDETERMINED"),        # nothing usable: a person has to look
])
def test_recall_first_rule(age, share, usable, expected):
    assert A.classify_minor(age, share, usable) == expected


def test_missing_crop_is_the_normalised_black_image():
    z = A._letterbox_normalise(None)
    assert z.shape == (3, 384, 384)
    assert z[0, 0, 0] == pytest.approx((0 - A.MEAN[0]) / A.STD[0])


def test_letterbox_keeps_aspect_ratio():
    out = A._letterbox_normalise(np.full((100, 50, 3), 255, np.uint8))
    assert out.shape == (3, 384, 384)
    inside = out[:, 192, 192]
    outside = out[:, 192, 5]                   # the padding column is the black image
    assert inside[0] > outside[0]


def test_model_loads_and_ages_a_synthetic_person():
    est = A.AgeEstimator()
    if not est.available:
        pytest.skip("age weights not present")
    img = np.full((480, 320, 3), 128, np.uint8)
    r = est.assess(img, faces=[(110, 60, 100, 120)], persons=[(40, 20, 240, 440)])
    assert r["status"] == "OK" and r["n_subjects"] == 1
    assert 0.0 <= r["subjects"][0]["age"] <= 122.0
    assert r["subjects"][0]["evidence"] == "face+body"


def test_unavailable_model_fails_closed(tmp_path):
    est = A.AgeEstimator(weights=tmp_path / "missing.safetensors")
    r = est.assess(np.zeros((200, 200, 3), np.uint8), faces=[(50, 50, 60, 60)], persons=[])
    assert r["status"] == "UNAVAILABLE" and r["review_required"] is True and r["contains_minor"] is False


def test_no_people_means_nothing_to_review():
    r = A.AgeEstimator().assess(np.zeros((200, 200, 3), np.uint8), faces=[], persons=[])
    assert r["status"] == "NO_PEOPLE" and r["review_required"] is False


class _FakeFinder:
    def __init__(self, hits):
        self.hits = hits

    def find(self, image):
        return []

    def find_rotated(self, image, rotations=(90, 270, 180)):
        return self.hits


def _estimator_with(monkeypatch, variant_ages, shares=None, hits=None):
    """An AgeEstimator whose model is replaced by scripted ages (one per (face, body) variant, in order)."""
    from core.perception import face_scan

    monkeypatch.setattr(face_scan, "get_screening_finder", lambda: _FakeFinder(hits if hits is not None else []))
    est = A.AgeEstimator()
    monkeypatch.setattr(est, "_ensure", lambda: object())
    monkeypatch.setattr(est, "_ages", lambda faces, bodies: list(variant_ages)[: len(faces)])
    monkeypatch.setattr(A.AgeEstimator, "_age_group_shares", staticmethod(lambda crops: list(shares or [0.0] * len(crops))[: len(crops)]))
    return est


_HIT = [{"box": (50, 40, 60, 60), "rot": 270, "rot_box": (40, 50, 60, 60), "score": 0.9}]
_IMG = np.full((400, 300, 3), 120, np.uint8)
_PERSON = (20, 20, 200, 360)


def test_rotated_face_found_in_a_person_without_an_upright_face_lowers_the_age(monkeypatch):
    est = _estimator_with(monkeypatch, variant_ages=[40.0, 15.0], hits=_HIT)       # body-only says 40, the rotated face says 15
    r = est.assess(_IMG, faces=[], persons=[_PERSON])
    s = r["subjects"][0]
    assert s["age"] == 15.0 and s["assessment"] == "LIKELY_MINOR" and s["rotation"] == 270
    assert s["evidence"].startswith("face+body (rotated") and s["face_box"] == [60, 42, 60, 60]


def test_a_false_rotated_face_cannot_remove_a_flag(monkeypatch):
    est = _estimator_with(monkeypatch, variant_ages=[20.0, 45.0], hits=_HIT)       # body says 20; the (false) rotated face says 45
    s = est.assess(_IMG, faces=[], persons=[_PERSON])["subjects"][0]
    assert s["age"] == 20.0 and s["assessment"] == "POSSIBLE_MINOR"


def test_rotated_scan_can_be_switched_off_and_is_not_used_when_a_face_was_found(monkeypatch):
    est = _estimator_with(monkeypatch, variant_ages=[40.0, 15.0], hits=_HIT)
    off = est.assess(_IMG, faces=[], persons=[_PERSON], scan_rotated=False)["subjects"][0]
    assert off["age"] == 40.0 and off["rotation"] == 0
    est2 = _estimator_with(monkeypatch, variant_ages=[40.0], hits=_HIT)
    with_face = est2.assess(_IMG, faces=[(60, 40, 60, 60)], persons=[_PERSON])["subjects"][0]
    assert with_face["rotation"] == 0 and with_face["evidence"] == "face+body"


def test_no_rotated_face_found_keeps_the_body_only_judgement(monkeypatch):
    est = _estimator_with(monkeypatch, variant_ages=[22.0], hits=[])
    s = est.assess(_IMG, faces=[], persons=[_PERSON])["subjects"][0]
    assert s["evidence"] == "body" and s["assessment"] == "POSSIBLE_MINOR"


def test_loading_the_age_model_leaves_autograd_enabled_for_the_rest_of_the_process():
    """Retraining in the same process (the app retrains detectors) needs gradients; loading MiVOLO must not switch them off."""
    import torch

    est = A.get_age_estimator()
    if est._ensure() is None:
        pytest.skip("age model weights not available")
    assert torch.is_grad_enabled()
    assert not any(p.requires_grad for p in est._ensure().parameters())


def test_a_git_lfs_pointer_is_reported_clearly_not_as_a_parse_error(tmp_path):
    from core.perception.hub import ensure_not_lfs_pointer

    pointer = tmp_path / "model.safetensors"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 115078528\n")
    with pytest.raises(RuntimeError, match="git lfs pull"):
        ensure_not_lfs_pointer(pointer)
    real = tmp_path / "real.bin"
    real.write_bytes(b"x" * 5000)
    ensure_not_lfs_pointer(real)                                   # a genuine file passes
    ensure_not_lfs_pointer(tmp_path / "missing.bin")               # absent is the caller's concern


# --- fail-safe behaviour: a screening that could not look must never read as "nobody there" ---------------------------------------

def test_missing_person_detector_with_no_face_is_unavailable_not_no_people(monkeypatch):
    est = _estimator_with(monkeypatch, variant_ages=[30.0])
    r = est.assess(_IMG, faces=[], persons=[], persons_available=False)
    assert r["status"] == "UNAVAILABLE" and r["review_required"] is True and "person detector" in r["reason"]


def test_missing_person_detector_but_a_face_found_is_still_screened(monkeypatch):
    est = _estimator_with(monkeypatch, variant_ages=[12.0])
    r = est.assess(_IMG, faces=[(60, 40, 60, 60)], persons=[], persons_available=False)
    assert r["status"] == "OK" and r["contains_minor"] is True and r["person_detector_available"] is False


def test_an_inference_crash_becomes_unavailable_and_review_required(monkeypatch):
    est = _estimator_with(monkeypatch, variant_ages=[30.0])
    monkeypatch.setattr(est, "_ages", lambda faces, bodies: (_ for _ in ()).throw(MemoryError("out of memory")))
    r = est.assess(_IMG, faces=[], persons=[_PERSON])
    assert r["status"] == "UNAVAILABLE" and r["review_required"] is True and "MemoryError" in r["reason"]


def test_a_crowd_is_aged_in_chunks_not_one_giant_batch(monkeypatch):
    import torch

    est = A.AgeEstimator()
    sizes = []

    class Net:
        def __call__(self, batch):
            sizes.append(batch.shape[0])
            return torch.zeros(batch.shape[0], 3)

    monkeypatch.setattr(est, "_model", Net())
    crops = [np.full((60, 60, 3), 100, np.uint8)] * (A.AGE_BATCH * 2 + 3)
    ages = est._ages(crops, crops)
    assert len(ages) == len(crops) and sizes == [A.AGE_BATCH, A.AGE_BATCH, 3]


def test_age_eval_stops_with_a_reason_when_the_screening_did_not_run_and_summarises_unaged_faces(tmp_path):
    import json

    import pytest

    from services import age_eval

    class Dead:
        def assess(self, img, faces=None, persons=None):
            return {"status": "UNAVAILABLE", "subjects": [], "reason": "weights missing"}

    with pytest.raises(RuntimeError, match="weights missing"):
        age_eval._single_face_subject(Dead(), np.zeros((40, 40, 3), np.uint8))
    rows = [{"file": "a", "true": 30, "est": None, "share": None, "assessment": "UNDETERMINED"},
            {"file": "b", "true": 31, "est": 29.0, "share": 0.0, "assessment": "ADULT"}]
    out = tmp_path / "r.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert "26-35" in age_eval.summary(out)                                   # no TypeError on the face without an age
