"""Face authenticity: cropping, graceful absence of a model, and the training-data preparation that removes shortcuts."""
import numpy as np
from PIL import Image

from image_detector.face_authenticity import INPUT_SIZE, FaceAuthenticityClassifier, crop_face
from image_detector.face_training import prepare, split


def test_crop_face_is_fixed_size_even_at_the_border():
    img = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    assert crop_face(img, (0, 0, 80, 80)).shape == (INPUT_SIZE, INPUT_SIZE, 3)
    assert crop_face(img, (350, 250, 80, 80)).shape == (INPUT_SIZE, INPUT_SIZE, 3)


def test_without_a_checkpoint_the_classifier_reports_unavailable(tmp_path):
    clf = FaceAuthenticityClassifier(tmp_path / "missing.pt")
    assert not clf.available
    out = clf.analyse(np.zeros((200, 200, 3), np.uint8), [(10, 10, 100, 100, 0.9)])
    assert out["available"] is False and out["faces"] == []


def test_small_faces_are_not_judged(tmp_path):
    clf = FaceAuthenticityClassifier(tmp_path / "missing.pt")
    assert clf.analyse(np.zeros((200, 200, 3), np.uint8), [(0, 0, 20, 20, 0.9)])["skipped_small"] == 1


def test_prepare_gives_the_same_shape_for_any_source_size(tmp_path):
    box = (20, 20, 100, 100)
    for name, size in (("a.jpg", (178, 218)), ("b.jpg", (256, 256)), ("c.png", (640, 480))):
        Image.fromarray(np.random.default_rng(1).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(tmp_path / name)
        for train, stress in ((True, False), (False, False), (False, True)):
            assert prepare(tmp_path / name, box, train, stress).shape == (INPUT_SIZE, INPUT_SIZE, 3)


def test_validation_preparation_is_deterministic(tmp_path):
    Image.fromarray(np.random.default_rng(2).integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(tmp_path / "d.jpg")
    assert np.array_equal(prepare(tmp_path / "d.jpg", (10, 10, 120, 120), False, True), prepare(tmp_path / "d.jpg", (10, 10, 120, 120), False, True))


def test_split_is_stable_and_disjoint(tmp_path):
    files = []
    for i in range(30):
        p = tmp_path / f"{i}.png"
        Image.fromarray(np.full((8, 8, 3), i * 7, np.uint8)).save(p)
        files.append(p)
    tr1, va1 = split(files)
    tr2, va2 = split(list(reversed(files)))
    assert set(tr1) == set(tr2) and set(va1) == set(va2) and not set(tr1) & set(va1)


def test_near_duplicates_never_straddle_the_train_validation_split(tmp_path):
    """A re-saved or resized copy of a picture must land on the same side as the original, or validation accuracy is inflated."""
    import numpy as np
    from PIL import Image

    from image_detector.face_training import difference_hash, near_duplicate_groups, split

    rng = np.random.default_rng(3)
    files = []
    for i in range(40):
        base = Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).resize((256, 256), Image.BICUBIC)
        orig = tmp_path / f"p{i}.png"
        base.save(orig)
        copy = tmp_path / f"p{i}_copy.jpg"
        base.resize((200, 200)).save(copy, quality=70)             # same picture, different size and compression
        files += [orig, copy]
    hashes = [difference_hash(p) for p in files]
    groups = near_duplicate_groups(hashes)
    assert all(groups[2 * i] == groups[2 * i + 1] for i in range(40))      # each copy joins its original
    assert len(set(groups)) < 60                                           # while unrelated pictures stay apart
    train, val = split(files)
    assert val and train
    for i in range(40):
        assert (files[2 * i] in val) == (files[2 * i + 1] in val)


class _FakeClassifier:
    def __init__(self, available, faces=None, skipped=0):
        self.available, self._faces, self._skipped = available, faces or [], skipped

    def analyse(self, image, faces):
        return {"available": self.available, "faces": self._faces, "skipped_small": self._skipped}


def _run_check(monkeypatch, tmp_path, classifier, faces):
    from core.forensics.registry import CheckContext
    from image_detector.dimension_checks import faces as faces_check

    path = tmp_path / "p.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)).save(path)
    monkeypatch.setattr(faces_check, "get_face_authenticity", lambda: classifier)
    monkeypatch.setattr(faces_check, "get_face_finder", lambda: type("F", (), {"find": staticmethod(lambda img: faces)})())
    return faces_check.check_face_authenticity(CheckContext(path=path, modality="image"))


def test_face_check_reports_each_outcome_honestly(monkeypatch, tmp_path):
    from core.forensics.schemas import FindingStatus

    big = [(50, 50, 200, 200, 0.9)]
    assert _run_check(monkeypatch, tmp_path, _FakeClassifier(False), big).status == FindingStatus.NOT_CALIBRATED          # no model: says so
    assert _run_check(monkeypatch, tmp_path, _FakeClassifier(True), []).status == FindingStatus.NOT_APPLICABLE            # no face
    assert _run_check(monkeypatch, tmp_path, _FakeClassifier(True, []), big).status == FindingStatus.INFO                 # all too small to judge
    ai = _run_check(monkeypatch, tmp_path, _FakeClassifier(True, [{"bbox": [50, 50, 200, 200], "p_ai": 0.97}]), big)
    assert ai.status == FindingStatus.WARN and ai.llr == 0.40 and ai.data["face_dominant"] is True and ai.data["ai_like"] is True
    real = _run_check(monkeypatch, tmp_path, _FakeClassifier(True, [{"bbox": [50, 50, 200, 200], "p_ai": 0.05}]), big)
    assert real.status == FindingStatus.PASS and not real.llr                                  # a real-looking face never adds evidence
    unsure = _run_check(monkeypatch, tmp_path, _FakeClassifier(True, [{"bbox": [50, 50, 200, 200], "p_ai": 0.7}]), big)
    assert unsure.status == FindingStatus.INFO and not unsure.llr
