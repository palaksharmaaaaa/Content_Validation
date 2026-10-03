"""Face authenticity: cropping, graceful absence of a model, and the training-data preparation that removes shortcuts."""
import numpy as np
import pytest
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
