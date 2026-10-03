"""Face detection: the bundled YuNet model loads and behaves sensibly on degenerate and face-free input."""
import numpy as np
import pytest

from core.face_detection import MODEL_PATH, FaceFinder, get_face_finder
from image_detector.face import FaceDeepfakeDetector


def test_model_is_bundled_and_loads():
    assert MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 100_000
    assert get_face_finder().available


def test_missing_model_degrades_to_no_faces(tmp_path):
    finder = FaceFinder(tmp_path / "nope.onnx")
    assert not finder.available
    assert finder.find(np.zeros((200, 200, 3), np.uint8)) == []


@pytest.mark.parametrize("img", [None, np.zeros((3,), np.uint8), np.zeros((8, 8, 3), np.uint8)])
def test_degenerate_inputs_return_no_faces(img):
    assert FaceDeepfakeDetector().detect_faces(img) == []


def test_noise_and_flat_images_have_no_faces():
    rng = np.random.default_rng(0)
    det = FaceDeepfakeDetector()
    assert det.detect_faces(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)) == []
    assert det.detect_faces(np.full((480, 640, 3), 128, np.uint8)) == []
    # skin-coloured ellipse: fooled the old colour heuristic, must not be called a face
    import cv2
    img = np.full((480, 640, 3), 90, np.uint8)
    cv2.ellipse(img, (320, 240), (90, 130), 0, 0, 360, (110, 150, 200), -1)
    assert det.detect_faces(img) == []


def test_analyze_faces_reports_zero_for_no_faces():
    out = FaceDeepfakeDetector().analyze_faces(np.full((300, 300, 3), 100, np.uint8))
    assert out["faces_detected"] == 0 and out["deepfake_risk"] == "NO_FACES_DETECTED"
