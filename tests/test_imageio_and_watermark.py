"""Reading unusual files, and the AI-watermark detector's resistance to ordinary texture."""
import cv2
import numpy as np

from core.imageio import imread
from image_detector.features import detect_ai_watermark


def test_imread_handles_non_ascii_names_and_sixteen_bit_png(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    p = tmp_path / "evening-street-1920×1080.png"
    cv2.imencode(".png", img)[1].tofile(str(p))
    out = imread(p)
    assert out is not None and out.shape == img.shape and out.dtype == np.uint8
    wide = (img.astype(np.uint16) * 257)
    q = tmp_path / "deep.png"
    cv2.imencode(".png", wide)[1].tofile(str(q))
    deep = imread(q, cv2.IMREAD_UNCHANGED)
    assert deep.dtype == np.uint8 and np.abs(deep.astype(int) - img.astype(int)).max() <= 1


def test_imread_returns_none_for_missing_or_broken_files(tmp_path):
    assert imread(tmp_path / "nope.png") is None
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert imread(bad) is None


def _canvas(star_size=0, alpha=1.0, bg=90):
    img = np.full((900, 1200, 3), bg, np.uint8)
    img = np.clip(img.astype(int) + np.random.default_rng(1).integers(-12, 12, img.shape), 0, 255).astype(np.uint8)
    if star_size:
        t = np.linspace(0, 2 * np.pi, 400)
        r = star_size / 2
        pts = np.stack([r + r * np.cos(t) ** 3, r + r * np.sin(t) ** 3], 1).astype(np.int32)
        layer = np.zeros((star_size, star_size), np.uint8)
        cv2.fillPoly(layer, [pts], 255)
        y0, x0 = 900 - star_size - 40, 1200 - star_size - 40
        roi = img[y0:y0 + star_size, x0:x0 + star_size].astype(float)
        img[y0:y0 + star_size, x0:x0 + star_size] = np.where((layer > 0)[..., None], roi * (1 - alpha) + 255 * alpha, roi).astype(np.uint8)
    return img


def test_a_four_pointed_sparkle_in_the_corner_is_detected():
    for size in (40, 56, 72):
        for alpha in (0.6, 1.0):
            assert detect_ai_watermark("x", _canvas(size, alpha))["watermark_detected"], (size, alpha)


def test_plain_and_textured_corners_are_not_watermarks():
    assert not detect_ai_watermark("x", _canvas())["watermark_detected"]
    rng = np.random.default_rng(3)
    textured = rng.integers(0, 255, (900, 1200, 3), dtype=np.uint8)
    textured = cv2.GaussianBlur(textured, (0, 0), 3)         # blobby texture: many roundish contours in the corner
    assert not detect_ai_watermark("x", textured)["watermark_detected"]
