"""End-to-end: register media in place -> retrain -> promote, then rollback. Tiny synthetic media."""
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from core.checkpoint_log import read_last_accuracy
from core.media_library import MediaLibrary


def _write_wav(path: Path, freq: float, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.5, 22050 * 3 // 2, endpoint=False)
    sig = 0.4 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(t.size)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
        w.writeframes((np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes())


class TestAudioInPlaceRetrain(unittest.TestCase):
    def test_retrain_promotes_then_rolls_back_without_copying(self):
        from audio_detector.retrain import count_pending_corrections, run_retrain
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"; media.mkdir()
            lib = MediaLibrary(root / "library.json")
            for i in range(40):
                ai = i % 2 == 0
                f = media / f"a{i}.wav"
                _write_wav(f, 220 + i * 7, 0.001 if ai else 0.15, i)
                lib.add(f, "ai_generated" if ai else "real")
            before = sorted(p.name for p in media.iterdir())
            log, ckpt = root / "models" / "LOG.md", root / "models" / "a.pt"

            self.assertEqual(count_pending_corrections(lib), 40)
            res = run_retrain(min_new=1, epochs=3, library=lib, log_path=log, checkpoint_path=ckpt)
            self.assertTrue(res.promoted, res.message)
            self.assertTrue(ckpt.is_file())
            self.assertEqual(count_pending_corrections(lib), 0)
            self.assertEqual(sorted(p.name for p in media.iterdir()), before)  # nothing added/removed

            # Seed an unbeatable prior accuracy -> next run must roll back.
            from core.checkpoint_log import append_row
            append_row(log, 2, "2026-10-02", 0.1, 1.01, 0, 40)
            extra = media / "extra.wav"; _write_wav(extra, 500, 0.001, 99)
            lib.add(extra, "ai_generated")
            live_bytes = ckpt.read_bytes()
            res2 = run_retrain(min_new=1, epochs=1, library=lib, log_path=log, checkpoint_path=ckpt)
            self.assertFalse(res2.promoted)
            self.assertEqual(ckpt.read_bytes(), live_bytes)
            self.assertEqual(count_pending_corrections(lib), 1)
            self.assertEqual(read_last_accuracy(log), 1.01)


if __name__ == "__main__":
    unittest.main()


class TestImageInPlaceRetrain(unittest.TestCase):
    def test_retrain_reads_in_place_and_skips_corrupt_files(self):
        from image_detector.retrain import run_retrain
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"; media.mkdir()
            lib = MediaLibrary(root / "library.json")
            rng = np.random.default_rng(0)
            for i in range(40):
                ai = i % 2 == 0
                arr = (np.full((64, 64, 3), 120 if ai else 0, np.uint8) if ai
                       else rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
                f = media / f"i{i}.png"
                Image.fromarray(arr).save(f)
                lib.add(f, "ai_generated" if ai else "real")
            bad = media / "corrupt.png"; bad.write_bytes(b"not an image")
            lib.add(bad, "real")
            before = sorted(p.name for p in media.iterdir())
            res = run_retrain(min_new=1, epochs=1, library=lib,
                              log_path=root / "m" / "LOG.md", checkpoint_path=root / "m" / "i.pt")
            self.assertTrue(res.promoted, res.message)
            self.assertTrue((root / "m" / "i.pt").is_file())
            self.assertEqual(sorted(p.name for p in media.iterdir()), before)


class TestVideoInPlaceRetrain(unittest.TestCase):
    def test_retrain_reads_videos_in_place(self):
        import cv2
        from video_detector.retrain import run_retrain
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"; media.mkdir()
            lib = MediaLibrary(root / "library.json")
            rng = np.random.default_rng(1)
            for i in range(30):
                ai = i % 2 == 0
                f = media / f"v{i}.avi"
                w = cv2.VideoWriter(str(f), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
                for k in range(12):
                    frame = (np.full((64, 64, 3), 100 + k, np.uint8) if ai
                             else rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
                    w.write(frame)
                w.release()
                lib.add(f, "ai_generated" if ai else "real")
            before = sorted(p.name for p in media.iterdir())
            res = run_retrain(min_new=1, epochs=1, library=lib,
                              log_path=root / "m" / "LOG.md", checkpoint_path=root / "m" / "v.pt")
            self.assertTrue(res.promoted, res.message)
            self.assertEqual(sorted(p.name for p in media.iterdir()), before)
