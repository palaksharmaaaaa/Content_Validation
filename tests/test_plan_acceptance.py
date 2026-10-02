"""Plan acceptance criteria (feedback-driven retraining), adapted to the in-place library design."""
import importlib
import tempfile
import unittest
from pathlib import Path

from core.checkpoint_log import append_row, read_last_accuracy
from core.media_library import MediaLibrary

MODALITIES = ("image", "audio", "video")


class TestLearnerQueuesCorrections(unittest.TestCase):
    def test_record_feedback_queues_by_reference_for_each_modality(self):
        for mod in MODALITIES:
            with self.subTest(mod=mod), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                media = root / "sample.bin"
                media.write_bytes(b"verified-content-" + mod.encode())
                lib = MediaLibrary(root / "lib.json")
                learner_mod = importlib.import_module(f"{mod}_detector.learner")
                cls = next(getattr(learner_mod, n) for n in dir(learner_mod) if n.endswith("SelfImprover"))
                learner = cls(memory_file=root / "mem.json", calibration_file=root / "cal.json", library=lib)
                kwargs = {f"{mod}_path": str(media), "user_label": "AI"}
                if mod == "image":
                    kwargs["forensic_metrics"] = {}
                elif mod == "audio":
                    kwargs["acoustic_metrics"] = {}
                else:
                    kwargs["video_metrics"] = {}
                learner.record_feedback(**kwargs)
                self.assertEqual(lib.counts()["ai_generated"], 1)
                self.assertEqual(lib.counts()["new"], 1)
                self.assertEqual(media.read_bytes(), b"verified-content-" + mod.encode())


class TestRetrainWrappers(unittest.TestCase):
    def test_noop_when_nothing_pending(self):
        for mod in MODALITIES:
            with self.subTest(mod=mod), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                retrain = importlib.import_module(f"{mod}_detector.retrain")
                lib = MediaLibrary(root / "lib.json")
                self.assertEqual(retrain.count_pending_corrections(lib), 0)
                res = retrain.run_retrain(library=lib, log_path=root / "LOG.md", checkpoint_path=root / "x.pt")
                self.assertFalse(res.promoted)
                self.assertFalse((root / "x.pt").exists())

    def test_rollback_keeps_live_checkpoint_and_queue(self):
        import numpy as np
        from PIL import Image
        for mod in ("image", "video"):
            with self.subTest(mod=mod), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                media = root / "m"; media.mkdir()
                lib = MediaLibrary(root / "lib.json")
                rng = np.random.default_rng(3)
                for i in range(30):
                    if mod == "image":
                        f = media / f"{i}.png"
                        Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(f)
                    else:
                        import cv2
                        f = media / f"{i}.avi"
                        w = cv2.VideoWriter(str(f), cv2.VideoWriter_fourcc(*"MJPG"), 10, (48, 48))
                        for _ in range(8):
                            w.write(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8))
                        w.release()
                    lib.add(f, "real" if i % 2 else "ai_generated")
                log, live = root / "o" / "LOG.md", root / "o" / "live.pt"
                append_row(log, 1, "2026-10-02", 0.1, 1.01, 5, 5)  # unbeatable prior accuracy
                tr_mod = importlib.import_module(f"{mod}_detector.trainer")
                tr = getattr(tr_mod, "ImageDetectorTrainer" if mod == "image" else "VideoDetectorTrainer")(checkpoint_path=live)
                tr.save_checkpoint()
                live_bytes = live.read_bytes()
                res = importlib.import_module(f"{mod}_detector.retrain").run_retrain(
                    epochs=1, library=lib, log_path=log, checkpoint_path=live)
                self.assertFalse(res.promoted)
                self.assertEqual(live.read_bytes(), live_bytes)
                self.assertEqual(lib.counts()["new"], 30)
                self.assertEqual(read_last_accuracy(log), 1.01)
                self.assertFalse((root / "o" / "live.candidate.pt").exists())


if __name__ == "__main__":
    unittest.main()
