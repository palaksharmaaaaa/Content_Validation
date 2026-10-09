"""Tests for core.retrain_engine and learner -> library hooks."""
import tempfile
import unittest
from pathlib import Path

from core.checkpoint_log import read_last_accuracy
from core.media_library import MediaLibrary, register_feedback
from core.retrain_engine import run_retrain


class TestRetrainEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lib = MediaLibrary(self.root / "library.json")
        for i in range(40):
            f = self.root / f"m{i}.bin"
            f.write_bytes(f"content-{i}".encode())
            self.lib.add(f, "real" if i % 2 else "ai_generated")
        self.log = self.root / "models" / "CHECKPOINT_LOG.md"
        self.live = self.root / "models" / "m.pt"

    def tearDown(self):
        self.tmp.cleanup()

    def _train(self, acc):
        def fn(train, val, candidate):
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(b"weights")
            return 0.25, acc
        return fn

    def test_bootstrap_promotes_and_marks_seen(self):
        res = run_retrain(self.lib, self.log, self.live, self._train(0.9))
        self.assertTrue(res.promoted)
        self.assertEqual(res.samples_folded_in, 40)
        self.assertEqual(self.lib.counts()["new"], 0)
        self.assertTrue(self.live.is_file())
        self.assertFalse(self.live.with_name("m.candidate.pt").exists())
        self.assertAlmostEqual(read_last_accuracy(self.log), 0.9)

    def test_no_new_samples_is_noop(self):
        run_retrain(self.lib, self.log, self.live, self._train(0.9))
        res = run_retrain(self.lib, self.log, self.live, self._train(0.9))
        self.assertFalse(res.promoted)

    def test_rollback_when_worse_keeps_live_and_labels_queued(self):
        run_retrain(self.lib, self.log, self.live, self._train(0.9))
        self.live.write_bytes(b"old-live")
        extra = self.root / "extra.bin"
        extra.write_bytes(b"brand-new-file")
        self.lib.add(extra, "ai_generated")
        res = run_retrain(self.lib, self.log, self.live, self._train(0.5))
        self.assertFalse(res.promoted)
        self.assertEqual(self.live.read_bytes(), b"old-live")
        self.assertEqual(self.lib.counts()["new"], 1)
        self.assertFalse(self.live.with_name("m.candidate.pt").exists())
        self.assertAlmostEqual(read_last_accuracy(self.log), 0.9)

    def test_training_error_cleans_candidate(self):
        def boom(train, val, candidate):
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(b"partial")
            raise RuntimeError("x")
        with self.assertRaises(RuntimeError):
            run_retrain(self.lib, self.log, self.live, boom)
        self.assertFalse(self.live.with_name("m.candidate.pt").exists())
        self.assertEqual(self.lib.counts()["new"], 40)


class TestRegisterFeedback(unittest.TestCase):
    def test_maps_labels_without_copying_and_tolerates_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lib = MediaLibrary(root / "lib.json")
            a = root / "a.bin"; a.write_bytes(b"aaa")
            b = root / "b.bin"; b.write_bytes(b"bbb")
            self.assertTrue(register_feedback(lib, a, "AI"))
            self.assertTrue(register_feedback(lib, b, "REAL"))
            self.assertFalse(register_feedback(lib, a, "ai"))  # duplicate no-op
            self.assertFalse(register_feedback(lib, root / "gone.bin", "AI"))
            self.assertFalse(register_feedback(None, a, "AI"))
            c = lib.counts()
            self.assertEqual((c["ai_generated"], c["real"]), (1, 1))
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["a.bin", "b.bin", "lib.json"])


if __name__ == "__main__":
    unittest.main()
