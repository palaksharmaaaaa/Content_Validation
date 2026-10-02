"""Tests for core.checkpoint_log."""
import tempfile
import unittest
from pathlib import Path

from core.checkpoint_log import append_row, next_version, read_cumulative_samples, read_last_accuracy


class TestCheckpointLog(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "models" / "CHECKPOINT_LOG.md"
            self.assertIsNone(read_last_accuracy(log))
            self.assertEqual(next_version(log), 1)
            self.assertEqual(read_cumulative_samples(log), 0)

            append_row(log, 1, "2026-10-02", 0.3, 0.94, 203, 203)
            self.assertAlmostEqual(read_last_accuracy(log), 0.94)
            self.assertEqual(next_version(log), 2)

            append_row(log, 2, "2026-10-15", 0.2, 0.96, 18, 221)
            self.assertAlmostEqual(read_last_accuracy(log), 0.96)
            self.assertEqual(next_version(log), 3)
            self.assertEqual(read_cumulative_samples(log), 221)

    def test_log_contains_no_paths_or_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "CHECKPOINT_LOG.md"
            append_row(log, 1, "2026-10-02", 0.3, 0.94, 5, 5)
            text = log.read_text(encoding="utf-8")
            self.assertNotIn("\\", text.replace("|", ""))
            self.assertNotIn(".jpg", text)


if __name__ == "__main__":
    unittest.main()
