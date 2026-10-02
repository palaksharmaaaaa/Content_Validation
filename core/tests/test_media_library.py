"""Tests for core.media_library -- content-addressed, copy-free registry of labeled media."""
import os
import tempfile
import unittest
from pathlib import Path

from core.media_library import MediaLibrary, compute_file_sha256, partition


class TestMediaLibrary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.media = self.root / "media"
        self.media.mkdir()
        self.lib = MediaLibrary(self.root / "library.json")

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name: str, content: bytes) -> Path:
        p = self.media / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    def test_add_references_file_without_copying_it(self):
        p = self._write("a.jpg", b"alpha-bytes")
        self.assertTrue(self.lib.add(p, "real"))
        self.assertEqual(self.lib.counts()["real"], 1)
        # Nothing but the manifest was written outside the media folder.
        self.assertEqual(sorted(x.name for x in self.root.iterdir()), ["library.json", "media"])
        self.assertEqual(p.read_bytes(), b"alpha-bytes")

    def test_manifest_stores_no_filenames_except_path_hint(self):
        p = self._write("secret_name.jpg", b"alpha")
        self.lib.add(p, "real")
        entry = self.lib.entries()[0]
        self.assertNotIn("filename", entry)
        self.assertNotIn("name", entry)

    def test_identity_is_content_not_name_or_path(self):
        p = self._write("one.jpg", b"same-content")
        self.lib.add(p, "real")
        renamed = self.media / "renamed_later.jpg"
        p.rename(renamed)
        # Old path is stale; rescan re-links by content hash.
        self.assertEqual(len(self.lib.missing()), 1)
        self.assertEqual(self.lib.rescan([self.media]), 1)
        self.assertEqual(self.lib.missing(), [])
        self.assertEqual(self.lib.counts()["total"], 1)
        self.assertEqual(self.lib.resolve(compute_file_sha256(renamed)), renamed.resolve())

    def test_duplicate_content_is_one_entry(self):
        a = self._write("a.jpg", b"dup")
        b = self._write("copy_of_a.jpg", b"dup")
        self.lib.add(a, "real")
        self.assertFalse(self.lib.add(b, "real"))
        self.assertEqual(self.lib.counts()["total"], 1)

    def test_relabel_marks_entry_new_again(self):
        p = self._write("a.jpg", b"flip")
        self.lib.add(p, "real")
        sha = compute_file_sha256(p)
        self.lib.mark_seen([sha])
        self.assertEqual(self.lib.counts()["new"], 0)
        self.assertTrue(self.lib.add(p, "ai_generated"))
        self.assertEqual(self.lib.counts()["ai_generated"], 1)
        self.assertEqual(self.lib.counts()["real"], 0)
        self.assertEqual(self.lib.counts()["new"], 1)

    def test_invalid_label_raises(self):
        p = self._write("a.jpg", b"x")
        with self.assertRaises(ValueError):
            self.lib.add(p, "maybe")

    def test_add_tree_filters_extensions_recursively(self):
        self._write("a.jpg", b"1")
        self._write("sub/b.png", b"2")
        self._write("sub/readme.txt", b"3")
        added = self.lib.add_tree(self.media, "real", {".jpg", ".png"})
        self.assertEqual(added, 2)

    def test_persists_across_instances(self):
        p = self._write("a.jpg", b"persist")
        self.lib.add(p, "ai_generated")
        again = MediaLibrary(self.root / "library.json")
        self.assertEqual(again.counts()["ai_generated"], 1)

    def test_changed_file_content_is_not_resolved_under_old_hash(self):
        p = self._write("a.jpg", b"original")
        self.lib.add(p, "real")
        sha = compute_file_sha256(p)
        p.write_bytes(b"edited-in-place")
        os.utime(p, (1, 1))  # force a different mtime
        self.assertIsNone(self.lib.resolve(sha))

    def test_partition_is_stable_and_roughly_20_percent_val(self):
        shas = [compute_file_sha256_bytes(str(i).encode()) for i in range(1000)]
        first = [partition(s) for s in shas]
        self.assertEqual(first, [partition(s) for s in shas])
        val_fraction = first.count("val") / len(first)
        self.assertTrue(0.15 < val_fraction < 0.25, val_fraction)

    def test_samples_split_and_status_filters(self):
        for i in range(30):
            self._write(f"r{i}.jpg", f"real-{i}".encode())
        self.lib.add_tree(self.media, "real", {".jpg"})
        train = self.lib.samples(split="train")
        val = self.lib.samples(split="val")
        self.assertEqual(len(train) + len(val), 30)
        self.assertTrue(all(label == "real" for _, _, label in train + val))
        self.assertEqual(len(self.lib.samples(status="new")), 30)

    def test_select_for_retrain_replays_seen_and_keeps_val_separate(self):
        for i in range(60):
            self._write(f"r{i}.jpg", f"real-{i}".encode())
        self.lib.add_tree(self.media, "real", {".jpg"})
        first_train, first_val = self.lib.select_for_retrain()
        self.lib.mark_seen([sha for sha, _, _ in first_train + first_val])
        self._write("fresh.jpg", b"fresh-content-unique")
        self.lib.add(self.media / "fresh.jpg", "ai_generated")

        train, val = self.lib.select_for_retrain(min_replay=5)
        train_shas = {sha for sha, _, _ in train}
        val_shas = {sha for sha, _, _ in val}
        self.assertTrue(train_shas.isdisjoint(val_shas))
        self.assertTrue(all(partition(s) == "train" for s in train_shas))
        self.assertTrue(all(partition(s) == "val" for s in val_shas))
        # replay includes some old files, deterministically
        again, _ = self.lib.select_for_retrain(min_replay=5)
        self.assertEqual([s for s, _, _ in train], [s for s, _, _ in again])


def compute_file_sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    unittest.main()
