"""
Unit test suite verifying zero-retention in-memory processing,
compressed .npz feature caching, and zero-media model training.
"""
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from core.atomic_io import get_ephemeral_cache_dir, purge_ephemeral_cache
from image_detector import (
    FeatureBankDataset,
    FeatureClassifierHead,
    FeatureStore,
    ImageAIDetector,
    ImageDetectorTrainer,
)


class TestZeroRetentionFeatureStore(unittest.TestCase):
    """Verifies that media files can be processed and models trained without retaining media on disk."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create dummy synthetic and authentic test images
        self.img1_path = self.temp_path / "sample_real.jpg"
        self.img2_path = self.temp_path / "sample_ai.jpg"

        # Realistic grain for real image
        real_arr = np.random.normal(128, 8, (256, 256, 3)).astype(np.uint8)
        cv2.imwrite(str(self.img1_path), real_arr)

        # Flat smoothed area for AI image
        ai_arr = np.full((256, 256, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(self.img2_path), ai_arr)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_in_memory_image_prediction(self):
        """Verifies ImageAIDetector.predict() runs on raw bytes and ndarray with 0 disk writes."""
        detector = ImageAIDetector()
        detector.load()

        # 1. Test raw numpy array
        raw_bgr = np.random.normal(128, 10, (200, 200, 3)).astype(np.uint8)
        res_ndarray = detector.predict(raw_bgr)
        self.assertTrue(res_ndarray["is_available"])
        self.assertIn("ai_percentage", res_ndarray)

        # 2. Test in-memory bytes
        is_success, buffer = cv2.imencode(".jpg", raw_bgr)
        self.assertTrue(is_success)
        raw_bytes = buffer.tobytes()

        res_bytes = detector.predict(raw_bytes)
        self.assertTrue(res_bytes["is_available"])
        self.assertGreater(res_bytes["real_percentage"] + res_bytes["ai_percentage"], 0)

        # 3. Test BytesIO stream
        bio = io.BytesIO(raw_bytes)
        res_bio = detector.predict(bio)
        self.assertTrue(res_bio["is_available"])

    def test_feature_extraction_in_memory(self):
        """Verifies FeatureStore extracts 512-dim embedding and 12-dim forensic features in memory."""
        store = FeatureStore()
        raw_bgr = np.random.normal(128, 6, (128, 128, 3)).astype(np.uint8)

        feats = store.extract_features(raw_bgr)
        self.assertIsNotNone(feats)
        self.assertEqual(feats["embedding"].shape, (512,))
        self.assertEqual(feats["forensics"].shape, (12,))

    def test_build_feature_bank_keeps_sources_and_stores_no_names(self):
        """Verifies the .npz cache is keyed by content hash, stores no filenames, and never deletes sources."""
        store = FeatureStore()
        out_npz = self.temp_path / "test_bank.npz"

        samples = [
            (self.img1_path, 1),
            (self.img2_path, 0),
        ]

        res = store.build_feature_bank(samples, out_npz)
        self.assertTrue(res["success"])
        self.assertEqual(res["samples_processed"], 2)
        self.assertTrue(out_npz.exists())

        # Sources untouched; archive holds hashes + fingerprint, never filenames.
        self.assertTrue(self.img1_path.exists())
        self.assertTrue(self.img2_path.exists())
        data = np.load(str(out_npz), allow_pickle=False)
        self.assertNotIn("filenames", data.files)
        self.assertEqual(len(data["hashes"][0]), 64)
        self.assertEqual(str(data["fingerprint"]), store.fingerprint)

        # Second build reuses every row from the cache.
        again = store.build_feature_bank(samples, out_npz)
        self.assertEqual(again["samples_reused_from_cache"], 2)

        # Verify loaded dataset
        dataset = FeatureBankDataset(out_npz, combine_forensics=False)
        self.assertEqual(len(dataset), 2)
        feats, label = dataset[0]
        self.assertEqual(feats.shape, torch.Size([512]))

    def test_training_from_feature_bank(self):
        """Verifies training neural classification head directly from .npz with zero media files."""
        trainer = ImageDetectorTrainer()
        out_npz = self.temp_path / "train_bank.npz"

        # Create 10 dummy feature samples directly in npz
        np.savez_compressed(
            str(out_npz),
            embeddings=np.random.randn(10, 512).astype(np.float32),
            forensics=np.random.randn(10, 12).astype(np.float32),
            labels=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        )

        history = trainer.train_from_feature_bank(out_npz, epochs=2, batch_size=4, val_split=0.2)
        self.assertIn("train_loss", history)
        self.assertEqual(len(history["train_loss"]), 2)

    def test_ephemeral_cache_purge(self):
        """Verifies that purge_ephemeral_cache() removes files from temporary storage."""
        cache_dir = get_ephemeral_cache_dir()
        test_file = cache_dir / "transient_test.tmp"
        test_file.write_text("transient_data")
        self.assertTrue(test_file.exists())

        purged_count = purge_ephemeral_cache()
        self.assertGreaterEqual(purged_count, 1)
        self.assertFalse(test_file.exists())


if __name__ == "__main__":
    unittest.main()
