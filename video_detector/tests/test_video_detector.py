"""
video_detector.tests.test_video_detector: Autonomous test suite for Video AI Detection module.
Verifies all submodules, temporal analysis, optical flow, and pipeline stages.
"""
import unittest
from pathlib import Path
import tempfile
import cv2
import numpy as np

from video_detector.attribution import VideoModelAttributionEngine
from video_detector.batch import VideoBatchProcessor
from video_detector.content import VideoContentAnalyzer
from video_detector.cross_modal import CrossModalConsistencyEngine
from video_detector.detector import VideoAIDetector
from video_detector.extractor import VideoFrameExtractor
from video_detector.face import VideoFaceDeepfakeDetector
from video_detector.learner import VideoSelfImprover
from video_detector.pipeline import VideoForensicPipeline
from video_detector.profiler import VideoProfiler
from video_detector.provenance import VideoProvenanceValidator
from video_detector.scoring import normalize_percentages
from video_detector.temporal import compute_interframe_motion_variance, detect_diffusion_flickering
from video_detector.validator import VideoValidator


class TestVideoDetector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create synthetic test MP4 video
        self.vid_file = self.temp_path / "test_sample.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.vid_file), fourcc, 10.0, (128, 128))
        for _ in range(15):
            frame = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scoring_invariants(self):
        ai, real, u = normalize_percentages(60.0, 30.0, 10.0)
        self.assertAlmostEqual(ai + real + u, 100.0, places=1)
        self.assertGreaterEqual(u, 3.0)

    def test_extractor(self):
        extractor = VideoFrameExtractor()
        frames, timestamps, step = extractor.extract_frames(self.vid_file, max_frames=5)
        self.assertGreater(len(frames), 0)
        self.assertGreater(len(timestamps), 0)
        self.assertGreater(step, 0)

    def test_temporal(self):
        frames = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(6)]
        motion = compute_interframe_motion_variance(frames)
        self.assertIn("motion_variance", motion)

        flicker = detect_diffusion_flickering(frames)
        self.assertIn("has_diffusion_flicker", flicker)

    def test_validator(self):
        validator = VideoValidator()
        res = validator.validate(self.vid_file)
        self.assertTrue(res.valid)
        self.assertGreater(res.duration_seconds, 0.0)

    def test_profiler(self):
        profiler = VideoProfiler()
        prof = profiler.profile_video(self.vid_file)
        self.assertTrue(prof["valid"])
        self.assertIn("sha256", prof)
        self.assertGreater(prof["fps"], 0.0)

    def test_face_detector(self):
        detector = VideoFaceDeepfakeDetector()
        frames = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        res = detector.analyze_video_frames(frames)
        self.assertIn("deepfake_risk", res)

    def test_content_analyzer(self):
        analyzer = VideoContentAnalyzer()
        frames = [np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        res = analyzer.analyze_video_frames(frames)
        self.assertIn("environment", res)

    def test_attribution_engine(self):
        engine = VideoModelAttributionEngine()
        res = engine.attribute_video(self.vid_file)
        self.assertIn("attributed_model", res)

    def test_provenance_validator(self):
        prov = VideoProvenanceValidator()
        res = prov.analyze_provenance(self.vid_file)
        self.assertIn("c2pa_present", res)

    def test_cross_modal_engine(self):
        engine = CrossModalConsistencyEngine()
        res = engine.evaluate_consistency(
            video_forensics={"ai_percentage": 70.0},
            audio_forensics={"has_audio_track": True, "ai_percentage": 20.0},
        )
        self.assertTrue(res["is_multimodal"])
        self.assertGreater(res["asymmetry_score"], 0.0)

    def test_detector_predict(self):
        detector = VideoAIDetector()
        detector.load()
        res = detector.analyze_video(self.vid_file)
        self.assertTrue(res.get("valid", True))
        self.assertIn("ai_percentage", res)
        self.assertIn("label", res)

    def test_learner(self):
        learner = VideoSelfImprover(
            memory_file=self.temp_path / "vid_memory.json",
            calibration_file=self.temp_path / "vid_calib.json",
        )
        calib = learner.record_feedback(str(self.vid_file), "REAL", {"motion_variance": 45.0})
        self.assertEqual(calib["samples_processed"], 1)

    def test_pipeline_end_to_end(self):
        pipeline = VideoForensicPipeline()
        report = pipeline.analyze(self.vid_file)
        self.assertTrue(report["content_valid"])
        self.assertIn("final_status", report)
        self.assertIn("authenticity_probabilities", report)
        self.assertIn("evidence_trail", report)

    def test_batch_processor(self):
        batch = VideoBatchProcessor()
        res = batch.process_files([self.vid_file])
        self.assertEqual(res["total_processed"], 1)
        self.assertEqual(len(res["results"]), 1)


if __name__ == "__main__":
    unittest.main()
