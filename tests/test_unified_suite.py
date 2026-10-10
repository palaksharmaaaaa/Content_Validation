import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.decision import generate_final_decision, normalize_percentages
from audio_detector.explain import build_audio_nine_dimensions_dossier, generate_audio_newbie_explanation
from video_detector.explain import build_video_nine_dimensions_dossier, generate_video_newbie_explanation
from video_detector.learner import VideoSelfImprover


class TestCoreDecision(unittest.TestCase):
    def test_normalize_percentages_valid(self):
        p_ai, p_real, p_und = normalize_percentages(75.0, 25.0)
        self.assertAlmostEqual(p_ai + p_real + p_und, 100.0, places=1)
        self.assertGreater(p_ai, p_real)

    def test_normalize_percentages_zero_sum(self):
        p_ai, p_real, p_und = normalize_percentages(0.0, 0.0)
        self.assertEqual(p_und, 100.0)
        self.assertEqual(p_ai, 0.0)
        self.assertEqual(p_real, 0.0)

    def test_normalize_percentages_negative(self):
        p_ai, p_real, p_und = normalize_percentages(-10.0, 90.0)
        self.assertAlmostEqual(p_ai + p_real + p_und, 100.0, places=1)
        self.assertEqual(p_ai, 0.0)

    def test_camera_exif_alone_is_not_scored(self):
        """Unauthenticated camera EXIF is forgeable: with no other evidence the verdict stays undetermined."""
        decision = generate_final_decision(
            file_validation={'readable': True},
            quality_result={},
            provenance_result={'camera_make': 'Sony', 'exif_valid': True},
        )
        self.assertEqual(decision['final_status'], 'UNDETERMINED')
        self.assertEqual(decision['authenticity_probabilities']['p_undecided'], 100.0)
        self.assertTrue(any('not scored' in e for e in decision.get('evidence_trail', [])) or decision['final_status'] == 'UNDETERMINED')

    def test_video_result_drives_video_verdict_not_keyframe(self):
        decision = generate_final_decision(
            file_validation={'readable': True},
            quality_result={},
            video_result={'ai_percentage': 93.0, 'real_percentage': 5.0, 'forensic_cues': ['flicker']},
            ai_result={'ai_percentage': 4.0, 'real_percentage': 94.0, 'taxonomy_state': 'AUTHENTIC_REAL_PHOTOGRAPH', 'label': 'LIKELY REAL'},
        )
        self.assertEqual(decision['decision_mode'], 'fused')
        self.assertNotEqual(decision['final_status'], 'LIKELY REAL')  # keyframe label must not override the video verdict
        only_video = generate_final_decision(
            file_validation={'readable': True}, quality_result={},
            video_result={'ai_percentage': 93.0, 'real_percentage': 5.0})
        self.assertEqual(only_video['final_status'], 'LIKELY_SYNTHETIC')
        self.assertGreater(only_video['authenticity_probabilities']['p_ai'], 65.0)

    def test_image_result_is_authoritative_and_marked_uncalibrated(self):
        decision = generate_final_decision(
            file_validation={'readable': True},
            quality_result={},
            ai_result={'ai_percentage': 12.0, 'real_percentage': 80.0, 'undecided_percentage': 8.0,
                       'taxonomy_state': 'AUTHENTIC_REAL_PHOTOGRAPH', 'label': 'LIKELY REAL'},
        )
        self.assertEqual(decision['decision_mode'], 'image_authoritative')
        self.assertEqual(decision['calibration_status'], 'UNCALIBRATED_HEURISTIC')

    def test_attribution_is_explanation_not_evidence(self):
        base = dict(file_validation={'readable': True}, quality_result={},
                    video_result={'ai_percentage': 60.0, 'real_percentage': 35.0})
        a = generate_final_decision(**base)
        b = generate_final_decision(**base, attribution_result={'attribution_confidence': 0.95, 'model_key': 'sora', 'attributed_model': 'Sora'})
        self.assertEqual(a['authenticity_probabilities'], b['authenticity_probabilities'])

    def test_generate_final_decision_ai_detected(self):
        decision = generate_final_decision(
            file_validation={'readable': True},
            quality_result={},
            ai_result={'confidence': 0.95, 'ai_percentage': 95.0, 'real_percentage': 5.0, 'is_ai': True},
        )
        self.assertTrue('SYNTHETIC' in decision['final_status'] or 'AI' in decision['final_status'])
        self.assertGreater(decision['authenticity_probabilities']['p_ai'], 50.0)


class TestModalExplainers(unittest.TestCase):
    def test_audio_explainer_dossier_and_newbie(self):
        prof = {'sample_rate': 44100, 'duration': 4.5, 'channels': 2}
        aud_res = {'authenticity_score': 0.85, 'has_vocoder_cutoff': True, 'acoustic_features': {'cutoff_freq_hz': 16000.0, 'has_vocoder_cutoff': True}}
        decision = {
            'final_status': 'AI_GENERATED_CONTENT',
            'authenticity_probabilities': {'p_ai': 85.0, 'p_real': 15.0},
        }
        dossier = build_audio_nine_dimensions_dossier(prof, aud_res)
        self.assertEqual(len(dossier), 9)
        self.assertIn('dimension_1', dossier)
        self.assertIn('dimension_9', dossier)

        narrative = generate_audio_newbie_explanation('test.wav', prof, {}, aud_res, decision)
        self.assertIsInstance(narrative, str)
        self.assertIn('test.wav', narrative)
        self.assertTrue('AI' in narrative or 'generator' in narrative)

    def test_video_explainer_dossier_and_newbie(self):
        prof = {'geometry': {'width': 1920, 'height': 1080, 'fps': 30.0, 'duration_seconds': 5.0, 'total_frames': 150}}
        vid_res = {'ai_score': 0.1, 'details': {'ai_duration_pct': 0.0}}
        decision = {
            'final_status': 'HUMAN_AUTHENTIC_CAPTURE',
            'authenticity_probabilities': {'p_ai': 5.0, 'p_real': 95.0},
        }
        dossier = build_video_nine_dimensions_dossier(prof, vid_res)
        self.assertEqual(len(dossier), 9)
        self.assertIn('dimension_1', dossier)
        self.assertIn('dimension_9', dossier)

        narrative = generate_video_newbie_explanation('test.mp4', prof, {}, vid_res, decision)
        self.assertIsInstance(narrative, str)
        self.assertIn('test.mp4', narrative)
        self.assertTrue('genuine' in narrative or 'camera' in narrative)


class TestVideoLearnerAtomic(unittest.TestCase):
    def test_atomic_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_file = Path(tmp_dir) / "video_feedback.json"
            learner = VideoSelfImprover(memory_file=memory_file, calibration_file=Path(tmp_dir) / "video_calibration.json")
            res = learner.record_feedback(
                video_path="test_clip.mp4",
                user_label="AI",
                metrics={"temporal_score": 0.85, "face_score": 0.75},
            )
            self.assertIsInstance(res, dict)
            self.assertTrue(memory_file.exists())
            loaded = learner.load_memory()
            self.assertGreaterEqual(len(loaded), 1)
            self.assertEqual(loaded[-1]["video_path"], "test_clip.mp4")
            self.assertEqual(loaded[-1]["user_label"], "AI")


from audio_detector.validator import AudioValidator
from video_detector.face import VideoFaceDeepfakeDetector
from services.forensic_service import ForensicService
import numpy as np


class TestForensicServiceFacade(unittest.TestCase):
    def setUp(self):
        self.service = ForensicService()

    def test_service_health_check(self):
        health = self.service.health_check()
        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        self.assertIn("components", health)
        self.assertIn("security", health)
        self.assertEqual(health["status"], "HEALTHY")


    def test_pipelines_lazy_initialized(self):
        self.assertIsNotNone(self.service.image_pipeline)
        self.assertIsNotNone(self.service.video_pipeline)
        self.assertIsNotNone(self.service.audio_pipeline)

    def test_audio_validator_rejects_a_missing_file(self):
        res = AudioValidator().validate("non_existent_file.wav")
        self.assertFalse(res.is_valid)
        self.assertEqual(res.get("readable"), False)


class TestVideoFaceGating(unittest.TestCase):
    def test_a_blank_frame_has_no_faces(self):
        faces = VideoFaceDeepfakeDetector().detect_faces(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(faces, [])


def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTests(loader.loadTestsFromTestCase(TestCoreDecision))
    s.addTests(loader.loadTestsFromTestCase(TestModalExplainers))
    s.addTests(loader.loadTestsFromTestCase(TestVideoLearnerAtomic))
    s.addTests(loader.loadTestsFromTestCase(TestForensicServiceFacade))
    s.addTests(loader.loadTestsFromTestCase(TestVideoFaceGating))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)

