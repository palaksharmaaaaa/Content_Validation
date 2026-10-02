import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.decision import generate_final_decision, normalize_percentages
from core.security import validate_secure_url, SecureUrlFetcher
from image_detector.downloader import ImageDownloader
from audio_detector.downloader import AudioDownloader
from video_detector.downloader import VideoDownloader
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

    def test_generate_final_decision_human_override(self):
        decision = generate_final_decision(
            file_validation={'readable': True},
            quality_result={'authenticity_score': 0.05, 'ai_percentage': 5.0, 'real_percentage': 95.0},
            provenance_result={'camera_make': 'Sony', 'exif_valid': True},
        )
        self.assertIn('AUTHENTIC', decision['final_status'])
        self.assertGreater(decision['authenticity_probabilities']['p_real'], 50.0)

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
        aud_res = {'authenticity_score': 0.85, 'has_vocoder_cutoff': True, 'acoustic_features': {'vocoder_cutoff_hz': 16000}}
        decision = {
            'final_status': 'AI_GENERATED_CONTENT',
            'authenticity_probabilities': {'p_ai': 85.0, 'p_real': 15.0},
        }
        dossier = build_audio_nine_dimensions_dossier(prof, aud_res)
        self.assertEqual(len(dossier), 9)
        self.assertIn('dimension_1_hardware_provenance', dossier)
        self.assertIn('dimension_9_generative_attribution', dossier)

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
        self.assertIn('dimension_1_hardware_provenance', dossier)
        self.assertIn('dimension_9_generative_attribution', dossier)

        narrative = generate_video_newbie_explanation('test.mp4', prof, {}, vid_res, decision)
        self.assertIsInstance(narrative, str)
        self.assertIn('test.mp4', narrative)
        self.assertTrue('genuine' in narrative or 'camera' in narrative)


class TestDownloaderSecurityDelegation(unittest.TestCase):
    def test_ssrf_blocked_across_all_downloaders(self):
        malicious_urls = [
            'http://127.0.0.1:8080/secret',
            'http://localhost/admin',
            'http://169.254.169.254/latest/meta-data',
            'http://10.0.0.1/internal',
            'file:///etc/passwd',
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir)
            img_dl = ImageDownloader()
            aud_dl = AudioDownloader()
            vid_dl = VideoDownloader()

            for url in malicious_urls:
                img_res = img_dl.download_image(url, destination_dir=dest)
                self.assertFalse(img_res['success'])
                self.assertIn('error', img_res)

                aud_res = aud_dl.download_audio(url, destination_dir=dest)
                self.assertFalse(aud_res['success'])
                self.assertIn('error', aud_res)

                vid_res = vid_dl.download_video(url, destination_dir=dest)
                self.assertFalse(vid_res['success'])
                self.assertIn('error', vid_res)


class TestVideoLearnerAtomic(unittest.TestCase):
    def test_atomic_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            memory_file = Path(tmp_dir) / "video_feedback.json"
            learner = VideoSelfImprover(memory_file=memory_file)
            res = learner.record_feedback(
                video_path="test_clip.mp4",
                user_label="ai_generated",
                metrics={"temporal_score": 0.85, "face_score": 0.75},
            )
            self.assertIsInstance(res, dict)
            self.assertTrue(memory_file.exists())
            loaded = learner.load_memory()
            self.assertGreaterEqual(len(loaded), 1)
            self.assertEqual(loaded[-1]["video_path"], "test_clip.mp4")
            self.assertEqual(loaded[-1]["user_label"], "AI_GENERATED")


def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTests(loader.loadTestsFromTestCase(TestCoreDecision))
    s.addTests(loader.loadTestsFromTestCase(TestModalExplainers))
    s.addTests(loader.loadTestsFromTestCase(TestDownloaderSecurityDelegation))
    s.addTests(loader.loadTestsFromTestCase(TestVideoLearnerAtomic))
    return s


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)
