"""
audio_detector.tests.test_audio_detector: Autonomous test suite for Audio AI Detection module.
Verifies all submodules, vocoder cutoffs, spectral flatness, and pipeline stages.
"""
import unittest
from pathlib import Path
import tempfile
import wave
import numpy as np

from audio_detector.attribution import AudioModelAttributionEngine
from audio_detector.batch import AudioBatchProcessor
from audio_detector.content import AudioContentAnalyzer
from audio_detector.detector import AudioAIDetector
from audio_detector.downloader import AudioDownloader
from audio_detector.features import compute_spectral_features, generate_spectrogram_image
from audio_detector.learner import AudioSelfImprover
from audio_detector.pipeline import AudioForensicPipeline
from audio_detector.profiler import AudioProfiler
from audio_detector.provenance import AudioProvenanceValidator
from audio_detector.scoring import normalize_percentages, pool_acoustic_evidence
from audio_detector.validator import AudioValidator


class TestAudioDetector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create synthetic test WAV audio
        self.aud_file = self.temp_path / "test_sample.wav"
        sr = 16000
        dur_s = 1.5
        t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
        waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)

        with wave.open(str(self.aud_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(waveform.tobytes())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scoring_invariants(self):
        ai, real, u = normalize_percentages(60.0, 30.0, 10.0)
        self.assertAlmostEqual(ai + real + u, 100.0, places=1)
        self.assertGreaterEqual(u, 3.0)

    def test_features(self):
        samples = np.sin(np.linspace(0, 50, 4000)).astype(np.float32)
        feats = compute_spectral_features(samples, 16000)
        self.assertIn("has_vocoder_cutoff", feats)
        self.assertIn("spectral_flatness", feats)

        spec_img = generate_spectrogram_image(samples, 16000)
        self.assertIsNotNone(spec_img)

    def test_validator(self):
        validator = AudioValidator()
        res = validator.validate(self.aud_file)
        self.assertTrue(res.valid)
        self.assertGreater(res.duration_seconds, 0.0)

    def test_profiler(self):
        profiler = AudioProfiler()
        prof = profiler.profile_audio(self.aud_file)
        self.assertTrue(prof["valid"])
        self.assertIn("sha256", prof)
        self.assertGreater(prof["sample_rate"], 0)

    def test_content_analyzer(self):
        analyzer = AudioContentAnalyzer()
        samples = np.sin(np.linspace(0, 50, 4000)).astype(np.float32)
        res = analyzer.analyze_audio_scene(samples, 16000)
        self.assertIn("setting", res)
        self.assertIn("dominant_modality", res)

    def test_attribution_engine(self):
        engine = AudioModelAttributionEngine()
        res = engine.attribute_audio(self.aud_file)
        self.assertIn("attributed_model", res)

    def test_provenance_validator(self):
        prov = AudioProvenanceValidator()
        res = prov.analyze_provenance(self.aud_file)
        self.assertIn("c2pa_present", res)

    def test_detector_predict(self):
        detector = AudioAIDetector()
        detector.load()
        res = detector.predict(self.aud_file)
        self.assertTrue(res.get("valid", True))
        self.assertIn("ai_percentage", res)
        self.assertIn("label", res)

    def test_learner(self):
        learner = AudioSelfImprover(
            memory_file=self.temp_path / "aud_memory.json",
            calibration_file=self.temp_path / "aud_calib.json",
        )
        calib = learner.record_feedback(str(self.aud_file), "REAL", {"spectral_flatness": 0.04})
        self.assertEqual(calib["samples_processed"], 1)

    def test_pipeline_end_to_end(self):
        pipeline = AudioForensicPipeline()
        report = pipeline.analyze(self.aud_file)
        self.assertTrue(report["content_valid"])
        self.assertIn("final_status", report)
        self.assertIn("authenticity_probabilities", report)
        self.assertIn("evidence_trail", report)

    def test_batch_processor(self):
        batch = AudioBatchProcessor()
        res = batch.process_files([self.aud_file])
        self.assertEqual(res["total_processed"], 1)
        self.assertEqual(len(res["results"]), 1)


if __name__ == "__main__":
    unittest.main()
