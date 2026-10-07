"""
image_detector.tests.test_image_detector: Autonomous test suite for Image AI Detection module.
Verifies all submodules, scoring invariants, neural backbones, and pipeline stages.
"""
import unittest
from pathlib import Path
import tempfile
import cv2
import numpy as np

from image_detector.attribution import ImageModelAttributionEngine
from image_detector.batch import ImageBatchProcessor
from image_detector.content import ImageContentAnalyzer
from image_detector.detector import ImageAIDetector
from image_detector.downloader import ImageDownloader
from image_detector.face import FaceDeepfakeDetector
from image_detector.features import (
    analyze_fft_radial_power_spectrum,
    calculate_sensor_noise_profile,
    calculate_surface_smoothness,
    compute_ela,
)
from image_detector.learner import ImageSelfImprover
from image_detector.pipeline import ImageForensicPipeline
from image_detector.profiler import ImageProfiler, compute_pixel_entropy
from image_detector.provenance import ImageProvenanceValidator
from image_detector.scoring import normalize_percentages, pool_bayesian_log_odds
from image_detector.validator import ImageValidator


class TestImageDetector(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create synthetic test image
        self.img_file = self.temp_path / "test_sample.png"
        dummy = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(self.img_file), dummy)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_scoring_invariants(self):
        ai, real, u = normalize_percentages(60.0, 30.0, 10.0)
        self.assertAlmostEqual(ai + real + u, 100.0, places=1)
        self.assertGreaterEqual(u, 3.0)

    def test_features(self):
        img = cv2.imread(str(self.img_file))
        noise_mean, noise_std = calculate_sensor_noise_profile(img)
        self.assertGreater(noise_mean, 0.0)

        smooth = calculate_surface_smoothness(img)
        self.assertGreater(smooth, 0.0)

        fft = analyze_fft_radial_power_spectrum(img)
        self.assertIn("spectral_decay_alpha", fft)

        ela_score_arr, ela_map_arr = compute_ela(img)
        self.assertGreaterEqual(ela_score_arr, 0.0)
        self.assertIsNotNone(ela_map_arr)

        ela_score_path, ela_map_path = compute_ela(self.img_file)
        self.assertGreaterEqual(ela_score_path, 0.0)
        self.assertIsNotNone(ela_map_path)

    def test_validator(self):
        validator = ImageValidator()
        res = validator.validate(self.img_file)
        self.assertTrue(res.valid)
        self.assertGreater(res.file_size_mb, 0.0)

    def test_profiler(self):
        profiler = ImageProfiler()
        prof = profiler.profile_image(self.img_file)
        self.assertTrue(prof["valid"])
        self.assertIn("sha256", prof)
        self.assertGreater(prof["pixel_entropy"], 0.0)

    def test_face_detector(self):
        detector = FaceDeepfakeDetector()
        img = cv2.imread(str(self.img_file))
        res = detector.analyze_faces(img)
        self.assertIn("faces_detected", res)

    def test_content_analyzer(self):
        analyzer = ImageContentAnalyzer()
        res = analyzer.analyze_image_content(self.img_file)
        self.assertIn("environment", res)
        self.assertIn("lighting_and_daytime", res)

    def test_attribution_engine(self):
        engine = ImageModelAttributionEngine()
        res = engine.attribute_image(self.img_file)
        self.assertIn("attributed_model", res)

    def test_provenance_validator(self):
        prov = ImageProvenanceValidator()
        res = prov.analyze_provenance(self.img_file)
        self.assertIn("c2pa_present", res)

    def test_detector_predict(self):
        detector = ImageAIDetector()
        detector.load()
        res = detector.predict(self.img_file)
        self.assertTrue(res.get("valid", True))
        self.assertIn("ai_percentage", res)
        self.assertIn("label", res)

    def test_learner(self):
        learner = ImageSelfImprover(
            memory_file=self.temp_path / "memory.json",
            calibration_file=self.temp_path / "calib.json",
        )
        calib = learner.record_feedback(str(self.img_file), "REAL", {"noise_residual_mean": 2.5})
        self.assertEqual(calib["samples_processed"], 1)

    def test_pipeline_end_to_end(self):
        pipeline = ImageForensicPipeline()
        report = pipeline.analyze(self.img_file)
        self.assertTrue(report["content_valid"])
        self.assertIn("final_status", report)
        self.assertIn("authenticity_probabilities", report)
        self.assertIn("evidence_trail", report)
        self.assertIn("taxonomy_state", report)
        self.assertIn("taxonomy_label", report)
        self.assertIn("taxonomy_description", report)
        self.assertIn("taxonomy_reasons", report)

    def test_taxonomy_classification_logic(self):
        from image_detector.scoring import evaluate_taxonomy_classification
        from image_detector.schemas import ImageTaxonomyState

        # 1. Fully AI Generated
        s1, l1, d1, r1 = evaluate_taxonomy_classification(
            prob_ai=0.95, prob_real=0.05,
            watermark_detected=True,
            metadata={"ai_signature_found": True},
            cutout_detected=False,
            scanned_detected=False,
            face_swap_detected=False
        )
        self.assertEqual(s1, ImageTaxonomyState.FULLY_AI_GENERATED)
        self.assertEqual(l1, "Fully AI Generated")

        # 2. AI-Enhanced / Composite (Face Swap)
        s2, l2, d2, r2 = evaluate_taxonomy_classification(
            prob_ai=0.75, prob_real=0.25,
            watermark_detected=False,
            metadata={},
            cutout_detected=False,
            scanned_detected=False,
            face_swap_detected=True
        )
        self.assertEqual(s2, ImageTaxonomyState.AI_ENHANCED_COMPOSITE)
        self.assertEqual(l2, "AI-Enhanced / Composite (Mix)")

        # 3. AI-Enhanced / Composite (Topaz / Remini)
        s3, l3, d3, r3 = evaluate_taxonomy_classification(
            prob_ai=0.70, prob_real=0.30,
            watermark_detected=False,
            metadata={"ai_enhancer_signature_found": True, "signature_details": "Topaz Photo AI"},
            cutout_detected=False,
            scanned_detected=False,
            face_swap_detected=False
        )
        self.assertEqual(s3, ImageTaxonomyState.AI_ENHANCED_COMPOSITE)

        # 4. Authentic Photograph (Edited / Graphic Design - Cutout)
        s4, l4, d4, r4 = evaluate_taxonomy_classification(
            prob_ai=0.30, prob_real=0.70,
            watermark_detected=False,
            metadata={},
            cutout_detected=True,
            scanned_detected=False,
            face_swap_detected=False
        )
        self.assertEqual(s4, ImageTaxonomyState.AUTHENTIC_EDITED)
        self.assertEqual(l4, "Authentic Created Photograph (Edited / Graphic Design)")

        # 5. Authentic Photograph (Edited / Graphic Design - Canva)
        s5, l5, d5, r5 = evaluate_taxonomy_classification(
            prob_ai=0.20, prob_real=0.80,
            watermark_detected=False,
            metadata={"graphic_editor_signature_found": True, "software": "Canva"},
            cutout_detected=False,
            scanned_detected=False,
            face_swap_detected=False
        )
        self.assertEqual(s5, ImageTaxonomyState.AUTHENTIC_EDITED)

        # 6. Authentic Real Photograph
        s6, l6, d6, r6 = evaluate_taxonomy_classification(
            prob_ai=0.10, prob_real=0.90,
            watermark_detected=False,
            metadata={"camera_make": "Canon", "camera_model": "EOS"},
            cutout_detected=False,
            scanned_detected=False,
            face_swap_detected=False
        )
        self.assertEqual(s6, ImageTaxonomyState.AUTHENTIC_REAL_PHOTOGRAPH)
        self.assertEqual(l6, "Authentic Real Photograph")

    def test_background_cutout_detection(self):
        from image_detector.features import detect_background_cutout
        # Create image with transparent alpha channel
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[20:80, 20:80, :3] = 200
        rgba[20:80, 20:80, 3] = 255  # only center opaque, rest transparent
        rgba_path = self.temp_path / "cutout.png"
        cv2.imwrite(str(rgba_path), rgba)

        res = detect_background_cutout(rgba_path)
        self.assertTrue(res["is_cutout"])
        self.assertTrue(res["is_transparent_png"])

    def test_digital_art_detection(self):
        from image_detector.features import detect_digital_art_and_painting
        # Synthetic saturated digital painting image
        hsv_art = np.zeros((100, 100, 3), dtype=np.uint8)
        hsv_art[:, :, 0] = 30   # Hue (gold/yellow)
        hsv_art[:, :, 1] = 180  # Saturation > 120
        hsv_art[:, :, 2] = 200  # Value
        bgr_art = cv2.cvtColor(hsv_art, cv2.COLOR_HSV2BGR)
        art_path = self.temp_path / "art_sample.png"
        cv2.imwrite(str(art_path), bgr_art)

        res = detect_digital_art_and_painting(art_path, bgr_art)
        self.assertTrue(res["is_digital_art"])
        self.assertGreaterEqual(res["mean_saturation"], 115.0)
        self.assertGreaterEqual(res["high_sat_pct"], 45.0)

    def test_taxonomy_digital_art_and_3d_cutout(self):
        from image_detector.scoring import evaluate_taxonomy_classification
        from image_detector.schemas import ImageTaxonomyState

        # 3D Render with cutout (e.g. ganesh-idol-emblem.png)
        s, l, d, r = evaluate_taxonomy_classification(
            prob_ai=0.90, prob_real=0.10,
            watermark_detected=False,
            metadata={},
            cutout_detected=True,
            art_detected=True,
            scanned_detected=False,
            face_swap_detected=False
        )
        self.assertEqual(s, ImageTaxonomyState.FULLY_AI_GENERATED)
        self.assertIn("Synthetic 3D asset with transparent alpha background cutout", r)

    def test_canonical_imagen_resolution(self):
        from image_detector.config import CANONICAL_RESOLUTIONS
        self.assertIn((896, 1200), CANONICAL_RESOLUTIONS)
        self.assertIn((1200, 896), CANONICAL_RESOLUTIONS)

    def test_screenshot_detection_mobile_and_desktop(self):
        from image_detector.features import detect_screenshot
        from image_detector.scoring import evaluate_taxonomy_classification
        from image_detector.schemas import ImageTaxonomyState

        # 1. Create simulated Mobile Portrait Screenshot (1080x2400)
        mobile_shot = np.zeros((2400, 1080, 3), dtype=np.uint8)
        # Top status bar simulation
        mobile_shot[:60, :] = 25
        # Bottom gesture nav bar simulation
        mobile_shot[-40:, :] = 35
        # Inner content (UI card)
        mobile_shot[200:1800, 50:1030] = 240
        m_path = self.temp_path / "Screenshot_20260101_120000.png"
        cv2.imwrite(str(m_path), mobile_shot)

        res_m = detect_screenshot(m_path, mobile_shot, meta={})
        self.assertTrue(res_m["is_screenshot"])
        self.assertEqual(res_m["device_type"], "Mobile Phone")
        self.assertEqual(res_m["orientation"], "Portrait")

        # 2. Create simulated Desktop Landscape Screenshot (1920x1080)
        desktop_shot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        desktop_shot[-50:, :] = 30 # Taskbar
        d_path = self.temp_path / "desktop_capture.png"
        cv2.imwrite(str(d_path), desktop_shot)

        res_d = detect_screenshot(d_path, desktop_shot, meta={"software": "Snipping Tool", "screenshot_software_found": True})
        self.assertTrue(res_d["is_screenshot"])
        self.assertEqual(res_d["device_type"], "Laptop / Desktop")
        self.assertEqual(res_d["orientation"], "Landscape")

        # 3. Test Screenshot Taxonomy Routing (Authentic Screenshot vs AI Screenshot)
        # Authentic UI screenshot
        s_auth, l_auth, _, _ = evaluate_taxonomy_classification(
            screenshot_data=res_m,
            screenshot_detected=True,
            noise_mean=0.3,
            metadata={},
        )
        self.assertEqual(s_auth, ImageTaxonomyState.AUTHENTIC_SCREENSHOT)
        self.assertEqual(l_auth, "Authentic Device Screenshot")

        # Screenshot displaying fully AI generated image (with watermark)
        s_ai, l_ai, _, _ = evaluate_taxonomy_classification(
            screenshot_data=res_m,
            screenshot_detected=True,
            watermark_detected=True,
            watermark_data={"watermark_detected": True, "details": "Gemini watermark"},
            noise_mean=0.3,
            metadata={},
        )
        self.assertEqual(s_ai, ImageTaxonomyState.AI_GENERATED_SCREENSHOT)
        self.assertEqual(l_ai, "AI-Generated Content Screenshot")

        # Screenshot displaying AI face swap / composite
        s_comp, l_comp, _, _ = evaluate_taxonomy_classification(
            screenshot_data=res_m,
            screenshot_detected=True,
            face_swap_detected=True,
            face_swap_data={"is_face_swap": True, "details": "Neural face-swap graft"},
            noise_mean=0.3,
            metadata={},
        )
        self.assertEqual(s_comp, ImageTaxonomyState.AI_ENHANCED_SCREENSHOT)
        self.assertEqual(l_comp, "AI-Enhanced / Composite Screenshot")

    def test_inpainting_and_manipulation_detection(self):
        from image_detector.features import detect_inpainting_and_manipulation
        from image_detector.scoring import evaluate_taxonomy_classification
        from image_detector.schemas import ImageTaxonomyState

        # Create composite image with localized noise discrepancy:
        # Background has noisy camera grain, center square is completely denoised/smoothed
        comp_img = np.random.normal(128, 12, (512, 512, 3)).astype(np.uint8)
        # Center inpainting patch (completely flat/zero noise)
        comp_img[150:350, 150:350] = 128

        comp_path = self.temp_path / "inpaint_sample.jpg"
        cv2.imwrite(str(comp_path), comp_img)

        inpaint_res = detect_inpainting_and_manipulation(comp_path, comp_img)
        self.assertTrue(inpaint_res["is_manipulated"])

        # Test taxonomy routing for inpainting
        meta = {"camera_make": "Sony", "camera_model": "DSC-T99"}
        # Uneven noise with a real-leaning score is ordinary local processing, never "partly AI"
        s, _, _, _ = evaluate_taxonomy_classification(inpainting_data=inpaint_res, inpainting_detected=True, metadata=meta, ai_pct=2.0, real_pct=95.0)
        self.assertEqual(s, ImageTaxonomyState.AUTHENTIC_EDITED)
        # The same noise pattern together with a synthetic-leaning score is a composite
        s, l, _, _ = evaluate_taxonomy_classification(inpainting_data=inpaint_res, inpainting_detected=True, metadata=meta, ai_pct=65.0, real_pct=30.0)
        self.assertEqual(s, ImageTaxonomyState.AI_ENHANCED_COMPOSITE)
        self.assertEqual(l, "AI-Enhanced / Composite (Mix)")
        # A middling score is not enough to claim AI involvement from uneven noise alone
        s, _, _, _ = evaluate_taxonomy_classification(inpainting_data=inpaint_res, inpainting_detected=True, metadata=meta, ai_pct=45.0, real_pct=40.0)
        self.assertNotEqual(s, ImageTaxonomyState.AI_ENHANCED_COMPOSITE)

    def test_digital_art_alone_does_not_name_an_image_ai(self):
        from image_detector.scoring import evaluate_taxonomy_classification
        from image_detector.schemas import ImageTaxonomyState

        art = {"is_digital_art": True, "visual_medium": "Digital 3D CGI / AI Neural Painting", "details": "saturated"}
        # saturation alone also describes corals and sunsets: with a middling score it is not called AI or CGI
        s, _, _, _ = evaluate_taxonomy_classification(art_data=art, ai_pct=40.0, real_pct=50.0)
        self.assertNotIn(s, (ImageTaxonomyState.PROCEDURAL_CGI_SYNTHETIC, ImageTaxonomyState.FULLY_AI_GENERATED))
        # with a score that agrees, it still is
        s, _, _, _ = evaluate_taxonomy_classification(art_data=art, ai_pct=75.0, real_pct=20.0)
        self.assertEqual(s, ImageTaxonomyState.PROCEDURAL_CGI_SYNTHETIC)

    def test_screen_rephotography_moire_detection(self):
        from image_detector.features import detect_screen_rephotography_moire
        from image_detector.scoring import evaluate_taxonomy_classification
        from image_detector.schemas import ImageTaxonomyState

        # Synthetic Moiré frequency grid simulation (diagonal sinusoidal subpixel raster)
        moire_img = np.zeros((512, 512, 3), dtype=np.uint8)
        y, x = np.ogrid[:512, :512]
        # Diagonal spatial wave (45 deg) with period ~3.5 pixels (mid-to-high spatial frequency)
        moire_pattern = (128 + 120 * np.sin(2 * np.pi * (x + y) / 3.5)).astype(np.uint8)
        moire_img[:, :, 0] = moire_pattern
        moire_img[:, :, 1] = moire_pattern
        moire_img[:, :, 2] = moire_pattern

        moire_path = self.temp_path / "screen_moire_sample.png"
        cv2.imwrite(str(moire_path), moire_img)

        res = detect_screen_rephotography_moire(moire_path, moire_img, {"camera_make": "Samsung", "camera_model": "S24 Ultra"})
        self.assertTrue(res["is_screen_recapture"])
        self.assertGreaterEqual(res["peak_ratio"], 4.5)

        s, l, _, _ = evaluate_taxonomy_classification(
            screen_recapture_data=res,
            screen_recapture_detected=True,
            metadata={"camera_make": "Samsung", "camera_model": "S24 Ultra"},
        )
        self.assertEqual(s, ImageTaxonomyState.AUTHENTIC_RECAPTURED_SCREEN)
        self.assertEqual(l, "Authentic Screen Re-photography (Recaptured)")

    def test_spectral_modality_detection(self):
        from image_detector.features import detect_spectral_modality

        # True Monochrome test
        mono_img = np.full((128, 128, 3), 120, dtype=np.uint8)
        mono_path = self.temp_path / "mono_sample.png"
        cv2.imwrite(str(mono_path), mono_img)
        res_mono = detect_spectral_modality(mono_path, mono_img)
        self.assertTrue(res_mono["is_monochrome"])
        self.assertEqual(res_mono["sensor_spectrum"], "Monochrome / Grayscale Sensor")

        # Color Bayer RGB test
        color_img = np.zeros((128, 128, 3), dtype=np.uint8)
        color_img[:, :, 0] = 50   # Blue
        color_img[:, :, 1] = 120  # Green
        color_img[:, :, 2] = 200  # Red
        color_path = self.temp_path / "color_sample.png"
        cv2.imwrite(str(color_path), color_img)
        res_color = detect_spectral_modality(color_path, color_img)
        self.assertFalse(res_color["is_monochrome"])
        self.assertEqual(res_color["sensor_spectrum"], "Visible Spectrum (Bayer RGB)")

    def test_procedural_cgi_synthetic_state(self):
        from image_detector.scoring import evaluate_taxonomy_classification
        from image_detector.schemas import ImageTaxonomyState

        # Procedural 3D CGI without diffusion watermark or generative IPTC
        s, l, d, r = evaluate_taxonomy_classification(
            prob_ai=0.85, prob_real=0.15,
            watermark_detected=False,
            art_detected=True,
            art_data={"is_digital_art": True, "visual_medium": "Digital 3D CGI / AI Neural Painting"},
            metadata={},
        )
        self.assertEqual(s, ImageTaxonomyState.PROCEDURAL_CGI_SYNTHETIC)
        self.assertEqual(l, "Procedural CGI Synthetic (3D Render)")

    def test_2026_foundation_model_attribution(self):
        from PIL import Image as PILImage
        from image_detector.attribution import ImageModelAttributionEngine

        engine = ImageModelAttributionEngine()

        # Ideogram 2.0 resolution and typography detection
        dummy_ideo = self.temp_path / "ideogram_sample.png"
        img = PILImage.new("RGB", (1280, 720), color=(100, 150, 200))
        img.save(str(dummy_ideo))

        attr_ideo = engine.attribute_image(
            dummy_ideo,
            forensic_data={"taxonomy_state": "FULLY_AI_GENERATED", "digital_art_detected": True},
            provenance_data={"metadata": {"software": "Ideogram 2.0"}},
        )
        self.assertIn("Ideogram", attr_ideo["attributed_model"])

        # Recraft v3 attribution via software tag
        dummy_recraft = self.temp_path / "recraft_sample.png"
        img.save(str(dummy_recraft))
        attr_recraft = engine.attribute_image(
            dummy_recraft,
            forensic_data={"taxonomy_state": "FULLY_AI_GENERATED"},
            provenance_data={"metadata": {"software": "Recraft v3"}},
        )
        self.assertIn("Recraft", attr_recraft["attributed_model"])

    def test_pipeline_multidimensional_fields(self):
        from image_detector.pipeline import ImageForensicPipeline

        pipe = ImageForensicPipeline()
        res = pipe.analyze(self.img_file)
        self.assertIn("subject_genre", res)
        self.assertIn("visual_medium", res)
        self.assertIn("sensor_spectrum", res)
        self.assertIn("document_layout", res)
        self.assertIn("screen_recapture_detected", res)
        self.assertIn("screen_recapture_analysis", res)

    def test_batch_processor(self):
        batch = ImageBatchProcessor()
        res = batch.process_files([self.img_file])
        self.assertEqual(res["total_processed"], 1)
        self.assertEqual(len(res["results"]), 1)

    def test_stylized_character_cutout_classification(self):
        from image_detector.scoring import evaluate_taxonomy_classification
        from image_detector.schemas import ImageTaxonomyState

        # Synthetic character cutout on solid background without camera hardware
        state, label, desc, reasons = evaluate_taxonomy_classification(
            ai_pct=75.0,
            real_pct=15.0,
            watermark_detected=False,
            cutout_detected=True,
            cutout_data={"is_cutout": True, "details": "Solid canvas background"},
            art_detected=True,
            art_data={"is_digital_art": True, "visual_medium": "Cel-Shaded / Ink Cross-Hatched Digital Art"},
            metadata={},
            noise_mean=0.15,
            smoothness=2.5,
        )
        self.assertEqual(state, ImageTaxonomyState.FULLY_AI_GENERATED)
        self.assertEqual(label, "Fully AI Generated")


if __name__ == "__main__":
    unittest.main()





def test_profiler_reads_exif_optics_with_real_rationals_and_survives_bad_tag(tmp_path):
    import numpy as np
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    from image_detector.profiler import ImageProfiler

    ex = Image.Exif()
    ex[271], ex[272] = "Canon", "EOS R5"
    sub = ex.get_ifd(0x8769)
    sub[0x829A] = (1, 250)  # malformed (tuple, not a rational): must not discard the tags below
    sub[0x829D] = IFDRational(28, 10)
    sub[0x8827] = 100
    sub[0x920A] = IFDRational(50, 1)
    p = tmp_path / "e.jpg"
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(p, "JPEG", exif=ex)
    info = ImageProfiler().profile_image(p)["exif_device_details"]
    assert info["aperture"] == "f/2.8" and info["iso"] == 100 and info["focal_length"] == "50.0mm"
    assert info["exposure_time"] is None
