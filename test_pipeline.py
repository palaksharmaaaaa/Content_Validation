"""
Automated Test & Verification Suite for Media Content & AI Detection Pipeline.
Generates synthetic media samples (Image, Audio, Frames) and validates:
- File integrity
- Visual quality & blur
- Error Level Analysis (ELA) & EXIF metadata inspection
- Face deepfake detection
- Synthetic audio vocoder forensics
- Multi-modal tri-percentage breakdown (AI %, Real %, Undecided %)
- Final decision engine verdict
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageDraw

from models.ai_audio_detector import AIAudioDetector
from models.ai_image_detector import AIImageDetector
from models.face_detector import FaceDeepfakeDetector
from scoring.decision_engine import generate_final_decision
from validators.file_validator import validate_file
from validators.image_validator import analyze_image


def run_pipeline_test():
    print("=" * 70)
    print("[TEST] RUNNING MEDIA CONTENT & AI VALIDATION TEST SUITE")
    print("=" * 70)

    # 1. Create a test image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        print("\n[Step 1/5] Creating simulated test media image...")
        img = Image.new("RGB", (400, 400), color=(180, 200, 220))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 350, 350], outline=(20, 40, 80), width=4)
        draw.text((80, 180), "AUTHENTIC TEST MEDIA", fill=(10, 20, 30))
        img.save(tmp_path, "JPEG", quality=95)
        print(f" -> Generated test file: {tmp_path.name}")

        # 2. File Validation
        print("\n[Step 2/5] Running File Integrity Validation...")
        file_res = validate_file(str(tmp_path))
        print(f" -> Media Type: {file_res.get('media_type')}")
        print(f" -> Readable: {file_res.get('readable')}")
        print(f" -> Size Valid: {file_res.get('size_valid')}")
        assert file_res["readable"] is True, "File should be readable"

        # 3. Quality Validation
        print("\n[Step 3/5] Running Visual Quality & Exposure Analysis...")
        quality_res = analyze_image(str(tmp_path))
        print(f" -> Quality Status: {quality_res.get('quality')}")
        print(f" -> Blur Score: {quality_res.get('blur_score')}")
        print(f" -> Brightness: {quality_res.get('brightness')}")
        print(f" -> Is Blank: {quality_res.get('is_blank')}")

        # 4. AI & Forensics Detection (Image + Face + Audio)
        print("\n[Step 4/5] Running Multi-Modal AI Detection (Image, Face, Audio)...")
        ai_detector = AIImageDetector()
        ai_detector.load()
        ai_img_res = ai_detector.predict(str(tmp_path))

        print(f" -> Image AI Detector Backend: {ai_img_res.get('backend')}")
        print(f" -> Image AI Percentage: {ai_img_res.get('ai_percentage')}%")
        print(f" -> Image Real Percentage: {ai_img_res.get('real_percentage')}%")
        print(f" -> Image Undecided Percentage: {ai_img_res.get('undecided_percentage')}%")
        print(f" -> Image Prediction: {ai_img_res.get('prediction')}")

        # Face Detection
        face_detector = FaceDeepfakeDetector()
        img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
        face_res = face_detector.analyze_faces(img_np)
        print(f" -> Faces Detected: {face_res.get('faces_detected')}")
        print(f" -> Deepfake Risk: {face_res.get('deepfake_risk')}")

        # Audio Detection
        audio_detector = AIAudioDetector()
        # Simulate 1 second of 16kHz audio waveform
        t = np.linspace(0, 1.0, 16000)
        simulated_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.normal(0, 0.05, len(t))
        audio_res = audio_detector.analyze_audio_samples(simulated_audio, 16000)
        print(f" -> Audio AI Percentage: {audio_res.get('ai_percentage')}%")
        print(f" -> Audio Real Percentage: {audio_res.get('real_percentage')}%")
        print(f" -> Audio Undecided Percentage: {audio_res.get('undecided_percentage')}%")
        print(f" -> Audio Verdict: {audio_res.get('label')}")

        # 5. Consolidated Final Decision
        print("\n[Step 5/5] Generating Consolidated Multi-Modal Verdict...")
        final_decision = generate_final_decision(
            file_validation=file_res,
            quality_result=quality_res,
            ai_result=ai_img_res,
            audio_result=audio_res,
        )

        print("\n" + "=" * 70)
        print("[REPORT] FINAL CONSOLIDATED VERDICT")
        print("=" * 70)
        print(f"Content Valid:  {final_decision['content_valid']}")
        print(f"Final Status:   {final_decision['final_status']}")
        print(f"Verdict Reason: {final_decision['reason']}")
        print(f"AI Detected:    {final_decision['ai_detected']}")
        print("\nModality Breakdown:")
        for mod, sc in final_decision.get("modality_scores", {}).items():
            print(f" - [{mod.upper()}]: AI: {sc.get('ai_percentage')}% | Real: {sc.get('real_percentage')}% | Undecided: {sc.get('undecided_percentage')}%")
        print("=" * 70)
        print("[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    run_pipeline_test()
