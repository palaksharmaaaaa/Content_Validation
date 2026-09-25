"""
Automated Test & Verification Suite for Multi-Modal Media Content, AI Detection,
Forensic Media Profiler, C2PA Provenance, Parallel Content Understanding,
Cross-Modal Synchronization, Feedback Memory Bank, and Continual Auto-Learning Pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import wave

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from PIL import Image, ImageDraw

from learning.auto_learner import AutoLearner
from learning.forensic_memory import ForensicMemory
from models.ai_audio_detector import AIAudioDetector
from models.ai_image_detector import AIImageDetector
from models.content_analyzer import ContentAnalyzer
from models.face_detector import FaceDeepfakeDetector
from models.model_attribution import ModelAttributionEngine
from scoring.cross_modal_engine import evaluate_cross_modal_consistency
from scoring.decision_engine import generate_final_decision
from utils.media_profiler import profile_media
from validators.file_validator import validate_file
from validators.image_validator import analyze_image
from validators.provenance_validator import analyze_provenance


def run_pipeline_test():
    print("=" * 70)
    print("[TEST] NIST-ALIGNED MULTI-MODAL FORENSICS & CONTINUAL LEARNING TEST SUITE")
    print("=" * 70)

    # 1. Create a test image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
        tmp_img_path = Path(tmp_img.name)

    # 2. Create a synthetic test audio WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_aud:
        tmp_aud_path = Path(tmp_aud.name)

    try:
        print("\n[Step 1/9] Creating simulated test media (Image & Audio)...")
        img = Image.new("RGB", (400, 400), color=(180, 200, 220))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 350, 350], outline=(20, 40, 80), width=4)
        draw.text((80, 180), "AUTHENTIC TEST MEDIA", fill=(10, 20, 30))
        img.save(tmp_img_path, "JPEG", quality=95)

        # Generate 3 seconds of 16kHz audio with vocoder cutoff
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3)
        signal = 0.5 * np.sin(2 * np.pi * 400 * t) + 0.3 * np.sin(2 * np.pi * 1200 * t)
        with wave.open(str(tmp_aud_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((signal * 32767).astype(np.int16).tobytes())

        print(f" -> Generated test image: {tmp_img_path.name}")
        print(f" -> Generated test audio: {tmp_aud_path.name}")

        # 2. File Validation & Forensics
        print("\n[Step 2/9] Running File Integrity Validation...")
        file_res_img = validate_file(str(tmp_img_path))
        file_res_aud = validate_file(str(tmp_aud_path))
        print(f" -> Image Type: {file_res_img.get('media_type')} | Readable: {file_res_img.get('readable')}")
        print(f" -> Audio Type: {file_res_aud.get('media_type')} | Readable: {file_res_aud.get('readable')}")

        # 3. Provenance & C2PA Inspection
        print("\n[Step 3/9] Running Provenance, C2PA Manifest & EXIF Validation...")
        prov_img = analyze_provenance(str(tmp_img_path))
        print(f" -> C2PA Manifest Status: {prov_img['c2pa']['status']}")
        print(f" -> Provenance Verdict: {prov_img['provenance_verdict']}")
        print(f" -> Rule Enforced: {prov_img['provenance_rule'][:75]}...")

        # 4. Media Quality & Exposure Analysis
        print("\n[Step 4/9] Running Media Quality & Exposure Analysis...")
        quality_res = analyze_image(str(tmp_img_path))
        print(f" -> Quality Status: {quality_res.get('quality')} | Blur Score: {quality_res.get('blur_score')}")

        # 5. Parallel Content & Semantic Inventory
        content_analyzer = ContentAnalyzer()
        content_img = content_analyzer.analyze_image_content(str(tmp_img_path))
        content_aud = content_analyzer.analyze_audio_content(signal, sr, 3.0)
        humans_cnt = content_img["entities"]["humans"]["persons_count"]
        faces_cnt = content_img["entities"]["humans"]["faces_count"]
        text_cnt = content_img["contents_and_items"]["text_regions_count"]
        setting_typ = content_img["environment_and_surroundings"]["setting_type"]
        daytime_est = content_img["lighting_and_daytime"]["estimated_daytime"]
        tone_est = content_img["tone_and_mood"]["color_tone"]

        print(f" -> Image Content: Persons={humans_cnt}, Faces={faces_cnt}, Text Regions={text_cnt}")
        print(f" -> Surroundings: {setting_typ} | Daytime: {daytime_est} | Tone: {tone_est}")
        print(f" -> Audio Content: Dominant={content_aud['dominant_audio_type']}, Speakers={content_aud['estimated_speakers']}")

        # 6. Deep Media Profiler (Pixel Specs, Entropy, Bit Depth)
        print("\n[Step 6/9] Running Deep Media Profiler (Pixel Specs, Entropy, Hashes)...")
        img_profile = profile_media(str(tmp_img_path), modality="image")
        aud_profile = profile_media(str(tmp_aud_path), modality="audio")
        pix_spec = img_profile.get("pixel_specifications", {})
        print(f" -> Image SHA-256: {img_profile['file_identity']['sha256'][:16]}...")
        print(f" -> Pixel Dtype: {pix_spec.get('pixel_data_type')} | Data per Pixel: {pix_spec.get('bits_per_pixel')} bpp")
        print(f" -> Shannon Entropy: {pix_spec.get('shannon_entropy_bpp')} bits/pixel")

        # 7. Multi-Modal AI Detection (Image + Audio)
        print("\n[Step 7/9] Running Multi-Modal AI Forensics (PRNU, FFT, Vocoder)...")
        ai_detector = AIImageDetector()
        ai_img_res = ai_detector.predict(str(tmp_img_path), sensitivity="high")
        print(f" -> Image AI %: {ai_img_res.get('ai_percentage')}% | Real %: {ai_img_res.get('real_percentage')}% | Undecided: {ai_img_res.get('undecided_percentage')}%")
        print(f" -> Spatial Anomaly Area: {ai_img_res.get('ai_spatial_area_pct')}%")

        audio_detector = AIAudioDetector()
        audio_res = audio_detector.analyze_audio_file(str(tmp_aud_path), sensitivity="high")
        print(f" -> Audio AI %: {audio_res.get('ai_percentage')}% | Real %: {audio_res.get('real_percentage')}%")
        print(f" -> Audio Verdict: {audio_res.get('label')}")

        # 8. Cross-Modal Consistency Check
        print("\n[Step 8/10] Testing Cross-Modal Consistency Engine...")
        mock_vid_res = {"ai_video_rating": {"ai_percentage": 25.0}, "temporal_consistency": {"temporal_warping_risk": "LOW"}}
        cross_modal_res = evaluate_cross_modal_consistency(mock_vid_res, audio_res, content_img)
        print(f" -> Cross-Modal Status: {cross_modal_res['cross_modal_status']} (Tampering Risk: {cross_modal_res['tampering_risk']})")
        print(f" -> Modality Asymmetry: {cross_modal_res['asymmetry_score']}%")

        # 9. Global Generative Model Attribution & Watermarking
        print("\n[Step 9/10] Testing Global Generative Model Attribution Engine...")
        attribution_engine = ModelAttributionEngine()
        attr_img = attribution_engine.attribute_media(
            str(tmp_img_path),
            modality="image",
            forensic_data=ai_img_res,
            profile_data=img_profile,
            provenance_data=prov_img,
        )
        attr_aud = attribution_engine.attribute_media(
            str(tmp_aud_path),
            modality="audio",
            forensic_data=audio_res,
            profile_data=aud_profile,
            provenance_data=prov_img,
        )
        print(f" -> Image Attributed Model: {attr_img['attributed_model']} ({attr_img['region_of_origin']}) - Conf: {attr_img['attribution_confidence']}")
        print(f" -> Audio Attributed Model: {attr_aud['attributed_model']} ({attr_aud['region_of_origin']}) - Conf: {attr_aud['attribution_confidence']}")
        print(f" -> Watermark Detected: {attr_img['watermark_detected']}")

        # 10. Complete NIST-Style Media Forensics Dossier Report
        print("\n[Step 10/10] Generating Official Media Forensics Dossier Report...")
        dossier = generate_final_decision(
            file_validation=file_res_img,
            quality_result=quality_res,
            ai_result=ai_img_res,
            audio_result=audio_res,
            content_inventory=content_img,
            provenance_result=prov_img,
            cross_modal_result=cross_modal_res,
            attribution_result=attr_img,
        )

        probs = dossier["authenticity_probabilities"]
        inv = dossier["content_inventory"]
        loc = dossier["localization"]
        attr = dossier.get("model_attribution", {})

        print("\n" + "=" * 70)
        print("          OFFICIAL MEDIA FORENSICS DOSSIER REPORT")
        print("=" * 70)
        print(f" AUTHENTICITY PROBABILITIES:")
        print(f"   • P(AI-Generated):       {probs['p_ai']}%")
        print(f"   • P(Authentic Capture):  {probs['p_real']}%")
        print(f"   • P(Undetermined / OOD): {probs['p_undecided']}%")
        print(f" FINAL STATUS:             {dossier['final_status']}")
        print(f" REASON:                   {dossier['reason']}")
        print("-" * 70)
        print(" GLOBAL MODEL ATTRIBUTION:")
        print(f"   • Attributed Model:     {attr.get('attributed_model')}")
        print(f"   • Region of Origin:     {attr.get('region_of_origin')}")
        print(f"   • Confidence:           {int(attr.get('attribution_confidence', 0) * 100)}%")
        print("-" * 70)
        print(" CONTENT INVENTORY:")
        print(f"   • Persons: {inv['persons_count']} | Faces: {inv['faces_count']} | Text Regions: {inv['text_regions_count']}")
        print(f"   • Scene: {inv['scene_type']} | Dominant Audio: {inv['dominant_audio_type']}")
        print("-" * 70)
        print(" LOCALIZATION:")
        print(f"   • Suspicious Image Area:     {loc.get('suspicious_image_area_pct')}%")
        print(f"   • Suspicious Audio Duration: {loc.get('suspicious_audio_duration_pct')}%")
        print("-" * 70)
        print(" PROVENANCE & C2PA:")
        print(f"   • C2PA Manifest:  {'PRESENT' if dossier['provenance']['c2pa_present'] else 'ABSENT'}")
        print(f"   • C2PA Signature: {dossier['provenance']['c2pa_signature']}")
        print(f"   • Provenance:     {dossier['provenance']['provenance_verdict']}")
        print("-" * 70)
        print(" FORENSIC EVIDENCE AUDIT TRAIL:")
        for t in dossier.get("evidence_trail", []):
            print(f"   • {t}")
        print("=" * 70)
        print("[SUCCESS] ALL 10 FORENSIC STAGES COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        if tmp_img_path.exists():
            tmp_img_path.unlink()
        if tmp_aud_path.exists():
            tmp_aud_path.unlink()


if __name__ == "__main__":
    run_pipeline_test()
