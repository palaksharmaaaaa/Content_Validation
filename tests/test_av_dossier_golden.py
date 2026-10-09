"""Seeded golden for the audio and video nine-dimension dossiers (random key presence)."""
import json
import random

from audio_detector.explain import build_audio_nine_dimensions_dossier
from video_detector.explain import build_video_nine_dimensions_dossier


from tests.golden_support import assert_golden


def put(r, d, k, v, p=.6):
    if r.random() < p:
        d[k] = v


def audio_case(r):
    geom = {}
    put(r, geom, "sample_rate", r.choice([8000, 16000, 44100, 48000])); put(r, geom, "duration_seconds", round(r.uniform(.5, 60), 2)); put(r, geom, "channels", r.choice([1, 2]))
    prof = {"geometry": geom}
    for k, v in (("sample_rate", 22050), ("duration", 3.3), ("channels", 2), ("format", "MP3"), ("bit_depth", "24-bit")):
        put(r, prof, k, v, .4)
    ac = {}
    put(r, ac, "cutoff_freq_hz", r.choice([0.0, 8000.0, 16000.0, 12345.0])); put(r, ac, "spectral_flatness", r.uniform(0, .1)); put(r, ac, "digital_silence_ratio", r.uniform(0, .6)); put(r, ac, "measured", r.random() < .8, .7)
    put(r, ac, "has_vocoder_cutoff", r.random() < .5, .5)
    res = {"acoustic_features": ac}
    put(r, res, "has_vocoder_cutoff", r.random() < .5, .4); put(r, res, "synthesis_medium", "TTS", .4); put(r, res, "is_synthetic", True, .4); put(r, res, "ai_duration_pct", 33.3, .5)
    inv = {}
    put(r, inv, "dominant_modality", "Music-like"); put(r, inv, "signal_level", "Strong signal"); put(r, inv, "delivery_style", "Natural dynamics")
    prov = {}
    put(r, prov, "c2pa_present", True); put(r, prov, "provenance_status", "X")
    attr = {}
    put(r, attr, "attributed_model", "ElevenLabs"); put(r, attr, "attribution_confidence", r.random()); put(r, attr, "watermark_detected", True)
    return prof, res, inv, (prov if r.random() < .8 else None), (attr if r.random() < .8 else None)


def video_case(r):
    geom = {}
    put(r, geom, "width", 1920); put(r, geom, "height", 1080); put(r, geom, "fps", 29.97); put(r, geom, "duration_seconds", 12.5); put(r, geom, "total_frames", 375); put(r, geom, "aspect_ratio", 1.78)
    prof = {"geometry": geom}
    put(r, prof, "bitrate_kbps", 4500.0)
    temp = {}
    put(r, temp, "motion_variance", r.uniform(0, 100)); put(r, temp, "temporal_warping_risk", r.choice(["LOW", "HIGH_WARPING_DETECTED", "SUSPICIOUS_FLICKER", "UNNATURAL_FREEZE"]))
    res = {"temporal_consistency": temp}
    put(r, res, "diffusion_flicker", {"flicker_score": round(r.uniform(0, 1), 2), "has_diffusion_flicker": r.random() < .3, "frames_compared": r.choice([2, 12, 30])})
    put(r, res, "mean_frame_noise", r.uniform(0, 3)); put(r, res, "visual_medium", "CGI"); put(r, res, "is_synthetic", True); put(r, res, "details", {"ai_duration_pct": 12.0})
    inv = {}
    put(r, inv, "living_entities", {"humans": {"persons_count": 2, "faces_count": 1}}); put(r, inv, "entities", {"humans": {"persons_count": 3}}, .3)
    put(r, inv, "environment_and_surroundings", {"setting_type": "Street", "setting": "Out"}); put(r, inv, "lighting_and_daytime", {"daytime": "Night"})
    put(r, inv, "contents_and_items", {"identified_items": ["cup"]}); put(r, inv, "persons_count", 5, .3); put(r, inv, "faces_count", 4, .3); put(r, inv, "tone_and_mood", {}, .2)
    prov = {}
    put(r, prov, "c2pa_present", True); put(r, prov, "container_atoms", ["moov"]); put(r, prov, "camera_make", "GoPro"); put(r, prov, "provenance_status", "Y")
    attr = {}
    put(r, attr, "attributed_model", "Sora"); put(r, attr, "attribution_confidence", r.random()); put(r, attr, "watermark_detected", True)
    return prof, res, inv, (prov if r.random() < .8 else None), (attr if r.random() < .8 else None), ({"x": 1} if r.random() < .3 else None)



def test_av_dossiers_match_golden():
    r = random.Random(33)
    out = {"audio": [], "video": []}
    for _ in range(800):
        p, a, i, pr, at = audio_case(r)
        out["audio"].append(build_audio_nine_dimensions_dossier(p, a, i, pr, at))
    for _ in range(800):
        p, a, i, pr, at, cm = video_case(r)
        out["video"].append(build_video_nine_dimensions_dossier(p, a, i, pr, at, cm))
    assert_golden("av_dossier_golden", json.loads(json.dumps(out, sort_keys=True, default=str)))
