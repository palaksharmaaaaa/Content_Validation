"""Golden regression for AudioAIDetector.analyze_audio_file across fixtures, sensitivities and dimension terms."""
import json
from pathlib import Path

import numpy as np

from audio_detector import AudioAIDetector
from tests.golden_support import assert_golden, skip_unless_state_matches
from audio_detector.tests.audio_fixtures import tone, write_wav


def clean(r):
    r = dict(r)
    r.pop("spectrogram_image", None)
    return r



def test_audio_analysis_matches_golden(tmp_path):
    skip_unless_state_matches("audio_analyze_golden.meta.json", ["audio_detector/data/audio_calibration.json"])
    tmp = tmp_path

    det = AudioAIDetector()
    rng = np.random.default_rng(1)
    files = {
        "tone": write_wav(tmp / "a.wav", tone(3.0, noise=0.01)),
        "pure": write_wav(tmp / "b.wav", tone(4.0, amp=0.5)),
        "noise": write_wav(tmp / "c.wav", (rng.standard_normal(16000 * 5) * 0.1).astype(np.float32)),
        "silence_gaps": write_wav(tmp / "d.wav", np.concatenate([tone(1.0), np.zeros(16000 * 2, np.float32), tone(1.0, freq=300)])),
        "short": write_wav(tmp / "e.wav", tone(0.02)),
        "hf": write_wav(tmp / "f.wav", (tone(3.0, freq=7000, amp=0.3) + (rng.standard_normal(16000 * 3) * 0.05)).astype(np.float32)),
    }
    res = {}
    for k, p in files.items():
        for sens in ("balanced", "high", "aggressive"):
            res[f"{k}|{sens}"] = clean(det.analyze_audio_file(p, sensitivity=sens))
        res[f"{k}|extra"] = clean(det.analyze_audio_file(p, extra_log_lrs={"enf": 0.3, "z": -0.1, "zero": 0.0}))
        res[f"{k}|extra_neg"] = clean(det.analyze_audio_file(p, extra_log_lrs={"enf": -0.4}))
    res["missing"] = clean(det.analyze_audio_file(tmp / "nope.wav"))
    res["preextracted"] = clean(det.analyze_audio_file(files["tone"], pre_extracted=(tone(2.0), 16000, 2.0)))
    res["pre_none"] = clean(det.analyze_audio_file(files["tone"], pre_extracted=(None, 16000, 0.0)))
    res["spectro"] = {k: v for k, v in det.analyze_audio_file(files["tone"], generate_spectrogram=True).items() if k != "spectrogram_image"}
    assert_golden("audio_analyze_golden", json.loads(json.dumps(res, sort_keys=True, default=str)))
