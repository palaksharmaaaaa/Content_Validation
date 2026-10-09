"""Golden regression for VideoAIDetector.analyze_video on synthetic clips (noise, still, moving, flat, flicker)."""
import hashlib
import json

import cv2
import numpy as np

from tests.golden_support import assert_golden, skip_unless_state_matches
from video_detector import VideoAIDetector


def write(path, frames, fps=10.0):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def clean(r):
    r = dict(r)
    kf = r.pop("keyframes", None)
    r["_keyframes"] = None if kf is None else [len(kf), hashlib.sha256(b"".join(k.tobytes() for k in kf)).hexdigest()[:16]]
    return r



def test_video_analysis_matches_golden(tmp_path):
    skip_unless_state_matches("video_analyze_golden.meta.json", ["video_detector/data/video_calibration.json"])
    tmp = tmp_path

    rng = np.random.default_rng(6)
    vids = {}
    vids["noise"] = [rng.integers(0, 255, (96, 128, 3), dtype=np.uint8) for _ in range(30)]
    base = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
    vids["still"] = [base.copy() for _ in range(30)]
    moving = []
    for i in range(40):
        f = np.full((96, 128, 3), 120, np.uint8)
        cv2.rectangle(f, (i * 2, 20), (i * 2 + 30, 60), (30, 200, 90), -1)
        moving.append((f + rng.integers(0, 6, f.shape, dtype=np.uint8)).astype(np.uint8))
    vids["moving"] = moving
    vids["flat"] = [np.full((96, 128, 3), 128, np.uint8) for _ in range(20)]
    vids["flicker"] = [np.clip(base.astype(int) + (40 if i % 2 else -40), 0, 255).astype(np.uint8) for i in range(30)]
    det = VideoAIDetector()
    res = {}
    progress = []
    for k, frames in vids.items():
        p = tmp / f"{k}.mp4"
        write(p, frames)
        for sens in ("balanced", "high"):
            res[f"{k}|{sens}"] = clean(det.analyze_video(p, sensitivity=sens))
        res[f"{k}|extra"] = clean(det.analyze_video(p, extra_log_lrs={"a": 0.3, "b": -0.05, "z": 0.0}))
        res[f"{k}|extra_neg"] = clean(det.analyze_video(p, extra_log_lrs={"a": -0.4}))
    det.analyze_video(tmp / "moving.mp4", progress_callback=lambda i, n, m: progress.append((i, n)))
    res["progress"] = {"calls": len(progress), "last": progress[-1] if progress else None}
    (tmp / "junk.mp4").write_bytes(b"not a video")
    res["junk"] = clean(det.analyze_video(tmp / "junk.mp4"))
    res["missing"] = clean(det.analyze_video(tmp / "nope.mp4"))
    assert_golden("video_analyze_golden", json.loads(json.dumps(res, sort_keys=True, default=str)))
