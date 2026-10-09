"""
video_detector.dimension_checks.fit_ood: fit the video OOD gate from the in-place media library.

Usage:  python -m video_detector.dimension_checks.fit_ood [--max 200]

Runs the temporal analysis (motion variance / diffusion flicker) on every resolvable library video
(real + ai_generated: the 'known' distribution), fits the Mahalanobis reference, and writes
``video_detector/data/ood_stats.npz`` (gitignored local data). Nothing is copied or deleted; only
mean/covariance statistics are stored.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from core.forensics.ood import fit_gate_from_files

logger = logging.getLogger("video_detector.dimension_checks.fit_ood")


def fit_ood_gate(paths: Iterable[Path], out_path: Path, vectorizer: Callable[[Path], np.ndarray], max_samples: int = 200) -> dict:
    return fit_gate_from_files(paths, out_path, vectorizer, max_samples, "analyzable videos")


def _default_vectorizer() -> Callable[[Path], np.ndarray]:
    from video_detector.dimension_checks import temporal_vector
    from video_detector.extractor import VideoFrameExtractor
    from video_detector.temporal import compute_interframe_motion_variance, detect_diffusion_flickering

    extractor = VideoFrameExtractor()

    def vec(p: Path) -> np.ndarray:
        frames, _ts, step = extractor.extract_sampled_frames(p)
        if not frames or len(frames) < 3:
            raise ValueError("too few frames")
        result = {"temporal_consistency": compute_interframe_motion_variance(frames, temporal_step=step),
                  "diffusion_flicker": detect_diffusion_flickering(frames)}
        v = temporal_vector(result)
        if v is None:
            raise ValueError("no features")
        return v

    return vec


def main(argv: Optional[list] = None) -> None:
    from core.media_library import library_for
    from video_detector.dimension_checks import DEFAULT_OOD_STATS

    ap = argparse.ArgumentParser(description="Fit the video OOD gate from the media library.")
    ap.add_argument("--max", type=int, default=200, help="maximum number of videos to analyze")
    args = ap.parse_args(argv)
    paths = [p for _sha, p, _label in library_for("video").samples()]
    print(fit_ood_gate(paths, DEFAULT_OOD_STATS, _default_vectorizer(), max_samples=args.max))


if __name__ == "__main__":
    main()
