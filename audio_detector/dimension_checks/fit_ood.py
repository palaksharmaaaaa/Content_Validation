"""
audio_detector.dimension_checks.fit_ood: fit the audio OOD gate from the in-place media library.

Usage:  python -m audio_detector.dimension_checks.fit_ood [--max 2000]

Computes the 5-dim acoustic feature vector for every resolvable library audio file (real + ai_generated:
the 'known' distribution), fits the Mahalanobis reference, and writes ``audio_detector/data/ood_stats.npz``
(gitignored local data). Nothing is copied or deleted; only mean/covariance statistics are stored.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from core.forensics.ood import fit_gate_from_files

logger = logging.getLogger("audio_detector.dimension_checks.fit_ood")


def fit_ood_gate(paths: Iterable[Path], out_path: Path, vectorizer: Callable[[Path], np.ndarray], max_samples: int = 2000) -> dict:
    return fit_gate_from_files(paths, out_path, vectorizer, max_samples, "analyzable files")


def _default_vectorizer() -> Callable[[Path], np.ndarray]:
    from audio_detector.dimension_checks import acoustic_vector
    from audio_detector.features import compute_spectral_features
    from audio_detector.validator import AudioValidator

    validator = AudioValidator()

    def vec(p: Path) -> np.ndarray:
        samples, sr, _dur = validator.extract_pcm_samples(p)
        if samples is None or len(samples) < 1000:
            raise ValueError("no decodable audio")
        v = acoustic_vector(compute_spectral_features(samples, sr))
        if v is None:
            raise ValueError("no features")
        return v

    return vec


def main(argv: Optional[list] = None) -> None:
    from audio_detector.dimension_checks import DEFAULT_OOD_STATS
    from core.media_library import library_for

    ap = argparse.ArgumentParser(description="Fit the audio OOD gate from the media library.")
    ap.add_argument("--max", type=int, default=2000, help="maximum number of files to analyze")
    args = ap.parse_args(argv)
    paths = [p for _sha, p, _label in library_for("audio").samples()]
    print(fit_ood_gate(paths, DEFAULT_OOD_STATS, _default_vectorizer(), max_samples=args.max))


if __name__ == "__main__":
    main()
