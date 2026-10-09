"""
image_detector.dimension_checks.fit_ood: fit the OOD gate from the in-place media library.

Usage:  python -m image_detector.dimension_checks.fit_ood [--max 2000]

Embeds every resolvable library image (real + ai_generated: the 'known' distribution), fits the
Mahalanobis reference and writes ``image_detector/data/ood_stats.npz`` (gitignored local data).
Nothing is copied or deleted; only the mean/covariance statistics are stored.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from core.forensics.ood import fit_gate_from_files

logger = logging.getLogger("image_detector.dimension_checks.fit_ood")


def fit_ood_gate(paths: Iterable[Path], out_path: Path, embedder: Callable[[Path], np.ndarray], max_samples: int = 2000) -> dict:
    return fit_gate_from_files(paths, out_path, embedder, max_samples, "embeddable images")


def main(argv: Optional[list] = None) -> None:
    from core.media_library import library_for
    from image_detector.dimension_checks import DEFAULT_OOD_STATS, _default_embedder

    ap = argparse.ArgumentParser(description="Fit the image OOD gate from the media library.")
    ap.add_argument("--max", type=int, default=2000, help="maximum number of images to embed")
    args = ap.parse_args(argv)
    embedder = _default_embedder()
    if embedder is None:
        raise SystemExit("No embedding backend available (torch/torchvision backbone failed to load).")
    paths = [p for _sha, p, _label in library_for("image").samples()]
    print(fit_ood_gate(paths, DEFAULT_OOD_STATS, embedder, max_samples=args.max))


if __name__ == "__main__":
    main()
