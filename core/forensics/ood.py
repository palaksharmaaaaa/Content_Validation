"""
core.forensics.ood: epistemic out-of-distribution gate (Mahalanobis distance on embeddings).

The gate is only meaningful once fitted on reference embeddings (e.g. from the user's media
library). Unfitted, it reports NOT_CALIBRATED -- it never invents a threshold.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

MIN_FIT_SAMPLES = 20
THRESHOLD_PERCENTILE = 99.0


class OODGate:
    """Mahalanobis out-of-distribution gate. Reports ``NOT_CALIBRATED`` until fitted on enough embeddings; never invents a threshold."""
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.n_samples: Optional[int] = None          # how many reference files the gate was fitted on

    @property
    def calibrated(self) -> bool:
        """True once a mean, covariance and threshold exist."""
        return self.mean is not None and self.inv_cov is not None and self.threshold is not None

    def fit(self, embeddings: np.ndarray) -> "OODGate":
        """Fit on an (n, d) embedding matrix; too few samples leaves the gate uncalibrated."""
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim == 2:
            x = x[np.all(np.isfinite(x), axis=1)]         # rows with NaN or infinity cannot describe a distribution
        # With fewer samples than dimensions the covariance is singular and every training point sits unnaturally close to the mean,
        # so the percentile threshold would be far too tight and ordinary files would read as out of distribution.
        if x.ndim != 2 or x.shape[0] < max(MIN_FIT_SAMPLES, x.shape[1]):
            self.mean = self.inv_cov = self.threshold = self.n_samples = None
            return self
        self.n_samples = int(x.shape[0])
        mean = x.mean(axis=0)
        centered = x - mean
        cov = (centered.T @ centered) / max(1, x.shape[0] - 1)
        ridge = 1e-3 * (np.trace(cov) / cov.shape[0] + 1e-12)
        inv = np.linalg.pinv(cov + ridge * np.eye(cov.shape[0]))
        d = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", centered, inv, centered)))
        self.mean, self.inv_cov = mean, inv
        self.threshold = float(np.percentile(d, THRESHOLD_PERCENTILE))
        return self

    def score(self, vector: np.ndarray) -> Dict[str, Any]:
        """Distance of one vector from the fitted distribution and its IN/OUT-of-distribution status."""
        if not self.calibrated:
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": None}
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(v)):
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": self.threshold, "note": "The feature vector holds a non-finite value."}
        if v.shape[0] != self.mean.shape[0]:
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": self.threshold}
        diff = v - self.mean
        dist = float(np.sqrt(max(0.0, diff @ self.inv_cov @ diff)))
        status = "OUT_OF_DISTRIBUTION" if dist > self.threshold else "IN_DISTRIBUTION"
        return {"status": status, "distance": dist, "threshold": self.threshold}

    def save(self, path: Union[str, Path]) -> None:
        """Write the fitted statistics to a compressed ``.npz``."""
        if not self.calibrated:
            raise ValueError("Cannot save an uncalibrated OODGate")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        partial = p.with_name(p.name + ".partial")
        with open(partial, "wb") as handle:                       # written aside, then swapped in: a crash never leaves half a file
            np.savez_compressed(handle, mean=self.mean, inv_cov=self.inv_cov, threshold=np.array([self.threshold]),
                                n_samples=np.array([self.n_samples or 0]))
        os.replace(partial, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OODGate":
        """Load statistics saved by ``save``; returns an uncalibrated gate if the file is missing."""
        gate = cls()
        p = Path(path)
        if not p.is_file():
            return gate
        try:
            with np.load(p) as data:
                gate.mean = data["mean"]
                gate.inv_cov = data["inv_cov"]
                gate.threshold = float(data["threshold"][0])
                gate.n_samples = int(data["n_samples"][0]) if "n_samples" in data.files else None
        except Exception:  # corrupt stats file -> treat as uncalibrated
            return cls()
        return gate


def fit_gate_from_files(
    paths: Iterable[Path],
    out_path: Union[str, Path],
    vectorizer: Callable[[Path], np.ndarray],
    max_samples: int,
    noun: str,
) -> Dict[str, Any]:
    """Vectorise up to ``max_samples`` files (one that cannot be read is skipped), fit an ``OODGate`` and save it.

    ``noun`` names what is counted in the "need at least N" message (images, files, videos).
    """
    vectors = []
    for p in list(paths)[:max_samples]:
        try:
            vectors.append(np.asarray(vectorizer(Path(p)), dtype=np.float64).reshape(-1))
        except Exception as exc:  # unreadable file: skip, keep fitting
            logger.warning("skipping %s: %s", p, exc)
    if len(vectors) < MIN_FIT_SAMPLES:
        return {"fitted": False, "samples": len(vectors), "message": f"Need at least {MIN_FIT_SAMPLES} {noun}; got {len(vectors)}."}
    gate = OODGate().fit(np.vstack(vectors))
    if not gate.calibrated:
        dim = len(vectors[0])
        return {"fitted": False, "samples": len(vectors), "message": f"Fit failed: need at least {max(MIN_FIT_SAMPLES, dim)} usable {noun} for {dim} features (and finite values); got {len(vectors)}."}
    gate.save(out_path)
    return {"fitted": True, "samples": len(vectors), "threshold": gate.threshold, "path": str(out_path)}
