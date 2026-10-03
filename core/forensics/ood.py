"""
core.forensics.ood: epistemic out-of-distribution gate (Mahalanobis distance on embeddings).

The gate is only meaningful once fitted on reference embeddings (e.g. from the user's media
library). Unfitted, it reports NOT_CALIBRATED -- it never invents a threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

MIN_FIT_SAMPLES = 20
THRESHOLD_PERCENTILE = 99.0


class OODGate:
    def __init__(self) -> None:
        self.mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    @property
    def calibrated(self) -> bool:
        return self.mean is not None and self.inv_cov is not None and self.threshold is not None

    def fit(self, embeddings: np.ndarray) -> "OODGate":
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < MIN_FIT_SAMPLES:
            self.mean = self.inv_cov = self.threshold = None
            return self
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
        if not self.calibrated:
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": None}
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.mean.shape[0]:
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": self.threshold}
        diff = v - self.mean
        dist = float(np.sqrt(max(0.0, diff @ self.inv_cov @ diff)))
        status = "OUT_OF_DISTRIBUTION" if dist > self.threshold else "IN_DISTRIBUTION"
        return {"status": status, "distance": dist, "threshold": self.threshold}

    def save(self, path: Union[str, Path]) -> None:
        if not self.calibrated:
            raise ValueError("Cannot save an uncalibrated OODGate")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, mean=self.mean, inv_cov=self.inv_cov, threshold=np.array([self.threshold]))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OODGate":
        gate = cls()
        p = Path(path)
        if not p.is_file():
            return gate
        try:
            data = np.load(p)
            gate.mean = data["mean"]
            gate.inv_cov = data["inv_cov"]
            gate.threshold = float(data["threshold"][0])
        except Exception:  # corrupt stats file -> treat as uncalibrated
            return cls()
        return gate
