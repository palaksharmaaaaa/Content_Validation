"""
image_detector.dimension_checks: report-dimension checks wired through core.forensics.

Public API
----------
* ``check_image_gates(path)``  -- hard-block + scientific-format recognition gates (run first).
* ``ImageDimensionAnalysis``   -- ``run_pre()`` (before the detector; returns capped score terms for
  ``ImageAIDetector.predict(extra_log_lrs=...)``) and ``run_post()`` (after detector/content/attribution;
  returns the full report dict incl. confidence band, OOD status and open-set attribution flag).
* ``normalize_provenance``     -- accepts both the pipeline's flat provenance shape and the UI's nested one.
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from core.forensics.gates import HardBlockGate, recognize_scientific_format
from core.forensics.ood import OODGate
from core.forensics.registry import CheckContext, build_report, registry
from core.forensics.reporting import UNKNOWN_SOURCE_MIN_AI_PERCENT as _UNKNOWN_MIN
from core.forensics.reporting import add_band_and_open_set, summarize_for_evidence_trail
from core.forensics.schemas import Finding

# Importing the modules registers their checks. Order = execution order within a phase.
from image_detector.dimension_checks import (
    context,  # noqa: F401
    faces,  # noqa: F401
    formats,  # noqa: F401
    integrity,  # noqa: F401
    legal,  # noqa: F401
    lifecycle,  # noqa: F401
    metadata,  # noqa: F401
    reliability,  # noqa: F401
)

logger = logging.getLogger(__name__)

DEFAULT_OOD_STATS = Path(__file__).resolve().parents[1] / "data" / "ood_stats.npz"
UNKNOWN_SOURCE_MIN_AI_PERCENT = _UNKNOWN_MIN  # re-exported for callers

__all__ = [
    "ImageDimensionAnalysis",
    "check_image_gates",
    "gate_short_circuit_result",
    "normalize_provenance",
    "summarize_for_evidence_trail",
]


def normalize_provenance(prov: dict[str, Any] | None) -> dict[str, Any]:
    """Flat {c2pa_present, has_camera_hardware} view of either provenance shape (pipeline or UI)."""
    prov = prov or {}
    c2pa = prov.get("c2pa") if isinstance(prov.get("c2pa"), dict) else {}
    exif = prov.get("exif") if isinstance(prov.get("exif"), dict) else {}
    make = prov.get("camera_make") or exif.get("camera_make")
    model = prov.get("camera_model") or exif.get("camera_model")
    return {
        "c2pa_present": bool(prov.get("c2pa_present") or c2pa.get("c2pa_present")),
        "has_camera_hardware": bool(prov.get("has_camera_hardware") or (make and model)),
    }


def check_image_gates(path: str | Path) -> dict[str, Any]:
    hb = HardBlockGate().check(path)
    recog = recognize_scientific_format(path)
    final_status = "HARD_BLOCK_ESCALATE" if hb.triggered else ("RECOGNIZED_OUT_OF_SCOPE" if recog else None)
    return {
        "hard_block": hb.to_dict(),
        "recognition": recog,
        "triggered": final_status is not None,
        "final_status": final_status,
    }


def gate_short_circuit_result(path: str | Path, gates: dict[str, Any]) -> dict[str, Any]:
    """Result dict returned instead of a verdict when a gate fires (no probabilities are computed)."""
    status = gates["final_status"]
    if status == "HARD_BLOCK_ESCALATE":
        reason = "File hash matches the operator-supplied hard-block list. Analysis stopped; escalate per policy."
    else:
        rec = gates.get("recognition") or {}
        reason = (f"Recognized {rec.get('type', 'scientific')} data ({rec.get('description', '')}). "
                  "Outside this engine's authenticity-scoring scope; no verdict is produced.")
    return {
        "content_valid": True,
        "filename": Path(path).name,
        "path": str(path),
        "final_status": status,
        "ai_detected": False,
        "reason": reason,
        "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
        "gate": gates,
        "evidence_trail": [reason],
    }


@functools.lru_cache(maxsize=1)
def _shared_feature_store():
    """One FeatureStore (and thus one backbone load) per process instead of one per analysed image."""
    from image_detector.feature_store import FeatureStore

    return FeatureStore()


def _default_embedder() -> Callable[[Path], np.ndarray] | None:
    try:
        store = _shared_feature_store()

        def embed(p: Path) -> np.ndarray:
            feats = store.extract_features(str(p))
            if feats is None:
                raise ValueError("no embedding could be computed for this image")
            return np.asarray(feats["embedding"], dtype=np.float64)

        return embed
    except Exception as exc:
        logger.debug("_default_embedder: ignored %s: %s", type(exc).__name__, exc)
        return None


class ImageDimensionAnalysis:
    def __init__(
        self,
        path: str | Path,
        profile: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        ood_gate: OODGate | None = None,
        ood_embedder: Callable[[Path], np.ndarray] | None = None,
    ):
        self.path = Path(path)
        self.profile = profile or {}
        self.provenance = normalize_provenance(provenance)
        self._ood_gate = ood_gate
        self._ood_embedder = ood_embedder
        self._pre: list[Finding] = []
        self._pre_terms: dict[str, float] = {}

    def _ctx(self, content=None, ai_result=None) -> CheckContext:
        return CheckContext(path=self.path, modality="image", profile=self.profile, provenance=self.provenance,
                            content=content or {}, ai_result=ai_result or {})

    def run_pre(self) -> dict[str, float]:
        """Runs integrity/format/metadata checks; returns capped log-odds terms for the detector."""
        self._pre = registry.run("image", self._ctx(), phase="pre")
        self._pre_terms = build_report(self._pre).score_terms
        return dict(self._pre_terms)

    def pre_finding(self, check_id: str) -> Finding | None:
        """A finding produced by ``run_pre`` (or None)."""
        return next((f for f in self._pre if f.check_id == check_id), None)

    def _ood(self) -> dict[str, Any]:
        gate = self._ood_gate or OODGate.load(DEFAULT_OOD_STATS)
        if not gate.calibrated:
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": None,
                    "note": "OOD gate is not fitted; fit it from your media library to enable out-of-distribution detection."}
        embed = self._ood_embedder or _default_embedder()
        if embed is None:
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": gate.threshold,
                    "note": "No embedding backend available."}
        try:
            return gate.score(embed(self.path))
        except Exception as exc:  # embedding failure must never break the pipeline
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": gate.threshold, "note": f"Embedding failed: {exc}"}

    def run_post(
        self,
        content: dict[str, Any] | None = None,
        ai_result: dict[str, Any] | None = None,
        attribution: dict[str, Any] | None = None,
        gates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self._ctx(content, ai_result)
        post = registry.run("image", ctx, phase="post")
        report = build_report(self._pre + post, gates=gates)
        out = report.to_dict()

        out["ood"] = self._ood()
        add_band_and_open_set(out, ai_result, attribution, "image")
        return out
