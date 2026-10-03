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

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from core.forensics.reporting import UNKNOWN_SOURCE_MIN_AI_PERCENT as _UNKNOWN_MIN
from core.forensics.reporting import add_band_and_open_set, summarize_for_evidence_trail
from core.forensics.gates import HardBlockGate, recognize_scientific_format
from core.forensics.ood import OODGate
from core.forensics.registry import CheckContext, build_report, registry
from core.forensics.schemas import Finding

# Importing the modules registers their checks. Order = execution order within a phase.
from image_detector.dimension_checks import integrity  # noqa: F401
from image_detector.dimension_checks import formats  # noqa: F401
from image_detector.dimension_checks import metadata  # noqa: F401
from image_detector.dimension_checks import context  # noqa: F401
from image_detector.dimension_checks import legal  # noqa: F401
from image_detector.dimension_checks import lifecycle  # noqa: F401
from image_detector.dimension_checks import reliability  # noqa: F401

DEFAULT_OOD_STATS = Path(__file__).resolve().parents[1] / "data" / "ood_stats.npz"
UNKNOWN_SOURCE_MIN_AI_PERCENT = _UNKNOWN_MIN  # re-exported for callers

__all__ = [
    "ImageDimensionAnalysis",
    "check_image_gates",
    "gate_short_circuit_result",
    "normalize_provenance",
    "summarize_for_evidence_trail",
]


def normalize_provenance(prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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


def check_image_gates(path: Union[str, Path]) -> Dict[str, Any]:
    hb = HardBlockGate().check(path)
    recog = recognize_scientific_format(path)
    final_status = "HARD_BLOCK_ESCALATE" if hb.triggered else ("RECOGNIZED_OUT_OF_SCOPE" if recog else None)
    return {
        "hard_block": hb.to_dict(),
        "recognition": recog,
        "triggered": final_status is not None,
        "final_status": final_status,
    }


def gate_short_circuit_result(path: Union[str, Path], gates: Dict[str, Any]) -> Dict[str, Any]:
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


def _default_embedder() -> Optional[Callable[[Path], np.ndarray]]:
    try:
        store = _shared_feature_store()

        def embed(p: Path) -> np.ndarray:
            feats = store.extract_features(str(p))
            return np.asarray(feats["embedding"], dtype=np.float64)

        return embed
    except Exception:
        return None


class ImageDimensionAnalysis:
    def __init__(
        self,
        path: Union[str, Path],
        profile: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        ood_gate: Optional[OODGate] = None,
        ood_embedder: Optional[Callable[[Path], np.ndarray]] = None,
    ):
        self.path = Path(path)
        self.profile = profile or {}
        self.provenance = normalize_provenance(provenance)
        self._ood_gate = ood_gate
        self._ood_embedder = ood_embedder
        self._pre: List[Finding] = []
        self._pre_terms: Dict[str, float] = {}

    def _ctx(self, content=None, ai_result=None) -> CheckContext:
        return CheckContext(path=self.path, modality="image", profile=self.profile, provenance=self.provenance,
                            content=content or {}, ai_result=ai_result or {})

    def run_pre(self) -> Dict[str, float]:
        """Runs integrity/format/metadata checks; returns capped log-odds terms for the detector."""
        self._pre = registry.run("image", self._ctx(), phase="pre")
        self._pre_terms = build_report(self._pre).score_terms
        return dict(self._pre_terms)

    def _ood(self) -> Dict[str, Any]:
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
        content: Optional[Dict[str, Any]] = None,
        ai_result: Optional[Dict[str, Any]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        gates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self._ctx(content, ai_result)
        post = registry.run("image", ctx, phase="post")
        report = build_report(self._pre + post, gates=gates)
        out = report.to_dict()

        out["ood"] = self._ood()
        add_band_and_open_set(out, ai_result, attribution, "image")
        return out
