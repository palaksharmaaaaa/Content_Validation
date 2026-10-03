"""
audio_detector.dimension_checks: report-dimension checks wired through core.forensics.

Public API
----------
* ``check_audio_gates(path)``  -- hard-block + symbolic-music recognition gates (run first).
* ``AudioDimensionAnalysis``   -- ``run_pre()`` (before the detector; returns capped score terms for
  ``AudioAIDetector.analyze_audio_file(extra_log_lrs=...)``) and ``run_post()`` (after detector, scene and
  attribution; returns the full report dict incl. confidence band, OOD status, open-set attribution flag).
* ``normalize_provenance``     -- accepts both the pipeline's flat provenance and the UI's nested shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.forensics.reporting import UNKNOWN_SOURCE_MIN_AI_PERCENT as _UNKNOWN_MIN
from core.forensics.reporting import add_band_and_open_set, summarize_for_evidence_trail
from core.forensics.gates import HardBlockGate, recognize_symbolic_music
from core.forensics.ood import OODGate
from core.forensics.registry import CheckContext, build_report, registry
from core.forensics.schemas import Finding

# Importing the modules registers their checks. Order = execution order within a phase.
from audio_detector.dimension_checks import integrity  # noqa: F401
from audio_detector.dimension_checks import container  # noqa: F401
from audio_detector.dimension_checks import signal  # noqa: F401
from audio_detector.dimension_checks import context  # noqa: F401
from audio_detector.dimension_checks import legal  # noqa: F401
from audio_detector.dimension_checks import lifecycle  # noqa: F401
from audio_detector.dimension_checks import reliability  # noqa: F401

DEFAULT_OOD_STATS = Path(__file__).resolve().parents[1] / "data" / "ood_stats.npz"
UNKNOWN_SOURCE_MIN_AI_PERCENT = _UNKNOWN_MIN  # re-exported for callers

__all__ = [
    "AudioDimensionAnalysis",
    "acoustic_vector",
    "check_audio_gates",
    "gate_short_circuit_result",
    "normalize_provenance",
    "summarize_for_evidence_trail",
]


def normalize_provenance(prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    prov = prov or {}
    c2pa = prov.get("c2pa") if isinstance(prov.get("c2pa"), dict) else {}
    return {"c2pa_present": bool(prov.get("c2pa_present") or c2pa.get("c2pa_present"))}


def check_audio_gates(path: Union[str, Path]) -> Dict[str, Any]:
    hb = HardBlockGate().check(path)
    recog = recognize_symbolic_music(path)
    final_status = "HARD_BLOCK_ESCALATE" if hb.triggered else ("RECOGNIZED_OUT_OF_SCOPE" if recog else None)
    return {"hard_block": hb.to_dict(), "recognition": recog, "triggered": final_status is not None, "final_status": final_status}


def gate_short_circuit_result(path: Union[str, Path], gates: Dict[str, Any]) -> Dict[str, Any]:
    status = gates["final_status"]
    if status == "HARD_BLOCK_ESCALATE":
        reason = "File hash matches the operator-supplied hard-block list. Analysis stopped; escalate per policy."
    else:
        rec = gates.get("recognition") or {}
        reason = (f"Recognized {rec.get('type', 'symbolic')} data ({rec.get('description', '')}). "
                  "Outside this engine's waveform-authenticity scope; no verdict is produced.")
    return {
        "content_valid": True, "filename": Path(path).name, "path": str(path), "final_status": status, "ai_detected": False,
        "reason": reason, "authenticity_probabilities": {"p_ai": 0.0, "p_real": 0.0, "p_undecided": 100.0},
        "gate": gates, "evidence_trail": [reason],
    }


def acoustic_vector(features: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """5-dim acoustic feature vector used by the audio OOD gate (None if features are unavailable)."""
    if not features:
        return None
    try:
        return np.array([
            float(bool(features.get("has_vocoder_cutoff", False))),
            float(features.get("cutoff_freq_hz", 0.0)) / 10000.0,
            float(features.get("spectral_flatness", 0.0)) * 100.0,
            float(features.get("digital_silence_ratio", 0.0)),
            float(features.get("high_freq_ratio", 0.0)),
        ], dtype=np.float64)
    except (TypeError, ValueError):
        return None


class AudioDimensionAnalysis:
    def __init__(
        self,
        path: Union[str, Path],
        profile: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        samples: Optional[Tuple[Optional[np.ndarray], int, float]] = None,
        ood_gate: Optional[OODGate] = None,
    ):
        self.path = Path(path)
        self.profile = profile or {}
        self.provenance = normalize_provenance(provenance)
        self.samples = samples
        self._ood_gate = ood_gate
        self._pre: List[Finding] = []

    def _ctx(self, content=None, ai_result=None) -> CheckContext:
        extra: Dict[str, Any] = {}
        if self.samples is not None and self.samples[0] is not None:
            extra["samples"] = self.samples
        return CheckContext(path=self.path, modality="audio", profile=self.profile, provenance=self.provenance,
                            content=content or {}, ai_result=ai_result or {}, extra=extra)

    def run_pre(self) -> Dict[str, float]:
        self._pre = registry.run("audio", self._ctx(), phase="pre")
        return dict(build_report(self._pre).score_terms)

    def _ood(self, ai_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        gate = self._ood_gate or OODGate.load(DEFAULT_OOD_STATS)
        if not gate.calibrated:
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": None,
                    "note": "OOD gate is not fitted; fit it from your audio library to enable out-of-distribution detection."}
        vec = acoustic_vector((ai_result or {}).get("acoustic_features") or (ai_result or {}).get("spectral_features"))
        if vec is None:
            return {"status": "NOT_CALIBRATED", "distance": None, "threshold": gate.threshold, "note": "No acoustic features available."}
        return gate.score(vec)

    def run_post(
        self,
        content: Optional[Dict[str, Any]] = None,
        ai_result: Optional[Dict[str, Any]] = None,
        attribution: Optional[Dict[str, Any]] = None,
        gates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        post = registry.run("audio", self._ctx(content, ai_result), phase="post")
        out = build_report(self._pre + post, gates=gates).to_dict()
        out["ood"] = self._ood(ai_result)
        add_band_and_open_set(out, ai_result, attribution, "audio")
        return out
