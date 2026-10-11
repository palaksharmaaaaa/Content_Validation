"""
core.forensics.schemas: typed result objects shared by every modality's dimension checks.

A ``Finding`` is one check's outcome. Only PHYSICAL_SIGNAL, LEARNED_SIGNAL and METADATA_WEAK findings may carry
a log-odds contribution (``llr``); every other evidence class is advisory and never moves P.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from core.forensics.config import PER_FINDING_LLR_CAP


class Severity(StrEnum):
    """How serious a finding is."""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FindingStatus(StrEnum):
    """Outcome of a check (pass, warn, fail, info, not applicable, not calibrated, recognised out of scope, error)."""
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_CALIBRATED = "NOT_CALIBRATED"
    RECOGNIZED_OOS = "RECOGNIZED_OOS"
    ERROR = "ERROR"


class EvidenceClass(StrEnum):
    """What kind of evidence a finding is. Only physical-signal, learned-signal and weak-metadata findings may affect the score."""
    PHYSICAL_SIGNAL = "PHYSICAL_SIGNAL"
    LEARNED_SIGNAL = "LEARNED_SIGNAL"
    METADATA_WEAK = "METADATA_WEAK"
    SECURITY = "SECURITY"
    LEGAL_FLAG = "LEGAL_FLAG"
    CONTEXT = "CONTEXT"
    RELIABILITY = "RELIABILITY"


# Classes allowed to contribute log-odds to the authenticity score.
SCORE_ELIGIBLE_CLASSES = frozenset({EvidenceClass.PHYSICAL_SIGNAL, EvidenceClass.METADATA_WEAK, EvidenceClass.LEARNED_SIGNAL})


@dataclass
class Finding:
    """One check result: id, title, status, evidence class, detail and an optional capped log-likelihood ratio."""
    check_id: str
    dimension: str
    stage: str
    title: str
    status: FindingStatus
    severity: Severity
    evidence_class: EvidenceClass
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    llr: float | None = None
    llr_cap: float = PER_FINDING_LLR_CAP

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for reports."""
        d = asdict(self)
        d["status"] = self.status.value
        d["severity"] = self.severity.value
        d["evidence_class"] = self.evidence_class.value
        return d


@dataclass
class DimensionReport:
    """All findings of one analysis, grouped by stage, plus the capped total log-odds contribution."""
    findings: list[Finding] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    score_terms: dict[str, float] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for reports."""
        by_stage: dict[str, list[dict[str, Any]]] = {}
        counts: dict[str, int] = {}
        for f in self.findings:
            by_stage.setdefault(f.stage, []).append(f.to_dict())
            counts[f.status.value] = counts.get(f.status.value, 0) + 1
        return {
            "findings_by_stage": by_stage,
            "gates": dict(self.gates),
            "score_terms": dict(self.score_terms),
            "reliability": dict(self.reliability),
            "summary": {"total": len(self.findings), "by_status": counts},
        }
