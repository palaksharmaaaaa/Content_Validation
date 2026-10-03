"""
core.forensics.schemas: typed result objects shared by every modality's dimension checks.

A ``Finding`` is one check's outcome. Only PHYSICAL_SIGNAL / METADATA_WEAK findings may carry
a log-odds contribution (``llr``); every other evidence class is advisory and never moves P.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from core.forensics.config import PER_FINDING_LLR_CAP


class Severity(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FindingStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_CALIBRATED = "NOT_CALIBRATED"
    RECOGNIZED_OOS = "RECOGNIZED_OOS"
    ERROR = "ERROR"


class EvidenceClass(str, Enum):
    PHYSICAL_SIGNAL = "PHYSICAL_SIGNAL"
    METADATA_WEAK = "METADATA_WEAK"
    SECURITY = "SECURITY"
    LEGAL_FLAG = "LEGAL_FLAG"
    CONTEXT = "CONTEXT"
    RELIABILITY = "RELIABILITY"


# Classes allowed to contribute log-odds to the authenticity score.
SCORE_ELIGIBLE_CLASSES = frozenset({EvidenceClass.PHYSICAL_SIGNAL, EvidenceClass.METADATA_WEAK})


@dataclass
class Finding:
    check_id: str
    dimension: str
    stage: str
    title: str
    status: FindingStatus
    severity: Severity
    evidence_class: EvidenceClass
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    llr: Optional[float] = None
    llr_cap: float = PER_FINDING_LLR_CAP

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["severity"] = self.severity.value
        d["evidence_class"] = self.evidence_class.value
        return d


@dataclass
class DimensionReport:
    findings: List[Finding] = field(default_factory=list)
    gates: Dict[str, Any] = field(default_factory=dict)
    score_terms: Dict[str, float] = field(default_factory=dict)
    reliability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        by_stage: Dict[str, List[Dict[str, Any]]] = {}
        counts: Dict[str, int] = {}
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
