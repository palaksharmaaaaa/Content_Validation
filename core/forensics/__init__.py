"""core.forensics: modality-agnostic dimension-check foundation."""
from core.forensics.registry import CheckContext, CheckRegistry, build_report, registry, run_checks
from core.forensics.schemas import (
    DimensionReport,
    EvidenceClass,
    Finding,
    FindingStatus,
    Severity,
)

__all__ = [
    "CheckContext",
    "CheckRegistry",
    "DimensionReport",
    "EvidenceClass",
    "Finding",
    "FindingStatus",
    "Severity",
    "build_report",
    "registry",
    "run_checks",
]
