"""
core.forensics.registry: isolated check execution + hybrid-score cap enforcement.

Checks are small functions ``fn(ctx) -> Finding | List[Finding] | None`` registered per
modality and phase ("pre" = before the AI detector runs, "post" = after). A failing check
never raises out of ``run``; it becomes an ERROR finding.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.forensics.config import TOTAL_LLR_CAP
from core.forensics.schemas import (
    SCORE_ELIGIBLE_CLASSES,
    DimensionReport,
    EvidenceClass,
    Finding,
    FindingStatus,
    Severity,
)

logger = logging.getLogger("core.forensics.registry")


@dataclass
class CheckContext:
    """Everything a check may read: file path, modality and the profile, provenance, content and detector results so far."""
    path: Path
    modality: str
    profile: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    content: Dict[str, Any] = field(default_factory=dict)
    ai_result: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)


CheckFn = Callable[[CheckContext], Any]


class CheckRegistry:
    """Holds the dimension checks per modality and phase and runs each one in isolation."""
    def __init__(self) -> None:
        self._checks: Dict[Tuple[str, str], List[Tuple[str, CheckFn]]] = {}

    def register(self, modality: str, check_id: str, phase: str = "pre") -> Callable[[CheckFn], CheckFn]:
        """Decorator that registers a check under ``check_id``. Re-registering replaces the old one."""
        def deco(fn: CheckFn) -> CheckFn:
            bucket = self._checks.setdefault((modality, phase), [])
            # Idempotent re-registration (module reloads in tests / Streamlit reruns).
            bucket[:] = [(cid, f) for cid, f in bucket if cid != check_id]
            bucket.append((check_id, fn))
            return fn

        return deco

    def run(self, modality: str, ctx: CheckContext, phase: str = "pre") -> List[Finding]:
        """Run every check for a modality and phase. A check that raises becomes an ``ERROR`` finding; the rest still run."""
        out: List[Finding] = []
        for check_id, fn in list(self._checks.get((modality, phase), [])):
            try:
                result = fn(ctx)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                logger.warning("dimension check %s failed: %s", check_id, exc)
                out.append(
                    Finding(
                        check_id=check_id,
                        dimension="-",
                        stage="errors",
                        title=check_id,
                        status=FindingStatus.ERROR,
                        severity=Severity.NONE,
                        evidence_class=EvidenceClass.RELIABILITY,
                        detail=f"Check raised {type(exc).__name__}: {exc}",
                    )
                )
                continue
            if result is None:
                continue
            if isinstance(result, Finding):
                out.append(result)
            else:
                out.extend(result)
        return out


registry = CheckRegistry()


def run_checks(modality: str, ctx: CheckContext, phase: str = "pre") -> List[Finding]:
    """Run the default registry's checks for a modality and phase."""
    return registry.run(modality, ctx, phase)


def build_report(findings: List[Finding], gates: Optional[Dict[str, Any]] = None) -> DimensionReport:
    """Applies the hybrid-scoring rules and assembles the report."""
    terms: Dict[str, float] = {}
    for f in findings:
        if f.evidence_class not in SCORE_ELIGIBLE_CLASSES:
            f.llr = None
            continue
        if f.llr is None or f.llr == 0.0:
            continue
        cap = abs(float(f.llr_cap))
        f.llr = max(-cap, min(cap, float(f.llr)))
        terms[f.check_id] = terms.get(f.check_id, 0.0) + f.llr

    total = sum(terms.values())
    if abs(total) > TOTAL_LLR_CAP:
        scale = TOTAL_LLR_CAP / abs(total)
        terms = {k: v * scale for k, v in terms.items()}

    reliability: Dict[str, Any] = {}
    for f in findings:
        if f.evidence_class == EvidenceClass.RELIABILITY and "level" in f.data:
            reliability = dict(f.data)
            break

    return DimensionReport(findings=list(findings), gates=dict(gates or {}), score_terms=terms, reliability=reliability)


def ctx_memo(key: str) -> Callable[[CheckFn], CheckFn]:
    """Memoise a check's result on ``ctx.extra`` so checks that reuse another check's data don't recompute it."""

    def deco(fn: CheckFn) -> CheckFn:
        @functools.wraps(fn)
        def wrapper(ctx: CheckContext):
            memo = ctx.extra.setdefault("_memo", {})
            if key not in memo:
                memo[key] = fn(ctx)
            return memo[key]

        return wrapper

    return deco
