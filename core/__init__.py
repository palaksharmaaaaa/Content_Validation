"""
core: Enterprise-grade modular forensic architecture.
Provides thread-safe atomic I/O, OWASP security hardening, anti-SSRF protections,
and unified forensic service orchestration decoupled from presentation layers.
"""
from core.atomic_io import atomic_read_json, atomic_update_json, atomic_write_json
from core.decision import generate_final_decision, normalize_percentages
from core.security import SecureUrlFetcher, sanitize_filename, validate_secure_url


def __getattr__(name: str):
    if name == "ForensicService":
        from core.forensic_service import ForensicService
        return ForensicService
    raise AttributeError(f"module 'core' has no attribute '{name}'")


__all__ = [
    "atomic_write_json",
    "atomic_read_json",
    "atomic_update_json",
    "generate_final_decision",
    "normalize_percentages",
    "SecureUrlFetcher",
    "validate_secure_url",
    "sanitize_filename",
    "ForensicService",
]
