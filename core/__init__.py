"""
core: Enterprise-grade modular forensic architecture.
Provides thread-safe atomic I/O, OWASP security hardening, anti-SSRF protections,
and the shared decision layer. `core` never imports a detector package; the orchestration facade
lives in `services/forensic_service.py`.
"""
from core.atomic_io import atomic_read_json, atomic_update_json, atomic_write_json
from core.decision import generate_final_decision, normalize_percentages
from core.security import SecureUrlFetcher, sanitize_filename, validate_secure_url


__all__ = [
    "atomic_write_json",
    "atomic_read_json",
    "atomic_update_json",
    "generate_final_decision",
    "normalize_percentages",
    "SecureUrlFetcher",
    "validate_secure_url",
    "sanitize_filename",
]
