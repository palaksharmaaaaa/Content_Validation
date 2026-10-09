"""
core.tests.test_enterprise_hardening: Test suite verifying OWASP security,
anti-SSRF protections, atomic file persistence, and headless service orchestration.
"""
import concurrent.futures
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from core.atomic_io import atomic_read_json, atomic_update_json, atomic_write_json
from services.forensic_service import ForensicService
from core.security import (
    SAFE_MAX_IMAGE_PIXELS,
    is_ip_restricted,
    sanitize_filename,
    validate_secure_url,
)
import ipaddress


class TestEnterpriseHardening(unittest.TestCase):
    """Verifies enterprise security boundaries and resilience."""

    def test_decompression_bomb_limit(self):
        """Verifies that Pillow's decompression bomb ceiling is safely locked."""
        self.assertEqual(Image.MAX_IMAGE_PIXELS, SAFE_MAX_IMAGE_PIXELS)
        self.assertLessEqual(Image.MAX_IMAGE_PIXELS, 100_000_000)

    def test_anti_ssrf_validation(self):
        """Verifies zero-trust anti-SSRF blocks private, loopback, and metadata IPs."""
        # Loopback
        valid, msg, _ = validate_secure_url("http://127.0.0.1:8080/admin")
        self.assertFalse(valid)
        self.assertIn("prohibited", msg.lower())

        valid, msg, _ = validate_secure_url("http://localhost/test")
        self.assertFalse(valid)

        # Cloud Metadata (AWS/GCP/Azure)
        valid, msg, _ = validate_secure_url("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(valid)

        # RFC 1918 Private ranges
        valid, _, _ = validate_secure_url("http://10.0.0.1/status")
        self.assertFalse(valid)

        valid, _, _ = validate_secure_url("http://192.168.1.1/gateway")
        self.assertFalse(valid)

        valid, _, _ = validate_secure_url("http://172.16.0.1/internal")
        self.assertFalse(valid)

        # Non-HTTP protocol rejection
        valid, msg, _ = validate_secure_url("file:///etc/passwd")
        self.assertFalse(valid)
        self.assertIn("only http and https", msg.lower())

        valid, _, _ = validate_secure_url("ftp://ftp.example.com/file")
        self.assertFalse(valid)

    def test_restricted_ip_ranges(self):
        """Verifies low-level restricted IP identification."""
        self.assertTrue(is_ip_restricted(ipaddress.ip_address("127.0.0.1")))
        self.assertTrue(is_ip_restricted(ipaddress.ip_address("10.10.10.10")))
        self.assertTrue(is_ip_restricted(ipaddress.ip_address("192.168.0.100")))
        self.assertTrue(is_ip_restricted(ipaddress.ip_address("169.254.169.254")))
        self.assertTrue(is_ip_restricted(ipaddress.ip_address("::1")))
        self.assertTrue(is_ip_restricted(ipaddress.ip_address("100.64.0.1")))  # Carrier-grade NAT

        # Public IP should not be restricted
        self.assertFalse(is_ip_restricted(ipaddress.ip_address("8.8.8.8")))
        self.assertFalse(is_ip_restricted(ipaddress.ip_address("1.1.1.1")))

    def test_filename_sanitization(self):
        """Verifies directory traversal and path injection sanitization."""
        self.assertEqual(sanitize_filename("../../../etc/passwd"), "passwd")
        self.assertEqual(sanitize_filename("..\\..\\windows\\system32\\cmd.exe"), "cmd.exe")
        self.assertEqual(sanitize_filename(r"C:\Users\x/..\mixed/separators.png"), "separators.png")   # same result on every OS
        cleaned = sanitize_filename("bad;rm -rf /;evil.png")
        self.assertNotIn(";", cleaned)
        self.assertNotIn(" ", cleaned)
        self.assertTrue(cleaned.endswith(".png"))

    def test_atomic_file_persistence_and_concurrency(self):
        """Verifies transactional atomic JSON updates under high concurrency."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_json = Path(tmp_dir) / "concurrent_state.json"
            atomic_write_json(target_json, {"counter": 0, "workers": []})

            def worker_task(worker_id: int):
                for _ in range(10):
                    def update_fn(data):
                        d = data or {"counter": 0, "workers": []}
                        d["counter"] = d.get("counter", 0) + 1
                        d.setdefault("workers", []).append(worker_id)
                        return d

                    atomic_update_json(target_json, update_fn)

            # Run 5 threads concurrently updating the same file
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker_task, i) for i in range(5)]
                for f in futures:
                    f.result()

            final_data = atomic_read_json(target_json)
            self.assertEqual(final_data["counter"], 50)
            self.assertEqual(len(final_data["workers"]), 50)

    def test_forensic_service_facade_health(self):
        """Verifies that ForensicService starts up cleanly and reports HEALTHY."""
        service = ForensicService.get_instance()
        health = service.health_check()
        self.assertEqual(health["status"], "HEALTHY")
        self.assertEqual(health["security"]["anti_ssrf"], "ACTIVE")
        self.assertEqual(health["security"]["decompression_bomb_protection"], "ACTIVE")
        self.assertEqual(health["security"]["atomic_storage"], "ACTIVE")


class TestSessionCacheIsolation(unittest.TestCase):
    def test_sessions_are_isolated_and_purge_is_scoped(self):
        from core.atomic_io import get_session_cache_dir, purge_ephemeral_cache

        a, b = get_session_cache_dir("sess-a"), get_session_cache_dir("sess-b")
        self.assertNotEqual(a, b)
        (a / "x.bin").write_bytes(b"1")
        (b / "y.bin").write_bytes(b"2")
        self.assertEqual(purge_ephemeral_cache(a), 1)
        self.assertFalse((a / "x.bin").exists())
        self.assertTrue((b / "y.bin").exists())
        purge_ephemeral_cache(b)

    def test_session_id_cannot_traverse(self):
        from core.atomic_io import get_session_cache_dir

        d = get_session_cache_dir("../../etc")
        self.assertEqual(d.parent, get_session_cache_dir("zz").parent)
        with self.assertRaises(ValueError):
            get_session_cache_dir("../..")

if __name__ == "__main__":
    unittest.main()
