"""Outbound fetches are limited to ports 80/443 (blocks internal-service probing on public hosts)."""
import pytest

from core.security import validate_secure_url


@pytest.mark.parametrize("url", ["http://example.com:22/x", "https://example.com:6379/", "http://example.com:8080/a"])
def test_non_web_ports_rejected(url):
    ok, msg, _ = validate_secure_url(url)
    assert not ok
    assert "port" in msg.lower()


def test_invalid_port_rejected():
    ok, msg, _ = validate_secure_url("http://example.com:99999/x")
    assert not ok
